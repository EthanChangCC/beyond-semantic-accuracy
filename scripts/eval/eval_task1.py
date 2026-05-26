"""
eval_task1.py
-------------
Task 1 linear evaluation: NER-F1, Risk-NER, Action-Exact, Action-Risk score.

Usage:
    # single file
    python scripts/eval/eval_task1.py output/task1/zeroshot/gpt-4o-mini.jsonl

    # whole directory  →  prints each model in sequence
    python scripts/eval/eval_task1.py output/task1/zeroshot/

    # multiple directories
    python scripts/eval/eval_task1.py \
        output/task1/zeroshot/ \
        output/task1/ablation_fewshot/ \
        output/task1/ablation_knowledge/ \
        output/task1/ablation_full_aligned/
"""

import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import json

# ======================
# Config
# ======================

root = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = root / "output/task1/zeroshot/gpt-4o-mini.jsonl"

ENTITY_WEIGHTS = {
    "callsign": 1.00,
    "runway": 1.00,
    "altitude": 1.00,
    "heading": 0.80,
    "speed": 0.40,
    "frequency": 0.40,
    "controller": 0.20,
    "waypoint": 0.80,
    "condition": 0.50,
    "attribute": 0.40,
    "relation": 0.40,
    "traffic": 0.40,
    "taxiway": 0.80,
    "O": 0.0,
    "altitude_control": 1.00,
    "clearance": 1.00,
    "heading_control": 1.00,
    "runway_instruction": 1.00,
    "speed_control": 1.00,
    "taxiway_instruction": 1.00,
    "contact": 1.00,
    "sequence_control": 1.00,
}

ACTION_WEIGHTS = {
    "altitude_control": 1.00,
    "clearance": 1.00,
    "heading_control": 0.60,
    "runway_instruction": 1.00,
    "taxiway_instruction": 0.80,
    "speed_control": 0.60,
    "contact": 0.30,
    "sequence_control": 0.40,
}

ACTION_SCHEMA = {
    "altitude_control":   {"slots": ["callsign", "altitude", "condition"]},
    "clearance":          {"slots": ["callsign", "runway", "waypoint", "condition"]},
    "heading_control":    {"slots": ["callsign", "heading", "condition"]},
    "runway_instruction": {"slots": ["callsign", "runway", "condition"]},
    "taxiway_instruction":{"slots": ["callsign", "taxiway", "waypoint", "runway", "condition"]},
    "speed_control":      {"slots": ["callsign", "speed", "condition"]},
    "contact":            {"slots": ["callsign", "frequency", "controller"]},
    "sequence_control":   {"slots": ["callsign", "condition", "traffic"]},
}

TOKEN_OVERLAP_THRESHOLD = 1.0
HIGH_RISK_THRESHOLD = 0.8


# ======================
# Utils
# ======================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_words(text):
    return [part for part in text.strip().split() if part]


def tokenize(text):
    return [part.lower() for part in split_words(text)]


def token_overlap(gt, pred):
    gt_tokens = tokenize(gt)
    pr_tokens = tokenize(pred)
    if not gt_tokens:
        return 0.0
    gt_counter = Counter(gt_tokens)
    pr_counter = Counter(pr_tokens)
    overlap = sum((gt_counter & pr_counter).values())
    return overlap / len(gt_tokens)


def normalize_actions(actions):
    if isinstance(actions, list):
        return [str(x) for x in actions]
    if isinstance(actions, str):
        return [actions]
    if actions:
        return [str(actions)]
    return []


def is_high_risk_sample(item, threshold=HIGH_RISK_THRESHOLD):
    gt_actions = normalize_action_list(item["original"]["action"]["type"])
    return any(ACTION_WEIGHTS.get(atype, 0.0) >= threshold for atype in gt_actions)


def is_multi_intent_sample(item):
    gt_actions = normalize_action_list(item["original"]["action"]["type"])
    return len(gt_actions) > 1


def get_source(item):
    return item["meta"].get("source", "unknown")


def is_augmented(item):
    return bool(item["meta"].get("augmented", False))


def project_span_labels_to_words_exact(sentence, label_response):
    sentence_words = split_words(sentence)
    projected = []
    cursor = 0

    for token in label_response.get("tokens", []):
        token_words = split_words(token.get("text", ""))
        label = str(token.get("label", "O"))
        if not token_words:
            continue

        if cursor + len(token_words) > len(sentence_words):
            raise ValueError(f"Token span exceeds sentence boundary: {token}")

        expected_words = sentence_words[cursor : cursor + len(token_words)]
        if expected_words != token_words:
            raise ValueError(
                f"Token words do not align with sentence at position {cursor}: "
                f"expected={expected_words}, token={token_words}, sentence='{sentence}'"
            )

        for word in token_words:
            projected.append({"text": word, "label": label})
        cursor += len(token_words)

    if cursor != len(sentence_words):
        raise ValueError(
            f"Label tokens do not fully cover the sentence. covered={cursor}, total={len(sentence_words)}, sentence='{sentence}'"
        )

    return projected


def project_span_labels_to_words_fuzzy(sentence, label_response):
    sentence_words = split_words(sentence)
    tokens = [token for token in label_response.get("tokens", []) if split_words(token.get("text", ""))]
    projected = []
    cursor = 0

    for i, token in enumerate(tokens):
        token_words = split_words(token.get("text", ""))
        label = str(token.get("label", "O"))
        remaining_tokens = len(tokens) - i - 1

        if cursor >= len(sentence_words):
            break

        if i == len(tokens) - 1:
            projected.extend({"text": word, "label": label} for word in sentence_words[cursor:])
            cursor = len(sentence_words)
            continue

        min_len = max(1, len(token_words) - 1)
        max_len = min(len(sentence_words) - cursor - remaining_tokens, len(token_words) + 2)
        if max_len < min_len:
            max_len = min_len

        best_end = None
        best_score = -1.0
        token_text = " ".join(token_words)
        for window_len in range(min_len, max_len + 1):
            end = cursor + window_len
            window_text = " ".join(sentence_words[cursor:end])
            score = SequenceMatcher(None, token_text, window_text).ratio() - 0.03 * abs(window_len - len(token_words))
            if score > best_score:
                best_score = score
                best_end = end

        if best_end is None:
            best_end = min(len(sentence_words), cursor + max(1, len(token_words)))

        projected.extend({"text": word, "label": label} for word in sentence_words[cursor:best_end])
        cursor = best_end

    if cursor < len(sentence_words):
        fill_label = projected[-1]["label"] if projected else "O"
        projected.extend({"text": word, "label": fill_label} for word in sentence_words[cursor:])

    return projected


def project_span_labels_to_words(sentence, label_response):
    try:
        return project_span_labels_to_words_exact(sentence, label_response)
    except ValueError:
        return project_span_labels_to_words_fuzzy(sentence, label_response)


def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        if tp == 0:
            f1s.append(0.0)
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))
    return sum(f1s) / len(f1s) if f1s else 0.0


def match_entities_pairs(gt_entities, pred_entities):
    pairs = []
    used_pred = set()
    for i, gt in enumerate(gt_entities):
        for j, pr in enumerate(pred_entities):
            if j in used_pred:
                continue
            if gt["label"] != pr["label"]:
                continue
            if token_overlap(gt["text"], pr["text"]) >= TOKEN_OVERLAP_THRESHOLD:
                pairs.append((i, j))
                used_pred.add(j)
                break
    return pairs


def word_projection_to_entities(sentence, label_response):
    words = project_span_labels_to_words(sentence, label_response)
    entities = []
    current = None

    for token in words:
        text = token["text"]
        label = token["label"]
        if current is None or current["label"] != label:
            if current is not None:
                entities.append(current)
            current = {"text": text, "label": label}
        else:
            current["text"] += f" {text}"

    if current is not None:
        entities.append(current)

    return entities


def build_action_slot_view(sentence, label_response):
    words = project_span_labels_to_words(sentence, label_response)
    grouped = defaultdict(list)
    for token in words:
        grouped[token["label"]].append(token["text"])

    entities = []
    current = None
    for token in words:
        text = token["text"]
        label = token["label"]
        if current is None or current["label"] != label:
            if current is not None:
                entities.append(current)
            current = {"text": text, "label": label}
        else:
            current["text"] += f" {text}"
    if current is not None:
        entities.append(current)

    slot_values = defaultdict(list)
    for entity in entities:
        if entity["label"] == "O":
            continue
        slot_values[entity["label"]].append(entity["text"])

    slots = {}
    for label, values in slot_values.items():
        if not values:
            continue
        slots[label] = values[0] if len(values) == 1 else values
    slots["type"] = normalize_actions(label_response.get("utterance_action", []))
    return slots


def is_empty_slot_value(val):
    if val is None:
        return True
    if isinstance(val, (list, tuple, set)) and len(val) == 0:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def has_slot_value(obj, slot):
    return slot in obj and not is_empty_slot_value(obj[slot])


def slot_value_match(gt_val, pred_val, threshold=TOKEN_OVERLAP_THRESHOLD):
    if is_empty_slot_value(gt_val) or is_empty_slot_value(pred_val):
        return False

    gt_vals = [gt_val] if isinstance(gt_val, str) else gt_val
    pr_vals = [pred_val] if isinstance(pred_val, str) else pred_val

    for g in gt_vals:
        for p in pr_vals:
            if token_overlap(str(g), str(p)) >= threshold:
                return True
    return False


def unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def normalize_action_list(actions):
    normalized = normalize_actions(actions)
    return normalized if isinstance(normalized, list) else [normalized]


def combined_action_slots(actions, action_schema):
    slots = []
    for atype in actions:
        schema = action_schema.get(atype)
        if not schema:
            continue
        slots.extend(schema["slots"])
    return unique_preserve_order(slots)


def combined_action_weight(actions):
    valid = [ACTION_WEIGHTS.get(atype, 0.0) for atype in actions if atype in ACTION_WEIGHTS]
    return sum(valid) / len(valid) if valid else 0.0


def evaluate_action_weighted_f1(dataset):
    weighted_sum = 0.0
    weight_total = 0.0

    for atype, weight in ACTION_WEIGHTS.items():
        tp = fp = fn = 0
        for item in dataset:
            gt_actions = set(normalize_action_list(item["original"]["action"]["type"]))
            pred_actions = set(normalize_action_list(item["pred"]["action"]["type"]))
            in_gt = atype in gt_actions
            in_pred = atype in pred_actions
            if in_gt and in_pred:
                tp += 1
            elif (not in_gt) and in_pred:
                fp += 1
            elif in_gt and (not in_pred):
                fn += 1

        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

        weighted_sum += weight * f1
        weight_total += weight

    return weighted_sum / weight_total if weight_total > 0 else 0.0


def evaluate_action_exact(dataset):
    matched = 0
    for item in dataset:
        gt_actions = normalize_action_list(item["original"]["action"]["type"])
        pred_actions = normalize_action_list(item["pred"]["action"]["type"])
        if gt_actions == pred_actions:
            matched += 1
    return matched / len(dataset) if dataset else 0.0


# ======================
# Entity Evaluation
# ======================

def evaluate_entities(dataset):
    slot_stats = defaultdict(lambda: {"hit": 0, "total": 0})
    total_gt = total_hit = total_pred = 0

    for item in dataset:
        gt = [e for e in item["original"]["entities"] if e["label"] != "O"]
        pr = [e for e in item["pred"]["entities"] if e["label"] != "O"]

        matched = {i for i, _ in match_entities_pairs(gt, pr)}

        total_gt += len(gt)
        total_pred += len(pr)
        total_hit += len(matched)

        for i, e in enumerate(gt):
            slot_stats[e["label"]]["total"] += 1
            if i in matched:
                slot_stats[e["label"]]["hit"] += 1

    recall = total_hit / total_gt if total_gt else 0.0
    precision = total_hit / total_pred if total_pred else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1, slot_stats


def evaluate_risk_ner(dataset):
    num = 0.0
    den = 0.0
    for item in dataset:
        gt = [e for e in item["original"]["entities"] if e["label"] != "O"]
        pr = [e for e in item["pred"]["entities"] if e["label"] != "O"]
        matched = {i for i, _ in match_entities_pairs(gt, pr)}
        for i, e in enumerate(gt):
            w = ENTITY_WEIGHTS.get(e["label"], 0.0)
            den += w
            if i in matched:
                num += w
    return num / den if den > 0 else 0.0


def evaluate_risk_ner_full(dataset):
    num_r = den_r = 0.0
    num_p = den_p = 0.0

    for item in dataset:
        gt = [e for e in item["original"]["entities"] if e["label"] != "O"]
        pr = [e for e in item["pred"]["entities"] if e["label"] != "O"]
        pairs = match_entities_pairs(gt, pr)

        matched_gt = {i for i, _ in pairs}
        matched_pr = {j for _, j in pairs}

        for i, e in enumerate(gt):
            w = ENTITY_WEIGHTS.get(e["label"], 0.0)
            den_r += w
            if i in matched_gt:
                num_r += w

        for j, e in enumerate(pr):
            w = ENTITY_WEIGHTS.get(e["label"], 0.0)
            den_p += w
            if j in matched_pr:
                num_p += w

    rec = num_r / den_r if den_r > 0 else 0.0
    prec = num_p / den_p if den_p > 0 else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return rec, prec, f1


# ======================
# Action Evaluation
# ======================

def evaluate_action_detail(dataset, action_schema):
    stats = defaultdict(lambda: defaultdict(lambda: {"hit": 0, "total": 0}))

    for item in dataset:
        gt_actions = item["original"]["action"]["type"]
        pred_actions = item["pred"]["action"]["type"]
        gt = item["original"]["action"]
        pr = item["pred"]["action"]

        action_list = gt_actions if isinstance(gt_actions, list) else [gt_actions]
        pred_list = pred_actions if isinstance(pred_actions, list) else [pred_actions]

        for atype in action_list:
            schema = action_schema.get(atype)
            if not schema:
                continue

            stats[atype]["type"]["total"] += 1
            if atype in pred_list:
                stats[atype]["type"]["hit"] += 1

            for slot in schema["slots"]:
                if slot == "type" or not has_slot_value(gt, slot):
                    continue
                stats[atype][slot]["total"] += 1
                if has_slot_value(pr, slot) and slot_value_match(gt[slot], pr[slot]):
                    stats[atype][slot]["hit"] += 1

    return stats


def evaluate_action_risk(dataset, action_schema):
    scores = []
    for item in dataset:
        gt_actions = item["original"]["action"]["type"]
        pred_actions = item["pred"]["action"]["type"]
        gt = item["original"]["action"]
        pr = item["pred"]["action"]

        action_list = normalize_action_list(gt_actions)
        pred_list = normalize_action_list(pred_actions)
        sample_weighted = 0.0
        sample_total = 0.0

        for atype in action_list:
            action_weight = ACTION_WEIGHTS.get(atype, 0.0)
            if action_weight <= 0.0:
                continue

            schema = action_schema.get(atype)
            if not schema:
                continue

            weighted_hit = 0.0
            weighted_total = 0.0
            for slot in schema["slots"]:
                slot_weight = ENTITY_WEIGHTS.get(slot, 0.0)
                if not has_slot_value(gt, slot):
                    continue
                weighted_total += slot_weight
                if has_slot_value(pr, slot) and slot_value_match(gt[slot], pr[slot]):
                    weighted_hit += slot_weight

            slot_recall = weighted_hit / weighted_total if weighted_total > 0 else 1.0
            type_factor = 1.0 if atype in pred_list else (1.0 - action_weight)
            sample_weighted += action_weight * type_factor * slot_recall
            sample_total += action_weight

        scores.append(sample_weighted / sample_total if sample_total > 0 else 1.0)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_action_risk_absolute(dataset, action_schema):
    scores = []
    for item in dataset:
        gt_actions = item["original"]["action"]["type"]
        pred_actions = item["pred"]["action"]["type"]
        gt = item["original"]["action"]
        pr = item["pred"]["action"]

        action_list = normalize_action_list(gt_actions)
        pred_list = normalize_action_list(pred_actions)
        slot_list = combined_action_slots(action_list, action_schema)

        all_correct = True
        weighted_hit = 0.0
        weighted_total = 0.0

        for slot in slot_list:
            w = ENTITY_WEIGHTS.get(slot, 0.0)
            if not has_slot_value(gt, slot):
                continue
            weighted_total += w
            if has_slot_value(pr, slot) and slot_value_match(gt[slot], pr[slot]):
                weighted_hit += w
            else:
                all_correct = False

        if any(atype not in pred_list for atype in action_list):
            all_correct = False

        if not all_correct:
            scores.append(0.0)
            continue

        base = weighted_hit / weighted_total if weighted_total > 0 else 1.0
        scores.append(base)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_action_macro(dataset, action_schema):
    scores = []
    for item in dataset:
        gt_actions = item["original"]["action"]["type"]
        gt = item["original"]["action"]
        pr = item["pred"]["action"]
        action_list = normalize_action_list(gt_actions)
        pred_list = normalize_action_list(pr.get("type", []))
        slot_list = combined_action_slots(action_list, action_schema)

        hit = 0
        tot = 0
        for slot in slot_list:
            if not has_slot_value(gt, slot):
                continue
            tot += 1
            if has_slot_value(pr, slot) and slot_value_match(gt[slot], pr[slot]):
                hit += 1

        for atype in action_list:
            tot += 1
            if atype in pred_list:
                hit += 1

        scores.append(hit / tot if tot > 0 else 1.0)

    return sum(scores) / len(scores) if scores else 0.0


# ======================
# Main Eval
# ======================

def convert_dataset(rows):
    dataset = []
    for row in rows:
        sentence = row["sentence"]
        gt_lr = row["groundtruth"]
        pr_lr = row["predict"]
        dataset.append(
            {
                "meta": row.get("meta", {}),
                "sentence": sentence,
                "original": {
                    "entities": word_projection_to_entities(sentence, gt_lr),
                    "action": build_action_slot_view(sentence, gt_lr),
                },
                "pred": {
                    "entities": word_projection_to_entities(sentence, pr_lr),
                    "action": build_action_slot_view(sentence, pr_lr),
                },
            }
        )
    return dataset


def run_eval(dataset):
    risk_rec, risk_prec, risk_f1 = evaluate_risk_ner_full(dataset)
    return {
        "action_type_f1": evaluate_action_weighted_f1(dataset),
        "action_exact": evaluate_action_exact(dataset),
        "ner": evaluate_entities(dataset),
        "risk_ner": evaluate_risk_ner(dataset),
        "risk_ner_full": (risk_rec, risk_prec, risk_f1),
        "action_detail": evaluate_action_detail(dataset, ACTION_SCHEMA),
        "action_risk": evaluate_action_risk(dataset, ACTION_SCHEMA),
        "action_risk_absolute": evaluate_action_risk_absolute(dataset, ACTION_SCHEMA),
        "action_macro": evaluate_action_macro(dataset, ACTION_SCHEMA),
    }


def print_results(res):
    ner_r, ner_p, ner_f, _ = res["ner"]
    risk_rec, risk_prec, risk_f1 = res["risk_ner_full"]

    print(f"** Action-Exact: {res['action_exact']:.4f}**")
    print(
        f"** risk-ner f1: {risk_f1:.4f} (risk-recall: {risk_rec:.4f}, risk-prec: {risk_prec:.4f})**    "
        f"(ner-f1 (unweighted): {ner_f:.4f} [rec {ner_r:.4f}, prec {ner_p:.4f}])"
    )
    print(
        f"** action-risk score: {res['action_risk']:.4f}**    "
        f"(action-risk strict: {res['action_risk_absolute']:.4f}    "
        f"action w/t : {res['action_risk_no_type'] if 'action_risk_no_type' in res else 'n/a'}    "
        f"action macro (slots,type=1): {res['action_macro']:.4f})"
    )

    print("-" * 80)
    print("action detail:")
    for atype, slots in res["action_detail"].items():
        parts = []
        for slot in ACTION_SCHEMA[atype]["slots"]:
            if slot in slots and slots[slot]["total"] > 0:
                h = slots[slot]["hit"]
                t = slots[slot]["total"]
                parts.append(f"{slot}: {h/t:.4f} ({h}/{t})")
        type_stats = slots.get("type")
        if type_stats and type_stats["total"] > 0:
            h = type_stats["hit"]
            t = type_stats["total"]
            parts.insert(0, f"type: {h/t:.4f} ({h}/{t})")
        print(f"[{atype}]  " + "   ".join(parts))


def print_subset_results(name, dataset):
    print("-" * 80)
    print(f"{name}: {len(dataset)} samples")
    if not dataset:
        print("No samples.")
        return

    res = run_eval(dataset)
    ner_r, ner_p, ner_f, _ = res["ner"]
    risk_rec, risk_prec, risk_f1 = res["risk_ner_full"]

    print(f"** Action-Exact: {res['action_exact']:.4f}**")
    print(
        f"** risk-ner f1: {risk_f1:.4f} (risk-recall: {risk_rec:.4f}, risk-prec: {risk_prec:.4f})**    "
        f"(ner-f1 (unweighted): {ner_f:.4f} [rec {ner_r:.4f}, prec {ner_p:.4f}])"
    )
    print(
        f"** action-risk score: {res['action_risk']:.4f}**    "
        f"(action-risk strict: {res['action_risk_absolute']:.4f}    "
        f"action macro (slots,type=1): {res['action_macro']:.4f})"
    )


def _eval_path(p):
    """Returns list of (model_name, dataset) for a file or directory."""
    p = Path(p)
    if p.is_file():
        return [(p.stem, convert_dataset(load_jsonl(str(p))))]
    elif p.is_dir():
        results = []
        for f in sorted(p.glob("*.jsonl")):
            if "_failed" in f.stem:
                continue
            rows = load_jsonl(str(f))
            if rows:
                results.append((f.stem, convert_dataset(rows)))
        return results
    else:
        print(f"[SKIP] Not found: {p}")
        return []


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    for arg in args:
        entries = _eval_path(arg)
        for model_name, dataset in entries:
            print(f"\n{'='*72}")
            print(f"  Model: {model_name}   (N={len(dataset)})")
            print(f"{'='*72}")
            results = run_eval(dataset)
            print_results(results)
            print_subset_results("high-risk only", [i for i in dataset if is_high_risk_sample(i)])
            print_subset_results("multi-intent only", [i for i in dataset if is_multi_intent_sample(i)])
            print_subset_results("augmented (filler words)", [i for i in dataset if is_augmented(i)])
            print_subset_results("non-augmented", [i for i in dataset if not is_augmented(i)])
            sources = sorted(set(get_source(i) for i in dataset))
            for src in sources:
                print_subset_results(f"source={src}", [i for i in dataset if get_source(i) == src])
