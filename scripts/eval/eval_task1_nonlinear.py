"""
eval_task1_nonlinear.py
-----------------------
Task 1 nonlinear consequence-aware evaluation.
Produces a side-by-side comparison of three scoring methods.

Methods
-------
  linear : weighted average (baseline, same as eval_task1.py)
  geo    : weighted geometric mean  score = Π_s (ε + m_s)^(w_s / Σ_s w_s)
           ε = 1e-5 prevents log(0); any high-weight miss collapses score
  erp    : Exponential Risk Penalty  score = exp(-λ * Σ w_i * (1 - m_i))
           λ = 3.0 (action-risk), 2.0 (NER); errors compound multiplicatively

Usage:
    # use default MODEL_PATHS (output/task1/zeroshot/)
    python scripts/eval/eval_task1_nonlinear.py

    # specific files
    python scripts/eval/eval_task1_nonlinear.py \
        output/task1/zeroshot/gpt-4o-mini.jsonl \
        output/task1/zeroshot/deepseek-chat.jsonl

    # whole directory
    python scripts/eval/eval_task1_nonlinear.py output/task1/zeroshot/

    # multiple directories (each dir runs as a separate comparison table)
    python scripts/eval/eval_task1_nonlinear.py \
        output/task1/zeroshot/ \
        output/task1/ablation_fewshot/
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# ======================
# Config
# ======================

root = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "deepseek-chat":      root / "output/task1/zeroshot/deepseek-chat.jsonl",
    "gpt-4o-mini":        root / "output/task1/zeroshot/gpt-4o-mini.jsonl",
    "gpt-5.1":            root / "output/task1/zeroshot/gpt-5.1.jsonl",
    "gpt-5.4":            root / "output/task1/zeroshot/gpt-5.4.jsonl",
    "qwen-plus":          root / "output/task1/zeroshot/qwen-plus.jsonl",
    "qwen3-14b":          root / "output/task1/zeroshot/qwen3-14b.jsonl",
    "qwen3-8b":           root / "output/task1/zeroshot/qwen3-8b.jsonl",
    "claude-haiku-4-5":   root / "output/task1/zeroshot/claude-haiku-4-5-20251001.jsonl",
}

# Slot-level risk weights
ENTITY_WEIGHTS = {
    "callsign":   1.00,
    "runway":     1.00,
    "altitude":   1.00,
    "heading":    0.80,
    "speed":      0.40,
    "frequency":  0.40,
    "controller": 0.20,
    "waypoint":   0.80,
    "taxiway":    0.80,
    "condition":  0.50,
    "traffic":    0.40,
    "attribute":  0.40,
    "relation":   0.40,
    "O": 0.0,
}

# Action-level risk weights
ACTION_WEIGHTS = {
    "altitude_control":   1.00,
    "clearance":          1.00,
    "heading_control":    0.60,
    "runway_instruction": 1.00,
    "taxiway_instruction":0.80,
    "speed_control":      0.60,
    "contact":            0.30,
    "sequence_control":   0.40,
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
HIGH_RISK_THRESHOLD     = 0.8

ERP_LAMBDA     = 3.0
ERP_LAMBDA_NER = 2.0
GEO_EPS        = 1e-5


# ======================
# I/O
# ======================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ======================
# Text utils
# ======================

def split_words(text):
    return [p for p in text.strip().split() if p]


def tokenize(text):
    return [p.lower() for p in split_words(text)]


def token_overlap(gt, pred):
    gt_tokens = tokenize(gt)
    pr_tokens = tokenize(pred)
    if not gt_tokens:
        return 0.0
    gt_counter = Counter(gt_tokens)
    pr_counter = Counter(pr_tokens)
    overlap = sum((gt_counter & pr_counter).values())
    return overlap / len(gt_tokens)


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


# ======================
# Action utils
# ======================

def normalize_actions(actions):
    if isinstance(actions, list):
        return [str(x) for x in actions]
    if isinstance(actions, str):
        return [actions]
    if actions:
        return [str(actions)]
    return []


def normalize_action_list(actions):
    normalized = normalize_actions(actions)
    return normalized if isinstance(normalized, list) else [normalized]


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def combined_action_slots(actions, action_schema):
    slots = []
    for atype in actions:
        schema = action_schema.get(atype)
        if schema:
            slots.extend(schema["slots"])
    return unique_preserve_order(slots)


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
# Span projection
# ======================

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
        expected = sentence_words[cursor: cursor + len(token_words)]
        if expected != token_words:
            raise ValueError(
                f"Mismatch at pos {cursor}: expected={expected}, got={token_words}"
            )
        for word in token_words:
            projected.append({"text": word, "label": label})
        cursor += len(token_words)
    if cursor != len(sentence_words):
        raise ValueError(f"Labels do not cover full sentence (covered {cursor}/{len(sentence_words)})")
    return projected


def project_span_labels_to_words_fuzzy(sentence, label_response):
    sentence_words = split_words(sentence)
    tokens = [t for t in label_response.get("tokens", []) if split_words(t.get("text", ""))]
    projected = []
    cursor = 0
    for i, token in enumerate(tokens):
        token_words = split_words(token.get("text", ""))
        label = str(token.get("label", "O"))
        remaining = len(tokens) - i - 1
        if cursor >= len(sentence_words):
            break
        if i == len(tokens) - 1:
            projected.extend({"text": w, "label": label} for w in sentence_words[cursor:])
            cursor = len(sentence_words)
            continue
        min_len = max(1, len(token_words) - 1)
        max_len = min(len(sentence_words) - cursor - remaining, len(token_words) + 2)
        if max_len < min_len:
            max_len = min_len
        best_end, best_score = None, -1.0
        token_text = " ".join(token_words)
        for wlen in range(min_len, max_len + 1):
            end = cursor + wlen
            window = " ".join(sentence_words[cursor:end])
            score = SequenceMatcher(None, token_text, window).ratio() - 0.03 * abs(wlen - len(token_words))
            if score > best_score:
                best_score = score
                best_end = end
        if best_end is None:
            best_end = min(len(sentence_words), cursor + max(1, len(token_words)))
        projected.extend({"text": w, "label": label} for w in sentence_words[cursor:best_end])
        cursor = best_end
    if cursor < len(sentence_words):
        fill = projected[-1]["label"] if projected else "O"
        projected.extend({"text": w, "label": fill} for w in sentence_words[cursor:])
    return projected


def project_span_labels_to_words(sentence, label_response):
    try:
        return project_span_labels_to_words_exact(sentence, label_response)
    except ValueError:
        return project_span_labels_to_words_fuzzy(sentence, label_response)


def word_projection_to_entities(sentence, label_response):
    words = project_span_labels_to_words(sentence, label_response)
    entities = []
    current = None
    for token in words:
        text, label = token["text"], token["label"]
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


def convert_dataset(rows):
    dataset = []
    for row in rows:
        sentence = row["sentence"]
        gt_lr = row["groundtruth"]
        pr_lr = row["predict"]
        dataset.append({
            "meta":     row.get("meta", {}),
            "sentence": sentence,
            "original": {
                "entities": word_projection_to_entities(sentence, gt_lr),
                "action":   build_action_slot_view(sentence, gt_lr),
            },
            "pred": {
                "entities": word_projection_to_entities(sentence, pr_lr),
                "action":   build_action_slot_view(sentence, pr_lr),
            },
        })
    return dataset


# ======================
# Sample filters
# ======================

def is_high_risk_sample(item):
    gt_actions = normalize_action_list(item["original"]["action"]["type"])
    return any(ACTION_WEIGHTS.get(a, 0.0) >= HIGH_RISK_THRESHOLD for a in gt_actions)


def is_multi_intent_sample(item):
    return len(normalize_action_list(item["original"]["action"]["type"])) > 1


def get_source(item):
    return item["meta"].get("source", "unknown")


def is_augmented(item):
    return bool(item["meta"].get("augmented", False))


# ======================
# Macro F1
# ======================

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


# ======================
# Entity matching
# ======================

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


# ======================
# NER scoring helpers
# ======================

def _ner_linear_score(gt, pr, entity_weights):
    pairs = match_entities_pairs(gt, pr)
    matched_gt = {i for i, _ in pairs}
    matched_pr = {j for _, j in pairs}

    num_r = sum(entity_weights.get(e["label"], 0.0) for i, e in enumerate(gt) if i in matched_gt)
    den_r = sum(entity_weights.get(e["label"], 0.0) for e in gt)
    num_p = sum(entity_weights.get(e["label"], 0.0) for j, e in enumerate(pr) if j in matched_pr)
    den_p = sum(entity_weights.get(e["label"], 0.0) for e in pr)

    rec  = num_r / den_r if den_r > 0 else 1.0
    prec = num_p / den_p if den_p > 0 else 1.0
    return rec, prec


def _ner_geo_score(gt, pr, entity_weights, eps=GEO_EPS):
    pairs = match_entities_pairs(gt, pr)
    matched_gt = {i for i, _ in pairs}
    matched_pr = {j for _, j in pairs}

    def wgm(items, matched_set, weights):
        ws = [weights.get(e["label"], 0.0) for e in items]
        ms = [1.0 if i in matched_set else 0.0 for i, _ in enumerate(items)]
        total_w = sum(ws)
        if total_w <= 0:
            return 1.0
        acc = sum(w * math.log(eps + m) for w, m in zip(ws, ms))
        return math.exp(acc / total_w)

    rec  = wgm(gt, matched_gt, entity_weights)
    prec = wgm(pr, matched_pr, entity_weights)
    return rec, prec


def _ner_erp_score(gt, pr, entity_weights, lam=ERP_LAMBDA_NER):
    pairs = match_entities_pairs(gt, pr)
    matched_gt = {i for i, _ in pairs}
    matched_pr = {j for _, j in pairs}

    def erp_score(items, matched_set, weights):
        penalty = sum(
            weights.get(e["label"], 0.0) * (0.0 if i in matched_set else 1.0)
            for i, e in enumerate(items)
        )
        return math.exp(-lam * penalty)

    rec  = erp_score(gt, matched_gt, entity_weights)
    prec = erp_score(pr, matched_pr, entity_weights)
    return rec, prec


# ======================
# NER evaluation
# ======================

def evaluate_ner_all(dataset):
    lin_recs, lin_precs = [], []
    geo_recs, geo_precs = [], []
    erp_recs, erp_precs = [], []

    for item in dataset:
        gt = [e for e in item["original"]["entities"] if e["label"] != "O"]
        pr = [e for e in item["pred"]["entities"] if e["label"] != "O"]

        r, p = _ner_linear_score(gt, pr, ENTITY_WEIGHTS)
        lin_recs.append(r); lin_precs.append(p)

        r, p = _ner_geo_score(gt, pr, ENTITY_WEIGHTS)
        geo_recs.append(r); geo_precs.append(p)

        r, p = _ner_erp_score(gt, pr, ENTITY_WEIGHTS)
        erp_recs.append(r); erp_precs.append(p)

    def f1(recs, precs):
        r = sum(recs)  / len(recs)
        p = sum(precs) / len(precs)
        return r, p, 2*r*p/(r+p) if r+p > 0 else 0.0

    return {
        "linear": f1(lin_recs, lin_precs),
        "geo":    f1(geo_recs, geo_precs),
        "erp":    f1(erp_recs, erp_precs),
    }


# ======================
# Action scoring helpers
# ======================

def _action_slot_matches_for_type(gt, pr, atype, action_schema):
    schema = action_schema.get(atype)
    if not schema:
        return []

    result = []
    for slot in schema["slots"]:
        w = ENTITY_WEIGHTS.get(slot, 0.0)
        if not has_slot_value(gt, slot):
            continue
        m = 1.0 if (has_slot_value(pr, slot) and slot_value_match(gt[slot], pr[slot])) else 0.0
        result.append((slot, w, m))
    return result


def _per_action_components(item, action_schema):
    gt_actions = normalize_action_list(item["original"]["action"]["type"])
    pred_actions = set(normalize_action_list(item["pred"]["action"]["type"]))
    gt = item["original"]["action"]
    pr = item["pred"]["action"]

    components = []
    for atype in gt_actions:
        action_weight = ACTION_WEIGHTS.get(atype, 0.0)
        if action_weight <= 0.0:
            continue
        type_factor = 1.0 if atype in pred_actions else (1.0 - action_weight)
        slot_matches = _action_slot_matches_for_type(gt, pr, atype, action_schema)
        components.append((atype, action_weight, type_factor, slot_matches))
    return components


def _slot_score_linear(slot_matches):
    total_w = sum(w for _, w, _ in slot_matches)
    hit_w   = sum(w * m for _, w, m in slot_matches)
    return hit_w / total_w if total_w > 0 else 1.0


def _slot_score_geo(slot_matches, eps=GEO_EPS):
    total_w = sum(w for _, w, _ in slot_matches)
    if total_w <= 0:
        return 1.0
    acc = sum(w * math.log(eps + m) for _, w, m in slot_matches)
    return math.exp(acc / total_w)


def _slot_score_erp(slot_matches, lam=ERP_LAMBDA):
    penalty = sum(w * (1.0 - m) for _, w, m in slot_matches)
    return math.exp(-lam * penalty)


# ======================
# Action evaluation
# ======================

def evaluate_action_all(dataset, action_schema):
    lin_scores, geo_scores, erp_scores = [], [], []

    for item in dataset:
        components = _per_action_components(item, action_schema)
        total_action_weight = sum(action_weight for _, action_weight, _, _ in components)
        if total_action_weight <= 0:
            lin_scores.append(1.0)
            geo_scores.append(1.0)
            erp_scores.append(1.0)
            continue

        lin_num = geo_num = erp_num = 0.0
        for _, action_weight, type_factor, slot_matches in components:
            lin_num += action_weight * type_factor * _slot_score_linear(slot_matches)
            geo_num += action_weight * type_factor * _slot_score_geo(slot_matches)
            erp_num += action_weight * type_factor * _slot_score_erp(slot_matches)

        lin_scores.append(lin_num / total_action_weight)
        geo_scores.append(geo_num / total_action_weight)
        erp_scores.append(erp_num / total_action_weight)

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    return {
        "linear": avg(lin_scores),
        "geo":    avg(geo_scores),
        "erp":    avg(erp_scores),
    }


def evaluate_action_detail(dataset, action_schema):
    stats = defaultdict(lambda: defaultdict(lambda: {"hit": 0, "total": 0}))
    for item in dataset:
        gt_actions   = item["original"]["action"]["type"]
        pred_actions = item["pred"]["action"]["type"]
        gt = item["original"]["action"]
        pr = item["pred"]["action"]

        action_list = gt_actions if isinstance(gt_actions, list) else [gt_actions]
        pred_list   = pred_actions if isinstance(pred_actions, list) else [pred_actions]

        for atype in action_list:
            schema = action_schema.get(atype)
            if not schema:
                continue
            stats[atype]["type"]["total"] += 1
            if atype in pred_list:
                stats[atype]["type"]["hit"] += 1
            for slot in schema["slots"]:
                if not has_slot_value(gt, slot):
                    continue
                stats[atype][slot]["total"] += 1
                if has_slot_value(pr, slot) and slot_value_match(gt[slot], pr[slot]):
                    stats[atype][slot]["hit"] += 1
    return stats


# ======================
# Full eval
# ======================

def run_eval(dataset):
    ner_scores    = evaluate_ner_all(dataset)
    action_scores = evaluate_action_all(dataset, ACTION_SCHEMA)
    action_detail = evaluate_action_detail(dataset, ACTION_SCHEMA)

    action_type_f1 = evaluate_action_weighted_f1(dataset)
    action_exact   = evaluate_action_exact(dataset)
    return {
        "action_type_f1": action_type_f1,
        "action_exact":   action_exact,
        "ner":            ner_scores,
        "action":         action_scores,
        "action_detail":  action_detail,
    }


# ======================
# Print
# ======================

def print_results(model_name, res, n):
    w = 14
    ner    = res["ner"]
    action = res["action"]

    print(f"\n{'='*72}")
    print(f"  Model: {model_name}   (N={n})")
    print(f"{'='*72}")
    print(f"  Action-Exact   : {res['action_exact']:.4f}")
    print()
    print(f"  {'Metric':<28}  {'Linear':>{w}}  {'Geo-Mean':>{w}}  {'ERP (λ={:.0f})'.format(ERP_LAMBDA):>{w}}")
    print(f"  {'-'*28}  {'-'*w}  {'-'*w}  {'-'*w}")

    nr, np_, nf = ner["linear"]
    gr, gp, gf  = ner["geo"]
    er, ep, ef  = ner["erp"]
    print(f"  {'NER F1 (weighted)':<28}  {nf:>{w}.4f}  {gf:>{w}.4f}  {ef:>{w}.4f}")
    print(f"  {'  recall':<28}  {nr:>{w}.4f}  {gr:>{w}.4f}  {er:>{w}.4f}")
    print(f"  {'  precision':<28}  {np_:>{w}.4f}  {gp:>{w}.4f}  {ep:>{w}.4f}")
    print()
    print(f"  {'Action-Risk Score':<28}  {action['linear']:>{w}.4f}  {action['geo']:>{w}.4f}  {action['erp']:>{w}.4f}")
    print()

    print(f"  action detail:")
    for atype, slots in sorted(res["action_detail"].items()):
        parts = []
        type_s = slots.get("type")
        if type_s and type_s["total"] > 0:
            h, t = type_s["hit"], type_s["total"]
            flag = " !" if h/t < 0.8 else ""
            parts.append(f"type: {h/t:.4f} ({h}/{t}){flag}")
        for slot in ACTION_SCHEMA.get(atype, {}).get("slots", []):
            if slot in slots and slots[slot]["total"] > 0:
                h, t = slots[slot]["hit"], slots[slot]["total"]
                w_slot = ENTITY_WEIGHTS.get(slot, 0.0)
                flag = " !!" if w_slot >= HIGH_RISK_THRESHOLD and h/t < 0.5 else ""
                parts.append(f"{slot}: {h/t:.4f} ({h}/{t}){flag}")
        print(f"    [{atype}]  " + "   ".join(parts))


def print_comparison_table(results_by_model):
    methods = ["linear", "geo", "erp"]
    col = 13
    mc  = 20

    models = list(results_by_model.keys())

    def header(title):
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

    header("Action-Risk Score  (linear / geo / erp)")
    print(f"  {'Model':<{mc}}  {'N':>5}  " + "  ".join(f"{m:>{col}}" for m in methods))
    print(f"  {'-'*mc}  {'-'*5}  " + "  ".join("-"*col for _ in methods))
    for name, (res, n) in results_by_model.items():
        row = "  ".join(f"{res['action'][m]:>{col}.4f}" for m in methods)
        print(f"  {name:<{mc}}  {n:>5}  {row}")

    header("NER F1 — weighted  (linear / geo / erp)")
    print(f"  {'Model':<{mc}}  {'N':>5}  " + "  ".join(f"{m:>{col}}" for m in methods))
    print(f"  {'-'*mc}  {'-'*5}  " + "  ".join("-"*col for _ in methods))
    for name, (res, n) in results_by_model.items():
        row = "  ".join(f"{res['ner'][m][2]:>{col}.4f}" for m in methods)
        print(f"  {name:<{mc}}  {n:>5}  {row}")

    header("Action-Exact")
    print(f"  {'Model':<{mc}}  {'N':>5}  {'Action-Exact':>{col}}")
    print(f"  {'-'*mc}  {'-'*5}  {'-'*col}")
    for name, (res, n) in results_by_model.items():
        print(f"  {name:<{mc}}  {n:>5}  {res['action_exact']:>{col}.4f}")

    header("Action-Risk Score — High-risk subset only  (action weight ≥ 0.8)")
    print(f"  {'Model':<{mc}}  {'N':>5}  " + "  ".join(f"{m:>{col}}" for m in methods))
    print(f"  {'-'*mc}  {'-'*5}  " + "  ".join("-"*col for _ in methods))
    for name, (res, n) in results_by_model.items():
        hr = res.get("high_risk")
        if hr:
            row = "  ".join(f"{hr['action'][m]:>{col}.4f}" for m in methods)
            print(f"  {name:<{mc}}  {hr['n']:>5}  {row}")
        else:
            print(f"  {name:<{mc}}  {'—':>5}")

    header("Action-Risk Score — Multi-intent subset only")
    print(f"  {'Model':<{mc}}  {'N':>5}  " + "  ".join(f"{m:>{col}}" for m in methods))
    print(f"  {'-'*mc}  {'-'*5}  " + "  ".join("-"*col for _ in methods))
    for name, (res, n) in results_by_model.items():
        mi = res.get("multi_intent")
        if mi:
            row = "  ".join(f"{mi['action'][m]:>{col}.4f}" for m in methods)
            print(f"  {name:<{mc}}  {mi['n']:>5}  {row}")
        else:
            print(f"  {name:<{mc}}  {'—':>5}")

    header("Per-action Slot Accuracy  (hit/total  across models)")
    all_atypes = sorted(ACTION_SCHEMA.keys())

    for atype in all_atypes:
        slots = ["type"] + ACTION_SCHEMA[atype]["slots"]
        print(f"\n  [{atype}]")
        print(f"  {'slot':<14}  {'weight':>6}  " + "  ".join(f"{m:>22}" for m in models))
        print(f"  {'-'*14}  {'-'*6}  " + "  ".join("-"*22 for _ in models))
        for slot in slots:
            w = ACTION_WEIGHTS.get(atype, 0.0) if slot == "type" else ENTITY_WEIGHTS.get(slot, 0.0)
            cells = []
            for name, (res, _) in results_by_model.items():
                detail = res["action_detail"].get(atype, {}).get(slot)
                if detail and detail["total"] > 0:
                    h, t = detail["hit"], detail["total"]
                    rate = h / t
                    flag = " !!" if w >= HIGH_RISK_THRESHOLD and rate < 0.5 else (
                           " !" if rate < 0.75 else "")
                    cells.append(f"{rate:.4f} ({h:>3}/{t:<3}){flag:3}")
                else:
                    cells.append(f"{'—':>22}")
            print(f"  {slot:<14}  {w:>6.2f}  " + "  ".join(f"{c:>22}" for c in cells))


# ======================
# Entry point
# ======================

def _add_subsets(res, dataset):
    high_risk = [item for item in dataset if is_high_risk_sample(item)]
    multi_int = [item for item in dataset if is_multi_intent_sample(item)]
    augmented = [item for item in dataset if is_augmented(item)]
    non_aug   = [item for item in dataset if not is_augmented(item)]

    if high_risk:
        hr = run_eval(high_risk); hr["n"] = len(high_risk); res["high_risk"] = hr
    if multi_int:
        mi = run_eval(multi_int); mi["n"] = len(multi_int); res["multi_intent"] = mi
    if augmented:
        aug = run_eval(augmented); aug["n"] = len(augmented); res["augmented"] = aug
    if non_aug:
        na = run_eval(non_aug); na["n"] = len(non_aug); res["non_augmented"] = na

    sources = sorted(set(get_source(item) for item in dataset))
    res["by_source"] = {}
    for src in sources:
        subset = [item for item in dataset if get_source(item) == src]
        if subset:
            sr = run_eval(subset); sr["n"] = len(subset)
            res["by_source"][src] = sr

    return res


def print_subset_summary(results_by_model):
    col = 13
    mc  = 20

    def header(title):
        print(f"\n{'='*90}")
        print(f"  {title}")
        print(f"{'='*90}")

    header("Action-Risk ERP (λ=3) — Augmented vs Non-augmented")
    print(f"  {'Model':<{mc}}  {'Augmented N':>11}  {'Aug ERP':>10}  {'Non-aug N':>9}  {'Non-aug ERP':>11}")
    print(f"  {'-'*mc}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*11}")
    for name, (res, _) in results_by_model.items():
        aug = res.get("augmented")
        na  = res.get("non_augmented")
        aug_str = f"{aug['action']['erp']:.4f}" if aug else "—"
        na_str  = f"{na['action']['erp']:.4f}"  if na  else "—"
        aug_n   = aug["n"] if aug else "—"
        na_n    = na["n"]  if na  else "—"
        print(f"  {name:<{mc}}  {aug_n:>11}  {aug_str:>10}  {na_n:>9}  {na_str:>11}")

    header("Action-Risk ERP (λ=3) — Per Source Breakdown")
    all_sources = sorted(set(
        src for res, _ in results_by_model.values()
        for src in res.get("by_source", {}).keys()
    ))
    header_cols = "  ".join(f"{src:>{col}}" for src in all_sources)
    print(f"  {'Model':<{mc}}  {'N':>5}  {header_cols}")
    print(f"  {'-'*mc}  {'-'*5}  " + "  ".join("-"*col for _ in all_sources))
    for name, (res, n) in results_by_model.items():
        cells = []
        for src in all_sources:
            sr = res.get("by_source", {}).get(src)
            cells.append(f"{sr['action']['erp']:.4f} (n={sr['n']})" if sr else "—")
        row = "  ".join(f"{c:>{col}}" for c in cells)
        print(f"  {name:<{mc}}  {n:>5}  {row}")


def _build_model_dict_from_dir(dirpath):
    p = Path(dirpath)
    result = {}
    for f in sorted(p.glob("*.jsonl")):
        if "_failed" not in f.stem:
            result[f.stem] = f
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task1 nonlinear consequence-aware evaluation")
    parser.add_argument("paths", nargs="*",
                        help="JSONL files or directories to evaluate. "
                             "If a directory is given, all *.jsonl inside are evaluated. "
                             "If omitted, uses the hardcoded MODEL_PATHS (output/task1/zeroshot/).")
    args = parser.parse_args()

    if args.paths:
        model_dict = {}
        for p in args.paths:
            path = Path(p)
            if path.is_dir():
                model_dict.update(_build_model_dict_from_dir(path))
            elif path.is_file():
                model_dict[path.stem] = path
            else:
                print(f"[SKIP] Not found: {p}")
    else:
        model_dict = MODEL_PATHS

    results_by_model = {}
    for model_name, path in model_dict.items():
        if not Path(path).exists():
            print(f"[SKIP] {model_name}: file not found at {path}")
            continue
        raw     = load_jsonl(str(path))
        dataset = convert_dataset(raw)
        res     = run_eval(dataset)
        _add_subsets(res, dataset)
        results_by_model[model_name] = (res, len(dataset))

    if results_by_model:
        print_comparison_table(results_by_model)
        print_subset_summary(results_by_model)
