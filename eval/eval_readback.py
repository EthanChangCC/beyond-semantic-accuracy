"""
eval_readback.py
----------------
Evaluate Task 2 readback safety predictions against ground truth.

Usage:
    # single file
    python scripts/eval/eval_readback.py output/task2/zeroshot/gpt-4o-mini.jsonl

    # whole directory  →  prints per-model table
    python scripts/eval/eval_readback.py output/task2/zeroshot/

    # compare multiple dirs  →  prints comparison table
    python scripts/eval/eval_readback.py \
        output/task2/zeroshot/ \
        output/task2/ablation_fewshot/ \
        output/task2/ablation_knowledge/

Metrics:
    1. is_correct   Accuracy
    2. error_type   Macro-F1   (10 classes)
    3. risk_level   Exact Accuracy
    4. risk_level   Danger Downgrade Rate  (GT risk > predicted risk)
    5. risk_level   Weighted Downgrade Severity
    6. CONSTRAINT_HIGH  F1  (key blind-spot metric)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# ── Schema ────────────────────────────────────────────────────────────────────

ERROR_TYPES = [
    "CORRECT", "FILLER_CORRECT", "NUMBER_FORMAT",
    "VALUE_CRITICAL", "OMISSION_CRITICAL", "TARGET_CRITICAL",
    "VALUE_HIGH", "OMISSION_HIGH", "CONSTRAINT_HIGH",
    "MULTI_ERROR",
]

RISK_LEVELS = ["CORRECT", "HIGH", "CRITICAL", "EXTREME"]
RISK_ORDER  = {"CORRECT": 0, "HIGH": 1, "CRITICAL": 2, "EXTREME": 3}

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_field(record, *keys):
    for k in keys:
        if k in record:
            return record[k]
    return None


def extract(records):
    gt_correct, pr_correct = [], []
    gt_etype,   pr_etype   = [], []
    gt_risk,    pr_risk    = [], []
    ids = []

    for r in records:
        gt  = r.get("groundtruth", {})
        pr  = r.get("predict", {})
        ids.append(r.get("meta", {}).get("id", ""))
        gt_correct.append(gt.get("is_correct"))
        pr_correct.append(pr.get("is_correct"))
        gt_etype.append(gt.get("error_type", ""))
        pr_etype.append(pr.get("error_type", ""))
        gt_risk.append(gt.get("risk_level", ""))
        pr_risk.append(pr.get("risk_level", ""))

    return ids, gt_correct, pr_correct, gt_etype, pr_etype, gt_risk, pr_risk


# ── Metrics ───────────────────────────────────────────────────────────────────

def prf1(y_true, y_pred, label):
    tp = sum(yt == label and yp == label for yt, yp in zip(y_true, y_pred))
    fp = sum(yt != label and yp == label for yt, yp in zip(y_true, y_pred))
    fn = sum(yt == label and yp != label for yt, yp in zip(y_true, y_pred))
    p  = tp / (tp + fp) if tp + fp else 0.0
    r  = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*p*r / (p+r) if p+r else 0.0
    return p, r, f1


def macro_f1(y_true, y_pred, labels):
    present = [l for l in labels if any(yt == l for yt in y_true)]
    if not present:
        return 0.0
    return sum(prf1(y_true, y_pred, l)[2] for l in present) / len(present)


def danger_downgrade_rate(gt_risk, pr_risk):
    """Fraction of samples where predicted risk is LOWER than GT risk."""
    errors = [(g, p) for g, p in zip(gt_risk, pr_risk) if g != "CORRECT"]
    if not errors:
        return 0.0
    downgraded = sum(RISK_ORDER.get(p, 0) < RISK_ORDER.get(g, 0) for g, p in errors)
    return downgraded / len(errors)


def weighted_downgrade_severity(gt_risk, pr_risk):
    """
    Average normalized downgrade distance on erroneous samples.

    0.0 = no underestimation
    1.0 = every erroneous sample is downgraded by the maximum possible gap
          (e.g. EXTREME -> CORRECT)
    """
    errors = [(g, p) for g, p in zip(gt_risk, pr_risk) if g != "CORRECT"]
    if not errors:
        return 0.0

    max_gap = max(RISK_ORDER.values()) - min(RISK_ORDER.values())
    severity = sum(
        max(0, RISK_ORDER.get(g, 0) - RISK_ORDER.get(p, 0)) / max_gap
        for g, p in errors
    )
    return severity / len(errors)


# ── Core evaluate ─────────────────────────────────────────────────────────────

def evaluate(records):
    ids, gt_c, pr_c, gt_e, pr_e, gt_r, pr_r = extract(records)
    n = len(records)

    correct_acc = sum(g == p for g, p in zip(gt_c, pr_c)) / n
    et_mf1      = macro_f1(gt_e, pr_e, ERROR_TYPES)
    risk_acc    = sum(g == p for g, p in zip(gt_r, pr_r)) / n
    ddr         = danger_downgrade_rate(gt_r, pr_r)
    wds         = weighted_downgrade_severity(gt_r, pr_r)
    _, _, ch_f1 = prf1(gt_e, pr_e, "CONSTRAINT_HIGH")

    # per error_type F1
    et_detail = {}
    for et in ERROR_TYPES:
        if any(g == et for g in gt_e):
            p, r, f1 = prf1(gt_e, pr_e, et)
            et_detail[et] = {"p": p, "r": r, "f1": f1,
                             "n": sum(g == et for g in gt_e)}

    # risk confusion  (GT row × Pred col)
    risk_conf = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gt_r, pr_r):
        risk_conf[g][p] += 1

    return {
        "n": n,
        "correct_acc": correct_acc,
        "et_mf1": et_mf1,
        "risk_acc": risk_acc,
        "danger_downgrade_rate": ddr,
        "weighted_downgrade_severity": wds,
        "constraint_high_f1": ch_f1,
        "et_detail": et_detail,
        "risk_conf": {k: dict(v) for k, v in risk_conf.items()},
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_single(res, model_name=""):
    tag = f"  Model: {model_name}" if model_name else ""
    print(f"\n{'='*68}")
    print(f"  Readback Safety Evaluation  (N={res['n']}){tag}")
    print(f"{'='*68}")

    print(f"\n  {'Metric':<38}  {'Score':>8}")
    print(f"  {'-'*38}  {'-'*8}")
    print(f"  {'1. is_correct  Accuracy':<38}  {res['correct_acc']:>8.4f}")
    print(f"  {'2. error_type  Macro-F1':<38}  {res['et_mf1']:>8.4f}")
    print(f"  {'3. risk_level  Exact Accuracy':<38}  {res['risk_acc']:>8.4f}")
    print(f"  {'4. Danger Downgrade Rate':<38}  {res['danger_downgrade_rate']:>8.4f}")
    print(f"  {'5. Weighted Downgrade Severity':<38}  {res['weighted_downgrade_severity']:>8.4f}")
    print(f"  {'6. CONSTRAINT_HIGH F1':<38}  {res['constraint_high_f1']:>8.4f}")

    # per error_type
    print(f"\n  Per Error-Type  (sorted by GT count)")
    print(f"  {'error_type':<22}  {'N':>4}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"  {'-'*22}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}")
    for et, d in sorted(res["et_detail"].items(), key=lambda x: -x[1]["n"]):
        print(f"  {et:<22}  {d['n']:>4}  {d['p']:>6.3f}  {d['r']:>6.3f}  {d['f1']:>6.3f}")

    # risk confusion
    print(f"\n  Risk Confusion  (rows=GT, cols=Pred)")
    col_w = 9
    print(f"  {'GT \\ Pred':<10}" + "".join(f"{l:>{col_w}}" for l in RISK_LEVELS))
    print(f"  {'-'*10}" + "-"*col_w*len(RISK_LEVELS))
    for g in RISK_LEVELS:
        row = res["risk_conf"].get(g, {})
        if not any(row.values()):
            continue
        cells = "".join(f"{row.get(p, 0):>{col_w}}" for p in RISK_LEVELS)
        print(f"  {g:<10}{cells}")


def print_comparison(results_map):
    """results_map: { label: {model: res_dict} }"""
    # collect all model names in order
    all_models = []
    for m_dict in results_map.values():
        for m in m_dict:
            if m not in all_models:
                all_models.append(m)

    col = 10
    header_pad = 22

    for label, m_dict in results_map.items():
        print(f"\n{'='*68}")
        print(f"  Condition: {label}")
        print(f"{'='*68}")
        print(f"  {'Metric':<{header_pad}}" + "".join(f"{m[:col]:>{col}}" for m in all_models if m in m_dict))
        print(f"  {'-'*header_pad}" + "-"*col*len(m_dict))

        rows = [
            ("is_correct Acc",    "correct_acc"),
            ("error_type MF1",    "et_mf1"),
            ("risk_level Acc",    "risk_acc"),
            ("Danger Downgrade",  "danger_downgrade_rate"),
            ("Weighted Down Sev", "weighted_downgrade_severity"),
            ("CONSTRAINT_H F1",   "constraint_high_f1"),
        ]
        for row_name, key in rows:
            vals = "".join(
                f"{m_dict[m][key]:>{col}.4f}" if m in m_dict else f"{'—':>{col}}"
                for m in all_models if m in m_dict
            )
            print(f"  {row_name:<{header_pad}}{vals}")


# ── Main ──────────────────────────────────────────────────────────────────────

def eval_path(p):
    """Returns {model_name: result_dict} for a file or directory."""
    p = Path(p)
    if p.is_file():
        recs = load_jsonl(p)
        return {p.stem: evaluate(recs)}
    elif p.is_dir():
        out = {}
        for f in sorted(p.glob("*.jsonl")):
            if "_failed" in f.stem:
                continue
            recs = load_jsonl(f)
            if recs:
                out[f.stem] = evaluate(recs)
        return out
    else:
        print(f"Not found: {p}")
        return {}


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    if len(args) == 1:
        results = eval_path(args[0])
        for model, res in results.items():
            print_single(res, model)
    else:
        # multi-dir comparison
        comparison = {}
        for arg in args:
            p = Path(arg)
            label = p.name
            comparison[label] = eval_path(p)
        print_comparison(comparison)
