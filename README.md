# Beyond Semantic Accuracy: Consequence-Aware Evaluation for Safety-Critical Language Understanding

This repository contains the benchmark data, evaluation code, and inference prompts for the paper:

> **Beyond Semantic Accuracy: Consequence-Aware Evaluation for Safety-Critical Language Understanding**
> *(Under review)*

---

## Overview

Standard NLP metrics treat all errors equally. In safety-critical domains such as Air Traffic Control (ATC), the same surface-level error can have vastly different operational consequences. This work proposes **consequence-aware evaluation** that explicitly accounts for operational risk when measuring language model performance.

We introduce a dual-task ATC benchmark:

- **Task 1 — Structured Understanding**: Extract utterance-level actions and entity slots from controller–pilot communications. Evaluated with **AR-Geo**, a consequence-weighted metric where missing critical slots (altitude, heading, runway) collapse the score.
- **Task 2 — Readback Safety Judgment**: Classify whether a pilot readback is operationally correct, identify the error type (10 classes), and assign a risk level (CORRECT / HIGH / CRITICAL / EXTREME). Evaluated with **DDR** and **WDS** (dangerous-downgrade metrics).

---

## Repository Structure

```
atc-consequence-eval/
├── README.md
├── requirements.txt
├── schema/
│   ├── task1_schema.json        # action types, entity slots, risk weights
│   └── task2_schema.json        # error types, risk levels, definitions
├── prompts/
│   ├── task1/
│   │   ├── zeroshot.txt
│   │   ├── knowledge.txt
│   │   ├── fewshot.txt
│   │   └── full_aligned.txt
│   └── task2/
│       ├── zeroshot.txt
│       ├── knowledge.txt
│       ├── fewshot.txt
│       ├── fewshot_cot.txt
│       └── full_aligned.txt
├── scripts/
│   └── eval/
│       ├── eval_task1.py        # NER-F1, rNER-F1, AR-Lin, AR-Geo, Strict
│       ├── eval_task1_nonlinear.py
│       └── eval_readback.py     # isCorr, ET-F1, RL-Acc, DDR, WDS, CH-F1
└── data/
    └── README.md                # benchmark data access instructions
```

> **Note**: Inference scripts and fine-tuning code will be added prior to the rebuttal period.

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Task 1 Evaluation

```bash
# Evaluate a single prediction file
python scripts/eval/eval_task1.py output/task1/zeroshot/gpt-4o-mini.jsonl

# Evaluate all files in a directory
python scripts/eval/eval_task1.py output/task1/zeroshot/
```

**Input format** (`predictions.jsonl`): each line is a JSON object with `predict` and `groundtruth` fields matching the Task 1 schema.

### Task 2 Evaluation

```bash
# Evaluate a single prediction file
python scripts/eval/eval_readback.py output/task2/zeroshot/gpt-4o-mini.jsonl

# Compare multiple conditions
python scripts/eval/eval_readback.py \
    output/task2/zeroshot/ \
    output/task2/ablation_fewshot/ \
    output/task2/ablation_full_aligned/
```

---

## Benchmark Data

The benchmark consists of:

- **Task 1**: `hard_final.jsonl` — 500 ATC utterances with expert-reviewed action and entity annotations (8 action types, 14 entity slots). Includes 50 filler-augmented samples.
- **Task 2**: `readback_1000.jsonl` — 1,000 readback evaluation pairs (10 error types × 4 risk levels), balanced across error categories.

See `data/README.md` for access instructions. The full benchmark will be released upon paper acceptance.

---

## Metrics

### Task 1

| Metric | Description |
|---|---|
| NER-F1 | Token-span exact match F1 |
| rNER-F1 | Risk-weighted NER-F1 (altitude/runway/heading weighted ≥ 0.8) |
| ActWF1 | Action-set macro-F1 weighted by action risk |
| AR-Lin | Linear consequence-aware score (weighted slot recall) |
| **AR-Geo** ★ | Geometric consequence-aware score; missing critical slot collapses score |
| Strict | Full utterance exact match (actions + all slots) |

### Task 2

| Metric | Description |
|---|---|
| isCorr | Readback-correctness accuracy |
| ET F1 | Error-type macro-F1 (10 classes) |
| RL Acc | Risk-level exact accuracy |
| DDR ↓ | Danger Downgrade Rate: fraction of predictions below GT risk |
| WDS ↓ | Weighted Downgrade Severity: severity-weighted DDR |
| CH-F1 | CONSTRAINT_HIGH F1 (hardest error type) |

---

## Citation

```bibtex
@article{anonymous2026atc,
  title={Beyond Semantic Accuracy: Consequence-Aware Evaluation for Safety-Critical Language Understanding},
  author={Anonymous},
  year={2026},
  note={Under review}
}
```

*Full citation will be provided upon acceptance.*

---

## License

Data and code are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for research use. The ATC audio corpus is derived from publicly available or licensed sources; see the paper for data provenance details.
