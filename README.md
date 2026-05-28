# Beyond Semantic Accuracy: Consequence-Aware Evaluation for Safety-Critical Language Understanding

This repository contains datasets, prompts, fine-tuning scripts, and model weights to reproduce the experiments in our paper.

---

## Repository Structure

```
├── data-finetune/
│   ├── task1/
│   │   ├── train.jsonl             # Task 1 train set (2035 utterances)
│   │   ├── val.jsonl               # Task 1 val set   (200 utterances)
│   │   └── eval.jsonl              # Task 1 test set  (500 utterances,)
│   └── task2/
│       ├── readback_1000.jsonl     # Task 2 test set  (1000 readback pairs, 10×100)
│       ├── train_2853.jsonl        # Task 2 train set (2853 entries, augmented)
│       └── val.jsonl               # Task 2 val set   (90 entries)
├── checkpoints/
│   ├── task1-ce-lora/              # Task 1 LoRA — standard CE loss  (AR-Geo 0.515)
│   ├── task1-risk-lora/            # Task 1 LoRA — Risk-Aware loss   (AR-Geo 0.686)
│   └── task2-risk-lora/            # Task 2 LoRA — Risk-Aware loss   (RL-Acc 0.906)
├── prompt/
│   ├── prompt_label_ft.txt                  # Task 1 — FT system prompt (used for fine-tuning & FT inference)
│   ├── prompt_operational_zeroshot.txt      # Task 1 — Condition A: Zero-Shot
│   ├── prompt_operational_knowledge_pure.txt# Task 1 — Condition C: Knowledge
│   ├── prompt_operational_fewshot_pure.txt  # Task 1 — Condition D: Few-Shot
│   ├── prompt_operational_full_aligned.txt  # Task 1 — Condition B: Full-Aligned
│   ├── prompt_readback_zeroshot.txt         # Task 2 — Condition A: Zero-Shot (also used for FT fine-tuning & FT inference)
│   ├── prompt_readback_knowledge.txt        # Task 2 — Condition C: Knowledge
│   ├── prompt_readback_fewshot.txt          # Task 2 — Condition D: Few-Shot
│   ├── prompt_readback_fewshot_cot.txt      # Task 2 — Condition D+CoT
│   ├── prompt_readback_full_aligned.txt     # Task 2 — Condition B: Full-Aligned
│   └── infer_qwen3_8b_lora_local.py        # Local LoRA inference helper
├── risk_aware/                     # Custom HuggingFace Trainer with Risk-Aware loss
│   ├── slot_weights.py             # Slot → loss weight mapping
│   ├── data_utils.py               # Token-level weight encoding
│   ├── risk_trainer.py             # RiskAwareTrainer + RiskWeightedCollator
│   ├── train_task1.py              # Task 1 training entry point
│   ├── train_task2.py              # Task 2 training entry point
│   ├── infer_task1.py              # Task 1 batch inference (LoRA)
│   └── infer_task2.py              # Task 2 batch inference (LoRA)
└── configs/                        # LLaMA-Factory YAML configs (CE baseline)
```

---

## Tasks

### Task 1 — Consequence-Aware Semantic Annotation

Given an ATC controller utterance, extract:
- `utterance_action`: list of action types (e.g., `altitude_control`, `heading_control`)
- `tokens`: token-level semantic slot labels (e.g., `altitude`, `heading`, `runway`)

**Evaluation metric:** AR-Geo (weighted geometric mean; missing critical slots collapse the score)

### Task 2 — Safety-Aware Readback Verification

Given a controller utterance and a pilot readback, classify:
- `is_correct`: whether the readback is operationally correct
- `error_type`: one of 10 error categories (e.g., `VALUE_CRITICAL`, `OMISSION_HIGH`, `CONSTRAINT_HIGH`)
- `risk_level`: `CORRECT` / `HIGH` / `CRITICAL` / `EXTREME`
- `affected_slot` and `explanation`

**Evaluation metrics:** ET-MF1, RL-Acc, DDR (Danger Downgrade Rate), WDS (Weighted Downgrade Severity)

---

## Dataset Format

### Task 1 (`train.jsonl` / `val.jsonl` / `eval.jsonl`)

```json
{
  "id": "aston2_train_000100",
  "source": "hard_v1",
  "sentence": "ex three twenty nine descent your discretion after landing two two left ...",
  "utterance_action": ["altitude_control", "runway_instruction", "sequence_control"],
  "tokens": [
    {"text": "ex three twenty nine", "label": "callsign"},
    {"text": "descent",             "label": "altitude_control"},
    {"text": "your discretion after landing", "label": "condition"},
    {"text": "two two left",        "label": "runway"}
  ]
}
```

### Task 2 (`readback_1000.jsonl` / `train_2853.jsonl` / `val.jsonl`)

The test and train sets use different formats:

**Test set** (`readback_1000.jsonl`) — raw evaluation format:
```json
{
  "id": "rb1k_0001",
  "source_id": "aston2_train_000100",
  "controller_utterance": "csa seven two seven reduce speed one six zero knots ...",
  "correct_readback":     "reduce speed one six zero knots ...",
  "pilot_readback":       "reduce speed one five zero knots ...",
  "error_type":  "VALUE_HIGH",
  "risk_level":  "HIGH",
  "is_correct":  false,
  "corrupted_slot":  "speed",
  "original_value":  "one six zero knots",
  "corrupted_value": "one five zero knots"
}
```

**Train set** (`train_2853.jsonl`) — chat message format for fine-tuning:
```json
{
  "messages": [
    {"role": "system",    "content": "<system prompt>"},
    {"role": "user",      "content": "{\"controller_utterance\": \"...\", \"pilot_readback\": \"...\"}"},
    {"role": "assistant", "content": "{\"is_correct\": false, \"error_type\": \"VALUE_HIGH\", \"risk_level\": \"HIGH\", \"affected_slot\": \"speed\", \"explanation\": \"...\"}"}
  ]
}
```

---

## Model Weights

All three checkpoints are LoRA adapters for **Qwen3-8B** (r=32, alpha=64).  
Load them with the `peft` library on top of the base model.

| Checkpoint dir | Paper model ID | Task | Loss | AR-Geo | RL-Acc |
|---|---|---|---|---|---|
| `task1-ce-lora` | `qwen3-8b-atc-lora-fast` | Task 1 | Standard CE | 0.515 | — |
| `task1-risk-lora` | `qwen3-8b-atc-task1-risk-lora-v3` | Task 1 | Risk-Aware | **0.686** | — |
| `task2-risk-lora` | `qwen3-8b-task2-risk-lora` | Task 2 | Risk-Aware | — | **0.906** |

### Quick Inference (Task 1)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, json

base_model = "Qwen/Qwen3-8B"
lora_path  = "checkpoints/task1-risk-lora"

tokenizer = AutoTokenizer.from_pretrained(lora_path)
model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

system_prompt = open("prompt/prompt_label_ft.txt").read().strip()
utterance = "csa seven two seven reduce speed one six zero knots till four miles contact tower"

messages = [
    {"role": "system",  "content": system_prompt},
    {"role": "user",    "content": json.dumps({"sentence": utterance})},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(json.loads(response))
```

### Quick Inference (Task 2)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, json

base_model = "Qwen/Qwen3-8B"
lora_path  = "checkpoints/task2-risk-lora"

tokenizer = AutoTokenizer.from_pretrained(lora_path)
model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

system_prompt = open("prompt/prompt_readback_zeroshot.txt").read().strip()
controller = "csa seven two seven reduce speed one six zero knots till four miles"
pilot_rb   = "reduce speed one five zero knots till four miles csa seven two seven"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": json.dumps({
        "controller_utterance": controller,
        "pilot_readback": pilot_rb,
    })},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(json.loads(response))
# → {"is_correct": false, "error_type": "VALUE_HIGH", "risk_level": "HIGH",
#    "affected_slot": "speed", "explanation": "..."}
```

---

## Fine-Tuning

### Requirements

```bash
pip install transformers>=4.51 peft>=0.18 torch>=2.3 accelerate datasets
```

### Risk-Aware Training (Task 1)

```bash
python risk_aware/train_task1.py \
    --base_model  Qwen/Qwen3-8B \
    --train_file  data-finetune/task1/train.jsonl \
    --val_file    data-finetune/task1/val.jsonl \
    --output_dir  output/task1-risk-lora \
    --epochs      3 \
    --max_length  8192
```

Training time: ~89 min on a single NVIDIA A800 80 GB.

### Risk-Aware Training (Task 2)

```bash
python risk_aware/train_task2.py \
    --base_model  Qwen/Qwen3-8B \
    --train_file  data-finetune/task2/train_2853.jsonl \
    --val_file    data-finetune/task2/val.jsonl \
    --output_dir  output/task2-risk-lora \
    --epochs      5
```

Training time: ~17 min on a single NVIDIA A800 80 GB.

### CE Baseline (Task 1, via LLaMA-Factory)

The standard CE baseline (`task1-ce-lora`) was trained with LLaMA-Factory using the config in `configs/task1_lora_fast.yaml`.

---

## Zero-Shot / Few-Shot / Full-Aligned Evaluation

Select the prompt corresponding to each ablation condition and pass it as the system message. All frontier-model experiments used greedy decoding (temperature=0).

| Condition | Task 1 Prompt | Task 2 Prompt |
|---|---|---|
| A — Zero-Shot | `prompt_operational_zeroshot.txt` | `prompt_readback_zeroshot.txt` |
| B — Full-Aligned | `prompt_operational_full_aligned.txt` | `prompt_readback_full_aligned.txt` |
| C — Knowledge | `prompt_operational_knowledge_pure.txt` | `prompt_readback_knowledge.txt` |
| D — Few-Shot | `prompt_operational_fewshot_pure.txt` | `prompt_readback_fewshot.txt` |
| D+CoT | — | `prompt_readback_fewshot_cot.txt` |
| FT inference | `prompt_label_ft.txt` | `prompt_readback_zeroshot.txt` |

> **Note:** Task 2 fine-tuning uses `prompt_readback_zeroshot.txt` as the system prompt for both the training data construction and FT inference — it is identical to Condition A. Task 1 fine-tuning uses the dedicated `prompt_label_ft.txt`.

Input format for all API calls:

```python
# Task 1
user_content = json.dumps({"sentence": "<controller utterance>"})

# Task 2
user_content = json.dumps({
    "controller_utterance": "<controller utterance>",
    "pilot_readback": "<pilot readback>",
})
```

---

## Task 2 Dataset

### Train set (`train_2853.jsonl`)

2853 entries across 10 error types, built from real ATC utterances with systematic corruptions and value-substitution augmentation:

| Error Type | Count | Risk Level |
|---|---|---|
| VALUE_CRITICAL | 403 | CRITICAL |
| TARGET_CRITICAL | 330 | CRITICAL |
| OMISSION_CRITICAL | 300 | CRITICAL |
| CORRECT | 310 | CORRECT |
| FILLER_CORRECT | 310 | CORRECT |
| NUMBER_FORMAT | 310 | CORRECT |
| MULTI_ERROR | 300 | EXTREME |
| VALUE_HIGH | 263 | HIGH |
| CONSTRAINT_HIGH | 177 | HIGH |
| OMISSION_HIGH | 150 | HIGH |

### Test set (`readback_1000.jsonl`)

Balanced evaluation set — 10 error types × 100 entries each.

## Task 2 Test Set — Error Type Distribution

| Error Type | Count | Risk Level |
|---|---|---|
| VALUE_CRITICAL | 100 | CRITICAL |
| OMISSION_CRITICAL | 100 | CRITICAL |
| TARGET_CRITICAL | 100 | CRITICAL |
| VALUE_HIGH | 100 | HIGH |
| OMISSION_HIGH | 100 | HIGH |
| CONSTRAINT_HIGH | 100 | HIGH |
| MULTI_ERROR | 100 | EXTREME |
| FILLER_CORRECT | 100 | CORRECT |
| NUMBER_FORMAT | 100 | CORRECT |
| CORRECT | 100 | CORRECT |

---

## Citation

```bibtex
@inproceedings{anonymous2025beyond,
  title     = {Beyond Semantic Accuracy: Consequence-Aware Evaluation for Safety-Critical Language Understanding},
  booktitle = {Proceedings of EMNLP},
  year      = {2025},
  note      = {Under review}
}
```
