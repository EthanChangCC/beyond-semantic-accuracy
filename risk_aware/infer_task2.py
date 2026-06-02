"""
Inference script for Task 2 fine-tuned model (LoRA adapter).

Loads the LoRA adapter, reads the Task 2 evaluation set (readback_1000.jsonl),
runs inference, and saves predictions to output_jsonl in the same format as API-based scripts.

Usage:
  python infer_task2.py \
      --adapter_dir   checkpoints/task2-risk-lora \
      --base_model    Qwen/Qwen3-8B \
      --input_jsonl   data-finetune/task2/readback_1000.jsonl \
      --prompt_txt    prompt/prompt_readback_zeroshot.txt \
      --output_jsonl  output/task2/finetune/qwen3-8b-task2-risk-lora.jsonl
"""

import json
import re
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir",  type=str,
                   default="checkpoints/task2-risk-lora")
    p.add_argument("--base_model",   type=str,
                   default="Qwen/Qwen3-8B")
    p.add_argument("--input_jsonl",  type=str,
                   default="data-finetune/task2/readback_1000.jsonl")
    p.add_argument("--prompt_txt",   type=str,
                   default="prompt/prompt_readback_zeroshot.txt")
    p.add_argument("--output_jsonl", type=str,
                   default="output/task2/finetune/qwen3-8b-task2-risk-lora.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--start_index",    type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Extract first valid JSON object from model output."""
    # Strip markdown code fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load system prompt -----------------------------------------------
    with open(args.prompt_txt, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # ---- Load data -----------------------------------------------------------
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    records = records[args.start_index:]
    logger.info(f"Loaded {len(records)} examples (start_index={args.start_index})")

    # ---- Load model + adapter -----------------------------------------------
    logger.info(f"Loading base model from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    logger.info(f"Loading LoRA adapter from {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    # ---- Output setup -------------------------------------------------------
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    # Resume: collect already-done IDs
    done_ids = set()
    if Path(args.output_jsonl).exists():
        with open(args.output_jsonl, "r") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        logger.info(f"Resuming: {len(done_ids)} examples already done")

    out_f = open(args.output_jsonl, "a", encoding="utf-8")

    # ---- Inference (batch) --------------------------------------------------
    batch_size = args.batch_size
    todo = [r for r in records if r.get("id") not in done_ids]
    logger.info(f"Running inference on {len(todo)} examples ...")

    for i in tqdm(range(0, len(todo), batch_size)):
        batch = todo[i : i + batch_size]

        # Build messages for each example
        messages_batch = []
        for rec in batch:
            user_content = json.dumps({
                "controller_utterance": rec["controller_utterance"],
                "pilot_readback":       rec["pilot_readback"],
            }, ensure_ascii=False)
            messages_batch.append([
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_content},
            ])

        # Tokenize
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        for rec, out_ids in zip(batch, outputs):
            new_ids = out_ids[input_len:]
            raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            pred = extract_json(raw_text)

            result = {
                "id":                   rec.get("id"),
                "controller_utterance": rec["controller_utterance"],
                "pilot_readback":       rec["pilot_readback"],
                "raw_output":           raw_text,
                "groundtruth": {
                    "is_correct":    rec["is_correct"],
                    "error_type":    rec["error_type"],
                    "risk_level":    rec["risk_level"],
                    "affected_slot": rec.get("corrupted_slot", ""),
                },
                "predict": pred if pred else {
                    "is_correct":    None,
                    "error_type":    None,
                    "risk_level":    None,
                    "affected_slot": None,
                    "explanation":   raw_text,
                },
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    out_f.close()
    logger.info(f"Done. Results saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
