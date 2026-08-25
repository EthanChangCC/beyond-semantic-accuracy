"""
Risk-aware fine-tuning for Task 2 (ATC readback safety classification).

Two-level weighting:
  Field-level:  risk_level→2.0, error_type→1.5, affected_slot→≥1.5, explanation→0.5
  Sample-level: EXTREME×2.0, CRITICAL×1.5, HIGH×1.0, CORRECT×0.7

Usage (on AutoDL):
  python train_task2.py \
      --model_name_or_path /root/autodl-tmp/models/Qwen3-8B \
      --train_file      /root/autodl-tmp/data/task2/train_2853.jsonl \
      --val_file        /root/autodl-tmp/data/task2/val.jsonl \
      --output_dir      /root/autodl-tmp/output/qwen3-8b-atc-task2-risk-lora \
      --num_train_epochs 5
"""

import os
import json
import logging
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import encode_with_risk_weights_task2
from risk_trainer import RiskAwareTrainer, RiskWeightedCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str,
                   default="/root/autodl-tmp/models/Qwen3-8B")
    p.add_argument("--train_file",  type=str,
                   default="/root/autodl-tmp/data/task2/train_2853.jsonl")
    p.add_argument("--val_file",    type=str,
                   default="/root/autodl-tmp/data/task2/val.jsonl")
    p.add_argument("--output_dir",  type=str,
                   default="/root/autodl-tmp/output/qwen3-8b-atc-task2-risk-lora")
    p.add_argument("--max_length",  type=int,   default=2048)
    p.add_argument("--num_train_epochs",             type=int,   default=5)
    p.add_argument("--per_device_train_batch_size",  type=int,   default=4)
    p.add_argument("--gradient_accumulation_steps",  type=int,   default=4)
    p.add_argument("--learning_rate",                type=float, default=1e-4)
    p.add_argument("--lora_rank",    type=int,   default=32)
    p.add_argument("--lora_alpha",   type=int,   default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--logging_steps", type=int,  default=10)
    p.add_argument("--save_steps",    type=int,  default=100)
    p.add_argument("--eval_steps",    type=int,  default=100)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(
    jsonl_path: str,
    tokenizer,
    max_length: int,
    split_name: str = "train",
) -> Dataset:
    raw = load_jsonl(jsonl_path)
    logger.info(f"[{split_name}] Loaded {len(raw)} examples from {jsonl_path}")

    encoded, skipped = [], 0
    for item in raw:
        messages = item.get("messages", [])
        result = encode_with_risk_weights_task2(messages, tokenizer, max_length)
        if result is None:
            skipped += 1
            continue
        encoded.append(result)

    logger.info(f"[{split_name}] Encoded {len(encoded)}, skipped {skipped}")
    return Dataset.from_list(encoded)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- Tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Datasets ----------------------------------------------------------
    train_dataset = build_dataset(args.train_file, tokenizer, args.max_length, "train")
    eval_dataset  = build_dataset(args.val_file,   tokenizer, args.max_length, "val")

    # ---- Model + LoRA ------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training arguments ------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    collator = RiskWeightedCollator(tokenizer=tokenizer)

    trainer = RiskAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    logger.info("Starting risk-aware fine-tuning (Task 2)...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
