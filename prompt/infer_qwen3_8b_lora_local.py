import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]

BASE_MODEL = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
ADAPTER_PATH = "/root/autodl-tmp/output/qwen3-8b-atc-lora-fast"
MODEL_NAME = "qwen3-8b-atc-lora-fast"

INPUT_JSONL = os.getenv("INPUT_JSONL", f"{ROOT}/data/task1/hard_final.jsonl")
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", f"{ROOT}/output/task1/finetune/{MODEL_NAME}.jsonl")
FAILED_JSONL = os.getenv("FAILED_JSONL", f"{ROOT}/output/task1/finetune/{MODEL_NAME}_failed.jsonl")
SYSTEM_PROMPT_TXT = os.getenv("SYSTEM_PROMPT_TXT", f"{ROOT}/prompt/prompt_operational_zeroshot.txt")

START_INDEX = int(os.getenv("START_INDEX", "0"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "0"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str, mode: str = "w") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_structured_prediction(pred: dict, sentence: str) -> dict:
    normalized = {"utterance": sentence, "utterance_action": [], "tokens": []}
    if not isinstance(pred, dict):
        return normalized

    utterance = pred.get("utterance")
    if isinstance(utterance, str) and utterance.strip():
        normalized["utterance"] = utterance.strip()

    actions = pred.get("utterance_action", [])
    if isinstance(actions, str):
        normalized["utterance_action"] = [actions]
    elif isinstance(actions, list):
        normalized["utterance_action"] = [str(x) for x in actions]
    elif actions:
        normalized["utterance_action"] = [str(actions)]

    tokens = pred.get("tokens", [])
    if isinstance(tokens, list):
        cleaned_tokens = []
        for token in tokens:
            if not isinstance(token, dict):
                continue
            text = token.get("text", "")
            label = token.get("label", "")
            cleaned_tokens.append({"text": str(text).strip(), "label": str(label).strip()})
        normalized["tokens"] = cleaned_tokens

    return normalized


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No JSON object found in model output.")


def build_messages(system_prompt: str, sentence: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"sentence": sentence}, ensure_ascii=False)},
    ]


def generate_one(model, tokenizer, system_prompt: str, sentence: str) -> tuple[dict, str, float]:
    messages = build_messages(system_prompt, sentence)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = round(time.time() - start, 4)

    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    parsed = extract_json_object(raw)
    return parsed, raw, latency


def main():
    out_dir = Path(OUTPUT_JSONL).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(SYSTEM_PROMPT_TXT, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    data = load_jsonl(INPUT_JSONL)

    for idx, item in enumerate(tqdm(data, desc="Local FT Inference")):
        if idx < START_INDEX:
            continue
        if MAX_ITEMS and idx >= START_INDEX + MAX_ITEMS:
            break

        sentence = item.get("sentence", "")
        try:
            pred, raw, latency = generate_one(model, tokenizer, system_prompt, sentence)
            out = {
                "meta": {
                    "index": item.get("index"),
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "augmented": item.get("augmented", False),
                    "latency_sec": latency,
                },
                "sentence": sentence,
                "model": MODEL_NAME,
                "groundtruth": {
                    "utterance": sentence,
                    "utterance_action": item.get("utterance_action", []),
                    "tokens": item.get("tokens", []),
                },
                "predict": normalize_structured_prediction(pred, sentence),
            }
            save_jsonl([out], OUTPUT_JSONL, mode="a")
        except Exception as e:
            failed = {
                "meta": {
                    "index": item.get("index"),
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "augmented": item.get("augmented", False),
                },
                "sentence": sentence,
                "model": MODEL_NAME,
                "error": {"type": "local_infer_error", "detail": repr(e)},
            }
            save_jsonl([failed], FAILED_JSONL, mode="a")


if __name__ == "__main__":
    main()
