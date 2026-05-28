"""
Build per-token risk weights for Task 1 and Task 2 training data.

Task 1 strategy (ATC semantic annotation):
  - text-value chars → weight by slot criticality (heading/altitude → 2.0, O → 0.4)
  - label-value chars → weight by slot criticality
  - utterance_action value chars → ACTION_TEXT_WEIGHT

Task 2 strategy (readback safety classification):
  - risk_level value → 2.0  (the safety verdict)
  - error_type value → 1.5
  - affected_slot value → max(field_weight=1.5, slot_weight)
  - is_correct value → 1.0
  - explanation chars → 0.5  (narrative, less safety-critical)
  - sample-level multiplier: EXTREME×2.0, CRITICAL×1.5, HIGH×1.0, CORRECT×0.7

All tasks:
  3. Tokenize via apply_chat_template + offset_mapping.
  4. Map char weights → token weights using offset_mapping.
  5. Return input_ids / attention_mask / labels / risk_weights.
"""

from __future__ import annotations
import json
import re
import numpy as np
from typing import Optional

from slot_weights import (
    label_to_text_weight,
    label_to_label_weight,
    ACTION_TEXT_WEIGHT,
    DEFAULT_WEIGHT,
)


# ---------------------------------------------------------------------------
# Character-level weight construction
# ---------------------------------------------------------------------------

def _find_next(pattern: str, text: str, pos: int) -> Optional[re.Match]:
    return re.search(pattern, text[pos:])


def build_char_weights(assistant_content: str) -> np.ndarray:
    """
    Return float32 array of length len(assistant_content) with per-char weights.
    Chars that correspond to high-risk slot values get higher weights.
    """
    weights = np.full(len(assistant_content), DEFAULT_WEIGHT, dtype=np.float32)

    try:
        parsed = json.loads(assistant_content)
    except json.JSONDecodeError:
        return weights

    # ---- 1. utterance_action values ----------------------------------------
    ua_section_end = 0
    m = re.search(r'"utterance_action"\s*:\s*\[', assistant_content)
    if m:
        ua_section_end = m.end()
        # Find closing ']' for this array (simple scan, no nesting expected)
        bracket_end = assistant_content.find(']', ua_section_end)
        if bracket_end != -1:
            ua_section_end = bracket_end + 1
            ua_section = assistant_content[m.end():bracket_end]
            for action_match in re.finditer(r'"([^"]+)"', ua_section):
                abs_start = m.end() + action_match.start(1)
                abs_end   = m.end() + action_match.end(1)
                weights[abs_start:abs_end] = ACTION_TEXT_WEIGHT

    # ---- 2. tokens array: text values and label values ----------------------
    tokens_list = parsed.get("tokens", [])
    # Start searching after utterance_action section
    pos = ua_section_end

    for tok in tokens_list:
        text_val  = tok.get("text", "")
        label_val = tok.get("label", "O")
        tw = label_to_text_weight(label_val)
        lw = label_to_label_weight(label_val)

        # --- text value --------------------------------------------------
        if text_val:
            pattern = r'"text"\s*:\s*"' + re.escape(text_val) + r'"'
            m = _find_next(pattern, assistant_content, pos)
            if m:
                full_match = m.group(0)
                # prefix length: everything before the actual value
                prefix_m = re.match(r'"text"\s*:\s*"', full_match)
                prefix_len = prefix_m.end() if prefix_m else 0
                abs_start = pos + m.start() + prefix_len
                abs_end   = abs_start + len(text_val)
                weights[abs_start:abs_end] = tw
                # advance pos to just after text value; label is nearby
                text_abs_end = pos + m.end()
                pos = pos + m.start() + 1  # keep pos before this object for label search

        # --- label value -------------------------------------------------
        if label_val:
            # Search for "label": "LABEL" near the current position
            near_pos = max(0, pos - 5)
            search_window = min(near_pos + 300, len(assistant_content))
            pattern = r'"label"\s*:\s*"' + re.escape(label_val) + r'"'
            m = re.search(pattern, assistant_content[near_pos:search_window])
            if m:
                full_match = m.group(0)
                prefix_m = re.match(r'"label"\s*:\s*"', full_match)
                prefix_len = prefix_m.end() if prefix_m else 0
                abs_start = near_pos + m.start() + prefix_len
                abs_end   = abs_start + len(label_val)
                weights[abs_start:abs_end] = lw
                # advance past this token object
                pos = near_pos + m.end()

    return weights


# ---------------------------------------------------------------------------
# Character → token weight mapping
# ---------------------------------------------------------------------------

def char_to_token_weights(
    char_weights: np.ndarray,
    offset_mapping: list[tuple[int, int]],
    response_char_offset: int,
) -> list[float]:
    """
    Map char_weights (over assistant_content) to per-token weights using
    the offset_mapping from the full-conversation tokenization.

    offset_mapping[i] = (start, end) char positions in the full formatted text.
    response_char_offset = start position of assistant_content in full text.
    """
    n = len(char_weights)
    token_weights = []
    for start, end in offset_mapping:
        # Convert to position within assistant_content
        rel_start = start - response_char_offset
        rel_end   = end   - response_char_offset
        if rel_end <= 0 or rel_start >= n:
            # Outside assistant response → will be masked by label=-100, weight irrelevant
            token_weights.append(DEFAULT_WEIGHT)
        elif rel_start < 0:
            token_weights.append(DEFAULT_WEIGHT)
        else:
            rel_start = max(0, rel_start)
            rel_end   = min(n, rel_end)
            if rel_start >= rel_end:
                token_weights.append(DEFAULT_WEIGHT)
            else:
                # Use max weight among all chars covered by this token
                token_weights.append(float(np.max(char_weights[rel_start:rel_end])))
    return token_weights


# ---------------------------------------------------------------------------
# Full example encoding
# ---------------------------------------------------------------------------

def _find_assistant_start(input_ids: list[int], im_start_id: int,
                           assistant_ids: list[int]) -> int:
    """
    Find the token position right after <|im_start|>assistant\n in input_ids.
    Returns len(input_ids) (all-prefix) if not found.
    """
    pat_len = 1 + len(assistant_ids)  # im_start + "assistant[\n]" tokens
    for i in range(len(input_ids) - pat_len, -1, -1):
        if input_ids[i] == im_start_id:
            if input_ids[i + 1 : i + pat_len] == assistant_ids:
                return i + pat_len
    return len(input_ids)


def encode_with_risk_weights(
    messages: list[dict],
    tokenizer,
    max_length: int = 2048,
) -> Optional[dict]:
    """
    Tokenize a conversation with LEFT truncation so the assistant response
    is always preserved even when the system prompt is very long.

    labels[i] = -100 for non-assistant tokens, else token_id.
    risk_weights[i] = per-token risk multiplier.
    """
    if not messages or messages[-1]["role"] != "assistant":
        return None

    assistant_content = messages[-1]["content"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Left-truncate: keep the LAST max_length tokens so the assistant
    # response (at the end) is never cut off even with long system prompts.
    tokenizer.truncation_side = "left"
    enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    tokenizer.truncation_side = "right"  # restore default

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offset_mapping = enc["offset_mapping"]

    # Locate assistant start by finding <|im_start|>assistant\n in token ids
    im_start_id     = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_ids   = tokenizer.encode("assistant\n", add_special_tokens=False)
    prefix_len      = _find_assistant_start(input_ids, im_start_id, assistant_ids)

    labels = [-100] * prefix_len + input_ids[prefix_len:]

    if all(l == -100 for l in labels):
        return None

    # Character offset of assistant_content in full_text (search from right)
    response_char_offset = full_text.rfind(assistant_content)
    if response_char_offset == -1:
        risk_weights = [DEFAULT_WEIGHT] * len(input_ids)
    else:
        char_weights = build_char_weights(assistant_content)
        risk_weights = char_to_token_weights(
            char_weights, offset_mapping, response_char_offset
        )

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "risk_weights":   risk_weights,
    }


# ===========================================================================
# Task 2 — Readback Safety Classification
# ===========================================================================

# Field-level weights: how important is it to predict each field correctly?
_TASK2_FIELD_WEIGHTS: dict[str, float] = {
    "risk_level":    2.0,   # safety verdict — top priority
    "error_type":    1.5,   # error taxonomy
    "affected_slot": 1.5,   # further boosted by slot criticality below
    "is_correct":    1.0,
    "explanation":   0.5,   # narrative text, less safety-critical
}

# Sample-level multipliers by ground-truth risk_level
_SAMPLE_WEIGHTS: dict[str, float] = {
    "EXTREME":  2.0,
    "CRITICAL": 1.5,
    "HIGH":     1.0,
    "CORRECT":  0.7,   # slight downweight; model must still learn negatives
}

from slot_weights import SLOT_TEXT_WEIGHTS as _SLOT_W


def _slot_weight_for_name(slot_name) -> float:
    """Return slot text weight for an affected_slot value like 'altitude'."""
    if not slot_name:
        return 1.0
    for key, w in _SLOT_W.items():
        if slot_name == key or slot_name.startswith(key):
            return w
    return 1.0


def build_char_weights_task2(assistant_content: str) -> tuple[np.ndarray, float]:
    """
    Return (char_weights, sample_weight) for a Task 2 assistant response.

    char_weights: float32 array, shape=(len(assistant_content),)
    sample_weight: scalar multiplier derived from ground-truth risk_level
    """
    weights = np.full(len(assistant_content), DEFAULT_WEIGHT, dtype=np.float32)

    try:
        parsed = json.loads(assistant_content)
    except json.JSONDecodeError:
        return weights, 1.0

    risk_level  = parsed.get("risk_level", "CORRECT")
    sample_w    = _SAMPLE_WEIGHTS.get(risk_level, 1.0)
    slot_name   = parsed.get("affected_slot", "")
    affected_w  = max(_TASK2_FIELD_WEIGHTS["affected_slot"],
                      _slot_weight_for_name(slot_name))

    # Actual per-field weights (override affected_slot with combined weight)
    field_weights = dict(_TASK2_FIELD_WEIGHTS)
    field_weights["affected_slot"] = affected_w

    # For each scalar field, find  "field_name": "value"  or  "field_name": true/false
    for field, fw in field_weights.items():
        if field == "explanation":
            # Find the explanation value span and apply uniform low weight
            m = re.search(r'"explanation"\s*:\s*"', assistant_content)
            if m:
                val_start = m.end()
                # Find the closing quote (not escaped)
                close = re.search(r'(?<!\\)"', assistant_content[val_start:])
                if close:
                    val_end = val_start + close.start()
                    weights[val_start:val_end] = fw
        else:
            val = parsed.get(field)
            if val is None:
                continue
            val_str = json.dumps(val) if not isinstance(val, str) else val
            # Match: "field_name": "VALUE"  or  "field_name": true/false
            if isinstance(val, str):
                pattern = (r'"' + re.escape(field) + r'"\s*:\s*"'
                           + re.escape(val_str) + r'"')
            else:
                pattern = (r'"' + re.escape(field) + r'"\s*:\s*'
                           + re.escape(val_str))
            m = re.search(pattern, assistant_content)
            if m:
                full = m.group(0)
                # Value starts after the colon + optional quote
                colon_pos = full.index(':') + 1
                # Skip whitespace and optional opening quote
                offset = colon_pos
                while offset < len(full) and full[offset] in (' ', '"'):
                    offset += 1
                abs_val_start = m.start() + offset
                abs_val_end   = abs_val_start + len(val_str)
                weights[abs_val_start:abs_val_end] = fw

    # Apply sample-level multiplier to all assistant token weights
    weights *= sample_w

    return weights, sample_w


def encode_with_risk_weights_task2(
    messages: list[dict],
    tokenizer,
    max_length: int = 2048,
) -> Optional[dict]:
    """
    Encode a Task 2 example with per-token risk weights.
    Same interface as encode_with_risk_weights; uses Task 2 weighting scheme.
    """
    if not messages or messages[-1]["role"] != "assistant":
        return None

    assistant_content = messages[-1]["content"]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
    )

    enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offset_mapping = enc["offset_mapping"]

    prefix_enc = tokenizer(prefix_text, add_special_tokens=False)
    prefix_len = len(prefix_enc["input_ids"])

    labels = [-100] * min(prefix_len, len(input_ids)) + input_ids[prefix_len:]
    labels = labels[:len(input_ids)]

    if all(l == -100 for l in labels):
        return None

    response_char_offset = full_text.rfind(assistant_content)
    if response_char_offset == -1:
        risk_weights = [DEFAULT_WEIGHT] * len(input_ids)
    else:
        char_weights, _ = build_char_weights_task2(assistant_content)
        risk_weights = char_to_token_weights(
            char_weights, offset_mapping, response_char_offset
        )

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "risk_weights":   risk_weights,
    }
