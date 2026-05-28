"""
RiskAwareTrainer: replaces standard cross-entropy with per-token risk-weighted CE.

  loss = Σ_i  w_i · CE(logit_i, label_i)  /  Σ_i  w_i
         (sum over non-masked positions only)

Where w_i is the risk weight for token i (high-risk slot values → higher weight).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union, Any
from transformers import Trainer, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
import numpy as np


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class RiskWeightedCollator(DataCollatorMixin):
    """
    Pads input_ids, attention_mask, labels, and risk_weights to the same length.
    Drops the risk_weights key only if it is absent in all examples.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    return_tensors: str = "pt"

    def torch_call(self, features: list[dict]) -> dict[str, torch.Tensor]:
        has_weights = "risk_weights" in features[0]

        if has_weights:
            raw_weights = [f.pop("risk_weights") for f in features]

        # Manual padding — avoids tokenizer.pad() mishandling labels/-100
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id  = self.tokenizer.pad_token_id or 0

        input_ids_pad, attn_mask_pad, labels_pad = [], [], []
        for f in features:
            plen = max_len - len(f["input_ids"])
            input_ids_pad.append(f["input_ids"]      + [pad_id] * plen)
            attn_mask_pad.append(f["attention_mask"]  + [0]      * plen)
            labels_pad.append(   f["labels"]          + [-100]   * plen)

        batch = {
            "input_ids":      torch.tensor(input_ids_pad, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask_pad, dtype=torch.long),
            "labels":         torch.tensor(labels_pad,    dtype=torch.long),
        }

        if has_weights:
            padded = [w + [1.0] * (max_len - len(w)) for w in raw_weights]
            batch["risk_weights"] = torch.tensor(padded, dtype=torch.float32)
            for f, w in zip(features, raw_weights):
                f["risk_weights"] = w

        return batch


# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------

class RiskAwareTrainer(Trainer):
    """
    Trainer that computes risk-weighted cross-entropy loss.
    Falls back to standard CE if risk_weights is not in the batch.
    """

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        risk_weights: Optional[torch.Tensor] = inputs.pop("risk_weights", None)
        labels: torch.Tensor = inputs["labels"]

        outputs = model(**inputs)

        if risk_weights is None:
            loss = outputs.loss
        else:
            logits = outputs.logits  # (B, T, V)

            # Causal LM: shift by one
            shift_logits = logits[:, :-1, :].contiguous()     # (B, T-1, V)
            shift_labels = labels[:, 1:].contiguous()          # (B, T-1)
            shift_weights = risk_weights[:, 1:].contiguous()   # (B, T-1)

            # Flat views
            B, T_1, V = shift_logits.shape
            flat_logits  = shift_logits.view(-1, V)
            flat_labels  = shift_labels.view(-1)
            flat_weights = shift_weights.view(-1)

            # Per-token CE (no reduction)
            ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            per_token_loss = ce(flat_logits, flat_labels)  # (B*T-1,)

            # Mask padded / prefix positions (label == -100)
            mask = (flat_labels != -100).float()

            # Weighted mean: Σ w_i * loss_i / Σ w_i
            weighted = per_token_loss * mask * flat_weights.to(per_token_loss.device)
            denom = (mask * flat_weights.to(mask.device)).sum().clamp(min=1e-8)
            loss = weighted.sum() / denom

        return (loss, outputs) if return_outputs else loss
