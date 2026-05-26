# Benchmark Data

The full benchmark will be released upon paper acceptance.

## Files

| File | Task | Size | Description |
|---|---|---|---|
| `task1/hard_final.jsonl` | Task 1 | 500 | ATC utterances with expert-reviewed action and entity annotations |
| `task2/readback_1000.jsonl` | Task 2 | 1,000 | Readback evaluation pairs (10 error types × 4 risk levels) |

## Task 1 Data Format

Each line is a JSON object:

```json
{
  "id": "aston2_train_000100",
  "source": "hard_v1 | hard_v2 | agreed_v1 | synthetic",
  "augmented": false,
  "sentence": "SIA123 climb to flight level 350",
  "utterance_action": ["altitude_control"],
  "tokens": [
    {"text": "SIA123", "label": "callsign"},
    {"text": "climb to flight level 350", "label": "altitude"}
  ]
}
```

The `augmented` flag marks 50 samples (10%) where filler words (uh, um, er, ah, ehm) are inserted after critical value tokens to test robustness.

## Task 2 Data Format

Each line is a JSON object:

```json
{
  "meta": {"id": "rb_0001", "error_type": "VALUE_CRITICAL", "risk_level": "CRITICAL"},
  "input": {
    "clearance": "SIA123 climb to flight level 350",
    "readback":  "SIA123 climb to flight level 250"
  },
  "groundtruth": {
    "is_correct": false,
    "error_type": "VALUE_CRITICAL",
    "risk_level": "CRITICAL",
    "affected_slot": "altitude",
    "explanation": "Pilot read back FL250 instead of FL350."
  }
}
```

## Accessing the Data

During the review period, the benchmark data is available in the supplementary materials submitted with the paper.

The public release (post-acceptance) will include all data files, model predictions, and fine-tuning training sets.
