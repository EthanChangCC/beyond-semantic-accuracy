"""
Risk weight constants for ATC semantic annotation.
Weights reflect operational consequence severity (ICAO Doc 4444 §12.4.10).
"""

# Per-slot multiplier for text-value tokens (content words that carry slot meaning)
SLOT_TEXT_WEIGHTS: dict[str, float] = {
    # CRITICAL — direct navigation/safety impact
    "altitude":    2.0,
    "heading":     2.0,
    "callsign":    2.0,
    "runway":      2.0,
    # HIGH — route/ground movement
    "waypoint":    1.5,
    "taxiway":     1.5,
    # MODERATE
    "condition":   1.0,
    "frequency":   0.8,
    "speed":       0.8,
    "traffic":     0.8,
    "controller":  0.8,
    "gate":        0.6,
    # NON-SLOT
    "O":           0.4,
}

# Multiplier for action-chunk text and utterance_action name tokens
ACTION_TEXT_WEIGHT: float = 1.2

# Multiplier for label-string tokens (e.g. the string "heading" inside "label": "heading")
SLOT_LABEL_WEIGHTS: dict[str, float] = {
    "altitude":  1.5,
    "heading":   1.5,
    "callsign":  1.5,
    "runway":    1.5,
    "waypoint":  1.2,
    "taxiway":   1.2,
    "condition": 1.0,
    "frequency": 0.8,
    "speed":     0.8,
    "traffic":   0.8,
    "controller":0.8,
    "gate":      0.6,
    "O":         0.4,
}

ACTION_LABEL_WEIGHT: float = 1.2
DEFAULT_WEIGHT: float = 1.0  # JSON structure, utterance echo, unknown tokens


def label_to_text_weight(label: str) -> float:
    """Weight for the *content words* of a span bearing this label."""
    if label == "O":
        return SLOT_TEXT_WEIGHTS["O"]
    # action labels end with "_control" or are bare action names
    for slot, w in SLOT_TEXT_WEIGHTS.items():
        if label == slot or label.startswith(slot + "_"):
            return w
    # fallback: treat as action
    return ACTION_TEXT_WEIGHT


def label_to_label_weight(label: str) -> float:
    """Weight for the *label string token itself* (e.g. the word 'heading' in label field)."""
    if label == "O":
        return SLOT_LABEL_WEIGHTS["O"]
    for slot, w in SLOT_LABEL_WEIGHTS.items():
        if label == slot or label.startswith(slot + "_"):
            return w
    return ACTION_LABEL_WEIGHT
