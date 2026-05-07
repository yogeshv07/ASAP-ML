"""Antibiogram generation for ASAP-ML."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from preprocessing import transform_input


def probability_level(probability: float) -> str:
    if probability > 0.8:
        return "High"
    if probability >= 0.6:
        return "Moderate"
    return "Low"


def model_probability(model: Any, features) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0][1])
    score = float(model.decision_function(features)[0])
    return 1.0 / (1.0 + math.exp(-score))


def generate_antibiogram(
    sequence: str,
    amr_identifier: str,
    model_map: dict[str, Any],
    seq_vectorizer,
    amr_vectorizer,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    encoded_input = transform_input(sequence, amr_identifier, seq_vectorizer, amr_vectorizer)

    for antibiotic_class, model in sorted(model_map.items()):
        probability = model_probability(model, encoded_input)
        rows.append(
            {
                "antibiotic": antibiotic_class,
                "probability": probability,
                "level": probability_level(probability),
            }
        )

    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)
