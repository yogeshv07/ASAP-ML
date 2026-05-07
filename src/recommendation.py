"""Recommendation logic built on top of antibiogram probabilities."""

from __future__ import annotations

import pandas as pd


def decision_from_probability(probability: float) -> str:
    if probability > 0.8:
        return "Avoid ❌"
    if probability >= 0.6:
        return "Caution ⚠️"
    return "Recommended ✅"


def add_recommendations(antibiogram_df: pd.DataFrame) -> pd.DataFrame:
    recommendations = antibiogram_df.copy()
    recommendations["decision"] = recommendations["probability"].apply(decision_from_probability)
    recommendations = recommendations.sort_values("probability", ascending=True).reset_index(drop=True)
    recommendations["rank"] = recommendations.index + 1
    return recommendations[["rank", "antibiotic", "probability", "level", "decision"]]


def top_recommended(recommendation_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    return recommendation_df[recommendation_df["decision"] == "Recommended ✅"].head(top_n).reset_index(drop=True)


def is_multi_drug_resistant(antibiogram_df: pd.DataFrame, threshold: float = 0.8, min_high: int = 3) -> bool:
    return int((antibiogram_df["probability"] > threshold).sum()) >= min_high
