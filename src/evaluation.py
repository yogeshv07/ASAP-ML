"""Evaluation utilities for ASAP-ML."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model: Any, x_test, y_test, model_name: str, antibiotic_class: str) -> dict[str, Any]:
    predictions = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(x_test)
        probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        probabilities = predictions

    return {
        "antibiotic_class": antibiotic_class,
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def evaluations_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "antibiotic_class",
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "best_params",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    dataframe = pd.DataFrame(rows)
    return dataframe[columns].sort_values(
        by=["antibiotic_class", "recall", "accuracy"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
