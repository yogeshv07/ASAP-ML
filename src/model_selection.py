"""Train, tune, and select the best model per antibiotic class."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from evaluation import evaluate_model, evaluations_to_frame
from models import get_models, get_param_grids
from preprocessing import encode_features, split_dataset
from utils import MODELS_DIR, RESULTS_DIR, save_csv, save_joblib


def train_class_models(
    df: pd.DataFrame,
    random_state: int = 42,
    preset_train_df: pd.DataFrame | None = None,
    preset_test_df: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    all_results: list[dict[str, Any]] = []
    best_model_map: dict[str, Any] = {}
    best_model_rows: list[dict[str, Any]] = []

    if preset_train_df is not None and preset_test_df is not None:
        train_df = preset_train_df.reset_index(drop=True)
        test_df = preset_test_df.reset_index(drop=True)
    else:
        train_df, test_df = split_dataset(df, test_size=0.2, random_state=random_state)

    split_data = encode_features(train_df, test_df, model_dir=MODELS_DIR)
    train_class_array = split_data.train_df["antibiotic_class"].to_numpy()
    test_class_array = split_data.test_df["antibiotic_class"].to_numpy()
    param_grids = get_param_grids()

    for antibiotic_class in sorted(df["antibiotic_class"].unique()):
        train_idx = np.where(train_class_array == antibiotic_class)[0]
        test_idx = np.where(test_class_array == antibiotic_class)[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        y_train_class = split_data.y_train.iloc[train_idx]
        y_test_class = split_data.y_test.iloc[test_idx]

        if y_train_class.nunique() < 2 or y_test_class.nunique() < 2:
            continue

        candidate_results: list[tuple[str, Any, dict[str, Any]]] = []
        models = get_models(random_state=random_state)

        for model_name, model in models.items():
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                scoring={"recall": "recall", "accuracy": "accuracy"},
                refit="recall",
                cv=5,
                n_jobs=1,
            )
            search.fit(split_data.X_train[train_idx], y_train_class)
            best_estimator = search.best_estimator_
            metrics = evaluate_model(
                best_estimator,
                split_data.X_test[test_idx],
                y_test_class,
                model_name=model_name,
                antibiotic_class=antibiotic_class,
            )
            metrics["best_params"] = str(search.best_params_)
            all_results.append(metrics)
            candidate_results.append((model_name, best_estimator, metrics))

        candidate_results.sort(
            key=lambda item: (item[2]["recall"], item[2]["accuracy"]),
            reverse=True,
        )
        best_model_name, best_estimator, best_metrics = candidate_results[0]
        best_model_map[antibiotic_class] = best_estimator
        save_joblib(best_estimator, MODELS_DIR / f"best_model_{antibiotic_class}.joblib")

        best_model_rows.append(
            {
                "antibiotic_class": antibiotic_class,
                "selected_model": best_model_name,
                "recall": best_metrics["recall"],
                "accuracy": best_metrics["accuracy"],
                "f1_score": best_metrics["f1_score"],
                "roc_auc": best_metrics["roc_auc"],
            }
        )

    evaluation_df = evaluations_to_frame(all_results)
    best_model_df = pd.DataFrame(best_model_rows).sort_values("antibiotic_class").reset_index(drop=True)
    save_csv(evaluation_df, RESULTS_DIR / "evaluation_results.csv")
    save_csv(best_model_df, RESULTS_DIR / "best_model_map.csv")
    save_joblib(list(best_model_map.keys()), MODELS_DIR / "supported_antibiotic_classes.joblib")
    return best_model_map, evaluation_df


def load_best_model_map(model_dir: Path = MODELS_DIR) -> dict[str, Any]:
    supported_classes_path = model_dir / "supported_antibiotic_classes.joblib"
    if not supported_classes_path.exists():
        return {}
    antibiotic_classes = list(joblib.load(supported_classes_path))
    best_model_map: dict[str, Any] = {}
    for antibiotic_class in antibiotic_classes:
        model_path = model_dir / f"best_model_{antibiotic_class}.joblib"
        if model_path.exists():
            best_model_map[antibiotic_class] = joblib.load(model_path)
    return best_model_map
