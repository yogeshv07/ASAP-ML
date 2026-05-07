"""Model registry and hyperparameter grids."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        ),
        "svm": Pipeline(
            [
                ("scale", MaxAbsScaler()),
                ("model", SVC(probability=True, random_state=random_state)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scale", MaxAbsScaler()),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("scale", MaxAbsScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "naive_bayes": MultinomialNB(),
    }


def get_param_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "random_forest": {
            "n_estimators": [150, 250],
            "max_depth": [None, 12],
            "min_samples_split": [2, 6],
        },
        "xgboost": {
            "n_estimators": [120, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
        "svm": {
            "model__C": [0.5, 1.0, 2.0],
            "model__kernel": ["linear", "rbf"],
        },
        "knn": {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"],
        },
        "logistic_regression": {
            "model__C": [0.5, 1.0, 2.0],
        },
        "naive_bayes": {
            "alpha": [0.1, 0.5, 1.0],
        },
    }
