"""Shared project paths and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
CHARTS_DIR = OUTPUTS_DIR / "charts"
RESULTS_DIR = OUTPUTS_DIR / "results"


def ensure_directories() -> None:
    for directory in (DATA_DIR, OUTPUTS_DIR, MODELS_DIR, CHARTS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def save_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
