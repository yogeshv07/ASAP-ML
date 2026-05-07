"""Train ASAP-ML using only data/dataset.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_selection import train_class_models
from utils import CHARTS_DIR, PROJECT_ROOT, RESULTS_DIR
from visualization import (
    plot_best_model_frequency,
    plot_class_model_heatmap,
    plot_model_metric_comparison,
    plot_model_recall_summary,
)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.csv not found at {dataset_path}")

    dataframe = pd.read_csv(dataset_path)
    _, evaluation_df = train_class_models(dataframe, random_state=42)
    best_model_df = pd.read_csv(RESULTS_DIR / "best_model_map.csv")

    plot_model_recall_summary(evaluation_df, save_path=CHARTS_DIR / "model_recall_summary.png")
    plot_model_metric_comparison(
        evaluation_df,
        metrics=("accuracy", "precision", "recall", "f1_score"),
        save_path=CHARTS_DIR / "model_metric_comparison.png",
    )
    plot_class_model_heatmap(evaluation_df, metric="recall", save_path=CHARTS_DIR / "model_recall_heatmap.png")
    plot_best_model_frequency(best_model_df, save_path=CHARTS_DIR / "best_model_frequency.png")

    print("ASAP-ML dataset.csv-only training complete")
    print(f"Training source: {dataset_path}")
    print(f"Rows: {len(dataframe)}")
    print(f"Saved evaluation: {RESULTS_DIR / 'evaluation_results.csv'}")
    print(f"Saved model map: {RESULTS_DIR / 'best_model_map.csv'}")
    print(f"Saved charts: {CHARTS_DIR}")
    print()
    print(best_model_df.to_string(index=False))


if __name__ == "__main__":
    main()
