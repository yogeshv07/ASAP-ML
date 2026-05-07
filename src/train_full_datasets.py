"""Train ASAP-ML using the full external training/testing datasets."""

from __future__ import annotations

import pandas as pd

from model_selection import train_class_models
from run_pipeline import enrich_antibiotic_classes, load_existing_training_sources
from utils import CHARTS_DIR, RESULTS_DIR
from visualization import (
    plot_best_model_frequency,
    plot_class_model_heatmap,
    plot_model_metric_comparison,
    plot_model_recall_summary,
)


def main() -> None:
    train_df, test_df, source_name = load_existing_training_sources()
    if train_df is None:
        raise FileNotFoundError("No external training datasets were found in the data directory.")

    train_df = enrich_antibiotic_classes(train_df)
    if test_df is not None:
        test_df = enrich_antibiotic_classes(test_df)
        dataset = pd.concat([train_df, test_df], ignore_index=True)
        _, evaluation_df = train_class_models(
            dataset,
            random_state=42,
            preset_train_df=train_df,
            preset_test_df=test_df,
        )
    else:
        dataset = train_df
        _, evaluation_df = train_class_models(dataset, random_state=42)

    best_model_df = pd.read_csv(RESULTS_DIR / "best_model_map.csv")

    plot_model_recall_summary(evaluation_df, save_path=CHARTS_DIR / "model_recall_summary.png")
    plot_model_metric_comparison(
        evaluation_df,
        metrics=("accuracy", "precision", "recall", "f1_score"),
        save_path=CHARTS_DIR / "model_metric_comparison.png",
    )
    plot_class_model_heatmap(evaluation_df, metric="recall", save_path=CHARTS_DIR / "model_recall_heatmap.png")
    plot_best_model_frequency(best_model_df, save_path=CHARTS_DIR / "best_model_frequency.png")

    print("ASAP-ML full-dataset training complete")
    print(f"Training source: {source_name}")
    print(f"Rows: {len(dataset)}")
    print(f"Saved evaluation: {RESULTS_DIR / 'evaluation_results.csv'}")
    print(f"Saved model map: {RESULTS_DIR / 'best_model_map.csv'}")
    print(f"Saved charts: {CHARTS_DIR}")
    print()
    print(best_model_df.to_string(index=False))


if __name__ == "__main__":
    main()
