"""CLI entrypoint for generating data and training ASAP-ML."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_generation import generate_dataset
from model_selection import train_class_models
from utils import CHARTS_DIR, DATA_DIR, RESULTS_DIR, ensure_directories
from visualization import (
    plot_best_model_frequency,
    plot_class_model_heatmap,
    plot_model_metric_comparison,
    plot_model_recall_summary,
)


def infer_antibiotic_class_from_amr(amr_identifier: str) -> str:
    normalized = str(amr_identifier).lower()
    prefix_map = {
        "bla": "beta-lactam",
        "ctx": "beta-lactam",
        "oxa": "beta-lactam",
        "tet": "tetracycline",
        "otr": "tetracycline",
        "aac": "aminoglycoside",
        "aad": "aminoglycoside",
        "aph": "aminoglycoside",
        "qnr": "fluoroquinolone",
        "gyr": "fluoroquinolone",
        "par": "fluoroquinolone",
        "erm": "macrolide",
        "mef": "macrolide",
        "mph": "macrolide",
        "sul": "sulfonamide",
        "dfr": "sulfonamide",
        "fol": "sulfonamide",
        "van": "glycopeptide",
    }
    for prefix, antibiotic_class in prefix_map.items():
        if normalized.startswith(prefix):
            return antibiotic_class
    return "unknown"


def enrich_antibiotic_classes(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    unknown_mask = enriched["antibiotic_class"].astype(str).str.lower().eq("unknown")
    enriched.loc[unknown_mask, "antibiotic_class"] = enriched.loc[unknown_mask, "amr_identifier"].apply(
        infer_antibiotic_class_from_amr
    )
    return enriched


def load_existing_training_sources() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    training_synth_path = DATA_DIR / "training_dataset_with_synthetic_negatives.csv"
    testing_synth_path = DATA_DIR / "testing_dataset_with_synthetic_negatives.csv"
    training_path = DATA_DIR / "training_dataset.csv"
    testing_path = DATA_DIR / "testing_dataset.csv"

    if training_synth_path.exists() and testing_synth_path.exists():
        return (
            enrich_antibiotic_classes(pd.read_csv(training_synth_path)),
            enrich_antibiotic_classes(pd.read_csv(testing_synth_path)),
            "training_dataset_with_synthetic_negatives.csv + testing_dataset_with_synthetic_negatives.csv",
        )

    if training_path.exists() and testing_path.exists():
        return (
            enrich_antibiotic_classes(pd.read_csv(training_path)),
            enrich_antibiotic_classes(pd.read_csv(testing_path)),
            "training_dataset.csv + testing_dataset.csv",
        )

    if training_synth_path.exists():
        return enrich_antibiotic_classes(pd.read_csv(training_synth_path)), None, "training_dataset_with_synthetic_negatives.csv"

    if training_path.exists():
        return enrich_antibiotic_classes(pd.read_csv(training_path)), None, "training_dataset.csv"

    return None, None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ASAP-ML data generation and training pipeline.")
    parser.add_argument("--num-sequences", type=int, default=1400, help="Number of FASTA sequences to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    ensure_directories()

    train_df, test_df, source_name = load_existing_training_sources()
    if train_df is not None:
        if test_df is not None:
            dataset = pd.concat([train_df, test_df], ignore_index=True)
            _, evaluation_df = train_class_models(
                dataset,
                random_state=arguments.seed,
                preset_train_df=train_df,
                preset_test_df=test_df,
            )
        else:
            dataset = train_df
            _, evaluation_df = train_class_models(dataset, random_state=arguments.seed)
    else:
        dataset = generate_dataset(
            fasta_path=DATA_DIR / "sequences.fasta",
            csv_path=DATA_DIR / "dataset.csv",
            num_sequences=arguments.num_sequences,
            seed=arguments.seed,
        )
        _, evaluation_df = train_class_models(dataset, random_state=arguments.seed)
        source_name = "generated FASTA dataset"

    best_model_df = pd.read_csv(RESULTS_DIR / "best_model_map.csv")
    plot_model_recall_summary(evaluation_df, save_path=CHARTS_DIR / "model_recall_summary.png")
    plot_model_metric_comparison(
        evaluation_df,
        metrics=("accuracy", "precision", "recall", "f1_score"),
        save_path=CHARTS_DIR / "model_metric_comparison.png",
    )
    plot_class_model_heatmap(evaluation_df, metric="recall", save_path=CHARTS_DIR / "model_recall_heatmap.png")
    plot_best_model_frequency(best_model_df, save_path=CHARTS_DIR / "best_model_frequency.png")

    print(f"Generated dataset with {len(dataset)} rows")
    print(f"Training source: {source_name}")
    print(f"Saved evaluation results to {RESULTS_DIR / 'evaluation_results.csv'}")
    print(f"Saved charts to {CHARTS_DIR}")
    print(best_model_df.to_string(index=False))


if __name__ == "__main__":
    main()
