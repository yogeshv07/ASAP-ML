"""Terminal prediction entrypoint for ASAP-ML."""

from __future__ import annotations

import argparse

from antibiogram import generate_antibiogram
from model_selection import load_best_model_map
from recommendation import add_recommendations, is_multi_drug_resistant, top_recommended
from utils import MODELS_DIR, load_joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASAP-ML prediction in the terminal.")
    parser.add_argument("--sequence", required=True, help="Biological sequence to score.")
    parser.add_argument("--amr-id", required=True, help="AMR identifier, for example bla_7.")
    return parser.parse_args()


def validate_sequence(sequence: str) -> str:
    cleaned = "".join(sequence.upper().split())
    if not cleaned:
        raise ValueError("Sequence must not be empty.")
    if not cleaned.isalpha():
        raise ValueError("Sequence must contain alphabetic biological sequence characters only.")
    if len(cleaned) < 50:
        raise ValueError("Sequence must be at least 50 characters long.")
    return cleaned


def main() -> None:
    args = parse_args()
    sequence = validate_sequence(args.sequence)
    amr_identifier = args.amr_id.strip()

    seq_vectorizer = load_joblib(MODELS_DIR / "seq_vectorizer.joblib")
    amr_vectorizer = load_joblib(MODELS_DIR / "amr_vectorizer.joblib")
    model_map = load_best_model_map()
    if not model_map:
        raise RuntimeError("No trained models found. Run training first with python src/run_pipeline.py")

    antibiogram_df = generate_antibiogram(
        sequence=sequence,
        amr_identifier=amr_identifier,
        model_map=model_map,
        seq_vectorizer=seq_vectorizer,
        amr_vectorizer=amr_vectorizer,
    )
    recommendation_df = add_recommendations(antibiogram_df)
    top3_df = top_recommended(recommendation_df, top_n=3)
    mdr_flag = is_multi_drug_resistant(antibiogram_df)

    print("ASAP-ML Prediction")
    print("==================")
    print(f"AMR identifier: {amr_identifier}")
    print(f"Sequence length: {len(sequence)}")
    print()
    print("Antibiogram")
    print(antibiogram_df.to_string(index=False))
    print()
    print("Recommendations")
    print(recommendation_df.to_string(index=False))
    print()
    print("Top 3 Recommended")
    if top3_df.empty:
        print("No recommendations fell into the Recommended band.")
    else:
        print(top3_df.to_string(index=False))
    print()
    print(f"MDR Warning: {'Yes' if mdr_flag else 'No'}")


if __name__ == "__main__":
    main()
