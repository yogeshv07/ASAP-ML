from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
FRONTEND_DIST_DIR = PROJECT_ROOT / "app" / "frontend" / "dist"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from antibiogram import generate_antibiogram
from data_generation import generate_dataset
from model_selection import load_best_model_map, train_class_models
from recommendation import add_recommendations, is_multi_drug_resistant, top_recommended
from utils import CHARTS_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR, ensure_directories, load_joblib
from visualization import (
    plot_antibiogram,
    plot_best_model_frequency,
    plot_class_model_heatmap,
    plot_model_metric_comparison,
    plot_model_recall_summary,
)


class TrainRequest(BaseModel):
    num_sequences: int = Field(default=1400, ge=1200, le=10000)
    seed: int = Field(default=42, ge=0, le=100000)
    regenerate_dataset: bool = True


class PredictRequest(BaseModel):
    gene_sequence: str
    amr_identifier: str = Field(min_length=1)


def load_vectorizers():
    seq_vectorizer_path = MODELS_DIR / "seq_vectorizer.joblib"
    amr_vectorizer_path = MODELS_DIR / "amr_vectorizer.joblib"
    if not (seq_vectorizer_path.exists() and amr_vectorizer_path.exists()):
        return None, None
    return load_joblib(seq_vectorizer_path), load_joblib(amr_vectorizer_path)


def validate_sequence(sequence: str) -> str:
    cleaned = "".join(sequence.upper().split())
    if not cleaned:
        raise ValueError("Please provide a sequence.")
    if not cleaned.isalpha():
        raise ValueError("Sequence must contain alphabetic biological sequence characters only.")
    if len(cleaned) < 50:
        raise ValueError("Sequence must be at least 50 characters long.")
    return cleaned


def image_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def get_dataset_frame(num_sequences: int, seed: int, regenerate_dataset: bool) -> pd.DataFrame:
    fasta_path = DATA_DIR / "sequences.fasta"
    csv_path = DATA_DIR / "dataset.csv"
    if regenerate_dataset or not csv_path.exists() or not fasta_path.exists():
        return generate_dataset(
            fasta_path=fasta_path,
            csv_path=csv_path,
            num_sequences=num_sequences,
            seed=seed,
        )
    return pd.read_csv(csv_path)


def load_existing_training_sources() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    training_synth_path = DATA_DIR / "training_dataset_with_synthetic_negatives.csv"
    testing_synth_path = DATA_DIR / "testing_dataset_with_synthetic_negatives.csv"
    training_path = DATA_DIR / "training_dataset.csv"
    testing_path = DATA_DIR / "testing_dataset.csv"

    if training_synth_path.exists() and testing_synth_path.exists():
        return (
            pd.read_csv(training_synth_path),
            pd.read_csv(testing_synth_path),
            "training_dataset_with_synthetic_negatives.csv + testing_dataset_with_synthetic_negatives.csv",
        )

    if training_path.exists() and testing_path.exists():
        return (
            pd.read_csv(training_path),
            pd.read_csv(testing_path),
            "training_dataset.csv + testing_dataset.csv",
        )

    if training_synth_path.exists():
        return pd.read_csv(training_synth_path), None, "training_dataset_with_synthetic_negatives.csv"

    if training_path.exists():
        return pd.read_csv(training_path), None, "training_dataset.csv"

    return None, None, None


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
    if "antibiotic_class" not in enriched.columns:
        enriched["antibiotic_class"] = enriched["amr_identifier"].apply(infer_antibiotic_class_from_amr)
        return enriched

    unknown_mask = enriched["antibiotic_class"].astype(str).str.lower().eq("unknown")
    enriched.loc[unknown_mask, "antibiotic_class"] = enriched.loc[unknown_mask, "amr_identifier"].apply(
        infer_antibiotic_class_from_amr
    )
    return enriched


def run_training(num_sequences: int, seed: int, regenerate_dataset: bool) -> dict:
    train_df, test_df, source_name = load_existing_training_sources()
    if train_df is not None:
        train_df = enrich_antibiotic_classes(train_df)
        if test_df is not None:
            test_df = enrich_antibiotic_classes(test_df)
            dataset = pd.concat([train_df, test_df], ignore_index=True)
            _, evaluation_df = train_class_models(
                dataset,
                random_state=seed,
                preset_train_df=train_df,
                preset_test_df=test_df,
            )
        else:
            dataset = train_df
            _, evaluation_df = train_class_models(dataset, random_state=seed)
    else:
        dataset = get_dataset_frame(num_sequences=num_sequences, seed=seed, regenerate_dataset=regenerate_dataset)
        _, evaluation_df = train_class_models(dataset, random_state=seed)
        source_name = "generated FASTA dataset"

    best_model_df = pd.read_csv(RESULTS_DIR / "best_model_map.csv")
    recall_fig = plot_model_recall_summary(evaluation_df, save_path=CHARTS_DIR / "model_recall_summary.png")
    comparison_fig = plot_model_metric_comparison(
        evaluation_df,
        metrics=("accuracy", "precision", "recall", "f1_score"),
        save_path=CHARTS_DIR / "model_metric_comparison.png",
    )
    heatmap_fig = plot_class_model_heatmap(
        evaluation_df,
        metric="recall",
        save_path=CHARTS_DIR / "model_recall_heatmap.png",
    )
    best_model_fig = plot_best_model_frequency(
        best_model_df,
        save_path=CHARTS_DIR / "best_model_frequency.png",
    )

    class_distribution = (
        dataset["antibiotic_class"].value_counts().sort_index().rename_axis("antibiotic_class").reset_index(name="count")
    )
    label_distribution = (
        dataset["resistance_label"].value_counts().sort_index().rename_axis("resistance_label").reset_index(name="count")
    )

    return {
        "message": "Training completed successfully on the selected full dataset.",
        "training_source": source_name,
        "dataset_rows": int(len(dataset)),
        "class_distribution": class_distribution.to_dict(orient="records"),
        "label_distribution": label_distribution.to_dict(orient="records"),
        "evaluation": evaluation_df.to_dict(orient="records"),
        "best_models": best_model_df.to_dict(orient="records"),
        "recall_chart_base64": image_to_base64(recall_fig),
        "metric_comparison_chart_base64": image_to_base64(comparison_fig),
        "recall_heatmap_chart_base64": image_to_base64(heatmap_fig),
        "best_model_frequency_chart_base64": image_to_base64(best_model_fig),
    }


def run_prediction(sequence: str, amr_identifier: str) -> dict:
    model_map = load_best_model_map()
    seq_vectorizer, amr_vectorizer = load_vectorizers()
    if not model_map or seq_vectorizer is None or amr_vectorizer is None:
        raise RuntimeError("Models are not available yet. Train the models first.")

    antibiogram_df = generate_antibiogram(sequence, amr_identifier, model_map, seq_vectorizer, amr_vectorizer)
    recommendation_df = add_recommendations(antibiogram_df)
    chart_fig = plot_antibiogram(antibiogram_df, save_path=CHARTS_DIR / "latest_antibiogram.png")

    return {
        "antibiogram": antibiogram_df.to_dict(orient="records"),
        "recommendations": recommendation_df.to_dict(orient="records"),
        "top_recommended": top_recommended(recommendation_df, top_n=3).to_dict(orient="records"),
        "mdr_warning": is_multi_drug_resistant(antibiogram_df),
        "chart_base64": image_to_base64(chart_fig),
    }


ensure_directories()
app = FastAPI(title="ASAP-ML API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    dataset_path = DATA_DIR / "dataset.csv"
    return {
        "status": "ok",
        "dataset_exists": dataset_path.exists(),
        "models_available": bool(load_best_model_map()),
    }


@app.get("/api/about")
def about() -> dict:
    return {
        "title": "ASAP-ML",
        "summary": (
            "ASAP-ML replicates a FASTA-first AMR workflow inspired by Antibiotic Susceptibility and "
            "Antibiogram Prediction and extends it with a recommendation engine."
        ),
        "workflow": [
            "Generate FASTA-style sequences",
            "Convert FASTA to CSV",
            "Encode gene_sequence using 2-gram CountVectorizer",
            "Encode amr_identifier using 1-gram CountVectorizer",
            "Train Random Forest, XGBoost, SVM, KNN, Logistic Regression, and Naive Bayes",
            "Select the best model per antibiotic class using recall, then accuracy",
            "Generate antibiograms and ranked antibiotic recommendations",
        ],
    }


@app.post("/api/train")
def train_models(payload: TrainRequest) -> dict:
    try:
        return run_training(
            num_sequences=payload.num_sequences,
            seed=payload.seed,
            regenerate_dataset=payload.regenerate_dataset,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict:
    try:
        sequence = validate_sequence(payload.gene_sequence)
        return run_prediction(sequence=sequence, amr_identifier=payload.amr_identifier.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="assets")

    @app.get("/")
    def serve_frontend() -> FileResponse:
        return FileResponse(FRONTEND_DIST_DIR / "index.html")

    @app.get("/{path:path}")
    def serve_frontend_routes(path: str) -> FileResponse:
        candidate = FRONTEND_DIST_DIR / path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
