"""Train an XGBoost model on the first 10,000 rows of training_dataset_with_synthetic_negatives.csv."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, ensure_directories


INPUT_CSV = DATA_DIR / "training_dataset_with_synthetic_negatives.csv"
ROW_LIMIT = 10_000
MODEL_PATH = MODELS_DIR / "xgboost_first_10000.joblib"
SEQ_VECTORIZER_PATH = MODELS_DIR / "xgboost_first_10000_seq_vectorizer.joblib"
AMR_VECTORIZER_PATH = MODELS_DIR / "xgboost_first_10000_amr_vectorizer.joblib"
CLASS_ENCODER_PATH = MODELS_DIR / "xgboost_first_10000_class_encoder.joblib"
RESULTS_PATH = RESULTS_DIR / "xgboost_first_10000_metrics.csv"


def load_first_rows(csv_path: Path, row_limit: int = ROW_LIMIT) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path, nrows=row_limit, low_memory=False)
    expected_columns = {"gene_sequence", "antibiotic_class", "amr_identifier", "resistance_label"}
    missing_columns = expected_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.drop_duplicates().dropna(subset=list(expected_columns)).reset_index(drop=True)
    dataframe["gene_sequence"] = dataframe["gene_sequence"].astype(str).str.upper().str.strip()
    dataframe["antibiotic_class"] = dataframe["antibiotic_class"].astype(str).str.strip()
    dataframe["amr_identifier"] = dataframe["amr_identifier"].astype(str).str.strip()
    dataframe["resistance_label"] = dataframe["resistance_label"].astype(int)
    return dataframe


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    seq_vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2), lowercase=False)
    amr_vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 1), lowercase=False)
    class_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    x_seq_train = seq_vectorizer.fit_transform(train_df["gene_sequence"])
    x_seq_test = seq_vectorizer.transform(test_df["gene_sequence"])

    x_amr_train = amr_vectorizer.fit_transform(train_df["amr_identifier"])
    x_amr_test = amr_vectorizer.transform(test_df["amr_identifier"])

    x_class_train = class_encoder.fit_transform(train_df[["antibiotic_class"]])
    x_class_test = class_encoder.transform(test_df[["antibiotic_class"]])

    x_train = hstack([x_seq_train, x_amr_train, x_class_train], format="csr")
    x_test = hstack([x_seq_test, x_amr_test, x_class_test], format="csr")

    return x_train, x_test, seq_vectorizer, amr_vectorizer, class_encoder


def train_xgboost(x_train, y_train):
    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test) -> dict[str, float | str]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }


def main() -> None:
    ensure_directories()
    dataframe = load_first_rows(INPUT_CSV, row_limit=ROW_LIMIT)
    train_df, test_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42,
        stratify=dataframe["resistance_label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    x_train, x_test, seq_vectorizer, amr_vectorizer, class_encoder = build_features(train_df, test_df)
    y_train = train_df["resistance_label"]
    y_test = test_df["resistance_label"]

    model = train_xgboost(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(seq_vectorizer, SEQ_VECTORIZER_PATH)
    joblib.dump(amr_vectorizer, AMR_VECTORIZER_PATH)
    joblib.dump(class_encoder, CLASS_ENCODER_PATH)

    metrics_df = pd.DataFrame(
        [
            {
                "rows_used": len(dataframe),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
            }
        ]
    )
    metrics_df.to_csv(RESULTS_PATH, index=False)

    print("XGBoost first-10000 training complete")
    print(f"Input file: {INPUT_CSV}")
    print(f"Rows used: {len(dataframe)}")
    print(metrics_df.to_string(index=False))
    print()
    print("Classification report")
    print(metrics["classification_report"])
    print()
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
