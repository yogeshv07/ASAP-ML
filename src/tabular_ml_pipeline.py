"""Generic end-to-end tabular classification pipeline for CSV datasets."""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover - optional dependency
    SMOTE = None
    ImbPipeline = None


warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class PipelineArtifacts:
    best_model_name: str
    best_estimator: Any
    comparison_df: pd.DataFrame
    label_encoder: LabelEncoder | None
    feature_names: list[str]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop highly correlated numeric features using the training split only."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.keep_columns_: list[str] = []
        self.drop_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        X_frame = X.copy()
        numeric_columns = X_frame.select_dtypes(include=[np.number]).columns.tolist()
        self.keep_columns_ = X_frame.columns.tolist()
        if not numeric_columns:
            self.drop_columns_ = []
            return self

        correlation = X_frame[numeric_columns].corr().abs()
        upper_triangle = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        self.drop_columns_ = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.threshold)]
        self.keep_columns_ = [column for column in X_frame.columns if column not in self.drop_columns_]
        return self

    def transform(self, X: pd.DataFrame):
        return X.loc[:, self.keep_columns_].copy()


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers using IQR fences learned from the training split."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: dict[int, float] = {}
        self.upper_bounds_: dict[int, float] = {}

    def fit(self, X, y: pd.Series | None = None):
        array = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(array, 25, axis=0)
        q3 = np.nanpercentile(array, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = {index: q1[index] - self.factor * iqr[index] for index in range(array.shape[1])}
        self.upper_bounds_ = {index: q3[index] + self.factor * iqr[index] for index in range(array.shape[1])}
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float).copy()
        for index in range(array.shape[1]):
            array[:, index] = np.clip(array[:, index], self.lower_bounds_[index], self.upper_bounds_[index])
        return array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full tabular classification pipeline on a CSV dataset.")
    parser.add_argument("--csv-path", type=Path, required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", type=str, required=True, help="Name of the target column.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tabular_pipeline"), help="Output directory.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row sample size for large datasets.")
    parser.add_argument("--missing-threshold", type=float, default=0.40, help="Drop columns above this missing ratio.")
    parser.add_argument("--corr-threshold", type=float, default=0.95, help="Correlation threshold for dropping features.")
    parser.add_argument("--variance-threshold", type=float, default=0.0, help="Low-variance threshold after encoding.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def sample_csv_in_chunks(csv_path: Path, sample_rows: int, random_state: int) -> pd.DataFrame:
    """Reservoir sample a large CSV without loading all rows into memory."""
    rng = np.random.default_rng(random_state)
    reservoir: list[dict[str, Any]] = []
    total_seen = 0

    for chunk in pd.read_csv(csv_path, chunksize=50000, low_memory=False):
        for row in chunk.to_dict(orient="records"):
            total_seen += 1
            if len(reservoir) < sample_rows:
                reservoir.append(row)
                continue
            replace_index = rng.integers(0, total_seen)
            if replace_index < sample_rows:
                reservoir[replace_index] = row

    return pd.DataFrame(reservoir)


def load_dataset(csv_path: Path, sample_rows: int | None, random_state: int) -> pd.DataFrame:
    if sample_rows is not None:
        return sample_csv_in_chunks(csv_path, sample_rows=sample_rows, random_state=random_state)
    return pd.read_csv(csv_path, low_memory=False)


def optimize_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
    optimized = dataframe.copy()
    for column in optimized.columns:
        if optimized[column].dtype == "object":
            try:
                optimized[column] = pd.to_numeric(optimized[column])
            except (TypeError, ValueError):
                pass
            if optimized[column].dtype == "object":
                cardinality = optimized[column].nunique(dropna=True)
                if 0 < cardinality <= max(50, int(0.05 * len(optimized))):
                    optimized[column] = optimized[column].astype("category")
        elif pd.api.types.is_integer_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
        elif pd.api.types.is_float_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], downcast="float")
    return optimized


def clean_dataset(dataframe: pd.DataFrame, target: str, missing_threshold: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = optimize_dtypes(dataframe)
    initial_rows, initial_cols = cleaned.shape

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    dropped_columns = []
    for column in cleaned.columns:
        missing_ratio = cleaned[column].isna().mean()
        if column != target and missing_ratio > missing_threshold:
            dropped_columns.append(column)
    cleaned = cleaned.drop(columns=dropped_columns)

    if target not in cleaned.columns:
        raise KeyError(f"Target column '{target}' was dropped or not found after cleaning.")

    cleaned = cleaned.dropna(subset=[target]).reset_index(drop=True)
    summary = {
        "initial_shape": (initial_rows, initial_cols),
        "cleaned_shape": cleaned.shape,
        "duplicates_removed": initial_rows - dataframe.shape[0] + (dataframe.shape[0] - cleaned.shape[0]),
        "dropped_columns": dropped_columns,
    }
    return cleaned, summary


def split_features_target(dataframe: pd.DataFrame, target: str):
    X = dataframe.drop(columns=[target])
    y = dataframe[target]
    label_encoder = None

    if y.dtype == "object" or str(y.dtype).startswith("category") or not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), name=target)
    else:
        y = y.astype(int) if set(pd.Series(y).dropna().unique()).issubset({0, 1}) else y

    return X, y, label_encoder


def detect_imbalance(y: pd.Series) -> tuple[bool, dict[int, int]]:
    class_counts = y.value_counts().sort_index().to_dict()
    if len(class_counts) <= 1:
        return False, class_counts
    ratio = min(class_counts.values()) / max(class_counts.values())
    return ratio < 0.6, class_counts


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outlier_capper", OutlierCapper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_features, categorical_features


def build_model_specs(
    imbalance_detected: bool,
    class_counts: dict[int, int],
    random_state: int,
) -> dict[str, dict[str, Any]]:
    binary_scale_pos_weight = None
    if imbalance_detected and len(class_counts) == 2:
        counts = sorted(class_counts.items(), key=lambda item: item[0])
        negative = counts[0][1]
        positive = counts[1][1]
        binary_scale_pos_weight = negative / max(positive, 1)

    logistic = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced" if imbalance_detected else None,
        random_state=random_state,
    )
    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced" if imbalance_detected else None,
        n_jobs=1,
        random_state=random_state,
    )
    svm = SVC(
        probability=True,
        class_weight="balanced" if imbalance_detected else None,
        random_state=random_state,
    )
    xgboost = XGBClassifier(
        objective="binary:logistic" if len(class_counts) == 2 else "multi:softprob",
        eval_metric="logloss",
        num_class=None if len(class_counts) == 2 else len(class_counts),
        scale_pos_weight=binary_scale_pos_weight if binary_scale_pos_weight is not None else 1.0,
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
    )

    return {
        "Logistic Regression": {
            "model": logistic,
            "search": "grid",
            "params": {
                "model__C": [0.1, 1.0, 10.0],
            },
        },
        "Random Forest": {
            "model": random_forest,
            "search": "random",
            "params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [None, 8, 12, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "SVM": {
            "model": svm,
            "search": "random",
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
            },
        },
        "XGBoost": {
            "model": xgboost,
            "search": "random",
            "params": {
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        },
    }


def build_estimator_pipeline(
    preprocessor: ColumnTransformer,
    corr_threshold: float,
    variance_threshold: float,
    model: Any,
    use_smote: bool,
    random_state: int,
):
    steps: list[tuple[str, Any]] = [
        ("correlation_filter", CorrelationFilter(threshold=corr_threshold)),
        ("preprocessor", preprocessor),
        ("variance_selector", VarianceThreshold(threshold=variance_threshold)),
    ]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("model", model))
        return ImbPipeline(steps=steps)

    steps.append(("model", model))
    return Pipeline(steps=steps)


def get_scoring_refit(y_train: pd.Series) -> tuple[dict[str, str], str]:
    if y_train.nunique() == 2:
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        return scoring, "roc_auc"

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc_ovr_weighted",
    }
    return scoring, "f1"


def evaluate_model(name: str, estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }

    if probabilities is not None:
        if y_test.nunique() == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, probabilities[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted")
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def extract_feature_names(best_estimator: Any, original_X: pd.DataFrame) -> list[str]:
    correlation_filter = best_estimator.named_steps["correlation_filter"]
    filtered_columns = correlation_filter.keep_columns_
    filtered_X = original_X[filtered_columns]

    preprocessor = best_estimator.named_steps["preprocessor"]
    encoded_names = preprocessor.get_feature_names_out(filtered_X.columns)

    variance_selector = best_estimator.named_steps["variance_selector"]
    support_mask = variance_selector.get_support()
    return [encoded_names[index] for index, keep in enumerate(support_mask) if keep]


def compute_feature_importance(
    best_estimator: Any,
    feature_names: list[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    model = best_estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        importance_values = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    else:
        transformed_pipeline = Pipeline(best_estimator.steps[:-1])
        transformed_X = transformed_pipeline.transform(X_test)
        permutation = permutation_importance(
            model,
            transformed_X,
            y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        importance_values = permutation.importances_mean

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance_values})
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_confusion_matrix(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    label_encoder: LabelEncoder | None,
) -> None:
    predictions = estimator.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = label_encoder.classes_ if label_encoder is not None else None
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=labels,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(importance_df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    top_features = importance_df.head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(top_features) + 2)))
    sns.barplot(data=top_features, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top {min(top_n, len(top_features))} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    corr_threshold: float,
    variance_threshold: float,
    random_state: int,
) -> PipelineArtifacts:
    imbalance_detected, class_counts = detect_imbalance(y_train)
    preprocessor, _, _ = build_preprocessor(X_train)
    scoring, refit_metric = get_scoring_refit(y_train)
    model_specs = build_model_specs(imbalance_detected=imbalance_detected, class_counts=class_counts, random_state=random_state)

    use_smote = imbalance_detected and SMOTE is not None and y_train.nunique() >= 2
    comparison_rows = []
    best_name = ""
    best_estimator = None
    best_score = -math.inf

    for model_name, spec in model_specs.items():
        estimator_pipeline = build_estimator_pipeline(
            preprocessor=preprocessor,
            corr_threshold=corr_threshold,
            variance_threshold=variance_threshold,
            model=spec["model"],
            use_smote=use_smote,
            random_state=random_state,
        )

        if spec["search"] == "grid":
            search = GridSearchCV(
                estimator=estimator_pipeline,
                param_grid=spec["params"],
                scoring=scoring,
                refit=refit_metric,
                cv=5,
                n_jobs=1,
            )
        else:
            search = RandomizedSearchCV(
                estimator=estimator_pipeline,
                param_distributions=spec["params"],
                n_iter=8,
                scoring=scoring,
                refit=refit_metric,
                cv=5,
                n_jobs=1,
                random_state=random_state,
            )

        search.fit(X_train, y_train)
        metrics = evaluate_model(model_name, search.best_estimator_, X_test, y_test)
        metrics["best_params"] = search.best_params_
        metrics["cv_best_score"] = search.best_score_
        comparison_rows.append(metrics)

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = model_name
            best_estimator = search.best_estimator_

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["roc_auc", "f1_score"], ascending=False).reset_index(drop=True)
    feature_names = extract_feature_names(best_estimator, X_train)
    return PipelineArtifacts(
        best_model_name=best_name,
        best_estimator=best_estimator,
        comparison_df=comparison_df,
        label_encoder=None,
        feature_names=feature_names,
    )


def save_outputs(
    artifacts: PipelineArtifacts,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    label_encoder: LabelEncoder | None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "model_comparison.csv"
    best_model_path = output_dir / "best_model.joblib"
    confusion_matrix_path = output_dir / "confusion_matrix.png"
    feature_importance_path = output_dir / "feature_importance.png"
    importance_csv_path = output_dir / "feature_importance.csv"

    artifacts.comparison_df.to_csv(comparison_path, index=False)
    joblib.dump(artifacts.best_estimator, best_model_path)

    importance_df = compute_feature_importance(
        artifacts.best_estimator,
        artifacts.feature_names,
        X_test,
        y_test,
    )
    importance_df.to_csv(importance_csv_path, index=False)

    save_confusion_matrix(
        artifacts.best_estimator,
        X_test,
        y_test,
        output_path=confusion_matrix_path,
        label_encoder=label_encoder,
    )
    save_feature_importance_plot(importance_df, output_path=feature_importance_path, top_n=20)
    return importance_df


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.csv_path, sample_rows=args.sample_rows, random_state=args.random_state)
    cleaned_df, cleaning_summary = clean_dataset(dataframe, target=args.target, missing_threshold=args.missing_threshold)

    X, y, label_encoder = split_features_target(cleaned_df, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

    artifacts = train_and_compare_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        corr_threshold=args.corr_threshold,
        variance_threshold=args.variance_threshold,
        random_state=args.random_state,
    )
    artifacts.label_encoder = label_encoder
    importance_df = save_outputs(artifacts, X_test, y_test, args.output_dir, label_encoder)

    print("Cleaning Summary")
    print("================")
    print(f"Initial shape: {cleaning_summary['initial_shape']}")
    print(f"Cleaned shape: {cleaning_summary['cleaned_shape']}")
    print(f"Dropped columns for missingness: {cleaning_summary['dropped_columns']}")
    print()
    print("Model Comparison")
    print("================")
    print(artifacts.comparison_df.to_string(index=False))
    print()
    print(f"Best model: {artifacts.best_model_name}")
    print(f"Outputs saved to: {args.output_dir.resolve()}")
    print()
    print("Top Features")
    print("============")
    print(importance_df.head(20).to_string(index=False))
    if SMOTE is None:
        print()
        print("Note: imbalanced-learn is not installed, so SMOTE was not used. Class-weight balancing was applied instead.")


if __name__ == "__main__":
    main()
