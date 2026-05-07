"""Preprocessing helpers for FASTA-derived AMR classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from utils import MODELS_DIR, save_joblib


@dataclass
class SplitData:
    X_train: object
    X_test: object
    y_train: pd.Series
    y_test: pd.Series
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    seq_vectorizer: CountVectorizer
    amr_vectorizer: CountVectorizer


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["resistance_label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def encode_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: Path = MODELS_DIR,
) -> SplitData:
    seq_vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2), lowercase=False)
    amr_vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 1), lowercase=False)

    x_seq_train = seq_vectorizer.fit_transform(train_df["gene_sequence"])
    x_seq_test = seq_vectorizer.transform(test_df["gene_sequence"])

    x_amr_train = amr_vectorizer.fit_transform(train_df["amr_identifier"])
    x_amr_test = amr_vectorizer.transform(test_df["amr_identifier"])

    x_train = hstack([x_seq_train, x_amr_train], format="csr")
    x_test = hstack([x_seq_test, x_amr_test], format="csr")

    y_train = train_df["resistance_label"].astype(int)
    y_test = test_df["resistance_label"].astype(int)

    save_joblib(seq_vectorizer, model_dir / "seq_vectorizer.joblib")
    save_joblib(amr_vectorizer, model_dir / "amr_vectorizer.joblib")

    return SplitData(
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_df=train_df,
        test_df=test_df,
        seq_vectorizer=seq_vectorizer,
        amr_vectorizer=amr_vectorizer,
    )


def transform_input(
    sequence: str,
    amr_identifier: str,
    seq_vectorizer: CountVectorizer,
    amr_vectorizer: CountVectorizer,
):
    x_seq = seq_vectorizer.transform([sequence.upper().strip()])
    x_amr = amr_vectorizer.transform([amr_identifier.strip()])
    return hstack([x_seq, x_amr], format="csr")
