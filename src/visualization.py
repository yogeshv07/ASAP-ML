"""Visualization helpers for ASAP-ML outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bar_color(probability: float) -> str:
    if probability > 0.8:
        return "#C0392B"
    if probability >= 0.6:
        return "#F1C40F"
    return "#27AE60"


def plot_antibiogram(dataframe: pd.DataFrame, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = dataframe["probability"].apply(_bar_color).tolist()
    ax.bar(dataframe["antibiotic"], dataframe["probability"], color=colors)
    ax.axhline(0.6, linestyle="--", color="#F39C12", linewidth=1.5, label="0.6 threshold")
    ax.axhline(0.8, linestyle="--", color="#C0392B", linewidth=1.5, label="0.8 threshold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Resistance probability")
    ax.set_title("Antibiogram probability profile")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_recall_summary(evaluation_df: pd.DataFrame, save_path: Path | None = None):
    summary = (
        evaluation_df.groupby("model_name", as_index=False)["recall"]
        .mean()
        .sort_values("recall", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary["model_name"], summary["recall"], color="#2874A6")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean recall")
    ax.set_title("Average recall across candidate models")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_metric_comparison(
    evaluation_df: pd.DataFrame,
    metrics: Iterable[str] = ("accuracy", "precision", "recall", "f1_score", "roc_auc"),
    save_path: Path | None = None,
):
    metric_names = [metric for metric in metrics if metric in evaluation_df.columns]
    summary = evaluation_df.groupby("model_name", as_index=False)[metric_names].mean()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x_positions = np.arange(len(summary["model_name"]))
    width = 0.14 if metric_names else 0.2
    palette = ["#0D7A6D", "#2874A6", "#F39C12", "#C0392B", "#7D3C98"]

    for index, metric in enumerate(metric_names):
        offset = (index - (len(metric_names) - 1) / 2) * width
        ax.bar(
            x_positions + offset,
            summary[metric],
            width=width,
            label=metric.replace("_", " ").title(),
            color=palette[index % len(palette)],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary["model_name"], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean score")
    ax.set_title("Average model performance across evaluation metrics")
    ax.legend(ncol=min(3, len(metric_names)))
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_model_heatmap(
    evaluation_df: pd.DataFrame,
    metric: str = "recall",
    save_path: Path | None = None,
):
    pivot = evaluation_df.pivot_table(
        index="antibiotic_class",
        columns="model_name",
        values=metric,
        aggfunc="mean",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.7 * len(pivot.index) + 2)))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{metric.replace('_', ' ').title()} by antibiotic class and model")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iat[row, col]
            if pd.notna(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.55 else "#163240",
                    fontsize=9,
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(metric.replace("_", " ").title())
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_best_model_frequency(best_model_df: pd.DataFrame, save_path: Path | None = None):
    summary = (
        best_model_df["selected_model"]
        .value_counts()
        .rename_axis("model_name")
        .reset_index(name="count")
        .sort_values(["count", "model_name"], ascending=[False, True])
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary["model_name"], summary["count"], color="#34495E")
    ax.set_ylabel("Classes selected")
    ax.set_title("How often each model was selected as class winner")
    ax.tick_params(axis="x", rotation=25)

    for idx, count in enumerate(summary["count"]):
        ax.text(idx, count + 0.03, str(int(count)), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
