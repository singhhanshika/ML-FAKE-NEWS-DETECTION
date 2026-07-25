"""Exploratory analysis artifacts for the prepared dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _top_ngrams(texts: pd.Series, ngram: int, limit: int = 20) -> list[dict[str, Any]]:
    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(ngram, ngram), max_features=5000, min_df=2
    )
    try:
        matrix = vectorizer.fit_transform(texts.fillna(""))
    except ValueError:
        return []
    totals = matrix.sum(axis=0).A1
    ranked = totals.argsort()[::-1][:limit]
    names = vectorizer.get_feature_names_out()
    return [{"feature": str(names[index]), "count": int(totals[index])} for index in ranked]


def create_eda_artifacts(
    frame: pd.DataFrame, figures_dir: Path, report_path: Path
) -> dict[str, Any]:
    """Generate compact professional EDA plots and a JSON report."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated(subset=["title", "text"]).sum()),
        "class_distribution": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
    }
    analysis = frame.copy()
    analysis["headline_words"] = analysis["title"].fillna("").str.split().str.len()
    analysis["article_words"] = analysis["text"].fillna("").str.split().str.len()
    analysis["characters"] = analysis["content"].fillna("").str.len()
    report["length_statistics"] = (
        analysis[["headline_words", "article_words", "word_count", "characters"]]
        .describe()
        .round(2)
        .to_dict()
    )
    report["ngrams"] = {}
    for label, label_name in ((0, "fake"), (1, "real")):
        texts = analysis.loc[analysis["label"] == label, "content"]
        report["ngrams"][label_name] = {
            "unigrams": _top_ngrams(texts, 1),
            "bigrams": _top_ngrams(texts, 2),
            "trigrams": _top_ngrams(texts, 3),
        }
    if "subject" in analysis.columns:
        report["subject_distribution"] = (
            analysis.groupby(["label", "subject"])
            .size()
            .sort_values(ascending=False)
            .head(50)
            .to_dict()
        )
        report["subject_distribution"] = {
            f"{key[0]}::{key[1]}": int(value)
            for key, value in report["subject_distribution"].items()
        }
    if "date" in analysis.columns:
        parsed = pd.to_datetime(analysis["date"], errors="coerce")
        report["valid_date_fraction"] = round(float(parsed.notna().mean()), 4)
        report["date_range"] = {
            "minimum": parsed.min().isoformat() if parsed.notna().any() else None,
            "maximum": parsed.max().isoformat() if parsed.notna().any() else None,
        }

    colors = ["#d97706", "#2563eb"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = analysis["label"].value_counts().reindex([0, 1], fill_value=0)
    axes[0].bar(["Fake", "Real"], counts, color=colors)
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Articles")
    for index, value in enumerate(counts):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")
    for label, name, color in ((0, "Fake", colors[0]), (1, "Real", colors[1])):
        clipped = analysis.loc[analysis["label"] == label, "word_count"].clip(
            upper=analysis["word_count"].quantile(0.99)
        )
        axes[1].hist(clipped, bins=35, alpha=0.55, label=name, color=color)
    axes[1].set_title("Article word-count distribution (clipped at 99th percentile)")
    axes[1].set_xlabel("Words")
    axes[1].set_ylabel("Articles")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "dataset_overview.png", dpi=160)
    plt.close(fig)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
