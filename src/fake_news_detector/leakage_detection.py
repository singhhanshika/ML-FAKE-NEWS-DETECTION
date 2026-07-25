"""Detect suspicious class-specific tokens and dataset artifacts."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

KNOWN_PATTERNS = {
    "reuters_marker": re.compile(r"\breuters\b", re.IGNORECASE),
    "associated_press_marker": re.compile(r"\bassociated press\b", re.IGNORECASE),
    "url": re.compile(r"(?:https?://|www\.)", re.IGNORECASE),
    "copyright": re.compile(r"\bcopyright\b", re.IGNORECASE),
    "date_pattern": re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)" r"\w* \d{1,2}, \d{4}\b",
        re.IGNORECASE,
    ),
}


def analyze_leakage(
    frame: pd.DataFrame,
    *,
    top_n: int = 30,
    minimum_document_frequency: int = 10,
) -> dict[str, Any]:
    """Report tokens and formatting markers strongly associated with one class."""
    texts = frame["content"].fillna("").astype(str)
    labels = frame["label"].to_numpy()
    pattern_rows = []
    for name, pattern in KNOWN_PATTERNS.items():
        present = texts.str.contains(pattern, regex=True)
        counts = Counter(labels[present])
        pattern_rows.append(
            {
                "feature": name,
                "fake_rows": int(counts.get(0, 0)),
                "real_rows": int(counts.get(1, 0)),
                "removed": name in {"reuters_marker", "associated_press_marker", "copyright"},
                "reason": (
                    "Removed as likely source/boilerplate leakage."
                    if name in {"reuters_marker", "associated_press_marker", "copyright"}
                    else "Normalized or retained because it may carry legitimate context."
                ),
            }
        )

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        binary=True,
        min_df=minimum_document_frequency,
        max_features=25000,
        ngram_range=(1, 2),
    )
    suspicious_tokens: list[dict[str, Any]] = []
    try:
        matrix = vectorizer.fit_transform(texts)
        fake_counts = np.asarray(matrix[labels == 0].sum(axis=0)).ravel()
        real_counts = np.asarray(matrix[labels == 1].sum(axis=0)).ravel()
        total = fake_counts + real_counts
        real_share = (real_counts + 1) / (total + 2)
        association = np.abs(real_share - 0.5) * np.log1p(total)
        features = vectorizer.get_feature_names_out()
        for index in np.argsort(association)[::-1][:top_n]:
            suspicious_tokens.append(
                {
                    "feature": str(features[index]),
                    "fake_documents": int(fake_counts[index]),
                    "real_documents": int(real_counts[index]),
                    "real_share_smoothed": round(float(real_share[index]), 4),
                    "review_status": "flagged_for_manual_review",
                }
            )
    except ValueError:
        suspicious_tokens = []

    return {
        "known_artifacts": pattern_rows,
        "suspicious_class_specific_features": suspicious_tokens,
        "mitigation": [
            "Raw file origin is never supplied as a feature.",
            "Subject and date are excluded from the modeling pipeline.",
            "Known agency and copyright markers are removed by the persisted preprocessor.",
            "Normalized duplicates and contradictory duplicate labels are removed "
            "before splitting.",
            "The default split groups identical normalized documents together.",
        ],
    }
