"""Structured misclassification analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fake_news_detector.evaluation import get_positive_scores
from fake_news_detector.explainability import explain_prediction


def build_error_analysis(
    model: Any,
    frame: pd.DataFrame,
    *,
    top_n_features: int = 5,
) -> pd.DataFrame:
    """Return false-positive and false-negative records with local explanations."""
    texts = frame["content"]
    labels = frame["label"].to_numpy()
    predictions = model.predict(texts)
    scores, score_kind = get_positive_scores(model, texts)
    rows = []
    for position, (index, record) in enumerate(frame.iterrows()):
        prediction = int(predictions[position])
        truth = int(labels[position])
        if prediction == truth:
            continue
        explanation = explain_prediction(model, record["content"], top_n=top_n_features)
        confidence = None
        if scores is not None and score_kind == "probability":
            probability_real = float(scores[position])
            confidence = probability_real if prediction == 1 else 1 - probability_real
        rows.append(
            {
                "row_index": int(index),
                "headline": str(record.get("title", ""))[:500],
                "true_label": truth,
                "predicted_label": prediction,
                "confidence": confidence,
                "score_kind": score_kind,
                "toward_fake_features": ", ".join(
                    item["feature"] for item in explanation["toward_fake"]
                ),
                "toward_real_features": ", ".join(
                    item["feature"] for item in explanation["toward_real"]
                ),
                "word_count": int(record.get("word_count", 0)),
                "possible_reason": _possible_reason(record, confidence),
            }
        )
    return pd.DataFrame(rows)


def _possible_reason(record: pd.Series, confidence: float | None) -> str:
    word_count = int(record.get("word_count", 0))
    if word_count < 20:
        return "Very short input provides limited evidence."
    if confidence is not None and confidence >= 0.85:
        return "High-confidence error; review source artifacts, duplication, and temporal drift."
    if '"' in str(record.get("content", "")):
        return "Quoted or mixed-voice language may resemble the opposite class."
    return "Vocabulary, topic, satire/opinion, or source style may differ from training patterns."
