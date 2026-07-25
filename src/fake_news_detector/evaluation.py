"""Model evaluation utilities with probability-aware metrics."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def get_positive_scores(
    model: BaseEstimator, texts: list[str] | Any
) -> tuple[np.ndarray | None, str | None]:
    """Return positive-class scores and explicitly identify their meaning."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        return np.asarray(probabilities)[:, 1], "probability"
    if hasattr(model, "decision_function"):
        decisions = np.asarray(model.decision_function(texts))
        return decisions.ravel(), "decision_score"
    return None, None


def evaluate_predictions(
    y_true: np.ndarray | Any,
    y_pred: np.ndarray | Any,
    positive_scores: np.ndarray | None = None,
    score_kind: str | None = None,
) -> dict[str, Any]:
    """Calculate classification and applicable probability metrics."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Fake News", "Real News"],
            zero_division=0,
            output_dict=True,
        ),
        "score_kind": score_kind,
    }
    if positive_scores is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, positive_scores))
        if score_kind == "probability":
            clipped = np.clip(positive_scores, 1e-7, 1 - 1e-7)
            metrics["log_loss"] = float(log_loss(y_true, clipped))
            metrics["brier_score"] = float(brier_score_loss(y_true, clipped))
            metrics["expected_calibration_error"] = expected_calibration_error(
                np.asarray(y_true), clipped
            )
    return metrics


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Calculate weighted absolute calibration error."""
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (probabilities > lower) & (probabilities <= upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y_true[mask]))
        confidence = float(np.mean(probabilities[mask]))
        error += float(np.sum(mask)) / total * abs(accuracy - confidence)
    return error


def evaluate_model(model: BaseEstimator, texts: Any, labels: Any) -> dict[str, Any]:
    """Evaluate a fitted model and measure average per-record latency."""
    started = time.perf_counter()
    predictions = model.predict(texts)
    elapsed = time.perf_counter() - started
    scores, score_kind = get_positive_scores(model, texts)
    metrics = evaluate_predictions(labels, predictions, scores, score_kind)
    metrics["inference_latency_ms_per_record"] = 1000 * elapsed / max(len(texts), 1)
    return metrics


def cross_validation_summary(
    model: BaseEstimator,
    texts: Any,
    labels: Any,
    *,
    folds: int,
    seed: int,
) -> dict[str, float]:
    """Measure stratified cross-validation macro F1."""
    effective_folds = min(folds, int(np.bincount(np.asarray(labels)).min()))
    if effective_folds < 2:
        return {"cv_macro_f1_mean": math.nan, "cv_macro_f1_std": math.nan}
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, texts, labels, cv=splitter, scoring="f1_macro", n_jobs=-1)
    return {
        "cv_macro_f1_mean": float(scores.mean()),
        "cv_macro_f1_std": float(scores.std()),
    }


def serialized_model_size(model: BaseEstimator, temporary_path: Path) -> int:
    """Measure compressed serialized size without retaining a duplicate artifact."""
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, temporary_path, compress=3)
    size = temporary_path.stat().st_size
    temporary_path.unlink(missing_ok=True)
    return size
