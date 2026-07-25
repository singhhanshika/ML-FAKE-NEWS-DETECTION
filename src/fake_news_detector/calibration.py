"""Probability calibration comparison."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from fake_news_detector.evaluation import evaluate_model


def fit_calibration_candidates(
    base_model: BaseEstimator,
    train_texts: Any,
    train_labels: Any,
    validation_texts: Any,
    validation_labels: Any,
    methods: list[str],
) -> list[tuple[str, BaseEstimator, dict[str, Any]]]:
    """Fit CV-based calibrators and return validation measurements."""
    results: list[tuple[str, BaseEstimator, dict[str, Any]]] = []
    for method in methods:
        calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=3,
            ensemble=True,
            n_jobs=-1,
        )
        calibrated.fit(train_texts, train_labels)
        metrics = evaluate_model(calibrated, validation_texts, validation_labels)
        results.append((method, calibrated, metrics))
    return results


def calibration_improved(
    uncalibrated: dict[str, Any],
    calibrated: dict[str, Any],
    *,
    f1_tolerance: float,
) -> bool:
    """Choose calibration only when Brier score improves without material F1 loss."""
    baseline_brier = uncalibrated.get("brier_score")
    calibrated_brier = calibrated.get("brier_score")
    if baseline_brier is None or calibrated_brier is None:
        return False
    return (
        calibrated_brier < baseline_brier
        and calibrated["macro_f1"] >= uncalibrated["macro_f1"] - f1_tolerance
    )
