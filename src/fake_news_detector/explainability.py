"""Global and local coefficient-based explanations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def _direct_pipeline_contributions(
    pipeline: Pipeline, text: str
) -> tuple[dict[str, float], int, int]:
    """Calculate feature contributions for a fitted linear pipeline."""
    matrix = pipeline[:-1].transform([text])
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return {}, int(matrix.count_nonzero()), int(matrix.shape[1])
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefficients = np.asarray(classifier.coef_)
    weights = coefficients[0] if coefficients.shape[0] == 1 else coefficients[1]
    row = matrix.tocsr().getrow(0)
    contributions = {
        str(feature_names[index]): float(value * weights[index])
        for index, value in zip(row.indices, row.data, strict=True)
    }
    return contributions, int(row.count_nonzero()), int(matrix.shape[1])


def _all_contributions(model: Any, text: str) -> tuple[dict[str, float], float]:
    """Support direct pipelines and CV-calibrated ensembles."""
    if isinstance(model, Pipeline):
        contributions, nonzero, total = _direct_pipeline_contributions(model, text)
        return contributions, nonzero / max(total, 1)
    if isinstance(model, CalibratedClassifierCV) and hasattr(model, "calibrated_classifiers_"):
        totals: dict[str, list[float]] = defaultdict(list)
        coverage_values = []
        for calibrated in model.calibrated_classifiers_:
            pipeline = calibrated.estimator
            if not isinstance(pipeline, Pipeline):
                continue
            contribution, nonzero, total = _direct_pipeline_contributions(pipeline, text)
            coverage_values.append(nonzero / max(total, 1))
            for feature, value in contribution.items():
                totals[feature].append(value)
        averaged = {feature: float(np.mean(values)) for feature, values in totals.items()}
        return averaged, float(np.mean(coverage_values)) if coverage_values else 0.0
    return {}, 0.0


def explain_prediction(model: Any, text: str, top_n: int = 8) -> dict[str, Any]:
    """Rank word and phrase features that push the linear score toward each class."""
    contributions, sparse_coverage = _all_contributions(model, text)
    word_contributions = {
        feature.removeprefix("word__"): value
        for feature, value in contributions.items()
        if feature.startswith("word__") and len(feature.removeprefix("word__").strip()) > 1
    }
    toward_real = sorted(
        (
            {"feature": feature, "contribution": value}
            for feature, value in word_contributions.items()
            if value > 0
        ),
        key=lambda item: item["contribution"],
        reverse=True,
    )[:top_n]
    toward_fake = sorted(
        (
            {"feature": feature, "contribution": abs(value)}
            for feature, value in word_contributions.items()
            if value < 0
        ),
        key=lambda item: item["contribution"],
        reverse=True,
    )[:top_n]
    return {
        "toward_fake": toward_fake,
        "toward_real": toward_real,
        "vocabulary_coverage": sparse_coverage,
        "method": "local_tfidf_times_linear_coefficient",
        "disclaimer": (
            "Highlighted terms are statistical signals learned from the training dataset. "
            "They do not independently prove that the article is true or false."
        ),
    }


def global_linear_features(model: Any, top_n: int = 20) -> dict[str, list[dict[str, float]]]:
    """Extract the strongest global word coefficients from a direct pipeline."""
    if not isinstance(model, Pipeline):
        return {"fake": [], "real": []}
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return {"fake": [], "real": []}
    names = model.named_steps["tfidf"].get_feature_names_out()
    coefficients = np.asarray(classifier.coef_)[0]
    word_indices = np.array(
        [index for index, name in enumerate(names) if str(name).startswith("word__")]
    )
    if not len(word_indices):
        return {"fake": [], "real": []}
    ranked_real = word_indices[np.argsort(coefficients[word_indices])[-top_n:][::-1]]
    ranked_fake = word_indices[np.argsort(coefficients[word_indices])[:top_n]]

    def rows(indices: np.ndarray) -> list[dict[str, float]]:
        return [
            {
                "feature": str(names[index]).removeprefix("word__"),
                "coefficient": float(coefficients[index]),
            }
            for index in indices
        ]

    return {"fake": rows(ranked_fake), "real": rows(ranked_real)}
