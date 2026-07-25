"""Tests for coefficient-based explanations."""

from sklearn.pipeline import Pipeline

from fake_news_detector.explainability import (
    explain_prediction,
    global_linear_features,
)


def test_local_explanation_schema(fitted_linear_pipeline: Pipeline) -> None:
    result = explain_prediction(
        fitted_linear_pipeline,
        "official report confirms a shocking rumor",
        top_n=3,
    )
    assert set(result) == {
        "toward_fake",
        "toward_real",
        "vocabulary_coverage",
        "method",
        "disclaimer",
    }
    assert len(result["toward_fake"]) <= 3
    assert len(result["toward_real"]) <= 3
    assert all(item["contribution"] >= 0 for item in result["toward_fake"])


def test_global_features_available(fitted_linear_pipeline: Pipeline) -> None:
    result = global_linear_features(fitted_linear_pipeline, top_n=3)
    assert len(result["fake"]) == 3
    assert len(result["real"]) == 3
