"""Tests for persisted inference behavior."""

from pathlib import Path

import pytest

from fake_news_detector.exceptions import InputValidationError, ModelNotReadyError
from fake_news_detector.inference import FakeNewsPredictor


def test_missing_pipeline_raises(tmp_path: Path, inference_config: dict[str, object]) -> None:
    with pytest.raises(ModelNotReadyError):
        FakeNewsPredictor(
            tmp_path / "missing.joblib",
            tmp_path / "missing.json",
            inference_config,
        )


def test_prediction_schema_and_probability(
    model_artifacts: tuple[Path, Path], inference_config: dict[str, object]
) -> None:
    model_path, metadata_path = model_artifacts
    predictor = FakeNewsPredictor(model_path, metadata_path, inference_config)
    result = predictor.predict(
        "official public report confirms the committee approved documented results"
    )
    assert result["label"] in {0, 1}
    assert result["class_name"] in {"Fake News", "Real News"}
    assert 0.0 <= result["estimated_model_probability"] <= 1.0
    assert result["score_kind"] == "probability"
    assert isinstance(result["explanation"]["toward_real"], list)


def test_empty_prediction_rejected(
    model_artifacts: tuple[Path, Path], inference_config: dict[str, object]
) -> None:
    predictor = FakeNewsPredictor(*model_artifacts, inference_config)
    with pytest.raises(InputValidationError):
        predictor.predict(" ")


def test_short_input_warns(
    model_artifacts: tuple[Path, Path], inference_config: dict[str, object]
) -> None:
    predictor = FakeNewsPredictor(*model_artifacts, inference_config)
    result = predictor.predict("rumor")
    assert result["warnings"]
