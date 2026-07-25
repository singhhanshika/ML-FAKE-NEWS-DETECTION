"""Safe model loading and prediction service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from fake_news_detector.data_validation import assess_input
from fake_news_detector.evaluation import get_positive_scores
from fake_news_detector.exceptions import ModelNotReadyError
from fake_news_detector.explainability import explain_prediction

LOGGER = logging.getLogger(__name__)


class FakeNewsPredictor:
    """Load a persisted pipeline and return UI-ready prediction records."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        inference_config: dict[str, Any],
    ) -> None:
        if not model_path.is_file() or not metadata_path.is_file():
            raise ModelNotReadyError(
                "The trained model is not available. Run `python train.py` first."
            )
        try:
            self.model = joblib.load(model_path)
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.exception("Model artifact could not be loaded.")
            raise ModelNotReadyError(
                "The trained model artifact is invalid or incompatible."
            ) from exc
        self.config = inference_config

    def predict(self, text: object) -> dict[str, Any]:
        """Validate and classify one news text without retaining it."""
        assessment = assess_input(
            text,
            max_characters=int(self.config["maximum_characters"]),
            minimum_words=int(self.config["minimum_recommended_words"]),
            ascii_ratio_threshold=float(self.config["supported_language_ascii_ratio"]),
        )
        prediction = int(self.model.predict([assessment.text])[0])
        scores, score_kind = get_positive_scores(self.model, [assessment.text])
        probability_real: float | None = None
        confidence: float | None = None
        if score_kind == "probability" and scores is not None:
            probability_real = float(np.clip(scores[0], 0.0, 1.0))
            confidence = probability_real if prediction == 1 else 1.0 - probability_real
        explanation = explain_prediction(
            self.model,
            assessment.text,
            top_n=int(self.config["explanation_features_per_class"]),
        )
        if explanation["vocabulary_coverage"] < float(self.config["minimum_vocabulary_coverage"]):
            assessment.warnings.append(
                "Very few learned vocabulary features matched this input; "
                "it may be out of distribution."
            )
        if confidence is None:
            band = "Unavailable"
        elif confidence >= 0.85:
            band = "High"
        elif confidence >= 0.65:
            band = "Moderate"
        else:
            band = "Low"
        return {
            "label": prediction,
            "class_name": "Real News" if prediction == 1 else "Fake News",
            "estimated_model_probability": confidence,
            "probability_real": probability_real,
            "confidence_band": band,
            "score_kind": score_kind,
            "word_count": assessment.word_count,
            "character_count": assessment.character_count,
            "warnings": assessment.warnings,
            "explanation": explanation,
            "model_version": self.metadata.get("model_version", "unknown"),
            "disclaimer": (
                "This model identifies statistical language patterns. It does not verify facts, "
                "consult sources, or determine objective truth."
            ),
        }
