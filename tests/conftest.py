"""Synthetic fixtures shared by unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from fake_news_detector.preprocessing import CleanTextTransformer


@pytest.fixture()
def fitted_linear_pipeline() -> Pipeline:
    """Fit a tiny deterministic pipeline without production data."""
    texts = [
        "official council report confirms the approved public budget",
        "agency documents show the committee published verified results",
        "shocking secret plot exposed click now unbelievable conspiracy",
        "miracle rumor they do not want you to know viral hoax",
        "government audit provides evidence and named public records",
        "anonymous source reveals unbelievable hidden scheme share now",
    ]
    labels = [1, 1, 0, 0, 1, 0]
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, norm="l2"),
            )
        ]
    )
    pipeline = Pipeline(
        [
            ("clean", CleanTextTransformer()),
            ("tfidf", features),
            (
                "classifier",
                LogisticRegression(random_state=42, solver="liblinear"),
            ),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


@pytest.fixture()
def model_artifacts(tmp_path: Path, fitted_linear_pipeline: Pipeline) -> tuple[Path, Path]:
    """Persist a synthetic model and matching metadata."""
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(fitted_linear_pipeline, model_path)
    metadata_path.write_text(
        json.dumps({"model_version": "test", "label_mapping": {"0": "Fake", "1": "Real"}}),
        encoding="utf-8",
    )
    return model_path, metadata_path


@pytest.fixture()
def inference_config() -> dict[str, object]:
    """Return small deterministic inference thresholds."""
    return {
        "maximum_characters": 500,
        "minimum_recommended_words": 5,
        "minimum_vocabulary_coverage": 0.0001,
        "supported_language_ascii_ratio": 0.55,
        "explanation_features_per_class": 4,
    }
