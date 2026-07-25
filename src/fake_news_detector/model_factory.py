"""Candidate model construction."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from fake_news_detector.feature_engineering import build_feature_pipeline


def build_classifier(name: str, config: dict[str, Any], seed: int) -> Any:
    """Create one configured classifier."""
    model_config = config["models"]
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent", random_state=seed)
    if name == "naive_bayes":
        return MultinomialNB(alpha=float(model_config["naive_bayes"]["alpha"]))
    if name == "logistic_regression":
        settings = model_config["logistic_regression"]
        return LogisticRegression(
            C=float(settings["C"]),
            max_iter=int(settings["max_iter"]),
            class_weight="balanced",
            random_state=seed,
            solver="liblinear",
        )
    if name == "passive_aggressive":
        settings = model_config["passive_aggressive"]
        return PassiveAggressiveClassifier(
            C=float(settings["C"]),
            max_iter=int(settings["max_iter"]),
            class_weight="balanced",
            average=True,
            random_state=seed,
            early_stopping=False,
        )
    if name == "linear_svm":
        settings = model_config["linear_svm"]
        return LinearSVC(
            C=float(settings["C"]),
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"Unknown model candidate: {name}")


def build_model_pipeline(name: str, config: dict[str, Any], seed: int) -> Pipeline:
    """Create a complete preprocessing, feature, and model pipeline."""
    features = build_feature_pipeline(config)
    return Pipeline(
        [
            ("clean", clone(features.named_steps["clean"])),
            ("tfidf", clone(features.named_steps["tfidf"])),
            ("classifier", build_classifier(name, config, seed)),
        ]
    )
