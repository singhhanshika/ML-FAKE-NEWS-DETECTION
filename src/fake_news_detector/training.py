"""End-to-end training orchestration."""

from __future__ import annotations

import copy
import importlib.metadata
import logging
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fake_news_detector.calibration import (
    calibration_improved,
    fit_calibration_candidates,
)
from fake_news_detector.config import ProjectConfig
from fake_news_detector.data_loader import load_raw_dataset
from fake_news_detector.eda import create_eda_artifacts
from fake_news_detector.error_analysis import build_error_analysis
from fake_news_detector.evaluation import (
    cross_validation_summary,
    evaluate_model,
    get_positive_scores,
    serialized_model_size,
)
from fake_news_detector.explainability import global_linear_features
from fake_news_detector.leakage_detection import analyze_leakage
from fake_news_detector.model_factory import build_model_pipeline
from fake_news_detector.splitting import assign_splits
from fake_news_detector.utils import file_sha256, set_random_seed, write_json

LOGGER = logging.getLogger(__name__)


def train_project(config: ProjectConfig) -> dict[str, Any]:
    """Train, compare, select, evaluate, and persist the complete project."""
    values = config.values
    seed = config.random_seed
    set_random_seed(seed)
    fake_path, real_path = config.path("fake_data"), config.path("real_data")
    frame, data_summary = load_raw_dataset(
        fake_path,
        real_path,
        required_columns=list(values["data"]["required_columns"]),
        separator=str(values["data"]["title_body_separator"]),
        minimum_words=int(values["data"]["minimum_training_words"]),
    )
    leakage_report = analyze_leakage(frame)
    frame, split_summary = assign_splits(frame, values, seed=seed)
    processed_path = config.path("processed_data")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed_path, index=False)
    create_eda_artifacts(
        frame,
        config.path("figures"),
        config.root / "reports" / "eda_report.json",
    )

    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    candidates = list(values["models"]["candidates"])
    fitted_models: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    validation_details: dict[str, Any] = {}
    folds = int(values["evaluation"]["cross_validation_folds"])

    for name in candidates:
        LOGGER.info("Training candidate model: %s", name)
        model = build_model_pipeline(name, values, seed)
        started = time.perf_counter()
        model.fit(train["content"], train["label"])
        training_seconds = time.perf_counter() - started
        validation_metrics = evaluate_model(model, validation["content"], validation["label"])
        cv_metrics = cross_validation_summary(
            clone(model),
            train["content"],
            train["label"],
            folds=folds,
            seed=seed,
        )
        size_bytes = serialized_model_size(
            model, config.root / "work" / f"{name}_size_check.joblib"
        )
        fitted_models[name] = model
        validation_details[name] = validation_metrics
        comparison_rows.append(
            {
                "model": name,
                "calibration": "native_or_none",
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_roc_auc": validation_metrics.get("roc_auc"),
                "validation_brier_score": validation_metrics.get("brier_score"),
                **cv_metrics,
                "training_seconds": training_seconds,
                "inference_latency_ms_per_record": validation_metrics[
                    "inference_latency_ms_per_record"
                ],
                "serialized_size_bytes": size_bytes,
                "score_kind": validation_metrics.get("score_kind"),
            }
        )

    selected_name = max(candidates, key=lambda name: validation_details[name]["macro_f1"])
    selected_model = fitted_models[selected_name]
    selected_validation = validation_details[selected_name]
    deployed_calibration = "native_or_none"

    methods = list(values["evaluation"]["calibration_methods"])
    if len(train) < 1000:
        methods = [method for method in methods if method != "isotonic"]
    LOGGER.info("Comparing probability calibration for %s.", selected_name)
    calibration_results = fit_calibration_candidates(
        build_model_pipeline(selected_name, values, seed),
        train["content"],
        train["label"],
        validation["content"],
        validation["label"],
        methods,
    )
    for method, calibrated_model, metrics in calibration_results:
        comparison_rows.append(
            {
                "model": selected_name,
                "calibration": method,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_roc_auc": metrics.get("roc_auc"),
                "validation_brier_score": metrics.get("brier_score"),
                "cv_macro_f1_mean": None,
                "cv_macro_f1_std": None,
                "training_seconds": None,
                "inference_latency_ms_per_record": metrics["inference_latency_ms_per_record"],
                "serialized_size_bytes": serialized_model_size(
                    calibrated_model,
                    config.root / "work" / f"{selected_name}_{method}_size_check.joblib",
                ),
                "score_kind": metrics.get("score_kind"),
            }
        )

    if calibration_results:
        best_method, best_calibrated, best_metrics = min(
            calibration_results,
            key=lambda item: (
                item[2].get("brier_score", float("inf")),
                -item[2]["macro_f1"],
            ),
        )
        native_has_probability = selected_validation.get("brier_score") is not None
        use_calibrated = not native_has_probability or calibration_improved(
            selected_validation,
            best_metrics,
            f1_tolerance=float(values["evaluation"]["calibration_f1_tolerance"]),
        )
        if use_calibrated:
            selected_model = best_calibrated
            selected_validation = best_metrics
            deployed_calibration = best_method

    LOGGER.info(
        "Selected %s with %s calibration using validation macro F1.",
        selected_name,
        deployed_calibration,
    )
    test_metrics = evaluate_model(selected_model, test["content"], test["label"])
    test_errors = build_error_analysis(selected_model, test)
    error_path = config.path("error_analysis")
    error_path.parent.mkdir(parents=True, exist_ok=True)
    test_errors.to_csv(error_path, index=False)

    comparison = pd.DataFrame(comparison_rows).sort_values("validation_macro_f1", ascending=False)
    comparison_path = config.path("model_comparison")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    leakage_report["performance_before_after_mitigation"] = _leakage_benchmark(
        values, seed, train, validation, validation_details
    )
    leakage_report["data_summary"] = data_summary
    write_json(config.path("leakage_report"), leakage_report)

    model_path = config.path("model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, model_path, compress=3)
    metrics = {
        "selected_model": selected_name,
        "deployed_calibration": deployed_calibration,
        "selection_policy": "highest validation macro F1, then calibration quality",
        "validation": selected_validation,
        "test": test_metrics,
        "split_summary": split_summary,
        "model_comparison": comparison.to_dict(orient="records"),
        "global_features": global_linear_features(selected_model),
    }
    write_json(config.path("metrics"), metrics)
    _create_evaluation_figures(selected_model, test, config.path("figures"))

    metadata = {
        "model_name": selected_name,
        "model_version": values["project"]["version"],
        "training_timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset_name": "Kaggle Fake and Real News Dataset",
        "dataset_row_count": int(len(frame)),
        "dataset_fingerprint_sha256": file_sha256([fake_path, real_path]),
        "feature_configuration": values["vectorizer"],
        "preprocessing_configuration": values["preprocessing"],
        "validation_metrics": selected_validation,
        "test_metrics": test_metrics,
        "label_mapping": {"0": "Fake News", "1": "Real News"},
        "python_version": platform.python_version(),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": importlib.metadata.version("joblib"),
        },
        "git_commit": _git_commit(config.root),
        "calibration": deployed_calibration,
    }
    write_json(config.path("metadata"), metadata)
    LOGGER.info("Saved deployable model to %s.", model_path)
    return {"metadata": metadata, "metrics": metrics, "data_summary": data_summary}


def _leakage_benchmark(
    values: dict[str, Any],
    seed: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    validation_details: dict[str, Any],
) -> dict[str, Any]:
    """Compare Logistic Regression with source markers retained and removed."""
    leaky_config = copy.deepcopy(values)
    leaky_config["preprocessing"]["remove_source_markers"] = False
    model = build_model_pipeline("logistic_regression", leaky_config, seed)
    model.fit(train["content"], train["label"])
    before = evaluate_model(model, validation["content"], validation["label"])
    after = validation_details["logistic_regression"]
    return {
        "benchmark_model": "logistic_regression",
        "before_source_marker_removal": {
            "accuracy": before["accuracy"],
            "macro_f1": before["macro_f1"],
        },
        "after_source_marker_removal": {
            "accuracy": after["accuracy"],
            "macro_f1": after["macro_f1"],
        },
        "interpretation": (
            "A large drop after mitigation suggests the unmitigated model relied on "
            "publisher or agency artifacts. Review the suspicious-feature table."
        ),
    }


def _create_evaluation_figures(model: Any, test: pd.DataFrame, output: Path) -> None:
    """Save confusion matrix, ROC, PR, and reliability figures."""
    output.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(test["content"])
    fig, axis = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        test["label"],
        predictions,
        labels=[0, 1],
        display_labels=["Fake", "Real"],
        cmap="Blues",
        ax=axis,
    )
    axis.set_title("Untouched test-set confusion matrix")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    scores, score_kind = get_positive_scores(model, test["content"])
    if scores is None or len(set(test["label"])) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(test["label"], scores, ax=axes[0])
    axes[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(test["label"], scores, ax=axes[1])
    axes[1].set_title("Precision–recall curve")
    fig.tight_layout()
    fig.savefig(output / "discrimination_curves.png", dpi=160)
    plt.close(fig)
    if score_kind == "probability":
        probability_true, probability_predicted = calibration_curve(
            test["label"], scores, n_bins=10, strategy="quantile"
        )
        fig, axis = plt.subplots(figsize=(5.5, 5))
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        axis.plot(
            probability_predicted,
            probability_true,
            marker="o",
            label="Deployed model",
        )
        axis.set(
            title="Reliability diagram",
            xlabel="Mean estimated probability",
            ylabel="Observed fraction of Real News",
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / "calibration_curve.png", dpi=160)
        plt.close(fig)


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
