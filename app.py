"""Streamlit application for leakage-aware fake news pattern classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fake_news_detector.config import load_config  # noqa: E402
from fake_news_detector.exceptions import (  # noqa: E402
    FakeNewsDetectorError,
    InputValidationError,
)
from fake_news_detector.inference import FakeNewsPredictor  # noqa: E402
from fake_news_detector.logging_config import configure_logging  # noqa: E402

configure_logging()
CONFIG = load_config()

st.set_page_config(
    page_title="Fake News Pattern Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_predictor(model_mtime: float, metadata_mtime: float) -> FakeNewsPredictor:
    """Cache model loading while invalidating when artifacts change."""
    del model_mtime, metadata_mtime
    return FakeNewsPredictor(
        CONFIG.path("model"),
        CONFIG.path("metadata"),
        CONFIG.section("inference"),
    )


@st.cache_data(show_spinner=False)
def load_json(path: str, modified: float) -> dict[str, Any]:
    """Read generated JSON artifacts with modification-aware caching."""
    del modified
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_if_available(path: Path) -> dict[str, Any] | None:
    """Return an artifact or None when training has not generated it."""
    if not path.is_file():
        return None
    return load_json(str(path), path.stat().st_mtime)


def probability_text(value: float | None) -> str:
    """Format an honest probability label."""
    return "Unavailable" if value is None else f"{value:.1%}"


def contribution_frame(items: list[dict[str, Any]], direction: str) -> pd.DataFrame:
    """Prepare contribution rows for display."""
    return pd.DataFrame(
        [
            {
                "Word or phrase": item["feature"],
                "Contribution strength": round(float(item["contribution"]), 4),
                "Direction": direction,
            }
            for item in items
        ]
    )


def prediction_tab(predictor: FakeNewsPredictor | None) -> None:
    """Render the main prediction workflow."""
    st.subheader("Analyze language patterns")
    st.caption(
        "Enter English news text. The model recognizes dataset-derived writing patterns; "
        "it does not browse the web or verify a claim."
    )
    mode = st.radio(
        "Input mode",
        ["Headline", "Full article", "Headline and article"],
        horizontal=True,
    )
    if mode == "Headline and article":
        headline = st.text_input("Headline", key="headline", placeholder="Paste the news headline")
        article = st.text_area(
            "Article",
            key="article",
            height=220,
            placeholder="Paste the article body",
            max_chars=int(CONFIG.section("inference")["maximum_characters"]),
        )
        submitted_text = f"{headline} bodytext {article}".strip()
    else:
        height = 130 if mode == "Headline" else 300
        submitted_text = st.text_area(
            mode,
            key="single_text",
            height=height,
            placeholder=(
                "Paste a headline" if mode == "Headline" else "Paste the full news article"
            ),
            max_chars=int(CONFIG.section("inference")["maximum_characters"]),
        )
    word_count = len(submitted_text.split())
    st.caption(f"{len(submitted_text):,} characters · {word_count:,} words")
    with st.expander("Example input"):
        st.write(
            "City council approves a revised public-transit budget after a three-hour "
            "meeting, according to documents released Tuesday."
        )
    analyze, clear = st.columns([1, 1])
    analyze_clicked = analyze.button("Analyze", type="primary", use_container_width=True)
    if clear.button("Clear", use_container_width=True):
        for key in ("single_text", "headline", "article"):
            st.session_state[key] = ""
        st.rerun()
    if not analyze_clicked:
        return
    if predictor is None:
        st.error(
            "No trained model is deployed. Add the dataset and run `python train.py`, "
            "then restart this app."
        )
        return
    try:
        result = predictor.predict(submitted_text)
    except InputValidationError as exc:
        st.warning(str(exc))
        return
    except Exception:
        st.error("The text could not be analyzed safely. Please try a shorter plain-text input.")
        return

    st.divider()
    st.subheader("Model output")
    first, second, third = st.columns(3)
    first.metric("Predicted pattern class", result["class_name"])
    second.metric(
        "Estimated model probability",
        probability_text(result["estimated_model_probability"]),
    )
    third.metric("Confidence band", result["confidence_band"])
    st.caption(
        "The probability is the model's calibrated estimate for its predicted label—not "
        "the probability that the article is objectively true or false."
    )
    if result["warnings"]:
        st.warning("Input quality: " + " ".join(result["warnings"]))
    else:
        st.info("Input quality: no basic length, language, or coverage warnings were detected.")

    st.markdown("#### Influential words and phrases")
    fake = contribution_frame(result["explanation"]["toward_fake"], "Toward Fake News")
    real = contribution_frame(result["explanation"]["toward_real"], "Toward Real News")
    features = pd.concat([fake, real], ignore_index=True)
    if features.empty:
        st.caption("No interpretable word-level contributions were available.")
    else:
        figure = px.bar(
            features,
            x="Contribution strength",
            y="Word or phrase",
            color="Direction",
            orientation="h",
            barmode="group",
            color_discrete_map={
                "Toward Fake News": "#d97706",
                "Toward Real News": "#2563eb",
            },
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
        st.plotly_chart(figure, use_container_width=True)
    st.caption(result["explanation"]["disclaimer"])
    st.warning(result["disclaimer"])


def performance_tab(metrics: dict[str, Any] | None) -> None:
    """Render measured model performance only."""
    st.subheader("Model performance")
    if metrics is None:
        st.info("Performance artifacts appear here after `python train.py` completes.")
        return
    validation = metrics["validation"]
    test = metrics["test"]
    st.write(
        f"Deployed model: **{metrics['selected_model']}** · "
        f"Calibration: **{metrics['deployed_calibration']}**"
    )
    columns = st.columns(4)
    columns[0].metric("Validation macro F1", f"{validation['macro_f1']:.3f}")
    columns[1].metric("Test macro F1", f"{test['macro_f1']:.3f}")
    columns[2].metric("Test accuracy", f"{test['accuracy']:.3f}")
    columns[3].metric("Latency / record", f"{test['inference_latency_ms_per_record']:.2f} ms")
    comparison_path = CONFIG.path("model_comparison")
    if comparison_path.is_file():
        table = pd.read_csv(comparison_path)
        st.markdown("#### Validation model comparison")
        st.dataframe(table, use_container_width=True, hide_index=True)
    for filename, caption in (
        ("confusion_matrix.png", "Untouched test-set confusion matrix"),
        ("discrimination_curves.png", "ROC and precision–recall curves"),
        ("calibration_curve.png", "Reliability diagram"),
    ):
        path = CONFIG.path("figures") / filename
        if path.is_file():
            st.image(str(path), caption=caption, use_container_width=True)
    st.caption(
        "The test split is evaluated only after model and calibration selection on validation data."
    )


def explainability_tab(metrics: dict[str, Any] | None) -> None:
    """Explain global and local interpretation semantics."""
    st.subheader("Explainability")
    st.write(
        "For a linear model, each active TF-IDF value is multiplied by its learned "
        "coefficient. Positive contributions move the score toward Real News; negative "
        "contributions move it toward Fake News. Character features remain active in the "
        "model but are omitted from the user-facing explanation when they are fragmentary."
    )
    if metrics:
        global_features = metrics.get("global_features", {})
        left, right = st.columns(2)
        left.markdown("#### Globally associated with Fake News")
        left.dataframe(pd.DataFrame(global_features.get("fake", [])), hide_index=True)
        right.markdown("#### Globally associated with Real News")
        right.dataframe(pd.DataFrame(global_features.get("real", [])), hide_index=True)
    st.warning(
        "Highlighted terms are statistical signals learned from the training dataset. "
        "They do not independently prove that the article is true or false."
    )


def dataset_tab() -> None:
    """Render generated EDA and leakage artifacts."""
    st.subheader("Dataset analysis")
    report = read_json_if_available(ROOT / "reports" / "eda_report.json")
    if report is None:
        st.info("Dataset analysis is generated during training.")
        return
    one, two, three = st.columns(3)
    one.metric("Prepared rows", f"{report['shape'][0]:,}")
    two.metric("Columns", report["shape"][1])
    three.metric("Exact duplicates remaining", report["duplicate_rows"])
    overview = CONFIG.path("figures") / "dataset_overview.png"
    if overview.is_file():
        st.image(str(overview), use_container_width=True)
    leakage = read_json_if_available(CONFIG.path("leakage_report"))
    if leakage:
        st.markdown("#### Leakage investigation")
        st.dataframe(
            pd.DataFrame(leakage["known_artifacts"]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Suspicious class-specific features")
        st.dataframe(
            pd.DataFrame(leakage["suspicious_class_specific_features"]),
            use_container_width=True,
            hide_index=True,
        )
        benchmark = leakage.get("performance_before_after_mitigation")
        if benchmark:
            st.json(benchmark, expanded=False)


def error_tab() -> None:
    """Render generated held-out errors without full article bodies."""
    st.subheader("Error analysis")
    path = CONFIG.path("error_analysis")
    if not path.is_file():
        st.info("Held-out errors are generated during training.")
        return
    errors = pd.read_csv(path)
    st.metric("Misclassified test examples", len(errors))
    if errors.empty:
        st.info("No errors were recorded on this test split.")
    else:
        st.dataframe(errors, use_container_width=True, hide_index=True)
    st.caption(
        "Review high-confidence errors first. Common risks include source style, temporal "
        "drift, satire, opinion, quotations, unfamiliar events, and very short inputs."
    )


def architecture_tab() -> None:
    """Explain data and runtime boundaries."""
    st.subheader("Project architecture")
    st.code(
        """
Fake.csv + True.csv
        │
        ▼
validation → leakage audit → duplicate-safe split
        │
        ▼
persisted cleaner → word + character TF-IDF → linear classifier → calibration
        │
        ├── metrics, figures, error analysis, metadata
        ▼
joblib pipeline → cached Streamlit inference → local coefficient explanation
""".strip(),
        language="text",
    )
    st.write(
        "Training is an offline command. Streamlit only loads the saved pipeline and never "
        "fits a vectorizer or model on submitted input. Submitted text is not stored by default."
    )


def ethics_tab() -> None:
    """Present model limitations prominently."""
    st.subheader("Limitations and ethics")
    st.markdown(
        """
- The model recognizes language patterns, not objective truth, and performs no
  external fact-checking.
- Dataset and publisher bias can influence every prediction, even after leakage mitigation.
- News language changes over time; new topics and events may be out of distribution.
- Satire, opinion, mixed factual/misleading claims, and short headlines may be misclassified.
- A high estimated model probability does not guarantee correctness.
- Do not use this system for censorship, legal decisions, or automated content removal.
- Verify important claims with primary sources and trusted independent fact-checkers.
"""
    )


def main() -> None:
    """Render all application tabs."""
    st.title("Fake News Pattern Detector")
    st.write(
        "A leakage-aware TF-IDF and linear-classifier system for educational analysis "
        "of English news writing patterns."
    )
    predictor: FakeNewsPredictor | None = None
    model_path = CONFIG.path("model")
    metadata_path = CONFIG.path("metadata")
    if model_path.is_file() and metadata_path.is_file():
        try:
            predictor = load_predictor(model_path.stat().st_mtime, metadata_path.stat().st_mtime)
            version = predictor.metadata.get("model_version", "unknown")
            st.caption(f"Deployed model version: {version}")
        except FakeNewsDetectorError as exc:
            st.error(str(exc))
    else:
        st.info("Demo setup required: place the dataset in `data/raw/` and run `python train.py`.")
    metrics = read_json_if_available(CONFIG.path("metrics"))
    tabs = st.tabs(
        [
            "Prediction",
            "Model Performance",
            "Explainability",
            "Dataset Analysis",
            "Error Analysis",
            "Project Architecture",
            "Limitations & Ethics",
        ]
    )
    with tabs[0]:
        prediction_tab(predictor)
    with tabs[1]:
        performance_tab(metrics)
    with tabs[2]:
        explainability_tab(metrics)
    with tabs[3]:
        dataset_tab()
    with tabs[4]:
        error_tab()
    with tabs[5]:
        architecture_tab()
    with tabs[6]:
        ethics_tab()


if __name__ == "__main__":
    main()
