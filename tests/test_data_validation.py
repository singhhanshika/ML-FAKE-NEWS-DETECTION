"""Tests for dataset and inference data contracts."""

import pandas as pd
import pytest

from fake_news_detector.data_validation import (
    assess_input,
    combine_text,
    validate_dataset,
)
from fake_news_detector.exceptions import DataValidationError, InputValidationError


def test_missing_values_are_normalized() -> None:
    frame = pd.DataFrame({"title": [None], "text": ["body"], "label": [0]})
    validated = validate_dataset(frame, ["title", "text"])
    assert validated.loc[0, "title"] == ""


def test_missing_required_column_raises() -> None:
    with pytest.raises(DataValidationError):
        validate_dataset(pd.DataFrame({"title": ["x"], "label": [1]}), ["title", "text"])


def test_combines_title_and_body_with_separator() -> None:
    frame = pd.DataFrame({"title": ["Title"], "text": ["Body"]})
    assert combine_text(frame).iloc[0] == "Title bodytext Body"


def test_empty_input_rejected() -> None:
    with pytest.raises(InputValidationError):
        assess_input("", max_characters=100, minimum_words=5, ascii_ratio_threshold=0.5)


def test_long_input_rejected() -> None:
    with pytest.raises(InputValidationError):
        assess_input(
            "word " * 30,
            max_characters=20,
            minimum_words=5,
            ascii_ratio_threshold=0.5,
        )


def test_non_english_warning() -> None:
    result = assess_input(
        "यह एक समाचार परीक्षण वाक्य है",
        max_characters=500,
        minimum_words=2,
        ascii_ratio_threshold=0.55,
    )
    assert any("non-English" in warning for warning in result.warnings)
