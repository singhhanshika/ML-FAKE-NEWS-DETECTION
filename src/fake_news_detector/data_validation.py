"""Dataset and user-input validation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd

from fake_news_detector.exceptions import DataValidationError, InputValidationError
from fake_news_detector.preprocessing import normalize_text


@dataclass(frozen=True)
class InputAssessment:
    """Quality signals calculated before prediction."""

    text: str
    word_count: int
    character_count: int
    warnings: list[str]


def validate_dataset(frame: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Validate columns and normalize missing textual fields."""
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {', '.join(missing)}")
    cleaned = frame.copy()
    for column in required_columns:
        cleaned[column] = cleaned[column].fillna("").astype(str)
    if "label" not in cleaned.columns:
        raise DataValidationError("Dataset must contain a numeric label column.")
    invalid_labels = set(cleaned["label"].dropna().unique()) - {0, 1}
    if invalid_labels:
        raise DataValidationError(f"Unsupported labels: {sorted(invalid_labels)}")
    return cleaned


def combine_text(frame: pd.DataFrame, separator: str = " bodytext ") -> pd.Series:
    """Combine title and body with an explicit, vocabulary-safe separator."""
    titles = frame["title"].fillna("").astype(str).str.strip()
    bodies = frame["text"].fillna("").astype(str).str.strip()
    return (titles + separator + bodies).str.strip()


def document_fingerprint(text: str) -> str:
    """Create a stable fingerprint for duplicate-safe grouping."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def remove_duplicate_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove exact and normalized duplicates while reporting removals."""
    before = len(frame)
    exact = frame.drop_duplicates(subset=["title", "text", "label"]).copy()
    exact_removed = before - len(exact)
    exact["document_hash"] = exact["content"].map(document_fingerprint)
    contradictory = exact.groupby("document_hash")["label"].nunique()
    contradictory_hashes = set(contradictory[contradictory > 1].index)
    contradiction_count = int(exact["document_hash"].isin(contradictory_hashes).sum())
    exact = exact[~exact["document_hash"].isin(contradictory_hashes)]
    deduplicated = exact.drop_duplicates(subset=["document_hash"], keep="first").copy()
    normalized_removed = len(exact) - len(deduplicated)
    return deduplicated.reset_index(drop=True), {
        "exact_duplicates_removed": exact_removed,
        "normalized_duplicates_removed": normalized_removed,
        "contradictory_duplicate_rows_removed": contradiction_count,
    }


def assess_input(
    text: object,
    *,
    max_characters: int,
    minimum_words: int,
    ascii_ratio_threshold: float,
) -> InputAssessment:
    """Validate one inference request and produce non-blocking quality warnings."""
    if text is None or not str(text).strip():
        raise InputValidationError("Enter a headline or article before analyzing.")
    value = str(text).strip()
    if len(value) > max_characters:
        raise InputValidationError(f"Input exceeds the {max_characters:,}-character safety limit.")
    words = re.findall(r"\b[\w'-]+\b", value, flags=re.UNICODE)
    if not words:
        raise InputValidationError("The input does not contain analyzable words.")
    warnings: list[str] = []
    if len(words) < minimum_words:
        warnings.append(f"Only {len(words)} words were provided; short inputs are less reliable.")
    letter_count = sum(char.isalpha() for char in value)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in value)
    if letter_count and ascii_letters / letter_count < ascii_ratio_threshold:
        warnings.append(
            "The text may be non-English or mixed-language; this model was trained on English news."
        )
    if len(set(word.lower() for word in words)) <= 2 and len(words) >= 5:
        warnings.append("The input is highly repetitive and unlike a normal news article.")
    return InputAssessment(
        text=value,
        word_count=len(words),
        character_count=len(value),
        warnings=warnings,
    )
