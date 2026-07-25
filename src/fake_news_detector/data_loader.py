"""Safe loading and preparation of the Fake and Real News Dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fake_news_detector.data_validation import (
    combine_text,
    remove_duplicate_records,
    validate_dataset,
)
from fake_news_detector.exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with a tolerant encoding fallback."""
    if not path.is_file():
        raise DataValidationError(
            f"Required dataset file is missing: {path.name}. "
            "Place Fake.csv and True.csv in data/raw/."
        )
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        LOGGER.warning("UTF-8 decoding failed for %s; using latin-1 fallback.", path.name)
        return pd.read_csv(path, encoding="latin-1")


def load_raw_dataset(
    fake_path: Path,
    real_path: Path,
    *,
    required_columns: list[str],
    separator: str,
    minimum_words: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load immutable raw files, add labels, and return a clean modeling table."""
    LOGGER.info("Loading raw news datasets.")
    fake = _read_csv(fake_path)
    real = _read_csv(real_path)
    fake["label"] = 0
    real["label"] = 1
    frame = pd.concat([fake, real], ignore_index=True, sort=False)
    frame = validate_dataset(frame, required_columns)
    frame["content"] = combine_text(frame, separator)
    frame["word_count"] = frame["content"].str.split().str.len()
    empty_count = int((frame["word_count"] == 0).sum())
    short_count = int((frame["word_count"] < minimum_words).sum())
    frame = frame[frame["word_count"] > 0].copy()
    frame, duplicate_stats = remove_duplicate_records(frame)
    summary: dict[str, object] = {
        "raw_rows": len(fake) + len(real),
        "usable_rows": len(frame),
        "empty_rows_removed": empty_count,
        "short_rows_retained": short_count,
        "class_distribution": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        **duplicate_stats,
    }
    LOGGER.info("Prepared %d usable records.", len(frame))
    return frame, summary
