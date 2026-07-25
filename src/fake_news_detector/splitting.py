"""Reproducible baseline, duplicate-safe, and time-aware splits."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from fake_news_detector.preprocessing import normalize_text


def _title_group(title: object, fallback: str) -> str:
    normalized = normalize_text(title)
    key = " ".join(normalized.split()[:30]) or fallback
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def assign_splits(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign split labels without using the test set for model selection."""
    split_config = config["split"]
    strategy = split_config["strategy"]
    output = frame.copy()
    output["split"] = ""

    if strategy == "time_aware" and "date" in output.columns:
        parsed = pd.to_datetime(output["date"], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            output = output.assign(_parsed_date=parsed).sort_values("_parsed_date")
            train_end = int(len(output) * float(split_config["train_size"]))
            validation_end = train_end + int(len(output) * float(split_config["validation_size"]))
            output.iloc[:train_end, output.columns.get_loc("split")] = "train"
            output.iloc[train_end:validation_end, output.columns.get_loc("split")] = "validation"
            output.iloc[validation_end:, output.columns.get_loc("split")] = "test"
            output = output.drop(columns="_parsed_date")
            return output, _split_summary(output, "time_aware")

    if strategy == "baseline":
        train, remainder = train_test_split(
            output,
            train_size=float(split_config["train_size"]),
            stratify=output["label"],
            random_state=seed,
        )
        relative_test = float(split_config["test_size"]) / (
            float(split_config["validation_size"]) + float(split_config["test_size"])
        )
        validation, test = train_test_split(
            remainder,
            test_size=relative_test,
            stratify=remainder["label"],
            random_state=seed,
        )
    else:
        output["_group"] = [
            _title_group(title, fallback)
            for title, fallback in zip(output["title"], output["document_hash"], strict=True)
        ]
        first = GroupShuffleSplit(
            n_splits=1,
            train_size=float(split_config["train_size"]),
            random_state=seed,
        )
        train_index, remainder_index = next(
            first.split(output, output["label"], groups=output["_group"])
        )
        train = output.iloc[train_index]
        remainder = output.iloc[remainder_index]
        relative_test = float(split_config["test_size"]) / (
            float(split_config["validation_size"]) + float(split_config["test_size"])
        )
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_test,
            random_state=seed + 1,
        )
        validation_index, test_index = next(
            second.split(remainder, remainder["label"], groups=remainder["_group"])
        )
        validation = remainder.iloc[validation_index]
        test = remainder.iloc[test_index]

    output.loc[train.index, "split"] = "train"
    output.loc[validation.index, "split"] = "validation"
    output.loc[test.index, "split"] = "test"
    output = output.drop(columns=["_group"], errors="ignore")
    return output, _split_summary(output, strategy)


def _split_summary(frame: pd.DataFrame, strategy: str) -> dict[str, Any]:
    """Summarize split sizes, balance, dates, and group overlap."""
    summary: dict[str, Any] = {"strategy": strategy, "splits": {}}
    for name in ("train", "validation", "test"):
        subset = frame[frame["split"] == name]
        details: dict[str, Any] = {
            "rows": int(len(subset)),
            "class_distribution": {
                str(key): int(value)
                for key, value in subset["label"].value_counts().sort_index().items()
            },
        }
        if "date" in subset.columns:
            parsed = pd.to_datetime(subset["date"], errors="coerce")
            if parsed.notna().any():
                details["date_min"] = parsed.min().isoformat()
                details["date_max"] = parsed.max().isoformat()
        summary["splits"][name] = details
    hashes = {
        name: set(frame.loc[frame["split"] == name, "document_hash"])
        for name in ("train", "validation", "test")
    }
    summary["exact_hash_overlap"] = {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }
    return summary
