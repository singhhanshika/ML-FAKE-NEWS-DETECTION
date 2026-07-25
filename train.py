"""Command-line entry point for model training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fake_news_detector.config import load_config
from fake_news_detector.exceptions import FakeNewsDetectorError
from fake_news_detector.logging_config import configure_logging
from fake_news_detector.training import train_project


def main() -> int:
    """Run the complete training workflow."""
    configure_logging()
    try:
        result = train_project(load_config())
    except FakeNewsDetectorError as exc:
        print(f"Training could not start: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "model": result["metadata"]["model_name"],
                "version": result["metadata"]["model_version"],
                "validation_macro_f1": result["metrics"]["validation"]["macro_f1"],
                "test_macro_f1": result["metrics"]["test"]["macro_f1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
