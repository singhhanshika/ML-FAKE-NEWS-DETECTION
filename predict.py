"""Command-line prediction helper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fake_news_detector.config import load_config
from fake_news_detector.exceptions import FakeNewsDetectorError
from fake_news_detector.inference import FakeNewsPredictor


def main() -> int:
    """Classify a supplied text and emit JSON."""
    parser = argparse.ArgumentParser(description="Classify news language patterns.")
    parser.add_argument("text", help="Headline or article text")
    args = parser.parse_args()
    config = load_config()
    try:
        predictor = FakeNewsPredictor(
            config.path("model"),
            config.path("metadata"),
            config.section("inference"),
        )
        print(json.dumps(predictor.predict(args.text), indent=2))
    except FakeNewsDetectorError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
