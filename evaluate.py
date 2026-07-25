"""Re-evaluate the persisted model on the saved untouched test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import pandas as pd

from fake_news_detector.config import load_config
from fake_news_detector.evaluation import evaluate_model
from fake_news_detector.logging_config import configure_logging


def main() -> int:
    """Load saved artifacts and print test metrics."""
    configure_logging()
    config = load_config()
    model_path = config.path("model")
    data_path = config.path("processed_data")
    if not model_path.is_file() or not data_path.is_file():
        print("Run `python train.py` before evaluation.", file=sys.stderr)
        return 2
    model = joblib.load(model_path)
    frame = pd.read_csv(data_path)
    test = frame[frame["split"] == "test"]
    metrics = evaluate_model(model, test["content"], test["label"])
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
