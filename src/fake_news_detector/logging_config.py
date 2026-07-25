"""Central logging configuration."""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Configure concise, structured application logging once."""
    level = os.getenv("FAKE_NEWS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
