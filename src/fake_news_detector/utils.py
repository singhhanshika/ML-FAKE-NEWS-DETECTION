"""General serialization and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set reproducible pseudo-random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def file_sha256(paths: list[Path]) -> str:
    """Hash source dataset bytes without modifying them."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write readable UTF-8 JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
