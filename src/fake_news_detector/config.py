"""Typed project configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration with paths resolved relative to the repository root."""

    values: dict[str, Any]
    root: Path = PROJECT_ROOT

    def section(self, name: str) -> dict[str, Any]:
        """Return a required top-level section."""
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Missing configuration section: {name}")
        return value

    def path(self, name: str) -> Path:
        """Resolve a configured project path."""
        raw = self.section("paths").get(name)
        if not raw:
            raise KeyError(f"Missing configured path: {name}")
        return self.root / str(raw)

    @property
    def random_seed(self) -> int:
        """Return the reproducible project seed."""
        return int(self.section("project")["random_seed"])


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load YAML configuration from disk."""
    config_path = path or PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration root must be a mapping.")
    return ProjectConfig(values=values, root=PROJECT_ROOT)
