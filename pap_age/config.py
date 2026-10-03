from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def output_path(cfg: dict[str, Any], *parts: str) -> Path:
    p = Path(cfg["project"]["output_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
