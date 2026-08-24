"""Small, strict JSON configuration helpers for reproducible experiments."""

from __future__ import annotations

import json
from pathlib import Path


def load_json_object(path: str | Path, allowed_keys: set[str]) -> dict[str, object]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        values = json.load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    unknown = set(values) - allowed_keys
    if unknown:
        raise ValueError(f"{config_path} contains unsupported keys: {', '.join(sorted(unknown))}")
    return values
