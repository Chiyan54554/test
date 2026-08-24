"""Minimal dependency-free .env loading for local API credentials."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    """Load unset environment variables from a simple KEY=VALUE file."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value.strip("\"'"))
