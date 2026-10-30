"""Load KEY=VALUE pairs from a local .env file into the process environment.

Kept dependency-free on purpose. Values already set in the environment win,
so an exported variable always overrides the file. The file holds secrets
(API keys) and must never be committed — see .gitignore.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_file(text: str) -> dict[str, str]:
    """Parse .env text: comments, blank lines and 'export ' prefixes allowed."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_env_file(path: Path) -> list[str]:
    """Set variables from ``path`` that are not already set; return their names."""
    if not path.is_file():
        return []
    loaded = []
    for key, value in parse_env_file(path.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
