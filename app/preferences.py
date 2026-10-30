"""Saved display preferences: text size, interface language, section order.

Stored as a small JSON file outside the repository (no patient data — only
section keys and display settings). Safety sections stay pinned at the top
whatever order is saved, so personalisation can never hide allergies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULTS = {"font_scale": 100, "ui_lang": "en", "section_order": {}}
PINNED_FIRST = ("allergy",)


def preferences_path() -> Path:
    override = os.environ.get("CLINIDOC_PREFS_PATH")
    if override:
        return Path(override)
    return Path.home() / ".config" / "clinidocai" / "preferences.json"


def load_preferences(path: Path | None = None) -> dict:
    path = path or preferences_path()
    prefs = {**DEFAULTS, "section_order": {}}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return prefs
    if isinstance(stored.get("font_scale"), int):
        prefs["font_scale"] = stored["font_scale"]
    if stored.get("ui_lang") in ("en", "de"):
        prefs["ui_lang"] = stored["ui_lang"]
    if isinstance(stored.get("section_order"), dict):
        prefs["section_order"] = {
            str(role): [str(key) for key in keys]
            for role, keys in stored["section_order"].items()
            if isinstance(keys, list)}
    return prefs


def save_preferences(prefs: dict, path: Path | None = None) -> Path:
    path = path or preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        "font_scale": int(prefs.get("font_scale", DEFAULTS["font_scale"])),
        "ui_lang": prefs.get("ui_lang", DEFAULTS["ui_lang"]),
        "section_order": prefs.get("section_order", {}),
    }
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return path


def order_section_keys(keys: list[str], preferred: list[str] | None) -> list[str]:
    """Apply a saved order to the sections of the current screen.

    Unknown saved keys are ignored, new sections keep their default position
    after the saved ones, and pinned safety sections always come first.
    """
    preferred = preferred or []
    ordered = [key for key in preferred if key in keys]
    ordered += [key for key in keys if key not in ordered]
    pinned = [key for key in PINNED_FIRST if key in ordered]
    return pinned + [key for key in ordered if key not in pinned]


def move_section(keys: list[str], key: str, step: int) -> list[str]:
    """Move ``key`` by ``step`` positions without passing a pinned section."""
    keys = list(keys)
    if key not in keys or key in PINNED_FIRST:
        return keys
    index = keys.index(key)
    floor = sum(1 for pinned in PINNED_FIRST if pinned in keys)
    target = max(floor, min(len(keys) - 1, index + step))
    keys.insert(target, keys.pop(index))
    return keys
