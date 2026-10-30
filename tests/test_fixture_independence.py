"""Public test files must not copy wording from the supplied patient letters.

Runs only where the (unpublished) data folder exists; skipped in public CI.
"""

import html
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "data" / "patient-0000-0001" / "documents"
RUN = 6  # words in a row that may not be shared


def _words(text: str) -> list[str]:
    text = html.unescape(text).replace("<BR>", " ")
    return re.findall(r"[\wäöüÄÖÜß%./+-]+", text.lower())


def _runs(words: list[str]) -> set[str]:
    return {" ".join(words[i:i + RUN]) for i in range(len(words) - RUN + 1)}


@pytest.mark.skipif(not DOCUMENTS.is_dir(), reason="supplied data not present")
def test_test_files_share_no_long_word_runs_with_patient_letters():
    letters = set()
    for path in DOCUMENTS.glob("*.json"):
        letters |= _runs(_words(json.loads(path.read_text(encoding="utf-8"))["text"]))
    findings = []
    for path in sorted(ROOT.joinpath("tests").rglob("*")):
        if path.suffix not in {".py", ".txt", ".json"}:
            continue
        shared = _runs(_words(path.read_text(encoding="utf-8"))) & letters
        if shared:
            findings.append(f"{path.relative_to(ROOT)}: {sorted(shared)[:3]}")
    assert not findings, "\n".join(findings)
