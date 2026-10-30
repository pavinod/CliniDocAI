"""Hackathon metric checks: 5 situations x 15 expected answers, traceability,
contradictions and load time. Run: pytest eval/test_metrics.py -v
"""
import json
import time
from pathlib import Path

import pytest

from app.contradictions import detect_contradictions
from app.extract import build_fact_store
from app.needs import build_screen
from app.pipeline import load_patient, resolve_data_dir

DATA = Path(__file__).resolve().parent.parent / "data"
EXPECTED = json.loads((Path(__file__).parent / "expected.json").read_text())

SITUATIONS = {
    "Pre-operative anaesthesia check":
        "You are the anaesthetist. She is scheduled for a quadrantectomy "
        "(partial removal of the breast) with intraoperative radiotherapy on 16 May 2016.",
    "Surgery planning after the tumour board, April 2016":
        "You are the surgeon planning her operation after the tumour board.",
    "Diabetes review at the GP":
        "You are her general practitioner (GP), reviewing her diabetes in spring 2016.",
    "Emergency department at night":
        "You are the emergency physician on night duty. She arrives at 02:00 "
        "with dizziness and blurred vision in the left eye.",
    "Open follow-ups from the 2015 hospital stay":
        "You are her GP. You check which follow-ups from her June 2015 "
        "hospital stay are still open.",
}

# question id -> (required substrings anywhere in the screen's item texts,
#                docs that must appear among those items' sources (any-of))
REQUIRED = {
    "1.1": (["thrombo ass", "acetylsalicylic", "100 mg", "1-0-0"], ["12340012"]),
    "1.2": (["contrast media allergy", "belkeit"], ["12340012"]),
    "1.3": (["60%"], ["12340012"]),
    "2.1": (["invasive ductal carcinoma", "t2 n0 m0"], ["12340016"]),
    "2.2": (["oestrogen receptor", "allred", "progesterone receptor",
             "her2 negative", "ki-67", "20%"], ["12340016"]),
    "2.3": (["quadrantectomy", "iort", "16 may 2016"], ["12340018"]),
    "3.1": (["metformin", "1000 mg", "pioglitazone", "30 mg",
             "exenatide", "5 µg"], ["12340018"]),
    "3.2": (["hba1c", "6.4%", "june 2015"], ["12340012"]),
    "3.3": (["78 kg", "67 kg", "66 kg"], ["12340012", "12340014", "12340018"]),
    "4.1": (["hypertensive crisis", "amaurosis", "carotid", "60%"], ["12340011", "12340012"]),
    "4.2": (["contrast media allergy", "metformin", "without contrast",
             "not documented"], ["12340012"]),
    "4.3": (["candesartan", "16 mg", "1-0-1"], ["12340018"]),
    "5.1": (["carotid duplex", "ophthalmology", "blood-pressure"], ["12340012"]),
    "5.2": (["not documented"], []),
    "5.3": (["left", "60%"], ["12340008"]),
}


@pytest.fixture(scope="module")
def store():
    t0 = time.time()
    data = load_patient(resolve_data_dir(DATA))
    facts = build_fact_store(data)
    contras = detect_contradictions(data)
    return data, facts, contras, time.time() - t0


@pytest.fixture(scope="module")
def screens(store):
    data, facts, _, _ = store
    return {name: build_screen(text, data, facts)
            for name, text in SITUATIONS.items()}


def _all_items(screen):
    return [it for sec in screen["sections"] for it in sec["items"]]


def _screen_text(screen):
    return "\n".join(
        " ".join(part for part in (it["text"], it.get("original")) if part)
        for it in _all_items(screen)
    ).lower()


def _screen_docs(screen):
    return {s["doc"] for it in _all_items(screen) for s in it.get("sources", [])}


@pytest.mark.parametrize("exp", EXPECTED, ids=[e["id"] for e in EXPECTED])
def test_expected_answer(exp, screens):
    screen = screens[exp["situation"]]
    text = _screen_text(screen)
    need_frag, need_docs = REQUIRED[exp["id"]]
    missing = [f for f in need_frag if f.lower() not in text]
    assert not missing, f"{exp['id']}: screen missing fragments {missing}"
    if need_docs:
        missing_docs = [d for d in need_docs if d not in _screen_docs(screen)]
        assert not missing_docs, f"{exp['id']}: sources missing docs {missing_docs}"


def test_fifteen_situations_covered(screens):
    assert len(screens) == 5
    for name, scr in screens.items():
        assert scr["sections"], name


def test_traceability(screens):
    """Every displayed fact links to >=1 source with a quote (except
    not-documented items, which need none)."""
    bad = []
    for scr in screens.values():
        for it in _all_items(scr):
            if it["not_documented"]:
                continue
            if not it.get("sources") or not all(s.get("quote") for s in it["sources"]):
                bad.append(it["text"][:60])
    assert not bad, f"items without sources: {bad}"


def test_contradictions(store):
    _, _, contras, _ = store
    assert len(contras) >= 5
    for c in contras:
        assert c.title and c.detail
        docs = {s.doc_id for s in c.evidence}
        assert len(docs) >= 1


def test_load_time(store):
    assert store[3] < 10.0, f"pipeline took {store[3]:.1f}s"
