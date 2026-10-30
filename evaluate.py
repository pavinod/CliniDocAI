"""
Evaluation script for the five hackathon situations.

Usage:
    python evaluate.py [--data-root <path>]

Reads the five situations from cases/situations.md (descriptions only),
runs the pipeline for each, and reports metrics WITHOUT reading expected.json
at runtime.

Expected answers are loaded separately for comparison and kept out of the
pipeline code entirely — they are only used here, in the evaluation pipeline.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Reconfigure stdout to UTF-8 on Windows so emoji print without crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from clinidocai.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# The five situations (title → role, task description)
# ---------------------------------------------------------------------------

SITUATIONS = [
    {
        "id": "S1",
        "title": "Pre-operative anaesthesia check",
        "role": "Anaesthetist",
        "task": (
            "She is scheduled for a quadrantectomy (partial removal of the breast) "
            "with intraoperative radiotherapy on 16 May 2016."
        ),
        "questions": [
            "Which antithrombotic drug (blood thinner) does she take and at what dose?",
            "Which allergy is documented?",
            "What was the last documented left-ventricular ejection fraction?",
        ],
    },
    {
        "id": "S2",
        "title": "Surgery planning after the tumour board, April 2016",
        "role": "Surgeon",
        "task": "Planning her operation after the tumour board.",
        "questions": [
            "What is the histology (tissue type) and the TNM stage?",
            "What is the receptor status and the Ki-67 rate?",
            "What did the tumour board recommend and when is it scheduled?",
        ],
    },
    {
        "id": "S3",
        "title": "Diabetes review at the GP",
        "role": "General Practitioner",
        "task": "Reviewing her diabetes in spring 2016.",
        "questions": [
            "Which glucose-lowering drugs does she take and at what doses?",
            "What is her latest HbA1c and from when?",
            "How did her weight change between June 2015 and April 2016?",
        ],
    },
    {
        "id": "S4",
        "title": "Emergency department at night",
        "role": "Emergency Physician",
        "task": (
            "She arrives at 02:00 with dizziness and blurred vision in the left eye."
        ),
        "questions": [
            "Has this happened before and what was found to be the cause?",
            "Is anything documented that matters before giving iodinated contrast?",
            "Which antihypertensive does she take and at what dose?",
        ],
    },
    {
        "id": "S5",
        "title": "Open follow-ups from the 2015 hospital stay",
        "role": "General Practitioner",
        "task": "Checking which follow-ups from her June 2015 hospital stay are still open.",
        "questions": [
            "Which follow-ups were recommended at discharge in June 2015?",
            "Is there a record that the follow-up carotid ultrasound took place?",
            "On which side is the carotid stenosis?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Simple answer-presence checker
# ---------------------------------------------------------------------------

def _screen_text(screen) -> str:
    """Flatten the screen to plain text for substring search."""
    parts = []
    for section in screen.sections:
        parts.append(section.heading)
        for fact in section.facts:
            parts.append(fact.label)
            if fact.value:
                parts.append(str(fact.value))
        for m in section.missing:
            parts.append(m.label)
            parts.append(m.reason)
    for c in screen.contradictions:
        parts.append(c.description)
    return " ".join(parts).lower()


def _count_facts_with_evidence(screen) -> tuple[int, int]:
    """Return (facts_with_evidence, total_facts)."""
    total = 0
    with_ev = 0
    for section in screen.sections:
        for fact in section.facts:
            total += 1
            if fact.evidence:
                with_ev += 1
    return with_ev, total


def _count_contradictions(screen) -> tuple[int, int]:
    """Return (real_contradictions, false_alarms)."""
    real = sum(1 for c in screen.contradictions if not c.is_temporal_change)
    temporal = sum(1 for c in screen.contradictions if c.is_temporal_change)
    return real, temporal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_root: str = "data/patient-0000-0001") -> None:
    print("=" * 70)
    print("CliniDocAI Evaluation Report")
    print(f"Data root : {data_root}")
    print(f"Reference : 2016-04-15")
    print("=" * 70)

    # Load expected answers (evaluation pipeline only)
    expected_path = ROOT / "cases" / "expected.json"
    expected: list[dict] = []
    if expected_path.exists():
        with open(expected_path, encoding="utf-8") as fh:
            expected = json.load(fh)

    total_questions  = 0
    correct_answers  = 0
    total_facts      = 0
    facts_with_ev    = 0
    all_contradictions: list = []
    slowest_time     = 0.0
    cold_start_times: list[float] = []
    warm_start_times: list[float] = []

    for sit in SITUATIONS:
        print("\n" + "-" * 60)
        print(f"Situation {sit['id']}: {sit['title']}")
        print(f"Role: {sit['role']}")
        print(f"Task: {sit['task'][:80]}…")

        # Cold start
        t0 = time.perf_counter()
        screen, elapsed_cold, _ = run_pipeline(
            role=sit["role"],
            task=sit["task"],
            data_root=data_root,
        )
        cold_start_times.append(elapsed_cold)
        if elapsed_cold > slowest_time:
            slowest_time = elapsed_cold

        # Warm start (second call — loader is cached)
        screen2, elapsed_warm, _ = run_pipeline(
            role=sit["role"],
            task=sit["task"],
            data_root=data_root,
        )
        warm_start_times.append(elapsed_warm)

        screen_txt = _screen_text(screen)

        print(f"  Load time (cold): {elapsed_cold:.3f}s | (warm): {elapsed_warm:.3f}s")

        # Check expected answers
        sit_expected = [e for e in expected if sit["title"] in e.get("situation", "")]
        print(f"  Expected answers found: {len(sit_expected)}")

        for q_idx, question in enumerate(sit["questions"], 1):
            q_id = f"{sit['id'][1:]}.{q_idx}"   # e.g. "1.1"
            exp = next((e for e in sit_expected if e.get("id") == f"{sit['id'][1:]}.{q_idx}"), None)
            total_questions += 1

            # Heuristic: check if key answer keywords appear in screen text
            answered = False
            if exp:
                expected_ans = exp["answer"].lower()

                if not exp.get("sources"):
                    # "Not documented" / no-source answers
                    answered = (
                        "not documented" in screen_txt
                        or "no record" in screen_txt
                        or "none of the" in screen_txt
                        or "not found" in screen_txt
                    )
                else:
                    # Extract key clinical terms (numbers + words >3 chars, skip stopwords)
                    _STOPWORDS = {
                        "from", "with", "that", "this", "also", "does", "were",
                        "have", "june", "april", "2015", "2016", "and", "the",
                        "she", "was", "for", "her", "its", "not",
                    }
                    answer_words = expected_ans.split()
                    significant = [
                        w.strip(".,()%:;") for w in answer_words
                        if (
                            len(w.strip(".,()%:;")) > 3
                            or w.strip(".,()%:;").replace(",", "").replace(".", "").isdigit()
                        )
                        and w.strip(".,()%:;") not in _STOPWORDS
                    ][:8]
                    matched = sum(1 for w in significant if w in screen_txt)
                    answered = matched >= min(2, len(significant))

            if answered:
                correct_answers += 1
                tick = "✅"
            else:
                tick = "❌"

            print(f"  Q{q_id}: {tick} {question[:60]}…")
            if exp:
                print(f"       Expected: {exp['answer'][:80]}")

        # Evidence traceability
        ev, tot = _count_facts_with_evidence(screen)
        facts_with_ev += ev
        total_facts   += tot
        print(f"  Facts with evidence: {ev}/{tot}")

        # Contradictions
        real, temporal = _count_contradictions(screen)
        all_contradictions.extend([c for c in screen.contradictions if not c.is_temporal_change])
        print(f"  Contradictions: {real} real, {temporal} temporal")
        for c in screen.contradictions:
            marker = "REAL" if not c.is_temporal_change else "TEMPORAL"
            print(f"    [{marker}] {c.contradiction_id}: {c.description[:70]}…")

    # Deduplicate contradictions by ID
    seen_ids: set[str] = set()
    unique_real = []
    for c in all_contradictions:
        if c.contradiction_id not in seen_ids:
            seen_ids.add(c.contradiction_id)
            unique_real.append(c)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    traceability_pct = (facts_with_ev / total_facts * 100) if total_facts else 0
    print(f"Correct answers        : {correct_answers}/{total_questions}")
    print(f"Passage traceability   : {facts_with_ev}/{total_facts} = {traceability_pct:.1f}%")
    print(f"Unique contradictions  : {len(unique_real)}")
    print(f"Slowest screen (cold)  : {slowest_time:.3f}s  (target < 10s)")
    print(f"Avg cold-start         : {sum(cold_start_times)/len(cold_start_times):.3f}s")
    print(f"Avg warm-start         : {sum(warm_start_times)/len(warm_start_times):.3f}s")
    print()
    print("Targets:")
    print(f"  15/15 answers        : {'✅' if correct_answers == 15 else f'❌ ({correct_answers}/15)'}")
    print(f"  100% traceability    : {'✅' if traceability_pct == 100 else f'❌ ({traceability_pct:.1f}%)'}")
    print(f"  ≥5 contradictions    : {'✅' if len(unique_real) >= 5 else f'❌ ({len(unique_real)}/5)'}")
    print(f"  <10s each screen     : {'✅' if slowest_time < 10 else f'❌ ({slowest_time:.2f}s)'}")
    print()
    print("NOTE: Automated answer-matching is heuristic. Human review of semantic")
    print("correctness and contradiction labels is required for final evaluation.")
    print()
    print("Invented facts: manual review required — not automatically detectable.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CliniDocAI evaluation")
    parser.add_argument("--data-root", default="data/patient-0000-0001")
    args = parser.parse_args()
    main(data_root=args.data_root)
