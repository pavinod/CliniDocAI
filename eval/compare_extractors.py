"""Compare rule-based and LLM-based fact extraction.

Evaluation pipeline only (it reads the expected answers); the app never
imports this file.

Usage:
    python eval/compare_extractors.py                          # rule-based only
    python eval/compare_extractors.py --model openrouter:<id>  # plus LLM arm
    python eval/compare_extractors.py --model lmstudio --corpus path/to/txt

* Supplied patient: for each of the 15 expected answers, checks whether the
  answer's key values appear in the facts each extractor produced. This
  measures extraction coverage, not the screen (see evaluate.py for that).
* --corpus: a folder of plain-text clinical documents (for example a GraSCCo
  download). There is no answer key there, so the script writes every fact
  of both extractors to a CSV for manual annotation and reports counts only.

Matching is a keyword heuristic and must be confirmed by human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_env_file  # noqa: E402
from app.extract import build_fact_store  # noqa: E402
from app.llm_extract import llm_extract  # noqa: E402
from app.model import HttpAdapter, adapter_for_choice  # noqa: E402
from app.pipeline import Document, PatientData, clean_text, load_patient  # noqa: E402

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß-]{4,}")
_STOP = {"documented", "which", "there", "their", "from", "with", "between",
         "about", "during", "after", "before", "record", "other"}


def answer_found(answer: str, text: str) -> bool:
    """All numbers and at least half of the key words of the answer present."""
    text = text.casefold().replace(",", ".")
    numbers = [n.replace(",", ".") for n in _NUMBER.findall(answer)]
    words = [w.casefold() for w in _WORD.findall(answer)
             if w.casefold() not in _STOP]
    if any(n not in text for n in numbers):
        return False
    if not words:
        return bool(numbers)
    return sum(w in text for w in words) * 2 >= len(words)


def _load_expected() -> list[dict]:
    for path in (ROOT / "eval" / "expected.json", ROOT / "cases" / "expected.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit("expected.json not found in eval/ or cases/")


def _corpus(folder: Path) -> PatientData:
    docs = {}
    for path in sorted(folder.glob("*.txt")):
        text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        lines = text.splitlines()
        offsets, offset = [], 0
        for line in lines:
            offsets.append(offset)
            offset += len(line) + 1
        docs[path.stem] = Document(
            doc_id=path.stem, title=path.stem, date="", doc_type="text",
            text=text, doc_hash=path.stem, lines=lines, line_offsets=offsets)
    return PatientData(patient={}, documents=docs, doc_order=list(docs), index={})


def _run_llm(data: PatientData, adapter):
    started = time.perf_counter()
    facts, rejected = llm_extract(
        data, lambda prompt: adapter.complete(prompt, max_tokens=4000))
    return facts, rejected, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", default="data/patient-0000-0001")
    parser.add_argument("--model", default="",
                        help="openrouter:<model id> or lmstudio")
    parser.add_argument("--corpus", type=Path,
                        help="folder of .txt documents (e.g. GraSCCo)")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "comparison")
    args = parser.parse_args()
    load_env_file(ROOT / ".env")
    args.out.mkdir(parents=True, exist_ok=True)

    adapter = None
    if args.model:
        adapter, problem = adapter_for_choice(args.model)
        if problem or not isinstance(adapter, HttpAdapter):
            raise SystemExit(f"LLM arm unavailable: {problem or 'not an LLM choice'}")

    # ---- supplied patient vs expected answers
    data = load_patient(Path(args.data_root))
    started = time.perf_counter()
    rule_facts = build_fact_store(data)
    rule_time = time.perf_counter() - started
    arms = {"rule-based": (rule_facts, [], rule_time)}
    if adapter:
        arms[f"llm ({adapter.model_name})"] = _run_llm(data, adapter)

    expected = _load_expected()
    report = {"patient": {}, "corpus": {}}
    print(f"{'arm':38} {'answers':>8} {'facts':>6} {'rejected':>9} {'seconds':>8}")
    for arm, (facts, rejected, seconds) in arms.items():
        text = " ".join(f.text + " " + (f.original or "") for f in facts)
        hits = [e["id"] for e in expected if answer_found(e["answer"], text)]
        report["patient"][arm] = {
            "answers_covered": hits, "facts": len(facts),
            "rejected": rejected, "seconds": round(seconds, 2)}
        print(f"{arm:38} {len(hits):>5}/{len(expected):<2} {len(facts):>6} "
              f"{len(rejected):>9} {seconds:>8.2f}")

    # ---- external corpus (no answer key): facts for manual annotation
    if args.corpus:
        corpus = _corpus(args.corpus)
        corpus_arms = {"rule-based": (build_fact_store(corpus), [], 0.0)}
        if adapter:
            corpus_arms[f"llm ({adapter.model_name})"] = _run_llm(corpus, adapter)
        sheet = args.out / "corpus_facts.csv"
        with sheet.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["arm", "document", "category", "fact", "quote",
                             "correct? (annotate)"])
            for arm, (facts, rejected, _) in corpus_arms.items():
                for fact in facts:
                    writer.writerow([arm, fact.evidence[0].doc_id, fact.category,
                                     fact.text, fact.original or
                                     fact.evidence[0].quote, ""])
                report["corpus"][arm] = {"facts": len(facts),
                                         "rejected": len(rejected)}
        print(f"\nCorpus: {len(corpus.documents)} documents; facts for manual "
              f"annotation written to {sheet}")

    (args.out / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetails: {args.out / 'comparison.json'}")
    print("Keyword matching is heuristic — confirm by human review.")


if __name__ == "__main__":
    main()
