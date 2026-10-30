# CliniDocAI

Hackathon prototype (Track 1 — Clinical Documentation). Given a clinician's
role and task, CliniDocAI displays **one screen** of relevant,
source-supported patient facts, missing information, and contradictions.

> Rule-based extraction (no AI model) — pending clinician review. Not clinically validated.

## Quick start

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the supplied synthetic patient dataset under data/patient-0000-0001/
#   (documents.json + documents/<id>.json + patient.json + index jsons)
streamlit run app.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
# place the supplied synthetic patient dataset under data\patient-0000-0001\
streamlit run app.py
```

In the sidebar, choose a role and enter a short task, or select an example to
fill both fields. Then press **Show patient facts**.

The reference date is shown in the sidebar settings and defaults to
2016-04-15.

Voice input is optional and runs local Whisper in the application process.
Install it separately with `pip install -r requirements-voice.txt`; install
`ffmpeg` and make it available on `PATH` for audio formats other than WAV.
English and German are offered in the selector; real transcription has not
been validated in either language.

## Architecture

```
app.py                     UI: role/task input, facts/missing/contradictions,
                          configured examples, highlighted source viewer,
                          text-size control, PDF export
app/
  pipeline.py             document loading, HTML normalisation (entities + <BR>),
                          offset-preserving text model, doc hashes
  schemas.py              pydantic models: Document, EvidenceSpan, Fact,
                          TaskProfile, MissingItem, Contradiction, Screen
  extract.py              deterministic fact extraction (demographics, diagnoses,
                          allergies, vitals, measurements, medication, carotid,
                          histology, procedures, plans, history, admin)
  needs.py                role/task → required fact categories + sections;
                          relevance fallback for unfamiliar task text
  contradictions.py       dynamic conflict detectors (laterality, numeric,
                          dates, case ids, event dates, negation, temporal order)
  model.py                configurable model adapter for claim-support checks
                          (deterministic default + optional HTTP LLM adapter)
  screen.py               orchestration → typed Screen model, content-hash cache,
                          support pass, missing items, recoverable errors
  pdf.py                  PDF export preserving provenance + review status
  lexicon.py              drug brand→INN map, diagnosis translations
config/profiles.yaml      configurable role/task relevance rules
tests/                    pytest suite on independently authored synthetic
                          fixtures (safe for public CI)
eval/                     evaluation pipeline (real data + expected.json —
                          kept out of the public repo; see .gitignore)
```

### Design notes

- **Documents are untrusted data.** Commands embedded in patient documents are
  never executed or followed; text is only pattern-matched as data.
- **Evidence.** Every fact carries `EvidenceSpan`s: document id, byte offsets,
  doc hash, doc date, verbatim quote, extractor version. Normalisation maps
  every character back to the original text.
- **Claim support.** A valid quote proves provenance, not entailment. The
  configured model adapter (`CLINIDOC_MODEL`) assesses each displayed claim;
  claims judged unsupported are dropped rather than re-cited. The default
  adapter is deterministic (no network, no secrets).
- **Clinical semantics.** Positive / negative / suspected / historical /
  recommended / completed / planned are distinct statuses; a missing record
  is *not* a negative finding; a documented medication is not proof of present-day use.
  Scenarios are interpreted as of 2016-04-15; partial dates keep precision.
- **Caching.** The fact store is cached under
  `sha256(doc hashes + extraction/model/prompt versions)` — any document or
  model/prompt change recomputes it.

## Configuration

- `config/profiles.yaml` — role patterns, task keyword rules, section lists.
  Edit to add roles or change relevance; no code change needed.
- `.env.example` — model adapter selection (`deterministic` | `http`),
  endpoint, key, timeout; data dir override.

## Tests

```bash
python -m pytest tests/          # public CI: synthetic fixtures only
python -m pytest eval/           # full evaluation on the supplied dataset
```

The eval suite needs `eval/expected.json` and `eval/situations.md`
(hackathon fixtures, intentionally not committed) and the patient dataset
under `data/patient-0000-0001/` (also not committed).

## Evaluation

See `EVALUATION.md` — 15/15 expected answers across the five supplied
situations, 157/157 (100%) passage traceability, 7 net contradictions, and a
slowest cold-start screen of 80 ms. Known limitations are listed there.

## Known limitations

- Deterministic extraction is tuned to German clinical letters; unusual
  layouts may fall back to keyword search only.
- The deterministic support double is conservative — it marks most
  translated claims "unchecked" rather than verifying them; enable the HTTP
  adapter for real claim-support assessment.
- n8n integration is intentionally not implemented (optional scope).
- No clinical validation; prototype only.

## Future work

- Patient-facing translations (e.g. Turkish, Arabic, Serbian) of a clinician-reviewed summary, labelled as machine translation pending review — not implemented; language support untested

## Verification

```bash
python3 -m pytest tests/ eval/ -q   # 47 passing checks
python3 evaluate.py --data-root /tmp/t01/data/patient-0000-0001
streamlit run app.py --server.port 8799 --server.headless true
```

Unresolved: none known at submission time.
