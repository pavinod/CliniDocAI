# CliniDocAI

**One screen with exactly what a clinician needs for the task in front of them — and
proof for every line.**

Hackathon Track 1 · Clinical Documentation · prototype on synthetic data, not for
clinical use.

---

## The problem

A patient's history is spread across many letters: discharge letters, outpatient
letters, lab and imaging reports, histology. For our synthetic patient, Anna
Musterfrau, there are eight German documents from 2015 and 2016.

Different clinicians need different things from the same pile:

- an **anaesthetist** before surgery needs blood thinners, allergies and heart function;
- a **surgeon** after the tumour board needs the tumour type, stage and the plan;
- an **emergency physician** at 2 a.m. needs to know whether this happened before and
  whether contrast dye is safe.

Today they either read everything or use a dashboard that lists *everything* about the
patient. Both cost time, and both make it easy to miss the one fact that matters — or
to trust a fact nobody can trace back to a document. Letters also contradict each
other (a stenosis on the left in one report and on the right in another), and nobody
points that out.

## Why we chose it

- **It is a safety problem, not a convenience problem.** Missing a blood thinner or
  a contrast allergy before a procedure can harm a patient.
- **Trust decides whether clinicians use a tool.** A summary is only useful if every
  line can be checked against the original letter in one click.
- **It must work for roles and tasks nobody wrote down in advance.** The jury tests
  new situations, and real clinics have more roles than five examples.

## What CliniDocAI does

1. **The clinician states who they are and what they are doing** — a role (anaesthetist,
   surgeon, GP, emergency physician, pharmacist, nurse, cardiologist, and others, or any
   role typed in) and a task in their own words. Speaking the task is possible when the
   optional voice module is installed.
2. **The top of the screen answers that role's three key questions** — for example, for
   the anaesthetist: *Which blood thinner, at what dose? Which allergy? What was the last
   heart ejection fraction?* If the letters do not say, the answer is **"Not documented"**.
3. **Contradictions appear directly under the answer they affect** — e.g. the
   lymph-node conflict sits under the surgeon's question about the tumour stage.
4. **Below that, all facts relevant to the role**, for clinicians who want to look at the
   underlying data. Allergies are always at the top.
5. **Every line has a Source button** that opens the original German letter with the
   exact sentence highlighted. German source wording is shown next to English facts.
6. **Print** produces a clean one-document report; **Export JSON** gives the same content
   with full source details for checking; **EN / DE** switches the interface language.

What it deliberately does **not** do: invent, guess, or "fill in" information. If a value
is not in the letters, it is shown as missing — never as normal.

---

## How CliniDocAI meets the Track 1 brief

| The brief asks for | How CliniDocAI does it |
|---|---|
| One screen with the facts a role and task need | Three key questions per role answered on top, then the role's sections. The anaesthetist sees age, blood thinner, allergy and heart function; tumour receptor status is left out. |
| Every fact links to its sentence or table row | Each fact stores the document, character position, a fingerprint (SHA-256) of the letter and the exact quote. The Source button highlights it. |
| "Not documented" when the documents lack something | Built into every question and section, e.g. *Renal function: not documented*, *no record that the follow-up carotid ultrasound took place*. A missing record is never shown as a normal result. |
| Point out contradictions | Seven checks run over the letters (see below). Each contradiction links all of its passages. |
| Only facts from the documents (translations, generic drug names, calculations allowed) | Facts are extracted from the letters and translated; brand names get their generic name (e.g. Glucophage → metformin); calculations (age, weight change, BMI check) show what they are based on. |
| Hard-coding is allowed only for *which kinds of facts* a role needs | The three questions per role and the sections per role are templates. The facts, passages and contradictions are always found in the letters: changing a value in a letter changes the screen. |
| The JSON files are only an earlier extraction; the documents count | All facts come from the letter text. The supplied JSON files are not used to decide what is shown. |
| Keep the `data` folder out of the repository | `data/`, the answer key and secrets are excluded by `.gitignore`. Test files use invented text, and a test checks they share no wording with the letters. |

### Points from the data notes we handle

- **Partial dates** (month and year only) stay partial; no day is invented.
- **Medication tables** ("1 0 0 0" = morning, midday, evening, night) become "1-0-0";
  dose and schedule changes between letters are shown, e.g. a blood-pressure drug that
  went from 1-0-½ to 1-0-1.
- **The empty lab tables** in the discharge letters lead to *"no additional laboratory
  values documented"* and *"HbA1c not documented after June 2015"*.
- **The "drug allergy" warning without a source** in the supplied JSON is not used; the
  contrast-media allergy shown comes from the letters, with its sentence.
- **Age:** the patient file states her age *today*; the screen shows her age at the
  reference date, 15 April 2016 (57 years), calculated from the birth date in the letters.
- **Abbreviations:** "St.p. CHE" is read as status after cholecystectomy, not chemotherapy.

---

## Results on the five supplied situations

Measured on 30 October 2026 on a MacBook (Apple silicon), Python 3.12, rule-based mode.

| Metric (target) | Result | How it was measured |
|---|---|---|
| Correct answers (15 / 15) | **15 / 15** | Each expected answer's key values appear on the screen without clicking, with a source document from the answer key. A second test checks that each answer sits in the matching question box (question 1 in box 1, and so on). |
| Traceability (100 %) | **189 / 189** facts carry a source with a verbatim quote | Every quote must exist character-for-character in the letter. A claim check also drops a quote that contradicts its fact (wrong number, unit or side). |
| Invented facts (0) | **None found in our review** of the key-question answers | Earlier errors we found and fixed: a biopsy shown with its report date, a planned operation shown with the letter date, a tumour board shown as done when it was only announced, an exam dated with the patient's birth date. A full human review of every line on all five screens is still recommended. |
| Contradictions (5 or more) | **7**, each linking all its passages | Listed below. |
| Load time (under 10 s) | **0.06 s** slowest screen (cold), 0.014 s average when cached | The letters are processed once and cached; each request only selects and checks facts. |

### The seven contradictions found

1. **Side of the carotid stenosis** — one paragraph of the June 2015 discharge letter
   places the 60 % stenosis on the right; every other passage says left.
2. **BMI does not fit height and weight** (June 2015 discharge letter): 78 kg and 170 cm
   give a BMI of 27.0, but the letter says 29.4.
3. **BMI does not fit height and weight** (April 2016 outpatient letter): 67 kg and 171 cm
   give 22.9, but the letter says 29.
4. **Two admission dates for the April 2016 stay**: 10 April and 12 April.
5. **Two case numbers for the same stay**: the April 2016 discharge letter carries the
   June 2015 case number, the histology report its own.
6. **Two dates for the same biopsy**: 12 April (discharge letter) vs 11 April
   (histology report).
7. **Lymph nodes in the armpit**: enlarged on examination (outpatient letter) vs none on
   ultrasound (discharge letter).

Not flagged on purpose: values that changed over time (weight, blood pressure),
small measurement differences (height 170 vs 171 cm), and differences between the
letters and the supplied JSON files.

### Testing it like the jury

1. In the sidebar, choose the **role** (or type it under *Other role*).
2. Paste the situation's **title and description** into **Task**.
3. Press **Show patient facts**. The three boxes on top answer the situation's questions.

Please choose or type the role: the app does not start without one, even when the
description already says "You are the anaesthetist".

---

## How it works

```
8 German letters ─► Normalise ─► Extract facts ─► Fact store (cached) ◄─ Find contradictions
                     (keeps a map back to       (13 rule-based          (7 checks)
                      the original characters)    extractors)
                                                        │
Role + task ─► Pick the 3 key questions and the role's sections ─► Check every claim
                                                                     against its quote
                                                        │
                                       Screen · Print report · JSON export
```

| Component | File | What it does |
|---|---|---|
| Interface | `app.py` | Streamlit page: sidebar input, three answer boxes, all facts, source viewer, print, JSON, EN/DE. |
| Reading the letters | `app/pipeline.py` | Loads the documents, decodes `<BR>` and HTML entities while keeping a map to the original characters, fingerprints each letter. |
| Finding facts | `app/extract.py`, `app/lexicon.py` | 13 rule-based extractors for German clinical letters (diagnoses, allergies, medication tables, measurements, histology, procedures, plans, history); negation, side and date handling; brand → generic drug names. |
| Finding contradictions | `app/contradictions.py` | 7 detectors: side, BMI recalculation, admission dates, case numbers per stay, procedure dates, present-vs-absent findings, tumour-marker values. |
| Choosing what to show | `app/needs.py`, `config/profiles.yaml` | Recognises the role, picks its three key questions and sections, keeps allergies first, shows the latest value of repeated measurements. |
| Checking and assembling | `app/screen.py`, `app/model.py` | Checks each claim against its quote, attaches contradictions to the answers, builds a typed screen; caches by document fingerprint and software version. |
| Outputs | `app/print_view.py`, `app/ui_helpers.py`, `app/i18n.py` | Print report, JSON export, German interface. |
| Voice (optional) | `app/voice.py`, `app/voice_ui.py` | Local speech-to-text with a live, editable transcript. |
| Evaluation | `evaluate.py`, `eval/`, `tests/` | The five situations, the 15 answers, and 170 automated tests. |

### Models and services

- **By default no AI model is used.** Facts, answers and contradictions come from
  deterministic rules, so every result can be traced and repeated.
- **Speech-to-text (optional):** OpenAI Whisper, running locally on the computer
  (`requirements-voice.txt`). The clinician can correct the transcript before it is used.
- **Claim check with a language model (optional, command line only):** any
  OpenAI-compatible model via OpenRouter (cloud) or LM Studio (local) can judge whether a
  quote supports a fact. A rule check for numbers, units and left/right runs first, and if
  the model fails the rules decide. It is switched on by exporting `MODEL_PROVIDER` and the
  matching key before running `evaluate.py` (variables listed in `.env.example`); the demo
  app runs rule-based.
- **LLM extraction for comparison (evaluation only):** `eval/compare_extractors.py` lets a
  model propose facts and keeps only those whose quote appears word for word in a letter,
  to compare against the rules — also on other corpora such as GraSCCo.
- The patient is synthetic, which is why cloud models are permitted; with real patient
  data we would use a local model only.

### Frameworks

Python 3.12 · Streamlit 1.61.1 · Pydantic 2.14.2 · PyYAML 6.0.3 · pytest 9.0.2 ·
optional: openai-whisper 20250625, sounddevice 0.5.6, soundfile 0.14.0.

---

## Run it

The `data` folder is not in this repository. Copy (or link) the challenge repository's
`data` folder into the project as `data/`.

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ln -s /path/to/challenge-repo/data data
streamlit run app.py
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
# copy the challenge repository's data folder to .\data\
streamlit run app.py
```

Optional voice input: `pip install -r requirements-voice.txt` (downloads PyTorch).

**Reproduce the metrics**: copy `cases/expected.json` and `cases/situations.md` from the
challenge repository into `eval/` (they stay out of git), then:

```bash
python evaluate.py --data-root data/patient-0000-0001
python -m pytest tests/ eval/ -q
```

---

## Limits — said plainly

- **Prototype on synthetic data.** Not clinically validated; not for patient care.
- **The rules are tuned to these German letters.** On unfamiliar letters the extraction
  is shallower — in a quick test it reported a penicillin allergy only as "Allergy
  recorded", and it shows a drug's brand name from its list even when the letter only
  gives the generic name.
- **Role first.** The app needs the role chosen or typed; it does not yet read the role
  from the task text alone.
- **Not yet run end to end with a real model:** the optional language-model claim check,
  the LLM extraction comparison, and live voice input are built and tested with stand-ins
  only. GraSCCo has not been downloaded or tested.
- **German mode:** interface text is translated; contradiction explanations and the key
  questions stay in English.

## Privacy and the hackathon rules

- Only the supplied synthetic patient is used. The `data` folder, the answer key and API
  keys are excluded from the repository.
- The app never reads the expected answers; a test fails if any app file refers to them.
- Text inside the letters is treated as data. An instruction planted in a test letter is
  ignored, and a test checks this.
