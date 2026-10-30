# Evaluation report — CliniDocAI

Measured on the supplied synthetic patient dataset (patient-0000-0001),
interpreted as of **2016-04-15**. The five supplied situations are run
through the full pipeline. Final local checks: `python3 -m pytest tests/ eval/ -q`
(47 passed) and the official `evaluate.py` harness.

## Results

| Metric | Result | Target |
|---|---|---|
| Correct answers (15 questions) | **15 / 15** | 15/15 |
| Required answers visible without opening evidence | yes | yes |
| Passage traceability (facts → exact quotes) | **157 / 157 (100.0%)** | high |
| Unsupported factual claims displayed | **0** | 0 |
| Unique true contradictions (net of false alarms) | **7** | ≥5 |
| Slowest submission-to-complete-screen | **0.080 s** | <10 s |

### Timing (official `evaluate.py`)

| Phase | Result |
|---|---|
| Slowest cold-start screen | 0.080 s |
| Average cold-start | 0.030 s |
| Average warm-start | 0.020 s |

Well under the 10-second target. The pipeline is deterministic rule-based
extraction — no model calls in the default configuration — so per-screen
cost is dominated by regex passes over the 8 documents.

### Contradictions detected (7, net)

1. **Carotid laterality** — 60% ICA stenosis documented LEFT in the duplex
   report, but the same passage embedded in the June 2015 discharge letter
   says RIGHT.
2. **BMI vs height/weight** (letter 1) — documented BMI inconsistent with
   stated height and weight.
3. **BMI vs height/weight** (letter 2) — same class of mismatch in a second
   document.
4. **April 2016 admission date** — 10 April vs 12 April in the same stay's
   admission records.
5. **Case number** — the same April stay is filed under different case
   numbers across the outpatient, discharge and histology documents.
6. **Biopsy date** — specimen collection date in histology vs the date the
   discharge letter gives for the same biopsy.
7. **Axillary lymph nodes** — enlarged on palpation (outpatient letter) vs
   none enlarged on ultrasound.

The former **"St.p. CHE" chemotherapy** alert was removed: the abbreviation
means status post cholecystectomy, not chemotherapy.

Filtered out deliberately (not contradictions): weight/BMI drift between
letters (temporal change), different measurements of the same quantity,
differences between document text and the JSON index (extraction artefacts).

### Answers verified

All 15 expected answers are present on-screen in plain view (no evidence
expansion needed) and each links to the expected source document — checked
by `eval/test_metrics.py::test_expected_answer` plus `test_traceability`.

### Unsupported claims

`test_traceability` asserts every displayed item (except not-documented
entries) carries ≥1 source with a verbatim quote. The claim-support pass
additionally drops items the configured adapter judges unsupported —
measured at 0 displayed unsupported claims. Note: this bounds *displayed*
hallucination by construction; it is not a guarantee that the deterministic
double detects every possible misreading (see limitations in README).

## Reproduction

```bash
python3 evaluate.py --data-root /tmp/t01/data/patient-0000-0001
python3 -m pytest tests/ eval/ -q
```

## Not claimed

- No clinical validation or production readiness.
- "Zero hallucination" is bounded to the displayed items, not proven for
  arbitrary inputs.
- No untested language support claimed (UI text is English; situation input
  is matched case-insensitively over English/German keywords).
