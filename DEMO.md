# Two-minute demo script — CliniDocAI

Setup (off camera): `streamlit run app.py`, data linked under `data/`,
browser at the app, interface in English, rule-based model.

**0:00–0:15 — Problem.**
"Clinicians get stacks of German letters. For a given role and task,
CliniDocAI turns them into one screen of sourced facts, missing information
and contradictions."

**0:15–0:40 — Situation 1.**
Role: Anaesthetist. Task: "Pre-operative anaesthesia check. She is scheduled
for a quadrantectomy with intraoperative radiotherapy on 16 May 2016."
Press **Show patient facts**. Point at: allergy pinned on top in red
(contrast media allergy), aspirin 100 mg 1-0-0, EF 60 % (June 2015), patient
57 years, reference date 2016-04-15, "Draft – please verify against sources".
Scroll to *Missing information*: renal function not documented — shown as
missing, not assumed normal.

**0:40–1:00 — Evidence.**
Click **Source** under the EF line → the discharge letter opens on the right
with the exact sentence highlighted. "Every line is pinned to its passage."

**1:00–1:20 — Contradictions.**
Show the carotid stenosis documented left in the duplex report but right in
one paragraph of the discharge letter (**Show passages**), and the biopsy
dated 11 vs 12 April. "Detected from the text, never hard-coded."

**1:20–1:40 — A new role.**
Role: Pharmacist, task: "Medication reconciliation before her surgery."
The screen reorganises: allergies, then the full medication list with doses,
then "renal function: not documented". Move a section with ↑/↓, *Save layout*.

**1:40–2:00 — Honest close.**
Switch to **DE** (interface in German, German source lines where safe),
press **Export JSON** (all facts with full source passages for validation).
"15/15 expected answers and 7 contradictions on the five situations, each
screen in under 0.1 s. Rule-based by default; an optional AI model checks
claims. Not done yet: the LLM-vs-rules comparison on GraSCCo, and live
voice was not tested end to end."
