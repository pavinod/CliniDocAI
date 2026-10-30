# Two-minute demo script — CliniDocAI

Setup (off camera): `streamlit run app.py`, dataset loaded, browser
at the app.

**0:00–0:15 — Problem.**
"Clinicians get stacks of letters. For a given role and task, CliniDocAI
distils them into one screen of sourced facts, missing info, and
contradictions."

**0:15–0:35 — Situation input.**
Paste: "You are the anaesthetist. She is scheduled for a quadrantectomy with
intraoperative radiotherapy on 16 May 2016." Press *Build my screen*.
Point at the header: detected role *anaesthetist*, reference date
2016-04-15, "generated draft — pending clinician review" banner.

**0:35–0:55 — Relevant facts.**
Scroll the sections — allergies (contrast media allergy), antithrombotics
(Thrombo ASS → acetylsalicylic acid, 1-0-0), cardiac, airway, renal.
Note *not documented* badges (no eGFR, no prior anaesthesia) — missing data
is shown as missing, not assumed normal.

**0:55–1:15 — Evidence.**
Click a fact's source button → the original German letter opens on the
right with the exact sentence highlighted. "Every claim is pinned to its
verbatim passage — nothing invented."

**1:15–1:40 — Contradictions.**
Open Contradictions: the carotid stenosis is left in the duplex report but
right in the discharge letter; the April admission has two dates; the same
stay carries two case numbers; "st.p. chemotherapy" predates the cancer
diagnosis. Click a passage to jump to it.

**1:40–2:00 — Other roles + close.**
Switch to "You are the GP reviewing her diabetes" → the screen reorganises
(meds, HbA1c missing since June 2015, weight trend). Mention: config-driven
profiles, content-hash cache, PDF export keeps provenance and review
status. Close: "Deterministic pipeline — every screen in tens of
milliseconds, 15/15 expected answers, zero invented facts."
