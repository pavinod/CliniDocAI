"""EN/DE UI strings. Keys are the English source text; DE maps them.

Clinical content extracted from the record (fact text, contradiction
titles/details, German "original" quotes) is intentionally not translated —
it stays in the language it was extracted in. Only the interface chrome is
bilingual.
"""

from __future__ import annotations

import re

LANGUAGES = {"en": "EN", "de": "DE"}

_ENGLISH_TO_GERMAN_MONTH = {
    "January": "Januar",
    "February": "Februar",
    "March": "März",
    "April": "April",
    "May": "Mai",
    "June": "Juni",
    "July": "Juli",
    "August": "August",
    "September": "September",
    "October": "Oktober",
    "November": "November",
    "December": "Dezember",
}

DE: dict[str, str] = {
    # ---------------------------------------------------------- sidebar
    "Role": "Rolle",
    "Other (type below)": "Andere (unten eingeben)",
    "Other role (optional)": "Andere Rolle (optional)",
    "Enter another clinical role": "Andere klinische Rolle eingeben",
    "Task": "Aufgabe",
    "What should this clinician review?": "Was soll dieser Kliniker prüfen?",
    "Voice input": "Spracheingabe",
    "Voice transcript — please check before use":
        "Sprach-Transkript — bitte vor Verwendung prüfen",
    "Voice transcription unavailable: {error}":
        "Sprachtranskription nicht verfügbar: {error}",
    "Examples": "Beispiele",
    "Choose an example…": "Beispiel wählen…",
    "Show patient facts": "Patientenfakten anzeigen",
    "Settings": "Einstellungen",
    "Text size (%)": "Schriftgröße (%)",
    "Reference date": "Referenzdatum",
    "Voice language": "Sprache der Spracheingabe",
    "English": "Englisch",
    "German": "Deutsch",
    "Voice input unavailable (install requirements-voice.txt)":
        "Spracheingabe nicht verfügbar (requirements-voice.txt installieren)",
    # ---------------------------------------------------------- run/messages
    "Reading documents…": "Dokumente werden gelesen…",
    "Role changed — enter a task for the new role":
        "Rolle geändert — bitte eine Aufgabe für die neue Rolle eingeben",
    "Enter a task before showing patient facts.":
        "Bitte eine Aufgabe eingeben, bevor Patientenfakten angezeigt werden.",
    "Choose a role or enter another role.":
        "Bitte eine Rolle wählen oder eine andere Rolle eingeben.",
    "Choose a role and task in the sidebar to view source-supported facts.":
        "Rolle und Aufgabe in der Seitenleiste wählen, um quellenbelegte "
        "Fakten anzuzeigen.",
    # ---------------------------------------------------------- content
    "Relevant facts": "Relevante Fakten",
    "Missing information": "Fehlende Informationen",
    "Contradictions": "Widersprüche",
    "Original source": "Originalquelle",
    "Select a fact's source to open the highlighted original.":
        "Quelle eines Fakts wählen, um das hervorgehobene Original zu öffnen.",
    "Draft – please verify against sources":
        "Entwurf – bitte anhand der Quellen prüfen",
    "Showing facts for: {role} — {task}":
        "Angezeigte Fakten für: {role} — {task}",
    "Source": "Quelle",
    "Sources ({n} documents)": "Quellen ({n} Dokumente)",
    "Show passages ({n})": "Textstellen anzeigen ({n})",
    "Document IDs: {ids}": "Dokumenten-IDs: {ids}",
    "Selected passage:": "Ausgewählte Textstelle:",
    "not documented": "nicht dokumentiert",
    "no record found": "kein Eintrag gefunden",
    "No additional laboratory values documented":
        "Keine weiteren Laborwerte dokumentiert",
    "HbA1c: not documented": "HbA1c: nicht dokumentiert",
    "Follow-up carotid duplex ultrasound: not documented as performed":
        "Kontroll-Duplexsonographie der Karotiden: nicht als durchgeführt "
        "dokumentiert",
    "on board, check protocol around iodinated contrast":
        "eingenommen – Vorgehen bei jodhaltigem Kontrastmittel prüfen",
    "done without contrast because of the allergy":
        "wegen der Allergie ohne Kontrastmittel durchgeführt",
    "{n} more": "{n} weitere",
    "original": "Original",
    "Key questions for this role": "Kernfragen für diese Rolle",
    "Q": "F",
    "All facts for this role": "Alle Fakten für diese Rolle",
    "The full documented data behind the answers above.":
        "Die vollständigen dokumentierten Daten hinter den Antworten oben.",
    "Other contradictions in the documents": "Weitere Widersprüche in den Dokumenten",
    "Clinical summary": "Klinische Zusammenfassung",
    "Patient": "Patientin",
    "Printed": "Gedruckt",
    "Section": "Abschnitt",
    "Fact": "Fakt",
    "Date": "Datum",
    "Generated from the supplied documents only. Draft pending clinician "
    "review — not for clinical use.":
        "Ausschließlich aus den vorliegenden Dokumenten erstellt. Entwurf, "
        "ärztliche Prüfung ausstehend — nicht für den klinischen Einsatz.",
    "🖨 Print": "🖨 Drucken",
    "Print this screen with sources and review status":
        "Diese Ansicht mit Quellen und Prüfstatus drucken",
    "⬇ Export JSON": "⬇ JSON exportieren",
    "All displayed facts with full source passages, for validation":
        "Alle angezeigten Fakten mit vollständigen Quellstellen, zur Prüfung",
    "JSON output (for validation)": "JSON-Ausgabe (zur Prüfung)",
    "Smaller text": "Kleinere Schrift",
    "Larger text": "Größere Schrift",
    "AI model (claim check)": "KI-Modell (Aussageprüfung)",
    "Model help":
        "Regelbasiert ist der Standard. Ein KI-Modell prüft zusätzlich, ob "
        "jede zitierte Textstelle die Aussage stützt.",
    "Rule-based only (no AI model)": "Nur regelbasiert (kein KI-Modell)",
    "LM Studio (local model)": "LM Studio (lokales Modell)",
    "Cloud models receive the cited passages. Only use them with synthetic data.":
        "Cloud-Modelle erhalten die zitierten Textstellen. Nur mit "
        "synthetischen Daten verwenden.",
    "AI model not used ({problem}) — showing rule-based results.":
        "KI-Modell nicht verwendet ({problem}) — regelbasierte Ergebnisse "
        "werden angezeigt.",
    "Layout": "Layout",
    "Save layout": "Layout speichern",
    "Reset layout": "Layout zurücksetzen",
    "Layout saved at {time}": "Layout gespeichert um {time}",
    "Allergies always stay at the top.": "Allergien bleiben immer oben.",
    "Layout preferences file: {path}": "Datei mit Layout-Einstellungen: {path}",
    "Inputs changed — press Show patient facts to update the screen.":
        "Eingaben geändert — „Patientenfakten anzeigen“ drücken, um die "
        "Ansicht zu aktualisieren.",
    "🎙 Start live voice input": "🎙 Live-Spracheingabe starten",
    "Listening… {n} s": "Aufnahme läuft… {n} s",
    "Live transcript": "Live-Transkript",
    "⏹ Stop and use as task": "⏹ Stoppen und als Aufgabe übernehmen",
    "Female": "Weiblich",
    "Male": "Männlich",
    "Move section up": "Abschnitt nach oben",
    "Move section down": "Abschnitt nach unten",
    "Download JSON": "JSON herunterladen",
    "Language": "Sprache",
    "years": "Jahre",
    "born": "geb.",
    "Age: not documented": "Alter: nicht dokumentiert",
    "needed for": "benötigt für",
    # ---------------------------------------------------------- about
    "About this system": "Über dieses System",
    "Generated by: rule-based extraction (no AI model)":
        "Erstellt durch: regelbasierte Extraktion (kein KI-Modell)",
    "Pending clinician review": "Ausstehende ärztliche Prüfung",
    "Load time": "Ladezeit",
    "Facts": "Fakten",
    "Dropped unsupported spans": "Verworfene nicht belegte Textstellen",
    "not yet loaded": "noch nicht geladen",
    "Versions": "Versionen",
    "Documents are treated as untrusted input; instructions inside them "
    "are never executed.":
        "Dokumente werden als nicht vertrauenswürdige Eingabe behandelt; "
        "darin enthaltene Anweisungen werden niemals ausgeführt.",
    # ---------------------------------------------------------- statuses
    "negative": "negativ",
    "suspected": "Verdacht",
    "historical": "anamnestisch",
    "recommended": "empfohlen",
    "completed": "abgeschlossen",
    "planned": "geplant",
    "positive": "positiv",
    # ---------------------------------------------------------- section titles
    "Allergies & warnings": "Allergien & Warnhinweise",
    "Current medication (latest list)": "Aktuelle Medikation (neueste Liste)",
    "Antithrombotic medication": "Antithrombotische Medikation",
    "Glucose-lowering medication": "Blutzuckersenkende Medikation",
    "Blood-pressure medication": "Blutdruckmedikation",
    "Before iodinated contrast": "Vor jodhaltigem Kontrastmittel",
    "Cardiac function": "Herzfunktion",
    "Vitals & measurements (latest each)":
        "Vitalwerte & Messungen (jeweils aktuellste)",
    "Weight course": "Gewichtsverlauf",
    "HbA1c": "HbA1c",
    "Laboratory results": "Laborwerte",
    "Diagnoses": "Diagnosen",
    "Tumour diagnosis & histology": "Tumordiagnose & Histologie",
    "Carotid stenosis": "Karotisstenose",
    "Previous episode": "Frühere Episode",
    "Examinations & procedures": "Untersuchungen & Eingriffe",
    "Breast imaging & biopsy": "Mamma-Bildgebung & Biopsie",
    "Recommended follow-ups": "Empfohlene Nachsorge",
    "Treatment plan": "Behandlungsplan",
    "Hospital stays & documents": "Klinikaufenthalte & Dokumente",
    "Renal function": "Nierenfunktion",
    "Anaesthesia history": "Anästhesie-Vorgeschichte",
    "Airway / aspiration risk notes": "Atemweg- / Aspirationsrisiko",
    "Lifestyle (smoking, alcohol)": "Lebensstil (Rauchen, Alkohol)",
    "Infections & antibiotic therapy": "Infektionen & Antibiotikatherapie",
    "Pain": "Schmerzen",
    "Pregnancy": "Schwangerschaft",
    "Functional status": "Funktionsstatus",
    "Oral anticoagulation (Marcumar/DOAK/INR)":
        "Orale Antikoagulation (Marcumar/DOAK/INR)",
    "Blood group / antibodies": "Blutgruppe / Antikörper",
    "Also relevant": "Ebenfalls relevant",
    # ---------------------------------------------------------- not documented
    "No allergy documented": "Keine Allergie dokumentiert",
    "No antithrombotic medication documented":
        "Keine antithrombotische Medikation dokumentiert",
    "No glucose-lowering medication documented":
        "Keine blutzuckersenkende Medikation dokumentiert",
    "No antihypertensive medication documented":
        "Keine Blutdruckmedikation dokumentiert",
    "Renal function (creatinine/eGFR): not documented":
        "Nierenfunktion (Kreatinin/eGFR): nicht dokumentiert",
    "Previous anaesthesia: not documented":
        "Frühere Anästhesie: nicht dokumentiert",
    "No airway assessment documented":
        "Keine Atemwegbeurteilung dokumentiert",
    "Smoking/alcohol: not documented": "Rauchen/Alkohol: nicht dokumentiert",
    "No infection or antibiotic therapy documented":
        "Keine Infektion oder Antibiotikatherapie dokumentiert",
    "Pain assessment: not documented": "Schmerzbeurteilung: nicht dokumentiert",
    "Pregnancy status: not documented":
        "Schwangerschaftsstatus: nicht dokumentiert",
    "Functional assessment: not documented":
        "Funktionsbeurteilung: nicht dokumentiert",
    "No oral anticoagulation documented":
        "Keine orale Antikoagulation dokumentiert",
    "Blood group: not documented": "Blutgruppe: nicht dokumentiert",
    # ---------------------------------------------------------- doc types
    "24-hour blood pressure report": "24-h-Blutdruckmessung",
    "Outpatient letter": "Ambulanter Arztbrief",
    "Discharge letter": "Entlassbrief",
    "Histology report": "Histologiebefund",
    "Long-term ECG report": "Langzeit-EKG-Befund",
    "Echocardiography report": "Echokardiographie-Befund",
    "Ultrasound report": "Ultraschallbefund",
    "Imaging report": "Bildgebungsbefund",
    "Consultation note": "Konsiliarbericht",
    # ---------------------------------------------------------- derived
    "Calculated from {n} documented weights":
        "Berechnet aus {n} dokumentierten Gewichten",
    "Calculated from documented weight and height":
        "Berechnet aus dokumentiertem Gewicht und Größe",
    "Calculated from the documented birth date":
        "Berechnet aus dem dokumentierten Geburtsdatum",
    "Calculated value": "Berechneter Wert",
    # ---------------------------------------------------------- contradictions
    "Left/right mismatch": "Links/rechts widersprüchlich",
    "Values don't add up": "Werte widersprüchlich",
    "Case number mismatch": "Fallnummer widersprüchlich",
    "Finding present vs absent": "Befund vorhanden vs. fehlend",
    "Timeline inconsistency": "Zeitlicher Widerspruch",
    "Date mismatch": "Datums-Widerspruch",
    # ---------------------------------------------------------- roles
    "Anaesthetist": "Anästhesist",
    "Surgeon": "Chirurg",
    "GP": "Hausarzt",
    "Emergency": "Notfallmediziner",
    "Oncologist": "Onkologe",
    "Cardiologist": "Kardiologe",
    "Radiologist": "Radiologe",
    "Diabetologist": "Diabetologe",
    "Nurse": "Pflegekraft",
    "Pharmacist": "Apotheker",
    "Ophthalmologist": "Augenarzt",
    "Neurologist": "Neurologe",
    "Pathologist": "Pathologe",
    "Internist": "Internist",
    "Gynaecologist": "Gynäkologe",
    "Urologist": "Urologe",
    "Orthopaedist": "Orthopäde",
    "Physiotherapist": "Physiotherapeut",
    "Dietitian": "Diätologe",
    "Psychiatrist": "Psychiater",
    "Dentist": "Zahnarzt",
    "Clinician": "Kliniker",
    # ---------------------------------------------------------- examples
    "Review relevant findings and risks before the scheduled operation.":
        "Relevante Befunde und Risiken vor der geplanten Operation prüfen.",
    "Review the available findings and proposed plan for the scheduled "
    "operation.":
        "Verfügbare Befunde und den vorgeschlagenen Plan für die geplante "
        "Operation prüfen.",
    "Review recent treatment, measurements, and follow-up needs.":
        "Aktuelle Behandlung, Messwerte und Nachsorgebedarf prüfen.",
    "Review prior episodes and relevant findings for the current urgent "
    "presentation.":
        "Frühere Episoden und relevante Befunde für die aktuelle akute "
        "Vorstellung prüfen.",
    "Review open follow-up items from the previous hospital stay.":
        "Offene Nachsorgepunkte aus dem früheren Klinikaufenthalt prüfen.",
    "Pre-operative anaesthesia check": "Präoperative Anästhesie-Prüfung",
    "Surgery planning after the tumour board, April 2016":
        "Operationsplanung nach dem Tumorboard, April 2016",
    "Diabetes review at the GP": "Diabetes-Kontrolle beim Hausarzt",
    "Emergency department at night": "Notaufnahme in der Nacht",
    "Open follow-ups from the 2015 hospital stay":
        "Offene Nachsorge aus dem Krankenhausaufenthalt 2015",
    "Task description": "Aufgabenbeschreibung",
    "Choose a task…": "Aufgabe wählen…",
    "Medication reconciliation": "Medikationsabgleich",
    "Check the documented medicines and identify reconciliation issues "
    "before the operation.":
        "Die dokumentierten Medikamente prüfen und Abgleichprobleme vor der "
        "Operation identifizieren.",
    "Pre-op cardiovascular review": "Präoperative kardiovaskuläre Beurteilung",
    "Review available findings and relevant risks before the planned "
    "operation.":
        "Verfügbare Befunde und relevante Risiken vor der geplanten "
        "Operation prüfen.",
}


def current_lang() -> str:
    """UI language from Streamlit session state; 'en' outside Streamlit."""
    try:
        import streamlit as st
        lang = st.session_state.get("ui_lang", "en")
    except Exception:
        lang = "en"
    return lang if lang in LANGUAGES else "en"


def t(key: str, **kwargs) -> str:
    """Translate a UI string; English (the key) is the fallback."""
    lang = current_lang()
    if lang == "en":
        text = key
    else:
        text = DE.get(key)
        if text is None:
            match = re.fullmatch(
                r"HbA1c not documented after (\w+) (\d{4})", key)
            month = _ENGLISH_TO_GERMAN_MONTH.get(match.group(1)) if match else None
            text = (f"HbA1c nach {month} {match.group(2)} nicht dokumentiert"
                    if match and month else key)
    return text.format(**kwargs) if kwargs else text


def role_label(display: str) -> str:
    """Localized label for a role dropdown value / built role name."""
    if display == "Other (type below)":
        return t("Other (type below)")
    pretty = display.replace("_", " ")
    pretty = pretty.upper() if len(pretty) <= 3 else pretty.title()
    return t(pretty)
