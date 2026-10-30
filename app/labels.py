"""Plain-language labels for source document types."""

import re


def doc_type_label(title: str) -> str:
    lowered = title.casefold()
    if re.search(r"24h-rr|langzeit-rr|blutdruckmessung", lowered):
        return "24-hour blood pressure report"
    if "patientenbrief" in lowered and "ambulant" in lowered:
        return "Outpatient letter"
    if ("patientenbrief" in lowered
            and re.search(r"\bstat\.?|\bstationär\b|\binterne\b", lowered)):
        return "Discharge letter"
    if re.search(r"histolog|patholog", lowered):
        return "Histology report"
    if re.search(r"langzeit[\s-]*ekg|\bekg\b", lowered):
        return "Long-term ECG report"
    if "echo" in lowered:
        return "Echocardiography report"
    if re.search(r"duplex|doppler|sono", lowered):
        return "Ultrasound report"
    if re.search(r"\bct\b|\bmrt\b|röntgen|radiolog", lowered):
        return "Imaging report"
    if re.search(r"konsil|augen", lowered):
        return "Consultation note"
    return title
