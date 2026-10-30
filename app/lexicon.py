"""Medical lexicon: drug identities, diagnosis translations, role/task terms.

Used to translate German document facts into English display text (allowed by
the task rules) and to map free-text clinical situations onto fact categories.
"""

import re

# drug name -> (INN / generic name, medication class key, english class label)
DRUGS = {
    "thrombo ass": ("acetylsalicylic acid", "antithrombotic", "antiplatelet"),
    "ass": ("acetylsalicylic acid", "antithrombotic", "antiplatelet"),
    "acetylsalicyl": ("acetylsalicylic acid", "antithrombotic", "antiplatelet"),
    "aspirin": ("acetylsalicylic acid", "antithrombotic", "antiplatelet"),
    "glucophage": ("metformin", "antidiabetic", "glucose-lowering"),
    "metformin": ("metformin", "antidiabetic", "glucose-lowering"),
    "actos": ("pioglitazone", "antidiabetic", "glucose-lowering"),
    "pioglitazon": ("pioglitazone", "antidiabetic", "glucose-lowering"),
    "pioglitazone": ("pioglitazone", "antidiabetic", "glucose-lowering"),
    "byetta": ("exenatide", "antidiabetic", "glucose-lowering"),
    "exenatid": ("exenatide", "antidiabetic", "glucose-lowering"),
    "exenatide": ("exenatide", "antidiabetic", "glucose-lowering"),
    "blopress": ("candesartan", "antihypertensive", "blood-pressure lowering"),
    "candesartan": ("candesartan", "antihypertensive", "blood-pressure lowering"),
    "zocor": ("simvastatin", "lipid", "lipid-lowering"),
    "simvastatin": ("simvastatin", "lipid", "lipid-lowering"),
    "oleovit": ("cholecalciferol", "supplement", "vitamin-D supplement"),
    "vitamin d": ("cholecalciferol", "supplement", "vitamin-D supplement"),
    "alendronat": ("alendronate", "bone", "osteoporosis treatment"),
    "alendronate": ("alendronate", "bone", "osteoporosis treatment"),
    "nitro": ("nitroglycerin", "nitrate", "acute antihypertensive"),
    "nitroglycerin": ("nitroglycerin", "nitrate", "acute antihypertensive"),
}

# ordered: longer names first so 'thrombo ass' wins over 'ass'
DRUG_NAMES = sorted(DRUGS, key=len, reverse=True)

# German diagnosis keywords -> english label
DIAGNOSIS_EN = [
    (r"invasiv duktal", "Invasive ductal carcinoma"),
    (r"mammakarzinom|mamma-?ca", "Breast cancer"),
    (r"hypertensiv\w* krise", "Hypertensive crisis"),
    (r"amaurosis fugax", "Amaurosis fugax (transient vision loss)"),
    (r"carotissklerose|karotisstenose|carotisstenose", "Carotid artery stenosis"),
    (r"gastroösophageale reflux|gerd", "Gastro-oesophageal reflux disease (GERD)"),
    (r"hiatushernie", "Hiatus hernia"),
    (r"lupus|\bsle\b", "Systemic lupus erythematosus (SLE)"),
    (r"diab", "Diabetes mellitus type 2"),
    (r"hypertonie|art\.\s*htn|art\.\s*hypertonie", "Arterial hypertension"),
    (r"vitamin-?d-?mangel", "Vitamin-D deficiency"),
    (r"osteoporose", "Osteoporosis"),
    (r"hyperlipidämie|hlp", "Hyperlipidaemia"),
    (r"hypercholesterinämie", "Hypercholesterolaemia"),
    (r"schlafapnoe", "Sleep apnoea"),
    (r"appendektom\w*", "Status post appendectomy"),
    (r"st\.?\s*p\.?\s*che", "Status post cholecystectomy"),
]

FINDINGS_EN = [
    (r"\bkeine\s+(?:anzeichen\s+für\s+(?:eine\s+)?)?blutung\b",
     "no haemorrhage"),
]


def finding_translations_en(text: str) -> list[str]:
    """Return configured English labels for recognised source findings."""
    return [label for pattern, label in FINDINGS_EN
            if re.search(pattern, text, re.IGNORECASE)]

ALLERGY_REACTIONS_EN = {
    "übelkeit": "nausea",
    "schwindel": "dizziness",
    "erbrechen": "vomiting",
}


def allergy_reactions_en(text: str) -> list[str]:
    return [label for source, label in ALLERGY_REACTIONS_EN.items()
            if re.search(re.escape(source), text, re.IGNORECASE)]


# diagnosis line -> subcategory
def diagnosis_subcategory(line: str) -> str:
    l = line.lower()
    if any(k in l for k in ("mammakarzinom", "mamma-ca", "histolog", "hormonrezeptor", "her2", "ki67", "tumor", "karzinom")):
        return "oncology"
    if any(k in l for k in ("karotis", "carotis", "stenose", "amaurosis")):
        return "vascular"
    if any(k in l for k in ("hypertonie", "hypertensive", "htn")):
        return "cardiac"
    if any(k in l for k in ("diabetes", "hyperlipid", "hypercholesterin", "vitamin", "osteoporose")):
        return "metabolic"
    return "other"


# role detection: pattern -> role key
ROLES = [
    (r"anaesth|anesth|narkose|anästhesi", "anaesthetist"),
    (r"surgeon|chirurg|surgical", "surgeon"),
    (r"general practitioner|\bgp\b|family (doctor|physician|practitioner)|hausarzt|primary care", "gp"),
    (r"emergency|notfall|\bed\b|\ber\b|night (duty|shift)", "emergency"),
    (r"oncolog|onkolog|tumou?r ?board", "oncologist"),
    (r"cardiolog|kardiolog", "cardiologist"),
    (r"radiolog", "radiologist"),
    (r"diabetolog|endocrinolog", "diabetologist"),
    (r"nurse|pflege", "nurse"),
    (r"pharmacist|apothek", "pharmacist"),
    (r"ophthalmolog|augen(arzt|heilkunde)|eye doctor", "ophthalmologist"),
    (r"neurolog", "neurologist"),
    (r"patholog", "pathologist"),
    (r"internist", "internist"),
    (r"gynaecolog|gynecolog|gynäkolog", "gynaecologist"),
    (r"urolog", "urologist"),
    (r"orthop", "orthopaedist"),
    (r"physiotherap|physio", "physiotherapist"),
    (r"dietitian|nutritionist|diätolog|ernährungsberat", "dietitian"),
    (r"psychiatr|psycholog", "psychiatrist"),
    (r"dentist|dental|zahnarzt|zahnärzt", "dentist"),
]

# task keyword -> extra need keys (additive on top of the role's needs)
TASK_NEEDS = [
    (r"pre-?op|preoperat|voroper|anaesth|anesth|narkose", ["anaesthesia"]),
    (r"tumou?r ?board|operation|surgery|op\b|resektion|quadrantectomy|mastectom", ["surgery_planning"], ["surgeon", "oncologist", "gynaecologist", "radiologist", "pathologist"]),
    (r"tumou?r ?board|operation|surgery|op\b|resektion|quadrantectomy|mastectom", ["surgery_date"]),
    (r"admission|aufnahme|admit", ["admission"]),
    (r"diabet|glucose|blood sugar|hba1c|blutzucker", ["diabetes"]),
    (r"contrast|dye|ct scan|angiograph|km-?|jod", ["contrast"]),
    (r"dizziness|schwindel|vision|seh|stroke|schlaganfall|tia\b|neurolog|amaurosis|visual|eye|blind", ["neuro_emergency"]),
    (r"follow-?up|nachsorge|kontroll|discharge|entlass", ["followup"]),
    (r"blood ?thinner|anticoag|antithromb|antiplatelet|marcumar|gerinnung", ["antithrombotic"]),
    (r"weight|gewicht|obesity|übergewicht", ["weight"]),
    (r"heart|herz|cardiac|ejection|ejektion|echo", ["cardiac"]),
    (r"kidney|renal|niere|kreatinin|creatinine|gfr", ["renal"]),
    (r"allerg|unverträglich", ["allergy"]),
    (r"hyperten|blood pressure|blutdruck", ["blood_pressure"]),
    (r"infection|infekt|antibiot|sepsis|fieber|fever", ["infection"]),
    (r"pain|schmerz", ["pain"]),
    (r"pregnan|schwanger", ["pregnancy"]),
    (r"smok|rauch|alcohol|alkohol", ["lifestyle"]),
    (r"driving|drive|fahrtüchtig|fitness to (work|fly)", ["functional"]),
]
