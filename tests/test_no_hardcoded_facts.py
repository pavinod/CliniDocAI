import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [
    r"\b1234\d{4}\b",
    r"15\.03\.1959",
    r"Musterfrau",
    r"Beispielpatientin",
    r"185/110",
    r"Allred score [0-9]",
    r"16 May 2016",
    r"T2 N0 M0",
    r"3 x 1\.5",
    r"20%",
    r"100%",
    r"70%",
    r"2015-06-28",
    r"2016-04-1[2-5]",
]


def test_application_code_has_no_patient_specific_fact_literals():
    application_files = sorted(
        [*ROOT.joinpath("app").rglob("*"), *ROOT.joinpath("clinidocai").rglob("*")])
    findings = []
    for path in application_files:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            if line.strip() == 'REFERENCE_DATE = "2016-04-15"':
                continue
            for pattern in FORBIDDEN:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not findings, "\n".join(findings)
