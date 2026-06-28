import re

_BANDS: list[tuple[str, float]] = [
    (r"\b(trainee|intern)\b", 25.0),
    (r"\b(associate partner|leadership)\b", 190.0),
    (r"\bpartner\b", 240.0),
    (r"\b(principal|technology solutions architect)\b", 145.0),
    (r"\b(senior solutions consultant|technical solutions architect|manager)\b", 110.0),
    (r"\bsenior software engineer\b", 65.0),
    (r"\bsenior associate consultant\b", 65.0),
    (r"\b(solutions consultant|senior consultant)\b", 85.0),
    (r"\bassociate consultant\b", 45.0),
    (r"\bconsultant\b", 65.0),
    (r"\b(solutions enabler|software engineer)\b", 45.0),
]

NON_BILLABLE_RATE = None

_NON_BILLABLE_TITLES = {
    "admin manager", "fp&a business partner", "fp&a manager", "it manager",
    "marketing manager", "office manager", "people partner", "resourcing manager",
    "senior hr leader consultant", "talent acquisition partner",
}

def get_hourly_rate(job_name) -> float | None:
    if not isinstance(job_name, str) or not job_name.strip():
        return NON_BILLABLE_RATE
    text = job_name.lower().strip()
    if text in _NON_BILLABLE_TITLES:
        return NON_BILLABLE_RATE
    for pattern, rate in _BANDS:
        if re.search(pattern, text):
            return rate
    return NON_BILLABLE_RATE

def get_rate_card(job_names: list[str]) -> list[dict]:
    seen = sorted(set(j for j in job_names if j))
    return [{"job_name": j, "hourly_rate_usd": get_hourly_rate(j), "source": "illustrative"} for j in seen]
