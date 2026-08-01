from app.engines.role_hierarchy import TITLE_TO_LEVEL

_LEVEL_BY_LOWER_TITLE: dict[str, int] = {title.lower(): level for title, level in TITLE_TO_LEVEL.items()}

# One rate per canonical org level (app.engines.role_hierarchy) -- every title at the
# same level bills the same, regardless of which naming family (UK/USA vs India) it
# belongs to. Reuses today's real numbers at every level except Senior Software
# Engineer, which used to share a band with plain Software Engineer (both $45) despite
# being a different real level -- it now matches its true peer, Senior Associate
# Consultant, at $85.
LEVEL_RATES: dict[int, float] = {
    1: 25.0,
    2: 45.0,
    3: 85.0,
    4: 65.0,
    5: 70.0,
    6: 110.0,
    7: 145.0,
    8: 190.0,
    9: 240.0,
}

NON_BILLABLE_RATE = None

_NON_BILLABLE_TITLES = {
    "admin manager", "fp&a business partner", "fp&a manager", "it manager",
    "marketing manager", "office manager", "people partner", "resourcing manager",
    "senior hr leader consultant", "talent acquisition partner",
}

def get_hourly_rate(job_name) -> float | None:
    if not isinstance(job_name, str) or not job_name.strip():
        return NON_BILLABLE_RATE
    text = job_name.strip().lower()
    if text in _NON_BILLABLE_TITLES:
        return NON_BILLABLE_RATE
    level = _LEVEL_BY_LOWER_TITLE.get(text)
    if level is None:
        return NON_BILLABLE_RATE
    return LEVEL_RATES.get(level)

def get_rate_card(job_names: list[str]) -> list[dict]:
    seen = sorted(set(j for j in job_names if j))
    return [{"job_name": j, "hourly_rate_usd": get_hourly_rate(j), "source": "illustrative"} for j in seen]
