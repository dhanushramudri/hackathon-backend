import re

from app.core.adapter import get_adapter

def get_revenue_trend() -> list[dict]:
    df = get_adapter().get_pipeline_revenue()
    col = df.columns[0]

    rows = []
    for raw in df[col].dropna():
        match = re.match(r"\s*([A-Za-z]+)\s*:\s*([\d.]+)\s*([KkMm]?)", str(raw))
        if not match:
            continue
        month_name, value, unit = match.groups()
        multiplier = {"K": 1_000, "M": 1_000_000}.get(unit.upper(), 1)
        rows.append({"month": month_name, "value": float(value) * multiplier, "raw": raw})

    return rows
