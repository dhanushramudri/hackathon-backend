"""Shared upsert-by-project_code CSV helper for the wizard's app-authored
steps (GDPR, Budget, Kickoff) -- these aren't derived from any source system,
so a single row per project_code, overwritten on every save, is enough."""
import pandas as pd

def upsert_row(csv_path, project_code: str, fields: dict) -> dict:
    row = {"project_code": project_code, **fields}
    row_str = {k: ("" if v is None else str(v)) for k, v in row.items()}
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        df = df[df["project_code"] != project_code]
        df = pd.concat([df, pd.DataFrame([row_str])], ignore_index=True)
    else:
        df = pd.DataFrame([row_str])
    df.to_csv(csv_path, index=False)
    return row

def get_row(csv_path, project_code: str) -> dict | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    match = df[df["project_code"] == project_code]
    if match.empty:
        return None
    return {k: (v if v != "" else None) for k, v in match.iloc[-1].to_dict().items()}
