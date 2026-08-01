import pandas as pd

from app.core.config import APP_STATE_DIR

SOW_UPLOADS_DIR = APP_STATE_DIR / "sow_uploads"
SOW_METADATA_CSV = APP_STATE_DIR / "project_sow.csv"

def save_sow_file(project_code: str, filename: str, content: bytes) -> dict:
    project_dir = SOW_UPLOADS_DIR / project_code
    project_dir.mkdir(parents=True, exist_ok=True)
    file_path = project_dir / filename
    file_path.write_bytes(content)

    uploaded_at = pd.Timestamp.now().isoformat()
    row = {"project_code": project_code, "filename": filename, "size_bytes": len(content), "uploaded_at": uploaded_at}
    if SOW_METADATA_CSV.exists():
        df = pd.read_csv(SOW_METADATA_CSV, dtype=str)
        df = df[~((df["project_code"] == project_code) & (df["filename"] == filename))]
        df = pd.concat([df, pd.DataFrame([{k: str(v) for k, v in row.items()}])], ignore_index=True)
    else:
        df = pd.DataFrame([{k: str(v) for k, v in row.items()}])
    df.to_csv(SOW_METADATA_CSV, index=False)
    return row

def list_sow_files(project_code: str) -> list[dict]:
    if not SOW_METADATA_CSV.exists():
        return []
    df = pd.read_csv(SOW_METADATA_CSV, dtype=str)
    rows = df[df["project_code"] == project_code].sort_values("uploaded_at", ascending=False)
    return [
        {"filename": r["filename"], "size_bytes": int(r["size_bytes"]), "uploaded_at": r["uploaded_at"]}
        for _, r in rows.iterrows()
    ]

def get_sow_file_path(project_code: str, filename: str):
    path = SOW_UPLOADS_DIR / project_code / filename
    return path if path.exists() else None
