import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = BACKEND_ROOT / "data"
TRANSFORMED_DIR = DATA_ROOT / "Transformed"
PIPELINE_XLSX = TRANSFORMED_DIR / "07_Pipeline_Details.xlsx"

DUCKDB_PATH = ":memory:"

CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
