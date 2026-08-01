import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = BACKEND_ROOT / "data"
TRANSFORMED_DIR = DATA_ROOT / "Transformed"
PIPELINE_XLSX = TRANSFORMED_DIR / "07_Pipeline_Details.xlsx"
HEADCOUNT_PREDICTION_DIR = DATA_ROOT / "HeadcountPrediction"
# Data the app itself originates (GDPR/budget/SOW/kickoff wizard steps) rather
# than derives from a source system -- kept separate from Transformed so it's
# obvious at a glance which files are pipeline output vs. app-authored state.
APP_STATE_DIR = DATA_ROOT / "AppState"
APP_STATE_DIR.mkdir(exist_ok=True)

DUCKDB_PATH = ":memory:"

CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
