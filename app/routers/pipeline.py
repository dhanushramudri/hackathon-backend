import pandas as pd
from fastapi import APIRouter

from app.core.adapter import get_adapter
from app.engines.skillset_classifier import classify_skillset

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

LATE_NOTICE_THRESHOLD_DAYS = 14

@router.get("/forecast")
def forecast() -> list[dict]:
    df = get_adapter().get_pipeline_forecast().reset_index().rename(columns={"index": "row_index"})
    cols = [
        "row_index", "deal_id", "cluster", "client", "client_priority", "em", "solution", "status", "priority",
        "resources_requested", "requested_pct", "skillset", "request_received", "original_requested_start_date",
        "request_type", "start_date_confirmed", "number_of_weeks", "deal_stage_hubspot", "comments",
        "likely_start_date", "sow_signed",
    ]
    df = df[cols].copy()

    notice_days_raw = (df["likely_start_date"] - df["request_received"]).dt.days
    notice_days = [int(d) if pd.notna(d) else None for d in notice_days_raw]
    is_late_notice = [(d < LATE_NOTICE_THRESHOLD_DAYS) if d is not None else None for d in notice_days]

    date_cols = {}
    for date_col in ["likely_start_date", "request_received", "original_requested_start_date"]:
        date_cols[date_col] = [d.strftime("%Y-%m-%d") if pd.notna(d) else None for d in df[date_col]]

    df["skillset_coe_categories"] = df["skillset"].apply(classify_skillset)
    records = df.to_dict(orient="records")
    for i, record in enumerate(records):
        record["notice_days"] = notice_days[i]
        record["is_late_notice"] = is_late_notice[i]
        for date_col, values in date_cols.items():
            record[date_col] = values[i]
    return records
