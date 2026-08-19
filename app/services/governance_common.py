"""Shared helpers for the Cluster Governance services."""
import pandas as pd

def current_week_start() -> str:
    """Monday of the current ISO week, as YYYY-MM-DD -- the anchor key for
    this-week-only governance entries (spotlight, kickoff tracking). Every
    row is still stamped with this, so a future "browse past weeks" view can
    be added without a data migration -- this first version only ever reads/
    writes the row for today's week."""
    today = pd.Timestamp.now().normalize()
    monday = today - pd.Timedelta(days=today.dayofweek)
    return monday.strftime("%Y-%m-%d")

def week_end(week_start_date: str) -> str:
    return (pd.Timestamp(week_start_date) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
