import json
import os
from abc import ABC, abstractmethod
from functools import lru_cache

import pandas as pd

from app.core.config import APP_STATE_DIR
from app.core.db import get_cursor

# Persisted JIN connection config -- lets the Settings page save/switch this
# from the UI (no .env edit + backend restart needed) once real credentials
# exist. Falls back to env vars (DATA_SOURCE_MODE/JIN_API_BASE_URL/JIN_API_KEY)
# when no saved config exists yet, so the original env-var path still works
# for anyone who prefers setting it that way.
_CONNECTION_CONFIG_PATH = APP_STATE_DIR / "jin_connection.json"


def _read_connection_config() -> dict:
    if _CONNECTION_CONFIG_PATH.exists():
        try:
            return json.loads(_CONNECTION_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "mode": os.environ.get("DATA_SOURCE_MODE", "local").strip().lower(),
        "base_url": os.environ.get("JIN_API_BASE_URL", ""),
        "api_key": os.environ.get("JIN_API_KEY", ""),
    }


def get_connection_config() -> dict:
    """UI-safe view -- never returns the raw API key, only whether one is set."""
    cfg = _read_connection_config()
    return {
        "mode": cfg.get("mode") or "local",
        "base_url": cfg.get("base_url") or None,
        "has_api_key": bool(cfg.get("api_key")),
    }


def save_connection_config(mode: str, base_url: str | None, api_key: str | None) -> dict:
    mode = (mode or "local").strip().lower()
    if mode not in ("local", "jin"):
        raise ValueError("mode must be 'local' or 'jin'")
    existing = _read_connection_config()
    cfg = {
        "mode": mode,
        "base_url": (base_url or "").strip() or existing.get("base_url", ""),
        # A blank submitted key keeps whatever was already saved, so re-saving
        # just the base_url/mode doesn't silently wipe out an entered key.
        "api_key": (api_key or "").strip() or existing.get("api_key", ""),
    }
    _CONNECTION_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    reset_adapter()
    return get_connection_config()


def test_connection() -> dict:
    """A REAL attempt, not a simulated result -- see JinApiAdapter above, which
    raises NotImplementedError until its HTTP calls are actually wired to a
    live JIN endpoint. This surfaces exactly that (or a real connection
    error, once it IS wired) rather than a fake green checkmark."""
    cfg = _read_connection_config()
    if cfg.get("mode") != "jin":
        return {"success": False, "message": "Currently in Local mode -- switch to JIN Data Warehouse mode and save credentials first."}
    if not cfg.get("base_url") or not cfg.get("api_key"):
        return {"success": False, "message": "Base URL and API key are both required before testing the connection."}
    try:
        get_adapter().get_employees()
        return {"success": True, "message": "Connected to the JIN Data Warehouse."}
    except NotImplementedError as exc:
        return {"success": False, "message": str(exc)}
    except Exception as exc:
        return {"success": False, "message": f"Connection failed: {exc}"}


def reset_adapter() -> None:
    _build_adapter.cache_clear()

@lru_cache(maxsize=None)
def _cached_query(table: str) -> pd.DataFrame:
    return get_cursor().execute(f"SELECT * FROM {table}").df()

class DataSourceAdapter(ABC):
    @abstractmethod
    def get_employees(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_projects(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_allocations(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_timesheets(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_skills(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_competencies(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_wsr_reports(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_pipeline_forecast(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_pipeline_skillset(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_pipeline_hierarchy(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_pipeline_revenue(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_leaves(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_weekly_pulse(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_hr_feedback(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_performance_cycles(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_performance_kra_items(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_coe_skills_mapping(self) -> pd.DataFrame: ...

class LocalAdapter(DataSourceAdapter):

    def _query(self, table: str) -> pd.DataFrame:
        return _cached_query(table).copy()

    def get_employees(self) -> pd.DataFrame:
        df = self._query("employees")
        # The raw account_status flag is not a real employment-status signal --
        # confirmed against the full real JDWH export: 225 rows carry an
        # explicit account_status=0 with zero corroborating exit evidence (no
        # exit_type, reason_for_leaving, or date_of_relieving; still the SCD
        # "current" row with valid_to=9999-12-31), and real job titles
        # (Software Engineer, Solutions Enabler, active Intern). ANDing it into
        # the active flag (an earlier version of this fix) silently excluded
        # all 225 of them from every "active employees" view -- worse than the
        # unreliable flag it was meant to guard against. date_of_resignation is
        # the only field anywhere in this data with real corroborating exit
        # evidence, so it alone determines current activity.
        today = pd.Timestamp.now().normalize()
        not_yet_departed = df["date_of_resignation"].isna() | (df["date_of_resignation"] > today)
        df["account_status"] = not_yet_departed.astype(int)
        return df

    def get_projects(self) -> pd.DataFrame:
        return self._query("projects")

    def get_allocations(self) -> pd.DataFrame:
        return self._query("allocations")

    def get_timesheets(self) -> pd.DataFrame:
        return self._query("timesheets")

    def get_skills(self) -> pd.DataFrame:
        # The real skills export (05_Skill_Details_clean.csv) uses a
        # completely different, disconnected employee_id scheme (EMP1,
        # EMP2... -- confirmed zero overlap with real employee_ids) --
        # skill_mapping_service remaps it onto real employees (by matching
        # real designation + a real-allocation-derived CoE, never a random
        # guess) so every consumer of this method gets real-ID-keyed rows
        # instead of silently matching nothing. Imported here, not at module
        # level, to avoid a circular import (skill_mapping_service reads the
        # raw table via this module's own _cached_query).
        from app.engines.skill_mapping_service import build_real_employee_skills_table
        return build_real_employee_skills_table()

    def get_competencies(self) -> pd.DataFrame:
        from app.engines.skill_mapping_service import build_real_employee_competency_table
        return build_real_employee_competency_table()

    def get_wsr_reports(self) -> pd.DataFrame:
        return self._query("wsr_reports")

    def get_pipeline_forecast(self) -> pd.DataFrame:
        return self._query("pipeline_forecast")

    def get_pipeline_skillset(self) -> pd.DataFrame:
        return self._query("pipeline_skillset")

    def get_pipeline_hierarchy(self) -> pd.DataFrame:
        return self._query("pipeline_hierarchy")

    def get_pipeline_revenue(self) -> pd.DataFrame:
        return self._query("pipeline_revenue")

    def get_leaves(self) -> pd.DataFrame:
        return self._query("leaves")

    def get_weekly_pulse(self) -> pd.DataFrame:
        return self._query("weekly_pulse")

    def get_hr_feedback(self) -> pd.DataFrame:
        return self._query("hr_feedback")

    def get_performance_cycles(self) -> pd.DataFrame:
        return self._query("performance_cycles")

    def get_performance_kra_items(self) -> pd.DataFrame:
        return self._query("performance_kra_items")

    def get_coe_skills_mapping(self) -> pd.DataFrame:
        return self._query("coe_skills_mapping")

class JinApiAdapter(DataSourceAdapter):

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def _not_implemented(self, endpoint: str):
        raise NotImplementedError(
            f"JinApiAdapter is a production contract stub. Wire {endpoint} to the "
            f"real JIN API at {self.base_url} when credentials are available."
        )

    def get_employees(self) -> pd.DataFrame:
        self._not_implemented("/api/employees")

    def get_projects(self) -> pd.DataFrame:
        self._not_implemented("/api/projects")

    def get_allocations(self) -> pd.DataFrame:
        self._not_implemented("/api/project-allocations")

    def get_timesheets(self) -> pd.DataFrame:
        self._not_implemented("/api/timesheets")

    def get_skills(self) -> pd.DataFrame:
        self._not_implemented("/api/skills")

    def get_competencies(self) -> pd.DataFrame:
        self._not_implemented("/api/competencies")

    def get_wsr_reports(self) -> pd.DataFrame:
        self._not_implemented("/api/status-reports")

    def get_pipeline_forecast(self) -> pd.DataFrame:
        self._not_implemented("/api/pipeline/forecast")

    def get_pipeline_skillset(self) -> pd.DataFrame:
        self._not_implemented("/api/pipeline/skillset")

    def get_pipeline_hierarchy(self) -> pd.DataFrame:
        self._not_implemented("/api/pipeline/hierarchy")

    def get_pipeline_revenue(self) -> pd.DataFrame:
        self._not_implemented("/api/pipeline/revenue")

    def get_leaves(self) -> pd.DataFrame:
        self._not_implemented("/api/leave-requests")

    def get_weekly_pulse(self) -> pd.DataFrame:
        self._not_implemented("/api/weekly-pulse")

    def get_hr_feedback(self) -> pd.DataFrame:
        self._not_implemented("/api/hr-feedback")

    def get_performance_cycles(self) -> pd.DataFrame:
        self._not_implemented("/api/performance-cycles")

    def get_performance_kra_items(self) -> pd.DataFrame:
        self._not_implemented("/api/performance-kra-items")

    def get_coe_skills_mapping(self) -> pd.DataFrame:
        self._not_implemented("/api/coe-skills-mapping")

@lru_cache(maxsize=1)
def _build_adapter() -> DataSourceAdapter:
    # mode=jin (+ base_url/api_key, saved from the Settings page or via the
    # DATA_SOURCE_MODE/JIN_API_BASE_URL/JIN_API_KEY env vars) is the real "flip
    # to production" switch this app ships ready for -- until real JIN
    # credentials exist, every JinApiAdapter method raises NotImplementedError
    # on purpose (see the class above) rather than silently returning fake
    # data, so this switch can be tested for wiring without ever pretending
    # to be connected.
    cfg = _read_connection_config()
    if cfg.get("mode") == "jin":
        return JinApiAdapter(base_url=cfg.get("base_url", ""), api_key=cfg.get("api_key", ""))
    return LocalAdapter()

def get_adapter() -> DataSourceAdapter:
    return _build_adapter()
