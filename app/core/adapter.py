from abc import ABC, abstractmethod
from functools import lru_cache

import pandas as pd

from app.core.db import get_cursor

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

class LocalAdapter(DataSourceAdapter):

    def _query(self, table: str) -> pd.DataFrame:
        return _cached_query(table).copy()

    def get_employees(self) -> pd.DataFrame:
        df = self._query("employees")
        # account_status=1 alone isn't enough: it's a static HR-record flag (all 377
        # account_status=0 rows have no date_of_resignation at all, so it's tracking
        # something else entirely, not departure) -- but it's also never revised as
        # time passes, so someone whose date_of_resignation has already gone by can
        # still carry account_status=1, silently leaking departed people into every
        # "active employees" candidate pool (recommendations, redeployment, semantic
        # match, free pool, search). Combine both signals instead of trusting either
        # one alone.
        today = pd.Timestamp.now().normalize()
        not_yet_departed = df["date_of_resignation"].isna() | (df["date_of_resignation"] > today)
        df["account_status"] = ((df["account_status"] == 1) & not_yet_departed).astype(int)
        return df

    def get_projects(self) -> pd.DataFrame:
        return self._query("projects")

    def get_allocations(self) -> pd.DataFrame:
        return self._query("allocations")

    def get_timesheets(self) -> pd.DataFrame:
        return self._query("timesheets")

    def get_skills(self) -> pd.DataFrame:
        return self._query("skills")

    def get_competencies(self) -> pd.DataFrame:
        return self._query("competencies")

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

def get_adapter() -> DataSourceAdapter:
    return LocalAdapter()
