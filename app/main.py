from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.core.db import get_connection, table_counts
from app.core.safe_json import SafeJSONResponse
from app.routers import allocations, buddy, employees, forecast, free_pool, leave, pipeline, recommendations, revenue, role_mix
from app.routers import health as health_monitor_router

app = FastAPI(
    title="ResourceIQ API",
    description="JMAN resourcing co-pilot -- backend for the 5 use-case engines.",
    version="0.1.0",
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_data() -> None:
    get_connection()

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/meta/tables")
def meta_tables() -> dict[str, int]:
    return table_counts()

app.include_router(role_mix.router)
app.include_router(allocations.router)
app.include_router(recommendations.router)
app.include_router(health_monitor_router.router)
app.include_router(forecast.router)
app.include_router(pipeline.router)
app.include_router(buddy.router)
app.include_router(free_pool.router)
app.include_router(revenue.router)
app.include_router(leave.router)
app.include_router(employees.router)
