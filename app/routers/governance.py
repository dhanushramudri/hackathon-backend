from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.governance_service import (
    add_or_update_spotlight, add_risk, assign_cluster, current_week_start, get_cluster_dashboard,
    get_unassigned_projects, list_clusters, remove_from_spotlight, resolve_risk, save_kickoff_tracking,
)

router = APIRouter(prefix="/governance", tags=["governance"])

@router.get("/clusters")
def clusters() -> list[dict]:
    return list_clusters()

@router.get("/clusters/{cluster_number}")
def cluster_dashboard(cluster_number: int) -> dict:
    try:
        return get_cluster_dashboard(cluster_number)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/unassigned-projects")
def unassigned_projects() -> list[dict]:
    return get_unassigned_projects()

class AssignClusterRequest(BaseModel):
    project_code: str
    cluster_number: int

@router.post("/assign-cluster")
def assign_cluster_route(body: AssignClusterRequest) -> dict:
    try:
        return assign_cluster(body.project_code, body.cluster_number)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

class AddRiskRequest(BaseModel):
    project_code: str
    risk_description: str
    risk_type: str | None = None
    mitigation_steps: str | None = None

@router.post("/risks")
def add_risk_route(body: AddRiskRequest) -> dict:
    return add_risk(body.project_code, body.risk_description, body.risk_type, body.mitigation_steps)

@router.post("/risks/{risk_id}/resolve")
def resolve_risk_route(risk_id: str) -> dict:
    result = resolve_risk(risk_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Risk not found")
    return result

class SpotlightRequest(BaseModel):
    project_code: str
    action_plan: str | None = None

@router.post("/spotlight")
def save_spotlight_route(body: SpotlightRequest) -> dict:
    fields = body.model_dump(exclude={"project_code"})
    return add_or_update_spotlight(body.project_code, current_week_start(), fields)

@router.delete("/spotlight/{project_code}")
def remove_spotlight_route(project_code: str) -> dict:
    remove_from_spotlight(project_code, current_week_start())
    return {"removed": True}

class KickoffTrackingRequest(BaseModel):
    project_code: str
    kickoff_completed: str | None = None
    scope_approved: str | None = None
    devops_setup: str | None = None
    comment: str | None = None

@router.post("/kickoff-tracking")
def save_kickoff_tracking_route(body: KickoffTrackingRequest) -> dict:
    fields = body.model_dump(exclude={"project_code"})
    return save_kickoff_tracking(body.project_code, current_week_start(), fields)
