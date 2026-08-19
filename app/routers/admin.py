from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.services.admin_data_service import (
    DatasetValidationError,
    get_connection_status,
    get_dataset_preview,
    get_dataset_schema,
    list_data_sources,
    save_connection,
    test_connection,
    upload_dataset,
)
from app.services.jdwh_connection_service import (
    JdwhConnectionError,
    connect_and_discover_tables,
    get_expected_tables,
    get_jdwh_connection,
    list_jdwh_backups,
    pull_and_load_tables,
    revert_to_backup,
    save_jdwh_connection,
)
from app.services.jdwh_upload_service import (
    JdwhUploadError,
    load_tables_from_files,
    load_tables_from_workbook,
    preview_file,
    preview_workbook,
)

router = APIRouter(prefix="/admin", tags=["admin"])


class ConnectionConfigRequest(BaseModel):
    mode: str
    base_url: str | None = None
    api_key: str | None = None


class JdwhConnectionConfigRequest(BaseModel):
    profile_name: str | None = None
    server: str
    port: int = 1433
    database: str
    auth_type: str
    account: str | None = None
    encrypt: str = "Mandatory"
    trust_server_certificate: bool = False


class JdwhRevertRequest(BaseModel):
    timestamp: str | None = None


class JdwhConnectRequest(JdwhConnectionConfigRequest):
    # SQL Login only -- never persisted, only used for this one connection
    # attempt (see jdwh_connection_service.build_connection_string).
    password: str | None = None


@router.get("/data-sources")
def data_sources() -> list[dict]:
    return list_data_sources()


@router.get("/connection-status")
def connection_status() -> dict:
    return get_connection_status()


@router.post("/connection")
def save_connection_endpoint(req: ConnectionConfigRequest) -> dict:
    try:
        return save_connection(req.mode, req.base_url, req.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/connection/test")
def test_connection_endpoint() -> dict:
    return test_connection()


@router.get("/jdwh/connection")
def jdwh_connection() -> dict:
    return get_jdwh_connection()


@router.get("/jdwh/expected-tables")
def jdwh_expected_tables() -> dict:
    """The 6 provisioned tables and their real columns, straight from the
    data-warehouse team's own schema export -- structural metadata only,
    shown before (or without ever) connecting."""
    return {"tables": get_expected_tables()}


@router.post("/jdwh/connection")
def jdwh_save_connection(req: JdwhConnectionConfigRequest) -> dict:
    try:
        return save_jdwh_connection(req.model_dump())
    except JdwhConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jdwh/connect")
def jdwh_connect(req: JdwhConnectRequest) -> dict:
    """Opens the real connection (the Resource Manager's own Microsoft sign-in
    handles Entra ID/MFA) and confirms the 6 provisioned tables' real columns
    via INFORMATION_SCHEMA only -- never a row of real data."""
    payload = req.model_dump()
    password = payload.pop("password", None)
    try:
        return connect_and_discover_tables(payload, password=password)
    except JdwhConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jdwh/load-tables")
def jdwh_load_tables(req: JdwhConnectRequest) -> dict:
    """The real, consequential action: pulls the 5 real tables this app's
    data model uses, column-maps them (see jdwh_table_mapping.py), backs up
    the current local CSVs, and replaces them -- this changes what every
    page in the app reads from. The Settings UI gates this behind an
    explicit confirmation separate from Connect."""
    payload = req.model_dump()
    password = payload.pop("password", None)
    try:
        return pull_and_load_tables(payload, password=password)
    except JdwhConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/jdwh/backups")
def jdwh_backups() -> dict:
    return {"backups": list_jdwh_backups()}


@router.post("/jdwh/revert")
def jdwh_revert(req: JdwhRevertRequest) -> dict:
    """Undoes a Load Tables action -- restores the local CSVs from the given
    backup (or the most recent one) and reloads."""
    try:
        return revert_to_backup(req.timestamp)
    except JdwhConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jdwh/preview-workbook")
async def jdwh_preview_workbook(file: UploadFile = File(...)) -> dict:
    """Real feedback that a file was actually read -- sheet names and row/
    column counts only, never a real cell value. Called automatically as
    soon as a file is picked, before the destructive Load step."""
    content = await file.read()
    try:
        return preview_workbook(content, file.filename or "")
    except JdwhUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jdwh/preview-file")
async def jdwh_preview_file(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        return preview_file(content, file.filename or "")
    except JdwhUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jdwh/upload-workbook")
async def jdwh_upload_workbook(file: UploadFile = File(...)) -> dict:
    """No live connection needed (e.g. no JMAN VPN available right now) --
    one uploaded workbook with all 6 tables as separate sheets, real data
    already exported (see jdwh_upload_service.py's module docstring for the
    expected sheet names/shape)."""
    content = await file.read()
    try:
        return load_tables_from_workbook(content, file.filename or "")
    except JdwhUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/jdwh/upload-files")
async def jdwh_upload_files(
    employee: UploadFile = File(...),
    project: UploadFile = File(...),
    project_allocation: UploadFile = File(...),
    timesheet: UploadFile = File(...),
    weekly_status_report: UploadFile = File(...),
    designation_history: UploadFile | None = File(None),
    project_rolebased_user: UploadFile | None = File(None),
) -> dict:
    """Same as /jdwh/upload-workbook, but 6 (or 7) separate files instead of
    one workbook -- designation_history is accepted for consistency with the
    rest of the JDWH UI but genuinely unused (see jdwh_table_mapping.py).
    project_rolebased_user is optional and, when provided, is combined with
    project_allocation's own rows (see map_rolebased_user_table)."""
    uploads = {
        "employee": employee, "project": project, "project_allocation": project_allocation,
        "timesheet": timesheet, "weekly_status_report": weekly_status_report,
    }
    if designation_history is not None:
        uploads["designation_history"] = designation_history
    if project_rolebased_user is not None:
        uploads["project_rolebased_user"] = project_rolebased_user

    files: dict[str, tuple[bytes, str]] = {}
    for table, upload in uploads.items():
        files[table] = (await upload.read(), upload.filename or "")

    try:
        return load_tables_from_files(files)
    except JdwhUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/data-sources/{key}/preview")
def preview(key: str, rows: int = Query(default=20, ge=1, le=5000)) -> dict:
    try:
        return get_dataset_preview(key, max_rows=rows)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/data-sources/{key}/schema")
def schema(key: str) -> dict:
    try:
        return get_dataset_schema(key)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/data-sources/{key}/upload")
async def upload(key: str, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        return upload_dataset(key, file.filename or "", content)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
