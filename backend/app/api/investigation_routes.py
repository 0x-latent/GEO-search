"""Public API for GEO anomaly scans and controlled investigations."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services import investigation_store
from .auth_routes import _current_user, _require_admin
from .routes import _check_dataset_access, _dataset_scope


router = APIRouter(prefix="/api")


class ScanPayload(BaseModel):
    current_dataset_id: str
    baseline_dataset_id: str | None = None
    auto_start: bool | None = None


class InvestigationPayload(BaseModel):
    current_dataset_id: str
    baseline_dataset_id: str | None = None
    product_code: str
    metric: str
    stage: str | None = None
    model: str | None = None
    search_enabled: int | None = Field(default=None, ge=0, le=1)
    expected_value: float | None = Field(default=None, ge=0, le=1)
    severity: Literal["low", "medium", "high"] = "medium"
    auto_start: bool = True


class TargetPayload(BaseModel):
    product_code: str
    metric: str
    stage: str | None = None
    model: str | None = None
    search_enabled: int | None = Field(default=None, ge=0, le=1)
    operator: str = "gte"
    target_value: float = Field(ge=0, le=1)
    is_active: bool = True


class TargetPatch(BaseModel):
    product_code: str | None = None
    metric: str | None = None
    stage: str | None = None
    model: str | None = None
    search_enabled: int | None = Field(default=None, ge=0, le=1)
    operator: str | None = None
    target_value: float | None = Field(default=None, ge=0, le=1)
    is_active: bool | None = None


class SettingsPayload(BaseModel):
    enabled: bool | None = None
    auto_start: bool | None = None
    primary_model_key: str | None = None
    primary_model_id: str | None = None
    fallback_model_key: str | None = None
    fallback_model_id: str | None = None
    max_reasoning_calls: int | None = Field(default=None, ge=1)
    max_probe_calls: int | None = Field(default=None, ge=0)
    max_web_fetches: int | None = Field(default=None, ge=0)
    max_iterations: int | None = Field(default=None, ge=1)
    max_auto_cases: int | None = Field(default=None, ge=0)
    request_timeout_seconds: int | None = Field(default=None, ge=10, le=600)


def _owned(investigation_id: str, request: Request) -> dict[str, Any]:
    item = investigation_store.get_investigation(investigation_id)
    if item is None:
        raise HTTPException(status_code=404, detail="调查不存在")
    allowed = _dataset_scope(request)
    if allowed is not None and item["current_dataset_id"] not in allowed:
        raise HTTPException(status_code=404, detail="调查不存在")
    return item


@router.post("/investigations/scan")
def scan_dataset(payload: ScanPayload, request: Request) -> dict[str, Any]:
    user = _current_user(request)
    allowed = _dataset_scope(request)
    _check_dataset_access(payload.current_dataset_id, allowed)
    if payload.baseline_dataset_id:
        _check_dataset_access(payload.baseline_dataset_id, allowed)
    try:
        return investigation_store.scan_dataset(
            payload.current_dataset_id,
            owner_username=user["username"],
            baseline_dataset_id=payload.baseline_dataset_id,
            auto_start=payload.auto_start,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/investigations")
def investigations(
    request: Request, status: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    user = _current_user(request)
    allowed = _dataset_scope(request)
    return investigation_store.list_investigations(
        username=None if user["role"] == "admin" else user["username"],
        allowed_dataset_ids=allowed,
        status=status,
        limit=limit,
    )


@router.post("/investigations")
def create_investigation(
    payload: InvestigationPayload, request: Request
) -> dict[str, Any]:
    user = _current_user(request)
    allowed = _dataset_scope(request)
    _check_dataset_access(payload.current_dataset_id, allowed)
    if payload.baseline_dataset_id:
        _check_dataset_access(payload.baseline_dataset_id, allowed)
    try:
        return investigation_store.create_manual_investigation(
            user["username"], payload.model_dump(exclude_none=True)
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/investigations/{investigation_id}")
def investigation(investigation_id: str, request: Request) -> dict[str, Any]:
    return _owned(investigation_id, request)


@router.get("/investigations/{investigation_id}/events")
def investigation_events(
    investigation_id: str, request: Request, after: int = 0
) -> list[dict[str, Any]]:
    _owned(investigation_id, request)
    return investigation_store.list_events(investigation_id, max(0, after))


@router.post("/investigations/{investigation_id}/start")
def start_investigation(investigation_id: str, request: Request) -> dict[str, Any]:
    _owned(investigation_id, request)
    try:
        return investigation_store.start_investigation(investigation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/investigations/{investigation_id}/retry")
def retry_investigation(investigation_id: str, request: Request) -> dict[str, Any]:
    _owned(investigation_id, request)
    try:
        return investigation_store.retry_investigation(investigation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/investigations/{investigation_id}/cancel")
def cancel_investigation(investigation_id: str, request: Request) -> dict[str, Any]:
    _owned(investigation_id, request)
    try:
        return investigation_store.cancel_investigation(investigation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/investigation-targets")
def investigation_targets(request: Request) -> list[dict[str, Any]]:
    user = _current_user(request)
    return investigation_store.list_targets(
        None if user["role"] == "admin" else user["username"]
    )


@router.post("/investigation-targets")
def create_target(payload: TargetPayload, request: Request) -> dict[str, Any]:
    user = _current_user(request)
    try:
        return investigation_store.create_target(user["username"], payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _owned_target(target_id: str, request: Request) -> dict[str, Any]:
    user = _current_user(request)
    target = investigation_store.get_target(target_id)
    if target is None or (
        user["role"] != "admin" and target["owner_username"] != user["username"]
    ):
        raise HTTPException(status_code=404, detail="监测目标不存在")
    return target


@router.patch("/investigation-targets/{target_id}")
def update_target(
    target_id: str, payload: TargetPatch, request: Request
) -> dict[str, Any]:
    _owned_target(target_id, request)
    try:
        return investigation_store.update_target(
            target_id, payload.model_dump(exclude_unset=True)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/investigation-targets/{target_id}")
def delete_target(target_id: str, request: Request) -> dict[str, str]:
    _owned_target(target_id, request)
    try:
        investigation_store.delete_target(target_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok"}


@router.get("/investigation-settings")
def investigation_settings(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return investigation_store.get_settings()


@router.put("/investigation-settings")
def update_investigation_settings(
    payload: SettingsPayload, request: Request
) -> dict[str, Any]:
    user = _require_admin(request)
    return investigation_store.update_settings(
        payload.model_dump(exclude_unset=True), user["username"]
    )
