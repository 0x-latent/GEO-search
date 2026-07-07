from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..core.paths import ROOT_DIR
from ..services.sqlite_dashboard import (
    build_split_performance,
    build_sqlite_overview,
    delete_dataset,
    get_owned_dataset_ids,
    list_answer_samples,
    list_sqlite_datasets,
)
from ..services.yaml_store import load_brands, load_models, save_brands
from .auth_routes import _current_user, _require_admin


router = APIRouter(prefix="/api")


def _dataset_scope(request: Request) -> list[str] | None:
    """admin 返回 None（不限制），普通用户返回自己拥有的 dataset_id 列表。"""
    user = _current_user(request)
    if user["role"] == "admin":
        return None
    return get_owned_dataset_ids(user["username"])


def _check_dataset_access(dataset_id: str, allowed: list[str] | None) -> None:
    if allowed is not None and dataset_id != "all" and dataset_id not in allowed:
        raise HTTPException(status_code=404, detail=f"数据集不存在或无权访问: {dataset_id}")


class BrandsPayload(BaseModel):
    data: dict[str, Any]


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "root": str(ROOT_DIR)}


@router.get("/config/brands")
def get_brands(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return load_brands()


@router.put("/config/brands")
def put_brands(payload: BrandsPayload, request: Request) -> dict[str, Any]:
    _require_admin(request)
    try:
        return save_brands(payload.data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/config/models")
def get_models() -> dict[str, Any]:
    return load_models()


@router.get("/sqlite/datasets")
def get_sqlite_datasets(request: Request) -> list[dict[str, Any]]:
    allowed = _dataset_scope(request)
    try:
        return list_sqlite_datasets(allowed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sqlite/overview")
def get_sqlite_overview(request: Request, dataset_id: str = "all") -> dict[str, Any]:
    allowed = _dataset_scope(request)
    _check_dataset_access(dataset_id, allowed)
    try:
        return build_sqlite_overview(dataset_id, allowed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sqlite/answers")
def get_sqlite_answers(
    request: Request, dataset_id: str = "all", limit: int = 100
) -> list[dict[str, Any]]:
    allowed = _dataset_scope(request)
    _check_dataset_access(dataset_id, allowed)
    try:
        return list_answer_samples(dataset_id, limit, allowed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/sqlite/datasets/{dataset_id}")
def remove_dataset(dataset_id: str, request: Request) -> dict[str, str]:
    _require_admin(request)
    try:
        delete_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok"}


@router.get("/sqlite/splits")
def get_sqlite_splits(request: Request, dataset_id: str = "all") -> dict[str, Any]:
    allowed = _dataset_scope(request)
    _check_dataset_access(dataset_id, allowed)
    try:
        return build_split_performance(dataset_id, allowed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
