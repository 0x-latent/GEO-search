from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..core.paths import QUESTIONS_DIR, ROOT_DIR
from ..services.analysis_store import build_bi_overview, list_analysis_files, read_csv_preview
from ..services.run_store import create_run, list_runs
from ..services.sqlite_dashboard import (
    build_split_performance,
    build_sqlite_overview,
    delete_dataset,
    get_owned_dataset_ids,
    list_answer_samples,
    list_sqlite_datasets,
)
from ..services.task_runner import list_catalog, list_tasks, read_task_log, start_task
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


class RunPayload(BaseModel):
    config: dict[str, Any]


class TaskPayload(BaseModel):
    script_key: str
    args: list[str] = Field(default_factory=list)
    run_id: str | None = None


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


@router.get("/questions")
def get_questions(request: Request) -> dict[str, Any]:
    _require_admin(request)
    for name in ("questions_expanded.json", "questions_base.json"):
        path = QUESTIONS_DIR / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"source": name, "total": len(data), "questions": data}
    return {"source": None, "total": 0, "questions": []}


@router.post("/runs")
def create_run_config(payload: RunPayload, request: Request) -> dict[str, Any]:
    _require_admin(request)
    return create_run(payload.config)


@router.get("/runs")
def get_runs(request: Request) -> list[dict[str, Any]]:
    _require_admin(request)
    return list_runs()


@router.get("/tasks/catalog")
def get_task_catalog(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return list_catalog()


@router.post("/tasks")
def post_task(payload: TaskPayload, request: Request) -> dict[str, Any]:
    _require_admin(request)
    try:
        return start_task(payload.script_key, payload.args, payload.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/tasks")
def get_tasks(request: Request) -> list[dict[str, Any]]:
    _require_admin(request)
    return list_tasks()


@router.get("/tasks/{task_id}/log")
def get_task_log(task_id: str, request: Request) -> dict[str, str]:
    _require_admin(request)
    return {"log": read_task_log(task_id)}


@router.get("/analysis/files")
def get_analysis_files(request: Request) -> list[dict[str, Any]]:
    _require_admin(request)
    return list_analysis_files()


@router.get("/analysis/table/{filename}")
def get_analysis_table(filename: str, request: Request, limit: int = 100) -> dict[str, Any]:
    _require_admin(request)
    try:
        return read_csv_preview(filename, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/bi/overview")
def get_bi_overview(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return build_bi_overview()


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
