from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.paths import QUESTIONS_DIR, ROOT_DIR
from ..services.analysis_store import build_bi_overview, list_analysis_files, read_csv_preview
from ..services.run_store import create_run, list_runs
from ..services.sqlite_dashboard import (
    build_split_performance,
    build_sqlite_overview,
    list_answer_samples,
    list_sqlite_datasets,
)
from ..services.task_runner import list_catalog, list_tasks, read_task_log, start_task
from ..services.yaml_store import load_brands, load_models, save_brands


router = APIRouter(prefix="/api")


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
def get_brands() -> dict[str, Any]:
    return load_brands()


@router.put("/config/brands")
def put_brands(payload: BrandsPayload) -> dict[str, Any]:
    try:
        return save_brands(payload.data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/config/models")
def get_models() -> dict[str, Any]:
    return load_models()


@router.get("/questions")
def get_questions() -> dict[str, Any]:
    for name in ("questions_expanded.json", "questions_base.json"):
        path = QUESTIONS_DIR / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {"source": name, "total": len(data), "questions": data}
    return {"source": None, "total": 0, "questions": []}


@router.post("/runs")
def create_run_config(payload: RunPayload) -> dict[str, Any]:
    return create_run(payload.config)


@router.get("/runs")
def get_runs() -> list[dict[str, Any]]:
    return list_runs()


@router.get("/tasks/catalog")
def get_task_catalog() -> dict[str, Any]:
    return list_catalog()


@router.post("/tasks")
def post_task(payload: TaskPayload) -> dict[str, Any]:
    try:
        return start_task(payload.script_key, payload.args, payload.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/tasks")
def get_tasks() -> list[dict[str, Any]]:
    return list_tasks()


@router.get("/tasks/{task_id}/log")
def get_task_log(task_id: str) -> dict[str, str]:
    return {"log": read_task_log(task_id)}


@router.get("/analysis/files")
def get_analysis_files() -> list[dict[str, Any]]:
    return list_analysis_files()


@router.get("/analysis/table/{filename}")
def get_analysis_table(filename: str, limit: int = 100) -> dict[str, Any]:
    try:
        return read_csv_preview(filename, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/bi/overview")
def get_bi_overview() -> dict[str, Any]:
    return build_bi_overview()


@router.get("/sqlite/datasets")
def get_sqlite_datasets() -> list[dict[str, Any]]:
    try:
        return list_sqlite_datasets()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sqlite/overview")
def get_sqlite_overview(dataset_id: str = "all") -> dict[str, Any]:
    try:
        return build_sqlite_overview(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sqlite/answers")
def get_sqlite_answers(dataset_id: str = "all", limit: int = 100) -> list[dict[str, Any]]:
    try:
        return list_answer_samples(dataset_id, limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sqlite/splits")
def get_sqlite_splits(dataset_id: str = "all") -> dict[str, Any]:
    try:
        return build_split_performance(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
