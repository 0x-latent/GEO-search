from __future__ import annotations

import csv
import io
import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..services import job_store
from ..services.question_parser import parse_upload
from ..services.yaml_store import load_models
from .auth_routes import _current_user, _require_admin

router = APIRouter(prefix="/api/jobs")

TEMPLATE_COLUMNS = ["问题", "产品", "层级", "场景"]
TEMPLATE_ROWS = [
    ["感冒了吃什么药好得快？", "感冒灵", "q1_overall", "日常用药"],
    ["999感冒灵和连花清瘟哪个效果好？", "感冒灵", "q2_compare", ""],
]


class JobPayload(BaseModel):
    dataset_name: str
    questions: list[dict[str, Any]]
    models: list[str] = Field(default_factory=list)
    model_overrides: dict[str, str] = Field(default_factory=dict)
    search_mode: str = "both"
    rounds: int = 1
    route: str | None = None


@router.get("/options")
def get_job_options(request: Request) -> dict[str, Any]:
    """上传界面的模型选项：各厂商可选型号（第一个为默认推荐）。"""
    user = _current_user(request)
    config = load_models()
    models = []
    for key, spec in (config.get("models") or {}).items():
        if not spec.get("enabled", False):
            continue
        variants = spec.get("variants") or [{"id": spec.get("model_id", ""), "label": spec.get("model_id", "")}]
        models.append({
            "key": key,
            "name": spec.get("name", key),
            "supports_search": spec.get("supports_search", False),
            "default_model": variants[0]["id"],
            "variants": variants,
        })
    job_settings = config.get("job_settings") or {}
    return {
        "models": models,
        "max_questions": job_store.MAX_QUESTIONS,
        "default_rounds": job_settings.get("rounds", 1),
        "can_choose_route": user["role"] == "admin",
        "default_route": "relay" if (config.get("relay") or {}).get("enabled") else "direct",
    }


@router.post("/parse")
async def parse_questions_file(
    request: Request,
    file: UploadFile = File(...),
    default_product: str = Form(""),
) -> dict[str, Any]:
    _current_user(request)
    content = await file.read()
    try:
        questions = parse_upload(file.filename or "upload", content, default_product)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "total": len(questions),
        "questions": questions,
        "preview": questions[:20],
    }


@router.post("")
def create_job(payload: JobPayload, request: Request) -> dict[str, Any]:
    user = _current_user(request)
    name = payload.dataset_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="请填写数据集名称")
    try:
        return job_store.create_job(
            username=user["username"],
            role=user["role"],
            dataset_name=name,
            questions=payload.questions,
            models=payload.models,
            model_overrides=payload.model_overrides,
            search_mode=payload.search_mode,
            rounds=payload.rounds,
            route=payload.route,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def get_jobs(request: Request) -> list[dict[str, Any]]:
    user = _current_user(request)
    username = None if user["role"] == "admin" else user["username"]
    return job_store.list_jobs(username)


@router.get("/{job_id}/log")
def get_job_log(job_id: str, request: Request) -> dict[str, str]:
    user = _current_user(request)
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if user["role"] != "admin" and job["username"] != user["username"]:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"log": job_store.read_job_log(job_id)}


# ---------- 问题模板下载 ----------

template_router = APIRouter(prefix="/api/templates")


@template_router.get("/questions.csv")
def template_csv() -> Response:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(TEMPLATE_COLUMNS)
    writer.writerows(TEMPLATE_ROWS)
    data = ("﻿" + buffer.getvalue()).encode("utf-8")
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="questions_template.csv"'},
    )


@template_router.get("/questions.json")
def template_json() -> Response:
    rows = [dict(zip(TEMPLATE_COLUMNS, row)) for row in TEMPLATE_ROWS]
    data = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=data,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="questions_template.json"'},
    )


@template_router.get("/questions.xlsx")
def template_xlsx() -> Response:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "问题清单"
    sheet.append(TEMPLATE_COLUMNS)
    for row in TEMPLATE_ROWS:
        sheet.append(row)
    sheet.column_dimensions["A"].width = 50
    buffer = io.BytesIO()
    workbook.save(buffer)
    return Response(
        content=buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="questions_template.xlsx"'},
    )
