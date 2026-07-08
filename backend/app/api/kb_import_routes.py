from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ..services import kb_import_store
from .auth_routes import _current_user

router = APIRouter(prefix="/api/kb-imports")


@router.post("")
async def create_kb_import(
    request: Request,
    file: UploadFile = File(...),
    product_key: str = Form(...),
    scope: str = Form("user"),
) -> dict[str, Any]:
    """上传产品资料（PDF/PPTX/TXT/图片），后台多模态识别并结构化为知识库草稿。"""
    user = _current_user(request)
    content = await file.read()
    try:
        return kb_import_store.create_import(
            username=user["username"],
            role=user["role"],
            product_key=product_key,
            scope=scope,
            filename=file.filename or "upload",
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def list_kb_imports(request: Request) -> list[dict[str, Any]]:
    user = _current_user(request)
    username = None if user["role"] == "admin" else user["username"]
    return kb_import_store.list_imports(username)


def _owned(import_id: str, request: Request) -> dict[str, Any]:
    user = _current_user(request)
    record = kb_import_store.get_import(import_id)
    if record is None or (user["role"] != "admin" and record["username"] != user["username"]):
        raise HTTPException(status_code=404, detail="导入任务不存在")
    return record


@router.get("/{import_id}")
def get_kb_import(import_id: str, request: Request) -> dict[str, Any]:
    """任务详情（含草稿模块，供审核界面做 diff）。"""
    record = _owned(import_id, request)
    if record.get("draft_json"):
        try:
            record["draft"] = json.loads(record.pop("draft_json"))
        except (TypeError, ValueError):
            record["draft"] = None
    return record


@router.post("/{import_id}/retry")
def retry_kb_import(import_id: str, request: Request) -> dict[str, Any]:
    """失败任务断点重试（已识别页面不重复调用）。"""
    _owned(import_id, request)
    try:
        return kb_import_store.retry_import(import_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ApplyPayload(BaseModel):
    modules: dict[str, str]
    scope: str | None = None


@router.post("/{import_id}/apply")
def apply_kb_import(import_id: str, payload: ApplyPayload, request: Request) -> dict[str, Any]:
    """把审核采纳的模块（可在前端修改后提交）合并进知识库。"""
    user = _current_user(request)
    _owned(import_id, request)
    try:
        return kb_import_store.apply_import(
            import_id, user["username"], user["role"], payload.modules, payload.scope
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
