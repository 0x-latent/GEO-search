from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..services import user_config_store
from .auth_routes import _current_user, _require_admin

router = APIRouter(prefix="/api/config")


class ConfigPayload(BaseModel):
    data: dict[str, Any]


# ---------- 用户级配置（brands / knowledge_base），未自定义时回退全局默认 ----------

@router.get("/my/brands")
def get_my_brands(request: Request) -> dict[str, Any]:
    user = _current_user(request)
    return user_config_store.load_effective_brands(user["username"])


@router.put("/my/brands")
def put_my_brands(payload: ConfigPayload, request: Request) -> dict[str, str]:
    user = _current_user(request)
    try:
        user_config_store.save_user_brands(user["username"], payload.data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@router.delete("/my/brands")
def reset_my_brands(request: Request) -> dict[str, str]:
    user = _current_user(request)
    user_config_store.reset_user_brands(user["username"])
    return {"status": "ok"}


@router.get("/my/knowledge-base")
def get_my_kb(request: Request) -> dict[str, Any]:
    user = _current_user(request)
    return user_config_store.load_effective_kb(user["username"])


@router.put("/my/knowledge-base")
def put_my_kb(payload: ConfigPayload, request: Request) -> dict[str, str]:
    user = _current_user(request)
    try:
        user_config_store.save_user_kb(user["username"], payload.data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@router.delete("/my/knowledge-base")
def reset_my_kb(request: Request) -> dict[str, str]:
    user = _current_user(request)
    user_config_store.reset_user_kb(user["username"])
    return {"status": "ok"}


# ---------- 全局 knowledge_base（admin，与 brands 的全局编辑对齐） ----------

@router.get("/knowledge-base")
def get_global_kb(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return user_config_store.load_global_kb()


@router.put("/knowledge-base")
def put_global_kb(payload: ConfigPayload, request: Request) -> dict[str, str]:
    _require_admin(request)
    try:
        user_config_store.save_global_kb(payload.data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}
