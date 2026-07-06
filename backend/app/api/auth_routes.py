from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ..services import auth_store

SESSION_COOKIE = "geo_session"

router = APIRouter(prefix="/api/auth")


class LoginPayload(BaseModel):
    username: str
    password: str


class UserPayload(BaseModel):
    username: str
    password: str
    role: str = "user"


class PasswordPayload(BaseModel):
    password: str


class RolePayload(BaseModel):
    role: str


def _current_user(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    return user


def _require_admin(request: Request) -> dict[str, Any]:
    user = _current_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


@router.post("/login")
def login(payload: LoginPayload, response: Response) -> dict[str, Any]:
    user = auth_store.verify_credentials(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = auth_store.create_session(user["username"])
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=auth_store.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {"username": user["username"], "role": user["role"]}


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    auth_store.delete_session(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"status": "ok"}


@router.get("/me")
def me(request: Request) -> dict[str, Any]:
    return _current_user(request)


@router.put("/me/password")
def change_own_password(payload: PasswordPayload, request: Request) -> dict[str, str]:
    user = _current_user(request)
    try:
        auth_store.set_password(user["username"], payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@router.get("/users")
def get_users(request: Request) -> list[dict[str, Any]]:
    _require_admin(request)
    return auth_store.list_users()


@router.post("/users")
def post_user(payload: UserPayload, request: Request) -> dict[str, Any]:
    _require_admin(request)
    try:
        return auth_store.create_user(payload.username, payload.password, payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/users/{username}")
def remove_user(username: str, request: Request) -> dict[str, str]:
    admin = _require_admin(request)
    try:
        auth_store.delete_user(username, operator=admin["username"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@router.put("/users/{username}/password")
def reset_password(username: str, payload: PasswordPayload, request: Request) -> dict[str, str]:
    _require_admin(request)
    try:
        auth_store.set_password(username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@router.put("/users/{username}/role")
def change_role(username: str, payload: RolePayload, request: Request) -> dict[str, str]:
    admin = _require_admin(request)
    try:
        auth_store.set_role(username, payload.role, operator=admin["username"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}
