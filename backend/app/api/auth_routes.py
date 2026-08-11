from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

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


def _portal_managed() -> None:
    raise HTTPException(status_code=403, detail="账户、密码和权限由门户统一管理")


@router.post("/login")
def login(payload: LoginPayload, response: Response) -> dict[str, Any]:
    del payload, response
    _portal_managed()


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    del request
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"status": "ok"}


@router.get("/me")
def me(request: Request) -> dict[str, Any]:
    return _current_user(request)


@router.put("/me/password")
def change_own_password(payload: PasswordPayload, request: Request) -> dict[str, str]:
    del payload, request
    _portal_managed()


@router.get("/users")
def get_users(request: Request) -> list[dict[str, Any]]:
    del request
    _portal_managed()


@router.post("/users")
def post_user(payload: UserPayload, request: Request) -> dict[str, Any]:
    del payload, request
    _portal_managed()


@router.delete("/users/{username}")
def remove_user(username: str, request: Request) -> dict[str, str]:
    del username, request
    _portal_managed()


@router.put("/users/{username}/password")
def reset_password(username: str, payload: PasswordPayload, request: Request) -> dict[str, str]:
    del username, payload, request
    _portal_managed()


@router.put("/users/{username}/role")
def change_role(username: str, payload: RolePayload, request: Request) -> dict[str, str]:
    del username, payload, request
    _portal_managed()
