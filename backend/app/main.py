from __future__ import annotations

import os
import secrets

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api.auth_routes import SESSION_COOKIE, router as auth_router
from .api.config_routes import router as config_router
from .api.insight_routes import products_router, router as insight_router
from .api.job_routes import router as job_router, template_router
from .api.routes import router
from .core.paths import APP_DIR
from .services import auth_store, job_store, product_master

app = FastAPI(title="GEO Search Workbench", version="0.1.0")

auth_store.init_db()
# GEO 库：建齐/迁移 schema（含物化指标表），并从 brands.yaml 同步产品主数据
product_master.ensure_geo_schema()
product_master.sync_products_from_brands()
job_store.init_db()
job_store.ensure_worker()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_PATHS = {"/login.html", "/api/auth/login", "/api/health", "/favicon.ico"}

# 门户统一登录（portal SSO）：仅当前置网关携带正确的共享密钥时，
# 才信任其注入的 X-Portal-User / X-Portal-Role 身份头。
# 未配置 GEO_PORTAL_SECRET 时该通道完全关闭，防止本机进程伪造身份。
PORTAL_SECRET = os.environ.get("GEO_PORTAL_SECRET", "")


def _portal_user(request: Request) -> dict | None:
    if not PORTAL_SECRET:
        return None
    provided = request.headers.get("x-portal-secret", "")
    if not provided or not secrets.compare_digest(provided, PORTAL_SECRET):
        return None
    username = request.headers.get("x-portal-user")
    if not username:
        return None
    role = request.headers.get("x-portal-role") or "user"
    return {"username": username, "role": role if role in ("admin", "user") else "user"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC_PATHS:
        response = await call_next(request)
    else:
        user = _portal_user(request) or auth_store.get_session_user(
            request.cookies.get(SESSION_COOKIE)
        )
        if user is None:
            if path.startswith("/api"):
                return JSONResponse({"detail": "未登录"}, status_code=401)
            return RedirectResponse("/login.html", status_code=302)
        request.state.user = user
        response = await call_next(request)
    # 缓存策略：入口 HTML 每次协商（发版即生效），带内容哈希的资源长缓存
    if path.startswith("/assets/"):
        response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
    elif not path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.include_router(auth_router)
app.include_router(config_router)
app.include_router(insight_router)
app.include_router(products_router)
app.include_router(job_router)
app.include_router(template_router)
app.include_router(router)
app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
