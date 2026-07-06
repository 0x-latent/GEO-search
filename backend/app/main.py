from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api.auth_routes import SESSION_COOKIE, router as auth_router
from .api.config_routes import router as config_router
from .api.job_routes import router as job_router, template_router
from .api.routes import router
from .core.paths import APP_DIR
from .services import auth_store, job_store


from .services.sqlite_dashboard import ensure_owner_column

app = FastAPI(title="GEO Search Workbench", version="0.1.0")

auth_store.init_db()
ensure_owner_column()
job_store.init_db()
job_store.ensure_worker()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_PATHS = {"/login.html", "/api/auth/login", "/favicon.ico"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC_PATHS:
        return await call_next(request)
    user = auth_store.get_session_user(request.cookies.get(SESSION_COOKIE))
    if user is None:
        if path.startswith("/api"):
            return JSONResponse({"detail": "未登录"}, status_code=401)
        return RedirectResponse("/login.html", status_code=302)
    request.state.user = user
    return await call_next(request)


app.include_router(auth_router)
app.include_router(config_router)
app.include_router(job_router)
app.include_router(template_router)
app.include_router(router)
app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
