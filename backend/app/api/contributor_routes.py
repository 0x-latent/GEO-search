from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from ..services import contributor_store
from ..services.document_extract import MAX_FILE_BYTES
from ..services.yaml_store import load_models
from .auth_routes import _require_admin


router = APIRouter()


def _failure(exc: ValueError, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(exc))


def _contributor(request: Request) -> dict[str, Any]:
    session = contributor_store.get_contributor_session(
        request.cookies.get(contributor_store.CONTRIBUTOR_COOKIE)
    )
    if session is None:
        raise HTTPException(status_code=401, detail="投稿会话无效或已过期")
    return session


class SessionPayload(BaseModel):
    invite_id: str
    token: str


class PublicationPayload(BaseModel):
    platform: str
    url: str
    published_at: str | None = None


@router.post("/api/contributor/session")
def exchange_session(payload: SessionPayload, response: Response) -> dict[str, Any]:
    try:
        token, workspace, max_age = contributor_store.exchange_invite(payload.invite_id, payload.token)
    except ValueError as exc:
        raise _failure(exc, 401) from exc
    response.set_cookie(
        contributor_store.CONTRIBUTOR_COOKIE, token, max_age=max_age,
        httponly=True, samesite="lax", path="/api/contributor",
    )
    return workspace


@router.get("/api/contributor/context")
def contributor_context(request: Request) -> dict[str, Any]:
    return _contributor(request)["workspace"]


@router.post("/api/contributor/logout")
def contributor_logout(request: Request, response: Response) -> dict[str, str]:
    contributor_store.delete_contributor_session(
        request.cookies.get(contributor_store.CONTRIBUTOR_COOKIE)
    )
    response.delete_cookie(contributor_store.CONTRIBUTOR_COOKIE, path="/api/contributor")
    return {"status": "ok"}


@router.get("/api/contributor/submissions")
def contributor_submissions(request: Request) -> list[dict[str, Any]]:
    session = _contributor(request)
    return contributor_store.list_submissions(session["company_id"], external=True)


@router.post("/api/contributor/submissions")
async def create_submission(
    request: Request,
    file: UploadFile = File(...),
    product_code: str = Form(...),
    title: str = Form(...),
    submitter_name: str = Form(...),
    submitter_email: str = Form(...),
    campaign: str | None = Form(None),
    published_platform: str | None = Form(None),
    published_url: str | None = Form(None),
    published_at: str | None = Form(None),
) -> dict[str, Any]:
    session = _contributor(request)
    try:
        return contributor_store.create_submission(
            session, file.filename or "article", await file.read(MAX_FILE_BYTES + 1), product_code,
            title, submitter_name, submitter_email, campaign,
            published_platform, published_url, published_at,
        )
    except ValueError as exc:
        raise _failure(exc) from exc


@router.get("/api/contributor/submissions/{submission_id}")
def contributor_submission(submission_id: str, request: Request) -> dict[str, Any]:
    session = _contributor(request)
    try:
        return contributor_store.get_submission(
            submission_id, company_id=session["company_id"], external=True
        )
    except ValueError as exc:
        raise _failure(exc, 404) from exc


@router.post("/api/contributor/submissions/{submission_id}/revision")
async def contributor_revision(
    submission_id: str, request: Request, file: UploadFile = File(...),
) -> dict[str, Any]:
    try:
        return contributor_store.add_revision(
            _contributor(request), submission_id, file.filename or "article",
            await file.read(MAX_FILE_BYTES + 1)
        )
    except ValueError as exc:
        raise _failure(exc) from exc


@router.put("/api/contributor/submissions/{submission_id}/publication")
def contributor_publication(
    submission_id: str, payload: PublicationPayload, request: Request,
) -> dict[str, Any]:
    try:
        return contributor_store.update_publication(
            _contributor(request), submission_id, payload.platform,
            payload.url, payload.published_at,
        )
    except ValueError as exc:
        raise _failure(exc) from exc


class CompanyPayload(BaseModel):
    name: str
    contact_name: str | None = None
    contact_email: str | None = None


class CompanyPatch(BaseModel):
    name: str | None = None
    contact_name: str | None = None
    contact_email: str | None = None
    is_active: bool | None = None


class InvitePayload(BaseModel):
    company_id: str
    allowed_product_codes: list[str] = Field(default_factory=list)
    expires_at: str
    max_submissions: int = 20


class ReviewActionPayload(BaseModel):
    feedback: str | None = None
    finding_ids: list[str] = Field(default_factory=list)


class ReviewSettingsPayload(BaseModel):
    auto_start: bool | None = None
    queue_paused: bool | None = None
    primary_model_key: str | None = None
    primary_model_id: str | None = None
    fallback_model_key: str | None = None
    fallback_model_id: str | None = None
    ai_concurrency: int | None = None
    request_timeout_seconds: int | None = None
    retry_count: int | None = None
    similarity_threshold: float | None = None
    similarity_top_k: int | None = None


@router.get("/api/admin/contributor-companies")
def admin_companies(request: Request) -> list[dict[str, Any]]:
    _require_admin(request)
    return contributor_store.list_companies()


@router.post("/api/admin/contributor-companies")
def admin_create_company(payload: CompanyPayload, request: Request) -> dict[str, Any]:
    user = _require_admin(request)
    try:
        return contributor_store.create_company(
            payload.name, user["username"], payload.contact_name, payload.contact_email
        )
    except ValueError as exc:
        raise _failure(exc) from exc


@router.patch("/api/admin/contributor-companies/{company_id}")
def admin_update_company(
    company_id: str, payload: CompanyPatch, request: Request,
) -> dict[str, Any]:
    _require_admin(request)
    try:
        return contributor_store.update_company(
            company_id, **payload.model_dump(exclude_none=True)
        )
    except ValueError as exc:
        raise _failure(exc) from exc


@router.get("/api/admin/contributor-invites")
def admin_invites(request: Request, company_id: str | None = None) -> list[dict[str, Any]]:
    _require_admin(request)
    return contributor_store.list_invites(company_id)


@router.post("/api/admin/contributor-invites")
def admin_create_invite(payload: InvitePayload, request: Request) -> dict[str, Any]:
    user = _require_admin(request)
    try:
        return contributor_store.create_invite(
            payload.company_id, user["username"], payload.allowed_product_codes,
            payload.expires_at, payload.max_submissions,
        )
    except ValueError as exc:
        raise _failure(exc) from exc


@router.delete("/api/admin/contributor-invites/{invite_id}")
def admin_revoke_invite(invite_id: str, request: Request) -> dict[str, str]:
    _require_admin(request)
    try:
        contributor_store.revoke_invite(invite_id)
        return {"status": "ok"}
    except ValueError as exc:
        raise _failure(exc, 404) from exc


@router.get("/api/admin/article-submissions")
def admin_submissions(
    request: Request, company_id: str | None = None, status: str | None = None,
) -> list[dict[str, Any]]:
    _require_admin(request)
    return contributor_store.list_submissions(company_id, status)


@router.get("/api/admin/article-submissions/{submission_id}")
def admin_submission(submission_id: str, request: Request) -> dict[str, Any]:
    _require_admin(request)
    try:
        return contributor_store.get_submission(submission_id)
    except ValueError as exc:
        raise _failure(exc, 404) from exc


def _review_action(
    submission_id: str, action: str, payload: ReviewActionPayload, request: Request,
) -> dict[str, Any]:
    user = _require_admin(request)
    try:
        return contributor_store.review_action(
            submission_id, action, user["username"], payload.feedback, payload.finding_ids
        )
    except ValueError as exc:
        raise _failure(exc) from exc


@router.post("/api/admin/article-submissions/{submission_id}/approve")
def approve_submission(submission_id: str, payload: ReviewActionPayload, request: Request):
    return _review_action(submission_id, "approve", payload, request)


@router.post("/api/admin/article-submissions/{submission_id}/request-revision")
def request_revision(submission_id: str, payload: ReviewActionPayload, request: Request):
    return _review_action(submission_id, "request_revision", payload, request)


@router.post("/api/admin/article-submissions/{submission_id}/reject")
def reject_submission(submission_id: str, payload: ReviewActionPayload, request: Request):
    return _review_action(submission_id, "reject", payload, request)


@router.post("/api/admin/article-submissions/{submission_id}/retry")
def retry_submission(submission_id: str, request: Request) -> dict[str, str]:
    user = _require_admin(request)
    try:
        contributor_store.retry_review(submission_id, user["username"])
        return {"status": "ok"}
    except ValueError as exc:
        raise _failure(exc) from exc


@router.post("/api/admin/article-submissions/{submission_id}/cancel")
def cancel_submission(submission_id: str, request: Request) -> dict[str, str]:
    user = _require_admin(request)
    try:
        contributor_store.cancel_review(submission_id, user["username"])
        return {"status": "ok"}
    except ValueError as exc:
        raise _failure(exc) from exc


def _model_options() -> list[dict[str, Any]]:
    result = []
    for key, spec in (load_models().get("models") or {}).items():
        if spec.get("enabled", False):
            result.append({
                "key": key, "name": spec.get("name", key),
                "variants": spec.get("variants") or [{"id": spec.get("model_id", ""), "label": spec.get("model_id", "")}],
            })
    return result


@router.get("/api/admin/article-review/settings")
def review_settings(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return {"settings": contributor_store.get_review_settings(), "models": _model_options()}


@router.put("/api/admin/article-review/settings")
def save_review_settings(payload: ReviewSettingsPayload, request: Request) -> dict[str, Any]:
    user = _require_admin(request)
    try:
        values = payload.model_dump(exclude_none=True)
        valid = {(m["key"], v["id"]) for m in _model_options() for v in m["variants"]}
        if values.get("primary_model_key") and (values["primary_model_key"], values.get("primary_model_id")) not in valid:
            raise ValueError("主审模型不在已启用模型列表中")
        if values.get("fallback_model_key") and (values["fallback_model_key"], values.get("fallback_model_id")) not in valid:
            raise ValueError("备用模型不在已启用模型列表中")
        return contributor_store.update_review_settings(values, user["username"])
    except ValueError as exc:
        raise _failure(exc) from exc


@router.get("/api/admin/article-review/dashboard")
def review_dashboard(request: Request) -> dict[str, Any]:
    _require_admin(request)
    return contributor_store.review_dashboard()
