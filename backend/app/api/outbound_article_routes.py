from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..services import outbound_article_store
from .auth_routes import _current_user
from .routes import _dataset_scope


router = APIRouter(prefix="/api/outbound-articles")


def _csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _visibility(request: Request) -> tuple[str | None, list[str] | None]:
    user = _current_user(request)
    owner = None if user["role"] == "admin" else user["username"]
    return owner, _dataset_scope(request)


@router.post("")
async def import_article(
    request: Request,
    file: UploadFile = File(...),
    platform: str = Form(...),
    url: str = Form(...),
    published_at: str | None = Form(None),
    title: str | None = Form(None),
    product_code: str | None = Form(None),
    campaign: str | None = Form(None),
) -> dict[str, Any]:
    """导入一篇外发文章及其发布记录（MD / TXT / DOCX / PDF）。"""
    user = _current_user(request)
    content = await file.read()
    try:
        article = outbound_article_store.create_article(
            username=user["username"], filename=file.filename or "article",
            content=content, platform=platform, url=url, published_at=published_at,
            title=title, product_code=product_code, campaign=campaign,
        )
        outbound_article_store.refresh_matches(user["username"], _dataset_scope(request))
        return article
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("")
def article_dashboard(
    request: Request,
    dataset_ids: str | None = None,
    product_codes: str | None = None,
    models: str | None = None,
    search_modes: str | None = None,
) -> dict[str, Any]:
    owner, allowed = _visibility(request)
    modes = _csv(search_modes)
    if any(item not in {"0", "1"} for item in modes):
        raise HTTPException(status_code=422, detail="search_modes 仅支持 0/1")
    return outbound_article_store.list_dashboard(
        username=owner,
        allowed=allowed,
        filters={
            "dataset_ids": _csv(dataset_ids),
            "product_codes": _csv(product_codes),
            "models": _csv(models),
            "search_modes": [int(item) for item in modes],
        },
    )


@router.get("/{article_id}/citations")
def article_citations(
    article_id: str,
    request: Request,
    dataset_ids: str | None = None,
    models: str | None = None,
    search_modes: str | None = None,
) -> list[dict[str, Any]]:
    owner, allowed = _visibility(request)
    modes = _csv(search_modes)
    if any(item not in {"0", "1"} for item in modes):
        raise HTTPException(status_code=422, detail="search_modes 仅支持 0/1")
    try:
        return outbound_article_store.list_citations(
            article_id, username=owner, allowed=allowed,
            filters={
                "dataset_ids": _csv(dataset_ids), "models": _csv(models),
                "search_modes": [int(item) for item in modes],
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{article_id}")
def remove_article(article_id: str, request: Request) -> dict[str, str]:
    owner, _ = _visibility(request)
    try:
        outbound_article_store.delete_article(article_id, owner)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok"}
