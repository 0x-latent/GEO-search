from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..services import insight_store, product_master
from .routes import _dataset_scope

router = APIRouter(prefix="/api/insight")
products_router = APIRouter(prefix="/api/products")


@router.get("/products")
def get_product_insights(request: Request) -> list[dict[str, Any]]:
    """品牌总览：每产品最新批次健康卡 + 环比 delta。"""
    allowed = _dataset_scope(request)
    try:
        return insight_store.list_product_insights(allowed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/products/{product_code}/journey")
def get_product_journey(
    product_code: str,
    request: Request,
    dataset_id: str | None = None,
    model: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """产品详情：三阶段（病症→品类→品牌）结论、竞品、趋势与证据计数。"""
    allowed = _dataset_scope(request)
    try:
        return insight_store.product_journey(product_code, dataset_id, model, search, allowed)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/products/{product_code}/trend")
def get_product_trend(
    product_code: str,
    request: Request,
    metric: str,
    stage: str | None = None,
    model: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    allowed = _dataset_scope(request)
    try:
        return insight_store.product_trend(product_code, metric, stage, model, search, allowed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/evidence")
def get_evidence(
    request: Request,
    dataset_id: str,
    type: str,
    product_code: str | None = None,
    stage: str | None = None,
    model: str | None = None,
    search_enabled: int | None = None,
    rec_product: str | None = None,
    strength: str | None = None,
    verdict: str | None = None,
    page: int = 1,
    size: int = 50,
) -> dict[str, Any]:
    """证据链：指标 → 问题级明细（推荐/负面/准确率/品类）。"""
    allowed = _dataset_scope(request)
    try:
        return insight_store.evidence_list(
            dataset_id, type, product_code, stage, model, search_enabled,
            rec_product, strength, verdict, page, size, allowed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/answers/{dataset_id}/{question_id}")
def get_answer_full(
    dataset_id: str,
    question_id: str,
    request: Request,
    model: str | None = None,
    search_enabled: int | None = None,
    round: int | None = None,
) -> dict[str, Any]:
    """证据链终点：AI 原始回答全文 + 信源列表。"""
    allowed = _dataset_scope(request)
    try:
        return insight_store.answer_full(
            dataset_id, question_id, model, search_enabled, round, allowed
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@products_router.get("")
def get_products(request: Request) -> list[dict[str, Any]]:
    """产品主数据（上传界面下拉、总览页产品清单）。"""
    return product_master.list_products(active_only=True)
