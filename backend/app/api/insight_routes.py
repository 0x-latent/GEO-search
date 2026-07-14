from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..services import insight_store, product_master, source_insight_store
from .routes import _dataset_scope

router = APIRouter(prefix="/api/insight")
products_router = APIRouter(prefix="/api/products")


def _csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _source_filters(
    dataset_ids: str | None,
    product_codes: str | None,
    models: str | None,
    search_modes: str | None,
    categories: str | None,
    domains: str | None,
    stages: str | None,
    scenarios: str | None,
) -> dict[str, list[Any]]:
    mode_values = _csv_values(search_modes)
    invalid_modes = [item for item in mode_values if item not in {"0", "1"}]
    if invalid_modes:
        raise HTTPException(
            status_code=422,
            detail=f"search_modes 仅接受 0/1，收到：{','.join(invalid_modes)}",
        )
    return {
        "dataset_ids": _csv_values(dataset_ids),
        "product_codes": _csv_values(product_codes),
        "models": _csv_values(models),
        "search_modes": [int(item) for item in mode_values],
        "categories": _csv_values(categories),
        "domains": _csv_values(domains),
        "stages": _csv_values(stages),
        "scenarios": _csv_values(scenarios),
    }


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


@router.get("/sources/options")
def get_source_options(request: Request) -> dict[str, Any]:
    """信源工作台筛选项；只返回当前用户可见数据集中的内容。"""
    allowed = _dataset_scope(request)
    try:
        return source_insight_store.list_options(allowed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sources/analysis")
def get_source_analysis(
    request: Request,
    dataset_ids: str | None = None,
    product_codes: str | None = None,
    models: str | None = None,
    search_modes: str | None = None,
    categories: str | None = None,
    domains: str | None = None,
    stages: str | None = None,
    scenarios: str | None = None,
) -> dict[str, Any]:
    """组合信源分析：覆盖率、分类、域名、产品对比与信源缺口。"""
    allowed = _dataset_scope(request)
    filters = _source_filters(
        dataset_ids, product_codes, models, search_modes,
        categories, domains, stages, scenarios,
    )
    try:
        return source_insight_store.analyze(filters, allowed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sources/answers")
def get_source_answers(
    request: Request,
    domain: str | None = None,
    dataset_ids: str | None = None,
    product_codes: str | None = None,
    models: str | None = None,
    search_modes: str | None = None,
    categories: str | None = None,
    stages: str | None = None,
    scenarios: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """域名/分类 → 回答明细，供信源排行下钻。"""
    allowed = _dataset_scope(request)
    filters = _source_filters(
        dataset_ids, product_codes, models, search_modes,
        categories, None, stages, scenarios,
    )
    try:
        return source_insight_store.source_answers(filters, domain, limit, allowed)
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
    scenario: str | None = None,
    page: int = 1,
    size: int = 50,
) -> dict[str, Any]:
    """证据链：指标 → 问题级明细（推荐/负面/准确率/品类），可按场景过滤。"""
    allowed = _dataset_scope(request)
    try:
        return insight_store.evidence_list(
            dataset_id, type, product_code, stage, model, search_enabled,
            rec_product, strength, verdict, scenario, page, size, allowed,
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
