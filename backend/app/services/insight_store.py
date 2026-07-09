"""
业务洞察查询层：品牌总览健康卡、产品三阶段旅程、跨批次趋势、证据链下钻。

数据全部来自物化表（metrics_summary / metrics_recommendation / metric_evidence /
dataset_products），由 `manage_geo_sqlite.py materialize` 在导入时写入——这里只查询，
不做重聚合，保证毫秒级响应。

行的类型判别约定（见 materialize）：
- mention 行：total_answers 非空（提及/推荐率主指标，附负面计数）
- accuracy 行：accuracy_rate 非空（07 校验聚合）
- 养胃舒专项行：search_enabled = 'agg'
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from ..core.paths import GEO_SQLITE_PATH
from utils.sqlite_schema import STAGES

SEARCH_LABELS = {"1": "联网", "0": "非联网", "agg": "汇总"}

# 趋势/卡片可用的指标列白名单
METRIC_COLUMNS = {
    "category_mention_rate", "brand_mention_rate", "brand_rec_rate",
    "generic_mention_rate", "generic_rec_rate",
    "competitor_mention_rate", "competitor_rec_rate",
    "negative_rate", "accuracy_rate", "first_rate", "top3_rate", "avg_rank",
}

# 每个阶段的"核心结论指标"（总览卡片和趋势默认线）
STAGE_KEY_METRIC = {
    "symptom": "category_mention_rate",
    "category": "brand_mention_rate",
    "brand": "accuracy_rate",
}


def _connect() -> sqlite3.Connection:
    if not GEO_SQLITE_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {GEO_SQLITE_PATH}")
    conn = sqlite3.connect(GEO_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def _wavg(pairs: list[tuple[Any, Any]]) -> float | None:
    points = [(float(v), float(w) if w else 1.0) for v, w in pairs if v is not None]
    if not points:
        return None
    total_weight = sum(w for _, w in points)
    if total_weight <= 0:
        return None
    return round(sum(v * w for v, w in points) / total_weight, 4)


def _scope_clause(allowed: list[str] | None, column: str) -> tuple[str, list[Any]]:
    if allowed is None:
        return "", []
    if not allowed:
        return f" AND 1 = 0", []
    placeholders = ",".join("?" for _ in allowed)
    return f" AND {column} IN ({placeholders})", list(allowed)


def _product_batches(
    conn: sqlite3.Connection, product_code: str, allowed: list[str] | None
) -> list[dict[str, Any]]:
    scope_sql, scope_params = _scope_clause(allowed, "d.dataset_id")
    return _rows(conn.execute(
        f"""
        SELECT d.dataset_id, d.name, d.owner_username,
               COALESCE(d.batch_date, substr(d.imported_at, 1, 10)) AS batch_date,
               dp.question_set_id, dp.question_count
        FROM dataset_products dp
        JOIN datasets d ON d.dataset_id = dp.dataset_id
        WHERE dp.product_code = ?{scope_sql}
        ORDER BY batch_date, d.dataset_id
        """,
        [product_code, *scope_params],
    ))


def _summary_rows(
    conn: sqlite3.Connection,
    dataset_id: str,
    product_code: str,
    model: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    conds = ["dataset_id = ?", "product_code = ?"]
    params: list[Any] = [dataset_id, product_code]
    if model:
        conds.append("model = ?")
        params.append(model)
    if search:
        conds.append("search_enabled = ?")
        params.append(search)
    return _rows(conn.execute(
        f"SELECT * FROM metrics_summary WHERE {' AND '.join(conds)}", params
    ))


def _stage_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把一个批次的 metrics_summary 行聚成三阶段结论指标（按回答量加权平均）。"""
    result: dict[str, Any] = {stage: {} for stage in STAGES}
    negative_total = 0
    answers_total = 0
    negative_rate_points: list[tuple[Any, Any]] = []
    for stage in STAGES:
        mention = [r for r in rows if r["stage"] == stage and r["total_answers"] is not None]
        agg = {
            "total_answers": sum(r["total_answers"] or 0 for r in mention) or None,
            "category_mention_rate": _wavg([(r["category_mention_rate"], r["total_answers"]) for r in mention]),
            "brand_mention_rate": _wavg([(r["brand_mention_rate"], r["total_answers"]) for r in mention]),
            "brand_rec_rate": _wavg([(r["brand_rec_rate"], r["total_answers"]) for r in mention]),
            "generic_mention_rate": _wavg([(r["generic_mention_rate"], r["total_answers"]) for r in mention]),
            "generic_rec_rate": _wavg([(r["generic_rec_rate"], r["total_answers"]) for r in mention]),
            "competitor_mention_rate": _wavg([(r["competitor_mention_rate"], r["total_answers"]) for r in mention]),
            "competitor_rec_rate": _wavg([(r["competitor_rec_rate"], r["total_answers"]) for r in mention]),
            "first_rate": _wavg([(r["first_rate"], r["total_answers"]) for r in mention]),
            "top3_rate": _wavg([(r["top3_rate"], r["total_answers"]) for r in mention]),
            "avg_rank": _wavg([(r["avg_rank"], r["total_answers"]) for r in mention]),
            "negative_count": sum(r["negative_count"] or 0 for r in mention) or 0,
            "negative_rate": _wavg([(r["negative_rate"], r["total_answers"]) for r in mention]),
        }
        accuracy = [r for r in rows if r["stage"] == stage and r["accuracy_rate"] is not None]
        agg["accuracy_rate"] = _wavg([(r["accuracy_rate"], r["total_claims"]) for r in accuracy])
        agg["wrong_claims"] = sum(r["wrong_claims"] or 0 for r in accuracy) or 0
        agg["total_claims"] = sum(r["total_claims"] or 0 for r in accuracy) or 0
        result[stage] = agg
        negative_total += agg["negative_count"]
        answers_total += agg["total_answers"] or 0
        negative_rate_points.extend(
            (r["negative_rate"], r["total_answers"]) for r in mention
        )

    # 准确率不分阶段兜底：07 对用户上传（统一 q4）也会产出，按全量聚合一份
    accuracy_all = [r for r in rows if r["accuracy_rate"] is not None]
    result["overall"] = {
        "accuracy_rate": _wavg([(r["accuracy_rate"], r["total_claims"]) for r in accuracy_all]),
        "wrong_claims": sum(r["wrong_claims"] or 0 for r in accuracy_all) or 0,
        "total_claims": sum(r["total_claims"] or 0 for r in accuracy_all) or 0,
        "negative_count": negative_total,
        # 率按行级 negative_rate（回答去重口径）加权平均；negative_count 是条目数，二者口径不同不可互除
        "negative_rate": _wavg(negative_rate_points),
        "total_answers": answers_total or None,
    }
    return result


def list_product_insights(allowed: list[str] | None = None) -> list[dict[str, Any]]:
    """品牌总览健康卡：由数据驱动——只展示当前可见数据集里真实存在的产品。

    产品显示名/品类优先取主数据（products 表），数据中存在但主数据没有的
    产品（如用户上传的新品）用数据里的名称兜底展示。
    """
    with _connect() as conn:
        scope_sql, scope_params = _scope_clause(allowed, "d.dataset_id")
        data_products = _rows(conn.execute(
            f"""
            SELECT dp.product_code,
                   MAX(dp.product_name) AS data_name,
                   COUNT(DISTINCT dp.dataset_id) AS batch_count
            FROM dataset_products dp
            JOIN datasets d ON d.dataset_id = dp.dataset_id
            WHERE dp.product_code <> ''{scope_sql}
            GROUP BY dp.product_code
            """,
            scope_params,
        ))
        master = {
            row["product_code"]: dict(row)
            for row in conn.execute(
                "SELECT product_code, product_name, category, is_active, display_order FROM products"
            )
        }
        # 主数据排序优先，数据中的新产品排在后面
        data_products.sort(key=lambda item: (
            master.get(item["product_code"], {}).get("display_order", 999),
            item["product_code"],
        ))
        cards = []
        for item in data_products:
            code = item["product_code"]
            info = master.get(code, {})
            batches = _product_batches(conn, code, allowed)
            if not batches:
                continue
            card: dict[str, Any] = {
                "product_code": code,
                "product_name": info.get("product_name") or item["data_name"] or code,
                "category": info.get("category"),
                "is_active": bool(info.get("is_active", 0)),
                "batch_count": len(batches),
                "latest_batch": None,
                "metrics": None,
                "delta": None,
            }
            if batches:
                latest = batches[-1]
                rollup = _stage_rollup(_summary_rows(conn, latest["dataset_id"], code))
                card["latest_batch"] = {
                    "dataset_id": latest["dataset_id"],
                    "batch_date": latest["batch_date"],
                    "name": latest["name"],
                }
                card["metrics"] = {
                    "symptom_category_rate": rollup["symptom"]["category_mention_rate"],
                    "category_brand_rate": rollup["category"]["brand_mention_rate"],
                    "category_brand_rec_rate": rollup["category"]["brand_rec_rate"],
                    "accuracy_rate": rollup["overall"]["accuracy_rate"],
                    "negative_count": rollup["overall"]["negative_count"],
                    "negative_rate": rollup["overall"]["negative_rate"],
                }
                # 环比：同问题集指纹的上一批次才可比
                comparable = [
                    b for b in batches[:-1]
                    if b["question_set_id"] == latest["question_set_id"]
                ]
                if comparable:
                    prev = comparable[-1]
                    prev_rollup = _stage_rollup(_summary_rows(conn, prev["dataset_id"], code))
                    def _delta(new: float | None, old: float | None) -> float | None:
                        if new is None or old is None:
                            return None
                        return round(new - old, 4)
                    card["delta"] = {
                        "vs_dataset_id": prev["dataset_id"],
                        "vs_batch_date": prev["batch_date"],
                        "symptom_category_rate": _delta(
                            rollup["symptom"]["category_mention_rate"],
                            prev_rollup["symptom"]["category_mention_rate"]),
                        "category_brand_rate": _delta(
                            rollup["category"]["brand_mention_rate"],
                            prev_rollup["category"]["brand_mention_rate"]),
                        "accuracy_rate": _delta(
                            rollup["overall"]["accuracy_rate"],
                            prev_rollup["overall"]["accuracy_rate"]),
                        "negative_rate": _delta(
                            rollup["overall"]["negative_rate"],
                            prev_rollup["overall"]["negative_rate"]),
                    }
            cards.append(card)
        return cards


def _competitors(
    conn: sqlite3.Connection,
    dataset_id: str,
    product_code: str,
    stage: str,
    model: str | None,
    search: str | None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """竞品/品类推荐排行。跨模型×联网聚合成一行（未筛选时），点开明细再拆。

    品牌阶段没有竞品排行（看准确率和负面），返回空。
    """
    if stage == "brand":
        return []
    conds = ["dataset_id = ?", "product_code = ?"]
    params: list[Any] = [dataset_id, product_code]
    if stage == "symptom":
        conds.append("name_type = '品类'")
    else:
        conds.append("name_type IN ('999品牌', '竞品品牌', '通用名', '目标品牌')")
    if model:
        conds.append("model = ?")
        params.append(model)
    if search:
        conds.append("search_enabled = ?")
        params.append(search)
    rows = _rows(conn.execute(
        f"""
        SELECT rec_product, name_type,
               SUM(mention_count) AS mention_count,
               ROUND(AVG(mention_rate), 4) AS mention_rate,
               SUM(strong_count) AS strong_count,
               ROUND(AVG(strong_rate), 4) AS strong_rate,
               MIN(rank) AS best_rank,
               COUNT(*) AS row_count
        FROM metrics_recommendation
        WHERE {' AND '.join(conds)}
        GROUP BY rec_product, name_type
        ORDER BY mention_count DESC
        LIMIT ?
        """,
        [*params, limit],
    ))
    return rows


def _evidence_counts(
    conn: sqlite3.Connection, dataset_id: str, product_code: str, stage: str
) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT evidence_type, COUNT(*) FROM metric_evidence
        WHERE dataset_id = ? AND product_code = ? AND stage = ?
        GROUP BY evidence_type
        """,
        (dataset_id, product_code, stage),
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _trend_points(
    conn: sqlite3.Connection,
    batches: list[dict[str, Any]],
    question_set_id: str | None,
    product_code: str,
    stage: str,
    metric: str,
    model: str | None,
    search: str | None,
) -> list[dict[str, Any]]:
    points = []
    for batch in batches:
        if question_set_id and batch["question_set_id"] != question_set_id:
            continue
        rollup = _stage_rollup(
            _summary_rows(conn, batch["dataset_id"], product_code, model, search)
        )
        source = rollup["overall"] if metric in ("accuracy_rate",) else rollup[stage]
        points.append({
            "dataset_id": batch["dataset_id"],
            "batch_date": batch["batch_date"],
            "value": source.get(metric),
        })
    return points


def _scenario_breakdown(
    conn: sqlite3.Connection,
    dataset_id: str,
    product_code: str,
    model: str | None,
    search: str | None,
) -> list[dict[str, Any]]:
    """场景拆解：跨模型/联网聚合每个场景的品牌表现（按回答量加权）。"""
    conds = ["dataset_id = ?", "product_code = ?"]
    params: list[Any] = [dataset_id, product_code]
    if model:
        conds.append("model = ?")
        params.append(model)
    if search:
        conds.append("search_enabled = ?")
        params.append(search)
    rows = _rows(conn.execute(
        f"SELECT * FROM metrics_scenario WHERE {' AND '.join(conds)}", params
    ))
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = grouped.setdefault(row["scenario"], {
            "scenario": row["scenario"],
            "question_count": 0,
            "total_answers": 0,
            "_mention": [], "_rec": [], "_neg": [],
            "negative_count": 0,
            "_categories": {},
            "sources": set(),
        })
        item["question_count"] = max(item["question_count"], row["question_count"] or 0)
        item["total_answers"] += row["total_answers"] or 0
        weight = row["total_answers"] or 1
        if row["brand_mention_rate"] is not None:
            item["_mention"].append((row["brand_mention_rate"], weight))
        if row["brand_rec_rate"] is not None:
            item["_rec"].append((row["brand_rec_rate"], weight))
        if row["negative_rate"] is not None:
            item["_neg"].append((row["negative_rate"], weight))
        item["negative_count"] += row["negative_count"] or 0
        item["sources"].add(row["search_enabled"])
        try:
            for cat in json.loads(row["top_categories_json"] or "[]"):
                item["_categories"][cat["category"]] = (
                    item["_categories"].get(cat["category"], 0) + cat["count"]
                )
        except (TypeError, ValueError, KeyError):
            pass
    result = []
    for item in grouped.values():
        top = sorted(item["_categories"].items(), key=lambda x: -x[1])[:3]
        result.append({
            "scenario": item["scenario"],
            "question_count": item["question_count"],
            "total_answers": item["total_answers"],
            "brand_mention_rate": _wavg(item["_mention"]),
            "brand_rec_rate": _wavg(item["_rec"]),
            "negative_count": item["negative_count"],
            "negative_rate": _wavg(item["_neg"]),
            "top_categories": [{"category": c, "count": n} for c, n in top],
            "is_vendor_agg": item["sources"] == {"agg"},
        })
    result.sort(key=lambda x: -(x["brand_mention_rate"] or 0))
    return result


def product_journey(
    product_code: str,
    dataset_id: str | None = None,
    model: str | None = None,
    search: str | None = None,
    allowed: list[str] | None = None,
) -> dict[str, Any]:
    """产品详情页主接口：批次列表 + 选中批次的三阶段结论/明细/竞品/趋势/证据计数。"""
    with _connect() as conn:
        product = conn.execute(
            "SELECT product_code, product_name, category FROM products WHERE product_code = ?",
            (product_code,),
        ).fetchone()
        batches = _product_batches(conn, product_code, allowed)
        if not batches:
            return {
                "product_code": product_code,
                "product_name": product["product_name"] if product else product_code,
                "category": product["category"] if product else None,
                "batches": [],
                "selected": None,
                "stages": {},
            }
        selected = batches[-1]
        if dataset_id:
            match = [b for b in batches if b["dataset_id"] == dataset_id]
            if not match:
                raise ValueError(f"数据集不存在或无权访问: {dataset_id}")
            selected = match[0]

        all_rows = _summary_rows(conn, selected["dataset_id"], product_code, model, search)
        rollup = _stage_rollup(all_rows)
        available = _rows(conn.execute(
            """
            SELECT DISTINCT model, search_enabled FROM metrics_summary
            WHERE dataset_id = ? AND product_code = ?
            """,
            (selected["dataset_id"], product_code),
        ))

        stages: dict[str, Any] = {}
        for stage in STAGES:
            stage_rows = [r for r in all_rows if r["stage"] == stage]
            for row in stage_rows:
                row["search_label"] = SEARCH_LABELS.get(row["search_enabled"], row["search_enabled"])
                row.pop("extra_json", None)
            key_metric = STAGE_KEY_METRIC[stage]
            stages[stage] = {
                "summary": rollup[stage],
                "key_metric": key_metric,
                "metrics_by_model": stage_rows,
                "competitors": _competitors(
                    conn, selected["dataset_id"], product_code, stage, model, search
                ),
                "trend": {
                    key_metric: _trend_points(
                        conn, batches, selected["question_set_id"], product_code,
                        stage, key_metric, model, search,
                    ),
                    "negative_rate": _trend_points(
                        conn, batches, selected["question_set_id"], product_code,
                        stage, "negative_rate", model, search,
                    ),
                },
                "evidence_counts": _evidence_counts(
                    conn, selected["dataset_id"], product_code, stage
                ),
            }

        return {
            "product_code": product_code,
            "product_name": product["product_name"] if product else product_code,
            "category": product["category"] if product else None,
            "batches": batches,
            "selected": selected,
            "overall": rollup["overall"],
            "filters": {
                "models": sorted({r["model"] for r in available}),
                "search_modes": sorted({r["search_enabled"] for r in available}),
            },
            "stages": stages,
            "scenarios": _scenario_breakdown(
                conn, selected["dataset_id"], product_code, model, search
            ),
        }


def product_trend(
    product_code: str,
    metric: str,
    stage: str | None = None,
    model: str | None = None,
    search: str | None = None,
    allowed: list[str] | None = None,
) -> dict[str, Any]:
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"不支持的指标: {metric}")
    stage = stage or next(
        (s for s, m in STAGE_KEY_METRIC.items() if m == metric), "category"
    )
    with _connect() as conn:
        batches = _product_batches(conn, product_code, allowed)
        if not batches:
            return {"product_code": product_code, "series": []}
        # 按问题集指纹分组：同指纹批次才连成一条可比趋势线
        groups: dict[str, list[dict[str, Any]]] = {}
        for batch in batches:
            groups.setdefault(batch["question_set_id"], []).append(batch)
        series = []
        for question_set_id, group in groups.items():
            series.append({
                "question_set_id": question_set_id,
                "points": _trend_points(
                    conn, group, question_set_id, product_code, stage, metric, model, search
                ),
            })
        series.sort(key=lambda s: len(s["points"]), reverse=True)
        return {
            "product_code": product_code,
            "metric": metric,
            "stage": stage,
            "series": series,
        }


def evidence_list(
    dataset_id: str,
    evidence_type: str,
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
    allowed: list[str] | None = None,
) -> dict[str, Any]:
    if allowed is not None and dataset_id not in allowed:
        raise ValueError(f"数据集不存在或无权访问: {dataset_id}")
    size = max(1, min(size, 200))
    page = max(1, page)
    conds = ["e.dataset_id = ?", "e.evidence_type = ?"]
    params: list[Any] = [dataset_id, evidence_type]
    for column, value in (
        ("e.product_code", product_code),
        ("e.stage", stage),
        ("e.model", model),
        ("e.rec_product", rec_product),
        ("e.strength", strength),
        ("e.verdict", verdict),
        ("q.scenario", scenario),
    ):
        if value:
            conds.append(f"{column} = ?")
            params.append(value)
    if search_enabled is not None:
        conds.append("e.search_enabled = ?")
        params.append(search_enabled)
    where = " AND ".join(conds)
    with _connect() as conn:
        total = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM metric_evidence e
            LEFT JOIN questions q
              ON q.dataset_id = e.dataset_id AND q.question_id = e.question_id
            WHERE {where}
            """,
            params,
        ).fetchone()[0]
        rows = _rows(conn.execute(
            f"""
            SELECT e.rowid AS evidence_id, e.evidence_type, e.product_code, e.stage,
                   e.question_level, e.question_id, e.model, e.search_enabled, e.round,
                   e.rec_product, e.name_type, e.rank, e.strength, e.verdict, e.detail,
                   e.payload_json, q.question_text
            FROM metric_evidence e
            LEFT JOIN questions q
              ON q.dataset_id = e.dataset_id AND q.question_id = e.question_id
            WHERE {where}
            ORDER BY e.rowid
            LIMIT ? OFFSET ?
            """,
            [*params, size, (page - 1) * size],
        ))
    for row in rows:
        try:
            row["payload"] = json.loads(row.pop("payload_json") or "{}")
        except (TypeError, ValueError):
            row["payload"] = {}
    return {"total": total, "page": page, "size": size, "items": rows}


def answer_full(
    dataset_id: str,
    question_id: str,
    model: str | None = None,
    search_enabled: int | None = None,
    round_num: int | None = None,
    allowed: list[str] | None = None,
) -> dict[str, Any]:
    """证据链终点：AI 原始回答全文 + 信源。精确五元组未命中时返回该问题的全部回答清单。"""
    if allowed is not None and dataset_id not in allowed:
        raise ValueError(f"数据集不存在或无权访问: {dataset_id}")
    with _connect() as conn:
        conds = ["a.dataset_id = ?", "a.question_id = ?"]
        params: list[Any] = [dataset_id, question_id]
        if model:
            conds.append("a.model = ?")
            params.append(model)
        if search_enabled is not None:
            conds.append("a.search_enabled = ?")
            params.append(search_enabled)
        if round_num is not None:
            conds.append("a.round = ?")
            params.append(round_num)
        rows = _rows(conn.execute(
            f"""
            SELECT a.answer_id, a.question_id, q.question_text, q.source_level, q.scenario,
                   a.product_name, a.model, a.model_name, a.search_enabled, a.round,
                   a.timestamp, a.answer_text, a.answer_chars, a.source_count
            FROM answers a
            JOIN questions q ON q.dataset_id = a.dataset_id AND q.question_id = a.question_id
            WHERE {' AND '.join(conds)}
            ORDER BY a.model, a.search_enabled, a.round
            LIMIT 20
            """,
            params,
        ))
        for row in rows:
            row["sources"] = _rows(conn.execute(
                """
                SELECT source_index, title, url, domain FROM sources
                WHERE dataset_id = ? AND answer_id = ?
                ORDER BY source_index
                """,
                (dataset_id, row["answer_id"]),
            ))
    exact = rows[0] if len(rows) == 1 else None
    return {"answer": exact, "candidates": rows if exact is None else []}
