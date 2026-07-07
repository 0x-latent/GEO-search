from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.paths import GEO_SQLITE_PATH


def _connect() -> sqlite3.Connection:
    if not GEO_SQLITE_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {GEO_SQLITE_PATH}")
    conn = sqlite3.connect(GEO_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def _scope_conditions(
    dataset_id: str | None, allowed: list[str] | None, column: str
) -> tuple[list[str], list[Any]]:
    conds: list[str] = []
    params: list[Any] = []
    if dataset_id and dataset_id != "all":
        conds.append(f"{column} = ?")
        params.append(dataset_id)
    if allowed is not None:
        if not allowed:
            conds.append("1 = 0")
        else:
            placeholders = ",".join("?" for _ in allowed)
            conds.append(f"{column} IN ({placeholders})")
            params.extend(allowed)
    return conds, params


def _dataset_where(
    dataset_id: str | None, alias: str = "a", allowed: list[str] | None = None
) -> tuple[str, list[Any]]:
    conds, params = _scope_conditions(dataset_id, allowed, f"{alias}.dataset_id")
    if conds:
        return "WHERE " + " AND ".join(conds), params
    return "", params


def _dataset_filter(
    dataset_id: str | None, column: str = "dataset_id", allowed: list[str] | None = None
) -> tuple[str, list[Any]]:
    conds, params = _scope_conditions(dataset_id, allowed, column)
    if conds:
        return "WHERE " + " AND ".join(conds), params
    return "", params


def get_owned_dataset_ids(username: str) -> list[str]:
    if not GEO_SQLITE_PATH.exists():
        return []
    with _connect() as conn:
        rows = conn.execute(
            "SELECT dataset_id FROM datasets WHERE owner_username = ?", (username,)
        ).fetchall()
    return [row["dataset_id"] for row in rows]


def delete_dataset(dataset_id: str) -> None:
    with _connect() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
        if cursor.rowcount == 0:
            raise ValueError(f"数据集不存在: {dataset_id}")
        conn.commit()


def list_sqlite_datasets(allowed: list[str] | None = None) -> list[dict[str, Any]]:
    # 每张表先各自按 dataset_id 聚合再 JOIN，避免多表 LEFT JOIN 笛卡尔积
    # （旧写法 questions×answers×sources 行数爆炸，低配服务器上单次查询 20s+）
    where, params = _dataset_filter("all", "d.dataset_id", allowed)
    with _connect() as conn:
        return _rows(
            conn.execute(
                f"""
                SELECT d.dataset_id, d.name, d.description, d.source_type, d.source_path,
                       d.owner_username, d.product_code, d.batch_date, d.question_set_id,
                       COALESCE(q.questions, 0) AS questions,
                       COALESCE(a.answers, 0) AS answers,
                       COALESCE(q.products, 0) AS products,
                       COALESCE(a.models, 0) AS models,
                       COALESCE(s.source_urls, 0) AS source_urls,
                       COALESCE(ext.tables, 0) AS external_tables,
                       COALESCE(ext.rows, 0) AS external_rows
                FROM datasets d
                LEFT JOIN (
                    SELECT dataset_id, COUNT(DISTINCT question_id) AS questions,
                           COUNT(DISTINCT product_code) AS products
                    FROM questions GROUP BY dataset_id
                ) q ON q.dataset_id = d.dataset_id
                LEFT JOIN (
                    SELECT dataset_id, COUNT(*) AS answers, COUNT(DISTINCT model) AS models
                    FROM answers GROUP BY dataset_id
                ) a ON a.dataset_id = d.dataset_id
                LEFT JOIN (
                    SELECT dataset_id, COUNT(DISTINCT url) AS source_urls
                    FROM sources GROUP BY dataset_id
                ) s ON s.dataset_id = d.dataset_id
                LEFT JOIN (
                    SELECT dataset_id, COUNT(*) AS tables, SUM(row_count) AS rows
                    FROM external_tables
                    GROUP BY dataset_id
                ) ext ON ext.dataset_id = d.dataset_id
                {where}
                ORDER BY d.dataset_id
                """,
                params,
            )
        )


def build_sqlite_overview(
    dataset_id: str | None = None, allowed: list[str] | None = None
) -> dict[str, Any]:
    with _connect() as conn:
        answer_where, answer_params = _dataset_where(dataset_id, "a", allowed)
        question_where, question_params = _dataset_filter(dataset_id, "dataset_id", allowed)
        external_where, external_params = _dataset_filter(dataset_id, "dataset_id", allowed)

        cards = dict(
            conn.execute(
                f"""
                SELECT
                  COUNT(DISTINCT a.dataset_id) AS datasets,
                  COUNT(DISTINCT q.product_code) AS products,
                  COUNT(DISTINCT q.question_id) AS questions,
                  COUNT(DISTINCT a.answer_id) AS answers,
                  COUNT(DISTINCT a.model) AS models,
                  COUNT(DISTINCT s.url) AS source_urls,
                  ROUND(AVG(a.answer_chars), 0) AS avg_answer_chars
                FROM answers a
                JOIN questions q ON q.dataset_id = a.dataset_id AND q.question_id = a.question_id
                LEFT JOIN sources s ON s.dataset_id = a.dataset_id AND s.answer_id = a.answer_id
                {answer_where}
                """,
                answer_params,
            ).fetchone()
        )

        external = dict(
            conn.execute(
                f"""
                SELECT COUNT(*) AS external_tables, COALESCE(SUM(row_count), 0) AS external_rows
                FROM external_tables
                {external_where}
                """,
                external_params,
            ).fetchone()
        )
        cards.update(external)

        dataset_rows = list_sqlite_datasets(allowed)
        if dataset_id and dataset_id != "all":
            dataset_rows = [row for row in dataset_rows if row["dataset_id"] == dataset_id]

        product_rows = _rows(
            conn.execute(
                f"""
                SELECT q.dataset_id, q.product_name, q.product_code,
                       COUNT(DISTINCT q.question_id) AS questions,
                       COUNT(DISTINCT a.answer_id) AS answers,
                       COUNT(DISTINCT a.model) AS models,
                       SUM(CASE WHEN a.search_enabled = 1 THEN 1 ELSE 0 END) AS search_answers,
                       SUM(CASE WHEN a.search_enabled = 0 THEN 1 ELSE 0 END) AS nosearch_answers,
                       ROUND(AVG(a.answer_chars), 0) AS avg_answer_chars
                FROM answers a
                JOIN questions q ON q.dataset_id = a.dataset_id AND q.question_id = a.question_id
                {answer_where}
                GROUP BY q.dataset_id, q.product_code, q.product_name
                ORDER BY answers DESC, q.product_name
                """,
                answer_params,
            )
        )

        model_rows = _rows(
            conn.execute(
                f"""
                SELECT a.dataset_id, a.model, COALESCE(a.model_name, a.model) AS model_name,
                       COUNT(*) AS answers,
                       COUNT(DISTINCT a.question_id) AS questions,
                       SUM(CASE WHEN a.search_enabled = 1 THEN 1 ELSE 0 END) AS search_answers,
                       SUM(CASE WHEN a.search_enabled = 0 THEN 1 ELSE 0 END) AS nosearch_answers,
                       ROUND(AVG(a.answer_chars), 0) AS avg_answer_chars,
                       SUM(a.source_count) AS source_refs
                FROM answers a
                {answer_where}
                GROUP BY a.dataset_id, a.model, a.model_name
                ORDER BY a.dataset_id, answers DESC
                """,
                answer_params,
            )
        )

        level_rows = _rows(
            conn.execute(
                f"""
                SELECT dataset_id,
                       COALESCE(source_level, '') AS source_level,
                       COALESCE(level, '') AS level,
                       COUNT(DISTINCT question_id) AS questions
                FROM questions
                {question_where}
                GROUP BY dataset_id, source_level, level
                ORDER BY dataset_id, questions DESC
                """,
                question_params,
            )
        )

        source_where, source_params = _dataset_filter(dataset_id, "s.dataset_id", allowed)
        source_rows = _rows(
            conn.execute(
                f"""
                SELECT s.dataset_id, s.domain, COUNT(*) AS refs
                FROM sources s
                {source_where}
                GROUP BY s.dataset_id, s.domain
                HAVING s.domain <> ''
                ORDER BY refs DESC
                LIMIT 30
                """,
                source_params,
            )
        )

        ext_where, ext_params = _dataset_filter(dataset_id, "et.dataset_id", allowed)
        external_rows = _rows(
            conn.execute(
                f"""
                SELECT et.dataset_id, inf.file_name, et.table_name, et.sheet_name, et.row_count
                FROM external_tables et
                JOIN import_files inf ON inf.file_id = et.file_id
                {ext_where}
                ORDER BY et.row_count DESC
                LIMIT 80
                """,
                ext_params,
            )
        )

        search_rows = _rows(
            conn.execute(
                f"""
                SELECT a.dataset_id,
                       CASE WHEN a.search_enabled = 1 THEN '联网' ELSE '离线' END AS search_mode,
                       COUNT(*) AS answers,
                       ROUND(AVG(a.answer_chars), 0) AS avg_answer_chars,
                       SUM(a.source_count) AS source_refs
                FROM answers a
                {answer_where}
                GROUP BY a.dataset_id, a.search_enabled
                ORDER BY a.dataset_id, a.search_enabled
                """,
                answer_params,
            )
        )

        scenario_rows = _rows(
            conn.execute(
                f"""
                SELECT dataset_id, COALESCE(source_level, level) AS source_level,
                       COALESCE(scenario, '') AS scenario,
                       COUNT(DISTINCT question_id) AS questions
                FROM questions
                {question_where}
                GROUP BY dataset_id, source_level, scenario
                HAVING scenario <> ''
                ORDER BY questions DESC, scenario
                LIMIT 80
                """,
                question_params,
            )
        )

    return {
        "cards": cards,
        "datasets": dataset_rows,
        "products": product_rows,
        "models": model_rows,
        "levels": level_rows,
        "search_modes": search_rows,
        "sources": source_rows,
        "external_tables": external_rows,
        "scenarios": scenario_rows,
        "database": str(Path(GEO_SQLITE_PATH)),
    }


def list_answer_samples(
    dataset_id: str | None = None, limit: int = 100, allowed: list[str] | None = None
) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 500))
    with _connect() as conn:
        where, params = _dataset_where(dataset_id, "a", allowed)
        params.append(limit)
        return _rows(
            conn.execute(
                f"""
                SELECT a.dataset_id, a.product_name, q.source_level, q.scenario,
                       a.question_id, q.question_text, a.model,
                       CASE WHEN a.search_enabled = 1 THEN '联网' ELSE '离线' END AS search_mode,
                       a.round, a.answer_chars, a.source_count,
                       SUBSTR(a.answer_text, 1, 220) AS answer_preview
                FROM answers a
                JOIN questions q ON q.dataset_id = a.dataset_id AND q.question_id = a.question_id
                {where}
                ORDER BY a.dataset_id, a.product_name, a.model, a.question_id, a.round
                LIMIT ?
                """,
                params,
            )
        )


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _yang_recommendation_strength(row: dict[str, Any]) -> str:
    if _num(row.get("首位率")) > 0:
        return "首位"
    if _num(row.get("前三率")) > 0:
        return "前三"
    if _num(row.get("能见度")) > 0:
        return "提及"
    return "未提及"


_SEARCH_LABEL = {"1": "是", "0": "否", "agg": "汇总"}


def _search_label(value: Any) -> str:
    return _SEARCH_LABEL.get(str(value), str(value))


def _json(value: Any) -> dict[str, Any]:
    import json

    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}


def _product_labels(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    rows = conn.execute(
        "SELECT DISTINCT dataset_id, product_code, product_name FROM questions"
    ).fetchall()
    return {
        (row["dataset_id"], row["product_code"] or ""): row["product_name"] or row["product_code"] or ""
        for row in rows
    }


def build_split_performance(
    dataset_id: str | None = None, allowed: list[str] | None = None
) -> dict[str, Any]:
    """拆分表现：读物化表（metrics_* / metric_evidence），保持旧 JSON 形状。

    聚合已在导入时由 `manage_geo_sqlite.py materialize` 落库，这里只做查询和
    形状映射；未物化的数据集返回空（请先跑 materialize）。
    """
    with _connect() as conn:
        labels = _product_labels(conn)

        def label_of(ds: str, code: str, extra: dict[str, Any]) -> str:
            return extra.get("label") or labels.get((ds, code or ""), code or "")

        where_ms, params_ms = _dataset_filter(dataset_id, "dataset_id", allowed)
        summary_rows = _rows(conn.execute(
            f"SELECT * FROM metrics_summary {where_ms}", params_ms
        ))
        rec_rows = _rows(conn.execute(
            f"SELECT * FROM metrics_recommendation {where_ms}", params_ms
        ))
        where_ev, params_ev = _dataset_filter(dataset_id, "dataset_id", allowed)
        type_rows = _rows(conn.execute(
            f"""
            SELECT dataset_id, product_code, question_level, model, search_enabled, name_type,
                   COUNT(*) AS entries,
                   SUM(CASE WHEN strength = 'strong' THEN 1 ELSE 0 END) AS strong,
                   COUNT(DISTINCT CASE WHEN rec_product <> '' THEN rec_product END) AS rec_products
            FROM metric_evidence
            {where_ev} {'AND' if where_ev else 'WHERE'} evidence_type = 'recommendation'
            GROUP BY dataset_id, product_code, question_level, model, search_enabled, name_type
            """,
            params_ev,
        ))
        detail_rows = _rows(conn.execute(
            f"""
            SELECT * FROM metric_evidence
            {where_ev} {'AND' if where_ev else 'WHERE'} evidence_type = 'recommendation'
            LIMIT 1500
            """,
            params_ev,
        ))
        yang_evidence = _rows(conn.execute(
            f"""
            SELECT * FROM metric_evidence
            {where_ev} {'AND' if where_ev else 'WHERE'} evidence_type = 'yang_metric'
            """,
            params_ev,
        ))

    # ---- mention_summary：标准 mention 行 + 养胃舒目标品牌行 ----
    mention_summary = []
    for row in summary_rows:
        extra = _json(row.get("extra_json"))
        if extra.get("_source") != "mention":
            continue
        mention_summary.append({
            "dataset_id": row["dataset_id"],
            "产品": label_of(row["dataset_id"], row["product_code"], extra),
            "问题层级": row["question_level"],
            "模型": row["model"],
            "联网": _search_label(row["search_enabled"]),
            "总回答数": row["total_answers"],
            "品类提及率": row["category_mention_rate"],
            "999品牌提及率": row["brand_mention_rate"],
            "999品牌推荐率": row["brand_rec_rate"],
            "通用名提及率": row["generic_mention_rate"],
            "通用名推荐率": row["generic_rec_rate"],
            "竞品品牌提及率": row["competitor_mention_rate"],
            "竞品品牌推荐率": row["competitor_rec_rate"],
            "负面提及数": row["negative_count"],
            "负面提及率": row["negative_rate"],
        })

    yang_rec_rows = [r for r in rec_rows if r["name_type"] == "目标品牌"]
    std_rec_rows = [r for r in rec_rows if r["name_type"] not in ("目标品牌", "品类")]
    cat_rec_rows = [r for r in rec_rows if r["name_type"] == "品类"]

    yang_summary = []
    for row in yang_rec_rows:
        extra = _json(row.get("extra_json"))
        yang_summary.append({
            "dataset_id": row["dataset_id"],
            "层级": row["question_level"],
            "模型": row["model"],
            "目标品牌": row["rec_product"],
            "样本数": extra.get("样本数"),
            "平均能见度": row["mention_rate"],
            "平均前三率": row["strong_rate"],
            "平均首位率": extra.get("平均首位率"),
            "平均位次": extra.get("平均位次"),
        })
    yang_summary.sort(key=lambda item: (_num(item.get("平均能见度")), _num(item.get("平均前三率"))), reverse=True)

    mention_summary.extend({
        "dataset_id": item["dataset_id"],
        "产品": "养胃舒专项",
        "问题层级": item["层级"],
        "模型": item["模型"],
        "联网": "汇总",
        "总回答数": item["样本数"],
        "目标品牌": item["目标品牌"],
        "目标品牌提及率": item["平均能见度"],
        "目标品牌前三率": item["平均前三率"],
        "目标品牌首位率": item["平均首位率"],
        "平均位次": item["平均位次"],
    } for item in yang_summary)

    # ---- rec_overview：标准推荐排行 + 养胃舒排行 ----
    rec_overview = []
    for row in std_rec_rows:
        extra = _json(row.get("extra_json"))
        rec_overview.append({
            "dataset_id": row["dataset_id"],
            "产品": label_of(row["dataset_id"], row["product_code"], extra),
            "模型": row["model"],
            "联网": _search_label(row["search_enabled"]),
            "排名": row["rank"],
            "被推荐产品": row["rec_product"],
            "名称类型": row["name_type"],
            "提及次数": row["mention_count"],
            "提及率": row["mention_rate"],
            "强推荐次数": row["strong_count"],
            "强推荐率": row["strong_rate"],
            "可选次数": extra.get("可选次数"),
        })
    for row in yang_rec_rows:
        extra = _json(row.get("extra_json"))
        rec_overview.append({
            "dataset_id": row["dataset_id"],
            "产品": "养胃舒专项",
            "问题层级": row["question_level"],
            "模型": row["model"],
            "联网": "汇总",
            "排名": row["rank"],
            "被推荐产品": row["rec_product"],
            "名称类型": "目标品牌",
            "提及次数": row["mention_count"],
            "提及率": row["mention_rate"],
            "强推荐次数": row["strong_count"],
            "强推荐率": row["strong_rate"],
            "平均首位率": extra.get("平均首位率"),
            "平均位次": extra.get("平均位次"),
            "可选次数": extra.get("样本数"),
        })
    rec_overview.sort(key=lambda item: (_num(item.get("提及次数")), _num(item.get("强推荐率"))), reverse=True)

    # ---- type_summary：名称类型汇总（SQL 聚合证据表） + 养胃舒 ----
    type_summary = []
    for row in type_rows:
        entries = row["entries"] or 0
        type_summary.append({
            "dataset_id": row["dataset_id"],
            "产品": labels.get((row["dataset_id"], row["product_code"] or ""), row["product_code"]),
            "问题层级": row["question_level"],
            "模型": row["model"],
            "联网": _search_label(row["search_enabled"]),
            "名称类型": row["name_type"],
            "推荐条目数": entries,
            "强推荐数": row["strong"] or 0,
            "强推荐占比": round((row["strong"] or 0) / entries, 4) if entries else 0,
            "涉及推荐产品数": row["rec_products"] or 0,
        })
    yang_type_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in yang_summary:
        yang_type_groups[(item["dataset_id"], item["层级"], item["模型"])].append(item)
    for (ds, level, model), items in yang_type_groups.items():
        strong_count = sum(1 for item in items if _num(item.get("平均前三率")) > 0)
        type_summary.append({
            "dataset_id": ds,
            "产品": "养胃舒专项",
            "问题层级": level,
            "模型": model,
            "联网": "汇总",
            "名称类型": "目标品牌",
            "推荐条目数": len(items),
            "强推荐数": strong_count,
            "强推荐占比": round(strong_count / len(items), 4) if items else 0,
            "涉及推荐产品数": len({item.get("目标品牌") for item in items if item.get("目标品牌")}),
        })
    type_summary.sort(key=lambda item: item["推荐条目数"], reverse=True)

    # ---- category_summary：品类结构 + 养胃舒层级汇总 ----
    category_summary_rows = []
    for row in cat_rec_rows:
        extra = _json(row.get("extra_json"))
        category_summary_rows.append({
            "dataset_id": row["dataset_id"],
            "产品": label_of(row["dataset_id"], row["product_code"], extra),
            "模型": row["model"],
            "联网": _search_label(row["search_enabled"]),
            "品类": row["rec_product"],
            "推荐次数": row["mention_count"],
            "强推荐数": row["strong_count"],
        })
    yang_category: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in yang_rec_rows:
        key = (row["dataset_id"], "养胃舒专项", row["model"], "汇总", row["question_level"])
        bucket = yang_category.setdefault(
            key,
            {
                "dataset_id": key[0],
                "产品": key[1],
                "模型": key[2],
                "联网": key[3],
                "品类": key[4],
                "推荐次数": 0,
                "强推荐数": 0,
            },
        )
        bucket["推荐次数"] += _num(row["mention_count"])
        bucket["强推荐数"] += _num(row["strong_count"])
    category_summary_rows.extend(yang_category.values())
    category_summary_rows.sort(key=lambda item: _num(item["推荐次数"]), reverse=True)

    # ---- question_details：推荐明细（限 1500） + 养胃舒逐题指标 ----
    question_details = []
    for row in detail_rows:
        payload = _json(row.get("payload_json"))
        question_details.append({
            "dataset_id": row["dataset_id"],
            "问题ID": row["question_id"],
            "产品": payload.get("label") or labels.get((row["dataset_id"], row["product_code"] or ""), ""),
            "问题层级": row["question_level"],
            "模型": row["model"],
            "联网": "是" if row["search_enabled"] else "否",
            "轮次": row["round"],
            "推荐排名": row["rank"],
            "推荐产品": row["rec_product"],
            "名称类型": row["name_type"],
            "推荐强度": row["strength"],
            "推荐原因": row["detail"],
        })
    yang_question_rows = []
    for row in yang_evidence:
        payload = _json(row.get("payload_json"))
        yang_question_rows.append({
            "dataset_id": row["dataset_id"],
            "层级": row["question_level"],
            "提问词": row["detail"],
            "模型": row["model"],
            "目标品牌": row["rec_product"],
            "能见度": payload.get("能见度"),
            "位次": payload.get("位次"),
            "前三率": payload.get("前三率"),
            "首位率": payload.get("首位率"),
            "轮数": payload.get("轮数"),
        })
    question_details.extend({
        "dataset_id": row["dataset_id"],
        "问题ID": "",
        "产品": "养胃舒专项",
        "问题层级": row["层级"],
        "模型": row["模型"],
        "联网": "汇总",
        "轮次": row.get("轮数"),
        "推荐排名": row.get("位次"),
        "推荐产品": row.get("目标品牌"),
        "名称类型": "目标品牌",
        "推荐强度": _yang_recommendation_strength(row),
        "推荐原因": f"提问词：{row.get('提问词')}；能见度：{row.get('能见度')}；前三率：{row.get('前三率')}；首位率：{row.get('首位率')}",
    } for row in yang_question_rows)

    return {
        "mention_summary": mention_summary,
        "rec_overview": rec_overview[:800],
        "type_summary": type_summary[:500],
        "category_summary": category_summary_rows[:500],
        "question_details": question_details,
        "yangweishu_brand_summary": yang_summary[:800],
        "yangweishu_question_metrics": yang_question_rows[:1500],
    }
