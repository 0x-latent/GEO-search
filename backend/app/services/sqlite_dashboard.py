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


def ensure_owner_column() -> None:
    if not GEO_SQLITE_PATH.exists():
        return
    with _connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)")}
        if "owner_username" not in columns:
            conn.execute("ALTER TABLE datasets ADD COLUMN owner_username TEXT")
            conn.commit()


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
    where, params = _dataset_filter("all", "d.dataset_id", allowed)
    with _connect() as conn:
        return _rows(
            conn.execute(
                f"""
                SELECT d.dataset_id, d.name, d.description, d.source_type, d.source_path,
                       d.owner_username,
                       COUNT(DISTINCT q.question_id) AS questions,
                       COUNT(DISTINCT a.answer_id) AS answers,
                       COUNT(DISTINCT q.product_code) AS products,
                       COUNT(DISTINCT a.model) AS models,
                       COUNT(DISTINCT s.url) AS source_urls,
                       COALESCE(ext.tables, 0) AS external_tables,
                       COALESCE(ext.rows, 0) AS external_rows
                FROM datasets d
                LEFT JOIN questions q ON q.dataset_id = d.dataset_id
                LEFT JOIN answers a ON a.dataset_id = d.dataset_id
                LEFT JOIN sources s ON s.dataset_id = a.dataset_id AND s.answer_id = a.answer_id
                LEFT JOIN (
                    SELECT dataset_id, COUNT(*) AS tables, SUM(row_count) AS rows
                    FROM external_tables
                    GROUP BY dataset_id
                ) ext ON ext.dataset_id = d.dataset_id
                {where}
                GROUP BY d.dataset_id
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


def _safe_value(value: Any) -> Any:
    if value != value:
        return None
    return value


def _json_rows(
    conn: sqlite3.Connection,
    table_names: set[str] | None = None,
    dataset_id: str | None = None,
    allowed: list[str] | None = None,
) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if table_names:
        placeholders = ",".join("?" for _ in table_names)
        clauses.append(f"et.table_name IN ({placeholders})")
        params.extend(sorted(table_names))
    scope_conds, scope_params = _scope_conditions(dataset_id, allowed, "et.dataset_id")
    clauses.extend(scope_conds)
    params.extend(scope_params)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT et.dataset_id, et.table_name, et.sheet_name, er.row_json
        FROM external_rows er
        JOIN external_tables et ON et.table_id = er.table_id
        {where}
        """,
        params,
    ).fetchall()
    result = []
    for row in rows:
        payload = __import__("json").loads(row["row_json"])
        cleaned = {key: _safe_value(value) for key, value in payload.items()}
        cleaned["_dataset_id"] = row["dataset_id"]
        cleaned["_table_name"] = row["table_name"]
        cleaned["_sheet_name"] = row["sheet_name"]
        result.append(cleaned)
    return result


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


def build_split_performance(
    dataset_id: str | None = None, allowed: list[str] | None = None
) -> dict[str, Any]:
    with _connect() as conn:
        mention_rows = _json_rows(conn, {"mention_report"}, dataset_id, allowed)
        rec_rows = _json_rows(conn, {"rec_overview"}, dataset_id, allowed)
        detail_rows = _json_rows(conn, {"brand_generic_detail"}, dataset_id, allowed)
        category_rows = _json_rows(conn, {"recommendation_detail"}, dataset_id, allowed)

        yang_dataset_visible = allowed is None or "weitai_yangweishu_20260602" in allowed
        yang_rows = []
        if yang_dataset_visible and (not dataset_id or dataset_id in {"all", "weitai_yangweishu_20260602"}):
            rows = conn.execute(
                """
                SELECT et.dataset_id, et.table_name, et.sheet_name, er.row_json
                FROM external_rows er
                JOIN external_tables et ON et.table_id = er.table_id
                WHERE et.dataset_id = 'weitai_yangweishu_20260602'
                  AND et.table_name LIKE '养胃舒-%汇总统计数据%'
                  AND et.sheet_name <> '字段说明'
                """
            ).fetchall()
            for row in rows:
                payload = __import__("json").loads(row["row_json"])
                payload = {key: _safe_value(value) for key, value in payload.items()}
                payload["_dataset_id"] = row["dataset_id"]
                payload["_table_name"] = row["table_name"]
                payload["_sheet_name"] = row["sheet_name"]
                if "提问词" in payload and "AI模型" in payload and "目标品牌" in payload:
                    yang_rows.append(payload)

    mention_summary = []
    for row in mention_rows:
        mention_summary.append({
            "dataset_id": row["_dataset_id"],
            "产品": row.get("产品"),
            "问题层级": row.get("问题层级"),
            "模型": row.get("模型"),
            "联网": row.get("联网"),
            "总回答数": row.get("总回答数"),
            "品类提及率": row.get("品类提及率"),
            "999品牌提及率": row.get("999品牌提及率"),
            "999品牌推荐率": row.get("999品牌推荐率"),
            "通用名提及率": row.get("通用名提及率"),
            "通用名推荐率": row.get("通用名推荐率"),
            "竞品品牌提及率": row.get("竞品品牌提及率"),
            "竞品品牌推荐率": row.get("竞品品牌推荐率"),
        })

    rec_overview = [
        {
            "dataset_id": row["_dataset_id"],
            "产品": row.get("产品"),
            "模型": row.get("模型"),
            "联网": row.get("联网"),
            "排名": row.get("排名"),
            "被推荐产品": row.get("被推荐产品"),
            "名称类型": row.get("名称类型"),
            "提及次数": row.get("提及次数"),
            "提及率": row.get("提及率"),
            "强推荐次数": row.get("强推荐次数"),
            "强推荐率": row.get("强推荐率"),
            "可选次数": row.get("可选次数"),
        }
        for row in rec_rows
    ]
    rec_overview.sort(key=lambda item: (_num(item.get("提及次数")), _num(item.get("强推荐率"))), reverse=True)

    product_type: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in detail_rows:
        key = (
            row.get("_dataset_id"),
            row.get("产品"),
            row.get("问题层级"),
            row.get("模型"),
            row.get("联网"),
            row.get("名称类型"),
        )
        item = product_type.setdefault(
            key,
            {
                "dataset_id": key[0],
                "产品": key[1],
                "问题层级": key[2],
                "模型": key[3],
                "联网": key[4],
                "名称类型": key[5],
                "推荐条目数": 0,
                "强推荐数": 0,
                "涉及推荐产品数": set(),
            },
        )
        item["推荐条目数"] += 1
        if row.get("推荐强度") == "strong":
            item["强推荐数"] += 1
        if row.get("推荐产品"):
            item["涉及推荐产品数"].add(row.get("推荐产品"))
    type_summary = []
    for item in product_type.values():
        item = dict(item)
        item["涉及推荐产品数"] = len(item["涉及推荐产品数"])
        item["强推荐占比"] = round(item["强推荐数"] / item["推荐条目数"], 4) if item["推荐条目数"] else 0
        type_summary.append(item)
    type_summary.sort(key=lambda item: item["推荐条目数"], reverse=True)

    category_summary: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in category_rows:
        key = (row.get("_dataset_id"), row.get("产品"), row.get("模型"), row.get("联网"), row.get("品类"))
        item = category_summary.setdefault(
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
        item["推荐次数"] += 1
        if row.get("推荐强度") == "strong":
            item["强推荐数"] += 1
    category_summary_rows = sorted(category_summary.values(), key=lambda item: item["推荐次数"], reverse=True)

    question_details = [
        {
            "dataset_id": row["_dataset_id"],
            "问题ID": row.get("问题ID"),
            "产品": row.get("产品"),
            "问题层级": row.get("问题层级"),
            "模型": row.get("模型"),
            "联网": row.get("联网"),
            "轮次": row.get("轮次"),
            "推荐排名": row.get("推荐排名"),
            "推荐产品": row.get("推荐产品"),
            "名称类型": row.get("名称类型"),
            "推荐强度": row.get("推荐强度"),
            "推荐原因": row.get("推荐原因"),
        }
        for row in detail_rows[:1500]
    ]

    yang_agg: dict[tuple[Any, ...], dict[str, Any]] = {}
    yang_question_rows = []
    for row in yang_rows:
        source_level = row["_table_name"].replace("养胃舒-", "").split("汇总统计数据")[0]
        key = (row["_dataset_id"], source_level, row.get("AI模型"), row.get("目标品牌"))
        item = yang_agg.setdefault(
            key,
            {
                "dataset_id": key[0],
                "层级": key[1],
                "模型": key[2],
                "目标品牌": key[3],
                "样本数": 0,
                "_visibility": [],
                "_top3": [],
                "_first": [],
                "_rank": [],
            },
        )
        item["样本数"] += 1
        item["_visibility"].append(_num(row.get("能见度")))
        if row.get("前三率") is not None:
            item["_top3"].append(_num(row.get("前三率")))
        if row.get("首位率") is not None:
            item["_first"].append(_num(row.get("首位率")))
        rank = _num(row.get("位次"))
        if rank > 0:
            item["_rank"].append(rank)
        yang_question_rows.append({
            "dataset_id": row["_dataset_id"],
            "层级": source_level,
            "提问词": row.get("提问词"),
            "模型": row.get("AI模型"),
            "目标品牌": row.get("目标品牌"),
            "能见度": row.get("能见度"),
            "位次": row.get("位次"),
            "前三率": row.get("前三率"),
            "首位率": row.get("首位率"),
            "轮数": row.get("轮数"),
        })

    yang_summary = []
    for item in yang_agg.values():
        yang_summary.append({
            "dataset_id": item["dataset_id"],
            "层级": item["层级"],
            "模型": item["模型"],
            "目标品牌": item["目标品牌"],
            "样本数": item["样本数"],
            "平均能见度": _avg(item["_visibility"]),
            "平均前三率": _avg(item["_top3"]),
            "平均首位率": _avg(item["_first"]),
            "平均位次": _avg(item["_rank"]),
        })
    yang_summary.sort(key=lambda item: (_num(item.get("平均能见度")), _num(item.get("平均前三率"))), reverse=True)

    yang_mention_summary = [
        {
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
        }
        for item in yang_summary
    ]
    mention_summary.extend(yang_mention_summary)

    yang_rank_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in yang_summary:
        yang_rank_groups[(item["dataset_id"], item["层级"], item["模型"])].append(item)

    yang_rec_overview = []
    for (_dataset_id, _level, _model), items in yang_rank_groups.items():
        ranked = sorted(
            items,
            key=lambda item: (
                _num(item.get("平均能见度")),
                _num(item.get("平均前三率")),
                _num(item.get("平均首位率")),
            ),
            reverse=True,
        )
        for index, item in enumerate(ranked, start=1):
            mention_count = round(_num(item.get("平均能见度")) * _num(item.get("样本数")), 2)
            top3_count = min(round(_num(item.get("平均前三率")) * _num(item.get("样本数")), 2), mention_count)
            yang_rec_overview.append({
                "dataset_id": item["dataset_id"],
                "产品": "养胃舒专项",
                "问题层级": item["层级"],
                "模型": item["模型"],
                "联网": "汇总",
                "排名": index,
                "被推荐产品": item["目标品牌"],
                "名称类型": "目标品牌",
                "提及次数": mention_count,
                "提及率": item["平均能见度"],
                "强推荐次数": top3_count,
                "强推荐率": item["平均前三率"],
                "平均首位率": item["平均首位率"],
                "平均位次": item["平均位次"],
                "可选次数": item["样本数"],
            })
    rec_overview.extend(yang_rec_overview)
    rec_overview.sort(key=lambda item: (_num(item.get("提及次数")), _num(item.get("强推荐率"))), reverse=True)

    for (_dataset_id, level, model), items in yang_rank_groups.items():
        strong_count = sum(1 for item in items if _num(item.get("平均前三率")) > 0)
        type_summary.append({
            "dataset_id": _dataset_id,
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

    yang_category: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in yang_rec_overview:
        key = (item["dataset_id"], item["产品"], item["模型"], item["联网"], item["问题层级"])
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
        bucket["推荐次数"] += _num(item.get("提及次数"))
        bucket["强推荐数"] += _num(item.get("强推荐次数"))
    category_summary_rows.extend(yang_category.values())
    category_summary_rows.sort(key=lambda item: _num(item["推荐次数"]), reverse=True)

    yang_question_details = [
        {
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
        }
        for row in yang_question_rows
    ]
    question_details.extend(yang_question_details)

    return {
        "mention_summary": mention_summary,
        "rec_overview": rec_overview[:800],
        "type_summary": type_summary[:500],
        "category_summary": category_summary_rows[:500],
        "question_details": question_details,
        "yangweishu_brand_summary": yang_summary[:800],
        "yangweishu_question_metrics": yang_question_rows[:1500],
    }
