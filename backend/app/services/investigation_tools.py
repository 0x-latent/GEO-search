"""Deterministic and controlled tools available to the GEO investigation Agent."""
from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from ..core.paths import GEO_SQLITE_PATH
from . import outbound_article_store, source_insight_store, user_config_store
from utils.sqlite_schema import stage_for_level


MAX_FETCH_BYTES = 2 * 1024 * 1024
MAX_EXTRACT_CHARS = 200_000
ALLOWED_CONTENT_TYPES = {
    "text/html", "text/plain", "application/json", "application/xhtml+xml",
    "application/pdf",
}


def _connect() -> sqlite3.Connection:
    if not GEO_SQLITE_PATH.exists():
        raise FileNotFoundError(f"GEO SQLite 不存在: {GEO_SQLITE_PATH}")
    conn = sqlite3.connect(GEO_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _metric_row(
    conn: sqlite3.Connection, dataset_id: str, item: dict[str, Any]
) -> dict[str, Any] | None:
    conditions = ["dataset_id=?", "product_code=?"]
    params: list[Any] = [dataset_id, item["product_code"]]
    if item.get("stage_filter"):
        conditions.append("stage=?")
        params.append(item["stage_filter"])
    if item.get("model_filter"):
        conditions.append("model=?")
        params.append(item["model_filter"])
    if item.get("search_enabled_filter") is not None:
        conditions.append("search_enabled=?")
        params.append(str(item["search_enabled_filter"]))
    rows = conn.execute(
        f"""SELECT * FROM metrics_summary
            WHERE {' AND '.join(conditions)}
            ORDER BY total_answers DESC""",
        params,
    ).fetchall()
    metric = item["metric"]
    for row in rows:
        value = dict(row)
        if metric in value and value[metric] is not None:
            return value
    return None


def derived_rates(
    dataset_id: str,
    product_code: str,
    stage: str | None = None,
    model: str | None = None,
    search_enabled: int | None = None,
) -> dict[str, Any]:
    """Compute source/article rates that are not materialized in metrics_summary."""
    filters = {
        "dataset_ids": [dataset_id],
        "product_codes": [product_code],
        "stages": [stage] if stage else [],
        "models": [model] if model else [],
        "search_modes": [search_enabled] if search_enabled is not None else [],
    }
    source_summary = source_insight_store.analyze(filters, None)["summary"]
    item = {
        "product_code": product_code,
        "stage_filter": stage,
        "model_filter": model,
        "search_enabled_filter": search_enabled,
    }
    with closing(_connect()) as conn:
        answer_rows = _answer_rows(conn, dataset_id, item)
        answer_ids = {row["answer_id"] for row in answer_rows}
        dataset_owner = conn.execute(
            "SELECT owner_username FROM datasets WHERE dataset_id=?",
            (dataset_id,),
        ).fetchone()
        owner = dataset_owner[0] if dataset_owner else None
        publication_rows = conn.execute(
            """SELECT ap.url_match_key
               FROM article_publications ap
               JOIN outbound_articles oa ON oa.article_id=ap.article_id
               WHERE oa.product_code=?
                 AND (? IS NULL OR oa.owner_username=?)""",
            (product_code, owner, owner),
        ).fetchall()
        publication_keys = {row[0] for row in publication_rows if row[0]}
        cited_ids: set[str] = set()
        if publication_keys and answer_ids:
            for row in conn.execute(
                """SELECT answer_id,url FROM sources
                   WHERE dataset_id=? AND COALESCE(url,'')<>''""",
                (dataset_id,),
            ):
                if row["answer_id"] not in answer_ids:
                    continue
                try:
                    _, match_key = outbound_article_store.url_match_key(row["url"])
                except ValueError:
                    continue
                if match_key in publication_keys:
                    cited_ids.add(row["answer_id"])
    denominator = len(answer_ids)
    online_or_all = int(source_summary.get("online_answers") or 0) or denominator
    official_n = int(source_summary.get("official_eligible_answers") or 0)
    return {
        "official_coverage_rate": source_summary.get("official_coverage_rate"),
        "authority_coverage_rate": source_summary.get("authority_coverage_rate"),
        "article_citation_rate": (
            round(len(cited_ids) / denominator, 4) if denominator else None
        ),
        "sample_sizes": {
            "official_coverage_rate": official_n,
            "authority_coverage_rate": online_or_all,
            "article_citation_rate": denominator,
        },
        "cited_answers": len(cited_ids),
        "total_answers": denominator,
    }


def compare_metrics(item: dict[str, Any], arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    metric = item["metric"]
    if metric in {
        "official_coverage_rate", "authority_coverage_rate", "article_citation_rate"
    }:
        current_rates = derived_rates(
            item["current_dataset_id"], item["product_code"],
            item.get("stage_filter"), item.get("model_filter"),
            item.get("search_enabled_filter"),
        )
        baseline_rates = (
            derived_rates(
                item["baseline_dataset_id"], item["product_code"],
                item.get("stage_filter"), item.get("model_filter"),
                item.get("search_enabled_filter"),
            )
            if item.get("baseline_dataset_id") else {}
        )
        current_value = current_rates.get(metric)
        baseline_value = baseline_rates.get(metric)
        delta = (
            float(current_value) - float(baseline_value)
            if current_value is not None and baseline_value is not None else None
        )
        return {
            "comparable": baseline_value is not None,
            "metric": metric,
            "current": {
                "value": current_value,
                "total_answers": current_rates.get("sample_sizes", {}).get(metric),
                "stage": item.get("stage_filter"),
                "model": item.get("model_filter"),
                "search_enabled": item.get("search_enabled_filter"),
            },
            "baseline": {
                "value": baseline_value,
                "total_answers": baseline_rates.get("sample_sizes", {}).get(metric),
                "stage": item.get("stage_filter"),
                "model": item.get("model_filter"),
                "search_enabled": item.get("search_enabled_filter"),
            },
            "delta": delta,
            "expected_value": item.get("expected_value"),
        }
    with closing(_connect()) as conn:
        current = _metric_row(conn, item["current_dataset_id"], item)
        baseline = (
            _metric_row(conn, item["baseline_dataset_id"], item)
            if item.get("baseline_dataset_id") else None
        )
    if not current:
        return {"comparable": False, "reason": "current_metric_missing", "metric": metric}
    current_value = current.get(metric)
    baseline_value = baseline.get(metric) if baseline else None
    delta = (
        float(current_value) - float(baseline_value)
        if current_value is not None and baseline_value is not None else None
    )
    return {
        "comparable": baseline_value is not None,
        "metric": metric,
        "current": {
            "value": current_value,
            "total_answers": current.get("total_answers"),
            "stage": current.get("stage"),
            "question_level": current.get("question_level"),
            "model": current.get("model"),
            "search_enabled": current.get("search_enabled"),
        },
        "baseline": {
            "value": baseline_value,
            "total_answers": (baseline or {}).get("total_answers"),
            "stage": (baseline or {}).get("stage"),
            "question_level": (baseline or {}).get("question_level"),
            "model": (baseline or {}).get("model"),
            "search_enabled": (baseline or {}).get("search_enabled"),
        },
        "delta": delta,
        "expected_value": item.get("expected_value"),
    }


def _dataset_sample(
    conn: sqlite3.Connection, dataset_id: str, item: dict[str, Any]
) -> dict[str, Any]:
    conditions = ["q.dataset_id=?", "q.product_code=?"]
    params: list[Any] = [dataset_id, item["product_code"]]
    if item.get("model_filter"):
        conditions.append("a.model=?")
        params.append(item["model_filter"])
    if item.get("search_enabled_filter") is not None:
        conditions.append("a.search_enabled=?")
        params.append(item["search_enabled_filter"])
    rows = conn.execute(
        f"""SELECT q.question_id,q.question_text,q.level,q.source_level,q.scenario,
                   a.answer_id,a.model,a.model_id,a.search_enabled,a.search_triggered,
                   a.round,a.route,a.client_mode,a.timestamp,a.source_count
            FROM questions q
            LEFT JOIN answers a
              ON a.dataset_id=q.dataset_id AND a.question_id=q.question_id
            WHERE {' AND '.join(conditions)}""",
        params,
    ).fetchall()
    if item.get("stage_filter"):
        rows = [
            row for row in rows
            if stage_for_level(row["level"], row["source_level"])
            == item["stage_filter"]
        ]
    question_ids = {row["question_id"] for row in rows}
    answers = [row for row in rows if row["answer_id"]]
    envs = Counter(
        (
            row["model"], row["model_id"] or "unknown", row["route"] or "unknown",
            row["client_mode"] or "unknown", row["search_enabled"],
        )
        for row in answers
    )
    return {
        "question_ids": question_ids,
        "question_count": len(question_ids),
        "answer_count": len(answers),
        "models": sorted({row["model"] for row in answers if row["model"]}),
        "rounds": sorted({row["round"] for row in answers if row["round"] is not None}),
        "search_modes": sorted({row["search_enabled"] for row in answers if row["search_enabled"] is not None}),
        "collection_environments": [
            {
                "model": key[0], "model_id": key[1], "route": key[2],
                "client_mode": key[3], "search_enabled": key[4], "answers": count,
            }
            for key, count in envs.items()
        ],
        "answers_without_sources": sum(
            1 for row in answers if row["search_enabled"] and not row["source_count"]
        ),
    }


def validate_sample(item: dict[str, Any], arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    with closing(_connect()) as conn:
        current = _dataset_sample(conn, item["current_dataset_id"], item)
        baseline = (
            _dataset_sample(conn, item["baseline_dataset_id"], item)
            if item.get("baseline_dataset_id") else None
        )
        current_info = conn.execute(
            """SELECT dataset_id,name,batch_date,question_set_id,metadata_json
               FROM datasets WHERE dataset_id=?""",
            (item["current_dataset_id"],),
        ).fetchone()
        baseline_info = (
            conn.execute(
                """SELECT dataset_id,name,batch_date,question_set_id,metadata_json
                   FROM datasets WHERE dataset_id=?""",
                (item["baseline_dataset_id"],),
            ).fetchone()
            if item.get("baseline_dataset_id") else None
        )
    overlap = (
        len(current["question_ids"] & baseline["question_ids"])
        if baseline else 0
    )
    union = (
        len(current["question_ids"] | baseline["question_ids"])
        if baseline else len(current["question_ids"])
    )
    current.pop("question_ids", None)
    if baseline:
        baseline.pop("question_ids", None)
    comparable = bool(
        baseline
        and current["question_count"] >= 5 and baseline["question_count"] >= 5
        and current["answer_count"] >= 10 and baseline["answer_count"] >= 10
        and overlap / max(1, union) >= .8
    )
    warnings = []
    if not baseline:
        warnings.append("缺少基线数据集")
    if overlap / max(1, union) < .8:
        warnings.append("前后问题集合重合度低于80%")
    if current["answer_count"] < 10 or (baseline and baseline["answer_count"] < 10):
        warnings.append("样本量低于自动归因门槛")
    current_env = {
        (row["model"], row["model_id"], row["route"], row["client_mode"])
        for row in current["collection_environments"]
    }
    baseline_env = {
        (row["model"], row["model_id"], row["route"], row["client_mode"])
        for row in (baseline or {}).get("collection_environments", [])
    }
    if baseline_env and current_env != baseline_env:
        warnings.append("采集模型ID、链路或客户端环境发生变化")
    return {
        "comparable": comparable,
        "question_overlap": overlap,
        "question_union": union,
        "question_overlap_rate": round(overlap / max(1, union), 4),
        "current": current,
        "baseline": baseline,
        "current_dataset": dict(current_info) if current_info else None,
        "baseline_dataset": dict(baseline_info) if baseline_info else None,
        "warnings": warnings,
    }


def _answer_rows(
    conn: sqlite3.Connection, dataset_id: str, item: dict[str, Any]
) -> list[dict[str, Any]]:
    conditions = ["a.dataset_id=?", "a.product_code=?"]
    params: list[Any] = [dataset_id, item["product_code"]]
    if item.get("model_filter"):
        conditions.append("a.model=?")
        params.append(item["model_filter"])
    if item.get("search_enabled_filter") is not None:
        conditions.append("a.search_enabled=?")
        params.append(item["search_enabled_filter"])
    if item.get("stage_filter"):
        stage_levels = {
            "symptom": ("symptom", "症状", "病症", "Q3", "Q4"),
            "category": ("category", "品类", "Q5"),
            "brand": ("brand", "品牌", "Q1", "Q2"),
        }.get(item["stage_filter"], (item["stage_filter"],))
        conditions.append(
            f"(q.level IN ({','.join('?' for _ in stage_levels)}) "
            f"OR q.source_level IN ({','.join('?' for _ in stage_levels)}))"
        )
        params.extend(stage_levels)
        params.extend(stage_levels)
    rows = conn.execute(
        f"""SELECT a.dataset_id,a.answer_id,a.question_id,q.question_text,q.scenario,
                   a.model,a.model_id,a.search_enabled,a.search_triggered,a.round,
                   a.timestamp,a.answer_text,a.source_count,a.route,a.client_mode
            FROM answers a JOIN questions q
              ON q.dataset_id=a.dataset_id AND q.question_id=a.question_id
            WHERE {' AND '.join(conditions)}
            ORDER BY a.question_id,a.model,a.search_enabled,a.round""",
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]{2,}", (text or "").lower()))


def diff_answers(item: dict[str, Any], arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    with closing(_connect()) as conn:
        current = _answer_rows(conn, item["current_dataset_id"], item)
        baseline = (
            _answer_rows(conn, item["baseline_dataset_id"], item)
            if item.get("baseline_dataset_id") else []
        )
    key = lambda row: (
        row["question_id"], row["model"], row["search_enabled"], row["round"]
    )
    current_map = {key(row): row for row in current}
    baseline_map = {key(row): row for row in baseline}
    pairs = []
    for answer_key in sorted(current_map.keys() & baseline_map.keys()):
        now, old = current_map[answer_key], baseline_map[answer_key]
        left, right = _token_set(old["answer_text"]), _token_set(now["answer_text"])
        similarity = len(left & right) / max(1, len(left | right))
        if similarity > .82 and old["source_count"] == now["source_count"]:
            continue
        pairs.append({
            "question_id": now["question_id"],
            "question": now["question_text"],
            "model": now["model"],
            "search_enabled": now["search_enabled"],
            "round": now["round"],
            "semantic_token_similarity": round(similarity, 4),
            "source_count_before": old["source_count"],
            "source_count_after": now["source_count"],
            "baseline_excerpt": old["answer_text"][:1200],
            "current_excerpt": now["answer_text"][:1200],
        })
    pairs.sort(key=lambda row: row["semantic_token_similarity"])
    return {
        "matched_answers": len(current_map.keys() & baseline_map.keys()),
        "current_only": len(current_map.keys() - baseline_map.keys()),
        "baseline_only": len(baseline_map.keys() - current_map.keys()),
        "changed_answers": len(pairs),
        "most_changed": pairs[:25],
        "average_similarity": round(
            sum(row["semantic_token_similarity"] for row in pairs) / len(pairs), 4
        ) if pairs else None,
    }


def _source_rows(
    conn: sqlite3.Connection, dataset_id: str, item: dict[str, Any]
) -> list[dict[str, Any]]:
    conditions = ["a.dataset_id=?", "a.product_code=?"]
    params: list[Any] = [dataset_id, item["product_code"]]
    if item.get("model_filter"):
        conditions.append("a.model=?")
        params.append(item["model_filter"])
    if item.get("search_enabled_filter") is not None:
        conditions.append("a.search_enabled=?")
        params.append(item["search_enabled_filter"])
    rows = conn.execute(
        f"""SELECT s.url,s.domain,s.title,a.question_id,a.model,a.search_enabled
            FROM answers a JOIN sources s
              ON s.dataset_id=a.dataset_id AND s.answer_id=a.answer_id
            WHERE {' AND '.join(conditions)}""",
        params,
    ).fetchall()
    result = []
    for row in rows:
        value = dict(row)
        value["canonical_url"] = source_insight_store.normalize_url(
            value.get("url"), value.get("domain")
        )
        result.append(value)
    return result


def diff_sources(item: dict[str, Any], arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    with closing(_connect()) as conn:
        current = _source_rows(conn, item["current_dataset_id"], item)
        baseline = (
            _source_rows(conn, item["baseline_dataset_id"], item)
            if item.get("baseline_dataset_id") else []
        )
    current_urls = {row["canonical_url"] for row in current if row["canonical_url"]}
    baseline_urls = {row["canonical_url"] for row in baseline if row["canonical_url"]}
    current_domains = Counter(row["domain"] for row in current if row["domain"])
    baseline_domains = Counter(row["domain"] for row in baseline if row["domain"])
    gained_domains = sorted(
        (
            {"domain": domain, "refs": count, "before_refs": baseline_domains.get(domain, 0)}
            for domain, count in current_domains.items()
            if count > baseline_domains.get(domain, 0)
        ),
        key=lambda row: row["refs"] - row["before_refs"], reverse=True,
    )
    lost_domains = sorted(
        (
            {"domain": domain, "refs": count, "after_refs": current_domains.get(domain, 0)}
            for domain, count in baseline_domains.items()
            if count > current_domains.get(domain, 0)
        ),
        key=lambda row: row["refs"] - row["after_refs"], reverse=True,
    )
    return {
        "baseline_refs": len(baseline),
        "current_refs": len(current),
        "baseline_domains": len(baseline_domains),
        "current_domains": len(current_domains),
        "gained_urls": sorted(current_urls - baseline_urls)[:100],
        "lost_urls": sorted(baseline_urls - current_urls)[:100],
        "gained_domains": gained_domains[:50],
        "lost_domains": lost_domains[:50],
        "url_overlap_rate": round(
            len(current_urls & baseline_urls) / max(1, len(current_urls | baseline_urls)), 4
        ),
    }


def trace_articles(item: dict[str, Any], arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    with closing(_connect()) as conn:
        dataset_ids = [
            value for value in (
                item.get("baseline_dataset_id"), item["current_dataset_id"]
            ) if value
        ]
        owner_row = conn.execute(
            "SELECT owner_username FROM datasets WHERE dataset_id=?",
            (item["current_dataset_id"],),
        ).fetchone()
        owner = owner_row[0] if owner_row else None
        articles = [
            dict(row) for row in conn.execute(
                """SELECT oa.article_id,oa.title,oa.campaign,oa.created_at,
                          ap.publication_id,ap.platform,ap.url,ap.url_match_key,
                          ap.published_at
                   FROM outbound_articles oa
                   JOIN article_publications ap ON ap.article_id=oa.article_id
                   WHERE oa.product_code=?
                     AND (? IS NULL OR oa.owner_username=?)""",
                (item["product_code"], owner, owner),
            ).fetchall()
        ]
        if not articles:
            return {"articles": [], "summary": {"total_articles": 0}}
        eligible_answers: dict[str, set[str]] = {}
        for dataset_id in dataset_ids:
            eligible_answers[dataset_id] = {
                row["answer_id"] for row in _answer_rows(conn, dataset_id, item)
            }
        placeholders = ",".join("?" for _ in dataset_ids)
        sources = conn.execute(
            f"""SELECT dataset_id,answer_id,url FROM sources
                WHERE dataset_id IN ({placeholders})
                  AND COALESCE(url,'')<>''""",
            dataset_ids,
        ).fetchall()
    by_key: dict[str, list[str]] = {}
    for article in articles:
        if article.get("url_match_key"):
            by_key.setdefault(article["url_match_key"], []).append(
                article["publication_id"]
            )
    by_pub: dict[str, dict[str, set[str]]] = {}
    for source in sources:
        if source["answer_id"] not in eligible_answers.get(source["dataset_id"], set()):
            continue
        try:
            _, match_key = outbound_article_store.url_match_key(source["url"])
        except ValueError:
            continue
        for publication_id in by_key.get(match_key, []):
            by_pub.setdefault(publication_id, {}).setdefault(
                source["dataset_id"], set()
            ).add(source["answer_id"])
    result = []
    for article in articles:
        counts = by_pub.get(article["publication_id"], {})
        before = len(counts.get(item.get("baseline_dataset_id"), set()))
        after = len(counts.get(item["current_dataset_id"], set()))
        clean_article = {key: value for key, value in article.items() if key != "url_match_key"}
        result.append({
            **clean_article,
            "baseline_citation_answers": before,
            "current_citation_answers": after,
            "citation_delta": after - before,
        })
    return {
        "summary": {
            "total_articles": len({row["article_id"] for row in result}),
            "publications": len(result),
            "cited_before": sum(1 for row in result if row["baseline_citation_answers"]),
            "cited_after": sum(1 for row in result if row["current_citation_answers"]),
        },
        "articles": sorted(result, key=lambda row: row["citation_delta"])[:100],
    }


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_flatten_text(item) for item in value)
    return str(value) if value is not None else ""


def check_knowledge_consistency(
    item: dict[str, Any], arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    kb = user_config_store.load_global_kb()
    with closing(_connect()) as conn:
        product = conn.execute(
            "SELECT product_name,aliases_json FROM products WHERE product_code=?",
            (item["product_code"],),
        ).fetchone()
        answers = _answer_rows(conn, item["current_dataset_id"], item)[:50]
    candidates = [item["product_code"]]
    if product:
        candidates.append(product["product_name"])
        try:
            candidates.extend(json.loads(product["aliases_json"] or "[]"))
        except json.JSONDecodeError:
            pass
    selected_key = next((key for key in candidates if key in kb), None)
    if not selected_key:
        return {
            "knowledge_base_found": False,
            "product_candidates": candidates,
            "warning": "未找到产品知识库，无法核验事实一致性",
        }
    kb_text = _flatten_text(kb[selected_key])
    kb_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", kb_text))
    unsupported = Counter()
    samples = []
    for answer in answers:
        answer_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", answer["answer_text"]))
        missing = sorted(answer_numbers - kb_numbers)
        for number in missing:
            unsupported[number] += 1
        if missing:
            samples.append({
                "question_id": answer["question_id"],
                "question": answer["question_text"],
                "numbers_not_found_in_kb": missing[:20],
                "answer_excerpt": answer["answer_text"][:800],
            })
    return {
        "knowledge_base_found": True,
        "knowledge_key": selected_key,
        "knowledge_chars": len(kb_text),
        "answers_checked": len(answers),
        "numeric_claims_not_in_kb": [
            {"value": value, "answers": count}
            for value, count in unsupported.most_common(30)
        ],
        "samples": samples[:20],
        "caveat": "未在知识库出现的数字只能标记为待核验，不能自动判错",
    }


def _validated_host(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("仅允许带有效主机名的 HTTP/HTTPS URL")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError("禁止访问本机地址")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"域名解析失败: {host}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if (
            address.is_private or address.is_loopback or address.is_link_local
            or address.is_multicast or address.is_reserved or address.is_unspecified
        ):
            raise ValueError("禁止访问私网、本机或保留地址")
    return parsed.scheme, host


def _html_text(content: bytes, encoding: str | None) -> str:
    text = content.decode(encoding or "utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _pdf_text(content: bytes) -> str:
    try:
        import fitz
        document = fitz.open(stream=content, filetype="pdf")
        return "\n".join(page.get_text() for page in document)[:MAX_EXTRACT_CHARS]
    except Exception as exc:  # noqa: BLE001 - a failed PDF parse is an audit result
        raise ValueError(f"PDF正文提取失败: {exc}") from exc


def audit_url(item: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url") or "").strip()
    if not url:
        raise ValueError("audit_url 需要 url")
    query = str(arguments.get("query") or "")
    timeout = min(max(float(arguments.get("timeout") or 10), 1), 20)
    current_url = url
    redirects = []
    with httpx.Client(
        timeout=httpx.Timeout(timeout),
        follow_redirects=False,
        headers={"User-Agent": "GeoInvestigationBot/1.0 (+read-only audit)"},
    ) as client:
        for _ in range(5):
            _validated_host(current_url)
            with client.stream("GET", current_url) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("重定向响应缺少 Location")
                    next_url = urljoin(current_url, location)
                    _validated_host(next_url)
                    redirects.append({"from": current_url, "to": next_url})
                    current_url = next_url
                    continue
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if content_type not in ALLOWED_CONTENT_TYPES:
                    raise ValueError(f"不支持的内容类型: {content_type or 'unknown'}")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_FETCH_BYTES:
                        raise ValueError("网页响应超过 2MB 限制")
                raw = bytes(body)
                if content_type == "application/pdf":
                    text = _pdf_text(raw)
                elif content_type in {"text/html", "application/xhtml+xml"}:
                    encoding = response.encoding if response.encoding != "utf-8" else None
                    text = _html_text(raw, encoding)
                else:
                    text = raw.decode(response.encoding or "utf-8", errors="replace")
                tokens = _token_set(query)
                content_tokens = _token_set(text)
                matched = sorted(tokens & content_tokens)
                return {
                    "requested_url": url,
                    "final_url": current_url,
                    "redirects": redirects,
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "content_bytes": len(raw),
                    "content_chars": len(text),
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                    "last_modified": response.headers.get("last-modified"),
                    "etag": response.headers.get("etag"),
                    "query_token_coverage": round(len(matched) / max(1, len(tokens)), 4),
                    "matched_query_tokens": matched[:50],
                    "text_excerpt": text[:5000],
                }
    raise ValueError("网页重定向超过5次")


TOOL_REGISTRY = {
    "compare_metrics": compare_metrics,
    "validate_sample": validate_sample,
    "diff_answers": diff_answers,
    "diff_sources": diff_sources,
    "trace_articles": trace_articles,
    "audit_url": audit_url,
    "check_knowledge_consistency": check_knowledge_consistency,
}
