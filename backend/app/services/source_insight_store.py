"""信源分析查询层：任意数据集/产品组合、域名分类、缺口与回答下钻。"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import sqlite3
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml

from ..core.paths import CONFIG_DIR, GEO_SQLITE_PATH
from utils.sqlite_schema import stage_for_level


SOURCE_CONFIG_PATH = CONFIG_DIR / "source_domains.yaml"
PRESENTATION_SUBDOMAINS = {"www", "m", "mip", "wap", "amp", "mobile"}
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "spm", "from", "source", "src", "ref", "referrer", "share_token",
}


def _connect() -> sqlite3.Connection:
    if not GEO_SQLITE_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {GEO_SQLITE_PATH}")
    conn = sqlite3.connect(GEO_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_domain(domain: str | None, url: str | None = None) -> str:
    raw = (domain or "").strip().lower().rstrip(".")
    if not raw and url:
        try:
            raw = (urlsplit(url).hostname or "").lower().rstrip(".")
        except ValueError:
            raw = ""
    if raw.startswith("["):
        raw = raw[1:].split("]", 1)[0]
    elif raw.count(":") == 1:
        raw = raw.split(":", 1)[0]
    if ":" in raw:
        # IPv6 字面量没有可归并的域名标签，原样返回。
        return raw
    labels = [part for part in raw.split(".") if part]
    if len(labels) > 2 and labels[0] in PRESENTATION_SUBDOMAINS:
        labels = labels[1:]
    return ".".join(labels)


def normalize_url(url: str | None, domain: str | None = None) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text
    host = normalize_domain(domain, text)
    if not parsed.scheme or not host:
        return text
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS and not key.lower().startswith("utm_")
    ]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), host, path, urlencode(sorted(query)), ""))


@lru_cache(maxsize=4)
def _load_catalog(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    path = Path(path_text)
    if not path.exists():
        return {"categories": {"other": "其他"}, "domain_overrides": {}, "suffix_rules": [], "keyword_rules": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("categories", {"other": "其他"})
    data.setdefault("domain_overrides", {})
    data.setdefault("suffix_rules", [])
    data.setdefault("keyword_rules", [])
    return data


def source_catalog() -> dict[str, Any]:
    modified = SOURCE_CONFIG_PATH.stat().st_mtime_ns if SOURCE_CONFIG_PATH.exists() else 0
    return _load_catalog(str(SOURCE_CONFIG_PATH), modified)


def classify_domain(domain: str, catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    catalog = catalog or source_catalog()
    normalized = normalize_domain(domain)
    override = catalog.get("domain_overrides", {}).get(normalized)
    if override:
        item = dict(override)
    else:
        item = {}
        for rule in catalog.get("suffix_rules", []):
            suffix = normalize_domain(rule.get("suffix"))
            if suffix and (normalized == suffix or normalized.endswith(f".{suffix}")):
                item = dict(rule)
                break
        if not item:
            for rule in catalog.get("keyword_rules", []):
                if any(str(keyword).lower() in normalized for keyword in rule.get("keywords", [])):
                    item = dict(rule)
                    break
    category = item.get("category") or "other"
    categories = catalog.get("categories", {})
    ownership = item.get("ownership") or "third_party"
    return {
        "domain": normalized,
        "name": item.get("name") or normalized,
        "category": category,
        "category_label": categories.get(category, category),
        "authority": item.get("authority") or "U",
        "ownership": ownership,
        "product_codes": item.get("product_codes") or [],
        "is_official": ownership in {"owned", "competitor"} or category in {"brand_official", "competitor_official"},
        # 官方覆盖类指标只认我方官方渠道；竞品官网属于 is_official 但不参与我方覆盖。
        "is_own_official": ownership == "owned" or category == "brand_official",
        "is_authoritative": item.get("authority") == "A",
    }


def _scope_condition(
    allowed: list[str] | None, column: str, conds: list[str], params: list[Any]
) -> None:
    if allowed is None:
        return
    if not allowed:
        conds.append("1 = 0")
        return
    conds.append(f"{column} IN ({','.join('?' for _ in allowed)})")
    params.extend(allowed)


def _in_condition(
    values: Iterable[Any] | None, column: str, conds: list[str], params: list[Any]
) -> None:
    clean = [value for value in (values or []) if value not in (None, "")]
    if not clean:
        return
    conds.append(f"{column} IN ({','.join('?' for _ in clean)})")
    params.extend(clean)


def _answer_where(filters: dict[str, Any], allowed: list[str] | None) -> tuple[str, list[Any]]:
    conds: list[str] = []
    params: list[Any] = []
    _scope_condition(allowed, "a.dataset_id", conds, params)
    _in_condition(filters.get("dataset_ids"), "a.dataset_id", conds, params)
    _in_condition(filters.get("product_codes"), "a.product_code", conds, params)
    _in_condition(filters.get("models"), "a.model", conds, params)
    _in_condition(filters.get("search_modes"), "a.search_enabled", conds, params)
    scenarios = filters.get("scenarios") or []
    _in_condition(scenarios, "q.scenario", conds, params)
    return (" AND ".join(conds) if conds else "1 = 1"), params


def _answer_rows(
    conn: sqlite3.Connection, filters: dict[str, Any], allowed: list[str] | None
) -> list[dict[str, Any]]:
    where, params = _answer_where(filters, allowed)
    rows = [dict(row) for row in conn.execute(
        f"""
        SELECT a.dataset_id, a.answer_id, a.question_id, a.product_code,
               COALESCE(a.product_name, a.product_code) AS product_name,
               a.model, COALESCE(a.model_name, a.model) AS model_name,
               a.search_enabled, a.round,
               substr(COALESCE(a.answer_text, ''), 1, 220) AS answer_preview,
               q.question_text, q.level, q.source_level, COALESCE(q.scenario, '') AS scenario,
               EXISTS (
                 SELECT 1 FROM metric_evidence e
                 WHERE e.dataset_id = a.dataset_id
                   AND e.product_code = a.product_code
                   AND e.question_id = a.question_id
                   AND e.model IN (a.model, COALESCE(a.model_name, a.model))
                   AND (e.search_enabled IS NULL OR e.search_enabled = a.search_enabled)
                   AND (e.round IS NULL OR e.round = a.round)
                   AND (
                     (e.evidence_type = 'recommendation'
                      AND e.name_type IN ('999品牌', '目标品牌'))
                     OR (e.evidence_type = 'yang_metric'
                         AND e.name_type = '目标品牌'
                         AND (CAST(COALESCE(json_extract(e.payload_json, '$."位次"'), 0) AS REAL) > 0
                              OR CAST(COALESCE(json_extract(e.payload_json, '$."前三率"'), 0) AS REAL) > 0))
                   )
               ) AS brand_recommended
        FROM answers a
        JOIN questions q
          ON q.dataset_id = a.dataset_id AND q.question_id = a.question_id
        WHERE {where}
        ORDER BY a.dataset_id, a.product_code, a.question_id, a.model, a.round
        """,
        params,
    )]
    stages = set(filters.get("stages") or [])
    for row in rows:
        row["stage"] = stage_for_level(row.get("level"), row.get("source_level"))
    if stages:
        rows = [row for row in rows if row["stage"] in stages]
    return rows


def _source_rows(
    conn: sqlite3.Connection,
    answers: dict[tuple[str, str], dict[str, Any]],
    filters: dict[str, Any],
    allowed: list[str] | None,
) -> list[dict[str, Any]]:
    if not answers:
        return []
    where, params = _answer_where(filters, allowed)
    catalog = source_catalog()
    category_filter = set(filters.get("categories") or [])
    domain_filter = {normalize_domain(value) for value in (filters.get("domains") or [])}
    classified: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    cursor = conn.execute(
        f"""
        SELECT s.dataset_id, s.answer_id, s.source_index, s.title, s.url, s.domain
        FROM sources s
        JOIN answers a ON a.dataset_id = s.dataset_id AND a.answer_id = s.answer_id
        JOIN questions q ON q.dataset_id = a.dataset_id AND q.question_id = a.question_id
        WHERE {where}
        """,
        params,
    )
    for raw in cursor:
        row = dict(raw)
        key = (row["dataset_id"], row["answer_id"])
        if key not in answers:
            continue
        normalized = normalize_domain(row.get("domain"), row.get("url"))
        info = classified.get(normalized)
        if info is None:
            info = classified[normalized] = classify_domain(normalized, catalog)
        if category_filter and info["category"] not in category_filter:
            continue
        if domain_filter and normalized not in domain_filter:
            continue
        row.update(info)
        row["canonical_url"] = normalize_url(row.get("url"), normalized)
        rows.append(row)
    return rows


def list_options(allowed: list[str] | None = None) -> dict[str, Any]:
    catalog = source_catalog()
    with closing(_connect()) as conn:
        conds: list[str] = []
        params: list[Any] = []
        _scope_condition(allowed, "d.dataset_id", conds, params)
        where = f"WHERE {' AND '.join(conds)}" if conds else ""
        datasets = [dict(row) for row in conn.execute(
            f"""
            SELECT d.dataset_id, d.name, d.batch_date, d.owner_username
            FROM datasets d {where}
            ORDER BY COALESCE(d.batch_date, d.imported_at) DESC, d.dataset_id
            """,
            params,
        )]

        conds = []
        params = []
        _scope_condition(allowed, "a.dataset_id", conds, params)
        where = f"WHERE {' AND '.join(conds)}" if conds else ""
        products = [dict(row) for row in conn.execute(
            f"""
            SELECT a.product_code, MAX(COALESCE(a.product_name, a.product_code)) AS product_name,
                   COUNT(DISTINCT a.dataset_id) AS dataset_count
            FROM answers a {where}
            GROUP BY a.product_code ORDER BY product_name
            """,
            params,
        )]
        models = [dict(row) for row in conn.execute(
            f"""
            SELECT a.model, MAX(COALESCE(a.model_name, a.model)) AS model_name
            FROM answers a {where}
            GROUP BY a.model ORDER BY model_name
            """,
            params,
        )]
        raw_domains = list(conn.execute(
            f"""
            SELECT s.domain, MAX(s.url) AS sample_url, COUNT(*) AS refs
            FROM sources s JOIN answers a
              ON a.dataset_id = s.dataset_id AND a.answer_id = s.answer_id
            {where}
            GROUP BY s.domain ORDER BY refs DESC
            """,
            params,
        ))
    merged: dict[str, dict[str, Any]] = {}
    for row in raw_domains:
        domain = normalize_domain(row["domain"], row["sample_url"])
        if not domain:
            continue
        item = merged.setdefault(domain, {**classify_domain(domain, catalog), "refs": 0})
        item["refs"] += row["refs"] or 0
    domains = sorted(merged.values(), key=lambda item: (-item["refs"], item["domain"]))
    return {
        "datasets": datasets,
        "products": products,
        "models": models,
        "categories": [
            {"value": key, "label": label}
            for key, label in catalog.get("categories", {}).items()
        ],
        "domains": domains,
        "stages": [
            {"value": "symptom", "label": "病症阶段"},
            {"value": "category", "label": "品类阶段"},
            {"value": "brand", "label": "品牌阶段"},
        ],
    }


def analyze(filters: dict[str, Any], allowed: list[str] | None = None) -> dict[str, Any]:
    catalog = source_catalog()
    configured_official_products: set[str] = set()
    official_wildcard = False
    for config in catalog.get("domain_overrides", {}).values():
        if config.get("ownership") == "owned" or config.get("category") == "brand_official":
            codes = config.get("product_codes") or []
            if codes:
                configured_official_products.update(codes)
            else:
                # 未标注产品的官方域名视为对全部产品生效。
                official_wildcard = True
    with closing(_connect()) as conn:
        answer_list = _answer_rows(conn, filters, allowed)
        answers = {(row["dataset_id"], row["answer_id"]): row for row in answer_list}
        # 缺口判定必须看回答的全部信源，不受信源分类/域名筛选影响；
        # 分类/域名筛选在内存中收窄，避免同一 SQL 跑两遍。
        all_source_filters = {**filters, "categories": [], "domains": []}
        all_sources = _source_rows(conn, answers, all_source_filters, allowed)
    category_filter = set(filters.get("categories") or [])
    domain_filter = {normalize_domain(value) for value in (filters.get("domains") or [])}
    selected_sources = [
        source for source in all_sources
        if (not category_filter or source["category"] in category_filter)
        and (not domain_filter or source["domain"] in domain_filter)
    ]

    selected_by_answer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_by_answer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in selected_sources:
        selected_by_answer[(source["dataset_id"], source["answer_id"])].append(source)
    for source in all_sources:
        all_by_answer[(source["dataset_id"], source["answer_id"])].append(source)

    total_answers = len(answer_list)
    online_keys = {
        (row["dataset_id"], row["answer_id"])
        for row in answer_list if row["search_enabled"] == 1
    }
    cited_keys = set(selected_by_answer)
    cited_online = cited_keys & online_keys
    official_keys: set[tuple[str, str]] = set()
    authority_keys: set[tuple[str, str]] = set()
    for key, sources in selected_by_answer.items():
        answer_product = answers[key]["product_code"]
        if any(
            source["is_own_official"]
            and (not source["product_codes"] or answer_product in source["product_codes"])
            for source in sources
        ):
            official_keys.add(key)
        if any(source["is_authoritative"] for source in sources):
            authority_keys.add(key)

    denominator_keys = online_keys or set(answers)
    denominator = len(denominator_keys)
    official_denominator_keys = {
        key for key in denominator_keys
        if official_wildcard or answers[key]["product_code"] in configured_official_products
    }
    summary = {
        "total_answers": total_answers,
        "online_answers": len(online_keys),
        "cited_answers": len(cited_keys),
        "cited_online_answers": len(cited_online),
        "coverage_rate": round(len(cited_keys & denominator_keys) / denominator, 4) if denominator else None,
        "source_refs": len(selected_sources),
        "avg_sources_per_cited_answer": round(len(selected_sources) / len(cited_keys), 2) if cited_keys else None,
        "distinct_domains": len({source["domain"] for source in selected_sources if source["domain"]}),
        "distinct_urls": len({source["canonical_url"] for source in selected_sources if source["canonical_url"]}),
        "official_eligible_answers": len(official_denominator_keys),
        "official_coverage_rate": (
            round(len(official_keys & official_denominator_keys) / len(official_denominator_keys), 4)
            if official_denominator_keys else None
        ),
        "authority_coverage_rate": round(len(authority_keys & denominator_keys) / denominator, 4) if denominator else None,
        "online_without_sources": len(online_keys - set(all_by_answer)),
    }

    category_groups: dict[str, dict[str, Any]] = {}
    domain_groups: dict[str, dict[str, Any]] = {}
    for source in selected_sources:
        key = (source["dataset_id"], source["answer_id"])
        category = source["category"]
        cat = category_groups.setdefault(category, {
            "category": category,
            "label": source["category_label"],
            "refs": 0, "answers": set(), "domains": set(), "urls": set(),
        })
        cat["refs"] += 1
        cat["answers"].add(key)
        cat["domains"].add(source["domain"])
        if source["canonical_url"]:
            cat["urls"].add(source["canonical_url"])

        dom = domain_groups.setdefault(source["domain"], {
            "domain": source["domain"], "name": source["name"],
            "category": category, "category_label": source["category_label"],
            "authority": source["authority"], "ownership": source["ownership"],
            "refs": 0, "answers": set(), "urls": set(), "products": set(), "models": set(),
        })
        dom["refs"] += 1
        dom["answers"].add(key)
        if source["canonical_url"]:
            dom["urls"].add(source["canonical_url"])
        dom["products"].add(answers[key]["product_name"])
        dom["models"].add(answers[key]["model"])

    categories = []
    for item in category_groups.values():
        answer_keys = item.pop("answers")
        categories.append({
            **item,
            "answer_count": len(answer_keys),
            "domain_count": len(item.pop("domains")),
            "url_count": len(item.pop("urls")),
            "coverage_rate": round(len(answer_keys & denominator_keys) / denominator, 4) if denominator else None,
        })
    categories.sort(key=lambda item: (-item["answer_count"], item["label"]))

    domains = []
    for item in domain_groups.values():
        answer_keys = item.pop("answers")
        domains.append({
            **item,
            "answer_count": len(answer_keys),
            "url_count": len(item.pop("urls")),
            "products": sorted(item.pop("products")),
            "models": sorted(item.pop("models")),
            "coverage_rate": round(len(answer_keys & denominator_keys) / denominator, 4) if denominator else None,
        })
    domains.sort(key=lambda item: (-item["answer_count"], -item["refs"], item["domain"]))

    product_groups: dict[str, dict[str, Any]] = {}
    for row in answer_list:
        code = row["product_code"]
        item = product_groups.setdefault(code, {
            "product_code": code, "product_name": row["product_name"],
            "answers": set(), "online": set(), "cited": set(),
            "official": set(), "authority": set(), "domains": set(),
        })
        key = (row["dataset_id"], row["answer_id"])
        item["answers"].add(key)
        if row["search_enabled"] == 1:
            item["online"].add(key)
        if key in cited_keys:
            item["cited"].add(key)
            item["domains"].update(source["domain"] for source in selected_by_answer[key])
        if key in official_keys:
            item["official"].add(key)
        if key in authority_keys:
            item["authority"].add(key)
    products = []
    for item in product_groups.values():
        product_denominator = item["online"] or item["answers"]
        count = len(product_denominator)
        products.append({
            "product_code": item["product_code"],
            "product_name": item["product_name"],
            "answers": len(item["answers"]),
            "online_answers": len(item["online"]),
            "cited_answers": len(item["cited"] & product_denominator),
            "coverage_rate": round(len(item["cited"] & product_denominator) / count, 4) if count else None,
            "official_coverage_rate": (
                round(len(item["official"] & product_denominator) / count, 4)
                if count and (official_wildcard or item["product_code"] in configured_official_products) else None
            ),
            "authority_coverage_rate": round(len(item["authority"] & product_denominator) / count, 4) if count else None,
            "domain_count": len(item["domains"]),
        })
    products.sort(key=lambda item: item["product_name"])

    gaps = []
    for key, row in answers.items():
        if row["search_enabled"] != 1:
            continue
        sources = all_by_answer.get(key, [])
        reasons = []
        if not sources:
            reasons.append(("online_without_source", "联网回答无信源", "high"))
        if row["brand_recommended"] and not sources:
            reasons.append(("recommendation_without_source", "推荐产品但无信源", "high"))
        if row["brand_recommended"] and sources and not any(source["is_authoritative"] for source in sources):
            reasons.append(("recommendation_without_authority", "推荐产品但缺少权威信源", "medium"))
        if (
            row["brand_recommended"]
            and (official_wildcard or row["product_code"] in configured_official_products)
            and sources
            and not any(
                source["is_own_official"]
                and (not source["product_codes"] or row["product_code"] in source["product_codes"])
                for source in sources
            )
        ):
            reasons.append(("recommendation_without_official", "推荐产品但缺少官方信源", "medium"))
        for gap_type, label, severity in reasons:
            gaps.append({
                "gap_type": gap_type, "gap_label": label, "severity": severity,
                "dataset_id": row["dataset_id"], "answer_id": row["answer_id"],
                "question_id": row["question_id"], "question_text": row["question_text"],
                "product_code": row["product_code"], "product_name": row["product_name"],
                "model": row["model"], "model_name": row["model_name"],
                "search_enabled": row["search_enabled"], "round": row["round"],
                "stage": row["stage"], "scenario": row["scenario"],
                "source_count": len(sources),
            })
    severity_order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda item: (severity_order.get(item["severity"], 9), item["product_name"], item["model"]))

    return {
        "summary": summary,
        "categories": categories,
        "domains": domains[:300],
        "products": products,
        "gaps": gaps[:500],
        "gap_total": len(gaps),
        "selected_product_codes": sorted(set(filters.get("product_codes") or [])),
    }


def source_answers(
    filters: dict[str, Any], domain: str | None = None, limit: int = 100,
    allowed: list[str] | None = None,
) -> list[dict[str, Any]]:
    if domain:
        filters = {**filters, "domains": [domain]}
    with closing(_connect()) as conn:
        answer_list = _answer_rows(conn, filters, allowed)
        answers = {(row["dataset_id"], row["answer_id"]): row for row in answer_list}
        sources = _source_rows(conn, answers, filters, allowed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        grouped[(source["dataset_id"], source["answer_id"])].append(source)
    result = []
    cap = max(1, min(limit, 300))
    # 按 _answer_rows 的 ORDER BY 顺序输出，截断结果对同一请求保持稳定。
    for answer in answer_list:
        key = (answer["dataset_id"], answer["answer_id"])
        source_list = grouped.get(key)
        if not source_list:
            continue
        row = dict(answer)
        row["sources"] = [
            {
                "title": source.get("title") or "",
                "url": source.get("url") or "",
                "domain": source["domain"],
                "category": source["category"],
                "category_label": source["category_label"],
            }
            for source in source_list
        ]
        result.append(row)
        if len(result) >= cap:
            break
    return result
