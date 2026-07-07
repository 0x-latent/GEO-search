"""
产品主数据：以 config/brands.yaml 的 brand_999 段为唯一维护入口，
同步到 GEO SQLite 的 products 表（品牌总览/上传下拉/趋势锚点都读这张表）。

历史库里 answers/questions 用的是导入时的短名 code（ganmaolin/piyanping/...），
同步时优先复用这些既有 code，避免趋势维度断裂；brands.yaml 里新出现的品牌
生成稳定哈希 code。brands.yaml 匹配不上的存量产品保留但 is_active=0。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from ..core.paths import GEO_SQLITE_PATH
from utils.sqlite_schema import ensure_schema, ensure_schema_at

from . import yaml_store

# 现库既有短名 → brands.yaml 品牌键（03 采集时代的命名约定，一次性对照）
LEGACY_NAME_MAP = {
    "感冒灵": "999感冒灵",
    "皮炎平": "999皮炎平",
    "胃泰": "三九胃泰",
    "抗病毒": "999抗病毒口服液",
    "小感": "999小儿感冒药",
    "强枇": "强力枇杷露",
    "澳诺": "澳诺葡萄糖酸锌钙",
    "易善复": "易善复",
}


def _stable_code(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def ensure_geo_schema() -> None:
    """backend 启动时建齐/迁移 GEO 库 schema（库不存在则创建空库）。"""
    ensure_schema_at(GEO_SQLITE_PATH)


def _existing_products(conn: sqlite3.Connection) -> tuple[dict[str, str], set[str]]:
    rows = conn.execute("SELECT product_code, product_name FROM products").fetchall()
    existing = {code: (name or "") for code, name in rows}
    used = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT product_code FROM answers WHERE product_code <> ''"
        )
    }
    return existing, used


def _match_code(
    brand_key: str,
    aliases: list[str],
    existing: dict[str, str],
    used: set[str],
    claimed: set[str],
) -> str | None:
    candidates = [brand_key, *aliases]
    matched: list[str] = []
    for code, name in existing.items():
        if code in claimed or not name:
            continue
        if LEGACY_NAME_MAP.get(name) == brand_key:
            matched.append(code)
            continue
        if any(name in cand or cand in name for cand in candidates if cand):
            matched.append(code)
    if not matched:
        return None
    # 同名重复 code（历史脏数据）时，优先选 answers 里真正在用的那个
    matched.sort(key=lambda code: (code not in used, code))
    return matched[0]


def sync_products_from_brands(brands: dict[str, Any] | None = None) -> dict[str, Any]:
    if brands is None:
        brands = yaml_store.load_brands()
    brand_999 = brands.get("brand_999") or {}
    conn = sqlite3.connect(GEO_SQLITE_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_schema(conn)
        existing, used = _existing_products(conn)
        claimed: set[str] = set()
        synced: list[dict[str, str]] = []
        for order, (brand_key, spec) in enumerate(brand_999.items()):
            spec = spec or {}
            aliases = [str(a) for a in (spec.get("aliases") or [])]
            code = _match_code(brand_key, aliases, existing, used, claimed)
            if code is None:
                code = _stable_code(brand_key)
            claimed.add(code)
            conn.execute(
                """
                INSERT INTO products (
                    product_code, product_name, metadata_json,
                    category, aliases_json, is_active, display_order
                )
                VALUES (?, ?, '{}', ?, ?, 1, ?)
                ON CONFLICT(product_code) DO UPDATE SET
                    product_name = excluded.product_name,
                    category = excluded.category,
                    aliases_json = excluded.aliases_json,
                    is_active = 1,
                    display_order = excluded.display_order
                """,
                (
                    code,
                    brand_key,
                    str(spec.get("category") or ""),
                    json.dumps(aliases, ensure_ascii=False),
                    order,
                ),
            )
            synced.append({"product_code": code, "product_name": brand_key})
        if claimed:
            placeholders = ",".join("?" for _ in claimed)
            conn.execute(
                f"UPDATE products SET is_active = 0 WHERE product_code NOT IN ({placeholders})",
                sorted(claimed),
            )
        conn.commit()
        return {"synced": synced, "deactivated_unmatched": len(existing) - len(claimed & set(existing))}
    finally:
        conn.close()


def list_products(active_only: bool = True) -> list[dict[str, Any]]:
    if not GEO_SQLITE_PATH.exists():
        return []
    conn = sqlite3.connect(GEO_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        where = "WHERE is_active = 1" if active_only else ""
        rows = conn.execute(
            f"""
            SELECT product_code, product_name, category, aliases_json,
                   is_active, display_order
            FROM products {where}
            ORDER BY display_order, product_name
            """
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["aliases"] = json.loads(item.pop("aliases_json") or "[]")
            except json.JSONDecodeError:
                item["aliases"] = []
            result.append(item)
        return result
    finally:
        conn.close()
