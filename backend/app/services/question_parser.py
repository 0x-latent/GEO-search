from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

MAX_QUESTIONS = 500

# 支持的列名映射（中英文均可）
COLUMN_ALIASES = {
    "question": "question",
    "问题": "question",
    "提问": "question",
    "geo关键词": "question",
    "关键词": "question",
    "product_name": "product",
    "product": "product",
    "产品": "product",
    "产品名": "product",
    "产品名称": "product",
    "level": "level",
    "层级": "level",
    "问题层级": "level",
    "scenario": "scenario",
    "场景": "scenario",
    "关键词类型": "scenario",
    "用户画像": "persona",
}


def _normalize_header(header: str) -> str | None:
    return COLUMN_ALIASES.get(str(header).strip().lower()) or COLUMN_ALIASES.get(str(header).strip())


def _slug(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "", text)[:16]
    if cleaned:
        return cleaned.lower()
    if text.strip():
        # 纯中文产品名：取稳定 hash 作为 code，避免多个产品共用同一 code 互相覆盖
        import hashlib

        return "p" + hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:8]
    return fallback


def _build_questions(rows: list[dict[str, str]], default_product: str) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_product = ""
    for row in rows:
        text = str(row.get("question") or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        # 产品列空时沿用上一行（Excel 合并单元格导出后只有首行有值）
        row_product = str(row.get("product") or "").strip()
        if row_product:
            last_product = row_product
        product = row_product or last_product or str(default_product or "").strip()
        index = len(questions) + 1
        product_code = _slug(product, "custom") if product else "custom"
        questions.append({
            # id 带 _q4_ 使 05 推荐抽取覆盖全部用户上传问题（05 按 _q3_/_q4_/_q5_ 识别推荐类）
            "id": f"{product_code}_q4_{index:04d}",
            "product": product,
            "product_code": product_code,
            "category": "user_upload",
            "level": str(row.get("level") or "").strip(),
            "scenario": str(row.get("scenario") or "").strip(),
            "persona": str(row.get("persona") or "").strip(),
            "question": text,
            "has_brand_name": False,
            "is_variant": False,
            "variant_of": None,
        })
        if len(questions) > MAX_QUESTIONS:
            raise ValueError(f"问题数超过上限 {MAX_QUESTIONS}，请拆分后分批上传")
    if not questions:
        raise ValueError("未解析到任何问题，请检查文件格式（需包含“问题”或 question 列）")
    return questions


def _rows_to_records(rows: list[list[str]]) -> list[dict[str, str]]:
    """在前 10 行内自动定位表头行（含“问题/GEO关键词”等列的行），其余行按表头映射。

    找不到表头时退化为“第一列即问题”。"""
    if not rows:
        return []
    header_index = -1
    headers: list[str | None] = []
    for i, row in enumerate(rows[:10]):
        candidate = [_normalize_header(h) for h in row]
        if "question" in candidate:
            header_index = i
            headers = candidate
            break
    if header_index < 0:
        return [{"question": row[0]} for row in rows if row and str(row[0]).strip()]
    result = []
    for row in rows[header_index + 1 :]:
        item: dict[str, str] = {}
        for idx, header in enumerate(headers):
            if header and idx < len(row):
                item[header] = row[idx]
        result.append(item)
    return result


def _parse_csv(content: bytes) -> list[dict[str, str]]:
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    return _rows_to_records(list(reader))


def _parse_xlsx(content: bytes) -> list[dict[str, str]]:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sheet = workbook.active
    rows = [[("" if c is None else str(c)) for c in row] for row in sheet.iter_rows(values_only=True)]
    workbook.close()
    return _rows_to_records(rows)


def _parse_json(content: bytes) -> list[dict[str, str]]:
    data = json.loads(content.decode("utf-8-sig", errors="replace"))
    if not isinstance(data, list):
        raise ValueError("JSON 必须是数组")
    rows = []
    for item in data:
        if isinstance(item, str):
            rows.append({"question": item})
        elif isinstance(item, dict):
            row: dict[str, str] = {}
            for key, value in item.items():
                normalized = _normalize_header(key)
                if normalized and value is not None:
                    row[normalized] = str(value)
            rows.append(row)
    return rows


def parse_upload(filename: str, content: bytes, default_product: str = "") -> list[dict[str, Any]]:
    """解析上传文件为标准问题列表（id/product/level/scenario/question）。"""
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("文件超过 10MB")
    name = filename.lower()
    if name.endswith(".xlsx"):
        rows = _parse_xlsx(content)
    elif name.endswith(".csv"):
        rows = _parse_csv(content)
    elif name.endswith(".json"):
        rows = _parse_json(content)
    else:
        raise ValueError("仅支持 .xlsx / .csv / .json 文件")
    return _build_questions(rows, default_product)
