"""
Manage the unified GEO SQLite data store.

This script imports both existing raw JSON results and externally delivered
spreadsheets into one durable SQLite database. Excel/CSV files remain archived
sources; analysis should read from SQLite or from exports generated from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.sqlite_schema import ensure_schema, stage_for_level  # noqa: E402

DEFAULT_DB = BASE_DIR / "data" / "geo_datasets" / "geo_answers.sqlite"

YANGWEISHU_LEVEL_MAP = {
    "解决方案": {"level": "q3_solution", "category": "mention_recommend"},
    "泛式吃药": {"level": "q4_medicine", "category": "mention_recommend"},
    "中药相关": {"level": "q5_tcm", "category": "mention_recommend"},
}

MODEL_MAP = {
    "deepseek": ("deepseek", "DeepSeek"),
    "DeepSeek": ("deepseek", "DeepSeek"),
    "Deepseek": ("deepseek", "DeepSeek"),
    "Kimi": ("kimi", "Kimi"),
    "kimi": ("kimi", "Kimi"),
    "元宝": ("yuanbao", "元宝"),
    "千问": ("qwen", "通义千问"),
    "通义千问": ("qwen", "通义千问"),
    "百度AI": ("baidu", "百度AI"),
    "豆包": ("doubao", "豆包"),
    "doubao": ("doubao", "豆包"),
    "hunyuan": ("hunyuan", "腾讯混元"),
    "qwen": ("qwen", "通义千问"),
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def iso_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stable_hash(*parts: Any, length: int = 16) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def split_sources(value: Any) -> list[dict[str, str]]:
    text = clean_text(value)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[，,;\n\r\t ]+", text) if p.strip()]
    sources = []
    seen = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        sources.append({"title": "", "url": part})
    return sources


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    ensure_schema(conn)


def reset_dataset(conn: sqlite3.Connection, dataset_id: str) -> None:
    conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
    conn.commit()


def upsert_dataset(
    conn: sqlite3.Connection,
    dataset_id: str,
    name: str,
    description: str,
    source_type: str,
    source_path: str,
    metadata: dict[str, Any] | None = None,
    owner: str | None = None,
    product_code: str | None = None,
    batch_date: str | None = None,
    question_set_id: str | None = None,
) -> None:
    ensure_schema(conn)
    conn.execute(
        """
        INSERT INTO datasets (
            dataset_id, name, description, source_type, source_path,
            created_at, imported_at, metadata_json, owner_username,
            product_code, batch_date, question_set_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
            name = excluded.name,
            description = excluded.description,
            source_type = excluded.source_type,
            source_path = excluded.source_path,
            imported_at = excluded.imported_at,
            metadata_json = excluded.metadata_json,
            owner_username = COALESCE(excluded.owner_username, datasets.owner_username),
            product_code = COALESCE(excluded.product_code, datasets.product_code),
            batch_date = COALESCE(excluded.batch_date, datasets.batch_date),
            question_set_id = COALESCE(excluded.question_set_id, datasets.question_set_id)
        """,
        (
            dataset_id,
            name,
            description,
            source_type,
            source_path,
            now_iso(),
            now_iso(),
            json_dumps(metadata or {}),
            owner,
            product_code,
            batch_date,
            question_set_id,
        ),
    )


def upsert_product(conn: sqlite3.Connection, product_code: str, product_name: str, metadata: dict[str, Any] | None = None) -> None:
    # 产品主数据由 backend/app/services/product_master.py 从 brands.yaml 同步维护，
    # 导入侧只补缺失行，不覆盖已有的名称/分类/别名。
    if not product_code and not product_name:
        return
    product_code = product_code or stable_hash(product_name, length=10)
    conn.execute(
        """
        INSERT OR IGNORE INTO products (product_code, product_name, metadata_json)
        VALUES (?, ?, ?)
        """,
        (product_code, product_name or product_code, json_dumps(metadata or {})),
    )


def insert_question(conn: sqlite3.Connection, dataset_id: str, q: dict[str, Any]) -> None:
    product_code = clean_text(q.get("product_code"))
    product_name = clean_text(q.get("product"))
    upsert_product(conn, product_code, product_name)
    conn.execute(
        """
        INSERT INTO questions (
            dataset_id, question_id, product_code, product_name, category, level,
            source_level, scenario, question_text, has_brand_name, is_variant,
            variant_of, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, question_id) DO UPDATE SET
            product_code = excluded.product_code,
            product_name = excluded.product_name,
            category = excluded.category,
            level = excluded.level,
            source_level = excluded.source_level,
            scenario = excluded.scenario,
            question_text = excluded.question_text,
            has_brand_name = excluded.has_brand_name,
            is_variant = excluded.is_variant,
            variant_of = excluded.variant_of,
            metadata_json = excluded.metadata_json
        """,
        (
            dataset_id,
            clean_text(q.get("id")),
            product_code,
            product_name,
            clean_text(q.get("category")),
            clean_text(q.get("level")),
            clean_text(q.get("source_level")),
            clean_text(q.get("scenario")),
            clean_text(q.get("question")),
            1 if q.get("has_brand_name") else 0,
            1 if q.get("is_variant") else 0,
            clean_text(q.get("variant_of")),
            json_dumps({k: v for k, v in q.items() if k not in {
                "id", "product_code", "product", "category", "level", "source_level",
                "scenario", "question", "has_brand_name", "is_variant", "variant_of"
            }}),
        ),
    )


def insert_answer(
    conn: sqlite3.Connection,
    dataset_id: str,
    q: dict[str, Any],
    response: dict[str, Any],
    raw_path: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    insert_question(conn, dataset_id, q)
    model = clean_text(response.get("model"))
    model_name = clean_text(response.get("model_name"))
    if model in MODEL_MAP:
        model, default_name = MODEL_MAP[model]
        model_name = model_name or default_name
    search_enabled = 1 if response.get("search_enabled") else 0
    round_num = safe_int(response.get("round"), 1)
    answer_text = clean_text(response.get("answer"))
    answer_id = stable_hash(dataset_id, q["id"], model, search_enabled, round_num, length=24)
    raw_sources = response.get("sources") or []
    sources = []
    for item in raw_sources:
        if isinstance(item, dict):
            url = clean_text(item.get("url"))
            title = clean_text(item.get("title"))
        else:
            url = clean_text(item)
            title = ""
        if url or title:
            sources.append({"title": title, "url": url})
    source_count = safe_int(response.get("source_count"), len(sources)) or len(sources)

    conn.execute(
        """
        INSERT OR REPLACE INTO answers (
            dataset_id, answer_id, question_id, product_code, product_name,
            model, model_name, search_enabled, round, timestamp, answer_text,
            answer_chars, latency_ms, source_count, raw_path, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            answer_id,
            q["id"],
            clean_text(q.get("product_code")),
            clean_text(q.get("product")),
            model,
            model_name,
            search_enabled,
            round_num,
            clean_text(response.get("timestamp")),
            answer_text,
            len(answer_text),
            response.get("latency_ms"),
            source_count,
            raw_path,
            json_dumps(metadata or {}),
        ),
    )

    for i, item in enumerate(sources, start=1):
        url = item["url"]
        conn.execute(
            """
            INSERT OR REPLACE INTO sources (
                dataset_id, answer_id, source_index, title, url, domain, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                answer_id,
                i,
                item["title"],
                url,
                source_domain(url),
                "{}",
            ),
        )


def import_file_record(conn: sqlite3.Connection, dataset_id: str, path: Path, root: Path | None = None) -> str:
    stat = path.stat()
    rel = str(path.resolve())
    if root:
        try:
            rel = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            rel = str(path.resolve())
    sha = file_sha256(path)
    file_id = stable_hash(dataset_id, rel, sha, length=24)
    conn.execute(
        """
        INSERT OR REPLACE INTO import_files (
            file_id, dataset_id, source_path, file_name, file_type, sha256,
            size_bytes, modified_at, imported_at, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file_id,
            dataset_id,
            rel,
            path.name,
            path.suffix.lower().lstrip("."),
            sha,
            stat.st_size,
            datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            now_iso(),
            "{}",
        ),
    )
    return file_id


def frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.where(pd.notnull(df), None)
    records = []
    for row in df.to_dict(orient="records"):
        cleaned = {}
        for key, value in row.items():
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            cleaned[str(key)] = value
        records.append(cleaned)
    return records


def insert_external_table(
    conn: sqlite3.Connection,
    dataset_id: str,
    file_id: str,
    table_name: str,
    sheet_name: str,
    df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> int:
    records = frame_records(df)
    table_id = stable_hash(dataset_id, file_id, sheet_name or table_name, table_name, length=24)
    conn.execute("DELETE FROM external_rows WHERE table_id = ?", (table_id,))
    conn.execute(
        """
        INSERT OR REPLACE INTO external_tables (
            table_id, dataset_id, file_id, table_name, sheet_name,
            row_count, columns_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            table_id,
            dataset_id,
            file_id,
            table_name,
            sheet_name,
            len(records),
            json_dumps([str(c) for c in df.columns]),
            json_dumps(metadata or {}),
        ),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO external_rows (table_id, row_index, row_json) VALUES (?, ?, ?)",
        [(table_id, i + 1, json_dumps(row)) for i, row in enumerate(records)],
    )
    return len(records)


def import_tabular_file(conn: sqlite3.Connection, dataset_id: str, path: Path, root: Path) -> int:
    if path.name.startswith("~$"):
        return 0
    file_id = import_file_record(conn, dataset_id, path, root)
    imported_rows = 0
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            imported_rows += insert_external_table(
                conn,
                dataset_id,
                file_id,
                table_name=path.stem,
                sheet_name=sheet,
                df=df,
                metadata={"source_kind": "spreadsheet"},
            )
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=sep)
        except pd.errors.EmptyDataError:
            return imported_rows
        imported_rows += insert_external_table(
            conn,
            dataset_id,
            file_id,
            table_name=path.stem,
            sheet_name="",
            df=df,
            metadata={"source_kind": "csv"},
        )
    return imported_rows


def load_questions_map(paths: list[Path]) -> dict[str, dict[str, Any]]:
    qmap: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for q in data:
            if "id" in q:
                qmap[q["id"]] = q
    return qmap


def import_baseline(args: argparse.Namespace) -> dict[str, Any]:
    db_path = Path(args.db).resolve()
    conn = connect(db_path)
    init_db(conn)
    if args.reset:
        reset_dataset(conn, args.dataset_id)
    upsert_dataset(
        conn,
        args.dataset_id,
        args.name,
        args.description,
        "raw_json_pipeline",
        str(Path(args.raw_dir).resolve()),
        {"question_files": [args.questions, args.questions_base]},
        owner=getattr(args, "owner", None),
        product_code=getattr(args, "product_code", None),
        batch_date=getattr(args, "batch_date", None),
        question_set_id=getattr(args, "question_set_id", None),
    )

    qmap = load_questions_map([Path(args.questions).resolve(), Path(args.questions_base).resolve()])
    for q in qmap.values():
        insert_question(conn, args.dataset_id, q)

    raw_root = Path(args.raw_dir).resolve()
    raw_files = sorted(raw_root.rglob("*.json"))
    inserted = 0
    missing_questions = 0
    for path in raw_files:
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        qid = clean_text(response.get("question_id"))
        answer = clean_text(response.get("answer"))
        if not qid or not answer:
            continue
        q = qmap.get(qid)
        if not q:
            missing_questions += 1
            q = {
                "id": qid,
                "product": clean_text(response.get("product")),
                "product_code": "",
                "category": "",
                "level": "",
                "question": clean_text(response.get("question_text")),
                "has_brand_name": False,
                "is_variant": False,
                "variant_of": None,
            }
        rel = str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else str(path)
        if path.stem.endswith("_nosearch"):
            response["search_enabled"] = False
            source_mode = "filename_nosearch"
        elif path.stem.endswith("_search"):
            response["search_enabled"] = True
            source_mode = "filename_search"
        else:
            source_mode = "payload"
        insert_answer(
            conn,
            args.dataset_id,
            q,
            response,
            raw_path=rel,
            metadata={"source": "results/raw", "search_mode_source": source_mode},
        )
        inserted += 1

    external_rows = 0
    analysis_root = Path(args.analysis_dir).resolve()
    if analysis_root.exists():
        for path in sorted(analysis_root.glob("*.csv")):
            external_rows += import_tabular_file(conn, args.dataset_id, path, analysis_root)

    conn.commit()
    unique_answers = conn.execute(
        "SELECT COUNT(*) FROM answers WHERE dataset_id = ?",
        (args.dataset_id,),
    ).fetchone()[0]
    return {
        "dataset_id": args.dataset_id,
        "raw_json_files": inserted,
        "unique_answers": unique_answers,
        "questions": len(qmap),
        "missing_questions": missing_questions,
        "analysis_external_rows": external_rows,
        "db": str(db_path),
    }


def yangweishu_question_map(df: pd.DataFrame, product: str, product_code: str, brand_terms: list[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    qmap: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        level_label = clean_text(row.get("提问层级"))
        scenario = clean_text(row.get("场景"))
        question = clean_text(row.get("问题"))
        if not question:
            continue
        level_info = YANGWEISHU_LEVEL_MAP.get(level_label, {"level": f"external_{stable_hash(level_label, length=8)}", "category": "external"})
        qid = f"{product_code}_{level_info['level']}_{stable_hash(level_label, scenario, question, length=8)}"
        key = (level_label, scenario, question)
        if key not in qmap:
            qmap[key] = {
                "id": qid,
                "product": product,
                "product_code": product_code,
                "category": level_info["category"],
                "level": level_info["level"],
                "source_level": level_label,
                "scenario": scenario,
                "question": question,
                "has_brand_name": any(term and term in question for term in brand_terms),
                "is_variant": False,
                "variant_of": None,
            }
    return qmap


def import_yangweishu(args: argparse.Namespace) -> dict[str, Any]:
    db_path = Path(args.db).resolve()
    conn = connect(db_path)
    init_db(conn)
    if args.reset:
        reset_dataset(conn, args.dataset_id)
    source_root = Path(args.source_root).resolve()
    answer_path = Path(args.answers_xlsx).resolve()
    upsert_dataset(
        conn,
        args.dataset_id,
        args.name,
        args.description,
        "external_spreadsheet_bundle",
        str(source_root),
        {"answers_xlsx": str(answer_path)},
    )

    df = pd.read_excel(answer_path, sheet_name=args.answers_sheet)
    df = df[df["AI回答"].notna()].copy()
    brand_terms = [t.strip() for t in args.brand_terms.split(",") if t.strip()]
    qmap = yangweishu_question_map(df, args.product, args.product_code, brand_terms)

    inserted = 0
    for idx, row in enumerate(df.to_dict(orient="records"), start=2):
        level_label = clean_text(row.get("提问层级"))
        scenario = clean_text(row.get("场景"))
        question = clean_text(row.get("问题"))
        q = qmap.get((level_label, scenario, question))
        if not q:
            continue
        model_label = clean_text(row.get("AI模型"))
        model_key, model_name = MODEL_MAP.get(model_label, (model_label or "unknown", model_label or "unknown"))
        sources = split_sources(row.get("引用信源"))
        source_count = safe_int(row.get("信源数量"), len(sources))
        response = {
            "answer": clean_text(row.get("AI回答")),
            "model": model_key,
            "model_name": model_name,
            "search_enabled": source_count > 0 or bool(sources),
            "round": safe_int(row.get("查询轮次"), 1),
            "timestamp": iso_timestamp(row.get("查询时间")),
            "latency_ms": None,
            "sources": sources,
            "source_count": source_count,
        }
        insert_answer(
            conn,
            args.dataset_id,
            q,
            response,
            raw_path=str(answer_path.relative_to(BASE_DIR)) if answer_path.is_relative_to(BASE_DIR) else str(answer_path),
            metadata={"source": "external_answers_xlsx", "source_row": idx},
        )
        inserted += 1

    external_rows = 0
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv", ".tsv"}:
            external_rows += import_tabular_file(conn, args.dataset_id, path, source_root)

    conn.commit()
    return {
        "dataset_id": args.dataset_id,
        "answers": inserted,
        "questions": len(qmap),
        "external_rows": external_rows,
        "db": str(db_path),
    }


def export_dataset_to_raw(args: argparse.Namespace) -> dict[str, Any]:
    db_path = Path(args.db).resolve()
    conn = connect(db_path)
    output_root = Path(args.output_raw_dir).resolve()
    questions_output = Path(args.questions_output).resolve()
    if args.reset_output and output_root.exists():
        resolved = output_root.resolve()
        workspace = BASE_DIR.resolve()
        if not str(resolved).startswith(str(workspace)):
            raise ValueError(f"Refusing to delete outside workspace: {resolved}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    questions_output.parent.mkdir(parents=True, exist_ok=True)

    question_rows = conn.execute(
        """
        SELECT question_id, product_name, product_code, category, level, question_text,
               has_brand_name, is_variant, variant_of, source_level, scenario, metadata_json
        FROM questions
        WHERE dataset_id = ?
        ORDER BY question_id
        """,
        (args.dataset_id,),
    ).fetchall()
    question_payload = []
    for row in question_rows:
        question_payload.append({
            "id": row[0],
            "product": row[1],
            "product_code": row[2],
            "category": row[3],
            "level": row[4],
            "question": row[5],
            "has_brand_name": bool(row[6]),
            "is_variant": bool(row[7]),
            "variant_of": row[8] or None,
            "source_level": row[9] or "",
            "scenario": row[10] or "",
        })
    questions_output.write_text(json.dumps(question_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    answer_rows = conn.execute(
        """
        SELECT a.answer_id, a.question_id, q.question_text, a.product_name, a.model,
               a.model_name, a.search_enabled, a.round, a.timestamp, a.answer_text,
               a.latency_ms, a.source_count, a.metadata_json
        FROM answers a
        JOIN questions q ON q.dataset_id = a.dataset_id AND q.question_id = a.question_id
        WHERE a.dataset_id = ?
        ORDER BY a.model, a.question_id, a.round, a.search_enabled
        """,
        (args.dataset_id,),
    ).fetchall()
    written = 0
    for row in answer_rows:
        answer_id, qid, question_text, product, model, model_name, search, round_num, ts, answer, latency, source_count, meta = row
        sources = conn.execute(
            """
            SELECT title, url
            FROM sources
            WHERE dataset_id = ? AND answer_id = ?
            ORDER BY source_index
            """,
            (args.dataset_id, answer_id),
        ).fetchall()
        payload = {
            "answer": answer,
            "sources": [{"title": s[0] or "", "url": s[1] or ""} for s in sources],
            "model": model,
            "model_name": model_name,
            "latency_ms": latency,
            "search_enabled": bool(search),
            "question_id": qid,
            "question_text": question_text,
            "product": product,
            "round": round_num,
            "timestamp": ts,
            "source_count": source_count,
            "metadata": json.loads(meta or "{}"),
        }
        search_tag = "search" if search else "nosearch"
        path = output_root / model / f"{qid}_r{round_num}_{search_tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
    return {"dataset_id": args.dataset_id, "answers_written": written, "questions_written": len(question_payload), "output_raw_dir": str(output_root)}


# ---------------------------------------------------------------------------
# 物化：把 splits 聚合从"请求时"搬到"入库时"（性能根治 + 趋势/洞察数据源）
# ---------------------------------------------------------------------------

SEARCH_FLAG = {"是": "1", "否": "0", "1": "1", "0": "0", "True": "1", "False": "0"}

YANG_TABLE_PATTERN = "养胃舒-%汇总统计数据%"
YANG_BRAND_TERMS = ("养胃舒", "三九养胃舒", "999养胃舒")


def _search_flag(value: Any) -> str:
    text = clean_text(value)
    return SEARCH_FLAG.get(text, text or "0")


def _level_bucket(level: str) -> str:
    """detail 表的逐题层级归并到 mention_report 的聚合桶（Q1/Q2、Q3/Q4、Q5）。"""
    text = clean_text(level).lower()
    if text.startswith(("q1", "q2")):
        return "Q1/Q2"
    if text.startswith(("q3", "q4")):
        return "Q3/Q4"
    if text.startswith("q5"):
        return "Q5"
    return clean_text(level)


def _num_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fingerprint(texts: list[str]) -> str:
    joined = "\n".join(sorted(t.strip() for t in texts if t and t.strip()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _ext_table_rows(conn: sqlite3.Connection, dataset_id: str, table_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT er.row_json FROM external_rows er
        JOIN external_tables et ON et.table_id = er.table_id
        WHERE et.dataset_id = ? AND et.table_name = ?
        """,
        (dataset_id, table_name),
    ).fetchall()
    result = []
    for (row_json,) in rows:
        payload = json.loads(row_json)
        result.append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in payload.items()})
    return result


def _avg_or_none(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def materialize_dataset(conn: sqlite3.Connection, dataset_id: str) -> dict[str, Any]:
    ensure_schema(conn)
    exists = conn.execute("SELECT 1 FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    if not exists:
        raise ValueError(f"dataset not found: {dataset_id}")

    label_rows = conn.execute(
        "SELECT DISTINCT product_name, product_code FROM questions WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchall()
    label_to_code = {clean_text(name): clean_text(code) for name, code in label_rows if clean_text(name)}

    def pcode(label: Any) -> str:
        text = clean_text(label)
        return label_to_code.get(text) or (text and stable_hash(text, length=10)) or "unknown"

    for table in ("metrics_summary", "metrics_recommendation", "metric_evidence", "dataset_products"):
        conn.execute(f"DELETE FROM {table} WHERE dataset_id = ?", (dataset_id,))

    stats = {"summary": 0, "recommendation": 0, "evidence": 0}

    def insert_summary(**kw: Any) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO metrics_summary (
                dataset_id, product_code, stage, question_level, model, search_enabled,
                total_answers, category_mention_rate, brand_mention_rate, brand_rec_rate,
                generic_mention_rate, generic_rec_rate, competitor_mention_rate,
                competitor_rec_rate, first_rate, top3_rate, avg_rank,
                negative_count, negative_rate, accuracy_rate, wrong_claims, total_claims, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id, kw["product_code"], kw["stage"], kw["question_level"], kw["model"],
                kw["search_enabled"], kw.get("total_answers"), kw.get("category_mention_rate"),
                kw.get("brand_mention_rate"), kw.get("brand_rec_rate"),
                kw.get("generic_mention_rate"), kw.get("generic_rec_rate"),
                kw.get("competitor_mention_rate"), kw.get("competitor_rec_rate"),
                kw.get("first_rate"), kw.get("top3_rate"), kw.get("avg_rank"),
                kw.get("negative_count"), kw.get("negative_rate"), kw.get("accuracy_rate"),
                kw.get("wrong_claims"), kw.get("total_claims"),
                json_dumps(kw.get("extra") or {}),
            ),
        )
        stats["summary"] += 1

    def insert_rec(**kw: Any) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO metrics_recommendation (
                dataset_id, product_code, stage, question_level, model, search_enabled,
                rank, rec_product, name_type, mention_count, mention_rate,
                strong_count, strong_rate, moderate_count, negative_count, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id, kw["product_code"], kw["stage"], kw.get("question_level", ""),
                kw["model"], kw["search_enabled"], kw.get("rank"), kw["rec_product"],
                kw.get("name_type", ""), kw.get("mention_count"), kw.get("mention_rate"),
                kw.get("strong_count"), kw.get("strong_rate"), kw.get("moderate_count"),
                kw.get("negative_count"), json_dumps(kw.get("extra") or {}),
            ),
        )
        stats["recommendation"] += 1

    def insert_evidence(**kw: Any) -> None:
        conn.execute(
            """
            INSERT INTO metric_evidence (
                dataset_id, evidence_type, product_code, stage, question_level, question_id,
                model, search_enabled, round, rec_product, name_type, rank,
                strength, verdict, detail, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id, kw["evidence_type"], kw.get("product_code"), kw.get("stage"),
                kw.get("question_level", ""), kw.get("question_id", ""), kw.get("model", ""),
                kw.get("search_enabled"), kw.get("round"), kw.get("rec_product"),
                kw.get("name_type"), kw.get("rank"), kw.get("strength"), kw.get("verdict"),
                kw.get("detail"), json_dumps(kw.get("payload") or {}),
            ),
        )
        stats["evidence"] += 1

    # ---- 1) mention_report → metrics_summary（提及/推荐率主指标） ----
    # 负面口径：negative_count/rate = 对"我方品牌"的负面定性（品牌健康的核心问题）；
    # 全部产品的负面提及数（含竞品被警告）保留在 extra_json 供参考。
    for row in _ext_table_rows(conn, dataset_id, "mention_report"):
        label = clean_text(row.get("产品"))
        level = clean_text(row.get("问题层级"))
        total = safe_int(row.get("总回答数"))
        own_negative = (
            safe_int(row.get("999品牌负面提及数"))
            if row.get("999品牌负面提及数") is not None
            else None
        )
        insert_summary(
            product_code=pcode(label),
            stage=stage_for_level(level),
            question_level=level,
            model=clean_text(row.get("模型")),
            search_enabled=_search_flag(row.get("联网")),
            total_answers=total,
            category_mention_rate=_num_or_none(row.get("品类提及率")),
            brand_mention_rate=_num_or_none(row.get("999品牌提及率")),
            brand_rec_rate=_num_or_none(row.get("999品牌推荐率")),
            generic_mention_rate=_num_or_none(row.get("通用名提及率")),
            generic_rec_rate=_num_or_none(row.get("通用名推荐率")),
            competitor_mention_rate=_num_or_none(row.get("竞品品牌提及率")),
            competitor_rec_rate=_num_or_none(row.get("竞品品牌推荐率")),
            negative_count=own_negative,
            negative_rate=round(own_negative / total, 4) if (own_negative is not None and total) else None,
            extra={
                "_source": "mention",
                "label": label,
                "999品牌提及次数": row.get("999品牌提及次数"),
                "通用名提及次数": row.get("通用名提及次数"),
                "竞品品牌提及次数": row.get("竞品品牌提及次数"),
                "成分品类级提及次数": row.get("成分品类级提及次数"),
                "负面提及数_全部产品": row.get("负面提及数"),
                "负面提及率_全部产品": row.get("负面提及率"),
            },
        )

    # ---- 2) accuracy_detail → 品牌阶段准确率（聚合）+ 逐条证据 ----
    acc_agg: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for row in _ext_table_rows(conn, dataset_id, "accuracy_detail"):
        if row.get("问题ID") is None or row.get("准确率") is None:
            continue  # 旧格式或空行
        label = clean_text(row.get("产品"))
        level = clean_text(row.get("问题类型"))
        model = clean_text(row.get("模型"))
        sea = _search_flag(row.get("联网"))
        key = (label, level, model, sea)
        agg = acc_agg.setdefault(key, {"correct": 0, "wrong": 0, "claims": 0})
        correct, wrong = safe_int(row.get("正确")), safe_int(row.get("错误"))
        agg["correct"] += correct
        agg["wrong"] += wrong
        agg["claims"] += safe_int(row.get("知识点数"))
        insert_evidence(
            evidence_type="accuracy",
            product_code=pcode(label),
            stage=stage_for_level(level),
            question_level=level,
            question_id=clean_text(row.get("问题ID")),
            model=model,
            search_enabled=1 if sea == "1" else 0,
            round=safe_int(row.get("轮次"), 1),
            verdict="wrong" if wrong > 0 else ("correct" if correct > 0 else "unverified"),
            detail=clean_text(row.get("错误摘要")),
            payload={
                "知识点数": row.get("知识点数"), "正确": correct, "错误": wrong,
                "无依据": row.get("无依据"), "准确率": row.get("准确率"),
            },
        )
    for (label, level, model, sea), agg in acc_agg.items():
        denom = agg["correct"] + agg["wrong"]
        insert_summary(
            product_code=pcode(label),
            stage=stage_for_level(level),
            question_level=level,
            model=model,
            search_enabled=sea,
            accuracy_rate=round(agg["correct"] / denom, 4) if denom else None,
            wrong_claims=agg["wrong"],
            total_claims=agg["claims"],
            extra={"_source": "accuracy", "label": label},
        )

    # ---- 3) rec_overview → 竞品推荐排行 ----
    for row in _ext_table_rows(conn, dataset_id, "rec_overview"):
        label = clean_text(row.get("产品"))
        insert_rec(
            product_code=pcode(label),
            stage="all",
            question_level="",
            model=clean_text(row.get("模型")),
            search_enabled=_search_flag(row.get("联网")),
            rank=safe_int(row.get("排名")) or None,
            rec_product=clean_text(row.get("被推荐产品")),
            name_type=clean_text(row.get("名称类型")),
            mention_count=_num_or_none(row.get("提及次数")),
            mention_rate=_num_or_none(row.get("提及率")),
            strong_count=_num_or_none(row.get("强推荐次数")),
            strong_rate=_num_or_none(row.get("强推荐率")),
            extra={"label": label, "应答总数": row.get("应答总数"), "可选次数": row.get("可选次数")},
        )

    # ---- 4) brand_generic_detail → 推荐证据链 + 负面证据/计数 ----
    neg_agg: dict[tuple[str, str, str, str], int] = {}
    detail_has_sentiment = False
    for row in _ext_table_rows(conn, dataset_id, "brand_generic_detail"):
        label = clean_text(row.get("产品"))
        level = clean_text(row.get("问题层级"))
        model = clean_text(row.get("模型"))
        sea = _search_flag(row.get("联网"))
        sentiment = clean_text(row.get("情感")).lower()
        if sentiment:
            detail_has_sentiment = True
        is_negative = sentiment in {"negative", "负面"}
        common = dict(
            product_code=pcode(label),
            stage=stage_for_level(level),
            question_level=level,
            question_id=clean_text(row.get("问题ID")),
            model=model,
            search_enabled=1 if sea == "1" else 0,
            round=safe_int(row.get("轮次"), 1),
            rec_product=clean_text(row.get("推荐产品")),
            name_type=clean_text(row.get("名称类型")),
            rank=safe_int(row.get("推荐排名")) or None,
            strength=clean_text(row.get("推荐强度")),
            detail=clean_text(row.get("推荐原因")),
            payload={"label": label, "是否推荐": row.get("是否推荐"), "情感": sentiment or None},
        )
        insert_evidence(evidence_type="recommendation", **common)
        if is_negative:
            insert_evidence(evidence_type="negative", **common)
            # 汇总口径只统计我方品牌的负面（与 mention_report 的 999品牌负面提及数一致）
            if common["name_type"] == "999品牌":
                neg_agg[(label, _level_bucket(level), model, sea)] = (
                    neg_agg.get((label, _level_bucket(level), model, sea), 0) + 1
                )
    # mention_report 没带负面列时（老版 05 输出），从 detail 聚合回填
    if detail_has_sentiment:
        for (label, bucket, model, sea), count in neg_agg.items():
            conn.execute(
                """
                UPDATE metrics_summary
                SET negative_count = ?,
                    negative_rate = CASE WHEN COALESCE(total_answers, 0) > 0
                                         THEN ROUND(? * 1.0 / total_answers, 4) END
                WHERE dataset_id = ? AND product_code = ? AND question_level = ?
                  AND model = ? AND search_enabled = ? AND negative_count IS NULL
                """,
                (count, count, dataset_id, pcode(label), bucket, model, sea),
            )

    # ---- 5) recommendation_detail → 品类推荐结构 + 品类证据 ----
    cat_agg: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _ext_table_rows(conn, dataset_id, "recommendation_detail"):
        label = clean_text(row.get("产品"))
        category = clean_text(row.get("品类"))
        model = clean_text(row.get("模型"))
        sea = _search_flag(row.get("联网"))
        strength = clean_text(row.get("推荐强度"))
        if category:
            item = cat_agg.setdefault(
                (label, model, sea, category), {"count": 0, "strong": 0}
            )
            item["count"] += 1
            if strength == "strong":
                item["strong"] += 1
        insert_evidence(
            evidence_type="category",
            product_code=pcode(label),
            stage=stage_for_level(""),
            question_level="",
            question_id=clean_text(row.get("问题ID")),
            model=model,
            search_enabled=1 if sea == "1" else 0,
            round=safe_int(row.get("轮次"), 1),
            rec_product=clean_text(row.get("推荐产品")),
            rank=safe_int(row.get("推荐排名")) or None,
            strength=strength,
            detail=clean_text(row.get("推荐理由")),
            payload={"label": label, "品类": category},
        )
    for (label, model, sea, category), item in cat_agg.items():
        insert_rec(
            product_code=pcode(label),
            stage="category",
            question_level="品类",
            model=model,
            search_enabled=sea,
            rec_product=category,
            name_type="品类",
            mention_count=item["count"],
            strong_count=item["strong"],
            extra={"label": label},
        )

    # ---- 6) 养胃舒专项（厂商预聚合 Excel，无逐题证据，联网维度为 agg） ----
    yang_rows = conn.execute(
        """
        SELECT et.table_name, er.row_json
        FROM external_rows er
        JOIN external_tables et ON et.table_id = er.table_id
        WHERE et.dataset_id = ? AND et.table_name LIKE ? AND et.sheet_name <> '字段说明'
        """,
        (dataset_id, YANG_TABLE_PATTERN),
    ).fetchall()
    yang_agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    yang_evidence_count = 0
    for table_name, row_json in yang_rows:
        payload = json.loads(row_json)
        payload = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in payload.items()}
        if not ("提问词" in payload and "AI模型" in payload and "目标品牌" in payload):
            continue
        source_level = table_name.replace("养胃舒-", "").split("汇总统计数据")[0]
        model = clean_text(payload.get("AI模型"))
        brand = clean_text(payload.get("目标品牌"))
        item = yang_agg.setdefault(
            (source_level, model, brand),
            {"count": 0, "visibility": [], "top3": [], "first": [], "rank": []},
        )
        item["count"] += 1
        item["visibility"].append(_num_or_none(payload.get("能见度")) or 0.0)
        if payload.get("前三率") is not None:
            item["top3"].append(_num_or_none(payload.get("前三率")))
        if payload.get("首位率") is not None:
            item["first"].append(_num_or_none(payload.get("首位率")))
        rank_val = _num_or_none(payload.get("位次"))
        if rank_val and rank_val > 0:
            item["rank"].append(rank_val)
        insert_evidence(
            evidence_type="yang_metric",
            product_code="weitai",
            stage=stage_for_level(source_level),
            question_level=source_level,
            question_id="",
            model=model,
            rec_product=brand,
            name_type="目标品牌",
            detail=clean_text(payload.get("提问词")),
            payload={
                "能见度": payload.get("能见度"), "位次": payload.get("位次"),
                "前三率": payload.get("前三率"), "首位率": payload.get("首位率"),
                "轮数": payload.get("轮数"),
            },
        )
        yang_evidence_count += 1

    yang_groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for (source_level, model, brand), item in yang_agg.items():
        yang_groups.setdefault((source_level, model), []).append((brand, item))
    for (source_level, model), items in yang_groups.items():
        ranked = sorted(
            items,
            key=lambda pair: (
                _avg_or_none(pair[1]["visibility"]) or 0,
                _avg_or_none(pair[1]["top3"]) or 0,
                _avg_or_none(pair[1]["first"]) or 0,
            ),
            reverse=True,
        )
        for index, (brand, item) in enumerate(ranked, start=1):
            visibility = _avg_or_none(item["visibility"])
            top3 = _avg_or_none(item["top3"])
            first = _avg_or_none(item["first"])
            avg_rank = _avg_or_none(item["rank"])
            mention_count = round((visibility or 0) * item["count"], 2)
            strong_count = min(round((top3 or 0) * item["count"], 2), mention_count)
            insert_rec(
                product_code="weitai",
                stage=stage_for_level(source_level),
                question_level=source_level,
                model=model,
                search_enabled="agg",
                rank=index,
                rec_product=brand,
                name_type="目标品牌",
                mention_count=mention_count,
                mention_rate=visibility,
                strong_count=strong_count,
                strong_rate=top3,
                extra={"平均首位率": first, "平均位次": avg_rank, "样本数": item["count"]},
            )
            if any(term in brand for term in YANG_BRAND_TERMS):
                insert_summary(
                    product_code="weitai",
                    stage=stage_for_level(source_level),
                    question_level=source_level,
                    model=model,
                    search_enabled="agg",
                    total_answers=item["count"],
                    brand_mention_rate=visibility,
                    first_rate=first,
                    top3_rate=top3,
                    avg_rank=avg_rank,
                    extra={"_source": "yang", "label": "养胃舒专项", "目标品牌": brand},
                )

    # ---- 7) 批次归属：dataset_products + 问题集指纹 + 批次日期 ----
    q_rows = conn.execute(
        "SELECT product_code, product_name, question_text FROM questions WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchall()
    per_product: dict[str, dict[str, Any]] = {}
    all_texts: list[str] = []
    for code, name, text in q_rows:
        code = clean_text(code) or "unknown"
        item = per_product.setdefault(code, {"name": clean_text(name), "texts": []})
        item["texts"].append(clean_text(text))
        all_texts.append(clean_text(text))
    for code, item in per_product.items():
        conn.execute(
            """
            INSERT OR REPLACE INTO dataset_products (
                dataset_id, product_code, product_name, question_set_id, question_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (dataset_id, code, item["name"], _fingerprint(item["texts"]), len(item["texts"])),
        )
    batch_match = re.search(r"_(\d{8})$", dataset_id)
    batch_date = None
    if batch_match:
        raw = batch_match.group(1)
        batch_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    conn.execute(
        """
        UPDATE datasets
        SET question_set_id = ?,
            batch_date = COALESCE(batch_date, ?),
            product_code = COALESCE(product_code, ?)
        WHERE dataset_id = ?
        """,
        (
            _fingerprint(all_texts) if all_texts else None,
            batch_date,
            list(per_product)[0] if len(per_product) == 1 else None,
            dataset_id,
        ),
    )

    conn.commit()
    stats["dataset_products"] = len(per_product)
    stats["yang_evidence"] = yang_evidence_count
    return stats


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    conn = connect(Path(args.db).resolve())
    init_db(conn)
    if args.dataset_id == "all":
        ids = [row[0] for row in conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id")]
    else:
        ids = [args.dataset_id]
    results = {}
    for dataset_id in ids:
        results[dataset_id] = materialize_dataset(conn, dataset_id)
    return results


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    conn = connect(Path(args.db).resolve())
    rows = conn.execute(
        """
        SELECT d.dataset_id, d.name,
               COUNT(DISTINCT q.question_id) AS questions,
               COUNT(DISTINCT a.answer_id) AS answers,
               COUNT(DISTINCT p.product_code) AS products,
               COUNT(DISTINCT a.model) AS models,
               COUNT(DISTINCT s.url) AS source_urls
        FROM datasets d
        LEFT JOIN questions q ON q.dataset_id = d.dataset_id
        LEFT JOIN answers a ON a.dataset_id = d.dataset_id
        LEFT JOIN products p ON p.product_code = q.product_code
        LEFT JOIN sources s ON s.dataset_id = a.dataset_id AND s.answer_id = a.answer_id
        GROUP BY d.dataset_id, d.name
        ORDER BY d.dataset_id
        """
    ).fetchall()
    external = conn.execute(
        """
        SELECT dataset_id, COUNT(*) AS tables, COALESCE(SUM(row_count), 0) AS rows
        FROM external_tables
        GROUP BY dataset_id
        """
    ).fetchall()
    return {
        "datasets": [
            {
                "dataset_id": row[0],
                "name": row[1],
                "questions": row[2],
                "answers": row[3],
                "products": row[4],
                "models": row[5],
                "source_urls": row[6],
            }
            for row in rows
        ],
        "external_tables": [
            {"dataset_id": row[0], "tables": row[1], "rows": row[2]}
            for row in external
        ],
    }


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import and query unified GEO SQLite datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Initialize SQLite schema.")
    add_common(p)

    p = sub.add_parser("import-baseline", help="Import the existing 8-product raw JSON pipeline outputs.")
    add_common(p)
    p.add_argument("--dataset-id", default="baseline_8products_20260423")
    p.add_argument("--name", default="8产品基线数据")
    p.add_argument("--description", default="现有 results/raw 采集结果及 results/analysis 分析输出")
    p.add_argument("--raw-dir", default=str(BASE_DIR / "results" / "raw"))
    p.add_argument("--analysis-dir", default=str(BASE_DIR / "results" / "analysis"))
    p.add_argument("--questions", default=str(BASE_DIR / "questions" / "questions_expanded.json"))
    p.add_argument("--questions-base", default=str(BASE_DIR / "questions" / "questions_base.json"))
    p.add_argument("--owner", default=None, help="Owner username for this dataset (user-scoped access).")
    p.add_argument("--product-code", default=None, help="Primary product code for this batch (trend anchor).")
    p.add_argument("--batch-date", default=None, help="Collection batch date YYYY-MM-DD.")
    p.add_argument("--question-set-id", default=None, help="Question set fingerprint for trend comparability.")
    p.add_argument("--reset", action="store_true", help="Delete and re-import this dataset.")

    p = sub.add_parser("import-yangweishu", help="Import the full 三九养胃舒 source folder.")
    add_common(p)
    p.add_argument("--dataset-id", default="weitai_yangweishu_20260602")
    p.add_argument("--name", default="三九胃泰/养胃舒专项数据")
    p.add_argument("--description", default="养胃舒 AI 回答明细、信源明细、提及推荐率、社媒热度、问题库与选词建议")
    p.add_argument("--source-root", default=str(BASE_DIR / "questions" / "三九养胃舒数据源"))
    p.add_argument("--answers-xlsx", default=str(BASE_DIR / "questions" / "三九养胃舒数据源" / "【养胃舒】明细数据（AI回答）.xlsx"))
    p.add_argument("--answers-sheet", default="明细数据")
    p.add_argument("--product", default="胃泰")
    p.add_argument("--product-code", default="weitai")
    p.add_argument("--brand-terms", default="养胃舒,三九胃泰,999胃泰,胃泰")
    p.add_argument("--reset", action="store_true", help="Delete and re-import this dataset.")

    p = sub.add_parser("export-raw", help="Export one SQLite dataset back to raw JSON files for existing scripts.")
    add_common(p)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--output-raw-dir", required=True)
    p.add_argument("--questions-output", required=True)
    p.add_argument("--reset-output", action="store_true")

    p = sub.add_parser("materialize", help="Materialize metrics tables for dashboards/insight APIs.")
    add_common(p)
    p.add_argument("--dataset-id", default="all", help="Dataset id, or 'all' for every dataset.")

    p = sub.add_parser("summary", help="Print dataset counts.")
    add_common(p)

    p = sub.add_parser("import-all", help="Import both baseline and yangweishu datasets.")
    add_common(p)
    p.add_argument("--reset", action="store_true", help="Reset both known datasets before import.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init":
        conn = connect(Path(args.db).resolve())
        init_db(conn)
        result = {"db": str(Path(args.db).resolve()), "status": "initialized"}
    elif args.command == "import-baseline":
        result = import_baseline(args)
    elif args.command == "import-yangweishu":
        result = import_yangweishu(args)
    elif args.command == "export-raw":
        result = export_dataset_to_raw(args)
    elif args.command == "materialize":
        result = materialize(args)
    elif args.command == "summary":
        result = summarize(args)
    elif args.command == "import-all":
        common_db = args.db
        baseline_args = argparse.Namespace(
            db=common_db,
            dataset_id="baseline_8products_20260423",
            name="8产品基线数据",
            description="现有 results/raw 采集结果及 results/analysis 分析输出",
            raw_dir=str(BASE_DIR / "results" / "raw"),
            analysis_dir=str(BASE_DIR / "results" / "analysis"),
            questions=str(BASE_DIR / "questions" / "questions_expanded.json"),
            questions_base=str(BASE_DIR / "questions" / "questions_base.json"),
            reset=args.reset,
        )
        yangweishu_args = argparse.Namespace(
            db=common_db,
            dataset_id="weitai_yangweishu_20260602",
            name="三九胃泰/养胃舒专项数据",
            description="养胃舒 AI 回答明细、信源明细、提及推荐率、社媒热度、问题库与选词建议",
            source_root=str(BASE_DIR / "questions" / "三九养胃舒数据源"),
            answers_xlsx=str(BASE_DIR / "questions" / "三九养胃舒数据源" / "【养胃舒】明细数据（AI回答）.xlsx"),
            answers_sheet="明细数据",
            product="胃泰",
            product_code="weitai",
            brand_terms="养胃舒,三九胃泰,999胃泰,胃泰",
            reset=args.reset,
        )
        result = {
            "baseline": import_baseline(baseline_args),
            "yangweishu": import_yangweishu(yangweishu_args),
            "summary": summarize(argparse.Namespace(db=common_db)),
        }
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
