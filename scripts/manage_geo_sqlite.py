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
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
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


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    source_type TEXT NOT NULL,
    source_path TEXT,
    created_at TEXT,
    imported_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS products (
    product_code TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS questions (
    dataset_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    product_code TEXT,
    product_name TEXT,
    category TEXT,
    level TEXT,
    source_level TEXT,
    scenario TEXT,
    question_text TEXT NOT NULL,
    has_brand_name INTEGER NOT NULL DEFAULT 0,
    is_variant INTEGER NOT NULL DEFAULT 0,
    variant_of TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, question_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS answers (
    dataset_id TEXT NOT NULL,
    answer_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    product_code TEXT,
    product_name TEXT,
    model TEXT NOT NULL,
    model_name TEXT,
    search_enabled INTEGER NOT NULL DEFAULT 0,
    round INTEGER NOT NULL,
    timestamp TEXT,
    answer_text TEXT NOT NULL,
    answer_chars INTEGER,
    latency_ms REAL,
    source_count INTEGER NOT NULL DEFAULT 0,
    raw_path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, answer_id),
    UNIQUE (dataset_id, question_id, model, search_enabled, round),
    FOREIGN KEY (dataset_id, question_id)
        REFERENCES questions(dataset_id, question_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sources (
    dataset_id TEXT NOT NULL,
    answer_id TEXT NOT NULL,
    source_index INTEGER NOT NULL,
    title TEXT,
    url TEXT,
    domain TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, answer_id, source_index),
    FOREIGN KEY (dataset_id, answer_id)
        REFERENCES answers(dataset_id, answer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS import_files (
    file_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    modified_at TEXT,
    imported_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS external_tables (
    table_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    sheet_name TEXT,
    row_count INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES import_files(file_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS external_rows (
    table_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    row_json TEXT NOT NULL,
    PRIMARY KEY (table_id, row_index),
    FOREIGN KEY (table_id) REFERENCES external_tables(table_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_answers_dataset_product
    ON answers(dataset_id, product_code, model, search_enabled);
CREATE INDEX IF NOT EXISTS idx_answers_question
    ON answers(dataset_id, question_id);
CREATE INDEX IF NOT EXISTS idx_sources_domain
    ON sources(dataset_id, domain);
CREATE INDEX IF NOT EXISTS idx_external_tables_dataset
    ON external_tables(dataset_id, table_name);
"""


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
    conn.executescript(SCHEMA)
    conn.commit()


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
) -> None:
    conn.execute(
        """
        INSERT INTO datasets (
            dataset_id, name, description, source_type, source_path,
            created_at, imported_at, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
            name = excluded.name,
            description = excluded.description,
            source_type = excluded.source_type,
            source_path = excluded.source_path,
            imported_at = excluded.imported_at,
            metadata_json = excluded.metadata_json
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
        ),
    )


def upsert_product(conn: sqlite3.Connection, product_code: str, product_name: str, metadata: dict[str, Any] | None = None) -> None:
    if not product_code and not product_name:
        return
    product_code = product_code or stable_hash(product_name, length=10)
    conn.execute(
        """
        INSERT INTO products (product_code, product_name, metadata_json)
        VALUES (?, ?, ?)
        ON CONFLICT(product_code) DO UPDATE SET
            product_name = excluded.product_name,
            metadata_json = excluded.metadata_json
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
        df = pd.read_csv(path, encoding="utf-8-sig", sep=sep)
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
        SELECT answer_id, question_id, question_text, product_name, model, model_name,
               search_enabled, round, timestamp, answer_text, latency_ms, source_count,
               metadata_json
        FROM answers
        JOIN questions USING (dataset_id, question_id)
        WHERE answers.dataset_id = ?
        ORDER BY model, question_id, round, search_enabled
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
