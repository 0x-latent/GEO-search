"""
GEO SQLite 统一 schema 定义与迁移。

单一真相源：scripts/manage_geo_sqlite.py（CLI 导入）和 backend 启动都从这里
拿 DDL 并调用 ensure_schema()，避免两侧建表口径漂移。

迁移策略沿用 owner_username 的先例：新表用 CREATE TABLE IF NOT EXISTS，
已有表的新列用 PRAGMA table_info 检查后 ALTER TABLE ADD COLUMN（惰性、幂等）。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# 基础表（原 scripts/manage_geo_sqlite.py 的 SCHEMA，原样迁入）
# ---------------------------------------------------------------------------

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
    metadata_json TEXT NOT NULL DEFAULT '{}',
    owner_username TEXT,
    product_code TEXT,
    batch_date TEXT,
    question_set_id TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_code TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    category TEXT,
    aliases_json TEXT DEFAULT '[]',
    is_active INTEGER DEFAULT 1,
    display_order INTEGER DEFAULT 0
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

# ---------------------------------------------------------------------------
# 扩展表：批次归属、物化指标、证据链（materialize 子命令写入）
# ---------------------------------------------------------------------------

EXTENSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_products (
    dataset_id TEXT NOT NULL,
    product_code TEXT NOT NULL,
    product_name TEXT,
    question_set_id TEXT NOT NULL,
    question_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (dataset_id, product_code),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics_summary (
    dataset_id TEXT NOT NULL,
    product_code TEXT NOT NULL,
    stage TEXT NOT NULL,
    question_level TEXT NOT NULL,
    model TEXT NOT NULL,
    search_enabled TEXT NOT NULL,
    total_answers INTEGER,
    category_mention_rate REAL,
    brand_mention_rate REAL,
    brand_rec_rate REAL,
    generic_mention_rate REAL,
    generic_rec_rate REAL,
    competitor_mention_rate REAL,
    competitor_rec_rate REAL,
    first_rate REAL,
    top3_rate REAL,
    avg_rank REAL,
    negative_count INTEGER,
    negative_rate REAL,
    accuracy_rate REAL,
    wrong_claims INTEGER,
    total_claims INTEGER,
    extra_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, product_code, question_level, model, search_enabled),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics_recommendation (
    dataset_id TEXT NOT NULL,
    product_code TEXT NOT NULL,
    stage TEXT NOT NULL,
    question_level TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    search_enabled TEXT NOT NULL,
    rank INTEGER,
    rec_product TEXT NOT NULL,
    name_type TEXT NOT NULL DEFAULT '',
    mention_count REAL,
    mention_rate REAL,
    strong_count REAL,
    strong_rate REAL,
    moderate_count REAL,
    negative_count REAL,
    extra_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, product_code, question_level, model, search_enabled, rec_product, name_type),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metric_evidence (
    dataset_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    product_code TEXT,
    stage TEXT,
    question_level TEXT DEFAULT '',
    question_id TEXT NOT NULL,
    model TEXT NOT NULL,
    search_enabled INTEGER,
    round INTEGER,
    rec_product TEXT,
    name_type TEXT,
    rank INTEGER,
    strength TEXT,
    verdict TEXT,
    detail TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evidence_lookup
    ON metric_evidence(dataset_id, product_code, evidence_type, model, search_enabled);
CREATE INDEX IF NOT EXISTS idx_metrics_summary_product
    ON metrics_summary(product_code, stage, model);
CREATE INDEX IF NOT EXISTS idx_metrics_rec_lookup
    ON metrics_recommendation(dataset_id, product_code, question_level, model, search_enabled);
CREATE INDEX IF NOT EXISTS idx_dataset_products_product
    ON dataset_products(product_code, question_set_id);
"""

# 已有库的惰性加列（CREATE TABLE IF NOT EXISTS 对已存在的表不生效）
_LAZY_COLUMNS: dict[str, dict[str, str]] = {
    "datasets": {
        "owner_username": "TEXT",
        "product_code": "TEXT",
        "batch_date": "TEXT",
        "question_set_id": "TEXT",
    },
    "products": {
        "category": "TEXT",
        "aliases_json": "TEXT DEFAULT '[]'",
        "is_active": "INTEGER DEFAULT 1",
        "display_order": "INTEGER DEFAULT 0",
    },
    "metric_evidence": {
        "question_level": "TEXT DEFAULT ''",
    },
}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if not existing:
        return
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """建齐所有表并补齐新列，幂等，可在每次连接/启动时调用。"""
    conn.executescript(SCHEMA)
    conn.executescript(EXTENSION_SCHEMA)
    for table, columns in _LAZY_COLUMNS.items():
        _ensure_columns(conn, table, columns)
    conn.commit()


def ensure_schema_at(db_path: Path | str) -> None:
    """按路径打开（不存在则创建）并确保 schema。供 backend 启动使用。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_schema(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 消费者链路阶段映射（分析框架 V6：Q3/Q4=病症，Q5=品类，Q1/Q2=品牌）
# ---------------------------------------------------------------------------

STAGE_SYMPTOM = "symptom"
STAGE_CATEGORY = "category"
STAGE_BRAND = "brand"
STAGES = (STAGE_SYMPTOM, STAGE_CATEGORY, STAGE_BRAND)

_STAGE_KEYWORDS = (
    (STAGE_BRAND, ("negative", "负面", "安全", "品牌", "产品认知", "详情", "怎么样")),
    (STAGE_SYMPTOM, ("病症", "症状", "解决方案", "场景", "泛式")),
    (STAGE_CATEGORY, ("品类", "用药", "吃什么", "哪个牌子", "中药", "top3", "对比")),
)


def match_stage(raw: str | None) -> str | None:
    """单个层级标签 → 阶段；识别不出返回 None（供上传校验区分"未识别层级"）。"""
    text = (raw or "").strip().lower()
    if not text:
        return None
    if text.startswith(("q1", "q2")) or text.startswith("negative"):
        return STAGE_BRAND
    if text.startswith(("q3", "q4")):
        return STAGE_SYMPTOM
    if text.startswith("q5"):
        return STAGE_CATEGORY
    for stage, keywords in _STAGE_KEYWORDS:
        if any(k in text for k in keywords):
            return stage
    return None


def stage_for_level(level: str | None, source_level: str | None = None) -> str:
    """
    把问题层级（q1_overall / q4_scenario2 / 解决方案 / 品类层 …）映射到
    消费者链路三阶段。前缀规则优先（与 V6 框架一致），中文标签走关键词，
    匹配不到时默认 symptom（用户上传的泛式问题默认按病症层处理）。
    """
    for raw in (level, source_level):
        stage = match_stage(raw)
        if stage:
            return stage
    return STAGE_SYMPTOM
