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

CREATE TABLE IF NOT EXISTS outbound_articles (
    article_id TEXT PRIMARY KEY,
    owner_username TEXT NOT NULL,
    title TEXT NOT NULL,
    content_text TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    product_code TEXT,
    campaign TEXT,
    source_filename TEXT NOT NULL,
    file_ext TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS article_publications (
    publication_id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    url_match_key TEXT NOT NULL,
    published_at TEXT,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (article_id, url_match_key),
    FOREIGN KEY (article_id) REFERENCES outbound_articles(article_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_article_matches (
    dataset_id TEXT NOT NULL,
    answer_id TEXT NOT NULL,
    source_index INTEGER NOT NULL,
    publication_id TEXT NOT NULL,
    match_method TEXT NOT NULL,
    confidence REAL NOT NULL,
    matched_at TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, answer_id, source_index, publication_id),
    FOREIGN KEY (dataset_id, answer_id, source_index)
        REFERENCES sources(dataset_id, answer_id, source_index) ON DELETE CASCADE,
    FOREIGN KEY (publication_id) REFERENCES article_publications(publication_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outbound_articles_owner
    ON outbound_articles(owner_username, created_at);
CREATE INDEX IF NOT EXISTS idx_article_publications_url
    ON article_publications(url_match_key);
CREATE INDEX IF NOT EXISTS idx_source_article_matches_publication
    ON source_article_matches(publication_id, dataset_id, answer_id);

CREATE TABLE IF NOT EXISTS contributor_companies (
    company_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    contact_name TEXT,
    contact_email TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contributor_invites (
    invite_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    allowed_product_codes_json TEXT NOT NULL DEFAULT '[]',
    expires_at TEXT NOT NULL,
    max_submissions INTEGER NOT NULL DEFAULT 20,
    submission_count INTEGER NOT NULL DEFAULT 0,
    revoked_at TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    FOREIGN KEY (company_id) REFERENCES contributor_companies(company_id)
);

CREATE TABLE IF NOT EXISTS contributor_sessions (
    session_hash TEXT PRIMARY KEY,
    invite_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    FOREIGN KEY (invite_id) REFERENCES contributor_invites(invite_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS article_submissions (
    submission_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    invite_id TEXT NOT NULL,
    product_code TEXT NOT NULL,
    title TEXT NOT NULL,
    campaign TEXT,
    submitter_name TEXT NOT NULL,
    submitter_email TEXT NOT NULL,
    status TEXT NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 1,
    published_platform TEXT,
    published_url TEXT,
    published_at TEXT,
    article_id TEXT,
    admin_feedback TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    approved_by TEXT,
    approved_at TEXT,
    rejected_by TEXT,
    rejected_at TEXT,
    FOREIGN KEY (company_id) REFERENCES contributor_companies(company_id),
    FOREIGN KEY (invite_id) REFERENCES contributor_invites(invite_id),
    FOREIGN KEY (article_id) REFERENCES outbound_articles(article_id)
);

CREATE TABLE IF NOT EXISTS article_submission_versions (
    submission_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    file_ext TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    relative_path TEXT NOT NULL,
    content_text TEXT,
    content_sha256 TEXT,
    parse_error TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (submission_id, version),
    FOREIGN KEY (submission_id) REFERENCES article_submissions(submission_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS article_submission_events (
    event_id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    action TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (submission_id) REFERENCES article_submissions(submission_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS article_review_jobs (
    job_id TEXT PRIMARY KEY,
    submission_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER NOT NULL DEFAULT 0,
    settings_snapshot_json TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0,
    progress REAL NOT NULL DEFAULT 0,
    error_message TEXT,
    lease_owner TEXT,
    lease_expires_at TEXT,
    heartbeat_at TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE (submission_id, version),
    FOREIGN KEY (submission_id, version)
        REFERENCES article_submission_versions(submission_id, version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS article_review_reports (
    job_id TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    summary TEXT NOT NULL,
    model_key TEXT,
    model_id TEXT,
    model_version TEXT,
    prompt_version TEXT NOT NULL,
    knowledge_base_sha256 TEXT NOT NULL,
    config_snapshot_json TEXT NOT NULL DEFAULT '{}',
    structured_json TEXT NOT NULL DEFAULT '{}',
    raw_response TEXT,
    duration_ms INTEGER,
    retry_log_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES article_review_jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS article_review_findings (
    finding_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    excerpt TEXT,
    verdict TEXT NOT NULL,
    kb_module TEXT,
    evidence TEXT,
    suggestion TEXT,
    blocks_publication INTEGER NOT NULL DEFAULT 0,
    external_visible INTEGER NOT NULL DEFAULT 0,
    reviewer_note TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES article_review_jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS article_similarity_matches (
    match_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    matched_kind TEXT NOT NULL,
    matched_id TEXT NOT NULL,
    matched_version INTEGER,
    exact_hash INTEGER NOT NULL DEFAULT 0,
    lexical_score REAL NOT NULL DEFAULT 0,
    semantic_score REAL,
    similarity_level TEXT NOT NULL,
    overlap_summary TEXT,
    source_excerpt TEXT,
    matched_excerpt TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES article_review_jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS article_review_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    auto_start INTEGER NOT NULL DEFAULT 1,
    queue_paused INTEGER NOT NULL DEFAULT 0,
    primary_model_key TEXT,
    primary_model_id TEXT,
    fallback_model_key TEXT,
    fallback_model_id TEXT,
    ai_concurrency INTEGER NOT NULL DEFAULT 5,
    request_timeout_seconds INTEGER NOT NULL DEFAULT 120,
    retry_count INTEGER NOT NULL DEFAULT 2,
    similarity_threshold REAL NOT NULL DEFAULT 0.68,
    similarity_top_k INTEGER NOT NULL DEFAULT 10,
    updated_by TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS article_review_worker_state (
    worker_id TEXT PRIMARY KEY,
    hostname TEXT,
    pid INTEGER,
    configured_concurrency INTEGER NOT NULL,
    environment_max INTEGER NOT NULL,
    effective_concurrency INTEGER NOT NULL,
    active_requests INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    heartbeat_at TEXT NOT NULL,
    started_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_invites_company ON contributor_invites(company_id, created_at);
CREATE INDEX IF NOT EXISTS idx_submissions_queue ON article_submissions(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_submissions_company ON article_submissions(company_id, created_at);
CREATE INDEX IF NOT EXISTS idx_review_jobs_queue ON article_review_jobs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_review_findings_job ON article_review_findings(job_id, severity);
CREATE INDEX IF NOT EXISTS idx_similarity_job ON article_similarity_matches(job_id, lexical_score);
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

CREATE TABLE IF NOT EXISTS metrics_scenario (
    dataset_id TEXT NOT NULL,
    product_code TEXT NOT NULL,
    scenario TEXT NOT NULL,
    model TEXT NOT NULL,
    search_enabled TEXT NOT NULL,
    question_count INTEGER,
    total_answers INTEGER,
    brand_mention_count INTEGER,
    brand_mention_rate REAL,
    brand_rec_count INTEGER,
    brand_rec_rate REAL,
    generic_mention_count INTEGER,
    competitor_mention_count INTEGER,
    negative_count INTEGER,
    negative_rate REAL,
    top_categories_json TEXT NOT NULL DEFAULT '[]',
    extra_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, product_code, scenario, model, search_enabled),
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
CREATE INDEX IF NOT EXISTS idx_metrics_scenario_lookup
    ON metrics_scenario(dataset_id, product_code, scenario);
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
    "outbound_articles": {
        "company_id": "TEXT",
        "submission_id": "TEXT",
        "approved_by": "TEXT",
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
    conn.execute(
        """INSERT OR IGNORE INTO article_review_settings
        (id, updated_at) VALUES (1, datetime('now'))"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_outbound_articles_submission "
        "ON outbound_articles(submission_id)"
    )
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
