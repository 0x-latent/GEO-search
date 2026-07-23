"""Durable storage, anomaly detection, and queue operations for GEO investigations."""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta
from statistics import median
from typing import Any
from uuid import uuid4

from ..core.paths import DATA_DIR, GEO_SQLITE_PATH
from utils.sqlite_schema import stage_for_level


DB_PATH = DATA_DIR / "jobs.sqlite"
_QUEUE: "queue.Queue[str]" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()
_CANCEL_FLAGS: set[str] = set()
_WORKER_ID = f"{os.getpid()}-{uuid4().hex[:12]}"
_LEASE_SECONDS = max(
    180, int(os.environ.get("GEO_INVESTIGATION_LEASE_SECONDS", "600"))
)

RATE_METRICS: dict[str, dict[str, Any]] = {
    "category_mention_rate": {"label": "品类提及率", "higher_is_better": True},
    "brand_mention_rate": {"label": "品牌提及率", "higher_is_better": True},
    "brand_rec_rate": {"label": "品牌推荐率", "higher_is_better": True},
    "generic_mention_rate": {"label": "通用名提及率", "higher_is_better": True},
    "generic_rec_rate": {"label": "通用名推荐率", "higher_is_better": True},
    "competitor_mention_rate": {"label": "竞品提及率", "higher_is_better": False},
    "competitor_rec_rate": {"label": "竞品推荐率", "higher_is_better": False},
    "first_rate": {"label": "首推率", "higher_is_better": True},
    "top3_rate": {"label": "TOP3率", "higher_is_better": True},
    "negative_rate": {"label": "负向率", "higher_is_better": False},
    "accuracy_rate": {"label": "准确率", "higher_is_better": True},
}

RATE_METRICS.update({
    "official_coverage_rate": {"label": "官方信源覆盖率", "higher_is_better": True},
    "authority_coverage_rate": {"label": "权威信源覆盖率", "higher_is_better": True},
    "article_citation_rate": {"label": "文章引用率", "higher_is_better": True},
})
DERIVED_RATE_METRICS = {
    "official_coverage_rate", "authority_coverage_rate", "article_citation_rate"
}
SUMMARY_RATE_METRICS = set(RATE_METRICS) - DERIVED_RATE_METRICS


DEFAULT_SETTINGS = {
    "enabled": 1,
    "auto_start": 1,
    "primary_model_key": None,
    "primary_model_id": None,
    "fallback_model_key": None,
    "fallback_model_id": None,
    "max_reasoning_calls": 6,
    "max_probe_calls": 12,
    "max_web_fetches": 10,
    "max_iterations": 4,
    "max_auto_cases": 3,
    "request_timeout_seconds": 120,
}

HARD_LIMITS = {
    "max_reasoning_calls": max(1, int(os.environ.get("GEO_INVESTIGATION_MAX_REASONING_CALLS", "6"))),
    "max_probe_calls": max(0, int(os.environ.get("GEO_INVESTIGATION_MAX_PROBE_CALLS", "12"))),
    "max_web_fetches": max(0, int(os.environ.get("GEO_INVESTIGATION_MAX_WEB_FETCHES", "10"))),
    "max_iterations": max(1, int(os.environ.get("GEO_INVESTIGATION_MAX_ITERATIONS", "4"))),
    "max_auto_cases": max(0, int(os.environ.get("GEO_INVESTIGATION_MAX_AUTO_CASES", "3"))),
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _lease_until() -> str:
    return (datetime.now() + timedelta(seconds=_LEASE_SECONDS)).isoformat(
        timespec="seconds"
    )


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _geo_connect() -> sqlite3.Connection:
    if not GEO_SQLITE_PATH.exists():
        raise FileNotFoundError(f"GEO SQLite 不存在: {GEO_SQLITE_PATH}")
    conn = sqlite3.connect(GEO_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS investigation_settings (
        id INTEGER PRIMARY KEY CHECK (id=1),
        enabled INTEGER NOT NULL DEFAULT 1,
        auto_start INTEGER NOT NULL DEFAULT 1,
        primary_model_key TEXT,
        primary_model_id TEXT,
        fallback_model_key TEXT,
        fallback_model_id TEXT,
        max_reasoning_calls INTEGER NOT NULL DEFAULT 6,
        max_probe_calls INTEGER NOT NULL DEFAULT 12,
        max_web_fetches INTEGER NOT NULL DEFAULT 10,
        max_iterations INTEGER NOT NULL DEFAULT 4,
        max_auto_cases INTEGER NOT NULL DEFAULT 3,
        request_timeout_seconds INTEGER NOT NULL DEFAULT 120,
        updated_by TEXT,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS investigation_targets (
        target_id TEXT PRIMARY KEY,
        owner_username TEXT NOT NULL,
        product_code TEXT NOT NULL,
        metric TEXT NOT NULL,
        stage TEXT,
        model TEXT,
        search_enabled INTEGER,
        operator TEXT NOT NULL DEFAULT 'gte',
        target_value REAL NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS investigation_scans (
        scan_id TEXT PRIMARY KEY,
        owner_username TEXT NOT NULL,
        current_dataset_id TEXT NOT NULL,
        baseline_dataset_id TEXT,
        status TEXT NOT NULL,
        candidate_count INTEGER NOT NULL DEFAULT 0,
        error_message TEXT,
        created_at TEXT NOT NULL,
        finished_at TEXT
    );

    CREATE TABLE IF NOT EXISTS investigations (
        investigation_id TEXT PRIMARY KEY,
        scan_id TEXT,
        owner_username TEXT NOT NULL,
        trigger_type TEXT NOT NULL,
        current_dataset_id TEXT NOT NULL,
        baseline_dataset_id TEXT,
        product_code TEXT NOT NULL,
        metric TEXT NOT NULL,
        stage_filter TEXT,
        model_filter TEXT,
        search_enabled_filter INTEGER,
        expected_value REAL,
        direction TEXT,
        severity TEXT NOT NULL DEFAULT 'medium',
        status TEXT NOT NULL DEFAULT 'candidate',
        stage TEXT NOT NULL DEFAULT 'signal_validation',
        progress REAL NOT NULL DEFAULT 0,
        signal_json TEXT NOT NULL DEFAULT '{}',
        budget_json TEXT NOT NULL DEFAULT '{}',
        conclusion_json TEXT,
        error_message TEXT,
        cancellation_requested INTEGER NOT NULL DEFAULT 0,
        worker_owner TEXT,
        lease_expires_at TEXT,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        updated_at TEXT NOT NULL,
        UNIQUE (
            current_dataset_id, baseline_dataset_id, product_code, metric,
            stage_filter, model_filter, search_enabled_filter, trigger_type
        ),
        FOREIGN KEY (scan_id) REFERENCES investigation_scans(scan_id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS investigation_hypotheses (
        hypothesis_id TEXT PRIMARY KEY,
        investigation_id TEXT NOT NULL,
        category TEXT NOT NULL,
        statement TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        confidence REAL NOT NULL DEFAULT 0,
        supporting_evidence_json TEXT NOT NULL DEFAULT '[]',
        opposing_evidence_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS investigation_tool_calls (
        call_id TEXT PRIMARY KEY,
        investigation_id TEXT NOT NULL,
        iteration INTEGER NOT NULL,
        tool_name TEXT NOT NULL,
        arguments_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL,
        result_json TEXT,
        error_message TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS investigation_evidence (
        evidence_id TEXT PRIMARY KEY,
        investigation_id TEXT NOT NULL,
        evidence_type TEXT NOT NULL,
        title TEXT NOT NULL,
        source_ref TEXT,
        summary TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS investigation_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        investigation_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL,
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_investigations_owner
        ON investigations(owner_username, created_at);
    CREATE INDEX IF NOT EXISTS idx_investigations_queue
        ON investigations(status, severity, created_at);
    CREATE INDEX IF NOT EXISTS idx_investigation_events
        ON investigation_events(investigation_id, event_id);
    CREATE INDEX IF NOT EXISTS idx_investigation_targets
        ON investigation_targets(owner_username, product_code, metric);
    """
    with _connect() as conn:
        conn.executescript(schema)
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(investigations)")
        }
        for name, ddl in {
            "worker_owner": "TEXT",
            "lease_expires_at": "TEXT",
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            if name not in existing_columns:
                conn.execute(f"ALTER TABLE investigations ADD COLUMN {name} {ddl}")
        conn.execute(
            """INSERT OR IGNORE INTO investigation_settings (
               id, enabled, auto_start, max_reasoning_calls, max_probe_calls,
               max_web_fetches, max_iterations, max_auto_cases,
               request_timeout_seconds, updated_at
               ) VALUES (1,1,1,6,12,10,4,3,120,?)""",
            (_now(),),
        )
        # Legacy rows have no lease. A valid future lease may belong to another
        # process, so only unleased/expired work is recovered.
        conn.execute(
            """UPDATE investigations
               SET status='queued', stage='signal_validation', error_message=NULL,
                   cancellation_requested=0,worker_owner=NULL,
                   lease_expires_at=NULL,updated_at=?
               WHERE status='running'
                 AND (lease_expires_at IS NULL OR lease_expires_at<?)""",
            (_now(), _now()),
        )


def get_settings() -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM investigation_settings WHERE id=1").fetchone()
    settings = dict(row) if row else dict(DEFAULT_SETTINGS)
    for key, limit in HARD_LIMITS.items():
        settings[key] = min(int(settings.get(key) or 0), limit)
    settings["hard_limits"] = dict(HARD_LIMITS)
    return settings


def update_settings(values: dict[str, Any], updated_by: str) -> dict[str, Any]:
    allowed = set(DEFAULT_SETTINGS)
    updates = {key: value for key, value in values.items() if key in allowed}
    for key, limit in HARD_LIMITS.items():
        if key in updates:
            if updates[key] is None:
                updates.pop(key)
                continue
            updates[key] = min(max(0, int(updates[key])), limit)
    for key in ("enabled", "auto_start"):
        if key in updates:
            if updates[key] is None:
                updates.pop(key)
                continue
            updates[key] = int(bool(updates[key]))
    updates["updated_by"] = updated_by
    updates["updated_at"] = _now()
    clause = ", ".join(f"{key}=?" for key in updates)
    with _connect() as conn:
        conn.execute(
            f"UPDATE investigation_settings SET {clause} WHERE id=1",
            tuple(updates.values()),
        )
    return get_settings()


def list_targets(username: str | None = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        if username is None:
            rows = conn.execute(
                "SELECT * FROM investigation_targets ORDER BY updated_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM investigation_targets
                   WHERE owner_username=? ORDER BY updated_at DESC""",
                (username,),
            ).fetchall()
    return [dict(row) for row in rows]


def create_target(owner_username: str, values: dict[str, Any]) -> dict[str, Any]:
    if not str(values.get("product_code") or "").strip():
        raise ValueError("product_code 必填")
    metric = str(values.get("metric") or "")
    if metric not in RATE_METRICS:
        raise ValueError(f"不支持的监测指标: {metric}")
    operator = str(values.get("operator") or "gte")
    if operator not in {"gte", "lte"}:
        raise ValueError("operator 仅支持 gte/lte")
    value = float(values["target_value"])
    if value < 0 or value > 1:
        raise ValueError("target_value 必须在 0-1 之间")
    target_id = f"itgt_{uuid4().hex}"
    now = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO investigation_targets (
               target_id,owner_username,product_code,metric,stage,model,
               search_enabled,operator,target_value,is_active,created_at,updated_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                target_id, owner_username, str(values.get("product_code") or ""),
                metric, values.get("stage"), values.get("model"),
                values.get("search_enabled"), operator, value,
                int(bool(values.get("is_active", True))), now, now,
            ),
        )
    return get_target(target_id) or {}


def get_target(target_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM investigation_targets WHERE target_id=?", (target_id,)
        ).fetchone()
    return dict(row) if row else None


def update_target(target_id: str, values: dict[str, Any]) -> dict[str, Any]:
    current = get_target(target_id)
    if current is None:
        raise ValueError("监测目标不存在")
    allowed = {
        "product_code", "metric", "stage", "model", "search_enabled",
        "operator", "target_value", "is_active",
    }
    updates = {key: value for key, value in values.items() if key in allowed}
    for key in {"product_code", "metric", "operator", "target_value", "is_active"}:
        if key in updates and updates[key] is None:
            raise ValueError(f"{key} 不能设为 null")
    if "metric" in updates and updates["metric"] not in RATE_METRICS:
        raise ValueError("不支持的监测指标")
    if "operator" in updates and updates["operator"] not in {"gte", "lte"}:
        raise ValueError("operator 仅支持 gte/lte")
    if "target_value" in updates:
        updates["target_value"] = float(updates["target_value"])
        if not 0 <= updates["target_value"] <= 1:
            raise ValueError("target_value 必须在 0-1 之间")
    if "is_active" in updates:
        updates["is_active"] = int(bool(updates["is_active"]))
    if not updates:
        return current
    updates["updated_at"] = _now()
    clause = ", ".join(f"{key}=?" for key in updates)
    with _connect() as conn:
        conn.execute(
            f"UPDATE investigation_targets SET {clause} WHERE target_id=?",
            (*updates.values(), target_id),
        )
    return get_target(target_id) or {}


def delete_target(target_id: str) -> None:
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM investigation_targets WHERE target_id=?", (target_id,)
        )
        if cursor.rowcount == 0:
            raise ValueError("监测目标不存在")


def _decode_investigation(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["signal"] = _loads(item.pop("signal_json", None), {})
    item["budget"] = _loads(item.pop("budget_json", None), {})
    item["conclusion"] = _loads(item.pop("conclusion_json", None), None)
    return item


def get_investigation(investigation_id: str, include_details: bool = True) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM investigations WHERE investigation_id=?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            return None
        item = _decode_investigation(row)
        if include_details:
            hypotheses = conn.execute(
                """SELECT * FROM investigation_hypotheses
                   WHERE investigation_id=? ORDER BY confidence DESC,created_at""",
                (investigation_id,),
            ).fetchall()
            item["hypotheses"] = []
            for hypothesis in hypotheses:
                value = dict(hypothesis)
                value["supporting_evidence"] = _loads(
                    value.pop("supporting_evidence_json", None), []
                )
                value["opposing_evidence"] = _loads(
                    value.pop("opposing_evidence_json", None), []
                )
                item["hypotheses"].append(value)
            item["tool_calls"] = [
                {
                    **dict(call),
                    "arguments": _loads(call["arguments_json"], {}),
                    "result": _loads(call["result_json"], None),
                }
                for call in conn.execute(
                    """SELECT * FROM investigation_tool_calls
                       WHERE investigation_id=? ORDER BY started_at,call_id""",
                    (investigation_id,),
                ).fetchall()
            ]
            for call in item["tool_calls"]:
                call.pop("arguments_json", None)
                call.pop("result_json", None)
            item["evidence"] = [
                {
                    **dict(evidence),
                    "payload": _loads(evidence["payload_json"], {}),
                }
                for evidence in conn.execute(
                    """SELECT * FROM investigation_evidence
                       WHERE investigation_id=? ORDER BY created_at,evidence_id""",
                    (investigation_id,),
                ).fetchall()
            ]
            for evidence in item["evidence"]:
                evidence.pop("payload_json", None)
    return item


def list_investigations(
    username: str | None = None,
    allowed_dataset_ids: list[str] | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    conditions = ["1=1"]
    params: list[Any] = []
    if username is not None:
        conditions.append("owner_username=?")
        params.append(username)
    if allowed_dataset_ids is not None:
        if not allowed_dataset_ids:
            return []
        conditions.append(
            f"current_dataset_id IN ({','.join('?' for _ in allowed_dataset_ids)})"
        )
        params.extend(allowed_dataset_ids)
    if status:
        conditions.append("status=?")
        params.append(status)
    params.append(min(max(1, limit), 500))
    with _connect() as conn:
        rows = conn.execute(
            f"""SELECT * FROM investigations
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()
    return [_decode_investigation(row) for row in rows]


def add_event(
    investigation_id: str, event_type: str, message: str,
    details: dict[str, Any] | None = None,
) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            """INSERT INTO investigation_events (
               investigation_id,event_type,message,details_json,created_at
               ) VALUES (?,?,?,?,?)""",
            (investigation_id, event_type, message, _json(details or {}), _now()),
        )
    return int(cursor.lastrowid)


def list_events(investigation_id: str, after: int = 0) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM investigation_events
               WHERE investigation_id=? AND event_id>?
               ORDER BY event_id LIMIT 500""",
            (investigation_id, after),
        ).fetchall()
    return [
        {
            **dict(row),
            "details": _loads(row["details_json"], {}),
        }
        for row in rows
    ]


def update_investigation(investigation_id: str, **updates: Any) -> None:
    if "signal" in updates:
        updates["signal_json"] = _json(updates.pop("signal"))
    if "budget" in updates:
        updates["budget_json"] = _json(updates.pop("budget"))
    if "conclusion" in updates:
        updates["conclusion_json"] = _json(updates.pop("conclusion"))
    if updates.get("status") == "running":
        updates.setdefault("worker_owner", _WORKER_ID)
        updates.setdefault("lease_expires_at", _lease_until())
    elif updates.get("status") in {
        "candidate", "queued", "needs_review", "completed", "failed", "cancelled"
    }:
        updates.setdefault("worker_owner", None)
        updates.setdefault("lease_expires_at", None)
    updates["updated_at"] = _now()
    clause = ", ".join(f"{key}=?" for key in updates)
    with _connect() as conn:
        conn.execute(
            f"UPDATE investigations SET {clause} WHERE investigation_id=?",
            (*updates.values(), investigation_id),
        )


def renew_lease(investigation_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            """UPDATE investigations
               SET lease_expires_at=?,updated_at=?
               WHERE investigation_id=? AND status='running'
                 AND worker_owner=?""",
            (_lease_until(), _now(), investigation_id, _WORKER_ID),
        )


def _recover_expired_investigations() -> list[str]:
    now = _now()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """SELECT investigation_id FROM investigations
               WHERE status='running'
                 AND (lease_expires_at IS NULL OR lease_expires_at<?)""",
            (now,),
        ).fetchall()
        ids = [row["investigation_id"] for row in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"""UPDATE investigations
                    SET status='queued',stage='signal_validation',
                        cancellation_requested=0,worker_owner=NULL,
                        lease_expires_at=NULL,error_message=NULL,updated_at=?
                    WHERE investigation_id IN ({placeholders})""",
                (now, *ids),
            )
        conn.commit()
    return ids


def _claim_investigation(investigation_id: str) -> bool:
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """UPDATE investigations
               SET status='running',worker_owner=?,lease_expires_at=?,
                   attempt_count=attempt_count+1,
                   started_at=COALESCE(started_at,?),updated_at=?
               WHERE investigation_id=? AND status='queued'""",
            (
                _WORKER_ID, _lease_until(), _now(), _now(),
                investigation_id,
            ),
        )
        conn.commit()
    return cursor.rowcount == 1


def add_evidence(
    investigation_id: str, evidence_type: str, title: str, summary: str,
    payload: dict[str, Any], source_ref: str | None = None,
) -> str:
    evidence_id = f"iev_{uuid4().hex}"
    with _connect() as conn:
        conn.execute(
            """INSERT INTO investigation_evidence (
               evidence_id,investigation_id,evidence_type,title,source_ref,
               summary,payload_json,created_at
               ) VALUES (?,?,?,?,?,?,?,?)""",
            (
                evidence_id, investigation_id, evidence_type, title, source_ref,
                summary[:4000], _json(payload), _now(),
            ),
        )
    return evidence_id


def start_tool_call(
    investigation_id: str, iteration: int, tool_name: str, arguments: dict[str, Any]
) -> str:
    call_id = f"itc_{uuid4().hex}"
    with _connect() as conn:
        conn.execute(
            """INSERT INTO investigation_tool_calls (
               call_id,investigation_id,iteration,tool_name,arguments_json,
               status,started_at
               ) VALUES (?,?,?,?,?,'running',?)""",
            (call_id, investigation_id, iteration, tool_name, _json(arguments), _now()),
        )
    return call_id


def finish_tool_call(
    call_id: str, result: dict[str, Any] | None = None, error: str | None = None
) -> None:
    with _connect() as conn:
        conn.execute(
            """UPDATE investigation_tool_calls
               SET status=?,result_json=?,error_message=?,finished_at=?
               WHERE call_id=?""",
            (
                "failed" if error else "success",
                _json(result) if result is not None else None,
                error[:2000] if error else None,
                _now(), call_id,
            ),
        )


def replace_hypotheses(investigation_id: str, values: list[dict[str, Any]]) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM investigation_hypotheses WHERE investigation_id=?",
            (investigation_id,),
        )
        for value in values[:20]:
            conn.execute(
                """INSERT INTO investigation_hypotheses (
                   hypothesis_id,investigation_id,category,statement,status,
                   confidence,supporting_evidence_json,opposing_evidence_json,
                   created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"ihyp_{uuid4().hex}", investigation_id,
                    str(value.get("category") or "unknown")[:80],
                    str(value.get("statement") or "")[:4000],
                    str(value.get("status") or "active")[:30],
                    min(1, max(0, float(value.get("confidence") or 0))),
                    _json(value.get("supporting_evidence") or []),
                    _json(value.get("opposing_evidence") or []),
                    now, now,
                ),
            )


def _dataset_info(conn: sqlite3.Connection, dataset_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT dataset_id,name,owner_username,product_code,batch_date,
                  question_set_id,imported_at
           FROM datasets WHERE dataset_id=?""",
        (dataset_id,),
    ).fetchone()
    return dict(row) if row else None


def dataset_owner(dataset_id: str) -> str | None:
    with closing(_geo_connect()) as conn:
        info = _dataset_info(conn, dataset_id)
    return info.get("owner_username") if info else None


def _product_question_count(
    conn: sqlite3.Connection, dataset_id: str, product_code: str
) -> int:
    row = conn.execute(
        """SELECT COUNT(DISTINCT question_id)
           FROM questions WHERE dataset_id=? AND product_code=?""",
        (dataset_id, product_code),
    ).fetchone()
    return int(row[0] or 0)


def _slice_question_count(
    conn: sqlite3.Connection,
    dataset_id: str,
    product_code: str,
    stage: str | None,
    model: str | None,
    search_enabled: str | int | None,
) -> int:
    conditions = ["a.dataset_id=?", "a.product_code=?"]
    params: list[Any] = [dataset_id, product_code]
    if model:
        conditions.append("a.model=?")
        params.append(model)
    if search_enabled is not None and str(search_enabled) in {"0", "1"}:
        conditions.append("a.search_enabled=?")
        params.append(int(search_enabled))
    rows = conn.execute(
        f"""SELECT DISTINCT q.question_id,q.level,q.source_level
            FROM answers a
            JOIN questions q
              ON q.dataset_id=a.dataset_id AND q.question_id=a.question_id
            WHERE {' AND '.join(conditions)}""",
        params,
    ).fetchall()
    if not stage:
        return len(rows)
    return sum(
        1 for row in rows
        if stage_for_level(row["level"], row["source_level"]) == stage
    )


def select_baseline(
    current_dataset_id: str, product_code: str, explicit: str | None = None
) -> str | None:
    with closing(_geo_connect()) as conn:
        current = _dataset_info(conn, current_dataset_id)
        if current is None:
            raise ValueError("当前数据集不存在")
        if explicit:
            exists = conn.execute(
                """SELECT 1 FROM dataset_products
                   WHERE dataset_id=? AND product_code=?""",
                (explicit, product_code),
            ).fetchone()
            if not exists:
                raise ValueError("基线数据集不包含该产品")
            return explicit
        question_set_id = current.get("question_set_id")
        if not question_set_id:
            row = conn.execute(
                """SELECT question_set_id FROM dataset_products
                   WHERE dataset_id=? AND product_code=?""",
                (current_dataset_id, product_code),
            ).fetchone()
            question_set_id = row[0] if row else None
        if not question_set_id:
            return None
        row = conn.execute(
            """SELECT dp.dataset_id
               FROM dataset_products dp
               JOIN datasets d ON d.dataset_id=dp.dataset_id
               WHERE dp.product_code=? AND dp.question_set_id=?
                 AND dp.dataset_id<>?
                 AND COALESCE(d.batch_date,d.imported_at,'') <
                     COALESCE(?, ?, '9999')
               ORDER BY COALESCE(d.batch_date,d.imported_at,'') DESC
               LIMIT 1""",
            (
                product_code, question_set_id, current_dataset_id,
                current.get("batch_date"), current.get("imported_at"),
            ),
        ).fetchone()
        return row[0] if row else None


def _metric_rows(
    conn: sqlite3.Connection, dataset_id: str, product_code: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT stage,question_level,model,search_enabled,total_answers,
                  category_mention_rate,brand_mention_rate,brand_rec_rate,
                  generic_mention_rate,generic_rec_rate,
                  competitor_mention_rate,competitor_rec_rate,
                  first_rate,top3_rate,negative_rate,accuracy_rate
           FROM metrics_summary
           WHERE dataset_id=? AND product_code=?""",
        (dataset_id, product_code),
    ).fetchall()
    return [dict(row) for row in rows]


def _target_for(
    targets: list[dict[str, Any]], product_code: str, metric: str,
    stage: str | None, model: str | None, search_enabled: str | None,
) -> dict[str, Any] | None:
    matches = []
    for target in targets:
        if not target["is_active"] or target["product_code"] != product_code:
            continue
        if target["metric"] != metric:
            continue
        if target["stage"] and target["stage"] != stage:
            continue
        if target["model"] and target["model"] != model:
            continue
        if (
            target["search_enabled"] is not None
            and str(target["search_enabled"]) != str(search_enabled)
        ):
            continue
        specificity = sum(
            value is not None and value != ""
            for value in (target["stage"], target["model"], target["search_enabled"])
        )
        matches.append((specificity, target))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def _candidate_signal(
    metric: str, current_value: float, baseline_value: float | None,
    target: dict[str, Any] | None, current_n: int, baseline_n: int,
) -> dict[str, Any] | None:
    spec = RATE_METRICS[metric]
    delta = None if baseline_value is None else current_value - baseline_value
    regression = False
    improvement = False
    if delta is not None:
        regression = delta <= -0.15 if spec["higher_is_better"] else delta >= 0.15
        improvement = delta >= 0.15 if spec["higher_is_better"] else delta <= -0.15
    missed_target = False
    if target:
        missed_target = (
            current_value < target["target_value"]
            if target["operator"] == "gte"
            else current_value > target["target_value"]
        )
    if not (regression or improvement or missed_target):
        return None
    magnitude = abs(delta or (current_value - float(target["target_value"])))
    direction = "regression" if regression or missed_target else "improvement"
    severity = "high" if magnitude >= 0.30 or (
        metric in {"negative_rate", "accuracy_rate"} and magnitude >= 0.20
    ) else "medium"
    return {
        "metric": metric,
        "metric_label": spec["label"],
        "current_value": current_value,
        "baseline_value": baseline_value,
        "delta": delta,
        "current_n": current_n,
        "baseline_n": baseline_n,
        "direction": direction,
        "severity": severity,
        "target": target,
        "reason": "target_missed" if missed_target else "batch_change",
    }


def _insert_investigation(
    *, owner_username: str, trigger_type: str, current_dataset_id: str,
    baseline_dataset_id: str | None, product_code: str, metric: str,
    stage: str | None, model: str | None, search_enabled: int | None,
    expected_value: float | None, signal: dict[str, Any], scan_id: str | None = None,
    status: str = "candidate",
) -> dict[str, Any]:
    investigation_id = f"inv_{uuid4().hex}"
    settings = get_settings()
    budget = {
        key: settings[key] for key in (
            "max_reasoning_calls", "max_probe_calls", "max_web_fetches", "max_iterations"
        )
    }
    now = _now()
    try:
        with _connect() as conn:
            conn.execute(
                """INSERT INTO investigations (
                   investigation_id,scan_id,owner_username,trigger_type,
                   current_dataset_id,baseline_dataset_id,product_code,metric,
                   stage_filter,model_filter,search_enabled_filter,expected_value,
                   direction,severity,status,stage,progress,signal_json,budget_json,
                   created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    investigation_id, scan_id, owner_username, trigger_type,
                    current_dataset_id, baseline_dataset_id, product_code, metric,
                    stage, model, search_enabled, expected_value,
                    signal.get("direction"), signal.get("severity", "medium"),
                    status, "signal_validation", 0, _json(signal), _json(budget),
                    now, now,
                ),
            )
    except sqlite3.IntegrityError:
        with _connect() as conn:
            row = conn.execute(
                """SELECT * FROM investigations
                   WHERE current_dataset_id=? AND baseline_dataset_id IS ?
                     AND product_code=? AND metric=? AND stage_filter IS ?
                     AND model_filter IS ? AND search_enabled_filter IS ?
                     AND trigger_type=?""",
                (
                    current_dataset_id, baseline_dataset_id, product_code, metric,
                    stage, model, search_enabled, trigger_type,
                ),
            ).fetchone()
        return _decode_investigation(row)
    add_event(investigation_id, "created", "调查已创建", {"signal": signal})
    return get_investigation(investigation_id, include_details=False) or {}


def create_manual_investigation(owner_username: str, values: dict[str, Any]) -> dict[str, Any]:
    metric = str(values.get("metric") or "")
    if metric not in RATE_METRICS:
        raise ValueError(f"手动调查暂不支持指标: {metric}")
    current_dataset_id = str(values.get("current_dataset_id") or "")
    product_code = str(values.get("product_code") or "")
    if not current_dataset_id or not product_code:
        raise ValueError("current_dataset_id 和 product_code 必填")
    baseline = select_baseline(
        current_dataset_id, product_code, values.get("baseline_dataset_id")
    )
    if not baseline:
        raise ValueError("找不到可比基线数据集，请显式指定 baseline_dataset_id")
    stage = values.get("stage")
    model = values.get("model")
    search_enabled = values.get("search_enabled")
    with closing(_geo_connect()) as conn:
        current_rows = _metric_rows(conn, current_dataset_id, product_code)
        baseline_rows = _metric_rows(conn, baseline, product_code)
    if metric in DERIVED_RATE_METRICS:
        from . import investigation_tools
        current_rates = investigation_tools.derived_rates(
            current_dataset_id, product_code, stage, model, search_enabled
        )
        baseline_rates = investigation_tools.derived_rates(
            baseline, product_code, stage, model, search_enabled
        )
        current_raw = current_rates.get(metric)
        baseline_raw = baseline_rates.get(metric)
        if current_raw is None:
            raise ValueError("当前数据集没有可计算的派生指标")
        current_value = float(current_raw)
        baseline_value = float(baseline_raw) if baseline_raw is not None else None
        current_n = int(current_rates.get("sample_sizes", {}).get(metric) or 0)
        baseline_n = int(baseline_rates.get("sample_sizes", {}).get(metric) or 0)
    else:
        def find(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
            for row in rows:
                if stage and row["stage"] != stage:
                    continue
                if model and row["model"] != model:
                    continue
                if (
                    search_enabled is not None
                    and str(row["search_enabled"]) != str(search_enabled)
                ):
                    continue
                if row.get(metric) is not None:
                    return row
            return None
        current_row, baseline_row = find(current_rows), find(baseline_rows)
        if not current_row:
            raise ValueError("当前数据集没有匹配的指标")
        current_value = float(current_row[metric])
        baseline_value = (
            float(baseline_row[metric])
            if baseline_row and baseline_row[metric] is not None else None
        )
        current_n = int(current_row.get("total_answers") or 0)
        baseline_n = int((baseline_row or {}).get("total_answers") or 0)
    expected = values.get("expected_value")
    signal = {
        "metric": metric,
        "metric_label": RATE_METRICS[metric]["label"],
        "current_value": current_value,
        "baseline_value": baseline_value,
        "delta": current_value - baseline_value if baseline_value is not None else None,
        "current_n": current_n,
        "baseline_n": baseline_n,
        "direction": "manual",
        "severity": str(values.get("severity") or "medium"),
        "expected_value": expected,
        "reason": "manual",
    }
    item = _insert_investigation(
        owner_username=owner_username, trigger_type="manual",
        current_dataset_id=current_dataset_id, baseline_dataset_id=baseline,
        product_code=product_code, metric=metric, stage=stage, model=model,
        search_enabled=search_enabled, expected_value=expected, signal=signal,
        status="queued" if values.get("auto_start", True) else "candidate",
    )
    if item["status"] == "queued":
        enqueue(item["investigation_id"])
    return item


def scan_dataset(
    current_dataset_id: str, owner_username: str | None = None,
    baseline_dataset_id: str | None = None, auto_start: bool | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    owner = owner_username or dataset_owner(current_dataset_id) or "__system__"
    scan_id = f"iscan_{uuid4().hex}"
    now = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO investigation_scans (
               scan_id,owner_username,current_dataset_id,baseline_dataset_id,
               status,created_at
               ) VALUES (?,?,?,?,'running',?)""",
            (scan_id, owner, current_dataset_id, baseline_dataset_id, now),
        )
    created: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        with closing(_geo_connect()) as conn:
            current = _dataset_info(conn, current_dataset_id)
            if current is None:
                raise ValueError("当前数据集不存在")
            products = [
                row[0] for row in conn.execute(
                    "SELECT product_code FROM dataset_products WHERE dataset_id=?",
                    (current_dataset_id,),
                ).fetchall()
            ]
            targets = [] if owner == "__system__" else list_targets(owner)
            for product_code in products:
                baseline = select_baseline(current_dataset_id, product_code, baseline_dataset_id)
                if not baseline:
                    continue
                if _product_question_count(conn, current_dataset_id, product_code) < 5:
                    continue
                if _product_question_count(conn, baseline, product_code) < 5:
                    continue
                current_rows = _metric_rows(conn, current_dataset_id, product_code)
                baseline_rows = _metric_rows(conn, baseline, product_code)
                baseline_map = {
                    (
                        row["stage"], row["question_level"], row["model"],
                        str(row["search_enabled"]),
                    ): row
                    for row in baseline_rows
                }
                question_count_cache: dict[
                    tuple[str, str | None, str | None, str], int
                ] = {}

                def slice_question_count(
                    dataset_id: str, stage: str | None,
                    model: str | None, search_mode: str | int | None,
                ) -> int:
                    cache_key = (dataset_id, stage, model, str(search_mode))
                    if cache_key not in question_count_cache:
                        question_count_cache[cache_key] = _slice_question_count(
                            conn, dataset_id, product_code, stage, model, search_mode
                        )
                    return question_count_cache[cache_key]

                for row in current_rows:
                    if (
                        slice_question_count(
                            current_dataset_id, row["stage"], row["model"],
                            row["search_enabled"],
                        ) < 5
                        or slice_question_count(
                            baseline, row["stage"], row["model"],
                            row["search_enabled"],
                        ) < 5
                    ):
                        continue
                    key = (
                        row["stage"], row["question_level"], row["model"],
                        str(row["search_enabled"]),
                    )
                    old = baseline_map.get(key)
                    current_n = int(row.get("total_answers") or 0)
                    baseline_n = int((old or {}).get("total_answers") or 0)
                    if current_n < 10 or baseline_n < 10:
                        continue
                    for metric in SUMMARY_RATE_METRICS:
                        if row.get(metric) is None:
                            continue
                        current_value = float(row[metric])
                        baseline_value = (
                            float(old[metric])
                            if old and old.get(metric) is not None else None
                        )
                        target = _target_for(
                            targets, product_code, metric, row["stage"],
                            row["model"], str(row["search_enabled"]),
                        )
                        signal = _candidate_signal(
                            metric, current_value, baseline_value, target,
                            current_n, baseline_n,
                        )
                        if signal is None:
                            continue
                        signal["question_level"] = row["question_level"]
                        item = _insert_investigation(
                            owner_username=owner, trigger_type="auto",
                            current_dataset_id=current_dataset_id,
                            baseline_dataset_id=baseline, product_code=product_code,
                            metric=metric, stage=row["stage"], model=row["model"],
                            search_enabled=(
                                int(row["search_enabled"])
                                if str(row["search_enabled"]) in {"0", "1"} else None
                            ),
                            expected_value=(
                                float(target["target_value"]) if target else None
                            ),
                            signal=signal, scan_id=scan_id, status="candidate",
                        )
                        created.append(item)

                # Source and outbound-article rates are derived from answer/source
                # evidence rather than stored as columns in metrics_summary.
                from . import investigation_tools
                seen_slices: set[tuple[str | None, str | None, str]] = set()
                for row in current_rows:
                    search_mode = str(row["search_enabled"])
                    slice_key = (row["stage"], row["model"], search_mode)
                    if slice_key in seen_slices:
                        continue
                    seen_slices.add(slice_key)
                    search_value = int(search_mode) if search_mode in {"0", "1"} else None
                    if (
                        slice_question_count(
                            current_dataset_id, row["stage"], row["model"], search_value
                        ) < 5
                        or slice_question_count(
                            baseline, row["stage"], row["model"], search_value
                        ) < 5
                    ):
                        continue
                    try:
                        current_rates = investigation_tools.derived_rates(
                            current_dataset_id, product_code, row["stage"], row["model"],
                            search_value,
                        )
                        baseline_rates = investigation_tools.derived_rates(
                            baseline, product_code, row["stage"], row["model"], search_value,
                        )
                    except Exception as exc:  # derived rates must not hide core signals
                        warning = (
                            f"派生信源指标计算失败 "
                            f"({product_code}/{row['model']}/{search_mode}): {exc}"
                        )
                        if warning not in warnings:
                            warnings.append(warning[:1000])
                        continue
                    for metric in DERIVED_RATE_METRICS:
                        current_value = current_rates.get(metric)
                        baseline_value = baseline_rates.get(metric)
                        current_n = int(
                            current_rates.get("sample_sizes", {}).get(metric) or 0
                        )
                        baseline_n = int(
                            baseline_rates.get("sample_sizes", {}).get(metric) or 0
                        )
                        if (
                            current_value is None or baseline_value is None
                            or current_n < 10 or baseline_n < 10
                        ):
                            continue
                        target = _target_for(
                            targets, product_code, metric, row["stage"],
                            row["model"], search_mode,
                        )
                        signal = _candidate_signal(
                            metric, float(current_value), float(baseline_value),
                            target, current_n, baseline_n,
                        )
                        if signal is None:
                            continue
                        signal["question_level"] = row["question_level"]
                        item = _insert_investigation(
                            owner_username=owner, trigger_type="auto",
                            current_dataset_id=current_dataset_id,
                            baseline_dataset_id=baseline, product_code=product_code,
                            metric=metric, stage=row["stage"], model=row["model"],
                            search_enabled=search_value,
                            expected_value=(
                                float(target["target_value"]) if target else None
                            ),
                            signal=signal, scan_id=scan_id, status="candidate",
                        )
                        created.append(item)

                # Current-only peer divergence is evidence-worthy even without a large batch delta.
                peer_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
                for row in current_rows:
                    peer_groups.setdefault(
                        (row["stage"], row["question_level"], str(row["search_enabled"])), []
                    ).append(row)
                for (stage, question_level, search_mode), peers in peer_groups.items():
                    if len(peers) < 3:
                        continue
                    for metric in SUMMARY_RATE_METRICS:
                        values = [float(row[metric]) for row in peers if row.get(metric) is not None]
                        if len(values) < 3:
                            continue
                        center = median(values)
                        for row in peers:
                            if row.get(metric) is None or int(row.get("total_answers") or 0) < 10:
                                continue
                            if (
                                slice_question_count(
                                    current_dataset_id, stage, row["model"], search_mode
                                ) < 5
                                or slice_question_count(
                                    baseline, stage, row["model"], search_mode
                                ) < 5
                            ):
                                continue
                            deviation = float(row[metric]) - center
                            spec = RATE_METRICS[metric]
                            bad = deviation <= -0.20 if spec["higher_is_better"] else deviation >= 0.20
                            if not bad:
                                continue
                            old = baseline_map.get(
                                (stage, question_level, row["model"], search_mode)
                            )
                            signal = {
                                "metric": metric,
                                "metric_label": spec["label"],
                                "current_value": float(row[metric]),
                                "baseline_value": (
                                    float(old[metric])
                                    if old and old.get(metric) is not None else None
                                ),
                                "delta": (
                                    float(row[metric]) - float(old[metric])
                                    if old and old.get(metric) is not None else None
                                ),
                                "peer_median": center,
                                "peer_deviation": deviation,
                                "current_n": int(row.get("total_answers") or 0),
                                "baseline_n": int((old or {}).get("total_answers") or 0),
                                "direction": "regression",
                                "severity": "high" if abs(deviation) >= .30 else "medium",
                                "reason": "peer_divergence",
                                "question_level": question_level,
                            }
                            item = _insert_investigation(
                                owner_username=owner, trigger_type="auto",
                                current_dataset_id=current_dataset_id,
                                baseline_dataset_id=baseline, product_code=product_code,
                                metric=metric, stage=stage, model=row["model"],
                                search_enabled=(
                                    int(search_mode) if search_mode in {"0", "1"} else None
                                ),
                                expected_value=None, signal=signal, scan_id=scan_id,
                                status="candidate",
                            )
                            created.append(item)
        unique = {item["investigation_id"]: item for item in created}
        created = list(unique.values())
        should_start = settings["auto_start"] if auto_start is None else int(bool(auto_start))
        if settings["enabled"] and should_start:
            ranked = sorted(
                (
                    item for item in created
                    if (
                        item.get("direction") != "improvement"
                        and item["severity"] == "high"
                    )
                ),
                key=lambda item: (
                    0 if item["severity"] == "high" else 1,
                    -abs(float((item.get("signal") or {}).get("delta") or 0)),
                ),
            )
            for item in ranked[: settings["max_auto_cases"]]:
                start_investigation(item["investigation_id"])
        with _connect() as conn:
            conn.execute(
                """UPDATE investigation_scans
                   SET status='completed',candidate_count=?,error_message=?,finished_at=?
                   WHERE scan_id=?""",
                (
                    len(created), _json(warnings) if warnings else None,
                    _now(), scan_id,
                ),
            )
    except Exception as exc:
        with _connect() as conn:
            conn.execute(
                """UPDATE investigation_scans
                   SET status='failed',error_message=?,finished_at=?
                   WHERE scan_id=?""",
                (str(exc)[:2000], _now(), scan_id),
            )
        raise
    return {
        "scan_id": scan_id,
        "current_dataset_id": current_dataset_id,
        "baseline_dataset_id": baseline_dataset_id,
        "candidate_count": len(created),
        "candidates": created,
        "warnings": warnings,
    }


def enqueue(investigation_id: str) -> None:
    ensure_worker()
    _QUEUE.put(investigation_id)


def start_investigation(investigation_id: str) -> dict[str, Any]:
    item = get_investigation(investigation_id, include_details=False)
    if item is None:
        raise ValueError("调查不存在")
    if item["status"] not in {"candidate", "needs_review", "failed", "cancelled"}:
        if item["status"] in {"queued", "running"}:
            return item
        raise ValueError(f"当前状态不能启动: {item['status']}")
    _CANCEL_FLAGS.discard(investigation_id)
    update_investigation(
        investigation_id, status="queued", stage="signal_validation",
        progress=0, error_message=None, cancellation_requested=0,
        finished_at=None,
    )
    add_event(investigation_id, "queued", "调查已进入执行队列")
    enqueue(investigation_id)
    return get_investigation(investigation_id, include_details=False) or {}


def retry_investigation(investigation_id: str) -> dict[str, Any]:
    item = get_investigation(investigation_id, include_details=False)
    if item is None:
        raise ValueError("调查不存在")
    if item["status"] not in {"failed", "needs_review", "cancelled"}:
        raise ValueError(f"当前状态不能重试: {item['status']}")
    return start_investigation(investigation_id)


def cancel_investigation(investigation_id: str) -> dict[str, Any]:
    item = get_investigation(investigation_id, include_details=False)
    if item is None:
        raise ValueError("调查不存在")
    if item["status"] not in {"queued", "running"}:
        raise ValueError(f"当前状态不能取消: {item['status']}")
    _CANCEL_FLAGS.add(investigation_id)
    update_investigation(
        investigation_id, status="cancelled", cancellation_requested=1,
        finished_at=_now(), error_message="用户取消",
    )
    add_event(investigation_id, "cancelled", "调查已取消")
    return get_investigation(investigation_id, include_details=False) or {}


def is_cancelled(investigation_id: str) -> bool:
    if investigation_id in _CANCEL_FLAGS:
        return True
    with _connect() as conn:
        row = conn.execute(
            "SELECT cancellation_requested,status FROM investigations WHERE investigation_id=?",
            (investigation_id,),
        ).fetchone()
    return bool(row and (row["cancellation_requested"] or row["status"] == "cancelled"))


def ensure_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True
        _recover_expired_investigations()
        with _connect() as conn:
            rows = conn.execute(
                """SELECT investigation_id FROM investigations
                   WHERE status='queued' ORDER BY created_at"""
            ).fetchall()
        for row in rows:
            _QUEUE.put(row["investigation_id"])
        threading.Thread(target=_worker_loop, daemon=True, name="geo-investigation-worker").start()


def _worker_loop() -> None:
    while True:
        try:
            investigation_id = _QUEUE.get(timeout=5)
        except queue.Empty:
            for recovered_id in _recover_expired_investigations():
                _QUEUE.put(recovered_id)
            continue
        try:
            if not _claim_investigation(investigation_id):
                continue
            # Lazy import avoids a storage/agent import cycle.
            from .investigation_agent import run_investigation
            run_investigation(investigation_id)
        except Exception as exc:  # noqa: BLE001 - worker must survive
            update_investigation(
                investigation_id, status="failed", stage="conclusion",
                error_message=str(exc)[:2000], finished_at=_now(),
            )
            add_event(
                investigation_id, "failed", "调查执行失败",
                {"error": str(exc)[:1000]},
            )
        finally:
            _QUEUE.task_done()
