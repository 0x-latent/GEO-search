from __future__ import annotations

import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.paths import CONFIG_DIR, DATA_DIR, ROOT_DIR, SCRIPTS_DIR
from .user_config_store import GLOBAL_KB_PATH, user_brands_path, user_kb_path

JOBS_DB_PATH = DATA_DIR / "jobs.sqlite"
JOBS_DIR = DATA_DIR / "jobs"

MAX_QUESTIONS = 500

# 流水线阶段：(阶段名, 脚本, 失败是否中止整个 job)
STAGES = [
    ("collect", "03_query_models.py", True),
    ("analyze", "04_analyze_results.py", False),
    ("extract", "05_extract_recommendations.py", False),
    ("verify", "07_verify_accuracy.py", False),
]

_QUEUE: "queue.Queue[str]" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    JOBS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(JOBS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                stage TEXT,
                models_json TEXT NOT NULL,
                model_overrides_json TEXT NOT NULL DEFAULT '{}',
                search_mode TEXT NOT NULL DEFAULT 'both',
                rounds INTEGER NOT NULL DEFAULT 1,
                route TEXT NOT NULL DEFAULT 'relay',
                question_count INTEGER NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                product_code TEXT,
                batch_date TEXT
            )
            """
        )
        # 已有库的惰性迁移
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        for name in ("product_code", "batch_date"):
            if name not in columns:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} TEXT")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _update_job(job_id: str, **updates: Any) -> None:
    keys = ", ".join(f"{k} = ?" for k in updates)
    with _connect() as conn:
        conn.execute(f"UPDATE jobs SET {keys} WHERE job_id = ?", (*updates.values(), job_id))


def get_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(username: str | None = None) -> list[dict[str, Any]]:
    """username=None 返回全部（admin），否则只返回该用户的。"""
    with _connect() as conn:
        if username is None:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 200").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE username = ? ORDER BY created_at DESC LIMIT 200",
                (username,),
            ).fetchall()
    return [dict(row) for row in rows]


def read_job_log(job_id: str) -> str:
    path = JOBS_DIR / job_id / "job.log"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    # 只回传末尾 200KB，避免大日志拖垮页面
    return text[-200_000:]


def create_job(
    username: str,
    role: str,
    dataset_name: str,
    questions: list[dict[str, Any]],
    models: list[str],
    model_overrides: dict[str, str] | None = None,
    search_mode: str = "both",
    rounds: int = 1,
    route: str | None = None,
    product_code: str | None = None,
    batch_date: str | None = None,
) -> dict[str, Any]:
    if not questions:
        raise ValueError("问题列表为空")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"单次最多 {MAX_QUESTIONS} 个问题，当前 {len(questions)} 个")
    if not models:
        raise ValueError("至少选择一个模型")
    if search_mode not in ("both", "search", "nosearch"):
        raise ValueError("search_mode 必须是 both / search / nosearch")
    rounds = max(1, min(int(rounds), 5))
    # 链路控制：普通用户强制走中继；admin 可显式选 direct
    if role != "admin" or route not in ("relay", "direct"):
        route = "relay"
    # 批次归属：产品必须在主数据中（趋势锚点），批次日期默认当天
    product_code = (product_code or "").strip() or None
    if product_code:
        from . import product_master

        active = {p["product_code"] for p in product_master.list_products(active_only=True)}
        if product_code not in active:
            raise ValueError(f"产品不在主数据中：{product_code}（请先在品牌配置里维护）")
    batch_date = (batch_date or "").strip() or datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(batch_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("batch_date 格式必须是 YYYY-MM-DD")

    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    dataset_id = f"user_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, username, dataset_id, dataset_name, status, models_json,
                model_overrides_json, search_mode, rounds, route, question_count,
                created_at, product_code, batch_date
            ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                username,
                dataset_id,
                dataset_name,
                json.dumps(models, ensure_ascii=False),
                json.dumps(model_overrides or {}, ensure_ascii=False),
                search_mode,
                rounds,
                route,
                len(questions),
                _now(),
                product_code,
                batch_date,
            ),
        )

    ensure_worker()
    _QUEUE.put(job_id)
    return get_job(job_id)


def ensure_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True
        # 恢复上次进程退出时排队/中断的任务（03 有断点续跑，重跑安全）
        with _connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
            ).fetchall()
        for row in rows:
            _QUEUE.put(row["job_id"])
        thread = threading.Thread(target=_worker_loop, daemon=True)
        thread.start()


def _worker_loop() -> None:
    while True:
        job_id = _QUEUE.get()
        try:
            _run_job(job_id)
        except Exception as exc:
            _update_job(job_id, status="failed", error=str(exc), finished_at=_now())
        finally:
            _QUEUE.task_done()


def _stage_env(job: dict[str, Any], job_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["GEO_QUESTIONS_FILE"] = str(job_dir / "questions.json")
    env["GEO_RAW_DIR"] = str(job_dir / "raw")
    env["GEO_ANALYSIS_DIR"] = str(job_dir / "analysis")
    env["GEO_EXTRACT_DIR"] = str(job_dir / "extractions")
    env["GEO_EXEC_LOG"] = str(job_dir / "execution_log.json")
    env["GEO_ROUTE"] = job["route"]
    env["GEO_ROUNDS"] = str(job["rounds"])
    env["GEO_MODELS"] = ",".join(json.loads(job["models_json"]))
    env["GEO_SEARCH_MODES"] = job["search_mode"]
    overrides = job.get("model_overrides_json") or "{}"
    if overrides != "{}":
        env["GEO_MODEL_OVERRIDES"] = overrides
    brands = user_brands_path(job["username"])
    if brands is not None:
        env["GEO_BRANDS_FILE"] = str(brands)
    # 07 准确率校验：优先用户知识库，放宽层级到全部（用户上传问题统一是 q4 前缀）
    kb = user_kb_path(job["username"]) or GLOBAL_KB_PATH
    env["GEO_KB_FILE"] = str(kb)
    env["GEO_ACCURACY_LEVELS"] = "all"
    return env


def _run_stage(name: str, command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== [{_now()}] stage: {name} =====\n$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        return process.wait()


def _run_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None or job["status"] not in ("queued", "running"):
        return
    job_dir = JOBS_DIR / job_id
    log_path = job_dir / "job.log"
    _update_job(job_id, status="running", started_at=job["started_at"] or _now(), error=None)

    env = _stage_env(job, job_dir)
    for stage_name, script, critical in STAGES:
        _update_job(job_id, stage=stage_name)
        exit_code = _run_stage(
            stage_name, [sys.executable, str(SCRIPTS_DIR / script)], env, log_path
        )
        if exit_code != 0:
            if critical:
                _update_job(
                    job_id, status="failed", stage=stage_name,
                    error=f"{stage_name} 阶段失败 (exit {exit_code})", finished_at=_now(),
                )
                return
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[warn] {stage_name} 阶段失败 (exit {exit_code})，继续后续阶段\n")

    # 入库为该用户的 dataset
    _update_job(job_id, stage="import")
    import_cmd = [
        sys.executable, str(SCRIPTS_DIR / "manage_geo_sqlite.py"), "import-baseline",
        "--dataset-id", job["dataset_id"],
        "--name", job["dataset_name"],
        "--description", f"用户 {job['username']} 上传的问题分析（{job['question_count']} 题）",
        "--raw-dir", str(job_dir / "raw"),
        "--analysis-dir", str(job_dir / "analysis"),
        "--questions", str(job_dir / "questions.json"),
        "--questions-base", str(job_dir / "questions.json"),
        "--owner", job["username"],
        "--reset",
    ]
    if job.get("product_code"):
        import_cmd += ["--product-code", job["product_code"]]
    if job.get("batch_date"):
        import_cmd += ["--batch-date", job["batch_date"]]
    exit_code = _run_stage("import", import_cmd, env, log_path)
    if exit_code != 0:
        _update_job(
            job_id, status="failed", stage="import",
            error=f"入库失败 (exit {exit_code})", finished_at=_now(),
        )
        return

    # 物化指标（品牌总览/产品详情/趋势的数据源）——失败则新页面无数据，按失败处理
    _update_job(job_id, stage="materialize")
    materialize_cmd = [
        sys.executable, str(SCRIPTS_DIR / "manage_geo_sqlite.py"), "materialize",
        "--dataset-id", job["dataset_id"],
    ]
    exit_code = _run_stage("materialize", materialize_cmd, env, log_path)
    if exit_code != 0:
        _update_job(
            job_id, status="failed", stage="materialize",
            error=f"指标物化失败 (exit {exit_code})", finished_at=_now(),
        )
        return

    _update_job(job_id, status="success", stage="done", finished_at=_now())
