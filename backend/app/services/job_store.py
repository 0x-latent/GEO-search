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
from .yaml_store import load_models

JOBS_DB_PATH = DATA_DIR / "jobs.sqlite"
JOBS_DIR = DATA_DIR / "jobs"

MAX_QUESTIONS = 500
DEFAULT_CONCURRENCY = 20
MAX_CONCURRENCY = 50

# 运行中任务的子进程句柄与取消标记（单 worker 线程，dict 足够）
_RUNNING_PROCS: dict[str, subprocess.Popen] = {}
_CANCEL_FLAGS: set[str] = set()

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
        lazy = {
            "product_code": "TEXT",
            "batch_date": "TEXT",
            "concurrency": "INTEGER",
            "model_concurrency_json": "TEXT",
            "total_calls": "INTEGER",
        }
        for name, ddl in lazy.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}")


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


def collect_progress(job: dict[str, Any]) -> dict[str, Any] | None:
    """采集阶段进度：execution_log 已完成的唯一任务数 / 预估总调用数。"""
    total = job.get("total_calls") or 0
    if not total:
        return None
    log_path = JOBS_DIR / job["job_id"] / "execution_log.json"
    if not log_path.exists():
        return {"done": 0, "total": total}
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"done": 0, "total": total}
    keys = {
        (e.get("question_id"), e.get("model"), e.get("search_enabled"), e.get("round"))
        for e in log.get("executions", [])
    }
    return {"done": min(len(keys), total), "total": total}


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
    jobs = [dict(row) for row in rows]
    for job in jobs:
        # 只为进行中的采集任务读日志（避免列表接口扫全部历史文件）
        if job["status"] == "running" and job.get("stage") == "collect":
            job["collect_progress"] = collect_progress(job)
    return jobs


def retry_job(job_id: str) -> dict[str, Any]:
    """失败/已取消任务断点续跑：03/05/07 有各自的断点日志，只补没完成的部分。"""
    job = get_job(job_id)
    if job is None:
        raise ValueError("任务不存在")
    if job["status"] not in ("failed", "cancelled"):
        raise ValueError(f"当前状态不能重试：{job['status']}")
    _CANCEL_FLAGS.discard(job_id)
    _update_job(job_id, status="queued", error=None, finished_at=None)
    ensure_worker()
    _QUEUE.put(job_id)
    return get_job(job_id)


def cancel_job(job_id: str) -> dict[str, Any]:
    """取消任务：排队中直接标记；执行中终止当前阶段子进程（已完成部分保留，可重试续跑）。"""
    job = get_job(job_id)
    if job is None:
        raise ValueError("任务不存在")
    if job["status"] not in ("queued", "running"):
        raise ValueError(f"当前状态不能取消：{job['status']}")
    _CANCEL_FLAGS.add(job_id)
    _update_job(job_id, status="cancelled", error="用户取消", finished_at=_now())
    process = _RUNNING_PROCS.get(job_id)
    if process is not None and process.poll() is None:
        process.terminate()
    return get_job(job_id)


def read_job_log(job_id: str) -> str:
    path = JOBS_DIR / job_id / "job.log"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    # 只回传末尾 200KB，避免大日志拖垮页面
    return text[-200_000:]


def _estimate_total_calls(
    questions_count: int, models: list[str], search_mode: str, rounds: int
) -> int:
    """采集阶段总调用数（进度分母）。按各模型是否支持联网精确计算。"""
    config = load_models()
    specs = config.get("models") or {}
    per_question = 0
    for key in models:
        supports_search = (specs.get(key) or {}).get("supports_search", False)
        if search_mode == "both":
            per_question += 2 if supports_search else 1
        elif search_mode == "search":
            per_question += 1 if supports_search else 0
        else:
            per_question += 1
    return questions_count * per_question * rounds


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
    concurrency: int | None = None,
    model_concurrency: dict[str, int] | None = None,
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
    # 并发：默认 20；仅 admin 可调（全局与按模型）
    if role != "admin":
        concurrency = DEFAULT_CONCURRENCY
        model_concurrency = None
    concurrency = max(1, min(int(concurrency or DEFAULT_CONCURRENCY), MAX_CONCURRENCY))
    if model_concurrency:
        model_concurrency = {
            key: max(1, min(int(value), MAX_CONCURRENCY))
            for key, value in model_concurrency.items()
            if key in models
        } or None
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
                created_at, product_code, batch_date,
                concurrency, model_concurrency_json, total_calls
            ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                concurrency,
                json.dumps(model_concurrency, ensure_ascii=False) if model_concurrency else None,
                _estimate_total_calls(len(questions), models, search_mode, rounds),
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
    # 并发：全局 + 按模型覆盖
    env["GEO_CONCURRENCY"] = str(job.get("concurrency") or DEFAULT_CONCURRENCY)
    if job.get("model_concurrency_json"):
        env["GEO_MODEL_CONCURRENCY"] = job["model_concurrency_json"]
    return env


def _run_stage(
    name: str, command: list[str], env: dict[str, str], log_path: Path, job_id: str = ""
) -> int:
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
        if job_id:
            _RUNNING_PROCS[job_id] = process
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
            return process.wait()
        finally:
            _RUNNING_PROCS.pop(job_id, None)


def _run_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None or job["status"] not in ("queued", "running"):
        return
    job_dir = JOBS_DIR / job_id
    log_path = job_dir / "job.log"
    _update_job(job_id, status="running", started_at=job["started_at"] or _now(), error=None)

    env = _stage_env(job, job_dir)
    for stage_name, script, critical in STAGES:
        if job_id in _CANCEL_FLAGS:
            return  # 已被取消，状态由 cancel_job 写入
        _update_job(job_id, stage=stage_name)
        exit_code = _run_stage(
            stage_name, [sys.executable, str(SCRIPTS_DIR / script)], env, log_path, job_id
        )
        if job_id in _CANCEL_FLAGS:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[cancelled] 任务被用户取消于 {stage_name} 阶段\n")
            return
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
    if job_id in _CANCEL_FLAGS:
        return
    exit_code = _run_stage("import", import_cmd, env, log_path, job_id)
    if job_id in _CANCEL_FLAGS:
        return
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
    exit_code = _run_stage("materialize", materialize_cmd, env, log_path, job_id)
    if job_id in _CANCEL_FLAGS:
        return
    if exit_code != 0:
        _update_job(
            job_id, status="failed", stage="materialize",
            error=f"指标物化失败 (exit {exit_code})", finished_at=_now(),
        )
        return

    _update_job(job_id, status="success", stage="done", finished_at=_now())

    # 调查扫描不反向影响数据采集任务成功状态。没有可比基线是正常情况；
    # 扫描异常只记录到 job 日志，用户仍可通过 API 手动发起调查。
    try:
        from . import investigation_store
        scan = investigation_store.scan_dataset(
            job["dataset_id"], owner_username=job["username"]
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[investigation] scan={scan['scan_id']} "
                f"candidates={scan['candidate_count']}\n"
            )
    except Exception as exc:  # noqa: BLE001 - 调查是非阻塞后处理
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[investigation] 自动扫描跳过/失败: {exc}\n")
