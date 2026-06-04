from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.paths import ROOT_DIR, SCRIPTS_DIR, TASKS_DIR


TASKS_PATH = TASKS_DIR / "tasks.json"
TASK_LOCK = threading.Lock()

SCRIPT_CATALOG = {
    "01_parse_questions": {"label": "01 解析问题", "script": "01_parse_questions.py", "args": []},
    "02_expand_questions": {"label": "02 生成问题变体", "script": "02_expand_questions.py", "args": []},
    "03_query_models": {"label": "03 采集模型回答", "script": "03_query_models.py", "args": []},
    "04_analyze_results": {"label": "04 基础统计分析", "script": "04_analyze_results.py", "args": []},
    "05_extract_recommendations": {"label": "05 推荐信息抽取", "script": "05_extract_recommendations.py", "args": []},
    "06_build_knowledge_base": {"label": "06 构建知识库", "script": "06_build_knowledge_base.py", "args": []},
    "06_build_knowledge_base_force": {"label": "06 构建知识库 force", "script": "06_build_knowledge_base.py", "args": ["--force"]},
    "07_verify_accuracy": {"label": "07 准确性校验", "script": "07_verify_accuracy.py", "args": []},
    "08_generate_report": {"label": "08 生成分析报告", "script": "08_generate_report.py", "args": []},
}


def _load_tasks() -> dict[str, Any]:
    if not TASKS_PATH.exists():
        return {"tasks": []}
    try:
        return json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"tasks": []}


def _save_tasks(data: dict[str, Any]) -> None:
    TASKS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_task(task_id: str, **updates: Any) -> None:
    with TASK_LOCK:
        data = _load_tasks()
        for task in data["tasks"]:
            if task["id"] == task_id:
                task.update(updates)
                break
        _save_tasks(data)


def list_catalog() -> dict[str, Any]:
    return SCRIPT_CATALOG


def list_tasks() -> list[dict[str, Any]]:
    return _load_tasks()["tasks"]


def read_task_log(task_id: str) -> str:
    path = TASKS_DIR / f"{task_id}.log"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def start_task(script_key: str, extra_args: list[str] | None = None, run_id: str | None = None) -> dict[str, Any]:
    if script_key not in SCRIPT_CATALOG:
        raise ValueError(f"Unknown script: {script_key}")

    spec = SCRIPT_CATALOG[script_key]
    script_path = SCRIPTS_DIR / spec["script"]
    if not script_path.exists():
        raise ValueError(f"Script not found: {script_path}")

    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    log_path = TASKS_DIR / f"{task_id}.log"
    args = list(extra_args if extra_args is not None else spec.get("args", []))

    task = {
        "id": task_id,
        "run_id": run_id,
        "script_key": script_key,
        "label": spec["label"],
        "status": "queued",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "args": args,
        "log_path": str(log_path),
    }
    with TASK_LOCK:
        data = _load_tasks()
        data["tasks"].insert(0, task)
        _save_tasks(data)

    thread = threading.Thread(target=_run_task, args=(task_id, script_path, args, log_path), daemon=True)
    thread.start()
    return task


def _run_task(task_id: str, script_path: Path, args: list[str], log_path: Path) -> None:
    _update_task(task_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
    command = [sys.executable, str(script_path), *args]

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
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
            exit_code = process.wait()
            status = "success" if exit_code == 0 else "failed"
            _update_task(
                task_id,
                status=status,
                exit_code=exit_code,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            log.write(f"\nTask runner error: {exc}\n")
            _update_task(
                task_id,
                status="failed",
                exit_code=-1,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
