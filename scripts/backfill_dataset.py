# -*- coding: utf-8 -*-
"""
历史数据集回刷：用新版 05（含情感/负面维度）重新抽取，并重新物化指标。

流程：export-raw → 全新抽取目录跑 05 →（可选 07）→ import-baseline 回灌
     （不加 --reset，external 同名表覆盖，原始 answers 不动）→ materialize。

用法：
  # 抽样对比：新旧抽取结果差异率（不写库）
  python scripts/backfill_dataset.py --dataset-id baseline_8products_20260423 --sample 50

  # 全量回刷（05 + 物化；07 已有数据默认跳过，需要时加 --run-verify）
  python scripts/backfill_dataset.py --dataset-id baseline_8products_20260423

回刷前请备份 data/geo_datasets/geo_answers.sqlite。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

PY = sys.executable
SCRIPTS = BASE_DIR / "scripts"
DB_PATH = BASE_DIR / "data" / "geo_datasets" / "geo_answers.sqlite"


def dataset_meta(dataset_id: str) -> tuple[str, str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT name, description FROM datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise SystemExit(f"数据集不存在: {dataset_id}")
    return row[0] or dataset_id, row[1] or ""


def run(cmd: list[str], env: dict[str, str] | None = None, name: str = "") -> None:
    print(f"\n===== {name or cmd[1]} =====", flush=True)
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"步骤失败（exit {proc.returncode}）：{name}")


def stage_env(workdir: Path, route: str) -> dict[str, str]:
    env = os.environ.copy()
    env["GEO_RAW_DIR"] = str(workdir / "raw")
    env["GEO_ANALYSIS_DIR"] = str(workdir / "analysis")
    env["GEO_EXTRACT_DIR"] = str(workdir / "extractions")
    env["GEO_ROUTE"] = route
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def export_raw(dataset_id: str, workdir: Path) -> None:
    run(
        [
            PY, "-X", "utf8", str(SCRIPTS / "manage_geo_sqlite.py"), "export-raw",
            "--dataset-id", dataset_id,
            "--output-raw-dir", str(workdir / "raw"),
            "--questions-output", str(workdir / "questions.json"),
            "--reset-output",
        ],
        name="export-raw",
    )


def load_old_extractions() -> dict:
    path = BASE_DIR / "results" / "extractions" / "extraction_log.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("completed", {})


def sample_compare(dataset_id: str, workdir: Path, sample: int, route: str) -> None:
    """抽 N 条旧抽取里有缓存的应答双跑对比，评估新 prompt 的口径漂移。"""
    export_raw(dataset_id, workdir)
    old = load_old_extractions()
    raw_files = sorted((workdir / "raw").rglob("*.json"))
    candidates = []
    for path in raw_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        key = f"{data.get('question_id','')}_{data.get('model','')}_{data.get('search_enabled','')}_{data.get('round','')}"
        if key in old:
            candidates.append((path, key))
    random.seed(42)
    picked = random.sample(candidates, min(sample, len(candidates)))
    print(f"旧缓存可对比应答 {len(candidates)} 条，抽样 {len(picked)} 条")

    sample_dir = workdir / "sample"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    (sample_dir / "raw" / "mixed").mkdir(parents=True, exist_ok=True)
    for path, _ in picked:
        shutil.copy(path, sample_dir / "raw" / "mixed" / path.name)

    env = stage_env(sample_dir, route)
    run([PY, "-X", "utf8", str(SCRIPTS / "05_extract_recommendations.py")], env=env, name="05 抽样重抽")

    new_log = json.loads((sample_dir / "extractions" / "extraction_log.json").read_text(encoding="utf-8"))["completed"]
    same_products = same_strength = total = 0
    sentiment_filled = 0
    for _, key in picked:
        if key not in new_log:
            continue
        total += 1
        old_recs, new_recs = old[key], new_log[key]
        old_set = {str(r.get("product", "")) for r in old_recs}
        new_set = {str(r.get("product", "")) for r in new_recs}
        if old_set == new_set:
            same_products += 1
        old_st = {(str(r.get("product", "")), r.get("strength")) for r in old_recs}
        new_st = {(str(r.get("product", "")), r.get("strength")) for r in new_recs}
        if old_st == new_st:
            same_strength += 1
        if any(r.get("sentiment") for r in new_recs):
            sentiment_filled += 1
    print("\n===== 抽样对比结果 =====")
    print(f"对比条数: {total}")
    print(f"推荐产品集合一致: {same_products}/{total} ({same_products/total:.0%})" if total else "无")
    print(f"产品+强度完全一致: {same_strength}/{total} ({same_strength/total:.0%})" if total else "无")
    print(f"sentiment 有值比例: {sentiment_filled}/{total}" if total else "无")
    print("\n（口径漂移可接受则执行全量回刷：去掉 --sample 参数）")


def backfill(dataset_id: str, workdir: Path, route: str, run_verify: bool, verify_levels: str) -> None:
    export_raw(dataset_id, workdir)
    env = stage_env(workdir, route)
    # 全新抽取目录（不带旧缓存）——新 prompt 必须全量重抽
    run([PY, "-X", "utf8", str(SCRIPTS / "05_extract_recommendations.py")], env=env, name="05 全量重抽")
    if run_verify:
        env["GEO_ACCURACY_LEVELS"] = verify_levels
        env["GEO_KB_FILE"] = str(BASE_DIR / "config" / "knowledge_base.json")
        run([PY, "-X", "utf8", str(SCRIPTS / "07_verify_accuracy.py")], env=env, name="07 准确率校验")
    # 回灌：不加 --reset（保留原始 answers/accuracy 等外部表，仅覆盖同名分析表）
    name, description = dataset_meta(dataset_id)
    run(
        [
            PY, "-X", "utf8", str(SCRIPTS / "manage_geo_sqlite.py"), "import-baseline",
            "--dataset-id", dataset_id,
            "--name", name,
            "--description", description,
            "--raw-dir", str(workdir / "raw"),
            "--analysis-dir", str(workdir / "analysis"),
            "--questions", str(workdir / "questions.json"),
            "--questions-base", str(workdir / "questions.json"),
        ],
        name="import-baseline 回灌",
    )
    run(
        [PY, "-X", "utf8", str(SCRIPTS / "manage_geo_sqlite.py"), "materialize", "--dataset-id", dataset_id],
        name="materialize",
    )
    print("\n回刷完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--sample", type=int, default=0, help="抽样对比条数（不写库）")
    parser.add_argument("--route", default="relay", choices=["relay", "direct"])
    parser.add_argument("--run-verify", action="store_true", help="重跑 07（prompt 未变时通常不需要）")
    parser.add_argument("--verify-levels", default="q1,q2")
    args = parser.parse_args()

    workdir = Path(args.workdir) if args.workdir else BASE_DIR / "backend" / "data" / "backfill" / args.dataset_id
    workdir.mkdir(parents=True, exist_ok=True)
    if args.sample:
        sample_compare(args.dataset_id, workdir, args.sample, args.route)
    else:
        backfill(args.dataset_id, workdir, args.route, args.run_verify, args.verify_levels)


if __name__ == "__main__":
    main()
