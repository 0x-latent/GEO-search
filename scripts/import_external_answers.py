"""
Import externally collected AI answer spreadsheets into the existing raw JSON format.

The analyzer reads results/raw/**/*.json. This script converts a normalized Excel
detail sheet into that boundary format and writes a companion question map under
questions/imported_questions_<dataset_id>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = [
    "提问层级",
    "场景",
    "问题",
    "查询轮次",
    "查询时间",
    "AI模型",
    "AI回答",
    "引用信源",
    "信源数量",
]

LEVEL_MAP = {
    "解决方案": {"level": "q3_solution", "category": "mention_recommend"},
    "泛式吃药": {"level": "q4_medicine", "category": "mention_recommend"},
    "中药相关": {"level": "q5_tcm", "category": "mention_recommend"},
}

MODEL_MAP = {
    "deepseek": ("deepseek", "DeepSeek"),
    "kimi": ("kimi", "Kimi"),
    "元宝": ("yuanbao", "元宝"),
    "千问": ("qwen", "通义千问"),
    "百度AI": ("baidu", "百度AI"),
    "豆包": ("doubao", "豆包"),
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _iso_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return datetime.now().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return datetime.now().isoformat()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.isoformat()


def _question_hash(level_label: str, scenario: str, question: str) -> str:
    raw = f"{level_label}|{scenario}|{question}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:8]


def _split_sources(value: Any) -> list[dict[str, str]]:
    text = _clean_text(value)
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


def _infer_has_brand(question: str, brand_terms: list[str]) -> bool:
    return any(term and term in question for term in brand_terms)


def _load_external_sheet(path: Path, sheet_name: str | None) -> pd.DataFrame:
    if sheet_name:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"输入表缺少必要字段: {', '.join(missing)}")
    return df


def _build_question_map(
    df: pd.DataFrame,
    product: str,
    product_code: str,
    brand_terms: list[str],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    question_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        level_label = _clean_text(row["提问层级"])
        scenario = _clean_text(row["场景"])
        question = _clean_text(row["问题"])
        if not question:
            continue

        level_info = LEVEL_MAP.get(level_label)
        if not level_info:
            level_code = f"external_{_question_hash(level_label, '', '')}"
            category = "external"
        else:
            level_code = level_info["level"]
            category = level_info["category"]

        key = (level_label, scenario, question)
        if key in question_map:
            continue

        qid = f"{product_code}_{level_code}_{_question_hash(level_label, scenario, question)}"
        question_map[key] = {
            "id": qid,
            "product": product,
            "product_code": product_code,
            "category": category,
            "level": level_code,
            "question": question,
            "has_brand_name": _infer_has_brand(question, brand_terms),
            "is_variant": False,
            "variant_of": None,
            "source_level": level_label,
            "scenario": scenario,
        }
    return question_map


def import_answers(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = _load_external_sheet(input_path, args.sheet)
    df = df[df["AI回答"].notna()].copy()

    brand_terms = [term.strip() for term in args.brand_terms.split(",") if term.strip()]
    question_map = _build_question_map(df, args.product, args.product_code, brand_terms)

    raw_output_dir = Path(args.raw_output_dir).resolve()
    question_output = Path(args.question_output).resolve()
    manifest_output = Path(args.manifest_output).resolve()

    written = 0
    skipped_existing = 0
    rows = 0
    errors: list[str] = []

    for idx, row in enumerate(df.to_dict(orient="records"), start=2):
        rows += 1
        level_label = _clean_text(row["提问层级"])
        scenario = _clean_text(row["场景"])
        question = _clean_text(row["问题"])
        qinfo = question_map.get((level_label, scenario, question))
        if not qinfo:
            errors.append(f"第 {idx} 行缺少问题文本，已跳过")
            continue

        model_label = _clean_text(row["AI模型"])
        model_key, model_name = MODEL_MAP.get(model_label, (model_label or "unknown", model_label or "unknown"))
        round_num = _safe_int(row["查询轮次"], default=1)
        sources = _split_sources(row.get("引用信源"))
        source_count = _safe_int(row.get("信源数量"), default=len(sources))
        # 优先使用表内的联网标注；缺列时才退回“有信源即联网”的推断
        # （推断口径下联网必然带信源，信源覆盖率会被固定在 100%）。
        search_text = _clean_text(row.get("联网"))
        if search_text:
            search_enabled = search_text in {"是", "1", "True", "true"}
        else:
            search_enabled = source_count > 0 or bool(sources)
        search_tag = "search" if search_enabled else "nosearch"

        payload = {
            "answer": _clean_text(row["AI回答"]),
            "model": model_key,
            "model_name": model_name,
            "question_id": qinfo["id"],
            "question_text": question,
            "product": args.product,
            "round": round_num,
            "timestamp": _iso_timestamp(row["查询时间"]),
            "search_enabled": search_enabled,
            "latency_ms": None,
            "sources": sources,
            "source_count": source_count,
            "external_meta": {
                "dataset_id": args.dataset_id,
                "source_file": str(input_path),
                "source_row": idx,
                "source_level": level_label,
                "scenario": scenario,
            },
        }

        output_path = raw_output_dir / model_key / f"{qinfo['id']}_r{round_num}_{search_tag}.json"
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    questions = sorted(question_map.values(), key=lambda item: item["id"])
    if not args.dry_run:
        question_output.parent.mkdir(parents=True, exist_ok=True)
        question_output.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest = {
            "dataset_id": args.dataset_id,
            "source_file": str(input_path),
            "imported_at": datetime.now().isoformat(),
            "raw_output_dir": str(raw_output_dir),
            "question_output": str(question_output),
            "rows_seen": rows,
            "questions": len(questions),
            "written": written,
            "skipped_existing": skipped_existing,
            "errors": errors,
        }
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "dataset_id": args.dataset_id,
        "rows_seen": rows,
        "questions": len(questions),
        "written": written,
        "skipped_existing": skipped_existing,
        "errors": errors,
        "dry_run": args.dry_run,
        "question_output": str(question_output),
        "raw_output_dir": str(raw_output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import external AI answer Excel into results/raw JSON files.")
    parser.add_argument("--input", required=True, help="Excel file path.")
    parser.add_argument("--sheet", default=None, help="Sheet name. Defaults to the first sheet.")
    parser.add_argument("--dataset-id", default="weitai_yangweishu_external", help="Stable dataset id.")
    parser.add_argument("--product", default="胃泰", help="Canonical product short name used by reports.")
    parser.add_argument("--product-code", default="weitai", help="Canonical product code.")
    parser.add_argument(
        "--brand-terms",
        default="养胃舒,三九胃泰,999胃泰,胃泰",
        help="Comma-separated terms used to flag whether the question includes a brand/product name.",
    )
    parser.add_argument("--raw-output-dir", default=str(BASE_DIR / "results" / "raw"), help="Output raw JSON root.")
    parser.add_argument(
        "--question-output",
        default=str(BASE_DIR / "questions" / "imported_questions_weitai_yangweishu_external.json"),
        help="Output imported question map JSON.",
    )
    parser.add_argument(
        "--manifest-output",
        default=str(BASE_DIR / "results" / "imports" / "weitai_yangweishu_external_manifest.json"),
        help="Output import manifest JSON.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report counts without writing files.")
    return parser.parse_args()


def main() -> None:
    summary = import_answers(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
