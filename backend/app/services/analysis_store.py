from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..core.paths import ANALYSIS_DIR


def list_analysis_files() -> list[dict[str, Any]]:
    if not ANALYSIS_DIR.exists():
        return []
    files = []
    for path in sorted(ANALYSIS_DIR.glob("*.csv")):
        files.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
        )
    return files


def _csv_path(filename: str) -> Path:
    path = (ANALYSIS_DIR / filename).resolve()
    if not str(path).startswith(str(ANALYSIS_DIR.resolve())):
        raise ValueError("Invalid analysis file path")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported")
    return path


def read_csv_preview(filename: str, limit: int = 100) -> dict[str, Any]:
    path = _csv_path(filename)
    if not path.exists():
        return {"columns": [], "rows": [], "total": 0}
    df = pd.read_csv(path, encoding="utf-8-sig")
    total = len(df)
    df = df.head(max(1, min(limit, 1000))).fillna("")
    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "total": total,
    }


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_bi_overview() -> dict[str, Any]:
    overview: dict[str, Any] = {
        "cards": [],
        "product_metrics": [],
        "model_metrics": [],
        "top_competitors": [],
        "source_domains": [],
    }

    raw_path = ANALYSIS_DIR / "raw_data.csv"
    mention_path = ANALYSIS_DIR / "mention_report.csv"
    rec_path = ANALYSIS_DIR / "rec_overview.csv"
    source_path = ANALYSIS_DIR / "source_report.csv"

    if raw_path.exists():
        raw = pd.read_csv(raw_path, encoding="utf-8-sig")
        overview["cards"].extend(
            [
                {"label": "原始回答", "value": int(len(raw))},
                {"label": "产品数", "value": int(raw["产品"].nunique()) if "产品" in raw.columns else 0},
                {"label": "模型数", "value": int(raw["模型"].nunique()) if "模型" in raw.columns else 0},
            ]
        )

    if mention_path.exists():
        mention = pd.read_csv(mention_path, encoding="utf-8-sig").fillna("")
        product_col = _find_col(mention, ["产品", "浜у搧"])
        model_col = _find_col(mention, ["模型", "妯″瀷"])
        search_col = _find_col(mention, ["联网", "鑱旂綉"])
        total_col = _find_col(mention, ["总回答数", "鎬诲洖绛旀暟"])
        brand_rate_col = _find_col(mention, ["999品牌提及率", "999鍝佺墝鎻愬強鐜?"])
        brand_rec_col = _find_col(mention, ["999品牌推荐率", "999鍝佺墝鎺ㄨ崘鐜?"])
        generic_rate_col = _find_col(mention, ["通用名提及率", "閫氱敤鍚嶆彁鍙婄巼"])

        if product_col:
            rows = []
            for product, group in mention.groupby(product_col):
                rows.append(
                    {
                        "product": product,
                        "answers": int(pd.to_numeric(group.get(total_col, 0), errors="coerce").sum()) if total_col else 0,
                        "brandMentionRate": float(pd.to_numeric(group.get(brand_rate_col, 0), errors="coerce").mean()) if brand_rate_col else 0,
                        "brandRecommendationRate": float(pd.to_numeric(group.get(brand_rec_col, 0), errors="coerce").mean()) if brand_rec_col else 0,
                        "genericMentionRate": float(pd.to_numeric(group.get(generic_rate_col, 0), errors="coerce").mean()) if generic_rate_col else 0,
                    }
                )
            overview["product_metrics"] = sorted(rows, key=lambda r: r["brandRecommendationRate"], reverse=True)

        if model_col:
            rows = []
            group_cols = [model_col] + ([search_col] if search_col else [])
            for key, group in mention.groupby(group_cols):
                if not isinstance(key, tuple):
                    key = (key, "")
                rows.append(
                    {
                        "model": key[0],
                        "search": key[1] if len(key) > 1 else "",
                        "brandMentionRate": float(pd.to_numeric(group.get(brand_rate_col, 0), errors="coerce").mean()) if brand_rate_col else 0,
                        "brandRecommendationRate": float(pd.to_numeric(group.get(brand_rec_col, 0), errors="coerce").mean()) if brand_rec_col else 0,
                    }
                )
            overview["model_metrics"] = rows

    if rec_path.exists():
        rec = pd.read_csv(rec_path, encoding="utf-8-sig").fillna("")
        name_col = _find_col(rec, ["被推荐产品", "琚帹鑽愪骇鍝?"])
        type_col = _find_col(rec, ["名称类型", "鍚嶇О绫诲瀷"])
        mention_col = _find_col(rec, ["提及次数", "鎻愬強娆℃暟"])
        if name_col:
            filtered = rec
            if type_col:
                filtered = rec[~rec[type_col].astype(str).str.contains("999", na=False)]
            grouped = filtered.groupby(name_col)[mention_col].sum() if mention_col else filtered.groupby(name_col).size()
            overview["top_competitors"] = [
                {"name": str(name), "mentions": int(value)}
                for name, value in grouped.sort_values(ascending=False).head(12).items()
            ]

    if source_path.exists():
        source = pd.read_csv(source_path, encoding="utf-8-sig").fillna("")
        domain_col = _find_col(source, ["domain", "域名", "淇℃伅婧愬煙鍚?"])
        count_col = _find_col(source, ["count", "引用次数", "寮曠敤娆℃暟"])
        if domain_col:
            grouped = source.groupby(domain_col)[count_col].sum() if count_col else source.groupby(domain_col).size()
            overview["source_domains"] = [
                {"domain": str(domain), "count": int(value)}
                for domain, value in grouped.sort_values(ascending=False).head(10).items()
            ]

    return overview
