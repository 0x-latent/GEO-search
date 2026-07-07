from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..core.paths import CONFIG_DIR, RUNS_DIR


BRANDS_PATH = CONFIG_DIR / "brands.yaml"
MODELS_PATH = CONFIG_DIR / "models.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, indent=2)


def load_brands() -> dict[str, Any]:
    return _read_yaml(BRANDS_PATH)


def save_brands(data: dict[str, Any]) -> dict[str, str]:
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = RUNS_DIR / f"brands_{version}.yaml"
    _write_yaml(snapshot_path, data)
    _write_yaml(BRANDS_PATH, data)
    from . import product_master  # 延迟导入避免循环依赖

    product_master.sync_products_from_brands(data)
    return {"version": version, "snapshot": str(snapshot_path)}


def load_models() -> dict[str, Any]:
    return _read_yaml(MODELS_PATH)
