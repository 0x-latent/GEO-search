from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import yaml

from ..core.paths import CONFIG_DIR, DATA_DIR

USER_CONFIGS_DIR = DATA_DIR / "user_configs"

GLOBAL_BRANDS_PATH = CONFIG_DIR / "brands.yaml"
GLOBAL_KB_PATH = CONFIG_DIR / "knowledge_base.json"

MAX_CONFIG_BYTES = 2 * 1024 * 1024


def _user_dir(username: str) -> Path:
    normalized = username.strip()
    if not normalized:
        raise ValueError("非法用户名")
    # 门户用户名可能包含点号、斜线或非 ASCII 字符；直接清洗会把不同用户
    # 映射到同一目录（例如 a.b 与 ab），因此使用不可逆且碰撞概率可忽略的键。
    key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return USER_CONFIGS_DIR / key


def _legacy_user_dir(username: str) -> Path:
    """旧版目录键，仅用于读取/清理历史配置。新写入一律使用哈希键。"""
    safe = "".join(c for c in username.strip() if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError("非法用户名")
    return USER_CONFIGS_DIR / safe


def _user_config_path(username: str, filename: str) -> Path:
    current = _user_dir(username) / filename
    if current.exists():
        return current
    legacy = _legacy_user_dir(username) / filename
    return legacy if legacy.exists() else current


def user_brands_path(username: str) -> Path | None:
    """用户自定义 brands.yaml 的路径；未自定义时返回 None（表示用全局默认）。"""
    path = _user_config_path(username, "brands.yaml")
    return path if path.exists() else None


def user_kb_path(username: str) -> Path | None:
    path = _user_config_path(username, "knowledge_base.json")
    return path if path.exists() else None


def load_effective_brands(username: str) -> dict[str, Any]:
    path = user_brands_path(username) or GLOBAL_BRANDS_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"source": "user" if path != GLOBAL_BRANDS_PATH else "default", "data": data}


def load_effective_kb(username: str) -> dict[str, Any]:
    path = user_kb_path(username) or GLOBAL_KB_PATH
    if not path.exists():
        return {"source": "default", "data": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"source": "user" if path != GLOBAL_KB_PATH else "default", "data": data}


def save_user_brands(username: str, data: dict[str, Any]) -> None:
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ValueError("配置过大（超过 2MB）")
    # 先校验能被 pipeline 消费的关键结构
    if not isinstance(data, dict):
        raise ValueError("brands 配置必须是对象")
    directory = _user_dir(username)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "brands.yaml").write_text(text, encoding="utf-8")


def save_user_kb(username: str, data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("knowledge_base 必须是 JSON 对象")
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise ValueError("配置过大（超过 2MB）")
    directory = _user_dir(username)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "knowledge_base.json").write_text(text, encoding="utf-8")


def reset_user_brands(username: str) -> None:
    paths = {
        _user_dir(username) / "brands.yaml",
        _legacy_user_dir(username) / "brands.yaml",
    }
    for path in paths:
        if path.exists():
            path.unlink()


def reset_user_kb(username: str) -> None:
    paths = {
        _user_dir(username) / "knowledge_base.json",
        _legacy_user_dir(username) / "knowledge_base.json",
    }
    for path in paths:
        if path.exists():
            path.unlink()


def load_global_kb() -> dict[str, Any]:
    if not GLOBAL_KB_PATH.exists():
        return {}
    return json.loads(GLOBAL_KB_PATH.read_text(encoding="utf-8"))


def save_global_kb(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("knowledge_base 必须是 JSON 对象")
    GLOBAL_KB_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
