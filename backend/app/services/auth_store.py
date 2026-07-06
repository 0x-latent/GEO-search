from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from typing import Any

from ..core.paths import DATA_DIR

AUTH_DB_PATH = DATA_DIR / "auth.sqlite"

SESSION_TTL_SECONDS = 7 * 24 * 3600
PBKDF2_ITERATIONS = 200_000

VALID_ROLES = ("admin", "user")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
    _ensure_default_admin()


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    )
    return digest.hex()


def _ensure_default_admin() -> None:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        if row["n"] == 0:
            password = os.environ.get("GEO_ADMIN_PASSWORD", "admin123")
            salt = secrets.token_hex(16)
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, 'admin', ?)",
                ("admin", _hash_password(password, salt), salt, time.time()),
            )


def verify_credentials(username: str, password: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, password_hash, salt, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        # 用相同耗时的哈希避免用户名枚举的时间侧信道
        _hash_password(password, secrets.token_hex(16))
        return None
    if not secrets.compare_digest(row["password_hash"], _hash_password(password, row["salt"])):
        return None
    return {"username": row["username"], "role": row["role"]}


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        conn.execute(
            "INSERT INTO sessions (token, username, expires_at) VALUES (?, ?, ?)",
            (token, username, time.time() + SESSION_TTL_SECONDS),
        )
    return token


def get_session_user(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT u.username, u.role FROM sessions s
            JOIN users u ON u.username = s.username
            WHERE s.token = ? AND s.expires_at > ?
            """,
            (token, time.time()),
        ).fetchone()
    if row is None:
        return None
    return {"username": row["username"], "role": row["role"]}


def delete_session(token: str | None) -> None:
    if not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def list_users() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT username, role, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(row) for row in rows]


def create_user(username: str, password: str, role: str) -> dict[str, Any]:
    username = username.strip()
    if not username:
        raise ValueError("用户名不能为空")
    if len(password) < 6:
        raise ValueError("密码至少 6 位")
    if role not in VALID_ROLES:
        raise ValueError(f"角色必须是 {VALID_ROLES} 之一")
    salt = secrets.token_hex(16)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, _hash_password(password, salt), salt, role, time.time()),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"用户 {username} 已存在") from exc
    return {"username": username, "role": role}


def delete_user(username: str, operator: str) -> None:
    if username == operator:
        raise ValueError("不能删除当前登录的账号")
    with _connect() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            raise ValueError(f"用户 {username} 不存在")
        if row["role"] == "admin":
            admins = conn.execute(
                "SELECT COUNT(*) AS n FROM users WHERE role = 'admin'"
            ).fetchone()
            if admins["n"] <= 1:
                raise ValueError("至少需要保留一个管理员账号")
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.execute("DELETE FROM sessions WHERE username = ?", (username,))


def set_password(username: str, password: str) -> None:
    if len(password) < 6:
        raise ValueError("密码至少 6 位")
    salt = secrets.token_hex(16)
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
            (_hash_password(password, salt), salt, username),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"用户 {username} 不存在")
        conn.execute("DELETE FROM sessions WHERE username = ?", (username,))


def set_role(username: str, role: str, operator: str) -> None:
    if role not in VALID_ROLES:
        raise ValueError(f"角色必须是 {VALID_ROLES} 之一")
    if username == operator and role != "admin":
        raise ValueError("不能取消自己的管理员权限")
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE users SET role = ? WHERE username = ?", (role, username)
        )
        if cursor.rowcount == 0:
            raise ValueError(f"用户 {username} 不存在")
