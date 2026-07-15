"""External contributor workspaces, submissions and review administration."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Any
from uuid import uuid4

from ..core.paths import DATA_DIR, GEO_SQLITE_PATH
from .document_extract import validate_document
from .outbound_article_store import url_match_key
from .product_master import list_products


CONTRIBUTOR_COOKIE = "geo_contributor_session"
SUBMISSION_DIR = DATA_DIR / "article_submissions"
SESSION_TTL_SECONDS = 7 * 24 * 3600
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ALLOWED_STATUS = {
    "queued", "reviewing", "awaiting_admin", "revision_requested",
    "approved_waiting_publication", "rejected", "tracked",
    "review_failed", "blocked_missing_kb",
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(GEO_SQLITE_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now_dt()).isoformat(timespec="seconds")


def _parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None, fallback: Any) -> Any:
    try:
        return json.loads(value or "")
    except (json.JSONDecodeError, TypeError):
        return fallback


def _event(
    conn: sqlite3.Connection, submission_id: str, actor_type: str,
    actor_id: str | None, action: str, from_status: str | None,
    to_status: str | None, details: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """INSERT INTO article_submission_events
        (event_id, submission_id, actor_type, actor_id, action, from_status,
         to_status, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (f"evt_{uuid4().hex}", submission_id, actor_type, actor_id, action,
         from_status, to_status, _json(details or {}), _iso()),
    )


def create_company(
    name: str, created_by: str, contact_name: str | None = None,
    contact_email: str | None = None,
) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise ValueError("请填写公司名称")
    company_id = f"cmp_{uuid4().hex}"
    now = _iso()
    try:
        with closing(_connect()) as conn:
            conn.execute(
                """INSERT INTO contributor_companies
                (company_id,name,contact_name,contact_email,is_active,created_by,created_at,updated_at)
                VALUES (?,?,?,?,1,?,?,?)""",
                (company_id, name, (contact_name or "").strip() or None,
                 (contact_email or "").strip() or None, created_by, now, now),
            )
            conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("公司名称已存在") from exc
    return get_company(company_id)


def get_company(company_id: str) -> dict[str, Any]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM contributor_companies WHERE company_id=?", (company_id,)
        ).fetchone()
    if not row:
        raise ValueError("公司不存在")
    return dict(row)


def list_companies() -> list[dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """SELECT c.*,
            (SELECT count(*) FROM article_submissions s WHERE s.company_id=c.company_id) submission_count,
            (SELECT count(*) FROM contributor_invites i WHERE i.company_id=c.company_id
             AND i.revoked_at IS NULL) invite_count
            FROM contributor_companies c ORDER BY c.created_at DESC"""
        ).fetchall()
    return [dict(row) for row in rows]


def update_company(company_id: str, **values: Any) -> dict[str, Any]:
    allowed = {"name", "contact_name", "contact_email", "is_active"}
    fields = {key: value for key, value in values.items() if key in allowed}
    if not fields:
        return get_company(company_id)
    sets = ", ".join(f"{key}=?" for key in fields)
    with closing(_connect()) as conn:
        cur = conn.execute(
            f"UPDATE contributor_companies SET {sets}, updated_at=? WHERE company_id=?",
            (*fields.values(), _iso(), company_id),
        )
        if not cur.rowcount:
            raise ValueError("公司不存在")
        conn.commit()
    return get_company(company_id)


def create_invite(
    company_id: str, created_by: str, allowed_product_codes: list[str],
    expires_at: str, max_submissions: int = 20,
) -> dict[str, Any]:
    company = get_company(company_id)
    if not company["is_active"]:
        raise ValueError("该公司已停用")
    expires = _parse_time(expires_at)
    if expires <= _now_dt():
        raise ValueError("邀请有效期必须晚于当前时间")
    if not 1 <= max_submissions <= 10000:
        raise ValueError("最大投稿数必须在 1 到 10000 之间")
    known = {item["product_code"] for item in list_products()}
    requested = list(dict.fromkeys(code.strip() for code in allowed_product_codes if code.strip()))
    if requested and not set(requested).issubset(known):
        raise ValueError("邀请中包含无效产品")
    token = secrets.token_urlsafe(32)
    invite_id = f"inv_{uuid4().hex}"
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT INTO contributor_invites
            (invite_id,company_id,token_hash,allowed_product_codes_json,expires_at,
             max_submissions,submission_count,created_by,created_at)
            VALUES (?,?,?,?,?,?,0,?,?)""",
            (invite_id, company_id, _hash_token(token), _json(requested),
             _iso(expires), max_submissions, created_by, _iso()),
        )
        conn.commit()
    return {"invite_id": invite_id, "token": token, "company_id": company_id,
            "company_name": company["name"], "expires_at": _iso(expires),
            "max_submissions": max_submissions, "allowed_product_codes": requested}


def list_invites(company_id: str | None = None) -> list[dict[str, Any]]:
    params: tuple[Any, ...] = ()
    where = ""
    if company_id:
        where, params = "WHERE i.company_id=?", (company_id,)
    with closing(_connect()) as conn:
        rows = conn.execute(
            f"""SELECT i.invite_id,i.company_id,c.name company_name,
            i.allowed_product_codes_json,i.expires_at,i.max_submissions,
            i.submission_count,i.revoked_at,i.created_by,i.created_at,i.last_used_at
            FROM contributor_invites i JOIN contributor_companies c USING(company_id)
            {where} ORDER BY i.created_at DESC""", params,
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["allowed_product_codes"] = _loads(item.pop("allowed_product_codes_json"), [])
        item["active"] = not item["revoked_at"] and _parse_time(item["expires_at"]) > _now_dt()
        result.append(item)
    return result


def revoke_invite(invite_id: str) -> None:
    with closing(_connect()) as conn:
        cur = conn.execute(
            "UPDATE contributor_invites SET revoked_at=? WHERE invite_id=? AND revoked_at IS NULL",
            (_iso(), invite_id),
        )
        if not cur.rowcount:
            raise ValueError("邀请不存在或已撤销")
        conn.execute("DELETE FROM contributor_sessions WHERE invite_id=?", (invite_id,))
        conn.commit()


def exchange_invite(invite_id: str, token: str) -> tuple[str, dict[str, Any], int]:
    with closing(_connect()) as conn:
        row = conn.execute(
            """SELECT i.*,c.name company_name,c.is_active company_active
            FROM contributor_invites i JOIN contributor_companies c USING(company_id)
            WHERE i.invite_id=?""", (invite_id,),
        ).fetchone()
        if not row or not secrets.compare_digest(row["token_hash"], _hash_token(token)):
            raise ValueError("邀请链接无效")
        if row["revoked_at"] or not row["company_active"]:
            raise ValueError("邀请已撤销")
        expires = _parse_time(row["expires_at"])
        if expires <= _now_dt():
            raise ValueError("邀请已过期")
        if row["submission_count"] >= row["max_submissions"]:
            raise ValueError("邀请投稿次数已用完")
        session = secrets.token_urlsafe(32)
        session_expiry = min(expires, _now_dt() + timedelta(seconds=SESSION_TTL_SECONDS))
        conn.execute(
            """INSERT INTO contributor_sessions
            (session_hash,invite_id,expires_at,created_at,last_seen_at) VALUES (?,?,?,?,?)""",
            (_hash_token(session), invite_id, _iso(session_expiry), _iso(), _iso()),
        )
        conn.execute("UPDATE contributor_invites SET last_used_at=? WHERE invite_id=?", (_iso(), invite_id))
        conn.commit()
    return session, _workspace(dict(row)), max(1, int((session_expiry - _now_dt()).total_seconds()))


def _workspace(invite: dict[str, Any]) -> dict[str, Any]:
    allowed = _loads(invite.get("allowed_product_codes_json"), [])
    products = list_products()
    if allowed:
        products = [p for p in products if p["product_code"] in allowed]
    return {
        "invite_id": invite["invite_id"], "company_id": invite["company_id"],
        "company_name": invite.get("company_name"), "expires_at": invite["expires_at"],
        "remaining_submissions": max(0, invite["max_submissions"] - invite["submission_count"]),
        "products": products,
    }


def get_contributor_session(token: str | None, *, touch: bool = True) -> dict[str, Any] | None:
    if not token:
        return None
    with closing(_connect()) as conn:
        row = conn.execute(
            """SELECT s.session_hash,s.expires_at session_expires,i.*,
            c.name company_name,c.is_active company_active
            FROM contributor_sessions s JOIN contributor_invites i USING(invite_id)
            JOIN contributor_companies c USING(company_id)
            WHERE s.session_hash=?""", (_hash_token(token),),
        ).fetchone()
        if not row:
            return None
        if (row["revoked_at"] or not row["company_active"] or
                _parse_time(row["expires_at"]) <= _now_dt() or
                _parse_time(row["session_expires"]) <= _now_dt()):
            conn.execute("DELETE FROM contributor_sessions WHERE session_hash=?", (_hash_token(token),))
            conn.commit()
            return None
        if touch:
            conn.execute("UPDATE contributor_sessions SET last_seen_at=? WHERE session_hash=?", (_iso(), row["session_hash"]))
            conn.commit()
    result = dict(row)
    result["workspace"] = _workspace(result)
    return result


def delete_contributor_session(token: str | None) -> None:
    if not token:
        return
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM contributor_sessions WHERE session_hash=?", (_hash_token(token),))
        conn.commit()


def _store_file(company_id: str, submission_id: str, version: int, filename: str, content: bytes) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)[:120] or "article"
    relative = Path(company_id) / submission_id / f"v{version:03d}" / safe_name
    target = SUBMISSION_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return relative.as_posix()


def _queue_job(conn: sqlite3.Connection, submission_id: str, version: int) -> str:
    job_id = f"rev_{uuid4().hex}"
    now = _iso()
    conn.execute(
        """INSERT INTO article_review_jobs
        (job_id,submission_id,version,status,stage,created_at,updated_at)
        VALUES (?,?,?,'queued','queued',?,?)""",
        (job_id, submission_id, version, now, now),
    )
    return job_id


def create_submission(
    session: dict[str, Any], filename: str, content: bytes, product_code: str,
    title: str, submitter_name: str, submitter_email: str,
    campaign: str | None = None, published_platform: str | None = None,
    published_url: str | None = None, published_at: str | None = None,
) -> dict[str, Any]:
    ext = validate_document(filename, content)
    title, submitter_name, submitter_email = title.strip(), submitter_name.strip(), submitter_email.strip()
    if not title or len(title) > 300:
        raise ValueError("请填写不超过 300 字的标题")
    if not submitter_name:
        raise ValueError("请填写投稿人姓名")
    if not EMAIL_RE.match(submitter_email):
        raise ValueError("请填写有效的投稿人邮箱")
    allowed = _loads(session["allowed_product_codes_json"], [])
    known = {p["product_code"] for p in list_products()}
    if product_code not in known or (allowed and product_code not in allowed):
        raise ValueError("所选产品不在邀请允许范围内")
    platform = (published_platform or "").strip() or None
    url = (published_url or "").strip() or None
    if bool(platform) != bool(url):
        raise ValueError("发布平台和发布链接需要同时填写")
    if url:
        url, _ = url_match_key(url)
    with closing(_connect()) as conn:
        invite_check = conn.execute(
            "SELECT revoked_at,submission_count,max_submissions FROM contributor_invites WHERE invite_id=?",
            (session["invite_id"],),
        ).fetchone()
    if (not invite_check or invite_check["revoked_at"] or
            invite_check["submission_count"] >= invite_check["max_submissions"]):
        raise ValueError("邀请投稿次数已用完或邀请已撤销")
    submission_id = f"sub_{uuid4().hex}"
    relative = _store_file(session["company_id"], submission_id, 1, filename, content)
    now = _iso()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        invite = conn.execute("SELECT * FROM contributor_invites WHERE invite_id=?", (session["invite_id"],)).fetchone()
        if not invite or invite["revoked_at"] or invite["submission_count"] >= invite["max_submissions"]:
            raise ValueError("邀请投稿次数已用完或邀请已撤销")
        conn.execute(
            """INSERT INTO article_submissions
            (submission_id,company_id,invite_id,product_code,title,campaign,submitter_name,
             submitter_email,status,current_version,published_platform,published_url,published_at,
             created_at,updated_at) VALUES (?,?,?,?,?,?,?,?, 'queued',1,?,?,?,?,?)""",
            (submission_id, session["company_id"], session["invite_id"], product_code,
             title, (campaign or "").strip() or None, submitter_name, submitter_email,
             platform, url, published_at, now, now),
        )
        conn.execute(
            """INSERT INTO article_submission_versions
            (submission_id,version,original_filename,file_ext,file_sha256,size_bytes,relative_path,created_at)
            VALUES (?,1,?,?,?,?,?,?)""",
            (submission_id, filename, ext, hashlib.sha256(content).hexdigest(), len(content), relative, now),
        )
        job_id = _queue_job(conn, submission_id, 1)
        conn.execute("UPDATE contributor_invites SET submission_count=submission_count+1 WHERE invite_id=?", (session["invite_id"],))
        _event(conn, submission_id, "contributor", submitter_email, "submitted", None, "queued", {"job_id": job_id})
        conn.commit()
    return get_submission(submission_id, company_id=session["company_id"], external=True)


def add_revision(session: dict[str, Any], submission_id: str, filename: str, content: bytes) -> dict[str, Any]:
    ext = validate_document(filename, content)
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=? AND company_id=?", (submission_id, session["company_id"])).fetchone()
        if not row:
            raise ValueError("投稿不存在")
        if row["status"] not in {"revision_requested", "review_failed", "blocked_missing_kb"}:
            raise ValueError("当前状态不能上传修订版")
        version = row["current_version"] + 1
    relative = _store_file(session["company_id"], submission_id, version, filename, content)
    now = _iso()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=? AND company_id=?", (submission_id, session["company_id"])).fetchone()
        old_status = row["status"]
        conn.execute(
            """INSERT INTO article_submission_versions
            (submission_id,version,original_filename,file_ext,file_sha256,size_bytes,relative_path,created_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (submission_id, version, filename, ext, hashlib.sha256(content).hexdigest(), len(content), relative, now),
        )
        conn.execute("UPDATE article_submissions SET current_version=?,status='queued',admin_feedback=NULL,updated_at=? WHERE submission_id=?", (version, now, submission_id))
        job_id = _queue_job(conn, submission_id, version)
        _event(conn, submission_id, "contributor", None, "revision_uploaded", old_status, "queued", {"version": version, "job_id": job_id})
        conn.commit()
    return get_submission(submission_id, company_id=session["company_id"], external=True)


def update_publication(
    session: dict[str, Any], submission_id: str, platform: str, url: str,
    published_at: str | None = None,
) -> dict[str, Any]:
    platform = platform.strip()
    if not platform:
        raise ValueError("请填写发布平台")
    canonical, _ = url_match_key(url.strip())
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=? AND company_id=?", (submission_id, session["company_id"])).fetchone()
        if not row:
            raise ValueError("投稿不存在")
        conn.execute(
            "UPDATE article_submissions SET published_platform=?,published_url=?,published_at=?,updated_at=? WHERE submission_id=?",
            (platform, canonical, published_at, _iso(), submission_id),
        )
        _event(conn, submission_id, "contributor", None, "publication_updated", row["status"], row["status"])
        conn.commit()
    if row["status"] == "approved_waiting_publication":
        promote_submission(submission_id, row["approved_by"] or "admin")
    return get_submission(submission_id, company_id=session["company_id"], external=True)


def list_submissions(company_id: str | None = None, status: str | None = None, *, external: bool = False) -> list[dict[str, Any]]:
    clauses, params = [], []
    if company_id:
        clauses.append("s.company_id=?"); params.append(company_id)
    if status:
        clauses.append("s.status=?"); params.append(status)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with closing(_connect()) as conn:
        rows = conn.execute(
            f"""SELECT s.*,c.name company_name,j.job_id,j.status review_job_status,j.stage review_stage,
            j.progress review_progress,j.error_message review_error
            FROM article_submissions s JOIN contributor_companies c USING(company_id)
            LEFT JOIN article_review_jobs j ON j.submission_id=s.submission_id AND j.version=s.current_version
            {where} ORDER BY s.updated_at DESC""", params,
        ).fetchall()
    return [_submission_public(dict(row), external) for row in rows]


def _submission_public(item: dict[str, Any], external: bool) -> dict[str, Any]:
    if external:
        for key in ("approved_by", "rejected_by", "review_error", "content_text"):
            item.pop(key, None)
        for version in item.get("versions", []):
            version.pop("parse_error", None)
        for event in item.get("events", []):
            event.pop("details_json", None)
    return item


def get_submission(submission_id: str, company_id: str | None = None, *, external: bool = False) -> dict[str, Any]:
    clauses, params = ["s.submission_id=?"], [submission_id]
    if company_id:
        clauses.append("s.company_id=?"); params.append(company_id)
    with closing(_connect()) as conn:
        row = conn.execute(
            f"""SELECT s.*,c.name company_name,j.job_id,j.status review_job_status,
            j.stage review_stage,j.progress review_progress,j.error_message review_error
            ,(SELECT content_text FROM article_submission_versions cv
              WHERE cv.submission_id=s.submission_id AND cv.version=s.current_version) content_text
            FROM article_submissions s JOIN contributor_companies c USING(company_id)
            LEFT JOIN article_review_jobs j ON j.submission_id=s.submission_id AND j.version=s.current_version
            WHERE {' AND '.join(clauses)}""", params,
        ).fetchone()
        if not row:
            raise ValueError("投稿不存在")
        result = dict(row)
        result["versions"] = [dict(r) for r in conn.execute(
            """SELECT version,original_filename,file_ext,file_sha256,size_bytes,content_sha256,
            parse_error,created_at FROM article_submission_versions WHERE submission_id=? ORDER BY version DESC""",
            (submission_id,),
        )]
        result["events"] = [dict(r) for r in conn.execute(
            """SELECT actor_type,action,from_status,to_status,details_json,created_at
            FROM article_submission_events WHERE submission_id=? ORDER BY created_at DESC""", (submission_id,),
        )]
        if result.get("job_id"):
            visibility = "AND external_visible=1" if external else ""
            result["findings"] = [dict(r) for r in conn.execute(
                f"SELECT * FROM article_review_findings WHERE job_id=? {visibility} ORDER BY blocks_publication DESC,severity DESC",
                (result["job_id"],),
            )]
            if not external:
                result["similarities"] = [dict(r) for r in conn.execute(
                    "SELECT * FROM article_similarity_matches WHERE job_id=? ORDER BY exact_hash DESC,coalesce(semantic_score,lexical_score) DESC",
                    (result["job_id"],),
                )]
                for match in result["similarities"]:
                    if match["matched_kind"] == "article":
                        source = conn.execute(
                            "SELECT title,content_text FROM outbound_articles WHERE article_id=?",
                            (match["matched_id"],),
                        ).fetchone()
                    else:
                        source = conn.execute(
                            """SELECT s.title,v.content_text FROM article_submission_versions v
                            JOIN article_submissions s USING(submission_id)
                            WHERE v.submission_id=? AND v.version=?""",
                            (match["matched_id"], match["matched_version"]),
                        ).fetchone()
                    match["matched_title"] = source["title"] if source else None
                    match["matched_content_text"] = source["content_text"] if source else None
                report = conn.execute("SELECT * FROM article_review_reports WHERE job_id=?", (result["job_id"],)).fetchone()
                result["report"] = dict(report) if report else None
    return _submission_public(result, external)


def review_action(
    submission_id: str, action: str, admin: str, feedback: str | None = None,
    visible_finding_ids: list[str] | None = None,
) -> dict[str, Any]:
    mapping = {"request_revision": "revision_requested", "reject": "rejected"}
    if action not in {*mapping, "approve"}:
        raise ValueError("无效审核操作")
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=?", (submission_id,)).fetchone()
        if not row:
            raise ValueError("投稿不存在")
        if row["status"] != "awaiting_admin":
            raise ValueError("仅待管理员确认的投稿可以执行此操作")
        if action == "approve":
            # Promotion is a second atomic transaction. Keep the submission in
            # the recoverable waiting state until the formal article is created.
            new_status = "approved_waiting_publication"
            conn.execute("UPDATE article_submissions SET status=?,approved_by=?,approved_at=?,admin_feedback=?,updated_at=? WHERE submission_id=?", (new_status, admin, _iso(), feedback, _iso(), submission_id))
        else:
            new_status = mapping[action]
            rejected = action == "reject"
            conn.execute("""UPDATE article_submissions SET status=?,admin_feedback=?,
                rejected_by=?,rejected_at=?,updated_at=? WHERE submission_id=?""",
                (new_status, feedback, admin if rejected else None, _iso() if rejected else None, _iso(), submission_id))
        conn.execute("UPDATE article_review_findings SET external_visible=0 WHERE job_id=(SELECT job_id FROM article_review_jobs WHERE submission_id=? AND version=?)", (submission_id, row["current_version"]))
        ids = visible_finding_ids or []
        if ids:
            marks = ",".join("?" for _ in ids)
            conn.execute(f"UPDATE article_review_findings SET external_visible=1 WHERE job_id=(SELECT job_id FROM article_review_jobs WHERE submission_id=? AND version=?) AND finding_id IN ({marks})", (submission_id, row["current_version"], *ids))
        _event(conn, submission_id, "admin", admin, action, row["status"], new_status, {"feedback": feedback or "", "visible_findings": ids})
        conn.commit()
    if action == "approve" and row["published_url"]:
        promote_submission(submission_id, admin)
    return get_submission(submission_id)


def promote_submission(submission_id: str, approved_by: str) -> str:
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("""SELECT s.*,v.original_filename,v.file_ext,v.file_sha256,v.size_bytes,
            v.content_text,v.content_sha256 FROM article_submissions s
            JOIN article_submission_versions v ON v.submission_id=s.submission_id AND v.version=s.current_version
            WHERE s.submission_id=?""", (submission_id,)).fetchone()
        if not row or not row["published_url"] or not row["content_text"]:
            raise ValueError("投稿缺少发布链接或可解析正文")
        if row["article_id"]:
            return row["article_id"]
        canonical, match_key = url_match_key(row["published_url"])
        article_id, publication_id, now = f"art_{uuid4().hex}", f"pub_{uuid4().hex}", _iso()
        conn.execute("""INSERT INTO outbound_articles
            (article_id,owner_username,title,content_text,content_sha256,product_code,campaign,
             source_filename,file_ext,file_sha256,size_bytes,created_at,metadata_json,
             company_id,submission_id,approved_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (article_id, approved_by, row["title"], row["content_text"], row["content_sha256"],
             row["product_code"], row["campaign"], row["original_filename"], row["file_ext"],
             row["file_sha256"], row["size_bytes"], now, "{}", row["company_id"], submission_id, approved_by))
        conn.execute("""INSERT INTO article_publications
            (publication_id,article_id,platform,url,canonical_url,url_match_key,published_at,created_at,metadata_json)
            VALUES (?,?,?,?,?,?,?,?, '{}')""",
            (publication_id, article_id, row["published_platform"], row["published_url"], canonical,
             match_key, row["published_at"], now))
        conn.execute("UPDATE article_submissions SET article_id=?,status='tracked',updated_at=? WHERE submission_id=?", (article_id, now, submission_id))
        _event(conn, submission_id, "system", approved_by, "promoted_to_tracking", row["status"], "tracked", {"article_id": article_id})
        conn.commit()
    return article_id


def retry_review(submission_id: str, admin: str) -> None:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=?", (submission_id,)).fetchone()
        if not row or row["status"] not in {"review_failed", "blocked_missing_kb"}:
            raise ValueError("当前任务不能重试")
        conn.execute("""UPDATE article_review_jobs SET status='queued',stage='queued',error_message=NULL,
            lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE submission_id=? AND version=?""", (_iso(), submission_id, row["current_version"]))
        conn.execute("UPDATE article_submissions SET status='queued',updated_at=? WHERE submission_id=?", (_iso(), submission_id))
        _event(conn, submission_id, "admin", admin, "review_retried", row["status"], "queued")
        conn.commit()


def cancel_review(submission_id: str, admin: str) -> None:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=?", (submission_id,)).fetchone()
        if not row or row["status"] not in {"queued", "reviewing"}:
            raise ValueError("当前任务不能取消")
        conn.execute("UPDATE article_review_jobs SET status='cancelled',stage='cancelled',updated_at=? WHERE submission_id=? AND version=?", (_iso(), submission_id, row["current_version"]))
        conn.execute("UPDATE article_submissions SET status='review_failed',updated_at=? WHERE submission_id=?", (_iso(), submission_id))
        _event(conn, submission_id, "admin", admin, "review_cancelled", row["status"], "review_failed")
        conn.commit()


def get_review_settings() -> dict[str, Any]:
    cap = max(1, min(100, int(__import__("os").environ.get("GEO_ARTICLE_REVIEW_CONCURRENCY_MAX", "5"))))
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM article_review_settings WHERE id=1").fetchone()
    result = dict(row)
    result["environment_max"] = cap
    result["effective_concurrency"] = min(result["ai_concurrency"], cap)
    return result


def update_review_settings(values: dict[str, Any], admin: str) -> dict[str, Any]:
    allowed = {"auto_start", "queue_paused", "primary_model_key", "primary_model_id",
               "fallback_model_key", "fallback_model_id", "ai_concurrency",
               "request_timeout_seconds", "retry_count", "similarity_threshold", "similarity_top_k"}
    fields = {key: value for key, value in values.items() if key in allowed}
    concurrency = int(fields.get("ai_concurrency", 5))
    if not 1 <= concurrency <= 100:
        raise ValueError("并发数必须在 1 到 100 之间")
    if "retry_count" in fields and not 0 <= int(fields["retry_count"]) <= 10:
        raise ValueError("重试次数必须在 0 到 10 之间")
    if "request_timeout_seconds" in fields and not 10 <= int(fields["request_timeout_seconds"]) <= 900:
        raise ValueError("超时时间必须在 10 到 900 秒之间")
    if "similarity_threshold" in fields and not .4 <= float(fields["similarity_threshold"]) <= .95:
        raise ValueError("相似度阈值必须在 0.4 到 0.95 之间")
    if "similarity_top_k" in fields and not 1 <= int(fields["similarity_top_k"]) <= 50:
        raise ValueError("Top K 必须在 1 到 50 之间")
    sets = ",".join(f"{key}=?" for key in fields)
    with closing(_connect()) as conn:
        conn.execute(f"UPDATE article_review_settings SET {sets},updated_by=?,updated_at=? WHERE id=1", (*fields.values(), admin, _iso()))
        conn.commit()
    return get_review_settings()


def review_dashboard() -> dict[str, Any]:
    settings = get_review_settings()
    with closing(_connect()) as conn:
        counts = {row["status"]: row["n"] for row in conn.execute("SELECT status,count(*) n FROM article_review_jobs GROUP BY status")}
        running = conn.execute("SELECT count(*) FROM article_review_jobs WHERE status='running'").fetchone()[0]
        average = conn.execute("SELECT avg(duration_ms) FROM article_review_reports").fetchone()[0]
        worker = conn.execute("SELECT * FROM article_review_worker_state ORDER BY heartbeat_at DESC LIMIT 1").fetchone()
        error = conn.execute("SELECT error_message,updated_at FROM article_review_jobs WHERE error_message IS NOT NULL ORDER BY updated_at DESC LIMIT 1").fetchone()
    return {"counts": counts, "running": running, "average_duration_ms": round(average or 0),
            "settings": settings, "worker": dict(worker) if worker else None,
            "latest_error": dict(error) if error else None}
