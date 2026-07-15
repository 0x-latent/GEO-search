"""外发文章导入、发布记录及 AI 信源引用匹配。"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable
from urllib.parse import urlsplit
from uuid import uuid4
import xml.etree.ElementTree as ET
import zipfile

from ..core.paths import GEO_SQLITE_PATH
from .source_insight_store import normalize_url


ALLOWED_EXTS = {".md", ".markdown", ".txt", ".docx", ".pdf"}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_CONTENT_CHARS = 2_000_000
MAX_PDF_PAGES = 200


def _connect() -> sqlite3.Connection:
    if not GEO_SQLITE_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {GEO_SQLITE_PATH}")
    conn = sqlite3.connect(GEO_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _extract_docx(content: bytes) -> str:
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(content)) as archive:
            try:
                info = archive.getinfo("word/document.xml")
            except KeyError as exc:
                raise ValueError("DOCX 中缺少正文内容") from exc
            if info.file_size > MAX_FILE_BYTES * 4:
                raise ValueError("DOCX 解压后的正文过大")
            root = ET.fromstring(archive.read(info))
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError("DOCX 文件损坏或格式无效") from exc

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines: list[str] = []
    for paragraph in root.iter(f"{ns}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{ns}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{ns}tab":
                parts.append("\t")
            elif node.tag in {f"{ns}br", f"{ns}cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_pdf(content: bytes) -> str:
    try:
        import fitz

        with fitz.open(stream=content, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ValueError("暂不支持加密 PDF")
            if doc.page_count > MAX_PDF_PAGES:
                raise ValueError(f"PDF 超过 {MAX_PDF_PAGES} 页，请拆分后上传")
            pages = [page.get_text("text") for page in doc]
    except ValueError:
        raise
    except Exception as exc:  # PyMuPDF 会抛出多种格式异常
        raise ValueError("PDF 文件损坏或无法读取") from exc
    return "\n\n".join(page.strip() for page in pages if page.strip())


def extract_article_text(filename: str, content: bytes) -> tuple[str, str]:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError("不支持该文件类型（支持 MD / DOCX / PDF）")
    if not content:
        raise ValueError("上传文件为空")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("文件超过 20MB")
    if ext in {".md", ".markdown", ".txt"}:
        text = _decode_text(content)
    elif ext == ".docx":
        text = _extract_docx(content)
    else:
        text = _extract_pdf(content)
    text = text.replace("\x00", "").strip()
    if not text and ext != ".pdf":
        raise ValueError("未能从文件中提取到文字")
    if len(text) > MAX_CONTENT_CHARS:
        raise ValueError("文档正文超过 200 万字，请拆分后上传")
    return ext, text


def infer_title(filename: str, text: str) -> str:
    for raw in text.splitlines()[:30]:
        line = raw.strip().lstrip("#").strip()
        if 2 <= len(line) <= 200:
            return line
    return Path(filename).stem[:200] or "未命名文章"


def url_match_key(url: str) -> tuple[str, str]:
    canonical = normalize_url(url)
    try:
        parsed = urlsplit(canonical)
    except ValueError as exc:
        raise ValueError("发布链接格式无效") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("发布链接必须是 http/https URL")
    key = f"{parsed.netloc.lower()}{parsed.path or '/'}"
    if parsed.query:
        key += f"?{parsed.query}"
    return canonical, key


def _as_naive_utc(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def create_article(
    *, username: str, filename: str, content: bytes, platform: str, url: str,
    published_at: str | None = None, title: str | None = None,
    product_code: str | None = None, campaign: str | None = None,
) -> dict[str, Any]:
    platform = platform.strip()
    if not platform:
        raise ValueError("请填写发布平台")
    ext, text = extract_article_text(filename, content)
    canonical_url, match_key = url_match_key(url.strip())
    article_title = (title or "").strip() or infer_title(filename, text)
    if len(article_title) > 300:
        raise ValueError("文章标题不能超过 300 字")
    if published_at and _as_naive_utc(published_at) is None:
        raise ValueError("发布时间格式无效")

    file_sha256 = hashlib.sha256(content).hexdigest()
    content_sha256 = hashlib.sha256(text.encode("utf-8") if text else content).hexdigest()
    article_id = f"art_{uuid4().hex}"
    publication_id = f"pub_{uuid4().hex}"
    created_at = _now()
    with closing(_connect()) as conn:
        duplicate = conn.execute(
            """
            SELECT oa.article_id
            FROM article_publications ap
            JOIN outbound_articles oa ON oa.article_id = ap.article_id
            WHERE oa.owner_username = ? AND ap.url_match_key = ?
            """,
            (username, match_key),
        ).fetchone()
        if duplicate:
            raise ValueError("该发布链接已经导入")
        existing = conn.execute(
            """
            SELECT article_id FROM outbound_articles
            WHERE owner_username = ? AND content_sha256 = ?
            ORDER BY created_at LIMIT 1
            """,
            (username, content_sha256),
        ).fetchone()
        if existing:
            article_id = existing["article_id"]
        else:
            conn.execute(
                """
                INSERT INTO outbound_articles (
                    article_id, owner_username, title, content_text, content_sha256,
                    product_code, campaign, source_filename, file_ext, file_sha256,
                    size_bytes, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id, username, article_title, text, content_sha256,
                    (product_code or "").strip() or None,
                    (campaign or "").strip() or None,
                    Path(filename).name, ext, file_sha256, len(content), created_at,
                    json.dumps(
                        {"text_extraction": "success" if text else "empty_scanned_pdf"},
                        ensure_ascii=False,
                    ),
                ),
            )
        conn.execute(
            """
            INSERT INTO article_publications (
                publication_id, article_id, platform, url, canonical_url,
                url_match_key, published_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                publication_id, article_id, platform, url.strip(), canonical_url,
                match_key, published_at or None, created_at,
            ),
        )
        conn.commit()
    return get_article(article_id, username=username)


def _scope_clause(allowed: list[str] | None, column: str, params: list[Any]) -> str:
    if allowed is None:
        return "1 = 1"
    if not allowed:
        return "1 = 0"
    params.extend(allowed)
    return f"{column} IN ({','.join('?' for _ in allowed)})"


def _in_clause(values: Iterable[Any] | None, column: str, params: list[Any]) -> str | None:
    clean = [value for value in (values or []) if value not in (None, "")]
    if not clean:
        return None
    params.extend(clean)
    return f"{column} IN ({','.join('?' for _ in clean)})"


def refresh_matches(username: str | None, allowed: list[str] | None) -> int:
    """重建可见文章的精确 URL 匹配，保证新导入的回答批次会自动纳入。"""
    with closing(_connect()) as conn:
        article_params: list[Any] = []
        article_where = "1 = 1"
        if username is not None:
            article_where = "oa.owner_username = ?"
            article_params.append(username)
        publications = [dict(row) for row in conn.execute(
            f"""
            SELECT ap.*, oa.owner_username
            FROM article_publications ap
            JOIN outbound_articles oa ON oa.article_id = ap.article_id
            WHERE {article_where}
            """,
            article_params,
        )]
        publication_ids = [row["publication_id"] for row in publications]
        if not publication_ids:
            return 0
        conn.execute(
            f"DELETE FROM source_article_matches WHERE publication_id IN ({','.join('?' for _ in publication_ids)})",
            publication_ids,
        )

        source_params: list[Any] = []
        source_where = _scope_clause(allowed, "s.dataset_id", source_params)
        sources = conn.execute(
            f"""
            SELECT s.dataset_id, s.answer_id, s.source_index, s.url,
                   a.timestamp AS answer_timestamp
            FROM sources s
            JOIN answers a ON a.dataset_id = s.dataset_id AND a.answer_id = s.answer_id
            WHERE {source_where} AND COALESCE(s.url, '') <> ''
            """,
            source_params,
        ).fetchall()
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for publication in publications:
            by_key[publication["url_match_key"]].append(publication)

        inserts: list[tuple[Any, ...]] = []
        matched_at = _now()
        for source in sources:
            try:
                _, key = url_match_key(source["url"])
            except ValueError:
                continue
            for publication in by_key.get(key, []):
                answer_dt = _as_naive_utc(source["answer_timestamp"])
                published_dt = _as_naive_utc(publication["published_at"])
                if answer_dt and published_dt and answer_dt < published_dt:
                    continue
                evidence = json.dumps(
                    {"source_url": source["url"], "publication_url": publication["url"]},
                    ensure_ascii=False,
                )
                inserts.append((
                    source["dataset_id"], source["answer_id"], source["source_index"],
                    publication["publication_id"], "exact_url", 1.0, matched_at, evidence,
                ))
        conn.executemany(
            """
            INSERT INTO source_article_matches (
                dataset_id, answer_id, source_index, publication_id,
                match_method, confidence, matched_at, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            inserts,
        )
        conn.commit()
        return len(inserts)


def _article_visibility(username: str | None) -> tuple[str, list[Any]]:
    if username is None:
        return "1 = 1", []
    return "oa.owner_username = ?", [username]


def get_article(article_id: str, username: str | None) -> dict[str, Any] | None:
    owner_where, params = _article_visibility(username)
    with closing(_connect()) as conn:
        row = conn.execute(
            f"SELECT oa.* FROM outbound_articles oa WHERE oa.article_id = ? AND {owner_where}",
            [article_id, *params],
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item.pop("content_text", None)
        item["publications"] = [dict(pub) for pub in conn.execute(
            """
            SELECT publication_id, platform, url, canonical_url, published_at, created_at
            FROM article_publications WHERE article_id = ? ORDER BY created_at
            """,
            (article_id,),
        )]
    return item


def list_dashboard(
    *, username: str | None, allowed: list[str] | None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filters = filters or {}
    refresh_matches(username, allowed)
    owner_where, owner_params = _article_visibility(username)
    article_params: list[Any] = list(owner_params)
    article_conds = [owner_where]
    product_cond = _in_clause(filters.get("product_codes"), "oa.product_code", article_params)
    if product_cond:
        article_conds.append(product_cond)
    with closing(_connect()) as conn:
        articles = [dict(row) for row in conn.execute(
            f"""
            SELECT oa.article_id, oa.owner_username, oa.title, oa.product_code,
                   p.product_name, oa.campaign, oa.source_filename, oa.file_ext,
                   oa.size_bytes, oa.created_at
            FROM outbound_articles oa
            LEFT JOIN products p ON p.product_code = oa.product_code
            WHERE {' AND '.join(article_conds)}
            ORDER BY oa.created_at DESC
            """,
            article_params,
        )]
        article_ids = [row["article_id"] for row in articles]
        if not article_ids:
            return {
                "summary": {"total_articles": 0, "cited_articles": 0, "citation_rate": None,
                            "citation_answers": 0, "citation_refs": 0, "platforms": 0},
                "articles": [], "platforms": [],
            }
        placeholders = ",".join("?" for _ in article_ids)
        publications = [dict(row) for row in conn.execute(
            f"""
            SELECT publication_id, article_id, platform, url, canonical_url, published_at
            FROM article_publications WHERE article_id IN ({placeholders})
            ORDER BY created_at
            """,
            article_ids,
        )]

        citation_params: list[Any] = list(article_ids)
        citation_conds = [f"ap.article_id IN ({placeholders})"]
        citation_conds.append(_scope_clause(allowed, "m.dataset_id", citation_params))
        for key, column in (
            ("dataset_ids", "m.dataset_id"), ("models", "a.model"),
            ("search_modes", "a.search_enabled"),
        ):
            cond = _in_clause(filters.get(key), column, citation_params)
            if cond:
                citation_conds.append(cond)
        citations = [dict(row) for row in conn.execute(
            f"""
            SELECT ap.article_id, m.publication_id, m.dataset_id, m.answer_id,
                   m.source_index, m.match_method, m.confidence,
                   a.question_id, a.model, a.timestamp AS cited_at
            FROM source_article_matches m
            JOIN article_publications ap ON ap.publication_id = m.publication_id
            JOIN answers a ON a.dataset_id = m.dataset_id AND a.answer_id = m.answer_id
            WHERE {' AND '.join(citation_conds)}
            """,
            citation_params,
        )]

    pubs_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for publication in publications:
        pubs_by_article[publication["article_id"]].append(publication)
    citations_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for citation in citations:
        citations_by_article[citation["article_id"]].append(citation)

    for article in articles:
        rows = citations_by_article.get(article["article_id"], [])
        article["publications"] = pubs_by_article.get(article["article_id"], [])
        article["platforms"] = sorted({row["platform"] for row in article["publications"]})
        article["citation_refs"] = len(rows)
        article["citation_answers"] = len({(row["dataset_id"], row["answer_id"]) for row in rows})
        article["citation_questions"] = len({(row["dataset_id"], row["question_id"]) for row in rows})
        article["citation_models"] = sorted({row["model"] for row in rows})
        times = sorted(row["cited_at"] for row in rows if row["cited_at"])
        article["first_cited_at"] = times[0] if times else None
        article["last_cited_at"] = times[-1] if times else None

    cited_articles = sum(1 for article in articles if article["citation_refs"])
    answer_keys = {(row["dataset_id"], row["answer_id"]) for row in citations}
    platforms = sorted({pub["platform"] for pub in publications})
    return {
        "summary": {
            "total_articles": len(articles), "cited_articles": cited_articles,
            "citation_rate": round(cited_articles / len(articles), 4) if articles else None,
            "citation_answers": len(answer_keys), "citation_refs": len(citations),
            "platforms": len(platforms),
        },
        "articles": articles,
        "platforms": platforms,
    }


def list_citations(
    article_id: str, *, username: str | None, allowed: list[str] | None,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    filters = filters or {}
    if get_article(article_id, username) is None:
        raise ValueError("文章不存在")
    refresh_matches(username, allowed)
    params: list[Any] = [article_id]
    conds = [_scope_clause(allowed, "m.dataset_id", params)]
    for key, column in (
        ("dataset_ids", "m.dataset_id"), ("models", "a.model"),
        ("search_modes", "a.search_enabled"),
    ):
        cond = _in_clause(filters.get(key), column, params)
        if cond:
            conds.append(cond)
    with closing(_connect()) as conn:
        rows = [dict(row) for row in conn.execute(
            f"""
            SELECT m.dataset_id, m.answer_id, m.source_index, m.match_method,
                   m.confidence, s.title AS source_title, s.url AS source_url,
                   a.question_id, a.model, COALESCE(a.model_name, a.model) AS model_name,
                   a.search_enabled, a.round, a.timestamp AS cited_at,
                   substr(a.answer_text, 1, 220) AS answer_preview,
                   q.question_text, COALESCE(q.scenario, '') AS scenario,
                   d.name AS dataset_name, d.batch_date,
                   ap.platform, ap.url AS publication_url
            FROM source_article_matches m
            JOIN article_publications ap ON ap.publication_id = m.publication_id
            JOIN sources s ON s.dataset_id = m.dataset_id AND s.answer_id = m.answer_id
                          AND s.source_index = m.source_index
            JOIN answers a ON a.dataset_id = m.dataset_id AND a.answer_id = m.answer_id
            JOIN questions q ON q.dataset_id = a.dataset_id AND q.question_id = a.question_id
            JOIN datasets d ON d.dataset_id = m.dataset_id
            WHERE ap.article_id = ? AND {' AND '.join(conds)}
            ORDER BY COALESCE(a.timestamp, d.batch_date) DESC, a.model, a.question_id
            """,
            params,
        )]
    return rows


def delete_article(article_id: str, username: str | None) -> None:
    owner_where, params = _article_visibility(username)
    with closing(_connect()) as conn:
        cursor = conn.execute(
            f"DELETE FROM outbound_articles AS oa WHERE article_id = ? AND {owner_where}",
            [article_id, *params],
        )
        if cursor.rowcount == 0:
            raise ValueError("文章不存在")
        conn.commit()
