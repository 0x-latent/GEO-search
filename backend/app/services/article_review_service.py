"""AI-assisted article review pipeline and durable SQLite job operations."""
from __future__ import annotations

import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
import re
import socket
import time
from typing import Any
from uuid import uuid4

import yaml

from ..core.paths import CONFIG_DIR
from . import contributor_store, product_master, user_config_store
from .document_extract import extract_article_text
from .yaml_store import load_models


logger = logging.getLogger(__name__)
PROMPT_VERSION = "article-review-v1"
LEASE_SECONDS = 300


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json_answer(answer: str) -> dict[str, Any]:
    text = answer.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("AI 未返回有效 JSON")
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("AI 返回 JSON 无法解析") from exc
    if not isinstance(value, dict):
        raise ValueError("AI 审稿结果必须为 JSON 对象")
    return value


def _product_kb(product_code: str) -> tuple[dict[str, Any] | None, str]:
    global_kb = user_config_store.load_global_kb()
    product = next((p for p in product_master.list_products(False) if p["product_code"] == product_code), None)
    if not product:
        return None, ""
    candidates = [product_code, product["product_name"], *(product.get("aliases") or [])]
    selected = next((global_kb.get(key) for key in candidates if key in global_kb), None)
    if not isinstance(selected, dict):
        return None, ""
    canonical = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return selected, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _kb_text(kb: dict[str, Any]) -> str:
    modules = kb.get("modules")
    if isinstance(modules, dict):
        chunks = []
        for code, module in modules.items():
            if isinstance(module, dict):
                chunks.append(f"[{code} {module.get('name', '')}]\n{module.get('text', '')}")
        return "\n\n".join(chunks)[:160_000]
    return json.dumps(kb, ensure_ascii=False)[:160_000]


def _tokens(text: str) -> set[str]:
    try:
        import jieba
        words = jieba.cut(text.lower())
    except ImportError:
        words = re.findall(r"[\w\u4e00-\u9fff]{2,}", text.lower())
    return {word.strip() for word in words if len(word.strip()) >= 2}


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[i:i + 2] for i in range(max(0, len(compact) - 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def _simhash(tokens: set[str]) -> int:
    vector = [0] * 64
    for token in tokens:
        value = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for i in range(64):
            vector[i] += 1 if value & (1 << i) else -1
    return sum((1 << i) for i, count in enumerate(vector) if count >= 0)


def _lexical_candidates(
    submission_id: str, version: int, title: str, content: str,
    content_sha: str, top_k: int,
) -> list[dict[str, Any]]:
    own_tokens, own_title, own_hash = _tokens(content), _bigrams(title), None
    if own_tokens:
        own_hash = _simhash(own_tokens)
    candidates: list[dict[str, Any]] = []
    with closing(contributor_store._connect()) as conn:
        rows = conn.execute(
            """SELECT 'article' kind,a.article_id matched_id,NULL matched_version,a.title,
            a.content_text,a.content_sha256,c.name company_name FROM outbound_articles a
            LEFT JOIN contributor_companies c ON c.company_id=a.company_id
            WHERE a.submission_id IS NULL OR a.submission_id<>?
            UNION ALL
            SELECT 'submission',v.submission_id,v.version,s.title,v.content_text,v.content_sha256,c.name
            FROM article_submission_versions v JOIN article_submissions s USING(submission_id)
            JOIN contributor_companies c USING(company_id)
            WHERE v.content_text IS NOT NULL AND NOT (v.submission_id=? AND v.version=?)""",
            (submission_id, submission_id, version),
        ).fetchall()
    for row in rows:
        other = row["content_text"] or ""
        exact = bool(content_sha and row["content_sha256"] == content_sha)
        other_tokens = _tokens(other)
        token_score = _jaccard(own_tokens, other_tokens)
        title_score = _jaccard(own_title, _bigrams(row["title"] or ""))
        sim_score = 0.0
        if own_hash is not None and other_tokens:
            distance = (own_hash ^ _simhash(other_tokens)).bit_count()
            sim_score = 1 - distance / 64
        score = 1.0 if exact else max(0.55 * token_score + 0.25 * title_score + 0.2 * sim_score, title_score * .7)
        if exact or score >= .12:
            candidates.append({
                "kind": row["kind"], "matched_id": row["matched_id"],
                "matched_version": row["matched_version"], "title": row["title"],
                "company_name": row["company_name"], "exact_hash": exact,
                "lexical_score": round(score, 4), "excerpt": other[:2500],
            })
    candidates.sort(key=lambda item: (item["exact_hash"], item["lexical_score"]), reverse=True)
    return candidates[:top_k]


def _load_client(model_key: str | None, model_id: str | None) -> tuple[Any, str, str]:
    # Keep the OpenAI SDK out of web/schema-only imports; only the review worker
    # needs to load provider clients.
    from utils.api_clients import ModelClient, resolve_relay, resolve_route
    config = load_models()
    enabled = [(key, spec) for key, spec in (config.get("models") or {}).items() if spec.get("enabled")]
    if not enabled:
        raise ValueError("models.yaml 中没有已启用模型")
    key = model_key or enabled[0][0]
    spec = (config.get("models") or {}).get(key)
    if not spec or not spec.get("enabled"):
        raise ValueError(f"审稿模型 {key} 未启用")
    keys_path = CONFIG_DIR / "api_keys.yaml"
    keys = yaml.safe_load(keys_path.read_text(encoding="utf-8")) or {} if keys_path.exists() else {}
    route = resolve_route(config, keys)
    relay = resolve_relay(config, keys)
    api_key = (keys.get(key) or {}).get("api_key", "")
    if route != "relay" and (not api_key or api_key == "sk-xxx"):
        raise ValueError(f"审稿模型 {key} 未配置 API 密钥")
    selected_id = model_id or spec.get("model_id", "")
    return ModelClient(key, spec, api_key, route=route, relay_config=relay, model_override=selected_id), key, selected_id


async def _call_json(
    model_key: str | None, model_id: str | None, prompt: str,
    timeout: int, retries: int,
) -> tuple[dict[str, Any], str, str, str, list[dict[str, Any]]]:
    client, key, selected_id = _load_client(model_key, model_id)
    retry_log = []
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(
                client.query(prompt, enable_search=False, temperature=.1, max_tokens=6000, json_mode=True),
                timeout=timeout,
            )
            answer = result.get("answer", "")
            return _parse_json_answer(answer), answer, key, selected_id, retry_log
        except Exception as exc:
            last = exc
            retry_log.append({"attempt": attempt + 1, "error": str(exc)[:500], "at": contributor_store._iso()})
            if attempt < retries:
                await asyncio.sleep(min(8, 2 ** attempt))
    raise ValueError(f"模型调用失败：{last}")


def _review_prompt(title: str, content: str, kb: dict[str, Any]) -> str:
    return f"""你是产品文章事实审稿 Agent。只依据给定产品知识库审核，不使用外部知识。
重要规则：知识库没有证据只能标记 unsupported，不能自动判错；AI 不能批准文章。
检查事实准确性、证据充分性、夸大疗效、绝对安全、禁忌弱化、误导性比较等高风险表达。
返回严格 JSON：
{{"verdict":"pass|needs_revision|high_risk","risk_level":"low|medium|high",
"summary":"中文摘要","findings":[{{"issue_type":"fact|evidence|risk|wording",
"severity":"low|medium|high","excerpt":"文章原句","verdict":"supported|conflict|unsupported",
"kb_module":"模块编号或名称","evidence":"知识库依据或无依据说明","suggestion":"修改建议",
"blocks_publication":true}}]}}

产品知识库：
{_kb_text(kb)}

待审文章标题：{title}
正文：
{content[:120_000]}"""


def _similarity_prompt(title: str, content: str, candidates: list[dict[str, Any]]) -> str:
    compact = [{"index": i, "title": c["title"], "excerpt": c["excerpt"]} for i, c in enumerate(candidates)]
    return f"""判断目标文章与候选历史文章的语义相似度，只返回严格 JSON。
尺度：1 表示实质重复，0 表示无关。返回：
{{"matches":[{{"index":0,"score":0.0,"overlap_summary":"重合主题","source_excerpt":"目标段落",
"matched_excerpt":"历史段落"}}]}}
目标标题：{title}
目标正文：{content[:30000]}
候选：{json.dumps(compact, ensure_ascii=False)}"""


def claim_jobs(worker_id: str, limit: int) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    with closing(contributor_store._connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """UPDATE article_review_jobs SET status='queued',stage='queued',lease_owner=NULL,
            lease_expires_at=NULL,error_message='Worker 重启后回收超时任务',updated_at=?
            WHERE status='running' AND lease_expires_at<?""",
            (contributor_store._iso(now), contributor_store._iso(now)),
        )
        settings = dict(conn.execute("SELECT * FROM article_review_settings WHERE id=1").fetchone())
        if settings["queue_paused"] or not settings["auto_start"]:
            conn.commit()
            return []
        rows = conn.execute(
            """SELECT * FROM article_review_jobs WHERE status='queued'
            ORDER BY priority DESC,created_at LIMIT ?""", (limit,),
        ).fetchall()
        claimed = []
        for row in rows:
            lease = contributor_store._iso(now + timedelta(seconds=LEASE_SECONDS))
            cur = conn.execute(
                """UPDATE article_review_jobs SET status='running',stage='parsing',attempts=attempts+1,
                lease_owner=?,lease_expires_at=?,heartbeat_at=?,started_at=coalesce(started_at,?),
                settings_snapshot_json=?,updated_at=? WHERE job_id=? AND status='queued'""",
                (worker_id, lease, contributor_store._iso(now), contributor_store._iso(now),
                 _json(settings), contributor_store._iso(now), row["job_id"]),
            )
            if cur.rowcount:
                conn.execute("UPDATE article_submissions SET status='reviewing',updated_at=? WHERE submission_id=?", (contributor_store._iso(now), row["submission_id"]))
                claimed.append(dict(row) | {"settings": settings})
        conn.commit()
    return claimed


def _job_update(job_id: str, **values: Any) -> None:
    values["updated_at"] = contributor_store._iso()
    sets = ",".join(f"{key}=?" for key in values)
    with closing(contributor_store._connect()) as conn:
        conn.execute(f"UPDATE article_review_jobs SET {sets} WHERE job_id=?", (*values.values(), job_id))
        conn.commit()


async def process_job(job: dict[str, Any]) -> None:
    started = time.monotonic()
    settings = job["settings"]
    retry_log: list[dict[str, Any]] = []
    try:
        with closing(contributor_store._connect()) as conn:
            row = conn.execute(
                """SELECT s.*,v.original_filename,v.relative_path,v.content_text,v.content_sha256
                FROM article_review_jobs j JOIN article_submissions s USING(submission_id)
                JOIN article_submission_versions v ON v.submission_id=j.submission_id AND v.version=j.version
                WHERE j.job_id=?""", (job["job_id"],),
            ).fetchone()
        if not row:
            raise ValueError("投稿版本不存在")
        content = row["content_text"]
        if content is None:
            raw = (contributor_store.SUBMISSION_DIR / row["relative_path"]).read_bytes()
            try:
                _, content = await asyncio.to_thread(extract_article_text, row["original_filename"], raw)
                if not content:
                    raise ValueError("PDF 未提取到正文，可能是扫描件；请重新提供可解析文件")
            except ValueError as parse_exc:
                message = str(parse_exc)
                now = contributor_store._iso()
                with closing(contributor_store._connect()) as conn:
                    conn.execute("""UPDATE article_review_jobs SET status='failed',stage='parse_failed',
                        error_message=?,finished_at=?,lease_owner=NULL,lease_expires_at=NULL,updated_at=?
                        WHERE job_id=?""", (message, now, now, job["job_id"]))
                    conn.execute("""UPDATE article_submission_versions SET parse_error=?
                        WHERE submission_id=? AND version=?""", (message, row["submission_id"], job["version"]))
                    conn.execute("""UPDATE article_submissions SET status='revision_requested',
                        admin_feedback=?,updated_at=? WHERE submission_id=?""",
                        (message, now, row["submission_id"]))
                    contributor_store._event(
                        conn, row["submission_id"], "system", job["job_id"],
                        "document_parse_failed", "reviewing", "revision_requested",
                        {"message": message},
                    )
                    conn.commit()
                return
            content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            with closing(contributor_store._connect()) as conn:
                conn.execute("UPDATE article_submission_versions SET content_text=?,content_sha256=?,parse_error=NULL WHERE submission_id=? AND version=?", (content, content_sha, row["submission_id"], job["version"]))
                conn.commit()
        else:
            content_sha = row["content_sha256"] or hashlib.sha256(content.encode()).hexdigest()
        kb, kb_sha = _product_kb(row["product_code"])
        if kb is None:
            with closing(contributor_store._connect()) as conn:
                conn.execute("UPDATE article_review_jobs SET status='blocked_missing_kb',stage='blocked_missing_kb',error_message='产品知识库缺失',finished_at=?,updated_at=? WHERE job_id=?", (contributor_store._iso(), contributor_store._iso(), job["job_id"]))
                conn.execute("UPDATE article_submissions SET status='blocked_missing_kb',updated_at=? WHERE submission_id=?", (contributor_store._iso(), row["submission_id"]))
                conn.commit()
            return
        _job_update(job["job_id"], stage="fact_review", progress=.25)
        try:
            review, raw_answer, model_key, model_id, retry_log = await _call_json(
                settings["primary_model_key"], settings["primary_model_id"],
                _review_prompt(row["title"], content, kb),
                settings["request_timeout_seconds"], settings["retry_count"],
            )
        except Exception as primary_exc:
            if not settings.get("fallback_model_key"):
                raise
            retry_log.append({"fallback_after": str(primary_exc)[:500], "at": contributor_store._iso()})
            review, raw_answer, model_key, model_id, fallback_log = await _call_json(
                settings["fallback_model_key"], settings["fallback_model_id"],
                _review_prompt(row["title"], content, kb),
                settings["request_timeout_seconds"], settings["retry_count"],
            )
            retry_log.extend(fallback_log)
        _job_update(job["job_id"], stage="similarity", progress=.65)
        candidates = await asyncio.to_thread(
            _lexical_candidates, row["submission_id"], job["version"], row["title"],
            content, content_sha, settings["similarity_top_k"],
        )
        semantic: dict[int, dict[str, Any]] = {}
        if candidates:
            similarity, _, _, _, similarity_retries = await _call_json(
                model_key, model_id, _similarity_prompt(row["title"], content, candidates),
                settings["request_timeout_seconds"], settings["retry_count"],
            )
            retry_log.extend(similarity_retries)
            semantic = {int(item.get("index", -1)): item for item in similarity.get("matches", []) if str(item.get("index", "")).lstrip("-").isdigit()}
        now = contributor_store._iso()
        duration = int((time.monotonic() - started) * 1000)
        with closing(contributor_store._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT status FROM article_review_jobs WHERE job_id=?", (job["job_id"],)
            ).fetchone()
            if not current or current["status"] != "running":
                conn.rollback()
                return
            conn.execute("DELETE FROM article_review_findings WHERE job_id=?", (job["job_id"],))
            conn.execute("DELETE FROM article_similarity_matches WHERE job_id=?", (job["job_id"],))
            conn.execute("DELETE FROM article_review_reports WHERE job_id=?", (job["job_id"],))
            conn.execute("""INSERT INTO article_review_reports
                (job_id,verdict,risk_level,summary,model_key,model_id,model_version,prompt_version,
                 knowledge_base_sha256,config_snapshot_json,structured_json,raw_response,duration_ms,
                 retry_log_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (job["job_id"], review.get("verdict", "needs_revision"), review.get("risk_level", "medium"),
                 str(review.get("summary", ""))[:5000], model_key, model_id, model_id, PROMPT_VERSION,
                 kb_sha, _json(settings), _json(review), raw_answer, duration, _json(retry_log), now))
            for finding in review.get("findings", [])[:100]:
                conn.execute("""INSERT INTO article_review_findings
                    (finding_id,job_id,issue_type,severity,excerpt,verdict,kb_module,evidence,
                     suggestion,blocks_publication,external_visible,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,0,?)""",
                    (f"find_{uuid4().hex}", job["job_id"], str(finding.get("issue_type", "fact"))[:30],
                     str(finding.get("severity", "medium"))[:20], str(finding.get("excerpt", ""))[:5000],
                     str(finding.get("verdict", "unsupported"))[:30], str(finding.get("kb_module", ""))[:300],
                     str(finding.get("evidence", ""))[:10000], str(finding.get("suggestion", ""))[:5000],
                     int(bool(finding.get("blocks_publication"))), now))
            threshold = float(settings["similarity_threshold"])
            for index, candidate in enumerate(candidates):
                item = semantic.get(index, {})
                score = 1.0 if candidate["exact_hash"] else float(item.get("score", candidate["lexical_score"]))
                level = "high" if candidate["exact_hash"] or score >= threshold else "possible" if score >= threshold * .75 else "hidden"
                conn.execute("""INSERT INTO article_similarity_matches
                    (match_id,job_id,matched_kind,matched_id,matched_version,exact_hash,lexical_score,
                     semantic_score,similarity_level,overlap_summary,source_excerpt,matched_excerpt,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (f"sim_{uuid4().hex}", job["job_id"], candidate["kind"], candidate["matched_id"],
                     candidate["matched_version"], int(candidate["exact_hash"]), candidate["lexical_score"],
                     score, level, str(item.get("overlap_summary", ""))[:5000],
                     str(item.get("source_excerpt", ""))[:5000], str(item.get("matched_excerpt", ""))[:5000], now))
            conn.execute("""UPDATE article_review_jobs SET status='success',stage='completed',progress=1,
                finished_at=?,lease_owner=NULL,lease_expires_at=NULL,error_message=NULL,updated_at=? WHERE job_id=?""", (now, now, job["job_id"]))
            conn.execute("UPDATE article_submissions SET status='awaiting_admin',updated_at=? WHERE submission_id=?", (now, row["submission_id"]))
            contributor_store._event(conn, row["submission_id"], "system", job["job_id"], "ai_review_completed", "reviewing", "awaiting_admin", {"verdict": review.get("verdict"), "model": f"{model_key}:{model_id}"})
            conn.commit()
    except Exception as exc:
        logger.exception("article review job failed: %s", job["job_id"])
        now = contributor_store._iso()
        with closing(contributor_store._connect()) as conn:
            conn.execute("UPDATE article_review_jobs SET status='failed',stage='failed',error_message=?,finished_at=?,lease_owner=NULL,lease_expires_at=NULL,updated_at=? WHERE job_id=?", (str(exc)[:2000], now, now, job["job_id"]))
            conn.execute("UPDATE article_submissions SET status='review_failed',updated_at=? WHERE submission_id=?", (now, job["submission_id"]))
            conn.execute("UPDATE article_submission_versions SET parse_error=coalesce(parse_error,?) WHERE submission_id=? AND version=?", (str(exc)[:2000], job["submission_id"], job["version"]))
            conn.commit()


def update_worker_heartbeat(worker_id: str, active: int, last_error: str | None = None) -> None:
    settings = contributor_store.get_review_settings()
    now = contributor_store._iso()
    lease = contributor_store._iso(datetime.now(timezone.utc) + timedelta(seconds=LEASE_SECONDS))
    with closing(contributor_store._connect()) as conn:
        conn.execute(
            """UPDATE article_review_jobs SET heartbeat_at=?,lease_expires_at=?,updated_at=?
            WHERE lease_owner=? AND status='running'""",
            (now, lease, now, worker_id),
        )
        conn.execute("""INSERT INTO article_review_worker_state
            (worker_id,hostname,pid,configured_concurrency,environment_max,effective_concurrency,
             active_requests,last_error,heartbeat_at,started_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(worker_id) DO UPDATE SET configured_concurrency=excluded.configured_concurrency,
            environment_max=excluded.environment_max,effective_concurrency=excluded.effective_concurrency,
            active_requests=excluded.active_requests,last_error=excluded.last_error,heartbeat_at=excluded.heartbeat_at""",
            (worker_id, socket.gethostname(), os.getpid(), settings["ai_concurrency"],
             settings["environment_max"], settings["effective_concurrency"], active,
             last_error, now, now))
        conn.commit()
