from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from backend.app.services import article_review_service, contributor_store
from backend.app.services.document_extract import extract_article_text
from utils.sqlite_schema import ensure_schema


PRODUCTS = [{
    "product_code": "p1", "product_name": "测试产品", "category": "test",
    "aliases": [], "is_active": 1, "display_order": 0,
}]


class ContributorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "geo.sqlite"
        conn = sqlite3.connect(self.db_path)
        ensure_schema(conn)
        conn.close()
        self.patches = [
            patch.object(contributor_store, "GEO_SQLITE_PATH", self.db_path),
            patch.object(contributor_store, "SUBMISSION_DIR", Path(self.temp.name) / "files"),
            patch.object(contributor_store, "list_products", lambda: PRODUCTS),
            patch.object(article_review_service.product_master, "list_products", lambda active_only=False: PRODUCTS),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def _workspace(self, max_submissions=2):
        company = contributor_store.create_company("外部伙伴", "admin")
        invite = contributor_store.create_invite(
            company["company_id"], "admin", ["p1"],
            (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            max_submissions,
        )
        session_token, _, _ = contributor_store.exchange_invite(invite["invite_id"], invite["token"])
        return contributor_store.get_contributor_session(session_token), invite, session_token

    def test_invite_token_is_hashed_and_revocation_ends_session(self):
        session, invite, token = self._workspace()
        self.assertIsNotNone(session)
        with closing(contributor_store._connect()) as conn:
            saved = conn.execute("SELECT token_hash FROM contributor_invites").fetchone()[0]
        self.assertNotEqual(saved, invite["token"])
        contributor_store.revoke_invite(invite["invite_id"])
        self.assertIsNone(contributor_store.get_contributor_session(token))

    def test_submission_queues_review_and_revision_creates_version(self):
        session, _, _ = self._workspace()
        result = contributor_store.create_submission(
            session, "article.md", b"# Test\nbody", "p1", "测试文章",
            "张三", "zhang@example.com",
        )
        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["review_job_status"], "queued")
        with closing(contributor_store._connect()) as conn:
            conn.execute("UPDATE article_submissions SET status='revision_requested' WHERE submission_id=?", (result["submission_id"],))
            conn.commit()
        revised = contributor_store.add_revision(session, result["submission_id"], "v2.txt", b"new body")
        self.assertEqual(revised["current_version"], 2)
        self.assertEqual(revised["status"], "queued")
        self.assertEqual(len(revised["versions"]), 2)

    def test_product_scope_and_invite_quota(self):
        session, _, _ = self._workspace(max_submissions=1)
        with self.assertRaisesRegex(ValueError, "允许范围"):
            contributor_store.create_submission(session, "b.txt", b"body", "p2", "标题2", "李四", "li@example.com")
        contributor_store.create_submission(session, "a.txt", b"body", "p1", "标题", "李四", "li@example.com")
        with self.assertRaisesRegex(ValueError, "次数"):
            contributor_store.create_submission(session, "b.txt", b"body", "p1", "标题2", "李四", "li@example.com")

    def test_review_settings_are_capped_by_environment(self):
        with patch.dict(os.environ, {"GEO_ARTICLE_REVIEW_CONCURRENCY_MAX": "5"}):
            settings = contributor_store.update_review_settings({"ai_concurrency": 30}, "admin")
        self.assertEqual(settings["effective_concurrency"], 5)
        self.assertEqual(settings["ai_concurrency"], 30)

    def test_empty_review_settings_update_is_a_noop(self):
        settings = contributor_store.update_review_settings({}, "admin")
        self.assertIn("effective_concurrency", settings)

    def test_document_parser_supports_markdown_and_scanned_pdf_state(self):
        ext, text = extract_article_text("hello.md", "# 标题\n正文".encode())
        self.assertEqual(ext, ".md")
        self.assertIn("正文", text)

    def test_approval_promotes_published_submission_without_re_review(self):
        session, _, _ = self._workspace()
        result = contributor_store.create_submission(
            session, "article.md", b"# Test\nbody", "p1", "已发布文章",
            "张三", "zhang@example.com", published_platform="官网",
            published_url="https://example.com/article",
        )
        content = "# Test\nbody"
        with closing(contributor_store._connect()) as conn:
            conn.execute(
                "UPDATE article_submission_versions SET content_text=?,content_sha256=? WHERE submission_id=?",
                (content, hashlib.sha256(content.encode()).hexdigest(), result["submission_id"]),
            )
            conn.execute("UPDATE article_submissions SET status='awaiting_admin' WHERE submission_id=?", (result["submission_id"],))
            conn.commit()
        approved = contributor_store.review_action(result["submission_id"], "approve", "admin")
        self.assertEqual(approved["status"], "tracked")
        self.assertTrue(approved["article_id"])
        with closing(contributor_store._connect()) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM article_review_jobs").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT submission_id FROM outbound_articles").fetchone()[0], result["submission_id"])

    def test_missing_knowledge_base_blocks_worker_job(self):
        session, _, _ = self._workspace()
        result = contributor_store.create_submission(
            session, "article.txt", b"plain content", "p1", "知识库缺失",
            "王五", "wang@example.com",
        )
        jobs = article_review_service.claim_jobs("test-worker", 1)
        self.assertEqual(len(jobs), 1)
        with patch.object(article_review_service.user_config_store, "load_global_kb", return_value={}):
            asyncio.run(article_review_service.process_job(jobs[0]))
        blocked = contributor_store.get_submission(result["submission_id"])
        self.assertEqual(blocked["status"], "blocked_missing_kb")
        self.assertEqual(blocked["review_job_status"], "blocked_missing_kb")

    def test_unparseable_document_requests_new_revision(self):
        session, _, _ = self._workspace()
        result = contributor_store.create_submission(
            session, "scan.pdf", b"%PDF-fake", "p1", "扫描文件",
            "王五", "wang@example.com",
        )
        job = article_review_service.claim_jobs("test-worker", 1)[0]
        with patch.object(article_review_service, "extract_article_text", side_effect=ValueError("PDF 未提取到正文，请重新提供可解析文件")):
            asyncio.run(article_review_service.process_job(job))
        external = contributor_store.get_submission(
            result["submission_id"], company_id=session["company_id"], external=True
        )
        self.assertEqual(external["status"], "revision_requested")
        self.assertIn("重新提供", external["admin_feedback"])

    def test_ai_report_waits_for_admin_before_external_feedback(self):
        session, _, _ = self._workspace()
        result = contributor_store.create_submission(
            session, "article.txt", "产品很安全".encode(), "p1", "待审文章",
            "赵六", "zhao@example.com",
        )
        job = article_review_service.claim_jobs("test-worker", 1)[0]
        ai_result = ({
            "verdict": "needs_revision", "risk_level": "high", "summary": "存在绝对化表达",
            "findings": [{
                "issue_type": "risk", "severity": "high", "excerpt": "产品很安全",
                "verdict": "unsupported", "kb_module": "风险",
                "evidence": "知识库无绝对安全证据", "suggestion": "改为审慎表达",
                "blocks_publication": True,
            }],
        }, "raw", "qwen", "test-model", [])
        kb = {"测试产品": {"modules": {"01": {"name": "风险", "text": "不得宣称绝对安全"}}}}
        with patch.object(article_review_service.user_config_store, "load_global_kb", return_value=kb), patch.object(article_review_service, "_call_json", new=AsyncMock(return_value=ai_result)):
            asyncio.run(article_review_service.process_job(job))
        internal = contributor_store.get_submission(result["submission_id"])
        external = contributor_store.get_submission(result["submission_id"], company_id=session["company_id"], external=True)
        self.assertEqual(internal["status"], "awaiting_admin")
        self.assertEqual(len(internal["findings"]), 1)
        self.assertEqual(external["findings"], [])
        finding_id = internal["findings"][0]["finding_id"]
        contributor_store.review_action(
            result["submission_id"], "request_revision", "admin", "请修改",
            [finding_id],
        )
        external = contributor_store.get_submission(result["submission_id"], company_id=session["company_id"], external=True)
        self.assertEqual(len(external["findings"]), 1)
        self.assertEqual(external["admin_feedback"], "请修改")

    def test_claim_count_obeys_effective_concurrency(self):
        session, _, _ = self._workspace(max_submissions=10)
        for index in range(6):
            contributor_store.create_submission(
                session, f"{index}.txt", b"body", "p1", f"文章 {index}",
                "并发测试", f"test{index}@example.com",
            )
        with patch.dict(os.environ, {"GEO_ARTICLE_REVIEW_CONCURRENCY_MAX": "5"}):
            settings = contributor_store.update_review_settings({"ai_concurrency": 30}, "admin")
            jobs = article_review_service.claim_jobs("worker", settings["effective_concurrency"])
        self.assertEqual(len(jobs), 5)


    def _review_submission(self, session=None):
        if session is None:
            session, _, _ = self._workspace(max_submissions=10)
        result = contributor_store.create_submission(
            session, "article.txt", b"test body", "p1", "Test title", "Test",
            "test@example.com", published_platform="site", published_url="https://example.com/old",
        )
        return session, result["submission_id"]

    def test_publication_edit_updates_tracking(self):
        session, sid = self._review_submission()
        with closing(contributor_store._connect()) as conn:
            conn.execute("UPDATE article_submission_versions SET content_text='body',content_sha256='hash' WHERE submission_id=?", (sid,))
            conn.execute("UPDATE article_submissions SET status='awaiting_admin' WHERE submission_id=?", (sid,))
            conn.commit()
        contributor_store.review_action(sid, "approve", "admin")
        with closing(contributor_store._connect()) as conn:
            article_id = conn.execute("SELECT article_id FROM article_submissions WHERE submission_id=?", (sid,)).fetchone()[0]
            conn.execute(
                """INSERT INTO article_publications (publication_id,article_id,platform,url,
                   canonical_url,url_match_key,created_at) VALUES ('other',?,'other site',
                   'https://example.com/other','https://example.com/other','example.com/other','2026-09-01')""",
                (article_id,),
            )
            conn.commit()
        contributor_store.update_publication(session, sid, "new site", "https://example.com/new", "2026-09-15")
        with closing(contributor_store._connect()) as conn:
            row = conn.execute("SELECT * FROM article_publications WHERE publication_id<>'other'").fetchone()
            other = conn.execute("SELECT url FROM article_publications WHERE publication_id='other'").fetchone()[0]
        self.assertEqual(other, "https://example.com/other")
        self.assertEqual(row["url"], "https://example.com/new")
        self.assertEqual(row["url_match_key"], "example.com/new")
        self.assertEqual(row["platform"], "new site")
        self.assertEqual(row["published_at"], "2026-09-15")

    def test_concurrent_revisions_preserve_accepted_content(self):
        session, sid = self._review_submission()
        with closing(contributor_store._connect()) as conn:
            conn.execute("UPDATE article_submissions SET status='revision_requested' WHERE submission_id=?", (sid,))
            conn.commit()
        barrier = threading.Barrier(2)
        original_store = contributor_store._store_file

        def slow_write(*args):
            time.sleep(.1)
            return original_store(*args)

        def revise(content):
            barrier.wait(timeout=5)
            try:
                return contributor_store.add_revision(session, sid, "v2.txt", content)
            except ValueError:
                return None

        with patch.object(contributor_store, "_store_file", side_effect=slow_write), ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(revise, content) for content in (b"first content", b"second content")]
            values = [future.result(timeout=15) for future in futures]
        self.assertEqual(sum(value is not None for value in values), 1)
        with closing(contributor_store._connect()) as conn:
            row = conn.execute("SELECT * FROM article_submission_versions WHERE submission_id=? AND version=2", (sid,)).fetchone()
        content = (contributor_store.SUBMISSION_DIR / row["relative_path"]).read_bytes()
        self.assertEqual(row["file_sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(contributor_store.get_submission(sid)["current_version"], 2)

    def test_upload_paths_do_not_overwrite_a_previous_attempt(self):
        first = contributor_store._store_file("company", "submission", 2, "same.txt", b"first")
        second = contributor_store._store_file("company", "submission", 2, "same.txt", b"second")
        self.assertNotEqual(first, second)
        self.assertEqual((contributor_store.SUBMISSION_DIR / first).read_bytes(), b"first")

    def test_stale_review_cannot_complete_or_fail_retried_attempt(self):
        session, _, _ = self._workspace(max_submissions=10)
        for fail in (False, True):
            with self.subTest(fail=fail):
                _, sid = self._review_submission(session)
                old_job = article_review_service.claim_jobs("same-worker", 1)[0]

                async def retry_while_waiting(*args):
                    contributor_store.cancel_review(sid, "admin")
                    contributor_store.retry_review(sid, "admin")
                    new_job = article_review_service.claim_jobs("same-worker", 1)[0]
                    self.assertGreater(new_job["attempts"], old_job["attempts"])
                    if fail:
                        raise RuntimeError("stale timeout")
                    return ({"findings": []}, "raw", "qwen", "test", [])

                with patch.object(article_review_service, "_product_kb", return_value=({"modules": {}}, "hash")), patch.object(article_review_service, "_call_json", side_effect=retry_while_waiting):
                    asyncio.run(article_review_service.process_job(old_job))
                item = contributor_store.get_submission(sid)
                self.assertEqual(item["review_job_status"], "running")
                self.assertEqual(item["status"], "reviewing")
                self.assertIsNone(item["review_error"])
                self.assertIsNone(item["report"])

    def test_cancel_during_parse_cannot_restore_submission_state(self):
        _, sid = self._review_submission()
        job = article_review_service.claim_jobs("worker", 1)[0]

        def cancelled_parse(*args):
            contributor_store.cancel_review(sid, "admin")
            raise ValueError("bad document after cancellation")

        with patch.object(article_review_service, "extract_article_text", side_effect=cancelled_parse):
            asyncio.run(article_review_service.process_job(job))
        item = contributor_store.get_submission(sid)
        self.assertEqual(item["review_job_status"], "cancelled")
        self.assertEqual(item["status"], "review_failed")
        self.assertIsNone(item["admin_feedback"])


class ReviewHelpersTests(unittest.TestCase):
    def test_json_fence_is_accepted(self):
        result = article_review_service._parse_json_answer("```json\n{\"verdict\":\"pass\"}\n```")
        self.assertEqual(result["verdict"], "pass")

    def test_simhash_is_stable(self):
        tokens = {"产品", "说明", "安全"}
        self.assertEqual(article_review_service._simhash(tokens), article_review_service._simhash(tokens))


if __name__ == "__main__":
    unittest.main()
