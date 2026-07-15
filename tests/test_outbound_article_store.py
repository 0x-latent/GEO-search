from contextlib import closing
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

from backend.app.services import outbound_article_store as store
from utils.sqlite_schema import ensure_schema


class OutboundArticleStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db = store.GEO_SQLITE_PATH
        store.GEO_SQLITE_PATH = Path(self.tempdir.name) / "articles.sqlite"
        with closing(sqlite3.connect(store.GEO_SQLITE_PATH)) as conn:
            ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, name, source_type, imported_at, batch_date, owner_username
                ) VALUES ('ds1', '新查询', 'test', '2026-07-15', '2026-07-15', 'alice')
                """
            )
            conn.execute(
                """
                INSERT INTO questions (
                    dataset_id, question_id, product_code, product_name, question_text
                ) VALUES ('ds1', 'q1', 'p1', '产品一', '测试问题')
                """
            )
            conn.executemany(
                """
                INSERT INTO answers (
                    dataset_id, answer_id, question_id, product_code, product_name,
                    model, search_enabled, round, timestamp, answer_text
                ) VALUES ('ds1', ?, 'q1', 'p1', '产品一', 'm1', 1, ?, ?, ?)
                """,
                [
                    ("a1", 1, "2026-07-15T10:00:00", "引用文章的回答"),
                    ("a2", 2, "2026-07-10T10:00:00", "发布时间之前的回答"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO sources (
                    dataset_id, answer_id, source_index, title, url, domain
                ) VALUES ('ds1', ?, 0, '文章', ?, 'example.com')
                """,
                [
                    ("a1", "https://www.example.com/post/1?utm_source=ai"),
                    ("a2", "http://example.com/post/1"),
                ],
            )
            conn.commit()

    def tearDown(self) -> None:
        store.GEO_SQLITE_PATH = self.original_db
        self.tempdir.cleanup()

    def test_markdown_import_and_exact_url_match(self) -> None:
        article = store.create_article(
            username="alice", filename="article.md", content="# 我的文章\n\n正文".encode(),
            platform="示例平台", url="https://example.com/post/1",
            published_at="2026-07-12T09:00:00", product_code="p1",
        )
        self.assertEqual(article["title"], "我的文章")
        self.assertEqual(article["publications"][0]["platform"], "示例平台")
        self.assertEqual(store.refresh_matches("alice", ["ds1"]), 1)

        dashboard = store.list_dashboard(username="alice", allowed=["ds1"])
        self.assertEqual(dashboard["summary"]["total_articles"], 1)
        self.assertEqual(dashboard["summary"]["cited_articles"], 1)
        self.assertEqual(dashboard["summary"]["citation_answers"], 1)
        self.assertEqual(dashboard["articles"][0]["citation_refs"], 1)
        citations = store.list_citations(
            article["article_id"], username="alice", allowed=["ds1"]
        )
        self.assertEqual(citations[0]["answer_id"], "a1")

    def test_docx_text_extraction(self) -> None:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "word/document.xml",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body><w:p><w:r><w:t>Word 文章标题</w:t></w:r></w:p>
                  <w:p><w:r><w:t>正文内容</w:t></w:r></w:p></w:body>
                </w:document>""",
            )
        ext, text = store.extract_article_text("test.docx", buffer.getvalue())
        self.assertEqual(ext, ".docx")
        self.assertEqual(text, "Word 文章标题\n正文内容")

    def test_pdf_text_extraction(self) -> None:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "PDF article body")
        content = doc.tobytes()
        doc.close()
        ext, text = store.extract_article_text("test.pdf", content)
        self.assertEqual(ext, ".pdf")
        self.assertIn("PDF article body", text)

    def test_owner_and_dataset_scope(self) -> None:
        article = store.create_article(
            username="alice", filename="a.md", content=b"# Alice article",
            platform="平台", url="https://example.com/post/1",
        )
        hidden = store.list_dashboard(username="bob", allowed=[])
        self.assertEqual(hidden["summary"]["total_articles"], 0)
        store.delete_article(article["article_id"], username="alice")
        visible = store.list_dashboard(username="alice", allowed=["ds1"])
        self.assertEqual(visible["summary"]["total_articles"], 0)

    def test_same_content_across_platforms_is_one_article(self) -> None:
        first = store.create_article(
            username="alice", filename="a.md", content=b"# Shared article",
            platform="平台甲", url="https://example.com/a",
        )
        second = store.create_article(
            username="alice", filename="a.md", content=b"# Shared article",
            platform="平台乙", url="https://example.net/a",
        )
        self.assertEqual(first["article_id"], second["article_id"])
        self.assertEqual(len(second["publications"]), 2)
        with self.assertRaisesRegex(ValueError, "已经导入"):
            store.create_article(
                username="alice", filename="a.md", content=b"# Shared article",
                platform="平台甲", url="https://www.example.com/a?utm_source=x",
            )


if __name__ == "__main__":
    unittest.main()
