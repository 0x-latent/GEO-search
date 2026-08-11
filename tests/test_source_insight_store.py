from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest

from backend.app.services import source_insight_store as store
from utils.sqlite_schema import ensure_schema


class SourceInsightStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db = store.GEO_SQLITE_PATH
        store.GEO_SQLITE_PATH = Path(self.tempdir.name) / "sources.sqlite"
        with closing(sqlite3.connect(store.GEO_SQLITE_PATH)) as conn:
            ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, name, source_type, imported_at, batch_date
                ) VALUES ('ds1', '测试批次', 'test', '2026-07-14', '2026-07-14')
                """
            )
            conn.executemany(
                """
                INSERT INTO questions (
                    dataset_id, question_id, product_code, product_name,
                    level, scenario, question_text
                ) VALUES ('ds1', ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("q1", "p1", "产品一", "q3_solution", "场景一", "问题一"),
                    ("q2", "p1", "产品一", "q5_category", "场景二", "问题二"),
                    ("q3", "p2", "产品二", "q1_brand", "场景三", "问题三"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO answers (
                    dataset_id, answer_id, question_id, product_code, product_name,
                    model, model_name, search_enabled, round, answer_text, source_count
                ) VALUES ('ds1', ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                [
                    ("a1", "q1", "p1", "产品一", "m1", "模型一", 1, "有信源回答", 2),
                    ("a2", "q2", "p1", "产品一", "m1", "模型一", 1, "无信源回答", 0),
                    ("a3", "q3", "p2", "产品二", "m1", "模型一", 0, "离线回答", 0),
                ],
            )
            conn.executemany(
                """
                INSERT INTO sources (
                    dataset_id, answer_id, source_index, title, url, domain
                ) VALUES ('ds1', 'a1', ?, ?, ?, ?)
                """,
                [
                    (0, "药监局", "https://www.nmpa.gov.cn/a?utm_source=test", "www.nmpa.gov.cn"),
                    (1, "博禾", "https://m.bohe.cn/a", "m.bohe.cn"),
                ],
            )
            conn.execute(
                """
                INSERT INTO metric_evidence (
                    dataset_id, evidence_type, product_code, stage, question_level,
                    question_id, model, search_enabled, round, rec_product, name_type
                ) VALUES (
                    'ds1', 'recommendation', 'p1', 'category', 'q5_category',
                    'q2', 'm1', 1, 1, '产品一', '目标品牌'
                )
                """
            )
            conn.commit()

    def tearDown(self) -> None:
        store.GEO_SQLITE_PATH = self.original_db
        self.tempdir.cleanup()

    def test_domain_and_url_normalization(self) -> None:
        self.assertEqual(store.normalize_domain("www.nmpa.gov.cn"), "nmpa.gov.cn")
        self.assertEqual(store.normalize_domain("m.bohe.cn"), "bohe.cn")
        self.assertEqual(store.normalize_domain("bohe.cn:8080"), "bohe.cn")
        self.assertEqual(store.normalize_domain(None, "http://[2001:db8::1]/x"), "2001:db8::1")
        self.assertEqual(
            store.normalize_url("https://www.nmpa.gov.cn/a?utm_source=test&x=1"),
            "https://nmpa.gov.cn/a?x=1",
        )
        self.assertEqual(store.normalize_url("javascript:alert(1)"), "")
        self.assertEqual(store.normalize_url("data:text/html,pwned"), "")

    def test_classification_uses_catalog(self) -> None:
        regulator = store.classify_domain("nmpa.gov.cn")
        portal = store.classify_domain("bohe.cn")
        self.assertEqual(regulator["category"], "regulator")
        self.assertTrue(regulator["is_authoritative"])
        self.assertEqual(portal["category"], "medical_portal")

    def test_multi_dimension_analysis_and_gaps(self) -> None:
        result = store.analyze({"dataset_ids": ["ds1"], "product_codes": ["p1"]})
        self.assertEqual(result["summary"]["online_answers"], 2)
        self.assertEqual(result["summary"]["cited_online_answers"], 1)
        self.assertEqual(result["summary"]["coverage_rate"], 0.5)
        self.assertEqual(result["summary"]["distinct_domains"], 2)
        self.assertIsNone(result["summary"]["official_coverage_rate"])
        self.assertEqual({row["domain"] for row in result["domains"]}, {"nmpa.gov.cn", "bohe.cn"})
        self.assertIn("recommendation_without_source", {row["gap_type"] for row in result["gaps"]})

    def test_offline_sources_do_not_exceed_online_denominator(self) -> None:
        # 离线回答也带信源时，分类/域名覆盖率不得超过 100%（分母是联网回答）。
        with closing(sqlite3.connect(store.GEO_SQLITE_PATH)) as conn:
            conn.execute(
                """
                INSERT INTO sources (dataset_id, answer_id, source_index, title, url, domain)
                VALUES ('ds1', 'a3', 0, '离线信源', 'https://www.nmpa.gov.cn/b', 'www.nmpa.gov.cn')
                """
            )
            conn.commit()
        result = store.analyze({"dataset_ids": ["ds1"]})
        for row in result["categories"] + result["domains"]:
            self.assertLessEqual(row["coverage_rate"], 1.0)
        nmpa = next(row for row in result["domains"] if row["domain"] == "nmpa.gov.cn")
        # a1（联网）与 a3（离线）都引用 nmpa，分子只计联网的 a1。
        self.assertEqual(nmpa["answer_count"], 2)
        self.assertEqual(nmpa["coverage_rate"], 0.5)

    def test_yang_metric_evidence_counts_as_recommendation(self) -> None:
        # 厂商预聚合数据集：yang_metric 证据 search_enabled/round 为 NULL、model 为展示名。
        with closing(sqlite3.connect(store.GEO_SQLITE_PATH)) as conn:
            conn.execute(
                """
                INSERT INTO metric_evidence (
                    dataset_id, evidence_type, product_code, stage, question_level,
                    question_id, model, search_enabled, round, rec_product, name_type, payload_json
                ) VALUES (
                    'ds1', 'yang_metric', 'p2', 'brand', 'q1_brand',
                    'q3', '模型一', NULL, NULL, '产品二', '目标品牌', '{"位次": 2}'
                )
                """
            )
            conn.commit()
        with closing(store._connect()) as conn:
            rows = store._answer_rows(conn, {"dataset_ids": ["ds1"], "product_codes": ["p2"]}, None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["brand_recommended"], 1)

    def test_official_wildcard_enables_coverage(self) -> None:
        # brand_official 域名未标 product_codes 时按全产品生效，而不是永远“待配置”。
        catalog_path = Path(self.tempdir.name) / "source_domains.yaml"
        catalog_path.write_text(
            "categories:\n  brand_official: 品牌官方\n  other: 其他\n"
            "domain_overrides:\n  brand.example.com:\n    category: brand_official\n",
            encoding="utf-8",
        )
        original_config = store.SOURCE_CONFIG_PATH
        store.SOURCE_CONFIG_PATH = catalog_path
        try:
            with closing(sqlite3.connect(store.GEO_SQLITE_PATH)) as conn:
                conn.execute(
                    """
                    INSERT INTO sources (dataset_id, answer_id, source_index, title, url, domain)
                    VALUES ('ds1', 'a1', 2, '官方页', 'https://brand.example.com/p', 'brand.example.com')
                    """
                )
                conn.commit()
            result = store.analyze({"dataset_ids": ["ds1"]})
            self.assertIsNotNone(result["summary"]["official_coverage_rate"])
            self.assertGreater(result["summary"]["official_coverage_rate"], 0)
        finally:
            store.SOURCE_CONFIG_PATH = original_config

    def test_source_answers_preview_and_stable_order(self) -> None:
        rows = store.source_answers({"dataset_ids": ["ds1"]}, domain="nmpa.gov.cn", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answer_preview"], "有信源回答")
        self.assertNotIn("answer_text", rows[0])


if __name__ == "__main__":
    unittest.main()
