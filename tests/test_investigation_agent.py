from __future__ import annotations

import asyncio
import sqlite3
import socket
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.app.api import investigation_routes
from backend.app.api import routes as api_routes
from backend.app.services import (
    investigation_agent,
    investigation_store,
    investigation_tools,
    source_insight_store,
)
from utils.sqlite_schema import ensure_schema


def _seed_geo(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        conn.execute(
            """INSERT INTO products
               (product_code,product_name,metadata_json,aliases_json)
               VALUES ('p1','测试产品','{}','[]')"""
        )
        for dataset_id, date in (("base", "2026-01-01"), ("current", "2026-02-01")):
            conn.execute(
                """INSERT INTO datasets (
                   dataset_id,name,source_type,imported_at,metadata_json,
                   owner_username,product_code,batch_date,question_set_id
                   ) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    dataset_id, dataset_id, "test", f"{date}T00:00:00", "{}",
                    "alice", "p1", date, "qs1",
                ),
            )
            conn.execute(
                """INSERT INTO dataset_products (
                   dataset_id,product_code,product_name,question_set_id,question_count
                   ) VALUES (?,?,?,?,5)""",
                (dataset_id, "p1", "测试产品", "qs1"),
            )
            for question_index in range(5):
                question_id = f"q{question_index}"
                conn.execute(
                    """INSERT INTO questions (
                       dataset_id,question_id,product_code,product_name,level,
                       question_text,metadata_json
                       ) VALUES (?,?,?,?,?,?,?)""",
                    (
                        dataset_id, question_id, "p1", "测试产品", "symptom",
                        f"测试问题{question_index}", "{}",
                    ),
                )
                for round_num in (1, 2):
                    answer_id = f"{dataset_id}_{question_id}_{round_num}"
                    conn.execute(
                        """INSERT INTO answers (
                           dataset_id,answer_id,question_id,product_code,product_name,
                           model,model_name,model_id,search_enabled,search_triggered,
                           round,timestamp,answer_text,answer_chars,source_count,
                           route,client_mode,request_config_json,metadata_json
                           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            dataset_id, answer_id, question_id, "p1", "测试产品",
                            "qwen", "通义千问", "qwen-test", 1, 1, round_num,
                            f"{date}T12:00:00", "稳定回答内容", 6, 1,
                            "relay", "api", "{}", "{}",
                        ),
                    )
                    domain = "old.example.com" if dataset_id == "base" else "new.example.com"
                    conn.execute(
                        """INSERT INTO sources (
                           dataset_id,answer_id,source_index,title,url,domain,metadata_json
                           ) VALUES (?,?,?,?,?,?,?)""",
                        (
                            dataset_id, answer_id, 1, f"来源{question_index}",
                            f"https://{domain}/{question_index}/{round_num}",
                            domain, "{}",
                        ),
                    )
            conn.execute(
                """INSERT INTO metrics_summary (
                   dataset_id,product_code,stage,question_level,model,search_enabled,
                   total_answers,brand_rec_rate,brand_mention_rate,negative_rate,
                   accuracy_rate,extra_json
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    dataset_id, "p1", "symptom", "symptom", "qwen", "1", 10,
                    .8 if dataset_id == "base" else .4,
                    .9 if dataset_id == "base" else .6,
                    .05 if dataset_id == "base" else .10,
                    .95 if dataset_id == "base" else .90,
                    "{}",
                ),
            )
        conn.execute(
            """INSERT INTO outbound_articles (
               article_id,owner_username,title,content_text,content_sha256,
               product_code,campaign,source_filename,file_ext,file_sha256,
               size_bytes,created_at,metadata_json
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "article1", "alice", "稳定文章", "文章内容始终未变化", "content-sha",
                "p1", "campaign", "article.txt", ".txt", "file-sha", 24,
                "2025-12-01T00:00:00", "{}",
            ),
        )
        conn.execute(
            """INSERT INTO article_publications (
               publication_id,article_id,platform,url,canonical_url,
               url_match_key,published_at,created_at,metadata_json
               ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                "publication1", "article1", "website",
                "https://old.example.com/0/1", "https://old.example.com/0/1",
                "old.example.com/0/1", "2025-12-01T00:00:00",
                "2025-12-01T00:00:00", "{}",
            ),
        )
        conn.commit()


@pytest.fixture()
def investigation_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    jobs = tmp_path / "jobs.sqlite"
    geo = tmp_path / "geo.sqlite"
    _seed_geo(geo)
    monkeypatch.setattr(investigation_store, "DB_PATH", jobs)
    monkeypatch.setattr(investigation_store, "GEO_SQLITE_PATH", geo)
    monkeypatch.setattr(investigation_tools, "GEO_SQLITE_PATH", geo)
    monkeypatch.setattr(investigation_agent, "GEO_SQLITE_PATH", geo)
    monkeypatch.setattr(source_insight_store, "GEO_SQLITE_PATH", geo)
    monkeypatch.setattr(investigation_store, "_WORKER_STARTED", False)
    monkeypatch.setattr(investigation_store, "_QUEUE", __import__("queue").Queue())
    investigation_store.init_db()
    yield jobs, geo


def test_scan_selects_matching_baseline_and_creates_regression(investigation_env):
    result = investigation_store.scan_dataset(
        "current", owner_username="alice", auto_start=False
    )
    assert result["candidate_count"] >= 1
    candidate = next(
        row for row in result["candidates"] if row["metric"] == "brand_rec_rate"
    )
    assert candidate["baseline_dataset_id"] == "base"
    assert candidate["severity"] == "high"
    assert candidate["signal"]["delta"] == pytest.approx(-.4)
    assert candidate["status"] == "candidate"


def test_repeat_scan_is_idempotent(investigation_env):
    first = investigation_store.scan_dataset(
        "current", owner_username="alice", auto_start=False
    )
    second = investigation_store.scan_dataset(
        "current", owner_username="alice", auto_start=False
    )
    assert {
        row["investigation_id"] for row in first["candidates"]
    } == {
        row["investigation_id"] for row in second["candidates"]
    }


def test_auto_scan_starts_only_up_to_three_high_severity_cases(
    investigation_env, monkeypatch
):
    started = []

    def fake_start(investigation_id):
        started.append(investigation_id)
        return {"investigation_id": investigation_id}

    monkeypatch.setattr(investigation_store, "start_investigation", fake_start)
    result = investigation_store.scan_dataset(
        "current", owner_username="alice", auto_start=True
    )
    by_id = {
        row["investigation_id"]: row for row in result["candidates"]
    }
    assert 1 <= len(started) <= 3
    assert all(by_id[investigation_id]["severity"] == "high" for investigation_id in started)


def test_target_crud_and_hard_budget_limits(investigation_env):
    target = investigation_store.create_target(
        "alice",
        {
            "product_code": "p1",
            "metric": "brand_rec_rate",
            "target_value": .75,
        },
    )
    assert target["target_value"] == .75
    updated = investigation_store.update_target(
        target["target_id"], {"target_value": .8, "is_active": False}
    )
    assert updated["target_value"] == .8
    assert updated["is_active"] == 0
    settings = investigation_store.update_settings(
        {"max_probe_calls": 9999, "max_web_fetches": 9999}, "admin"
    )
    assert settings["max_probe_calls"] == investigation_store.HARD_LIMITS["max_probe_calls"]
    assert settings["max_web_fetches"] == investigation_store.HARD_LIMITS["max_web_fetches"]
    investigation_store.delete_target(target["target_id"])
    assert investigation_store.get_target(target["target_id"]) is None


def test_article_citation_target_creates_derived_metric_candidate(investigation_env):
    investigation_store.create_target(
        "alice",
        {
            "product_code": "p1",
            "metric": "article_citation_rate",
            "target_value": .2,
        },
    )
    result = investigation_store.scan_dataset(
        "current", owner_username="alice", auto_start=False
    )
    candidate = next(
        row for row in result["candidates"]
        if row["metric"] == "article_citation_rate"
    )
    assert candidate["signal"]["current_value"] == 0
    assert candidate["signal"]["baseline_value"] == .1
    assert candidate["signal"]["reason"] == "target_missed"


def test_sample_and_source_diff_tools(investigation_env, monkeypatch):
    monkeypatch.setattr(
        investigation_tools.user_config_store, "load_global_kb", lambda: {}
    )
    item = {
        "current_dataset_id": "current",
        "baseline_dataset_id": "base",
        "product_code": "p1",
        "metric": "brand_rec_rate",
        "stage_filter": "symptom",
        "model_filter": "qwen",
        "search_enabled_filter": 1,
    }
    sample = investigation_tools.validate_sample(item)
    assert sample["comparable"] is True
    assert sample["question_overlap_rate"] == 1
    sources = investigation_tools.diff_sources(item)
    assert sources["url_overlap_rate"] == 0
    assert len(sources["lost_urls"]) == 10
    assert len(sources["gained_urls"]) == 10
    articles = investigation_tools.trace_articles(item)
    assert articles["summary"]["total_articles"] == 1
    assert articles["summary"]["cited_before"] == 1
    assert articles["summary"]["cited_after"] == 0


def test_agent_preserves_evidence_when_planner_is_unavailable(
    investigation_env, monkeypatch
):
    monkeypatch.setattr(
        investigation_tools.user_config_store, "load_global_kb", lambda: {}
    )

    async def unavailable(*args, **kwargs):
        raise ValueError("no test API key")

    monkeypatch.setattr(investigation_agent, "_call_json", unavailable)
    item = investigation_store.create_manual_investigation(
        "alice",
        {
            "current_dataset_id": "current",
            "baseline_dataset_id": "base",
            "product_code": "p1",
            "metric": "brand_rec_rate",
            "stage": "symptom",
            "model": "qwen",
            "search_enabled": 1,
            "auto_start": False,
        },
    )
    investigation_store.update_investigation(item["investigation_id"], status="queued")
    investigation_agent.run_investigation(item["investigation_id"])
    completed = investigation_store.get_investigation(item["investigation_id"])
    assert completed["status"] == "needs_review"
    assert completed["conclusion"]["primary_cause"]["category"] == "search_retrieval"
    assert completed["evidence"]
    assert any(call["tool_name"] == "diff_sources" for call in completed["tool_calls"])


def test_running_investigation_is_recovered_after_restart(investigation_env):
    jobs, _ = investigation_env
    item = investigation_store.create_manual_investigation(
        "alice",
        {
            "current_dataset_id": "current",
            "baseline_dataset_id": "base",
            "product_code": "p1",
            "metric": "brand_rec_rate",
            "auto_start": False,
        },
    )
    investigation_store.update_investigation(
        item["investigation_id"], status="running", progress=.5
    )
    with sqlite3.connect(jobs) as conn:
        conn.execute(
            """UPDATE investigations SET lease_expires_at=?
               WHERE investigation_id=?""",
            ("2000-01-01T00:00:00", item["investigation_id"]),
        )
    investigation_store.init_db()
    recovered = investigation_store.get_investigation(
        item["investigation_id"], include_details=False
    )
    assert recovered["status"] == "queued"
    assert recovered["progress"] == .5


def test_worker_claim_is_atomic_and_records_attempt(investigation_env):
    item = investigation_store.create_manual_investigation(
        "alice",
        {
            "current_dataset_id": "current",
            "baseline_dataset_id": "base",
            "product_code": "p1",
            "metric": "brand_rec_rate",
            "auto_start": False,
        },
    )
    investigation_store.update_investigation(
        item["investigation_id"], status="queued"
    )
    assert investigation_store._claim_investigation(item["investigation_id"]) is True
    assert investigation_store._claim_investigation(item["investigation_id"]) is False
    claimed = investigation_store.get_investigation(
        item["investigation_id"], include_details=False
    )
    assert claimed["status"] == "running"
    assert claimed["attempt_count"] == 1
    assert claimed["worker_owner"]
    assert claimed["lease_expires_at"]
    investigation_store.init_db()
    assert investigation_store.get_investigation(
        item["investigation_id"], include_details=False
    )["status"] == "running"


def test_investigation_detail_uses_dataset_scope_as_404(
    investigation_env, monkeypatch
):
    item = investigation_store.create_manual_investigation(
        "alice",
        {
            "current_dataset_id": "current",
            "baseline_dataset_id": "base",
            "product_code": "p1",
            "metric": "brand_rec_rate",
            "auto_start": False,
        },
    )
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    request.state.user = {"username": "bob", "role": "user"}
    monkeypatch.setattr(api_routes, "get_owned_dataset_ids", lambda username: [])
    with pytest.raises(HTTPException) as exc_info:
        investigation_routes._owned(item["investigation_id"], request)
    assert exc_info.value.status_code == 404

    request.state.user = {"username": "admin", "role": "admin"}
    assert (
        investigation_routes._owned(item["investigation_id"], request)[
            "investigation_id"
        ]
        == item["investigation_id"]
    )


def test_private_and_loopback_hosts_are_blocked(monkeypatch):
    monkeypatch.setattr(
        investigation_tools.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    with pytest.raises(ValueError, match="禁止访问"):
        investigation_tools._validated_host("http://example.test/article")


def test_planner_uses_configured_fallback_within_budget(monkeypatch):
    attempts = []

    async def fake_call(model_key, model_id, prompt, timeout, retries):
        attempts.append((model_key, model_id, retries))
        if model_key == "primary":
            raise ValueError("primary unavailable")
        return (
            {"hypotheses": [], "next_actions": [], "stop": False},
            "{}",
            "fallback",
            "fallback-v1",
            [],
        )

    monkeypatch.setattr(investigation_agent, "_call_json", fake_call)
    result = asyncio.run(
        investigation_agent._call_planner(
            {
                "primary_model_key": "primary",
                "primary_model_id": "primary-v1",
                "fallback_model_key": "fallback",
                "fallback_model_id": "fallback-v1",
                "request_timeout_seconds": 10,
            },
            "test prompt",
            2,
        )
    )
    assert result[2:4] == ("fallback", "fallback-v1")
    assert result[-1] == 2
    assert attempts == [
        ("primary", "primary-v1", 0),
        ("fallback", "fallback-v1", 0),
    ]


def test_audit_url_allowlist_only_uses_existing_evidence():
    evidence = [
        {
            "payload": {
                "gained_urls": ["https://example.com/article/"],
                "note": "not a url",
            }
        }
    ]
    known = investigation_agent._known_evidence_urls(evidence)
    assert investigation_agent._url_key("https://example.com/article") in known
    assert investigation_agent._url_key("https://untrusted.example/path") not in known
