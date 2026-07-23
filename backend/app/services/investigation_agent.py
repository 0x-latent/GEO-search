"""Controlled investigation Agent runtime and model-assisted probe tools."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..core.paths import GEO_SQLITE_PATH
from . import investigation_store, investigation_tools
from .article_review_service import _call_json, _load_client


PROMPT_VERSION = "geo-investigation-agent-v1"
DETERMINISTIC_TOOLS = [
    "compare_metrics",
    "validate_sample",
    "diff_answers",
    "diff_sources",
    "trace_articles",
    "check_knowledge_consistency",
]
MODEL_TOOLS = {"replay_queries", "research_external_context"}
ALL_TOOLS = set(investigation_tools.TOOL_REGISTRY) | MODEL_TOOLS


class PlannerUnavailable(ValueError):
    """Carries the number of provider calls consumed before planning failed."""

    def __init__(self, message: str, calls_used: int) -> None:
        super().__init__(message)
        self.calls_used = calls_used


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _compact(value: Any, limit: int = 12_000) -> Any:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return value
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, list):
                compact[key] = item[:8]
            elif isinstance(item, str):
                compact[key] = item[:1000]
            else:
                compact[key] = item
        compact["_truncated"] = True
        return compact
    return {"excerpt": text[:limit], "_truncated": True}


def _url_key(value: str) -> str:
    """Normalize enough for allow-list comparison without altering the request URL."""
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname.rstrip(".").lower()
        port = parsed.port
    except ValueError:
        return ""
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path.rstrip("/") or "/", parsed.query, ""))


def _known_evidence_urls(evidence: list[dict[str, Any]]) -> set[str]:
    """Extract URLs from persisted tool evidence; planner-supplied URLs are excluded."""
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            key = _url_key(value)
            if key:
                found.add(key)

    for record in evidence:
        visit(record.get("payload"))
    return found


async def _call_planner(
    settings: dict[str, Any], prompt: str, remaining_calls: int
) -> tuple[dict[str, Any], str, str, str, list[dict[str, Any]], int]:
    """Try the configured primary and then fallback model within the hard budget."""
    candidates = [
        (settings.get("primary_model_key"), settings.get("primary_model_id"), "primary"),
    ]
    fallback = (settings.get("fallback_model_key"), settings.get("fallback_model_id"))
    primary = (settings.get("primary_model_key"), settings.get("primary_model_id"))
    if fallback[0] and fallback != primary:
        candidates.append((*fallback, "fallback"))

    attempts: list[dict[str, Any]] = []
    calls_used = 0
    for model_key, model_id, role in candidates:
        if calls_used >= remaining_calls:
            break
        calls_used += 1
        try:
            plan, raw, resolved_key, resolved_id, retry_log = await _call_json(
                model_key,
                model_id,
                prompt,
                int(settings.get("request_timeout_seconds") or 120),
                0,
            )
            attempts.append({
                "role": role,
                "model": resolved_key,
                "model_id": resolved_id,
                "status": "succeeded",
            })
            return plan, raw, resolved_key, resolved_id, [*attempts, *retry_log], calls_used
        except Exception as exc:  # noqa: BLE001 - fallback is the intended recovery path
            attempts.append({
                "role": role,
                "model": model_key,
                "model_id": model_id,
                "status": "failed",
                "error": str(exc)[:500],
            })
    raise PlannerUnavailable(
        "Primary and fallback investigation models are unavailable: "
        + "; ".join(item.get("error", "") for item in attempts),
        calls_used,
    )


def _summary(tool_name: str, result: dict[str, Any]) -> str:
    if tool_name == "compare_metrics":
        return (
            f"{result.get('metric')} 当前={result.get('current', {}).get('value')}，"
            f"基线={result.get('baseline', {}).get('value')}，变化={result.get('delta')}"
        )
    if tool_name == "validate_sample":
        return (
            f"样本可比={result.get('comparable')}，问题重合率="
            f"{result.get('question_overlap_rate')}，警告={result.get('warnings')}"
        )
    if tool_name == "diff_answers":
        return (
            f"匹配回答={result.get('matched_answers')}，显著变化="
            f"{result.get('changed_answers')}，平均相似度={result.get('average_similarity')}"
        )
    if tool_name == "diff_sources":
        return (
            f"信源URL重合率={result.get('url_overlap_rate')}，"
            f"新增URL={len(result.get('gained_urls') or [])}，"
            f"消失URL={len(result.get('lost_urls') or [])}"
        )
    if tool_name == "trace_articles":
        return f"文章引用追踪：{result.get('summary')}"
    if tool_name == "check_knowledge_consistency":
        return (
            f"知识库存在={result.get('knowledge_base_found')}，"
            f"待核验数字={len(result.get('numeric_claims_not_in_kb') or [])}"
        )
    if tool_name == "audit_url":
        return (
            f"网页状态={result.get('status_code')}，内容哈希="
            f"{result.get('content_sha256')}，Query覆盖={result.get('query_token_coverage')}"
        )
    if tool_name == "replay_queries":
        return (
            f"模型探针调用={result.get('probe_calls')}，"
            f"成功={result.get('successful_calls')}，结果数={len(result.get('results') or [])}"
        )
    if tool_name == "research_external_context":
        return (
            f"外部研究模型={result.get('model')}，"
            f"信源数={len(result.get('sources') or [])}"
        )
    return json.dumps(_compact(result, 1000), ensure_ascii=False)


async def _replay_queries(
    item: dict[str, Any], arguments: dict[str, Any],
    settings: dict[str, Any], remaining_probes: int,
) -> dict[str, Any]:
    max_questions = min(max(int(arguments.get("max_questions") or 3), 1), 3)
    rounds = min(max(int(arguments.get("rounds") or 1), 1), 2)
    requested_modes = arguments.get("search_modes")
    if not isinstance(requested_modes, list):
        requested_modes = [bool(item.get("search_enabled_filter"))]
    search_modes = list(dict.fromkeys(bool(mode) for mode in requested_modes))[:2]
    with sqlite3.connect(GEO_SQLITE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conditions = ["q.dataset_id=?", "q.product_code=?"]
        params: list[Any] = [item["current_dataset_id"], item["product_code"]]
        if item.get("stage_filter"):
            conditions.append(
                """EXISTS (
                   SELECT 1 FROM metrics_summary m
                   WHERE m.dataset_id=q.dataset_id AND m.product_code=q.product_code
                     AND m.stage=?
                )"""
            )
            params.append(item["stage_filter"])
        questions = [
            dict(row) for row in conn.execute(
                f"""SELECT DISTINCT q.question_id,q.question_text,q.scenario
                    FROM questions q WHERE {' AND '.join(conditions)}
                    ORDER BY q.question_id LIMIT ?""",
                (*params, max_questions),
            ).fetchall()
        ]
    if not questions:
        raise ValueError("找不到可用于模型探针的问题")
    model_key = str(arguments.get("model") or item.get("model_filter") or settings.get("primary_model_key") or "")
    model_id = str(arguments.get("model_id") or settings.get("primary_model_id") or "") or None
    client, resolved_key, resolved_id = _load_client(model_key or None, model_id)
    calls = []
    for question in questions:
        for search_enabled in search_modes:
            for round_num in range(1, rounds + 1):
                if len(calls) >= remaining_probes:
                    break
                calls.append((question, search_enabled, round_num))
    results = []
    errors = []
    for question, search_enabled, round_num in calls:
        try:
            response = await asyncio.wait_for(
                client.query(
                    question=question["question_text"],
                    enable_search=search_enabled,
                    temperature=.7,
                    max_tokens=2048,
                ),
                timeout=int(settings.get("request_timeout_seconds") or 120),
            )
            results.append({
                "question_id": question["question_id"],
                "question": question["question_text"],
                "scenario": question.get("scenario"),
                "model": resolved_key,
                "model_id": resolved_id,
                "search_enabled": search_enabled,
                "round": round_num,
                "answer": str(response.get("answer") or "")[:8000],
                "sources": (response.get("sources") or [])[:30],
                "latency_ms": response.get("latency_ms"),
                "search_triggered": response.get("search_triggered"),
            })
        except Exception as exc:  # noqa: BLE001 - partial probe results are evidence
            errors.append({
                "question_id": question["question_id"],
                "search_enabled": search_enabled,
                "round": round_num,
                "error": str(exc)[:1000],
            })
        finally:
            investigation_store.renew_lease(item["investigation_id"])
    return {
        "model": resolved_key,
        "model_id": resolved_id,
        "probe_calls": len(calls),
        "successful_calls": len(results),
        "results": results,
        "errors": errors,
    }


async def _research_external_context(
    item: dict[str, Any], arguments: dict[str, Any],
    settings: dict[str, Any], remaining_probes: int,
) -> dict[str, Any]:
    if remaining_probes < 1:
        raise ValueError("模型探针预算已用尽")
    topic = str(arguments.get("topic") or "").strip()
    if not topic:
        topic = (
            f"调查 {item['product_code']} 的 {item['metric']} 在近期发生异常变化，"
            "研究可能的行业、竞品、搜索或模型侧公开变化。"
        )
    model_key = str(arguments.get("model") or settings.get("primary_model_key") or "")
    model_id = str(arguments.get("model_id") or settings.get("primary_model_id") or "") or None
    client, resolved_key, resolved_id = _load_client(model_key or None, model_id)
    response = await asyncio.wait_for(
        client.query(
            question=(
            "你是GEO外部研究助手。只报告可以从当前联网搜索结果支持的事实，"
            "明确区分事实、推断和未知，不得宣称知道模型内部算法。"
            f"\n研究任务：{topic}"
            ),
            enable_search=True,
            temperature=.1,
            max_tokens=3000,
        ),
        timeout=int(settings.get("request_timeout_seconds") or 120),
    )
    return {
        "model": resolved_key,
        "model_id": resolved_id,
        "answer": str(response.get("answer") or "")[:12_000],
        "sources": (response.get("sources") or [])[:50],
        "search_triggered": response.get("search_triggered"),
        "probe_calls": 1,
    }


def _planner_prompt(
    item: dict[str, Any], evidence: list[dict[str, Any]],
    iteration: int, budget: dict[str, int],
) -> str:
    available = {
        "audit_url": {
            "args": {"url": "必须来自已有信源证据", "query": "可选目标Query"},
            "purpose": "检查文章可访问性、内容快照和Query匹配",
        },
        "replay_queries": {
            "args": {"max_questions": "1-3", "rounds": "1-2", "search_modes": [False, True]},
            "purpose": "受控复测少量Query",
        },
        "research_external_context": {
            "args": {"topic": "具体公开信息研究问题"},
            "purpose": "联网研究竞品、行业或模型公开变化",
        },
    }
    return f"""你是GEO异常调查 Agent，只能依据给定证据推理。
禁止宣称知道大模型内部算法；必须区分事实、推断和未知。
优先选择能排除不同原因的最少测试。不得要求修改知识库、发文或外部发布。

返回严格 JSON：
{{
  "hypotheses":[{{
    "category":"data_quality|random_variance|query_change|model_behavior|search_retrieval|article_quality|source_change|knowledge_conflict|competitor_change|external_environment|unknown",
    "statement":"原因假设",
    "status":"active|supported|weakened|rejected",
    "confidence":0.0,
    "supporting_evidence":["证据标题"],
    "opposing_evidence":["反证标题"]
  }}],
  "next_actions":[{{"tool":"audit_url|replay_queries|research_external_context","arguments":{{}},"reason":"为什么能区分原因"}}],
  "stop":false,
  "conclusion":null
}}

当证据已经足够或继续调用价值很低时，stop=true，next_actions=[]，conclusion必须为：
{{
  "primary_cause":{{"category":"分类","summary":"最可能原因","confidence":0.0}},
  "alternative_causes":[{{"category":"分类","summary":"备选原因","confidence":0.0}}],
  "supporting_evidence":["证据标题"],
  "opposing_evidence":["反证标题"],
  "unresolved_questions":["未解决问题"],
  "recommended_actions":["只读诊断后的建议，不直接执行"],
  "verification_experiments":["下一步如何验证"],
  "limitations":["数据或方法限制"]
}}

调查对象：
{json.dumps({
    "product_code": item["product_code"], "metric": item["metric"],
    "current_dataset_id": item["current_dataset_id"],
    "baseline_dataset_id": item.get("baseline_dataset_id"),
    "stage": item.get("stage_filter"), "model": item.get("model_filter"),
    "search_enabled": item.get("search_enabled_filter"),
    "signal": item.get("signal"),
}, ensure_ascii=False)}

当前迭代：{iteration}
剩余预算：{json.dumps(budget, ensure_ascii=False)}
可用增量工具：{json.dumps(available, ensure_ascii=False)}
已有证据：{json.dumps([{
    "title": value["title"], "summary": value["summary"],
    "payload": _compact(value["payload"]),
} for value in evidence], ensure_ascii=False)}
"""


def _fallback_conclusion(evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample = evidence.get("validate_sample", {})
    sources = evidence.get("diff_sources", {})
    answers = evidence.get("diff_answers", {})
    articles = evidence.get("trace_articles", {})
    if not sample.get("comparable"):
        category, summary, confidence = (
            "data_quality", "前后样本或采集环境不可比，当前异常可能由数据口径变化造成", .8
        )
    elif (
        len(sources.get("lost_urls") or []) >= 2
        or sources.get("url_overlap_rate", 1) < .35
        or articles.get("summary", {}).get("cited_before", 0)
        > articles.get("summary", {}).get("cited_after", 0)
    ):
        category, summary, confidence = (
            "search_retrieval",
            "前后引用信源发生显著替换，搜索召回或信源竞争变化比文章自身突变更符合现有证据",
            .72,
        )
    elif (answers.get("average_similarity") or 1) < .45:
        category, summary, confidence = (
            "model_behavior", "信源相对稳定但回答结构变化明显，存在模型行为变化可能", .55
        )
    else:
        category, summary, confidence = (
            "unknown", "现有确定性证据不足以区分随机波动、内容和模型侧变化", .35
        )
    return {
        "primary_cause": {
            "category": category, "summary": summary, "confidence": confidence,
        },
        "alternative_causes": [],
        "supporting_evidence": [
            name for name, result in evidence.items() if result
        ],
        "opposing_evidence": [],
        "unresolved_questions": ["需要模型探针或外部网页研究进一步验证"],
        "recommended_actions": ["在采取内容修改前先完成建议的对照实验"],
        "verification_experiments": [
            "对代表性Query进行联网/不联网多轮复测",
            "检查消失与新增信源的内容和可访问状态",
        ],
        "limitations": ["该结论由确定性规则生成，尚未完成LLM辅助假设评估"],
        "generated_by": "deterministic_fallback",
    }


def _normalize_conclusion(value: dict[str, Any]) -> dict[str, Any]:
    """Guarantee the stable conclusion contract even for imperfect model JSON."""
    result = dict(value)
    primary = result.get("primary_cause")
    if not isinstance(primary, dict):
        primary = {
            "category": "unknown",
            "summary": "现有证据不足以形成主原因判断",
            "confidence": 0,
        }
    primary["category"] = str(primary.get("category") or "unknown")[:80]
    primary["summary"] = str(primary.get("summary") or "未提供原因摘要")[:4000]
    try:
        primary["confidence"] = min(
            1.0, max(0.0, float(primary.get("confidence") or 0))
        )
    except (TypeError, ValueError):
        primary["confidence"] = 0.0
    result["primary_cause"] = primary

    alternatives = []
    for item in result.get("alternative_causes") or []:
        if not isinstance(item, dict):
            continue
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence") or 0)))
        except (TypeError, ValueError):
            confidence = 0.0
        alternatives.append({
            "category": str(item.get("category") or "unknown")[:80],
            "summary": str(item.get("summary") or "")[:4000],
            "confidence": confidence,
        })
    result["alternative_causes"] = alternatives[:10]
    for field in (
        "supporting_evidence", "opposing_evidence", "unresolved_questions",
        "recommended_actions", "verification_experiments", "limitations",
    ):
        raw = result.get(field)
        if not isinstance(raw, list):
            raw = []
        result[field] = [str(item)[:4000] for item in raw[:50]]
    return result


async def _execute_tool(
    item: dict[str, Any], tool_name: str, arguments: dict[str, Any],
    settings: dict[str, Any], counters: dict[str, int],
) -> dict[str, Any]:
    if tool_name in investigation_tools.TOOL_REGISTRY:
        if tool_name == "audit_url":
            if counters["web_fetches"] >= item["budget"]["max_web_fetches"]:
                raise ValueError("网页抓取预算已用尽")
            counters["web_fetches"] += 1
        return await asyncio.to_thread(
            investigation_tools.TOOL_REGISTRY[tool_name], item, arguments
        )
    remaining = item["budget"]["max_probe_calls"] - counters["probe_calls"]
    if remaining <= 0:
        raise ValueError("模型探针预算已用尽")
    if tool_name == "replay_queries":
        result = await _replay_queries(item, arguments, settings, remaining)
    elif tool_name == "research_external_context":
        result = await _research_external_context(item, arguments, settings, remaining)
    else:
        raise ValueError(f"不允许的调查工具: {tool_name}")
    counters["probe_calls"] += int(result.get("probe_calls") or 0)
    return result


def _run_async(investigation_id: str) -> None:
    asyncio.run(_run(investigation_id))


async def _run(investigation_id: str) -> None:
    item = investigation_store.get_investigation(investigation_id, include_details=False)
    if item is None or item["status"] not in {"queued", "running"}:
        return
    item["budget"] = item.get("budget") or {}
    settings = investigation_store.get_settings()
    counters = {"reasoning_calls": 0, "probe_calls": 0, "web_fetches": 0}
    evidence_by_tool: dict[str, dict[str, Any]] = {}
    evidence_records: list[dict[str, Any]] = []
    investigation_store.update_investigation(
        investigation_id, status="running", stage="signal_validation",
        progress=.05, started_at=item.get("started_at") or _now(),
    )
    investigation_store.add_event(
        investigation_id, "started", "开始校验异常信号",
        {"prompt_version": PROMPT_VERSION, "budget": item["budget"]},
    )

    for index, tool_name in enumerate(DETERMINISTIC_TOOLS):
        if investigation_store.is_cancelled(investigation_id):
            return
        investigation_store.renew_lease(investigation_id)
        call_id = investigation_store.start_tool_call(
            investigation_id, 0, tool_name, {}
        )
        try:
            result = await _execute_tool(item, tool_name, {}, settings, counters)
            investigation_store.finish_tool_call(call_id, result=result)
            evidence_by_tool[tool_name] = result
            summary = _summary(tool_name, result)
            evidence_id = investigation_store.add_evidence(
                investigation_id, tool_name, tool_name, summary, _compact(result),
                source_ref=f"tool:{call_id}",
            )
            evidence_records.append({
                "evidence_id": evidence_id, "title": tool_name,
                "summary": summary, "payload": _compact(result),
            })
            investigation_store.add_event(
                investigation_id, "tool_completed", f"{tool_name} 已完成",
                {"call_id": call_id, "summary": summary},
            )
        except Exception as exc:  # noqa: BLE001 - preserve partial deterministic evidence
            investigation_store.finish_tool_call(call_id, error=str(exc))
            investigation_store.add_event(
                investigation_id, "tool_failed", f"{tool_name} 执行失败",
                {"call_id": call_id, "error": str(exc)[:1000]},
            )
        investigation_store.update_investigation(
            investigation_id, progress=.08 + (index + 1) / len(DETERMINISTIC_TOOLS) * .30
        )

    if investigation_store.is_cancelled(investigation_id):
        return
    investigation_store.update_investigation(
        investigation_id, stage="hypothesis_generation", progress=.42
    )
    conclusion = None
    hypotheses: list[dict[str, Any]] = []
    planner_error = None

    for iteration in range(1, int(item["budget"].get("max_iterations") or 4) + 1):
        if investigation_store.is_cancelled(investigation_id):
            return
        investigation_store.renew_lease(investigation_id)
        remaining_budget = {
            "reasoning_calls": max(
                0, int(item["budget"].get("max_reasoning_calls") or 0)
                - counters["reasoning_calls"]
            ),
            "probe_calls": max(
                0, int(item["budget"].get("max_probe_calls") or 0)
                - counters["probe_calls"]
            ),
            "web_fetches": max(
                0, int(item["budget"].get("max_web_fetches") or 0)
                - counters["web_fetches"]
            ),
        }
        if remaining_budget["reasoning_calls"] <= 0:
            break
        try:
            plan, raw, model_key, model_id, retry_log, calls_used = await _call_planner(
                settings,
                _planner_prompt(item, evidence_records, iteration, remaining_budget),
                remaining_budget["reasoning_calls"],
            )
            counters["reasoning_calls"] += calls_used
            hypotheses = (
                [value for value in plan.get("hypotheses", []) if isinstance(value, dict)]
                if isinstance(plan.get("hypotheses"), list) else []
            )
            investigation_store.replace_hypotheses(investigation_id, hypotheses)
            investigation_store.add_event(
                investigation_id, "plan_created", f"已生成第{iteration}轮调查计划",
                {
                    "model": model_key, "model_id": model_id,
                    "actions": len(plan.get("next_actions") or []),
                    "retry_log": retry_log,
                },
            )
            if plan.get("stop"):
                value = plan.get("conclusion")
                if isinstance(value, dict) and value.get("primary_cause"):
                    conclusion = value
                    conclusion["generated_by"] = "agent"
                    conclusion["model"] = {"key": model_key, "id": model_id}
                    conclusion["prompt_version"] = PROMPT_VERSION
                    break
            actions = plan.get("next_actions") or []
            if not isinstance(actions, list) or not actions:
                break
            investigation_store.update_investigation(
                investigation_id, stage="evidence_collection",
                progress=min(.88, .45 + iteration * .10),
            )
            for action in actions[:3]:
                if investigation_store.is_cancelled(investigation_id):
                    return
                investigation_store.renew_lease(investigation_id)
                tool_name = str(action.get("tool") or "")
                arguments = action.get("arguments") or {}
                if tool_name not in ALL_TOOLS or tool_name in DETERMINISTIC_TOOLS:
                    investigation_store.add_event(
                        investigation_id, "tool_rejected",
                        f"拒绝非白名单或重复工具: {tool_name}",
                    )
                    continue
                if tool_name == "audit_url":
                    requested_url = str(arguments.get("url") or "")
                    if _url_key(requested_url) not in _known_evidence_urls(evidence_records):
                        investigation_store.add_event(
                            investigation_id,
                            "tool_rejected",
                            "拒绝审计未在已有证据中出现的 URL",
                            {"tool": tool_name, "url": requested_url[:1000]},
                        )
                        continue
                call_id = investigation_store.start_tool_call(
                    investigation_id, iteration, tool_name, arguments
                )
                try:
                    result = await _execute_tool(
                        item, tool_name, arguments, settings, counters
                    )
                    investigation_store.finish_tool_call(call_id, result=result)
                    summary = _summary(tool_name, result)
                    evidence_id = investigation_store.add_evidence(
                        investigation_id, tool_name, tool_name, summary,
                        _compact(result), source_ref=f"tool:{call_id}",
                    )
                    evidence_records.append({
                        "evidence_id": evidence_id, "title": tool_name,
                        "summary": summary, "payload": _compact(result),
                    })
                    evidence_by_tool[tool_name] = result
                    investigation_store.add_event(
                        investigation_id, "tool_completed",
                        f"{tool_name} 已完成", {"summary": summary},
                    )
                except Exception as exc:  # noqa: BLE001
                    investigation_store.finish_tool_call(call_id, error=str(exc))
                    investigation_store.add_event(
                        investigation_id, "tool_failed",
                        f"{tool_name} 执行失败", {"error": str(exc)[:1000]},
                    )
        except Exception as exc:  # model unavailable must not erase deterministic evidence
            counters["reasoning_calls"] += int(getattr(exc, "calls_used", 0))
            planner_error = str(exc)
            investigation_store.add_event(
                investigation_id, "planner_unavailable",
                "调查模型不可用，保留确定性证据并生成降级结论",
                {"error": planner_error[:1000]},
            )
            break

    if conclusion is None:
        conclusion = _fallback_conclusion(evidence_by_tool)
        if hypotheses:
            conclusion["agent_hypotheses"] = hypotheses
    conclusion = _normalize_conclusion(conclusion)
    conclusion["budgets_used"] = counters
    conclusion["evidence_ids"] = [
        value["evidence_id"] for value in evidence_records
    ]
    if planner_error:
        conclusion.setdefault("limitations", []).append(
            f"调查模型不可用：{planner_error[:500]}"
        )
    status = "needs_review" if planner_error else "completed"
    investigation_store.update_investigation(
        investigation_id, status=status, stage="conclusion", progress=1,
        conclusion=conclusion, error_message=planner_error,
        finished_at=_now(),
    )
    investigation_store.add_event(
        investigation_id,
        "needs_review" if status == "needs_review" else "completed",
        "调查已生成结论" if status == "completed" else "已生成降级结论，等待人工复核",
        {
            "primary_cause": conclusion.get("primary_cause"),
            "budgets_used": counters,
        },
    )


def run_investigation(investigation_id: str) -> None:
    """Synchronous worker entry point."""
    _run_async(investigation_id)
