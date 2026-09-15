"""Consume KB's fixed agency review contract using GEO's own server credential."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone, timedelta

import requests

from .knowledge_contract import ReviewContext, MAX_CONTEXT_BYTES


class KnowledgeUnavailable(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def screen(text, criteria):
    def norm(value):
        return re.sub(r"[\s　]+", "", value).translate(str.maketrans({"，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";", "（": "(", "）": ")", "“": '"', "”": '"', "‘": "'", "’": "'"}))
    haystack, out = norm(text), []
    for rule in criteria["forbidden_expressions"]:
        hit = next((v for v in (rule.get("match_variants") or [rule["expression"]]) if len(norm(v)) >= 3 and norm(v) in haystack), None)
        if hit:
            out.append({"issue_type": "risk", "severity": "high", "excerpt": hit, "verdict": "conflict",
                        "kb_module": rule["code"], "evidence": rule.get("reason") or rule["expression"],
                        "suggestion": "请对照标准表达修改：" + "、".join(rule.get("alternative_codes", [])), "blocks_publication": True})
    return out


def validate(payload, product, text):
    try:
        ReviewContext.model_validate(payload)
        if payload["product"]["code"] != product or payload["input_text_sha256"] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("产品或正文摘要不匹配")
        basis = {k: payload[k] for k in ("role", "product", "criteria", "standard_revisions")}
        if payload["criteria_sha256"] != hashlib.sha256(canonical(basis).encode()).hexdigest():
            raise ValueError("知识摘要校验失败")
        at = datetime.fromisoformat(payload["captured_at"].replace("Z", "+00:00"))
        if at.tzinfo is None or abs(datetime.now(timezone.utc) - at) > timedelta(minutes=10):
            raise ValueError("知识快照时间无效")
        if not any(payload["criteria"].values()):
            raise ValueError("知识库未提供有效判据")
        if len(canonical(payload).encode()) > MAX_CONTEXT_BYTES:
            raise ValueError("知识快照过大")
    except (ValueError, TypeError, KeyError) as exc:
        raise KnowledgeUnavailable("知识库固定判据校验失败，请核对产品映射及 KB 服务") from exc
    return payload


def fetch_context(product, text):
    base, key = os.environ.get("GEO_KB_URL", "").rstrip("/"), os.environ.get("GEO_KB_KEY", "")
    if not base or not key:
        raise KnowledgeUnavailable("尚未配置 GEO 专属知识库连接 GEO_KB_URL / GEO_KB_KEY")
    if len(text) > 20_000:
        raise KnowledgeUnavailable("稿件超过固定判据接口 20000 字上限，请拆分投稿")
    try:
        with requests.post(base + "/api/v1/brand/review-context",
                           headers={"X-KB-App": "geo", "X-KB-Key": key, "X-KB-Role": "agency"},
                           json={"product": product, "text": text}, timeout=(5, 30),
                           allow_redirects=False, stream=True) as response:
            if response.status_code != 200:
                raise KnowledgeUnavailable(f"知识库暂不可用（HTTP {response.status_code}），请核对 GEO 授权和产品映射")
            parts, size = [], 0
            for chunk in response.iter_content(16384):
                size += len(chunk)
                if size > MAX_CONTEXT_BYTES:
                    raise KnowledgeUnavailable("知识库响应超出大小限制")
                parts.append(chunk)
            payload = json.loads(b"".join(parts))
    except (requests.RequestException, ValueError) as exc:
        if isinstance(exc, KnowledgeUnavailable):
            raise
        raise KnowledgeUnavailable("无法读取知识库固定判据，请稍后重试") from exc
    return validate(payload, product, text)
