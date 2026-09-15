"""Validation shared by the job API and the command-line collector."""
from __future__ import annotations

import re
from typing import Any


def validate_question_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", value):
        raise ValueError("问题 ID 必须为 1–180 位字母、数字、下划线或短横线")
    return value


def validate_questions(questions: Any) -> None:
    if not isinstance(questions, list) or not questions:
        raise ValueError("问题列表为空或格式无效")
    seen = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("每个问题必须是对象")
        question_id = validate_question_id(question.get("id"))
        if question_id in seen:
            raise ValueError(f"问题 ID 重复：{question_id}")
        seen.add(question_id)
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"问题正文为空：{question_id}")
        if not isinstance(question.get("product"), str):
            raise ValueError(f"问题产品必须是字符串：{question_id}")
