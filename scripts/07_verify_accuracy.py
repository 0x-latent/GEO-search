"""
用知识库校对AI应答的准确性（Q1-Q2产品认知类问题）。
两种模式：
1. keyword — 关键词匹配（快速，离线）
2. llm — DeepSeek V3.2校对（精准，在线，10并发，全量5轮发送）
需手动触发执行。
"""
import asyncio
import json
import os
import sys
import glob
import yaml
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_clients import ModelClient
from utils.similarity import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "config", "knowledge_base.json")
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
ANALYSIS_DIR = os.path.join(BASE_DIR, "results", "analysis")

LLM_CONCURRENCY = 10

PRODUCT_KB_MAP = {
    "感冒灵": "感冒灵",
    "皮炎平": "皮炎平",
    "胃泰": "胃泰",
    "抗病毒": "抗病毒",
    "小感": "小感",
    "强枇": "强枇",
    "澳诺": "澳诺",
    "易善复": "易善复",
}


def load_knowledge_base() -> dict:
    if not os.path.exists(KB_PATH):
        print(f"错误: 知识库不存在，请先运行 06_build_knowledge_base.py")
        sys.exit(1)
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_accuracy_responses() -> list:
    responses = []
    pattern = os.path.join(RAW_DIR, "**", "*.json")
    for fpath in glob.glob(pattern, recursive=True):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "question_id" not in data or "answer" not in data:
                continue
            qid = data["question_id"]
            if "_q1_" in qid or "_q2_" in qid:
                responses.append(data)
        except Exception:
            continue
    return responses


# ===== 关键词校对 =====

def keyword_verify(answer: str, facts: list) -> list:
    results = []
    for fact in facts:
        keywords = fact.get("keywords", [])
        matched_kw = [kw for kw in keywords if kw in answer]
        coverage = len(matched_kw) / len(keywords) if keywords else 0

        wrong_claims = fact.get("wrong_claims", [])
        found_wrong = [w for w in wrong_claims if w in answer]

        results.append({
            "fact": fact["fact"],
            "category": fact.get("category", ""),
            "importance": fact.get("importance", ""),
            "keyword_coverage": round(coverage, 2),
            "matched_keywords": matched_kw,
            "wrong_claims_found": found_wrong,
            "has_error": len(found_wrong) > 0,
        })
    return results


async def run_keyword_verification():
    kb = load_knowledge_base()
    responses = load_accuracy_responses()
    print(f"加载 {len(responses)} 条准确率相关应答")

    rows = []
    for resp in responses:
        product = resp.get("product", "")
        kb_key = PRODUCT_KB_MAP.get(product)
        if not kb_key or kb_key not in kb:
            continue

        facts = kb[kb_key]["facts"]
        answer = resp.get("answer", "")
        results = keyword_verify(answer, facts)

        critical_facts = [r for r in results if r["importance"] == "critical"]
        critical_covered = sum(1 for r in critical_facts if r["keyword_coverage"] > 0.5)
        has_errors = any(r["has_error"] for r in results)

        rows.append({
            "问题ID": resp.get("question_id", ""),
            "产品": product,
            "模型": resp.get("model", ""),
            "联网": "是" if resp.get("search_enabled") else "否",
            "轮次": resp.get("round", ""),
            "关键知识点总数": len(critical_facts),
            "覆盖数": critical_covered,
            "覆盖率": round(critical_covered / len(critical_facts), 2) if critical_facts else 0,
            "发现错误": "是" if has_errors else "否",
            "错误内容": "; ".join(
                f"{r['fact']}: {r['wrong_claims_found']}"
                for r in results if r["has_error"]
            ),
        })

    df = pd.DataFrame(rows)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    path = os.path.join(ANALYSIS_DIR, "accuracy_keyword_verify.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"关键词校对报表 → {path}")


# ===== LLM校对（DeepSeek V3.2） =====

LLM_VERIFY_PROMPT = """你是一个药品信息准确性审核专家。你需要判断AI模型对某个药品的认知是否准确。

## 背景
同一个问题对同一个AI模型问了{round_count}轮，以下是全部{round_count}轮回答。

产品：{product_name}
问题：{question}

## 全部回答
{answers_text}

## 需要校对的知识点
{facts_text}

## 分析步骤
请按以下步骤逐条分析每个知识点：

1. **逐轮扫描**：这个知识点在每一轮回答中是否被提及？提及的内容是否正确？
2. **跨轮汇总**：综合所有轮次，给出结论
3. **错误优先**：只要有1轮出现与知识点矛盾的说法，verdict就是"wrong"（哪怕其他轮对了）
4. **区分遗漏和不相关**：如果知识点与问题本身无关（比如问功效时不需要提及包装规格），判为"not_applicable"；如果相关但没提到，判为"missing"

## 输出格式
返回JSON数组，每个元素：
- "fact_index": 知识点序号（从0开始）
- "verdict": "correct" | "wrong" | "missing" | "inconsistent" | "not_applicable"
  - correct: 多数轮次正确提及
  - wrong: 存在与知识点矛盾的回答
  - missing: 所有轮次均未提及（但与问题相关）
  - inconsistent: 部分轮次正确提及、部分轮次遗漏
  - not_applicable: 知识点与该问题无关
- "correct_rounds": 正确提及的轮次数（0-{round_count}）
- "wrong_rounds": 出现错误的轮次数（0-{round_count}）
- "matched_content": AI回答中与该知识点相关的原文摘录（50字以内）。如果verdict是missing或not_applicable则留空字符串
- "error_content": 如果verdict是wrong或inconsistent，说明AI具体错在哪里、正确说法应该是什么（80字以内）。其他情况留空字符串
- "detail": 简要说明（30字以内，如"5轮中4轮正确，1轮说法有误"）

只返回JSON数组。"""


def _build_answers_text(answers: list) -> str:
    """清洗并拼接全部轮次回答"""
    cleaned = [clean_text(a) for a in answers]
    parts = []
    for i, a in enumerate(cleaned):
        truncated = a[:1200] + ("...（截断）" if len(a) > 1200 else "")
        parts.append(f"--- 第{i+1}轮 ---\n{truncated}")
    return "\n\n".join(parts)


def _parse_json_response(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


async def llm_verify_aggregated(client: ModelClient, product_name: str,
                                 question: str, answers: list, facts: list) -> list:
    check_facts = [f for f in facts if f.get("importance") in ("critical", "important")]
    if not check_facts:
        return []

    facts_text = "\n".join(
        f"[{i}] [{f['importance']}] {f['fact']}"
        for i, f in enumerate(check_facts)
    )

    answers_text = _build_answers_text(answers)

    # 拼接为单条问题发给DeepSeek（走chat completions）
    system = "你是药品信息准确性审核专家。严格按要求返回JSON。"
    user = LLM_VERIFY_PROMPT.format(
        product_name=product_name,
        question=question,
        round_count=len(answers),
        answers_text=answers_text,
        facts_text=facts_text,
    )
    full_prompt = f"{system}\n\n{user}"

    try:
        result = await client.query(
            question=full_prompt,
            enable_search=False,
            temperature=0.1,
            max_tokens=4000,
            json_mode=True,
        )
        # json_mode保证返回合法JSON，但可能外层包了个对象，需要提取数组
        raw_json = json.loads(result["answer"])
        verdicts = raw_json if isinstance(raw_json, list) else raw_json.get("results", raw_json.get("data", []))

        for v in verdicts:
            idx = v.get("fact_index", 0)
            if 0 <= idx < len(check_facts):
                v["fact"] = check_facts[idx]["fact"]
                v["importance"] = check_facts[idx]["importance"]
                v["category"] = check_facts[idx].get("category", "")
        return verdicts
    except Exception as e:
        print(f"  LLM校对失败: {e}")
        return []


async def _verify_one_group(
    client: ModelClient,
    kb: dict,
    qid: str, model: str, search_enabled: bool,
    group_resps: list,
    semaphore: asyncio.Semaphore,
    counter: dict,
):
    """校对单个分组（带并发控制）"""
    product = group_resps[0].get("product", "")
    kb_key = PRODUCT_KB_MAP.get(product)
    if not kb_key or kb_key not in kb:
        counter["done"] += 1
        return []

    facts = kb[kb_key]["facts"]
    question = group_resps[0].get("question_text", "")
    answers = [r.get("answer", "") for r in group_resps]

    async with semaphore:
        search_tag = "联网" if search_enabled else "不联网"
        counter["done"] += 1
        print(f"({counter['done']}/{counter['total']}) {product}×{model}×{search_tag} ({len(answers)}轮)")

        verdicts = await llm_verify_aggregated(
            client, product, question, answers, facts,
        )
        for v in verdicts:
            v["question_id"] = qid
            v["product"] = product
            v["model"] = model
            v["search_enabled"] = search_enabled
            v["round_count"] = len(answers)
        return verdicts


async def run_llm_verification(api_key: str):
    kb = load_knowledge_base()
    responses = load_accuracy_responses()
    print(f"加载 {len(responses)} 条准确率相关应答")

    # 初始化DeepSeek客户端
    config_path = os.path.join(BASE_DIR, "config", "models.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_config = config["models"]["deepseek"]
    client = ModelClient("deepseek", ds_config, api_key)
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    # 按 问题ID×模型×联网模式 分组
    groups = defaultdict(list)
    for resp in responses:
        key = (resp.get("question_id", ""), resp.get("model", ""), resp.get("search_enabled", False))
        groups[key].append(resp)

    counter = {"done": 0, "total": len(groups)}
    print(f"共 {counter['total']} 个分组，并发 {LLM_CONCURRENCY}，全量发送全部轮次")

    tasks = [
        _verify_one_group(client, kb, qid, model, search_enabled, group_resps, semaphore, counter)
        for (qid, model, search_enabled), group_resps in groups.items()
    ]

    results = await asyncio.gather(*tasks)
    all_verdicts = [v for group_verdicts in results for v in group_verdicts]

    # 生成报表
    if all_verdicts:
        df = pd.DataFrame(all_verdicts)
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        path = os.path.join(ANALYSIS_DIR, "accuracy_llm_verify.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\nLLM校对明细 → {path}")

        # 汇总统计
        summary_rows = []
        for (product, model), group_v in df.groupby(["product", "model"]):
            total = len(group_v)
            na = sum(group_v["verdict"] == "not_applicable")
            applicable = total - na
            correct = sum(group_v["verdict"] == "correct")
            wrong = sum(group_v["verdict"] == "wrong")
            missing = sum(group_v["verdict"] == "missing")
            inconsistent = sum(group_v["verdict"] == "inconsistent")

            summary_rows.append({
                "产品": product,
                "模型": model,
                "校对知识点数": applicable,
                "正确": correct,
                "错误": wrong,
                "遗漏": missing,
                "不稳定": inconsistent,
                "准确率": round(correct / (correct + wrong), 2) if (correct + wrong) > 0 else None,
                "完整率": round(correct / applicable, 2) if applicable > 0 else None,
                "稳定率": round((correct + missing) / applicable, 2) if applicable > 0 else None,
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(ANALYSIS_DIR, "accuracy_llm_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"LLM校对汇总 → {summary_path}")
    else:
        print("未产生校对结果（可能知识库中没有对应产品）")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["keyword", "llm", "both"], default="keyword",
                        help="模式: keyword(关键词校对), llm(DeepSeek校对), both(两者都跑)")
    args = parser.parse_args()

    keys_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
    api_key = ""
    if args.mode in ("llm", "both"):
        with open(keys_path, "r", encoding="utf-8") as f:
            keys = yaml.safe_load(f)
        api_key = keys.get("deepseek", {}).get("api_key", "")
        if not api_key or api_key == "sk-xxx":
            print("错误: LLM模式需要DeepSeek API key")
            sys.exit(1)

    async def main():
        if args.mode in ("keyword", "both"):
            print("=== 关键词快速校对 ===")
            await run_keyword_verification()

        if args.mode in ("llm", "both"):
            print("\n=== DeepSeek LLM校对（全量5轮，10并发） ===")
            await run_llm_verification(api_key)

    asyncio.run(main())
