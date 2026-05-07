"""
用知识库校对AI应答的准确性（Q1-Q2产品认知类问题）。
方向：应答→知识库（查错），不查遗漏。
将产品完整知识库 + 多轮应答一起发给DeepSeek，检查应答中是否有与知识库矛盾的内容。

按 (问题ID × 模型 × 联网) 分组，每组5轮一起发。
10并发。需手动触发执行。
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

# 产品短名 → 知识库目录名（06脚本用目录名作key）
PRODUCT_KB_MAP = {
    "感冒灵": "感冒灵",
    "皮炎平": "皮炎平",
    "胃泰": "养胃舒",
    "抗病毒": "抗病毒口服液",
    "小感": "小儿氨酚黄那敏",
    "强枇": "强力枇杷露",
    "澳诺": "澳诺葡萄糖酸钙锌",
    "易善复": "易善复",
}


LLM_VERIFY_PROMPT = """你是药品信息准确性审核专家。请检查AI回答中是否有与产品知识库矛盾的内容。

## 任务
只检查AI回答中**说错的内容**，不要管AI没提到的内容（遗漏不算错）。

## 产品知识库（参考标准）
{kb_text}

## 待校验的AI回答
产品：{product_name}
问题：{question}
以下为同一问题的{round_count}轮回答：

{answers_text}

## 审核规则
1. 逐句审查AI回答中的**事实性陈述**（功效、成分、用法、禁忌、适应症等）
2. 将每条陈述与知识库对比：
   - **correct**：与知识库一致
   - **wrong**：与知识库矛盾（任何一轮说错即判wrong）
   - **unverified**：知识库中没有对应信息，无法判断对错
3. 相同陈述在多轮中重复出现的，合并为一条，标注出现在哪几轮
4. 纯主观表述（如"建议就医"）不需要审核
5. 重点关注：功效/适应症是否准确、成分是否正确、禁忌是否遗漏关键警告、用法用量是否正确

## 输出格式
返回JSON数组，每个元素代表AI回答中的一条事实性陈述：
- "claim": AI回答中的原文陈述（30字以内，保持原意）
- "rounds": 出现在哪几轮（如 [1,2,3,5]）
- "verdict": "correct" | "wrong" | "unverified"
- "evidence": 知识库中的对应依据原文（50字以内），unverified时填"知识库未涉及"
- "correction": 如果wrong，正确说法是什么（50字以内），否则留空字符串

只返回JSON数组，不要任何其他文本。"""


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


def _build_answers_text(answers: list) -> str:
    parts = []
    for i, a in enumerate(answers):
        text = clean_text(a)
        truncated = text[:1500] + ("...（截断）" if len(text) > 1500 else "")
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
    raw = json.loads(text)
    return raw if isinstance(raw, list) else raw.get("results", raw.get("data", []))


async def verify_one_group(
    client: ModelClient,
    kb: dict,
    qid: str, model: str, search_enabled: bool,
    group_resps: list,
    semaphore: asyncio.Semaphore,
    counter: dict,
) -> list:
    product = group_resps[0].get("product", "")
    kb_key = PRODUCT_KB_MAP.get(product)
    if not kb_key or kb_key not in kb:
        counter["done"] += 1
        return []

    kb_text = kb[kb_key].get("text", "")
    if not kb_text:
        counter["done"] += 1
        return []

    question = group_resps[0].get("question_text", "")
    answers = [r.get("answer", "") for r in group_resps]
    answers_text = _build_answers_text(answers)

    async with semaphore:
        search_tag = "联网" if search_enabled else "不联网"
        counter["done"] += 1
        print(f"({counter['done']}/{counter['total']}) {product}×{model}×{search_tag} ({len(answers)}轮)")

        user_prompt = LLM_VERIFY_PROMPT.format(
            kb_text=kb_text,
            product_name=product,
            question=question,
            round_count=len(answers),
            answers_text=answers_text,
        )

        try:
            result = await client.query(
                question=f"你是药品信息准确性审核专家。严格按要求返回JSON。\n\n{user_prompt}",
                enable_search=False,
                temperature=0.1,
                max_tokens=4000,
                json_mode=True,
            )
            verdicts = _parse_json_response(result["answer"])

            # 添加元数据
            for v in verdicts:
                v["问题ID"] = qid
                v["产品"] = product
                v["模型"] = model
                v["联网"] = "是" if search_enabled else "否"
                v["轮次数"] = len(answers)

            return verdicts
        except Exception as e:
            print(f"  校对失败: {e}")
            return []


async def run_verification(api_key: str):
    kb = load_knowledge_base()
    responses = load_accuracy_responses()
    print(f"加载 {len(responses)} 条Q1/Q2应答")

    # 初始化DeepSeek客户端
    config_path = os.path.join(BASE_DIR, "config", "models.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_config = config["models"]["deepseek"]
    client = ModelClient("deepseek", ds_config, api_key)
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    # 按 (问题ID × 模型 × 联网) 分组
    groups = defaultdict(list)
    for resp in responses:
        key = (resp.get("question_id", ""), resp.get("model", ""), resp.get("search_enabled", False))
        groups[key].append(resp)

    counter = {"done": 0, "total": len(groups)}
    print(f"共 {counter['total']} 个分组，{LLM_CONCURRENCY} 并发")

    tasks = [
        verify_one_group(client, kb, qid, model, search, resps, semaphore, counter)
        for (qid, model, search), resps in groups.items()
    ]

    results = await asyncio.gather(*tasks)
    all_verdicts = [v for group in results for v in group]

    if not all_verdicts:
        print("未产生校对结果")
        return

    # ===== 明细表 =====
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    df = pd.DataFrame(all_verdicts)

    # 整理列顺序
    col_order = ["问题ID", "产品", "模型", "联网", "轮次数",
                 "claim", "rounds", "verdict", "evidence", "correction"]
    existing_cols = [c for c in col_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in col_order]
    df = df[existing_cols + extra_cols]

    detail_path = os.path.join(ANALYSIS_DIR, "accuracy_llm_verify.csv")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n校对明细 → {detail_path} ({len(df)} 条)")

    # ===== 汇总表 =====
    summary_rows = []
    for (product, model, search), gdf in df.groupby(["产品", "模型", "联网"]):
        total = len(gdf)
        correct = sum(gdf["verdict"] == "correct")
        wrong = sum(gdf["verdict"] == "wrong")
        unverified = sum(gdf["verdict"] == "unverified")

        # Top3 错误摘要
        wrong_claims = gdf[gdf["verdict"] == "wrong"]
        top_errors = "; ".join(
            f"{row['claim']}→{row.get('correction', '')}"
            for _, row in wrong_claims.head(3).iterrows()
        )

        summary_rows.append({
            "产品": product,
            "模型": model,
            "联网": search,
            "总陈述数": total,
            "正确": correct,
            "错误": wrong,
            "无依据": unverified,
            "准确率": round(correct / (correct + wrong), 3) if (correct + wrong) > 0 else "",
            "主要错误": top_errors,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["产品", "模型", "联网"]).reset_index(drop=True)
    summary_path = os.path.join(ANALYSIS_DIR, "accuracy_llm_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"校对汇总 → {summary_path}")

    # 打印概况
    total_claims = len(df)
    total_wrong = sum(df["verdict"] == "wrong")
    print(f"\n概况: {total_claims} 条陈述, {total_wrong} 条错误")


if __name__ == "__main__":
    keys_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
    with open(keys_path, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f)
    api_key = keys.get("deepseek", {}).get("api_key", "")
    if not api_key or api_key == "sk-xxx":
        print("错误: 请在 api_keys.yaml 中填写 DeepSeek API key")
        sys.exit(1)

    asyncio.run(run_verification(api_key))
