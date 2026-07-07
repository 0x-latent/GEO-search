"""
用知识库校对AI应答的准确性（Q1-Q2产品认知类问题）。
方向：应答→知识库（查错），不查遗漏。
每条应答单独发给DeepSeek校验，可对比同一问题不同轮次的准确率差异。

逐条校验，10并发。需手动触发执行。
"""
import asyncio
import json
import os
import re
import sys
import glob
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_clients import ModelClient, resolve_relay, resolve_route
from utils.similarity import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 与 05 一致：认 GEO_* 环境变量，支持 per-job 目录与用户级知识库
KB_PATH = os.environ.get("GEO_KB_FILE") or os.path.join(BASE_DIR, "config", "knowledge_base.json")
RAW_DIR = os.environ.get("GEO_RAW_DIR") or os.path.join(BASE_DIR, "results", "raw")
ANALYSIS_DIR = os.environ.get("GEO_ANALYSIS_DIR") or os.path.join(BASE_DIR, "results", "analysis")

LLM_CONCURRENCY = 10

# 校验哪些问题层级：默认 q1/q2（产品认知类）；自助任务注入 GEO_ACCURACY_LEVELS=all
_LEVELS_ENV = (os.environ.get("GEO_ACCURACY_LEVELS") or "q1,q2").strip().lower()
ACCURACY_LEVELS = None if _LEVELS_ENV == "all" else {t.strip() for t in _LEVELS_ENV.split(",") if t.strip()}

# 产品短名 → 知识库目录名（历史 8 产品的固定对照，动态匹配的快速路径）
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


def _qid_level_tag(qid: str) -> str:
    m = re.search(r"_(q\d)_", qid)
    return m.group(1) if m else ""


def _load_brand_alias_groups() -> list:
    """brands.yaml 的品牌名+别名分组，用于产品名↔知识库 key 的模糊匹配。"""
    path = os.environ.get("GEO_BRANDS_FILE") or os.path.join(BASE_DIR, "config", "brands.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []
    groups = []
    for name, info in (data.get("brand_999") or {}).items():
        groups.append({name, *(info or {}).get("aliases", [])})
    return groups


def build_kb_resolver(kb: dict):
    """返回 product → kb_key 的解析函数。

    顺序：精确命中 → 历史对照表 → 名称包含 → brands.yaml 别名同组。
    匹配不到返回 None（该条应答跳过校验，不报错）。
    """
    alias_groups = _load_brand_alias_groups()
    cache: dict = {}

    def _group_of(text: str):
        for group in alias_groups:
            if any(term and (term in text or text in term) for term in group):
                return group
        return None

    def resolve(product: str):
        if not product:
            return None
        if product in cache:
            return cache[product]
        key = None
        if product in kb:
            key = product
        if key is None:
            legacy = PRODUCT_KB_MAP.get(product)
            if legacy and legacy in kb:
                key = legacy
        if key is None:
            for candidate in kb:
                name = kb[candidate].get("product_name", "") if isinstance(kb[candidate], dict) else ""
                if candidate in product or product in candidate or (
                    name and (name in product or product in name)
                ):
                    key = candidate
                    break
        if key is None:
            product_group = _group_of(product)
            if product_group:
                for candidate in kb:
                    if _group_of(candidate) is product_group:
                        key = candidate
                        break
        cache[product] = key
        return key

    return resolve

# 按问题类型选择知识库模块（编号前缀）
Q1_MODULES = {"01", "02", "03", "04", "06", "07", "08", "16"}
Q2_MODULES = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "15", "16"}

LEVEL_MODULE_MAP = {
    "q1_overall": Q1_MODULES,
    "q2_detail": Q2_MODULES,
}


LLM_VERIFY_PROMPT = """你是药品信息准确性审核专家。请检查AI回答中是否有与产品知识库矛盾的内容。

## 任务
只检查AI回答中**说错的内容**，不要管AI没提到的内容（遗漏不算错）。

## 产品知识库（参考标准）
{kb_text}

## 待校验的AI回答
产品：{product_name}
问题：{question}

{answer_text}

## 审核规则
1. 逐句审查AI回答中的**事实性陈述**（功效、成分、用法、禁忌、适应症等）
2. 将每条陈述与知识库对比：
   - **correct**：与知识库一致
   - **wrong**：与知识库矛盾
   - **unverified**：知识库中没有对应信息，无法判断对错
3. 纯主观表述（如"建议就医"）不需要审核
4. 重点关注：功效/适应症是否准确、成分是否正确、禁忌是否遗漏关键警告、用法用量是否正确

## 输出格式
返回JSON数组，每个元素代表AI回答中的一条事实性陈述：
- "claim": AI回答中的原文陈述（30字以内，保持原意）
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
            if ACCURACY_LEVELS is None or _qid_level_tag(qid) in ACCURACY_LEVELS:
                responses.append(data)
        except Exception:
            continue
    return responses


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


def _get_kb_text_for_question(kb_entry: dict, qid: str) -> str:
    """根据问题ID中的level选择相关知识库模块，拼接为文本。"""
    modules = kb_entry.get("modules", {})
    if not modules:
        return kb_entry.get("text", "")

    level = None
    for lvl in LEVEL_MODULE_MAP:
        if f"_{lvl}" in qid:
            level = lvl
            break

    selected_ids = LEVEL_MODULE_MAP.get(level) if level else None
    if not selected_ids:
        selected_ids = {mid for mid in modules if mid not in {"00", "12", "13", "14", "17", "18"}}

    parts = []
    for mid in sorted(modules.keys()):
        if mid in selected_ids:
            parts.append(modules[mid]["text"])

    return "\n\n---\n\n".join(parts)


async def verify_one_response(
    client: ModelClient,
    kb: dict,
    resolve_kb_key,
    resp: dict,
    semaphore: asyncio.Semaphore,
    counter: dict,
) -> dict:
    """校验单条应答，返回该条的汇总结果。"""
    qid = resp.get("question_id", "")
    product = resp.get("product", "")
    model = resp.get("model", "")
    search_enabled = resp.get("search_enabled", False)
    round_num = resp.get("round", 0)
    question = resp.get("question_text", "")
    answer = resp.get("answer", "")

    kb_key = resolve_kb_key(product)
    if not kb_key or kb_key not in kb:
        counter["done"] += 1
        counter["no_kb"] += 1
        return None

    kb_text = _get_kb_text_for_question(kb[kb_key], qid)
    if not kb_text:
        counter["done"] += 1
        return None

    answer_text = clean_text(answer)
    answer_text = answer_text[:2000] + ("...（截断）" if len(answer_text) > 2000 else "")

    async with semaphore:
        search_tag = "联网" if search_enabled else "不联网"
        counter["done"] += 1
        print(f"  ({counter['done']}/{counter['total']}) {product}×{model}×{search_tag}×R{round_num}")

        user_prompt = LLM_VERIFY_PROMPT.format(
            kb_text=kb_text,
            product_name=product,
            question=question,
            answer_text=answer_text,
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
        except Exception as e:
            print(f"    校对失败: {e}")
            return None

        total = len(verdicts)
        correct = sum(1 for v in verdicts if v.get("verdict") == "correct")
        wrong = sum(1 for v in verdicts if v.get("verdict") == "wrong")
        unverified = sum(1 for v in verdicts if v.get("verdict") == "unverified")

        # 提取 level（先认标准层级，其次退化为 q 前缀标签，如用户上传的 q4）
        level = ""
        for lvl in ("q1_overall", "q2_detail", "q3_scenario1", "q4_scenario2", "q5_top3"):
            if f"_{lvl}" in qid:
                level = lvl
                break
        if not level:
            level = _qid_level_tag(qid)

        # 错误摘要
        wrong_items = [v for v in verdicts if v.get("verdict") == "wrong"]
        error_summary = "; ".join(
            f"{v.get('claim', '')}→{v.get('correction', '')}"
            for v in wrong_items[:3]
        )

        return {
            "产品": product,
            "问题ID": qid,
            "问题类型": level,
            "问题": question,
            "模型": model,
            "联网": "是" if search_enabled else "否",
            "轮次": round_num,
            "知识点数": total,
            "正确": correct,
            "错误": wrong,
            "无依据": unverified,
            "正确-可验证": f"{correct}|{correct + wrong}" if (correct + wrong) > 0 else "-",
            "准确率": round(correct / (correct + wrong), 3) if (correct + wrong) > 0 else "",
            "错误摘要": error_summary,
        }


async def run_verification():
    kb = load_knowledge_base()
    responses = load_accuracy_responses()
    levels_label = "全部层级" if ACCURACY_LEVELS is None else "/".join(sorted(ACCURACY_LEVELS))
    print(f"加载 {len(responses)} 条应答（{levels_label}），逐条校验")

    # 初始化 DeepSeek 客户端（与 05 一致：支持 new-api 中继）
    keys_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
    config_path = os.path.join(BASE_DIR, "config", "models.yaml")
    with open(keys_path, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    route = resolve_route(config, keys, os.environ.get("GEO_ROUTE") or None)
    relay_conf = resolve_relay(config, keys) if route == "relay" else None
    api_key = keys.get("deepseek", {}).get("api_key", "")
    if route != "relay" and (not api_key or api_key == "sk-xxx"):
        print("错误: 需要DeepSeek API key（或配置 relay 中继）")
        sys.exit(1)

    ds_config = config["models"]["deepseek"]
    client = ModelClient("deepseek", ds_config, api_key, route=route, relay_config=relay_conf)
    print(f"校验链路: {route}")
    resolve_kb_key = build_kb_resolver(kb)
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    counter = {"done": 0, "no_kb": 0, "total": len(responses)}
    print(f"共 {counter['total']} 条，{LLM_CONCURRENCY} 并发")

    tasks = [
        verify_one_response(client, kb, resolve_kb_key, resp, semaphore, counter)
        for resp in responses
    ]
    results = await asyncio.gather(*tasks)
    detail_rows = [r for r in results if r is not None]
    if counter["no_kb"]:
        print(f"  跳过 {counter['no_kb']} 条（产品未匹配到知识库）")

    if not detail_rows:
        print("未产生校对结果")
        return

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # ===== 1. 明细表：每条应答一行 =====
    detail_df = pd.DataFrame(detail_rows)
    detail_df = detail_df.sort_values(
        ["产品", "问题类型", "模型", "联网", "轮次"]
    ).reset_index(drop=True)
    detail_path = os.path.join(ANALYSIS_DIR, "accuracy_detail.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n明细表 → {detail_path} ({len(detail_df)} 行)")

    # ===== 2. 交叉统计汇总表 =====
    detail_df["_correct"] = detail_df["正确"].astype(int)
    detail_df["_wrong"] = detail_df["错误"].astype(int)
    detail_df["_unverified"] = detail_df["无依据"].astype(int)
    detail_df["_total"] = detail_df["知识点数"].astype(int)

    def _agg(gdf):
        c = gdf["_correct"].sum()
        w = gdf["_wrong"].sum()
        u = gdf["_unverified"].sum()
        t = gdf["_total"].sum()
        rate = round(c / (c + w), 3) if (c + w) > 0 else None
        return pd.Series({
            "知识点数": t, "正确": c, "错误": w, "无依据": u, "准确率": rate,
        })

    summary_parts = []

    # 按模型×联网
    g = detail_df.groupby(["模型", "联网"]).apply(_agg, include_groups=False).reset_index()
    g.insert(0, "维度", "模型×联网")
    g["分组"] = g["模型"] + " / " + g["联网"]
    summary_parts.append(g[["维度", "分组", "知识点数", "正确", "错误", "无依据", "准确率"]])

    # 按产品×联网
    g = detail_df.groupby(["产品", "联网"]).apply(_agg, include_groups=False).reset_index()
    g.insert(0, "维度", "产品×联网")
    g["分组"] = g["产品"] + " / " + g["联网"]
    summary_parts.append(g[["维度", "分组", "知识点数", "正确", "错误", "无依据", "准确率"]])

    # 按产品×模型
    g = detail_df.groupby(["产品", "模型"]).apply(_agg, include_groups=False).reset_index()
    g.insert(0, "维度", "产品×模型")
    g["分组"] = g["产品"] + " / " + g["模型"]
    summary_parts.append(g[["维度", "分组", "知识点数", "正确", "错误", "无依据", "准确率"]])

    # 按轮次（核心：对比轮次间差异）
    g = detail_df.groupby(["轮次"]).apply(_agg, include_groups=False).reset_index()
    g.insert(0, "维度", "轮次")
    g["分组"] = "第" + g["轮次"].astype(str) + "轮"
    summary_parts.append(g[["维度", "分组", "知识点数", "正确", "错误", "无依据", "准确率"]])

    # 按模型×轮次（各模型的轮次稳定性）
    g = detail_df.groupby(["模型", "轮次"]).apply(_agg, include_groups=False).reset_index()
    g.insert(0, "维度", "模型×轮次")
    g["分组"] = g["模型"] + " / 第" + g["轮次"].astype(str) + "轮"
    summary_parts.append(g[["维度", "分组", "知识点数", "正确", "错误", "无依据", "准确率"]])

    # 按模型（总计）
    g = detail_df.groupby(["模型"]).apply(_agg, include_groups=False).reset_index()
    g.insert(0, "维度", "模型")
    g["分组"] = g["模型"]
    summary_parts.append(g[["维度", "分组", "知识点数", "正确", "错误", "无依据", "准确率"]])

    # 按产品（总计）
    g = detail_df.groupby(["产品"]).apply(_agg, include_groups=False).reset_index()
    g.insert(0, "维度", "产品")
    g["分组"] = g["产品"]
    summary_parts.append(g[["维度", "分组", "知识点数", "正确", "错误", "无依据", "准确率"]])

    # 总计
    overall = _agg(detail_df)
    overall_row = pd.DataFrame([{
        "维度": "总计", "分组": "全部", **overall.to_dict(),
    }])
    summary_parts.append(overall_row[["维度", "分组", "知识点数", "正确", "错误", "无依据", "准确率"]])

    summary_df = pd.concat(summary_parts, ignore_index=True)
    summary_path = os.path.join(ANALYSIS_DIR, "accuracy_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"汇总表 → {summary_path}")

    # 打印概况
    print(f"\n概况: {len(detail_df)} 条应答, "
          f"{detail_df['_total'].sum()} 个知识点, "
          f"准确率 {overall.get('准确率', '-')}")


if __name__ == "__main__":
    asyncio.run(run_verification())
