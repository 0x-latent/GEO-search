"""
用LLM提取AI应答中的产品推荐信息。
针对问题3-5（场景推荐类和Top3推荐类），从回答中提取：
- 推荐的产品列表（按推荐强度排序）
- 每个产品的推荐理由
- 推荐类型（强推荐/可选/提及）
- 产品所属品类

使用DeepSeek V3.2，成本低、中文理解好。
支持断点续跑、10并发。需手动触发执行。
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
ANALYSIS_DIR = os.path.join(BASE_DIR, "results", "analysis")
EXTRACT_DIR = os.path.join(BASE_DIR, "results", "extractions")
EXTRACT_LOG_PATH = os.path.join(EXTRACT_DIR, "extraction_log.json")

CONCURRENCY = 10

SYSTEM_PROMPT = """你是一个药品推荐信息提取专家。你的任务是从AI的回答中提取产品推荐信息。

请从回答中提取所有被提及的药品/产品，并判断其推荐强度。

输出格式：严格返回JSON数组，按推荐强度从高到低排列，每个元素包含：
- "rank": 推荐排名（1为最强推荐）
- "product": 产品名称（保持原文中的写法）
- "strength": 推荐强度
  - "strong": 明确推荐、首选、第一推荐
  - "moderate": 可以考虑、也不错、备选
  - "mention": 仅提及但未明确推荐
  - "caution": 提及但附带警告或不建议
- "reason": 推荐理由（从原文中提取，简明扼要，30字以内）
- "category": 产品品类（如"感冒药"、"止咳药"、"皮肤药"等）

注意：
1. 只提取具体的药品/产品名，不提取成分名（如"对乙酰氨基酚"是成分不是产品，除非回答中明确作为产品推荐）
2. 如果回答说"建议就医"而未推荐具体产品，返回空数组 []
3. "不建议使用X"算caution，不算推荐
4. 同一产品的不同规格/剂型合并为一条

只返回JSON数组，不要任何其他文本。"""

USER_PROMPT = """问题：{question}

AI的回答：
{answer}

请提取回答中所有被推荐或提及的药品产品。"""


def _make_extract_key(resp: dict) -> str:
    """生成唯一键：question_id + model + search + round"""
    return f"{resp.get('question_id', '')}_{resp.get('model', '')}_{resp.get('search_enabled', '')}_{resp.get('round', '')}"


def load_extract_log() -> dict:
    """加载提取日志（已完成的任务集合）"""
    if os.path.exists(EXTRACT_LOG_PATH):
        with open(EXTRACT_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}}


def save_extract_log(log: dict):
    with open(EXTRACT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def load_recommendation_responses() -> list:
    """加载推荐类问题（q3/q4/q5）的应答"""
    responses = []
    pattern = os.path.join(RAW_DIR, "**", "*.json")
    for fpath in glob.glob(pattern, recursive=True):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "question_id" not in data or "answer" not in data:
                continue
            qid = data["question_id"]
            if "_q3_" in qid or "_q4_" in qid or "_q5_" in qid:
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
    return json.loads(text)


async def extract_one(client, resp: dict, semaphore: asyncio.Semaphore,
                       log: dict, lock: asyncio.Lock, counter: dict) -> tuple:
    """提取单条应答的推荐信息（带断点和并发控制）"""
    key = _make_extract_key(resp)

    # 断点：已完成则直接返回缓存结果
    if key in log["completed"]:
        counter["skip"] += 1
        return resp, log["completed"][key]

    question = resp.get("question_text", "")
    answer = resp.get("answer", "")
    full_question = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(question=question, answer=answer)}"

    async with semaphore:
        try:
            result = await client.query(
                question=full_question,
                enable_search=False,
                temperature=0.1,
                max_tokens=2000,
            )
            recs = _parse_json_response(result["answer"])
        except Exception as e:
            counter["fail"] += 1
            if counter["fail"] <= 5:
                print(f"  提取失败: {e}")
            recs = []

        # 记录到日志
        async with lock:
            log["completed"][key] = recs
            counter["done"] += 1
            # 每50条保存一次
            if counter["done"] % 50 == 0:
                save_extract_log(log)
                print(f"  进度: {counter['done']}/{counter['total']} (跳过{counter['skip']}, 失败{counter['fail']})")

        return resp, recs


async def main():
    keys_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
    config_path = os.path.join(BASE_DIR, "config", "models.yaml")

    with open(keys_path, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_key = keys.get("deepseek", {}).get("api_key", "")
    if not ds_key or ds_key == "sk-xxx":
        print("错误: 需要DeepSeek API key")
        sys.exit(1)

    ds_config = config["models"]["deepseek"]
    client = ModelClient("deepseek", ds_config, ds_key)

    # 加载应答
    responses = load_recommendation_responses()
    print(f"加载 {len(responses)} 条推荐类应答")

    if not responses:
        print("没有找到推荐类应答数据")
        return

    # 加载断点日志
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    log = load_extract_log()
    already_done = sum(1 for r in responses if _make_extract_key(r) in log["completed"])
    print(f"已完成 {already_done}/{len(responses)}（断点续跑）")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    counter = {"done": 0, "skip": 0, "fail": 0, "total": len(responses) - already_done}

    print(f"开始提取，并发 {CONCURRENCY}，待处理 {counter['total']} 条...")

    tasks = [
        extract_one(client, resp, semaphore, log, lock, counter)
        for resp in responses
    ]
    all_results = await asyncio.gather(*tasks)

    # 最终保存日志
    save_extract_log(log)

    # 整理结果
    detail_rows = []
    for resp, recs in all_results:
        qid = resp.get("question_id", "")
        product = resp.get("product", "")
        model = resp.get("model", "")
        search = resp.get("search_enabled", False)
        round_num = resp.get("round", "")

        for rec in recs:
            detail_rows.append({
                "问题ID": qid,
                "产品": product,
                "模型": model,
                "联网": "是" if search else "否",
                "轮次": round_num,
                "推荐排名": rec.get("rank", ""),
                "推荐产品": rec.get("product", ""),
                "推荐强度": rec.get("strength", ""),
                "推荐理由": rec.get("reason", ""),
                "品类": rec.get("category", ""),
            })

    # 保存明细
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_path = os.path.join(ANALYSIS_DIR, "recommendation_detail.csv")
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"\n推荐提取明细 → {detail_path} ({len(detail_rows)} 条)")

        # 汇总
        summary = defaultdict(lambda: {"strong": 0, "moderate": 0, "mention": 0, "caution": 0, "total": 0})
        for row in detail_rows:
            key = (row["产品"], row["模型"], row["推荐产品"])
            strength = row["推荐强度"]
            summary[key]["total"] += 1
            if strength in summary[key]:
                summary[key][strength] += 1

        sum_rows = []
        for (product, model, rec_product), counts in summary.items():
            sum_rows.append({
                "场景产品": product,
                "模型": model,
                "被推荐产品": rec_product,
                "强推荐次数": counts["strong"],
                "可选次数": counts["moderate"],
                "仅提及次数": counts["mention"],
                "警告次数": counts["caution"],
                "总出现次数": counts["total"],
            })

        sum_df = pd.DataFrame(sum_rows)
        sum_df = sum_df.sort_values(["场景产品", "模型", "总出现次数"], ascending=[True, True, False])
        sum_path = os.path.join(ANALYSIS_DIR, "recommendation_extracted_summary.csv")
        sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")
        print(f"推荐提取汇总 → {sum_path}")

        # 生成统计分析报表
        print("\n生成统计分析...")
        generate_statistics(detail_df, responses)

        print("\n更新Dashboard...")
        generate_updated_dashboard(detail_df, responses)
    else:
        print("未提取到任何推荐信息")

    print(f"\n完成: 处理 {counter['done']}条, 跳过 {counter['skip']}条, 失败 {counter['fail']}条, 提取 {len(detail_rows)} 条推荐")


def _is_999_product(name: str) -> bool:
    """判断产品名是否属于999品牌"""
    keywords = ["999", "三九", "感冒灵", "皮炎平", "养胃舒", "胃泰",
                "抗病毒口服液", "小儿氨酚黄那敏", "强力枇杷露",
                "澳诺", "葡萄糖酸锌钙", "易善复", "多烯磷脂酰胆碱"]
    return any(kw in name for kw in keywords)


def generate_statistics(detail_df: pd.DataFrame, responses: list):
    """基于提取明细生成统计分析报表"""

    # ===== 1. 999推荐总览 =====
    # 以原始应答为基数，统计每个产品×模型×联网下999的推荐情况
    # 先构建应答级别的汇总（每条应答中999是否被推荐、排名多少）
    resp_index = {}
    for resp in responses:
        key = f"{resp.get('question_id', '')}_{resp.get('model', '')}_{resp.get('search_enabled', '')}_{resp.get('round', '')}"
        resp_index[key] = resp

    # 每条应答的999推荐情况
    answer_999 = defaultdict(lambda: {"has_strong": False, "has_any": False, "best_rank": None})
    for _, row in detail_df.iterrows():
        ans_key = f"{row['问题ID']}_{row['模型']}_{row['联网'] == '是'}_{row['轮次']}"
        if _is_999_product(row["推荐产品"]):
            info = answer_999[ans_key]
            info["has_any"] = True
            if row["推荐强度"] == "strong":
                info["has_strong"] = True
            rank = row["推荐排名"]
            if isinstance(rank, (int, float)) and (info["best_rank"] is None or rank < info["best_rank"]):
                info["best_rank"] = rank

    # 按产品×模型×联网聚合
    overview_groups = defaultdict(lambda: {"total": 0, "strong_count": 0, "any_count": 0, "ranks": []})
    for resp in responses:
        key = f"{resp.get('question_id', '')}_{resp.get('model', '')}_{resp.get('search_enabled', '')}_{resp.get('round', '')}"
        product = resp.get("product", "")
        model = resp.get("model", "")
        search = "是" if resp.get("search_enabled") else "否"
        gk = (product, model, search)
        overview_groups[gk]["total"] += 1

        info = answer_999.get(key, {})
        if info.get("has_strong"):
            overview_groups[gk]["strong_count"] += 1
        if info.get("has_any"):
            overview_groups[gk]["any_count"] += 1
        if info.get("best_rank") is not None:
            overview_groups[gk]["ranks"].append(info["best_rank"])

    overview_rows = []
    for (product, model, search), g in overview_groups.items():
        total = g["total"]
        ranks = g["ranks"]
        overview_rows.append({
            "产品": product,
            "模型": model,
            "联网": search,
            "应答总数": total,
            "999强推荐次数": g["strong_count"],
            "999强推荐率": round(g["strong_count"] / total, 3) if total else 0,
            "999被提及次数": g["any_count"],
            "999提及率": round(g["any_count"] / total, 3) if total else 0,
            "999平均排名": round(sum(ranks) / len(ranks), 2) if ranks else "",
            "999最佳排名": min(ranks) if ranks else "",
        })

    overview_df = pd.DataFrame(overview_rows)
    overview_df = overview_df.sort_values(["产品", "模型", "联网"]).reset_index(drop=True)
    path = os.path.join(ANALYSIS_DIR, "rec_999_overview.csv")
    overview_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  999推荐总览 → {path}")

    # ===== 2. 竞品推荐排行 =====
    # 每个场景产品×模型下，被推荐最多的Top15竞品
    comp_groups = defaultdict(lambda: defaultdict(lambda: {"strong": 0, "moderate": 0, "total": 0}))
    for _, row in detail_df.iterrows():
        if _is_999_product(row["推荐产品"]):
            continue
        key = (row["产品"], row["模型"])
        rec = row["推荐产品"]
        comp_groups[key][rec]["total"] += 1
        if row["推荐强度"] == "strong":
            comp_groups[key][rec]["strong"] += 1
        elif row["推荐强度"] == "moderate":
            comp_groups[key][rec]["moderate"] += 1

    comp_rows = []
    for (product, model), recs in comp_groups.items():
        sorted_recs = sorted(recs.items(), key=lambda x: x[1]["total"], reverse=True)[:15]
        for rank, (rec_name, counts) in enumerate(sorted_recs, 1):
            comp_rows.append({
                "场景产品": product,
                "模型": model,
                "排名": rank,
                "竞品": rec_name,
                "强推荐次数": counts["strong"],
                "可选次数": counts["moderate"],
                "总出现次数": counts["total"],
            })

    comp_df = pd.DataFrame(comp_rows)
    path = os.path.join(ANALYSIS_DIR, "rec_competitor_ranking.csv")
    comp_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  竞品推荐排行 → {path}")

    # ===== 3. 推荐理由分析 =====
    # 999 vs 竞品的推荐理由高频词
    from collections import Counter
    reasons_999 = []
    reasons_comp = []
    for _, row in detail_df.iterrows():
        reason = str(row.get("推荐理由", "")).strip()
        if not reason or reason == "nan":
            continue
        if _is_999_product(row["推荐产品"]):
            reasons_999.append(reason)
        else:
            reasons_comp.append(reason)

    reason_rows = []
    # 999产品的推荐理由
    reason_counter_999 = Counter(reasons_999)
    for reason, count in reason_counter_999.most_common(30):
        reason_rows.append({"类型": "999产品", "推荐理由": reason, "出现次数": count})

    # 竞品的推荐理由
    reason_counter_comp = Counter(reasons_comp)
    for reason, count in reason_counter_comp.most_common(30):
        reason_rows.append({"类型": "竞品", "推荐理由": reason, "出现次数": count})

    reason_df = pd.DataFrame(reason_rows)
    path = os.path.join(ANALYSIS_DIR, "rec_reasons.csv")
    reason_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  推荐理由分析 → {path}")

    # ===== 4. 联网影响分析 =====
    # 对比联网和不联网下999的推荐变化
    search_groups = defaultdict(lambda: {"nosearch": {"total": 0, "strong": 0, "any": 0, "ranks": []},
                                          "search": {"total": 0, "strong": 0, "any": 0, "ranks": []}})
    for resp in responses:
        product = resp.get("product", "")
        model = resp.get("model", "")
        mode = "search" if resp.get("search_enabled") else "nosearch"
        key = f"{resp.get('question_id', '')}_{resp.get('model', '')}_{resp.get('search_enabled', '')}_{resp.get('round', '')}"

        gk = (product, model)
        search_groups[gk][mode]["total"] += 1

        info = answer_999.get(key, {})
        if info.get("has_strong"):
            search_groups[gk][mode]["strong"] += 1
        if info.get("has_any"):
            search_groups[gk][mode]["any"] += 1
        if info.get("best_rank") is not None:
            search_groups[gk][mode]["ranks"].append(info["best_rank"])

    impact_rows = []
    for (product, model), modes in search_groups.items():
        ns = modes["nosearch"]
        s = modes["search"]
        if ns["total"] == 0 or s["total"] == 0:
            continue

        ns_strong_rate = round(ns["strong"] / ns["total"], 3)
        s_strong_rate = round(s["strong"] / s["total"], 3)
        ns_any_rate = round(ns["any"] / ns["total"], 3)
        s_any_rate = round(s["any"] / s["total"], 3)
        ns_avg_rank = round(sum(ns["ranks"]) / len(ns["ranks"]), 2) if ns["ranks"] else ""
        s_avg_rank = round(sum(s["ranks"]) / len(s["ranks"]), 2) if s["ranks"] else ""

        impact_rows.append({
            "产品": product,
            "模型": model,
            "不联网_强推荐率": ns_strong_rate,
            "联网_强推荐率": s_strong_rate,
            "强推荐率变化": round(s_strong_rate - ns_strong_rate, 3),
            "不联网_提及率": ns_any_rate,
            "联网_提及率": s_any_rate,
            "提及率变化": round(s_any_rate - ns_any_rate, 3),
            "不联网_平均排名": ns_avg_rank,
            "联网_平均排名": s_avg_rank,
        })

    if impact_rows:
        impact_df = pd.DataFrame(impact_rows)
        impact_df = impact_df.sort_values(["产品", "模型"]).reset_index(drop=True)
        path = os.path.join(ANALYSIS_DIR, "rec_search_impact.csv")
        impact_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  联网影响分析 → {path}")


def generate_updated_dashboard(detail_df: pd.DataFrame, responses: list):
    """
    基于LLM提取的推荐数据，生成修正后的Dashboard。
    只用Q3-Q5数据计算999提及率和推荐率（排除Q1-Q2带品牌名的问题）。
    """
    # 构建每条应答的999推荐情况
    answer_999 = {}
    for _, row in detail_df.iterrows():
        ans_key = f"{row['问题ID']}_{row['模型']}_{row['联网'] == '是'}_{row['轮次']}"
        if ans_key not in answer_999:
            answer_999[ans_key] = {"has_strong": False, "has_moderate": False, "has_any": False, "best_rank": None}
        if _is_999_product(row["推荐产品"]):
            info = answer_999[ans_key]
            info["has_any"] = True
            if row["推荐强度"] == "strong":
                info["has_strong"] = True
            if row["推荐强度"] == "moderate":
                info["has_moderate"] = True
            rank = row["推荐排名"]
            if isinstance(rank, (int, float)) and (info["best_rank"] is None or rank < info["best_rank"]):
                info["best_rank"] = rank

    # 按产品×模型聚合
    groups = defaultdict(lambda: {
        "nosearch": {"total": 0, "strong": 0, "any": 0, "ranks": []},
        "search": {"total": 0, "strong": 0, "any": 0, "ranks": []},
        "all": {"total": 0, "strong": 0, "any": 0, "ranks": []},
        "competitors": defaultdict(int),
    })

    for resp in responses:
        product = resp.get("product", "")
        model = resp.get("model", "")
        search = resp.get("search_enabled", False)
        mode = "search" if search else "nosearch"
        ans_key = f"{resp.get('question_id', '')}_{model}_{search}_{resp.get('round', '')}"

        gk = (product, model)
        groups[gk][mode]["total"] += 1
        groups[gk]["all"]["total"] += 1

        info = answer_999.get(ans_key, {})
        for m in [mode, "all"]:
            if info.get("has_strong"):
                groups[gk][m]["strong"] += 1
            if info.get("has_any"):
                groups[gk][m]["any"] += 1
            if info.get("best_rank") is not None:
                groups[gk][m]["ranks"].append(info["best_rank"])

    # 竞品统计（从detail_df中提取）
    for _, row in detail_df.iterrows():
        if _is_999_product(row["推荐产品"]):
            continue
        gk = (row["产品"], row["模型"])
        groups[gk]["competitors"][row["推荐产品"]] += 1

    # 生成Dashboard
    rows = []
    for (product, model), g in groups.items():
        a = g["all"]
        ns = g["nosearch"]
        s = g["search"]

        # Top3竞品
        top_comp = [c for c, _ in sorted(g["competitors"].items(), key=lambda x: -x[1])[:3]]

        def _rate(num, den):
            return round(num / den, 3) if den > 0 else ""

        def _avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else ""

        rows.append({
            "产品": product,
            "模型": model,
            "Q3-Q5应答总数": a["total"],
            "999强推荐率(整体)": _rate(a["strong"], a["total"]),
            "999提及率(整体)": _rate(a["any"], a["total"]),
            "999强推荐率(不联网)": _rate(ns["strong"], ns["total"]),
            "999提及率(不联网)": _rate(ns["any"], ns["total"]),
            "999强推荐率(联网)": _rate(s["strong"], s["total"]),
            "999提及率(联网)": _rate(s["any"], s["total"]),
            "999平均排名": _avg(a["ranks"]),
            "999最佳排名": min(a["ranks"]) if a["ranks"] else "",
            "Top3竞品": " / ".join(top_comp),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["产品", "模型"]).reset_index(drop=True)
    path = os.path.join(ANALYSIS_DIR, "dashboard.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Dashboard已更新（基于Q3-Q5 LLM提取数据） → {path}")


if __name__ == "__main__":
    asyncio.run(main())
