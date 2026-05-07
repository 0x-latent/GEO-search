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
                json_mode=True,
            )
            raw_json = json.loads(result["answer"])
            recs = raw_json if isinstance(raw_json, list) else raw_json.get("results", raw_json.get("data", []))
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

        # 生成统计分析报表
        print("\n生成统计分析...")
        generate_statistics(detail_df, responses)

        print("\n更新04报表（基于LLM提取数据覆盖正则版本）...")
        generate_updated_dashboard(detail_df, responses)
        generate_updated_stability(detail_df, responses)
        generate_updated_search_diff(detail_df, responses)
        generate_updated_optimization(detail_df, responses)

        print("\nV6框架补充报表...")
        generate_brand_generic_split(detail_df, responses)
        generate_unified_mention_report(detail_df, responses)
    else:
        print("未提取到任何推荐信息")

    print(f"\n完成: 处理 {counter['done']}条, 跳过 {counter['skip']}条, 失败 {counter['fail']}条, 提取 {len(detail_rows)} 条推荐")


def _load_brand_matchers():
    """从brands.yaml加载产品匹配表，返回 (999匹配列表, 竞品匹配列表)。
    每个匹配项为 (关键词, 标准名)，按关键词长度降序排列（优先匹配更长的名称）。
    数据来源：
      - 999: brand_999 (名称+aliases)
      - 竞品: known_brand_competitors + generic_names
    """
    config_path = os.path.join(BASE_DIR, "config", "brands.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    matchers_999 = []  # [(关键词, 标准名), ...]
    for name, info in cfg.get("brand_999", {}).items():
        matchers_999.append((name, name))
        for alias in info.get("aliases", []):
            matchers_999.append((alias, name))

    matchers_comp = []  # [(关键词, 标准名), ...]
    for name in cfg.get("known_brand_competitors", []):
        matchers_comp.append((name, name))
    for name in cfg.get("generic_names", []):
        matchers_comp.append((name, name))

    # 按关键词长度降序，优先匹配更长的名称
    matchers_999.sort(key=lambda x: len(x[0]), reverse=True)
    matchers_comp.sort(key=lambda x: len(x[0]), reverse=True)
    return matchers_999, matchers_comp


# 模块级缓存，首次调用时加载
_MATCHERS_999 = None
_MATCHERS_COMP = None


def _get_matchers():
    global _MATCHERS_999, _MATCHERS_COMP
    if _MATCHERS_999 is None:
        _MATCHERS_999, _MATCHERS_COMP = _load_brand_matchers()
    return _MATCHERS_999, _MATCHERS_COMP


def _match_products(name: str) -> tuple:
    """对推荐产品名做子串匹配，返回 (matched_999: set, matched_comp: set)。
    一条推荐产品可能同时命中999产品和竞品（如 '布洛芬（如"芬必得"）'）。
    使用位置级最长匹配：如果短关键词（如"泰诺"）的每次出现都被更长关键词
    （如"泰诺林"）完全覆盖，则该短关键词不计入结果。
    """
    matchers_999, matchers_comp = _get_matchers()

    # 第一步：找出所有匹配及其位置 [(start, end, std_name, type), ...]
    hits = []
    for kw, std_name in matchers_999:
        start = 0
        while True:
            pos = name.find(kw, start)
            if pos == -1:
                break
            hits.append((pos, pos + len(kw), std_name, "999"))
            start = pos + 1
    for kw, std_name in matchers_comp:
        start = 0
        while True:
            pos = name.find(kw, start)
            if pos == -1:
                break
            hits.append((pos, pos + len(kw), std_name, "comp"))
            start = pos + 1

    # 第二步：位置级去重——如果某次命中被另一个更长的命中完全覆盖，则丢弃
    # 按区间长度降序，长的优先保留
    hits.sort(key=lambda x: -(x[1] - x[0]))
    kept = []
    for h in hits:
        covered = any(k[0] <= h[0] and k[1] >= h[1] and (k[1] - k[0]) > (h[1] - h[0])
                       for k in kept)
        if not covered:
            kept.append(h)

    matched_999 = set(h[2] for h in kept if h[3] == "999")
    matched_comp = set(h[2] for h in kept if h[3] == "comp")
    return matched_999, matched_comp


def _is_999_product(name: str) -> bool:
    """判断产品名是否属于999品牌（基于brands.yaml子串匹配）"""
    matched_999, _ = _match_products(name)
    return len(matched_999) > 0


def generate_statistics(detail_df: pd.DataFrame, responses: list):
    """基于提取明细生成统计分析报表。
    输出：
      - rec_overview.csv    统一推荐总览（999本品+品牌竞品+通用名，口径一致）
      - rec_reasons.csv     推荐理由分析
    """
    config = _load_brand_config()
    known_brands = set(config.get("known_brand_competitors", []))
    generic_names = set(config.get("generic_names", []))
    component_kw = config.get("component_keywords", [])

    def _classify_name(name: str) -> str:
        """分类：999品牌 / 品牌竞品 / 通用名 / 成分品类级 / 未识别"""
        for b in known_brands:
            if b in name:
                return "品牌竞品"
        for g in generic_names:
            if g in name:
                return "通用名"
        for c in component_kw:
            if c in name:
                return "成分品类级"
        return "未识别"

    _strength_order = {"strong": 3, "moderate": 2, "mention": 1, "caution": 0}

    # ===== 1. 统一推荐总览 =====
    # 统一口径：按应答去重，每条应答中同一标准产品名只计1次，取最高强度
    # 维度：产品 × 模型 × 联网 × 被推荐产品 × 名称类型

    # 第一步：构建应答总数基数（产品×模型×联网）
    answer_totals = defaultdict(int)  # {(产品, 模型, 联网): 应答总数}
    for resp in responses:
        product = resp.get("product", "")
        model = resp.get("model", "")
        search = "是" if resp.get("search_enabled") else "否"
        answer_totals[(product, model, search)] += 1

    # 第二步：按应答聚合每个标准产品名的最高强度
    # answer_products[应答key] = {标准产品名: (最高强度, 名称类型)}
    answer_products = defaultdict(dict)
    for _, row in detail_df.iterrows():
        rec_raw = str(row["推荐产品"])
        matched_999, matched_comp = _match_products(rec_raw)
        ans_key = (row["产品"], row["模型"], row["联网"],
                   f"{row['问题ID']}_{row['轮次']}")
        new_s = row["推荐强度"]

        # 999产品
        for std_name in matched_999:
            cur = answer_products[ans_key].get(std_name)
            if cur is None or _strength_order.get(new_s, 0) > _strength_order.get(cur[0], 0):
                answer_products[ans_key][std_name] = (new_s, "999品牌")

        # 竞品（品牌竞品 / 通用名）
        for std_name in matched_comp:
            name_type = _classify_name(std_name)
            cur = answer_products[ans_key].get(std_name)
            if cur is None or _strength_order.get(new_s, 0) > _strength_order.get(cur[0], 0):
                answer_products[ans_key][std_name] = (new_s, name_type)

        # 未命中任何已知产品 → 保留原文
        if not matched_999 and not matched_comp:
            fallback = rec_raw.strip()
            name_type = _classify_name(fallback)
            cur = answer_products[ans_key].get(fallback)
            if cur is None or _strength_order.get(new_s, 0) > _strength_order.get(cur[0], 0):
                answer_products[ans_key][fallback] = (new_s, name_type)

    # 第三步：按 (产品, 模型, 联网, 被推荐产品, 名称类型) 聚合
    rec_groups = defaultdict(lambda: {"strong": 0, "moderate": 0, "total": 0})
    for (product, model, search, _ak), recs in answer_products.items():
        for rec_name, (strength, name_type) in recs.items():
            gk = (product, model, search, rec_name, name_type)
            rec_groups[gk]["total"] += 1
            if strength == "strong":
                rec_groups[gk]["strong"] += 1
            elif strength == "moderate":
                rec_groups[gk]["moderate"] += 1

    # 第四步：输出，每个 (产品, 模型, 联网) 内按总出现次数降序排名
    overview_rows = []
    # 按 (产品, 模型, 联网) 分组排名
    group_recs = defaultdict(list)
    for (product, model, search, rec_name, name_type), counts in rec_groups.items():
        group_recs[(product, model, search)].append((rec_name, name_type, counts))

    for (product, model, search), recs in group_recs.items():
        total_answers = answer_totals.get((product, model, search), 0)
        sorted_recs = sorted(recs, key=lambda x: x[2]["total"], reverse=True)
        for rank, (rec_name, name_type, counts) in enumerate(sorted_recs, 1):
            overview_rows.append({
                "产品": product,
                "模型": model,
                "联网": search,
                "应答总数": total_answers,
                "排名": rank,
                "被推荐产品": rec_name,
                "名称类型": name_type,
                "提及次数": counts["total"],
                "提及率": round(counts["total"] / total_answers, 3) if total_answers else 0,
                "强推荐次数": counts["strong"],
                "强推荐率": round(counts["strong"] / total_answers, 3) if total_answers else 0,
                "可选次数": counts["moderate"],
            })

    overview_df = pd.DataFrame(overview_rows)
    overview_df = overview_df.sort_values(
        ["产品", "模型", "联网", "排名"]).reset_index(drop=True)
    path = os.path.join(ANALYSIS_DIR, "rec_overview.csv")
    overview_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  统一推荐总览 → {path} ({len(overview_df)} 条)")

    # ===== 2. 推荐理由分析 =====
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
    reason_counter_999 = Counter(reasons_999)
    for reason, count in reason_counter_999.most_common(30):
        reason_rows.append({"类型": "999产品", "推荐理由": reason, "出现次数": count})
    reason_counter_comp = Counter(reasons_comp)
    for reason, count in reason_counter_comp.most_common(30):
        reason_rows.append({"类型": "竞品", "推荐理由": reason, "出现次数": count})

    reason_df = pd.DataFrame(reason_rows)
    path = os.path.join(ANALYSIS_DIR, "rec_reasons.csv")
    reason_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  推荐理由分析 → {path}")


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

    # 竞品统计（从detail_df中提取，用_match_products归一化）
    for _, row in detail_df.iterrows():
        rec_raw = str(row["推荐产品"])
        _, matched_comp = _match_products(rec_raw)
        if not matched_comp:
            # 未命中999也未命中竞品 → 保留原文
            matched_999, _ = _match_products(rec_raw)
            if not matched_999:
                matched_comp = {rec_raw.strip()}
            else:
                continue
        gk = (row["产品"], row["模型"])
        for comp_name in matched_comp:
            groups[gk]["competitors"][comp_name] += 1

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


def generate_updated_stability(detail_df: pd.DataFrame, responses: list):
    """
    基于LLM提取的推荐数据，重新生成稳定性报表。
    覆盖04脚本基于正则提取的版本，更准确。

    稳定性衡量两个维度：
    1. 推荐产品集合一致性（Jaccard）：多轮回答推荐的产品是否一样
    2. 推荐排序一致性：多轮的排序是否相同
    """
    # 按 (问题ID, 模型, 联网) 分组，每轮的推荐产品集合
    from collections import defaultdict

    # 先构建每条应答的推荐产品集合和排序
    answer_recs = defaultdict(lambda: {"products": set(), "ranking": []})
    for _, row in detail_df.iterrows():
        ans_key = f"{row['问题ID']}_{row['模型']}_{row['联网'] == '是'}_{row['轮次']}"
        answer_recs[ans_key]["products"].add(row["推荐产品"])
        # 按rank排序构建ranking
        rank = row["推荐排名"]
        if isinstance(rank, (int, float)) and row["推荐强度"] in ("strong", "moderate"):
            answer_recs[ans_key]["ranking"].append((int(rank), row["推荐产品"]))

    # 按 (问题ID, 模型, 联网) 分组
    groups = defaultdict(list)
    for resp in responses:
        qid = resp.get("question_id", "")
        model = resp.get("model", "")
        search = resp.get("search_enabled", False)
        round_num = resp.get("round", "")
        ans_key = f"{qid}_{model}_{search}_{round_num}"

        groups[(qid, model, search)].append({
            "round": round_num,
            "product": resp.get("product", ""),
            "products": answer_recs.get(ans_key, {}).get("products", set()),
            "ranking": answer_recs.get(ans_key, {}).get("ranking", []),
        })

    rows = []
    for (qid, model, search), rounds in groups.items():
        if len(rounds) < 2:
            continue

        product = rounds[0]["product"]

        # 1. 推荐产品集合的Jaccard相似度
        product_sets = [frozenset(r["products"]) for r in rounds]
        jaccards = []
        for i in range(len(product_sets)):
            for j in range(i + 1, len(product_sets)):
                union = product_sets[i] | product_sets[j]
                inter = product_sets[i] & product_sets[j]
                jaccards.append(len(inter) / len(union) if union else 1.0)

        avg_jaccard = round(sum(jaccards) / len(jaccards), 3) if jaccards else 1.0

        # 2. 推荐排序一致性
        sorted_rankings = []
        for r in rounds:
            ranking = sorted(r["ranking"], key=lambda x: x[0])
            sorted_rankings.append(tuple(name for _, name in ranking))

        non_empty_rankings = [r for r in sorted_rankings if r]
        if non_empty_rankings:
            ranking_consistent = len(set(non_empty_rankings)) == 1
            unique_count = len(set(non_empty_rankings))
        else:
            ranking_consistent = None
            unique_count = 0

        # 3. 999产品提及稳定性
        has_999_per_round = [
            any(_is_999_product(p) for p in r["products"])
            for r in rounds
        ]
        all_mention = all(has_999_per_round)
        none_mention = not any(has_999_per_round)
        if all_mention:
            stability_999 = "稳定提及"
        elif none_mention:
            stability_999 = "稳定未提及"
        else:
            mention_count = sum(has_999_per_round)
            stability_999 = f"不稳定({mention_count}/{len(has_999_per_round)}轮提及)"

        rows.append({
            "问题ID": qid,
            "产品": product,
            "模型": model,
            "联网": "是" if search else "否",
            "轮次数": len(rounds),
            "推荐产品Jaccard均值": avg_jaccard,
            "推荐排序完全一致": ranking_consistent,
            "排序变体数": unique_count,
            "999提及稳定性": stability_999,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["产品", "模型", "联网", "问题ID"]).reset_index(drop=True)

    # 合并 answer_similarity（04脚本生成的TF-IDF相似度）
    sim_path = os.path.join(ANALYSIS_DIR, "answer_similarity.csv")
    if os.path.exists(sim_path):
        sim_df = pd.read_csv(sim_path)
        # answer_similarity 的 key 是 问题ID×模型×联网
        sim_cols = {"TF-IDF相似度": "TF-IDF相似度", "平均字数": "平均字数", "字数标准差": "字数标准差"}
        sim_df = sim_df.rename(columns=sim_cols)
        df = df.merge(
            sim_df[["问题ID", "模型", "联网", "TF-IDF相似度", "平均字数", "字数标准差"]],
            on=["问题ID", "模型", "联网"], how="left"
        )

    path = os.path.join(ANALYSIS_DIR, "stability_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  稳定性报表已更新（含TF-IDF相似度） → {path}")


def generate_updated_search_diff(detail_df: pd.DataFrame, responses: list):
    """
    基于LLM提取数据覆盖联网差异分析。
    对比联网和不联网下的999推荐变化及竞品变化。
    """
    # 构建每条应答的推荐产品集合
    answer_products = defaultdict(set)
    answer_999 = {}
    for _, row in detail_df.iterrows():
        ans_key = f"{row['问题ID']}_{row['模型']}_{row['联网'] == '是'}_{row['轮次']}"
        answer_products[ans_key].add(row["推荐产品"])
        if _is_999_product(row["推荐产品"]):
            answer_999[ans_key] = True

    # 按 问题ID×模型 分组
    groups = defaultdict(lambda: {"search": [], "nosearch": []})
    for resp in responses:
        qid = resp.get("question_id", "")
        model = resp.get("model", "")
        search = resp.get("search_enabled", False)
        round_num = resp.get("round", "")
        ans_key = f"{qid}_{model}_{search}_{round_num}"
        mode = "search" if search else "nosearch"
        groups[(qid, model)][mode].append(ans_key)

    rows = []
    for (qid, model), pair in groups.items():
        if not pair["search"] or not pair["nosearch"]:
            continue

        product = None
        for resp in responses:
            if resp.get("question_id") == qid:
                product = resp.get("product", "")
                break

        # 999提及率
        ns_999 = sum(1 for k in pair["nosearch"] if answer_999.get(k)) / len(pair["nosearch"])
        s_999 = sum(1 for k in pair["search"] if answer_999.get(k)) / len(pair["search"])

        # 推荐产品差异
        ns_products = set()
        for k in pair["nosearch"]:
            ns_products.update(answer_products.get(k, set()))
        s_products = set()
        for k in pair["search"]:
            s_products.update(answer_products.get(k, set()))

        only_search = s_products - ns_products
        only_nosearch = ns_products - s_products

        rows.append({
            "问题ID": qid,
            "产品": product,
            "模型": model,
            "联网999提及率": round(s_999, 3),
            "不联网999提及率": round(ns_999, 3),
            "提及率差异": round(s_999 - ns_999, 3),
            "仅联网推荐的产品": ", ".join(sorted(only_search)) if only_search else "",
            "仅不联网推荐的产品": ", ".join(sorted(only_nosearch)) if only_nosearch else "",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["产品", "模型", "问题ID"]).reset_index(drop=True)
    path = os.path.join(ANALYSIS_DIR, "search_diff_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  联网差异已更新（基于LLM提取数据） → {path}")


def generate_updated_optimization(detail_df: pd.DataFrame, responses: list):
    """
    基于LLM提取数据覆盖优化建议报表。
    判断逻辑与04版本类似，但数据源更准确。
    """
    # 构建每条应答的999推荐信息
    answer_999 = {}
    for _, row in detail_df.iterrows():
        ans_key = f"{row['问题ID']}_{row['模型']}_{row['联网'] == '是'}_{row['轮次']}"
        if _is_999_product(row["推荐产品"]):
            if ans_key not in answer_999:
                answer_999[ans_key] = {"strong": False, "any": False, "rank": None}
            answer_999[ans_key]["any"] = True
            if row["推荐强度"] == "strong":
                answer_999[ans_key]["strong"] = True
            rank = row["推荐排名"]
            if isinstance(rank, (int, float)):
                cur = answer_999[ans_key]["rank"]
                if cur is None or rank < cur:
                    answer_999[ans_key]["rank"] = rank

    # 按产品×模型分组
    groups = defaultdict(lambda: {"nosearch": [], "search": [], "all": []})
    for resp in responses:
        product = resp.get("product", "")
        model = resp.get("model", "")
        search = resp.get("search_enabled", False)
        ans_key = f"{resp.get('question_id', '')}_{model}_{search}_{resp.get('round', '')}"
        mode = "search" if search else "nosearch"
        groups[(product, model)][mode].append(ans_key)
        groups[(product, model)]["all"].append(ans_key)

    # 竞品统计
    comp_by_group = defaultdict(lambda: defaultdict(int))
    for _, row in detail_df.iterrows():
        if _is_999_product(row["推荐产品"]):
            continue
        comp_by_group[(row["产品"], row["模型"])][row["推荐产品"]] += 1

    rows = []
    for (product, model), modes in groups.items():
        issues = []
        priorities = []

        # 1. 999强推荐率
        all_keys = modes["all"]
        if all_keys:
            strong_count = sum(1 for k in all_keys if answer_999.get(k, {}).get("strong"))
            any_count = sum(1 for k in all_keys if answer_999.get(k, {}).get("any"))
            strong_rate = strong_count / len(all_keys)
            any_rate = any_count / len(all_keys)

            if any_rate < 0.2:
                issues.append(f"999几乎不被提及(提及率{any_rate:.0%})")
                priorities.append("高")
            elif any_rate < 0.5:
                issues.append(f"999提及率偏低({any_rate:.0%})")
                priorities.append("中")

            if strong_rate == 0 and any_rate > 0:
                issues.append("999被提及但从未被强推荐")
                priorities.append("中")

        # 2. 999排名
        ranks = [answer_999[k]["rank"] for k in all_keys if answer_999.get(k, {}).get("rank") is not None]
        if ranks and sum(ranks) / len(ranks) > 3.0:
            issues.append(f"999平均排名靠后({sum(ranks)/len(ranks):.1f})")
            priorities.append("中")

        # 3. 联网影响
        ns_keys = modes["nosearch"]
        s_keys = modes["search"]
        if ns_keys and s_keys:
            ns_any = sum(1 for k in ns_keys if answer_999.get(k, {}).get("any")) / len(ns_keys)
            s_any = sum(1 for k in s_keys if answer_999.get(k, {}).get("any")) / len(s_keys)
            diff = s_any - ns_any
            if diff < -0.2:
                issues.append(f"联网后999提及率下降({diff:+.0%})，线上内容可能不利")
                priorities.append("高")
            elif diff > 0.2:
                issues.append(f"联网后999提及率提升({diff:+.0%})，线上内容有正面作用")
                priorities.append("低(正面)")

        # 4. 稳定性
        qid_groups = defaultdict(list)
        for k in all_keys:
            qid = k.rsplit("_", 3)[0]
            qid_groups[qid].append(answer_999.get(k, {}).get("any", False))

        unstable = sum(1 for vals in qid_groups.values() if len(vals) > 1 and True in vals and False in vals)
        if len(qid_groups) > 0 and unstable / len(qid_groups) > 0.4:
            issues.append(f"回答不稳定：{unstable}/{len(qid_groups)}个问题中999提及不一致")
            priorities.append("中")

        # 5. 主要竞品
        comps = comp_by_group.get((product, model), {})
        top_comp = [f"{c}({n}次)" for c, n in sorted(comps.items(), key=lambda x: -x[1])[:3]]

        if not issues:
            issues.append("表现良好，暂无明显薄弱环节")
            priorities.append("低")

        priority = "高" if "高" in priorities else ("中" if "中" in priorities else "低")

        suggestions = _generate_suggestions(issues)

        rows.append({
            "产品": product,
            "模型": model,
            "优先级": priority,
            "问题数": len(issues),
            "发现": " | ".join(issues),
            "主要竞品": ", ".join(top_comp) if top_comp else "",
            "建议方向": suggestions,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["优先级", "产品", "模型"]).reset_index(drop=True)
    path = os.path.join(ANALYSIS_DIR, "optimization_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  优化建议已更新（基于LLM提取数据） → {path}")


def _generate_suggestions(issues: list) -> str:
    suggestions = []
    issue_text = " ".join(issues)

    if "不被提及" in issue_text or "提及率偏低" in issue_text:
        suggestions.append("加强产品与症状/场景的内容关联，在权威平台发布对症用药指南")

    if "从未被强推荐" in issue_text:
        suggestions.append("强化产品差异化优势内容，突出独特卖点和使用场景")

    if "排名靠后" in issue_text:
        suggestions.append("在专业药学平台、百科、问答社区提升品牌品类关联度")

    if "联网后" in issue_text and "下降" in issue_text:
        suggestions.append("排查线上负面或竞品SEO内容，优化品牌搜索结果质量")

    if "不稳定" in issue_text:
        suggestions.append("模型对品牌认知不够强，需多渠道持续建设品牌内容")

    if not suggestions:
        suggestions.append("持续监测，保持当前内容建设")

    return " | ".join(suggestions)


def _load_brand_config():
    """加载brands.yaml配置"""
    path = os.path.join(BASE_DIR, "config", "brands.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_product_name_map():
    """
    构建产品名映射：原始数据中的产品短名 → yaml配置中的标准名。
    直接使用brands.yaml中的product_name_map配置。
    """
    config = _load_brand_config()
    mapping = config.get("product_name_map", {})
    # 标准名也映射到自身
    for v in set(mapping.values()):
        mapping[v] = v
    return mapping


def generate_brand_generic_split(detail_df: pd.DataFrame, responses: list):
    """
    推荐产品分类（任务二）：
    对recommendation_detail中的每一条推荐产品，标注归属和名称类型。
    - 归属：999自有 / 竞品
    - 名称类型：品牌名 / 通用名 / 成分品类级
    所有关键词来自brands.yaml，修改yaml后重跑即可。
    输出 brand_generic_detail.csv（全量明细） + brand_generic_summary.csv（999产品汇总）
    """
    config = _load_brand_config()
    # 从brand_999构建999品牌关键词（名称+所有aliases）
    brand_999_keywords = []
    for name, info in config.get("brand_999", {}).items():
        brand_999_keywords.append(name)
        brand_999_keywords.extend(info.get("aliases", []))
    competitor_brands = set(config.get("known_brand_competitors", []))
    generic_names = set(config.get("generic_names", []))
    component_kw = config.get("component_keywords", [])
    product_map = _build_product_name_map()

    def _get_level(qid):
        for tag in ["q3", "q4", "q5"]:
            if f"_{tag}_" in qid:
                return tag
        return ""

    def _classify(rec_product: str, cat_key: str) -> str:
        """
        返回名称类型。优先级：
        1. 999品牌：命中brand_999关键词
        2. 竞品品牌：命中known_brand_competitors
        3. 成分品类级：命中component_keywords
        4. 通用名：命中generic_names，或以上都没命中的默认值
        """
        # === 1. 999品牌 ===
        if any(kw in rec_product for kw in brand_999_keywords):
            return "999品牌"

        # === 2. 竞品品牌 ===
        for cb in competitor_brands:
            if cb in rec_product:
                return "竞品品牌"

        # === 3. 成分品类级（在通用名之前检查，避免成分词被当通用名）===
        for comp in component_kw:
            if comp in rec_product:
                return "成分品类级"

        # === 4. 通用名 ===
        return "通用名"

    detail_rows = []
    for _, row in detail_df.iterrows():
        qid = row["问题ID"]
        level = _get_level(qid)
        if not level:
            continue

        product = row["产品"]
        cat_key = product_map.get(product)
        if not cat_key:
            continue

        rec_product = str(row["推荐产品"])
        strength = row["推荐强度"]
        reason = row["推荐理由"]
        is_recommend = 1 if strength in ("strong", "moderate") else 0
        name_type = _classify(rec_product, cat_key)

        detail_rows.append({
            "问题ID": qid,
            "产品": product,
            "问题层级": level.upper(),
            "模型": row["模型"],
            "联网": row["联网"],
            "轮次": row["轮次"],
            "推荐排名": row["推荐排名"],
            "推荐产品": rec_product,
            "名称类型": name_type,
            "推荐强度": strength,
            "是否推荐": is_recommend,
            "推荐原因": reason if is_recommend else "",
        })

    if not detail_rows:
        print("  推荐产品分类: 无数据")
        return

    detail_out = pd.DataFrame(detail_rows)
    path = os.path.join(ANALYSIS_DIR, "brand_generic_detail.csv")
    detail_out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  推荐产品分类明细 → {path} ({len(detail_out)} 条)")

    # 分布统计
    counts = detail_out["名称类型"].value_counts()
    print(f"    名称类型: {dict(counts)}")

    # === 999产品汇总统计 ===
    answer_totals = defaultdict(int)
    for resp in responses:
        qid = resp.get("question_id", "")
        level = _get_level(qid)
        if not level:
            continue
        product = resp.get("product", "")
        model = resp.get("model", "")
        search = "是" if resp.get("search_enabled") else "否"
        level_group = "Q3/Q4" if level in ("q3", "q4") else "Q5"
        answer_totals[(product, level_group, model, search)] += 1

    # 按名称类型分别统计提及和推荐
    summary_groups = defaultdict(lambda: {
        "999brand_m": 0, "999brand_r": 0,
        "generic_m": 0, "generic_r": 0,
        "compbrand_m": 0, "compbrand_r": 0,
        "component_m": 0,
    })
    for row in detail_rows:
        level_group = "Q3/Q4" if row["问题层级"] in ("Q3", "Q4") else "Q5"
        gk = (row["产品"], level_group, row["模型"], row["联网"])
        nt = row["名称类型"]
        is_rec = row["是否推荐"] == 1
        if nt == "999品牌":
            summary_groups[gk]["999brand_m"] += 1
            if is_rec: summary_groups[gk]["999brand_r"] += 1
        elif nt == "竞品品牌":
            summary_groups[gk]["compbrand_m"] += 1
            if is_rec: summary_groups[gk]["compbrand_r"] += 1
        elif nt == "通用名":
            summary_groups[gk]["generic_m"] += 1
            if is_rec: summary_groups[gk]["generic_r"] += 1
        elif nt == "成分品类级":
            summary_groups[gk]["component_m"] += 1

    # brand_generic_summary 已合并到 mention_report，不再单独输出


def generate_unified_mention_report(detail_df: pd.DataFrame, responses: list):
    """
    统一提及率报表：合并品类提及、999品牌、通用名、竞品品牌的提及数据。
    每个 产品×问题层级×模型×联网 一行，一张表看全貌。
    品类提及从原始应答文本匹配（仅Q3/Q4），其余从recommendation_detail分类结果统计。
    """
    config = _load_brand_config()
    product_map = config.get("product_name_map", {})
    cat_keywords = config.get("category_keywords", {})
    # 从brand_999构建999品牌关键词
    brand_999_keywords = []
    for name, info in config.get("brand_999", {}).items():
        brand_999_keywords.append(name)
        brand_999_keywords.extend(info.get("aliases", []))
    competitor_brands = set(config.get("known_brand_competitors", []))
    component_kw = config.get("component_keywords", [])

    def _get_level(qid):
        for tag in ["q3", "q4", "q5"]:
            if f"_{tag}_" in qid:
                return tag
        return ""

    def _classify(rec_product, cat_key):
        if any(kw in rec_product for kw in brand_999_keywords):
            return "999品牌"
        for cb in competitor_brands:
            if cb in rec_product:
                return "竞品品牌"
        for comp in component_kw:
            if comp in rec_product:
                return "成分品类级"
        return "通用名"

    # === 1. 统计品类提及（从原始应答文本，仅Q3/Q4） ===
    category_hits = defaultdict(lambda: {"total": 0, "hit": 0})
    for resp in responses:
        qid = resp.get("question_id", "")
        level = _get_level(qid)
        if level not in ("q3", "q4"):
            continue
        product = resp.get("product", "")
        cat_key = product_map.get(product)
        if not cat_key or cat_key not in cat_keywords:
            continue
        model = resp.get("model", "")
        search = "是" if resp.get("search_enabled") else "否"
        answer = resp.get("answer", "")
        gk = (product, "Q3/Q4", model, search)
        category_hits[gk]["total"] += 1
        if any(kw in answer for kw in cat_keywords[cat_key]):
            category_hits[gk]["hit"] += 1

    # === 2. 统计品牌/通用名/竞品（从recommendation_detail） ===
    answer_totals = defaultdict(int)
    for resp in responses:
        qid = resp.get("question_id", "")
        level = _get_level(qid)
        if not level:
            continue
        product = resp.get("product", "")
        model = resp.get("model", "")
        search = "是" if resp.get("search_enabled") else "否"
        level_group = "Q3/Q4" if level in ("q3", "q4") else "Q5"
        answer_totals[(product, level_group, model, search)] += 1

    type_counts = defaultdict(lambda: {
        "999品牌_m": 0, "999品牌_r": 0,
        "通用名_m": 0, "通用名_r": 0,
        "竞品品牌_m": 0, "竞品品牌_r": 0,
        "成分品类级_m": 0,
    })
    for _, row in detail_df.iterrows():
        qid = row["问题ID"]
        level = _get_level(qid)
        if not level:
            continue
        product = row["产品"]
        cat_key = product_map.get(product)
        if not cat_key:
            continue
        model = row["模型"]
        search = row["联网"]
        level_group = "Q3/Q4" if level in ("q3", "q4") else "Q5"
        gk = (product, level_group, model, search)

        rec_product = str(row["推荐产品"])
        nt = _classify(rec_product, cat_key)
        is_rec = row["推荐强度"] in ("strong", "moderate")

        type_counts[gk][f"{nt}_m"] += 1
        if is_rec and nt != "成分品类级":
            type_counts[gk][f"{nt}_r"] += 1

    # === 3. 合并输出 ===
    all_keys = set(answer_totals.keys()) | set(category_hits.keys())
    rows = []
    for gk in all_keys:
        product, level_group, model, search = gk
        total = answer_totals.get(gk, 0)
        divisor = total if total > 0 else 1
        tc = type_counts.get(gk, {})
        ch = category_hits.get(gk, {})

        row = {
            "产品": product,
            "问题层级": level_group,
            "模型": model,
            "联网": search,
            "总回答数": total,
        }

        # 品类提及（仅Q3/Q4有值）
        if level_group == "Q3/Q4" and ch:
            row["品类提及率"] = round(ch["hit"] / ch["total"], 2) if ch["total"] else ""
        else:
            row["品类提及率"] = ""

        # 各类型提及
        row["999品牌提及次数"] = tc.get("999品牌_m", 0)
        row["999品牌提及率"] = round(tc.get("999品牌_m", 0) / divisor, 3)
        row["999品牌推荐率"] = round(tc.get("999品牌_r", 0) / divisor, 3)
        row["通用名提及次数"] = tc.get("通用名_m", 0)
        row["通用名提及率"] = round(tc.get("通用名_m", 0) / divisor, 3)
        row["通用名推荐率"] = round(tc.get("通用名_r", 0) / divisor, 3)
        row["竞品品牌提及次数"] = tc.get("竞品品牌_m", 0)
        row["竞品品牌提及率"] = round(tc.get("竞品品牌_m", 0) / divisor, 3)
        row["竞品品牌推荐率"] = round(tc.get("竞品品牌_r", 0) / divisor, 3)
        row["成分品类级提及次数"] = tc.get("成分品类级_m", 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["产品", "问题层级", "模型", "联网"]).reset_index(drop=True)
    path = os.path.join(ANALYSIS_DIR, "mention_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  统一提及率报表 → {path} ({len(df)} 条)")


if __name__ == "__main__":
    asyncio.run(main())
