"""
统计分析：读取全部原始应答，解析后生成各维度报表 + 多轮回答相似度分析。
"""
import json
import os
import sys
import glob
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parser import parse_single_response, discover_unknown_brands
from utils.reporter import generate_all_reports
from utils.similarity import clean_text, calc_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
ANALYSIS_DIR = os.path.join(BASE_DIR, "results", "analysis")


def load_questions_map() -> dict:
    qmap = {}
    for fname in ["questions_expanded.json", "questions_base.json"]:
        path = os.path.join(BASE_DIR, "questions", fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                questions = json.load(f)
            for q in questions:
                qmap[q["id"]] = q
            break
    return qmap


def load_all_responses() -> list:
    responses = []
    pattern = os.path.join(RAW_DIR, "**", "*.json")
    files = glob.glob(pattern, recursive=True)

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "question_id" in data and "answer" in data:
                responses.append(data)
        except Exception as e:
            print(f"跳过无法解析的文件: {fpath} ({e})")

    return responses


def generate_raw_data_csv(responses: list, qmap: dict):
    """生成原始数据总表，每条应答一行，包含完整问答信息"""
    rows = []
    for resp in responses:
        qid = resp.get("question_id", "")
        q = qmap.get(qid, {})

        # 提取联网引用源
        sources = resp.get("sources", [])
        source_urls = "; ".join(s.get("url", "") for s in sources if s.get("url")) if sources else ""
        source_titles = "; ".join(s.get("title", "") for s in sources if s.get("title")) if sources else ""

        rows.append({
            "问题ID": qid,
            "产品": resp.get("product", ""),
            "产品代码": q.get("product_code", ""),
            "问题层级": q.get("level", ""),
            "问题类别": q.get("category", ""),
            "是否变体": q.get("is_variant", False),
            "变体来源": q.get("variant_of", ""),
            "变体类型": q.get("variant_type", ""),
            "问题": resp.get("question_text", "") or q.get("question", ""),
            "模型": resp.get("model", ""),
            "模型名称": resp.get("model_name", ""),
            "联网": "是" if resp.get("search_enabled") else "否",
            "轮次": resp.get("round", ""),
            "应答": resp.get("answer", ""),
            "应答字数": len(resp.get("answer", "")),
            "响应耗时ms": resp.get("latency_ms", ""),
            "引用源URL": source_urls,
            "引用源标题": source_titles,
            "引用源数量": len(sources) if sources else 0,
            "时间戳": resp.get("timestamp", ""),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["产品", "问题层级", "问题ID", "模型", "联网", "轮次"]).reset_index(drop=True)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    path = os.path.join(ANALYSIS_DIR, "raw_data.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"原始数据总表 → {path} ({len(df)} 条)")


def run_similarity_analysis(responses: list):
    """对所有问题的多轮回答计算TF-IDF相似度"""
    groups = defaultdict(list)
    for resp in responses:
        key = (resp.get("question_id", ""), resp.get("model", ""), resp.get("search_enabled", False))
        groups[key].append(resp)

    print(f"共 {len(groups)} 个分组（问题×模型×联网）")

    rows = []
    for (qid, model, search_enabled), group_resps in groups.items():
        product = group_resps[0].get("product", "")
        question = group_resps[0].get("question_text", "")
        answers = [r.get("answer", "") for r in group_resps]

        cleaned = [clean_text(a) for a in answers]
        similarity = calc_similarity(cleaned)

        avg_len = sum(len(a) for a in cleaned) // len(cleaned) if cleaned else 0
        len_std = (sum((len(a) - avg_len) ** 2 for a in cleaned) / len(cleaned)) ** 0.5 if len(cleaned) > 1 else 0

        strategy = "取2条代表" if similarity > 0.85 else "全量发送"

        rows.append({
            "问题ID": qid,
            "产品": product,
            "模型": model,
            "联网": "是" if search_enabled else "否",
            "轮次数": len(answers),
            "平均字数": avg_len,
            "字数标准差": round(len_std),
            "TF-IDF相似度": round(similarity, 3),
            "LLM校对建议策略": strategy,
            "问题": question[:50],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("TF-IDF相似度").reset_index(drop=True)

    path = os.path.join(ANALYSIS_DIR, "answer_similarity.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")

    # 打印统计摘要
    total = len(df)
    high_sim = sum(df["TF-IDF相似度"] > 0.85)
    low_sim = total - high_sim
    avg_sim = df["TF-IDF相似度"].mean()

    print(f"相似度分析 → {path}")
    print(f"  总分组数: {total}")
    print(f"  平均相似度: {avg_sim:.3f}")
    print(f"  高相似度(>0.85): {high_sim} 组")
    print(f"  低相似度(≤0.85): {low_sim} 组")


def main():
    print("加载问题映射...")
    qmap = load_questions_map()

    print("加载原始应答...")
    responses = load_all_responses()
    print(f"共加载 {len(responses)} 条应答")

    if not responses:
        print("没有找到应答数据。请先运行 03_query_models.py")
        return

    print("解析应答内容...")
    parsed_results = []
    for resp in responses:
        parsed = parse_single_response(resp)

        qid = parsed["question_id"]
        if qid in qmap:
            q = qmap[qid]
            parsed["level"] = q.get("level", "")
            parsed["variant_of"] = q.get("variant_of")
            parsed["is_variant"] = q.get("is_variant", False)
            parsed["question_text"] = q.get("question", "")
            parsed["answer"] = resp.get("answer", "")
        else:
            parsed["level"] = ""
            parsed["variant_of"] = None
            parsed["is_variant"] = False
            parsed["question_text"] = resp.get("question_text", "")
            parsed["answer"] = resp.get("answer", "")

        parsed_results.append(parsed)

    print(f"解析完成，共 {len(parsed_results)} 条结果")

    # 未知品牌发现
    print("\n发现未收录品牌...")
    all_answers = [resp.get("answer", "") for resp in responses]
    unknown = discover_unknown_brands(all_answers)
    if unknown:
        unk_df = pd.DataFrame(unknown, columns=["候选品牌名", "出现次数"])
        unk_path = os.path.join(ANALYSIS_DIR, "unknown_brands.csv")
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        unk_df.to_csv(unk_path, index=False, encoding="utf-8-sig")
        print(f"发现 {len(unknown)} 个未收录候选品牌 → {unk_path}")
        print("  请审核后将有效品牌补充到 config/brands.yaml，再重新分析")
    else:
        print("  未发现新品牌")

    # 原始数据总表
    print("\n生成原始数据总表...")
    generate_raw_data_csv(responses, qmap)

    # 多轮回答相似度分析
    print("\n多轮回答相似度分析...")
    run_similarity_analysis(responses)

    # 生成报表
    print("\n生成报表...")
    generate_all_reports(parsed_results, ANALYSIS_DIR)
    print(f"\n全部报表已生成至: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
