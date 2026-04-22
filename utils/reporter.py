"""
CSV报表生成工具。
"""
import os
import pandas as pd
from collections import Counter, defaultdict


def generate_all_reports(parsed_results: list, output_dir: str):
    """生成全部分析报表"""
    os.makedirs(output_dir, exist_ok=True)

    generate_mention_report(parsed_results, output_dir)
    generate_recommendation_report(parsed_results, output_dir)
    generate_stability_report(parsed_results, output_dir)
    generate_search_diff_report(parsed_results, output_dir)
    generate_source_report(parsed_results, output_dir)
    generate_competitor_report(parsed_results, output_dir)
    generate_accuracy_summary(parsed_results, output_dir)
    generate_variant_sensitivity_report(parsed_results, output_dir)


def generate_mention_report(results: list, output_dir: str):
    """提及率分析"""
    rows = []
    # 按产品 × 模型 × 联网与否 × 问题层级分组
    groups = defaultdict(list)
    for r in results:
        key = (r["product"], r["model"], r["search_enabled"], r.get("level", ""))
        groups[key].append(r)

    for (product, model, search, level), group in groups.items():
        total = len(group)
        mentioned = sum(1 for r in group if r["mentions"]["has_999_mention"])
        rows.append({
            "产品": product,
            "模型": model,
            "联网": "是" if search else "否",
            "问题层级": level,
            "总回答数": total,
            "提及999次数": mentioned,
            "提及率": round(mentioned / total, 3) if total > 0 else 0,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "mention_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"提及率报表 → {path}")


def generate_recommendation_report(results: list, output_dir: str):
    """推荐率分析（针对q5_top3类问题）"""
    rows = []
    groups = defaultdict(list)
    for r in results:
        if not r.get("recommendation_ranking"):
            continue
        key = (r["product"], r["model"], r["search_enabled"])
        groups[key].append(r)

    for (product, model, search), group in groups.items():
        total = len(group)
        in_top3 = sum(1 for r in group if r["rank_999"] is not None)
        ranks = [r["rank_999"] for r in group if r["rank_999"] is not None]
        avg_rank = round(sum(ranks) / len(ranks), 2) if ranks else None

        # 排名分布
        rank_dist = Counter(ranks)

        rows.append({
            "产品": product,
            "模型": model,
            "联网": "是" if search else "否",
            "总回答数": total,
            "进入Top3次数": in_top3,
            "推荐率": round(in_top3 / total, 3) if total > 0 else 0,
            "平均排名": avg_rank,
            "排名1次数": rank_dist.get(1, 0),
            "排名2次数": rank_dist.get(2, 0),
            "排名3次数": rank_dist.get(3, 0),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "recommendation_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"推荐率报表 → {path}")


def generate_stability_report(results: list, output_dir: str):
    """稳定性分析：多轮回答的品牌提及一致性"""
    rows = []
    groups = defaultdict(list)
    for r in results:
        key = (r["question_id"], r["model"], r["search_enabled"])
        groups[key].append(r)

    for (qid, model, search), group in groups.items():
        if len(group) < 2:
            continue

        # 品牌提及一致性
        mention_sets = [
            frozenset(r["mentions"]["brand_999_mentioned"] + r["mentions"]["competitors_mentioned"])
            for r in group
        ]
        # 两两比较Jaccard相似度
        similarities = []
        for i in range(len(mention_sets)):
            for j in range(i + 1, len(mention_sets)):
                union = mention_sets[i] | mention_sets[j]
                inter = mention_sets[i] & mention_sets[j]
                sim = len(inter) / len(union) if union else 1.0
                similarities.append(sim)

        avg_sim = round(sum(similarities) / len(similarities), 3) if similarities else 1.0

        # 推荐排序一致性（如果有）
        rankings = [tuple(r["recommendation_ranking"]) for r in group if r["recommendation_ranking"]]
        ranking_consistency = len(set(rankings)) == 1 if rankings else None

        rows.append({
            "问题ID": qid,
            "产品": group[0]["product"],
            "模型": model,
            "联网": "是" if search else "否",
            "轮次数": len(group),
            "品牌提及Jaccard均值": avg_sim,
            "推荐排序完全一致": ranking_consistency,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "stability_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"稳定性报表 → {path}")


def generate_search_diff_report(results: list, output_dir: str):
    """联网vs不联网差异分析"""
    rows = []
    # 按问题×模型分组，对比联网和不联网
    groups = defaultdict(lambda: {"search": [], "nosearch": []})
    for r in results:
        key = (r["question_id"], r["model"])
        if r["search_enabled"]:
            groups[key]["search"].append(r)
        else:
            groups[key]["nosearch"].append(r)

    for (qid, model), pair in groups.items():
        if not pair["search"] or not pair["nosearch"]:
            continue

        # 提及差异
        search_mentions = set()
        for r in pair["search"]:
            search_mentions.update(r["mentions"]["brand_999_mentioned"])
            search_mentions.update(r["mentions"]["competitors_mentioned"])

        nosearch_mentions = set()
        for r in pair["nosearch"]:
            nosearch_mentions.update(r["mentions"]["brand_999_mentioned"])
            nosearch_mentions.update(r["mentions"]["competitors_mentioned"])

        only_search = search_mentions - nosearch_mentions
        only_nosearch = nosearch_mentions - search_mentions

        # 999提及率差异
        search_999_rate = sum(1 for r in pair["search"] if r["mentions"]["has_999_mention"]) / len(pair["search"])
        nosearch_999_rate = sum(1 for r in pair["nosearch"] if r["mentions"]["has_999_mention"]) / len(pair["nosearch"])

        rows.append({
            "问题ID": qid,
            "产品": pair["search"][0]["product"],
            "模型": model,
            "联网999提及率": round(search_999_rate, 3),
            "不联网999提及率": round(nosearch_999_rate, 3),
            "提及率差异": round(search_999_rate - nosearch_999_rate, 3),
            "仅联网出现的品牌": ", ".join(only_search) if only_search else "",
            "仅不联网出现的品牌": ", ".join(only_nosearch) if only_nosearch else "",
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "search_diff_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"联网差异报表 → {path}")


def generate_source_report(results: list, output_dir: str):
    """信息源分析"""
    domain_counter = Counter()
    source_details = []

    for r in results:
        if not r["search_enabled"] or not r.get("sources"):
            continue
        for s in r["sources"]:
            domain = s.get("domain", "")
            if domain:
                domain_counter[domain] += 1
                source_details.append({
                    "问题ID": r["question_id"],
                    "产品": r["product"],
                    "模型": r["model"],
                    "域名": domain,
                    "标题": s.get("title", ""),
                    "URL": s.get("url", ""),
                })

    # 域名频次汇总
    rows = [{"域名": domain, "引用次数": count} for domain, count in domain_counter.most_common()]
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "source_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")

    # 详细引用列表
    if source_details:
        df_detail = pd.DataFrame(source_details)
        detail_path = os.path.join(output_dir, "source_detail_report.csv")
        df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"信息源报表 → {path}")


def generate_competitor_report(results: list, output_dir: str):
    """竞品图谱"""
    rows = []
    # 按产品分组统计竞品出现频率
    groups = defaultdict(list)
    for r in results:
        groups[r["product"]].append(r)

    for product, group in groups.items():
        competitor_counter = Counter()
        for r in group:
            for comp in r["mentions"]["competitors_mentioned"]:
                competitor_counter[comp] += 1

        total = len(group)
        for comp, count in competitor_counter.most_common():
            rows.append({
                "产品": product,
                "竞品": comp,
                "出现次数": count,
                "出现率": round(count / total, 3) if total > 0 else 0,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "competitor_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"竞品图谱 → {path}")


def generate_accuracy_summary(results: list, output_dir: str):
    """准确率汇总（问题1-2的应答汇总表，供人工评分）"""
    rows = []
    for r in results:
        level = r.get("level", "")
        if level not in ("q1_overall", "q2_detail"):
            continue
        rows.append({
            "问题ID": r["question_id"],
            "产品": r["product"],
            "模型": r["model"],
            "联网": "是" if r["search_enabled"] else "否",
            "轮次": r["round"],
            "问题": r.get("question_text", ""),
            "应答": r.get("answer", ""),
            "人工评分": "",  # 留空供人工填写
            "备注": "",
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "accuracy_summary.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"准确率汇总 → {path}")


def generate_variant_sensitivity_report(results: list, output_dir: str):
    """问法敏感度分析"""
    rows = []
    # 按 (variant_of, model, search) 分组，对比原始问题和变体
    groups = defaultdict(lambda: {"original": [], "variants": []})
    for r in results:
        variant_of = r.get("variant_of")
        key = (variant_of or r["question_id"], r["model"], r["search_enabled"])
        if variant_of:
            groups[key]["variants"].append(r)
        else:
            groups[key]["original"].append(r)

    for (base_qid, model, search), pair in groups.items():
        if not pair["original"] or not pair["variants"]:
            continue

        # 原始问题的999提及率
        orig_rate = sum(1 for r in pair["original"] if r["mentions"]["has_999_mention"]) / len(pair["original"])

        # 各变体的999提及率
        variant_rates = []
        for r in pair["variants"]:
            variant_rates.append(1 if r["mentions"]["has_999_mention"] else 0)
        var_rate = sum(variant_rates) / len(variant_rates) if variant_rates else 0

        rows.append({
            "基础问题ID": base_qid,
            "产品": pair["original"][0]["product"],
            "模型": model,
            "联网": "是" if search else "否",
            "原始999提及率": round(orig_rate, 3),
            "变体999提及率": round(var_rate, 3),
            "差异": round(var_rate - orig_rate, 3),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "variant_sensitivity_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"问法敏感度报表 → {path}")
