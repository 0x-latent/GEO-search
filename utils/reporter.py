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
    generate_cross_model_report(parsed_results, output_dir)
    generate_dashboard(parsed_results, output_dir)
    generate_optimization_report(parsed_results, output_dir)


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


def generate_cross_model_report(results: list, output_dir: str):
    """
    跨模型对比分析：同一问题在不同模型下的回答差异。
    输出两张表：
    1. 品牌提及一致性 — 各模型对同一问题提及的品牌是否一致
    2. 推荐排序对比 — 各模型的Top3推荐差异
    """
    # --- 表1: 品牌提及一致性 ---
    # 按 (question_id, search_enabled) 分组，对比不同模型
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["question_id"], r["search_enabled"])
        groups[key][r["model"]].append(r)

    consistency_rows = []
    for (qid, search), model_results in groups.items():
        if len(model_results) < 2:
            continue

        # 每个模型的999提及率
        model_rates = {}
        model_competitors = {}
        for model, rs in model_results.items():
            rate_999 = sum(1 for r in rs if r["mentions"]["has_999_mention"]) / len(rs) if rs else 0
            model_rates[model] = round(rate_999, 3)

            # 汇总该模型提到的竞品
            comps = Counter()
            for r in rs:
                for c in r["mentions"]["competitors_mentioned"]:
                    comps[c] += 1
            model_competitors[model] = comps

        # 所有模型提及的品牌并集
        all_brands = set()
        for model, rs in model_results.items():
            for r in rs:
                all_brands.update(r["mentions"]["brand_999_mentioned"])
                all_brands.update(r["mentions"]["competitors_mentioned"])

        # 各模型都提及的品牌（交集）
        brand_sets = []
        for model, rs in model_results.items():
            s = set()
            for r in rs:
                s.update(r["mentions"]["brand_999_mentioned"])
                s.update(r["mentions"]["competitors_mentioned"])
            brand_sets.append(s)

        common_brands = brand_sets[0]
        for s in brand_sets[1:]:
            common_brands = common_brands & s

        row = {
            "问题ID": qid,
            "产品": list(model_results.values())[0][0]["product"],
            "联网": "是" if search else "否",
            "模型数": len(model_results),
            "品牌并集数": len(all_brands),
            "品牌交集数": len(common_brands),
            "品牌一致度": round(len(common_brands) / len(all_brands), 3) if all_brands else 1.0,
            "各模型一致提及": ", ".join(sorted(common_brands)) if common_brands else "",
        }
        # 每个模型的999提及率作为单独列
        for model, rate in model_rates.items():
            row[f"{model}_999提及率"] = rate

        consistency_rows.append(row)

    if consistency_rows:
        df = pd.DataFrame(consistency_rows)
        path = os.path.join(output_dir, "cross_model_consistency.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"跨模型一致性 → {path}")

    # 推荐排序对比已移至08脚本（LLM提取），此处不再生成


def generate_dashboard(results: list, output_dir: str):
    """
    总览Dashboard：一张表看全局。
    维度：产品 × 模型，指标汇总。
    """
    groups = defaultdict(list)
    for r in results:
        key = (r["product"], r["model"])
        groups[key].append(r)

    rows = []
    for (product, model), group in groups.items():
        total = len(group)
        nosearch = [r for r in group if not r["search_enabled"]]
        search = [r for r in group if r["search_enabled"]]

        # 999提及率
        mention_all = sum(1 for r in group if r["mentions"]["has_999_mention"]) / total if total else 0
        mention_nosearch = sum(1 for r in nosearch if r["mentions"]["has_999_mention"]) / len(nosearch) if nosearch else None
        mention_search = sum(1 for r in search if r["mentions"]["has_999_mention"]) / len(search) if search else None

        # 推荐率（仅Top3类问题）
        rec_group = [r for r in group if r.get("recommendation_ranking")]
        rec_rate = None
        avg_rank = None
        if rec_group:
            in_top3 = sum(1 for r in rec_group if r["rank_999"] is not None)
            rec_rate = round(in_top3 / len(rec_group), 3)
            ranks = [r["rank_999"] for r in rec_group if r["rank_999"] is not None]
            avg_rank = round(sum(ranks) / len(ranks), 2) if ranks else None

        # 稳定性（品牌提及Jaccard均值）
        q_groups = defaultdict(list)
        for r in group:
            q_groups[(r["question_id"], r["search_enabled"])].append(r)

        jaccard_values = []
        for _, q_rs in q_groups.items():
            if len(q_rs) < 2:
                continue
            mention_sets = [
                frozenset(r["mentions"]["brand_999_mentioned"] + r["mentions"]["competitors_mentioned"])
                for r in q_rs
            ]
            for i in range(len(mention_sets)):
                for j in range(i + 1, len(mention_sets)):
                    union = mention_sets[i] | mention_sets[j]
                    inter = mention_sets[i] & mention_sets[j]
                    if union:
                        jaccard_values.append(len(inter) / len(union))

        avg_jaccard = round(sum(jaccard_values) / len(jaccard_values), 3) if jaccard_values else None

        # Top竞品
        comp_counter = Counter()
        for r in group:
            for c in r["mentions"]["competitors_mentioned"]:
                comp_counter[c] += 1
        top_competitors = [c for c, _ in comp_counter.most_common(3)]

        rows.append({
            "产品": product,
            "模型": model,
            "总回答数": total,
            "999提及率(整体)": round(mention_all, 3),
            "999提及率(不联网)": round(mention_nosearch, 3) if mention_nosearch is not None else "",
            "999提及率(联网)": round(mention_search, 3) if mention_search is not None else "",
            "Top3推荐率": rec_rate if rec_rate is not None else "",
            "999平均排名": avg_rank if avg_rank is not None else "",
            "回答稳定性(Jaccard)": avg_jaccard if avg_jaccard is not None else "",
            "Top3竞品": " / ".join(top_competitors),
        })

    df = pd.DataFrame(rows)
    # 排序：按产品、模型
    df = df.sort_values(["产品", "模型"]).reset_index(drop=True)
    path = os.path.join(output_dir, "dashboard.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"总览Dashboard → {path}")


def generate_optimization_report(results: list, output_dir: str):
    """
    999优化建议报表：自动识别各产品在各模型上的薄弱环节，给出优化方向。
    判断逻辑：
    - 准确率层（q1/q2）：999被提及但信息可能不准 → 需要信息纠偏
    - 场景层（q3/q4）：999未被提及 → 需要加强场景关联
    - 推荐层（q5）：999未进Top3或排名靠后 → 需要提升品牌权重
    - 联网 vs 不联网差异大 → 线上内容需优化
    - 稳定性低 → 模型认知不稳定，优化空间大
    """
    # 按产品×模型汇总
    groups = defaultdict(list)
    for r in results:
        key = (r["product"], r["model"])
        groups[key].append(r)

    rows = []
    for (product, model), group in groups.items():
        issues = []
        priorities = []

        # 1. 场景题999提及率
        scenario_rs = [r for r in group if r.get("level") in ("q3_scenario1", "q4_scenario2") and not r["search_enabled"]]
        if scenario_rs:
            rate = sum(1 for r in scenario_rs if r["mentions"]["has_999_mention"]) / len(scenario_rs)
            if rate < 0.3:
                issues.append(f"场景题999提及率极低({rate:.0%})，模型在症状/场景描述中不会主动推荐")
                priorities.append("高")
            elif rate < 0.6:
                issues.append(f"场景题999提及率偏低({rate:.0%})，部分场景下不被推荐")
                priorities.append("中")

        # 2. Top3推荐情况
        top3_rs = [r for r in group if r.get("level") == "q5_top3"]
        if top3_rs:
            in_top3 = sum(1 for r in top3_rs if r["rank_999"] is not None)
            rec_rate = in_top3 / len(top3_rs)
            if rec_rate == 0:
                issues.append("品类推荐中完全未进入Top3")
                priorities.append("高")
            elif rec_rate < 0.5:
                issues.append(f"品类推荐Top3进入率低({rec_rate:.0%})")
                priorities.append("中")

            # 排名靠后
            ranks = [r["rank_999"] for r in top3_rs if r["rank_999"] is not None]
            if ranks and sum(ranks) / len(ranks) > 2.0:
                issues.append(f"即使进入Top3，平均排名靠后({sum(ranks)/len(ranks):.1f})")
                priorities.append("中")

        # 3. 联网vs不联网差异
        search_rs = [r for r in group if r["search_enabled"]]
        nosearch_rs = [r for r in group if not r["search_enabled"]]
        if search_rs and nosearch_rs:
            search_rate = sum(1 for r in search_rs if r["mentions"]["has_999_mention"]) / len(search_rs)
            nosearch_rate = sum(1 for r in nosearch_rs if r["mentions"]["has_999_mention"]) / len(nosearch_rs)
            diff = search_rate - nosearch_rate
            if diff < -0.2:
                issues.append(f"联网后999提及率反而下降({diff:+.0%})，线上内容可能对品牌不利")
                priorities.append("高")
            elif diff > 0.2:
                issues.append(f"联网后999提及率明显提升({diff:+.0%})，线上内容对品牌有正面作用")
                priorities.append("低(正面)")

        # 4. 回答稳定性
        q_groups = defaultdict(list)
        for r in group:
            q_groups[r["question_id"]].append(r)

        unstable_count = 0
        for _, q_rs in q_groups.items():
            if len(q_rs) < 2:
                continue
            mention_results = [r["mentions"]["has_999_mention"] for r in q_rs]
            # 如果同一问题多轮回答中，999时有时无，说明不稳定
            if True in mention_results and False in mention_results:
                unstable_count += 1

        total_questions = len(q_groups)
        if total_questions > 0:
            unstable_rate = unstable_count / total_questions
            if unstable_rate > 0.4:
                issues.append(f"回答不稳定：{unstable_rate:.0%}的问题中999提及情况不一致，优化空间大")
                priorities.append("中")

        # 5. 主要竞品威胁
        comp_counter = Counter()
        for r in group:
            for c in r["mentions"]["competitors_mentioned"]:
                comp_counter[c] += 1
        top_comp = [f"{c}({n}次)" for c, n in comp_counter.most_common(3)]

        # 汇总
        if not issues:
            issues.append("表现良好，暂无明显薄弱环节")
            priorities.append("低")

        priority = "高" if "高" in priorities else ("中" if "中" in priorities else "低")

        rows.append({
            "产品": product,
            "模型": model,
            "优先级": priority,
            "问题数": len(issues),
            "发现": " | ".join(issues),
            "主要竞品": ", ".join(top_comp) if top_comp else "",
            "建议方向": _generate_suggestions(issues),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["优先级", "产品", "模型"]).reset_index(drop=True)
    path = os.path.join(output_dir, "optimization_report.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"优化建议报表 → {path}")


def _generate_suggestions(issues: list) -> str:
    """根据发现的问题生成优化建议"""
    suggestions = []
    issue_text = " ".join(issues)

    if "场景题" in issue_text and "提及率" in issue_text:
        suggestions.append("加强产品与症状/场景的内容关联，在权威平台发布对症用药指南")

    if "Top3" in issue_text:
        suggestions.append("在专业药学平台、百科、问答社区提升品牌品类关联度")

    if "联网后" in issue_text and "下降" in issue_text:
        suggestions.append("排查线上负面或竞品SEO内容，优化品牌搜索结果质量")

    if "不稳定" in issue_text:
        suggestions.append("模型对品牌认知不够强，需多渠道持续建设品牌内容")

    if "排名靠后" in issue_text:
        suggestions.append("强化产品差异化优势内容，突出独特卖点")

    if not suggestions:
        suggestions.append("持续监测，保持当前内容建设")

    return " | ".join(suggestions)
