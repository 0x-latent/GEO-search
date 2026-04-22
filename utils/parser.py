"""
应答内容解析器：品牌提及检测、推荐排序提取、信息源提取。
"""
import re
from typing import Optional


# 999品牌产品词典（产品名 → 别名列表）
BRAND_999 = {
    "999感冒灵": ["三九感冒灵", "999感冒灵颗粒", "感冒灵颗粒", "感冒灵"],
    "999皮炎平": ["三九皮炎平", "皮炎平软膏", "皮炎平"],
    "养胃舒": ["养胃舒颗粒", "养胃舒胶囊"],
    "三九胃泰": ["三九胃泰颗粒", "999胃泰"],
    "999抗病毒口服液": ["三九抗病毒口服液", "抗病毒口服液", "999抗病毒"],
    "999小儿氨酚黄那敏": ["小儿氨酚黄那敏颗粒", "小儿氨酚黄那敏"],
    "强力枇杷露": ["999强力枇杷露", "三九强力枇杷露"],
    "澳诺葡萄糖酸锌钙": ["澳诺钙", "葡萄糖酸锌钙口服溶液", "澳诺"],
    "易善复": ["多烯磷脂酰胆碱胶囊", "易善复胶囊"],
}

# 主要竞品词典（按品类）
COMPETITOR_BRANDS = {
    # 感冒药
    "连花清瘟": ["连花清瘟胶囊", "连花清瘟颗粒"],
    "板蓝根": ["板蓝根颗粒"],
    "白加黑": ["白加黑感冒片"],
    "泰诺": ["泰诺感冒片"],
    "新康泰克": ["新康泰克胶囊"],
    "快克": ["快克感冒胶囊"],
    "感康": ["感康片", "复方氨酚烷胺片"],
    "仁和可立克": ["可立克"],
    "吴太感康": [],
    # 皮肤药
    "派瑞松": ["曲安奈德益康唑乳膏"],
    "艾洛松": ["糠酸莫米松乳膏"],
    "丹皮酚软膏": [],
    "无极膏": [],
    "冰黄肤乐软膏": [],
    # 胃药
    "达喜": ["铝碳酸镁咀嚼片"],
    "吗丁啉": ["多潘立酮片"],
    "奥美拉唑": ["奥美拉唑肠溶胶囊"],
    "斯达舒": [],
    "气滞胃痛颗粒": [],
    "摩罗丹": [],
    "胃苏颗粒": [],
    # 止咳药
    "川贝枇杷膏": ["京都念慈菴川贝枇杷膏", "念慈菴"],
    "急支糖浆": [],
    "复方甘草片": [],
    "肺力咳合剂": [],
    "蜜炼川贝枇杷膏": [],
    # 补钙/补锌
    "迪巧": ["迪巧儿童钙"],
    "钙尔奇": [],
    "龙牡壮骨颗粒": [],
    "三精葡萄糖酸锌": ["三精补锌"],
    "蓝瓶钙": ["三精蓝瓶钙"],
    "碳酸钙D3颗粒": [],
    # 护肝药
    "护肝片": ["葵花护肝片"],
    "水飞蓟宾": ["水飞蓟宾胶囊"],
    "双环醇": ["双环醇片"],
    "甘草酸二铵": [],
    "利加隆": [],
    "天晴甘美": [],
    # 儿童感冒药
    "小儿豉翘清热颗粒": [],
    "小儿感冒颗粒": [],
    "美林": ["布洛芬混悬液"],
    "泰诺林": ["对乙酰氨基酚混悬滴剂"],
    # 抗病毒
    "奥司他韦": ["磷酸奥司他韦"],
    "蒲地蓝消炎口服液": ["蒲地蓝"],
    "双黄连口服液": ["双黄连"],
    "清开灵": ["清开灵颗粒"],
}


def _build_search_patterns() -> list:
    """构建所有品牌的搜索模式列表"""
    patterns = []
    # 999产品
    for brand, aliases in BRAND_999.items():
        all_names = [brand] + aliases
        for name in all_names:
            patterns.append((name, brand, "999"))
    # 竞品
    for brand, aliases in COMPETITOR_BRANDS.items():
        all_names = [brand] + aliases
        for name in all_names:
            patterns.append((name, brand, "competitor"))

    # 按名称长度降序排列（优先匹配更长的名称，避免短名误匹配）
    patterns.sort(key=lambda x: len(x[0]), reverse=True)
    return patterns


SEARCH_PATTERNS = _build_search_patterns()


def extract_mentions(answer: str) -> dict:
    """
    从应答文本中提取品牌/产品提及。

    Returns:
        {
            "all_mentions": [{"brand": "xxx", "type": "999"|"competitor", "positions": [int]}],
            "brand_999_mentioned": ["999感冒灵", ...],
            "competitors_mentioned": ["连花清瘟", ...],
            "has_999_mention": bool,
        }
    """
    seen_brands = {}  # brand_name → {type, positions}

    for name, brand, brand_type in SEARCH_PATTERNS:
        if brand in seen_brands:
            # 已经通过更长的名称匹配到了
            # 但还要检查这个别名的位置
            pass

        start = 0
        while True:
            pos = answer.find(name, start)
            if pos == -1:
                break
            if brand not in seen_brands:
                seen_brands[brand] = {"type": brand_type, "positions": []}
            seen_brands[brand]["positions"].append(pos)
            start = pos + len(name)

    all_mentions = [
        {"brand": brand, "type": info["type"], "positions": info["positions"]}
        for brand, info in seen_brands.items()
    ]
    all_mentions.sort(key=lambda x: min(x["positions"]) if x["positions"] else 999999)

    brand_999 = [m["brand"] for m in all_mentions if m["type"] == "999"]
    competitors = [m["brand"] for m in all_mentions if m["type"] == "competitor"]

    return {
        "all_mentions": all_mentions,
        "brand_999_mentioned": brand_999,
        "competitors_mentioned": competitors,
        "has_999_mention": len(brand_999) > 0,
    }


def extract_recommendation_ranking(answer: str) -> list:
    """
    从Top3推荐类应答中提取推荐排序。

    Returns:
        有序列表，如 ["连花清瘟", "999感冒灵", "板蓝根"]
    """
    ranking = []

    # 模式1: "1. xxx 2. xxx 3. xxx" 或 "1、xxx 2、xxx"
    numbered = re.findall(r'[1-5][.、)）]\s*[*]*(.+?)(?=\n|[2-6][.、)）]|$)', answer)

    # 模式2: "第一...第二...第三..."
    if not numbered:
        ordinals = re.findall(r'第[一二三四五](?:个|款|种)?[：:是]?\s*(.+?)(?=第[一二三四五]|$)', answer, re.DOTALL)
        numbered = ordinals

    # 模式3: "首先推荐...其次...最后..."
    if not numbered:
        seq_patterns = [
            r'首先[推荐建议]?\s*(.+?)(?=其次|然后|$)',
            r'其次[推荐建议]?\s*(.+?)(?=最后|再次|$)',
            r'最后[推荐建议]?\s*(.+?)(?=$)',
        ]
        for pat in seq_patterns:
            match = re.search(pat, answer, re.DOTALL)
            if match:
                numbered.append(match.group(1))

    # 从每个匹配项中提取品牌名
    for item in numbered:
        item = item.strip()[:100]  # 截取前100字符
        for name, brand, _ in SEARCH_PATTERNS:
            if name in item:
                if brand not in ranking:
                    ranking.append(brand)
                break

    return ranking


def extract_sources(answer: str, raw_sources: list = None) -> list:
    """
    提取信息来源（联网模式下）。
    优先使用API返回的结构化sources，补充从文本中提取的URL。

    Returns:
        [{"title": str, "url": str, "domain": str}]
    """
    sources = []

    # 使用API返回的结构化sources
    if raw_sources:
        for s in raw_sources:
            url = s.get("url", "")
            domain = _extract_domain(url) if url else ""
            sources.append({
                "title": s.get("title", ""),
                "url": url,
                "domain": domain,
            })

    # 从文本中补充提取URL
    urls_in_text = re.findall(r'https?://[^\s\)）\]】"\'<>]+', answer)
    existing_urls = {s["url"] for s in sources}
    for url in urls_in_text:
        if url not in existing_urls:
            sources.append({
                "title": "",
                "url": url,
                "domain": _extract_domain(url),
            })

    return sources


def _extract_domain(url: str) -> str:
    """从URL提取域名"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def parse_single_response(response: dict) -> dict:
    """
    解析单条应答，返回完整的结构化分析结果。
    """
    answer = response.get("answer", "")
    raw_sources = response.get("sources", [])

    mentions = extract_mentions(answer)
    ranking = extract_recommendation_ranking(answer)
    sources = extract_sources(answer, raw_sources)

    # 判断999产品在推荐中的排名
    product = response.get("product", "")
    rank_999 = None
    for i, brand in enumerate(ranking):
        for b999 in BRAND_999:
            if brand == b999 or product in brand or brand in product:
                rank_999 = i + 1
                break
        if rank_999:
            break

    return {
        "question_id": response.get("question_id"),
        "model": response.get("model"),
        "search_enabled": response.get("search_enabled"),
        "round": response.get("round"),
        "product": product,
        "mentions": mentions,
        "recommendation_ranking": ranking,
        "rank_999": rank_999,
        "sources": sources,
        "answer_length": len(answer),
    }
