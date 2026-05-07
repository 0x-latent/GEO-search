"""
应答内容解析器：品牌提及检测、推荐排序提取、信息源提取。
品牌词典从 config/brands.yaml 加载，支持动态发现未收录品牌。
"""
import os
import re
import yaml
from collections import Counter
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRANDS_PATH = os.path.join(BASE_DIR, "config", "brands.yaml")


def load_brand_config() -> dict:
    """加载品牌词典配置"""
    with open(BRANDS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_search_patterns(brand_config: dict) -> list:
    """从配置构建搜索模式列表"""
    patterns = []

    # 999产品
    for brand, info in brand_config.get("brand_999", {}).items():
        aliases = info.get("aliases", [])
        # 主名 + 别名
        for name in [brand] + aliases:
            patterns.append((name, brand, "999", info.get("category", "")))
        # 短名（如果有）
        short = info.get("short_name")
        if short and short not in [brand] + aliases:
            patterns.append((short, brand, "999", info.get("category", "")))

    # 品牌竞品
    for name in brand_config.get("known_brand_competitors", []):
        patterns.append((name, name, "competitor", ""))

    # 通用名竞品
    for name in brand_config.get("generic_names", []):
        patterns.append((name, name, "competitor", ""))

    # 按名称长度降序（优先匹配更长的名称，避免短名误匹配）
    patterns.sort(key=lambda x: len(x[0]), reverse=True)
    return patterns


# 模块加载时初始化
_brand_config = load_brand_config()
SEARCH_PATTERNS = _build_search_patterns(_brand_config)

# 构建所有已知名称集合（用于未知品牌发现时排除）
_ALL_KNOWN_NAMES = set()
for name, brand, btype, cat in SEARCH_PATTERNS:
    _ALL_KNOWN_NAMES.add(name)
    _ALL_KNOWN_NAMES.add(brand)


def reload_brands():
    """重新加载品牌词典（更新brands.yaml后调用）"""
    global _brand_config, SEARCH_PATTERNS, _ALL_KNOWN_NAMES
    _brand_config = load_brand_config()
    SEARCH_PATTERNS = _build_search_patterns(_brand_config)
    _ALL_KNOWN_NAMES = set()
    for name, brand, btype, cat in SEARCH_PATTERNS:
        _ALL_KNOWN_NAMES.add(name)
        _ALL_KNOWN_NAMES.add(brand)


def extract_mentions(answer: str) -> dict:
    """
    从应答文本中提取品牌/产品提及。

    Returns:
        {
            "all_mentions": [{"brand": str, "type": "999"|"competitor", "category": str, "positions": [int], "count": int}],
            "brand_999_mentioned": ["999感冒灵", ...],
            "competitors_mentioned": ["连花清瘟", ...],
            "has_999_mention": bool,
        }
    """
    seen_brands = {}  # brand_name → {type, category, positions}

    for name, brand, brand_type, category in SEARCH_PATTERNS:
        start = 0
        while True:
            pos = answer.find(name, start)
            if pos == -1:
                break
            if brand not in seen_brands:
                seen_brands[brand] = {"type": brand_type, "category": category, "positions": []}
            seen_brands[brand]["positions"].append(pos)
            start = pos + len(name)

    all_mentions = [
        {
            "brand": brand,
            "type": info["type"],
            "category": info["category"],
            "positions": info["positions"],
            "count": len(info["positions"]),
        }
        for brand, info in seen_brands.items()
    ]
    # 按首次出现位置排序
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
        for name, brand, _, _ in SEARCH_PATTERNS:
            if name in item:
                if brand not in ranking:
                    ranking.append(brand)
                break

    return ranking


def extract_sources(answer: str, raw_sources: list = None) -> list:
    """
    提取信息来源（联网模式下）。
    优先使用API返回的结构化sources，补充从文本中提取的URL。
    """
    sources = []

    if raw_sources:
        for s in raw_sources:
            url = s.get("url", "")
            domain = _extract_domain(url) if url else ""
            sources.append({
                "title": s.get("title", ""),
                "url": url,
                "domain": domain,
            })

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
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def parse_single_response(response: dict) -> dict:
    """解析单条应答，返回完整的结构化分析结果。"""
    answer = response.get("answer", "")
    raw_sources = response.get("sources", [])

    mentions = extract_mentions(answer)
    ranking = extract_recommendation_ranking(answer)
    sources = extract_sources(answer, raw_sources)

    # 判断999产品在推荐中的排名
    product = response.get("product", "")
    rank_999 = None
    for i, brand in enumerate(ranking):
        # 检查brand是否属于999品牌
        for b999 in _brand_config.get("brand_999", {}):
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


# ===== 未知品牌发现 =====

# 药品名称常见后缀（用于从回答中发现潜在的药品名）
_DRUG_SUFFIXES = [
    "颗粒", "胶囊", "片", "口服液", "糖浆", "软膏", "乳膏", "滴剂",
    "混悬液", "散", "丸", "露", "合剂", "冲剂", "膏", "贴", "栓",
]

# 药品名称常见模式
_DRUG_PATTERNS = [
    # "XX牌XX" 或 "XX（XX）"
    re.compile(r'[\u4e00-\u9fa5]{2,8}(?:牌|®)[\u4e00-\u9fa5]{2,8}'),
    # 中文名+后缀
    re.compile(r'[\u4e00-\u9fa5]{2,10}(?:' + '|'.join(_DRUG_SUFFIXES) + ')'),
    # **加粗的药品名**
    re.compile(r'\*\*(.+?)\*\*'),
]


def discover_unknown_brands(answers: list) -> list:
    """
    从一批回答中发现词典未收录的潜在药品/品牌名。

    Args:
        answers: 回答文本列表

    Returns:
        [(名称, 出现次数)] 按频次降序排列，已过滤掉已知品牌
    """
    candidate_counter = Counter()

    for answer in answers:
        found = set()

        # 用正则模式提取候选药品名
        for pattern in _DRUG_PATTERNS:
            for match in pattern.finditer(answer):
                name = match.group(0).strip("*").strip()
                if 2 <= len(name) <= 15:
                    found.add(name)

        # 过滤掉已知品牌
        for name in found:
            is_known = False
            for known in _ALL_KNOWN_NAMES:
                if name == known or known in name or name in known:
                    is_known = True
                    break
            if not is_known:
                candidate_counter[name] += 1

    # 过滤掉只出现1次的（可能是噪音）和过于通用的词
    generic_words = {"感冒药", "止咳药", "消炎药", "退烧药", "胃药", "皮肤药",
                     "中成药", "西药", "处方药", "非处方药", "抗生素", "维生素",
                     "钙片", "补钙", "补锌", "护肝药", "保健品"}

    results = [
        (name, count)
        for name, count in candidate_counter.most_common()
        if count >= 2 and name not in generic_words
    ]
    return results
