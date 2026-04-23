"""
将产品资料txt解析为结构化JSON知识库。
支持增量更新：通过MD5哈希检测文件变更，只处理新增或修改的产品。
支持5并发调用LLM。
"""
import asyncio
import hashlib
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_clients import OpenAIClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_DIR = os.path.join(BASE_DIR, "docs", "products_db")
KB_PATH = os.path.join(BASE_DIR, "config", "knowledge_base.json")

CONCURRENCY = 5

SYSTEM_PROMPT = """你是一个药品知识库结构化专家。你的任务是将药品产品资料提取为结构化的知识点列表。

每个知识点（fact）包含以下字段：
- "category": 分类，取值范围：
  - "basic_info"：产品基本信息（名称、通用名、品牌等）
  - "efficacy"：功效与作用（主治、适应症）
  - "ingredients"：成分信息
  - "usage"：用法用量
  - "contraindication"：禁忌与不良反应
  - "caution"：注意事项
  - "differentiation"：产品区分/对比（如风寒风热、红皮绿皮）
  - "applicable_scene"：适用场景与人群
  - "mechanism"：作用机制与产品特点
- "fact": 知识点的准确陈述（一句话，明确无歧义）
- "keywords": 关键词列表（用于在AI回答中快速检索）
- "importance": 重要程度，"critical"（必须准确）| "important"（应该提及）| "nice_to_have"（加分项）
- "wrong_claims": 常见错误说法列表（如果有的话，AI回答中出现这些则判为错误）

要求：
1. 每个知识点必须是可验证的事实，不是主观评价
2. 优先提取与消费者问答最相关的知识点
3. "critical"级别用于：功效、适应症、禁忌、成分等核心信息
4. "important"级别用于：用法用量、注意事项、产品对比等
5. "nice_to_have"用于：包装规格、企业信息等
6. wrong_claims要列出常见的误解（如"感冒灵只治风热"就是错误的）

严格返回JSON数组，不要任何其他文本。"""

USER_PROMPT_TEMPLATE = """请将以下产品资料提取为结构化知识点：

产品名称：{product_name}

产品资料：
{product_text}

请提取所有重要知识点，尤其关注：
1. 产品的功效和作用（必须准确）
2. 适用症状和场景
3. 成分信息
4. 禁忌和注意事项
5. 与同类产品的区别
6. 常见误解和错误说法"""


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_product_texts() -> dict:
    products = {}
    if not os.path.exists(PRODUCTS_DIR):
        return products
    for filename in os.listdir(PRODUCTS_DIR):
        if filename.endswith(".txt"):
            product_name = filename.replace(".txt", "")
            filepath = os.path.join(PRODUCTS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                products[product_name] = f.read()
    return products


def load_existing_kb() -> dict:
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _parse_json_response(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


async def _process_one_product(
    client: OpenAIClient,
    product_name: str,
    product_text: str,
    semaphore: asyncio.Semaphore,
    counter: dict,
) -> tuple:
    """处理单个产品（带并发控制），返回 (product_name, result_dict) 或 (product_name, None)"""
    async with semaphore:
        counter["done"] += 1
        print(f"({counter['done']}/{counter['total']}) 提取: {product_name}")

        user_prompt = USER_PROMPT_TEMPLATE.format(
            product_name=product_name,
            product_text=product_text,
        )

        try:
            response = await client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
                max_completion_tokens=8000,
            )
            facts = _parse_json_response(response)

            critical = sum(1 for f in facts if f.get("importance") == "critical")
            important = sum(1 for f in facts if f.get("importance") == "important")
            print(f"  {product_name}: {len(facts)} 个知识点 (critical: {critical}, important: {important})")

            return product_name, {
                "product_name": product_name,
                "facts": facts,
                "fact_count": len(facts),
                "source_md5": _md5(product_text),
            }
        except Exception as e:
            print(f"  {product_name}: 提取失败 - {e}")
            return product_name, None


async def build_knowledge_base(api_key: str, force: bool = False):
    products = load_product_texts()
    if not products:
        print(f"错误: {PRODUCTS_DIR} 中没有找到产品资料文件")
        return

    print(f"找到 {len(products)} 个产品资料")

    # 加载已有知识库，检测变更
    existing_kb = load_existing_kb()

    to_process = {}
    skipped = []
    for name, text in products.items():
        current_md5 = _md5(text)
        existing = existing_kb.get(name, {})
        existing_md5 = existing.get("source_md5", "")

        if not force and current_md5 == existing_md5:
            skipped.append(name)
        else:
            reason = "新增" if name not in existing_kb else "内容已修改"
            to_process[name] = (text, reason)

    if skipped:
        print(f"跳过（未变更）: {', '.join(skipped)}")

    if not to_process:
        print("所有产品均无变更，无需处理")
        return

    print(f"需要处理: {len(to_process)} 个")
    for name, (_, reason) in to_process.items():
        print(f"  {name} ({reason})")

    client = OpenAIClient(api_key=api_key, model="gpt-5.4")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    counter = {"done": 0, "total": len(to_process)}

    # 并发处理
    tasks = [
        _process_one_product(client, name, text, semaphore, counter)
        for name, (text, _) in to_process.items()
    ]
    results = await asyncio.gather(*tasks)

    # 合并到知识库（保留未变更的旧数据）
    knowledge_base = dict(existing_kb)
    success = 0
    for name, result in results:
        if result is not None:
            knowledge_base[name] = result
            success += 1

    # 保存
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    total_facts = sum(p.get("fact_count", 0) for p in knowledge_base.values())
    print(f"\n知识库已更新: {KB_PATH}")
    print(f"  本次处理: {success}/{len(to_process)} 成功")
    print(f"  知识库总计: {len(knowledge_base)} 个产品, {total_facts} 个知识点")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="强制重新提取所有产品（忽略MD5缓存）")
    args = parser.parse_args()

    keys_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
    with open(keys_path, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f)

    openai_key = keys.get("openai", {}).get("api_key", "")
    if not openai_key or openai_key == "sk-xxx":
        print("错误: 请在 api_keys.yaml 中填写 OpenAI API key")
        sys.exit(1)

    asyncio.run(build_knowledge_base(openai_key, force=args.force))
