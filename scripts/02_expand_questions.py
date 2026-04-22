"""
调用OpenAI gpt-5.4生成问题变体，扩展问题集。
prompt配置统一从 config/prompts/expand_questions.yaml 加载。
"""
import asyncio
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_clients import OpenAIClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_prompt_config():
    """加载prompt配置（系统提示词、变体策略、负面问题模板）"""
    path = os.path.join(BASE_DIR, "config", "prompts", "expand_questions.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_json_response(text: str) -> list:
    """从LLM响应中提取JSON数组"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


async def expand_questions(base_path: str, output_path: str, api_key: str):
    config = load_prompt_config()
    system_prompt = config["system_prompt"]
    variant_strategy = config["variant_strategy"]
    negative_prompt_template = config["negative_prompt_template"]

    with open(base_path, "r", encoding="utf-8") as f:
        base_questions = json.load(f)

    client = OpenAIClient(api_key=api_key, model="gpt-5.4")
    expanded = list(base_questions)

    # 按产品分组
    products = {}
    for q in base_questions:
        products.setdefault(q["product_code"], []).append(q)

    for product_code, questions in products.items():
        product_name = questions[0]["product"]
        print(f"\n处理产品: {product_name} ({product_code})")

        # 1. 为每个基础问题生成变体
        for q in questions:
            level = q["level"]
            strategy = variant_strategy.get(level)
            if not strategy:
                continue

            user_prompt = (
                f"产品：{product_name}\n"
                f"基础问题：{q['question']}\n\n"
                f"请按以下要求生成{strategy['count']}个变体：\n"
                f"{strategy['instructions']}"
            )

            try:
                response = await client.generate(system_prompt, user_prompt)
                variants = _parse_json_response(response)

                for i, v in enumerate(variants):
                    variant_id = f"{q['id']}_v{i+1}"
                    expanded.append({
                        "id": variant_id,
                        "product": q["product"],
                        "product_code": q["product_code"],
                        "category": q["category"],
                        "level": q["level"],
                        "question": v["question"],
                        "has_brand_name": not (v.get("variant_type") == "no_brand"),
                        "is_variant": True,
                        "variant_of": q["id"],
                        "variant_type": v.get("variant_type", "unknown"),
                    })
                print(f"  {q['level']}: 生成 {len(variants)} 个变体")
            except Exception as e:
                print(f"  {q['level']}: 变体生成失败 - {e}")

        # 2. 生成负面/安全类问题
        base_q_texts = "\n".join(f"- {q['question']}" for q in questions)
        neg_prompt = negative_prompt_template.format(
            product_name=product_name, base_questions=base_q_texts
        )

        try:
            response = await client.generate(system_prompt, neg_prompt)
            neg_questions = _parse_json_response(response)

            for i, nq in enumerate(neg_questions):
                neg_id = f"{product_code}_negative_{i+1}"
                has_brand = any(kw in nq["question"]
                               for kw in ["999", "三九", product_name])
                expanded.append({
                    "id": neg_id,
                    "product": product_name,
                    "product_code": product_code,
                    "category": "negative_safety",
                    "level": "negative_safety",
                    "question": nq["question"],
                    "has_brand_name": has_brand,
                    "is_variant": False,
                    "variant_of": None,
                    "variant_type": "negative_safety",
                })
            print(f"  负面安全: 生成 {len(neg_questions)} 个问题")
        except Exception as e:
            print(f"  负面安全: 生成失败 - {e}")

        # API限流
        await asyncio.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(expanded, f, ensure_ascii=False, indent=2)

    base_count = len(base_questions)
    total_count = len(expanded)
    print(f"\n扩展完成: {base_count} → {total_count} 个问题 → {output_path}")


if __name__ == "__main__":
    keys_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
    if not os.path.exists(keys_path):
        print(f"错误: 请先创建 {keys_path}（参考 api_keys.yaml.example）")
        sys.exit(1)

    with open(keys_path, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f)

    openai_key = keys.get("openai", {}).get("api_key", "")
    if not openai_key or openai_key == "sk-xxx":
        print("错误: 请在 api_keys.yaml 中填写 OpenAI API key")
        sys.exit(1)

    base_path = os.path.join(BASE_DIR, "questions", "questions_base.json")
    if not os.path.exists(base_path):
        print(f"错误: 请先运行 01_parse_questions.py 生成 {base_path}")
        sys.exit(1)

    output_path = os.path.join(BASE_DIR, "questions", "questions_expanded.json")
    asyncio.run(expand_questions(base_path, output_path, openai_key))
