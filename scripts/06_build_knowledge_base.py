"""
将产品知识库目录（docs/products_db/{产品名}/*.md）合并为统一的JSON文件。
纯离线操作，不调用LLM。遍历每个产品目录下的md文件，按编号排序拼接为完整文本。
支持增量更新：通过MD5检测变更。
"""
import hashlib
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_DIR = os.path.join(BASE_DIR, "docs", "products_db")
KB_PATH = os.path.join(BASE_DIR, "config", "knowledge_base.json")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_product_texts() -> dict:
    """从 docs/products_db/{产品名}/ 读取所有 md 文件，按文件名排序拼接。
    跳过 _template 和 _raw 目录。
    返回 {产品名: 合并文本}
    """
    products = {}
    if not os.path.exists(PRODUCTS_DIR):
        return products

    for dirname in os.listdir(PRODUCTS_DIR):
        if dirname.startswith("_"):
            continue
        product_dir = os.path.join(PRODUCTS_DIR, dirname)
        if not os.path.isdir(product_dir):
            continue

        # 收集所有 md 文件，按文件名排序
        md_files = sorted(
            f for f in os.listdir(product_dir)
            if f.endswith(".md") and not f.startswith("_")
        )
        if not md_files:
            continue

        parts = []
        for md_file in md_files:
            filepath = os.path.join(product_dir, md_file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                parts.append(content)

        if parts:
            products[dirname] = "\n\n---\n\n".join(parts)

    return products


def load_existing_kb() -> dict:
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_knowledge_base(force: bool = False):
    products = load_product_texts()
    if not products:
        print(f"错误: {PRODUCTS_DIR} 中没有找到产品知识库目录")
        return

    print(f"找到 {len(products)} 个产品知识库")

    existing_kb = load_existing_kb()

    updated = 0
    skipped = 0
    knowledge_base = {}

    for name, text in products.items():
        current_md5 = _md5(text)
        existing = existing_kb.get(name, {})
        existing_md5 = existing.get("source_md5", "")

        if not force and current_md5 == existing_md5:
            knowledge_base[name] = existing
            skipped += 1
        else:
            reason = "新增" if name not in existing_kb else "内容已修改"
            print(f"  {name} ({reason}, {len(text)} 字)")
            knowledge_base[name] = {
                "product_name": name,
                "text": text,
                "char_count": len(text),
                "source_md5": current_md5,
            }
            updated += 1

    # 保存
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    total_chars = sum(p.get("char_count", len(p.get("text", ""))) for p in knowledge_base.values())
    print(f"\n知识库已更新: {KB_PATH}")
    print(f"  更新: {updated} 个, 跳过: {skipped} 个")
    print(f"  总计: {len(knowledge_base)} 个产品, {total_chars} 字")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="强制重新构建所有产品（忽略MD5缓存）")
    args = parser.parse_args()
    build_knowledge_base(force=args.force)
