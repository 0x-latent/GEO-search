"""
解析xlsx问题表 → 结构化JSON
"""
import json
import sys
import os
import pandas as pd

# 产品code映射
PRODUCT_CODE_MAP = {
    "感冒灵": "ganmaolin",
    "皮炎平": "piyanping",
    "胃泰": "weitai",
    "抗病毒": "kangbingdu",
    "小感": "xiaogan",
    "强枇": "qiangpi",
    "澳诺": "aonuo",
    "易善复": "yishanfu",
}

# 问题层级映射
LEVEL_MAP = {
    0: ("q1_overall", "accuracy"),
    1: ("q2_detail", "accuracy"),
    2: ("q3_scenario1", "mention_recommend"),
    3: ("q4_scenario2", "mention_recommend"),
    4: ("q5_top3", "mention_recommend"),
}

# 999品牌关键词（用于判断问题中是否带品牌名）
BRAND_KEYWORDS = ["999", "三九", "养胃舒", "感冒灵", "皮炎平", "抗病毒",
                  "小儿氨酚黄那敏", "强力枇杷露", "澳诺", "易善复"]


def parse_questions(xlsx_path: str, output_path: str):
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    questions = []
    product_columns = list(df.columns[1:])  # 跳过第一列"产品"

    for col in product_columns:
        product = col
        product_code = PRODUCT_CODE_MAP.get(product, product)

        for row_idx in range(len(df)):
            question_text = str(df[col].iloc[row_idx]).strip()
            if not question_text or question_text == "nan":
                continue

            level, category = LEVEL_MAP.get(row_idx, ("unknown", "unknown"))
            question_id = f"{product_code}_{level}"

            has_brand = any(kw in question_text for kw in BRAND_KEYWORDS)

            questions.append({
                "id": question_id,
                "product": product,
                "product_code": product_code,
                "category": category,
                "level": level,
                "question": question_text,
                "has_brand_name": has_brand,
                "is_variant": False,
                "variant_of": None,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"解析完成: {len(questions)} 个问题 → {output_path}")
    return questions


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xlsx_path = os.path.join(base_dir, "questions", "GEO产品摸底问题list.xlsx")

    # 如果questions目录下没有xlsx，尝试从项目根目录复制
    if not os.path.exists(xlsx_path):
        root_xlsx = os.path.join(base_dir, "GEO产品摸底问题list.xlsx")
        if os.path.exists(root_xlsx):
            import shutil
            shutil.copy2(root_xlsx, xlsx_path)

    output_path = os.path.join(base_dir, "questions", "questions_base.json")
    parse_questions(xlsx_path, output_path)
