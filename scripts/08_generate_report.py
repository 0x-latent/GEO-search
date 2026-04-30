"""
调用GPT-5.4生成完整的AI应答分析报告。
读取04脚本输出的CSV数据，组装成prompt，输出markdown报告。
需手动触发执行。
"""
import asyncio
import json
import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_clients import OpenAIClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(BASE_DIR, "results", "analysis")
REPORT_DIR = os.path.join(BASE_DIR, "results", "report")


def load_csv_as_text(filename: str, max_rows: int = 200) -> str:
    """读取CSV并转为紧凑的文本格式，供prompt使用"""
    path = os.path.join(ANALYSIS_DIR, filename)
    if not os.path.exists(path):
        return f"（{filename} 不存在）"

    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated = f"\n（仅展示前{max_rows}行）"
    else:
        truncated = ""

    return df.to_csv(index=False) + truncated


def load_prompt_config() -> dict:
    path = os.path.join(BASE_DIR, "config", "prompts", "analysis_report.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_query_settings() -> dict:
    path = os.path.join(BASE_DIR, "config", "models.yaml")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    enabled_models = [k for k, v in config.get("models", {}).items() if v.get("enabled")]
    rounds = config.get("query_settings", {}).get("rounds", 5)

    # 从dashboard读取产品数
    dashboard_path = os.path.join(ANALYSIS_DIR, "dashboard.csv")
    product_count = 0
    if os.path.exists(dashboard_path):
        df = pd.read_csv(dashboard_path, encoding="utf-8-sig")
        product_count = df["产品"].nunique() if "产品" in df.columns else 0

    return {
        "model_count": len(enabled_models),
        "product_count": product_count,
        "rounds": rounds,
    }


def build_prompt() -> tuple:
    """组装系统提示词和用户提示词"""
    prompt_config = load_prompt_config()
    settings = load_query_settings()

    system_prompt = prompt_config["system_prompt"]

    # 加载各维度数据
    data = {
        "model_count": settings["model_count"],
        "product_count": settings["product_count"],
        "rounds": settings["rounds"],
        "dashboard_data": load_csv_as_text("dashboard.csv"),
        "mention_data": load_csv_as_text("mention_report.csv"),
        "recommendation_data": load_csv_as_text("recommendation_report.csv"),
        "cross_model_data": load_csv_as_text("cross_model_consistency.csv", max_rows=100),
        "search_diff_data": load_csv_as_text("search_diff_report.csv", max_rows=100),
        "competitor_data": load_csv_as_text("competitor_report.csv"),
        "stability_data": load_csv_as_text("stability_report.csv", max_rows=100),
        "variant_data": load_csv_as_text("variant_sensitivity_report.csv", max_rows=100),
        "source_data": load_csv_as_text("source_report.csv", max_rows=50),
    }

    user_prompt = prompt_config["report_template"].format(**data)

    return system_prompt, user_prompt


async def generate_report(api_key: str):
    print("组装分析数据...")
    system_prompt, user_prompt = build_prompt()

    # 输出prompt长度供参考
    total_chars = len(system_prompt) + len(user_prompt)
    print(f"Prompt总长度: {total_chars} 字符 (约 {total_chars // 2} tokens)")

    print("调用GPT-5.4生成报告...")
    client = OpenAIClient(api_key=api_key, model="gpt-5.4")

    report = await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,  # 分析报告用低温度，保证准确性
        max_completion_tokens=16000,
    )

    # 保存报告
    os.makedirs(REPORT_DIR, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"analysis_report_{timestamp}.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已生成: {report_path}")
    print(f"报告长度: {len(report)} 字符")

    # 同时保存一份prompt日志（方便调试和复现）
    prompt_log_path = os.path.join(REPORT_DIR, f"prompt_log_{timestamp}.json")
    with open(prompt_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "report_length": len(report),
            "timestamp": timestamp,
        }, f, ensure_ascii=False, indent=2)

    return report_path


if __name__ == "__main__":
    keys_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
    if not os.path.exists(keys_path):
        print(f"错误: 请先创建 {keys_path}")
        sys.exit(1)

    with open(keys_path, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f)

    openai_key = keys.get("openai", {}).get("api_key", "")
    if not openai_key or openai_key == "sk-xxx":
        print("错误: 请在 api_keys.yaml 中填写 OpenAI API key")
        sys.exit(1)

    # 检查分析数据是否存在
    dashboard_path = os.path.join(ANALYSIS_DIR, "dashboard.csv")
    if not os.path.exists(dashboard_path):
        print("错误: 请先运行 04_analyze_results.py 生成分析数据")
        sys.exit(1)

    asyncio.run(generate_report(openai_key))
