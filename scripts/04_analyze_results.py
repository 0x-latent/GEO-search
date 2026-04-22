"""
统计分析：读取全部原始应答，解析后生成各维度报表。
"""
import json
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.parser import parse_single_response
from utils.reporter import generate_all_reports

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
ANALYSIS_DIR = os.path.join(BASE_DIR, "results", "analysis")


def load_questions_map() -> dict:
    """加载问题映射，用于补充解析结果中的元数据"""
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
    """加载所有原始应答文件"""
    responses = []
    pattern = os.path.join(RAW_DIR, "**", "*.json")
    files = glob.glob(pattern, recursive=True)

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 跳过执行日志等非应答文件
            if "question_id" in data and "answer" in data:
                responses.append(data)
        except Exception as e:
            print(f"跳过无法解析的文件: {fpath} ({e})")

    return responses


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

        # 补充问题元数据
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

    print("\n生成报表...")
    generate_all_reports(parsed_results, ANALYSIS_DIR)
    print(f"\n全部报表已生成至: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
