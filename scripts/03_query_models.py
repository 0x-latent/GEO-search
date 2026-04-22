"""
批量调用国内大模型API，回收应答结果。
支持：多模型并行、模型内并发、自适应限流、断点续跑、执行日志、去重。
"""
import asyncio
import json
import os
import sys
import time
import logging
import yaml
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_clients import ModelClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
LOG_PATH = os.path.join(RESULTS_DIR, "execution_log.json")


class AdaptiveThrottle:
    """
    自适应限流控制器（每个模型独立一个实例）。

    策略：
    - 正常时：按配置的 concurrency 和 request_interval 运行
    - 遇到429限流：全局暂停，等待后恢复，并自动降低并发
    - 连续成功一段时间后：尝试恢复并发到初始值
    """

    def __init__(self, model_key: str, initial_concurrency: int, initial_interval: float):
        self.model_key = model_key
        self.initial_concurrency = initial_concurrency
        self.initial_interval = initial_interval

        self.concurrency = initial_concurrency
        self.interval = initial_interval
        self.semaphore = asyncio.Semaphore(initial_concurrency)

        # 限流暂停锁：遇到429时所有请求等待
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # 初始未暂停

        self._consecutive_success = 0
        self._recover_threshold = 50  # 连续成功50次后尝试恢复

    async def acquire(self):
        """获取执行许可（等待暂停恢复 + 信号量）"""
        await self._pause_event.wait()
        await self.semaphore.acquire()
        await asyncio.sleep(self.interval)

    def release(self):
        self.semaphore.release()

    def on_success(self):
        self._consecutive_success += 1
        # 连续成功足够多次，尝试恢复并发
        if (self._consecutive_success >= self._recover_threshold
                and self.concurrency < self.initial_concurrency):
            self.concurrency = min(self.concurrency + 1, self.initial_concurrency)
            self.interval = max(self.interval - 0.05, self.initial_interval)
            self.semaphore = asyncio.Semaphore(self.concurrency)
            logger.info(f"[{self.model_key}] 恢复并发至 {self.concurrency}, 间隔 {self.interval:.2f}s")
            self._consecutive_success = 0

    async def on_rate_limit(self, retry_after: float = None):
        """
        遇到429限流时调用。
        暂停所有该模型的请求，等待后降速恢复。
        """
        self._consecutive_success = 0

        # 计算等待时间
        wait = retry_after if retry_after else 10.0

        # 降低并发（最低降到1）
        old_concurrency = self.concurrency
        self.concurrency = max(1, self.concurrency // 2)
        self.interval = min(self.interval * 1.5, 5.0)
        self.semaphore = asyncio.Semaphore(self.concurrency)

        logger.warning(
            f"[{self.model_key}] 触发限流！暂停 {wait:.0f}s, "
            f"并发 {old_concurrency} → {self.concurrency}, "
            f"间隔 → {self.interval:.2f}s"
        )

        # 暂停所有请求
        self._pause_event.clear()
        await asyncio.sleep(wait)
        self._pause_event.set()

        logger.info(f"[{self.model_key}] 限流等待结束，恢复执行")


def _is_rate_limit_error(e: Exception) -> tuple:
    """判断是否为限流错误，返回 (is_rate_limit, retry_after_seconds)"""
    err_str = str(e)
    # OpenAI SDK的429错误
    if "429" in err_str or "rate_limit" in err_str.lower() or "Rate limit" in err_str:
        # 尝试提取retry-after
        retry_after = None
        if hasattr(e, "response") and hasattr(e.response, "headers"):
            ra = e.response.headers.get("retry-after")
            if ra:
                try:
                    retry_after = float(ra)
                except ValueError:
                    pass
        return True, retry_after
    return False, None


def load_execution_log() -> dict:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"executions": []}


def save_execution_log(log: dict):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def build_completed_keys(log: dict) -> set:
    keys = set()
    for entry in log.get("executions", []):
        if entry.get("status") == "success":
            key = (entry["question_id"], entry["model"], entry["search_enabled"], entry["round"])
            keys.add(key)
    return keys


def make_task_key(question_id: str, model: str, search_enabled: bool, round_num: int) -> tuple:
    return (question_id, model, search_enabled, round_num)


def result_filename(question_id: str, round_num: int, search_enabled: bool) -> str:
    search_tag = "search" if search_enabled else "nosearch"
    return f"{question_id}_r{round_num}_{search_tag}.json"


async def execute_single_query(
    model_key: str,
    client: ModelClient,
    question: dict,
    round_num: int,
    search_enabled: bool,
    query_settings: dict,
    log: dict,
    completed_keys: set,
    lock: asyncio.Lock,
    throttle: AdaptiveThrottle,
    counter: dict,
):
    """执行单次查询（带自适应限流控制）"""
    task_key = make_task_key(question["id"], model_key, search_enabled, round_num)
    if task_key in completed_keys:
        counter["done"] += 1
        return

    model_dir = os.path.join(RAW_DIR, model_key)
    max_retries = query_settings.get("retry_max", 3)
    retry_delay = query_settings.get("retry_delay", 5)

    for attempt in range(1, max_retries + 1):
        await throttle.acquire()
        try:
            result = await client.query(
                question=question["question"],
                enable_search=search_enabled,
                temperature=query_settings.get("temperature", 0.7),
                max_tokens=query_settings.get("max_tokens", 2048),
            )

            result["question_id"] = question["id"]
            result["question_text"] = question["question"]
            result["product"] = question["product"]
            result["round"] = round_num
            result["timestamp"] = datetime.now().isoformat()

            fname = result_filename(question["id"], round_num, search_enabled)
            fpath = os.path.join(model_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            entry = {
                "question_id": question["id"],
                "model": model_key,
                "search_enabled": search_enabled,
                "round": round_num,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "result_file": f"results/raw/{model_key}/{fname}",
                "error": None,
            }
            async with lock:
                log["executions"].append(entry)
                completed_keys.add(task_key)
                counter["done"] += 1
                if counter["done"] % 10 == 0:
                    save_execution_log(log)

            throttle.release()
            throttle.on_success()

            search_tag = "联网" if search_enabled else "不联网"
            logger.info(
                f"[{model_key}] ({counter['done']}/{counter['total']}) "
                f"{question['id']} 轮{round_num} {search_tag} ✓ "
                f"({result['latency_ms']}ms)"
            )
            return

        except Exception as e:
            throttle.release()

            is_rate_limit, retry_after = _is_rate_limit_error(e)

            if is_rate_limit:
                # 限流：触发全局降速，然后重试（不计入重试次数）
                await throttle.on_rate_limit(retry_after)
                continue

            # 非限流错误：正常重试逻辑
            if attempt < max_retries:
                wait = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"[{model_key}] {question['id']} 轮{round_num} "
                    f"失败(第{attempt}次): {e}, {wait}秒后重试"
                )
                await asyncio.sleep(wait)
            else:
                entry = {
                    "question_id": question["id"],
                    "model": model_key,
                    "search_enabled": search_enabled,
                    "round": round_num,
                    "status": "failed",
                    "timestamp": datetime.now().isoformat(),
                    "result_file": None,
                    "error": str(e),
                }
                async with lock:
                    log["executions"].append(entry)
                    counter["done"] += 1
                    if counter["done"] % 10 == 0:
                        save_execution_log(log)
                logger.error(
                    f"[{model_key}] {question['id']} 轮{round_num} 最终失败: {e}"
                )


async def query_single_model(
    model_key: str,
    client: ModelClient,
    questions: list,
    rounds: int,
    query_settings: dict,
    log: dict,
    completed_keys: set,
    lock: asyncio.Lock,
):
    """单个模型的完整查询流程（自适应并发）"""
    model_dir = os.path.join(RAW_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)

    concurrency = client.config.get("concurrency", 1)
    interval = client.config.get("request_interval", 0.2)
    throttle = AdaptiveThrottle(model_key, concurrency, interval)

    search_modes = [False]
    if client.supports_search:
        search_modes.append(True)

    total = len(questions) * rounds * len(search_modes)
    counter = {"done": 0, "total": total}

    # 分阶段执行：先跑不联网，再跑联网
    for search_enabled in search_modes:
        phase = "联网" if search_enabled else "不联网"
        logger.info(f"[{model_key}] 开始阶段: {phase}")

        tasks = []
        for question in questions:
            for round_num in range(1, rounds + 1):
                tasks.append(
                    execute_single_query(
                        model_key, client, question, round_num, search_enabled,
                        query_settings, log, completed_keys, lock, throttle, counter,
                    )
                )

        await asyncio.gather(*tasks)
        logger.info(f"[{model_key}] 阶段完成: {phase}")

    async with lock:
        save_execution_log(log)

    logger.info(f"[{model_key}] 全部完成 ({counter['done']}/{total})")


async def main():
    config_path = os.path.join(BASE_DIR, "config", "models.yaml")
    keys_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(keys_path, "r", encoding="utf-8") as f:
        keys = yaml.safe_load(f)

    query_settings = config.get("query_settings", {})
    rounds = query_settings.get("rounds", 5)

    questions_path = os.path.join(BASE_DIR, "questions", "questions_expanded.json")
    if not os.path.exists(questions_path):
        questions_path = os.path.join(BASE_DIR, "questions", "questions_base.json")

    if not os.path.exists(questions_path):
        logger.error(f"问题文件不存在: {questions_path}")
        logger.error("请先运行 01_parse_questions.py")
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"加载 {len(questions)} 个问题，每题 {rounds} 轮")

    log = load_execution_log()
    completed_keys = build_completed_keys(log)
    logger.info(f"已完成 {len(completed_keys)} 个任务（断点续跑）")

    lock = asyncio.Lock()
    tasks = []

    for model_key, model_config in config.get("models", {}).items():
        if not model_config.get("enabled", False):
            logger.info(f"[{model_key}] 未启用，跳过")
            continue

        api_key = keys.get(model_key, {}).get("api_key", "")
        if not api_key or api_key == "sk-xxx":
            logger.warning(f"[{model_key}] API key未配置，跳过")
            continue

        client = ModelClient(model_key, model_config, api_key)
        concurrency = model_config.get("concurrency", 1)
        task = query_single_model(
            model_key, client, questions, rounds,
            query_settings, log, completed_keys, lock,
        )
        tasks.append(task)
        logger.info(f"[{model_key}] 已加入队列 (联网: {model_config.get('supports_search', False)}, 并发: {concurrency})")

    if not tasks:
        logger.warning("没有可用的模型，请检查 config/models.yaml 和 config/api_keys.yaml")
        return

    logger.info(f"开始执行，{len(tasks)} 个模型并行...")
    await asyncio.gather(*tasks)
    logger.info("全部执行完成！")

    log = load_execution_log()
    success = sum(1 for e in log["executions"] if e["status"] == "success")
    failed = sum(1 for e in log["executions"] if e["status"] == "failed")
    logger.info(f"统计: 成功 {success}, 失败 {failed}")


if __name__ == "__main__":
    asyncio.run(main())
