"""
统一API客户端封装
支持两种API模式：
- Chat Completions API：标准对话（DeepSeek、混元、以及千问/豆包的非联网模式）
- Responses API：联网搜索（千问、豆包的联网模式），通过OpenAI SDK调用
"""
import time
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ModelClient:
    """统一模型客户端"""

    def __init__(self, model_key: str, config: dict, api_key: str):
        self.model_key = model_key
        self.config = config
        self.name = config["name"]
        self.model_id = config["model_id"]
        self.search_model_id = config.get("search_model_id", self.model_id)
        self.supports_search = config.get("supports_search", False)
        self.search_api = config.get("search_api", "chat_completions")
        self.request_interval = config.get("request_interval", 1.0)
        self.api_key = api_key
        self.endpoint = config["endpoint"]

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config["endpoint"],
        )

    async def query(self, question: str, enable_search: bool = False,
                    temperature: float = 0.7, max_tokens: int = 2048) -> dict:
        """
        发送问题并返回结构化结果。
        根据模型配置自动选择Chat Completions或Responses API。
        """
        actual_search = enable_search and self.supports_search

        if actual_search and self.search_api == "responses":
            return await self._query_responses_api(question, temperature, max_tokens)
        else:
            return await self._query_chat_completions(question, actual_search, temperature, max_tokens)

    async def _query_chat_completions(self, question: str, enable_search: bool,
                                       temperature: float, max_tokens: int) -> dict:
        """标准Chat Completions API调用"""
        start = time.time()

        messages = [{"role": "user", "content": question}]
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if enable_search:
            search_param = self.config.get("search_param", "")
            if search_param:
                kwargs["extra_body"] = {search_param: True}

        try:
            response = await self.client.chat.completions.create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)

            answer = response.choices[0].message.content or ""
            sources = self._extract_sources_from_chat(response)

            return {
                "answer": answer,
                "sources": sources,
                "model": self.model_key,
                "model_name": self.name,
                "latency_ms": latency_ms,
                "search_enabled": enable_search,
                "raw_response": response.model_dump(),
            }
        except Exception as e:
            logger.error(f"[{self.model_key}] Chat Completions调用失败: {e}")
            raise

    async def _query_responses_api(self, question: str,
                                    temperature: float, max_tokens: int) -> dict:
        """
        Responses API调用（千问/豆包联网搜索模式）。
        通过OpenAI SDK的responses.create调用，SDK自动处理正确的endpoint路径。
        """
        start = time.time()

        tools = [{"type": "web_search"}]
        if self.model_key == "doubao":
            tools = [{"type": "web_search", "max_keyword": 3}]

        try:
            response = await self.client.responses.create(
                model=self.search_model_id,
                input=[{"role": "user", "content": question}],
                tools=tools,
            )
            latency_ms = int((time.time() - start) * 1000)

            # 解析response对象
            answer, sources = self._parse_responses_output(response)

            # 尝试序列化raw_response
            try:
                raw = response.model_dump()
            except Exception:
                raw = {"output_text": answer}

            return {
                "answer": answer,
                "sources": sources,
                "model": self.model_key,
                "model_name": self.name,
                "latency_ms": latency_ms,
                "search_enabled": True,
                "raw_response": raw,
            }
        except Exception as e:
            logger.error(f"[{self.model_key}] Responses API调用失败: {e}")
            raise

    def _parse_responses_output(self, response) -> tuple:
        """
        解析Responses API的返回。
        兼容SDK对象和dict两种格式。
        """
        answer = ""
        sources = []

        # SDK对象：直接取output_text属性
        if hasattr(response, "output_text") and response.output_text:
            answer = response.output_text

        # 遍历output提取详细信息
        output = []
        if hasattr(response, "output"):
            output = response.output
        elif isinstance(response, dict):
            output = response.get("output", [])

        for item in output:
            # 获取type（兼容对象和dict）
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else "")

            if item_type == "message":
                content = getattr(item, "content", None) or (item.get("content", []) if isinstance(item, dict) else [])
                for c in content:
                    c_type = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else "")
                    if c_type == "output_text":
                        text = getattr(c, "text", None) or (c.get("text", "") if isinstance(c, dict) else "")
                        if text:
                            answer = text
                    # annotations
                    annotations = getattr(c, "annotations", None) or (c.get("annotations", []) if isinstance(c, dict) else [])
                    for ann in annotations:
                        ann_type = getattr(ann, "type", None) or (ann.get("type") if isinstance(ann, dict) else "")
                        if ann_type == "url_citation":
                            title = getattr(ann, "title", "") or (ann.get("title", "") if isinstance(ann, dict) else "")
                            url = getattr(ann, "url", "") or (ann.get("url", "") if isinstance(ann, dict) else "")
                            sources.append({"title": title, "url": url})

            elif item_type == "web_search_call":
                action = getattr(item, "action", None) or (item.get("action", {}) if isinstance(item, dict) else {})
                action_sources = getattr(action, "sources", None) or (action.get("sources", []) if isinstance(action, dict) else [])
                for src in action_sources:
                    url = getattr(src, "url", "") or (src.get("url", "") if isinstance(src, dict) else "")
                    if url:
                        sources.append({"title": "", "url": url})

        return answer, sources

    def _extract_sources_from_chat(self, response) -> list:
        """从Chat Completions响应中提取引用源"""
        sources = []
        try:
            raw = response.model_dump()
            raw_str = str(raw)
            if "web_search_results" in raw_str:
                for choice in raw.get("choices", []):
                    msg = choice.get("message", {})
                    for ref in msg.get("web_search_results", []):
                        sources.append({
                            "title": ref.get("title", ""),
                            "url": ref.get("url", ""),
                        })
        except Exception:
            pass
        return sources


class OpenAIClient:
    """OpenAI客户端，用于问题变体生成"""

    def __init__(self, api_key: str, model: str = "gpt-5.4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, system_prompt: str, user_prompt: str,
                       temperature: float = 0.7, max_completion_tokens: int = 4096) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        return response.choices[0].message.content or ""
