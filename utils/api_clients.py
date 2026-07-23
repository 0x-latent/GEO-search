"""
统一API客户端封装
支持三种API模式：
- Chat Completions API：标准对话（DeepSeek、以及千问/豆包的非联网模式）
- Responses API：联网搜索（千问、豆包的联网模式），通过OpenAI SDK调用
- 腾讯云原生API：混元（使用SecretId+SecretKey签名认证）
"""
import asyncio
import hashlib
import json
import time
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _response_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str,
            separators=(",", ":"),
        )
    except Exception:
        payload = str(value)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def resolve_relay(models_config: dict, keys: dict) -> dict | None:
    """从 models.yaml 的 relay 段和 api_keys.yaml 的 relay key 组装中继配置。

    返回 {"base_url", "api_key"}；base_url 或 key 缺失时返回 None（回退直连）。
    """
    relay = dict(models_config.get("relay") or {})
    key = (keys.get("relay") or {}).get("api_key", "")
    if relay.get("base_url") and key and key != "sk-xxx":
        relay["api_key"] = key
        return relay
    return None


def resolve_route(models_config: dict, keys: dict, requested: str | None = None) -> str:
    """决定链路：requested 显式指定优先；否则 relay 可用且 enabled 时默认走 relay。"""
    relay = resolve_relay(models_config, keys)
    if requested == "direct":
        return "direct"
    if requested == "relay":
        if relay is None:
            logger.warning("请求走中继但 relay 未配置（缺 base_url 或 api_key），回退直连")
            return "direct"
        return "relay"
    if relay is not None and (models_config.get("relay") or {}).get("enabled", False):
        return "relay"
    return "direct"


class ModelClient:
    """统一模型客户端"""

    def __init__(self, model_key: str, config: dict, api_key: str,
                 route: str = "direct", relay_config: dict | None = None,
                 model_override: str | None = None):
        self.model_key = model_key
        self.config = config
        self.name = config["name"]
        self.model_id = model_override or config["model_id"]
        self.search_model_id = model_override or config.get("search_model_id", config["model_id"])
        self.supports_search = config.get("supports_search", False)
        self.search_api = config.get("search_api", "chat_completions")
        self.request_interval = config.get("request_interval", 1.0)
        self.api_key = api_key
        self.endpoint = config.get("endpoint", "")
        self.route = "direct"

        if route == "relay" and relay_config and relay_config.get("base_url") and relay_config.get("api_key"):
            # 统一走 new-api 中继（OpenAI 兼容），混元也不再需要腾讯 SDK
            self.route = "relay"
            self.endpoint = relay_config["base_url"]
            self.api_key = relay_config["api_key"]
            self._hunyuan_client = None
            if config.get("api_type") == "tencent_cloud":
                # 混元经中继降级为 chat_completions；联网增强参数由渠道透传
                self.search_api = "chat_completions"
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.endpoint,
            )
        elif config.get("api_type") == "tencent_cloud":
            # 混元直连：腾讯云原生SDK（保留作为中继故障时的回退链路）
            self.client = None
            self._hunyuan_client = self._init_hunyuan_client(api_key, config)
        else:
            self._hunyuan_client = None
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=config.get("endpoint", ""),
            )

    def _init_hunyuan_client(self, api_key: str, config: dict):
        """初始化腾讯云混元客户端"""
        try:
            from tencentcloud.common import credential
            from tencentcloud.hunyuan.v20230901 import hunyuan_client

            # api_key格式: "secret_id:secret_key"
            parts = api_key.split(":", 1)
            if len(parts) != 2:
                raise ValueError("混元API key格式应为 'SecretId:SecretKey'")

            secret_id, secret_key = parts
            cred = credential.Credential(secret_id, secret_key)
            region = config.get("region", "ap-guangzhou")
            client = hunyuan_client.HunyuanClient(cred, region)
            return client
        except ImportError:
            raise ImportError("请安装腾讯云SDK: pip install tencentcloud-sdk-python-hunyuan")

    async def query(self, question: str, enable_search: bool = False,
                    temperature: float = 0.7, max_tokens: int = 2048,
                    json_mode: bool = False) -> dict:
        actual_search = enable_search and self.supports_search

        if self._hunyuan_client:
            return await self._query_hunyuan(question, actual_search, temperature, max_tokens)
        elif self.search_api == "responses":
            return await self._query_responses_api(question, actual_search, temperature, max_tokens)
        else:
            return await self._query_chat_completions(question, actual_search, temperature, max_tokens, json_mode)

    async def _query_hunyuan(self, question: str, enable_search: bool,
                              temperature: float, max_tokens: int) -> dict:
        """腾讯云混元API调用（同步SDK，通过线程池转异步）"""
        from tencentcloud.hunyuan.v20230901 import models as hunyuan_models

        def _sync_call():
            req = hunyuan_models.ChatCompletionsRequest()
            params = {
                "Model": self.model_id,
                "Messages": [{"Role": "user", "Content": question}],
                "Temperature": temperature,
                "Stream": False,
            }
            if enable_search:
                params["EnableEnhancement"] = True
            req.from_json_string(json.dumps(params))
            return self._hunyuan_client.ChatCompletions(req)

        start = time.time()
        try:
            response = await asyncio.to_thread(_sync_call)
            latency_ms = int((time.time() - start) * 1000)

            # 解析混元返回
            answer = ""
            sources = []
            raw = {}

            try:
                raw_str = response.to_json_string()
                raw = json.loads(raw_str)
            except Exception as e:
                logger.warning(f"[{self.model_key}] 解析返回JSON失败: {e}, response type: {type(response)}")

            # 混元返回结构可能是 Response 包裹或直接返回
            # 尝试两种路径: raw.Choices 或 raw.Response.Choices
            data = raw
            if "Response" in raw:
                data = raw["Response"]

            choices = data.get("Choices") or []
            if choices:
                message = choices[0].get("Message") or {}
                answer = message.get("Content", "")

            # 联网搜索结果
            search_info = data.get("SearchInfo") or {}
            search_results = search_info.get("SearchResults") or []
            for sr in search_results:
                sources.append({
                    "title": sr.get("Title", ""),
                    "url": sr.get("Url", ""),
                })

            return {
                "answer": answer,
                "sources": sources,
                "model": self.model_key,
                "model_name": self.name,
                "model_id": self.model_id,
                "latency_ms": latency_ms,
                "search_enabled": enable_search,
                "search_triggered": bool(search_results),
                "route": self.route,
                "client_mode": "api",
                "request_config": {
                    "api": "tencent_cloud",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                "raw_response_sha256": _response_sha256(raw),
                "raw_response": raw,
            }
        except Exception as e:
            logger.error(f"[{self.model_key}] 混元API调用失败: {e}")
            raise

    async def _query_chat_completions(self, question: str, enable_search: bool,
                                       temperature: float, max_tokens: int,
                                       json_mode: bool = False) -> dict:
        """标准Chat Completions API调用"""
        start = time.time()

        messages = [{"role": "user", "content": question}]
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

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
                "model_id": self.model_id,
                "latency_ms": latency_ms,
                "search_enabled": enable_search,
                "search_triggered": bool(sources) if enable_search else False,
                "route": self.route,
                "client_mode": "api",
                "request_config": {
                    "api": "chat_completions",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "json_mode": bool(json_mode),
                },
                "raw_response_sha256": _response_sha256(response.model_dump()),
                "raw_response": response.model_dump(),
            }
        except Exception as e:
            logger.error(f"[{self.model_key}] Chat Completions调用失败: {e}")
            raise

    async def _query_responses_api(self, question: str, enable_search: bool,
                                    temperature: float, max_tokens: int) -> dict:
        """Responses API调用（千问/豆包，联网和非联网统一走此接口）"""
        start = time.time()

        kwargs = {
            "model": self.search_model_id,
            "input": [{"role": "user", "content": question}],
        }

        # 联网时带tools，不联网时不传
        if enable_search:
            tools = [{"type": "web_search"}]
            if self.model_key == "doubao":
                tools = [{"type": "web_search", "max_keyword": 3}]
            kwargs["tools"] = tools

        try:
            response = await self.client.responses.create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)

            answer, sources = self._parse_responses_output(response)

            try:
                raw = response.model_dump()
            except Exception:
                raw = {"output_text": answer}

            # 空答案警告 + 详细调试
            if not answer:
                status = raw.get("status", "") if isinstance(raw, dict) else getattr(response, "status", "")
                error = raw.get("error", "") if isinstance(raw, dict) else getattr(response, "error", "")
                text_field = raw.get("text", "") if isinstance(raw, dict) else getattr(response, "text", "")
                output_items = raw.get("output", []) if isinstance(raw, dict) else []
                logger.warning(
                    f"[{self.model_key}] 应答为空! search={enable_search}, "
                    f"status={status}, error={error}, "
                    f"text字段={str(text_field)[:200]}, "
                    f"output长度={len(output_items)}, "
                    f"output内容={str(output_items)[:300]}"
                )
                # 尝试从text字段兜底
                if text_field and isinstance(text_field, str):
                    answer = text_field

            return {
                "answer": answer,
                "sources": sources,
                "model": self.model_key,
                "model_name": self.name,
                "model_id": self.search_model_id,
                "latency_ms": latency_ms,
                "search_enabled": enable_search,
                "search_triggered": bool(sources) if enable_search else False,
                "route": self.route,
                "client_mode": "api",
                "request_config": {
                    "api": "responses",
                    "requested_temperature": temperature,
                    "requested_max_tokens": max_tokens,
                    "tools_enabled": bool(enable_search),
                },
                "raw_response_sha256": _response_sha256(raw),
                "raw_response": raw,
            }
        except Exception as e:
            logger.error(f"[{self.model_key}] Responses API调用失败: {e}")
            raise

    def _parse_responses_output(self, response) -> tuple:
        """解析Responses API的返回"""
        answer = ""
        sources = []

        if hasattr(response, "output_text") and response.output_text:
            answer = response.output_text

        output = []
        if hasattr(response, "output"):
            output = response.output
        elif isinstance(response, dict):
            output = response.get("output", [])

        for item in output:
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else "")

            if item_type == "message":
                content = getattr(item, "content", None) or (item.get("content", []) if isinstance(item, dict) else [])
                for c in content:
                    c_type = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else "")
                    if c_type == "output_text":
                        text = getattr(c, "text", None) or (c.get("text", "") if isinstance(c, dict) else "")
                        if text:
                            answer = text
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
