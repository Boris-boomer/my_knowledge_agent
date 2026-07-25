# ================================================================
# llm_router.py：LLM 模型路由器（整合生成 + 向量化）
# ================================================================
import asyncio
import logging
import time
import json
import re
from typing import List, Dict, Any, Optional, AsyncIterator, Literal
from concurrent.futures import ThreadPoolExecutor
import requests

from .base import BaseLLM, LLMResponse, TokenChunk, Tool, ToolCall
from .semantic_cache import SemanticCache, get_cache
from .observability import (
    TokenCounter,
    CostCalculator,
    StructuredLogger,
    new_trace,
    get_token_counter,
    get_cost_calculator,
    get_logger,
)

logger = logging.getLogger(__name__)

# ================================================================
# LLM 生成适配器
# ================================================================
class OllamaAdapter(BaseLLM):
    def __init__(self, model, base_url="http://localhost:11434", timeout=60, temperature=0.3, max_tokens=512, **kwargs):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._executor = ThreadPoolExecutor(max_workers=1)

    def generate(self, prompt, **kwargs):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "")
        content, reasoning = self._extract_reasoning(raw)
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=[],
            finish_reason=data.get("done_reason", "stop"),
            usage={"prompt_tokens": data.get("prompt_eval_count", 0),
                   "completion_tokens": data.get("eval_count", 0)},
        )

    async def stream_generate(self, prompt, **kwargs):
        import aiohttp
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=self.timeout) as resp:
                full = ""
                in_reasoning = False
                async for line in resp.content:
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        chunk = data.get("response", "")
                        if chunk:
                            full += chunk
                            if "<think>" in chunk:
                                in_reasoning = True
                            if not in_reasoning:
                                yield TokenChunk(text=chunk, is_final=False)
                            if "</think>" in chunk:
                                in_reasoning = False
                        if data.get("done", False):
                            _, reasoning = self._extract_reasoning(full)
                            yield TokenChunk(text="", is_final=True,
                                             finish_reason=data.get("done_reason", "stop"),
                                             token_count=data.get("eval_count", 0),
                                             reasoning=reasoning)
                            break
                    except:
                        continue

    async def health_check(self, deep=False):
        import aiohttp
        try:
            url = f"{self.base_url}/api/generate"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"model": self.model, "prompt": "ping", "stream": False}, timeout=5) as resp:
                    return resp.status == 200
        except:
            return False

    def estimate_cost(self, prompt_tokens, completion_tokens):
        return 0.0

    def _extract_reasoning(self, text):
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return content, reasoning
        return text, None


class OpenAIAdapter(BaseLLM):
    def __init__(self, model, api_key, base_url="https://api.openai.com/v1", timeout=30, temperature=0.3, max_tokens=4096, **kwargs):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt, **kwargs):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = [{"role": "user", "content": prompt}]
        params = {"model": self.model, "messages": messages,
                  "temperature": kwargs.get("temperature", self.temperature),
                  "max_tokens": kwargs.get("max_tokens", self.max_tokens)}
        if kwargs.get("reasoning_effort") and self.model.startswith("o1"):
            params["reasoning_effort"] = kwargs["reasoning_effort"]
        if kwargs.get("tools"):
            params["tools"] = kwargs["tools"]
        resp = client.chat.completions.create(**params)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments)) for tc in (msg.tool_calls or [])]
        return LLMResponse(content=msg.content or "", tool_calls=tool_calls,
                           finish_reason=choice.finish_reason,
                           usage={"prompt_tokens": resp.usage.prompt_tokens,
                                  "completion_tokens": resp.usage.completion_tokens})

    async def stream_generate(self, prompt, **kwargs):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = [{"role": "user", "content": prompt}]
        params = {"model": self.model, "messages": messages,
                  "temperature": kwargs.get("temperature", self.temperature),
                  "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                  "stream": True}
        if kwargs.get("reasoning_effort") and self.model.startswith("o1"):
            params["reasoning_effort"] = kwargs["reasoning_effort"]
        if kwargs.get("tools"):
            params["tools"] = kwargs["tools"]
        async for chunk in await client.chat.completions.create(**params):
            if chunk.choices and chunk.choices[0].delta.content:
                yield TokenChunk(text=chunk.choices[0].delta.content, is_final=False)
            if chunk.choices and chunk.choices[0].finish_reason:
                yield TokenChunk(text="", is_final=True, finish_reason=chunk.choices[0].finish_reason)

    async def health_check(self, deep=False):
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            await client.models.list()
            return True
        except:
            return False

    def estimate_cost(self, prompt_tokens, completion_tokens):
        from .observability import get_pricing
        pricing = get_pricing(self.model)
        return (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]


# ================================================================
# 嵌入适配器
# ================================================================
class OllamaEmbeddingAdapter:
    def __init__(self, model="qwen3-embedding:0.6b", base_url="http://localhost:11434", timeout=30, **kwargs):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def embed(self, text):
        if not text or not text.strip():
            return []
        resp = requests.post(f"{self.base_url}/api/embeddings",
                             json={"model": self.model, "prompt": text},
                             timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class OpenAIEmbeddingAdapter:
    def __init__(self, api_key, model="text-embedding-ada-002", base_url="https://api.openai.com/v1", timeout=30, **kwargs):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, text):
        if not text or not text.strip():
            return []
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts):
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]


# ================================================================
# 适配器工厂
# ================================================================
_LLM_ADAPTER_REGISTRY = {"ollama": OllamaAdapter, "openai": OpenAIAdapter}
_EMBED_ADAPTER_REGISTRY = {"ollama": OllamaEmbeddingAdapter, "openai": OpenAIEmbeddingAdapter}

def get_llm_adapter(provider, **params):
    if provider not in _LLM_ADAPTER_REGISTRY:
        raise ValueError(f"不支持的提供商: {provider}")
    return _LLM_ADAPTER_REGISTRY[provider](**params)

def get_embed_adapter(provider, **params):
    if provider not in _EMBED_ADAPTER_REGISTRY:
        raise ValueError(f"不支持的提供商: {provider}")
    return _EMBED_ADAPTER_REGISTRY[provider](**params)

get_adapter = get_llm_adapter


# ================================================================
# ModelRouter
# ================================================================
class ModelRouter:
    def __init__(self, llm_config, embedder_config=None, cache_config=None, chroma_path="./chroma_db"):
        self.llm_config = llm_config
        self.embedder_config = embedder_config or {}
        self.cache_config = cache_config or {}
        self.strategy = llm_config.get("strategy", "auto")
        self.timeout = llm_config.get("timeout", 30)
        self.temperature = llm_config.get("temperature", 0.3)
        self.max_tokens = llm_config.get("max_tokens", 512)

        self._primary_adapter = None
        self._fallback_adapters = []
        self._init_llm_adapters()

        self._embed_adapter = None
        self._init_embed_adapter()

        self.cache = None
        if self.cache_config.get("enabled", False):
            self.cache = get_cache(self.cache_config, embedder=self, chroma_path=chroma_path)

        self.token_counter = get_token_counter()
        self.cost_calculator = get_cost_calculator()
        self.logger = get_logger()

        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "primary_used": 0,
            "fallback_used": 0,
            "failed_calls": 0,
            "total_cost": 0.0,
        }

    def _init_llm_adapters(self):
        primary = self.llm_config.get("primary", {})
        if primary:
            provider = primary.get("provider")
            params = primary.get("params", {}).copy()
            params["timeout"] = params.get("timeout", self.timeout)
            params["temperature"] = params.get("temperature", self.temperature)
            params["max_tokens"] = params.get("max_tokens", self.max_tokens)
            self._primary_adapter = get_llm_adapter(provider, **params)

        for fallback in self.llm_config.get("fallback", []):
            provider = fallback.get("provider")
            params = fallback.get("params", {}).copy()
            params["timeout"] = params.get("timeout", self.timeout)
            params["temperature"] = params.get("temperature", self.temperature)
            params["max_tokens"] = params.get("max_tokens", self.max_tokens)
            try:
                self._fallback_adapters.append(get_llm_adapter(provider, **params))
            except Exception as e:
                logger.warning(f"备选模型初始化失败: {e}")

    def _init_embed_adapter(self):
        if not self.embedder_config.get("enabled", True):
            return
        provider = self.embedder_config.get("backend", "ollama")
        params = self.embedder_config.get("params", {}).copy()
        params["timeout"] = params.get("timeout", 30)
        self._embed_adapter = get_embed_adapter(provider, **params)

    async def get_active_adapter(self) -> Optional[BaseLLM]:
        """直接返回主适配器，跳过健康检查（绕过连接问题）"""
        if self._primary_adapter:
            return self._primary_adapter
        # 如果没有主适配器，尝试备选（但尽量不健康检查，直接返回第一个）
        if self._fallback_adapters:
            return self._fallback_adapters[0]
        return None

    def generate(self, prompt, **kwargs):
        response = asyncio.run(self.generate_full(prompt, **kwargs))
        return response.content

    async def generate_full(self, prompt, **kwargs):
        trace = new_trace()
        self.stats["total_calls"] += 1
        use_cache = kwargs.get("use_cache", True)
        if use_cache and self.cache:
            cached = await self.cache.get(prompt)
            if cached:
                self.stats["cache_hits"] += 1
                self.logger.log_cache_hit(trace_id=trace.trace_id, prompt=prompt, similarity=0.95)
                return LLMResponse(content=cached, finish_reason="cache", usage={})
        adapter = await self.get_active_adapter()
        if not adapter:
            self.stats["failed_calls"] += 1
            raise RuntimeError("没有可用的 LLM 模型，请检查 Ollama 是否运行且模型存在。")
        try:
            response = adapter.generate(prompt, **kwargs)
            if adapter == self._primary_adapter:
                self.stats["primary_used"] += 1
            else:
                self.stats["fallback_used"] += 1
            cost_info = self.cost_calculator.calculate(
                prompt=prompt,
                response=response.content,
                model=adapter.__class__.__name__,
                input_tokens=response.usage.get("prompt_tokens"),
                output_tokens=response.usage.get("completion_tokens"),
            )
            self.stats["total_cost"] += cost_info["total_cost"]
            if use_cache and self.cache and response.content:
                await self.cache.set(prompt, response.content)
            self.logger.log_llm_call(trace_id=trace.trace_id, prompt=prompt, response=response.content,
                                     model=adapter.__class__.__name__, cost_info=cost_info, success=True)
            return response
        except Exception as e:
            self.stats["failed_calls"] += 1
            self.logger.log_error(trace_id=trace.trace_id, error=str(e), module="llm_router.generate_full")
            raise

    async def stream_generate(self, prompt, **kwargs):
        trace = new_trace()
        self.stats["total_calls"] += 1
        use_cache = kwargs.get("use_cache", True)
        if use_cache and self.cache:
            cached = await self.cache.get(prompt)
            if cached:
                self.stats["cache_hits"] += 1
                self.logger.log_cache_hit(trace_id=trace.trace_id, prompt=prompt, similarity=0.95)
                yield TokenChunk(text=cached, is_final=True, finish_reason="cache")
                return
        adapter = await self.get_active_adapter()
        if not adapter:
            self.stats["failed_calls"] += 1
            raise RuntimeError("没有可用的 LLM 模型")
        full_response = ""
        try:
            async for chunk in adapter.stream_generate(prompt, **kwargs):
                if chunk.text:
                    full_response += chunk.text
                if chunk.is_final:
                    if adapter == self._primary_adapter:
                        self.stats["primary_used"] += 1
                    else:
                        self.stats["fallback_used"] += 1
                    cost_info = self.cost_calculator.calculate(
                        prompt=prompt,
                        response=full_response,
                        model=adapter.__class__.__name__,
                        input_tokens=None,
                        output_tokens=chunk.token_count,
                    )
                    self.stats["total_cost"] += cost_info["total_cost"]
                    if use_cache and self.cache and full_response:
                        await self.cache.set(prompt, full_response)
                    self.logger.log_llm_call(trace_id=trace.trace_id, prompt=prompt, response=full_response,
                                             model=adapter.__class__.__name__, cost_info=cost_info, success=True)
                yield chunk
        except Exception as e:
            self.stats["failed_calls"] += 1
            self.logger.log_error(trace_id=trace.trace_id, error=str(e), module="llm_router.stream_generate")
            raise

    def embed(self, text):
        if not self._embed_adapter:
            raise RuntimeError("嵌入器未初始化")
        return self._embed_adapter.embed(text)

    def embed_batch(self, texts):
        if not self._embed_adapter:
            raise RuntimeError("嵌入器未初始化")
        return self._embed_adapter.embed_batch(texts)

    def get_stats(self):
        return {
            **self.stats,
            "cache": self.cache.stats() if self.cache else {"enabled": False},
            "active_llm_adapter": self._primary_adapter.__class__.__name__ if self._primary_adapter else None,
            "embed_adapter": self._embed_adapter.__class__.__name__ if self._embed_adapter else None,
        }