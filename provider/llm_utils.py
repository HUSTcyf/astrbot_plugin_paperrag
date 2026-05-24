"""
统一 LLM 调用工具 — provider 解析、文本提取、JSON 解析、高级封装。

- get_llm_provider(): 4 步优先级 provider 解析链
- extract_text_from_response(): 从 LLM 响应提取文本
- parse_json_response(): 从文本解析 JSON（容忍尾逗号、code block 包裹）
- call_llm(): 完整 LLM 调用封装
- call_llm_json(): LLM 调用 + JSON 解析
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from astrbot.api import logger


def extract_text_from_response(response) -> str:
    """从 LLM 响应中提取文本（兼容 VLM、AstrBot cloud provider、dict）。"""
    if hasattr(response, 'content'):
        return response.content

    # AstrBot cloud provider 返回 result_chain 格式
    rc = getattr(response, 'result_chain', None)
    if rc is not None:
        chain = getattr(rc, 'chain', None)
        if chain and len(chain) > 0:
            first = chain[0]
            if hasattr(first, 'get_text'):
                return first.get_text()
            if hasattr(first, 'text'):
                return first.text

    if isinstance(response, dict):
        return response.get("content", "") or response.get("text", "")
    return str(response)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM JSON issue)."""
    for _ in range(3):
        text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def parse_json_response(text: str) -> Optional[dict]:
    """从文本中解析 JSON（支持 ```json 包裹，容忍尾逗号）"""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    text = _strip_trailing_commas(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def get_llm_provider(context, config=None):
    """
    统一的 LLM Provider 解析链（4 步优先级）。

    1. config["text_provider_id"] → inst_map[id]
    2. 本地 VLM (llama_cpp)
    3. context.get_using_provider()
    4. inst_map 中第一个有 text_chat 的

    Args:
        context: AstrBot Context 对象
        config: 可选，插件配置字典
    """
    if config is None:
        config = getattr(context, 'config', {}) if context else {}

    provider = None

    # Step 1: 用户显式配置的 text_provider_id
    text_provider_id = config.get("text_provider_id", "")
    if text_provider_id:
        pm = getattr(context, 'provider_manager', None)
        if pm:
            inst_map = getattr(pm, 'inst_map', None)
            if isinstance(inst_map, dict):
                provider = inst_map.get(text_provider_id)

    # Step 2: 本地 VLM (llama.cpp)
    if not provider:
        try:
            from provider.llama_cpp_vlm import get_llama_cpp_vlm_provider
            vlm = get_llama_cpp_vlm_provider()
            if vlm and getattr(vlm, "_initialized", False):
                provider = vlm
        except Exception as e:
            logger.debug(f"[Provider] Step 2 (local VLM) failed: {e}")

    # Step 3: 当前会话的云端 provider
    if not provider and context is not None:
        try:
            fn = getattr(context, 'get_using_provider', None)
            if fn:
                provider = fn()
        except Exception as e:
            logger.debug(f"[Provider] Step 3 (get_using_provider) failed: {e}")

    # Step 4: inst_map 中第一个有 text_chat 的 provider
    if not provider:
        pm = getattr(context, 'provider_manager', None)
        if pm:
            inst_map = getattr(pm, 'inst_map', None)
            if isinstance(inst_map, dict):
                for p in inst_map.values():
                    if callable(getattr(p, 'text_chat', None)):
                        provider = p
                        break

    # 验证返回的 provider 有 text_chat 方法
    if provider and not callable(getattr(provider, 'text_chat', None)):
        provider = None

    return provider


_UNSET = object()


async def call_llm(
    prompt: str,
    context: Any,
    config: Optional[dict] = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs,
) -> str:
    """
    完整 LLM 调用：解析 provider → text_chat → 提取文本。

    max_tokens 优先级：调用方显式传入 > config 配置 > 不传（由 provider 自行决定）。

    云端大模型不传 max_tokens，让其使用 provider 默认值（通常足够大）。
    本地 VLM 调用方应显式传入合适的 max_tokens。

    Returns:
        LLM 响应文本
    Raises:
        RuntimeError: 无可用 provider
        Exception: text_chat 调用失败
    """
    provider = get_llm_provider(context, config)
    if provider is None:
        raise RuntimeError("无可用 LLM provider")

    eff_temp = temperature
    eff_tokens = max_tokens
    if config:
        if temperature == 0.7 and "text_llm_temperature" in config:
            eff_temp = config["text_llm_temperature"]
        if max_tokens is None and "text_llm_max_tokens" in config:
            eff_tokens = config["text_llm_max_tokens"]

    chat_kwargs: dict = {
        "prompt": prompt,
        "contexts": [],
        "temperature": eff_temp,
        **kwargs,
    }
    if eff_tokens is not None:
        chat_kwargs["max_tokens"] = eff_tokens

    response = await provider.text_chat(**chat_kwargs)
    return extract_text_from_response(response)


async def call_llm_json(
    prompt: str,
    context: Any,
    config: Optional[dict] = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    **kwargs,
) -> Optional[dict]:
    """
    LLM 调用 + JSON 解析。

    max_tokens 优先级：调用方显式传入 > config 配置 > 不传（由 provider 自行决定）。

    Returns:
        解析后的 dict，或 None（解析失败）
    Raises:
        RuntimeError: 无可用 provider
    """
    text = await call_llm(prompt, context, config, temperature, max_tokens, **kwargs)
    return parse_json_response(text)


# ============================================================================
# LlamaIndex LLM Bridge
# ============================================================================

def _create_vlm_custom_llm(vlm_provider):
    """Create a LlamaIndex CustomLLM that wraps LlamaCppVLMProvider.

    Uses llama_index.core.llms.CustomLLM base class so that Pydantic
    validation in LlamaIndex components (SimpleLLMPathExtractor, etc.)
    accepts the instance.  Avoids llama_index.llms.openai.OpenAI which
    validates model names against OpenAI's known list and rejects 'local-vlm'.
    """
    from llama_index.core.base.llms.generic_utils import (
        completion_response_to_chat_response,
    )
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from typing import Sequence
    import asyncio

    class _Impl(CustomLLM):
        model_name: str = "local-vlm"
        _ctx_window: int = getattr(vlm_provider, 'n_ctx', 16384)

        class Config:
            arbitrary_types_allowed = True

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                model_name=self.model_name,
                context_window=self._ctx_window,
                is_chat_model=True,
            )

        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
            raise NotImplementedError

        def complete(self, prompt: str, formatted: bool = False, **kwargs):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.acomplete(prompt, formatted, **kwargs))
            raise RuntimeError(
                "complete() called inside running event loop — use acomplete() or achat() instead"
            )

        async def acomplete(self, prompt: str, formatted: bool = False, **kwargs):
            resp = await vlm_provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 256),
            )
            return CompletionResponse(
                text=resp.content if hasattr(resp, "content") else str(resp)
            )

        async def achat(
            self,
            messages: Sequence[ChatMessage],
            **kwargs: Any,
        ) -> ChatResponse:
            assert self.messages_to_prompt is not None
            prompt = self.messages_to_prompt(messages)
            completion_response = await self.acomplete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion_response)

    return _Impl()


async def get_llama_index_llm(context: Any = None, prefer_cloud: bool = False):
    """
    Create a LlamaIndex-compatible LLM from the provider resolution chain.

    Priority:
    1. Local VLM via _VLMCustomLLM (bypasses OpenAI SDK model validation)
    2. Cloud provider via llama_index.llms.openai.OpenAI

    Returns:
        LlamaIndex LLM object, or None if no provider available.
    """
    # Step 1: Local VLM (CustomLLM — no model name validation)
    if not prefer_cloud:
        try:
            from provider.llama_cpp_vlm import get_llama_cpp_vlm_provider
            vlm = get_llama_cpp_vlm_provider()
            if vlm and not getattr(vlm, '_initialized', False):
                await vlm.initialize()
            if vlm and getattr(vlm, '_initialized', False):
                return _create_vlm_custom_llm(vlm)
        except Exception as e:
            logger.debug(f"[Provider] LlamaIndex Step 1 (local VLM) failed: {e}")

    # Step 2: Cloud provider
    from astrbot.api import logger
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError:
        logger.warning("[Provider] llama_index.llms.openai 不可用")
        return None

    if context is None:
        return None

    provider = None
    pm = getattr(context, 'provider_manager', None)
    inst_map = getattr(pm, 'inst_map', None) if pm else None

    # Try context.get_using_provider()
    if not provider:
        try:
            provider = context.get_using_provider()
        except Exception as e:
            logger.debug(f"[Provider] LlamaIndex get_using_provider failed: {e}")

    # Fallback: first provider with text_chat from inst_map
    if not provider and isinstance(inst_map, dict):
        for p in inst_map.values():
            if callable(getattr(p, 'text_chat', None)):
                provider = p
                break

    if not provider:
        return None

    model = getattr(provider, 'model_name', '') or getattr(provider, 'provider_config', {}).get('model', '')
    if not model:
        return None

    api_key = getattr(provider, 'chosen_api_key', None)
    if not api_key:
        try:
            api_key = provider.get_current_key()
        except Exception as e:
            logger.debug(f"[Provider] get_current_key failed: {e}")
    if not api_key:
        return None

    api_base = getattr(provider, 'provider_config', {}).get('api_base', '')
    kwargs = {"model": model, "api_key": api_key}
    if api_base:
        kwargs["api_base"] = api_base
    logger.info(f"[Provider] 创建 LlamaIndex LLM: model={model}")
    return OpenAI(**kwargs)
