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
    # AstrBot cloud provider: result_chain 格式（优先，避免 .content 误判）
    rc = getattr(response, 'result_chain', None)
    if rc is not None and hasattr(rc, 'chain'):
        chain = rc.chain
        if chain and len(chain) > 0:
            first = chain[0]
            text = getattr(first, 'text', '') or getattr(first, 'get_text', lambda: '')()
            if text:
                return text

    # VLM / cloud raw response with .content
    content = getattr(response, 'content', None)
    if content is not None and isinstance(content, str) and content.strip():
        return content

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
    2. context.get_using_provider()（当前会话云端）
    3. inst_map 中第一个有 text_chat 的
    4. 本地 VLM (llama_cpp) — 最后兜底
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

    # Step 2: 当前会话的云端 provider（优先于本地 VLM）
    if not provider and context is not None:
        try:
            fn = getattr(context, 'get_using_provider', None)
            if fn:
                provider = fn()
        except Exception as e:
            logger.debug(f"[Provider] Step 2 (get_using_provider) failed: {e}")

    # Step 3: inst_map 中第一个有 text_chat 的 provider
    if not provider:
        pm = getattr(context, 'provider_manager', None)
        if pm:
            inst_map = getattr(pm, 'inst_map', None)
            if isinstance(inst_map, dict):
                for p in inst_map.values():
                    if callable(getattr(p, 'text_chat', None)):
                        provider = p
                        break

    # Step 4: 本地 VLM (llama.cpp) — 最后的兜底
    if not provider:
        try:
            from provider.llama_cpp_vlm import get_llama_cpp_vlm_provider
            logger.warning("[Provider] ⬇️ 无可用的云端 Provider，回退到本地 VLM")
            vlm = get_llama_cpp_vlm_provider()
            if vlm and getattr(vlm, "_initialized", False):
                provider = vlm
        except Exception as e:
            logger.debug(f"[Provider] Step 4 (local VLM) failed: {e}")

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

def _create_custom_llm(provider, model_name: str, ctx_window: int):
    """创建统一的 LlamaIndex CustomLLM 包装器，委托 text_chat 给任意 provider。

    Args:
        provider: 任何有 text_chat(prompt, contexts, temperature, max_tokens) 的 provider
        model_name: 模型名（用于 metadata）
        ctx_window: 上下文窗口大小
    """
    from llama_index.core.base.llms.generic_utils import (
        completion_response_to_chat_response,
    )
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from typing import Sequence
    import asyncio

    _name = model_name
    _ctx = ctx_window

    class _Impl(CustomLLM):
        class Config:
            arbitrary_types_allowed = True

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                model_name=_name,
                context_window=_ctx,
                is_chat_model=True,
            )

        @property
        def model_name(self) -> str:  # type: ignore[override]
            return _name

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
            resp = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 256),
            )
            return CompletionResponse(
                text=extract_text_from_response(resp)
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
    1. Cloud provider via _create_custom_llm (always tried first)
    2. Local VLM via _create_custom_llm (fallback when prefer_cloud=False)

    Returns:
        LlamaIndex LLM object, or None if no provider available.
    """
    from astrbot.api import logger

    # Step 1: Cloud provider (always preferred)
    if context is not None:
        provider = None
        pm = getattr(context, 'provider_manager', None)
        inst_map = getattr(pm, 'inst_map', None) if pm else None

        if not provider:
            try:
                provider = context.get_using_provider()
            except Exception as e:
                logger.debug(f"[Provider] LlamaIndex get_using_provider failed: {e}")

        if not provider and isinstance(inst_map, dict):
            for p in inst_map.values():
                if callable(getattr(p, 'text_chat', None)):
                    provider = p
                    break

        if provider:
            model = getattr(provider, 'model_name', '') or getattr(provider, 'provider_config', {}).get('model', '')
            ctx_window = getattr(provider, 'provider_config', {}).get('max_context_tokens', 204800)
            if model:
                logger.info(f"[Provider] ✅ LlamaIndex LLM → 云端: model={model}")
                return _create_custom_llm(provider, model, int(ctx_window))
            else:
                logger.warning(
                    f"[Provider] ⚠️ 云端 Provider 缺少 model 名，回退 VLM"
                )
        elif context is not None:
            logger.warning("[Provider] ⚠️ 未找到云端 Provider，回退 VLM")

    # Step 2: Local VLM (only when prefer_cloud=False)
    if not prefer_cloud:
        try:
            from provider.llama_cpp_vlm import get_llama_cpp_vlm_provider
            logger.warning("[Provider] ⬇️ 回退到本地 VLM")
            vlm = get_llama_cpp_vlm_provider()
            if vlm and not getattr(vlm, '_initialized', False):
                await vlm.initialize()
            if vlm and getattr(vlm, '_initialized', False):
                ctx_window = getattr(vlm, 'n_ctx', 16384)
                return _create_custom_llm(vlm, "local-vlm", ctx_window)
        except Exception as e:
            logger.debug(f"[Provider] LlamaIndex VLM fallback failed: {e}")

    return None
