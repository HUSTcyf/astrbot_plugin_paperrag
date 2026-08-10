# -*- coding: utf-8 -*-
"""MiniMax 端点专属兼容逻辑（与 ragas 核心分离）。

为什么单独一个文件：
  MiniMax-M3（Token Plan）与标准 OpenAI 兼容端点（智谱 GLM 等）有 3 处差异，
  上一会话把这些差异直接写进 ragas_generator.py，堆成"补丁山"。本模块把 MiniMax
  专属逻辑集中，ragas_generator/ragas_evaluator 只需调用这里的 helper，自身保持
  标准 OpenAI 逻辑。智谱等标准端点完全不走本模块的任何分支。

三处 MiniMax 专属差异：
  1. 思考模式：MiniMax-M3 默认输出 <think> 块，需在请求体加 thinking:disabled
  2. embedding 格式非标准：请求用 texts/type（非 input），响应用 vectors（非 data[].embedding）
  3. embedding 模型名不同：MiniMax 用 embo-01（非 text-embedding-v4 等标准命名）

不属于本模块的（留在 ragas_generator.py，因为对任何 LLM 都需要）：
  - _normalize_json_response（<think> 剥离 + JSON 修复）——通用防御
  - 7 个 extractor monkey-patch + Executor.results 过滤——ragas 0.4.3 run_async_tasks bug
  - 429 退避——通用限流处理
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def is_minimax_endpoint(api_base: str) -> bool:
    """判断 API 端点是否为 MiniMax（按 URL 关键字）。"""
    return "minimax" in (api_base or "").lower()


def needs_thinking_disabled(api_base: str) -> bool:
    """MiniMax-M3 思考模式会输出 <think> 块破坏 JSON 解析，需显式禁用。

    智谱 GLM 等标准端点无思考模式，返回 False（不发 thinking 字段）。
    """
    return is_minimax_endpoint(api_base)


def build_llm_request_fields(api_base: str) -> Dict[str, Any]:
    """返回需追加到 chat/completions 请求体的 MiniMax 专属字段。

    智谱等标准端点返回空 dict（请求体保持纯净）。
    MiniMax 返回 thinking:disabled + response_format:json_object。
    """
    if not needs_thinking_disabled(api_base):
        return {}
    return {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }


def apply_llm_request_fields(kwargs: Dict[str, Any], api_base: str) -> None:
    """把 MiniMax 专属请求字段合并进 openai SDK 的 chat.completions kwargs。

    openai SDK（>=2.x）不接受 thinking 作为顶层关键字参数（Completions.create() 会
    抛 TypeError），必须经 extra_body 透传；response_format 是 SDK 支持的顶层参数。
    标准端点（智谱等）build_llm_request_fields 返回空 dict，本函数不做任何改动。

    Args:
        kwargs: 传给 client.chat.completions.create(**kwargs) 的参数字典（原地修改）
        api_base: 端点 base_url，用于判断是否 MiniMax
    """
    for k, v in build_llm_request_fields(api_base).items():
        if k == "thinking":
            kwargs.setdefault("extra_body", {})["thinking"] = v
        else:
            kwargs[k] = v


def resolve_embedding_model(api_base: str, default_model: str) -> str:
    """按端点解析正确的 embedding 模型名。

    不同平台的 embedding 模型命名不同：
    - MiniMax：embo-01（非标准 OpenAI 命名）
    - 智谱 BigModel：embedding-3（非 text-embedding-v4，后者是阿里通义命名）
    - 其他标准端点：原样返回 default_model

    仅当用户未显式指定（仍是占位默认值）时自动切换，避免覆盖用户显式选择。
    """
    _placeholder_defaults = ("text-embedding-v4", "text-embedding-3-small", "text-embedding-ada-002")
    if default_model not in _placeholder_defaults:
        return default_model  # 用户显式指定了，尊重
    if is_minimax_endpoint(api_base):
        return "embo-01"
    if "bigmodel.cn" in api_base or "paas/v4" in api_base:
        return "embedding-3"  # 智谱
    return default_model


def build_embedding_request_data(api_base: str, model: str, texts: List[str]) -> Dict[str, Any]:
    """构造 /v1/embeddings 请求体。

    MiniMax：{model, texts, type:"query"}（非标准）
    标准：  {model, input: texts}
    """
    if is_minimax_endpoint(api_base):
        return {"model": model, "texts": texts, "type": "query"}
    return {"model": model, "input": texts}


def extract_embedding_vectors(api_base: str, result: Dict[str, Any]) -> List[List[float]]:
    """从 /v1/embeddings 响应提取向量列表。

    MiniMax：result["vectors"]（直接是数组的数组，无 index）
    标准：  result["data"][].embedding（按 index 排序）
    """
    if is_minimax_endpoint(api_base):
        vectors = result.get("vectors") or []
        if not isinstance(vectors, list) or not vectors:
            raise RuntimeError(f"MiniMax embeddings 返回空 vectors: {str(result)[:200]}")
        return [v if isinstance(v, list) else list(v) for v in vectors]
    embeddings = result["data"]
    embeddings.sort(key=lambda x: x["index"])
    return [e["embedding"] for e in embeddings]
