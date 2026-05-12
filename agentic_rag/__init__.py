"""
agentic_rag — Agentic RAG workflow using LangGraph.
"""

from __future__ import annotations

from typing import Optional, Any

from astrbot.api import logger
from .workflow import compile_workflow
from .react_workflow import compile_react_workflow
from .knowledge_extractor import run_knowledge_extraction

__all__ = [
    "compile_workflow", "run_agentic_rag", "run_agentic_rag_stream", "run_react_rag",
    "run_knowledge_extraction",
]
__version__ = "0.2.0"


async def run_agentic_rag(query: str, context, top_k: int = 5, config: Optional[dict[str, Any]] = None) -> str:
    """
    主要入口函数（非流式）。

    Args:
        query: 用户查询
        context: AstrBot Context 对象
        top_k: 召回数（默认5）
        config: 可选，直接传入插件配置字典（优先于 context.config）
    """
    app = compile_workflow()
    initial_state = {"query": query, "_context": context, "_config": config, "top_k": top_k, "steps": []}
    result = await app.ainvoke(initial_state)
    return result.get("final_answer", "")


async def run_agentic_rag_stream(query: str, context, top_k: int = 5, config: Optional[dict[str, Any]] = None):
    """
    流式版本入口函数。

    Yields:
        每步的事件 dict
    """
    app = compile_workflow()
    initial_state = {"query": query, "_context": context, "_config": config, "top_k": top_k, "steps": []}
    async for event in app.astream(initial_state, stream="updates"):
        yield event


async def run_react_rag(query: str, context, top_k: int = 5, config: Optional[dict[str, Any]] = None) -> str:
    """
    Tool-Using Agent 入口函数（ReAct 模式）。

    Args:
        query: 用户查询
        context: AstrBot Context 对象
        top_k: 召回数（默认5）
        config: 可选，直接传入插件配置字典
    """
    app = compile_react_workflow()
    initial_state = {
        "query": query,
        "scratchpad": "",
        "_context": context,
        "_config": config,
        "top_k": top_k,
        "steps": [],
    }
    result = await app.ainvoke(initial_state)
    return result.get("final_answer", "")
