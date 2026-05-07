"""
Vector Search node — 向量检索（复用 HybridRAGEngine）。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field, field_validator, model_validator

from astrbot.api import logger


class VectorSearchInput(BaseModel):
    """VectorSearch 节点输入。"""
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query cannot be empty or whitespace")
        return v


class VectorSearchOutput(BaseModel):
    """VectorSearch 节点输出。"""
    retrieved_nodes: list[dict] = Field(default_factory=list)
    search_successful: bool = False

    @model_validator(mode="after")
    def nodes_not_none(self) -> "VectorSearchOutput":
        if self.retrieved_nodes is None:
            logger.warning("[vector_search] retrieved_nodes 为 None，降级为空列表")
            self.retrieved_nodes = []
        return self


SEARCH_TIMEOUT = 0  # 0 = 禁用超时


async def vector_search_node(state: dict) -> dict:
    """
    LangGraph 节点：向量检索。

    调用 HybridRAGEngine.search(mode="retrieve") 获取检索结果。

    Args:
        state: AgenticRAGState（读取 query, _context）

    Returns:
        更新 state 的 dict（retrieved_nodes, steps）
    """
    try:
        input_data = VectorSearchInput(
            query=state["query"],
            top_k=state.get("top_k", 5),
        )
    except ValueError:
        raise

    query = input_data.query
    top_k = input_data.top_k

    logger.debug(f"[vector_search] 检索: query={query}, top_k={top_k}")

    retrieved_nodes: list[dict] = []
    search_successful = False

    context = state.get("_context")
    if context is None:
        logger.error("[vector_search] _context 未传入，无法获取引擎")
        return {
            "retrieved_nodes": [],
            "steps": ["vector_search: FAILED (no context)"],
        }

    try:
        # 获取 engine
        from ..engine_utils import get_engine
        config = state.get("_config")
        engine = get_engine(context, config)
        if engine is None:
            logger.error("[vector_search] HybridRAGEngine 未就绪")
            return {
                "retrieved_nodes": [],
                "steps": ["vector_search: FAILED (engine not ready)"],
            }

        # 带超时调用（SEARCH_TIMEOUT=0 表示禁用超时）
        try:
            if SEARCH_TIMEOUT > 0:
                result = await asyncio.wait_for(
                    engine.search(query, mode="retrieve", top_k=top_k),
                    timeout=SEARCH_TIMEOUT,
                )
            else:
                result = await engine.search(query, mode="retrieve", top_k=top_k)
        except asyncio.TimeoutError:
            logger.error(f"[vector_search] 检索超时（>{SEARCH_TIMEOUT}s）")
            return {
                "retrieved_nodes": [],
                "steps": [f"vector_search: TIMEOUT ({SEARCH_TIMEOUT}s)"],
            }

        # 解析 QueryResult → list[dict]
        if result is None:
            logger.warning("[vector_search] engine.search 返回 None")
            retrieved_nodes = []
        elif hasattr(result, "nodes"):
            # QueryResult 风格
            for i, node in enumerate(result.nodes):
                node_dict = {
                    "text": getattr(node, "text", ""),
                    "score": result.scores[i] if i < len(result.scores) else 1.0,
                    "metadata": getattr(node, "metadata", {}),
                }
                retrieved_nodes.append(node_dict)
            search_successful = True
        elif isinstance(result, list):
            # 直接 list[dict] 风格
            retrieved_nodes = result
            search_successful = True
        elif isinstance(result, dict):
            # dict 风格（某些封装）
            nodes = result.get("nodes", []) or result.get("results", [])
            if isinstance(nodes, list):
                for item in nodes:
                    if isinstance(item, dict):
                        retrieved_nodes.append(item)
                    elif hasattr(item, "text"):
                        retrieved_nodes.append({
                            "text": getattr(item, "text", ""),
                            "score": getattr(item, "score", 1.0),
                            "metadata": getattr(item, "metadata", {}),
                        })
            search_successful = True
        else:
            logger.warning(f"[vector_search] 未知的 result 类型: {type(result)}")
            retrieved_nodes = []

        logger.info(f"[vector_search] 检索成功: {len(retrieved_nodes)} 条结果")

    except Exception as e:
        logger.error(f"[vector_search] 检索失败: {e}")
        retrieved_nodes = []
        search_successful = False

    output = VectorSearchOutput(
        retrieved_nodes=retrieved_nodes,
        search_successful=search_successful,
    )

    status = "OK" if search_successful else "FAILED"
    return {
        "retrieved_nodes": output.retrieved_nodes,
        "steps": [f"vector_search: {status} ({len(output.retrieved_nodes)} nodes)"],
    }
