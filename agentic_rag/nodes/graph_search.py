"""
Graph Search node — 图谱检索（复用 GraphRAGEngine）。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field, field_validator

from astrbot.api import logger


class GraphSearchInput(BaseModel):
    """GraphSearch 节点输入。"""
    query: str = Field(..., min_length=1)
    graph_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query cannot be empty or whitespace")
        return v


class GraphSearchOutput(BaseModel):
    """GraphSearch 节点输出。"""
    graph_entities: list[dict] = Field(default_factory=list)
    graph_relations: list[dict] = Field(default_factory=list)
    retrieved_nodes: list[dict] = Field(default_factory=list)
    search_successful: bool = False


GRAPH_SEARCH_TIMEOUT = 0  # 0 = 禁用超时


async def graph_search_node(state: dict) -> dict:
    """
    LangGraph 节点：图谱检索。

    调用 GraphRAGEngine.search(mode="hybrid") 获取实体、关系和文本上下文。

    Args:
        state: AgenticRAGState（读取 query, graph_weight, _context）

    Returns:
        更新 state 的 dict（graph_entities, graph_relations, retrieved_nodes, steps）
    """
    try:
        input_data = GraphSearchInput(
            query=state["query"],
            graph_weight=state.get("graph_weight", 0.3),
        )
    except ValueError:
        raise

    query = input_data.query
    graph_weight = input_data.graph_weight

    # graph_weight=0 表示该查询类型不需要图谱，快速跳过
    if graph_weight == 0.0:
        logger.debug(f"[graph_search] graph_weight=0，跳过图谱检索")
        return {
            "graph_entities": [],
            "graph_relations": [],
            "retrieved_nodes": [],
            "steps": ["graph_search: SKIPPED (graph_weight=0)"],
        }

    logger.debug(f"[graph_search] 开始: query={query}, graph_weight={graph_weight}")

    entities: list[dict] = []
    relations: list[dict] = []
    retrieved_nodes: list[dict] = []
    search_successful = False

    context = state.get("_context")
    if context is None:
        logger.warning("[graph_search] _context 未传入，跳过图谱检索")
        return {
            "graph_entities": [],
            "graph_relations": [],
            "retrieved_nodes": [],
            "steps": ["graph_search: SKIPPED (no context)"],
        }

    try:
        from ..engine_utils import get_graph_engine
        config = state.get("_config")
        graph_engine = await get_graph_engine(context, config)

        if graph_engine is None:
            logger.warning("[graph_search] GraphRAGEngine 未就绪，跳过图谱检索")
            return {
                "graph_entities": [],
                "graph_relations": [],
                "retrieved_nodes": [],
                "steps": ["graph_search: SKIPPED (engine not ready)"],
            }

        # 带超时调用（GRAPH_SEARCH_TIMEOUT=0 表示禁用超时）
        try:
            if GRAPH_SEARCH_TIMEOUT > 0:
                result = await asyncio.wait_for(
                    graph_engine.search(query, mode="hybrid", top_k=5),
                    timeout=GRAPH_SEARCH_TIMEOUT,
                )
            else:
                result = await graph_engine.search(query, mode="hybrid", top_k=5)
        except asyncio.TimeoutError:
            logger.error(f"[graph_search] 检索超时（>{GRAPH_SEARCH_TIMEOUT}s）")
            return {
                "graph_entities": [],
                "graph_relations": [],
                "retrieved_nodes": [],
                "steps": [f"graph_search: TIMEOUT ({GRAPH_SEARCH_TIMEOUT}s)"],
            }

        if result is None:
            logger.warning("[graph_search] graph_engine.search 返回 None")
        elif isinstance(result, dict):
            # 提取实体和关系（从 entities/triplets 字段）
            entities = result.get("entities", []) or []
            triplets = result.get("triplets", []) or []

            # triplets: [{"head": str, "relation": str, "tail": str}, ...]
            for t in triplets:
                if isinstance(t, dict):
                    relations.append({
                        "head": t.get("head", ""),
                        "relation": t.get("relation", ""),
                        "tail": t.get("tail", ""),
                        "description": t.get("description", ""),
                    })

            # 提取 source_nodes 作为文本上下文
            sources = result.get("sources", [])
            for s in sources:
                if isinstance(s, dict):
                    retrieved_nodes.append({
                        "text": s.get("text", ""),
                        "score": s.get("score", 1.0),
                        "metadata": s.get("metadata", {}),
                        "source": "graph",
                    })
                elif hasattr(s, "text"):
                    retrieved_nodes.append({
                        "text": getattr(s, "text", ""),
                        "score": getattr(s, "score", 1.0),
                        "metadata": getattr(s, "metadata", {}),
                        "source": "graph",
                    })

            search_successful = True
        else:
            logger.warning(f"[graph_search] 未知的 result 类型: {type(result)}")

        logger.info(
            f"[graph_search] 完成: {len(entities)} 实体, {len(relations)} 关系, "
            f"{len(retrieved_nodes)} 文本块"
        )

    except Exception as e:
        logger.warning(f"[graph_search] 图谱检索失败（不阻断流程）: {e}")

    output = GraphSearchOutput(
        graph_entities=entities,
        graph_relations=relations,
        retrieved_nodes=retrieved_nodes,
        search_successful=search_successful,
    )

    status = "OK" if search_successful else "FAILED"
    return {
        "graph_entities": output.graph_entities,
        "graph_relations": output.graph_relations,
        "retrieved_nodes": output.retrieved_nodes,
        "steps": [
            f"graph_search: {status} "
            f"({len(output.graph_entities)} entities, {len(output.graph_relations)} relations)"
        ],
    }
