"""
Search node — 调用 IdeaEngine.search_knowledge。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from astrbot.api import logger


class SearchInput(BaseModel):
    topic_analysis: dict
    local_rag_top_k: int = Field(default=10, ge=0)
    web_top_k: int = Field(default=5, ge=0)


async def search_knowledge_node(state: dict) -> dict:
    """
    LangGraph 节点：多源知识检索。

    调用 IdeaEngine.search_knowledge()，执行本地 RAG + 网络搜索。

    Args:
        state: AgenticIdeaState（读取 topic_analysis, _context, _rag_engine）

    Returns:
        更新 state 的 dict（context_data, phase, steps）
    """
    try:
        input_data = SearchInput(
            topic_analysis=state["topic_analysis"],
            local_rag_top_k=state.get("_local_rag_top_k", 10),
            web_top_k=state.get("_web_top_k", 5),
        )
    except ValueError:
        raise

    topic_analysis = input_data.topic_analysis
    queries = (
        topic_analysis.get("search_queries", [])[:5] +
        topic_analysis.get("local_rag_queries", [])[:3]
    )

    if not queries:
        queries = [state.get("topic", "")]

    logger.debug(f"[search] 检索: queries={queries}")

    context = state.get("_context")
    if context is None:
        raise ValueError("[search] _context 未传入")

    rag_engine = state.get("_rag_engine")

    try:
        from idea import IdeaEngine
        engine = IdeaEngine(context=context, rag_engine=rag_engine)
    except Exception as e:
        logger.error(f"[search] IdeaEngine 创建失败: {e}")
        raise

    try:
        result = await engine.search_knowledge(
            queries=queries,
            local_rag_top_k=input_data.local_rag_top_k,
            web_top_k=input_data.web_top_k,
        )

        if result is None:
            result = {"local_results": [], "web_results": [], "fused_context": "", "stats": {"web_count": 0, "local_count": 0}}

        local_count = result.get("stats", {}).get("local_count", 0)
        web_count = result.get("stats", {}).get("web_count", 0)
        logger.info(f"[search] 完成: local={local_count}, web={web_count}")

        return {
            "context_data": result,
            "phase": "generate",
            "steps": [f"search: OK (local={local_count}, web={web_count})"],
        }

    except Exception as e:
        logger.error(f"[search] 检索失败: {e}")
        raise RuntimeError(f"[search] 检索失败: {e}") from e
