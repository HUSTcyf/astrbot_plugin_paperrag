"""
Analyze node — 调用 IdeaEngine.analyze_topic。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from astrbot.api import logger


class AnalyzeInput(BaseModel):
    topic: str = Field(..., min_length=1)
    depth: str = Field(default="standard")

    @property
    def topic_stripped(self) -> str:
        return self.topic.strip()


class AnalyzeOutput(BaseModel):
    topic_analysis: dict  # 序列化 TopicAnalysis


async def analyze_topic_node(state: dict) -> dict:
    """
    LangGraph 节点：分析研究主题。

    调用 IdeaEngine.analyze_topic()，输出 domain/keywords/search_queries 等。

    Args:
        state: AgenticIdeaState（读取 topic, depth, _context）

    Returns:
        更新 state 的 dict（topic_analysis, phase, steps）
    """
    try:
        input_data = AnalyzeInput(
            topic=state["topic"],
            depth=state.get("depth", "standard"),
        )
    except ValueError:
        raise

    topic = input_data.topic_stripped
    depth = input_data.depth

    logger.debug(f"[analyze] 分析主题: {topic}, depth={depth}")

    context = state.get("_context")
    if context is None:
        raise ValueError("[analyze] _context 未传入")

    rag_engine = state.get("_rag_engine")
    try:
        from idea import IdeaEngine
        engine = IdeaEngine(context=context, rag_engine=rag_engine)
    except Exception as e:
        logger.error(f"[analyze] IdeaEngine 创建失败: {e}")
        raise

    try:
        result = await engine.analyze_topic(topic, depth=depth)

        if result is None:
            raise RuntimeError("analyze_topic 返回 None")

        # TopicAnalysis → dict（可序列化）
        topic_analysis = {
            "domain": result.domain,
            "keywords": result.keywords,
            "search_queries": result.search_queries,
            "local_rag_queries": result.local_rag_queries,
            "exploration_angles": result.exploration_angles,
            "summary": result.summary,
        }

        logger.info(f"[analyze] 完成: domain={topic_analysis['domain']}, keywords={len(topic_analysis['keywords'])}")

        return {
            "topic_analysis": topic_analysis,
            "phase": "search",
            "steps": [f"analyze: OK (domain={topic_analysis['domain']}, keywords={len(topic_analysis['keywords'])})"],
        }

    except Exception as e:
        logger.error(f"[analyze] 分析失败: {e}")
        raise RuntimeError(f"[analyze] 分析失败: {e}") from e
