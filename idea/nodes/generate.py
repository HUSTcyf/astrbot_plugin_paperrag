"""
Generate node — 调用 IdeaEngine.generate_ideas。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from astrbot.api import logger


class GenerateInput(BaseModel):
    topic: str
    context_data: dict
    topic_analysis: dict | None = None
    num_ideas: int = Field(default=3, ge=1, le=10)
    idea_focus: str = Field(default="all")


async def generate_ideas_node(state: dict) -> dict:
    """
    LangGraph 节点：生成研究想法。

    调用 IdeaEngine.generate_ideas()，基于知识上下文生成 ResearchIdea 列表。

    Args:
        state: AgenticIdeaState（读取 topic, context_data, topic_analysis, _context, _rag_engine）

    Returns:
        更新 state 的 dict（ideas, phase, steps）
    """
    try:
        input_data = GenerateInput(
            topic=state["topic"],
            context_data=state.get("context_data") or {},
            topic_analysis=state.get("topic_analysis"),
            num_ideas=state.get("_num_ideas", 3),
            idea_focus=state.get("_idea_focus", "all"),
        )
    except ValueError:
        raise

    topic = input_data.topic.strip()
    context_data = input_data.context_data
    fused_context = context_data.get("fused_context", "") if context_data else ""

    if not fused_context:
        logger.warning("[generate] fused_context 为空，生成质量可能受影响")

    domain = ""
    if input_data.topic_analysis:
        domain = input_data.topic_analysis.get("domain", "")

    logger.debug(f"[generate] 生成 {input_data.num_ideas} 个想法: topic={topic}")

    context = state.get("_context")
    if context is None:
        raise ValueError("[generate] _context 未传入")

    rag_engine = state.get("_rag_engine")

    try:
        from idea import IdeaEngine
        engine = IdeaEngine(context=context, rag_engine=rag_engine)
    except Exception as e:
        logger.error(f"[generate] IdeaEngine 创建失败: {e}")
        raise

    try:
        result = await engine.generate_ideas(
            knowledge_context=fused_context,
            research_domain=domain,
            num_ideas=input_data.num_ideas,
            idea_focus=input_data.idea_focus,
            topic=topic,
        )

        if not result:
            raise RuntimeError("generate_ideas 返回空列表")

        # ResearchIdea list → list[dict]
        ideas = []
        for idea in result:
            ideas.append({
                "title": idea.title,
                "description": idea.description,
                "novelty": idea.novelty,
                "methodology": idea.methodology,
                "potential_challenges": idea.potential_challenges,
                "related_work": idea.related_work,
                "feasibility": idea.feasibility,
                "inspiration_sources": idea.inspiration_sources,
            })

        logger.info(f"[generate] 完成: {len(ideas)} 个想法")

        return {
            "ideas": ideas,
            "phase": "critique",
            "steps": [f"generate: OK ({len(ideas)} ideas)"],
        }

    except Exception as e:
        logger.error(f"[generate] 生成失败: {e}")
        raise RuntimeError(f"[generate] 生成失败: {e}") from e
