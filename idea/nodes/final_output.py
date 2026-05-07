"""
Final Output node — 格式化 ideas 为最终输出。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from astrbot.api import logger


class FinalOutputInput(BaseModel):
    ideas: list[dict]
    critique: str | None = None
    confidence: float = 0.0
    topic: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.ideas


def _format_ideas(ideas: list[dict], critique: str | None, confidence: float, topic: str) -> str:
    """将 ideas 格式化为可读输出。"""
    lines = [f"**💡 研究想法 — {topic}**\n"]

    if critique:
        lines.append(f"**📋 评审意见**: {critique}")
        lines.append(f"**📊 置信度**: {confidence:.0%}\n")

    for i, idea in enumerate(ideas, 1):
        feasibility = idea.get("feasibility", 0.5)
        feasibility_bar = "★" * int(feasibility * 5) + "☆" * (5 - int(feasibility * 5))

        lines.append(f"""---
**[{i}] {idea.get('title', '无标题')}**

**📝 描述**: {idea.get('description', '')[:200]}...

**✨ 创新点**: {idea.get('novelty', '')[:150]}

**🔧 方法论**: {idea.get('methodology', '')[:150]}

**⚠️ 挑战**: {', '.join(idea.get('potential_challenges', [])[:2])}

**📈 可行性**: {feasibility_bar} ({feasibility:.0%})
""")

    return "\n".join(lines)


async def final_output_node(state: dict) -> dict:
    """
    LangGraph 节点：格式化 ideas 为最终输出。

    Args:
        state: AgenticIdeaState（读取 ideas, critique, confidence, topic）

    Returns:
        更新 state 的 dict（final_output, steps）
    """
    try:
        input_data = FinalOutputInput(
            ideas=state.get("ideas", []),
            critique=state.get("critique"),
            confidence=state.get("confidence", 0.0),
            topic=state.get("topic", ""),
        )
    except ValueError:
        raise

    if input_data.is_empty:
        return {
            "final_output": "⚠️ 未能生成任何研究想法，请尝试调整研究主题。",
            "steps": ["final_output: EMPTY (no ideas)"],
        }

    final_output = _format_ideas(
        ideas=input_data.ideas,
        critique=input_data.critique,
        confidence=input_data.confidence,
        topic=input_data.topic,
    )

    logger.debug(f"[final_output] 格式化: {len(input_data.ideas)} ideas, {len(final_output)} chars")

    return {
        "final_output": final_output,
        "steps": [f"final_output: OK ({len(input_data.ideas)} ideas, {len(final_output)} chars)"],
    }
