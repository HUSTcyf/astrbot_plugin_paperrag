"""
Critique node — LLM 审查研究想法的质量。
"""

from __future__ import annotations

from pydantic import BaseModel

from astrbot.api import logger


class CritiqueInput(BaseModel):
    ideas: list[dict]
    context_data: dict


# 审查阈值
CONFIDENCE_THRESHOLD = 0.7


async def critique_ideas_node(state: dict) -> dict:
    """
    LangGraph 节点：审查研究想法质量。

    调用 LLM 评估 ideas 的引用支撑、创新性、可行性。

    Args:
        state: AgenticIdeaState（读取 ideas, context_data, _context）

    Returns:
        更新 state 的 dict（critique, confidence, missing_evidence, idea_scores, phase, steps）
    """
    try:
        input_data = CritiqueInput(
            ideas=state.get("ideas", []),
            context_data=state.get("context_data") or {},
        )
    except ValueError:
        raise

    ideas = input_data.ideas
    context_data = input_data.context_data
    fused_context = context_data.get("fused_context", "")[:3000] if context_data else ""

    if not ideas:
        logger.warning("[critique] ideas 为空，跳过审查")
        return {
            "critique": "无想法可审查",
            "confidence": 1.0,
            "missing_evidence": [],
            "idea_scores": [],
            "phase": "done",
            "steps": ["critique: SKIPPED (no ideas)"],
        }

    logger.debug(f"[critique] 审查 {len(ideas)} 个想法")

    context = state.get("_context")
    if context is None:
        raise ValueError("[critique] _context 未传入")

    # 获取 LLM provider（统一 4 步解析）
    from provider.llm_utils import call_llm_json

    ideas_text = "\n".join(
        f"[{i+1}] {idea['title']}\n  描述: {idea.get('description','')[:100]}\n"
        f"  创新点: {idea.get('novelty','')[:80]}\n"
        f"  方法论: {idea.get('methodology','')[:80]}"
        for i, idea in enumerate(ideas)
    )

    prompt = f"""你是一个学术研究评审员。请评估以下研究想法的质量。

证据上下文（相关论文摘要）：
{fused_context[:2000]}

研究想法：
{ideas_text}

请返回以下 JSON 格式的评审结果（只返回 JSON，不要包含其他文字）：
{{
    "critique": "整体评审意见（100字以内，中文）",
    "confidence": 0.0到1.0之间的置信度分数（综合评估）,
    "missing_evidence": ["缺失证据描述1（如：缺少对 Transformer 架构的对比）", ...]（最多3条）,
    "idea_scores": [
        {{
            "title": "想法标题",
            "score": 0到10的分数,
            "issues": ["具体问题1", ...]
        }},
        ...
    ]
}}"""

    try:
        result = await call_llm_json(prompt, context, state.get("_config"))

        critique = result.get("critique", "") if result else ""
        confidence = float(result.get("confidence", 0.5)) if result else 0.5
        missing_evidence = result.get("missing_evidence", []) if result else []
        idea_scores = result.get("idea_scores", []) if result else []

        # 置信度边界约束
        confidence = max(0.0, min(1.0, confidence))

        # 判断是否需要迭代
        needs_refine = (
            confidence < CONFIDENCE_THRESHOLD or
            len(missing_evidence) > 0
        )
        next_phase = "refine" if needs_refine else "done"

        logger.info(
            f"[critique] 完成: confidence={confidence:.2f}, "
            f"missing={len(missing_evidence)}, next_phase={next_phase}"
        )

        return {
            "critique": critique,
            "confidence": confidence,
            "missing_evidence": missing_evidence,
            "idea_scores": idea_scores,
            "phase": next_phase,
            "steps": [
                f"critique: OK (confidence={confidence:.2f}, missing={len(missing_evidence)}, next={next_phase})"
            ],
        }

    except Exception as e:
        logger.error(f"[critique] 审查失败: {e}")
        # 审查失败时不阻断，降级为 done
        return {
            "critique": f"审查失败: {e}",
            "confidence": 0.5,
            "missing_evidence": [],
            "idea_scores": [],
            "phase": "done",
            "steps": [f"critique: FAILED → done (error={e})"],
        }
