"""
Debate node — Ideator responds to Critic's feedback (defense + modification).
"""

from __future__ import annotations

import json

from astrbot.api import logger
from provider.llm_utils import call_llm_json

DEBATE_IDEATOR_PROMPT = """你是一个创新研究者。一位学术评审专家对你的研究想法提出了批评。

请仔细考虑评审意见：
1. 对于合理的批评，提出具体改进方案并修改对应想法
2. 对于你认为不合理的批评，基于证据给出反驳
3. 保持想法与研究主题「{topic}」的紧密相关性
4. 改进后的想法应更具体、更有说服力

{history_section}
评审意见：{critique}

各想法评分详情：
{scores_text}

当前想法：
{ideas_text}

相关知识上下文：
{fused_context}

请输出辩护/修改说明和修改后的想法列表（严格 JSON 格式）：
{{
    "defense": "整体辩护/修改说明（200字以内，说明哪些批评接受了、哪些反驳了）",
    "ideas": [
        {{
            "title": "想法标题",
            "description": "改进后的描述",
            "novelty": "改进后的创新点",
            "methodology": "改进后的方法论",
            "potential_challenges": ["挑战1", "挑战2"],
            "related_work": ["相关工作1", "相关工作2"],
            "feasibility": 0.8,
            "inspiration_sources": ["灵感来源1"]
        }}
    ]
}}"""


def _format_ideas_text(ideas: list[dict]) -> str:
    parts = []
    for i, idea in enumerate(ideas, 1):
        parts.append(
            f"[{i}] {idea.get('title', '')}\n"
            f"  描述: {idea.get('description', '')[:150]}\n"
            f"  创新点: {idea.get('novelty', '')[:100]}\n"
            f"  方法论: {idea.get('methodology', '')[:100]}"
        )
    return "\n".join(parts)


def _format_scores_text(idea_scores: list[dict]) -> str:
    if not idea_scores:
        return "（无详细评分）"
    parts = []
    for sc in idea_scores:
        title = sc.get("title", "")
        score = sc.get("score", 0)
        issues = sc.get("issues", [])
        issue_str = "; ".join(issues) if issues else "无"
        parts.append(f"- {title}: {score}/10 (问题: {issue_str})")
    return "\n".join(parts)


def _build_history_section(debate_history: list[str]) -> str:
    if not debate_history:
        return ""
    parts = ["前几轮辩论记录："]
    for entry in debate_history:
        parts.append(entry)
    return "\n".join(parts) + "\n\n"




async def debate_node(state: dict) -> dict:
    """
    LangGraph 节点：Ideator 回应 Critic 的评审反馈。

    让创新者有机会辩护或修改想法，然后交由 Critic 重新评估。
    """
    ideas = state.get("ideas", [])
    idea_scores = state.get("idea_scores") or []
    critique = state.get("critique") or ""
    context_data = state.get("context_data") or {}
    fused_context = context_data.get("fused_context", "")[:3000] if context_data else ""
    debate_round = state.get("debate_round", 0)
    debate_history = list(state.get("debate_history") or [])
    max_debate_rounds = state.get("_max_debate_rounds", 2)

    # 所有想法已达标 → 无需辩论
    if idea_scores and all(s.get("score", 0) >= 7 for s in idea_scores):
        logger.info("[debate] 所有想法评分 >= 7，跳过辩论")
        return {
            "phase": "done",
            "debate_round": debate_round,
            "steps": ["debate: ALL_GOOD (no debate needed)"],
        }

    # 辩论轮次耗尽 → 进入 refine
    if debate_round >= max_debate_rounds:
        logger.info(f"[debate] 辩论轮次耗尽 ({debate_round}/{max_debate_rounds})，进入 refine")
        return {
            "phase": "refine",
            "debate_round": debate_round,
            "steps": [f"debate: MAX_ROUNDS ({debate_round}/{max_debate_rounds})"],
        }

    context = state.get("_context")
    if context is None:
        raise ValueError("[debate] _context 未传入")


    # 构建辩论 prompt
    topic = state.get("topic", "")
    ideas_text = _format_ideas_text(ideas)
    scores_text = _format_scores_text(idea_scores)
    history_section = _build_history_section(debate_history)

    prompt = DEBATE_IDEATOR_PROMPT.format(
        topic=topic,
        history_section=history_section,
        critique=critique,
        scores_text=scores_text,
        ideas_text=ideas_text,
        fused_context=fused_context[:2000],
    )

    try:
        result = await call_llm_json(prompt, context, state.get("_config"), temperature=0.7)
    except Exception as e:
        logger.error(f"[debate] LLM 调用失败: {e}")
        return {
            "phase": "refine",
            "debate_round": debate_round,
            "steps": [f"debate: LLM_FAILED → refine ({e})"],
        }

    if result is None:
        logger.warning("[debate] 无法解析 JSON，保持原有想法")
        return {
            "phase": "refine",
            "debate_round": debate_round,
            "steps": ["debate: JSON_PARSE_FAILED → refine"],
        }

    # 提取辩护说明和修改后的想法
    defense = result.get("defense", "")
    modified_ideas_raw = result.get("ideas", [])

    if not modified_ideas_raw:
        logger.warning("[debate] 响应中无 ideas，保持原有想法")
        return {
            "phase": "refine",
            "debate_round": debate_round,
            "steps": ["debate: NO_IDEAS_IN_RESPONSE → refine"],
        }

    # 标准化修改后的想法（确保所有字段存在）
    modified_ideas = []
    for idea in modified_ideas_raw:
        if isinstance(idea, dict):
            modified_ideas.append({
                "title": idea.get("title", ""),
                "description": idea.get("description", ""),
                "novelty": idea.get("novelty", ""),
                "methodology": idea.get("methodology", ""),
                "potential_challenges": idea.get("potential_challenges", []),
                "related_work": idea.get("related_work", []),
                "feasibility": idea.get("feasibility", 0.5),
                "inspiration_sources": idea.get("inspiration_sources", []),
            })

    if not modified_ideas:
        modified_ideas = list(ideas)

    # 如果 LLM 返回的想法数量少于原始数量，补回缺失的
    if len(modified_ideas) < len(ideas):
        missing_count = len(ideas) - len(modified_ideas)
        modified_titles = {idea.get("title", "") for idea in modified_ideas}
        for original in ideas:
            if original.get("title", "") not in modified_titles:
                modified_ideas.append(original)
        logger.warning(f"[debate] LLM 返回 {len(ideas) - missing_count}/{len(ideas)} 想法，已补回缺失的")

    # 记录辩论历史
    debate_history.append(f"[Round {debate_round + 1}] Ideator: {defense}")

    logger.info(
        f"[debate] 完成: round={debate_round + 1}/{max_debate_rounds}, "
        f"defense_len={len(defense)}, ideas={len(modified_ideas)}"
    )

    return {
        "ideas": modified_ideas,
        "debate_history": debate_history,
        "debate_round": debate_round + 1,
        "phase": "critique",
        "steps": [f"debate: OK (round {debate_round + 1}/{max_debate_rounds}, ideas={len(modified_ideas)})"],
    }
