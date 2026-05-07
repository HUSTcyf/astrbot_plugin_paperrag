"""
Refine node — 基于 critique 定向修复低分 ideas（仅重新生成有问题的 ideas）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from astrbot.api import logger

WEAK_SCORE_THRESHOLD = 7


class RefineInput(BaseModel):
    topic: str
    ideas: list[dict]
    missing_evidence: list[str]
    iteration: int
    context_data: dict
    topic_analysis: dict | None
    max_iterations: int = Field(default=3)
    idea_scores: list[dict] = Field(default_factory=list)


def _split_ideas(ideas: list[dict], idea_scores: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """将 ideas 按 idea_scores 拆分为 keep（score >= 7）和 weak（score < 7）。

    Returns:
        (keep_ideas, weak_ideas, all_weak_issues)
    """
    score_lookup: dict[str, dict] = {}
    for sc in idea_scores:
        score_lookup[sc.get("title", "")] = {
            "score": sc.get("score", 0),
            "issues": sc.get("issues", []),
        }

    keep_ideas: list[dict] = []
    weak_ideas: list[dict] = []
    weak_issues: list[str] = []

    for idea in ideas:
        title = idea.get("title", "")
        info = score_lookup.get(title)
        if info and info["score"] < WEAK_SCORE_THRESHOLD:
            weak_ideas.append(idea)
            weak_issues.extend(info["issues"])
        else:
            keep_ideas.append(idea)

    return keep_ideas, weak_ideas, weak_issues


def _build_issue_context(weak_ideas: list[dict], idea_scores: list[dict]) -> str:
    """构建 issue 反馈文本，追加到 knowledge_context 中供 LLM 参考。"""
    score_lookup = {sc.get("title", ""): sc for sc in idea_scores}
    parts = ["\n\n[以下研究想法存在质量问题，需要改进：]"]
    for idea in weak_ideas:
        title = idea.get("title", "")
        sc = score_lookup.get(title, {})
        issues = sc.get("issues", [])
        parts.append(f"\n想法: {title}")
        parts.append(f"原始描述: {idea.get('description', '')[:150]}")
        if issues:
            parts.append(f"具体问题: {'; '.join(issues)}")
    return "\n".join(parts)


async def refine_ideas_node(state: dict) -> dict:
    """
    LangGraph 节点：基于 critique 反馈定向修复低分 ideas。

    如果 idea_scores 可用，仅重新生成分数低于阈值的 ideas，
    保留高分 ideas 不变。否则回退到全量重生成。

    Args:
        state: AgenticIdeaState

    Returns:
        更新 state 的 dict
    """
    try:
        input_data = RefineInput(
            topic=state["topic"],
            ideas=state.get("ideas", []),
            missing_evidence=state.get("missing_evidence", []),
            iteration=state.get("iteration", 0),
            context_data=state.get("context_data") or {},
            topic_analysis=state.get("topic_analysis"),
            max_iterations=state.get("_max_iterations", 3),
            idea_scores=state.get("idea_scores", []),
        )
    except ValueError:
        raise

    iteration = input_data.iteration

    # 强制终止条件
    if iteration >= input_data.max_iterations:
        logger.info(f"[refine] 达到最大迭代次数 {input_data.max_iterations}，强制终止")
        return {
            "phase": "done",
            "steps": [f"refine: MAX ITERATION ({input_data.max_iterations}), forcing done"],
        }

    missing = input_data.missing_evidence
    idea_scores = input_data.idea_scores

    logger.debug(f"[refine] 第 {iteration+1} 次迭代: missing_evidence={len(missing)}, idea_scores={len(idea_scores)}")

    context = state.get("_context")
    if context is None:
        raise ValueError("[refine] _context 未传入")

    rag_engine = state.get("_rag_engine")

    try:
        from idea import IdeaEngine
        engine = IdeaEngine(context=context, rag_engine=rag_engine)
    except Exception as e:
        logger.error(f"[refine] IdeaEngine 创建失败: {e}")
        raise RuntimeError(f"[refine] IdeaEngine 创建失败: {e}") from e

    # 1. 补充检索（如果有缺失证据）
    new_context_data = dict(input_data.context_data)
    if missing:
        logger.info(f"[refine] 针对缺失证据执行补充检索: {len(missing)} 条")
        try:
            extra_results = await engine.search_knowledge(
                queries=missing[:3],
                local_rag_top_k=5,
                web_top_k=3,
            )
            existing_local = new_context_data.get("local_results", [])
            existing_web = new_context_data.get("web_results", [])
            new_local = extra_results.get("local_results", [])
            new_web = extra_results.get("web_results", [])

            seen_texts = {r.get("text", "") for r in existing_local}
            merged_local = list(existing_local) + [r for r in new_local if r.get("text", "") not in seen_texts]

            seen_urls = {r.get("url", "") for r in existing_web}
            merged_web = list(existing_web) + [r for r in new_web if r.get("url", "") not in seen_urls]

            new_context_data = {
                "local_results": merged_local,
                "web_results": merged_web,
                "fused_context": engine._fuse_knowledge_context(merged_local, merged_web),
                "stats": {
                    "local_count": len(merged_local),
                    "web_count": len(merged_web),
                },
            }
            logger.info(f"[refine] 补充检索完成: local+={len(new_local)}, web+={len(new_web)}")
        except Exception as e:
            logger.warning(f"[refine] 补充检索失败: {e}，使用原有 context")
            new_context_data = input_data.context_data

    # 2. 决策：定向修复 vs 全量重生成
    domain = input_data.topic_analysis.get("domain", "") if input_data.topic_analysis else ""
    fused_context = new_context_data.get("fused_context", "")

    if idea_scores:
        keep_ideas, weak_ideas, _weak_issues = _split_ideas(input_data.ideas, idea_scores)

        if not weak_ideas:
            # 所有 ideas 分数都达标，无需重生成
            logger.info(f"[refine] 所有 {len(keep_ideas)} 个 ideas 质量达标，跳过重生成")
            return {
                "ideas": keep_ideas,
                "context_data": new_context_data,
                "iteration": iteration + 1,
                "missing_evidence": [],
                "phase": "done",
                "steps": [f"refine: ALL GOOD (iteration={iteration+1}, kept={len(keep_ideas)})"],
            }

        # 定向修复：只重新生成 weak ideas
        num_to_regenerate = len(weak_ideas)
        issue_context = _build_issue_context(weak_ideas, idea_scores)
        enhanced_context = fused_context + issue_context

        logger.info(
            f"[refine] 定向修复: keeping {len(keep_ideas)}, "
            f"regenerating {num_to_regenerate} weak ideas"
        )
    else:
        # 无 idea_scores，回退到全量重生成
        keep_ideas = []
        num_to_regenerate = len(input_data.ideas)
        enhanced_context = fused_context
        logger.info(f"[refine] 无 idea_scores，全量重生成 {num_to_regenerate} ideas")

    # 3. 调用 generate_ideas
    try:
        new_ideas_result = await engine.generate_ideas(
            knowledge_context=enhanced_context,
            research_domain=domain,
            num_ideas=num_to_regenerate,
            idea_focus="all",
            topic=input_data.topic,
        )

        if not new_ideas_result:
            logger.warning("[refine] 重新生成失败，保留原有 ideas")
            return {
                "ideas": input_data.ideas,
                "context_data": new_context_data,
                "iteration": iteration + 1,
                "missing_evidence": [],
                "phase": "critique",
                "steps": [f"refine: re-generate FAILED, keeping old ideas, iteration={iteration+1}"],
            }

        # 序列化新 ideas
        new_ideas = []
        for idea in new_ideas_result:
            new_ideas.append({
                "title": idea.title,
                "description": idea.description,
                "novelty": idea.novelty,
                "methodology": idea.methodology,
                "potential_challenges": idea.potential_challenges,
                "related_work": idea.related_work,
                "feasibility": idea.feasibility,
                "inspiration_sources": idea.inspiration_sources,
            })

        # 4. 合并：保留的 ideas + 新生成的 ideas
        final_ideas = list(keep_ideas) + new_ideas

        logger.info(
            f"[refine] 完成: kept={len(keep_ideas)}, "
            f"regenerated={len(new_ideas)}, total={len(final_ideas)}"
        )

        return {
            "ideas": final_ideas,
            "context_data": new_context_data,
            "iteration": iteration + 1,
            "missing_evidence": [],
            "phase": "critique",
            "steps": [
                f"refine: OK (iteration={iteration+1}, "
                f"kept={len(keep_ideas)}, new={len(new_ideas)}, total={len(final_ideas)})"
            ],
        }

    except Exception as e:
        logger.error(f"[refine] 重新生成失败: {e}")
        return {
            "ideas": input_data.ideas,
            "context_data": new_context_data,
            "iteration": iteration + 1,
            "missing_evidence": [],
            "phase": "done",
            "steps": [f"refine: FAILED → done (error={e})"],
        }
