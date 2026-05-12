"""
Save node — 持久化 ideas 到文件系统。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from astrbot.api import logger


class SaveInput(BaseModel):
    ideas: list[dict]
    topic: str
    context_data: dict | None = None
    topic_analysis: dict | None = None


async def save_ideas_node(state: dict) -> dict:
    """
    LangGraph 节点：持久化 ideas 到文件系统。

    调用 IdeaEngine.save_ideas_to_file() 保存，并更新 topic_index。

    Args:
        state: AgenticIdeaState（读取 ideas, topic, context_data, topic_analysis, _context, _rag_engine）

    Returns:
        更新 state 的 dict（saved_paths, steps）
    """
    try:
        input_data = SaveInput(
            ideas=state.get("ideas", []),
            topic=state["topic"],
            context_data=state.get("context_data"),
            topic_analysis=state.get("topic_analysis"),
        )
    except ValueError:
        raise

    ideas = input_data.ideas
    topic = input_data.topic.strip()

    if not ideas:
        logger.warning("[save] ideas 为空，跳过保存")
        return {
            "saved_paths": [],
            "steps": ["save: SKIPPED (no ideas)"],
        }

    logger.debug(f"[save] 保存 {len(ideas)} 个想法: topic={topic}")

    context = state.get("_context")
    if context is None:
        raise ValueError("[save] _context 未传入")

    rag_engine = state.get("_rag_engine")

    try:
        from idea import IdeaEngine
        from idea.datatypes import ResearchIdea
        engine = IdeaEngine(context=context, rag_engine=rag_engine)
    except Exception as e:
        logger.error(f"[save] IdeaEngine 创建失败: {e}")
        raise RuntimeError(f"[save] IdeaEngine 创建失败: {e}") from e

    # 转换 dict → ResearchIdea
    research_ideas = []
    for idea_dict in ideas:
        research_ideas.append(ResearchIdea(
            title=idea_dict.get("title", ""),
            description=idea_dict.get("description", ""),
            novelty=idea_dict.get("novelty", ""),
            methodology=idea_dict.get("methodology", ""),
            potential_challenges=idea_dict.get("potential_challenges", []),
            related_work=idea_dict.get("related_work", []),
            feasibility=idea_dict.get("feasibility", 0.5),
            inspiration_sources=idea_dict.get("inspiration_sources", []),
        ))

    try:
        # 构建 knowledge dict（同原有逻辑）
        knowledge = {
            "local_results": [],
            "web_results": [],
            "fused_context": "",
        }
        if input_data.context_data:
            knowledge["local_results"] = input_data.context_data.get("local_results", [])
            knowledge["web_results"] = input_data.context_data.get("web_results", [])
            knowledge["fused_context"] = input_data.context_data.get("fused_context", "")

        saved = engine.save_ideas_to_file(
            ideas=research_ideas,
            topic=topic,
            knowledge=knowledge,
        )

        saved_paths = [(uid, str(path)) for uid, path in saved]
        logger.info(f"[save] 保存完成: {len(saved_paths)} 个文件")

        # === Wiki 双写（非致命）===
        try:
            from idea.wiki import IdeaWikiEngine, slugify
            wiki = IdeaWikiEngine()
            wiki.ensure_schema()

            topic_slug = slugify(topic)
            wiki.add_topic_to_index(topic, topic_slug)
            wiki.append_log("ingest", f"topic={topic}, ideas={len(ideas)}")
            wiki.init_topic_index(topic, input_data.topic_analysis)

            ctx_rel = None
            if input_data.context_data:
                ctx_path = wiki.save_context(topic, input_data.context_data)
                ctx_rel = str(ctx_path.relative_to(wiki.root))

            idea_scores = state.get("idea_scores")
            for idea_dict, (uid, _) in zip(ideas, saved_paths):
                scores = None
                if idea_scores:
                    for s in idea_scores:
                        if s.get("title") == idea_dict.get("title"):
                            scores = dict(s)
                            # Normalize critique score (0-10) to feasibility (0-1)
                            if "score" in scores and "feasibility" not in scores:
                                scores["feasibility"] = scores["score"] / 10.0
                            break

                wiki.save_idea(
                    topic=topic,
                    idea_dict=idea_dict,
                    idea_id=uid,
                    context_path=ctx_rel,
                    scores=scores,
                    debate_rounds=state.get("debate_round", 0),
                )
                wiki.add_idea_to_topic_index(
                    topic,
                    slugify(idea_dict.get("title", "")) + "-" + uid[:8],
                    idea_dict.get("title", ""),
                    scores=scores,
                )

            logger.info(f"[save] Wiki 双写完成: {len(ideas)} ideas → {wiki.get_wiki_root()}")
        except Exception as e:
            logger.warning(f"[save] Wiki 写入失败（非致命）: {e}")

        return {
            "saved_paths": saved_paths,
            "steps": [f"save: OK ({len(saved_paths)} files saved)"],
        }

    except Exception as e:
        logger.error(f"[save] 保存失败: {e}")
        raise RuntimeError(f"[save] 保存失败: {e}") from e
