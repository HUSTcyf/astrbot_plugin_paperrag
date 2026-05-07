"""
Agentic Idea Engine — LangGraph StateGraph 工作流。

工作流：analyze → search → generate → critique → debate → critique → refine → save → final_output
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from idea.agentic_state import AgenticIdeaState
from idea.nodes.analyze import analyze_topic_node
from idea.nodes.search import search_knowledge_node
from idea.nodes.generate import generate_ideas_node
from idea.nodes.critique import critique_ideas_node
from idea.nodes.debate import debate_node
from idea.nodes.refine import refine_ideas_node
from idea.nodes.save import save_ideas_node
from idea.nodes.final_output import final_output_node


def route_after_analyze(state: AgenticIdeaState) -> Literal["search"]:
    """analyze 完成后路由到 search。"""
    return "search"


def route_after_search(state: AgenticIdeaState) -> Literal["generate"]:
    """search 完成后路由到 generate。"""
    return "generate"


def route_after_generate(state: AgenticIdeaState) -> Literal["critique"]:
    """generate 完成后路由到 critique。"""
    return "critique"


def route_after_critique(state: AgenticIdeaState) -> Literal["debate", "refine", "save"]:
    """critique 完成后：需要改进 → debate（有轮次）或 refine（轮次耗尽）；合格 → save。"""
    phase = state.get("phase", "done")
    if phase != "refine":
        return "save"

    debate_round = state.get("debate_round", 0)
    max_debate_rounds = state.get("_max_debate_rounds", 2)
    if debate_round < max_debate_rounds:
        return "debate"
    return "refine"


def route_after_debate(state: AgenticIdeaState) -> Literal["critique", "refine", "save"]:
    """debate 完成后：正常 → critique 重新评估；异常 → refine 或 save。"""
    phase = state.get("phase", "critique")
    if phase == "critique":
        return "critique"
    if phase == "refine":
        return "refine"
    return "save"


def route_after_refine(state: AgenticIdeaState) -> Literal["critique", "save"]:
    """refine 完成后：回到 critique 重新评估，或直接 save。"""
    iteration = state.get("iteration", 0)
    max_iter = state.get("_max_iterations", 3)
    if iteration >= max_iter:
        return "save"
    # 检查 phase 是否提前结束
    phase = state.get("phase", "critique")
    if phase == "done":
        return "save"
    return "critique"


def route_after_save(state: AgenticIdeaState) -> Literal["final_output"]:
    """save 完成后路由到 final_output。"""
    return "final_output"


def compile_workflow():
    """编译 LangGraph StateGraph。"""
    graph = StateGraph(AgenticIdeaState)

    graph.add_node("analyze", analyze_topic_node)
    graph.add_node("search", search_knowledge_node)
    graph.add_node("generate", generate_ideas_node)
    graph.add_node("critique", critique_ideas_node)
    graph.add_node("debate", debate_node)
    graph.add_node("refine", refine_ideas_node)
    graph.add_node("save", save_ideas_node)
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("analyze")

    graph.add_conditional_edges("analyze", route_after_analyze)
    graph.add_conditional_edges("search", route_after_search)
    graph.add_conditional_edges("generate", route_after_generate)
    graph.add_conditional_edges("critique", route_after_critique)
    graph.add_conditional_edges("debate", route_after_debate)
    graph.add_conditional_edges("refine", route_after_refine)
    graph.add_edge("save", "final_output")
    graph.add_edge("final_output", END)

    return graph.compile()


async def run_agentic_ideas(
    topic: str,
    context,
    depth: str = "standard",
    num_ideas: int = 3,
    idea_focus: str = "all",
    local_rag_top_k: int = 10,
    web_top_k: int = 5,
    max_iterations: int = 3,
    rag_engine=None,
    config: dict | None = None,
) -> dict:
    """
    Agentic Idea workflow 主入口。

    Args:
        topic: 研究主题
        context: AstrBot Context 对象
        depth: 分析深度 ("quick" | "standard" | "deep")
        num_ideas: 生成想法数量
        idea_focus: 侧重点 ("novelty" | "feasibility" | "impact" | "all")
        local_rag_top_k: 本地 RAG 召回数
        web_top_k: 网络搜索召回数
        max_iterations: critique-refine 最大迭代次数
        rag_engine: 可选，注入已有的 RAG engine 实例

    Returns:
        dict with keys: ideas, context_data, topic_analysis, critique,
                        confidence, idea_scores, final_output, saved_paths, steps
    """
    app = compile_workflow()
    initial_state = {
        "topic": topic,
        "depth": depth,
        "topic_analysis": None,
        "context_data": None,
        "ideas": [],
        "draft": None,
        "iteration": 0,
        "critique": None,
        "confidence": 0.0,
        "missing_evidence": [],
        "idea_scores": None,
        "phase": "analyze",
        "steps": [],
        "debate_round": 0,
        "debate_history": [],
        # 额外参数（State 内用 _ 前缀）
        "_num_ideas": num_ideas,
        "_idea_focus": idea_focus,
        "_local_rag_top_k": local_rag_top_k,
        "_web_top_k": web_top_k,
        "_max_iterations": max_iterations,
        "_max_debate_rounds": 2,
        "_context": context,
        "_config": config,
        "_rag_engine": rag_engine,
    }

    result = await app.ainvoke(initial_state)

    return {
        "ideas": result.get("ideas", []),
        "context_data": result.get("context_data"),
        "topic_analysis": result.get("topic_analysis"),
        "critique": result.get("critique"),
        "confidence": result.get("confidence", 0.0),
        "idea_scores": result.get("idea_scores"),
        "final_output": result.get("final_output", ""),
        "saved_paths": result.get("saved_paths", []),
        "steps": result.get("steps", []),
    }
