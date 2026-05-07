"""
ReAct Workflow — Tool-Using Agent 的 LangGraph StateGraph。

拓扑: agent ⇄ tool_executor → quality_check → final_output → END
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from .react_state import ReActRAGState, MAX_ITERATIONS
from .react_agent import react_agent_node
from .react_tools import react_tool_executor_node
from .nodes.quality_check import quality_check_node
from .nodes.final_output import final_output_node


def route_after_agent(state: ReActRAGState) -> Literal["tool_executor", "quality_check"]:
    """agent 输出后：有 pending action → tool_executor；有 draft 或超限 → quality_check。"""
    draft = state.get("draft")
    pending = state.get("_pending_action")
    iteration = state.get("iteration", 0)

    if draft:
        return "quality_check"

    if pending and iteration < MAX_ITERATIONS:
        return "tool_executor"

    return "quality_check"


def route_after_quality_check(state: ReActRAGState) -> Literal["agent", "final_output"]:
    """quality_check 后：有问题且可重试 → agent；否则 → final_output。"""
    quality_issues = state.get("quality_issues", [])
    retry_count = state.get("retry_count", 0)
    # quality_check_node 在 retry_count < MAX_RETRIES(=2) 时才返回 issues，
    # 所以 retry_count < 2 说明还有重试机会
    if quality_issues and retry_count < 2:
        return "agent"
    return "final_output"


def compile_react_workflow():
    """编译 ReAct Tool-Using Agent workflow。"""
    graph = StateGraph(ReActRAGState)

    graph.add_node("agent", react_agent_node)
    graph.add_node("tool_executor", react_tool_executor_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("agent")

    # agent → tool_executor or quality_check
    graph.add_conditional_edges("agent", route_after_agent)

    # tool_executor → agent (loop)
    graph.add_edge("tool_executor", "agent")

    # quality_check → agent (retry) or final_output
    graph.add_conditional_edges("quality_check", route_after_quality_check)

    # final_output → END
    graph.add_edge("final_output", END)

    return graph.compile()
