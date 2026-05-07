"""
Workflow — P0: router → [vector_search ‖ graph_search] → synthesize → quality_check → final_output.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Send
from langgraph.graph import StateGraph, END

from .state import AgenticRAGState
from .nodes.router import router_node
from .nodes.vector_search import vector_search_node
from .nodes.graph_search import graph_search_node
from .nodes.synthesize import synthesize_node
from .nodes.quality_check import quality_check_node
from .nodes.final_output import final_output_node


def route_after_router(state: AgenticRAGState) -> list[dict]:
    """router 之后分发到 vector_search 和 graph_search 并行执行。"""
    return [
        Send("vector_search", {"query": state["query"], "_context": state.get("_context"), "_config": state.get("_config")}),
        Send("graph_search", {"query": state["query"], "graph_weight": state.get("graph_weight", 0.0), "_context": state.get("_context"), "_config": state.get("_config")}),
    ]


def _barrier(state: AgenticRAGState) -> dict:
    """汇聚点：等待两个并行节点完成，然后路由到下一步。"""
    return {"steps": ["barrier: reached"]}


def route_after_parallel(state: AgenticRAGState) -> Literal["synthesize"]:
    """并行完成后路由到 synthesize。"""
    return "synthesize"


def route_after_quality_check(state: AgenticRAGState) -> Literal["synthesize", "final_output"]:
    """quality_check 完成后：有问题且未超重试次数则回到 synthesize，否则到 final_output。"""
    quality_issues = state.get("quality_issues", [])
    retry_count = state.get("retry_count", 0)
    if quality_issues and retry_count < 2:
        return "synthesize"
    return "final_output"


def compile_workflow() -> StateGraph:
    """编译 LangGraph StateGraph。"""
    graph = StateGraph(AgenticRAGState)

    graph.add_node("router", router_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("graph_search", graph_search_node)
    graph.add_node("_barrier", _barrier)  # 汇聚点
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("router")

    # 条件分发到并行节点
    graph.add_conditional_edges("router", route_after_router)

    # 两个并行节点完成后都汇聚到 barrier
    graph.add_edge("vector_search", "_barrier")
    graph.add_edge("graph_search", "_barrier")

    # barrier → synthesize
    graph.add_conditional_edges("_barrier", route_after_parallel)

    # synthesize → quality_check → final_output (or back to synthesize on retry)
    graph.add_edge("synthesize", "quality_check")
    graph.add_conditional_edges("quality_check", route_after_quality_check)
    graph.add_edge("final_output", END)

    return graph.compile()
