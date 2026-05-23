"""
LangGraph State 定义 + 状态压缩策略。
"""

from __future__ import annotations

from typing import Annotated, TypedDict, Any
from typing_extensions import NotRequired

import operator


def _keep_last_n(n: int):
    """
    状态压缩函数：保留列表中最后 n 条。
    配合 Annotated[list, _keep_last_n(10)] 使用。
    """
    def reducer(old: list, new: list) -> list:
        combined = old + new
        return combined[-n:] if len(combined) > n else combined
    return reducer


class AgenticRAGState(TypedDict):
    """LangGraph 工作流内部状态。"""

    # 用户查询
    query: str

    # 路由信息（router 节点输出）
    query_type: NotRequired[str]          # "fact" | "comparison" | "review" | "citation"
    graph_weight: NotRequired[float]     # 0.0-1.0，图谱权重

    # 规划（planner 节点输出，P1 启用）
    plan: NotRequired[list[str]]

    # 检索结果（vector_search + graph_search 并行追加）
    retrieved_nodes: Annotated[list[dict], _keep_last_n(10)]

    # 图谱结构化知识（graph_search 节点输出）
    graph_entities: NotRequired[list[dict]]    # [{"name": str, "type": str, "description": str}, ...]
    graph_relations: NotRequired[list[dict]]    # [{"head": str, "relation": str, "tail": str}, ...]

    # 生成（synthesize 节点输出）
    draft: NotRequired[str]
    citations: NotRequired[list[str]]      # [doi1, doi2, ...]

    # 最终答案（final_output 节点输出）
    final_answer: NotRequired[str]

    # 执行轨迹（可打印调试）
    steps: Annotated[list[str], operator.add]

    # 重试保护
    retry_count: NotRequired[int]
    quality_issues: NotRequired[list[str]]

    # 外部依赖（LangGraph State 需声明才传递）
    _context: NotRequired[Any]
    _rag_engine: NotRequired[Any]
    _config: NotRequired[Any]
    top_k: NotRequired[int]
