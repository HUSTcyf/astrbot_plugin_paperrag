"""
ReAct Agent State — Tool-Using Agent 状态定义。
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import NotRequired, TypedDict

import operator

from .state import _keep_last_n

MAX_TOOL_CALLS = 5
MAX_ITERATIONS = 10


class ReActRAGState(TypedDict):
    """Tool-Using Agent 状态。"""

    # 用户查询
    query: str

    # ReAct 轨迹（SYSTEM + THOUGHT/ACTION/OBSERVATION/FINISH）
    scratchpad: str

    # 检索结果（工具调用累积）
    retrieved_nodes: Annotated[list[dict], _keep_last_n(10)]

    # 图谱结构化知识
    graph_entities: NotRequired[list[dict]]
    graph_relations: NotRequired[list[dict]]

    # 生成（agent FINISH 输出）
    draft: NotRequired[str]
    citations: NotRequired[list[str]]

    # 最终答案
    final_answer: NotRequired[str]

    # 迭代控制
    iteration: NotRequired[int]
    tool_call_count: NotRequired[int]

    # 质量检查
    quality_issues: NotRequired[list[str]]
    retry_count: NotRequired[int]

    # 待执行的工具动作
    _pending_action: NotRequired[dict]  # {"tool": str, "args": str}

    # 执行轨迹
    steps: Annotated[list[str], operator.add]

    # 外部依赖
    _context: NotRequired[Any]
    _config: NotRequired[Any]
    top_k: NotRequired[int]
