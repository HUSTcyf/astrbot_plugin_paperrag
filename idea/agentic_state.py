"""
Agentic Idea State — TypedDict for Agentic Idea Engine LangGraph workflow.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict


class AgenticIdeaState(TypedDict):
    """Agentic Idea Engine 的 LangGraph 工作流状态。"""

    # 课题输入
    topic: str
    depth: str  # "quick" | "standard" | "deep"

    # 课题分析结果
    topic_analysis: NotRequired[dict | None]  # 序列化后的 TopicAnalysis

    # 知识检索结果
    context_data: NotRequired[dict | None]  # {local_results, web_results, fused_context, stats}

    # 生成状态
    ideas: NotRequired[list[dict]]  # ResearchIdea 序列化为 dict
    draft: NotRequired[str | None]

    # Agentic 控制
    iteration: NotRequired[int]  # 当前迭代轮次
    critique: NotRequired[str | None]  # 审查意见
    confidence: NotRequired[float]  # 置信度 [0, 1]
    missing_evidence: NotRequired[list[str]]  # 缺失证据列表
    idea_scores: NotRequired[list[dict] | None]  # 每个 idea 的评分

    # 保存结果
    saved_paths: NotRequired[list[str]]  # 已保存的想法文件路径

    # 辩论模式（Ideator ↔ Critic）
    debate_round: NotRequired[int]  # 当前辩论轮次
    debate_history: NotRequired[list[str]]  # 辩论历史记录

    # 工作流状态
    phase: NotRequired[Literal["analyze", "search", "generate", "critique", "refine", "done"]]
    steps: Annotated[list[str], operator.add]

    # 外部依赖（LangGraph State 需声明才传递）
    _context: NotRequired[Any]
    _rag_engine: NotRequired[Any]
    _num_ideas: NotRequired[int]
    _idea_focus: NotRequired[str]
    _local_rag_top_k: NotRequired[int]
    _web_top_k: NotRequired[int]
    _max_iterations: NotRequired[int]
    _max_debate_rounds: NotRequired[int]
