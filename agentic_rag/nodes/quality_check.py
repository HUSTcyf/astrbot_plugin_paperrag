"""
Quality Check node — 规则化质量检查（零 LLM 调用）。
"""

from __future__ import annotations

import re

from astrbot.api import logger

MAX_RETRIES = 2


async def quality_check_node(state: dict) -> dict:
    """
    规则化质量检查：验证 synthesize 输出的 draft 质量。
    不调用 LLM，纯规则判断。不达标时路由回 synthesize 重试。

    检查项：
    1. draft 长度 > 50 字符（排除"I don't know"等空答案）
    2. retrieved_nodes 非空（有检索结果支撑）
    3. draft 包含 [#n] 引用格式（仅当有 nodes 时检查）
    """
    draft = state.get("draft", "") or ""
    nodes = state.get("retrieved_nodes") or []
    retry_count = state.get("retry_count", 0)

    issues: list[str] = []

    # Check 1: minimum length
    if len(draft.strip()) <= 50:
        issues.append("回答过短（≤50字符），可能是空答案或'I don't know'")

    # Check 2: has retrieved sources
    if not nodes:
        issues.append("无检索结果支撑")

    # Check 3: has citations (only when nodes exist)
    if nodes and not re.search(r"\[#\d+\]", draft):
        issues.append("回答中无 [#n] 引用标注")

    if not issues:
        return {
            "quality_issues": [],
            "steps": ["quality_check: PASSED"],
        }

    # Has issues — decide whether to retry or give up
    if retry_count < MAX_RETRIES:
        logger.info(f"[quality_check] 发现 {len(issues)} 个问题，重试 {retry_count + 1}/{MAX_RETRIES}")
        return {
            "quality_issues": issues,
            "retry_count": retry_count + 1,
            "steps": [f"quality_check: RETRY ({len(issues)} issues, attempt {retry_count + 1}/{MAX_RETRIES})"],
        }

    # Retries exhausted — pass through anyway
    logger.warning(f"[quality_check] 重试耗尽，放行 ({len(issues)} issues remain)")
    return {
        "quality_issues": issues,
        "retry_count": retry_count + 1,
        "steps": [f"quality_check: PASSED WITH ISSUES ({len(issues)} issues, retries exhausted)"],
    }
