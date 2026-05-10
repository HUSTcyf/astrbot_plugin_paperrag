"""
ReAct Agent Node — LLM 推理 + 输出解析。
"""

from __future__ import annotations

import re

from astrbot.api import logger

from .react_state import MAX_ITERATIONS, MAX_TOOL_CALLS
from .react_tools import TOOL_DEFINITIONS
from provider.llm_utils import call_llm

SYSTEM_PROMPT = """你是一个学术论文问答助手。你可以使用以下工具来帮助回答问题：

{tool_descriptions}

使用工具时，请严格按以下格式输出：
THOUGHT: <你的推理过程>
ACTION: <工具名>(<参数>)

收到工具返回结果（OBSERVATION）后，继续推理。当你有足够信息回答问题时：
THOUGHT: <你的推理过程>
FINISH: <你的最终回答>

要求：
1. 回答准确，基于检索结果
2. 引用来源时使用 [#n] 格式（对应检索结果编号）
3. 如果信息不足，明确指出
4. 保持回答简洁有条理
5. 最多使用 {max_tool_calls} 次工具调用
"""


def _format_system_prompt() -> str:
    tool_desc = "\n".join(f"- {desc}" for desc in TOOL_DEFINITIONS.values())
    return SYSTEM_PROMPT.format(
        tool_descriptions=tool_desc,
        max_tool_calls=MAX_TOOL_CALLS,
    )


def _parse_response(text: str) -> tuple[str, dict | None]:
    """解析 LLM 输出为 ACTION 或 FINISH。

    Returns:
        ("finish", {"answer": str}) — FINISH 被解析
        ("action", {"tool": str, "args": str}) — ACTION 被解析
        ("unknown", None) — 两者都未解析
    """
    action_matches = list(re.finditer(
        r"ACTION:\s*(\w+)\((.+?)\)", text, re.DOTALL
    ))
    finish_matches = list(re.finditer(
        r"FINISH:\s*(.+)", text, re.DOTALL
    ))

    last_action = action_matches[-1] if action_matches else None
    last_finish = finish_matches[-1] if finish_matches else None

    if last_finish and (not last_action or last_finish.start() > last_action.start()):
        answer = last_finish.group(1).strip()
        return "finish", {"answer": answer}

    if last_action:
        tool_name = last_action.group(1).strip()
        tool_args = last_action.group(2).strip().strip('"').strip("'")
        return "action", {"tool": tool_name, "args": tool_args}

    return "unknown", None


def _extract_citations(draft: str, nodes: list[dict]) -> list[str]:
    """从 draft 中提取 DOI citations。"""
    cite_refs = re.findall(r"\[#(\d+)\]", draft)
    citations: list[str] = []
    for ref in set(cite_refs):
        idx = int(ref) - 1
        if 0 <= idx < len(nodes):
            node = nodes[idx]
            metadata = node.get("metadata", {})
            doi = metadata.get("doi", "")
            if doi and doi not in citations:
                citations.append(doi)
    return citations


async def react_agent_node(state: dict) -> dict:
    """ReAct Agent Loop: LLM 推理 + 输出解析。"""
    query = state["query"]
    scratchpad = state.get("scratchpad", "")
    iteration = state.get("iteration", 0)
    tool_call_count = state.get("tool_call_count", 0)

    # 首次调用初始化 scratchpad
    if not scratchpad:
        scratchpad = f"{_format_system_prompt()}\n\n用户问题: {query}\n"

    # 质量检查重试时追加反馈
    quality_issues = state.get("quality_issues", [])
    if quality_issues:
        scratchpad += "\n\n[你的上次回答质量不达标，请改进以下问题：]\n"
        for issue in quality_issues:
            scratchpad += f"- {issue}\n"
        scratchpad += "\n请重新回答，改进以上问题。\n"

    # 最大迭代保护
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"[react_agent] 达到最大迭代次数 {MAX_ITERATIONS}")
        draft = state.get("draft", "") or "抱歉，我无法在限定步骤内完成回答。"
        return {
            "draft": draft,
            "scratchpad": scratchpad,
            "iteration": iteration + 1,
            "_pending_action": None,
            "steps": [f"agent: MAX_ITERATIONS ({iteration})"],
        }

    # 获取 LLM provider
    context = state.get("_context")
    if context is None:
        raise ValueError("[react_agent] _context 未传入")


    # 调用 LLM
    try:
        text = await call_llm(scratchpad, context, state.get("_config"))
        if not text or not text.strip():
            raise RuntimeError("LLM 返回空内容")
    except Exception as e:
        logger.error(f"[react_agent] LLM 调用失败: {e}")
        draft = state.get("draft", "")
        if draft:
            return {
                "draft": draft,
                "scratchpad": scratchpad,
                "iteration": iteration + 1,
                "_pending_action": None,
                "steps": ["agent: LLM_FAILED (using existing draft)"],
            }
        raise RuntimeError(f"[react_agent] LLM 调用失败: {e}") from e

    text = text.strip()
    scratchpad += f"\n{text}\n"

    # 解析输出
    result_type, result_data = _parse_response(text)

    if result_type == "finish" and result_data is not None:
        draft = result_data["answer"]
        nodes = state.get("retrieved_nodes", [])
        citations = _extract_citations(draft, nodes)
        logger.info(f"[react_agent] FINISH (iter={iteration+1}, draft_len={len(draft)})")
        return {
            "draft": draft,
            "citations": citations,
            "scratchpad": scratchpad,
            "iteration": iteration + 1,
            "_pending_action": None,
            "steps": [f"agent: FINISH (iter={iteration+1}, draft_len={len(draft)})"],
        }

    if result_type == "action" and result_data is not None:
        tool_name = result_data["tool"]
        tool_args = result_data["args"]

        # 工具调用次数限制 — 直接用已有上下文生成回答
        if tool_call_count >= MAX_TOOL_CALLS:
            logger.warning(f"[react_agent] 工具调用次数达到上限 {MAX_TOOL_CALLS}，强制生成回答")
            fallback = text.strip() if text.strip() else "基于已收集的信息，我目前的回答如下。"
            if len(fallback) < 50:
                fallback = f"经过 {tool_call_count} 次工具调用后，我收集了相关信息但无法给出完整回答。"
            nodes = state.get("retrieved_nodes", [])
            citations = _extract_citations(fallback, nodes)
            return {
                "draft": fallback,
                "citations": citations,
                "scratchpad": scratchpad,
                "iteration": iteration + 1,
                "_pending_action": None,
                "steps": [f"agent: TOOL_LIMIT ({tool_call_count}, fallback draft)"],
            }

        logger.info(f"[react_agent] ACTION: {tool_name}({tool_args}) (iter={iteration+1})")
        return {
            "scratchpad": scratchpad,
            "iteration": iteration + 1,
            "_pending_action": {"tool": tool_name, "args": tool_args},
            "steps": [f"agent: ACTION {tool_name} (iter={iteration+1})"],
        }

    # 未解析出 ACTION/FINISH — 文本够长则当作直接回答
    if len(text) > 100:
        logger.info(f"[react_agent] UNKNOWN format, treating as direct answer (len={len(text)})")
        nodes = state.get("retrieved_nodes", [])
        citations = _extract_citations(text, nodes)
        return {
            "draft": text,
            "citations": citations,
            "scratchpad": scratchpad,
            "iteration": iteration + 1,
            "_pending_action": None,
            "steps": [f"agent: DIRECT_ANSWER (iter={iteration+1})"],
        }

    # 短文本未解析 → 如果快到迭代上限，强制当作回答；否则要求重新格式化
    if iteration + 1 >= MAX_ITERATIONS - 1:
        logger.warning(f"[react_agent] NO_PARSE 且接近迭代上限，强制输出")
        return {
            "draft": text.strip() or "抱歉，无法生成完整回答。",
            "scratchpad": scratchpad,
            "iteration": iteration + 1,
            "_pending_action": None,
            "steps": [f"agent: NO_PARSE_FORCE (iter={iteration+1})"],
        }

    scratchpad += "\n[请使用 THOUGHT/ACTION 或 THOUGHT/FINISH 格式回答]\n"
    return {
        "scratchpad": scratchpad,
        "iteration": iteration + 1,
        "_pending_action": None,
        "steps": [f"agent: NO_PARSE (iter={iteration+1})"],
    }
