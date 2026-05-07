"""
Final Output node — 格式化输出。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from astrbot.api import logger


class FinalOutputInput(BaseModel):
    """FinalOutput 节点输入。"""
    draft: str = ""
    citations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_draft(self) -> "FinalOutputInput":
        if not self.draft or not self.draft.strip():
            raise ValueError("draft cannot be empty")
        return self

    @model_validator(mode="after")
    def normalize_citations(self) -> "FinalOutputInput":
        if self.citations is None:
            self.citations = []
        return self


class FinalOutputOutput(BaseModel):
    """FinalOutput 节点输出。"""
    final_answer: str


def _format_answer(draft: str, citations: list[str], retrieved_nodes: list[dict] | None = None) -> str:
    """将草稿和引用格式化为最终答案，与普通 RAG 格式对齐。"""
    answer = draft.strip()

    # 检索结果（与普通 RAG 格式一致）
    if retrieved_nodes:
        answer += "\n\n**📚 检索结果**\n\n"
        for i, node in enumerate(retrieved_nodes, 1):
            metadata = node.get("metadata", {}) or {}
            filename = metadata.get("file_name", "unknown")
            chunk_index = metadata.get("chunk_index", 0)
            text = node.get("text", "")[:200]
            score = node.get("score", 0.0)
            arxiv_url = metadata.get("arxiv_url", "")

            if arxiv_url:
                ref_text = f"[{filename}]({arxiv_url})"
            else:
                ref_text = f"**{filename}**"

            answer += f"[{i}] {ref_text} (chunk #{chunk_index}, score={score:.3f})\n"
            answer += f"> {text}...\n\n"

    # 参考文献
    if citations:
        answer += "\n**参考文献**\n"
        for i, doi in enumerate(citations, 1):
            answer += f"[{i}] {doi}\n"

    return answer


async def final_output_node(state: dict) -> dict:
    """
    LangGraph 节点：格式化输出。

    将 draft 和 citations 格式化为最终答案。

    Args:
        state: AgenticRAGState（读取 draft, citations）

    Returns:
        更新 state 的 dict（final_answer, steps）
    """
    try:
        input_data = FinalOutputInput(
            draft=state["draft"],
            citations=state.get("citations", []),
        )
    except ValueError:
        raise

    logger.debug(f"[final_output] 格式化: draft_len={len(input_data.draft)}, citations={len(input_data.citations)}")

    final_answer = _format_answer(
        input_data.draft,
        input_data.citations,
        state.get("retrieved_nodes"),
    )
    output = FinalOutputOutput(final_answer=final_answer)

    return {
        "final_answer": output.final_answer,
        "steps": [f"final_output: OK (answer_len={len(output.final_answer)})"],
    }
