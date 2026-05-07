"""
Synthesize node — 多源聚合生成。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from astrbot.api import logger


class SynthesizeInput(BaseModel):
    """Synthesize 节点输入。"""
    query: str = Field(..., min_length=1)
    retrieved_nodes: list[dict] = Field(default_factory=list)
    graph_entities: list[dict] = Field(default_factory=list)
    graph_relations: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_none_to_empty(self) -> "SynthesizeInput":
        if self.retrieved_nodes is None:
            self.retrieved_nodes = []
        if self.graph_entities is None:
            self.graph_entities = []
        if self.graph_relations is None:
            self.graph_relations = []
        return self


class SynthesizeOutput(BaseModel):
    """Synthesize 节点输出。"""
    draft: str
    citations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def draft_not_empty(self) -> "SynthesizeOutput":
        if not self.draft or not self.draft.strip():
            raise ValueError("draft cannot be empty")
        return self


SYSTEM_PROMPT = """你是一个学术论文问答助手。基于检索结果和知识图谱背景知识，回答用户问题。

要求：
1. 回答准确，基于提供的上下文
2. 引用来源时使用 [#n] 格式
3. 如果知识图谱提供了实体关系，将其作为背景知识帮助理解
4. 如果信息不足，明确指出，不要编造
5. 保持回答简洁有条理
"""


def _build_context(input_data: SynthesizeInput) -> str:
    """构建包含图谱结构化知识的上下文字符串。"""
    parts = []

    # 1. 检索结果
    if input_data.retrieved_nodes:
        parts.append("[检索结果]")
        for i, node in enumerate(input_data.retrieved_nodes, 1):
            text = node.get("text", "")
            score = node.get("score", 0.0)
            parts.append(f"[#{i}] (score={score:.3f})\n{text[:300]}")
    else:
        parts.append("[检索结果]\n（无相关检索结果）")

    # 2. 知识图谱背景
    entities = input_data.graph_entities
    relations = input_data.graph_relations
    if entities or relations:
        parts.append("\n[知识图谱关系]")
        if entities:
            parts.append("实体:")
            for e in entities[:5]:
                parts.append(f"  - {e.get('name', '')} ({e.get('type', '')})")
        if relations:
            parts.append("关系:")
            for r in relations[:5]:
                parts.append(
                    f"  - {r.get('head', '')} → [{r.get('relation', '')}] → {r.get('tail', '')}"
                )
    else:
        parts.append("\n[知识图谱关系]\n（无图谱数据）")

    return "\n\n".join(parts)


async def synthesize_node(state: dict) -> dict:
    """
    LangGraph 节点：多源聚合生成。

    将检索结果和图谱数据拼接为上下文，调用 LLM 生成回答。

    Args:
        state: AgenticRAGState（读取 retrieved_nodes, graph_entities, graph_relations, query, _context）

    Returns:
        更新 state 的 dict（draft, citations, steps）
    """
    try:
        input_data = SynthesizeInput(
            query=state["query"],
            retrieved_nodes=state.get("retrieved_nodes", []),
            graph_entities=state.get("graph_entities", []),
            graph_relations=state.get("graph_relations", []),
        )
    except ValueError:
        raise

    logger.debug(
        f"[synthesize] 开始: {len(input_data.retrieved_nodes)} nodes, "
        f"{len(input_data.graph_entities)} entities, {len(input_data.graph_relations)} relations"
    )

    context_str = _build_context(input_data)

    draft = ""
    citations: list[str] = []

    context = state.get("_context")
    if context is None:
        logger.error("[synthesize] _context 未传入，无法调用 LLM")
        return {
            "draft": "",
            "citations": [],
            "steps": ["synthesize: FAILED (no context)"],
        }

    # 获取 LLM provider（统一 4 步解析）
    from provider.llm_utils import call_llm

    try:
        quality_feedback = ""
        quality_issues = state.get("quality_issues", [])
        if quality_issues:
            quality_feedback = (
                "\n\n[重要：上次回答质量不达标，请改进以下问题]\n"
                + "\n".join(f"- {issue}" for issue in quality_issues)
                + "\n请提供更详细、有引用支撑的回答。"
            )

        prompt = f"{SYSTEM_PROMPT}\n\n用户问题: {input_data.query}\n\n{context_str}{quality_feedback}\n\n请基于以上内容回答用户问题："

        draft = await call_llm(prompt, context, state.get("_config"))

        if not draft or not draft.strip():
            raise RuntimeError("LLM 返回空内容")

        draft = draft.strip()

        # 简单引用提取：找 [#n] 格式
        import re
        cite_refs = re.findall(r"\[#(\d+)\]", draft)
        for ref in set(cite_refs):
            idx = int(ref) - 1
            if idx < len(input_data.retrieved_nodes):
                node = input_data.retrieved_nodes[idx]
                metadata = node.get("metadata", {})
                doi = metadata.get("doi", "")
                if doi and doi not in citations:
                    citations.append(doi)

        logger.info(f"[synthesize] 生成完成: draft 长度={len(draft)}, citations={len(citations)}")

        output = SynthesizeOutput(draft=draft, citations=citations)

    except Exception as e:
        logger.error(f"[synthesize] LLM 生成失败: {e}")
        raise RuntimeError(f"[synthesize] 生成失败: {e}") from e

    return {
        "draft": output.draft,
        "citations": output.citations,
        "steps": [f"synthesize: OK (draft_len={len(output.draft)}, citations={len(output.citations)})"],
    }
