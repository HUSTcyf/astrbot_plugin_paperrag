"""
Router node — 查询分类 + 图谱权重决策。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from astrbot.api import logger


class RouterInput(BaseModel):
    query: str = Field(..., min_length=1)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("query must be a string")
        return v.strip()


class RouterOutput(BaseModel):
    query_type: Literal["fact", "comparison", "review", "citation"] = "fact"
    graph_weight: float = Field(default=0.0, ge=0.0, le=1.0)


_QUERY_TYPE_PATTERNS = {
    "comparison": ["比较", "对比", "difference", "compared", " vs ", " versus ", "与...区别", "和...区别", "哪个更好", "优势", "劣势", "优于", "差于"],
    "review": ["综述", "发展", "演进", "历史", "趋势", "survey", "overview", "state of the art", "sota", "总结", "概括"],
    "citation": ["引用", "被引", "溯源", "奠基", "cite", "cited by", "reference", "参考", "引用这篇", "谁是开创者"],
}


def _classify_by_keywords(query: str) -> tuple[str, float]:
    q_lower = query.lower()
    for qtype, keywords in _QUERY_TYPE_PATTERNS.items():
        if any(kw in q_lower for kw in keywords):
            weight_map = {"comparison": 0.6, "review": 0.3, "citation": 0.8}
            return qtype, weight_map.get(qtype, 0.0)
    return "fact", 0.0


async def router_node(state: dict) -> dict:
    """查询分类 + 图谱权重决策。"""
    query_raw = state.get("query", "")
    if not query_raw or not query_raw.strip():
        raise ValueError("query cannot be empty")

    query = query_raw.strip()
    logger.debug(f"[router] 分类查询: {query}")

    context = state.get("_context")

    from provider.llm_utils import get_llm_provider
    provider = get_llm_provider(context)

    query_type = "fact"
    graph_weight = 0.0

    if provider is not None:
        try:
            from provider.llm_utils import call_llm
            prompt = f"请判断以下查询的类型，只回答一个词（fact/comparison/review/citation）：\n\n查询: {query}\n\n类型: "
            text = await call_llm(prompt, context, temperature=0.0, max_tokens=20)
            text = text.strip().lower()

            if any(t in text for t in ["comparison", "compare", "比较", "对比"]):
                query_type = "comparison"; graph_weight = 0.6
            elif any(t in text for t in ["review", "综述", "survey", "overview", "趋势"]):
                query_type = "review"; graph_weight = 0.3
            elif any(t in text for t in ["citation", "引用", "reference", "cited", "参考"]):
                query_type = "citation"; graph_weight = 0.8
            else:
                query_type = "fact"; graph_weight = 0.0
        except Exception as e:
            logger.warning(f"[router] LLM分类失败，fallback: {e}")
            query_type, graph_weight = _classify_by_keywords(query)
    else:
        query_type, graph_weight = _classify_by_keywords(query)

    logger.debug(f"[router] 结果: query_type={query_type}, graph_weight={graph_weight}")

    return {
        "query_type": query_type,
        "graph_weight": graph_weight,
        "steps": [f"router: {query_type} (graph_weight={graph_weight})"],
    }
