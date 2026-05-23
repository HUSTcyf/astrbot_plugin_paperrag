"""Tests for router node."""
from unittest.mock import MagicMock

import pytest
from agentic_rag.nodes.router import _classify_by_keywords, router_node


class TestClassifyKeywords:
    def test_fact(self):
        qtype, weight = _classify_by_keywords("What is Attention?")
        assert qtype == "fact"
        assert weight == 0.0

    def test_comparison(self):
        qtype, weight = _classify_by_keywords("比较 ViT 和 CNN 的优劣")
        assert qtype == "comparison"
        assert weight == 0.6

    def test_citation(self):
        qtype, weight = _classify_by_keywords("这篇论文引用了哪些方法")
        assert qtype == "citation"
        assert weight == 0.8


@pytest.mark.asyncio
async def test_router_fallback_to_keywords():
    """无 LLM provider 时 fallback 到关键词规则。"""
    mock_ctx = MagicMock()
    mock_ctx.provider_manager = None
    state = {"query": "比较 ViT 和 CNN", "_context": mock_ctx}
    result = await router_node(state)
    assert result["query_type"] == "comparison"
    assert result["graph_weight"] == 0.6


@pytest.mark.asyncio
async def test_router_empty_query():
    """空 query 抛出 ValueError。"""
    state = {"query": "", "_context": None}
    with pytest.raises(ValueError):
        await router_node(state)


@pytest.mark.asyncio
async def test_router_whitespace_query():
    """空白 query 抛出 ValueError。"""
    state = {"query": "   ", "_context": None}
    with pytest.raises(ValueError):
        await router_node(state)


@pytest.mark.asyncio
async def test_router_non_string_query():
    """非字符串 query 抛出 TypeError。"""
    state = {"query": object(), "_context": None}
    with pytest.raises(TypeError, match="query must be a string"):
        await router_node(state)
