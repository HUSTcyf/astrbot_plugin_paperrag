"""Tests for graph_search node."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agentic_rag.nodes.graph_search import graph_search_node


@pytest.mark.asyncio
async def test_graph_search_skip_when_weight_zero():
    """graph_weight=0 时跳过。"""
    state = {"query": "attention", "graph_weight": 0.0, "_context": None}
    result = await graph_search_node(state)
    assert result["graph_entities"] == []
    assert "SKIPPED" in result["steps"][0]


@pytest.mark.asyncio
async def test_graph_search_no_context():
    """无 context 时跳过。"""
    state = {"query": "attention", "graph_weight": 0.3, "_context": None}
    result = await graph_search_node(state)
    assert result["graph_entities"] == []
    assert "SKIPPED" in result["steps"][0]


@pytest.mark.asyncio
async def test_graph_search_engine_not_ready():
    """engine 未就绪时跳过。"""
    mock_ctx = MagicMock()
    with patch("agentic_rag.engine_utils.get_graph_engine", AsyncMock(return_value=None)):
        state = {"query": "attention", "graph_weight": 0.3, "_context": mock_ctx}
        result = await graph_search_node(state)
        assert result["graph_entities"] == []


@pytest.mark.asyncio
async def test_graph_search_success():
    """正常返回。"""
    mock_engine = MagicMock()
    mock_engine.search = AsyncMock(return_value={
        "entities": [{"name": "Attention", "type": "Method"}],
        "triplets": [{"head": "Attention", "relation": "used_in", "tail": "Transformer"}],
        "sources": [{"text": "Attention mechanism", "score": 0.9}],
    })
    mock_ctx = MagicMock()

    with patch("agentic_rag.engine_utils.get_graph_engine", AsyncMock(return_value=mock_engine)):
        state = {"query": "attention", "graph_weight": 0.3, "_context": mock_ctx}
        result = await graph_search_node(state)
        assert len(result["graph_entities"]) == 1
        assert len(result["graph_relations"]) == 1


@pytest.mark.asyncio
async def test_graph_search_returns_none():
    """返回 None 时不抛异常。"""
    mock_engine = MagicMock()
    mock_engine.search = AsyncMock(return_value=None)
    mock_ctx = MagicMock()

    with patch("agentic_rag.engine_utils.get_graph_engine", AsyncMock(return_value=mock_engine)):
        state = {"query": "attention", "graph_weight": 0.3, "_context": mock_ctx}
        result = await graph_search_node(state)
        assert result["graph_entities"] == []
