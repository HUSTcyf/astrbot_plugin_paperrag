"""Tests for vector_search node."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.mark.asyncio
async def test_vector_search_no_context():
    """无 context 时返回空。"""
    from agentic_rag.nodes.vector_search import vector_search_node
    state = {"query": "attention", "_context": None}
    result = await vector_search_node(state)
    assert result["retrieved_nodes"] == []
    assert "FAILED" in result["steps"][0]


@pytest.mark.asyncio
async def test_vector_search_engine_not_ready():
    """engine 未就绪时返回空。"""
    from agentic_rag.nodes.vector_search import vector_search_node
    mock_ctx = MagicMock()
    mock_ctx.config = {}
    mock_ctx.provider_manager = MagicMock()

    with patch("agentic_rag.engine_utils.get_engine", return_value=None):
        state = {"query": "attention", "_context": mock_ctx}
        result = await vector_search_node(state)
        assert result["retrieved_nodes"] == []


@pytest.mark.asyncio
async def test_vector_search_success():
    """正常返回。"""
    from agentic_rag.nodes.vector_search import vector_search_node

    class FakeNode:
        text = "Attention is all you need"
        metadata = {"chunk_id": "c1"}

    class FakeResult:
        nodes = [FakeNode(), FakeNode()]
        scores = [0.9, 0.8]

    mock_engine = MagicMock()
    mock_engine.search = AsyncMock(return_value=FakeResult())
    mock_ctx = MagicMock()
    mock_ctx.config = {}

    with patch("agentic_rag.engine_utils.get_engine", return_value=mock_engine):
        state = {"query": "attention", "_context": mock_ctx}
        result = await vector_search_node(state)
        assert len(result["retrieved_nodes"]) == 2
        assert result["retrieved_nodes"][0]["text"] == "Attention is all you need"


@pytest.mark.asyncio
async def test_vector_search_timeout():
    """超时降级。"""
    import agentic_rag.nodes.vector_search as vs_module
    from agentic_rag.nodes.vector_search import vector_search_node

    original_timeout = vs_module.SEARCH_TIMEOUT
    vs_module.SEARCH_TIMEOUT = 1

    async def slow_search(*args, **kwargs):
        import asyncio
        await asyncio.sleep(20)

    mock_engine = MagicMock()
    mock_engine.search = slow_search
    mock_ctx = MagicMock()
    mock_ctx.config = {}

    try:
        with patch("agentic_rag.engine_utils.get_engine", return_value=mock_engine):
            state = {"query": "attention", "_context": mock_ctx}
            result = await vector_search_node(state)
            assert result["retrieved_nodes"] == []
            assert "TIMEOUT" in result["steps"][0]
    finally:
        vs_module.SEARCH_TIMEOUT = original_timeout


@pytest.mark.asyncio
async def test_vector_search_returns_none():
    """返回 None 时不抛异常。"""
    from agentic_rag.nodes.vector_search import vector_search_node

    mock_engine = MagicMock()
    mock_engine.search = AsyncMock(return_value=None)
    mock_ctx = MagicMock()
    mock_ctx.config = {}

    with patch("agentic_rag.engine_utils.get_engine", return_value=mock_engine):
        state = {"query": "attention", "_context": mock_ctx}
        result = await vector_search_node(state)
        assert result["retrieved_nodes"] == []
