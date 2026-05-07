"""Tests for synthesize node."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agentic_rag.nodes.synthesize import synthesize_node


@pytest.mark.asyncio
async def test_synthesize_no_context():
    """无 context 时失败。"""
    state = {"query": "test", "_context": None, "retrieved_nodes": [], "graph_entities": [], "graph_relations": []}
    result = await synthesize_node(state)
    assert result["draft"] == ""


@pytest.mark.asyncio
async def test_synthesize_no_provider():
    """无可用 provider 时失败。"""
    with patch("provider.llm_utils.call_llm", new_callable=AsyncMock, side_effect=RuntimeError("无可用 LLM provider")):
        mock_ctx = MagicMock()
        state = {"query": "test", "_context": mock_ctx, "retrieved_nodes": [], "graph_entities": [], "graph_relations": []}
        with pytest.raises(RuntimeError, match="生成失败"):
            await synthesize_node(state)


@pytest.mark.asyncio
async def test_synthesize_success():
    """正常生成。"""
    with patch("provider.llm_utils.call_llm", new_callable=AsyncMock, return_value="Answer: Attention mechanism"):
        mock_ctx = MagicMock()
        state = {
            "query": "What is attention?",
            "_context": mock_ctx,
            "retrieved_nodes": [{"text": "Attention is...", "score": 0.9}],
            "graph_entities": [],
            "graph_relations": [],
        }
        result = await synthesize_node(state)
        assert "Attention" in result["draft"]
