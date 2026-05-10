"""Tests for the ReAct Tool-Using Agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag.react_agent import (
    _parse_response,
    _extract_citations,
    react_agent_node,
    _format_system_prompt,
)
from agentic_rag.react_tools import react_tool_executor_node
from agentic_rag.react_state import MAX_TOOL_CALLS, MAX_ITERATIONS


class TestParseResponse:
    def test_finish_parsed(self):
        text = "THOUGHT: I have enough info.\nFINISH: The answer is 42."
        rtype, data = _parse_response(text)
        assert rtype == "finish"
        assert data is not None
        assert "42" in data["answer"]

    def test_action_parsed(self):
        text = "THOUGHT: Let me search.\nACTION: vector_search(attention mechanism)"
        rtype, data = _parse_response(text)
        assert rtype == "action"
        assert data is not None
        assert data["tool"] == "vector_search"
        assert data["args"] == "attention mechanism"

    def test_finish_after_action(self):
        """When both ACTION and FINISH present, take the last one."""
        text = "THOUGHT: search\nACTION: vector_search(query1)\n\nOBSERVATION: ...\n\nTHOUGHT: done\nFINISH: final answer"
        rtype, data = _parse_response(text)
        assert rtype == "finish"

    def test_action_after_finish(self):
        """When FINISH comes before ACTION, take the ACTION (it's last)."""
        text = "FINISH: early answer\n\nTHOUGHT: wait, let me check\nACTION: graph_search(entities)"
        rtype, data = _parse_response(text)
        assert rtype == "action"
        assert data["tool"] == "graph_search"

    def test_unknown_format(self):
        text = "This is just some text without any action or finish marker."
        rtype, data = _parse_response(text)
        assert rtype == "unknown"
        assert data is None

    def test_action_with_quoted_args(self):
        text = 'ACTION: vector_search("attention mechanism")'
        rtype, data = _parse_response(text)
        assert rtype == "action"
        assert data["args"] == "attention mechanism"


class TestExtractCitations:
    def test_extracts_doi(self):
        draft = "According to [#1], the method works."
        nodes = [{"metadata": {"doi": "10.1234/test"}}]
        citations = _extract_citations(draft, nodes)
        assert citations == ["10.1234/test"]

    def test_no_citations(self):
        draft = "No references here."
        nodes = []
        citations = _extract_citations(draft, nodes)
        assert citations == []

    def test_dedup_citations(self):
        draft = "[#1] and again [#1]"
        nodes = [{"metadata": {"doi": "10.1234/test"}}]
        citations = _extract_citations(draft, nodes)
        assert len(citations) == 1


class TestReactAgentNode:
    @pytest.mark.asyncio
    async def test_finish_on_first_call(self):
        """Agent returns FINISH immediately without tool calls."""
        state = {
            "query": "What is attention?",
            "scratchpad": "",
            "_context": MagicMock(),
            "_config": {},
            "top_k": 5,
            "iteration": 0,
            "tool_call_count": 0,
        }

        with patch("agentic_rag.react_agent.call_llm", new_callable=AsyncMock, return_value="THOUGHT: I know this.\nFINISH: The answer is attention."):
            result = await react_agent_node(state)

        assert result["draft"] == "The answer is attention."
        assert result["_pending_action"] is None
        assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_action_dispatch(self):
        """Agent returns ACTION, setting _pending_action."""
        state = {
            "query": "What is a transformer?",
            "scratchpad": "",
            "_context": MagicMock(),
            "_config": {},
            "top_k": 5,
            "iteration": 0,
            "tool_call_count": 0,
        }

        with patch("agentic_rag.react_agent.call_llm", new_callable=AsyncMock, return_value="THOUGHT: Need to search.\nACTION: vector_search(transformer architecture)"):
            result = await react_agent_node(state)

        assert result["_pending_action"] is not None
        assert result["_pending_action"]["tool"] == "vector_search"
        assert result["_pending_action"]["args"] == "transformer architecture"
        assert "draft" not in result or result.get("draft") is None

    @pytest.mark.asyncio
    async def test_max_iterations_guard(self):
        """Agent hits max iterations and forces a draft."""
        state = {
            "query": "test",
            "scratchpad": "some scratchpad content",
            "_context": MagicMock(),
            "_config": {},
            "iteration": MAX_ITERATIONS,
            "tool_call_count": 0,
        }

        result = await react_agent_node(state)
        assert result["draft"] is not None
        assert result["iteration"] == MAX_ITERATIONS + 1

    @pytest.mark.asyncio
    async def test_quality_feedback_appended(self):
        """Agent appends quality feedback to scratchpad on retry."""
        state = {
            "query": "test",
            "scratchpad": "SYSTEM PROMPT\n\n用户问题: test\n",
            "_context": MagicMock(),
            "_config": {},
            "iteration": 2,
            "tool_call_count": 0,
            "quality_issues": ["回答过短", "无引用"],
            "retry_count": 1,
        }

        with patch("agentic_rag.react_agent.call_llm", new_callable=AsyncMock, return_value="THOUGHT: Fix the issues.\nFINISH: Better answer with [#1] citation."):
            result = await react_agent_node(state)

        # The scratchpad should contain the quality feedback
        assert "回答过短" in result["scratchpad"]
        assert result["draft"] is not None


class TestReactToolExecutor:
    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Unknown tool returns error observation."""
        state = {
            "scratchpad": "SYSTEM...\nTHOUGHT: ...\n",
            "_pending_action": {"tool": "unknown_tool", "args": "test"},
            "_context": MagicMock(),
            "_config": {},
            "top_k": 5,
            "tool_call_count": 0,
        }

        result = await react_tool_executor_node(state)
        assert "未知工具" in result["scratchpad"]
        assert result["_pending_action"] is None

    @pytest.mark.asyncio
    async def test_no_action(self):
        """No pending action returns NO_ACTION."""
        state = {
            "scratchpad": "",
            "_context": MagicMock(),
            "_config": {},
            "top_k": 5,
        }

        result = await react_tool_executor_node(state)
        assert any("NO_ACTION" in s for s in result["steps"])

    @pytest.mark.asyncio
    async def test_vector_search_tool(self):
        """vector_search tool calls engine and returns nodes + observation."""
        mock_engine = MagicMock()
        mock_engine.search = AsyncMock(return_value=[
            {"text": "Attention is all you need", "score": 0.95, "metadata": {"file_name": "attention.pdf"}},
        ])

        state = {
            "scratchpad": "SYSTEM...\nTHOUGHT: ...\n",
            "_pending_action": {"tool": "vector_search", "args": "attention"},
            "_context": MagicMock(),
            "_config": {},
            "top_k": 5,
            "tool_call_count": 0,
        }

        with patch("agentic_rag.react_tools.get_engine", return_value=mock_engine):
            result = await react_tool_executor_node(state)

        assert result["tool_call_count"] == 1
        assert len(result["retrieved_nodes"]) == 1
        assert "OBSERVATION" in result["scratchpad"]
        assert "attention" in result["scratchpad"].lower()

    @pytest.mark.asyncio
    async def test_graph_search_tool(self):
        """graph_search tool calls graph engine and returns entities/relations."""
        mock_graph_engine = MagicMock()
        mock_graph_engine.search = AsyncMock(return_value={
            "type": "hybrid",
            "answer": "Transformer uses attention",
            "entities": [{"name": "Transformer", "type": "Model"}],
            "triplets": [{"head": "Transformer", "relation": "uses", "tail": "Attention"}],
            "sources": [{"text": "source text", "score": 0.9, "metadata": {}}],
        })

        state = {
            "scratchpad": "SYSTEM...\nTHOUGHT: ...\n",
            "_pending_action": {"tool": "graph_search", "args": "transformer"},
            "_context": MagicMock(),
            "_config": {},
            "top_k": 5,
            "tool_call_count": 0,
        }

        with patch("agentic_rag.react_tools.get_graph_engine", AsyncMock(return_value=mock_graph_engine)):
            result = await react_tool_executor_node(state)

        assert result["tool_call_count"] == 1
        assert len(result["graph_entities"]) == 1
        assert len(result["graph_relations"]) == 1
        assert "OBSERVATION" in result["scratchpad"]


class TestSystemPrompt:
    def test_contains_tool_descriptions(self):
        prompt = _format_system_prompt()
        assert "vector_search" in prompt
        assert "graph_search" in prompt
        assert "THOUGHT" in prompt
        assert "ACTION" in prompt
        assert "FINISH" in prompt
        assert str(MAX_TOOL_CALLS) in prompt
