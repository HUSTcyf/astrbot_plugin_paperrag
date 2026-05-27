"""Tests for the debate node (Ideator ↔ Critic)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from idea.nodes.debate import (
    debate_node,
    _format_ideas_text,
    _format_scores_text,
    _build_history_section,
)
from provider.llm_utils import parse_json_response as _parse_json
from idea.agentic_workflow import route_after_critique
from idea.agentic_workflow import route_after_debate


class TestFormatHelpers:
    def test_format_ideas_text(self):
        ideas = [
            {"title": "Idea A", "description": "Desc A", "novelty": "Novel A", "methodology": "Method A"},
            {"title": "Idea B", "description": "Desc B", "novelty": "Novel B", "methodology": "Method B"},
        ]
        text = _format_ideas_text(ideas)
        assert "[1] Idea A" in text
        assert "[2] Idea B" in text

    def test_format_scores_text(self):
        scores = [
            {"title": "Idea A", "score": 8, "issues": ["too vague"]},
            {"title": "Idea B", "score": 4, "issues": ["not novel", "unclear"]},
        ]
        text = _format_scores_text(scores)
        assert "8/10" in text
        assert "4/10" in text
        assert "not novel" in text

    def test_format_scores_empty(self):
        assert _format_scores_text([]) == "（无详细评分）"

    def test_build_history_section_empty(self):
        assert _build_history_section([]) == ""

    def test_build_history_section_with_entries(self):
        history = ["[Round 1] Ideator: defended idea A"]
        text = _build_history_section(history)
        assert "前几轮辩论记录" in text
        assert "Round 1" in text


class TestParseJson:
    def test_plain_json(self):
        result = _parse_json('{"defense": "ok", "ideas": []}')
        assert result["defense"] == "ok"

    def test_json_in_code_block(self):
        result = _parse_json('```json\n{"defense": "ok", "ideas": []}\n```')
        assert result["defense"] == "ok"

    def test_invalid_json(self):
        result = _parse_json("not json at all")
        assert result is None


class TestDebateNode:
    @pytest.mark.asyncio
    async def test_all_good_skips_debate(self):
        """All ideas score >= 7 → skip debate, phase='done'."""
        state = {
            "ideas": [{"title": "A"}],
            "idea_scores": [{"title": "A", "score": 8, "issues": []}],
            "critique": "Good",
            "context_data": {},
            "debate_round": 0,
            "debate_history": [],
            "_max_debate_rounds": 2,
            "_context": MagicMock(),
        }
        result = await debate_node(state)
        assert result["phase"] == "done"
        assert "ALL_GOOD" in result["steps"][0]

    @pytest.mark.asyncio
    async def test_max_rounds_exhausted(self):
        """Debate rounds exhausted → route to refine."""
        state = {
            "ideas": [{"title": "A"}],
            "idea_scores": [{"title": "A", "score": 4, "issues": ["weak"]}],
            "critique": "Not good",
            "context_data": {},
            "debate_round": 2,
            "debate_history": ["[Round 1] ...", "[Round 2] ..."],
            "_max_debate_rounds": 2,
            "_context": MagicMock(),
        }
        result = await debate_node(state)
        assert result["phase"] == "refine"
        assert "MAX_ROUNDS" in result["steps"][0]

    @pytest.mark.asyncio
    async def test_no_provider_skips_debate(self):
        """No LLM provider → skip debate, route to refine."""
        mock_ctx = MagicMock()
        mock_ctx.provider_manager = MagicMock()
        mock_ctx.provider_manager.inst_map = {}

        state = {
            "ideas": [{"title": "A"}],
            "idea_scores": [{"title": "A", "score": 4, "issues": ["weak"]}],
            "critique": "Not good",
            "context_data": {},
            "debate_round": 0,
            "debate_history": [],
            "_max_debate_rounds": 2,
            "_context": mock_ctx,
        }

        with patch("provider.llm_utils.call_llm_json", new_callable=AsyncMock, side_effect=RuntimeError("无可用 LLM provider")):
            result = await debate_node(state)

        assert result["phase"] == "refine"
        assert "LLM_FAILED" in result["steps"][0]

    @pytest.mark.asyncio
    async def test_successful_debate_round(self):
        """LLM returns valid defense + modified ideas → route back to critique."""
        llm_result = {
            "defense": "I disagree with the critique on Idea A. Here's why...",
            "ideas": [
                {
                    "title": "Improved Idea A",
                    "description": "Better description",
                    "novelty": "More novel",
                    "methodology": "Clearer method",
                    "potential_challenges": ["challenge1"],
                    "related_work": ["ref1"],
                    "feasibility": 0.8,
                    "inspiration_sources": ["src1"],
                }
            ]
        }

        state = {
            "topic": "transformer efficiency",
            "ideas": [{"title": "Idea A", "description": "old desc", "novelty": "old", "methodology": "old"}],
            "idea_scores": [{"title": "Idea A", "score": 4, "issues": ["not novel"]}],
            "critique": "Ideas are not novel enough",
            "context_data": {"fused_context": "some context"},
            "debate_round": 0,
            "debate_history": [],
            "_max_debate_rounds": 2,
            "_context": MagicMock(),
        }

        with patch("idea.nodes.debate.call_llm_json", new_callable=AsyncMock, return_value=llm_result):
            result = await debate_node(state)

        assert result["phase"] == "critique"
        assert result["debate_round"] == 1
        assert len(result["ideas"]) == 1
        assert result["ideas"][0]["title"] == "Improved Idea A"
        assert len(result["debate_history"]) == 1
        assert "Round 1" in result["debate_history"][0]

    @pytest.mark.asyncio
    async def test_llm_returns_unparseable_json(self):
        """LLM returns non-JSON → fallback to refine."""
        state = {
            "topic": "test",
            "ideas": [{"title": "A"}],
            "idea_scores": [{"title": "A", "score": 4, "issues": ["weak"]}],
            "critique": "Not good",
            "context_data": {},
            "debate_round": 0,
            "debate_history": [],
            "_max_debate_rounds": 2,
            "_context": MagicMock(),
        }

        with patch("idea.nodes.debate.call_llm_json", new_callable=AsyncMock, return_value=None):
            result = await debate_node(state)

        assert result["phase"] == "refine"
        assert "JSON_PARSE_FAILED" in result["steps"][0]

    @pytest.mark.asyncio
    async def test_debate_history_carried_over(self):
        """Debate history accumulates across rounds."""
        llm_result = {
            "defense": "Second defense",
            "ideas": [{"title": "A", "description": "d", "novelty": "n", "methodology": "m",
                       "potential_challenges": [], "related_work": [], "feasibility": 0.8,
                       "inspiration_sources": []}]
        }

        state = {
            "topic": "test",
            "ideas": [{"title": "A"}],
            "idea_scores": [{"title": "A", "score": 4, "issues": ["weak"]}],
            "critique": "Still not good",
            "context_data": {},
            "debate_round": 1,
            "debate_history": ["[Round 1] Ideator: first defense"],
            "_max_debate_rounds": 2,
            "_context": MagicMock(),
        }

        with patch("idea.nodes.debate.call_llm_json", new_callable=AsyncMock, return_value=llm_result):
            result = await debate_node(state)

        assert result["debate_round"] == 2
        assert len(result["debate_history"]) == 2
        assert "Round 1" in result["debate_history"][0]
        assert "Round 2" in result["debate_history"][1]


class TestWorkflowRouting:
    """Test routing logic that involves the debate node."""

    def test_route_after_critique_to_debate(self):
        """critique → debate when phase=refine and debate_rounds left."""

        state = {"phase": "refine", "debate_round": 0, "_max_debate_rounds": 2}
        assert route_after_critique(state) == "debate"

    def test_route_after_critique_to_refine(self):
        """critique → refine when debate rounds exhausted."""

        state = {"phase": "refine", "debate_round": 2, "_max_debate_rounds": 2}
        assert route_after_critique(state) == "refine"

    def test_route_after_critique_to_save(self):
        """critique → save when ideas are good (phase=done)."""

        state = {"phase": "done", "debate_round": 0, "_max_debate_rounds": 2}
        assert route_after_critique(state) == "save"

    def test_route_after_debate_to_critique(self):
        """debate → critique for re-evaluation."""

        state = {"phase": "critique"}
        assert route_after_debate(state) == "critique"

    def test_route_after_debate_to_refine(self):
        """debate → refine when phase=refine (e.g., LLM failed)."""

        state = {"phase": "refine"}
        assert route_after_debate(state) == "refine"

    def test_route_after_debate_to_save(self):
        """debate → save when phase=done (e.g., all good)."""

        state = {"phase": "done"}
        assert route_after_debate(state) == "save"
