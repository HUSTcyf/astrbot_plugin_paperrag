"""Tests for final_output node."""
import pytest
from agentic_rag.nodes.final_output import final_output_node, _format_answer


def test_format_answer_with_citations():
    result = _format_answer("Answer body", ["doi:10.1234/test"])
    assert "Answer body" in result
    assert "doi:10.1234/test" in result
    assert "[1]" in result


def test_format_answer_without_citations():
    result = _format_answer("Answer only", [])
    assert result == "Answer only"


@pytest.mark.asyncio
async def test_final_output_success():
    state = {"draft": "The answer", "citations": ["doi:1", "doi:2"]}
    result = await final_output_node(state)
    assert "The answer" in result["final_answer"]
    assert "doi:1" in result["final_answer"]


@pytest.mark.asyncio
async def test_final_output_empty_draft():
    state = {"draft": "", "citations": []}
    with pytest.raises(ValueError):
        await final_output_node(state)
