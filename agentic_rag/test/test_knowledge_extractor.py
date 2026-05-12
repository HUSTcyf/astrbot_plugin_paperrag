"""
Unit tests for knowledge_extractor — helpers, filtering logic, wiki writing.
Does NOT require a live LLM.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agentic_rag.knowledge_extractor import (
    _format_sources_for_prompt,
    _parse_json,
    _source_filenames,
    _coerce_page_type,
    _log_context_overflow,
    _MAX_SOURCE_TOKENS,
    _MAX_SOURCES_IN_PROMPT,
)
from rag.token_utils import count_tokens, truncate_text_to_tokens


# ---------------------------------------------------------------------------
# _format_sources_for_prompt
# ---------------------------------------------------------------------------

def test_format_sources_basic():
    sources = [
        {"text": "Short text.", "metadata": {"file_name": "paper1.pdf"}},
        {"text": "Another chunk.", "display_name": "paper2.pdf"},
    ]
    result = _format_sources_for_prompt(sources)
    assert "[1] paper1.pdf" in result
    assert "[2] paper2.pdf" in result
    assert "Short text." in result
    assert "Another chunk." in result


def test_format_sources_truncates_long_text():
    long_text = "token " * (_MAX_SOURCE_TOKENS + 100)
    sources = [{"text": long_text, "metadata": {"file_name": "long.pdf"}}]
    result = _format_sources_for_prompt(sources)
    truncated_tokens = count_tokens(sources[0]["text"])
    assert truncated_tokens > _MAX_SOURCE_TOKENS  # original was long
    assert "..." in result  # truncation marker


def test_format_sources_clips_to_max():
    sources = [
        {"text": f"chunk {i}", "metadata": {"file_name": f"paper{i}.pdf"}}
        for i in range(_MAX_SOURCES_IN_PROMPT + 5)
    ]
    result = _format_sources_for_prompt(sources)
    # Should only contain first _MAX_SOURCES_IN_PROMPT sources
    assert f"[{_MAX_SOURCES_IN_PROMPT}]" in result
    assert f"[{_MAX_SOURCES_IN_PROMPT + 1}]" not in result


def test_format_sources_missing_metadata():
    sources = [{"text": "No metadata here."}]
    result = _format_sources_for_prompt(sources)
    assert "[1] unknown" in result


def test_format_sources_empty():
    result = _format_sources_for_prompt([])
    assert result == ""


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

def test_parse_valid_json():
    result = _parse_json('{"pages": []}')
    assert result == {"pages": []}


def test_parse_json_with_chinese_quotes():
    # LLM outputs Chinese curly quotes: “ = ", ” = "
    result = _parse_json('{"title": “Hello”}')
    assert result is not None
    assert result["title"] == "Hello"


def test_parse_json_with_markdown_fence():
    result = _parse_json('```json\n{"pages": [{"type": "entity"}]}\n```')
    assert result is not None
    assert len(result["pages"]) == 1


def test_parse_invalid_json():
    result = _parse_json("not json at all {{{")
    assert result is None


# ---------------------------------------------------------------------------
# _source_filenames
# ---------------------------------------------------------------------------

def test_source_filenames_dedup():
    sources = [
        {"metadata": {"file_name": "a.pdf"}},
        {"metadata": {"file_name": "a.pdf"}},
        {"metadata": {"file_name": "b.pdf"}},
    ]
    result = _source_filenames(sources)
    assert result == ["a.pdf", "b.pdf"]


def test_source_filenames_max_n():
    sources = [
        {"metadata": {"file_name": f"paper{i}.pdf"}}
        for i in range(10)
    ]
    result = _source_filenames(sources, max_n=3)
    assert len(result) == 3


def test_source_filenames_display_name_priority():
    sources = [
        {"display_name": "nice_name.pdf", "metadata": {"file_name": "ugly.pdf"}},
    ]
    result = _source_filenames(sources)
    assert result == ["nice_name.pdf"]


# ---------------------------------------------------------------------------
# _coerce_page_type
# ---------------------------------------------------------------------------

def test_coerce_valid_types():
    assert _coerce_page_type("entity") == "entity"
    assert _coerce_page_type("concept") == "concept"
    assert _coerce_page_type("comparison") == "comparison"


def test_coerce_invalid_type():
    assert _coerce_page_type("garbage") == "concept"
    assert _coerce_page_type("") == "concept"


# ---------------------------------------------------------------------------
# _log_context_overflow
# ---------------------------------------------------------------------------

def test_log_context_overflow_warns():
    """15000 + 4096 = 19096 > 16384 → must warn."""
    from unittest.mock import patch
    with patch("agentic_rag.knowledge_extractor.logger") as mock_logger:
        _log_context_overflow(prompt_tokens=15000, output_max=4096, ctx_window=16384, label="Test")
        mock_logger.warning.assert_called_once()
        call_arg = mock_logger.warning.call_args[0][0]
        assert "Test" in call_arg
        assert "19096" in call_arg


def test_log_context_overflow_silent_when_fits():
    """5000 + 2048 = 7048 < 16384 → no warning."""
    from unittest.mock import patch
    with patch("agentic_rag.knowledge_extractor.logger") as mock_logger:
        _log_context_overflow(prompt_tokens=5000, output_max=2048, ctx_window=16384, label="Test")
        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# wiki integration: save_page writes valid markdown
# ---------------------------------------------------------------------------

def test_save_page_creates_valid_wiki_page(tmp_path, monkeypatch):
    from idea.wiki import IdeaWikiEngine

    wiki_root = tmp_path / ".paperrag-wiki-test"
    monkeypatch.setenv("PAPERRAG_WIKI_PATH", str(wiki_root))

    wiki = IdeaWikiEngine()
    wiki.ensure_schema()

    path = wiki.save_page(
        page_type="entity",
        title="Test Entity",
        slug="test-entity",
        content_md="This is a **test** entity page.\n\nSee also [[other-entity]].",
        tags=["test", "entity"],
        confidence="high",
        sources_list=["paper1.pdf", "paper2.pdf"],
    )

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "---" in content  # frontmatter
    assert "title: Test Entity" in content
    assert "slug: test-entity" in content
    assert "confidence: high" in content
    assert "paper1.pdf" in content
    assert "This is a **test** entity page." in content
    assert "[[other-entity]]" in content


def test_save_page_aliases(tmp_path, monkeypatch):
    from idea.wiki import IdeaWikiEngine

    wiki_root = tmp_path / ".paperrag-wiki-test-aliases"
    monkeypatch.setenv("PAPERRAG_WIKI_PATH", str(wiki_root))

    wiki = IdeaWikiEngine()
    wiki.ensure_schema()

    wiki.save_page(
        page_type="concept",
        title="Reciprocal Rank Fusion",
        slug="rrf",
        content_md="RRF combines rankings...",
        aliases=["RRF", "reciprocal-rank-fusion"],
        confidence="high",
    )

    content = (wiki_root / "concepts" / "rrf.md").read_text(encoding="utf-8")
    assert "aliases:" in content
    assert "RRF" in content


def test_save_page_updates_existing(tmp_path, monkeypatch):
    from idea.wiki import IdeaWikiEngine

    wiki_root = tmp_path / ".paperrag-wiki-test-update"
    monkeypatch.setenv("PAPERRAG_WIKI_PATH", str(wiki_root))

    wiki = IdeaWikiEngine()
    wiki.ensure_schema()

    wiki.save_page(
        page_type="concept",
        title="Test",
        slug="test-concept",
        content_md="Version 1.",
        confidence="medium",
    )
    wiki.save_page(
        page_type="concept",
        title="Test Updated",
        slug="test-concept",
        content_md="Version 2.",
        confidence="high",
    )

    content = (wiki_root / "concepts" / "test-concept.md").read_text(encoding="utf-8")
    assert "Version 2." in content
    assert "Version 1." not in content
    assert "confidence: high" in content


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
