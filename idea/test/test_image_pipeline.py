#!/usr/bin/env python3
"""
Tests for /idea tofeishu image insertion pipeline (lark-cli path).

Tests:
  1. _append_figure_section — returns figure_infos + anchor (no text modification)
  2. _find_figure_anchor — finds section boundary anchor for image positioning
  3. Non-existent path handling — silently skips missing files
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_plugin_root = Path(__file__).parent.parent.parent
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a minimal IdeaEngineFeishuDoc with mocked context."""
    from idea.feishu_doc import IdeaEngineFeishuDoc

    ctx = MagicMock()
    ctx.config = {}
    engine = IdeaEngineFeishuDoc(context=ctx)
    engine._get_vlm_provider_async = MagicMock(return_value=None)
    engine._get_ideas_dir = MagicMock(return_value=Path(tempfile.mkdtemp()))
    return engine


@pytest.fixture
def sample_png():
    """Create a minimal valid PNG file for testing."""
    import struct
    import zlib

    def _create_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack('>I', len(data)) + chunk + crc

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    ihdr = _create_chunk(b'IHDR', ihdr_data)
    raw = b'\x00\xff\x00\x00'
    compressed = zlib.compress(raw)
    idat = _create_chunk(b'IDAT', compressed)
    iend = _create_chunk(b'IEND', b'')

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.write(signature + ihdr + idat + iend)
    tmp.close()
    yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test _append_figure_section (new API: returns figure_infos + anchor)
# ---------------------------------------------------------------------------

class TestAppendFigureSection:
    """Verify _append_figure_section collects figures and finds anchors."""

    def test_figure_and_table_both_included(self, engine, sample_png):
        """Both image_path and table_png_path should produce entries."""
        knowledge = {
            "local_results": [
                {
                    "metadata": {
                        "image_path": sample_png,
                        "image_caption": "Figure 1",
                        "table_png_path": sample_png,
                        "table_caption": "Table 1",
                    }
                }
            ]
        }
        figure_infos, anchors = engine._append_figure_section(
            "## 2. 相关工作\nRelated work.\n\n## 3. 方法论\nMethods.\n", knowledge
        )
        assert len(figure_infos) == 2
        assert figure_infos[0]["type"] == "fig"
        assert "图 1" in figure_infos[0]["caption"]
        assert figure_infos[1]["type"] == "table"
        assert "表 1" in figure_infos[1]["caption"]
        assert anchors["related_work"] == "Related work."
        assert anchors["methodology"] == "Methods."

    def test_figure_only(self, engine, sample_png):
        """Only image_path, no table."""
        knowledge = {
            "local_results": [
                {"metadata": {"image_path": sample_png, "image_caption": "Solo Figure"}}
            ]
        }
        figure_infos, anchor = engine._append_figure_section(
            "## 2. 相关工作\nRelated work.\n\n## 3. 方法论\nMethods.\n", knowledge
        )
        assert len(figure_infos) == 1
        assert figure_infos[0]["type"] == "fig"
        assert "图 1" in figure_infos[0]["caption"]

    def test_empty_knowledge_returns_empty(self, engine):
        """None/empty knowledge should return ([], empty_anchors_dict)."""
        empty = {"related_work": None, "methodology": None}
        assert engine._append_figure_section("original", None) == ([], empty)
        assert engine._append_figure_section("original", {}) == ([], empty)

    def test_nonexistent_paths_are_skipped(self, engine, sample_png):
        """Paths that don't exist should be silently skipped."""
        knowledge = {
            "local_results": [
                {
                    "metadata": {
                        "image_path": "/nonexistent/figure.png",
                        "image_caption": "Ghost",
                        "table_png_path": sample_png,
                        "table_caption": "Real Table",
                    }
                }
            ]
        }
        figure_infos, anchor = engine._append_figure_section(
            "## 2. 相关工作\nRelated work.\n\n## 3. 方法论\nMethods.\n", knowledge
        )
        # Ghost figure not included, real table IS included
        assert len(figure_infos) == 1
        assert figure_infos[0]["type"] == "table"
        assert "表 1" in figure_infos[0]["caption"]

    def test_no_knowledge_key_returns_empty(self, engine, sample_png):
        """knowledge without 'local_results' key."""
        figure_infos, anchors = engine._append_figure_section("Some text", {"other": []})
        assert figure_infos == []
        assert anchors == {"related_work": None, "methodology": None}


# ---------------------------------------------------------------------------
# Test _find_figure_anchor
# ---------------------------------------------------------------------------

class TestFindFigureAnchors:
    """Verify anchor detection using str.find() on canonical headings."""

    def test_both_anchors_found(self, engine):
        """Both related_work and methodology anchors found with canonical headings."""
        text = (
            "## 1. 背景动机\nBackground.\n\n"
            "## 2. 相关工作\nRelated work content.\n\n"
            "## 3. 方法论\nMethodology content.\n\n"
            "## 4. 创新点\nInnovation.\n"
        )
        anchors = engine._find_figure_anchors(text)
        assert anchors["related_work"] == "Related work content."
        assert anchors["methodology"] == "Methodology content."

    def test_related_work_anchor_found(self, engine):
        """related_work anchor is last content line before 方法论."""
        text = (
            "## 1. 背景动机\nBackground.\n\n"
            "## 2. 相关工作\nRelated work discussion.\n\n"
            "## 3. 方法论\nMethods.\n"
        )
        anchors = engine._find_figure_anchors(text)
        assert anchors["related_work"] == "Related work discussion."

    def test_methodology_anchor_found(self, engine):
        """methodology anchor is last content line before 创新点."""
        text = (
            "## 3. 方法论\n"
            "Our method uses deep learning.\n\n"
            "## 4. 创新点\nInnovation.\n"
        )
        anchors = engine._find_figure_anchors(text)
        assert anchors["related_work"] is None  # no 相关工作 section
        assert anchors["methodology"] == "Our method uses deep learning."

    def test_missing_headings_return_none(self, engine):
        """Missing canonical headings return None anchors."""
        text = "## 1. 背景动机\nIntro.\n\n## 4. 创新点\nInnovation.\n"
        anchors = engine._find_figure_anchors(text)
        assert anchors["related_work"] is None
        assert anchors["methodology"] is None

    def test_no_canonical_headings(self, engine):
        """No canonical headings at all."""
        text = "Just some random text without section headings."
        anchors = engine._find_figure_anchors(text)
        assert anchors["related_work"] is None
        assert anchors["methodology"] is None

    def test_section_without_body_falls_back_to_heading(self, engine):
        """Section with only heading, no body — use heading as anchor."""
        text = (
            "## 2. 相关工作\n\n"
            "## 3. 方法论\nMethods.\n"
        )
        anchors = engine._find_figure_anchors(text)
        assert anchors["related_work"] == "2. 相关工作"


# ---------------------------------------------------------------------------
# Test non-existent paths are excluded
# ---------------------------------------------------------------------------

class TestNonExistentPaths:
    """Verify figures with non-existent paths are NOT included."""

    def test_nonexistent_path_skipped(self, engine):
        """Figure with non-existent path should be excluded."""
        text = "## 2. 相关工作\nRelated work.\n\n## 3. 方法论\nMethods.\n"
        knowledge = {
            "local_results": [
                {"metadata": {"image_path": "/nonexistent/ghost.png", "image_caption": "Ghost"}}
            ]
        }
        figure_infos, anchor = engine._append_figure_section(text, knowledge)
        assert figure_infos == []

    def test_mixed_existing_and_nonexistent(self, engine, sample_png):
        """Only existing figures are included; nonexistent ones are silently skipped."""
        text = "## 2. 相关工作\nRelated work.\n\n## 3. 方法论\nMethods.\n"
        knowledge = {
            "local_results": [
                {"metadata": {"image_path": "/nonexistent/ghost.png", "image_caption": "Ghost"}},
                {"metadata": {"image_path": sample_png, "image_caption": "Real Figure"}},
                {"metadata": {"image_path": "/another/missing.png", "image_caption": "Missing"}},
                {"metadata": {"table_png_path": "/nonexistent/table.png", "table_caption": "Ghost Table"}},
            ]
        }
        figure_infos, anchor = engine._append_figure_section(text, knowledge)
        assert len(figure_infos) == 1
        assert "图 1" in figure_infos[0]["caption"]
        assert "Real Figure" in figure_infos[0]["caption"]

    def test_all_nonexistent_returns_empty(self, engine):
        """All paths nonexistent → empty figure_infos."""
        text = "## 2. 相关工作\nRelated work.\n\n## 3. 方法论\nMethods.\n"
        knowledge = {
            "local_results": [
                {"metadata": {"image_path": "/nonexistent/ghost.png", "image_caption": "Ghost"}},
            ]
        }
        figure_infos, anchors = engine._append_figure_section(text, knowledge)
        assert figure_infos == []
        assert anchors == {"related_work": None, "methodology": None}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
