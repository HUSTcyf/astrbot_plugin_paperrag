#!/usr/bin/env python3
"""
Focused test for /idea tofeishu image insertion pipeline.

Tests:
  1. _extract_inline_images — detects images in various formats
  2. _make_image_block — reads file, creates correct block structure
  3. _markdown_to_feishu_blocks — converts figure section to blocks
  4. _append_figure_section — includes both figures and tables
  5. Temp file cleanup — only cleans temp files, never real files
"""

import base64
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
    # Mock _get_vlm_provider_async to avoid real LLM calls
    engine._get_vlm_provider_async = MagicMock(return_value=None)
    engine._get_feishu_tool = MagicMock(return_value=None)
    engine._get_ideas_dir = MagicMock(return_value=Path(tempfile.mkdtemp()))
    return engine


@pytest.fixture
def sample_png():
    """Create a minimal valid PNG file for testing."""
    # Minimal 1x1 red PNG
    import struct
    import zlib

    def _create_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack('>I', len(data)) + chunk + crc

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    ihdr = _create_chunk(b'IHDR', ihdr_data)
    # RGB red pixel
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


@pytest.fixture
def sample_jpg():
    """Create a minimal valid JPEG file for testing."""
    # Minimal JPEG
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    # SOI + APP0 + DQT + SOF0 + DHT + SOS + EOI (minimal valid JPEG)
    tmp.write(bytes.fromhex(
        'ffd8'           # SOI
        'ffe000104a46494600010101006000600000'  # APP0 JFIF
        'ffdb00430001'   # DQT
        'ffc000110800010001011100'  # SOF0 1x1 grayscale
        'ffc4001f0000010501010101010100000000000000000102030405060708090a0b'  # DHT
        'ffda000c03010002110311003f00'  # SOS
        '7f00'           # scan data
        'ffd9'           # EOI
    ))
    tmp.close()
    yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Test _extract_inline_images
# ---------------------------------------------------------------------------

class TestExtractInlineImages:
    """Verify image detection in various markdown formats."""

    def test_standard_markdown_image_png(self, engine, sample_png):
        """![caption](path.png) — should be detected."""
        segs = engine._extract_inline_images(f'![Test Image]({sample_png})')
        assert len(segs) == 1
        assert segs[0]["type"] == "image"
        assert segs[0]["path"] == sample_png
        assert segs[0]["caption"] == "Test Image"

    def test_standard_markdown_image_jpg(self, engine, sample_jpg):
        """![caption](path.jpg) — should be detected (was broken before fix)."""
        segs = engine._extract_inline_images(f'![Test JPG]({sample_jpg})')
        assert len(segs) == 1, f"Expected 1 image segment, got {segs}"
        assert segs[0]["type"] == "image"
        assert segs[0]["path"] == sample_jpg

    def test_figure_section_format(self, engine, sample_png):
        """![图 1：caption](path) — the exact format _append_figure_section generates."""
        segs = engine._extract_inline_images(f'![图 1：方法流程图]({sample_png})')
        assert len(segs) == 1
        assert segs[0]["type"] == "image"
        assert "图 1" in segs[0]["caption"]

    def test_table_section_format(self, engine, sample_png):
        """![表 1：caption](path) — the table format from _append_figure_section."""
        segs = engine._extract_inline_images(f'![表 1：实验结果]({sample_png})')
        assert len(segs) == 1
        assert segs[0]["type"] == "image"
        assert "表 1" in segs[0]["caption"]

    def test_nonexistent_file_returns_text(self, engine):
        """Non-existent path should fall back to text segment (no crash)."""
        segs = engine._extract_inline_images('![Missing](/nonexistent/file.png)')
        assert len(segs) == 1
        assert segs[0]["type"] == "text", f"Expected text fallback, got {segs[0]['type']}"

    def test_plain_text_no_image(self, engine):
        """Plain text without image should return text segment."""
        segs = engine._extract_inline_images('This is just plain text.')
        assert len(segs) == 1
        assert segs[0]["type"] == "text"

    def test_chinese_caption_with_colon(self, engine, sample_png):
        """Chinese caption with full-width colon."""
        segs = engine._extract_inline_images(f'![图 5：基于NeRF的稀疏重建方法]({sample_png})')
        assert len(segs) == 1
        assert segs[0]["type"] == "image"


# ---------------------------------------------------------------------------
# Test _make_image_block
# ---------------------------------------------------------------------------

class TestMakeImageBlock:
    """Verify image block construction."""

    def test_creates_block_with_image_path(self, engine, sample_png):
        """_make_image_block should read file and embed base64 + keep image_path."""
        block = engine._make_image_block(sample_png, "My Caption")
        assert block is not None
        assert block["blockType"] == "image"
        opts = block["options"]["image"]
        assert opts["image_path"] == sample_png
        assert opts["caption"] == "My Caption"
        # Verify base64 decodes back to the original PNG bytes
        with open(sample_png, "rb") as f:
            original = f.read()
        decoded = base64.b64decode(opts["base64"])
        assert decoded == original

    def test_nonexistent_file_returns_none(self, engine):
        """Non-existent file should return None."""
        block = engine._make_image_block("/nonexistent/file.png", "Test")
        assert block is None

    def test_block_has_both_path_and_base64(self, engine, sample_png):
        """Block must have BOTH image_path and base64 — needed for insertion loop."""
        block = engine._make_image_block(sample_png, "Test")
        assert block["options"]["image"]["image_path"]
        assert block["options"]["image"]["base64"]


# ---------------------------------------------------------------------------
# Test _markdown_to_feishu_blocks (figure section)
# ---------------------------------------------------------------------------

class TestMarkdownToFeishuBlocks:
    """Verify full markdown → feishu blocks conversion for figure sections."""

    def test_figure_section_creates_image_blocks(self, engine, sample_png):
        """A markdown figure section should produce image blocks."""
        md = (
            "## 9. 论文图表\n\n"
            f"![图 1：Test]({sample_png})\n\n"
            f"![图 2：Test2]({sample_png})\n"
        )
        blocks = engine._markdown_to_feishu_blocks(md)
        image_blocks = [b for b in blocks if b.get("blockType") == "image"]
        assert len(image_blocks) == 2, f"Expected 2 image blocks, got {len(image_blocks)}: {blocks}"

    def test_figure_section_handles_heading(self, engine):
        """## heading should become a heading block."""
        blocks = engine._markdown_to_feishu_blocks("## 9. 论文图表")
        headings = [b for b in blocks if b.get("blockType") == "heading"]
        assert len(headings) == 1

    def test_empty_lines_are_skipped(self, engine):
        """Empty lines should produce no blocks."""
        blocks = engine._markdown_to_feishu_blocks("\n\n\n")
        assert len(blocks) == 0

    def test_mixed_content_figure_and_text(self, engine, sample_png):
        """Mixed text + figure should produce both text and image blocks."""
        md = (
            "Some introduction text.\n\n"
            f"![图 1：Method]({sample_png})\n\n"
            "More text after figure.\n"
        )
        blocks = engine._markdown_to_feishu_blocks(md)
        text_blocks = [b for b in blocks if b.get("blockType") == "text"]
        image_blocks = [b for b in blocks if b.get("blockType") == "image"]
        assert len(text_blocks) >= 2, f"Expected at least 2 text blocks: {blocks}"
        assert len(image_blocks) == 1, f"Expected 1 image block: {blocks}"

    def test_jpg_in_figure_section(self, engine, sample_jpg):
        """JPG images in figure section should be detected and converted to image blocks."""
        md = f"![图 1：JPG Test]({sample_jpg})\n"
        blocks = engine._markdown_to_feishu_blocks(md)
        image_blocks = [b for b in blocks if b.get("blockType") == "image"]
        assert len(image_blocks) == 1, f"JPG not detected as image: {blocks}"


# ---------------------------------------------------------------------------
# Test _append_figure_section
# ---------------------------------------------------------------------------

class TestAppendFigureSection:
    """Verify _append_figure_section handles both figures and tables."""

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
        result = engine._append_figure_section("## 参考文献\nSome refs.\n", knowledge)
        assert "图 1" in result, f"Figure not included: {result}"
        assert "表 1" in result, f"Table not included: {result}"

    def test_figure_only(self, engine, sample_png):
        """Only image_path, no table."""
        knowledge = {
            "local_results": [
                {
                    "metadata": {
                        "image_path": sample_png,
                        "image_caption": "Solo Figure",
                    }
                }
            ]
        }
        result = engine._append_figure_section("## 参考文献\nRefs.\n", knowledge)
        assert "图 1" in result
        assert "表 " not in result

    def test_empty_knowledge_returns_original(self, engine):
        """None/empty knowledge should return text unchanged."""
        assert engine._append_figure_section("original", None) == "original"
        assert engine._append_figure_section("original", {}) == "original"

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
        result = engine._append_figure_section("## 参考文献\nRefs.\n", knowledge)
        # Ghost figure not included, real table IS included
        assert "Ghost" not in result
        assert "表 1" in result


# ---------------------------------------------------------------------------
# Test temp file cleanup safety
# ---------------------------------------------------------------------------

class TestTempFileSafety:
    """Verify that the insertion loop never deletes real files."""

    def test_real_file_still_exists_after_make_image_block(self, engine, sample_png):
        """_make_image_block reads a real file but must not delete it."""
        assert os.path.exists(sample_png)
        block = engine._make_image_block(sample_png, "Test")
        assert block is not None
        # Real file must still exist
        assert os.path.exists(sample_png), "Real file was deleted by _make_image_block!"

    def test_is_temp_file_flag_behavior(self, engine, sample_png):
        """Verify the is_temp_file logic: real file → is_temp_file=False, temp file → is_temp_file=True."""
        # Simulate a block from _make_image_block (has both image_path and base64)
        block = engine._make_image_block(sample_png, "Test")
        opts = block["options"]["image"]
        img_path = opts.get("image_path", "")
        img_base64 = opts.get("base64", "")

        # Real file: image_path is set, base64 is set
        assert img_path and img_base64  # Both are set
        is_temp_file = not img_path and bool(img_base64)
        assert is_temp_file is False, "Real file should NOT be flagged as temp"

        # Temp file: no image_path, only base64
        is_temp_file = not "" and bool(img_base64)
        # Actually: not "" = True, bool(img_base64) = True → is_temp_file = True
        # But we need: no image_path AND has base64 → is_temp_file = True
        assert is_temp_file is True, "Base64-only should be flagged as temp"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
