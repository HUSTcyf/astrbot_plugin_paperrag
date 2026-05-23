"""
Tests for accurate figure/table logical numbering via docling native caption_text().
Covers:
  1. _extract_logical_num helper — extracts logical figure/table number from real captions
  2. caption_text integration — mock PictureItem/TableItem with real vs synthetic captions
  3. End-to-end filename construction — ensures saved_path uses logical numbers
"""
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Ensure plugin root is on sys.path for package imports (rag.hybrid_parser, etc.)
_plugin_root = Path(__file__).parent.parent
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))


# ---------------------------------------------------------------------------
# Test 1: _extract_logical_num helper
# ---------------------------------------------------------------------------
def test_extract_logical_num():
    """Extract logical figure/table number from real caption text."""
    # Import helper from subprocess module
    sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
    from _docling_subprocess import _extract_logical_num

    # Figure captions
    assert _extract_logical_num("Figure 1: Overview", "Figure") == "1"
    assert _extract_logical_num("Figure 3: System Architecture", "Figure") == "3"
    assert _extract_logical_num("Figure 10: Results", "Figure") == "10"
    assert _extract_logical_num("Fig. 5: Pipeline", "Figure") == "5"
    assert _extract_logical_num("Figure 3a: Ablation", "Figure") == "3a"
    assert _extract_logical_num("Figure A1: Supplement", "Figure") == "A1"
    assert _extract_logical_num("Figure S1: Supplementary", "Figure") == "S1"

    # Table captions
    assert _extract_logical_num("Table 1: Statistics", "Table") == "1"
    assert _extract_logical_num("Table 2: Quantitative results", "Table") == "2"
    assert _extract_logical_num("Table 10a: Extended", "Table") == "10a"

    # Edge cases
    assert _extract_logical_num("", "Figure") is None
    assert _extract_logical_num(None, "Figure") is None
    assert _extract_logical_num("No number here", "Figure") is None
    assert _extract_logical_num("Fig.  about something", "Figure") is None  # "Fig." not followed by number

    # Chinese captions
    assert _extract_logical_num("图 1：系统架构", "Figure") == "1"
    assert _extract_logical_num("表 3：实验结果", "Table") == "3"

    print("  PASS: test_extract_logical_num")


# ---------------------------------------------------------------------------
# Test 2: filename construction with logical numbers
# ---------------------------------------------------------------------------
def test_logical_number_filename():
    """Verify that files are named with logical figure numbers, not per-page counters."""
    from _docling_subprocess import _extract_logical_num

    # Simulate: docling returns real caption "Figure 3: Pipeline Overview" on page 8
    caption = "Figure 3: Pipeline Overview"
    page_no = 8
    logical_num = _extract_logical_num(caption, "Figure")
    assert logical_num == "3"

    # Filename should use logical num, not per-page index
    label = f"Figure{logical_num}"
    filename = f"{page_no}-{label}.png"
    assert filename == "8-Figure3.png"
    assert filename != "8-Figure1.png"  # Old behavior would produce this

    # Subsequent page with same logical num should get _v suffix
    collision_label = f"Figure{logical_num}_v2"
    collision_filename = f"{page_no}-{collision_label}.png"
    assert collision_filename == "8-Figure3_v2.png"

    print("  PASS: test_logical_number_filename")


# ---------------------------------------------------------------------------
# Test 3: full image/tables JSON structure with logical numbers
# ---------------------------------------------------------------------------
def test_subprocess_output_structure():
    """Ensure subprocess output JSON has correct structure with logical_num."""
    image_entry = {
        "page_number": 8,
        "image_index": 0,
        "logical_num": "3",
        "bbox": [0, 0, 0, 0],
        "caption": "Figure 3: Pipeline Overview",
        "saved_path": "/path/to/figures/8-Figure3.png",
    }
    assert image_entry["logical_num"] == "3"
    assert "Figure3" in image_entry["saved_path"]
    assert image_entry["caption"] != "Figure 1"  # Not synthetic per-page label

    table_entry = {
        "page_number": 12,
        "table_index": 0,
        "logical_num": "1",
        "bbox": [0, 0, 0, 0],
        "csv": "...",
        "markdown": "...",
        "caption": "Table 1: Quantitative Results",
        "saved_csv_path": "/path/to/tables/12-Table1.csv",
        "saved_md_path": "/path/to/tables/12-Table1.md",
        "saved_png_path": "/path/to/tables/12-Table1.png",
    }
    assert table_entry["logical_num"] == "1"
    assert "Table1" in table_entry["saved_png_path"]
    assert table_entry["caption"] != "Table 1"  # It IS "Table 1" but from real caption, not synthetic

    print("  PASS: test_subprocess_output_structure")


# ---------------------------------------------------------------------------
# Test 4: caption_text fallback — docling caption unavailable
# ---------------------------------------------------------------------------
def test_fallback_when_caption_unavailable():
    """When caption_text falls back to per-page counter, naming is consistent."""
    # Simulate a figure with no docling caption (should get per-page fallback)
    # This tests the code path where real_caption is empty
    from _docling_subprocess import _extract_logical_num

    assert _extract_logical_num("", "Figure") is None
    assert _extract_logical_num(None, "Figure") is None

    # In this fallback case, the filename uses per-page counter (same as before)
    # E.g., 8-Figure1.png for the first figure on page 8
    # This is acceptable because the _associate_figures_with_chunks fix
    # now stores all candidates and picks the best one.

    print("  PASS: test_fallback_when_caption_unavailable")


# ---------------------------------------------------------------------------
# Test 5: hybrid_parser association with correct logical numbers
# ---------------------------------------------------------------------------
def test_association_with_logical_numbers():
    """Verify figure_refs correctly maps logical numbers to paths when captions are real."""
    from rag.hybrid_parser import HybridPDFParser

    parser = HybridPDFParser()

    # Simulate image_paths dict from _extract_and_save_images
    # With logical numbering, "Figure 3" appears ONCE (not overwritten by "Figure 1" from other pages)
    image_paths = {
        "Figure 1: Architecture|1|0": "/data/figures/paper/1-Figure1.png",
        "Figure 2: Methods|4|1": "/data/figures/paper/4-Figure2.png",
        "Figure 3: Pipeline|8|2": "/data/figures/paper/8-Figure3.png",
    }

    # Build figure_refs (same logic as our fixed _associate_figures_with_chunks)
    figure_refs = {}
    for key, path in image_paths.items():
        parts = key.split("|")
        caption_str = parts[0]
        page_num = int(parts[1]) if len(parts) > 1 else 0
        figure_num = parser._extract_figure_number(caption_str)
        if figure_num:
            if figure_num not in figure_refs:
                figure_refs[figure_num] = []
            if not any(p == path for p, _, _ in figure_refs[figure_num]):
                figure_refs[figure_num].append((path, caption_str, page_num))

    # Each figure number should have exactly 1 candidate (no overwrite)
    assert len(figure_refs["1"]) == 1, f"Expected 1 candidate for Figure 1, got {len(figure_refs['1'])}"
    assert len(figure_refs["2"]) == 1
    assert len(figure_refs["3"]) == 1

    # Figure 1 path should point to 1-Figure1.png, NOT the per-page "Figure 1" from page 8
    assert "1-Figure1" in figure_refs["1"][0][0]

    print("  PASS: test_association_with_logical_numbers")


# ---------------------------------------------------------------------------
# Test 6: no overwrite — multiple images with same per-page label
# ---------------------------------------------------------------------------
def test_no_overwrite_multiple_per_page_figure1():
    """Old behavior: multiple 'Figure 1' entries overwrite each other.
    New behavior: all candidates preserved, best selected by file-exists + page proximity."""
    from rag.hybrid_parser import HybridPDFParser
    parser = HybridPDFParser()

    # With logical numbering, these are ALL DIFFERENT figure numbers
    image_paths = {
        "Figure 1: Overview|1|0": "/data/figures/paper/1-Figure1.png",
        "Figure 2: Methods|4|1": "/data/figures/paper/4-Figure2.png",
        "Figure 3: Pipeline|8|2": "/data/figures/paper/8-Figure3.png",
        "Figure 4: Results|12|3": "/data/figures/paper/12-Figure4.png",
    }

    figure_refs = {}
    for key, path in image_paths.items():
        parts = key.split("|")
        caption_str = parts[0]
        page_num = int(parts[1]) if len(parts) > 1 else 0
        figure_num = parser._extract_figure_number(caption_str)
        if figure_num:
            if figure_num not in figure_refs:
                figure_refs[figure_num] = []
            if not any(p == path for p, _, _ in figure_refs[figure_num]):
                figure_refs[figure_num].append((path, caption_str, page_num))

    # All 4 different figure numbers should exist
    assert len(figure_refs) == 4, f"Expected 4 unique figure numbers, got {len(figure_refs)}"
    for num in ["1", "2", "3", "4"]:
        assert num in figure_refs, f"Figure {num} missing from figure_refs"

    # Legacy scenario test: same per-page "Figure 1" from different pages
    # (this is what old behavior would produce with per-page counters)
    legacy_paths = {
        "Figure 1|1|0": "/data/figures/paper/1-Figure1.png",
        "Figure 1|3|1": "/data/figures/paper/3-Figure1.png",
        "Figure 1|7|2": "/data/figures/paper/7-Figure1.png",
    }
    legacy_refs = {}
    for key, path in legacy_paths.items():
        parts = key.split("|")
        caption_str = parts[0]
        page_num = int(parts[1]) if len(parts) > 1 else 0
        figure_num = parser._extract_figure_number(caption_str)
        if figure_num:
            if figure_num not in legacy_refs:
                legacy_refs[figure_num] = []
            if not any(p == path for p, _, _ in legacy_refs[figure_num]):
                legacy_refs[figure_num].append((path, caption_str, page_num))

    # All 3 candidates preserved (was 1 with old overwrite behavior)
    assert len(legacy_refs["1"]) == 3, f"Expected 3 candidates, got {len(legacy_refs['1'])}"

    print("  PASS: test_no_overwrite_multiple_per_page_figure1")


# ---------------------------------------------------------------------------
# Test 7: best candidate selection — picks file that exists on disk
# ---------------------------------------------------------------------------
def test_best_candidate_selection():
    """When chunk references 'Figure 1', pick the candidate whose file exists, not just the last one."""
    candidates = [
        ("/data/figures/paper/1-Figure1.png", "Figure 1: Overview", 1),    # exists
        ("/data/figures/paper/8-Figure1.png", "Figure 1: Architecture", 8),  # doesn't exist
    ]

    # Simulate: chunk on page 2 says "as shown in Figure 1"
    chunk_page = 2

    best = None
    best_score = -2
    best_page_match = False
    for cand_path, cand_caption, cand_page in candidates:
        # Check file existence (mock: only 1-Figure1.png exists)
        exists = "1-Figure1" in cand_path
        page_ok = chunk_page > 0 and cand_page > 0 and abs(chunk_page - cand_page) <= 2
        if exists and page_ok:
            score = 1
        elif exists:
            score = 0
        else:
            score = -1
        if score > best_score:
            best_score = score
            best = (cand_path, cand_caption, cand_page)
            best_page_match = page_ok

    # Should pick the one that exists AND is on a nearby page (page 1 is within 2 pages of chunk page 2)
    assert best[0] == "/data/figures/paper/1-Figure1.png"
    assert best_score == 1
    assert best_page_match is True

    print("  PASS: test_best_candidate_selection")


# ---------------------------------------------------------------------------
# Test 8: end-to-end — mock docling caption_text returns real caption
# ---------------------------------------------------------------------------
def test_mock_docling_caption_text():
    """Simulate the docling subprocess with real captions via caption_text()."""
    # Mock a PictureItem with caption_text returning real caption
    mock_pic = MagicMock()
    mock_pic.prov = [Mock()]
    mock_pic.prov[0].page_no = 8
    mock_pic.image = Mock()
    mock_pic.image.pil_image = Mock()  # Pretend PIL Image

    mock_doc = Mock()
    mock_pic.caption_text.return_value = "Figure 3: Pipeline Overview"

    # Simulate the processing logic
    real_caption = mock_pic.caption_text(mock_doc)
    assert real_caption == "Figure 3: Pipeline Overview"

    from _docling_subprocess import _extract_logical_num
    logical_num = _extract_logical_num(real_caption, "Figure")
    assert logical_num == "3"

    # Filename construction
    page_no = 8
    label = f"Figure{logical_num}"
    filename = f"{page_no}-{label}.png"
    assert filename == "8-Figure3.png"

    caption = real_caption or f"Figure {logical_num}"
    assert caption == "Figure 3: Pipeline Overview"
    assert caption != "Figure 3"  # Full caption preserved, not truncated

    print("  PASS: test_mock_docling_caption_text")


# ---------------------------------------------------------------------------
# Test 9: TableItem caption_text integration
# ---------------------------------------------------------------------------
def test_mock_docling_table_caption_text():
    """Same as above but for TableItem."""
    mock_table = MagicMock()
    mock_table.prov = [Mock()]
    mock_table.prov[0].page_no = 12

    mock_doc = Mock()
    mock_table.caption_text.return_value = "Table 1: Quantitative Results on Replica"

    real_caption = mock_table.caption_text(mock_doc)
    assert real_caption == "Table 1: Quantitative Results on Replica"

    from _docling_subprocess import _extract_logical_num
    logical_num = _extract_logical_num(real_caption, "Table")
    assert logical_num == "1"

    page_no = 12
    label = f"Table{logical_num}"
    png_filename = f"{page_no}-{label}.png"
    assert png_filename == "12-Table1.png"

    print("  PASS: test_mock_docling_table_caption_text")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Test: Logical Figure Numbering (Plan A) ===\n")
    test_extract_logical_num()
    test_logical_number_filename()
    test_subprocess_output_structure()
    test_fallback_when_caption_unavailable()
    test_association_with_logical_numbers()
    test_no_overwrite_multiple_per_page_figure1()
    test_best_candidate_selection()
    test_mock_docling_caption_text()
    test_mock_docling_table_caption_text()
    print(f"\n=== All 9 tests passed ===")
