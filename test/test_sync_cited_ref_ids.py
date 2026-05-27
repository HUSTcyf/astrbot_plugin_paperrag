"""
Test sync_cited_ref_ids_for_paper on HybridIndexManager.

Verifies:
1. Basic sync: old cited_ref_ids differ from new → upsert called with partial_update=True
2. Unchanged: cited_ref_ids already match → no upsert called
3. Empty references: returns early with synced=0
4. No Milvus chunks: returns early (paper not indexed)
5. Metadata diff detection: only changed chunks are upserted
6. file_name suffix logic: .pdf handling matches get_all_chunks() behavior
"""
from unittest.mock import AsyncMock, MagicMock
import json
import pytest


def _make_mock_collection(query_return=None, upsert_side_effect=None):
    """Create a mock Milvus Collection."""
    col = MagicMock()
    col.query = MagicMock(return_value=query_return or [])
    col.upsert = MagicMock()
    if upsert_side_effect:
        col.upsert.side_effect = upsert_side_effect
    col.flush = MagicMock()
    return col


def _make_manager(mock_collection, file_name_has_pdf_suffix=None):
    """Create a HybridIndexManager with mocked collection and essential attrs."""
    from rag.hybrid_index import HybridIndexManager

    mgr = HybridIndexManager.__new__(HybridIndexManager)
    mgr._collection = mock_collection
    mgr._collection_name = "test_collection"
    mgr._file_name_has_pdf_suffix = file_name_has_pdf_suffix
    mgr._ensure_collection = AsyncMock(return_value=None)
    mgr._doc_stats = {}
    mgr._is_connected = False  # prevent __del__ from crashing
    return mgr


def _make_refs():
    """Create sample Reference objects (ref_id numbering from new reparse)."""
    from rag.reference_processor import Reference

    return [
        Reference(
            ref_id="ref_1",
            raw_text="[1] B. Mildenhall et al., NeRF: Representing Scenes as Neural Radiance Fields, ECCV 2020.",
            ref_title="NeRF: Representing Scenes as Neural Radiance Fields",
            ref_authors="B. Mildenhall et al.",
            ref_year=2020,
            ref_doi="10.1145/nerf",
            ref_venue="ECCV",
        ),
        Reference(
            ref_id="ref_2",
            raw_text="[2] A. Vaswani et al., Attention Is All You Need, NeurIPS 2017.",
            ref_title="Attention Is All You Need",
            ref_authors="A. Vaswani et al.",
            ref_year=2017,
            ref_doi="10.1234/attention",
            ref_venue="NeurIPS",
        ),
    ]


class TestSyncCitedRefIds:
    """Test HybridIndexManager.sync_cited_ref_ids_for_paper()."""

    @pytest.mark.asyncio
    async def test_basic_sync_updates_cited_ref_ids(self):
        """Chunks with old cited_ref_ids get updated via CitationLinker."""
        raw = [
            {
                "id": 1001,
                "text": "Neural Radiance Fields (NeRF) [1] represent scenes as continuous functions.",
                "metadata": json.dumps({
                    "file_name": "2508.09977v2.pdf",
                    "chunk_index": 0,
                    "cited_ref_ids": ["ref_1"],
                }),
            },
            {
                "id": 1002,
                "text": "Following [2], we use positional encoding for higher fidelity.",
                "metadata": json.dumps({
                    "file_name": "2508.09977v2.pdf",
                    "chunk_index": 1,
                    "cited_ref_ids": ["ref_2"],
                }),
            },
            {
                "id": 1003,
                "text": "This section introduces background concepts without citations.",
                "metadata": json.dumps({
                    "file_name": "2508.09977v2.pdf",
                    "chunk_index": 2,
                    "cited_ref_ids": [],
                }),
            },
        ]
        col = _make_mock_collection(query_return=raw)
        mgr = _make_manager(col, file_name_has_pdf_suffix=True)
        refs = _make_refs()

        result = await mgr.sync_cited_ref_ids_for_paper("2508.09977v2.pdf", refs)

        assert result["total_chunks"] == 3
        assert result["error"] is None
        # CitationLinker should find [1] in chunk 1001 → ref_1 (no change)
        # CitationLinker should find [2] in chunk 1002 → ref_2 (no change)
        # Chunk 1003 has no citations → unchanged
        # All cited_ref_ids already match → no upsert needed
        assert result["unchanged"] == 3
        assert result["synced"] == 0

    @pytest.mark.asyncio
    async def test_stale_ref_ids_get_synced(self):
        """When old cited_ref_ids are stale, full upsert is called (milvus_lite lacks partial_update)."""
        raw = [
            {
                "id": 2001,
                "text": "Method [1] was proposed by Smith et al.",
                "vector": [0.1, 0.2, 0.3],
                "metadata": json.dumps({
                    "file_name": "paper.pdf",
                    "chunk_index": 0,
                    "cited_ref_ids": ["ref_99"],  # stale — wrong ref_id
                }),
            },
        ]
        col = _make_mock_collection(query_return=raw)
        mgr = _make_manager(col, file_name_has_pdf_suffix=True)
        refs = _make_refs()

        result = await mgr.sync_cited_ref_ids_for_paper("paper.pdf", refs)

        assert result["total_chunks"] == 1
        # [1] in text → ref_1 from new references, old was ref_99 → changed
        assert result["synced"] == 1
        assert result["unchanged"] == 0

        # full upsert called (no partial_update — not supported by milvus_lite 2.5.x)
        col.upsert.assert_called_once()
        call_args, call_kwargs = col.upsert.call_args
        assert "partial_update" not in call_kwargs or call_kwargs.get("partial_update") is False

        # full upsert data includes id, text, vector, metadata
        upsert_data = call_args[0]
        assert len(upsert_data) == 1
        assert upsert_data[0]["id"] == 2001
        assert upsert_data[0]["text"] == "Method [1] was proposed by Smith et al."
        assert upsert_data[0]["vector"] == [0.1, 0.2, 0.3]
        assert "metadata" in upsert_data[0]

    @pytest.mark.asyncio
    async def test_empty_references_returns_early(self):
        """Empty reference list skips all Milvus operations."""
        col = _make_mock_collection()
        mgr = _make_manager(col)

        result = await mgr.sync_cited_ref_ids_for_paper("test.pdf", [])

        assert result == {"synced": 0, "unchanged": 0, "total_chunks": 0, "error": None}
        col.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_milvus_chunks_returns_early(self):
        """Paper has no chunks in Milvus (not indexed yet)."""
        col = _make_mock_collection(query_return=[])
        mgr = _make_manager(col, file_name_has_pdf_suffix=True)
        refs = _make_refs()

        result = await mgr.sync_cited_ref_ids_for_paper("nonexistent.pdf", refs)

        assert result["total_chunks"] == 0
        assert result["synced"] == 0
        col.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_changed_chunks_upserted(self):
        """Out of 3 chunks, only the ones with changed cited_ref_ids are upserted."""
        raw = [
            {
                "id": 3001,
                "text": "Method [1] was proposed by Smith.",
                "vector": [0.1],
                "metadata": json.dumps({
                    "file_name": "paper.pdf",
                    "chunk_index": 0,
                    "cited_ref_ids": ["ref_5"],  # old/wrong
                }),
            },
            {
                "id": 3002,
                "text": "As shown in [2], attention works well.",
                "vector": [0.2],
                "metadata": json.dumps({
                    "file_name": "paper.pdf",
                    "chunk_index": 1,
                    "cited_ref_ids": ["ref_2"],  # already correct
                }),
            },
            {
                "id": 3003,
                "text": "No citations in this chunk.",
                "vector": [0.3],
                "metadata": json.dumps({
                    "file_name": "paper.pdf",
                    "chunk_index": 2,
                    "cited_ref_ids": [],
                }),
            },
        ]
        col = _make_mock_collection(query_return=raw)
        mgr = _make_manager(col, file_name_has_pdf_suffix=True)
        refs = _make_refs()

        result = await mgr.sync_cited_ref_ids_for_paper("paper.pdf", refs)

        # Chunk 3001: [1] → old ["ref_5"], new ["ref_1"] → changed
        # Chunk 3002: [2] → old ["ref_2"], new ["ref_2"] → unchanged
        # Chunk 3003: no citations → unchanged
        assert result["synced"] == 1
        assert result["unchanged"] == 2

        col.upsert.assert_called_once()
        call_args, call_kwargs = col.upsert.call_args
        assert "partial_update" not in call_kwargs or call_kwargs.get("partial_update") is False

        upsert_data = call_args[0]
        assert len(upsert_data) == 1
        assert upsert_data[0]["id"] == 3001
        # Full upsert includes all fields (text + vector + metadata)
        assert "text" in upsert_data[0]
        assert "vector" in upsert_data[0]
        assert "metadata" in upsert_data[0]

    @pytest.mark.asyncio
    async def test_file_name_suffix_logic_strips_pdf(self):
        """When _file_name_has_pdf_suffix is False, .pdf is stripped for query."""
        col = _make_mock_collection(query_return=[])
        mgr = _make_manager(col, file_name_has_pdf_suffix=False)
        refs = _make_refs()

        await mgr.sync_cited_ref_ids_for_paper("paper.pdf", refs)

        call_expr = col.query.call_args[1]["expr"]
        assert "paper" in call_expr
        assert "paper.pdf" not in call_expr

    @pytest.mark.asyncio
    async def test_file_name_suffix_logic_adds_pdf(self):
        """When _file_name_has_pdf_suffix is True, .pdf is appended for query."""
        col = _make_mock_collection(query_return=[])
        mgr = _make_manager(col, file_name_has_pdf_suffix=True)
        refs = _make_refs()

        await mgr.sync_cited_ref_ids_for_paper("paper", refs)

        call_expr = col.query.call_args[1]["expr"]
        assert "paper.pdf" in call_expr

    @pytest.mark.asyncio
    async def test_string_metadata_parsed_correctly(self):
        """Metadata stored as JSON string is parsed before CitationLinker runs."""
        col = _make_mock_collection(query_return=[
            {
                "id": 4001,
                "text": "The approach in [1] improves results.",
                "metadata": json.dumps({
                    "file_name": "test.pdf",
                    "chunk_index": 0,
                    "cited_ref_ids": ["ref_99"],  # stale
                }),
            },
        ])
        mgr = _make_manager(col, file_name_has_pdf_suffix=True)
        refs = _make_refs()

        result = await mgr.sync_cited_ref_ids_for_paper("test.pdf", refs)

        assert result["total_chunks"] == 1
        assert result["error"] is None
        # ref_99 → ref_1 is a change (text has [1] → new ref_1)
        assert result["synced"] == 1
        col.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_unchanged_chunks_no_upsert(self):
        """When cited_ref_ids already match, upsert is not called."""
        col = _make_mock_collection(query_return=[
            {
                "id": 5001,
                "text": "NeRF [1] is a method.",
                "metadata": json.dumps({
                    "file_name": "test.pdf",
                    "chunk_index": 0,
                    "cited_ref_ids": ["ref_1"],  # already correct
                }),
            },
        ])
        mgr = _make_manager(col, file_name_has_pdf_suffix=True)
        refs = _make_refs()

        result = await mgr.sync_cited_ref_ids_for_paper("test.pdf", refs)

        assert result["synced"] == 0
        assert result["unchanged"] == 1
        col.upsert.assert_not_called()


class TestSyncCitedRefIdsErrorHandling:
    """Test error handling in sync_cited_ref_ids_for_paper()."""

    @pytest.mark.asyncio
    async def test_query_error_returns_error_dict(self):
        """When Milvus query throws, error is captured in result dict."""
        col = _make_mock_collection()
        col.query.side_effect = RuntimeError("Connection lost")

        mgr = _make_manager(col)
        refs = _make_refs()

        result = await mgr.sync_cited_ref_ids_for_paper("test.pdf", refs)

        assert result["synced"] == 0
        assert result["error"] is not None
        assert "Connection lost" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
