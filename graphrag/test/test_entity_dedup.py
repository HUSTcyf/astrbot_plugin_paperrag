"""Unit tests for entity name deduplication — acronyms, aliases, initials matching."""
import pytest
from graphrag.graph_builder import (
    _is_short_name,
    _initials_of,
    _FULL_NAME_WITH_ACRONYM,
)


class TestIsShortName:
    """_is_short_name is a trivial length gate, NOT a regex heuristic.

    It is intentionally permissive — false positives are harmless
    (just an extra dict lookup). The actual dedup decision is made by
    contextual evidence: canonical registry + same-paper co-occurrence.
    """

    def test_short_names_pass(self):
        assert _is_short_name("CNN")
        assert _is_short_name("BERT")
        assert _is_short_name("LSTM")
        assert _is_short_name("3DGS")
        assert _is_short_name("NeRF")
        assert _is_short_name("GPT3")
        assert _is_short_name("GPT-3")

    def test_too_short_rejected(self):
        assert not _is_short_name("X")
        assert not _is_short_name("A")

    def test_too_long_rejected(self):
        assert not _is_short_name("VERYLONGNAME")
        assert not _is_short_name("Convolutional")

    def test_spaces_rejected(self):
        assert not _is_short_name("our method")
        assert not _is_short_name("3D Gaussian")

    def test_boundary_lengths(self):
        assert _is_short_name("AB")       # len=2
        assert _is_short_name("12345678")  # len=8
        assert not _is_short_name("A")     # len=1
        assert not _is_short_name("123456789")  # len=9

    def test_short_regular_words_pass_by_design(self):
        """_is_short_name is intentionally permissive. Short regular words
        like 'method' pass the gate, but the actual dedup decision requires
        contextual evidence (registry hit or co-occurrence match)."""
        assert _is_short_name("method")
        assert _is_short_name("Gaussian")
        assert _is_short_name("cnn")
        assert _is_short_name("3D")

    def test_longer_regular_words_rejected_by_length(self):
        """Words > 8 chars are rejected by the length gate — they can't be acronyms."""
        assert not _is_short_name("Attention")   # 9 chars
        assert not _is_short_name("Transformer")  # 11 chars
        assert not _is_short_name("Classification")


class TestInitialsOf:
    def test_standard_full_name(self):
        assert _initials_of("Neural Radiance Fields") == "NRF"

    def test_digit_prefix_full_name(self):
        assert _initials_of("3D Gaussian Splatting") == "3DGS"

    def test_hyphenated_full_name(self):
        assert _initials_of("Generative Pre-trained Transformer") == "GPTT"

    def test_single_word(self):
        assert _initials_of("BERT") == "BERT"

    def test_with_parenthetical_acronym(self):
        assert _initials_of("3D Gaussian Splatting (3DGS)") == "3DGS"

    def test_with_numbered_variant(self):
        assert _initials_of("GPT 3") == "GPT3"


class TestFullNameWithAcronymRegex:
    def test_extracts_both_parts(self):
        m = _FULL_NAME_WITH_ACRONYM.match("3D Gaussian Splatting (3DGS)")
        assert m is not None
        assert m.group(1).strip() == "3D Gaussian Splatting"
        assert m.group(2) == "3DGS"

    def test_extracts_mixed_case_acronym(self):
        m = _FULL_NAME_WITH_ACRONYM.match("Neural Radiance Fields (NeRF)")
        assert m is not None
        assert m.group(1).strip() == "Neural Radiance Fields"
        assert m.group(2) == "NeRF"

    def test_no_acronym_parenthetical(self):
        assert _FULL_NAME_WITH_ACRONYM.match("BERT") is None
        assert _FULL_NAME_WITH_ACRONYM.match("Attention (is all you need)") is None

    def test_same_name_acronym(self):
        m = _FULL_NAME_WITH_ACRONYM.match("BERT (BERT)")
        assert m is not None
        assert m.group(1).strip() == "BERT"
        assert m.group(2) == "BERT"


class TestInitialsVsAcronymMatching:
    """End-to-end: initials extraction should match expected acronyms."""

    def test_3dgs_matches(self):
        assert _initials_of("3D Gaussian Splatting") == "3DGS"

    def test_nerf_matches(self):
        assert _initials_of("Neural Radiance Fields") == "NRF"

    def test_cnn_matches(self):
        assert _initials_of("Convolutional Neural Network") == "CNN"

    def test_rnn_matches(self):
        assert _initials_of("Recurrent Neural Network") == "RNN"


class TestSamePaperCooccurrence:
    """Simulate the same-paper co-occurrence flow:
    1. First, "3D Gaussian Splatting (3DGS)" is seen → registry populated
    2. Later, "3DGS" appears → is_short_name passes → registry hit → normalized
    """

    def test_full_then_acronym_normalization(self):
        full_name = "3D Gaussian Splatting"
        acronym = "3DGS"
        # Both are short names (gate passes)
        assert _is_short_name(acronym)
        # Registry lookup would find full_name from prior registration
        # (actual registry test is in _normalize_entity_name integration)

    def test_acronym_without_full_name_needs_cooccurrence(self):
        """If 'NRF' appears without prior 'Neural Radiance Fields (NRF)',
        it still passes the gate. Dedup requires same-paper co-occurrence."""
        assert _is_short_name("NRF")
        # But without a full name with matching initials in the same paper,
        # no normalization happens — this is correct behavior.


class TestNormalizeEntityNameIntegration:
    """Test the full _normalize_entity_name flow via MultimodalGraphBuilder."""

    @pytest.fixture
    def builder(self):
        from graphrag.graph_builder import MultimodalGraphBuilder
        from graphrag.graph_rag_engine import GraphRAGConfig
        return MultimodalGraphBuilder(config=GraphRAGConfig())

    def test_full_name_with_acronym(self, builder):
        result = builder._normalize_entity_name(
            "3D Gaussian Splatting (3DGS)",
            entity_type="Method",
            chunk_id="2103.00020.pdf"
        )
        assert result == "3D Gaussian Splatting"
        assert builder._canonical_registry["3dgs"] == "3D Gaussian Splatting"

    def test_acronym_after_registration(self, builder):
        # First: register full name+acronym
        builder._normalize_entity_name(
            "3D Gaussian Splatting (3DGS)",
            entity_type="Method",
            chunk_id="2103.00020.pdf"
        )
        # Then: bare acronym normalizes
        result = builder._normalize_entity_name(
            "3DGS",
            entity_type="Method",
            chunk_id="2103.00020.pdf"
        )
        assert result == "3D Gaussian Splatting"

    def test_same_paper_cooccurrence(self, builder):
        # First chunk registers full name
        builder._normalize_entity_name(
            "Neural Radiance Fields",
            entity_type="Method",
            chunk_id="2103.00020.pdf"
        )
        # Later chunk in SAME paper uses acronym
        result = builder._normalize_entity_name(
            "NRF",
            entity_type="Method",
            chunk_id="2103.00020.pdf"
        )
        assert result == "Neural Radiance Fields"

    def test_cross_paper_initials(self, builder):
        # Paper A registers full name
        builder._normalize_entity_name(
            "Neural Radiance Fields",
            entity_type="Method",
            chunk_id="paper_a.pdf"
        )
        # Paper B uses acronym — cross-paper initials match
        result = builder._normalize_entity_name(
            "NRF",
            entity_type="Method",
            chunk_id="paper_b.pdf"
        )
        assert result == "Neural Radiance Fields"

    def test_generic_self_reference_still_works(self, builder):
        result = builder._normalize_entity_name(
            "our method",
            entity_type="Method",
            chunk_id="2103.00020.pdf"
        )
        assert result == "2103.00020"

    def test_short_regular_word_no_false_dedup(self, builder):
        """Short regular words like 'CNN' pass the gate but without a
        matching full name in the same paper or cross-paper registry,
        they stay unchanged."""
        result = builder._normalize_entity_name(
            "method",
            entity_type="Method",
            chunk_id="paper_a.pdf"
        )
        assert result == "method"
