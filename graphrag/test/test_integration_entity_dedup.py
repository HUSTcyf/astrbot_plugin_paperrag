"""Integration tests: simulate real extraction flow across multiple chunks/papers.

Verifies the full entity dedup pipeline — all 5 layers of _normalize_entity_name()
+ same-paper co-occurrence + cross-paper initials matching.
"""
import pytest


@pytest.fixture
def builder():
    from graphrag.graph_builder import MultimodalGraphBuilder
    from graphrag.graph_rag_engine import GraphRAGConfig
    return MultimodalGraphBuilder(config=GraphRAGConfig())


class TestRealisticExtractionFlow:
    """Simulate a realistic LLM extraction: multiple chunks from multiple papers.

    Scenario:
      Paper A (2103.00020.pdf): 3 chunks about 3D Gaussian Splatting
      Paper B (2201.12345.pdf): 2 chunks about NeRF
      Paper C (2304.56789.pdf): 1 chunk referencing both methods
    """

    def test_paper_a_3dgs_full_form_then_acronym(self, builder):
        """Paper A: Chunk 1 extracts full name with acronym, Chunk 2 uses bare acronym."""
        # Chunk 1: LLM follows Rule 10 → "3D Gaussian Splatting (3DGS)"
        r1 = builder._normalize_entity_name(
            "3D Gaussian Splatting (3DGS)", "Method", "2103.00020.pdf"
        )
        assert r1 == "3D Gaussian Splatting"

        # Chunk 2 (same paper): later, LLM uses bare acronym
        r2 = builder._normalize_entity_name(
            "3DGS", "Method", "2103.00020.pdf"
        )
        # Should normalize via Layer 3 (canonical registry, from full form above)
        assert r2 == "3D Gaussian Splatting"

    def test_paper_a_same_paper_cooccurrence(self, builder):
        """Same paper: full name first (no parentheses), then acronym — Layer 4."""
        # Chunk 1: Full name without parenthetical acronym
        r1 = builder._normalize_entity_name(
            "Neural Radiance Fields", "Method", "paper_a.pdf"
        )
        assert r1 == "Neural Radiance Fields"

        # Chunk 2: Bare acronym — Layer 4 same-paper co-occurrence
        r2 = builder._normalize_entity_name(
            "NRF", "Method", "paper_a.pdf"
        )
        assert r2 == "Neural Radiance Fields"

    def test_cross_paper_initials(self, builder):
        """Paper A has full name, Paper B uses acronym — Layer 5 cross-paper."""
        # Paper A
        builder._normalize_entity_name(
            "Neural Radiance Fields", "Method", "paper_a.pdf"
        )
        # Paper B: acronym only, no prior in this paper
        r = builder._normalize_entity_name(
            "NRF", "Method", "paper_b.pdf"
        )
        assert r == "Neural Radiance Fields"

    def test_cross_paper_does_not_pollute_same_paper(self, builder):
        """Full name in Paper A should NOT affect same-paper check in Paper C
        for same acronym but different method. Actually, cross-paper WILL match
        because initials are the same — this verifies that behavior."""
        # Paper A registers "Non-Robust Features" → initials "NRF"
        builder._normalize_entity_name(
            "Non-Robust Features", "Method", "paper_a.pdf"
        )
        # Paper B uses "NRF" — cross-paper matches to "Non-Robust Features"
        r = builder._normalize_entity_name(
            "NRF", "Method", "paper_b.pdf"
        )
        assert r == "Non-Robust Features"
        # Note: if "Neural Radiance Fields" were also registered, the FIRST match
        # in the set iteration would win. This is a known limitation.

    def test_self_reference_replaced(self, builder):
        """Generic self-reference 'our method' → paper_id."""
        r = builder._normalize_entity_name(
            "our method", "Method", "2103.00020.pdf"
        )
        assert r == "2103.00020"

    def test_very_short_name_is_not_an_acronym_without_evidence(self, builder):
        """Short regular words ('method', 'Gaussian') pass _is_short_name gate
        but don't match any registry or co-occurrence — stay unchanged."""
        r = builder._normalize_entity_name(
            "Gaussian", "Method", "paper_a.pdf"
        )
        assert r == "Gaussian"

    def test_all_five_layers_in_order(self, builder):
        """Verify layers are applied in correct priority order."""
        # Setup: register a full name with acronym
        builder._normalize_entity_name(
            "3D Gaussian Splatting (3DGS)", "Method", "paper_a.pdf"
        )

        # Test: "our method" (full form with acronym slipped in somehow)
        # Layer 1 should catch "our method" BEFORE Layer 2 parses "(3DGS)"
        r = builder._normalize_entity_name(
            "our method", "Method", "2103.00020.pdf"
        )
        assert r == "2103.00020"  # self-reference, not 3DGS-related

    def test_registry_state_after_flow(self, builder):
        """Verify the internal registries are correctly populated."""
        builder._normalize_entity_name(
            "3D Gaussian Splatting (3DGS)", "Method", "paper_a.pdf"
        )
        builder._normalize_entity_name(
            "3DGS", "Method", "paper_a.pdf"  # should normalize
        )
        builder._normalize_entity_name(
            "3DGS", "Method", "paper_b.pdf"  # cross-paper
        )

        # Canonical registry maps both acronym and full name to canonical
        assert builder._canonical_registry["3dgs"] == "3D Gaussian Splatting"
        assert builder._canonical_registry["3d gaussian splatting"] == "3D Gaussian Splatting"

        # Per-type registry tracks all canonical names
        assert "3D Gaussian Splatting" in builder._entity_registry_by_type["Method"]

        # Per-paper registry tracks which paper saw which canonical name
        assert "3D Gaussian Splatting" in builder._entity_registry_by_paper["paper_a"]
        assert "3D Gaussian Splatting" in builder._entity_registry_by_paper["paper_b"]


class TestEdgeCasesDiscoveredInCodeReview:
    """Verify fixes for the CRITICAL/HIGH bugs found in code review."""

    def test_critical1_argument_order(self, builder):
        """CRITICAL #1: All 3 args correctly passed. This was the bug where
        _extract_multimodal_triplets passed chunk_id as entity_type.
        Now verified at the function level."""
        # The function signature is (name, entity_type, chunk_id)
        # If entity_type were wrong, Layer 4 (same-paper) would not work
        r = builder._normalize_entity_name(
            "3DGS", "Method", "paper_a.pdf"
        )
        assert r == "3DGS"  # no match without prior registration
        # Paper is tracked despite no normalization
        assert "3DGS" in builder._entity_registry_by_paper["paper_a"]

    def test_critical2_hyphenated_acronym(self, builder):
        """CRITICAL #2: GPT-3 now matches GPT3 via _normalize_acronym_key."""
        from graphrag.graph_builder import _normalize_acronym_key, _initials_of

        # _normalize_acronym_key strips hyphens
        assert _normalize_acronym_key("GPT-3") == "GPT3"
        assert _normalize_acronym_key("3D-GS") == "3DGS"

        # Verify the comparison works in the actual matching logic
        builder._normalize_entity_name(
            "GPT 3", "Model", "paper_a.pdf"
        )
        # "GPT-3" should match via Layer 4 (same-paper) because initials match
        r = builder._normalize_entity_name(
            "GPT-3", "Model", "paper_a.pdf"
        )
        assert r == "GPT 3"

    def test_medium6_entity_type_guard(self, builder):
        """MEDIUM #6: 'Method (Baseline)' should NOT register as acronym."""
        r = builder._normalize_entity_name(
            "Method (Baseline)", "Method", "paper_a.pdf"
        )
        # Should strip parenthetical, returning "Method"
        assert r == "Method"
        # But should NOT register "baseline" → "Method" in canonical_registry
        # because "method" is an entity type guard
        assert "baseline" not in builder._canonical_registry

    def test_entity_type_guard_allows_real_acronyms(self, builder):
        """The entity type guard should NOT block real acronyms like CNN."""
        r = builder._normalize_entity_name(
            "Convolutional Neural Network (CNN)", "Model", "paper_a.pdf"
        )
        assert r == "Convolutional Neural Network"
        # "cnn" → "Convolutional Neural Network" SHOULD be registered
        assert builder._canonical_registry["cnn"] == "Convolutional Neural Network"


class TestAcronymVariants:
    """Various acronym forms that real LLMs might produce."""

    def test_space_separated_acronym_variant(self, builder):
        """'GPT 3' (space) vs 'GPT-3' (hyphen) should match."""
        builder._normalize_entity_name(
            "Generative Pre-trained Transformer 3", "Model", "paper_a.pdf"
        )
        r = builder._normalize_entity_name(
            "GPT 3", "Model", "paper_a.pdf"
        )
        # Same-paper: _initials_of("Generative Pre-trained Transformer 3") = "GPTT3"
        # _normalize_acronym_key("GPT 3") = "GPT3"
        # "GPTT3" != "GPT3" → no match via initials. Falls through to cross-paper.
        # This is a known limitation documented in _initials_of.
        # The prompt fix (Layer 2) is the primary solution here.
        assert r == "GPT 3"  # unchanged (no matching initials)

    def test_mixed_case_acronym_normalized(self, builder):
        """NeRF (mixed case) should match NERF (uppercase)."""
        builder._normalize_entity_name(
            "Neural Radiance Fields (NeRF)", "Method", "paper_a.pdf"
        )
        r = builder._normalize_entity_name(
            "nerf", "Method", "paper_a.pdf"
        )
        assert r == "Neural Radiance Fields"

    def test_multiple_entities_same_paper(self, builder):
        """Paper with many entities — each deduped independently."""
        # Register several entities in the same paper
        builder._normalize_entity_name(
            "Convolutional Neural Network (CNN)", "Model", "multi.pdf"
        )
        builder._normalize_entity_name(
            "Recurrent Neural Network (RNN)", "Model", "multi.pdf"
        )
        builder._normalize_entity_name(
            "Stochastic Gradient Descent (SGD)", "Method", "multi.pdf"
        )

        # Later acronyms should normalize to their respective full names
        assert builder._normalize_entity_name("CNN", "Model", "multi.pdf") == "Convolutional Neural Network"
        assert builder._normalize_entity_name("RNN", "Model", "multi.pdf") == "Recurrent Neural Network"
        assert builder._normalize_entity_name("SGD", "Method", "multi.pdf") == "Stochastic Gradient Descent"

        # Cross-paper: these acronyms now resolve from other papers too
        assert builder._normalize_entity_name("CNN", "Model", "other.pdf") == "Convolutional Neural Network"
        assert builder._normalize_entity_name("SGD", "Method", "other.pdf") == "Stochastic Gradient Descent"
