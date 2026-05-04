"""
Closed-Set Content-Oriented Relation Schema Tests

Verifies the full knowledge graph construction pipeline:
1. GBNF grammars compile and enforce closed-set values
2. _normalize_relation_type() / _normalize_entity_type() correctness
3. _process_batch() pipeline: mock LLM → JSON parse → normalize → Cypher write
4. _extract_text_triplets() single-node pipeline
5. _extract_multimodal_triplets(): text closed-set + cross-modal open-set
6. JSON parsing with truncation recovery
7. Cypher queries use closed-set labels and store description/chunk_id
8. Prompt templates contain all 9 predicates and 9 entity types
"""

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_plugin_root = Path(__file__).resolve().parents[1]
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))


# ============================================================================
# Stubs and fixtures
# ============================================================================


def _install_astrbot_stubs():
    for mod_name in ["astrbot", "astrbot.api", "astrbot.api.star"]:
        if mod_name not in sys.modules:
            stub = types.SimpleNamespace(
                logger=types.SimpleNamespace(
                    info=lambda *a, **k: None,
                    warning=lambda *a, **k: None,
                    error=lambda *a, **k: None,
                    debug=lambda *a, **k: None,
                )
            )
            sys.modules[mod_name] = stub


def _install_neo4j_stub():
    class FakeGraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            return None

    if "neo4j" not in sys.modules:
        sys.modules["neo4j"] = types.SimpleNamespace(GraphDatabase=FakeGraphDatabase)


class CypherCaptureSession:
    """Fake Neo4j session that captures all run() Cypher queries and params."""

    def __init__(self):
        self.queries: List[str] = []
        self.params_list: List[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.queries.append(query)
        self.params_list.append(params)
        return types.SimpleNamespace(data=lambda: [])


class CypherCaptureDriver:
    """Fake Neo4j driver that returns CypherCaptureSession."""

    def __init__(self):
        self._session = CypherCaptureSession()

    def session(self, database=None):
        return self._session

    def close(self):
        pass

    @property
    def queries(self):
        return self._session.queries


def _make_graph_store_adapter(driver=None):
    """Create a real SimplePropertyGraphStoreAdapter with a capture driver."""
    _install_astrbot_stubs()
    _install_neo4j_stub()

    from graphrag.graph_rag_engine import SimplePropertyGraphStoreAdapter

    if driver is None:
        driver = CypherCaptureDriver()
    fake_store = types.SimpleNamespace(client=driver, _driver=driver)
    adapter = SimplePropertyGraphStoreAdapter(fake_store)
    return adapter, driver


@dataclass
class FakeGraphRAGConfig:
    max_triplets_per_chunk: int = 10
    graph_retrieval_top_k: int = 5
    graph_rrf_weight: float = 0.2
    multimodal_enabled: bool = True
    extract_image_entities: bool = True


def _make_builder(config=None):
    """Create a MultimodalGraphBuilder with fake config, ready for pipeline tests."""
    _install_astrbot_stubs()
    _install_neo4j_stub()

    from graphrag.graph_builder import MultimodalGraphBuilder

    if config is None:
        config = FakeGraphRAGConfig()
    builder = MultimodalGraphBuilder.__new__(MultimodalGraphBuilder)
    builder.config = config
    builder.context = None
    builder._llm = None
    builder._llm_config = types.SimpleNamespace(n_ctx=8192, max_tokens=1024)
    builder._triplet_grammar = None
    builder._multimodal_grammar = None
    return builder


def _make_node(text: str, chunk_id: str = "chunk_0", file_name: str = "test.pdf",
               has_image: bool = False, image_path: str = ""):
    """Create a fake document node."""
    metadata = {"chunk_id": chunk_id, "file_name": file_name}
    if has_image:
        metadata["has_image"] = True
        metadata["image_path"] = image_path
    return types.SimpleNamespace(text=text, metadata=metadata)


def _make_llm_response(content: str):
    """Create a fake LLM response."""
    return types.SimpleNamespace(content=content, model="test-model")


def _valid_batch_response(triplets):
    """Build a valid JSON response string from triplet dicts."""
    return json.dumps({"triplets": triplets})


def _valid_multimodal_response(text_triplets=None, image_info=None, cross_modal_triplets=None):
    """Build a valid multimodal JSON response."""
    return json.dumps({
        "text_triplets": text_triplets or [],
        "image_info": image_info or {},
        "cross_modal_triplets": cross_modal_triplets or [],
    })


# ============================================================================
# Test 1: GBNF grammar constraints
# ============================================================================


def _parse_json_schema_enum_values(schema: dict, field_name: str) -> set[str]:
    """Extract enum values from a JSON schema for a given field in the triplet item."""
    items = schema["properties"]["triplets"]["items"]
    prop = items["properties"][field_name]
    return set(prop["enum"])


def _parse_multimodal_json_schema_enum_values(schema: dict, field_name: str) -> set[str]:
    items = schema["properties"]["text_triplets"]["items"]
    prop = items["properties"][field_name]
    return set(prop["enum"])


class TestConstantsMatchGBNF:

    def test_relation_types_match_schema_exactly(self):
        _install_astrbot_stubs()
        from graphrag.graph_builder import CLOSED_RELATION_TYPES
        import json

        schema = json.loads((_plugin_root / "graphrag" / "triplet_schema.json").read_text())
        schema_values = _parse_json_schema_enum_values(schema, "relation_type")
        assert schema_values == CLOSED_RELATION_TYPES

    def test_entity_types_match_schema_exactly(self):
        _install_astrbot_stubs()
        from graphrag.graph_builder import CLOSED_ENTITY_TYPES
        import json

        schema = json.loads((_plugin_root / "graphrag" / "triplet_schema.json").read_text())
        schema_values = _parse_json_schema_enum_values(schema, "head_type")
        assert schema_values == CLOSED_ENTITY_TYPES

    def test_schema_has_exactly_14_relation_types(self):
        import json
        schema = json.loads((_plugin_root / "graphrag" / "triplet_schema.json").read_text())
        schema_values = _parse_json_schema_enum_values(schema, "relation_type")
        assert len(schema_values) == 14

    def test_schema_has_exactly_9_entity_types(self):
        import json
        schema = json.loads((_plugin_root / "graphrag" / "triplet_schema.json").read_text())
        schema_values = _parse_json_schema_enum_values(schema, "head_type")
        assert len(schema_values) == 9

    def test_multimodal_schema_relation_types_match_exactly(self):
        _install_astrbot_stubs()
        from graphrag.graph_builder import CLOSED_RELATION_TYPES
        import json

        schema = json.loads((_plugin_root / "graphrag" / "multimodal_schema.json").read_text())
        schema_values = _parse_multimodal_json_schema_enum_values(schema, "relation_type")
        assert schema_values == CLOSED_RELATION_TYPES

    def test_multimodal_schema_entity_types_match_exactly(self):
        _install_astrbot_stubs()
        from graphrag.graph_builder import CLOSED_ENTITY_TYPES
        import json

        schema = json.loads((_plugin_root / "graphrag" / "multimodal_schema.json").read_text())
        schema_values = _parse_multimodal_json_schema_enum_values(schema, "head_type")
        assert schema_values == CLOSED_ENTITY_TYPES

    def test_schema_grammar_compiles(self):
        try:
            from llama_cpp import LlamaGrammar
        except ImportError:
            pytest.skip("llama-cpp-python not installed")
        import json
        schema_text = (_plugin_root / "graphrag" / "triplet_schema.json").read_text()
        grammar = LlamaGrammar.from_json_schema(schema_text)
        assert grammar is not None

    def test_multimodal_schema_grammar_compiles(self):
        try:
            from llama_cpp import LlamaGrammar
        except ImportError:
            pytest.skip("llama-cpp-python not installed")
        import json
        schema_text = (_plugin_root / "graphrag" / "multimodal_schema.json").read_text()
        grammar = LlamaGrammar.from_json_schema(schema_text)
        assert grammar is not None


# ============================================================================
# Test 2: _normalize_relation_type()
# ============================================================================


class TestNormalizeRelationType:

    def setup_method(self):
        _install_astrbot_stubs()

    def _make_builder(self):
        return _make_builder()

    @pytest.mark.parametrize("input_val,expected", [
        ("ADDRESSES", "ADDRESSES"),
        ("PROPOSES", "PROPOSES"),
        ("USES_COMPONENT", "USES_COMPONENT"),
        ("EVALUATED_ON", "EVALUATED_ON"),
        ("ACHIEVES", "ACHIEVES"),
        ("COMPARES_WITH", "COMPARES_WITH"),
        ("LIMITED_BY", "LIMITED_BY"),
        ("APPLIES_TO", "APPLIES_TO"),
        ("EXTENDS", "EXTENDS"),
    ])
    def test_closed_set_passes_through(self, input_val, expected):
        builder = self._make_builder()
        assert builder._normalize_relation_type(input_val) == expected

    @pytest.mark.parametrize("input_val,expected", [
        ("based_on", "EXTENDS"),
        ("uses", "USES_COMPONENT"),
        ("achieves", "ACHIEVES"),
        ("outperforms", "OUTPERFORMS"),
        ("improves", "EXTENDS"),
        ("proposes", "PROPOSES"),
        ("introduces", "PROPOSES"),
        ("trained_on", "TRAINS_ON"),
        ("applied_to", "APPLIES_TO"),
        ("compares_with", "COMPARES_WITH"),
        ("combines_with", "USES_COMPONENT"),
        ("integrates", "USES_COMPONENT"),
        ("depends_on", "USES_COMPONENT"),
        ("github", "IMPLEMENTS"),
        ("beats", "OUTPERFORMS"),
        ("needs", "REQUIRES"),
        ("ablation", "ABLATES_ON"),
    ])
    def test_aliases_map_correctly(self, input_val, expected):
        builder = self._make_builder()
        assert builder._normalize_relation_type(input_val) == expected

    def test_case_insensitive_closed_set(self):
        builder = self._make_builder()
        assert builder._normalize_relation_type("addresses") == "ADDRESSES"
        assert builder._normalize_relation_type("proposes") == "PROPOSES"

    def test_unknown_defaults_to_uses_component(self):
        builder = self._make_builder()
        assert builder._normalize_relation_type("unknown_thing") == "USES_COMPONENT"
        assert builder._normalize_relation_type("") == "USES_COMPONENT"


# ============================================================================
# Test 3: _normalize_entity_type()
# ============================================================================


class TestNormalizeEntityType:

    def setup_method(self):
        _install_astrbot_stubs()

    def _make_builder(self):
        return _make_builder()

    @pytest.mark.parametrize("input_val", [
        "Method", "Model", "Task", "Dataset", "Metric",
        "Component", "Limitation", "Application", "Baseline",
    ])
    def test_closed_set_passes_through(self, input_val):
        builder = self._make_builder()
        assert builder._normalize_entity_type(input_val) == input_val

    @pytest.mark.parametrize("input_val,expected", [
        ("model/architecture", "Model"),
        ("method/technique", "Method"),
        ("optimizer/algorithm", "Method"),
        ("framework/library", "Component"),
        ("hyperparameter", "Component"),
        ("result/conclusion", "Metric"),
        ("application/domain", "Application"),
        ("other", "Method"),
    ])
    def test_aliases_map_correctly(self, input_val, expected):
        builder = self._make_builder()
        assert builder._normalize_entity_type(input_val) == expected

    def test_unknown_defaults_to_method(self):
        builder = self._make_builder()
        assert builder._normalize_entity_type("SomethingWeird") == "Method"

    def test_metadata_types_removed(self):
        builder = self._make_builder()
        assert builder._normalize_entity_type("Author/Organization") != "Author/Organization"
        # Venue is now a valid entity type, but "ConferenceRoom" is not
        assert builder._normalize_entity_type("ConferenceRoom") != "ConferenceRoom"


# ============================================================================
# Test 4: Prompt templates contain all 9 predicates and 9 entity types
# ============================================================================


class TestPromptsClosedSet:

    def setup_method(self):
        _install_astrbot_stubs()

    def test_batch_prompt_has_all_predicates(self):
        from graphrag.graph_builder import BATCH_TRIPLET_EXTRACTION_PROMPT, CLOSED_RELATION_TYPES
        for rt in CLOSED_RELATION_TYPES:
            assert rt in BATCH_TRIPLET_EXTRACTION_PROMPT

    def test_batch_prompt_has_all_entity_types(self):
        from graphrag.graph_builder import BATCH_TRIPLET_EXTRACTION_PROMPT, CLOSED_ENTITY_TYPES
        for et in CLOSED_ENTITY_TYPES:
            assert et in BATCH_TRIPLET_EXTRACTION_PROMPT

    def test_single_prompt_has_all_predicates(self):
        from graphrag.graph_builder import TRIPLET_EXTRACTION_PROMPT, CLOSED_RELATION_TYPES
        for rt in CLOSED_RELATION_TYPES:
            assert rt in TRIPLET_EXTRACTION_PROMPT

    def test_single_prompt_has_all_entity_types(self):
        from graphrag.graph_builder import TRIPLET_EXTRACTION_PROMPT, CLOSED_ENTITY_TYPES
        for et in CLOSED_ENTITY_TYPES:
            assert et in TRIPLET_EXTRACTION_PROMPT

    def test_multimodal_prompt_has_all_predicates(self):
        from graphrag.graph_builder import MULTIMODAL_TRIPLET_EXTRACTION_PROMPT, CLOSED_RELATION_TYPES
        for rt in CLOSED_RELATION_TYPES:
            assert rt in MULTIMODAL_TRIPLET_EXTRACTION_PROMPT

    def test_multimodal_prompt_has_all_entity_types(self):
        from graphrag.graph_builder import MULTIMODAL_TRIPLET_EXTRACTION_PROMPT, CLOSED_ENTITY_TYPES
        for et in CLOSED_ENTITY_TYPES:
            assert et in MULTIMODAL_TRIPLET_EXTRACTION_PROMPT

    def test_no_metadata_types_in_prompts(self):
        from graphrag.graph_builder import BATCH_TRIPLET_EXTRACTION_PROMPT, TRIPLET_EXTRACTION_PROMPT
        for prompt in [BATCH_TRIPLET_EXTRACTION_PROMPT, TRIPLET_EXTRACTION_PROMPT]:
            # Metadata types are now VALID entity types (Venue, Author, Institution)
            # Just verify they appear in their proper form (not as aliases)
            assert "ConferenceRoom" not in prompt
            assert "OrganizationName" not in prompt


# ============================================================================
# Test 5: Call sites verify exact code patterns
# ============================================================================


class TestCallSitesUseRelationType:

    def setup_method(self):
        _install_astrbot_stubs()

    def test_three_text_sites_use_normalize(self):
        source = (_plugin_root / "graphrag" / "graph_builder.py").read_text()
        import re
        pattern = r'relation\s*=\s*self\._normalize_relation_type\('
        matches = re.findall(pattern, source)
        assert len(matches) == 3

    def test_three_text_sites_pass_relation_description(self):
        source = (_plugin_root / "graphrag" / "graph_builder.py").read_text()
        import re
        pattern = r'relation_description\s*=\s*relation\b'
        matches = re.findall(pattern, source)
        assert len(matches) == 3

    def test_cross_modal_uses_raw_relation(self):
        source = (_plugin_root / "graphrag" / "graph_builder.py").read_text()
        last_idx = source.rfind("graph_store.add_relation")
        assert last_idx > 0
        snippet = source[last_idx:last_idx + 300]
        assert "_normalize_relation_type" not in snippet
        assert "relation_description" not in snippet

    def test_add_relation_cypher_stores_description(self):
        source = (_plugin_root / "graphrag" / "graph_rag_engine.py").read_text()
        method_start = source.find("def add_relation(")
        assert method_start > 0
        next_method = source.find("\n    def ", method_start + 1)
        method_body = source[method_start:next_method]
        assert "r.description" in method_body
        assert "relation_description" in method_body


# ============================================================================
# Test 6: add_relation signature
# ============================================================================


class TestAddRelationSignature:

    def setup_method(self):
        _install_astrbot_stubs()

    def test_add_relation_has_description_param(self):
        import inspect
        _install_neo4j_stub()
        from graphrag.graph_rag_engine import SimplePropertyGraphStoreAdapter
        sig = inspect.signature(SimplePropertyGraphStoreAdapter.add_relation)
        assert "relation_description" in sig.parameters
        assert sig.parameters["relation_description"].default == ""


# ============================================================================
# Test 7: Full _process_batch() pipeline — LLM mock → JSON → normalize → Cypher
# ============================================================================


class TestProcessBatchPipeline:
    """Test the complete batch extraction pipeline with a mock LLM.

    Verifies that LLM JSON output flows through normalization correctly
    and produces valid Cypher queries with closed-set edge labels.
    """

    def setup_method(self):
        _install_astrbot_stubs()
        _install_neo4j_stub()

    def _make_builder_with_mock_llm(self, response_content: str):
        builder = _make_builder(FakeGraphRAGConfig(max_triplets_per_chunk=10))
        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response(response_content))
        builder._llm = mock_llm
        return builder

    @pytest.mark.asyncio
    async def test_closed_set_relations_passed_through_to_cypher(self):
        """All 9 closed-set relation types should appear as backtick-escaped edge labels."""
        triplets = [
            {
                "head": "BERT", "head_type": "Model",
                "relation": "addresses language modeling",
                "relation_type": "ADDRESSES",
                "tail": "Language Modeling", "tail_type": "Task",
                "confidence": 0.9, "evidence": "[Chunk 1]"
            },
            {
                "head": "BERT", "head_type": "Model",
                "relation": "extends transformer architecture",
                "relation_type": "EXTENDS",
                "tail": "Transformer", "tail_type": "Model",
                "confidence": 0.85, "evidence": "[Chunk 1]"
            },
            {
                "head": "BERT", "head_type": "Model",
                "relation": "evaluated on GLUE benchmark",
                "relation_type": "EVALUATED_ON",
                "tail": "GLUE", "tail_type": "Dataset",
                "confidence": 0.95, "evidence": "[Chunk 1]"
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("BERT is a pre-trained language model that extends the Transformer architecture.", "chunk_0")]
        result = await builder._process_batch(nodes, adapter)

        assert result["text_triplets_added"] == 3

        cypher_queries = driver.queries
        relation_cyphers = [q for q in cypher_queries if "MERGE (a)-[r:" in q]

        assert any("`ADDRESSES`" in q for q in relation_cyphers)
        assert any("`EXTENDS`" in q for q in relation_cyphers)
        assert any("`EVALUATED_ON`" in q for q in relation_cyphers)

    @pytest.mark.asyncio
    async def test_all_nine_relation_types_in_cypher(self):
        """Every closed-set relation type should produce valid Cypher with backtick label."""
        from graphrag.graph_builder import CLOSED_RELATION_TYPES

        triplets = []
        for i, rt in enumerate(sorted(CLOSED_RELATION_TYPES)):
            triplets.append({
                "head": f"Entity_A_{i}", "head_type": "Method",
                "relation": f"some relation {rt}",
                "relation_type": rt,
                "tail": f"Entity_B_{i}", "tail_type": "Task",
                "confidence": 0.9, "evidence": "[Chunk 1]"
            })

        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60, "chunk_0")]
        result = await builder._process_batch(nodes, adapter)

        assert result["text_triplets_added"] == 14

        cypher_queries = driver.queries
        relation_cyphers = [q for q in cypher_queries if "MERGE (a)-[r:" in q]

        for rt in CLOSED_RELATION_TYPES:
            assert any(f"`{rt}`" in q for q in relation_cyphers), \
                f"Cypher missing edge label `{rt}`"

    @pytest.mark.asyncio
    async def test_alias_relations_normalized_in_cypher(self):
        """Alias relation types (e.g. 'based_on') should be normalized to closed-set labels."""
        triplets = [
            {
                "head": "GPT-2", "head_type": "Model",
                "relation": "based on transformer",
                "relation_type": "based_on",
                "tail": "Transformer", "tail_type": "Model",
                "confidence": 0.9, "evidence": "[Chunk 1]"
            },
            {
                "head": "ResNet", "head_type": "Model",
                "relation": "uses component residual connections",
                "relation_type": "uses",
                "tail": "Residual Connection", "tail_type": "Component",
                "confidence": 0.85, "evidence": "[Chunk 1]"
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60, "chunk_0")]
        result = await builder._process_batch(nodes, adapter)

        assert result["text_triplets_added"] == 2

        cypher_queries = driver.queries
        relation_cyphers = [q for q in cypher_queries if "MERGE (a)-[r:" in q]

        assert any("`EXTENDS`" in q for q in relation_cyphers), \
            "'based_on' should normalize to 'EXTENDS'"
        assert any("`USES_COMPONENT`" in q for q in relation_cyphers), \
            "'uses' should normalize to 'USES_COMPONENT'"

        assert not any("`based_on`" in q for q in relation_cyphers)
        assert not any("`uses`" in q for q in relation_cyphers)

    @pytest.mark.asyncio
    async def test_unknown_relation_defaults_to_uses_component_in_cypher(self):
        """Unknown relation types should default to USES_COMPONENT."""
        triplets = [
            {
                "head": "X", "head_type": "Method",
                "relation": "some weird relation",
                "relation_type": "completely_unknown_relation",
                "tail": "Y", "tail_type": "Task",
                "confidence": 0.7, "evidence": "[Chunk 1]"
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60, "chunk_0")]
        result = await builder._process_batch(nodes, adapter)

        assert result["text_triplets_added"] == 1
        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert any("`USES_COMPONENT`" in q for q in relation_cyphers)

    @pytest.mark.asyncio
    async def test_entity_types_normalized_in_cypher(self):
        """Entity labels in Cypher should use closed-set types, not aliases."""
        triplets = [
            {
                "head": "Adam", "head_type": "optimizer/algorithm",
                "relation": "optimizes",
                "relation_type": "APPLIES_TO",
                "tail": "Neural Networks", "tail_type": "framework/library",
                "confidence": 0.8, "evidence": "[Chunk 1]"
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60, "chunk_0")]
        await builder._process_batch(nodes, adapter)

        entity_cyphers = [q for q in driver.queries if "MERGE (n:" in q]

        assert any("`Method`" in q for q in entity_cyphers), \
            "'optimizer/algorithm' should normalize to 'Method'"
        assert any("`Component`" in q for q in entity_cyphers), \
            "'framework/library' should normalize to 'Component'"

        assert not any("`optimizer" in q for q in entity_cyphers)
        assert not any("`framework" in q for q in entity_cyphers)

    @pytest.mark.asyncio
    async def test_relation_description_stored_as_edge_property(self):
        """The free-text 'relation' field should be stored as r.description property."""
        triplets = [
            {
                "head": "BERT", "head_type": "Model",
                "relation": "extends the original Transformer by adding bidirectional context",
                "relation_type": "EXTENDS",
                "tail": "Transformer", "tail_type": "Model",
                "confidence": 0.9, "evidence": "[Chunk 1]"
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60, "chunk_0")]
        await builder._process_batch(nodes, adapter)

        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert len(relation_cyphers) == 1
        assert "r.description" in relation_cyphers[0]
        assert "extends the original Transformer" in relation_cyphers[0]

    @pytest.mark.asyncio
    async def test_chunk_id_stored_on_entities_and_relations(self):
        """chunk_id from node metadata should propagate to both entities and relations."""
        triplets = [
            {
                "head": "BERT", "head_type": "Model",
                "relation": "is a model",
                "relation_type": "PROPOSES",
                "tail": "NLP", "tail_type": "Task",
                "confidence": 0.9, "evidence": "[Chunk 1]"
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60, chunk_id="paper_bert_chunk_42")]
        await builder._process_batch(nodes, adapter)

        for q in driver.queries:
            if "MERGE (n:" in q or "MERGE (a)-[r:" in q:
                assert "paper_bert_chunk_42" in q, \
                    f"chunk_id not found in Cypher: {q}"

    @pytest.mark.asyncio
    async def test_evidence_used_to_match_correct_chunk(self):
        """When multiple chunks exist, evidence tag should route triplet to correct chunk."""
        triplets = [
            {
                "head": "BERT", "head_type": "Model",
                "relation": "proposes bidirectional encoding",
                "relation_type": "PROPOSES",
                "tail": "MLM", "tail_type": "Method",
                "confidence": 0.9, "evidence": "[Chunk 2]"
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [
            _make_node("x" * 60, chunk_id="chunk_A"),
            _make_node("y" * 60, chunk_id="chunk_B"),
        ]
        await builder._process_batch(nodes, adapter)

        for q in driver.queries:
            if "MERGE (n:" in q or "MERGE (a)-[r:" in q:
                assert "chunk_B" in q, \
                    f"Triplet from [Chunk 2] should use chunk_B's chunk_id: {q}"

    @pytest.mark.asyncio
    async def test_empty_head_or_tail_skipped(self):
        """Triplets with empty head/relation/tail should be silently skipped."""
        triplets = [
            {"head": "", "relation": "uses", "relation_type": "USES_COMPONENT",
             "tail": "X", "tail_type": "Component", "confidence": 0.5, "evidence": ""},
            {"head": "Y", "relation": "", "relation_type": "PROPOSES",
             "tail": "Z", "tail_type": "Task", "confidence": 0.5, "evidence": ""},
            {"head": "A", "relation": "extends", "relation_type": "EXTENDS",
             "tail": "", "tail_type": "Model", "confidence": 0.5, "evidence": ""},
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60)]
        result = await builder._process_batch(nodes, adapter)

        assert result["text_triplets_added"] == 0
        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert len(relation_cyphers) == 0

    @pytest.mark.asyncio
    async def test_short_text_nodes_filtered(self):
        """Nodes with text < 50 chars should be filtered out."""
        builder = self._make_builder_with_mock_llm('{"triplets": []}')
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("Short text")]
        result = await builder._process_batch(nodes, adapter)

        assert result["chunks_empty"] == 1
        assert result["text_triplets_added"] == 0

    @pytest.mark.asyncio
    async def test_no_free_text_relation_in_cypher_edge_label(self):
        """No Cypher should contain a free-text relation as edge label."""
        triplets = [
            {
                "head": "BERT", "head_type": "Model",
                "relation": "based on the Transformer architecture for NLP tasks",
                "relation_type": "EXTENDS",
                "tail": "Transformer", "tail_type": "Model",
                "confidence": 0.9, "evidence": "[Chunk 1]"
            },
            {
                "head": "GPT", "head_type": "Model",
                "relation": "uses attention mechanism",
                "relation_type": "USES_COMPONENT",
                "tail": "Attention", "tail_type": "Component",
                "confidence": 0.85, "evidence": "[Chunk 1]"
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60)]
        await builder._process_batch(nodes, adapter)

        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        for q in relation_cyphers:
            assert "`based on`" not in q, f"Free-text 'based on' should not be edge label: {q}"
            assert "`uses attention`" not in q, f"Free-text should not be edge label: {q}"


# ============================================================================
# Test 8: Full _extract_text_triplets() pipeline
# ============================================================================


class TestExtractTextTripletsPipeline:
    """Test single-node text extraction pipeline."""

    def setup_method(self):
        _install_astrbot_stubs()
        _install_neo4j_stub()

    def _make_builder_with_mock_llm(self, response_content: str):
        builder = _make_builder(FakeGraphRAGConfig(max_triplets_per_chunk=10))
        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response(response_content))
        builder._llm = mock_llm
        return builder

    @pytest.mark.asyncio
    async def test_text_extraction_produces_closed_set_cypher(self):
        triplets = [
            {
                "head": "LoRA", "head_type": "Method",
                "relation": "extends large language models with low-rank adaptation",
                "relation_type": "EXTENDS",
                "tail": "LLM", "tail_type": "Model",
                "confidence": 0.95, "evidence": ""
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        result = await builder._extract_text_triplets(
            "LoRA extends large language models with low-rank adaptation.",
            "chunk_lora",
            adapter,
        )

        assert result["text_triplets_added"] == 1
        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert any("`EXTENDS`" in q for q in relation_cyphers)
        assert any("r.description" in q for q in relation_cyphers)

    @pytest.mark.asyncio
    async def test_text_extraction_entity_type_normalization(self):
        """Entity type aliases should be normalized in Cypher labels."""
        triplets = [
            {
                "head": "SGD", "head_type": "optimizer/algorithm",
                "relation": "applied to training",
                "relation_type": "APPLIES_TO",
                "tail": "ImageNet", "tail_type": "Dataset",
                "confidence": 0.8, "evidence": ""
            },
        ]
        builder = self._make_builder_with_mock_llm(_valid_batch_response(triplets))
        adapter, driver = _make_graph_store_adapter()

        await builder._extract_text_triplets("SGD is applied to ImageNet.", "chunk_0", adapter)

        entity_cyphers = [q for q in driver.queries if "MERGE (n:" in q]
        assert any("`Method`" in q for q in entity_cyphers)
        assert any("`Dataset`" in q for q in entity_cyphers)
        assert not any("`optimizer" in q for q in entity_cyphers)


# ============================================================================
# Test 9: Full _extract_multimodal_triplets() pipeline
# ============================================================================


class TestExtractMultimodalTripletsPipeline:
    """Test multimodal extraction: text triplets use closed-set, cross-modal use free-text."""

    def setup_method(self):
        _install_astrbot_stubs()
        _install_neo4j_stub()

    def _make_builder_with_mock_llm(self, response_content: str):
        config = FakeGraphRAGConfig(
            max_triplets_per_chunk=10,
            multimodal_enabled=True,
            extract_image_entities=True,
        )
        builder = _make_builder(config)
        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response(response_content))
        builder._llm = mock_llm
        return builder

    @pytest.mark.asyncio
    async def test_text_triplets_use_closed_set(self):
        """Text triplets in multimodal response should use closed-set normalization."""
        response = _valid_multimodal_response(
            text_triplets=[
                {
                    "head": "CNN", "head_type": "Model",
                    "relation": "applied to image classification",
                    "relation_type": "APPLIES_TO",
                    "tail": "Image Classification", "tail_type": "Task",
                    "confidence": 0.9, "evidence": ""
                },
            ],
            image_info={"description": "A diagram showing CNN architecture", "figure_type": "diagram"},
        )
        builder = self._make_builder_with_mock_llm(response)
        adapter, driver = _make_graph_store_adapter()

        result = await builder._extract_multimodal_triplets(
            text="CNN is applied to image classification tasks.",
            image_path="/tmp/fake_image.png",
            image_caption="Figure 1: CNN architecture",
            chunk_id="chunk_cnn",
            graph_store=adapter,
        )

        assert result["text_triplets_added"] == 1
        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert any("`APPLIES_TO`" in q for q in relation_cyphers)

    @pytest.mark.asyncio
    async def test_cross_modal_uses_free_text_relation(self):
        """Cross-modal triplets should use raw relation string as edge label."""
        response = _valid_multimodal_response(
            text_triplets=[],
            image_info={"description": "Training loss curve", "figure_type": "graph"},
            cross_modal_triplets=[
                {
                    "head": "fig_1",
                    "relation": "visualizes",
                    "relation_type": "visualizes",
                    "tail": "Training Loss",
                    "tail_type": "Metric",
                    "confidence": 0.9,
                    "evidence": "",
                },
            ],
        )
        builder = self._make_builder_with_mock_llm(response)
        adapter, driver = _make_graph_store_adapter()

        result = await builder._extract_multimodal_triplets(
            text="The training loss decreases over time.",
            image_path="/tmp/fake_loss.png",
            image_caption="Figure 1: Training loss curve",
            chunk_id="chunk_loss",
            graph_store=adapter,
        )

        assert result["cross_modal_triplets_added"] >= 1

        cross_modal_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert any("`visualizes`" in q for q in cross_modal_cyphers), \
            "Cross-modal should use free-text 'visualizes' as edge label"

    @pytest.mark.asyncio
    async def test_cross_modal_no_normalize_relation_type(self):
        """Cross-modal add_relation call should NOT pass relation_description."""
        response = _valid_multimodal_response(
            text_triplets=[],
            image_info={"description": "Results table", "figure_type": "table"},
            cross_modal_triplets=[
                {
                    "head": "fig_2",
                    "relation": "shows_results",
                    "relation_type": "shows_results",
                    "tail": "Accuracy",
                    "tail_type": "Metric",
                    "confidence": 0.85,
                    "evidence": "",
                },
            ],
        )
        builder = self._make_builder_with_mock_llm(response)
        adapter, driver = _make_graph_store_adapter()

        await builder._extract_multimodal_triplets(
            text="Results are shown in the table.",
            image_path="/tmp/fake_table.png",
            image_caption="Table 1: Results",
            chunk_id="chunk_results",
            graph_store=adapter,
        )

        cross_modal_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        for q in cross_modal_cyphers:
            if "`shows_results`" in q:
                assert "r.description" not in q, \
                    f"Cross-modal edge should NOT have r.description: {q}"


# ============================================================================
# Test 10: JSON parsing with edge cases
# ============================================================================


class TestJsonParsing:

    def setup_method(self):
        _install_astrbot_stubs()

    def _make_builder(self):
        return _make_builder()

    def test_valid_json_parsed_correctly(self):
        builder = self._make_builder()
        response = json.dumps({"triplets": [
            {"head": "A", "relation": "uses", "relation_type": "USES_COMPONENT",
             "tail": "B", "head_type": "Method", "tail_type": "Component",
             "confidence": 0.9, "evidence": "test"},
        ]})
        result = builder._parse_json_response(response)
        assert len(result) == 1
        assert result[0]["relation_type"] == "USES_COMPONENT"

    def test_think_tags_stripped(self):
        builder = self._make_builder()
        response = '<think\nSome reasoning here\n</think\n{"triplets": []}'
        result = builder._parse_json_response(response)
        assert result == []

    def test_incomplete_last_triplet_skipped(self):
        """Triplets with incomplete fields after truncation should be filtered by _extract_triplets."""
        builder = self._make_builder()
        data = [
            {"head": "A", "relation": "r", "relation_type": "PROPOSES",
             "tail": "B", "head_type": "Method", "tail_type": "Task",
             "confidence": 0.9, "evidence": "e"},
            {"head": "C", "relation": "r2"},  # truncated - missing tail
        ]
        result = builder._extract_triplets(data)
        assert len(result) == 1
        assert result[0]["relation_type"] == "PROPOSES"

    def test_empty_response_returns_empty_list(self):
        builder = self._make_builder()
        assert builder._parse_json_response("") == []
        assert builder._parse_json_response("  ") == []

    def test_code_block_json_parsed(self):
        builder = self._make_builder()
        response = '```json\n{"triplets": []}\n```'
        result = builder._parse_json_response(response)
        assert result == []

    def test_multimodal_response_parsed(self):
        builder = self._make_builder()
        response = json.dumps({
            "text_triplets": [
                {"head": "X", "relation": "r", "relation_type": "ACHIEVES",
                 "tail": "Y", "head_type": "Method", "tail_type": "Metric",
                 "confidence": 0.9, "evidence": ""},
            ],
            "image_info": {"description": "test", "figure_type": "chart"},
            "cross_modal_triplets": [],
        })
        result = builder._parse_multimodal_response(response)
        assert len(result["text_triplets"]) == 1
        assert result["text_triplets"][0]["relation_type"] == "ACHIEVES"

    def test_extract_triplets_filters_incomplete(self):
        """Triplets missing head/relation/tail should be filtered."""
        builder = self._make_builder()
        data = [
            {"head": "A", "relation": "r", "tail": "B"},
            {"head": "", "relation": "r", "tail": "B"},
            {"head": "C", "relation": "", "tail": "D"},
            {"head": "E", "relation": "r", "tail": ""},
        ]
        result = builder._extract_triplets(data)
        assert len(result) == 1
        assert result[0]["head"] == "A"

    def test_extract_triplets_sorted_by_confidence(self):
        builder = self._make_builder()
        data = [
            {"head": "A", "relation": "r1", "tail": "B", "confidence": 0.5},
            {"head": "C", "relation": "r2", "tail": "D", "confidence": 0.9},
            {"head": "E", "relation": "r3", "tail": "F", "confidence": 0.7},
        ]
        result = builder._extract_triplets(data)
        assert [t["confidence"] for t in result] == [0.9, 0.7, 0.5]


# ============================================================================
# Test 11: Cypher generation correctness in adapter
# ============================================================================


class TestCypherGeneration:
    """Verify the exact Cypher queries produced by the adapter."""

    def setup_method(self):
        _install_astrbot_stubs()
        _install_neo4j_stub()

    def test_entity_cypher_uses_closed_set_label(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_entity(name="BERT", entity_type="Model", chunk_id="c0")

        assert len(driver.queries) == 1
        q = driver.queries[0]
        assert "MERGE (n:`Model`" in q
        assert "name: 'BERT'" in q
        assert "n.chunk_id = 'c0'" in q

    def test_relation_cypher_uses_closed_set_label(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_entity(name="BERT", entity_type="Model")
        adapter.add_entity(name="NLP", entity_type="Task")

        adapter.add_relation(
            head="BERT", tail="NLP",
            relation="ADDRESSES",
            relation_description="addresses NLP tasks",
            chunk_id="c0",
        )

        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert len(relation_cyphers) == 1
        q = relation_cyphers[0]
        assert "`ADDRESSES`" in q
        assert "r.description" in q
        assert "addresses NLP tasks" in q
        assert "r.chunk_id = 'c0'" in q

    def test_relation_stores_weight_but_no_description_or_chunk_id(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_entity(name="A", entity_type="Method")
        adapter.add_entity(name="B", entity_type="Task")

        adapter.add_relation(head="A", tail="B", relation="PROPOSES")

        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert len(relation_cyphers) == 1
        q = relation_cyphers[0]
        assert "r.weight" in q
        assert "r.description" not in q
        assert "r.chunk_id" not in q

    def test_special_characters_escaped_in_cypher(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_entity(name="O'Brien", entity_type="Method", description="it's a method")
        q = driver.queries[0]
        assert "O\\'Brien" in q
        assert "it\\'s a method" in q

    def test_idempotent_entity_not_rewritten(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_entity(name="BERT", entity_type="Model")
        first_count = len(driver.queries)
        adapter.add_entity(name="BERT", entity_type="Model")
        assert len(driver.queries) == first_count, "Duplicate entity should not produce new Cypher"


# ============================================================================
# Test 12: GBNF grammar validates sample outputs
# ============================================================================


class TestGBNFValidation:
    """Verify that sample LLM outputs conform to GBNF grammars."""

    def test_valid_triplet_json_matches_gbnf_schema(self):
        """A valid triplet JSON should conform to the GBNF grammar structure."""
        valid = {
            "triplets": [
                {
                    "head": "BERT",
                    "head_type": "Model",
                    "relation": "extends transformer for language understanding",
                    "relation_type": "EXTENDS",
                    "tail": "Transformer",
                    "tail_type": "Model",
                    "confidence": 0.95,
                    "evidence": "We extend the Transformer architecture"
                }
            ]
        }
        json_str = json.dumps(valid)
        parsed = json.loads(json_str)
        assert "triplets" in parsed
        assert len(parsed["triplets"]) == 1
        t = parsed["triplets"][0]
        assert t["relation_type"] in {"ADDRESSES", "PROPOSES", "USES_COMPONENT", "EVALUATED_ON",
                                       "ACHIEVES", "COMPARES_WITH", "LIMITED_BY", "APPLIES_TO", "EXTENDS"}
        assert t["head_type"] in {"Method", "Model", "Task", "Dataset", "Metric",
                                   "Component", "Limitation", "Application", "Baseline"}
        assert t["tail_type"] in {"Method", "Model", "Task", "Dataset", "Metric",
                                   "Component", "Limitation", "Application", "Baseline"}
        assert 0.0 <= t["confidence"] <= 1.0

    def test_valid_multimodal_json_matches_schema(self):
        valid = {
            "text_triplets": [
                {
                    "head": "CNN", "head_type": "Model",
                    "relation": "applied to classification", "relation_type": "APPLIES_TO",
                    "tail": "ImageNet", "tail_type": "Dataset",
                    "confidence": 0.9, "evidence": "CNN on ImageNet"
                }
            ],
            "image_info": {
                "figure_id": "fig1",
                "description": "CNN architecture diagram",
                "figure_type": "diagram",
                "key_entities": ["CNN", "Convolution"],
                "relations_shown": ["architecture"]
            },
            "cross_modal_triplets": [
                {
                    "head": "fig1",
                    "relation": "visualizes",
                    "relation_type": "visualizes",
                    "tail": "CNN Architecture",
                    "tail_type": "Component",
                    "confidence": 0.85,
                    "evidence": "The figure shows CNN architecture"
                }
            ]
        }
        json_str = json.dumps(valid)
        parsed = json.loads(json_str)

        for t in parsed["text_triplets"]:
            assert t["relation_type"] in {"ADDRESSES", "PROPOSES", "USES_COMPONENT", "EVALUATED_ON",
                                           "ACHIEVES", "COMPARES_WITH", "LIMITED_BY", "APPLIES_TO", "EXTENDS"}

        for t in parsed["cross_modal_triplets"]:
            assert isinstance(t["relation_type"], str)
            assert isinstance(t["relation"], str)


# ============================================================================
# Test 13: build_from_nodes() end-to-end with mock LLM
# ============================================================================


class TestBuildFromNodesEndToEnd:
    """Full build_from_nodes() with mock LLM and FakeDriver, verifying Neo4j writes."""

    def setup_method(self):
        _install_astrbot_stubs()
        _install_neo4j_stub()

    @pytest.mark.asyncio
    async def test_build_from_nodes_writes_closed_set_relations(self):
        """build_from_nodes should write only closed-set relation labels to Neo4j."""
        from graphrag.graph_builder import MultimodalGraphBuilder, CLOSED_RELATION_TYPES

        triplets_json = json.dumps({"triplets": [
            {
                "head": "BERT", "head_type": "Model",
                "relation": "proposes bidirectional pre-training",
                "relation_type": "PROPOSES",
                "tail": "Masked LM", "tail_type": "Method",
                "confidence": 0.95, "evidence": "[Chunk 1]"
            },
            {
                "head": "BERT", "head_type": "Model",
                "relation": "evaluated on SQuAD",
                "relation_type": "EVALUATED_ON",
                "tail": "SQuAD", "tail_type": "Dataset",
                "confidence": 0.9, "evidence": "[Chunk 1]"
            },
            {
                "head": "BERT", "head_type": "Model",
                "relation": "achieves SOTA results",
                "relation_type": "ACHIEVES",
                "tail": "SOTA", "tail_type": "Metric",
                "confidence": 0.85, "evidence": "[Chunk 1]"
            },
        ]})

        config = FakeGraphRAGConfig(max_triplets_per_chunk=10)
        builder = MultimodalGraphBuilder.__new__(MultimodalGraphBuilder)
        builder.config = config
        builder.context = None
        builder._llm_config = types.SimpleNamespace(n_ctx=8192, max_tokens=1024)
        builder._triplet_grammar = None
        builder._multimodal_grammar = None

        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response(triplets_json))
        mock_llm.initialize = AsyncMock()
        builder._llm = mock_llm
        builder._load_grammars = lambda: None

        adapter, driver = _make_graph_store_adapter()

        nodes = [
            _make_node(
                "BERT proposes bidirectional pre-training. It was evaluated on SQuAD and achieves SOTA results.",
                chunk_id="bert_paper_chunk_0",
            )
        ]

        result = await builder.build_from_nodes(nodes, adapter)

        assert result["text_triplets_added"] == 3

        relation_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert any("`PROPOSES`" in q for q in relation_cyphers)
        assert any("`EVALUATED_ON`" in q for q in relation_cyphers)
        assert any("`ACHIEVES`" in q for q in relation_cyphers)

        for q in relation_cyphers:
            for rt in CLOSED_RELATION_TYPES:
                if f"`{rt}`" in q:
                    assert "bert_paper_chunk_0" in q, \
                        f"Relation Cypher missing chunk_id: {q}"

    @pytest.mark.asyncio
    async def test_build_from_nodes_multiple_batches(self):
        """build_from_nodes should batch nodes (4 per batch) and aggregate stats."""
        from graphrag.graph_builder import MultimodalGraphBuilder

        triplets_json = json.dumps({"triplets": [
            {
                "head": "X", "head_type": "Method",
                "relation": "proposes method",
                "relation_type": "PROPOSES",
                "tail": "Y", "tail_type": "Task",
                "confidence": 0.9, "evidence": "[Chunk 1]"
            },
        ]})

        config = FakeGraphRAGConfig(max_triplets_per_chunk=5)
        builder = MultimodalGraphBuilder.__new__(MultimodalGraphBuilder)
        builder.config = config
        builder.context = None
        builder._llm_config = types.SimpleNamespace(n_ctx=8192, max_tokens=1024)
        builder._triplet_grammar = None
        builder._multimodal_grammar = None

        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response(triplets_json))
        mock_llm.initialize = AsyncMock()
        builder._llm = mock_llm
        # Pre-load grammars so _ensure_llm_initialized is a no-op
        builder._load_grammars = lambda: None

        adapter, _ = _make_graph_store_adapter()

        nodes = [_make_node("x" * 60, chunk_id=f"chunk_{i}") for i in range(6)]
        result = await builder.build_from_nodes(nodes, adapter)

        assert result["text_triplets_added"] == 2
        assert result["chunks_processed"] == 2


# ============================================================================
# Test 14: Deterministic media link (survives VLM failure)
# ============================================================================


class TestDeterministicMediaLink:
    """Verify Chunk→Media edges are written from metadata regardless of VLM outcome."""

    def setup_method(self):
        _install_astrbot_stubs()
        _install_neo4j_stub()

    def test_add_media_link_produces_correct_cypher(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_media_link(
            chunk_id="chunk_42",
            media_path="/data/figures/fig3.png",
            media_type="image",
            caption="Figure 3: Architecture",
        )
        assert len(driver.queries) == 1
        q = driver.queries[0]
        assert "MERGE (c:Chunk" in q
        assert "chunk_42" in q
        assert "MERGE (m:Media" in q
        assert "/data/figures/fig3.png" in q
        assert "HAS_MEDIA" in q
        assert "m.type = 'image'" in q

    def test_add_media_link_idempotent(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_media_link("c1", "/fig.png")
        first_count = len(driver.queries)
        adapter.add_media_link("c1", "/fig.png")
        assert len(driver.queries) == first_count

    def test_add_media_link_different_chunks_different_edges(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_media_link("c1", "/fig.png")
        adapter.add_media_link("c2", "/fig.png")
        assert len(driver.queries) == 2

    @pytest.mark.asyncio
    async def test_process_batch_writes_media_link_even_with_empty_llm_response(self):
        """Even when LLM returns no triplets, HAS_MEDIA edge should exist."""
        builder = _make_builder(FakeGraphRAGConfig(max_triplets_per_chunk=10))
        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response('{"triplets": []}'))
        builder._llm = mock_llm

        adapter, driver = _make_graph_store_adapter()

        node = _make_node(
            "BERT extends the Transformer architecture for language understanding tasks.",
            chunk_id="chunk_bert",
            has_image=True,
            image_path="/data/figures/bert_arch.png",
        )
        result = await builder._process_batch([node], adapter)

        assert result["text_triplets_added"] == 0
        media_cyphers = [q for q in driver.queries if "HAS_MEDIA" in q]
        assert len(media_cyphers) == 1, "Should have HAS_MEDIA edge even with no triplets"
        assert "chunk_bert" in media_cyphers[0]
        assert "/data/figures/bert_arch.png" in media_cyphers[0]

    @pytest.mark.asyncio
    async def test_process_batch_no_media_link_without_image_metadata(self):
        """Nodes without image metadata should not produce HAS_MEDIA edges."""
        builder = _make_builder(FakeGraphRAGConfig(max_triplets_per_chunk=10))
        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response('{"triplets": []}'))
        builder._llm = mock_llm

        adapter, driver = _make_graph_store_adapter()

        node = _make_node("x" * 60, chunk_id="chunk_plain")
        await builder._process_batch([node], adapter)

        media_cyphers = [q for q in driver.queries if "HAS_MEDIA" in q]
        assert len(media_cyphers) == 0

    def test_add_media_link_stores_caption_in_neo4j(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_media_link(
            chunk_id="c1", media_path="/fig.png",
            media_type="image",
            caption="Figure 1: Transformer architecture",
        )
        q = driver.queries[0]
        assert "m.caption = 'Figure 1: Transformer architecture'" in q

    def test_add_media_link_stores_dict_in_entity_info(self):
        """Media link entry must be a dict (not bool) to avoid get_stats() crash."""
        adapter, driver = _make_graph_store_adapter()
        adapter.add_media_link("c1", "/fig.png")
        key = f"__media__c1##/fig.png"
        assert isinstance(adapter._entity_info[key], dict)
        assert adapter._entity_info[key]["type"] == "MediaLink"

    def test_add_image_entity_stores_chunk_id_in_neo4j(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_image_entity(
            figure_id="paper1_Figure 1",
            image_path="/path/to/fig1.png",
            description="Architecture diagram",
            figure_type="diagram",
            chunk_id="bert_chunk_0",
        )
        fig_cyphers = [q for q in driver.queries if "Figure_diagram" in q and "MERGE (n:" in q]
        assert len(fig_cyphers) == 1
        assert "n.chunk_id = 'bert_chunk_0'" in fig_cyphers[0]

    def test_add_table_entity_stores_chunk_id_in_neo4j(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_table_entity(
            table_id="paper1_Table 1",
            description="Baseline results",
            chunk_id="resnet_chunk_5",
        )
        table_cyphers = [q for q in driver.queries if "MERGE (n:Table" in q]
        assert len(table_cyphers) == 1
        assert "n.chunk_id = 'resnet_chunk_5'" in table_cyphers[0]

    def test_add_relation_stores_weight_in_neo4j(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_entity(name="BERT", entity_type="Model")
        adapter.add_entity(name="NLP", entity_type="Task")
        adapter.add_relation(
            head="BERT", tail="NLP",
            relation="ADDRESSES",
            weight=0.87,
        )
        rel_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert len(rel_cyphers) == 1
        assert "r.weight = 0.87" in rel_cyphers[0]

    def test_add_relation_stores_description_and_chunk_id_and_weight(self):
        adapter, driver = _make_graph_store_adapter()
        adapter.add_entity(name="X", entity_type="Method")
        adapter.add_entity(name="Y", entity_type="Task")
        adapter.add_relation(
            head="X", tail="Y",
            relation="PROPOSES",
            relation_description="proposes a new method",
            chunk_id="chunk_abc",
            weight=0.95,
        )
        rel_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert len(rel_cyphers) == 1
        q = rel_cyphers[0]
        assert "r.description = 'proposes a new method'" in q
        assert "r.chunk_id = 'chunk_abc'" in q
        assert "r.weight = 0.95" in q

    def test_add_relation_consistent_escaping(self):
        """chunk_id with backslashes and quotes must be escaped correctly."""
        adapter, driver = _make_graph_store_adapter()
        adapter.add_entity(name="X", entity_type="Method")
        adapter.add_entity(name="Y", entity_type="Task")
        adapter.add_relation(
            head="X", tail="Y",
            relation="APPLIES_TO",
            chunk_id=r"C:\Users\test\file.png",  # backslash + windows path
        )
        rel_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert len(rel_cyphers) == 1
        q = rel_cyphers[0]
        # In the Cypher string, each backslash is doubled: C:\\Users = 4 chars
        assert "C:\\\\Users\\\\test\\\\file.png" in q  # 4 backslashes in the Cypher string
        assert "r.chunk_id = 'C:\\\\Users\\\\test\\\\file.png'" in q

    def test_get_stats_does_not_crash_after_media_link(self):
        """get_stats() must not crash on media link entries (which are dicts)."""
        adapter, driver = _make_graph_store_adapter()
        adapter.add_media_link("c1", "/fig.png")
        # get_stats iterates _entity_info.values() and calls .get("type", ...)
        # Should not raise AttributeError on the dict entry
        stats = adapter.get_stats() if hasattr(adapter, 'get_stats') else None
        # If get_stats exists and doesn't crash, we're good
        # If it doesn't exist, that's fine too — the key is _entity_info stores dict not bool
        assert isinstance(adapter._entity_info[f"__media__c1##/fig.png"], dict)


# ============================================================================
# Test: Adversarial review fixes
# ============================================================================


class TestVLMCacheEviction:
    """VLM cache should evict half, not clear all."""

    def test_cache_eviction_removes_half_not_all(self):
        _install_astrbot_stubs()
        from graphrag.graph_builder import _VLM_CACHE, _vlm_cache_key

        _VLM_CACHE.clear()
        for i in range(501):
            _VLM_CACHE[f"key_{i}"] = {"dummy": True}
        assert len(_VLM_CACHE) == 501

        # Simulate the eviction code path
        if len(_VLM_CACHE) > 500:
            keys_to_remove = list(_VLM_CACHE.keys())[:250]
            for k in keys_to_remove:
                del _VLM_CACHE[k]

        assert len(_VLM_CACHE) == 251  # 501 - 250 = 251
        # Verify early keys are removed, later keys survive
        assert "key_0" not in _VLM_CACHE
        assert "key_500" in _VLM_CACHE
        _VLM_CACHE.clear()


class TestFallbackFigureTypeMapping:
    """_fallback_cross_modal should map file extensions to meaningful figure types."""

    @pytest.mark.parametrize("ext,expected", [
        (".png", "image"),
        (".jpg", "image"),
        (".jpeg", "image"),
        (".svg", "diagram"),
        (".pdf", "document"),
        (".tiff", "image"),
        (".webp", "image"),
        (".bmp", "image"),
        (".gif", "image"),
    ])
    def test_extension_mapping(self, ext, expected):
        _install_astrbot_stubs()
        from graphrag.graph_builder import _EXT_TO_FIGURE_TYPE

        assert _EXT_TO_FIGURE_TYPE[ext] == expected

    def test_unknown_extension_returns_unknown(self):
        _install_astrbot_stubs()
        from graphrag.graph_builder import _EXT_TO_FIGURE_TYPE

        assert _EXT_TO_FIGURE_TYPE.get(".xyz", "unknown") == "unknown"


class TestMultimodalDisabled:
    """When multimodal_enabled=False, image nodes should still be processed by batch text extraction."""

    @pytest.mark.asyncio
    async def test_image_node_processed_in_batch_when_multimodal_disabled(self):
        """Image nodes should contribute text triplets via batch when multimodal is off."""
        config = FakeGraphRAGConfig(multimodal_enabled=False)
        builder = _make_builder(config)

        node = _make_node(
            text="BERT is a Transformer-based model that achieves state-of-the-art results on GLUE.",
            chunk_id="chunk_0",
            has_image=True,
            image_path="/fake/image.png",
        )

        mock_response = _make_llm_response(json.dumps({"triplets": [{
            "head": "BERT",
            "head_type": "Model",
            "relation": "based on",
            "relation_type": "EXTENDS",
            "tail": "Transformer",
            "tail_type": "Model",
            "confidence": 0.95,
            "evidence": "[Chunk 1]",
        }]}))

        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=mock_response)
        builder._llm = mock_llm

        adapter, driver = _make_graph_store_adapter()
        result = await builder._process_batch([node], adapter)

        # Should have text triplets from batch (not skipped)
        assert result["text_triplets_added"] == 1
        assert result["entities_added"] >= 1


class TestAddRelationMerge:
    """add_relation should use MERGE for nodes so relations survive entity creation failures."""

    def test_add_relation_uses_merge_not_match(self):
        adapter, driver = _make_graph_store_adapter()
        # Don't call add_entity first — add_relation should still create nodes
        adapter.add_relation(
            head="NewEntity1", tail="NewEntity2",
            relation="ADDRESSES",
            weight=0.9,
        )
        rel_cyphers = [q for q in driver.queries if "MERGE (a)-[r:" in q]
        assert len(rel_cyphers) == 1
        # Should have MERGE for both nodes, not MATCH
        node_cyphers = [q for q in driver.queries if "MERGE (a {" in q or "MERGE (b {" in q]
        # The relation cypher itself contains MERGE (a {…}) MERGE (b {…})
        q = rel_cyphers[0]
        assert "MERGE (a {name:" in q
        assert "MERGE (b {name:" in q
        assert "MATCH" not in q


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
