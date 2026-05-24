"""Neo4j integration tests: verify :ALIAS_OF creation and retrieval-time expansion.

Requires Neo4j running at localhost:7687.
Uses an isolated test session — creates test nodes, runs the merge,
verifies relationships, then cleans up.
"""
import pytest
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j_M73770"


@pytest.fixture
def driver():
    d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    d.verify_connectivity()
    yield d
    d.close()


@pytest.fixture(autouse=True)
def cleanup_test_nodes(driver):
    """Remove test nodes before and after each test."""
    with driver.session(database="neo4j") as s:
        s.run("MATCH (n {_test_dedup: true}) DETACH DELETE n")
    yield
    with driver.session(database="neo4j") as s:
        s.run("MATCH (n {_test_dedup: true}) DETACH DELETE n")


class TestPostBuildMergeAliases:
    """Layer 3: _post_build_merge_aliases integration with real Neo4j."""

    def test_creates_alias_of_relationship(self, driver):
        """Insert two entities where one is an acronym of the other,
        then run _post_build_merge_aliases — should create :ALIAS_OF."""
        from graphrag.graph_builder import MultimodalGraphBuilder
        from graphrag.graph_rag_engine import GraphRAGConfig

        # Insert test entities into Neo4j
        with driver.session(database="neo4j") as s:
            s.run(
                "MERGE (n:Method {name: '3D Gaussian Splatting'}) "
                "SET n._test_dedup = true, n.description = 'Test full name'"
            )
            s.run(
                "MERGE (n:Method {name: '3DGS'}) "
                "SET n._test_dedup = true, n.description = 'Test acronym'"
            )

        # Run post-build merge
        builder = MultimodalGraphBuilder(config=GraphRAGConfig())

        # Create a mock adapter that wraps the real driver
        class MockAdapter:
            def __init__(self, d):
                self._driver = d

        adapter = MockAdapter(driver)
        builder._post_build_merge_aliases(adapter)

        # Verify the :ALIAS_OF relationship exists between test nodes
        with driver.session(database="neo4j") as s:
            result = s.run(
                "MATCH (alias:Method {name: '3DGS'})-[r:ALIAS_OF]->(canon:Method {name: '3D Gaussian Splatting'}) "
                "RETURN count(r) AS c"
            )
            assert result.single()["c"] == 1

    def test_no_false_alias_for_unrelated_entities(self, driver):
        """Two unrelated short names should NOT get :ALIAS_OF."""
        from graphrag.graph_builder import MultimodalGraphBuilder
        from graphrag.graph_rag_engine import GraphRAGConfig

        with driver.session(database="neo4j") as s:
            s.run(
                "MERGE (n:Method {name: 'Attention'}) "
                "SET n._test_dedup = true"
            )
            s.run(
                "MERGE (n:Method {name: 'Optimization'}) "
                "SET n._test_dedup = true"
            )

        builder = MultimodalGraphBuilder(config=GraphRAGConfig())

        class MockAdapter:
            def __init__(self, d):
                self._driver = d

        adapter = MockAdapter(driver)
        builder._post_build_merge_aliases(adapter)

        # "Attention" and "Optimization" are both >8 chars, so _is_short_name rejects both.
        # Verify no :ALIAS_OF between these specific test nodes.
        with driver.session(database="neo4j") as s:
            result = s.run(
                "MATCH (a:Method {name: 'Attention'})-[r:ALIAS_OF]-(b:Method {name: 'Optimization'}) "
                "RETURN count(r) AS c"
            )
            assert result.single()["c"] == 0

    def test_multiple_aliases_in_same_type(self, driver):
        """Multiple acronym-full_name pairs within same entity type."""
        from graphrag.graph_builder import MultimodalGraphBuilder
        from graphrag.graph_rag_engine import GraphRAGConfig

        with driver.session(database="neo4j") as s:
            # Pair 1: 3DGS → 3D Gaussian Splatting
            s.run("MERGE (n:Method {name: '3D Gaussian Splatting'}) SET n._test_dedup = true")
            s.run("MERGE (n:Method {name: '3DGS'}) SET n._test_dedup = true")
            # Pair 2: NRF → Neural Radiance Fields
            s.run("MERGE (n:Method {name: 'Neural Radiance Fields'}) SET n._test_dedup = true")
            s.run("MERGE (n:Method {name: 'NRF'}) SET n._test_dedup = true")
            # Unrelated
            s.run("MERGE (n:Method {name: 'CNN'}) SET n._test_dedup = true")

        builder = MultimodalGraphBuilder(config=GraphRAGConfig())

        class MockAdapter:
            def __init__(self, d):
                self._driver = d

        builder._post_build_merge_aliases(MockAdapter(driver))

        # Verify both :ALIAS_OF relationships exist between test nodes
        with driver.session(database="neo4j") as s:
            r1 = s.run(
                "MATCH (:Method {name: '3DGS'})-[:ALIAS_OF]->(:Method {name: '3D Gaussian Splatting'}) "
                "RETURN count(*) AS c"
            ).single()["c"]
            r2 = s.run(
                "MATCH (:Method {name: 'NRF'})-[:ALIAS_OF]->(:Method {name: 'Neural Radiance Fields'}) "
                "RETURN count(*) AS c"
            ).single()["c"]
            assert r1 == 1, f"3DGS→3D Gaussian Splatting not found"
            assert r2 == 1, f"NRF→Neural Radiance Fields not found"


class TestKeywordAliasExpansion:
    """Layer 4: _expand_keyword_via_aliases — query-time alias expansion."""

    def test_expand_known_alias(self, driver):
        """Given :ALIAS_OF relationship, keyword '3DGS' should expand to include '3D Gaussian Splatting'."""

        # Insert alias relationship
        with driver.session(database="neo4j") as s:
            s.run(
                "MERGE (a:Method {name: '3DGS'}) SET a._test_dedup = true "
                "MERGE (c:Method {name: '3D Gaussian Splatting'}) SET c._test_dedup = true "
                "MERGE (a)-[:ALIAS_OF]->(c)"
            )

        # Create a minimal engine with adapter pointing to real driver
        class MockAdapter:
            def __init__(self, d):
                self._driver = d

        class MockEngine:
            def __init__(self, d):
                self._adapter = MockAdapter(d)

        engine = MockEngine(driver)

        # Import the method directly — it's an instance method on the engine
        import graphrag.graph_rag_engine as engine_module
        # Bind the method to our mock engine
        expand = engine_module.GraphRAGEngine._expand_keyword_via_aliases.__get__(engine)

        result = expand("3DGS")
        assert "3DGS" in result
        assert "3D Gaussian Splatting" in result
        assert len(result) >= 2

    def test_no_aliases_for_unknown_keyword(self, driver):
        """Keyword with no :ALIAS_OF links should return just [keyword]."""
        class MockAdapter:
            def __init__(self, d):
                self._driver = d

        class MockEngine:
            def __init__(self, d):
                self._adapter = MockAdapter(d)

        engine = MockEngine(driver)

        import graphrag.graph_rag_engine as engine_module
        expand = engine_module.GraphRAGEngine._expand_keyword_via_aliases.__get__(engine)

        result = expand("SomeUnknownKeyword")
        assert result == ["SomeUnknownKeyword"]

    def test_alias_expansion_with_no_neo4j(self):
        """When driver is None, should return [keyword] without error."""
        class MockEngine:
            def __init__(self):
                self._adapter = None

        engine = MockEngine()

        import graphrag.graph_rag_engine as engine_module
        expand = engine_module.GraphRAGEngine._expand_keyword_via_aliases.__get__(engine)

        result = expand("3DGS")
        assert result == ["3DGS"]


class TestEndToEndBuildThenQuery:
    """Full cycle: normalize entities, write to Neo4j, merge aliases, then query."""

    def test_build_and_query_cycle(self, driver):
        """Simulate the full pipeline on a small scale."""
        from graphrag.graph_builder import MultimodalGraphBuilder
        from graphrag.graph_rag_engine import GraphRAGConfig

        # Step 1: Normalize entity names (build-time Layers 1-2)
        builder = MultimodalGraphBuilder(config=GraphRAGConfig())

        # "3D Gaussian Splatting (3DGS)" → strips parenthetical, registers mapping
        n1 = builder._normalize_entity_name("3D Gaussian Splatting (3DGS)", "Method", "paper_a.pdf")
        assert n1 == "3D Gaussian Splatting"

        # "3DGS" (same paper, no parentheses) → normalizes via registry
        n2 = builder._normalize_entity_name("3DGS", "Method", "paper_a.pdf")
        assert n2 == "3D Gaussian Splatting"

        # Step 2: Write to Neo4j (simulate what add_entity would do)
        with driver.session(database="neo4j") as s:
            s.run(
                "MERGE (n:Method {name: '3D Gaussian Splatting'}) "
                "SET n._test_dedup = true"
            )
            # Intentionally also insert the acronym to test post-build merge
            s.run(
                "MERGE (n:Method {name: '3DGS'}) "
                "SET n._test_dedup = true"
            )

        # Step 3: Post-build merge
        class MockAdapter:
            def __init__(self, d):
                self._driver = d

        builder._post_build_merge_aliases(MockAdapter(driver))

        # Step 4: Verify :ALIAS_OF exists
        with driver.session(database="neo4j") as s:
            r = s.run(
                "MATCH (a:Method {name: '3DGS'})-[rel:ALIAS_OF]->(c:Method {name: '3D Gaussian Splatting'}) "
                "RETURN type(rel) AS t, a.name AS alias, c.name AS canonical"
            ).single()
            assert r["t"] == "ALIAS_OF"
            assert r["alias"] == "3DGS"
            assert r["canonical"] == "3D Gaussian Splatting"

        # Step 5: Query-time expansion should find canonical
        class MockEngine:
            def __init__(self, d):
                self._adapter = MockAdapter(d)

        engine = MockEngine(driver)

        import graphrag.graph_rag_engine as engine_module
        expand = engine_module.GraphRAGEngine._expand_keyword_via_aliases.__get__(engine)

        expanded = expand("3DGS")
        assert "3DGS" in expanded
        assert "3D Gaussian Splatting" in expanded
