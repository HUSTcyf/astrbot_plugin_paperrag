"""
Unit tests for _make_cypher_validator and _TEXT_TO_CYPHER_TEMPLATE.

Tests the two-stage Cypher validation:
  Stage 1 – first-token keyword check (no network)
  Stage 2 – Neo4j EXPLAIN pre-parse via mocked graph_store.structured_query()

Usage:
    cd astrbot_plugin_paperrag && python -m pytest test/test_cypher_validator.py -v
"""

from unittest.mock import MagicMock

import pytest

from graphrag.graph_rag_engine import (
    _TEXT_TO_CYPHER_TEMPLATE,
    _VALID_CYPHER_STARTS,
    _make_cypher_validator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_store(explain_outcome: str = "pass"):
    """Build a mock Neo4jPropertyGraphStore.

    ``explain_outcome``:
      "pass"       → structured_query returns successfully
      "syntax_err" → structured_query raises CypherSyntaxError
      "other_err"  → structured_query raises a generic RuntimeError
    """
    store = MagicMock(name="Neo4jPropertyGraphStore")
    if explain_outcome == "pass":
        store.structured_query.return_value = [{"plan": {}}]
    elif explain_outcome == "syntax_err":
        from neo4j.exceptions import CypherSyntaxError
        store.structured_query.side_effect = CypherSyntaxError(
            "Invalid input 'point map': expected an expression"
        )
    elif explain_outcome == "other_err":
        store.structured_query.side_effect = RuntimeError("connection refused")
    return store


@pytest.fixture
def passing_store():
    return _make_store("pass")


@pytest.fixture
def syntax_err_store():
    return _make_store("syntax_err")


@pytest.fixture
def broken_store():
    return _make_store("other_err")


# ---------------------------------------------------------------------------
# Stage 1 – first-token keyword check
# ---------------------------------------------------------------------------

_VALID_QUERIES = [
    "MATCH (h)-[r:USES]->(t) WHERE t.name CONTAINS '3D' RETURN h, r, t LIMIT 30",
    "CALL db.schema.visualization()",
    "MERGE (n:Model {name: 'BERT'})",
    "RETURN 1",
    "EXPLAIN MATCH (n) RETURN n",
]

_INVALID_FIRST_TOKENS = [
    ("WHERE l.name = 'noisy pose estimation'", "WHERE"),
    ("  \t\n  WHERE x = 1", "WHERE"),  # whitespace then WHERE
    ("SELECT * FROM nodes", "SELECT"),
    ("DELETE FROM nodes", "DELETE"),   # DELETE is not in _VALID_CYPHER_STARTS
]


@pytest.mark.parametrize("query", _VALID_QUERIES)
def test_stage1_passes_valid_first_token(query, passing_store):
    validator = _make_cypher_validator(passing_store)
    result = validator(query)
    assert result == query


@pytest.mark.parametrize("query,first_token", _INVALID_FIRST_TOKENS)
def test_stage1_rejects_invalid_first_token(query, first_token, passing_store):
    validator = _make_cypher_validator(passing_store)
    with pytest.raises(ValueError, match=f"keyword={first_token}"):
        validator(query)


def test_stage1_rejects_empty_query(passing_store):
    validator = _make_cypher_validator(passing_store)
    with pytest.raises(ValueError, match="空 Cypher"):
        validator("")
    with pytest.raises(ValueError, match="空 Cypher"):
        validator("   \t\n  ")


# ---------------------------------------------------------------------------
# Stage 2 – Neo4j EXPLAIN pre-parse
# ---------------------------------------------------------------------------

def test_stage2_explain_passes_valid_cypher(passing_store):
    validator = _make_cypher_validator(passing_store)
    cypher = "MATCH (h)-[r:EVALUATED_ON]->(t) WHERE t.name CONTAINS 'ImageNet' RETURN h, r, t LIMIT 30"
    result = validator(cypher)
    assert result == cypher
    # Verify EXPLAIN was called with the query
    passing_store.structured_query.assert_called_once()
    call_arg = passing_store.structured_query.call_args[0][0]
    assert call_arg.startswith("EXPLAIN ")


def test_stage2_explain_catches_contains_in_node_pattern(syntax_err_store):
    """The error that actually occurred: CONTAINS inside node property pattern."""
    validator = _make_cypher_validator(syntax_err_store)
    bad_cypher = (
        "MATCH (h:Model {name: 'MASt3R'})-[r:USES_COMPONENT]->"
        "(c:Component {name: 'Transformer'})-[r2:USES_COMPONENT|ADDRESSES]->"
        "(t:Task {name: CONTAINS 'point map'}) "
        "RETURN h, r2, t LIMIT 30"
    )
    with pytest.raises(ValueError, match="Neo4j EXPLAIN 校验"):
        validator(bad_cypher)


def test_stage2_explain_catches_standalone_where(syntax_err_store):
    validator = _make_cypher_validator(syntax_err_store)
    # This would also be caught by stage 1, but verify stage 2 handles it
    bad_cypher = "WHERE l.name = 'noisy pose estimation' RETURN l"
    # Stage 1 catches it first
    with pytest.raises(ValueError, match="keyword=WHERE"):
        validator(bad_cypher)


def test_stage2_explain_catches_other_errors(broken_store):
    validator = _make_cypher_validator(broken_store)
    with pytest.raises(ValueError, match="Neo4j EXPLAIN 校验"):
        validator("MATCH (n) RETURN n")


# ---------------------------------------------------------------------------
# Multi-hop query validation (the exact scenario from the user's error)
# ---------------------------------------------------------------------------

def test_multi_hop_correct_syntax_passes(passing_store):
    """The CORRECTED version of the failing query should pass both stages."""
    validator = _make_cypher_validator(passing_store)
    # Fixed: CONTAINS in WHERE, relationships bound, correct tail reference
    cypher = (
        "MATCH (h:Model {name: 'MASt3R'})-[r1:USES_COMPONENT]->"
        "(c:Component {name: 'Transformer'})-[r2:USES_COMPONENT|ADDRESSES]->"
        "(t:Task) "
        "WHERE t.name CONTAINS 'point map' OR t.name CONTAINS '3D visualization' "
        "RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, "
        "type(r2) AS relation, coalesce(t.name,'') AS tail, "
        "labels(t)[0] AS tail_type LIMIT 30"
    )
    result = validator(cypher)
    assert result == cypher


# ---------------------------------------------------------------------------
# Validator factory: closure captures graph_store correctly
# ---------------------------------------------------------------------------

def test_factory_returns_distinct_validators():
    store_a = _make_store("pass")
    store_b = _make_store("pass")
    val_a = _make_cypher_validator(store_a)
    val_b = _make_cypher_validator(store_b)
    assert val_a is not val_b
    val_a("MATCH (n) RETURN n")
    val_b("MATCH (n) RETURN n")
    store_a.structured_query.assert_called_once()
    store_b.structured_query.assert_called_once()


# ---------------------------------------------------------------------------
# Template correctness
# ---------------------------------------------------------------------------

def test_template_has_required_placeholders():
    assert "{schema}" in _TEXT_TO_CYPHER_TEMPLATE
    assert "{question}" in _TEXT_TO_CYPHER_TEMPLATE


def test_template_forbids_contains_in_node_pattern():
    assert "CONTAINS is a WHERE-only operator" in _TEXT_TO_CYPHER_TEMPLATE
    assert "NEVER use it inside node property patterns" in _TEXT_TO_CYPHER_TEMPLATE


def test_template_has_wrong_example():
    assert "WRONG — NEVER do this" in _TEXT_TO_CYPHER_TEMPLATE
    assert "CONTAINS in node pattern" in _TEXT_TO_CYPHER_TEMPLATE


def test_template_has_multi_hop_example():
    assert "multi-hop with WHERE on tail" in _TEXT_TO_CYPHER_TEMPLATE
    assert "r2:USES_COMPONENT|ADDRESSES" in _TEXT_TO_CYPHER_TEMPLATE


def test_template_binds_all_relationships():
    assert "both `r1` and `r2` must be bound" in _TEXT_TO_CYPHER_TEMPLATE
    assert "NEVER use anonymous" in _TEXT_TO_CYPHER_TEMPLATE


# ---------------------------------------------------------------------------
# _VALID_CYPHER_STARTS completeness
# ---------------------------------------------------------------------------

def test_valid_cypher_starts_covers_common_clauses():
    essential = {"MATCH", "CALL", "CREATE", "MERGE", "RETURN"}
    missing = essential - _VALID_CYPHER_STARTS
    assert not missing, f"Missing essential Cypher start keywords: {missing}"


def test_valid_cypher_starts_includes_with():
    # WITH is a valid Cypher start (e.g., WITH 1 AS x RETURN x)
    assert "WITH" in _VALID_CYPHER_STARTS
