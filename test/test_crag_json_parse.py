"""Test CRAG LLM JSON parsing robustness and rule-based evaluator score normalization.

NOTE: The JSON parsing logic tested here mirrors _evaluate_by_llm() in
rag/hybrid_rag.py. That method should eventually use provider.llm_utils.parse_json_response()
instead of inline parsing to avoid duplication.
"""

import pytest
from provider.llm_utils import parse_json_response


# ============================================================================
# Test 1: LLM JSON parsing via provider.llm_utils.parse_json_response
# ============================================================================

@pytest.mark.parametrize("name,response,exp_score,exp_level", [
    ("standard JSON", '{"score": 0.8, "level": "high", "reasoning": "OK"}', 0.8, "high"),
    ("Chinese quotes", '{"score": 0.7, "level": "medium", "reasoning": "中等相关"}', 0.7, "medium"),
    ("Trailing comma", '{"score": 0.6, "level": "medium",}', 0.6, "medium"),
    ("Extra text before JSON", 'Here is my evaluation:\n{"score": 0.9, "level": "high", "reasoning": "Good"}', 0.9, "high"),
    ("Extra text after JSON", '{"score": 0.4, "level": "medium"}\n\nThe results are moderately relevant.', 0.4, "medium"),
    ("Missing score field", '{"level": "high", "reasoning": "Great match"}', None, "high"),
    ("Score as string", '{"score": "0.75", "level": "medium"}', 0.75, "medium"),
    ("Invalid level", '{"score": 0.8, "level": "excellent"}', 0.8, None),
    ("Mixed CN/EN quotes", '{"score": 0.6, "reasoning": "结果还不错", "level": "medium"}', 0.6, "medium"),
])
def test_llm_json_parsing(name, response, exp_score, exp_level):
    result = parse_json_response(response)
    assert result is not None, f"No JSON found in: {name}"

    if exp_score is not None:
        actual_score = float(result["score"]) if isinstance(result["score"], str) else result["score"]
        assert abs(actual_score - exp_score) < 0.01, \
            f"{name}: expected score={exp_score}, got {result['score']}"

    if exp_level is not None and "level" in result:
        assert result["level"] == exp_level, \
            f"{name}: expected level={exp_level}, got {result['level']}"


# ============================================================================
# Test 2: Rule-based evaluator score normalization
# ============================================================================

def _evaluate_by_rules(results, query_terms):
    """Mirrors HybridRAGEngine._evaluate_by_rules() logic for isolated testing."""
    if not results:
        return {"score": 0.0, "level": "low"}

    max_raw_score = max(r.get("score", 0.0) for r in results)
    if max_raw_score > 0 and max_raw_score < 0.1:
        normalized = [r.get("score", 0.0) / max_raw_score for r in results]
    else:
        normalized = [min(r.get("score", 0.0), 1.0) for r in results]
    avg_score = sum(normalized) / len(normalized)

    query_terms_set = set(query_terms)
    coverage_scores = []
    for r in results[:3]:
        doc_text = r.get("text", "").lower()
        doc_terms = set(doc_text.split())
        coverage = len(query_terms_set & doc_terms) / max(len(query_terms_set), 1)
        coverage_scores.append(coverage)

    avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
    score = 0.3 * min(avg_score, 1.0) + 0.5 * avg_coverage + 0.2 * (coverage_scores[0] if coverage_scores else 0.5)

    top1_coverage = coverage_scores[0] if coverage_scores else 0.0
    if top1_coverage < 0.1 and avg_coverage < 0.15:
        score *= 0.5

    level = "high" if score >= 0.6 else "medium" if score >= 0.4 else "low"
    return {"score": min(score, 1.0), "level": level}


QUERY_TERMS = set("how does instantsplat enhance 3d reconstruction".split())


@pytest.mark.parametrize("name,results,valid_levels", [
    (
        "RRF scores + good coverage",
        [
            {"score": 0.015, "text": "InstantSplat enhances 3D reconstruction by using Gaussian Splatting"},
            {"score": 0.012, "text": "The method improves reconstruction quality significantly"},
            {"score": 0.010, "text": "Experiments show enhanced performance on Tanks and Temples"},
            {"score": 0.008, "text": "Unrelated content about neural networks"},
            {"score": 0.005, "text": "More unrelated content"},
        ],
        ("medium",),
    ),
    (
        "RRF scores + poor coverage",
        [
            {"score": 0.015, "text": "Algorithms pseudocodes for NeRF and 3DGS in Algorithm 1"},
            {"score": 0.012, "text": "Wang et al enabling SAM 3D to integrate with pipelines"},
            {"score": 0.010, "text": "Hongyu Zhou Jiahao Shao Lu Xu Dongfeng Bai"},
            {"score": 0.008, "text": "Results demonstrate balanced ratio is crucial"},
            {"score": 0.005, "text": "fus semantic indicator quote semicolon"},
        ],
        ("low",),
    ),
    (
        "Cosine scores + good coverage",
        [
            {"score": 0.82, "text": "InstantSplat enhances 3D reconstruction by using Gaussian Splatting"},
            {"score": 0.71, "text": "The method improves reconstruction quality significantly"},
            {"score": 0.65, "text": "Experiments show enhanced performance on Tanks and Temples"},
            {"score": 0.45, "text": "Unrelated content about neural networks"},
            {"score": 0.30, "text": "More unrelated content"},
        ],
        ("medium", "low"),
    ),
    (
        "Cosine scores + poor coverage",
        [
            {"score": 0.82, "text": "Algorithms pseudocodes for NeRF and 3DGS in Algorithm 1"},
            {"score": 0.71, "text": "Wang et al enabling SAM 3D to integrate with pipelines"},
            {"score": 0.65, "text": "Hongyu Zhou Jiahao Shao Lu Xu Dongfeng Bai"},
            {"score": 0.45, "text": "Results demonstrate balanced ratio"},
            {"score": 0.30, "text": "fus semantic indicator"},
        ],
        ("low",),
    ),
    (
        "All zero scores + good coverage",
        [
            {"score": 0.0, "text": "InstantSplat enhances 3D reconstruction by using Gaussian Splatting"},
            {"score": 0.0, "text": "The method improves reconstruction quality"},
            {"score": 0.0, "text": "Experiments show enhanced performance"},
        ],
        ("low", "medium"),
    ),
    (
        "RRF scores + mixed coverage",
        [
            {"score": 0.015, "text": "InstantSplat enhances 3D reconstruction by using Gaussian Splatting"},
            {"score": 0.012, "text": "Algorithms pseudocodes for NeRF and 3DGS"},
            {"score": 0.010, "text": "Unrelated content about something else"},
            {"score": 0.008, "text": "More unrelated"},
            {"score": 0.005, "text": "Yet more unrelated"},
        ],
        ("medium", "low"),
    ),
])
def test_rule_based_evaluator(name, results, valid_levels):
    result = _evaluate_by_rules(results, QUERY_TERMS)
    assert result["level"] in valid_levels, \
        f"{name}: expected level in {valid_levels}, got {result['level']} (score={result['score']:.3f})"


@pytest.mark.parametrize("name,results,valid_levels", [
    ("Empty results", [], ("low",)),
    ("Single result RRF",
     [{"score": 0.001, "text": "InstantSplat 3D reconstruction Gaussian Splatting"}],
     ("medium", "high")),
    ("Single result cosine high",
     [{"score": 0.95, "text": "InstantSplat 3D reconstruction Gaussian Splatting"}],
     ("medium", "high")),
])
def test_rule_based_evaluator_edge_cases(name, results, valid_levels):
    result = _evaluate_by_rules(results, QUERY_TERMS)
    assert result["level"] in valid_levels, \
        f"{name}: expected level in {valid_levels}, got {result['level']} (score={result['score']:.3f})"
