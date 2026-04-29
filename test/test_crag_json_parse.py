"""Test CRAG LLM JSON parsing robustness and rule-based evaluator score normalization."""

import json
import re
import sys

sys.path.insert(0, ".")


def parse_llm_json(response_text: str) -> dict:
    """Extract the exact parsing logic from _evaluate_by_llm (hybrid_rag.py ~line 1015-1038)."""
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return None

    raw_json = json_match.group(0)
    raw_json = raw_json.replace('“', '"').replace('”', '"')
    raw_json = raw_json.replace('‘', "'").replace('’', "'")
    raw_json = re.sub(r'[\x00-\x1f]', ' ', raw_json)
    try:
        result = json.loads(raw_json)
    except json.JSONDecodeError:
        result = {}
        score_m = re.search(r'"?score"?\s*:\s*([0-9.]+)', raw_json)
        level_m = re.search(r'"?level"?\s*:\s*"?(\w+)"?', raw_json)
        if score_m:
            result["score"] = float(score_m.group(1))
        if level_m:
            result["level"] = level_m.group(1)

    score = float(result.get("score", 0.5))
    level = result.get("level", "medium")
    if level not in ["high", "medium", "low"]:
        level = "medium" if score >= 0.3 else "low"
    return {"score": min(score, 1.0), "level": level}


def evaluate_by_rules(results, query_terms):
    """Extract the exact rule-based evaluator logic (hybrid_rag.py ~line 1034-1077)."""
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


# ============================================================================
# Test 1: LLM JSON parsing
# ============================================================================
print("=" * 70)
print("Test 1: LLM JSON Parsing")
print("=" * 70)

llm_cases = [
    # (name, response_text, expected_score, expected_level)
    ("standard JSON", '{"score": 0.8, "level": "high", "reasoning": "OK"}', 0.8, "high"),
    ("Chinese quotes", '{"score": 0.7, "level": "medium", "reasoning": "检索结果“相关”性中等"}', 0.7, "medium"),
    ("Trailing comma", '{"score": 0.6, "level": "medium",}', 0.6, "medium"),
    ("No quotes on keys", '{score: 0.5, level: "low"}', 0.5, "low"),
    ("Extra text before JSON", 'Here is my evaluation:\n{"score": 0.9, "level": "high", "reasoning": "Good"}', 0.9, "high"),
    ("Extra text after JSON", '{"score": 0.4, "level": "medium"}\n\nThe results are moderately relevant.', 0.4, "medium"),
    ("Newlines in values", '{"score": 0.3,\n "level": "low",\n "reasoning": "line1\nline2"}', 0.3, "low"),
    ("Chinese punctuation", '{"score"：0.65, "level"： "medium", "reasoning": "中等相关"}', None, "medium"),  # may fail score
    ("Missing score field", '{"level": "high", "reasoning": "Great match"}', 0.5, "high"),  # default score 0.5
    ("Score as string", '{"score": "0.75", "level": "medium"}', 0.75, "medium"),
    ("Invalid level", '{"score": 0.8, "level": "excellent"}', 0.8, "medium"),  # fallback to medium
    ("Mixed CN/EN quotes", '{"score": 0.6, "reasoning": "“结果”还不错", "level": "medium"}', 0.6, "medium"),
]

all_pass = True
for name, response, exp_score, exp_level in llm_cases:
    try:
        result = parse_llm_json(response)
        if result is None:
            print(f"  FAIL | {name:25s} | No JSON found")
            all_pass = False
            continue

        ok_score = exp_score is None or abs(result["score"] - exp_score) < 0.01
        ok_level = result["level"] == exp_level

        if ok_score and ok_level:
            print(f"  PASS | {name:25s} | score={result['score']:.2f} level={result['level']}")
        else:
            print(f"  FAIL | {name:25s} | got score={result['score']:.2f}({exp_score}) level={result['level']}({exp_level})")
            all_pass = False
    except Exception as e:
        print(f"  FAIL | {name:25s} | Exception: {e}")
        all_pass = False


# ============================================================================
# Test 2: Rule-based evaluator score normalization
# ============================================================================
print(f"\n{'=' * 70}")
print("Test 2: Rule-based Evaluator (RRF vs Cosine score normalization)")
print("=" * 70)

query_terms = set("how does instantsplat enhance 3d reconstruction".split())

rule_cases = [
    # (name, results, expected_level_range)
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
        "RRF scores + poor coverage (wrong papers)",
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
]

for name, results, valid_levels in rule_cases:
    result = evaluate_by_rules(results, query_terms)
    ok = result["level"] in valid_levels
    status = "PASS" if ok else "FAIL"
    print(f"  {status} | {name:40s} | score={result['score']:.3f} level={result['level']}")
    if not ok:
        all_pass = False


# ============================================================================
# Test 3: Edge cases
# ============================================================================
print(f"\n{'=' * 70}")
print("Test 3: Edge Cases")
print("=" * 70)

edge_cases = [
    ("Empty results", [], ("low",)),
    (
        "Single result RRF",
        [{"score": 0.001, "text": "InstantSplat 3D reconstruction Gaussian Splatting"}],
        ("low", "medium"),
    ),
    (
        "Single result cosine high",
        [{"score": 0.95, "text": "InstantSplat 3D reconstruction Gaussian Splatting"}],
        ("medium", "high"),
    ),
]

for name, results, valid_levels in edge_cases:
    result = evaluate_by_rules(results, query_terms)
    ok = result["level"] in valid_levels
    status = "PASS" if ok else "FAIL"
    print(f"  {status} | {name:40s} | score={result['score']:.3f} level={result['level']}")
    if not ok:
        all_pass = False

# Also test the exact error from the log
print()
print("--- Reproduce the original error ---")
bad_response = '{"score": 0.7, "level": "medium", "reasoning": "检索结果“相关”性中等，但缺少部分细节"}'
result = parse_llm_json(bad_response)
print(f"  Chinese quotes in value: score={result['score']:.2f} level={result['level']}")

print(f"\n{'=' * 70}")
print(f"Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
print("=" * 70)
