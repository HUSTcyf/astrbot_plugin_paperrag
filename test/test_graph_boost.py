"""
Test: RRF Graph Boost effectiveness on RAGAS failure queries.
Uses real BGE-M3 (local) + real Milvus + real Neo4j + real _rrf_fusion().
"""
import sys, os, json, re, asyncio, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "neo4j_M73770"
MILVUS_DB = os.path.join(os.path.dirname(__file__), "data", "milvus_papers.db")
BGE_PATH = os.path.join(os.path.dirname(__file__), "models", "bge-m3")

# ── Load BGE-M3 ──────────────────────────────────────────────
print("Loading BGE-M3...")
from FlagEmbedding import BGEM3FlagModel
bge = BGEM3FlagModel(BGE_PATH, use_fp16=True, device="cpu")
print(f"  dim={bge.model.model.config.hidden_size}")

# ── Load Milvus ──────────────────────────────────────────────
print("Connecting to Milvus...")
from pymilvus import connections, Collection
connections.connect(alias="test", uri=MILVUS_DB)
coll = Collection("paper_embeddings", using="test")
coll.load()
print(f"  {coll.num_entities} entities")

# ── Load Neo4j entities ──────────────────────────────────────
print("Loading KG entities...")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
with driver.session(database="neo4j") as session:
    result = session.run("""
        MATCH (n) WHERE n.chunk_id IS NOT NULL
        RETURN n.name AS name, n.chunk_id AS chunk_id, labels(n)[0] AS label
    """)
    kg_entities = [(r["name"], r["chunk_id"], r["label"]) for r in result]
driver.close()
print(f"  {len(kg_entities)} entities")

# ── Import actual RRF fusion ─────────────────────────────────
from rag.hybrid_rag import HybridRetriever

# Stopwords from _fetch_subgraph_keywords
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "has", "have", "had", "can", "could", "will",
    "would", "should", "may", "might", "shall", "must", "need",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "my", "our", "your", "his", "her", "their",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "and", "or", "not", "no", "nor", "but", "so", "if", "then",
    "than", "too", "very", "just", "also", "only", "such", "each", "all",
    "any", "both", "few", "more", "most", "other", "some", "up", "out",
    "method", "methods", "approach", "approaches", "model", "models",
    "dataset", "datasets", "result", "results", "compare", "comparison",
    "limitation", "limitations", "achieve", "achieves", "achieved",
    "evaluate", "evaluated", "evaluation", "use", "used", "using",
    "propose", "proposes", "proposed", "based", "performance",
    "novel", "new", "different", "improve", "improves", "improved",
    "work", "works", "paper", "study", "task", "tasks",
    "component", "components", "technique", "techniques",
    "better", "best", "state", "art", "recent", "efficient",
    "effective", "show", "shown", "describe", "described",
    "explain", "tell", "give", "list", "find", "found",
})

# ── Test queries ─────────────────────────────────────────────
queries = {
    "DPT-Hybrid": "What is DPT-Hybrid and how it works?",
    "Gaga": "What role does Abhijit Kundu play in the development of the Gaga framework?",
    "MipNeRF360": "What datasets are used for evaluating MipNeRF 360 in the experiments?",
    "FSGS": "What is FSGS and how does it relate to the initialization process in 3D Gaussian Splatting?",
    "CNN": "What role does CNN play in computer vision tasks?",
    "Octree-GS": "What is Octree-GS and how does it relate to the training strategy for large-scale scene reconstruction?",
    "InstantSplat": "What contributions has Boris Ivanovic made in the field of 3D reconstruction?",
    "RMSE": "How is RMSE used in the evaluation of 3D reconstruction quality?",
    "LUDVIG": "What advantages does LUDVIG provide over N3F in terms of object removal and segmentation performance?",
}

# ── Run tests ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RRF GRAPH BOOST TEST: Real BGE-M3 + Real Milvus + Real Neo4j")
print("=" * 80)

results_summary = []

for tag, query in queries.items():
    t0 = time.time()
    print(f"\n{'─'*70}")
    print(f"[{tag}] {query[:100]}")

    # 1. Vector search via Milvus
    q_emb = bge.encode([query], batch_size=1, max_length=512)['dense_vecs']
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    milvus_results = coll.search(
        data=q_emb.tolist(), anns_field="vector",
        param=search_params, limit=50,
        output_fields=["text", "metadata"]
    )

    vector_results = []
    for hits in milvus_results:
        for h in hits:
            meta_str = h.entity.get("metadata", "{}")
            meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
            vector_results.append({
                "text": h.entity.get("text", ""),
                "metadata": {
                    "file_name": meta.get("file_name", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                },
                "score": h.score,
            })

    # 2. Graph entity matching (keyword-based, simulating graph retriever)
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9_\-]*', query)
    tokens += [w for w in re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_]{2,}', query) if w not in tokens]
    raw = list(dict.fromkeys(w for w in tokens if w.lower() not in _STOPWORDS and len(w) >= 2))
    keywords = [w for w in raw if not any(w != o and w in o for o in raw)][:5]

    graph_chunk_boost = {}
    graph_text_results = []
    matched_names = []
    for name, cid, label in kg_entities:
        for kw in keywords:
            if kw.lower() in name.lower():
                score = min(1.0, 0.5 + len(kw) / max(len(name), 1) * 0.5)
                graph_chunk_boost[cid] = max(graph_chunk_boost.get(cid, 0.0), score)
                graph_text_results.append({
                    "text": f"Graph context for {name} ({label})",
                    "metadata": {"chunk_id": cid, "file_name": cid, "retrieval_type": "graph"},
                    "score": score,
                })
                matched_names.append(name)
                break

    n_papers = len(graph_chunk_boost)
    if n_papers == 0:
        print(f"  KG: NO match (keywords={keywords})")
        results_summary.append((tag, 0, 0, 0, 0))
        continue

    print(f"  Keywords: {keywords}")
    print(f"  KG: {len(matched_names)} entities → {n_papers} papers")

    # 3. Build a temporary HybridRetriever just to access _rrf_fusion
    # (We don't need full init since we only call the static-like method)
    class _MockIndex: pass
    class _MockEmbed: pass
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._graph_weight = 0.2
    retriever._alpha = 0.5
    retriever._rrf_k = 60

    # 4. Test different graph weights
    weights_to_test = [0.0, 0.1, 0.2, 0.3, 0.5]

    print(f"\n  {'Weight':<8} {'Boosted':<8} {'In Top10':<10} {'Avg Δrank':<10} {'Max Δrank':<10} Best boosted chunk")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*45}")

    best_result = None

    for w in weights_to_test:
        retriever._graph_weight = w

        if w == 0.0:
            fused = retriever._rrf_fusion(
                vector_results=vector_results,
                sparse_results={},
                bm25_results=None,
                graph_text_results=None,
                top_k=10,
            )
        else:
            fused = retriever._rrf_fusion(
                vector_results=vector_results,
                sparse_results={},
                bm25_results=None,
                graph_text_results=graph_text_results,
                top_k=10,
            )

        # Count boosted chunks in top 10
        boosted_in_top10 = 0
        for item in fused:
            meta = item.get("metadata", {})
            fn = meta.get("file_name", "")
            if fn and fn in graph_chunk_boost:
                boosted_in_top10 += 1

        # Compare with w=0 baseline
        if w == 0.0:
            baseline_texts = [item["text"] for item in fused]
            baseline_ranks = {t: i+1 for i, t in enumerate(baseline_texts)}
            print(f"  {w:<8.1f} {'─':<8} {'─':<10} {'─':<10} {'─':<10} (baseline)")
        else:
            # Calculate rank changes for boosted chunks
            rank_deltas = []
            best_delta = (0, "")
            for i, item in enumerate(fused):
                meta = item.get("metadata", {})
                fn = meta.get("file_name", "")
                if fn and fn in graph_chunk_boost:
                    old_rank = baseline_ranks.get(item["text"], 99)
                    new_rank = i + 1
                    delta = old_rank - new_rank
                    rank_deltas.append(delta)
                    if delta > best_delta[0]:
                        best_delta = (delta, item["text"][:50])

            n_boosted = len(rank_deltas)
            avg_delta = sum(rank_deltas) / n_boosted if n_boosted else 0
            max_delta = max(rank_deltas) if n_boosted else 0

            detail = ""
            if best_delta[0] > 0:
                detail = f"best +{best_delta[0]}: {best_delta[1]}..."

            print(f"  {w:<8.1f} {n_boosted:<8} {boosted_in_top10:<10} {avg_delta:+.1f}      {max_delta:+d}        {detail}")

            if w == 0.2:
                best_result = (tag, n_boosted, boosted_in_top10, avg_delta, max_delta)

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    results_summary.append((tag, n_papers,
        best_result[1] if best_result else 0,
        best_result[2] if best_result else 0,
        best_result[3] if best_result else 0))

# ── Final summary ────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: Graph Boost Effect at weight=0.2")
print("=" * 80)
print(f"{'Query':<16} {'Papers':<8} {'Boosted':<8} {'In Top10':<10} {'Avg Δrank':<10} Assessment")
print(f"{'─'*16} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*30}")
for tag, n_papers, n_boosted, in_top10, avg_delta in results_summary:
    if n_papers == 0:
        assessment = "KG miss — no effect"
    elif avg_delta > 2:
        assessment = "Strong positive impact"
    elif avg_delta > 0.5:
        assessment = "Moderate positive impact"
    elif n_boosted > 0:
        assessment = "Small positive impact"
    else:
        assessment = "No measurable effect"
    print(f"{tag:<16} {n_papers:<8} {n_boosted:<8} {in_top10:<10} {avg_delta:+.1f}       {assessment}")

# Cleanup
connections.disconnect("test")
print("\nDone.")
