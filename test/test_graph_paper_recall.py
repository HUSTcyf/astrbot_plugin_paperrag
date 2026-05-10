"""
Test: Graph paper recall in Stage 1 of two-stage retrieval.

Simulates the _graph_recall_papers flow:
1. PGRetriever.retrieve(query) → triplet texts
2. Parse entity names from triplets
3. Query Neo4j for chunk_ids (= paper filenames)
4. Return paper IDs, compare with abstract search coverage

Usage: cd astrbot_plugin_paperrag && python -m test.test_graph_paper_recall
"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "neo4j_M73770"
MILVUS_ABSTRACTS = str(Path(__file__).parent.parent / "data" / "milvus_abstracts.db")
BGE_PATH = str(Path(__file__).parent.parent / "models" / "bge-m3")

TEST_QUERIES = [
    "What is CF-3DGS and how does it compare in performance?",
    "What role does Abhijit Kundu play in the development of the Gaga framework?",
    "How does the MASt3R model contribute to the efficiency of neural 3D reconstruction?",
    "How does COLMAP contribute to the process of dense stereo reconstruction?",
    "What is the relationship between InstantSplat and DUSt3R?",
    "How is RMSE used in evaluating 3D reconstruction quality?",
]


def extract_entity_names(triplet_texts: list[str]) -> set:
    """Same logic as _graph_recall_papers."""
    entity_names = set()
    for text in triplet_texts:
        if not text:
            continue
        parts = [p.strip() for p in text.split(' -> ')]
        if len(parts) == 3:
            for idx in (0, 2):
                p = parts[idx]
                if p and len(p) > 1:
                    entity_names.add(p)
        else:
            for p in parts:
                if p and len(p) >= 2:
                    entity_names.add(p)
    return entity_names


def neo4j_paper_lookup(entity_names: set, driver) -> set:
    """Same logic as _graph_recall_papers — parameterized Cypher."""
    paper_ids = set()
    if not entity_names:
        return paper_ids
    with driver.session(database="neo4j") as session:
        result = session.run(
            "MATCH (n) WHERE n.name IN $names "
            "AND n.chunk_id IS NOT NULL "
            "RETURN DISTINCT n.chunk_id AS cid",
            names=list(entity_names)[:50],
        )
        for record in result:
            cid = record["cid"]
            if cid:
                if not cid.lower().endswith(".pdf"):
                    cid = cid + ".pdf"
                paper_ids.add(cid)
    return paper_ids


async def main():
    print("=" * 80)
    print("TEST: Graph Paper Recall in Stage 1")
    print("=" * 80)

    # ── 1. Init Neo4j ──
    print("\n[1] Connecting to Neo4j...")
    neo4j_driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with neo4j_driver.session(database="neo4j") as session:
        count = session.run(
            "MATCH (n) WHERE n.chunk_id IS NOT NULL RETURN count(n) AS c"
        ).single()["c"]
    print(f"  Neo4j: {count} entities with chunk_id")

    # ── 2. Init Graph Retriever (PGRetriever) ──
    print("\n[2] Initializing GraphRAGEngine + PGRetriever...")
    graph_store = None
    try:
        from graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig
        from rag.rag_engine import RAGConfig

        rag_config = RAGConfig(
            enable_graph_rag=True,
            graph_neo4j_uri=URI,
            graph_neo4j_user=USER,
            graph_neo4j_password=PASSWORD,
        )
        graph_config = GraphRAGConfig.from_rag_config(rag_config)

        # GraphRAGEngine needs base_engine; pass a minimal stub
        class _StubEngine:
            pass
        graph_engine = GraphRAGEngine(graph_config, _StubEngine())
        await graph_engine._init_index()
        pg_retriever = await graph_engine.get_retriever()

        if pg_retriever is None:
            print("  ❌ PGRetriever is None — cannot continue")
            neo4j_driver.close()
            return

        # Verify sub_retrievers structure
        sub_retrievers = getattr(pg_retriever, 'sub_retrievers', [])
        print(f"  PGRetriever: sub_retrievers={len(sub_retrievers)}")
        for i, sr in enumerate(sub_retrievers):
            print(f"    [{i}] {type(sr).__name__}")
            gs = getattr(sr, '_graph_store', None)
            print(f"        _graph_store: {type(gs).__name__ if gs else 'None'}")
            if gs:
                drv = getattr(gs, '_driver', None) or getattr(gs, 'client', None)
                print(f"        driver: {type(drv).__name__ if drv else 'None'}")

        # Get the graph_store for direct Neo4j access (same as production code)
        if sub_retrievers:
            graph_store = getattr(sub_retrievers[0], '_graph_store', None)

    except Exception as e:
        print(f"  ❌ GraphRAGEngine init failed: {e}")
        import traceback
        traceback.print_exc()
        neo4j_driver.close()
        return

    # ── 3. Init Milvus for abstract search (comparison) ──
    print("\n[3] Connecting to Milvus for abstract search...")
    try:
        from pymilvus import connections, Collection
        connections.connect(alias="test_abs", uri=MILVUS_ABSTRACTS)
        abstract_coll = Collection("paper_abstracts", using="test_abs")
        abstract_coll.load()
        print(f"  Abstracts: {abstract_coll.num_entities} papers")
        has_abstracts = True
    except Exception as e:
        print(f"  ⚠️ Abstract index not available: {e}")
        has_abstracts = False

    # ── 4. Load BGE-M3 for abstract vector search ──
    bge = None
    if has_abstracts:
        print("\n[4] Loading BGE-M3...")
        from FlagEmbedding import BGEM3FlagModel
        bge = BGEM3FlagModel(BGE_PATH, use_fp16=True, device="cpu")
        print(f"  BGE-M3 loaded")

    # ── 5. Run tests ──
    print("\n" + "=" * 80)
    print("GRAPH PAPER RECALL TEST")
    print("=" * 80)

    results_summary = []

    for qi, query in enumerate(TEST_QUERIES, 1):
        t0 = time.time()
        print(f"\n{'─' * 70}")
        print(f"[{qi}] {query}")

        # Step A: Graph retrieval → entity names → Neo4j paper IDs
        print(f"\n  ── Graph Retrieval ──")
        try:
            graph_result = await pg_retriever.retrieve(query)
            triplet_texts = []
            for nws in graph_result:
                text = getattr(nws.node, 'text', '') or ''
                if text:
                    triplet_texts.append(text)

            print(f"  Triplets retrieved: {len(triplet_texts)}")
            for t in triplet_texts[:5]:
                print(f"    {t[:100]}")

            entity_names = extract_entity_names(triplet_texts)
            print(f"  Entity names extracted: {len(entity_names)}")
            if entity_names:
                print(f"    {list(entity_names)[:10]}")

            # Neo4j lookup using the SAME driver access pattern as production code
            graph_paper_ids = set()
            if graph_store is not None:
                driver = getattr(graph_store, '_driver', None) or getattr(graph_store, 'client', None)
                if driver is not None:
                    graph_paper_ids = neo4j_paper_lookup(entity_names, driver)
            else:
                # Fallback: use direct neo4j_driver (same data, different access path)
                graph_paper_ids = neo4j_paper_lookup(entity_names, neo4j_driver)

            print(f"  Graph paper IDs: {len(graph_paper_ids)}")
            for pid in sorted(graph_paper_ids):
                print(f"    {pid}")

        except Exception as e:
            print(f"  ❌ Graph retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            graph_paper_ids = set()
            entity_names = set()

        # Step B: Abstract search (for comparison)
        abstract_paper_ids = set()
        if has_abstracts and bge:
            print(f"\n  ── Abstract Search ──")
            try:
                q_emb = bge.encode([query], batch_size=1, max_length=512)['dense_vecs']
                search_params = {"metric_type": "COSINE", "params": {}}
                abs_hits = abstract_coll.search(
                    data=q_emb.tolist(), anns_field="vector",
                    param=search_params, limit=20,
                    output_fields=["text", "metadata"]
                )
                for hits in abs_hits:
                    for h in hits:
                        meta_str = h.entity.get("metadata", "{}")
                        meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                        pid = meta.get("file_name", "")
                        if pid and not pid.lower().endswith(".pdf"):
                            pid = pid + ".pdf"
                        if pid:
                            abstract_paper_ids.add(pid)

                print(f"  Abstract paper IDs: {len(abstract_paper_ids)}")
                for pid in sorted(abstract_paper_ids)[:8]:
                    print(f"    {pid}")
                if len(abstract_paper_ids) > 8:
                    print(f"    ... +{len(abstract_paper_ids) - 8} more")

            except Exception as e:
                print(f"  ⚠️ Abstract search failed: {e}")

        # Step C: Compare
        print(f"\n  ── Comparison ──")
        if graph_paper_ids and abstract_paper_ids:
            overlap = graph_paper_ids & abstract_paper_ids
            graph_only = graph_paper_ids - abstract_paper_ids
            abstract_only = abstract_paper_ids - graph_paper_ids
            print(f"  Abstract top-20: {len(abstract_paper_ids)} papers")
            print(f"  Graph recall:    {len(graph_paper_ids)} papers")
            print(f"  Overlap:         {len(overlap)} papers")
            print(f"  Graph-only:      {len(graph_only)} papers ← NEW papers graph would add")
            print(f"  Abstract-only:   {len(abstract_only)} papers")
            if graph_only:
                for pid in sorted(graph_only):
                    print(f"    + {pid}")
            assessment = "SUPPLEMENTED" if graph_only else "NO NEW PAPERS"
        elif graph_paper_ids:
            print(f"  Graph recall: {len(graph_paper_ids)} papers (no abstract comparison)")
            graph_only = graph_paper_ids
            assessment = "GRAPH_ONLY"
        else:
            print(f"  ❌ Graph returned 0 papers")
            graph_only = set()
            assessment = "GRAPH_MISS"

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s  |  Assessment: {assessment}")

        results_summary.append({
            "query": query,
            "graph_papers": len(graph_paper_ids),
            "graph_only": len(graph_only) if isinstance(graph_only, set) else 0,
            "entities": len(entity_names),
            "assessment": assessment,
            "elapsed": elapsed,
        })

    # ── 6. Final summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Query':<60} {'GP':<4} {'New':<4} {'Ent':<4} {'Assessment'}")
    print(f"{'─' * 60} {'─' * 4} {'─' * 4} {'─' * 4} {'─' * 20}")
    total_graph_only = 0
    for r in results_summary:
        q_short = r['query'][:58]
        print(f"{q_short:<60} {r['graph_papers']:<4} {r['graph_only']:<4} {r['entities']:<4} {r['assessment']}")
        total_graph_only += r['graph_only']

    print(f"\nTotal new papers from graph: {total_graph_only}")
    if total_graph_only > 0:
        print("✅ Graph IS supplementing paper recall — new papers that abstract search missed")
    else:
        print("⚠️ Graph is NOT adding new papers — may need investigation")

    # Cleanup
    neo4j_driver.close()
    if has_abstracts:
        from pymilvus import connections as milvus_conns
        milvus_conns.disconnect("test_abs")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
