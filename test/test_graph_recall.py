"""
Test Plan B: graph → Neo4j chunk_id → Milvus paper-filtered vector search.

Simulates the full _build_graph_context flow:
1. Parse triplet texts → entity names
2. Query Neo4j for entity chunk_ids (= Milvus file_names)
3. Identify papers NOT already well-covered by vector search
4. Do a filtered Milvus search within those papers
5. Show what new real chunks the graph channel would bring in

Usage: .venv/bin/python -m test.test_graph_recall
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from pymilvus import connections, Collection
from FlagEmbedding import BGEM3FlagModel
from test._test_utils import get_neo4j_password

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = get_neo4j_password()
MILVUS_DB = str(Path(__file__).parent.parent / "data" / "milvus_papers.db")
BGE_PATH = str(Path(__file__).parent.parent / "models" / "bge-m3")

TEST_QUERIES = [
    "What is CF-3DGS and how does it compare in performance?",
    "What is DPT-Hybrid and how it works?",
    "What role does Abhijit Kundu play in the development of the Gaga framework?",
    "How does the MASt3R model contribute to the efficiency of neural 3D reconstruction?",
    "How does COLMAP contribute to the process of dense stereo reconstruction?",
]

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
    "propose", "proposes", "proposed", "based", "performance",
    "work", "works", "paper", "study", "task", "tasks",
})


def extract_keywords(query: str) -> list[str]:
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9_\-]*', query)
    tokens += [w for w in re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_]{2,}', query) if w not in tokens]
    raw = list(dict.fromkeys(w for w in tokens if w.lower() not in _STOPWORDS and len(w) >= 2))
    return [w for w in raw if not any(w != o and w in o for o in raw)][:5]


def main():
    # --- Init BGE-M3 ---
    print("Loading BGE-M3...")
    bge = BGEM3FlagModel(BGE_PATH, use_fp16=True, device="cpu")

    # --- Init Milvus ---
    print("Connecting to Milvus...")
    connections.connect(alias="test", uri=MILVUS_DB)
    coll = Collection("paper_embeddings", using="test")
    coll.load()

    # --- Init Neo4j ---
    neo4j_driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    # Pre-load all entities
    with neo4j_driver.session(database="neo4j") as session:
        result = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL AND n.chunk_id IS NOT NULL "
            "RETURN n.name AS name, n.chunk_id AS chunk_id"
        )
        all_entities = [(r["name"], r["chunk_id"]) for r in result]
    print(f"Loaded {len(all_entities)} entities from Neo4j\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"{'='*70}")
        print(f"[{i}] {query}")
        print(f"{'='*70}")

        keywords = extract_keywords(query)
        print(f"Keywords: {keywords}")

        # Step 1: Vector search (baseline)
        q_emb = bge.encode([query], batch_size=1, max_length=512)['dense_vecs']
        search_params = {"metric_type": "COSINE", "params": {}}
        milvus_hits = coll.search(
            data=q_emb.tolist(), anns_field="vector",
            param=search_params, limit=50,
            output_fields=["text", "metadata"]
        )
        vector_results = []
        for hits in milvus_hits:
            for h in hits:
                meta_str = h.entity.get("metadata", "{}")
                meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                vector_results.append({"text": h.entity.get("text", ""), "metadata": meta, "score": h.score})

        # Count vector coverage per paper
        vector_fn_counts = {}
        for r in vector_results:
            fn = r["metadata"].get("file_name", "")
            if fn:
                vector_fn_counts[fn] = vector_fn_counts.get(fn, 0) + 1

        print(f"Vector search: {len(vector_results)} chunks from {len(vector_fn_counts)} papers")

        # Step 2: Graph entity matching → chunk_ids
        # Simulate: use keyword matching against entity names (same as LLMSynonymRetriever)
        graph_file_names = set()
        for name, chunk_id in all_entities:
            for kw in keywords:
                if kw.lower() in name.lower():
                    graph_file_names.add(chunk_id)
                    break

        print(f"Graph entities: matched {len(graph_file_names)} papers")

        if not graph_file_names:
            print("  NO graph matches — skipping\n")
            continue

        # Step 3: Identify missed papers
        missed = [fn for fn in graph_file_names if vector_fn_counts.get(fn, 0) <= 1]
        well_covered = [fn for fn in graph_file_names if vector_fn_counts.get(fn, 0) > 1]

        print(f"  Well-covered by vector: {len(well_covered)} papers")
        print(f"  Missed / under-ranked: {len(missed)} papers")

        if missed:
            print(f"  Missed papers: {missed[:5]}")

        if not missed:
            print("  All graph papers already covered — no new chunks\n")
            continue

        # Step 4: Filtered Milvus search within missed papers
        paper_conditions = [f'metadata["file_name"] == "{pid}"' for pid in missed[:5]]
        filter_expr = " || ".join(paper_conditions) if len(paper_conditions) > 1 else paper_conditions[0]

        filtered_hits = coll.search(
            data=q_emb.tolist(), anns_field="vector",
            param=search_params, limit=10,
            expr=filter_expr,
            output_fields=["text", "metadata"]
        )

        # Step 5: Dedup and show results
        existing_prefixes = {r["text"][:100] for r in vector_results}
        new_chunks = 0
        for hits in filtered_hits:
            for h in hits:
                text = h.entity.get("text", "")
                if text[:100] in existing_prefixes:
                    continue
                meta_str = h.entity.get("metadata", "{}")
                meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
                fn = meta.get("file_name", "?")
                print(f"  NEW CHUNK [{new_chunks+1}] score={h.score:.4f} file={fn}")
                print(f"    {text[:120]}...")
                new_chunks += 1

        print(f"  Total new chunks from graph channel: {new_chunks}")
        if new_chunks == 0:
            print("  (all graph chunks already in vector results)")
        print()

    # Cleanup
    connections.disconnect("test")
    neo4j_driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
