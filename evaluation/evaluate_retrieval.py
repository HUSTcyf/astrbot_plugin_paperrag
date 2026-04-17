#!/usr/bin/env python3
"""
检索效率与质量评估脚本

对比新旧两套方案：
  新版: Unsloth BGE-M3 稀疏权重 + 稠密向量 RRF  + ColBERT reranking
  旧版: BM25 + Ollama 稠密向量 RRF + FlagReranker cross-encoder

使用方法:
    cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag
    python evaluation/evaluate_retrieval.py
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Dict

# 添加插件目录到路径
SCRIPT_DIR = Path(__file__).parent
PLUGIN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PLUGIN_DIR))

TESTSET_PATH = PLUGIN_DIR / "results" / "testset.json"
OUTPUT_DIR = PLUGIN_DIR / "results" / "retrieval_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 延迟跟踪器
# ============================================================================

class LatencyTracker:
    def __init__(self):
        self.timers: dict[str, list[float]] = {}

    def record(self, name: str, elapsed_s: float):
        self.timers.setdefault(name, []).append(elapsed_s)

    def summary(self) -> dict:
        out = {}
        for name, times in self.timers.items():
            times.sort()
            n = len(times)
            total = sum(times)
            out[name] = {
                "count": n,
                "mean_ms": total / n * 1000,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
                "p50_ms": times[n // 2] * 1000,
                "p95_ms": times[int(n * 0.95)] * 1000,
                "p99_ms": times[int(n * 0.99)] * 1000,
            }
        return out


# ============================================================================
# 新版检索 (Unsloth 稀疏+稠密 RRF + ColBERT)
# ============================================================================

async def new_dense_only(index_manager, embed_provider, query: str, top_k: int) -> Any:
    from rag.hybrid_rag import VectorRetriever
    r = VectorRetriever(index_manager, embed_provider)
    return await r.retrieve(query, top_k=top_k)


async def new_sparse_only(index_manager, embed_provider, query: str, top_k: int) -> Any:
    from rag.hybrid_rag import SparseRetriever
    r = SparseRetriever(index_manager, embed_provider)
    return await r.retrieve(query, top_k=top_k)


async def new_hybrid_rrf(index_manager, embed_provider, query: str, top_k: int) -> Any:
    from rag.hybrid_rag import HybridRetriever
    r = HybridRetriever(
        index_manager=index_manager,
        embed_provider=embed_provider,
        sparse_top_k=20, vector_top_k=50,
        alpha=0.5, rrf_k=60,
        enable_reranking=False, rerank_top_k=top_k,
    )
    return await r.retrieve(query, top_k=top_k)


async def new_rerank_colbert(index_manager, embed_provider, query: str, top_k: int) -> Any:
    from rag.hybrid_rag import HybridRetriever
    r = HybridRetriever(
        index_manager=index_manager,
        embed_provider=embed_provider,
        sparse_top_k=20, vector_top_k=50,
        alpha=0.5, rrf_k=60,
        enable_reranking=True, rerank_top_k=top_k,
    )
    return await r.retrieve(query, top_k=top_k)


# ============================================================================
# 旧版检索 (BM25 + Ollama 稠密向量 + FlagReranker)
# ============================================================================

async def legacy_full_rerank(query: str, doc_texts: List[str], top_k: int) -> Optional[Any]:
    """旧版: FlagReranker cross-encoder reranking"""
    try:
        sys.path.insert(0, str(PLUGIN_DIR / "legacy"))
        from legacy.embedding.reranker import ContentReranker, RerankerConfig
        from rag.hybrid_rag import QueryResult, Node
        config = RerankerConfig(model_name="BAAI/bge-reranker-v2-m3", device="auto")
        reranker = ContentReranker(config)
        reranker._ensure_initialized()  # synchronous init
        docs = [{"text": doc, "metadata": {}, "score": 0.0} for doc in doc_texts]
        reranked = await reranker.rerank(query, docs, top_k=top_k)
        # reranked = [{"text": ..., "metadata": ..., "score": ...}, ...]
        nodes = [Node(text=r["text"], metadata=r.get("metadata", {})) for r in reranked]
        scores = [r.get("score", 0.0) for r in reranked]
        return QueryResult(nodes=nodes, scores=scores)
    except Exception as e:
        print(f"    ⚠️ Legacy FlagReranker failed: {e}")
        return None


async def legacy_bm25_dense_rrf(query: str, top_k: int, embed_provider, index_manager) -> Any:
    """旧版: BM25 + 稠密向量 RRF"""
    try:
        from rag.hybrid_rag import VectorRetriever
        bm25_hits = await index_manager.bm25_search(query, top_k=top_k)
        dense_r = VectorRetriever(index_manager, embed_provider)
        dense_result = await dense_r.retrieve(query, top_k=top_k)
        return _rrf_fuse_bm25_and_dense(bm25_hits, dense_result, k=60)
    except Exception as e:
        print(f"    ⚠️ Legacy BM25+RRF failed: {e}")
        return None


def _rrf_fuse_bm25_and_dense(bm25_hits: List[Dict], dense_result: Any, k=60, alpha=0.5) -> Any:
    """
    RRF 融合 BM25 命中(列表) 和 稠密检索结果(QueryResult)
    与 HybridRetriever._rrf_fusion() 保持一致：纯 rank-based RRF 公式
    RRF score = alpha * 1/(k + v_rank) + (1-alpha) * 1/(k + s_rank)
    """
    from rag.hybrid_rag import QueryResult, Node

    # 构建 rank 映射（1-indexed，排名从1开始）
    bm25_rank_map = {}
    sorted_bm25 = sorted(bm25_hits, key=lambda x: x.get("score", 0), reverse=True)
    for i, hit in enumerate(sorted_bm25):
        bm25_rank_map[hash(hit["text"])] = i + 1

    n_dense = len(dense_result.nodes)
    dense_rank_map = {hash(n.text): i + 1 for i, n in enumerate(dense_result.nodes)}

    all_hashes = set(bm25_rank_map.keys()) | set(dense_rank_map.keys())
    default_bm25_rank = len(sorted_bm25) + 1
    default_dense_rank = n_dense + 1

    rrf_scores = {}
    for h in all_hashes:
        v_rank = dense_rank_map.get(h, default_dense_rank)
        s_rank = bm25_rank_map.get(h, default_bm25_rank)
        rrf_scores[h] = alpha * (1.0 / (k + v_rank)) + (1 - alpha) * (1.0 / (k + s_rank))

    sorted_hashes = sorted(rrf_scores, key=lambda h: rrf_scores[h], reverse=True)

    # 构建 text->metadata 映射
    text_to_meta = {}
    for n in dense_result.nodes:
        text_to_meta[n.text] = getattr(n, "metadata", {})
    for hit in bm25_hits:
        t = hit["text"]
        if t not in text_to_meta:
            text_to_meta[t] = hit.get("metadata", {})

    nodes, scs = [], []
    for h in sorted_hashes[:20]:
        for hit in bm25_hits:
            if hash(hit["text"]) == h:
                nodes.append(Node(text=hit["text"], metadata=text_to_meta.get(hit["text"], {})))
                scs.append(rrf_scores[h])
                break
        else:
            for n in dense_result.nodes:
                if hash(n.text) == h:
                    nodes.append(n)
                    scs.append(rrf_scores[h])
                    break

    return QueryResult(nodes=nodes, scores=scs)


# ============================================================================
# 主评估
# ============================================================================

async def evaluate_queries(
    queries: List[dict],
    index_manager,
    embed_provider,
    tracker: LatencyTracker,
    top_k: int,
):
    results = []

    for i, item in enumerate(queries):
        q = item["question"]
        print(f"\n[{i+1}/{len(queries)}] {q[:60]}...")

        # ---- 新版 Dense ----
        t0 = time.perf_counter()
        try:
            r = await new_dense_only(index_manager, embed_provider, q, top_k)
        except Exception as e:
            print(f"  ⚠️ new_dense: {e}"); r = None
        tracker.record("new_dense_ms", (time.perf_counter() - t0) * 1000)

        # ---- 新版 Sparse ----
        t0 = time.perf_counter()
        try:
            r2 = await new_sparse_only(index_manager, embed_provider, q, top_k)
        except Exception as e:
            print(f"  ⚠️ new_sparse: {e}"); r2 = None
        tracker.record("new_sparse_ms", (time.perf_counter() - t0) * 1000)

        # ---- 新版 RRF ----
        t0 = time.perf_counter()
        try:
            r3 = await new_hybrid_rrf(index_manager, embed_provider, q, top_k)
        except Exception as e:
            print(f"  ⚠️ new_rrf: {e}"); r3 = None
        tracker.record("new_rrf_ms", (time.perf_counter() - t0) * 1000)

        # ---- 新版 ColBERT rerank ----
        t0 = time.perf_counter()
        try:
            r4 = await new_rerank_colbert(index_manager, embed_provider, q, top_k=5)
        except Exception as e:
            print(f"  ⚠️ new_colbert: {e}"); r4 = None
        tracker.record("new_colbert_ms", (time.perf_counter() - t0) * 1000)

        # ---- 旧版 BM25+Dense RRF ----
        t0 = time.perf_counter()
        try:
            r5 = await legacy_bm25_dense_rrf(q, top_k, embed_provider, index_manager)
        except Exception as e:
            print(f"  ⚠️ legacy_bm25_rrf: {e}"); r5 = None
        tracker.record("legacy_bm25_rrf_ms", (time.perf_counter() - t0) * 1000)

        # ---- 旧版 FlagReranker (以 r5 的 doc_texts 为输入) ----
        if r5 and r5.nodes:
            doc_texts = [n.text[:200] for n in r5.nodes[:10]]
            t0 = time.perf_counter()
            try:
                r6 = await legacy_full_rerank(q, doc_texts, top_k=5)
            except Exception as e:
                print(f"  ⚠️ legacy_flag: {e}"); r6 = None
            tracker.record("legacy_flag_ms", (time.perf_counter() - t0) * 1000)
        else:
            tracker.record("legacy_flag_ms", 0)

        entry = {
            "question": q,
            "new_dense_count": len(r.nodes) if r else 0,
            "new_sparse_count": len(r2.nodes) if r2 else 0,
            "new_rrf_count": len(r3.nodes) if r3 else 0,
            "new_colbert_count": len(r4.nodes) if r4 else 0,
            "legacy_bm25_rrf_count": len(r5.nodes) if r5 else 0,
        }
        results.append(entry)

        print(f"  Dense:{tracker.timers['new_dense_ms'][-1]:.1f}ms | Sparse:{tracker.timers['new_sparse_ms'][-1]:.1f}ms | RRF:{tracker.timers['new_rrf_ms'][-1]:.1f}ms | ColBERT:{tracker.timers['new_colbert_ms'][-1]:.1f}ms | BM25+RRF:{tracker.timers['legacy_bm25_rrf_ms'][-1]:.1f}ms | Flag:{tracker.timers['legacy_flag_ms'][-1]:.1f}ms")

    return results


async def main():
    parser = argparse.ArgumentParser(description="新旧检索方案对比评估")
    parser.add_argument("--limit", "-n", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--qasper-db", type=str, default=None,
                        help="Qasper Milvus 数据库路径 (默认: ./data/milvus_qasper_text.db)")
    args = parser.parse_args()

    # 加载测试集
    if not TESTSET_PATH.exists():
        print(f"❌ 测试集不存在: {TESTSET_PATH}"); return
    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    queries = testset[:args.limit] if args.limit > 0 else testset
    print(f"📋 {len(queries)} 查询")

    # 初始化引擎
    print("\n🚀 初始化引擎...")
    from rag.rag_engine import RAGConfig, create_rag_engine

    class DummyContext:
        provider_manager = platform_manager = conversation_manager = persona_manager = None

    # 使用 papers 数据库（testset 对应 milvus_papers.db）
    db_path = args.qasper_db if args.qasper_db else str(PLUGIN_DIR / "data" / "milvus_papers.db")
    if args.qasper_db:
        print(f"📦 使用数据库: {db_path}")
    else:
        print(f"📦 使用默认 papers 数据库: {db_path}")

    config = RAGConfig(milvus_lite_path=db_path)
    engine = create_rag_engine(config, DummyContext())
    index_manager = engine._ensure_index_manager_initialized()
    embed_provider = await engine._ensure_embed_provider_initialized()
    print(f"✅ 完成 (sparse={config.enable_sparse_retrieval}, rerank={config.enable_multi_vector_rerank})")

    # 预热
    if args.warmup:
        print(f"🔥 预热 {args.warmup} 次...")
        for _ in range(args.warmup):
            await engine.search("What is 3D reconstruction?", top_k=5)
        print("✅ 预热完成")

    # 评估
    tracker = LatencyTracker()
    print(f"\n📊 开始评估...")
    results = await evaluate_queries(queries, index_manager, embed_provider, tracker, top_k=args.top_k)

    # 输出
    smry = tracker.summary()
    print("\n" + "=" * 70)
    print("📈 延迟统计 (ms)")
    print("=" * 70)
    labels = [
        ("new_dense_ms",       "新版 Dense"),
        ("new_sparse_ms",      "新版 Sparse(ABSPEC)"),
        ("new_rrf_ms",         "新版 RRF(稀疏+稠密)"),
        ("new_colbert_ms",     "新版 RRF+ColBERT"),
        ("legacy_bm25_rrf_ms",  "旧版 BM25+稠密RRF"),
        ("legacy_flag_ms",     "旧版 FlagReranker"),
    ]
    for key, label in labels:
        if key in smry:
            s = smry[key]
            print(f"  {label:<22} mean={s['mean_ms']:7.2f}  p50={s['p50_ms']:7.2f}  p95={s['p95_ms']:7.2f}  max={s['max_ms']:7.2f}")

    # 效率对比
    print("\n" + "=" * 70)
    print("⚡ 效率对比")
    print("=" * 70)
    new_rrf = smry.get("new_rrf_ms", {}).get("mean_ms", 0)
    new_colbert = smry.get("new_colbert_ms", {}).get("mean_ms", 0)
    legacy_rrf = smry.get("legacy_bm25_rrf_ms", {}).get("mean_ms", 0)
    legacy_flag = smry.get("legacy_flag_ms", {}).get("mean_ms", 0)

    if new_rrf and legacy_rrf:
        print(f"  RRF阶段: 新版={new_rrf:.1f}ms vs 旧版={legacy_rrf:.1f}ms  ({(1-new_rrf/legacy_rrf)*100:+.1f}%)")
    if new_colbert and legacy_flag:
        print(f"  Rerank阶段: 新版ColBERT={new_colbert:.1f}ms vs 旧版Flag={legacy_flag:.1f}ms  ({(1-new_colbert/legacy_flag)*100:+.1f}%)")
    total_new = new_rrf + new_colbert
    total_legacy = legacy_rrf + legacy_flag
    if total_new and total_legacy:
        print(f"  合计(新版): {total_new:.1f}ms")
        print(f"  合计(旧版): {total_legacy:.1f}ms  ({(1-total_new/total_legacy)*100:+.1f}%)")

    # 保存
    out = OUTPUT_DIR / f"eval_results_{int(time.time())}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary_ms": smry, "per_query": results}, f, indent=2, ensure_ascii=False)
    print(f"\n💾 已保存: {out}")


if __name__ == "__main__":
    asyncio.run(main())
