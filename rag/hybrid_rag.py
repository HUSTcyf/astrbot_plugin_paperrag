"""
混合架构RAG引擎 - BGE-M3 稀疏权重 + 多向量版

基于 Unsloth BGE-M3 本地加载，实现：
1. PDF解析（多模态）→ HybridPDFParser
2. 文档分块 → Node结构
3. 向量存储 → HybridIndexManager（避免与主进程冲突）
4. 检索：
   - 稠密向量检索 (dense vector)
   - 稀疏权重检索 (sparse weight from ABSPEC)
   - RRF 分数融合
   - ColBERT 多向量 reranking
5. 生成 → LLM（支持多模态）
"""

import asyncio
import os
import ast
import re
import sys
import shutil
import traceback
import json
from typing import List, Dict, Tuple, Any, Optional, Union, cast
from pathlib import Path
from itertools import zip_longest

# 抑制底层库的 gRPC/absl 警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# 添加插件根目录到 sys.path（支持 idea.* 模块导入）
_plugin_root = Path(__file__).parent.parent
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))

from astrbot.api import logger
import gc
import jieba
import numpy as np
from .colbert_storage import ColBERTStorage

# 获取插件根目录，用于解析相对路径
_PLUGIN_DIR = _plugin_root


def _is_mps_oom_error(error: Any) -> bool:
    """Return True for PyTorch MPS out-of-memory style errors."""
    text = str(error).lower()
    return "mps backend out of memory" in text or (
        "mps" in text and "out of memory" in text
    )


def _clear_accelerator_cache() -> None:
    """Best-effort cache cleanup after accelerator OOM."""

    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.debug(f"[Memory] 清理加速器缓存失败: {e}")

# 导入混合架构组件
try:
    from .hybrid_parser import HybridPDFParser, Node
    from .hybrid_index import HybridIndexManager
    from embedding.embedding_providers import (
        create_embedding_provider,
        UnslothEmbeddingProvider,
        AstrBotEmbeddingProvider,
        FlagEmbeddingProvider,
    )
    from .rag_engine import RAGConfig
except ImportError:
    from .hybrid_parser import HybridPDFParser, Node
    from .hybrid_index import HybridIndexManager
    from ..embedding.embedding_providers import (
        create_embedding_provider,
        UnslothEmbeddingProvider,
        AstrBotEmbeddingProvider,
        FlagEmbeddingProvider,
    )
    from .rag_engine import RAGConfig

# 导入 Unsloth embedding 获取稀疏权重和多向量
try:
    from embedding.unsloth_embedding import get_embedding_model
except ImportError:
    from ..embedding.unsloth_embedding import get_embedding_model

# 导入 FlagEmbedding 模型（跨平台备选）
try:
    from embedding.flag_embedding import get_flag_model
except ImportError:
    from ..embedding.flag_embedding import get_flag_model

# 导入 Llama.cpp VLM Provider（用于图片问答）
LLAMA_CPP_VLM_AVAILABLE = False
LLAMA_CPP_VLM_IMPORT_ERROR = None
try:
    from provider.llama_cpp_vlm import LlamaCppVLMProvider
    LLAMA_CPP_VLM_AVAILABLE = True
    logger.info("[Llama.cpp-VLM] Llama.cpp VLM Provider 已加载")
except ImportError as e:
    LLAMA_CPP_VLM_IMPORT_ERROR = e
    logger.warning(f"[Llama.cpp-VLM] Llama.cpp VLM Provider 导入失败: {e}")


class QueryResult:
    """查询结果封装类（llama-index风格）"""

    def __init__(
        self,
        nodes: List[Node],
        scores: Optional[List[float]] = None
    ):
        self.nodes = nodes
        self.scores = scores or [1.0] * len(nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, index: int) -> Node:
        return self.nodes[index]


class BaseRetriever:
    """检索器基类"""

    def __init__(self, index_manager: HybridIndexManager, embed_provider: Any):
        self._index_manager = index_manager
        self._embed_provider = embed_provider

    async def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        """检索相关文档"""
        raise NotImplementedError


class VectorRetriever(BaseRetriever):
    """向量检索器"""

    async def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        """使用向量相似度检索"""
        # 获取查询向量
        query_embedding = await self._embed_provider.get_text_embedding(query)

        # 执行向量搜索
        results = await self._index_manager.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 转换为Node列表
        nodes = []
        scores = []
        for item in results:
            node = Node(
                text=item["text"],
                metadata=item.get("metadata", {})
            )
            nodes.append(node)
            scores.append(item.get("score", 0.0))

        return QueryResult(nodes=nodes, scores=scores)


class SparseRetriever(BaseRetriever):
    """
    稀疏权重检索器（使用 BGE-M3 ABSPEC）

    利用 BGE-M3 的 token hidden states 计算查询关键词的重要性，
    结合文档的稀疏表示进行匹配
    """

    def __init__(self, index_manager: HybridIndexManager, embed_provider: Any):
        super().__init__(index_manager, embed_provider)
        self._embedding_model = None

    def _get_embedding_model(self):
        """获取 embedding 模型（优先已初始化的 FlagEmbedding，降级 Unsloth）"""
        if self._embedding_model is None:
            flag = get_flag_model()
            if flag._initialized:
                self._embedding_model = flag
            else:
                self._embedding_model = get_embedding_model()
            if self._embedding_model is None:
                logger.error("[PaperRAG] embedding 模型不可用（FlagEmbedding 和 Unsloth 均返回 None）")
        return self._embedding_model

    async def retrieve(self, query: str, top_k: int = 20) -> QueryResult:
        """使用稀疏权重检索"""
        try:
            # 获取查询向量（用于计算稀疏权重）
            query_embedding = await self._embed_provider.get_text_embedding(query)
            model = self._get_embedding_model()

            # 计算查询的稀疏权重
            query_sparse = model.get_sparse_weight(query, query_embedding)

            if not query_sparse:
                logger.debug("[SparseRetriever] 稀疏权重为空，降级到向量检索")
                fallback = VectorRetriever(self._index_manager, self._embed_provider)
                return await fallback.retrieve(query, top_k)

            # 先通过 Milvus 向量检索取 top_k 候选（与 dense 通道独立并行）
            vector_fallback = VectorRetriever(self._index_manager, self._embed_provider)
            candidates = await vector_fallback.retrieve(query, top_k=top_k)

            if not candidates.nodes:
                return QueryResult(nodes=[], scores=[])

            # 对候选文档计算稀疏匹配分数
            scored_docs = []
            for node, orig_score in zip(candidates.nodes, candidates.scores):
                doc_text = node.text
                if not doc_text:
                    continue
                score = model.get_query_sparse_vs_doc_dense(query, doc_text)
                scored_docs.append({
                    "text": doc_text,
                    "metadata": node.metadata,
                    "score": score
                })

            # 按稀疏分数降序排列
            scored_docs.sort(key=lambda x: x["score"], reverse=True)

            nodes = []
            scores = []
            for item in scored_docs[:top_k]:
                nodes.append(Node(
                    text=item["text"],
                    metadata=item.get("metadata", {})
                ))
                scores.append(item.get("score", 0.0))

            return QueryResult(nodes=nodes, scores=scores)

        except Exception as e:
            logger.error(f"[SparseRetriever] 稀疏检索失败: {e}")
            fallback = VectorRetriever(self._index_manager, self._embed_provider)
            return await fallback.retrieve(query, top_k)


class BM25Retriever:
    """
    BM25 精确匹配检索器

    用于专有名词、作者名、数字等需要精确匹配的查询。
    使用 jieba 分词 + rank_bm25 计算相关性分数。
    """

    def __init__(self, index_manager: HybridIndexManager):
        self._index_manager = index_manager
        self._bm25 = None
        self._corpus_texts: List[str] = []
        self._corpus_metadata: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        """使用 jieba 分词并标准化（与 legacy/hybrid_index.py 一致）"""

        # 1. 转小写
        text = text.lower()
        # 2. 将连字符/下划线连接的词合并（如 anti-scam -> antiscam）
        text = re.sub(r'([a-z])[-_]([a-z])', r'\1\2', text)
        # 3. 使用 jieba 分词
        tokens = list(jieba.cut(text))
        # 4. 去除纯标点 token
        tokens = [t for t in tokens if not re.match(r'^[\s\W]+$', t)]
        return tokens

    async def _ensure_index(self) -> bool:
        """确保 BM25 索引已构建"""
        if self._bm25 is not None:
            return True

        try:
            from rank_bm25 import BM25Okapi

            # 从 Milvus 获取所有 chunks
            chunks = await self._index_manager.get_all_chunks()
            if not chunks:
                logger.warning("[BM25Retriever] Milvus 中无数据，无法构建 BM25 索引")
                return False

            self._corpus_texts = [c["text"] for c in chunks]
            self._corpus_metadata = [c.get("metadata", {}) for c in chunks]

            # 使用 jieba 分词构建 BM25 索引
            tokenized_corpus = [self._tokenize(text) for text in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenized_corpus)

            logger.info(f"✅ BM25 索引构建完成: {len(self._corpus_texts)} 个 chunks")
            return True

        except ImportError:
            logger.error("rank_bm25 未安装，请运行: pip install rank-bm25")
            return False
        except Exception as e:
            logger.error(f"[BM25Retriever] BM25 索引构建失败: {e}")
            return False

    def is_exact_match_query(self, query: str) -> bool:
        """
        检测查询是否需要精确匹配（专有名词、作者名、数字等）

        返回 True 的情况：
        - 查询包含人名（带 "et al." 或 "等"）
        - 查询包含数字（年份、DOI、arXiv ID）
        - 查询包含具体名称（论文标题、机构名）
        - 查询包含特殊符号（括号、连字符等）
        - 查询是事实性问题（who, when, where, what role/contributions）
        """
        query_lower = query.lower()

        # 1. 检测是否包含具体名称模式（这些需要精确匹配）
        exact_match_patterns = [
            r'\b[A-Z][a-z]+\s+(?:et\s+al\.?|等)\b',  # "Smith et al."
            r'\b[A-Z][a-z]+,\s*[A-Z]\.\b',  # "Smith, J."
            r'\b\d{4}\b',  # 年份 "2024"
            r'arxiv:\s*\d+\.\d+',  # arXiv ID
            r'doi:\s*\d+',  # DOI
            r'\b\d{2,}\b',  # 数字（页码、版本号等）
            r'["""].*["""]',  # 引用的确切词语
            r'\([^)]{1,30}\)',  # 括号内的短文本（可能是专有名词）
            r'[-_]',  # 包含连字符的专有名词
            r'\b[A-Z]{2,}\b',  # 全大写缩写 (DTU, LERF, RMSE, SAC)
            r'[A-Z][a-z]+[A-Z][a-z]*',  # CamelCase (SparseNeRF, PointNeRF, InstantNGP)
        ]

        for pattern in exact_match_patterns:
            if re.search(pattern, query):
                return True

        # 2. 检测是否是事实性问题（who, when, where, which name）
        factual_patterns = [
            r'\bwho\s+(?:proposed|suggested|developed|created|invented|authored|is|are|was|were)\b',
            r'\bwhen\s+(?:was|did|were)\b',
            r'\bwhere\s+(?:was|did|did\s+it)\b',
            r'\bwhat\s+is\s+the\s+name\b',
            r'\bwhat\s+(?:role|contributions?)\s+(?:does|did|has|have)\b',
            r'\bwhich\s+(?:paper|author|method|model)\b',
        ]

        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                return True

        return False

    async def retrieve(self, query: str, top_k: int = 20) -> QueryResult:
        """使用 BM25 检索"""
        try:
            # 确保索引已构建
            if not await self._ensure_index():
                return QueryResult(nodes=[], scores=[])

            # 分词查询
            tokenized_query = self._tokenize(query)

            if not tokenized_query:
                return QueryResult(nodes=[], scores=[])

            # 计算 BM25 分数
            assert self._bm25 is not None
            scores = self._bm25.get_scores(tokenized_query)

            # 构建结果列表
            results = []
            for i, score in enumerate(scores):
                if score > 0:  # 只返回有匹配的
                    results.append({
                        "text": self._corpus_texts[i],
                        "metadata": self._corpus_metadata[i],
                        "score": float(score)
                    })

            # 按分数降序排列
            results.sort(key=lambda x: x["score"], reverse=True)

            nodes = []
            scores_list = []
            for item in results[:top_k]:
                nodes.append(Node(
                    text=item["text"],
                    metadata=item.get("metadata", {})
                ))
                scores_list.append(item.get("score", 0.0))

            return QueryResult(nodes=nodes, scores=scores_list)

        except Exception as e:
            logger.error(f"[BM25Retriever] BM25 检索失败: {e}")
            return QueryResult(nodes=[], scores=[])

    def refresh_index(self) -> None:
        """刷新 BM25 索引（如论文增删后）"""
        self._bm25 = None
        self._corpus_texts = []
        self._corpus_metadata = []
        logger.info("🔄 BM25 索引已重置，下次检索时将重新构建")


class MultiVectorReranker:
    """多向量 ColBERT 式 Reranker"""
    """
    ColBERT 式多向量 reranker

    使用预存储的 per-token vectors 做 MaxSim reranking
    """

    def __init__(self, embed_provider: Any):
        self._embed_provider = embed_provider
        self._embedding_model = None
        self._colbert_storage = None

    def set_colbert_storage(self, storage: Any) -> None:
        """设置 ColBERT 存储（延迟绑定）"""
        self._colbert_storage = storage

    def _get_embedding_model(self):
        """获取 embedding 模型（优先已初始化的 FlagEmbedding，降级 Unsloth）"""
        if self._embedding_model is None:
            flag = get_flag_model()
            if flag._initialized:
                self._embedding_model = flag
            else:
                self._embedding_model = get_embedding_model()
            if self._embedding_model is None:
                logger.error("[PaperRAG] embedding 模型不可用（FlagEmbedding 和 Unsloth 均返回 None）")
        return self._embedding_model

    async def rerank(
        self,
        query: str,
        nodes: List[Node],
        scores: List[float],
        top_k: int = 5,
    ) -> QueryResult:
        """
        使用 ColBERT 风格 reranking

        优先使用预存储的 ColBERT vectors；若无存储，则实时计算
        """
        if not nodes:
            return QueryResult(nodes=[], scores=[])

        try:
            # 提取文档文本和 chunk_id
            chunk_ids = [self._resolve_node_chunk_id(node) for node in nodes]
            has_stable_chunk_ids = any(chunk_ids)

            # 尝试使用预存储的 ColBERT vectors
            if self._colbert_storage is not None and self._colbert_storage.is_loaded and has_stable_chunk_ids:
                reranked = self._colbert_rerank_stored(
                    query, nodes, chunk_ids, top_k=len(nodes)
                )
                if not reranked:
                    logger.warning("[MultiVectorReranker] 预存 ColBERT 向量未匹配候选 chunks，保留原始顺序")
                    reranked = [(i, scores[i] if i < len(scores) else 0.0) for i in range(len(nodes))]
            else:
                if self._colbert_storage is not None and self._colbert_storage.is_loaded and not has_stable_chunk_ids:
                    logger.info("[MultiVectorReranker] 候选无稳定 chunk_id，跳过实时 ColBERT rerank，保留原始顺序")
                    reranked = [(i, scores[i] if i < len(scores) else 0.0) for i in range(len(nodes))]
                else:
                    model = self._get_embedding_model()
                    doc_texts = [node.text for node in nodes]
                    # Fallback: 实时计算（标准 ColBERT 做法）
                    reranked = model.colbert_rerank(query, doc_texts, top_k=len(nodes))

            # 构建新的结果
            reranked_nodes = []
            reranked_scores = []

            for idx, score in reranked:
                if idx < len(nodes):
                    reranked_nodes.append(nodes[idx])
                    reranked_scores.append(score)

            if reranked_scores:
                logger.info(
                    f"[MultiVectorReranker] rerank完成: candidates={len(nodes)}, "
                    f"returned={min(top_k, len(reranked_nodes))}, "
                    f"score_range={max(reranked_scores):.6f}..{min(reranked_scores):.6f}"
                )

            return QueryResult(
                nodes=reranked_nodes[:top_k],
                scores=reranked_scores[:top_k]
            )

        except Exception as e:
            logger.error(f"[MultiVectorReranker] reranking 失败: {e}")
            return QueryResult(nodes=nodes[:top_k], scores=scores[:top_k])

    def _resolve_node_chunk_id(self, node: Node) -> Optional[str]:
        """Resolve the stable ColBERT chunk id for a retrieved node."""
        metadata = node.metadata or {}
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)

        file_path = metadata.get("file_path") or metadata.get("source_path")
        chunk_index = metadata.get("chunk_index")
        if file_path is None or chunk_index is None:
            return None

        return f"{file_path}_{chunk_index}"

    def _colbert_rerank_stored(
        self,
        query: str,
        nodes: List[Node],
        chunk_ids: List[Optional[str]],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        使用预存储的 ColBERT vectors 做 MaxSim reranking

        标准 ColBERT 做法：
        1. query tokens → encode → colbert projection
        2. MaxSim: 每个 query token 与所有 doc tokens 做点积，取最大值
        3. 累加所有 query tokens 的 max-sim 分数
        """

        model = self._get_embedding_model()

        # 1. 计算 query 的 ColBERT vectors
        query_vectors = model.get_multi_vector(query)  # (M, 1024)
        if not query_vectors:
            return [(i, 0.0) for i in range(len(nodes))]
        query_arr = np.array(query_vectors, dtype=np.float32)

        storage = self._colbert_storage
        assert storage is not None

        # 3. 计算每个 chunk 的 MaxSim 分数
        chunk_maxsims: Dict[int, float] = {}
        matched_count = 0
        for node_idx, chunk_id in enumerate(chunk_ids):
            if not chunk_id:
                continue
            chunk_idx = storage.find_chunk_idx(chunk_id)
            if chunk_idx is None:
                continue

            # MaxSim 计算
            doc_vectors = storage.get_chunk_token_vectors(chunk_idx)
            if doc_vectors is None:
                continue
            doc_arr = np.array(doc_vectors, dtype=np.float32)
            # MaxSim: (M, 1024) @ (1024, N) = (M, N) → row sums → max
            sim_matrix = np.dot(query_arr, doc_arr.T)
            max_sim = float(np.max(sim_matrix, axis=1).sum())
            chunk_maxsims[node_idx] = max_sim
            matched_count += 1

        if matched_count == 0:
            logger.warning(
                f"[MultiVectorReranker] 预存 ColBERT chunk_id 匹配失败: "
                f"candidates={len(nodes)}, storage_chunks={len(storage._id_mapping)}"
            )
            return []

        # 4. 排序
        sorted_indices = sorted(chunk_maxsims.keys(), key=lambda i: chunk_maxsims[i], reverse=True)
        logger.info(
            f"[MultiVectorReranker] 预存ColBERT匹配: matched={matched_count}/{len(nodes)}, "
            f"storage_chunks={len(storage._id_mapping)}"
        )
        return [(i, chunk_maxsims[i]) for i in sorted_indices[:top_k]]


class HybridRetriever(BaseRetriever):
    """
    混合检索器：稠密向量 + 稀疏权重 + BM25精确匹配 + RRF 融合 + ColBERT Reranking

    流程：
    1. 并行执行向量搜索和稀疏权重检索
    2. 如果查询需要精确匹配（专有名词、作者名等），同时执行 BM25 检索
    3. 使用 Reciprocal Rank Fusion (RRF) 合并多路结果
    4. 可选：ColBERT 多向量 reranking
    """

    def __init__(
        self,
        index_manager: HybridIndexManager,
        embed_provider: Any,
        enable_sparse_retrieval: bool = True,
        sparse_top_k: int = 20,
        vector_top_k: int = 50,
        alpha: float = 0.5,
        rrf_k: int = 60,
        enable_reranking: bool = False,
        rerank_top_k: int = 5,
        enable_bm25: bool = True,
        bm25_top_k: int = 20,
        graph_retriever: Any = None,
    ):
        super().__init__(index_manager, embed_provider)
        self._enable_sparse_retrieval = enable_sparse_retrieval
        self._sparse_top_k = sparse_top_k
        self._vector_top_k = vector_top_k
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._enable_reranking = enable_reranking
        self._rerank_top_k = rerank_top_k
        self._enable_bm25 = enable_bm25
        self._bm25_top_k = bm25_top_k
        # Graph retriever is set externally by HybridRAGEngine for paper-level recall
        self._graph_retriever = graph_retriever
        self._sparse_retriever = SparseRetriever(index_manager, embed_provider)
        self._bm25_retriever = BM25Retriever(index_manager)
        self._reranker = MultiVectorReranker(embed_provider)

    async def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        """混合检索：向量 + 稀疏权重 + BM25（按需）+ RRF 融合"""

        try:
            logger.info(
                f"[HybridRetriever] 单阶段检索开始: top_k={top_k}, "
                f"vector_top_k={self._vector_top_k}, sparse={self._enable_sparse_retrieval}, "
                f"sparse_top_k={self._sparse_top_k}, bm25={self._enable_bm25}, "
                f"reranking={self._enable_reranking}"
            )

            # 1. 稠密向量搜索
            query_embedding = await self._embed_provider.get_text_embedding(query)
            vector_results = await self._index_manager.search(
                query_embedding=query_embedding,
                top_k=self._vector_top_k
            )
            logger.info(f"[HybridRetriever] dense召回: {len(vector_results)} chunks")

            # 2. 稀疏权重检索
            sparse_dict: Dict[str, float] = {}
            if self._enable_sparse_retrieval:
                sparse_results = await self._sparse_retriever.retrieve(
                    query, top_k=self._sparse_top_k
                )
                sparse_dict = {
                    node.text: score
                    for node, score in zip(sparse_results.nodes, sparse_results.scores)
                }
                logger.info(f"[HybridRetriever] sparse召回: {len(sparse_dict)} chunks")
            else:
                logger.info("[HybridRetriever] sparse召回已关闭")

            # 3. BM25 精确匹配（仅当查询需要精确匹配时）
            bm25_dict: Dict[str, float] = {}
            use_bm25 = self._enable_bm25 and self._bm25_retriever.is_exact_match_query(query)
            if use_bm25:
                logger.info(f"[HybridRetriever] BM25精确匹配召回启动: top_k={self._bm25_top_k}")
                bm25_result = await self._bm25_retriever.retrieve(
                    query, top_k=self._bm25_top_k
                )
                bm25_dict = {
                    node.text: score
                    for node, score in zip(bm25_result.nodes, bm25_result.scores)
                }
                logger.info(f"[HybridRetriever] BM25召回: {len(bm25_dict)} chunks")
            elif self._enable_bm25:
                logger.info("[HybridRetriever] BM25未触发：查询未判定为精确匹配型")
            else:
                logger.info("[HybridRetriever] BM25已关闭")

            # 4. RRF 融合
            fused = self._rrf_fusion(
                vector_results=vector_results,
                sparse_results=sparse_dict,
                bm25_results=bm25_dict if bm25_dict else None,
                top_k=top_k * 2 if self._enable_reranking else top_k,
            )
            if fused:
                logger.info(
                    f"[HybridRetriever] RRF融合完成: {len(fused)} candidates, "
                    f"score_range={fused[0].get('fused_score', 0.0):.6f}..{fused[-1].get('fused_score', 0.0):.6f}"
                )
            else:
                logger.info("[HybridRetriever] RRF融合无结果")

            # 构建 Node 列表
            nodes = []
            scores = []
            for item in fused:
                nodes.append(Node(
                    text=item["text"],
                    metadata=item.get("metadata", {})
                ))
                scores.append(item.get("fused_score", 0.0))

            # 5. 可选：ColBERT reranking
            if self._enable_reranking and len(nodes) > top_k:
                reranked = await self._reranker.rerank(
                    query, nodes, scores, top_k=top_k
                )
                if reranked.scores:
                    logger.info(
                        f"[HybridRetriever] ColBERT reranking完成: {len(nodes)} -> {len(reranked.nodes)}, "
                        f"score_range={max(reranked.scores):.6f}..{min(reranked.scores):.6f}"
                    )
                else:
                    logger.info(f"[HybridRetriever] ColBERT reranking完成: {len(nodes)} -> {len(reranked.nodes)}")
                if reranked.scores and all(abs(score) < 1e-12 for score in reranked.scores):
                    logger.warning(
                        "[HybridRetriever] ColBERT reranking 分数全为0，通常表示候选 chunk_id 与预存向量未匹配"
                    )
                return reranked

            result = QueryResult(nodes=nodes[:top_k], scores=scores[:top_k])
            if result.scores:
                logger.info(
                    f"[HybridRetriever] 单阶段检索完成: results={len(result.nodes)}, "
                    f"score_range={max(result.scores):.6f}..{min(result.scores):.6f}"
                )
            else:
                logger.info("[HybridRetriever] 单阶段检索完成: results=0")

            return result

        except Exception as e:
            logger.error(f"[HybridRetriever] 混合检索失败: {e}")
            if _is_mps_oom_error(e):
                _clear_accelerator_cache()
                raise RuntimeError(
                    "MPS 内存不足，已停止单阶段降级以避免继续放大内存压力。"
                    "请重启 AstrBot 让 PYTORCH_MPS_HIGH_WATERMARK_RATIO 生效，"
                    "或临时将 unsloth.device 设为 cpu / 关闭 enable_multi_vector_rerank。"
                ) from e
            # 降级为纯向量检索
            fallback = VectorRetriever(self._index_manager, self._embed_provider)
            return await fallback.retrieve(query, top_k)

    def _rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        sparse_results: Dict[str, float],
        bm25_results: Dict[str, float] | None = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion) 分数融合

        通道：vector + sparse + BM25(可选)
        """
        has_bm25 = bm25_results is not None and len(bm25_results) > 0

        # text -> vector_rank
        vector_rank_map: Dict[str, int] = {}
        for i, item in enumerate(vector_results):
            vector_rank_map[item["text"]] = i + 1

        # sparse rank
        sorted_sparse = sorted(sparse_results.items(), key=lambda x: x[1], reverse=True)
        sparse_rank_map: Dict[str, int] = {}
        for i, (text, _) in enumerate(sorted_sparse):
            sparse_rank_map[text] = i + 1

        # BM25 rank
        bm25_rank_map: Dict[str, int] = {}
        sorted_bm25: List[Tuple[str, float]] = []
        if has_bm25 and bm25_results:
            sorted_bm25 = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)
            for i, (text, _) in enumerate(sorted_bm25):
                bm25_rank_map[text] = i + 1

        all_texts = set(vector_rank_map.keys()) | set(sparse_rank_map.keys())
        if has_bm25:
            all_texts |= set(bm25_rank_map.keys())

        # 权重分配
        if has_bm25:
            alpha_v = 0.4
            alpha_s = 0.4
            alpha_b = 0.2
        else:
            alpha_v = self._alpha
            alpha_s = 1 - self._alpha
            alpha_b = 0.0

        n_vector = len(vector_results)
        n_sparse = len(sorted_sparse)
        n_bm25 = len(sorted_bm25) if has_bm25 else 0

        rrf_scores: Dict[str, float] = {}
        for text in all_texts:
            v_rank = vector_rank_map.get(text, n_vector + 1)
            s_rank = sparse_rank_map.get(text, n_sparse + 1)

            v_rrf = alpha_v * (1.0 / (self._rrf_k + v_rank)) if alpha_v > 0 else 0
            s_rrf = alpha_s * (1.0 / (self._rrf_k + s_rank)) if alpha_s > 0 else 0
            rrf_scores[text] = v_rrf + s_rrf

            if has_bm25:
                b_rank = bm25_rank_map.get(text, n_bm25 + 1)
                rrf_scores[text] += alpha_b * (1.0 / (self._rrf_k + b_rank))

        sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # metadata 合并
        text_to_metadata: Dict[str, Dict] = {}
        for item in vector_results:
            text_to_metadata[item["text"]] = item.get("metadata", {})

        fused = []
        for text, rrf_score in sorted_texts[:top_k]:
            if text not in text_to_metadata:
                text_to_metadata[text] = {}
            fused.append({
                "text": text,
                "metadata": text_to_metadata.get(text, {}),
                "score": rrf_score,
                "fused_score": rrf_score
            })

        return fused


FACT_TYPE_PATTERNS = [
    r'\bwhat\s+is\b', r'\bwhat\s+are\b', r'\bwhat\s+does\b', r'\bwhat\s+did\b',
    r'\bmean\b.*\bby\b', r'\bmeans\b', r'\brefer\s+to\b',
    r'\bdefine\b', r'\bdefinition\b',
    r'\bhow\s+do\b', r'\bhow\s+does\b',
    r'是什么', r'是指', r'定义是', r'含义是', r'意思是',
    r'什么叫', r'哪几个', r'有哪些', r'都有哪些',
]

ENTITY_NEED_EXACT_MATCH = [
    'freiburg1', 'CoRGS', 'DPT-Large', 'RegNeRF', 'NeRF', '3DGS', 'GS', 'SfM',
    'SAM', 'SAM 2', 'ViT', 'CNN', 'LoRA',
    'CVPR', 'ICCV', 'ECCV', 'NeurIPS', 'ICML',
    'Mip-Splatting', 'InstantSplat', 'DUSt3R', 'MASt3R',
]


def classify_query_complexity(query: str) -> str:
    """查询复杂度分类"""
    query_lower = query.lower()

    if len(query.split()) <= 8:
        if any(kw in query_lower for kw in [
            '是什么', '什么是', 'what is', 'what are',
            'means', 'meaning of', '定义是', '含义是',
            '哪个', 'what does', "what's"
        ]):
            return "simple_direct"

    if any(kw in query_lower for kw in [
        '比较', 'difference', 'compared', ' vs ', ' versus ',
        '与...区别', '和...区别', '哪个更好', '优势', '劣势',
    ]):
        return "multi_hop"

    if any(kw in query_lower for kw in [
        '如何', 'how does', 'how do', 'why', '为什么',
        '原因', '分析', '原因', 'explain', '原因分析',
        '原理', '机制'
    ]):
        return "complex"

    return "standard"


def detect_query_type(query: str) -> Dict[str, Any]:
    """检测查询类型"""
    query_lower = query.lower()
    is_fact_type = False

    for pattern in FACT_TYPE_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            is_fact_type = True
            break

    query_subtype = "general"
    if is_fact_type:
        if any(kw in query_lower for kw in ['what is', 'what are', '是什么', '是指', 'define', 'definition']):
            query_subtype = "definition"
        elif any(kw in query_lower for kw in ['what', 'which', '哪些', '哪个']):
            query_subtype = "entity_list"
        elif any(kw in query_lower for kw in ['how does', 'how do', '如何', '怎么', '过程']):
            query_subtype = "procedure"

    core_entities = []
    for entity in ENTITY_NEED_EXACT_MATCH:
        if entity.lower() in query_lower:
            core_entities.append(entity)

    needs_exact_match = len(core_entities) > 0

    return {
        "is_fact_type": is_fact_type,
        "query_subtype": query_subtype,
        "core_entities": core_entities,
        "needs_exact_match": needs_exact_match
    }


def filter_by_entity_match(
    results: List[Dict[str, Any]],
    core_entities: List[str],
    score_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """实体匹配过滤"""
    if not core_entities:
        return results

    filtered = []
    has_entity_match = False

    for r in results:
        text_lower = r.get("text", "").lower()
        score = r.get("score", 0.0)

        entity_matched = any(
            entity.lower() in text_lower for entity in core_entities
        )

        if entity_matched:
            has_entity_match = True
            filtered.append(r)
        elif score > score_threshold:
            r_copy = r.copy()
            r_copy["_entity_matched"] = False
            filtered.append(r_copy)

    if not has_entity_match and filtered:
        logger.debug(f"[EntityFilter] 警告：没有任何结果包含实体 {core_entities}")

    return filtered


# ============================================================================
# CRAG (Corrective RAG) - 保留原有逻辑
# ============================================================================

class CragEvaluator:
    """CRAG 检索质量评估器"""

    def __init__(self, llm_provider: Any = None):
        self._llm_provider = llm_provider

    async def evaluate_retrieval_quality(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """评估检索质量"""
        if not results:
            return {
                "score": 0.0,
                "level": "low",
                "reasoning": "无检索结果"
            }

        if self._llm_provider:
            try:
                return await self._evaluate_by_llm(query, results)
            except Exception as e:
                logger.debug(f"[CRAG] LLM评估失败，降级为规则评估: {e}")

        return self._evaluate_by_rules(query, results)

    async def _evaluate_by_llm(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """使用 LLM 评估检索质量"""
        top_results = results[:3]
        context_text = "\n\n".join([
            f"[文档 {i+1}] (相关性分数：{r.get('score', 0):.3f})\n{r.get('text', '')[:300]}..."
            for i, r in enumerate(top_results)
        ])

        prompt = f"""评估检索结果是否能回答以下查询：

【查询】
{query}

【检索结果】
{context_text}

请严格评估并返回 JSON：
{{"score": 0.0-1.0, "level": "high/medium/low", "reasoning": "理由", "missing_aspects": "缺失方面"}}"""

        try:
            response = await self._llm_provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.1,
            )

            from provider.llm_utils import extract_text_from_response
            response_text = extract_text_from_response(response)

            import json, re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                raw_json = json_match.group(0)
                # Sanitize common LLM JSON issues: Chinese quotes, unescaped newlines
                raw_json = raw_json.replace('“', '"').replace('”', '"')
                raw_json = raw_json.replace('‘', "'").replace('’', "'")
                raw_json = re.sub(r'[\x00-\x1f]', ' ', raw_json)
                try:
                    result = json.loads(raw_json)
                except json.JSONDecodeError:
                    # Fallback: extract key values with regex
                    result = {}
                    score_m = re.search(r'"?score"?\s*:\s*([0-9.]+)', raw_json)
                    level_m = re.search(r'"?level"?\s*:\s*"?(\w+)"?', raw_json)
                    if score_m:
                        result["score"] = float(score_m.group(1))
                    if level_m:
                        result["level"] = level_m.group(1)
                    if not result:
                        logger.error(f"[CRAG] LLM响应格式完全错误，JSON和正则提取均失败。原始响应: {raw_json}")
                        return self._evaluate_by_rules(query, results)
                score = float(result.get("score", 0.5))
                level = result.get("level", "medium")
                if level not in ["high", "medium", "low"]:
                    level = "medium" if score >= 0.3 else "low"
                return {
                    "score": min(score, 1.0),
                    "level": level,
                    "reasoning": result.get("reasoning", ""),
                    "missing_aspects": result.get("missing_aspects", "")
                }
        except Exception as e:
            logger.warning(f"[CRAG] LLM评估异常: {e}")

        return self._evaluate_by_rules(query, results)

    def _evaluate_by_rules(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """基于规则评估"""
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        if not results:
            return {"score": 0.0, "level": "low", "reasoning": "无检索结果", "missing_aspects": ""}

        # Normalize scores: RRF scores are 0.001-0.02, cosine similarity is 0-1
        max_raw_score = max(r.get("score", 0.0) for r in results)
        if max_raw_score > 0 and max_raw_score < 0.1:
            normalized = [r.get("score", 0.0) / max_raw_score for r in results]
        else:
            normalized = [min(r.get("score", 0.0), 1.0) for r in results]
        avg_score = sum(normalized) / len(normalized)

        coverage_scores = []
        for r in results[:3]:
            doc_text = r.get("text", "").lower()
            doc_terms = set(doc_text.split())
            coverage = len(query_terms & doc_terms) / max(len(query_terms), 1)
            coverage_scores.append(coverage)

        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        score = 0.3 * min(avg_score, 1.0) + 0.5 * avg_coverage + 0.2 * (coverage_scores[0] if coverage_scores else 0.5)

        top1_coverage = coverage_scores[0] if coverage_scores else 0.0
        if top1_coverage < 0.1 and avg_coverage < 0.15:
            score *= 0.5
            reasoning = "Top结果与查询相关性低"
        elif score > 0.6:
            reasoning = "检索结果与查询高度相关"
        elif score > 0.4:
            reasoning = "检索结果与查询中等相关"
        else:
            reasoning = "检索结果与查询相关性低"

        level = "high" if score >= 0.6 else "medium" if score >= 0.4 else "low"

        return {"score": min(score, 1.0), "level": level, "reasoning": reasoning, "missing_aspects": ""}


class CragCorrector:
    """CRAG 修正策略执行器"""

    def __init__(self, index_manager: Any, embed_provider: Any, llm_provider: Any = None):
        self._index_manager = index_manager
        self._embed_provider = embed_provider
        self._llm_provider = llm_provider

    async def correct(
        self,
        query: str,
        level: str,
        original_results: List[Dict[str, Any]],
        missing_aspects: str = "",
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """根据评估等级执行修正策略"""
        if level == "high":
            return original_results

        if level == "medium":
            return await self._retry_with_rewrite(query, original_results, paper_ids=paper_ids)

        return await self._fulltext_fallback(query, paper_ids=paper_ids)

    async def _retry_with_rewrite(
        self,
        query: str,
        original_results: List[Dict[str, Any]],
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """使用重写查询重试"""
        if self._llm_provider is None:
            logger.info("[CRAG] 查询重写跳过: LLM Provider 不可用")
            return original_results

        rewrite_prompt = f"""将以下查询重写为不同的表述：

原始查询：{query}

请生成1个不同的表述，直接输出："""

        try:
            response = await self._llm_provider.text_chat(
                prompt=rewrite_prompt,
                contexts=[],
                temperature=0.3,
            )

            from provider.llm_utils import extract_text_from_response
            rewritten = extract_text_from_response(response).strip()

            if rewritten:
                query_embedding = await self._embed_provider.get_text_embedding(rewritten)
                if paper_ids:
                    new_results = await self._index_manager.search_with_paper_filter(
                        query_embedding=query_embedding,
                        paper_ids=paper_ids,
                        top_k=10
                    )
                else:
                    new_results = await self._index_manager.search(
                        query_embedding=query_embedding,
                        top_k=10
                    )
                if new_results:
                    logger.info(f"[CRAG] 查询重写: '{query}' -> '{rewritten}'")
                    return new_results
        except Exception as e:
            logger.warning(f"[CRAG] 查询重写失败: {e}")

        return original_results

    async def _fulltext_fallback(self, query: str, paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """全文检索降级"""
        try:
            query_embedding = await self._embed_provider.get_text_embedding(query)
            if paper_ids:
                results = await self._index_manager.search_with_paper_filter(
                    query_embedding=query_embedding,
                    paper_ids=paper_ids,
                    top_k=20
                )
                logger.info(f"[CRAG] 论文内扩大检索，返回 {len(results)} 个结果")
            else:
                results = await self._index_manager.search(
                    query_embedding=query_embedding,
                    top_k=20
                )
                logger.info(f"[CRAG] 降级到全文检索，返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"[CRAG] 全文检索失败: {e}")
            return []


# ============================================================================
# HybridRAGEngine 主类
# ============================================================================

class HybridRAGEngine:
    """
    混合RAG引擎（BGE-M3 稀疏权重 + 多向量版）

    特性：
    - PDF解析（多模态）
    - Node结构存储
    - 混合检索（稀疏权重 + 稠密向量 + RRF）
    - ColBERT 多向量 reranking
    - LLM生成（支持多模态）
    """

    def __init__(self, config: RAGConfig, context):
        """
        初始化混合RAG引擎

        Args:
            config: RAG配置
            context: AstrBot上下文
        """
        self.config = config
        self.context = context

        # 延迟初始化
        self._parser: HybridPDFParser = cast(HybridPDFParser, None)
        self._index_manager: HybridIndexManager = cast(HybridIndexManager, None)
        self._abstract_manager: Optional[Any] = None
        self._embed_provider: Union[FlagEmbeddingProvider, UnslothEmbeddingProvider, AstrBotEmbeddingProvider, None] = None
        self._llm_client: Any = cast(Any, None)
        self._retriever: Union[VectorRetriever, HybridRetriever] = cast(Any, None)
        self._colbert_storage: Any = cast(Any, None)
        self._abstract_colbert_storage: Any = cast(Any, None)

        # 初始化标志
        self._parser_initialized = False
        self._index_initialized = False
        self._abstract_initialized = False
        self._embed_provider_initialized = False
        self._llm_initialized = False
        self._retriever_initialized = False

    def _ensure_parser_initialized(self) -> HybridPDFParser:
        """确保解析器已初始化"""
        if self._parser_initialized:
            return self._parser

        self._parser = HybridPDFParser(
            enable_multimodal=self.config.enable_multimodal,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._parser_initialized = True
        logger.info("✅ HybridPDFParser初始化完成")
        return self._parser

    async def _ensure_embed_provider_initialized(self) -> Union[FlagEmbeddingProvider, UnslothEmbeddingProvider, AstrBotEmbeddingProvider]:
        """确保Embedding Provider已初始化"""
        if self._embed_provider_initialized:
            assert self._embed_provider is not None
            return self._embed_provider

        try:
            self._embed_provider = create_embedding_provider(
                mode=self.config.embedding_mode,
                context=self.context,
                provider_id=self.config.embedding_provider_id,
                embed_batch_size=10,
                **(self.config.unsloth_config or {}),
            )

            if hasattr(self._embed_provider, 'initialize'):
                await self._embed_provider.initialize()
            else:
                logger.warning("[PaperRAG] Embedding provider 无 initialize 方法，跳过初始化")

            self._embed_provider_initialized = True
            logger.info(f"✅ Embedding Provider初始化完成: {self.config.embedding_mode}")
            assert self._embed_provider is not None
            return self._embed_provider

        except Exception as e:
            logger.error(f"❌ Embedding Provider初始化失败: {e}")
            raise

    async def _ensure_llm_provider_initialized(self) -> Any:
        """确保 LLM Provider 已初始化（用于噪声过滤），复用 _ensure_llm_initialized 的统一解析。"""
        if getattr(self, '_llm_provider', None) is not None:
            return self._llm_provider

        try:
            self._llm_provider = await self._ensure_llm_initialized()
            logger.info("✅ LLM Provider 初始化完成（噪声过滤）")
            return self._llm_provider
        except Exception as e:
            logger.warning(f"⚠️ LLM Provider 初始化失败，过滤将被跳过: {e}")
            self._llm_provider = None
            return None

    async def _ensure_llm_initialized(self) -> Any:
        """确保LLM Provider已初始化（单例，失败后标记避免重复加载）"""
        if self._llm_initialized:
            assert self._llm_client is not None
            return self._llm_client

        # 如果之前已尝试过且失败，不再重复
        if getattr(self, '_llm_init_failed', False):
            raise ValueError("LLM Provider 初始化之前已失败，不再重试")

        provider_id = self.config.text_provider_id
        if not provider_id:
            # text_provider_id 未配置时，默认使用本地VLM
            try:
                from provider.llama_cpp_vlm import get_llama_cpp_vlm_provider
                vlm_provider = get_llama_cpp_vlm_provider()
                if vlm_provider and vlm_provider._initialized:
                    self._llm_client = vlm_provider
                    self._llm_initialized = True
                    logger.info("✅ 使用本地VLM Provider (默认)")
                    return self._llm_client
                elif not vlm_provider or not vlm_provider._initialized:
                    logger.warning("⚠️ 本地VLM未初始化，尝试初始化...")
                    # 使用 _ensure_llm_provider_initialized 初始化
                    initialized = await self._ensure_llm_provider_initialized()
                    if initialized:
                        self._llm_client = initialized
                        self._llm_initialized = True
                        logger.info("✅ 使用本地VLM Provider (初始化后)")
                        return self._llm_client
            except Exception as e:
                logger.warning(f"⚠️ 获取本地VLM失败: {e}")

            # 本地VLM不可用，使用当前会话的云端Provider
            try:
                if self.context is not None:
                    self._llm_client = self.context.get_using_provider()
                    if self._llm_client:
                        logger.info("✅ 使用当前会话的 LLM Provider (云端备用)")
                        self._llm_initialized = True
                        return self._llm_client
            except Exception as e:
                logger.warning(f"⚠️ 获取云端Provider失败: {e}")

            raise ValueError(
                "无法获取LLM Provider。"
                "请在插件配置中设置 text_provider_id 或确保本地VLM可用。"
            )

        # 从 context 获取指定的 provider
        try:
            provider_manager = getattr(self.context, "provider_manager", None)
            if provider_manager:
                inst_map = getattr(provider_manager, "inst_map", None)
                if isinstance(inst_map, dict):
                    self._llm_client = inst_map.get(provider_id)
                    if self._llm_client:
                        logger.info(f"✅ 使用 LLM Provider: {provider_id}")
                        self._llm_initialized = True
                        return self._llm_client

            # 兼容旧版本
            self._llm_client = self.context.get_provider_by_id(provider_id)
            if self._llm_client:
                logger.info(f"✅ 使用 LLM Provider: {provider_id}")
                self._llm_initialized = True
                return self._llm_client

            raise ValueError(f"无法找到 Provider: {provider_id}")
        except Exception as e:
            logger.error(f"❌ LLM Provider初始化失败: {e}")
            raise

    NOISE_FILTER_PROMPT = """判断以下文本是否为无意义噪声（适合从 RAG 移除）：

噪声类型：
- 参考文献列表 [1] xxx
- 纯表格单元格（无完整语义）
- 机构 affiliation 信息
- 只有符号/数字/标点的无意义行
- 残缺公式片段或符号串
- 纯页码、页眉、页脚

文本：
{text}

只输出 JSON（不要其他内容）：{{"is_noise": true/false, "reason": "原因"}}
"""

    async def _filter_noise_nodes(
        self,
        nodes: List[Any],
        max_workers: int = 1,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        使用本地 LLM 过滤噪声 chunks

        Returns:
            (filtered_nodes, noise_report) — noise_report 包含被过滤的 chunks
        """
        llm = await self._ensure_llm_provider_initialized()
        if llm is None:
            return nodes, []

        async def classify_one(node: Any) -> Optional[Dict[str, Any]]:
            text = node.text
            if not text or len(text.strip()) < 5:
                return {"node": node, "is_noise": True, "reason": "文本过短"}
            try:
                prompt = self.NOISE_FILTER_PROMPT.format(text=text)
                response = await llm.text_chat(
                    prompt=prompt,
                    contexts=[],
                    temperature=0.1,
                )
                from provider.llm_utils import extract_text_from_response
                content = extract_text_from_response(response)
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    is_noise = result.get("is_noise", False)
                    reason = result.get("reason", "")
                    if is_noise:
                        return {"node": node, "is_noise": True, "reason": reason or "LLM判定"}
                return None
            except Exception as e:
                logger.debug(f"噪声分类失败: {e}")
                return None

        # 并发限制
        semaphore = asyncio.Semaphore(max_workers)
        async def bounded_classify(node: Any) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await classify_one(node)

        tasks = [bounded_classify(n) for n in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        filtered = []
        noise_report = []
        for node, result in zip(nodes, results):
            if isinstance(result, Exception):
                filtered.append(node)
                continue
            if result is None:
                filtered.append(node)
                continue
            r = cast(Dict[str, Any], result)
            if not r.get("is_noise"):
                filtered.append(node)
            else:
                noise_report.append({
                    "text": node.text,
                    "reason": r.get("reason", ""),
                    "metadata": node.metadata,
                })

        if noise_report:
            logger.info(f"[噪声过滤] 过滤了 {len(noise_report)}/{len(nodes)} 个 chunks")
        return filtered, noise_report

    def _ensure_colbert_storage_initialized(self) -> Any:
        """确保 ColBERT 存储已初始化"""
        if self._colbert_storage is not None:
            return self._colbert_storage

        data_dir = Path(__file__).parent.parent / "data" / "colbert_chunks"
        self._colbert_storage = ColBERTStorage(str(data_dir))

        if self._colbert_storage.load():
            logger.info(f"[ColBERTStorage] 已加载 {len(self._colbert_storage)} 个 chunks")
        else:
            logger.info("[ColBERTStorage] 未找到已有存储，将从头构建")

        return self._colbert_storage

    def _ensure_abstract_colbert_storage_initialized(self) -> Any:
        """确保摘要 ColBERT 存储已初始化"""
        if self._abstract_colbert_storage is not None:
            return self._abstract_colbert_storage

        data_dir = Path(__file__).parent.parent / "data" / "colbert_abstracts"
        self._abstract_colbert_storage = ColBERTStorage(str(data_dir))

        if self._abstract_colbert_storage.load():
            logger.info(f"[ColBERTStorage-Abstract] 已加载 {len(self._abstract_colbert_storage)} 个摘要")
        else:
            logger.info("[ColBERTStorage-Abstract] 未找到已有存储，将从头构建")

        return self._abstract_colbert_storage

    def _ensure_index_manager_initialized(self) -> HybridIndexManager:
        """确保索引管理器已初始化"""
        if self._index_initialized:
            assert self._index_manager is not None
            return self._index_manager

        mode = self.config.get_connection_mode()

        if mode == 'lite':
            lite_path = self.config.milvus_lite_path
            if not lite_path:
                # 设置默认路径为插件根目录的 data/milvus_papers.db
                lite_path = str(Path(__file__).parent.parent / "data" / "milvus_papers.db")
            uri: Optional[str] = None
        else:
            lite_path = None
            uri = self.config.address

        self._index_manager = HybridIndexManager(
            alias="paperrag_hybrid",
            lite_path=lite_path,
            uri=uri,
            collection_name=self.config.collection_name,
            embed_dim=self.config.embed_dim,
            authentication=self.config.authentication,
            db_name=self.config.db_name,
            hybrid_search=False
        )
        self._index_initialized = True
        logger.info(f"✅ HybridIndexManager初始化完成 (mode={mode}, collection={self.config.collection_name})")
        assert self._index_manager is not None
        return self._index_manager

    async def _ensure_abstract_manager_initialized(self) -> Any:
        """确保摘要索引管理器已初始化"""
        if self._abstract_initialized and self._abstract_manager is not None:
            return self._abstract_manager

        try:
            try:
                from .abstract_index import AbstractIndexManager
            except ImportError:
                from abstract_index import AbstractIndexManager

            milvus_uri = str(Path(__file__).parent.parent / "data" / "milvus_abstracts.db")

            self._abstract_manager = AbstractIndexManager(
                milvus_uri=milvus_uri,
                collection_name="paper_abstracts",
                embed_dim=self.config.embed_dim,
                core_api_key=getattr(self.config, "core_api_key", ""),
                use_arxiv_api=getattr(self.config, "use_arxiv_api", True),
            )

            embed_provider = await self._ensure_embed_provider_initialized()
            self._abstract_manager.set_embed_model(embed_provider)

            await self._abstract_manager.initialize()
            self._abstract_initialized = True
            logger.info("✅ AbstractIndexManager初始化完成")
            return self._abstract_manager

        except Exception as e:
            logger.warning(f"⚠️ AbstractIndexManager初始化失败: {e}")
            self._abstract_initialized = True
            self._abstract_manager = None
            return None

    async def _ensure_retriever_initialized(self) -> Union[VectorRetriever, HybridRetriever]:
        """确保检索器已初始化（异步以支持图谱检索器连接）"""
        if self._retriever_initialized:
            assert self._retriever is not None
            return self._retriever

        index_manager = self._ensure_index_manager_initialized()
        embed_provider = cast(Union[UnslothEmbeddingProvider, AstrBotEmbeddingProvider], self._embed_provider)

        if embed_provider is None:
            raise RuntimeError("Embed provider not initialized")

        # 使用稀疏+稠密混合检索（默认启用）
        self._retriever = HybridRetriever(
            index_manager=index_manager,
            embed_provider=embed_provider,
            enable_sparse_retrieval=self.config.enable_sparse_retrieval,
            sparse_top_k=self.config.sparse_top_k,
            vector_top_k=50,
            alpha=self.config.hybrid_alpha,
            rrf_k=self.config.hybrid_rrf_k,
            enable_reranking=self.config.enable_multi_vector_rerank,
            rerank_top_k=self.config.top_k,
            enable_bm25=self.config.enable_bm25,
            bm25_top_k=self.config.bm25_top_k,
            graph_retriever=None,
        )

        # 如果启用了 Graph RAG，连接图谱检索器（用于 paper-level 召回）
        if getattr(self.config, 'enable_graph_rag', False):
            try:
                try:
                    from ..graphrag.graph_rag_engine import GraphRAGConfig, create_graph_rag_engine
                except ImportError:
                    from graphrag.graph_rag_engine import GraphRAGConfig, create_graph_rag_engine
                graph_config = GraphRAGConfig.from_rag_config(self.config)
                graph_engine = create_graph_rag_engine(graph_config, self, self.context)
                pg_retriever = await graph_engine.get_retriever()
                if pg_retriever:
                    self._retriever._graph_retriever = pg_retriever
                    logger.info("[HybridRAGEngine] 图谱检索器已连接（paper-level 召回）")
            except Exception as e:
                logger.warning(f"[HybridRAGEngine] 图谱检索器连接失败: {e}")

        self._retriever_initialized = True
        self._bm25_retriever = self._retriever._bm25_retriever

        # 绑定 ColBERT 存储到 reranker
        if self.config.enable_multi_vector_rerank and hasattr(self._retriever, '_reranker'):
            colbert_storage = self._ensure_colbert_storage_initialized()
            self._retriever._reranker.set_colbert_storage(colbert_storage)
            logger.info("[ColBERTStorage] 已绑定到 MultiVectorReranker")

        # 绑定 ColBERT 存储到 BM25 retriever（用于刷新索引）
        if hasattr(self._retriever, '_bm25_retriever'):
            logger.info("[BM25Retriever] 已初始化")

        logger.info(
            f"✅ HybridRetriever初始化完成 "
            f"(sparse_top_k={self.config.sparse_top_k}, alpha={self.config.hybrid_alpha}, "
            f"bm25={self.config.enable_bm25}, bm25_top_k={self.config.bm25_top_k}, "
            f"reranking={self.config.enable_multi_vector_rerank})"
        )

        assert self._retriever is not None
        return self._retriever

    async def _get_bm25_retriever(self):
        """Lazily get BM25 retriever from HybridRetriever."""
        if hasattr(self, '_bm25_retriever') and self._bm25_retriever is not None:
            return self._bm25_retriever
        retriever = await self._ensure_retriever_initialized()
        return getattr(retriever, '_bm25_retriever', None)

    @staticmethod
    def _stage2_rrf_fusion(
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10,
        rrf_k: int = 60,
        alpha_dense: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Two-way RRF fusion for Stage 2: dense + BM25 within selected papers."""
        dense_rank = {r["text"]: i + 1 for i, r in enumerate(dense_results)}
        bm25_rank = {r["text"]: i + 1 for i, r in enumerate(bm25_results)}
        alpha_bm25 = 1.0 - alpha_dense

        all_texts = set(dense_rank.keys()) | set(bm25_rank.keys())
        n_dense = len(dense_results)
        n_bm25 = len(bm25_results)

        text_to_meta: Dict[str, Dict] = {}
        for r in dense_results:
            text_to_meta[r["text"]] = r.get("metadata", {})
        for r in bm25_results:
            text_to_meta.setdefault(r["text"], r.get("metadata", {}))

        scores: Dict[str, float] = {}
        for text in all_texts:
            dr = dense_rank.get(text, n_dense + 1)
            br = bm25_rank.get(text, n_bm25 + 1)
            scores[text] = (
                alpha_dense * (1.0 / (rrf_k + dr))
                + alpha_bm25 * (1.0 / (rrf_k + br))
            )

        fused = []
        for text, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            fused.append({
                "text": text,
                "metadata": text_to_meta.get(text, {}),
                "score": score,
                "fused_score": score,
            })
        return fused

    def _records_to_query_result(
        self,
        records: List[Dict[str, Any]],
        top_k: int,
    ) -> QueryResult:
        """Convert raw search records into the common QueryResult shape."""
        nodes = []
        scores = []
        for record in records[:top_k]:
            nodes.append(Node(
                text=record.get("text", ""),
                metadata=record.get("metadata", {}) or {},
            ))
            scores.append(float(record.get("score", 0.0)))
        return QueryResult(nodes=nodes, scores=scores)

    def _query_result_to_records(self, result: QueryResult) -> List[Dict[str, Any]]:
        """Convert QueryResult into CRAG-compatible records."""
        records = []
        for node, score in zip_longest(result.nodes, result.scores, fillvalue=0.0):
            if not isinstance(node, Node):
                continue
            records.append({
                "text": node.text,
                "metadata": node.metadata or {},
                "score": float(score or 0.0),
            })
        return records

    async def _colbert_rerank_query_result(
        self,
        query: str,
        result: QueryResult,
        embed_provider: Any,
        top_k: int,
        stage_label: str,
    ) -> QueryResult:
        """Apply stored ColBERT MaxSim reranking to an existing candidate set."""
        if not getattr(self.config, "enable_multi_vector_rerank", False):
            logger.info(f"[{stage_label}] ColBERT rerank跳过: enable_multi_vector_rerank=false")
            return QueryResult(nodes=result.nodes[:top_k], scores=result.scores[:top_k])

        if len(result.nodes) <= 1:
            logger.info(f"[{stage_label}] ColBERT rerank跳过: candidates={len(result.nodes)}")
            return QueryResult(nodes=result.nodes[:top_k], scores=result.scores[:top_k])

        try:
            colbert_storage = self._ensure_colbert_storage_initialized()
            if not getattr(colbert_storage, "is_loaded", False):
                logger.info(f"[{stage_label}] ColBERT rerank跳过: 未找到预存多向量")
                return QueryResult(nodes=result.nodes[:top_k], scores=result.scores[:top_k])

            reranker = MultiVectorReranker(embed_provider)
            reranker.set_colbert_storage(colbert_storage)

            logger.info(
                f"[{stage_label}] ColBERT chunk rerank开始: "
                f"candidates={len(result.nodes)}, top_k={top_k}"
            )
            reranked = await reranker.rerank(
                query=query,
                nodes=result.nodes,
                scores=result.scores,
                top_k=top_k,
            )
            if reranked.scores:
                logger.info(
                    f"[{stage_label}] ColBERT chunk rerank完成: "
                    f"{len(result.nodes)} -> {len(reranked.nodes)}, "
                    f"score_range={max(reranked.scores):.6f}..{min(reranked.scores):.6f}"
                )
            else:
                logger.info(f"[{stage_label}] ColBERT chunk rerank完成: results=0")
            return reranked
        except Exception as e:
            logger.warning(f"[{stage_label}] ColBERT chunk rerank失败，保留原始顺序: {e}")
            return QueryResult(nodes=result.nodes[:top_k], scores=result.scores[:top_k])

    async def _apply_crag_quality_eval(
        self,
        query: str,
        result: QueryResult,
        index_manager: HybridIndexManager,
        embed_provider: Any,
        top_k: int,
        paper_ids: Optional[List[str]] = None,
    ) -> QueryResult:
        """Evaluate retrieval quality and optionally run corrective retrieval."""
        if not getattr(self.config, "enable_crag_quality_eval", True):
            logger.info("[CRAG] 质量评估跳过: enable_crag_quality_eval=false")
            return result

        records = self._query_result_to_records(result)

        # 获取 LLM provider（用于 CRAG 评估和查询重写）
        llm_provider = None
        if self._llm_initialized:
            llm_provider = self._llm_client
        else:
            try:
                llm_provider = await self._ensure_llm_initialized()
            except Exception as e:
                logger.debug(f"[CRAG] LLM Provider 不可用，使用规则评估: {e}")
                # 标记失败，避免后续每次搜索都重新尝试加载 VLM
                self._llm_initialized = False
                self._llm_client = None
                self._llm_init_failed = True

        evaluator = CragEvaluator(llm_provider=llm_provider)
        assessment = await evaluator.evaluate_retrieval_quality(query, records)
        score = float(assessment.get("score", 0.0) or 0.0)
        level = str(assessment.get("level", "medium"))
        reasoning = assessment.get("reasoning", "")
        logger.info(
            f"[CRAG] 质量评估完成: level={level}, score={score:.3f}, "
            f"results={len(records)}, reasoning={reasoning}"
        )

        if not getattr(self.config, "crag_enable_correction", False):
            logger.info("[CRAG] 自动纠偏跳过: crag_enable_correction=false")
            return result

        min_score = float(getattr(self.config, "crag_min_score", 0.5) or 0.5)
        if level != "low" and score >= min_score:
            logger.info(f"[CRAG] 自动纠偏跳过: quality_ok (threshold={min_score:.3f})")
            return result

        try:
            logger.info(
                f"[CRAG] 自动纠偏启动: level={level}, score={score:.3f}, "
                f"threshold={min_score:.3f}"
            )
            corrector = CragCorrector(index_manager, embed_provider, llm_provider)
            corrected = await corrector.correct(
                query=query,
                level=level,
                original_results=records,
                missing_aspects=str(assessment.get("missing_aspects", "")),
                paper_ids=paper_ids,
            )
            if corrected:
                logger.info(f"[CRAG] 自动纠偏完成: corrected_results={len(corrected)}")
                return self._records_to_query_result(corrected, top_k=top_k)
            logger.warning("[CRAG] 自动纠偏未返回结果，保留原始检索结果")
        except Exception as e:
            logger.warning(f"[CRAG] 自动纠偏失败，保留原始检索结果: {e}")

        return result

    async def _graph_recall_papers(self, query: str) -> List[str]:
        """Recall paper IDs from knowledge graph to supplement abstract search."""
        if not getattr(self.config, 'enable_graph_rag', False):
            logger.debug("[HybridRAGEngine] graph召回: enable_graph_rag=False，跳过")
            return []

        try:
            retriever = await self._ensure_retriever_initialized()
            graph_retriever = getattr(retriever, '_graph_retriever', None)
            if not graph_retriever:
                logger.warning("[HybridRAGEngine] graph召回: _graph_retriever 不可用")
                return []

            graph_result = await graph_retriever.aretrieve(query)
            if not graph_result:
                logger.info("[HybridRAGEngine] graph召回: retriever 返回 0 条结果")
                return []

            # 精度优先：按 retriever score 排序，只取 Top-5 实体
            entity_scores: dict[str, float] = {}
            for nws in graph_result:
                score = getattr(nws, 'score', None)
                if score is not None and score <= 0:
                    continue
                node = nws.node
                text = getattr(node, 'text', '') or ''

                # 格式1: TextToCypher → "Cypher Response:\n[{...}]"
                cr_match = re.search(r'Cypher Response:\s*(\[.+?\](?:\s*$))', text, re.DOTALL)
                if cr_match:
                    try:
                        rows = ast.literal_eval(cr_match.group(1))
                        if isinstance(rows, list):
                            for row in rows:
                                for f in ('head', 'tail'):
                                    v = row.get(f, '')
                                    if v and isinstance(v, str) and len(v) >= 2:
                                        name = v.strip()
                                        entity_scores[name] = max(entity_scores.get(name, 0), score or 1.0)
                    except (ValueError, SyntaxError):
                        pass
                    continue

                # 格式2: LLMSynonymRetriever "EntityA -> REL -> EntityB"
                if ' -> ' in text:
                    for p in text.split(' -> '):
                        p = p.strip()
                        if len(p) >= 2:
                            entity_scores[p] = max(entity_scores.get(p, 0), score or 1.0)
                elif text:
                    stripped = text.strip()
                    if len(stripped) >= 2 and len(stripped) < 200:
                        entity_scores[stripped] = max(entity_scores.get(stripped, 0), score or 1.0)

            if not entity_scores:
                logger.warning("[HybridRAGEngine] graph召回: 0 entities，无实体名可查")
                return []

            _STOP_WORDS = {'lpips', 'psnr', 'ssim', 'fid', 'iou', 'map', 'auc',
                           'training', 'testing', 'evaluation', 'performance',
                           'accuracy', 'quality', 'efficiency', 'robustness'}
            scored = [
                (name, s) for name, s in entity_scores.items()
                if len(name) >= 5 and name.lower() not in _STOP_WORDS
            ]
            scored.sort(key=lambda x: -x[1])
            limited_names = [name for name, _ in scored[:5]]
            if not limited_names:
                return []

            logger.info(
                f"[HybridRAGEngine] graph召回: Top-5 entities → "
                f"{limited_names}"
            )

            # 精度优先：每个实体取 2 篇，按实体 score 加权累加
            paper_entity_count: dict[str, float] = {}
            graph_store = None
            try:
                if hasattr(graph_retriever, 'sub_retrievers') and graph_retriever.sub_retrievers:
                    graph_store = getattr(graph_retriever.sub_retrievers[0], '_graph_store', None)
            except Exception as e:
                logger.debug(f"[HybridRAGEngine] 无法从 PGRetriever 提取 graph_store: {e}")

            if graph_store is None:
                logger.warning("[HybridRAGEngine] graph论文召回: 无法获取 Neo4j graph_store")
            else:
                driver = getattr(graph_store, '_driver', None) or getattr(graph_store, 'client', None)
                if driver is not None:
                    with driver.session(database="neo4j") as session:
                        # 每个实体取最多 2 篇，按实体 score 加权累加
                        for name in limited_names:
                            safe_name = name.replace("\\", "\\\\").replace("'", "\\'")
                            weight = entity_scores.get(name, 1.0)
                            r = session.run(
                                f"MATCH (n) WHERE toLower(n.name) CONTAINS toLower('{safe_name}') "
                                "AND n.chunk_id IS NOT NULL "
                                "RETURN DISTINCT n.chunk_id AS cid LIMIT 2 "
                                "UNION "
                                f"MATCH (n)-[r]->() WHERE toLower(n.name) CONTAINS toLower('{safe_name}') "
                                "AND r.chunk_id IS NOT NULL "
                                "RETURN DISTINCT r.chunk_id AS cid LIMIT 2"
                            )
                            for rec in r:
                                cid = rec["cid"]
                                if cid:
                                    if not cid.lower().endswith(".pdf"):
                                        cid = cid + ".pdf"
                                    paper_entity_count[cid] = paper_entity_count.get(cid, 0) + weight

            # 按加权关联分排序（实体 score 加权），取 Top-3
            ranked = sorted(paper_entity_count.items(), key=lambda x: -x[1])
            graph_paper_ids = [cid for cid, _ in ranked[:3]]

            logger.info(
                f"[HybridRAGEngine] graph论文召回: {len(graph_paper_ids)} papers "
                f"from {len(limited_names)} entities (weighted={dict(list(ranked[:5]))})"
            )
            return graph_paper_ids

        except Exception as e:
            logger.warning(f"[HybridRAGEngine] graph论文召回失败: {e}")
            logger.warning(traceback.format_exc())
            return []

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "rag",
        **kwargs
    ) -> QueryResult:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            mode: 检索模式 ("rag", "retrieve", "graph_local", "graph_global", "hybrid")
            **kwargs: 其他参数

        Returns:
            QueryResult 对象
        """
        if top_k is None:
            top_k = self.config.top_k

        # Map graph modes to supported paths (graceful degradation, no ValueError crashes)
        actual_mode = mode
        if mode == "graph_global":
            # graph_global: global context, heavy entity aggregation → use retrieve (no generation)
            actual_mode = "retrieve"
            logger.info("[HybridRAGEngine] graph_global → retrieve (pure retrieval, no generation)")
        elif mode == "graph_local":
            # graph_local: local neighborhood → use rag (retrieval + generation)
            actual_mode = "rag"
            logger.info("[HybridRAGEngine] graph_local → rag (retrieval + generation)")

        # 确保组件已初始化。Retriever 依赖 embedding provider，搜索入口必须先初始化它。
        embed_provider = await self._ensure_embed_provider_initialized()
        logger.info(f"[HybridRAGEngine] Embedding Provider ready: {self.config.embedding_mode}")

        # 两阶段检索
        use_two_stage = getattr(self.config, 'enable_two_stage_retrieval', False)
        retriever: Optional[Union[VectorRetriever, HybridRetriever]] = None
        paper_ids_for_crag: Optional[List[str]] = None
        logger.info(
            f"[HybridRAGEngine] search开始: mode={mode}, retrieval={'two_stage' if use_two_stage else 'single_stage'}, "
            f"top_k={top_k}, two_stage_enabled={use_two_stage}, "
            f"two_stage_top_k={getattr(self.config, 'two_stage_top_k', 10)}, "
            f"two_stage_rerank_k={getattr(self.config, 'two_stage_rerank_k', 5)}, "
            f"reranking={getattr(self.config, 'enable_multi_vector_rerank', False)}, "
            f"crag_eval={getattr(self.config, 'enable_crag_quality_eval', True)}, "
            f"crag_correction={getattr(self.config, 'crag_enable_correction', False)}"
        )

        if use_two_stage:
            await self._ensure_abstract_manager_initialized()
            if self._abstract_manager is None:
                logger.warning("[HybridRAGEngine] 两阶段检索已开启，但摘要索引不可用，将降级到单阶段检索")

        if use_two_stage and self._abstract_manager is not None:
            # ========== 两阶段检索 ==========
            # 阶段1: 摘要检索找到相关论文
            abstract_top_k = 20
            logger.info(f"[两阶段] 阶段1摘要检索开始: abstract_top_k={abstract_top_k}")
            try:
                abstract_results = await self._abstract_manager.search_by_abstract(
                    query, top_k=abstract_top_k
                )
            except Exception as e:
                if _is_mps_oom_error(e):
                    _clear_accelerator_cache()
                    raise RuntimeError(
                        "MPS 内存不足，阶段1摘要检索无法生成查询向量。"
                        "已停止降级到单阶段检索，避免触发更高内存占用。"
                        "请重启 AstrBot 让 MPS high watermark 配置生效，"
                        "或临时切换 unsloth.device=cpu。"
                    ) from e
                raise

            if abstract_results:
                # 构建 paper_id 列表（支持 .pdf 后缀格式）
                paper_ids = []
                for r in abstract_results:
                    pid = r.get("paper_id", "")
                    if pid:
                        if not pid.lower().endswith(".pdf"):
                            pid = pid + ".pdf"
                        paper_ids.append(pid)

                top_abstract_score = abstract_results[0].get("score", 0.0)
                bottom_abstract_score = abstract_results[-1].get("score", 0.0)
                logger.info(
                    f"[两阶段] 阶段1摘要检索完成: papers={len(abstract_results)}, "
                    f"score_range={top_abstract_score:.6f}..{bottom_abstract_score:.6f}"
                )

                # 阶段1.5: Rerank 摘要（使用 ColBERT MaxSim）
                ABSTRACT_RERANK_QUOTA = 6
                GRAPH_RECALL_QUOTA = 3
                two_stage_rerank_k = ABSTRACT_RERANK_QUOTA
                abstract_colbert = self._ensure_abstract_colbert_storage_initialized()
                reranked_paper_ids = paper_ids[:two_stage_rerank_k]

                if (abstract_colbert is not None
                        and len(abstract_colbert) > 0
                        and getattr(self.config, 'enable_multi_vector_rerank', False)):
                    try:
                        model = get_embedding_model()
                        if model:
                            import numpy as np
                            qv = model.get_multi_vector(query)
                            if not qv:
                                raise ValueError("get_multi_vector returned empty")
                            query_vecs = np.array(qv, dtype=np.float32)
                            scored = []
                            for r in abstract_results:
                                pid = r.get("paper_id", "")
                                abstract_key = f"abstract_{pid}"
                                chunk_idx = abstract_colbert.find_chunk_idx(abstract_key)
                                if chunk_idx is not None:
                                    doc_vecs = abstract_colbert.get_chunk_token_vectors(chunk_idx)
                                    if doc_vecs is not None:
                                        sim = np.dot(query_vecs, doc_vecs.T)
                                        maxsim = float(np.sum(np.max(sim, axis=1)))
                                        scored.append((pid, maxsim))
                                    else:
                                        scored.append((pid, r.get("score", 0.0)))
                                else:
                                    scored.append((pid, r.get("score", 0.0)))
                            scored.sort(key=lambda x: x[1], reverse=True)
                            reranked_paper_ids = []
                            for s in scored[:two_stage_rerank_k]:
                                pid = s[0]
                                if not pid.lower().endswith(".pdf"):
                                    pid = pid + ".pdf"
                                reranked_paper_ids.append(pid)
                            score_str = f"{scored[0][1]:.4f}..{scored[min(len(scored)-1, two_stage_rerank_k-1)][1]:.4f}" if scored else "N/A"
                            logger.info(
                                f"[两阶段] 阶段1.5 ColBERT摘要rerank完成: "
                                f"selected={len(reranked_paper_ids)}/{len(paper_ids)} papers, "
                                f"score_range={score_str}"
                            )
                        else:
                            logger.error(
                                f"[两阶段] 阶段1.5摘要rerank跳过: embedding model 不可用，"
                                f"无法生成 query ColBERT 向量，降级使用原始分数: "
                                f"selected={len(reranked_paper_ids)}/{len(paper_ids)} papers"
                            )
                    except Exception as e:
                        logger.warning(f"[两阶段] 阶段1.5摘要rerank失败: {e}，使用原始分数")
                        reranked_paper_ids = paper_ids[:two_stage_rerank_k]
                        logger.info(
                            f"[两阶段] 阶段1.5摘要rerank跳过（失败），"
                            f"selected={len(reranked_paper_ids)}/{len(paper_ids)} papers"
                        )
                else:
                    logger.info(
                        f"[两阶段] 阶段1.5摘要rerank跳过: ColBERT未启用或无摘要向量，"
                        f"selected={len(reranked_paper_ids)}/{len(paper_ids)} papers"
                    )

                # 阶段1.6: 图谱独立通道召回（不走 ColBERT rerank）
                if getattr(self.config, 'enable_graph_rag', False):
                    graph_papers = await self._graph_recall_papers(query)
                    if graph_papers:
                        existing = set(reranked_paper_ids)
                        new_from_graph = [p for p in graph_papers if p not in existing]
                        if new_from_graph:
                            added = new_from_graph[:GRAPH_RECALL_QUOTA]
                            reranked_paper_ids.extend(added)
                            logger.info(
                                f"[两阶段] 阶段1.6 graph独立召回: +{len(added)} papers "
                                f"(e.g. {[os.path.splitext(p)[0] for p in added[:3]]}), "
                                f"abstract={ABSTRACT_RERANK_QUOTA}, graph={len(added)}, "
                                f"total={len(reranked_paper_ids)}"
                            )

                # 阶段2: 在选中的论文内检索 chunks
                paper_ids_for_crag = reranked_paper_ids
                index_manager = self._ensure_index_manager_initialized()
                try:
                    query_embedding = await embed_provider.get_text_embedding(query)
                except Exception as e:
                    if _is_mps_oom_error(e):
                        _clear_accelerator_cache()
                        raise RuntimeError(
                            "MPS 内存不足，阶段2论文内检索无法生成查询向量。"
                            "请重启 AstrBot 让 MPS high watermark 配置生效，"
                            "或临时切换 unsloth.device=cpu。"
                        ) from e
                    raise
                stage2_candidate_k = top_k
                if getattr(self.config, 'enable_multi_vector_rerank', False):
                    stage2_candidate_k = max(top_k * 2, top_k)

                logger.info(
                    f"[两阶段] 阶段2论文内chunk检索开始: "
                    f"papers={len(reranked_paper_ids)}, candidate_k={stage2_candidate_k}, final_top_k={top_k}"
                )
                chunk_results = await index_manager.search_with_paper_filter(
                    query_embedding=query_embedding,
                    paper_ids=reranked_paper_ids,
                    top_k=stage2_candidate_k
                )

                logger.info(f"[两阶段] 阶段2 dense检索完成: chunks={len(chunk_results)}")

                # 阶段2 BM25: 全库精确匹配检索（不受论文筛选限制）
                # BM25 的优势是跨全集词匹配，不应被语义抽象筛选的 paper_id 过滤
                paper_id_set = set(reranked_paper_ids)
                if getattr(self.config, 'enable_bm25', True):
                    bm25_retriever = await self._get_bm25_retriever()
                    if bm25_retriever and bm25_retriever.is_exact_match_query(query):
                        try:
                            bm25_raw = await bm25_retriever.retrieve(query, top_k=self.config.bm25_top_k)
                            # 不过滤 paper_id：精确词匹配不应受语义筛选限制
                            bm25_all = [
                                {"text": n.text, "metadata": n.metadata or {}, "score": float(s)}
                                for n, s in zip(bm25_raw.nodes, bm25_raw.scores)
                            ]
                            if bm25_all:
                                outside_count = sum(
                                    1 for n in bm25_raw.nodes
                                    if (n.metadata or {}).get("file_name", "") not in paper_id_set
                                )
                                logger.info(
                                    f"[两阶段] 阶段2 BM25全库召回: {len(bm25_all)} chunks "
                                    f"(含筛选论文外 {outside_count} chunks)"
                                )
                                chunk_results = self._stage2_rrf_fusion(
                                    chunk_results, bm25_all, top_k=stage2_candidate_k
                                )
                                logger.info(f"[两阶段] 阶段2 RRF融合完成: {len(chunk_results)} chunks")
                        except Exception as e:
                            logger.warning(f"[两阶段] 阶段2 BM25检索失败: {e}")

                logger.info(f"[两阶段] 阶段2论文内chunk检索完成: chunks={len(chunk_results)}")

                if chunk_results:
                    result = self._records_to_query_result(
                        chunk_results,
                        top_k=len(chunk_results),
                    )
                    # 阶段2.5: 对候选 chunks 做 ColBERT MaxSim 重排。
                    result = await self._colbert_rerank_query_result(
                        query=query,
                        result=result,
                        embed_provider=embed_provider,
                        top_k=top_k,
                        stage_label="两阶段 阶段2.5",
                    )
                else:
                    logger.warning("[两阶段] 阶段2论文内chunk检索无结果，返回空")
                    result = QueryResult(nodes=[], scores=[])
            else:
                # 摘要检索无结果，降级到普通检索
                logger.warning("[两阶段] 阶段1摘要检索无结果，降级到单阶段检索")
                retriever = await self._ensure_retriever_initialized()
                result = await retriever.retrieve(query, top_k=top_k)
        else:
            # 普通单阶段检索
            if use_two_stage:
                logger.warning("[HybridRAGEngine] 两阶段检索未执行，降级到单阶段检索")
            else:
                logger.info("[HybridRAGEngine] 两阶段检索未开启，执行单阶段检索")
            retriever = await self._ensure_retriever_initialized()
            result = await retriever.retrieve(query, top_k=top_k)

        index_manager_for_crag = self._ensure_index_manager_initialized()
        result = await self._apply_crag_quality_eval(
            query=query,
            result=result,
            index_manager=index_manager_for_crag,
            embed_provider=embed_provider,
            top_k=top_k,
            paper_ids=paper_ids_for_crag,
        )

        if result.scores:
            logger.info(
                f"[HybridRAGEngine] search完成: results={len(result.nodes)}, "
                f"score_range={max(result.scores):.6f}..{min(result.scores):.6f}"
            )
        else:
            logger.info("[HybridRAGEngine] search完成: results=0")

        return result

    async def _resolve_llm_config(self) -> dict:
        """Resolve LLM config for reference parsing.

        For OpenAI-compatible providers (ModelScope, etc.), extracts the
        provider's own API config so _call_via_http can call it directly,
        bypassing AstrBot's provider wrapper (which may have parsing issues).

        Falls back to freeapi if no provider is available.
        """
        if not getattr(self.config, 'enable_llm_reference_parsing', True):
            return {}

        config = {}

        # Priority 1: configured provider (AstrBot plugin LLM)
        text_provider_id = getattr(self.config, 'text_provider_id', '') or ''
        if text_provider_id:
            try:
                provider = await self._ensure_llm_initialized()
                if provider:
                    model = getattr(provider, 'model', None) or getattr(provider, 'model_name', None) or 'unknown'
                    config["provider"] = provider

                    # For OpenAI-compatible providers, extract API config
                    # so _call_via_http can call the provider directly as fallback
                    pc = getattr(provider, 'provider_config', {})
                    ptype = pc.get("type", "")
                    if "openai" in ptype:
                        api_base = pc.get("api_base", "")
                        keys = pc.get("key", [])
                        if api_base and keys:
                            config["model"] = model
                            config["api_base"] = api_base
                            config["api_key"] = keys[0] if isinstance(keys, list) else keys
                            logger.debug(f"📝 Provider raw HTTP fallback: {model} @ {api_base}")

                    logger.debug(f"📝 使用 Provider 进行 LLM 参考文献解析 ({model})")
                    return config
                logger.warning("⚠️ Provider 初始化失败")
            except Exception as e:
                logger.warning(f"⚠️ 无法获取 Provider: {e}")

        # Priority 2: freeapi fallback
        api_url = getattr(self.config, 'freeapi_url', '') or ''
        api_key = getattr(self.config, 'freeapi_key', '') or ''
        if api_url and api_key:
            config["model"] = "gpt-4o-mini"
            config["api_base"] = f"{api_url}/v1"
            config["api_key"] = api_key
            logger.debug("📝 使用 freeapi (gpt-4o-mini) 进行 LLM 参考文献解析")
            return config

        return {}

    async def add_papers(
        self,
        file_paths: List[str],
        llm_config: Dict[str, Any] = {},
        arxiv_client: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """添加论文到索引"""
        parser = self._ensure_parser_initialized()
        index_manager = self._ensure_index_manager_initialized()
        embed_provider = await self._ensure_embed_provider_initialized()

        # 初始化 ColBERT 存储（如果启用 reranking）
        colbert_storage = None
        if self.config.enable_multi_vector_rerank:
            colbert_storage = self._ensure_colbert_storage_initialized()
            # 首次仅做入库时，retriever 可能尚未初始化；这里强制初始化，
            # 以便 reranker 和 ColBERT storage 能正常绑定并写盘。
            retriever = await self._ensure_retriever_initialized()
            # 如果 reranker 已初始化，绑定新 storage
            reranker = getattr(retriever, '_reranker', None)
            if reranker is not None:
                reranker.set_colbert_storage(colbert_storage)
            else:
                logger.warning("[PaperRAG] HybridRetriever 无 _reranker，ColBERT storage 未绑定")

        # LLM 参考文献解析配置：显式传入优先，否则自动解析
        effective_llm_config = llm_config if llm_config else await self._resolve_llm_config()

        results = {
            "total": len(file_paths),
            "successful": 0,
            "failed": 0,
            "chunks_added": 0,
            "errors": []
        }

        for file_path in file_paths:
            try:
                # 解析 PDF（传递 LLM config 和 arXiv client 以支持 LLM-based 引用解析）
                nodes = await parser.parse_and_split(file_path, effective_llm_config, arxiv_client)

                if not nodes:
                    results["failed"] += 1
                    results["errors"].append({"file": file_path, "error": "无法解析 PDF"})
                    continue

                noise_report = []

                if not nodes:
                    results["failed"] += 1
                    results["errors"].append({"file": file_path, "error": "过滤后无有效 chunks"})
                    continue

                # 获取 embeddings
                texts = [node.text for node in nodes]
                embeddings = await embed_provider.get_embeddings(texts)

                # 插入 Milvus
                chunks_inserted = await index_manager.insert_nodes(nodes, embeddings)
                results["chunks_added"] += chunks_inserted

                # 预计算并存储 ColBERT per-token vectors
                if colbert_storage is not None:
                    model = get_embedding_model()
                    chunk_vectors = []
                    chunk_ids = []
                    if model is None:
                        logger.error(
                            "[ColBERTStorage] 未获取到 ColBERT embedding model，"
                            f"跳过 per-token vectors 预计算: {file_path}"
                        )
                    else:
                        for i, node in enumerate(nodes):
                            vec = model.get_multi_vector(node.text)
                            if vec:
                                import numpy as np
                                chunk_vectors.append(np.array(vec, dtype=np.float32))
                                chunk_id = node.metadata.get("chunk_id", f"{file_path}_{i}")
                                chunk_ids.append(chunk_id)
                            else:
                                logger.warning(f"[ColBERTStorage] get_multi_vector 返回空，跳过 chunk {i}: {file_path}")
                        if chunk_vectors:
                            colbert_storage.add_chunks(chunk_vectors, chunk_ids)
                            logger.debug(f"[ColBERTStorage] 为 {len(chunk_vectors)} 个 chunks 存储了 per-token vectors")
                        else:
                            logger.warning(
                                "[ColBERTStorage] 未生成任何 per-token vectors，"
                                f"跳过本篇 chunk 存储: {file_path}"
                            )

                results["successful"] += 1

                # 索引摘要（从第一个 node 的 metadata 中提取）
                try:
                    if nodes and file_path.lower().endswith(".pdf"):
                        abstract_manager = await self._ensure_abstract_manager_initialized()
                    else:
                        abstract_manager = None

                    if abstract_manager is not None and nodes:
                        first_node = nodes[0]
                        file_name = first_node.metadata.get("file_name", os.path.basename(file_path))
                        paper_id = os.path.splitext(file_name)[0]
                        extracted_title = first_node.metadata.get("extracted_title") or ""
                        extracted_abstract = first_node.metadata.get("extracted_abstract") or ""
                        abstract_metadata = {}
                        for key in (
                            "arxiv_url",
                            "github_url",
                            "doi_url",
                            "resolution_source",
                            "resolution_score",
                            "matched_title",
                            "matched_identifier",
                            "title_source",
                            "abstract_source",
                        ):
                            value = first_node.metadata.get(key)
                            if value:
                                abstract_metadata[key] = value
                        if extracted_title:
                            abstract_metadata.setdefault("extracted_title", extracted_title)
                        if extracted_abstract:
                            abstract_metadata.setdefault("extracted_abstract_chars", len(extracted_abstract))
                        await abstract_manager.index_paper(
                            pdf_path=file_path,
                            paper_id=paper_id,
                            file_name=file_name,
                            title=extracted_title,
                            abstract_text=extracted_abstract if extracted_abstract else None,
                            metadata=abstract_metadata or None,
                        )
                        indexed = abstract_manager._abstract_cache.get(paper_id)  # type: ignore[attr-defined]
                        indexed_title = indexed.title if indexed else ""
                        logger.info(f"📄 已索引摘要: {indexed_title or file_name}")

                        # 为摘要生成 ColBERT per-token vectors
                        # 优先从 abstract_manager 缓存获取（可能来自 arXiv 等外部源），
                        # 而非 chunk metadata（PDF 解析阶段不一定能提取到 abstract）
                        indexed_abstract_text = getattr(indexed, "abstract_text", "") or "" if indexed else ""
                        abstract_text_full = f"{indexed_title}\n\n{indexed_abstract_text}" if indexed_title and indexed_abstract_text else (indexed_abstract_text or extracted_abstract or "")
                        if abstract_text_full.strip():
                            try:
                                abstract_colbert = self._ensure_abstract_colbert_storage_initialized()
                                model = get_embedding_model()
                                if model:
                                    vec = model.get_multi_vector(abstract_text_full)
                                    if vec:
                                        import numpy as np
                                        abstract_colbert.add_chunks(
                                            [np.array(vec, dtype=np.float32)],
                                            [f"abstract_{paper_id}"],
                                        )
                                        logger.debug(f"[ColBERTStorage-Abstract] 已存储摘要向量: {paper_id}")
                                    else:
                                        logger.warning(f"[ColBERTStorage-Abstract] get_multi_vector 返回空，跳过摘要向量: {paper_id}")
                                else:
                                    logger.error(f"[ColBERTStorage-Abstract] embedding model 不可用，无法生成摘要 ColBERT 向量: {paper_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ 摘要ColBERT向量生成失败: {e}")
                        else:
                            logger.warning(f"[ColBERTStorage-Abstract] 摘要文本为空，跳过向量生成: {paper_id}")
                except Exception as e:
                    logger.warning(f"⚠️ 摘要索引失败: {e}")

            except Exception as e:
                logger.error(f"添加论文失败 {file_path}: {e}")
                results["failed"] += 1
                results["errors"].append({"file": file_path, "error": str(e)})

        # 索引完成后保存 ColBERT 存储
        if colbert_storage is not None and results["successful"] > 0:
            colbert_storage.save()
            logger.info(f"[ColBERTStorage] 已保存 {len(colbert_storage)} 个 chunks 的 per-token vectors")
        if self._abstract_colbert_storage is not None and len(self._abstract_colbert_storage) > 0:
            self._abstract_colbert_storage.save()
            logger.info(f"[ColBERTStorage-Abstract] 已保存 {len(self._abstract_colbert_storage)} 个摘要的 per-token vectors")

        return results

    async def add_paper(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        添加单个论文到索引

        Args:
            file_path: 论文文件路径

        Returns:
            {"status": "success", "chunks_added": N} 或 {"status": "error", "error": ...}
        """
        result = await self.add_papers([file_path], **kwargs)
        if result["failed"] > 0:
            return {"status": "error", "error": result["errors"][0]["error"] if result["errors"] else "Unknown error"}
        return {"status": "success", "chunks_added": result["chunks_added"]}

    async def delete_paper(self, file_name: str = "", file_path: str = "", **kwargs) -> Dict[str, Any]:
        """
        删除论文

        Args:
            file_name: 要删除的文件名
            file_path: 文件路径（可选）

        Returns:
            {"status": "success", "deleted": N} 或 {"status": "error", "error": ...}
        """
        try:
            index_manager = self._ensure_index_manager_initialized()
            result = await index_manager.delete_by_file_name(file_name or file_path)

            # 同时删除 ColBERT 存储中的 chunks
            colbert_storage = self._ensure_colbert_storage_initialized()
            prefix = file_name or file_path
            if prefix:
                colbert_storage.delete_by_file_prefix(prefix)
                colbert_storage.save()

            # 刷新 BM25 索引
            bm25_retriever = getattr(self._retriever, '_bm25_retriever', None)
            if bm25_retriever is not None:
                bm25_retriever.refresh_index()

            # 删除摘要索引条目
            try:
                abstract_manager = await self._ensure_abstract_manager_initialized()
                if abstract_manager is not None:
                    paper_id = os.path.splitext(os.path.basename(prefix))[0]
                    await abstract_manager.delete_paper(paper_id)
                    logger.info(f"🗑️ 已删除摘要索引: {paper_id}")
            except Exception as e:
                logger.warning(f"⚠️ 删除摘要索引失败: {e}")

            # 删除摘要 ColBERT 向量
            try:
                paper_id = os.path.splitext(os.path.basename(prefix))[0]
                abstract_colbert = self._ensure_abstract_colbert_storage_initialized()
                if abstract_colbert is not None and len(abstract_colbert) > 0:
                    abstract_colbert.delete_by_file_prefix(f"abstract_{paper_id}")
                    abstract_colbert.save()
            except Exception as e:
                logger.warning(f"⚠️ 删除摘要ColBERT向量失败: {e}")

            return {
                "status": "success",
                "deleted_count": result.get("deleted_count", 0),
                "message": result.get("message", "Paper deleted"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def clear(self, **kwargs) -> Dict[str, Any]:
        """
        清空索引

        Returns:
            {"status": "success"} 或 {"status": "error", "error": ...}
        """
        try:
            index_manager = self._ensure_index_manager_initialized()
            await index_manager.clear()

            # 同时清空 ColBERT 存储
            colbert_storage = self._ensure_colbert_storage_initialized()
            colbert_storage.clear_storage()
            self._colbert_storage = None

            # 清理旧路径残留（data/ 根目录下的 colbert_* 文件）
            plugin_data_dir = Path(__file__).parent.parent / "data"
            for stale_name in ("colbert_doc_vectors.npy", "colbert_faiss_index.bin", "colbert_id_mapping.json"):
                stale_file = plugin_data_dir / stale_name
                if stale_file.exists():
                    stale_file.unlink()

            # 刷新 BM25 索引
            bm25_retriever = getattr(self._retriever, '_bm25_retriever', None)
            if bm25_retriever is not None:
                bm25_retriever.refresh_index()

            # 清空摘要索引
            try:
                abstract_manager = await self._ensure_abstract_manager_initialized()
                if abstract_manager is not None:
                    abstract_manager.clear()
                    self._abstract_initialized = False
                    self._abstract_manager = None
                    logger.info("🗑️ 摘要索引已清空")
            except Exception as e:
                logger.warning(f"⚠️ 清空摘要索引失败: {e}")

            # 清空摘要 ColBERT 存储
            try:
                if self._abstract_colbert_storage is not None:
                    self._abstract_colbert_storage.clear_storage()
                    self._abstract_colbert_storage = None
                else:
                    # Storage was never initialized — clean stale files on disk if any
                    stale_dir = Path(__file__).parent.parent / "data" / "colbert_abstracts"
                    if stale_dir.exists():
                        shutil.rmtree(stale_dir)
                logger.info("🗑️ 摘要 ColBERT 存储已清空")
            except Exception as e:
                logger.warning(f"⚠️ 清空摘要 ColBERT 存储失败: {e}")

            # 清理物理文件：figures、tables、captions
            plugin_data_dir = Path(__file__).parent.parent / "data"
            for subdir in ["figures", "tables", "captions"]:
                subdir_path = plugin_data_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
                    subdir_path.mkdir(parents=True, exist_ok=True)

            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            index_manager = self._ensure_index_manager_initialized()
            return await index_manager.get_stats()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def list_papers(self) -> List[Dict[str, Any]]:
        """
        列出所有已索引的论文

        Returns:
            [{"file_name": str, "chunk_count": int, "added_time": str}, ...]
        """
        try:
            index_manager = self._ensure_index_manager_initialized()
            return await index_manager.list_unique_documents()
        except Exception as e:
            logger.error(f"列出论文失败: {e}")
            return []

    @property
    def index_manager(self) -> HybridIndexManager:
        """获取索引管理器"""
        return self._ensure_index_manager_initialized()

    @property
    def embed_provider(self):
        """获取 Embedding Provider"""
        return self._embed_provider

    @property
    def parser(self) -> HybridPDFParser:
        """获取 PDF 解析器"""
        return self._ensure_parser_initialized()
