"""
Shared engine factory — 单例模式复用 HybridRAGEngine 和 GraphRAGEngine。
从 commands/base.py 提取，避免 agentic_rag 重复初始化。
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Optional

from astrbot.api import logger
from rag.rag_engine import RAGConfig, create_rag_engine
from graphrag.graph_rag_engine import GraphRAGConfig, GraphRAGEngine

if TYPE_CHECKING:
    from ..rag.hybrid_rag import HybridRAGEngine


_engine_lock = threading.Lock()


def _configure_mps_memory() -> None:
    """Configure PyTorch MPS memory behavior."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')


def _create_rag_config(config: dict[str, Any]) -> "RAGConfig":
    """从插件配置字典构建 RAGConfig。"""

    raw_embedding_mode = config.get("embedding_mode", "unsloth")
    if raw_embedding_mode == "api":
        embedding_mode = "astrbot"
    elif raw_embedding_mode == "ollama":
        embedding_mode = "unsloth"
    else:
        embedding_mode = raw_embedding_mode

    return RAGConfig(
        embedding_mode=embedding_mode,
        embedding_provider_id=config.get("embedding_provider_id", ""),
        compress_provider_id=config.get("compress_provider_id", ""),
        text_provider_id=config.get("text_provider_id", ""),
        multimodal_provider_id=config.get("multimodal_provider_id", ""),
        unsloth_config=config.get("unsloth", {}),
        llama_vlm_model_path=config.get("llama_vlm_model_path", "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"),
        llama_vlm_mmproj_path=config.get("llama_vlm_mmproj_path", "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"),
        llama_vlm_max_tokens=config.get("llama_vlm_max_tokens", 25600),
        llama_vlm_temperature=config.get("llama_vlm_temperature", 0.7),
        llama_vlm_n_ctx=config.get("llama_vlm_n_ctx", 16384),
        llama_vlm_n_gpu_layers=config.get("llama_vlm_n_gpu_layers", 99),
        milvus_lite_path=config.get("milvus_lite_path", ""),
        address=config.get("address", ""),
        db_name=config.get("db_name", "default"),
        authentication=config.get("authentication", {}),
        collection_name=config.get("collection_name", "paper_embeddings"),
        embed_dim=config.get("embed_dim", 768),
        top_k=config.get("top_k", 5),
        similarity_cutoff=config.get("similarity_cutoff", 0.3),
        papers_dir=config.get("papers_dir", "./papers"),
        chunk_size=config.get("chunk_size", 512),
        chunk_overlap=config.get("chunk_overlap", 0),
        min_chunk_size=config.get("min_chunk_size", 100),
        use_semantic_chunking=config.get("use_semantic_chunking", True),
        enable_multimodal=config.get("multimodal", {}).get("enabled", True),
        figures_dir=config.get("figures_dir", ""),
        enable_sparse_retrieval=config.get("enable_sparse_retrieval", True),
        enable_multi_vector_rerank=config.get("enable_multi_vector_rerank", False),
        sparse_top_k=config.get("sparse_top_k", 20),
        hybrid_alpha=config.get("hybrid_alpha", 0.5),
        hybrid_rrf_k=config.get("hybrid_rrf_k", 60),
        enable_bm25=config.get("enable_bm25", True),
        bm25_top_k=config.get("bm25_top_k", 20),
        enable_two_stage_retrieval=bool(config.get("enable_two_stage_retrieval", False)),
        two_stage_top_k=config.get("two_stage_top_k") or 10,
        two_stage_rerank_k=config.get("two_stage_rerank_k") or 5,
        enable_crag_quality_eval=config.get("enable_crag_quality_eval", True),
        crag_enable_correction=config.get("crag_enable_correction", False),
        crag_min_score=config.get("crag_min_score", 0.3),
        enable_llm_reference_parsing=config.get("enable_llm_reference_parsing", True),
        freeapi_url=config.get("freeapi_url", ""),
        freeapi_key=config.get("freeapi_key", ""),
        core_api_key=config.get("core_api_key", ""),
        use_arxiv_api=config.get("use_arxiv_api", True),
        enable_graph_rag=config.get("enable_graph_rag", False),
        graph_storage_type=config.get("graph_rag", {}).get("storage_type", "neo4j"),
        graph_neo4j_uri=config.get("graph_rag", {}).get("neo4j_uri", "bolt://localhost:7687"),
        graph_neo4j_user=config.get("graph_rag", {}).get("neo4j_user", "neo4j"),
        graph_neo4j_password=config.get("graph_rag", {}).get("neo4j_password", ""),
        graph_max_triplets_per_chunk=config.get("graph_rag", {}).get("max_triplets_per_chunk", 5),
        graph_retrieval_top_k=config.get("graph_rag", {}).get("graph_retrieval_top_k", 5),
        graph_auto_build=config.get("graph_rag", {}).get("auto_build", False),
        graph_auto_build_threshold=config.get("graph_rag", {}).get("auto_build_threshold", 10),
    )


def get_engine(context: Any, config: Optional[dict[str, Any]] = None) -> Optional["HybridRAGEngine"]:
    """
    获取 HybridRAGEngine。

    Args:
        context: AstrBot Context 对象
        config: 可选，直接传入插件配置
    """
    if config is None:
        config = getattr(context, 'config', {})

    _configure_mps_memory()

    rag_config = _create_rag_config(config)
    logger.info(f"[agentic_rag] rag_config.enable_graph_rag={rag_config.enable_graph_rag}")
    valid, error_msg = rag_config.validate()
    if not valid:
        logger.error(f"[agentic_rag] RAG配置无效: {error_msg}")
        return None

    engine = create_rag_engine(rag_config, context)
    logger.info("[agentic_rag] HybridRAGEngine 已创建")
    return engine


async def get_graph_engine(
    context: Any,
    config: Optional[dict[str, Any]] = None
) -> Optional["GraphRAGEngine"]:
    """
    获取 GraphRAGEngine。

    Args:
        context: AstrBot Context 对象
        config: 可选，直接传入插件配置
    """
    if config is None:
        config = getattr(context, 'config', {})

    logger.info(f"[agentic_rag] get_graph_engine config: enable_graph_rag={config.get('enable_graph_rag', False)}")

    base_engine = get_engine(context, config)
    if base_engine is None:
        logger.warning("[agentic_rag] 基础引擎未就绪")
        return None


    graph_config = GraphRAGConfig.from_rag_config(base_engine.config)
    engine_instance = GraphRAGEngine(graph_config, base_engine, context)

    try:
        await engine_instance.initialize()
        if not getattr(engine_instance, '_initialized', False):
            logger.warning("[agentic_rag] Graph RAG 引擎初始化未完成")
            return None
        logger.info("[agentic_rag] GraphRAGEngine 已创建")
        return engine_instance
    except Exception as e:
        logger.error(f"[agentic_rag] Graph RAG 引擎创建失败: {e}")
        return None


# Re-export for backward compatibility
from provider.llm_utils import get_llm_provider  # noqa: F401
