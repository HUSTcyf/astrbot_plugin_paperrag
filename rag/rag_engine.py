"""
Paper RAG Plugin - 核心RAG引擎模块
混合架构版本：结合自定义PDF解析 + 本地向量存储
"""

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from astrbot.api import logger

# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    try:
        from .hybrid_rag import HybridRAGEngine
    except ImportError:
        from hybrid_rag import HybridRAGEngine

HYBRID_RAG_AVAILABLE = True


@dataclass
class RAGConfig:
    """RAG配置类"""

    # Embedding配置
    embedding_mode: str = "unsloth"  # "unsloth", "astrbot"
    embedding_provider_id: str = ""  # API模式下的Provider

    # LLM Provider配置
    compress_provider_id: str = ""  # LLM Provider ID（用于文本压缩）
    text_provider_id: str = ""  # LLM Provider ID（用于文本问答）
    multimodal_provider_id: str = ""  # LLM Provider ID（用于多模态问答）

    # Llama.cpp VLM配置（当multimodal_provider_id为空时使用）
    llama_vlm_model_path: str = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    llama_vlm_mmproj_path: str = "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"
    llama_vlm_max_tokens: int = 2560
    llama_vlm_temperature: float = 0.7
    llama_vlm_n_ctx: int = 4096
    llama_vlm_n_gpu_layers: int = 99

    # Unsloth配置（保留用于配置传递）
    ollama_config: dict = field(default_factory=dict)

    # Unsloth Embedding配置
    unsloth_config: dict = field(default_factory=dict)

    # Milvus配置
    milvus_lite_path: str = ""
    address: str = ""
    db_name: str = "default"
    authentication: Optional[dict] = None
    collection_name: str = "paper_embeddings"

    # 检索配置
    embed_dim: int = 768
    top_k: int = 5
    similarity_cutoff: float = 0.3

    # 论文目录
    papers_dir: str = "./papers"

    # 语义分块配置
    chunk_size: int = 512
    chunk_overlap: int = 0
    min_chunk_size: int = 100
    use_semantic_chunking: bool = True

    # 多模态配置
    enable_multimodal: bool = True
    figures_dir: str = ""

    # 混合检索配置（稀疏权重 + 稠密向量 + BM25精确匹配）
    enable_sparse_retrieval: bool = True  # 使用 BGE-M3 稀疏权重
    enable_multi_vector_rerank: bool = False  # 使用 ColBERT reranking
    enable_noise_filter: bool = True  # 使用本地 LLM 过滤噪声 chunks（参考文献/表格/符号等）
    sparse_top_k: int = 20        # 稀疏检索召回数量
    hybrid_alpha: float = 0.5    # RRF 融合权重（0=纯稀疏, 1=纯向量）
    hybrid_rrf_k: int = 60      # RRF 常数 k

    # BM25 精确匹配配置（用于专有名词、作者名等需要精确匹配的场景）
    enable_bm25: bool = True    # 启用 BM25 精确匹配（当检测到精确匹配意图时自动启用）
    bm25_top_k: int = 20       # BM25 召回数量

    # 重排序配置（向后兼容，内部使用 ColBERT reranking）
    enable_reranking: bool = False  # 使用 ColBERT reranking
    reranking_model: str = "BAAI/bge-reranker-v2-m3"  # 保留字段但不使用
    reranking_device: str = "auto"  # 保留字段但不使用
    reranking_adaptive: bool = True  # 保留字段但不使用
    reranking_threshold: float = 0.0  # 保留字段但不使用
    reranking_batch_size: int = 32  # 保留字段但不使用

    # LLM 参考文献解析配置
    enable_llm_reference_parsing: bool = True

    # FreeAPI 配置
    freeapi_url: str = ""
    freeapi_key: str = ""

    # Graph RAG 配置
    enable_graph_rag: bool = False
    graph_storage_type: str = "memory"
    graph_neo4j_uri: str = "bolt://localhost:7687"
    graph_neo4j_user: str = "neo4j"
    graph_neo4j_password: str = ""
    graph_max_triplets_per_chunk: int = 5
    graph_retrieval_top_k: int = 5
    graph_hybrid_alpha: float = 0.5
    graph_auto_build: bool = False
    graph_auto_build_threshold: int = 10

    # 两阶段检索配置
    enable_two_stage_retrieval: bool = False
    two_stage_top_k: int = 10
    two_stage_rerank_k: int = 5
    graph_multimodal_enabled: bool = True
    graph_max_images_per_chunk: int = 1
    graph_extract_image_entities: bool = True

    def __post_init__(self):
        """初始化后处理"""
        if self.authentication is None:
            self.authentication = {}
        if self.ollama_config is None:
            self.ollama_config = {}
        if self.unsloth_config is None:
            self.unsloth_config = {}

        # 向后兼容：如果设置了 enable_bm25，确保 sparse_top_k 有值
        if self.bm25_top_k > 0:
            self.sparse_top_k = self.bm25_top_k

        # 自动调整 embed_dim（BGE-M3 固定 1024 维）
        if self.embedding_mode == "unsloth":
            self.embed_dim = 1024

    def validate(self) -> tuple[bool, str]:
        """验证配置"""
        if self.embed_dim % 64 != 0:
            return False, "嵌入维度必须是64的倍数"
        return True, ""

    def get_connection_mode(self) -> str:
        """获取连接模式：'lite' 或 'remote'"""
        if self.milvus_lite_path and self.milvus_lite_path.strip():
            return 'lite'
        elif self.address and self.address.strip():
            return 'remote'
        else:
            return 'lite'

    def get_connection_uri(self) -> str:
        """获取连接 URI"""
        mode = self.get_connection_mode()
        if mode == 'lite':
            return self.milvus_lite_path
        else:
            return self.address


# ============================================================================
# 工厂函数
# ============================================================================

def create_rag_engine(config: RAGConfig, context) -> "HybridRAGEngine":
    """
    创建RAG引擎实例

    Args:
        config: RAG配置
        context: AstrBot上下文

    Returns:
        RAG引擎实例（HybridRAGEngine）
    """
    from .hybrid_rag import HybridRAGEngine

    logger.info("✅ 使用混合架构 RAG引擎（HybridRAGEngine）")
    logger.info("   - 自定义PDF解析（多模态）")
    logger.info("   - 语义分块")
    logger.info("   - Milvus向量存储")
    logger.info("   - BGE-M3 稀疏权重 + 稠密向量检索")
    if config.enable_multi_vector_rerank:
        logger.info("   - ColBERT 多向量 reranking")

    return HybridRAGEngine(config, context)
