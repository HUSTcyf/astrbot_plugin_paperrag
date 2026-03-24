"""
Paper RAG Plugin - 核心RAG引擎模块
混合架构版本：结合自定义PDF解析 + llama-index管理
"""

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from astrbot.api import logger

# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from .hybrid_rag import HybridRAGEngine

HYBRID_RAG_AVAILABLE = True


@dataclass
class RAGConfig:
    """RAG配置类"""
    # Embedding配置
    embedding_mode: str = "ollama"  # "api" 或 "ollama"
    embedding_provider_id: str = ""  # API模式下的Provider

    # GLM配置
    glm_api_key: str = ""  # 智谱AI API密钥
    glm_model: str = "glm-4.7-flash"  # GLM文本模型
    glm_multimodal_model: str = "glm-4.6v-flash"  # 多模态模型

    # Ollama配置
    ollama_config: dict = field(default_factory=dict)

    # Milvus配置
    milvus_lite_path: str = ""  # Lite 模式路径
    address: str = ""  # 远程 Milvus 服务器地址
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
    figures_dir: str = ""  # 空则使用插件目录下的 data/figures

    # 重排序配置
    enable_reranking: bool = False
    reranking_model: str = "BAAI/bge-reranker-v2-m3"
    reranking_device: str = "auto"
    reranking_adaptive: bool = True
    reranking_threshold: float = 0.0
    reranking_batch_size: int = 32

    def __post_init__(self):
        """初始化后处理"""
        if self.authentication is None:
            self.authentication = {}
        if self.ollama_config is None:
            self.ollama_config = {}

        # 自动调整embed_dim（根据embedding_mode和模型）
        if self.embedding_mode == "ollama":
            model = self.ollama_config.get("model", "bge-m3")
            if model == "bge-m3":
                self.embed_dim = 1024
            elif model == "nomic-embed-text":
                self.embed_dim = 768

    def validate(self) -> tuple[bool, str]:
        """验证配置"""
        logger.debug(f"🔍 [DEBUG] validate: milvus_lite_path='{self.milvus_lite_path}', address='{self.address}'")
        if self.embed_dim % 64 != 0:
            return False, "嵌入维度必须是64的倍数"

        # 检查 Milvus 配置 - 留空让 HybridIndexManager 使用插件目录下的默认路径
        # if not self.milvus_lite_path and not self.address:
        #     self.milvus_lite_path = ""  # 留空让管理器使用默认

        logger.debug(f"🔍 [DEBUG] validate after: milvus_lite_path='{self.milvus_lite_path}'")
        return True, ""

    def get_connection_mode(self) -> str:
        """获取连接模式：'lite' 或 'remote'"""
        # milvus_lite_path 优先级更高
        logger.debug(f"🔍 [DEBUG] get_connection_mode: milvus_lite_path='{self.milvus_lite_path}', address='{self.address}'")
        if self.milvus_lite_path and self.milvus_lite_path.strip():
            return 'lite'
        elif self.address and self.address.strip():
            return 'remote'
        else:
            # 默认使用 Lite 模式
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
    创建RAG引擎实例（混合架构版本）

    Args:
        config: RAG配置
        context: AstrBot上下文

    Returns:
        RAG引擎实例（HybridRAGEngine）

    Example:
        >>> engine = create_rag_engine(config, context)
        >>> result = await engine.search("attention机制")
    """
    # 延迟导入避免循环依赖
    from .hybrid_rag import HybridRAGEngine

    logger.info("✅ 使用混合架构 RAG引擎（HybridRAGEngine）")
    logger.info("   - 自定义PDF解析（多模态）")
    logger.info("   - 语义分块")
    logger.info("   - Milvus向量存储")
    logger.info("   - GLM LLM生成")

    return HybridRAGEngine(config, context)
