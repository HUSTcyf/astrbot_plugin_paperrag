"""
统一的 Embedding Provider 管理
支持 Unsloth 本地 BGE-M3（稀疏权重 + 稠密向量 + 多向量）
"""

from typing import List, Dict, Any, Optional, Union

from astrbot.api import logger

from .unsloth_embedding import (
    UnslothEmbeddingModel,
    init_embedding_model,
)


# ============================================================================
# Unsloth Embedding Provider
# ============================================================================

class UnslothEmbeddingProvider:
    """
    Unsloth BGE-M3 Embedding Provider

    使用 Unsloth 本地加载 BGE-M3，提供：
    - 稠密向量 (dense vector) - 用于 Milvus 检索
    - 稀疏权重 (sparse weight) - 替代 BM25 关键词检索
    - 多向量序列 (multi-vector) - 用于 ColBERT 式 reranking
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "mps",
        max_seq_length: int = 512,
    ):
        """
        Args:
            model_path: 模型路径，默认使用 ./models/bge-m3
            device: 运行设备 ("mps", "cuda", "cpu")
            max_seq_length: 最大序列长度
        """
        self.model_path = model_path
        self.device = device
        self.max_seq_length = max_seq_length
        self._model: Optional[UnslothEmbeddingModel] = None
        self._initialized = False

    async def initialize(self) -> None:
        """异步初始化模型"""
        if self._initialized:
            return

        self._model = await init_embedding_model(
            model_path=self.model_path,
            device=self.device,
            max_seq_length=self.max_seq_length,
        )
        self._initialized = True
        logger.info(f"[UnslothEmbeddingProvider] 初始化完成，维度: {self._model.embedding_dim}")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本embeddings

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 归一化的稠密向量列表
        """
        if not self._initialized:
            await self.initialize()

        assert self._model is not None
        return self._model.get_dense_embedding(texts)

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        兼容接口：批量获取embeddings

        Args:
            texts: 单个文本或文本列表

        Returns:
            List[List[float]]: 稠密向量列表
        """
        if isinstance(texts, str):
            texts = [texts]
        return await self.get_embeddings(texts)

    async def get_text_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的embedding

        Args:
            text: 文本字符串

        Returns:
            List[float]: 稠密向量
        """
        result = await self.embed([text])
        return result[0] if result else []

    async def get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询嵌入

        Args:
            query: 查询文本

        Returns:
            List[float]: 稠密向量
        """
        return await self.get_text_embedding(query)

    async def get_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本embeddings（兼容接口）

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 稠密向量列表
        """
        return await self.get_embeddings(texts)

    async def get_sparse_weight(self, text: str, query_embedding: Optional[List[float]] = None) -> Dict[int, float]:
        """
        获取稀疏权重（用于关键词检索）

        Args:
            text: 文本字符串
            query_embedding: 可选的查询向量

        Returns:
            Dict[int, float]: token_id -> weight
        """
        if not self._initialized:
            await self.initialize()

        assert self._model is not None
        return self._model.get_sparse_weight(text, query_embedding)

    async def get_multi_vector(self, text: str) -> List[List[float]]:
        """
        获取多向量序列（用于 ColBERT reranking）

        Args:
            text: 文本字符串

        Returns:
            List[List[float]]: 每个 token 的向量列表
        """
        if not self._initialized:
            await self.initialize()

        assert self._model is not None
        return self._model.get_multi_vector(text)

    async def colbert_rerank(
        self,
        query_text: str,
        doc_texts: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        ColBERT 式 reranking

        Args:
            query_text: 查询文本
            doc_texts: 文档文本列表
            top_k: 返回 top-k 结果

        Returns:
            List[tuple[int, float]]: (doc_index, score)
        """
        if not self._initialized:
            await self.initialize()

        assert self._model is not None
        return self._model.colbert_rerank(query_text, doc_texts, top_k)

    @property
    def embed_dim(self) -> int:
        """获取向量维度"""
        if self._model is not None:
            return self._model.embedding_dim
        # 默认 BGE-M3 1024 维
        return 1024


def create_unsloth_provider(
    model_path: Optional[str] = None,
    device: str = "mps",
    max_seq_length: int = 512,
) -> UnslothEmbeddingProvider:
    """
    创建 Unsloth Embedding Provider

    Args:
        model_path: 模型路径，默认 ./models/bge-m3
        device: 运行设备 ("mps", "cuda", "cpu")
        max_seq_length: 最大序列长度

    Returns:
        UnslothEmbeddingProvider 实例

    Example:
        >>> provider = create_unsloth_provider(device="mps")
        >>> await provider.initialize()
        >>> embeddings = await provider.get_embeddings(["hello", "world"])
    """
    logger.info(
        f"[UnslothEmbeddingProvider] 创建 Provider\n"
        f"   - 模型路径: {model_path or './models/bge-m3'}\n"
        f"   - 设备: {device}\n"
        f"   - 最大序列长度: {max_seq_length}"
    )
    return UnslothEmbeddingProvider(
        model_path=model_path,
        device=device,
        max_seq_length=max_seq_length,
    )




# ============================================================================
# AstrBot Embedding Provider（API）
# ============================================================================

class EmbeddingProviderWrapper:
    """AstrBot Embedding Provider 包装类 - 支持 OpenAI、Gemini 等API Provider"""

    def __init__(self, provider: Any):
        if not provider:
            raise ValueError("Embedding provider 不能为 None")
        self.provider = provider

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """批量获取文本嵌入"""
        try:
            if isinstance(texts, str):
                texts = [texts]

            # Gemini API 限制：单次批量请求最多100个
            BATCH_SIZE_LIMIT = 100

            if len(texts) > BATCH_SIZE_LIMIT:
                logger.info(f"📊 文本数量超过API限制 ({len(texts)}>{BATCH_SIZE_LIMIT})，自动分批处理")
                all_embeddings = []
                for i in range(0, len(texts), BATCH_SIZE_LIMIT):
                    batch = texts[i:i + BATCH_SIZE_LIMIT]
                    batch_embeddings = await self._embed_batch(batch)
                    all_embeddings.extend(batch_embeddings)
                return all_embeddings
            else:
                return await self._embed_batch(texts)

        except Exception as e:
            logger.error(f"❌ Embedding 失败: {e}")
            raise

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入（内部方法）"""
        try:
            if hasattr(self.provider, 'get_embeddings'):
                response = await self.provider.get_embeddings(texts)
                if hasattr(response, 'embeddings'):
                    return [e.values for e in response.embeddings]
                elif isinstance(response, list):
                    return response

            # 逐个调用
            embeddings = []
            for text in texts:
                if hasattr(self.provider, 'get_embedding'):
                    response = await self.provider.get_embedding(text)
                    if hasattr(response, 'values'):
                        embeddings.append(response.values)
                    elif isinstance(response, list):
                        embeddings.append(response)
                    else:
                        embeddings.append(response)
                elif hasattr(self.provider, 'embed'):
                    result = await self.provider.embed([text])
                    embeddings.append(result[0] if result else [])

            return embeddings

        except Exception as e:
            logger.error(f"❌ 批量嵌入失败: {e}")
            raise


class AstrBotEmbeddingProvider:
    """AstrBot API Embedding Provider"""

    def __init__(self, wrapper: EmbeddingProviderWrapper):
        self._wrapper = wrapper

    async def initialize(self) -> None:
        """AstrBot provider 已由 framework 初始化，此处为空操作"""
        pass

    @classmethod
    def from_context(
        cls,
        context: Any,
        provider_id: str,
        embed_batch_size: int = 10,
    ):
        """从 AstrBot context 创建实例"""
        provider_manager = getattr(context, "provider_manager", None)
        if provider_manager is None:
            raise ValueError("无法访问 context.provider_manager")

        inst_map = getattr(provider_manager, "inst_map", None)
        if not isinstance(inst_map, dict):
            raise ValueError("inst_map 不是 dict")

        provider = inst_map.get(provider_id)
        if provider is None:
            for pid, prov in inst_map.items():
                if hasattr(prov, 'get_embeddings') or hasattr(prov, 'embed'):
                    provider = prov
                    provider_id = pid
                    logger.info(f"✅ 使用第一个可用的 Embedding Provider: {provider_id}")
                    break

        if provider is None:
            raise ValueError(f"未找到可用的 Embedding Provider")

        wrapper = EmbeddingProviderWrapper(provider)
        logger.info(f"✅ 从 context 加载 Embedding Provider: {provider_id}")

        return cls(wrapper)

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """批量获取文本嵌入"""
        if isinstance(texts, str):
            texts = [texts]
        return await self._wrapper.embed(texts)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本 embeddings（兼容接口）"""
        return await self.embed(texts)

    async def get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding"""
        result = await self.embed([text])
        return result[0] if result else []

    async def get_query_embedding(self, query: str) -> List[float]:
        """获取查询的 embedding"""
        return await self.get_text_embedding(query)

    async def get_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本 embeddings"""
        return await self.embed(texts)

    @property
    def embed_dim(self) -> int:
        """获取向量维度（需要从 provider 获取）"""
        return 768  # Gemini 默认


# ============================================================================
# 统一的 Embedding Provider 工厂
# ============================================================================

class EmbeddingProviderType:
    """Embedding Provider 类型"""
    UNSLOTH = "unsloth"  # Unsloth 本地加载
    ASTRBOT = "astrbot"


def create_embedding_provider(
    mode: str = "unsloth",
    context: Any = None,
    provider_id: str = "",
    **kwargs,
) -> Union[UnslothEmbeddingProvider, AstrBotEmbeddingProvider]:
    """
    创建 Embedding Provider 的工厂函数

    Args:
        mode: Embedding 模式 ("unsloth", "astrbot")
        context: AstrBot 上下文（astrbot 模式需要）
        provider_id: Provider ID（astrbot 模式需要）
        **kwargs: 其他参数

    Returns:
        Embedding Provider 实例
    """
    if mode == EmbeddingProviderType.UNSLOTH:
        return create_unsloth_provider(
            model_path=kwargs.get("model_path"),
            device=kwargs.get("device", "mps"),
            max_seq_length=kwargs.get("max_seq_length", 512),
        )

    elif mode == EmbeddingProviderType.ASTRBOT:
        if not context:
            raise ValueError("AstrBot 模式需要 context")

        return AstrBotEmbeddingProvider.from_context(
            context=context,
            provider_id=provider_id or "gemini_embedding",
            embed_batch_size=kwargs.get("embed_batch_size", 10),
        )

    else:
        raise ValueError(f"无效的 Embedding 模式: {mode}")
