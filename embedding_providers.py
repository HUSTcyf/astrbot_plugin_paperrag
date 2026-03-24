"""
统一的 Embedding Provider 管理
支持多种 Embedding 方式：Ollama 本地、AstrBot API
避免依赖 llama-index 的全局状态
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import httpx

from astrbot.api import logger


# ============================================================================
# Ollama Embedding Provider（本地）
# ============================================================================

@dataclass
class OllamaEmbeddingConfig:
    """Ollama Embedding配置"""
    base_url: str = "http://localhost:11434"
    model: str = "bge-m3"
    timeout: float = 120.0
    batch_size: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0


class OllamaEmbeddingProvider:
    """Ollama Embedding Provider - 通过HTTP API调用本地Ollama服务"""

    def __init__(self, config: OllamaEmbeddingConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._embed_dim: Optional[int] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端（延迟初始化）"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout)
            )
        return self._client

    async def _close(self):
        """关闭HTTP客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _embed_single(self, text: str) -> List[float]:
        """获取单个文本的embedding"""
        client = await self._get_client()

        for attempt in range(self.config.retry_attempts):
            try:
                response = await client.post(
                    "/api/embeddings",
                    json={
                        "model": self.config.model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                result = response.json()

                if "embedding" not in result:
                    raise ValueError(f"Ollama响应缺少embedding字段: {result}")

                embedding = result["embedding"]

                # 缓存向量维度
                if self._embed_dim is None:
                    self._embed_dim = len(embedding)
                    logger.info(f"✅ Ollama Embedding向量维度: {self._embed_dim}")

                return embedding

            except httpx.HTTPStatusError as e:
                logger.warning(f"Ollama请求失败 (尝试 {attempt + 1}/{self.config.retry_attempts}): {e}")
                if e.response.status_code == 404:
                    raise Exception(
                        f"Ollama模型 '{self.config.model}' 不存在。"
                        f"请先运行: ollama pull {self.config.model}"
                    )
                elif e.response.status_code == 500:
                    raise Exception(
                        f"Ollama服务错误 (500)。请确保Ollama服务正在运行: "
                        f"ollama serve"
                    )
                elif attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    raise

            except httpx.ConnectError:
                logger.warning(f"无法连接到Ollama服务 (尝试 {attempt + 1}/{self.config.retry_attempts})")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    raise Exception(
                        f"无法连接到Ollama服务 ({self.config.base_url})。"
                        f"请确保Ollama服务正在运行: ollama serve"
                    )

            except Exception as e:
                logger.error(f"Ollama embedding请求失败: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    raise

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本embeddings（并发处理）"""
        if not texts:
            return []

        logger.info(f"🦙 Ollama批量处理: {len(texts)} 个文本（模型: {self.config.model}）")

        # 分批并发处理
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

            logger.debug(
                f"🔄 处理批次 {batch_num}/{total_batches} "
                f"({len(batch)} 个文本，并发度: {self.config.batch_size})"
            )

            # 并发处理当前批次
            tasks = [self._embed_single(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            # 检查异常
            for j, result in enumerate(batch_embeddings):
                if isinstance(result, Exception):
                    logger.error(f"批次 {batch_num} 第 {j+1} 个文本失败: {result}")
                    raise result

            all_embeddings.extend(batch_embeddings)

        logger.info(f"✅ Ollama完成: {len(all_embeddings)} 个向量（维度: {self._embed_dim or '未知'}）")
        return all_embeddings

    async def embed(self, texts: str | List[str]) -> List[List[float]]:
        """兼容接口：批量获取embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        return await self.get_embeddings(texts)

    async def get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的embedding"""
        result = await self.embed([text])
        return result[0] if result and len(result) > 0 else []

    async def get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入"""
        return await self.get_text_embedding(query)

    async def get_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本embeddings（兼容接口）"""
        return await self.get_embeddings(texts)

    @property
    def embed_dim(self) -> int:
        """获取向量维度"""
        if self._embed_dim is None:
            # BGE-M3默认1024维
            return 1024
        return self._embed_dim


def create_ollama_provider(
    base_url: str = "http://localhost:11434",
    model: str = "bge-m3",
    timeout: float = 120.0,
    batch_size: int = 10,
    retry_attempts: int = 3
) -> OllamaEmbeddingProvider:
    """
    创建Ollama Embedding Provider

    Args:
        base_url: Ollama服务地址，默认 http://localhost:11434
        model: 模型名称，默认 bge-m3
        timeout: 请求超时时间（秒），默认120
        batch_size: 并发批处理大小，默认10
        retry_attempts: 重试次数，默认3

    Returns:
        OllamaEmbeddingProvider实例
    """
    config = OllamaEmbeddingConfig(
        base_url=base_url,
        model=model,
        timeout=timeout,
        batch_size=batch_size,
        retry_attempts=retry_attempts
    )
    return OllamaEmbeddingProvider(config)


# ============================================================================
# AstrBot Embedding Provider（API）
# ============================================================================

class EmbeddingProviderWrapper:
    """AstrBot Embedding Provider 包装类 - 支持 OpenAI、Gemini 等API Provider"""

    def __init__(self, provider: Any):
        if not provider:
            raise ValueError("Embedding provider 不能为 None")
        self.provider = provider

    async def embed(self, texts: str | List[str]) -> List[List[float]]:
        """批量获取文本嵌入（优先使用批量API，自动分批以符合API限制）"""
        try:
            if isinstance(texts, str):
                texts = [texts]

            # Gemini API限制：单次批量请求最多100个
            BATCH_SIZE_LIMIT = 100

            # 如果超过限制，分批处理
            if len(texts) > BATCH_SIZE_LIMIT:
                logger.info(f"📊 文本数量超过API限制 ({len(texts)}>{BATCH_SIZE_LIMIT})，自动分批处理")
                all_embeddings = []

                for i in range(0, len(texts), BATCH_SIZE_LIMIT):
                    batch = texts[i:i + BATCH_SIZE_LIMIT]
                    logger.debug(f"处理批次 {i//BATCH_SIZE_LIMIT + 1}/{(len(texts) + BATCH_SIZE_LIMIT - 1)//BATCH_SIZE_LIMIT}")
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
            # 尝试使用批量 API（如果存在）
            if hasattr(self.provider, 'get_embeddings'):
                # Gemini 风格
                response = await self.provider.get_embeddings(texts)  # type: ignore
                if hasattr(response, 'embeddings'):
                    return [e.values for e in response.embeddings]  # type: ignore
                elif isinstance(response, list):
                    return response

            # 如果没有批量API，逐个调用
            embeddings = []
            for text in texts:
                if hasattr(self.provider, 'get_embedding'):
                    # 单个 embedding
                    response = await self.provider.get_embedding(text)  # type: ignore
                    if hasattr(response, 'values'):
                        embeddings.append(response.values)  # type: ignore
                    elif isinstance(response, list):
                        embeddings.append(response)
                    else:
                        embeddings.append(response)
                elif hasattr(self.provider, 'embed'):
                    # 统一接口
                    result = await self.provider.embed([text])  # type: ignore
                    embeddings.append(result[0] if result else [])

            return embeddings

        except Exception as e:
            logger.error(f"❌ 批量嵌入失败: {e}")
            raise


class AstrBotEmbeddingProvider:
    """AstrBot API Embedding Provider - 使用 AstrBot 配置的 Embedding Provider"""

    def __init__(self, wrapper: EmbeddingProviderWrapper):
        """
        初始化 AstrBot Embedding Provider

        Args:
            wrapper: EmbeddingProviderWrapper 实例
        """
        self._wrapper = wrapper

    @classmethod
    def from_context(
        cls,
        context: Any,
        provider_id: str,
        embed_batch_size: int = 10
    ):
        """
        从 AstrBot context 创建实例

        Args:
            context: AstrBot 上下文
            provider_id: Embedding Provider ID
            embed_batch_size: 批处理大小
        """
        provider_manager = getattr(context, "provider_manager", None)
        if provider_manager is None:
            raise ValueError("无法访问 context.provider_manager")

        inst_map = getattr(provider_manager, "inst_map", None)
        if not isinstance(inst_map, dict):
            raise ValueError("inst_map 不是 dict")

        provider = inst_map.get(provider_id)
        if provider is None:
            # 尝试使用第一个可用的 embedding provider
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

    async def embed(self, texts: str | List[str]) -> List[List[float]]:
        """批量获取文本嵌入"""
        if isinstance(texts, str):
            texts = [texts]
        return await self._wrapper.embed(texts)

    async def get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding"""
        result = await self.embed([text])
        return result[0] if result and len(result) > 0 else []

    async def get_query_embedding(self, query: str) -> List[float]:
        """获取查询的 embedding"""
        return await self.get_text_embedding(query)

    async def get_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本 embeddings"""
        return await self.embed(texts)

    @property
    def embed_dim(self) -> int:
        """获取向量维度（需要从 provider 获取）"""
        # Gemini 默认 768 维
        return 768


# ============================================================================
# 统一的 Embedding Provider 工厂
# ============================================================================

class EmbeddingProviderType:
    """Embedding Provider 类型"""
    OLLAMA = "ollama"
    ASTRBOT = "astrbot"


def create_embedding_provider(
    mode: str,
    context: Any = None,
    provider_id: str = None,
    ollama_config: dict = None,
    **kwargs
) -> Union[OllamaEmbeddingProvider, AstrBotEmbeddingProvider]:
    """
    创建 Embedding Provider 的工厂函数

    Args:
        mode: Embedding 模式 ("ollama" 或 "astrbot")
        context: AstrBot 上下文（astrbot 模式需要）
        provider_id: Provider ID（astrbot 模式需要）
        ollama_config: Ollama 配置（ollama 模式需要）
        **kwargs: 其他参数

    Returns:
        Embedding Provider 实例

    Raises:
        ValueError: 无效的模式或缺少必要参数
    """
    if mode == EmbeddingProviderType.OLLAMA:
        if not ollama_config:
            raise ValueError("Ollama 模式需要 ollama_config")

        return create_ollama_provider(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            model=ollama_config.get("model", "bge-m3"),
            timeout=ollama_config.get("timeout", 120.0),
            batch_size=ollama_config.get("batch_size", 10),
            retry_attempts=ollama_config.get("retry_attempts", 3)
        )

    elif mode == EmbeddingProviderType.ASTRBOT:
        if not context:
            raise ValueError("AstrBot 模式需要 context")

        return AstrBotEmbeddingProvider.from_context(
            context=context,
            provider_id=provider_id or "gemini_embedding",
            embed_batch_size=kwargs.get("embed_batch_size", 10)
        )

    else:
        raise ValueError(f"无效的 Embedding 模式: {mode}")


# ============================================================================
# 类型别名（向后兼容）
# ============================================================================

# 向后兼容：旧的类型名称
OllamaEmbedding = OllamaEmbeddingProvider
AstrBotEmbedding = AstrBotEmbeddingProvider
