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
    retry_delay: float = 2.0
    # LLM压缩配置（文本超限时使用）
    use_llm_compress: bool = True
    compress_provider: Any = None  # LLM Provider（用于文本压缩）
    compress_max_chars: int = 6400


class OllamaEmbeddingProvider:
    """Ollama Embedding Provider - 通过HTTP API调用本地Ollama服务"""

    def __init__(self, config: OllamaEmbeddingConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._embed_dim: Optional[int] = None
        self._llm_client: Optional[Any] = None

    async def _get_llm_client(self) -> Any:
        """获取LLM客户端（用于文本压缩）"""
        if self._llm_client is None and self.config.use_llm_compress and self.config.compress_provider:
            self._llm_client = self.config.compress_provider
            logger.info("✅ 使用配置的LLM Provider进行文本压缩")
        return self._llm_client

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

    async def _embed_single(self, text: str):
        """获取单个文本的embedding"""
        client = await self._get_client()

        # 处理文本：清理 -> 压缩（如需要）
        text = await self._process_text(text)

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
                    # 500 错误时增加重试间隔
                    # 记录错误文本信息以便调试
                    text_preview = text[:100] + "..." if len(text) > 100 else text
                    logger.warning(f"Ollama 500 错误 (尝试 {attempt + 1}/{self.config.retry_attempts}): 文本长度={len(text)}, 内容: {text_preview}")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        raise Exception(
                            f"Ollama服务错误 (500)。文本长度={len(text)}。"
                            f"可能是文本过长或包含特殊字符。可尝试: pkill -f 'ollama serve' && ollama serve"
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

    def _sanitize_text(self, text: str) -> str:
        """清理文本，避免特殊字符导致 Ollama 处理失败"""
        # 移除 null 字节
        text = text.replace('\x00', '')
        # 移除可能导致问题的控制字符
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        # 压缩多余空白
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def _compress_text(self, text: str) -> str:
        """
        使用LLM压缩文本，保留核心语义信息

        Args:
            text: 原始文本

        Returns:
            压缩后的文本（纯字符串）
        """
        llm_provider = await self._get_llm_client()
        if not llm_provider:
            # 如果没有LLM客户端，直接截断
            return text[:self.config.compress_max_chars]

        try:
            compress_prompt = f"""请将以下文本压缩到不超过6400字符，保留核心语义信息和关键细节。

原文：
{text}

压缩后的文本（直接输出，不要解释）："""

            # 优先使用 text_chat 接口（AstrBot 标准接口）
            if hasattr(llm_provider, 'text_chat'):
                response = await llm_provider.text_chat(
                    prompt=compress_prompt,
                    contexts=[],
                    temperature=0.3,
                    max_tokens=4000
                )
                # 安全提取纯文本内容
                if hasattr(response, 'choices') and hasattr(response.choices[0], 'message'):
                    content = response.choices[0].message.content
                    # 确保返回纯字符串
                    compressed = str(content).strip() if content else ""
                else:
                    compressed = str(response).strip()
            # 回退到 chat.completions.create 接口
            elif hasattr(llm_provider, 'chat'):
                response = await llm_provider.chat.completions.create(
                    messages=[{"role": "user", "content": compress_prompt}],
                    max_tokens=4000,
                    temperature=0.3
                )
                compressed = str(response.choices[0].message.content).strip()
            else:
                raise ValueError("Provider不支持 text_chat 或 chat 接口")

            logger.debug(f"📝 文本压缩: {len(text)} -> {len(compressed)} 字符")
            return compressed
        except Exception as e:
            logger.warning(f"⚠️ LLM压缩失败: {e}，使用直接截断")
            return text[:self.config.compress_max_chars]

    async def _process_text(self, text: str) -> str:
        """
        处理文本：清理 -> 压缩（如需要，支持多次压缩直到合适长度）

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        # 记录原始长度
        original_len = len(text)

        # 基础清理
        text = self._sanitize_text(text)

        # 如果超过限制且启用了LLM压缩，递归压缩直到合适长度
        max_retries = 3
        retry_count = 0
        if len(text) > self.config.compress_max_chars and self.config.use_llm_compress:
            while len(text) > self.config.compress_max_chars and retry_count < max_retries:
                logger.debug(f"📝 文本超限({len(text)}>{self.config.compress_max_chars})，使用LLM压缩 (第{retry_count + 1}次)")
                text = await self._compress_text(text)
                retry_count += 1

        # 记录处理后长度
        if original_len > self.config.compress_max_chars:
            logger.info(f"📊 文本压缩: {original_len} → {len(text)} 字符 (压缩了 {original_len - len(text)} 字符, 压缩次数: {retry_count})")

        return text

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
    retry_attempts: int = 3,
    use_llm_compress: bool = True,
    compress_provider: Any = None,
    compress_max_chars: int = 6400
) -> OllamaEmbeddingProvider:
    """
    创建Ollama Embedding Provider

    Args:
        base_url: Ollama服务地址，默认 http://localhost:11434
        model: 模型名称，默认 bge-m3
        timeout: 请求超时时间（秒），默认120
        batch_size: 并发批处理大小，默认10
        retry_attempts: 重试次数，默认3
        use_llm_compress: 文本超限时使用LLM压缩
        compress_provider: LLM Provider实例（用于文本压缩）
        compress_max_chars: 触发压缩的最大字符数

    Returns:
        OllamaEmbeddingProvider实例
    """
    config = OllamaEmbeddingConfig(
        base_url=base_url,
        model=model,
        timeout=timeout,
        batch_size=batch_size,
        retry_attempts=retry_attempts,
        use_llm_compress=use_llm_compress,
        compress_provider=compress_provider,
        compress_max_chars=compress_max_chars
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
    provider_id: str = '',
    ollama_config: dict = {},
    compress_provider_id: str = '',
    **kwargs
) -> Union[OllamaEmbeddingProvider, AstrBotEmbeddingProvider]:
    """
    创建 Embedding Provider 的工厂函数

    Args:
        mode: Embedding 模式 ("ollama" 或 "astrbot")
        context: AstrBot 上下文（astrbot 模式需要）
        provider_id: Provider ID（astrbot 模式需要）
        ollama_config: Ollama 配置（ollama 模式需要）
        compress_provider_id: LLM Provider ID（用于文本压缩）
        **kwargs: 其他参数

    Returns:
        Embedding Provider 实例

    Raises:
        ValueError: 无效的模式或缺少必要参数
    """
    if mode == EmbeddingProviderType.OLLAMA:
        if not ollama_config:
            raise ValueError("Ollama 模式需要 ollama_config")

        # 获取压缩用的 LLM Provider
        compress_provider = None
        if compress_provider_id and context:
            try:
                compress_provider = context.get_provider_by_id(compress_provider_id)
                if compress_provider:
                    logger.info(f"✅ 使用 LLM Provider '{compress_provider_id}' 进行文本压缩")
            except Exception as e:
                logger.warning(f"⚠️ 无法获取压缩Provider '{compress_provider_id}': {e}")

        return create_ollama_provider(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            model=ollama_config.get("model", "bge-m3"),
            timeout=ollama_config.get("timeout", 120.0),
            batch_size=ollama_config.get("batch_size", 10),
            retry_attempts=ollama_config.get("retry_attempts", 3),
            use_llm_compress=ollama_config.get("use_llm_compress", True),
            compress_provider=compress_provider,
            compress_max_chars=ollama_config.get("compress_max_chars", 6400)
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
