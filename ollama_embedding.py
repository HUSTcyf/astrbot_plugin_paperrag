"""
Ollama Embedding Provider - 通过HTTP API调用本地Ollama服务
避免直接加载模型导致的内存和进程冲突问题
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import httpx

from astrbot.api import logger

@dataclass
class OllamaEmbeddingConfig:
    """Ollama Embedding配置"""
    base_url: str = "http://localhost:11434"  # Ollama服务地址
    model: str = "bge-m3"  # 模型名称
    timeout: float = 120.0  # 请求超时时间（秒）
    batch_size: int = 10  # 并发批处理大小
    retry_attempts: int = 3  # 重试次数
    retry_delay: float = 1.0  # 重试延迟（秒）


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
                    # 服务器错误，可能Ollama服务未启动
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

    async def get_text_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本embeddings（兼容接口）"""
        return await self.get_embeddings(texts)

    async def get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的embedding（兼容接口）"""
        result = await self.embed([text])
        return result[0] if result and len(result) > 0 else []

    async def get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入（兼容接口）"""
        return await self.get_text_embedding(query)

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
    """创建Ollama Embedding Provider

    Args:
        base_url: Ollama服务地址，默认 http://localhost:11434
        model: 模型名称，默认 bge-m3
        timeout: 请求超时时间（秒），默认120
        batch_size: 并发批处理大小，默认10
        retry_attempts: 重试次数，默认3

    Returns:
        OllamaEmbeddingProvider实例

    Example:
        >>> provider = create_ollama_provider(model="bge-m3")
        >>> embeddings = await provider.get_embeddings(["hello", "world"])
    """
    config = OllamaEmbeddingConfig(
        base_url=base_url,
        model=model,
        timeout=timeout,
        batch_size=batch_size,
        retry_attempts=retry_attempts
    )

    logger.info(
        f"🦙 初始化Ollama Embedding Provider\n"
        f"   - 服务地址: {base_url}\n"
        f"   - 模型: {model}\n"
        f"   - 并发度: {batch_size}\n"
        f"   - 超时: {timeout}秒"
    )

    return OllamaEmbeddingProvider(config)


async def test_ollama_connection(
    base_url: str = "http://localhost:11434",
    model: str = "bge-m3"
) -> bool:
    """测试Ollama连接和模型可用性

    Args:
        base_url: Ollama服务地址
        model: 模型名称

    Returns:
        bool: True表示连接成功且模型可用

    Example:
        >>> if await test_ollama_connection():
        ...     print("✅ Ollama服务正常")
    """
    try:
        provider = create_ollama_provider(base_url=base_url, model=model)
        test_embedding = await provider._embed_single("test")
        return len(test_embedding) > 0
    except Exception as e:
        logger.error(f"❌ Ollama连接测试失败: {e}")
        return False
    finally:
        if 'provider' in locals():
            await provider._close()


if __name__ == "__main__":
    """测试Ollama Embedding功能"""
    import sys

    async def main():
        print("=== Ollama Embedding Provider 测试 ===\n")

        # 1. 测试连接
        print("1. 测试Ollama连接...")
        if not await test_ollama_connection():
            print(
                "❌ 连接失败！请确保：\n"
                "   1. Ollama服务正在运行: ollama serve\n"
                "   2. BGE-M3模型已下载: ollama pull bge-m3"
            )
            sys.exit(1)
        print("✅ 连接成功\n")

        # 2. 测试单个embedding
        print("2. 测试单个embedding...")
        provider = create_ollama_provider(model="bge-m3")
        embedding = await provider._embed_single("Hello, world!")
        print(f"✅ 向量维度: {len(embedding)}")
        print(f"   前5个值: {embedding[:5]}\n")

        # 3. 测试批量embedding
        print("3. 测试批量embedding...")
        texts = ["Hello", "World", "Ollama", "Embedding", "Test"]
        embeddings = await provider.get_embeddings(texts)
        print(f"✅ 成功处理 {len(embeddings)} 个文本")
        print(f"   向量维度: {len(embeddings[0])}\n")

        # 4. 测试并发性能
        print("4. 测试并发性能...")
        import time
        start = time.time()
        texts = [f"Text {i}" for i in range(50)]
        embeddings = await provider.get_embeddings(texts)
        elapsed = time.time() - start
        print(f"✅ 50个文本耗时: {elapsed:.2f}秒")
        print(f"   平均速度: {50/elapsed:.1f} 文本/秒\n")

        await provider._close()
        print("=== 测试完成 ===")

    asyncio.run(main())
