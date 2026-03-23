"""
RAG重排序模块
基于FlagEmbedding框架实现文档重排序，提升检索精度
支持MPS加速（Apple Silicon GPU）
"""

import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from astrbot.api import logger

# FlagEmbedding导入（优雅降级）
if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker as FlagRerankerType
else:
    FlagRerankerType = object  # type: ignore

try:
    from FlagEmbedding import FlagReranker
    FLAGRERANKER_AVAILABLE: bool = True
except ImportError:
    # 类型占位符，避免Pylance报告未绑定变量
    FlagReranker: Optional[type] = None  # type: ignore
    FLAGRERANKER_AVAILABLE = False
    logger.warning("⚠️ FlagEmbedding未安装，重排序功能将被禁用。安装: pip install -U FlagEmbedding")


@dataclass
class RerankerConfig:
    """重排序器配置"""
    model_name: str = "BAAI/bge-reranker-v2-m3"  # 多语言重排序模型
    device: str = "auto"  # 自动选择设备（优先MPS>CUDA>CPU）
    batch_size: int = 32  # 批处理大小（从配置文件读取）
    max_length: int = 512  # 最大序列长度
    use_fp16: bool = True  # 使用FP16精度（加速推理）
    score_threshold: float = 0.0  # 最低分数阈值


class ContentReranker:
    """
    文档重排序器
    使用交叉注意力模型精确计算查询-文档相关性
    """

    def __init__(self, config: RerankerConfig):
        self.config = config
        self._model: Optional[FlagRerankerType] = None
        self._initialized = False
        self._available: bool = FLAGRERANKER_AVAILABLE

    def _ensure_initialized(self):
        """延迟初始化模型"""
        if self._initialized or not FLAGRERANKER_AVAILABLE:
            return

        # 运行时检查FlagReranker是否可用
        if FlagReranker is None:  # type: ignore
            return

        try:
            import torch

            # 设备选择逻辑
            if self.config.device == "auto":
                if torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("🚀 使用MPS加速（Apple Silicon GPU）")
                elif torch.cuda.is_available():
                    device = "cuda"
                    logger.info("🚀 使用CUDA加速（NVIDIA GPU）")
                else:
                    device = "cpu"
                    logger.info("⚠️ 使用CPU运行重排序")
            else:
                device = self.config.device

            # 初始化重排序模型（类型断言，因为已检查FlagReranker不为None）
            assert FlagReranker is not None  # 让类型检查器知道FlagReranker可用
            self._model = FlagReranker(
                self.config.model_name,
                device=device,
                use_fp16=self.config.use_fp16
            )

            self._initialized = True
            logger.info(f"✅ 重排序模型加载完成: {self.config.model_name}")

        except Exception as e:
            logger.error(f"❌ 重排序模型初始化失败: {e}")
            self._initialized = False
            # 不要修改全局变量，使用实例变量标记失败
            self._available = False

    def is_available(self) -> bool:
        """检查重排序器是否可用"""
        return self._available and self._initialized and self._model is not None

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        重排序文档列表

        Args:
            query: 查询文本
            results: 文档列表，格式: [{"text": "...", "metadata": {}, "score": 0.8}, ...]
            top_k: 返回前K个结果，None表示返回所有

        Returns:
            重排序后的文档列表（按相关性降序）
        """
        if not self.is_available():
            logger.debug("重排序不可用，返回原始顺序")
            return results

        if not results:
            return []

        self._ensure_initialized()

        try:
            # 提取文档文本
            doc_texts = [doc["text"] for doc in results]

            # 构建查询-文档对
            pairs = [[query, doc_text] for doc_text in doc_texts]

            # 批量计算相关性分数
            scores = await self._compute_scores(pairs)

            # 更新文档分数
            reranked_docs = []
            for doc, score in zip(results, scores):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(score)
                reranked_docs.append(doc_copy)

            # 按重排序分数降序排列
            reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

            # 过滤低分结果
            if self.config.score_threshold > 0:
                reranked_docs = [
                    doc for doc in reranked_docs
                    if doc["rerank_score"] >= self.config.score_threshold
                ]

            # 返回Top-K
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]

            logger.debug(f"重排序完成: {len(results)} -> {len(reranked_docs)} 文档")
            return reranked_docs

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return results  # 失败时返回原始顺序

    async def _compute_scores(self, pairs: List[List[str]]) -> List[float]:
        """
        计算查询-文档对的相关性分数

        Args:
            pairs: [[query1, doc1], [query1, doc2], ...]

        Returns:
            相关性分数列表
        """
        if not self._model:
            return []

        try:
            # FlagReranker的compute_score方法在大量pairs时会阻塞
            # 使用run_in_executor避免阻塞事件循环
            loop = asyncio.get_event_loop()

            # 使用局部变量避免lambda中的self._model类型检查问题
            model = self._model
            assert model is not None  # 类型断言

            scores = await loop.run_in_executor(
                None,
                lambda: model.compute_score(
                    pairs, # type: ignore
                    batch_size=self.config.batch_size,
                    max_length=self.config.max_length
                )
            )

            return scores.tolist() if hasattr(scores, 'tolist') else list(scores) # type: ignore

        except Exception as e:
            logger.error(f"分数计算失败: {e}")
            return [0.0] * len(pairs)


class AdaptiveReranker:
    """
    自适应重排序器
    根据结果数量和相似度分布智能决定是否重排序
    """

    def __init__(self, config: RerankerConfig):
        self.reranker = ContentReranker(config)
        self.config = config

        # 自适应参数
        self.min_results_to_rerank = 3  # 至少3个结果才重排序
        self.max_results_to_rerank = 20  # 最多20个结果
        self.similarity_variance_threshold = 0.1  # 相似度方差阈值

    def should_rerank(self, results: List[Dict[str, Any]]) -> bool:
        """
        判断是否需要重排序

        Args:
            results: 检索结果列表

        Returns:
            是否需要重排序
        """
        if not self.reranker.is_available():
            return False

        n = len(results)

        # 结果数量检查
        if n < self.min_results_to_rerank:
            logger.debug(f"结果数量不足（{n}<{self.min_results_to_rerank}），跳过重排序")
            return False

        if n > self.max_results_to_rerank:
            logger.debug(f"结果数量过多（{n}>{self.max_results_to_rerank}），使用重排序优化")

        # 相似度分布检查
        if n >= 3:
            scores = [r.get("score", 0.0) for r in results]
            variance = self._compute_variance(scores)

            if variance < self.similarity_variance_threshold:
                logger.debug(f"相似度分布集中（方差={variance:.3f}），重排序可能提升效果")
                return True
            else:
                logger.debug(f"相似度分布分散（方差={variance:.3f}），跳过重排序")
                return False

        return True

    def _compute_variance(self, values: List[float]) -> float:
        """计算方差"""
        if not values:
            return 0.0

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return variance

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        自适应重排序

        Args:
            query: 查询文本
            results: 检索结果
            top_k: 返回前K个结果

        Returns:
            重排序后的结果（或原始结果）
        """
        if not self.should_rerank(results):
            return results

        return await self.reranker.rerank(query, results, top_k=top_k)


# 便捷函数
def create_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "auto",
    batch_size: int = 32,
    use_fp16: bool = True,
    adaptive: bool = True
) -> ContentReranker | AdaptiveReranker:
    """
    创建重排序器实例

    Args:
        model_name: 模型名称（默认bge-reranker-v2-m3，支持多语言）
        device: 设备（auto/mps/cuda/cpu）
        batch_size: 批处理大小（影响速度和内存）
        use_fp16: 使用FP16精度
        adaptive: 是否使用自适应模式

    Returns:
        重排序器实例
    """
    config = RerankerConfig(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        use_fp16=use_fp16
    )

    if adaptive:
        return AdaptiveReranker(config)
    else:
        return ContentReranker(config)
