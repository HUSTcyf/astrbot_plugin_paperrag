"""
llama-index 兼容的重排序器
支持自适应重排序和内容重排序
与 llama-index 的 NodeWithScore 格式完全兼容
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union, cast
from dataclasses import dataclass

from astrbot.api import logger

# 尝试导入 llama-index 类型（如果可用）
try:
    from llama_index.core.schema import Node, TextNode, NodeWithScore
    from llama_index.core import QueryBundle
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    # 运行时设为 None，类型检查时通过 TYPE_CHECKING 处理
    NodeWithScore = None  # type: ignore
    Node = None  # type: ignore
    TextNode = None  # type: ignore
    QueryBundle = None  # type: ignore

# 导入核心重排序器（兼容直接运行和包运行）
try:
    from .reranker import (
        ContentReranker,
        AdaptiveReranker as CoreAdaptiveReranker,
        RerankerConfig,
        create_reranker as create_core_reranker
    )
except ImportError:
    from reranker import (
        ContentReranker,
        AdaptiveReranker as CoreAdaptiveReranker,
        RerankerConfig,
        create_reranker as create_core_reranker
    )


@dataclass
class RerankResult:
    """重排序结果"""
    nodes: List[Any]  # llama-index NodeWithScore
    scores: List[float]  # 重排序分数
    metadata: Dict[str, Any]  # 元数据


class LlamaIndexContentReranker:
    """
    llama-index 兼容的内容重排序器

    特点：
    - 兼容 llama-index 的 NodeWithScore 格式
    - 支持同步和异步接口
    - 可独立使用或作为 Postprocessor
    """

    def __init__(self, config: RerankerConfig):
        """
        初始化重排序器

        Args:
            config: 重排序配置
        """
        self.config = config
        self._core = ContentReranker(config)
        self._available = self._core.is_available()

    def is_available(self) -> bool:
        """检查重排序器是否可用"""
        return self._available

    async def arerank(
        self,
        nodes: List[Any],
        query: str,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        异步重排序节点

        Args:
            nodes: llama-index NodeWithScore 列表
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            重排序后的节点列表
        """
        if not self.is_available():
            return nodes

        if not nodes:
            return []

        try:
            # 转换为字典格式
            results = []
            for node in nodes:
                results.append({
                    "text": node.node.get_content(),
                    "metadata": node.node.metadata,
                    "score": float(node.score if node.score is not None else 0.0)
                })

            # 使用核心重排序器
            reranked_results = await self._core.rerank(
                query=query,
                results=results,
                top_k=top_k
            )

            # 转换回 NodeWithScore 格式
            if LLAMA_INDEX_AVAILABLE:
                # 类型断言：在此块中 NodeWithScore 一定可用
                _NodeScoreClass = cast(type, NodeWithScore)

                reranked_nodes = []
                for i, result in enumerate(reranked_results):
                    # 保留原始节点，只更新分数
                    original_node = nodes[i].node
                    new_score = result.get("rerank_score", result.get("score", 0.0))

                    new_node_score = _NodeScoreClass(
                        node=original_node,
                        score=new_score
                    )
                    reranked_nodes.append(new_node_score)

                return reranked_nodes
            else:
                return nodes

        except Exception as e:
            logger.warning(f"⚠️ 重排序失败，返回原始顺序: {e}")
            return nodes

    def rerank(
        self,
        nodes: List[Any],
        query: str,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        同步重排序节点（包装异步方法）

        Args:
            nodes: llama-index NodeWithScore 列表
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            重排序后的节点列表
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # 在已有循环中创建任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.arerank(nodes, query, top_k)
                )
                return future.result()
        except RuntimeError:
            # 没有运行中的循环
            return asyncio.run(self.arerank(nodes, query, top_k))


class LlamaIndexAdaptiveReranker:
    """
    llama-index 兼容的自适应重排序器

    特点：
    - 根据结果数量和质量智能决定是否重排序
    - 兼容 llama-index 的 NodeWithScore 格式
    - 支持同步和异步接口
    """

    def __init__(self, config: RerankerConfig):
        """
        初始化自适应重排序器

        Args:
            config: 重排序配置
        """
        self.config = config
        self._core = CoreAdaptiveReranker(config)
        self._available = self._core.reranker.is_available()

        # 自适应参数
        self.min_results_to_rerank = 3
        self.max_results_to_rerank = 20
        self.similarity_variance_threshold = 0.1

    def is_available(self) -> bool:
        """检查重排序器是否可用"""
        return self._available

    def should_rerank(
        self,
        nodes: List[Any],
        query: Optional[str] = None
    ) -> bool:
        """
        判断是否需要重排序

        Args:
            nodes: 节点列表
            query: 查询文本（可选）

        Returns:
            是否需要重排序
        """
        if not self._available:
            return False

        n = len(nodes)

        # 结果数量检查
        if n < self.min_results_to_rerank:
            logger.debug(f"结果数量不足（{n}<{self.min_results_to_rerank}），跳过重排序")
            return False

        if n > self.max_results_to_rerank:
            logger.debug(f"结果数量过多（{n}>{self.max_results_to_rerank}），使用重排序优化")
            return True

        # 相似度分布检查
        if n >= 3:
            scores = [float(node.score if node.score is not None else 0.0) for node in nodes]
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

    async def arerank(
        self,
        nodes: List[Any],
        query: str,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        异步自适应重排序

        Args:
            nodes: llama-index NodeWithScore 列表
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            重排序后的节点列表
        """
        if not self.should_rerank(nodes, query):
            return nodes

        # 使用内容重排序器
        content_reranker = LlamaIndexContentReranker(self.config)
        return await content_reranker.arerank(nodes, query, top_k)

    def rerank(
        self,
        nodes: List[Any],
        query: str,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """
        同步自适应重排序（包装异步方法）

        Args:
            nodes: llama-index NodeWithScore 列表
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            重排序后的节点列表
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.arerank(nodes, query, top_k)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.arerank(nodes, query, top_k))


# ============================================================================
# llama-index Postprocessor 接口实现
# ============================================================================

if LLAMA_INDEX_AVAILABLE:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import QueryBundle

    class AdaptiveRerankerPostprocessor(BaseNodePostprocessor):
        """
        llama-index Postprocessor 接口的自适应重排序器

        可以直接用于 llama-index 的 QueryEngine 中：

        ```python
        from llama_index.core import VectorStoreIndex
        from llama_index_reranker import AdaptiveRerankerPostprocessor

        reranker = AdaptiveRerankerPostprocessor(
            model_name="BAAI/bge-reranker-v2-m3",
            device="auto",
            batch_size=32
        )

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[reranker]
        )
        ```
        """

        def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-v2-m3",
            device: str = "auto",
            batch_size: int = 32,
            use_fp16: bool = True,
            adaptive: bool = True,
            top_k: Optional[int] = None,
            score_threshold: float = 0.0
        ):
            """
            初始化 Postprocessor

            Args:
                model_name: 重排序模型名称
                device: 设备（auto/mps/cuda/cpu）
                batch_size: 批处理大小
                use_fp16: 使用 FP16 精度
                adaptive: 是否启用自适应模式
                top_k: 返回前 K 个结果
                score_threshold: 分数阈值
            """
            super().__init__()

            config = RerankerConfig(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                max_length=512,
                use_fp16=use_fp16,
                score_threshold=score_threshold
            )

            if adaptive:
                self._reranker = LlamaIndexAdaptiveReranker(config)
            else:
                self._reranker = LlamaIndexContentReranker(config)

            self._top_k = top_k

        def _postprocess_nodes(
            self,
            nodes: List[Any],
            query_bundle: Any = None,
            query_str: Optional[str] = None
        ) -> List[Any]:
            """
            同步后处理节点（llama-index Postprocessor 抽象方法实现）

            Args:
                nodes: 节点列表
                query_bundle: 查询包（包含查询文本）
                query_str: 查询文本字符串（优先级高于 query_bundle）

            Returns:
                重排序后的节点列表
            """
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.apostprocess_nodes(nodes, query_bundle, query_str)
                    )
                    return future.result()
            except RuntimeError:
                return asyncio.run(self.apostprocess_nodes(nodes, query_bundle, query_str))

        async def apostprocess_nodes(
            self,
            nodes: List[Any],
            query_bundle: Any = None,
            query_str: Optional[str] = None
        ) -> List[Any]:
            """
            异步后处理节点（llama-index Postprocessor 接口）

            Args:
                nodes: 节点列表
                query_bundle: 查询包（包含查询文本）
                query_str: 查询文本字符串（优先级高于 query_bundle）

            Returns:
                重排序后的节点列表
            """
            if not nodes:
                return nodes

            # 优先使用 query_str，否则从 QueryBundle 提取
            final_query_str = query_str or ""
            if query_bundle is not None and not final_query_str:
                final_query_str = query_bundle.query_str

            # 执行重排序
            reranked_nodes = await self._reranker.arerank(
                nodes=nodes,
                query=final_query_str,
                top_k=self._top_k
            )

            return reranked_nodes


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_llama_index_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "auto",
    batch_size: int = 32,
    use_fp16: bool = True,
    adaptive: bool = True,
    top_k: Optional[int] = None,
    score_threshold: float = 0.0
) -> Union[LlamaIndexAdaptiveReranker, LlamaIndexContentReranker]:
    """
    创建 llama-index 兼容的重排序器

    Args:
        model_name: 重排序模型名称
        device: 设备（auto/mps/cuda/cpu）
        batch_size: 批处理大小
        use_fp16: 使用 FP16 精度
        adaptive: 是否启用自适应模式
        top_k: 返回前 K 个结果
        score_threshold: 分数阈值

    Returns:
        重排序器实例

    Example:
        ```python
        # 创建自适应重排序器
        reranker = create_llama_index_reranker(
            model_name="BAAI/bge-reranker-v2-m3",
            adaptive=True
        )

        # 使用
        reranked_nodes = await reranker.arerank(nodes, query="查询文本")
        ```
    """
    config = RerankerConfig(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=512,
        use_fp16=use_fp16,
        score_threshold=score_threshold
    )

    if adaptive:
        return LlamaIndexAdaptiveReranker(config)
    else:
        return LlamaIndexContentReranker(config)


def create_adaptive_reranker_postprocessor(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "auto",
    batch_size: int = 32,
    use_fp16: bool = True,
    top_k: Optional[int] = None
) -> Optional['AdaptiveRerankerPostprocessor']:
    """
    创建 llama-index Postprocessor（仅当 llama-index 可用时）

    Args:
        model_name: 重排序模型名称
        device: 设备
        batch_size: 批处理大小
        use_fp16: 使用 FP16
        top_k: 返回前 K 个结果

    Returns:
        AdaptiveRerankerPostprocessor 实例，如果 llama-index 不可用则返回 None

    Example:
        ```python
        from llama_index_reranker import create_adaptive_reranker_postprocessor

        reranker = create_adaptive_reranker_postprocessor(
            model_name="BAAI/bge-reranker-v2-m3"
        )

        if reranker:
            query_engine = index.as_query_engine(
                node_postprocessors=[reranker]
            )
        ```
    """
    if not LLAMA_INDEX_AVAILABLE:
        logger.warning("⚠️ llama-index 不可用，无法创建 Postprocessor")
        return None

    return AdaptiveRerankerPostprocessor(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        use_fp16=use_fp16,
        adaptive=True,
        top_k=top_k
    )


# ============================================================================
# 简化的重排序函数（不依赖 NodeWithScore）
# ============================================================================

async def rerank_results(
    results: List[Dict[str, Any]],
    query: str,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "auto",
    batch_size: int = 32,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    简化的重排序函数（不依赖 llama-index）

    Args:
        results: 检索结果列表，格式: [{"text": "...", "metadata": {}, "score": 0.8}, ...]
        query: 查询文本
        model_name: 重排序模型名称
        device: 设备
        batch_size: 批处理大小
        top_k: 返回前 K 个结果

    Returns:
        重排序后的结果列表

    Example:
        ```python
        # 在 RAG 引擎中使用
        results = [
            {"text": "文档1", "metadata": {}, "score": 0.8},
            {"text": "文档2", "metadata": {}, "score": 0.7}
        ]

        reranked = await rerank_results(
            results=results,
            query="查询文本",
            top_k=5
        )
        ```
    """
    config = RerankerConfig(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=512,
        use_fp16=True,
        score_threshold=0.0
    )

    # 使用核心重排序器
    core_reranker = CoreAdaptiveReranker(config)

    return await core_reranker.rerank(
        query=query,
        results=results,
        top_k=top_k
    )


async def adaptive_rerank_results(
    results: List[Dict[str, Any]],
    query: str,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "auto",
    batch_size: int = 32,
    min_results: int = 3,
    max_results: int = 20,
    variance_threshold: float = 0.1,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    自适应重排序函数（智能决定是否重排序）

    Args:
        results: 检索结果列表
        query: 查询文本
        model_name: 重排序模型名称
        device: 设备
        batch_size: 批处理大小
        min_results: 最少结果数量才重排序
        max_results: 超过此数量强制重排序
        variance_threshold: 相似度方差阈值
        top_k: 返回前 K 个结果

    Returns:
        重排序后的结果列表（或原始结果）

    Example:
        ```python
        # 自适应重排序
        reranked = await adaptive_rerank_results(
            results=search_results,
            query="查询文本",
            min_results=3,
            max_results=20
        )
        ```
    """
    n = len(results)

    # 自适应判断
    if n < min_results:
        logger.debug(f"结果数量不足（{n}<{min_results}），跳过重排序")
        return results

    if n > max_results:
        logger.debug(f"结果数量过多（{n}>{max_results}），执行重排序")
    else:
        # 检查相似度分布
        scores = [r.get("score", 0.0) for r in results]
        if n >= 3:
            mean = sum(scores) / n
            variance = sum((x - mean) ** 2 for x in scores) / n

            if variance >= variance_threshold:
                logger.debug(f"相似度分布分散（方差={variance:.3f}），跳过重排序")
                return results

    # 执行重排序
    return await rerank_results(
        results=results,
        query=query,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        top_k=top_k
    )


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 类
    "LlamaIndexContentReranker",
    "LlamaIndexAdaptiveReranker",
    "AdaptiveRerankerPostprocessor",  # 仅当 llama-index 可用时
    "RerankResult",

    # 工厂函数
    "create_llama_index_reranker",
    "create_adaptive_reranker_postprocessor",

    # 简化函数
    "rerank_results",
    "adaptive_rerank_results",
]
