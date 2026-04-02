"""
混合架构RAG引擎 - 多模态支持版

完整的RAG流程（基于llama-index风格）：
1. PDF解析（多模态）→ HybridPDFParser
2. 文档分块 → Node结构
3. 向量存储 → HybridIndexManager（避免与主进程冲突）
4. 检索（向量搜索）→ HybridIndexManager
5. 生成 → GLM LLM（支持多模态：glm-4.6v-flash）
"""

import os
import time
import base64
from typing import List, Dict, Any, Optional, Union, cast
from pathlib import Path
from itertools import zip_longest

# 抑制底层库的 gRPC/absl 警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from astrbot.api import logger

# 获取插件目录，用于解析相对路径
_PLUGIN_DIR = Path(__file__).parent.resolve()

# 导入混合架构组件（兼容直接运行和包运行）
try:
    from .hybrid_parser import HybridPDFParser, Node
    from .hybrid_index import HybridIndexManager
    from .embedding_providers import (
        create_embedding_provider,
        OllamaEmbeddingProvider,
        AstrBotEmbeddingProvider
    )
    from .rag_engine import RAGConfig
    from .reranker import (
        AdaptiveReranker,
        RerankerConfig,
        create_reranker
    )
except ImportError:
    from hybrid_parser import HybridPDFParser, Node
    from hybrid_index import HybridIndexManager
    from embedding_providers import (
        create_embedding_provider,
        OllamaEmbeddingProvider,
        AstrBotEmbeddingProvider
    )
    from rag_engine import RAGConfig
    from reranker import (
        AdaptiveReranker,
        RerankerConfig,
        create_reranker
    )

# 导入 Llama.cpp VLM Provider（用于图片问答）
LLAMA_CPP_VLM_AVAILABLE = False
LLAMA_CPP_VLM_IMPORT_ERROR = None
try:
    try:
        from .llama_cpp_vlm_provider import LlamaCppVLMProvider
    except ImportError:
        from llama_cpp_vlm_provider import LlamaCppVLMProvider
    LLAMA_CPP_VLM_AVAILABLE = True
    logger.info("[Llama.cpp-VLM] Llama.cpp VLM Provider 已加载")
except ImportError as e:
    LLAMA_CPP_VLM_IMPORT_ERROR = e
    logger.warning(f"[Llama.cpp-VLM] Llama.cpp VLM Provider 导入失败: {e}")
    logger.warning("[Llama.cpp-VLM] 请确保已安装llama.cpp: brew install llama.cpp")


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
    """检索器基类（llama-index风格）"""

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


class HybridRetriever(BaseRetriever):
    """
    混合检索器：BM25 关键词检索 + 向量语义检索 + RRF 分数融合

    流程：
    1. 并行执行 BM25 和向量搜索
    2. 使用 Reciprocal Rank Fusion (RRF) 合并两路结果
    3. 可选：LLM 智能决定融合权重
    """

    def __init__(
        self,
        index_manager: HybridIndexManager,
        embed_provider: Any,
        bm25_top_k: int = 20,
        alpha: float = 0.5,
        rrf_k: int = 60,
        llm_provider: Any = None  # 可选：用于LLM权重决策
    ):
        super().__init__(index_manager, embed_provider)
        self._bm25_top_k = bm25_top_k
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._llm_provider = llm_provider

    def set_llm_provider(self, llm_provider: Any):
        """设置LLM provider（用于LLM权重决策）"""
        self._llm_provider = llm_provider

    async def retrieve(self, query: str, top_k: int = 5, use_llm_fusion: bool = False, initial_vector_k: int = 50, initial_bm25_k: int = 100) -> QueryResult:
        """混合检索：BM25 + 向量 + RRF 融合

        Args:
            query: 查询文本
            top_k: 返回结果数量（最终返回）
            use_llm_fusion: 是否使用LLM智能决定融合权重
            initial_vector_k: 向量初筛召回数量（默认50）
            initial_bm25_k: BM25初筛召回数量（默认100）
        """
        try:
            # 1. 向量语义搜索（初筛时使用更多结果）
            query_embedding = await self._embed_provider.get_text_embedding(query)
            vector_results = await self._index_manager.search(
                query_embedding=query_embedding,
                top_k=initial_vector_k
            )

            # 2. BM25 关键词搜索（传入 LLM provider 以复用）
            bm25_results = await self._index_manager.bm25_search(
                query=query,
                top_k=initial_bm25_k,
                llm_provider=self._llm_provider  # 复用已加载的 LLM
            )

            # 3. 决定融合权重
            if use_llm_fusion and self._llm_provider:
                alpha = await self._decide_fusion_weight_by_llm(query, vector_results, bm25_results)
                logger.info(f"[LLM融合权重] query='{query}', alpha={alpha:.2f} (0=BM25, 1=向量)")
            else:
                alpha = self._alpha

            # 4. RRF 融合
            fused = self._rrf_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                top_k=top_k,
                alpha=alpha
            )

            nodes = []
            scores = []
            for item in fused:
                nodes.append(Node(
                    text=item["text"],
                    metadata=item.get("metadata", {})
                ))
                scores.append(item.get("fused_score", 0.0))

            logger.debug(
                f"✅ 混合检索完成: query='{query}', "
                f"向量={len(vector_results)}, BM25={len(bm25_results)}, "
                f"融合后={len(fused)}"
            )
            return QueryResult(nodes=nodes, scores=scores)

        except Exception as e:
            logger.error(f"❌ 混合检索失败: {e}")
            # 降级为纯向量检索
            fallback = VectorRetriever(self._index_manager, self._embed_provider)
            return await fallback.retrieve(query, top_k)

    async def _decide_fusion_weight_by_llm(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> float:
        """让LLM分析查询和检索结果，决定融合权重

        Args:
            query: 用户查询
            vector_results: 向量检索结果（已按相关性排序）
            bm25_results: BM25检索结果（已按相关性排序）

        Returns:
            alpha值：0.0=纯BM25, 1.0=纯向量, 0.5=平等权重
        """
        try:
            # 构建分析prompt
            vector_texts = "\n".join([
                f"[向量{i+1}] {r['text'][:200]}..." if len(r['text']) > 200 else f"[向量{i+1}] {r['text']}"
                for i, r in enumerate(vector_results[:5])
            ])
            bm25_texts = "\n".join([
                f"[BM25{i+1}] {r['text'][:200]}..." if len(r['text']) > 200 else f"[BM25{i+1}] {r['text']}"
                for i, r in enumerate(bm25_results[:5])
            ])

            # 检查查询是否包含关键词特征
            keyword_indicators = ["多少", "几个", "who", "what", "how many", "which", "when", "where", "name"]
            has_keyword = any(ki in query.lower() for ki in keyword_indicators)

            # 检查查询是否包含专业术语或复杂语义
            semantic_indicators = ["为什么", "如何", "why", "how", "原理", "解释", "compare", "difference", "relationship"]
            has_semantic = any(si in query.lower() for si in semantic_indicators)

            prompt = f"""分析以下查询，决定使用BM25还是向量检索更适合：

查询：{query}

向量检索结果（Top 5）：
{vector_texts}

BM25检索结果（Top 5）：
{bm25_texts}

请分析：
1. 这个查询更依赖关键词匹配（BM25）还是语义理解（向量检索）？
2. BM25结果是否包含查询中的关键词？
3. 向量结果是否语义上与查询相关？

请从以下选项中选择最合适的融合策略：
- 如果查询包含具体关键词（人名、数字、专有名词等），关键词匹配更重要 → 选择偏重BM25
- 如果查询是复杂问题或需要语义理解，语义检索更重要 → 选择偏重向量
- 如果两者结果质量相近 → 选择平等权重

直接回答：BM25权重 / 向量权重（如 "0.7 / 0.3" 或 "0.5 / 0.5"）
只回答数字比例，不要其他解释。"""

            # 调用LLM
            if hasattr(self._llm_provider, 'text_chat'):
                response = await self._llm_provider.text_chat(
                    prompt=prompt,
                    contexts=[],
                    temperature=0.1,
                    max_tokens=100
                )
                response_text = self._extract_llm_response(response)
            else:
                return self._alpha

            # 解析LLM响应，提取权重
            import re
            # 尝试匹配 "0.7 / 0.3" 或 "0.7, 0.3" 或 "7:3" 等格式
            patterns = [
                r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)',  # 0.7 / 0.3
                r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',  # 0.7, 0.3
                r'(\d+)\s*:\s*(\d+)',              # 7:3
                r'BM25[:\s]+(\d+\.?\d*)',          # BM25: 0.7
            ]

            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    if ':' in pattern or '/' in pattern or ',' in pattern:
                        # 两个数字的情况
                        val1, val2 = float(match.group(1)), float(match.group(2))
                        total = val1 + val2
                        if total > 0:
                            # 返回BM25的权重（val1/total）
                            return val1 / total
                    else:
                        # 单个数字的情况
                        return float(match.group(1))

            # 默认返回配置的alpha
            return self._alpha

        except Exception as e:
            logger.warning(f"[LLM融合权重决策失败: {e}]")
            return self._alpha

    def _extract_llm_response(self, response: Any) -> str:
        """从LLM响应中提取文本"""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            return response.get("content", "") or response.get("text", "") or str(response)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
        alpha: Optional[float] = None  # 可选，默认使用self._alpha
    ) -> List[Dict[str, Any]]:
        """
        Score-based Linear Fusion 分数融合（改进版）

        使用实际BM25分数和向量分数进行线性融合，比纯RRF更准确

        Args:
            vector_results: 向量检索结果 [{"text", "metadata", "score"}, ...]
            bm25_results: BM25 检索结果 [{"text", "metadata", "score"}, ...]
            top_k: 返回结果数量
            alpha: 融合权重（0=纯BM25, 1=纯向量），默认使用self._alpha

        Returns:
            融合排序后的结果列表
        """
        # 使用传入的alpha或默认值
        if alpha is None:
            alpha = self._alpha

        # 构建 text -> vector_score 和 vector_rank 映射
        vector_score_map: Dict[str, float] = {}
        vector_rank_map: Dict[str, int] = {}
        for i, item in enumerate(vector_results):
            vector_score_map[item["text"]] = item.get("score", 0.0)
            vector_rank_map[item["text"]] = i + 1

        # 构建 text -> bm25_score 和 bm25_rank 映射
        bm25_score_map: Dict[str, float] = {}
        bm25_rank_map: Dict[str, int] = {}
        for i, item in enumerate(bm25_results):
            bm25_score_map[item["text"]] = item.get("score", 0.0)
            bm25_rank_map[item["text"]] = i + 1

        # 计算最大分数用于归一化
        max_vector_score = max(vector_score_map.values()) if vector_score_map else 1.0
        max_bm25_score = max(bm25_score_map.values()) if bm25_score_map else 1.0
        if max_vector_score <= 0:
            max_vector_score = 1.0
        if max_bm25_score <= 0:
            max_bm25_score = 1.0

        # 合并所有文本
        all_texts = set(vector_rank_map.keys()) | set(bm25_rank_map.keys())

        # 计算融合分数：结合分数和排名
        fused_scores: Dict[str, float] = {}
        for text in all_texts:
            # 归一化分数（0-1范围）
            v_score = vector_score_map.get(text, 0.0) / max_vector_score
            b_score = bm25_score_map.get(text, 0.0) / max_bm25_score

            # 获取排名（用于RRF平滑）
            v_rank = vector_rank_map.get(text, len(vector_results) + 1)
            b_rank = bm25_rank_map.get(text, len(bm25_results) + 1)

            # 混合策略：分数为主，RRF排名为辅
            # RRF平滑项：使用排名计算1/(rank+k)，k越小排名越重要
            rrf_smooth = 1.0 / (min(v_rank, b_rank) + 5)  # k=5，排名差距小时起作用

            # 融合分数 = 加权分数 + RRF平滑项
            # alpha控制向量vs BM25权重
            combined = alpha * v_score + (1 - alpha) * b_score
            # 加入微小的RRF平滑，避免分数相同时按排名分先后
            fused_scores[text] = combined + rrf_smooth * 0.001

        # 按融合分数降序排列
        sorted_texts = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果（保留完整 metadata）
        text_to_metadata: Dict[str, Dict] = {}
        for item in vector_results:
            text_to_metadata[item["text"]] = item.get("metadata", {})
        for item in bm25_results:
            if item["text"] not in text_to_metadata:
                text_to_metadata[item["text"]] = item.get("metadata", {})

        fused = []
        for text, fused_score in sorted_texts[:top_k]:
            fused.append({
                "text": text,
                "metadata": text_to_metadata.get(text, {}),
                "score": fused_score,
                "fused_score": fused_score
            })

        return fused


# ==================== CRAG (Corrective RAG) 评估和修正 ====================

class CragEvaluator:
    """
    CRAG 检索质量评估器

    评估检索结果与查询的相关性，返回质量等级和分数
    """

    def __init__(self, llm_provider: Any = None):
        """
        初始化评估器

        Args:
            llm_provider: 可选的 LLM provider，用于 LLM 评估（更准确但更慢）
        """
        self._llm_provider = llm_provider

    async def evaluate_retrieval_quality(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        评估检索质量（优先使用 LLM，基于规则作为降级）

        Args:
            query: 查询文本
            results: 检索结果 [{"text", "metadata", "score"}, ...]

        Returns:
            {
                "score": 0.0-1.0,  # 相关性分数
                "level": "high/medium/low",  # 质量等级
                "reasoning": str  # 评估理由
            }
        """
        if not results:
            return {
                "score": 0.0,
                "level": "low",
                "reasoning": "无检索结果"
            }

        # 如果有 LLM provider，优先使用 LLM 评估
        if self._llm_provider:
            try:
                return await self._evaluate_by_llm(query, results)
            except Exception as e:
                logger.debug(f"[CRAG] LLM评估失败，降级为规则评估: {e}")

        # 降级：基于规则评估
        return self._evaluate_by_rules(query, results)

    async def _evaluate_by_llm(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        使用 LLM 评估检索质量
        """
        # 构建评估 prompt
        top_results = results[:3]
        context_text = "\n\n".join([
            f"[文档 {i+1}] (相关性分数：{r.get('score', 0):.3f})\n{r.get('text', '')[:300]}..."
            for i, r in enumerate(top_results)
        ])

        prompt = f"""你是一个检索质量评估专家。请评估以下检索结果与查询的相关性。

【查询】
{query}

【检索结果】
{context_text}

【评估标准】
- 高相关性 (0.7-1.0): 检索结果直接包含答案或高度相关信息
- 中相关性 (0.3-0.7): 检索结果部分相关，需要补充信息
- 低相关性 (0.0-0.3): 检索结果与查询无关或信息不足

【输出格式】
严格的 JSON 格式：
{{
    "score": 0.0-1.0 的数字，
    "level": "high/medium/low",
    "reasoning": "简要说明评估理由"
}}

【评估】"""

        try:
            response = await self._llm_provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.1,
                max_tokens=1024
            )

            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, dict):
                response_text = response.get("content", "") or response.get("text", "")
            else:
                response_text = str(response)

            # 解析 JSON
            import json, re
            logger.debug(f"[CRAG LLM] 响应原文:\n{response_text}")
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                matched_json = json_match.group(0)
                logger.debug(f"[CRAG LLM] 匹配的JSON:\n{matched_json}")

                # 尝试标准 json.loads
                try:
                    result = json.loads(matched_json)
                    logger.info("[CRAG] JSON解析成功")
                except json.JSONDecodeError as e:
                    logger.warning(f"[CRAG] JSON解析失败: {e}，使用正则提取...")
                    # 正则提取
                    import re
                    score_match = re.search(r'"score"\s*:\s*([0-9.]+)', matched_json)
                    score = float(score_match.group(1)) if score_match else 0.5
                    level_match = re.search(r'"level"\s*:\s*"([^"]+)"', matched_json)
                    level = level_match.group(1) if level_match else "medium"
                    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', matched_json)
                    reasoning = reasoning_match.group(1) if reasoning_match else ""
                    result = {"score": score, "level": level, "reasoning": reasoning}
                    logger.info(f"[CRAG] 正则提取: score={score}, level={level}, reasoning={reasoning[:50]}...")

                score = float(result.get("score", 0.5))
                level = result.get("level", "medium")
                reasoning = result.get("reasoning", "")

                # 验证结果有效性
                if level not in ["high", "medium", "low"]:
                    level = "medium" if score >= 0.3 else "low"

                logger.debug(f"[CRAG LLM评估] score={score:.2f}, level={level}")
                return {
                    "score": min(score, 1.0),
                    "level": level,
                    "reasoning": reasoning
                }

        except Exception as e:
            logger.warning(f"[CRAG] LLM评估异常: {e}")

        # 解析失败，降级到规则评估
        return self._evaluate_by_rules(query, results)

    def _evaluate_by_rules(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        基于规则评估检索质量（轻量级降级方案）
        """
        # 1. 分析查询特征
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # 2. 分析检索结果
        top_result = results[0]
        avg_score = sum(r.get("score", 0.0) for r in results) / len(results)

        # 3. 计算关键词覆盖率
        coverage_scores = []
        for r in results[:3]:  # 只看top 3
            doc_text = r.get("text", "").lower()
            doc_terms = set(doc_text.split())
            coverage = len(query_terms & doc_terms) / max(len(query_terms), 1)
            coverage_scores.append(coverage)

        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0

        # 4. 综合评分
        score = 0.4 * min(avg_score * 2, 1.0) + 0.4 * avg_coverage + 0.2 * coverage_scores[0] if coverage_scores else 0.5

        # 5. 边界情况检查
        top1_text = top_result.get("text", "").lower()
        top1_coverage = len(query_terms & set(top1_text.split())) / max(len(query_terms), 1)
        if top1_coverage < 0.1 and avg_score < 0.3:
            score *= 0.5
            reasoning = "Top结果与查询相关性低"
        elif score > 0.6:
            reasoning = "检索结果与查询高度相关"
        elif score > 0.3:
            reasoning = "检索结果与查询中等相关"
        else:
            reasoning = "检索结果与查询相关性低"

        # 6. 确定等级
        if score >= 0.6:
            level = "high"
        elif score >= 0.3:
            level = "medium"
        else:
            level = "low"

        return {
            "score": min(score, 1.0),
            "level": level,
            "reasoning": reasoning
        }


class CragCorrector:
    """
    CRAG 修正策略执行器

    根据评估等级执行不同的修正策略
    """

    def __init__(self, index_manager: Any, embed_provider: Any, llm_provider: Any = None):
        self._index_manager = index_manager
        self._embed_provider = embed_provider
        self._llm_provider = llm_provider

    async def correct(
        self,
        query: str,
        level: str,
        original_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        根据评估等级执行修正策略

        Args:
            query: 原始查询
            level: high/medium/low
            original_results: 原始检索结果

        Returns:
            修正后的检索结果
        """
        if level == "high":
            # 高质量：直接使用原结果
            logger.debug(f"[CRAG] 检索质量高，直接使用原结果")
            return original_results

        elif level == "medium":
            # 中等质量：查询重写 + 补充检索
            logger.debug(f"[CRAG] 检索质量中等，执行查询重写补充")
            return await self._rewrite_and_retrieve(query, original_results)

        else:  # low
            # 低质量：执行查询重写 + 重新检索
            logger.debug(f"[CRAG] 检索质量低，执行查询重写重新检索")
            return await self._rewrite_and_merge(query, original_results)

    async def _rewrite_and_retrieve(
        self,
        query: str,
        original_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """中等质量修正：多查询重写 + 补充检索 + RRF融合"""
        rewritten_queries = await self._rewrite_query_multi(query)
        if len(rewritten_queries) <= 1:
            return original_results

        # 使用多查询执行补充检索
        try:
            all_results = list(original_results)
            for rewritten_query in rewritten_queries[1:]:  # 跳过原查询
                query_embedding = await self._embed_provider.get_text_embedding(rewritten_query)
                additional_results = await self._index_manager.search(
                    query_embedding=query_embedding,
                    top_k=5
                )
                all_results.extend(additional_results)

            # 使用RRF融合
            return self._rrf_fusion(all_results, top_k=10)
        except Exception as e:
            logger.warning(f"[CRAG] 补充检索失败: {e}")
            return original_results

    async def _rewrite_and_merge(
        self,
        query: str,
        original_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """低质量修正：多查询重写 + 混合检索 + RRF融合"""
        rewritten_queries = await self._rewrite_query_multi(query)

        try:
            all_results = []

            for rewritten_query in rewritten_queries:
                # 并行执行向量和BM25检索
                query_embedding = await self._embed_provider.get_text_embedding(rewritten_query)
                vector_results = await self._index_manager.search(
                    query_embedding=query_embedding,
                    top_k=15
                )
                bm25_results = await self._index_manager.bm25_search(
                    query=rewritten_query,
                    top_k=15,
                    llm_provider=self._llm_provider
                )

                # 简单融合后加入结果集
                combined = self._simple_fusion(vector_results, bm25_results, top_k=15)
                all_results.extend(combined)

            # 与原结果融合
            all_results.extend(original_results)

            # 使用RRF融合
            return self._rrf_fusion(all_results, top_k=10)
        except Exception as e:
            logger.warning(f"[CRAG] 重新检索失败: {e}")
            return original_results

    async def _rewrite_query_multi(self, query: str) -> List[str]:
        """使用LLM生成多个查询改写（多查询扩展）"""
        if not self._llm_provider:
            return [query]

        try:
            # 提取相关术语增强改写
            relevant_terms = await self._extract_relevant_terms(query)

            prompt = f"""将以下学术查询改写为3-5个不同的检索表述，覆盖不同角度。
保持核心术语不变，仅改变句式和表达方式。

原始查询：{query}
{relevant_terms}
改写查询（每行一个，不要编号）：
"""

            response = await self._llm_provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=300
            )

            rewritten_text = ""
            if hasattr(response, 'content'):
                rewritten_text = response.content
            elif isinstance(response, dict):
                rewritten_text = response.get("content", "") or response.get("text", "")
            else:
                rewritten_text = str(response)

            # 解析多行改写结果
            rewritten_queries = []
            for line in rewritten_text.strip().split('\n'):
                line = line.strip().strip('`').strip('0123456789.、、').strip()
                if line and len(line) > 5 and line != query:
                    rewritten_queries.append(line)

            if rewritten_queries:
                logger.debug(f"[CRAG] 多查询改写: '{query}' → {rewritten_queries}")
                return [query] + rewritten_queries[:4]  # 最多4个改写 + 原查询
        except Exception as e:
            logger.warning(f"[CRAG] 多查询改写失败: {e}")

        return [query]

    async def _extract_relevant_terms(self, query: str) -> str:
        """从已索引论文中提取与查询相关的术语"""
        try:
            # 获取部分chunk用于术语提取
            chunks = await self._index_manager.get_all_chunks()
            if not chunks:
                return ""

            # 简单采样：从开头和随机位置取一些chunks
            import random
            sample_size = min(50, len(chunks))
            sampled = random.sample(chunks, sample_size) if len(chunks) > sample_size else chunks

            # 提取文本片段
            sample_texts = []
            for chunk in sampled[:20]:
                text = chunk.get("text", "")[:500]  # 限制长度
                if text:
                    sample_texts.append(text)

            if not sample_texts:
                return ""

            context = "\n".join(sample_texts[:10])

            extract_prompt = f"""从以下论文片段中提取与查询相关的专业术语（仅提取，不解释）。
查询：{query}

片段：
{context}

相关术语（用逗号分隔）：
"""

            response = await self._llm_provider.text_chat(
                prompt=extract_prompt,
                contexts=[],
                temperature=0.1,
                max_tokens=100
            )

            terms = ""
            if hasattr(response, 'content'):
                terms = response.content
            elif isinstance(response, dict):
                terms = response.get("content", "") or response.get("text", "")
            else:
                terms = str(response)

            terms = terms.strip().strip('`').strip()
            if terms and len(terms) > 2:
                return f"本地论文库相关术语：{terms}"
        except Exception as e:
            logger.debug(f"[CRAG] 术语提取失败: {e}")

        return ""

    def _simple_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """简单的分数融合"""
        # 构建分数映射
        text_to_score: Dict[str, float] = {}

        # 向量分数归一化
        max_v = max((r.get("score", 0.0) for r in vector_results), default=1.0)
        if max_v <= 0:
            max_v = 1.0
        for r in vector_results:
            text_to_score[r["text"]] = r.get("score", 0.0) / max_v

        # BM25分数归一化并融合
        max_b = max((r.get("score", 0.0) for r in bm25_results), default=1.0)
        if max_b <= 0:
            max_b = 1.0
        for r in bm25_results:
            text = r["text"]
            bm25_norm = r.get("score", 0.0) / max_b
            if text in text_to_score:
                text_to_score[text] = 0.5 * text_to_score[text] + 0.5 * bm25_norm
            else:
                text_to_score[text] = bm25_norm

        # 按分数排序
        sorted_items = sorted(text_to_score.items(), key=lambda x: x[1], reverse=True)

        # 构建结果
        text_to_metadata: Dict[str, Dict] = {}
        for r in vector_results:
            text_to_metadata[r["text"]] = r.get("metadata", {})
        for r in bm25_results:
            if r["text"] not in text_to_metadata:
                text_to_metadata[r["text"]] = r.get("metadata", {})

        results = []
        for text, score in sorted_items[:top_k]:
            results.append({
                "text": text,
                "metadata": text_to_metadata.get(text, {}),
                "score": score
            })

        return results

    def _rrf_fusion(
        self,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        使用 Reciprocal Rank Fusion (RRF) 融合多轮检索结果

        Args:
            results: 检索结果列表（可能来自多轮检索）
            top_k: 返回结果数量
            k: RRF常数（默认60）
        """
        # 按查询轮次分组（通过metadata中的query标记，如果存在）
        # 否则假设所有结果来自同一轮
        text_to_rrf_score: Dict[str, float] = {}
        text_to_metadata: Dict[str, Dict] = {}

        for r in results:
            text = r["text"]
            score = r.get("score", 0.0)

            # 累加 RRF 分数（每个text在每轮检索中取最高分）
            if text not in text_to_rrf_score:
                # 首次出现，获得 1/k 的基础分数
                text_to_rrf_score[text] = 1.0 / (k + 1)
            else:
                # 已有该text，累加（同一text在多轮检索中都出现说明相关）
                text_to_rrf_score[text] += 1.0 / (k + 1)

            # 保留最高分数的metadata
            if text not in text_to_metadata or score > text_to_metadata[text].get("_raw_score", 0):
                metadata = dict(r.get("metadata", {}))
                metadata["_raw_score"] = score
                text_to_metadata[text] = metadata

        # 按 RRF 分数排序
        sorted_items = sorted(text_to_rrf_score.items(), key=lambda x: x[1], reverse=True)

        # 构建结果
        fused_results = []
        for text, rrf_score in sorted_items[:top_k]:
            metadata = text_to_metadata.get(text, {})
            # 移除内部字段
            if "_raw_score" in metadata:
                del metadata["_raw_score"]
            fused_results.append({
                "text": text,
                "metadata": metadata,
                "score": rrf_score
            })

        return fused_results

    def _deduplicate_and_merge(
        self,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """去重 + 合并"""
        seen_texts: set = set()
        deduped = []

        for r in results:
            text = r["text"]
            if text not in seen_texts:
                seen_texts.add(text)
                deduped.append(r)

            if len(deduped) >= top_k:
                break

        return deduped


class HybridRAGEngine:
    """
    混合RAG引擎（多模态支持版）

    基于llama-index风格设计：
    - PDF解析（多模态）
    - Node结构存储
    - 混合检索（BM25 + 向量 + RRF融合）或纯向量检索
    - AdaptiveReranker 自适应重排序
    - LLM生成（支持多模态：图片查询）
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

        # 延迟初始化 - 使用 cast 避免类型错误
        self._parser: HybridPDFParser = cast(HybridPDFParser, None)
        self._index_manager: HybridIndexManager = cast(HybridIndexManager, None)
        self._embed_provider: Union[OllamaEmbeddingProvider, AstrBotEmbeddingProvider, None] = None
        self._llm_client: Any = cast(Any, None)
        self._retriever: Union[VectorRetriever, "HybridRetriever"] = cast(Any, None)
        self._reranker: Optional[Any] = None

        # 初始化标志
        self._parser_initialized = False
        self._index_initialized = False
        self._embed_provider_initialized = False
        self._llm_initialized = False
        self._retriever_initialized = False
        self._reranker_initialized = False

    def _ensure_parser_initialized(self) -> HybridPDFParser:
        """确保解析器已初始化"""
        if self._parser_initialized:
            return self._parser

        try:
            self._parser = HybridPDFParser(
                enable_multimodal=self.config.enable_multimodal,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            self._parser_initialized = True
            logger.info("✅ HybridPDFParser初始化完成")
            return self._parser
        except Exception as e:
            logger.error(f"❌ HybridPDFParser初始化失败: {e}")
            raise

    async def _ensure_embed_provider_initialized(self) -> Union[OllamaEmbeddingProvider, AstrBotEmbeddingProvider]:
        """确保Embedding Provider已初始化"""
        if self._embed_provider_initialized:
            assert self._embed_provider is not None
            return self._embed_provider

        try:
            # 构建 Ollama 配置
            ollama_config = dict(self.config.ollama_config) if self.config.ollama_config else {}
            # 配置 LLM 压缩
            if self.config.compress_provider_id:
                ollama_config["use_llm_compress"] = True
                ollama_config["compress_max_chars"] = 6400

            self._embed_provider = create_embedding_provider(
                mode=self.config.embedding_mode,
                context=self.context,
                provider_id=self.config.embedding_provider_id,
                ollama_config=ollama_config,
                compress_provider_id=self.config.compress_provider_id,
                embed_batch_size=10
            )
            self._embed_provider_initialized = True
            logger.info(f"✅ Embedding Provider初始化完成: {self.config.embedding_mode}")
            assert self._embed_provider is not None
            return self._embed_provider
        except Exception as e:
            logger.error(f"❌ Embedding Provider初始化失败: {e}")
            raise

    def _ensure_index_manager_initialized(self) -> HybridIndexManager:
        """确保索引管理器已初始化"""
        if self._index_initialized:
            assert self._index_manager is not None
            return self._index_manager

        try:
            # 确定连接模式
            mode = self.config.get_connection_mode()

            if mode == 'lite':
                lite_path = self.config.milvus_lite_path
                uri: Optional[str] = None
            else:
                lite_path = None
                uri = self.config.address

            logger.info(f"🔍 [INFO] _ensure_index_manager_initialized: lite_path='{lite_path}', uri='{uri}'")

            self._index_manager = HybridIndexManager(
                alias="paperrag_hybrid",  # 唯一别名，避免冲突
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
        except Exception as e:
            logger.error(f"❌ HybridIndexManager初始化失败: {e}")
            raise

    def _ensure_retriever_initialized(self) -> Union[VectorRetriever, HybridRetriever]:
        """确保检索器已初始化"""
        if self._retriever_initialized:
            assert self._retriever is not None
            return self._retriever

        # 确保依赖组件已初始化
        index_manager = self._ensure_index_manager_initialized()
        embed_provider = cast(Union[OllamaEmbeddingProvider, AstrBotEmbeddingProvider], self._embed_provider)

        if embed_provider is None:
            raise RuntimeError("Embed provider not initialized")

        if self.config.enable_bm25:
            self._retriever = HybridRetriever(
                index_manager=index_manager,
                embed_provider=embed_provider,
                bm25_top_k=self.config.bm25_top_k,
                alpha=self.config.hybrid_alpha,
                rrf_k=self.config.hybrid_rrf_k
            )
            self._retriever_initialized = True
            logger.info(
                f"✅ HybridRetriever初始化完成 "
                f"(BM25 top_k={self.config.bm25_top_k}, alpha={self.config.hybrid_alpha})"
            )
        else:
            self._retriever = VectorRetriever(
                index_manager=index_manager,
                embed_provider=embed_provider
            )
            self._retriever_initialized = True
            logger.info("✅ VectorRetriever初始化完成")

        assert self._retriever is not None
        return self._retriever

    async def _ensure_llm_initialized(self) -> Any:
        """确保LLM Provider已初始化 - 优先使用本地模型"""
        if self._llm_initialized:
            assert self._llm_client is not None
            return self._llm_client

        # 优先使用配置的 provider_id
        provider_id = self.config.text_provider_id
        if not provider_id:
            # 优先使用 LlamaCpp 本地模型
            if LLAMA_CPP_VLM_AVAILABLE:
                try:
                    # 解析模型路径
                    llama_model_path_raw = getattr(self.config, 'llama_vlm_model_path', './models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf')
                    llama_mmproj_path_raw = getattr(self.config, 'llama_vlm_mmproj_path', './models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf')

                    llama_model_path = str((_PLUGIN_DIR / llama_model_path_raw).resolve())
                    llama_mmproj_path = str((_PLUGIN_DIR / llama_mmproj_path_raw).resolve())

                    llama_max_tokens = getattr(self.config, 'llama_vlm_max_tokens', 2560)
                    llama_temperature = getattr(self.config, 'llama_vlm_temperature', 0.7)
                    llama_n_ctx = getattr(self.config, 'llama_vlm_n_ctx', 4096)
                    llama_n_gpu_layers = getattr(self.config, 'llama_vlm_n_gpu_layers', 99)

                    try:
                        from .llama_cpp_vlm_provider import (
                            get_llama_cpp_vlm_provider,
                            get_cached_llama_cpp_provider,
                            init_llama_cpp_vlm_provider
                        )
                    except ImportError:
                        from llama_cpp_vlm_provider import (
                            get_llama_cpp_vlm_provider,
                            get_cached_llama_cpp_provider,
                            init_llama_cpp_vlm_provider
                        )

                    # 检查是否已有缓存的 Provider
                    cached = get_cached_llama_cpp_provider()
                    if cached is not None:
                        self._llm_client = cached
                        logger.info("✅ 复用已缓存的 LlamaCpp 本地模型")
                    else:
                        # 初始化新的 Provider（设置全局配置）
                        self._llm_client = init_llama_cpp_vlm_provider(
                            model_path=llama_model_path,
                            mmproj_path=llama_mmproj_path,
                            n_ctx=llama_n_ctx,
                            n_gpu_layers=llama_n_gpu_layers,
                            max_tokens=llama_max_tokens,
                            temperature=llama_temperature
                        )
                        logger.info("✅ 初始化 LlamaCpp 本地模型进行文本生成")

                    self._llm_initialized = True
                    return self._llm_client
                except Exception as e:
                    logger.warning(f"⚠️ LlamaCpp 模型加载失败: {e}")

            # 如果本地模型不可用，尝试获取AstrBot当前会话的Provider（云端）
            try:
                if self.context is not None:
                    self._llm_client = self.context.get_using_provider()
                    if self._llm_client:
                        logger.info("✅ 使用当前会话的 LLM Provider (云端备选)")
                        self._llm_initialized = True
                        return self._llm_client
            except Exception as e:
                logger.warning(f"⚠️ 获取当前Provider失败: {e}")

            # 如果都获取不到，报错
            raise ValueError(
                "未配置 text_provider_id、本地模型不可用、无法获取当前LLM Provider。"
                "请在插件配置中设置 text_provider_id 或确保 LlamaCpp 模型可用。"
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
            self._llm_initialized = True
            raise

    def _ensure_reranker_initialized(self) -> Optional[AdaptiveReranker]:
        """确保重排序器已初始化"""
        if self._reranker_initialized:
            return self._reranker

        if not self.config.enable_reranking:
            logger.debug("🔍 重排序未启用")
            return None

        try:
            reranker_config = RerankerConfig(
                model_name=self.config.reranking_model,
                device=self.config.reranking_device,
                batch_size=self.config.reranking_batch_size,
                score_threshold=self.config.reranking_threshold
            )
            self._reranker = create_reranker(
                model_name=self.config.reranking_model,
                device=self.config.reranking_device,
                batch_size=self.config.reranking_batch_size,
                adaptive=self.config.reranking_adaptive
            )
            self._reranker_initialized = True
            # 注意：reranker 使用延迟加载，模型在实际使用时才加载
            # 这避免了在初始化时阻塞或占用资源
            return self._reranker
        except Exception as e:
            logger.warning(f"⚠️ 重排序器初始化失败: {e}")
            self._reranker_initialized = True
            self._reranker = None
            return None

    async def add_paper(
        self,
        file_path: str,
        llm_config: Dict[str, Any] = {},
        arxiv_client: Any = None
    ) -> Dict[str, Any]:
        """
        添加论文到知识库

        Args:
            file_path: PDF文件路径
            llm_config: LLM 配置字典（可选），包含 model、api_base、api_key
            arxiv_client: arXiv MCP 客户端（可选）

        Returns:
            添加结果
        """
        try:
            # 确保组件已初始化
            parser = self._ensure_parser_initialized()
            embed_provider = await self._ensure_embed_provider_initialized()
            index_manager = self._ensure_index_manager_initialized()

            # 如果启用了 LLM 参考文献解析且没有显式提供 LLM config，
            # 则自动获取配置的 text_provider_id 并构建 config
            effective_llm_config = llm_config
            if not effective_llm_config and self.config.enable_llm_reference_parsing:
                try:
                    # 优先使用 freeapi 配置（从插件配置读取）
                    api_url = getattr(self.config, 'freeapi_url', '') or ''
                    api_key = getattr(self.config, 'freeapi_key', '') or ''
                    if api_url and api_key:
                        effective_llm_config = {
                            "model": "gpt-4o-mini",
                            "api_base": f"{api_url}/v1",
                            "api_key": api_key
                        }
                        logger.debug("📝 使用 freeapi 配置进行 LLM 参考文献解析")
                    else:
                        # 回退到从 provider 提取配置信息
                        provider = await self._ensure_llm_initialized()
                        if provider:
                            model = getattr(provider, 'model', None) or getattr(provider, 'model_name', None)
                            api_base = getattr(provider, 'api_base', None) or getattr(provider, 'base_url', None)
                            api_key = getattr(provider, 'api_key', None) or getattr(provider, 'key', None)
                            if model and api_base:
                                effective_llm_config = {
                                    "model": model,
                                    "api_base": api_base,
                                    "api_key": api_key or "sk-placeholder"
                                }
                                logger.debug("📝 使用 Provider 配置进行 LLM 参考文献解析")
                            else:
                                logger.warning(f"⚠️ 无法从 Provider 提取完整配置，LLM 参考文献解析被禁用")
                except Exception as e:
                    logger.warning(f"⚠️ 无法获取 LLM 配置，LLM 参考文献解析被禁用: {e}")

            # 解析PDF并分块（传递 LLM config 和 arXiv client 以支持 LLM-based 引用解析）
            logger.info(f"📄 处理文件: {file_path}")
            nodes = await parser.parse_and_split(file_path, effective_llm_config, arxiv_client)

            if not nodes:
                return {
                    "status": "error",
                    "message": f"无法解析文件: {file_path}"
                }

            logger.info(f"📑 解析完成: {len(nodes)} 个节点")

            # 批量获取embeddings
            texts = [node.text for node in nodes]
            embeddings = await embed_provider.get_text_embeddings_batch(texts)

            logger.info(f"🔍 [DEBUG] 生成embeddings: texts数量={len(texts)}, embeddings数量={len(embeddings) if embeddings else 'None'}")

            # 使用 zip_longest 配对，确保不漏掉任何 node（缺失的 embedding 填 None）
            pairs = list(zip_longest(nodes, embeddings if embeddings else [None] * len(nodes), fillvalue=None))

            # 过滤掉无效节点（文本为空或embedding为None/无效的节点）
            valid_nodes = []
            valid_embeddings = []
            for i, (node, embedding) in enumerate(pairs):
                # 跳过空文本
                if not node or not node.text or not node.text.strip():
                    if node:
                        logger.warning(f"⚠️ 跳过空文本节点 {i}: text={node.text[:30] if node.text else 'N/A'}...")
                    continue
                # 跳过无效 embedding
                if embedding is None or not isinstance(embedding, list) or len(embedding) == 0:
                    logger.warning(f"⚠️ 跳过embedding无效的节点 {i}: text={node.text[:50]}..., embedding={embedding}")
                    continue
                valid_nodes.append(node)
                valid_embeddings.append(embedding)

            logger.info(f"🔍 配对后节点数: {len(pairs)}, 有效节点数: {len(valid_nodes)}, 有效embeddings数: {len(valid_embeddings)}")

            # 插入到索引
            count = await index_manager.insert_nodes(valid_nodes, valid_embeddings)

            # 刷新 BM25 索引（下次检索时自动重建）
            if self.config.enable_bm25:
                index_manager.refresh_bm25_index()

            logger.info(f"✅ 论文添加成功: {count} 个chunks")

            return {
                "status": "success",
                "chunks_added": count,
                "message": f"已添加 {count} 个chunks"
            }

        except Exception as e:
            logger.error(f"❌ 添加论文失败: {e}")
            return {
                "status": "error",
                "message": f"添加论文失败: {e}"
            }

    async def retrieve(self, query: str, top_k: Optional[int] = None, use_llm_fusion: bool = False) -> QueryResult:
        """
        检索相关文档（llama-index风格接口）

        Args:
            query: 查询文本
            top_k: 返回数量
            use_llm_fusion: 是否使用LLM智能决定融合权重（仅HybridRetriever有效）

        Returns:
            QueryResult对象，包含节点列表和分数
        """
        if top_k is None:
            top_k = self.config.top_k

        # 确保组件已初始化
        await self._ensure_embed_provider_initialized()
        self._ensure_index_manager_initialized()
        retriever = self._ensure_retriever_initialized()
        reranker = self._ensure_reranker_initialized()

        # ========== 确保 LLM 只加载一次 ==========
        # 后续所有操作（LLM融合、CRAG评估、CRAG修正）都复用同一个 LLM 实例
        llm_provider = None
        if use_llm_fusion and isinstance(retriever, HybridRetriever):
            llm_provider = await self._ensure_llm_initialized()
            if llm_provider:
                retriever.set_llm_provider(llm_provider)

        try:
            # 执行检索（根据类型选择参数）
            if isinstance(retriever, HybridRetriever):
                query_result = await retriever.retrieve(query, top_k, use_llm_fusion=use_llm_fusion)
            else:
                query_result = await retriever.retrieve(query, top_k)

            # ========== CRAG: 分层检索 + 质量评估 + 修正 ==========
            # 将检索结果转换为字典格式（用于评估和修正）
            results_dict = []
            for i, node in enumerate(query_result.nodes):
                results_dict.append({
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": query_result.scores[i] if i < len(query_result.scores) else 0.0
                })

            # 阶段1: CRAG 评估检索质量（使用 LLM 评估）
            # 确保 LLM 已加载（如果还未加载）
            if llm_provider is None:
                try:
                    llm_provider = await self._ensure_llm_initialized()
                except Exception:
                    pass

            evaluator = CragEvaluator(llm_provider=llm_provider)
            evaluation = await evaluator.evaluate_retrieval_quality(query, results_dict)
            logger.debug(f"[CRAG评估] query='{query[:50]}...', score={evaluation['score']:.2f} ({evaluation['level']})")

            # 阶段2: 如果质量低或中，执行修正策略
            if evaluation['level'] in ['low', 'medium']:
                # 复用已加载的 LLM provider
                if llm_provider is None:
                    try:
                        llm_provider = await self._ensure_llm_initialized()
                    except Exception:
                        llm_provider = None

                # 创建 CRAG 修正器（复用同一个 LLM provider）
                corrector = CragCorrector(
                    index_manager=self._index_manager,
                    embed_provider=self._embed_provider,
                    llm_provider=llm_provider
                )

                # 执行修正
                corrected_results = await corrector.correct(
                    query=query,
                    level=evaluation['level'],
                    original_results=results_dict
                )

                # 更新结果
                if corrected_results != results_dict:
                    logger.info(f"[CRAG] 修正完成: {len(results_dict)} → {len(corrected_results)} 条结果")
                    results_dict = corrected_results
            # ========== CRAG 修正结束 ==========

            # 如果启用了重排序且结果数量足够，进行重排序
            if reranker and len(results_dict) >= 3:
                # 执行重排序（使用 CRAG 修正后的结果）
                reranked_results = await reranker.rerank(
                    query=query,
                    results=results_dict,  # 使用 CRAG 修正后的结果
                    top_k=top_k
                )

                # 转换回Node列表
                nodes = []
                scores = []
                for item in reranked_results:
                    nodes.append(Node(
                        text=item["text"],
                        metadata=item["metadata"]
                    ))
                    scores.append(item.get("rerank_score", item.get("score", 0.0)))

                logger.info(f"✅ 重排序完成: {len(results_dict)} → {len(nodes)} 个节点")
                return QueryResult(nodes=nodes, scores=scores)

            # 无重排序时，返回 CRAG 修正后的结果
            nodes = [Node(text=r["text"], metadata=r["metadata"]) for r in results_dict]
            scores = [r.get("score", 0.0) for r in results_dict]
            return QueryResult(nodes=nodes, scores=scores)
        except Exception as e:
            logger.error(f"❌ 检索失败: {e}")
            return QueryResult(nodes=[], scores=[])

    async def search(
        self,
        query: str,
        mode: str = "rag",
        images: Optional[List[str]] = None,
        force_english: bool = False
    ) -> Dict[str, Any]:
        """
        搜索论文（支持多模态查询）

        Args:
            query: 查询文本
            mode: 模式
                - "rag": 检索 + RAG生成
                - "retrieve": 仅检索
            images: 图片路径列表（支持多模态查询）
            force_english: 强制使用英文回答（用于评估场景）

        Returns:
            搜索结果
        """
        try:
            if mode == "retrieve":
                # 仅检索模式
                query_result = await self.retrieve(query, self.config.top_k)

                results = []
                for i, node in enumerate(query_result.nodes):
                    results.append({
                        "text": node.text,
                        "metadata": node.metadata,
                        "score": query_result.scores[i] if i < len(query_result.scores) else 0.0
                    })

                return {
                    "type": "retrieve",
                    "query": query,
                    "sources": results,
                    "count": len(results)
                }
            else:
                # RAG生成模式（支持多模态）
                return await self._rag_query(query, images=images, force_english=force_english)

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return {
                "type": "error",
                "message": f"搜索失败: {e}"
            }

    async def search_for_qasper(
        self,
        query: str,
        mode: str = "rag",
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        搜索论文（专门用于Qasper数据集评估，生成简短答案）

        与 search() 的区别：
        - 答案更简短、直接，符合Qasper评估要求
        - 不要求引用来源
        - 不使用复杂的markdown格式

        Args:
            query: 查询文本
            mode: 模式 ("rag" 或 "retrieve")
            images: 图片路径列表（可选）

        Returns:
            {
                "type": "rag",
                "query": str,
                "answer": str,  # 简短答案
                "sources": List[Dict],  # 检索来源
                "images": Optional[List[str]]
            }
        """
        try:
            # 确保LLM已初始化（用于回答生成，也用于LLM融合权重决策）
            llm_provider = await self._ensure_llm_initialized()

            # 执行检索（启用LLM融合权重决策）
            query_result = await self.retrieve(query, self.config.top_k, use_llm_fusion=True)

            # ========== 检查检索质量，若极低则返回空答案（模仿不可回答问题）==========
            # 使用 CRAG 评估器检查检索质量
            results_dict_for_eval = []
            for i, node in enumerate(query_result.nodes):
                results_dict_for_eval.append({
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": query_result.scores[i] if i < len(query_result.scores) else 0.0
                })

            # 快速检查：若检索结果为空或分数极低，直接返回空答案
            if len(results_dict_for_eval) == 0:
                logger.info(f"[Qasper] 检索无结果，返回空答案（不可回答）")
                return {
                    "type": "rag",
                    "query": query,
                    "answer": "",  # 空答案表示不可回答
                    "sources": [],
                    "unanswerable": True
                }

            # 使用 LLM 评估检索质量（仅在有结果但质量可能不足时）
            max_score = max(r["score"] for r in results_dict_for_eval) if results_dict_for_eval else 0.0

            # ========== 策略1: 动态检索阈值 ==========
            # 根据问题类型调整阈值：none类型用更严格标准
            unanswerable_patterns = [
                "does the paper", "does not", "doesn't", "is not mentioned",
                "not mention", "whether the paper", "is there any", "what is the name of",
                "who proposed", "who suggested", "when was", "where did"
            ]
            is_none_type_query = any(p in query.lower() for p in unanswerable_patterns)
            threshold = 0.35 if is_none_type_query else 0.15  # 不可回答问题用更高阈值

            if max_score < threshold:  # 分数低于动态阈值，认为不可回答
                logger.info(f"[Qasper] 检索质量低于阈值（max_score={max_score:.3f}, threshold={threshold}），返回空答案（不可回答）")
                return {
                    "type": "rag",
                    "query": query,
                    "answer": "",
                    "sources": [],
                    "unanswerable": True
                }

            if len(query_result) == 0 and not images:
                return {
                    "type": "error",
                    "message": "未找到相关文档",
                    "sources": []
                }

            # 转换为源文档格式
            sources = []
            for i, node in enumerate(query_result.nodes):
                sources.append({
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": query_result.scores[i] if i < len(query_result.scores) else 0.0
                })

            # VLM路由判断（复用原有逻辑）
            final_images = None
            if images:
                final_images = images
            elif self._should_use_vlm(query, sources):
                # 传入force_all_paper_images=True，当检测到视觉内容但无明确图片时获取论文所有图片
                final_images = self._extract_image_paths_from_sources(sources, force_all_paper_images=True)
                if not final_images:
                    logger.info(f"📝 Qasper评估检测到视觉内容但无关联图片，使用LLM模式")

            # 生成简短答案（使用专门的Qasper评估函数）
            answer = await self._generate_answer_for_qasper(
                llm_provider, query, sources, images=final_images
            )

            return {
                "type": "rag",
                "query": query,
                "answer": answer,
                "sources": sources,
                "images": final_images
            }

        except Exception as e:
            logger.error(f"❌ Qasper评估搜索失败: {e}")
            return {
                "type": "error",
                "message": f"Qasper评估搜索失败: {e}",
                "sources": []
            }

    async def search_for_qasper_llm_only(
        self,
        query: str,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        纯LLM回答（无检索）- 用于基线对比评估

        直接使用LLM回答问题，不进行任何检索。
        用于评估纯LLM基线性能。

        Args:
            query: 查询文本
            images: 图片路径列表（可选）

        Returns:
            {
                "type": "llm_only",
                "query": str,
                "answer": str,  # 简短答案
                "sources": [],
                "images": Optional[List[str]]
            }
        """
        try:
            # 确保LLM已初始化
            llm_provider = await self._ensure_llm_initialized()

            # 直接使用LLM生成答案（不进行检索）
            answer = await self._generate_answer_for_qasper(
                llm_provider, query, sources=[], images=images
            )

            return {
                "type": "llm_only",
                "query": query,
                "answer": answer,
                "sources": [],
                "images": images
            }

        except Exception as e:
            logger.error(f"❌ 纯LLM回答失败: {e}")
            return {
                "type": "error",
                "message": f"纯LLM回答失败: {e}",
                "sources": []
            }

    async def _rag_query(
        self,
        query: str,
        images: Optional[List[str]] = None,
        force_english: bool = False
    ) -> Dict[str, Any]:
        """
        检索 + RAG生成（支持多模态）

        多模态核心流程：
        1. 检索相关文档
        2. 路由判断：查询含视觉关键词 OR 检索结果有关联图片 → 使用VLM
        3. VLM模式：从检索结果提取图片路径，传递给多模态模型
        4. LLM模式：纯文本生成

        Args:
            query: 查询文本
            images: 用户直接上传的图片路径列表（可选）
            force_english: 强制使用英文回答
        """
        try:
            # 确保LLM已初始化
            llm_provider = await self._ensure_llm_initialized()

            # 执行检索
            query_result = await self.retrieve(query, self.config.top_k)

            if len(query_result) == 0 and not images:
                return {
                    "type": "error",
                    "message": "未找到相关文档",
                    "sources": []
                }

            # 转换为源文档格式
            sources = []
            for i, node in enumerate(query_result.nodes):
                sources.append({
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": query_result.scores[i] if i < len(query_result.scores) else 0.0
                })

            # VLM路由判断
            # 用户直接上传图片 → 直接使用VLM
            # 否则根据查询和检索结果自动判断
            if images:
                # 用户上传了图片，直接使用VLM
                final_images = images
                logger.info(f"🖼️ 用户上传 {len(images)} 张图片，使用VLM模式")
            else:
                # 自动路由：根据查询和检索结果判断
                if self._should_use_vlm(query, sources):
                    # 从检索结果提取关联图片（force_all_paper_images=True当检测到视觉内容时获取论文所有图片）
                    final_images = self._extract_image_paths_from_sources(sources, force_all_paper_images=True)
                    if final_images:
                        logger.info(f"🖼️ 检索到 {len(final_images)} 张关联图片，使用VLM模式")
                        for i, img in enumerate(final_images):
                            exists = os.path.exists(img) if img else False
                            size = os.path.getsize(img) if exists else 0
                    else:
                        # 有视觉关键词但没有关联图片，回退到LLM
                        logger.info(f"📝 检测到视觉关键词但无关联图片，使用LLM模式")
                        final_images = None
                else:
                    final_images = None
                    logger.info(f"📝 纯文本查询，使用LLM模式")


            # 生成答案
            answer = await self._generate_answer_with_llm(
                llm_provider, query, sources, images=final_images, force_english=force_english
            )

            return {
                "type": "rag",
                "query": query,
                "answer": answer,
                "sources": sources,
                "images": final_images
            }

        except Exception as e:
            logger.error(f"❌ RAG生成失败: {e}")
            return {
                "type": "error",
                "message": f"RAG生成失败: {e}",
                "sources": []  # 确保错误响应也包含 sources 键
            }

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """将图片编码为base64字符串"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_data
        except Exception as e:
            logger.error(f"❌ 图片编码失败 {image_path}: {e}")
            return None

    # VLM 图片 transform 配置
    VLM_MAX_IMAGE_SIZE = (1024, 1024)  # VLM 输入图片最大尺寸

    def _transform_image_for_vlm(self, image_path: str) -> Optional[str]:
        """
        将图片 transform 后返回 base64（用于 VLM）

        多模态：保存时存原图，查询时统一 transform
        - 压缩到 VLM 合适的大小
        - 转为 base64 供 VLM 使用

        Args:
            image_path: 图片路径

        Returns:
            base64 编码字符串，失败返回 None
        """
        try:
            from PIL import Image
            import io

            # 打开原图
            image = Image.open(image_path)
            original_size = image.size

            # 计算新的尺寸（保持宽高比）
            max_w, max_h = self.VLM_MAX_IMAGE_SIZE
            if image.size[0] > max_w or image.size[1] > max_h:
                image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            # 转换为 RGB（如果是 RGBA）
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            # 压缩为 JPEG 并返回 base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85, optimize=True)
            encoded_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # 计算压缩率
            new_size = len(buffered.getvalue()) / 1024  # KB
            original_size_kb = Path(image_path).stat().st_size / 1024

            logger.debug(f"🖼️ [VLM Transform] {Path(image_path).name}: {original_size[0]}x{original_size[1]} → {image.size[0]}x{image.size[1]}, {original_size_kb:.1f}KB → {new_size:.1f}KB")

            return encoded_data

        except Exception as e:
            logger.error(f"❌ 图片 transform 失败 {image_path}: {e}")
            return None

    def _extract_answer_from_response(self, response: Any) -> str:
        """
        从 LLM Response 中提取答案文本

        Args:
            response: LLM Provider 返回的响应对象

        Returns:
            答案文本字符串
        """
        # 方法1：检查是否有 choices 属性（OpenAI 格式）
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content

        # 方法2：检查是否是 LLMResponse 对象（AstrBot 格式）
        # LLMResponse.result_chain 包含 MessageChain
        if hasattr(response, 'result_chain'):
            chain = getattr(response.result_chain, 'chain', None)
            if chain and len(chain) > 0:
                first = chain[0]
                # 检查是否有 get_text 方法
                if hasattr(first, 'get_text'):
                    return first.get_text()
                # 检查是否有 text 属性
                if hasattr(first, 'text'):
                    return first.text

        # 方法3：尝试从 raw_completion 提取
        if hasattr(response, 'raw_completion'):
            raw = response.raw_completion
            if hasattr(raw, 'choices') and raw.choices:
                return raw.choices[0].message.content

        # 方法4：检查是否有 content 属性（我们的 LLMResponse 格式）
        if hasattr(response, 'content'):
            return response.content

        # 方法5：返回字符串形式
        return str(response)

    async def _generate_answer_with_llm(
        self,
        llm_provider: Any,
        query: str,
        sources: List[Dict[str, Any]],
        images: Optional[List[str]] = None,
        force_english: bool = False
    ) -> str:
        """
        使用LLM Provider生成答案（支持多模态）

        Args:
            llm_provider: LLM Provider
            query: 查询文本
            sources: 检索到的源文档
            images: 图片路径列表（可选）
            force_english: 强制使用英文回答
        """
        try:
            # 构建上下文
            context_parts = []
            for i, src in enumerate(sources):
                file_name = src['metadata'].get('file_name', 'unknown')
                text = src['text']
                context_parts.append(f"[来源{i+1}] {file_name}\n{text}")

            context = "\n\n".join(context_parts)

            # 构建提示（根据 force_english 选择语言）
            if force_english:
                text_prompt = f"""Based on the following paper content, answer the question. Respond in English only.

{context}

Question: {query}

Please provide a detailed answer and cite the relevant sources."""
            else:
                text_prompt = f"""基于以下论文内容回答问题：

{context}

问题：{query}

请提供详细的回答，并引用相关文献。"""

            # 判断使用哪个Provider
            # 多模态：优先使用配置的 multimodal_provider_id
            # 文本：使用 text_provider_id 或当前provider
            use_multimodal = images is not None
            provider_to_use = llm_provider  # 默认使用文本provider


            if use_multimodal and self.config.multimodal_provider_id:
                try:
                    provider_manager = getattr(self.context, "provider_manager", None)
                    if provider_manager:
                        inst_map = getattr(provider_manager, "inst_map", None)
                        if isinstance(inst_map, dict):
                            vlm_provider = inst_map.get(self.config.multimodal_provider_id)
                            if vlm_provider:
                                provider_to_use = vlm_provider
                                provider_name = getattr(vlm_provider, 'model_name', None) or getattr(vlm_provider, 'model', None) or 'unknown'
                                logger.info(f"🖼️ 使用VLM模式，多模态Provider: {self.config.multimodal_provider_id} (实际provider: {provider_name})")
                            else:
                                logger.warning(f"⚠️ 未找到多模态Provider: {self.config.multimodal_provider_id}，回退到文本模式")
                                use_multimodal = False
                except Exception as e:
                    logger.warning(f"⚠️ 获取多模态Provider失败: {e}，回退到文本模式")
                    use_multimodal = False
            elif use_multimodal:
                # 优先复用传入的 llm_provider（避免重复加载）
                if llm_provider is not None and hasattr(llm_provider, 'text_chat'):
                    provider_to_use = llm_provider
                    logger.debug("[VLM] 复用已加载的 LLM Provider 进行多模态生成")
                else:
                    # 尝试使用缓存的 Provider
                    try:
                        from .llama_cpp_vlm_provider import get_cached_llama_cpp_provider
                    except ImportError:
                        from llama_cpp_vlm_provider import get_cached_llama_cpp_provider

                    cached = get_cached_llama_cpp_provider()
                    if cached is not None:
                        provider_to_use = cached
                        logger.debug("[VLM] 复用缓存的 LlamaCpp Provider 进行多模态生成")
                    elif LLAMA_CPP_VLM_AVAILABLE:
                        # 需要初始化新的 LlamaCpp VLM Provider
                        try:
                            # 解析路径（相对路径基于插件目录）
                            llama_model_path_raw = getattr(self.config, 'llama_vlm_model_path', './models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf')
                            llama_mmproj_path_raw = getattr(self.config, 'llama_vlm_mmproj_path', './models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf')

                            llama_model_path = str((_PLUGIN_DIR / llama_model_path_raw).resolve())
                            llama_mmproj_path = str((_PLUGIN_DIR / llama_mmproj_path_raw).resolve())

                            llama_max_tokens = getattr(self.config, 'llama_vlm_max_tokens', 2560)
                            llama_temperature = getattr(self.config, 'llama_vlm_temperature', 0.7)
                            llama_n_ctx = getattr(self.config, 'llama_vlm_n_ctx', 4096)
                            llama_n_gpu_layers = getattr(self.config, 'llama_vlm_n_gpu_layers', 99)
                            try:
                                from .llama_cpp_vlm_provider import init_llama_cpp_vlm_provider
                            except ImportError:
                                from llama_cpp_vlm_provider import init_llama_cpp_vlm_provider
                            provider_to_use = init_llama_cpp_vlm_provider(
                                model_path=llama_model_path,
                                mmproj_path=llama_mmproj_path,
                                n_ctx=llama_n_ctx,
                                n_gpu_layers=llama_n_gpu_layers,
                                max_tokens=llama_max_tokens,
                                temperature=llama_temperature
                            )
                            logger.info(f"🖼️ 初始化 LlamaCpp VLM Provider: {llama_model_path}")
                        except Exception as e:
                            logger.error(f"❌ Llama.cpp VLM Provider初始化失败: {e}，回退到文本模式")
                            use_multimodal = False
                    else:
                        logger.warning("⚠️ Llama.cpp VLM Provider不可用，回退到文本模式")
                        use_multimodal = False

            # 使用多模态模式
            if use_multimodal:
                # 直接使用本地文件路径（不限制图片数量）
                assert images is not None
                image_urls = [str(Path(img_path).resolve()) for img_path in images]

                logger.info(f"🖼️ 使用VLM模式查询 (图片数: {len(image_urls)})")

                # 验证图片文件
                for i, img_url in enumerate(image_urls):
                    exists = os.path.exists(img_url)
                    size = os.path.getsize(img_url) if exists else 0
                    if not exists:
                        logger.warning(f"⚠️ 图片文件不存在: {img_url}")

                # 使用 text_chat 接口（AstrBot Provider 统一接口）
                # image_urls 支持本地文件路径或 URL
                start_time = time.time()
                if hasattr(provider_to_use, 'text_chat'):
                    try:
                        response = await provider_to_use.text_chat(
                            prompt=text_prompt,
                            image_urls=image_urls if image_urls else None,
                            contexts=[],
                            temperature=0.7
                        )

                        answer = self._extract_answer_from_response(response)
                        elapsed = time.time() - start_time
                        logger.info(f"✅ VLM模式完成，耗时: {elapsed:.2f}秒")
                        return answer
                    except Exception as e:
                        elapsed = time.time() - start_time
                        logger.error(f"❌ VLM模式失败: {e}，耗时: {elapsed:.2f}秒")
                        # 回退到LLM文本模式
                        use_multimodal = False
                        llm_provider = await self._ensure_llm_initialized()
                        if hasattr(llm_provider, 'text_chat'):
                            response = await llm_provider.text_chat(
                                prompt=text_prompt,
                                contexts=[],
                                temperature=0.7,
                                max_tokens=2000
                            )
                            answer = self._extract_answer_from_response(response)
                            return answer
                        elif hasattr(llm_provider, 'chat'):
                            messages = [
                                {"role": "user", "content": text_prompt}
                            ]
                            response = await llm_provider.chat.completions.create(
                                messages=messages,
                                temperature=0.7,
                                max_tokens=2000
                            )
                            answer = self._extract_answer_from_response(response)
                            return answer
                        else:
                            raise ValueError("Provider不支持文本聊天")
                else:
                    raise ValueError(f"Provider不支持多模态聊天: 缺少text_chat方法 (provider: {type(provider_to_use).__name__})")
            else:
                # 使用文本模式
                logger.info(f"📝 使用文本模式查询")

                if hasattr(llm_provider, 'text_chat'):
                    # 使用 text_chat 接口
                    response = await llm_provider.text_chat(
                        prompt=text_prompt,
                        contexts=[],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    answer = self._extract_answer_from_response(response)
                    return answer
                elif hasattr(llm_provider, 'chat'):
                    # 使用 chat.completions.create 接口
                    messages = [
                        {"role": "user", "content": text_prompt}
                    ]
                    response = await llm_provider.chat.completions.create(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    answer = self._extract_answer_from_response(response)
                    return answer
                else:
                    raise ValueError("Provider不支持文本聊天")

        except Exception as e:
            logger.error(f"❌ LLM生成失败: {e}")
            return f"生成答案失败: {e}"

    async def _generate_answer_for_qasper(
        self,
        llm_provider: Any,
        query: str,
        sources: List[Dict[str, Any]],
        images: Optional[List[str]] = None,
    ) -> str:
        """
        使用LLM Provider生成答案（专门用于Qasper数据集评估）

        与 _generate_answer_with_llm 的区别：
        - 生成简短、直接的答案（Qasper评估需要）
        - 不要求引用来源
        - 不使用复杂的markdown格式

        Qasper答案格式要求：
        - 布尔型：直接回答 "Yes" 或 "No"
        - 抽取型：关键短语/词组
        - 抽象型：1-2句话概括

        Args:
            llm_provider: LLM Provider
            query: 查询文本
            sources: 检索到的源文档
            images: 图片路径列表（可选）

        Returns:
            简短、直接的答案文本
        """
        try:
            # 构建上下文
            context_parts = []
            for i, src in enumerate(sources):
                file_name = src['metadata'].get('file_name', 'unknown')
                text = src['text']
                context_parts.append(f"[Source {i+1}] {text}")

            context = "\n\n".join(context_parts)

            # ========== 策略4: 证据链验证（仅在有检索上下文时启用）==========
            # 纯LLM模式（无检索上下文）跳过证据验证
            if sources:
                # 先让模型提取证据，再验证是否支持答案
                evidence_prompt = f"""Extract the specific sentences from the following paper content that directly answer the question.

{context}

Question: {query}

Output format:
- If evidence exists: list the exact sentences that contain the answer
- If no evidence exists: output "NO_EVIDENCE"

Evidence sentences:"""

                if hasattr(llm_provider, 'text_chat'):
                    evidence_response = await llm_provider.text_chat(
                        prompt=evidence_prompt,
                        contexts=[],
                        temperature=0.1,
                        max_tokens=4096
                    )
                    evidence_text = self._extract_answer_from_response(evidence_response)

                    if "NO_EVIDENCE" in evidence_text.upper() or not evidence_text.strip():
                        logger.info(f"[Qasper-Evidence] 未找到支持证据，返回空答案")
                        return ""

            # 构建简短答案的prompt（专门用于Qasper评估）
            if sources:
                # RAG模式：基于检索内容回答
                text_prompt = f"""Based ONLY on the following paper content, answer the question concisely and directly.

{context}

Question: {query}

Answer requirements:
- If the question is yes/no, answer ONLY "Yes" or "No" without any explanation
- If the answer is a specific fact, number, or name, give ONLY that answer
- If elaboration is needed, keep it to 1-2 short sentences maximum
- If the content does NOT contain information to answer the question, output exactly "NO_ANSWER"
- Do NOT use bullet points, lists, or markdown formatting
- Do NOT cite sources or reference numbers"""
            else:
                # 纯LLM模式：基于模型知识回答
                text_prompt = f"""Answer the following question about a research paper concisely and directly.

Question: {query}

Answer requirements:
- If the question is yes/no, answer ONLY "Yes" or "No" without any explanation
- If the answer is a specific fact, number, or name, give ONLY that answer
- If elaboration is needed, keep it to 1-2 short sentences maximum
- Do NOT use bullet points, lists, or markdown formatting"""

            # 判断使用哪个Provider
            use_multimodal = images is not None
            provider_to_use = llm_provider

            if use_multimodal and self.config.multimodal_provider_id:
                try:
                    provider_manager = getattr(self.context, "provider_manager", None)
                    if provider_manager:
                        inst_map = getattr(provider_manager, "inst_map", None)
                        if isinstance(inst_map, dict):
                            vlm_provider = inst_map.get(self.config.multimodal_provider_id)
                            if vlm_provider:
                                provider_to_use = vlm_provider
                                logger.info(f"🖼️ Qasper评估使用VLM模式: {self.config.multimodal_provider_id}")
                except Exception as e:
                    logger.warning(f"⚠️ 获取多模态Provider失败: {e}，回退到文本模式")
                    use_multimodal = False
            elif use_multimodal and LLAMA_CPP_VLM_AVAILABLE:
                # 优先复用传入的 llm_provider（避免重复加载）
                if llm_provider is not None and hasattr(llm_provider, 'text_chat'):
                    provider_to_use = llm_provider
                    logger.debug("[Qasper] 复用已加载的 LLM Provider 进行多模态生成")
                else:
                    # 需要加载新的 LlamaCpp VLM Provider
                    try:
                        llama_model_path_raw = getattr(self.config, 'llama_vlm_model_path', './models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf')
                        llama_mmproj_path_raw = getattr(self.config, 'llama_vlm_mmproj_path', './models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf')
                        llama_model_path = str((_PLUGIN_DIR / llama_model_path_raw).resolve())
                        llama_mmproj_path = str((_PLUGIN_DIR / llama_mmproj_path_raw).resolve())
                        llama_max_tokens = getattr(self.config, 'llama_vlm_max_tokens', 2560)
                        llama_temperature = getattr(self.config, 'llama_vlm_temperature', 0.7)
                        llama_n_ctx = getattr(self.config, 'llama_vlm_n_ctx', 4096)
                        llama_n_gpu_layers = getattr(self.config, 'llama_vlm_n_gpu_layers', 99)
                        try:
                            from .llama_cpp_vlm_provider import (
                                get_llama_cpp_vlm_provider,
                                get_cached_llama_cpp_provider,
                                init_llama_cpp_vlm_provider,
                            )
                        except ImportError:
                            from llama_cpp_vlm_provider import (
                                get_llama_cpp_vlm_provider,
                                get_cached_llama_cpp_provider,
                                init_llama_cpp_vlm_provider,
                            )
                        # 优先复用已初始化的单例
                        cached = get_cached_llama_cpp_provider()
                        if cached is not None:
                            provider_to_use = cached
                            logger.debug("[Qasper] 复用已缓存的 LlamaCppVLMProvider")
                        else:
                            provider_to_use = init_llama_cpp_vlm_provider(
                                model_path=llama_model_path,
                                mmproj_path=llama_mmproj_path,
                                n_ctx=llama_n_ctx,
                                n_gpu_layers=llama_n_gpu_layers,
                                max_tokens=llama_max_tokens,
                                temperature=llama_temperature
                            )
                        await provider_to_use.initialize()
                        logger.info(f"🖼️ Qasper评估使用Llama.cpp VLM Provider")
                    except Exception as e:
                        logger.warning(f"⚠️ Llama.cpp VLM初始化失败: {e}，回退到文本模式")
                        use_multimodal = False

            if use_multimodal:
                # 多模态模式
                image_urls = [str(Path(img_path).resolve()) for img_path in (images or [])]
                if hasattr(provider_to_use, 'text_chat'):
                    response = await provider_to_use.text_chat(
                        prompt=text_prompt,
                        image_urls=image_urls if image_urls else None,
                        contexts=[],
                        temperature=0.7,
                        max_tokens=500
                    )
                    return self._extract_answer_from_response(response)
                else:
                    raise ValueError(f"Provider不支持多模态聊天: {type(provider_to_use).__name__}")
            else:
                # 文本模式
                if hasattr(provider_to_use, 'text_chat'):
                    response = await provider_to_use.text_chat(
                        prompt=text_prompt,
                        contexts=[],
                        temperature=0.7,
                        max_tokens=500
                    )
                    answer = self._extract_answer_from_response(response)

                    # ========== 策略4: 检查NO_ANSWER ==========
                    # 如果模型输出NO_ANSWER，返回空字符串
                    if "NO_ANSWER" in answer.upper():
                        logger.info(f"[Qasper-Evidence] 模型判断无法回答，返回空答案")
                        return ""

                    return answer
                else:
                    raise ValueError(f"Provider不支持文本聊天: {type(provider_to_use).__name__}")

        except Exception as e:
            logger.error(f"❌ Qasper答案生成失败: {e}")
            return ""

    def _extract_evidence_spans(self, sources: List[Dict[str, Any]], query: str, answer: str = "", max_chars: int = 500) -> List[str]:
        """
        从检索到的chunks中提取精确的evidence spans

        与直接返回完整chunk的区别：
        - Gold evidence是标注员高亮的特定短文本（通常100-300字符）
        - 直接返回完整chunk会导致Evidence F1很低（因为precision低）

        策略：
        1. 将chunk分割成句子
        2. 找出包含答案关键词或查询关键词的句子
        3. 返回最相关的句子（总长度限制在max_chars内）

        Args:
            sources: 检索到的源文档列表
            query: 查询文本
            answer: 生成的答案（用于找关键词）
            max_chars: 最大总字符数

        Returns:
            精确的evidence spans列表
        """
        import re

        # 提取关键词
        query_keywords = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
        answer_keywords = set(re.findall(r'\b[a-z]{4,}\b', answer.lower())) if answer else set()

        # 合并关键词（答案关键词权重更高）
        all_keywords = query_keywords | answer_keywords

        evidence_spans = []

        for src in sources:
            text = src.get("text", "")
            if not text:
                continue

            # 分割句子（简单按句号/换行分割）
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # 对每个句子评分
            scored_sentences = []
            for sent in sentences:
                sent_lower = sent.lower()
                # 计算关键词匹配数
                query_matches = sum(1 for kw in query_keywords if kw in sent_lower)
                answer_matches = sum(1 for kw in answer_keywords if kw in sent_lower) * 2  # 答案词权重更高
                # 数字匹配（答案中可能有数字）
                answer_nums = re.findall(r'\b\d+\b', answer.lower()) if answer else []
                num_matches = sum(1 for num in answer_nums if num in sent_lower)

                score = query_matches + answer_matches + num_matches
                if score > 0:
                    scored_sentences.append((score, sent))

            # 按分数排序，选取高分的句子
            scored_sentences.sort(key=lambda x: x[0], reverse=True)

            # 选取句子直到达到max_chars
            current_len = sum(len(e) for e in evidence_spans)
            for score, sent in scored_sentences:
                if current_len + len(sent) <= max_chars:
                    evidence_spans.append(sent)
                    current_len += len(sent)
                # 限制每个source最多取2个句子
                src_sent_count = sum(1 for s in evidence_spans if s in [x[1] for x in scored_sentences[:scored_sentences.index((score, sent))+1]])
                if src_sent_count >= 2:
                    break

        # 去重并保持顺序
        seen = set()
        unique_spans = []
        for span in evidence_spans:
            span_key = span[:50]  # 用前50字符作为key去重
            if span_key not in seen:
                seen.add(span_key)
                unique_spans.append(span)

        return unique_spans[:5]  # 最多返回5个evidence spans

    # ==================== VLM路由逻辑 ====================

    # 视觉关键词（用于检测是否需要VLM）
    VISUAL_KEYWORDS = [
        "图", "figure", "chart", "plot", "graph",
        "表格", "table", "公式", "formula", "equation",
        "架构", "architecture", "diagram", "示意图",
        "图像", "picture", "photo", "image",
        "多少", "数值", "数据", "哪个大", "哪个小",
        "颜色", "曲线", "峰值", "趋势"
    ]

    # 更严格的视觉内容检测模式（考虑chunk文本内容）
    STRICT_VISUAL_PATTERNS = [
        r'\bfigure\s*\d+', r'\bfig\.\s*\d+', r'\btable\s*\d+',
        r'\bchart\s*\d+', r'\bplot\s*\d+', r'\balgorithm\s*\d+',
        r'\bstep\s*\d+', r'\bexperiment\s*\d+', r'\bexample\s*\d+',
        r'\bdiagram\s*\d+', r'\barchitecture\s*\d+',
    ]

    # 扩展模式：查询可能需要图表/视觉内容来回答（数量、比较、性能等）
    QUERY_NEEDS_VISUAL_PATTERNS = [
        # 数量/大小类 (12.3%)
        r'how\s+(big|large|many|much)',
        r'what\s+is\s+the\s+size',
        r'number\s+of',
        r'how\s+many',
        r'count\s+of',
        r'how\s+long\s+does',
        # 比较类 (3.7%)
        r'which\s+is\s+(better|larger|smaller|higher|lower|more|less)',
        r'compare\s+to',
        r'compared\s+with',
        r'outperform',
        r'better\s+than',
        r'worse\s+than',
        # 性能指标类 (2.2%)
        r'\baccuracy',
        r'\bprecision',
        r'\brecall',
        r'\bf1\s*score',
        r'\bauc\b',
        r'\broc\b',
        r'\bbleu\b',
        r'\brouge\b',
        r'\bmeteor\b',
        r'\bperplexity',
        r'\bloss\b',
        r'\berror\s+rate',
        r'\bperformance\b',
        r'\bresult\b',
        r'\bscore\b',
        # 分布/趋势类
        r'\bdistribution\b',
        r'\boverview\b',
        r'\bsummary\b',
        r'\bbreakdown\b',
    ]

    def _should_use_vlm(self, query: str, sources: List[Dict[str, Any]]) -> bool:
        """
        判断是否应该使用VLM（多模态模型）

        核心路由逻辑：
        1. 查询关键词检测（视觉相关词汇）
        2. 查询扩展模式检测（数量、比较、性能等）
        3. 检索结果检测（是否关联了原图）
        4. 检索文本检测（是否提及视觉内容如图、表等）

        Args:
            query: 用户查询
            sources: 检索到的源文档

        Returns:
            True: 使用VLM，False: 使用纯文本LLM
        """
        import re

        query_lower = query.lower()

        # 检查查询是否包含视觉关键词
        query_has_visual = any(
            keyword in query_lower
            for keyword in self.VISUAL_KEYWORDS
        )

        # 检查查询是否匹配扩展模式（数量、比较、性能等）
        query_needs_visual = any(
            re.search(pattern, query_lower)
            for pattern in self.QUERY_NEEDS_VISUAL_PATTERNS
        )

        # 检查检索结果是否关联了图片（普通PDF场景）
        sources_have_images = any(
            src.get("metadata", {}).get("image_path")
            for src in sources
        )

        # 检查检索结果是否包含图表caption节点（Qasper等数据集场景）
        sources_have_figure_captions = any(
            src.get("metadata", {}).get("node_type") == "figure_caption"
            or src.get("metadata", {}).get("figure_file")
            for src in sources
        )

        # 检查检索文本是否提及视觉内容（Figure X, Table Y, etc.）
        sources_text_mentions_visual = False
        for src in sources:
            text = src.get("text", "")
            for pattern in self.STRICT_VISUAL_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    sources_text_mentions_visual = True
                    break
            if sources_text_mentions_visual:
                break

        # VLM条件：
        # - 查询涉及视觉 OR
        # - 查询匹配数量/比较/性能等模式 OR
        # - 检索结果有图片 OR
        # - 检索结果有图表captions OR
        # - 检索文本提及视觉内容（Figure/Table等）
        use_vlm = (
            query_has_visual
            or query_needs_visual
            or sources_have_images
            or sources_have_figure_captions
            or sources_text_mentions_visual
        )

        logger.debug(f"🔍 [VLM路由] query={query[:50]}..., query_has_visual={query_has_visual}, query_needs_visual={query_needs_visual}, sources_have_images={sources_have_images}, sources_have_figure_captions={sources_have_figure_captions}, sources_text_mentions_visual={sources_text_mentions_visual} → use_vlm={use_vlm}")

        return use_vlm

    def _extract_image_paths_from_sources(self, sources: List[Dict[str, Any]], force_all_paper_images: bool = False) -> List[str]:
        """
        从检索结果中提取所有关联的图片路径

        Args:
            sources: 检索到的源文档
            force_all_paper_images: 若为True，当检测到视觉内容但无明确图片时，提取论文所有图片

        Returns:
            图片路径列表（去重，最多返回3个）
        """
        import re

        image_paths = []
        seen = set()
        paper_ids_with_visual_content = set()

        # 获取插件目录（用于构建Qasper图片路径）
        plugin_dir = Path(__file__).parent

        for src in sources:
            metadata = src.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            # 1. 直接的 image_path 字段（普通PDF场景）
            image_path = metadata.get("image_path")
            if image_path and image_path not in seen:
                seen.add(image_path)
                image_paths.append(image_path)
                continue  # 已找到图片，跳过figure_file处理

            # 2. figure_file 字段（Qasper等数据集场景）
            figure_file = metadata.get("figure_file")
            paper_id = metadata.get("paper_id")
            if figure_file and paper_id:
                # Qasper图片路径格式: {plugin_dir}/datasets/test_figures_and_tables/{paper_id}/{figure_file}
                qasper_fig_path = str(plugin_dir / "datasets" / "test_figures_and_tables" / paper_id / figure_file)
                if qasper_fig_path not in seen:
                    seen.add(qasper_fig_path)
                    image_paths.append(qasper_fig_path)
                continue

            # 3. 检查文本是否提及视觉内容
            text = src.get("text", "")
            for pattern in self.STRICT_VISUAL_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    if paper_id:
                        paper_ids_with_visual_content.add(paper_id)
                    break

        # 4. 若检测到视觉内容但没有找到明确图片，尝试获取论文所有图片
        if force_all_paper_images and not image_paths and paper_ids_with_visual_content:
            for pid in paper_ids_with_visual_content:
                fig_dir = plugin_dir / "datasets" / "test_figures_and_tables" / pid
                if fig_dir.exists():
                    for img_file in fig_dir.iterdir():
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                            img_path = str(img_file.resolve())
                            if img_path not in seen:
                                seen.add(img_path)
                                image_paths.append(img_path)
                                if len(image_paths) >= 3:
                                    break
                if len(image_paths) >= 3:
                    break

        # 限制最多3张图片
        return image_paths[:3]

    async def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档（按文件分组）"""
        try:
            index_manager = self._ensure_index_manager_initialized()
            documents = await index_manager.list_unique_documents()

            if not documents:
                return [{
                    "file_name": "No documents",
                    "chunk_count": 0,
                    "added_time": "unknown"
                }]

            return documents

        except Exception as e:
            logger.error(f"❌ 列出文档失败: {e}")
            return []

    async def clear_index(self) -> Dict[str, Any]:
        """清空知识库"""
        try:
            index_manager = self._ensure_index_manager_initialized()
            success = await index_manager.clear()

            if success:
                self._retriever_initialized = False
                self._retriever = cast(Any, None)
                # 刷新 BM25 索引
                if self.config.enable_bm25:
                    index_manager.refresh_bm25_index()
                return {
                    "status": "success",
                    "message": "知识库已清空"
                }
            else:
                return {
                    "status": "error",
                    "message": "清空失败"
                }

        except Exception as e:
            logger.error(f"❌ 清空知识库失败: {e}")
            return {
                "status": "error",
                "message": f"清空失败: {e}"
            }

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            index_manager = self._ensure_index_manager_initialized()
            return await index_manager.get_stats()

        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {}

    # ==================== 向后兼容的别名方法 ====================

    async def list_papers(self) -> List[Dict[str, Any]]:
        """列出所有论文（别名方法，向后兼容）"""
        return await self.list_documents()

    async def clear(self) -> Dict[str, Any]:
        """清空知识库（别名方法，向后兼容）"""
        return await self.clear_index()

    async def delete_paper(self, file_name: str) -> Dict[str, Any]:
        """
        删除指定论文的向量数据

        Args:
            file_name: 要删除的文件名

        Returns:
            删除结果
        """
        try:
            index_manager = self._ensure_index_manager_initialized()
            result = await index_manager.delete_by_file_name(file_name)

            if result.get("status") == "success":
                # 重置检索器（因为底层数据已改变）
                self._retriever_initialized = False
                self._retriever = cast(Any, None)
                # 刷新 BM25 索引
                if self.config.enable_bm25:
                    index_manager.refresh_bm25_index()

            return result

        except Exception as e:
            logger.error(f"❌ 删除论文 '{file_name}' 失败: {e}")
            return {
                "status": "error",
                "message": f"删除论文失败: {e}"
            }
