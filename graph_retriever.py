"""
Graph Retriever - 知识图谱检索器和融合检索器

包含：
1. KnowledgeGraphRetriever - 知识图谱检索器
2. FusionRetriever - 融合检索器（向量 + 图谱 RRF 融合）
"""

from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from astrbot.api import logger

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from .graph_rag_engine import MemoryGraphStore, GraphRAGConfig
    from .hybrid_rag import QueryResult


@dataclass
class KGQueryResult:
    """知识图谱查询结果"""
    entities: List[Dict[str, Any]]
    triplets: List[Dict[str, Any]]
    scores: List[float]

    def __len__(self) -> int:
        return len(self.entities)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.entities[index]


class KnowledgeGraphRetriever:
    """
    知识图谱检索器

    从知识图谱中检索相关实体和关系，支持：
    1. 基于实体名称的搜索
    2. 多跳关系扩展
    3. 子图提取
    """

    def __init__(
        self,
        graph_store: "MemoryGraphStore",
        embed_provider: Optional[Any] = None,
        top_k: int = 5
    ):
        """
        初始化知识图谱检索器

        Args:
            graph_store: 图谱存储实例
            embed_provider: 嵌入提供者（用于向量相似度搜索，可选）
            top_k: 返回结果数量
        """
        self.graph_store = graph_store
        self.embed_provider = embed_provider
        self.top_k = top_k

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        depth: int = 1
    ) -> KGQueryResult:
        """
        从知识图谱检索相关实体和关系

        Args:
            query: 查询文本
            top_k: 返回的实体数量
            depth: 关系扩展深度

        Returns:
            KGQueryResult，包含检索到的实体、三元组和分数
        """
        top_k = top_k or self.top_k

        # 1. 搜索相关实体
        related_entities = self.graph_store.search_related_entities(
            query,
            top_k=top_k
        )

        if not related_entities:
            return KGQueryResult(entities=[], triplets=[], scores=[])

        # 2. 获取每个实体的多跳三元组
        all_triplets = []
        entities = []
        scores = []

        for item in related_entities:
            entity_id = item["entity_id"]
            entity = item["entity"]
            score = item["score"]

            entities.append({
                "entity_id": entity_id,
                "name": entity.get("name", ""),
                "type": entity.get("type", ""),
                "description": entity.get("description", ""),
                "chunk_ids": entity.get("chunk_ids", [])
            })
            scores.append(score)

            # 获取多跳三元组
            triplets = self.graph_store.get_triplets(entity_id, depth=depth)
            all_triplets.extend(triplets)

        # 3. 去重
        unique_triplets = self._deduplicate_triplets(all_triplets)

        logger.debug(
            f"✅ KG检索完成: query='{query}', "
            f"实体={len(entities)}, 三元组={len(unique_triplets)}"
        )

        return KGQueryResult(
            entities=entities,
            triplets=unique_triplets,
            scores=scores
        )

    def _deduplicate_triplets(
        self,
        triplets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """去重三元组"""
        seen = set()
        unique = []

        for t in triplets:
            key = (
                t.get("head", ""),
                t.get("relation", ""),
                t.get("tail", "")
            )
            if key not in seen:
                seen.add(key)
                unique.append(t)

        return unique

    async def get_subgraph(
        self,
        entity_ids: List[str],
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        获取子图（以指定实体为中心的局部图谱）

        Args:
            entity_ids: 中心实体ID列表
            depth: 扩展深度

        Returns:
            子图数据，包含实体和关系
        """
        all_entities = {}
        all_relations = {}

        for entity_id in entity_ids:
            # 获取实体
            entity = self.graph_store.get_entity(entity_id)
            if entity:
                all_entities[entity_id] = entity

            # 获取相邻实体和关系
            for neighbor_id, relation, rel_id in self.graph_store.get_neighbors(
                entity_id, direction="both"
            ):
                # 添加关系
                if rel_id not in all_relations:
                    rel_data = self.graph_store.relations.get(rel_id, {})
                    all_relations[rel_id] = rel_data

                # 添加相邻实体
                neighbor = self.graph_store.get_entity(neighbor_id)
                if neighbor and neighbor_id not in all_entities:
                    all_entities[neighbor_id] = neighbor

        return {
            "entities": all_entities,
            "relations": all_relations
        }


class FusionRetriever:
    """
    融合检索器：向量检索 + 知识图谱检索

    使用 RRF (Reciprocal Rank Fusion) 分数融合两路检索结果：
    score = alpha * (1 / (k + rank_vector)) + (1 - alpha) * (1 / (k + rank_kg))

    其中 k 是 RRF 平滑常数（通常为 60），alpha 控制两路的权重
    """

    def __init__(
        self,
        vector_retriever: Any,  # HybridRetriever or VectorRetriever
        kg_retriever: KnowledgeGraphRetriever,
        alpha: float = 0.5,
        rrf_k: int = 60
    ):
        """
        初始化融合检索器

        Args:
            vector_retriever: 向量检索器
            kg_retriever: 知识图谱检索器
            alpha: 混合权重（0=纯图谱，1=纯向量）
            rrf_k: RRF 平滑常数
        """
        self.vector_retriever = vector_retriever
        self.kg_retriever = kg_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> "QueryResult":
        """
        混合检索：向量 + 图谱 RRF 融合

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            QueryResult - 融合后的检索结果
        """
        # 1. 并行执行两路检索
        vector_task = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        kg_task = self.kg_retriever.retrieve(query, top_k=top_k)

        vector_result, kg_result = await self._safe_gather(
            vector_task, kg_task
        )

        # 2. RRF 融合
        fused_results = self._rrf_fusion(
            vector_results=vector_result,
            kg_results=kg_result,
            top_k=top_k
        )

        # 3. 转换为 QueryResult 格式
        from .hybrid_rag import QueryResult, Node

        nodes = []
        scores = []

        for item in fused_results:
            node = Node(
                text=item.get("text", ""),
                metadata=item.get("metadata", {})
            )
            nodes.append(node)
            scores.append(item.get("fused_score", 0.0))

        logger.debug(
            f"✅ 融合检索完成: query='{query}', "
            f"向量={len(vector_result.nodes) if vector_result else 0}, "
            f"图谱={len(kg_result.entities) if kg_result else 0}, "
            f"融合={len(nodes)}"
        )

        return QueryResult(nodes=nodes, scores=scores)

    def _rrf_fusion(
        self,
        vector_results: Any,  # QueryResult
        kg_results: KGQueryResult,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        RRF 分数融合

        Args:
            vector_results: 向量检索结果 (QueryResult)
            kg_results: 图谱检索结果 (KGQueryResult)
            top_k: 返回数量

        Returns:
            融合后的结果列表
        """
        # 构建向量检索的分数映射 (chunk_id -> score)
        vector_scores: Dict[str, float] = {}
        if vector_results and hasattr(vector_results, 'nodes'):
            for i, node in enumerate(vector_results.nodes):
                chunk_id = node.metadata.get("chunk_id", f"vec_{i}")
                score = vector_results.scores[i] if i < len(vector_results.scores) else 0.0
                vector_scores[chunk_id] = 1.0 / (self.rrf_k + i + 1)  # RRF score

        # 构建图谱检索的分数映射 (entity_id -> score)
        kg_scores: Dict[str, float] = {}
        kg_texts: Dict[str, str] = {}
        kg_metadata: Dict[str, Dict] = {}

        if kg_results and kg_results.entities:
            for i, entity in enumerate(kg_results.entities):
                entity_id = entity.get("entity_id", f"kg_{i}")
                # 基于图谱分数和 RRF 计算
                base_score = kg_results.scores[i] if i < len(kg_results.scores) else 0.0
                kg_scores[entity_id] = base_score * (1.0 / (self.rrf_k + i + 1))

                # 保存文本和元数据
                kg_texts[entity_id] = self._entity_to_text(entity)
                kg_metadata[entity_id] = {
                    "entity_id": entity_id,
                    "entity_name": entity.get("name", ""),
                    "entity_type": entity.get("type", ""),
                    "source": "knowledge_graph"
                }

                # 添加关联的 chunk_ids
                chunk_ids = entity.get("chunk_ids", [])
                if chunk_ids:
                    kg_metadata[entity_id]["chunk_ids"] = chunk_ids

        # 计算融合分数
        # 融合策略：alpha * vector_score + (1 - alpha) * kg_score
        all_chunk_ids = set(list(vector_scores.keys()) + list(kg_scores.keys()))

        fused_scores = []
        for chunk_id in all_chunk_ids:
            vec_s = vector_scores.get(chunk_id, 0.0) * self.alpha
            kg_s = 0.0

            # 如果 chunk_id 对应图谱实体，计算图谱分数
            for entity_id, metadata in kg_metadata.items():
                if chunk_id in metadata.get("chunk_ids", []):
                    kg_s = kg_scores.get(entity_id, 0.0) * (1 - self.alpha)
                    break

            # 如果没有匹配到图谱实体但有图谱分数
            if chunk_id.startswith("kg_"):
                entity_id = chunk_id
                kg_s = kg_scores.get(entity_id, 0.0) * (1 - self.alpha)

            fused_score = vec_s + kg_s

            # 构建结果项
            if chunk_id.startswith("vec_"):
                # 向量结果
                idx = int(chunk_id.replace("vec_", ""))
                if vector_results and idx < len(vector_results.nodes):
                    node = vector_results.nodes[idx]
                    fused_scores.append({
                        "text": node.text,
                        "metadata": node.metadata,
                        "score": fused_scores[-1]["score"] if fused_scores and "score" in fused_scores[-1] else fused_score,
                        "fused_score": fused_score,
                        "source": "vector"
                    })
            elif chunk_id.startswith("kg_"):
                # 图谱结果
                entity_id = chunk_id
                fused_scores.append({
                    "text": kg_texts.get(entity_id, ""),
                    "metadata": kg_metadata.get(entity_id, {}),
                    "fused_score": fused_score,
                    "source": "knowledge_graph"
                })

        # 按融合分数排序
        fused_scores.sort(key=lambda x: x["fused_score"], reverse=True)

        return fused_scores[:top_k]

    def _entity_to_text(self, entity: Dict[str, Any]) -> str:
        """将实体转换为文本"""
        parts = []

        name = entity.get("name", "")
        if name:
            parts.append(f"实体: {name}")

        entity_type = entity.get("type", "")
        if entity_type:
            parts.append(f"类型: {entity_type}")

        description = entity.get("description", "")
        if description:
            parts.append(f"描述: {description}")

        return " | ".join(parts) if parts else name

    async def _safe_gather(
        self,
        *tasks
    ) -> List[Any]:
        """安全地并行执行任务，捕获异常"""
        results = []
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.warning(f"检索任务失败: {e}")
                results.append(None)
        return results


def create_fusion_retriever(
    vector_retriever: Any,
    kg_retriever: KnowledgeGraphRetriever,
    alpha: float = 0.5
) -> FusionRetriever:
    """
    创建融合检索器

    Args:
        vector_retriever: 向量检索器
        kg_retriever: 知识图谱检索器
        alpha: 混合权重

    Returns:
        FusionRetriever 实例
    """
    return FusionRetriever(
        vector_retriever=vector_retriever,
        kg_retriever=kg_retriever,
        alpha=alpha
    )
