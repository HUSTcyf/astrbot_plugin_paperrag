"""
Graph RAG Engine - 图谱增强检索引擎

支持三种检索模式：
1. vector - 纯向量检索
2. graph - 纯图谱检索
3. hybrid - 向量 + 图谱混合检索

存储后端支持：
- memory: LlamaIndex SimplePropertyGraphStore（适合小规模/测试，支持持久化）
- neo4j: LlamaIndex Neo4jGraphStore（适合生产环境）

重构说明：
- 删除自定义 MemoryGraphStore、Neo4jGraphStore 封装
- 直接使用 LlamaIndex PropertyGraphStore
- 创建 SimplePropertyGraphStoreAdapter 简化接口
- MultimodalGraphBuilder 的接口保持不变
"""

import asyncio
import gc
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import logger

if TYPE_CHECKING:
    from .rag_engine import RAGConfig

# Graph RAG 模块路径
_PLUGIN_DIR = Path(__file__).parent.resolve()
_DEFAULT_GRAPH_DIR = _PLUGIN_DIR / "data" / "graph_store"


@dataclass
class GraphRAGConfig:
    """Graph RAG 配置类"""
    enable_graph_rag: bool = False
    storage_type: str = "memory"  # "memory" 或 "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    max_triplets_per_chunk: int = 5
    graph_retrieval_top_k: int = 5
    hybrid_alpha: float = 0.5  # RRF 融合权重（0=纯图，1=纯向量）
    auto_build: bool = False  # 是否自动构建图谱
    auto_build_threshold: int = 10  # 自动构建阈值
    # 多模态配置
    multimodal_enabled: bool = True  # 是否启用多模态图谱抽取
    max_images_per_chunk: int = 1  # 每个chunk最多处理图片数
    extract_image_entities: bool = True  # 是否提取图片为实体

    @classmethod
    def from_rag_config(cls, config: "RAGConfig") -> "GraphRAGConfig":
        """从 RAGConfig 创建 GraphRAGConfig"""
        return cls(
            enable_graph_rag=getattr(config, 'enable_graph_rag', False),
            storage_type=getattr(config, 'graph_storage_type', 'memory'),
            neo4j_uri=getattr(config, 'graph_neo4j_uri', 'bolt://localhost:7687'),
            neo4j_user=getattr(config, 'graph_neo4j_user', 'neo4j'),
            neo4j_password=getattr(config, 'graph_neo4j_password', ''),
            max_triplets_per_chunk=getattr(config, 'graph_max_triplets_per_chunk', 5),
            graph_retrieval_top_k=getattr(config, 'graph_retrieval_top_k', 5),
            hybrid_alpha=getattr(config, 'graph_hybrid_alpha', 0.5),
            auto_build=getattr(config, 'graph_auto_build', False),
            auto_build_threshold=getattr(config, 'graph_auto_build_threshold', 10),
            multimodal_enabled=getattr(config, 'graph_multimodal_enabled', True),
            max_images_per_chunk=getattr(config, 'graph_max_images_per_chunk', 1),
            extract_image_entities=getattr(config, 'graph_extract_image_entities', True),
        )


class SimplePropertyGraphStoreAdapter:
    """
    简化的适配器 - 直接使用实体名作为标识符

    核心思路：
    - LlamaIndex PropertyGraphStore 内部使用 entity name 作为键
    - 我们直接用 entity name，不转换内部数字 ID
    - entity_id 只用于外部追踪统计，不参与业务逻辑
    """

    def __init__(self, graph_store: Any):
        """
        初始化适配器

        Args:
            graph_store: LlamaIndex PropertyGraphStore 实例
        """
        self._store = graph_store
        self._entity_info: Dict[str, Dict[str, Any]] = {}  # name.lower() -> {name, type, description}
        self._relation_count = 0  # 用于生成关系 ID

    def add_entity(
        self,
        name: str,
        entity_type: str = "UNKNOWN",
        description: str = "",
        chunk_id: str = ""
    ) -> str:
        """
        添加实体到图谱（如果实体已存在则不覆盖）

        Returns:
            实体名称（作为标识符）
        """
        try:
            from llama_index.core.graph_stores.types import EntityNode
            # 如果实体已存在，不覆盖原有信息
            if name.lower() not in self._entity_info:
                node = EntityNode(name=name, label=entity_type, properties={"description": description})
                self._store.upsert_nodes([node])
                self._entity_info[name.lower()] = {
                    "name": name,
                    "type": entity_type,
                    "description": description
                }
            return name
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加实体失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return name

    def add_relation(
        self,
        head: str,
        tail: str,
        relation: str,
        weight: float = 1.0,
        chunk_id: str = ""
    ) -> Optional[str]:
        """
        添加关系到图谱

        Returns:
            关系 ID（格式: head##relation##tail）
        """
        try:
            from llama_index.core.graph_stores.types import Relation
            rel = Relation(label=relation, source_id=head, target_id=tail, properties={})
            self._store.upsert_relations([rel])

            # 确保头尾实体都在缓存中
            if head.lower() not in self._entity_info:
                self._entity_info[head.lower()] = {"name": head, "type": "UNKNOWN", "description": ""}
            if tail.lower() not in self._entity_info:
                self._entity_info[tail.lower()] = {"name": tail, "type": "UNKNOWN", "description": ""}

            self._relation_count += 1
            return f"{head}##{relation}##{tail}"
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加关系失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return None

    def add_image_entity(
        self,
        figure_id: str,
        image_path: str,
        description: str = "",
        figure_type: str = "unknown",
        chunk_id: str = ""
    ) -> str:
        """添加图片实体"""
        try:
            from llama_index.core.graph_stores.types import EntityNode, Relation
            nodes = [
                EntityNode(name=figure_id, label=f"Figure:{figure_type}", properties={"description": description}),
                EntityNode(name=image_path, label="ImagePath", properties={}),
            ]
            rels = [
                Relation(label="__is_a__", source_id=figure_id, target_id=f"Figure:{figure_type}", properties={}),
                Relation(label="__has_path__", source_id=figure_id, target_id=image_path, properties={}),
            ]
            if description:
                rels.append(Relation(label="__has_description__", source_id=figure_id, target_id=description[:200], properties={}))
            self._store.upsert_nodes(nodes)
            self._store.upsert_relations(rels)
            self._entity_info[figure_id.lower()] = {
                "name": figure_id,
                "type": f"Figure:{figure_type}",
                "description": description
            }
            return figure_id
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加图片实体失败: {e}")
            return figure_id

    def add_table_entity(
        self,
        table_id: str,
        description: str = "",
        chunk_id: str = ""
    ) -> str:
        """添加表格实体"""
        try:
            from llama_index.core.graph_stores.types import EntityNode, Relation
            nodes = [EntityNode(name=table_id, label="Table", properties={"description": description})]
            rels = [Relation(label="__is_a__", source_id=table_id, target_id="Table", properties={})]
            if description:
                rels.append(Relation(label="__has_description__", source_id=table_id, target_id=description[:200], properties={}))
            self._store.upsert_nodes(nodes)
            self._store.upsert_relations(rels)
            self._entity_info[table_id.lower()] = {
                "name": table_id,
                "type": "Table",
                "description": description
            }
            return table_id
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加表格实体失败: {e}")
            return table_id

    def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """获取实体"""
        info = self._entity_info.get(entity_name.lower())
        if info:
            return {
                "id": info["name"],
                "name": info["name"],
                "type": info.get("type", "UNKNOWN"),
                "description": info.get("description", "")
            }
        return None

    def get_neighbors(
        self,
        entity_name: str,
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List:
        """获取实体的邻居节点"""
        try:
            from llama_index.core.graph_stores.types import EntityNode
            # get_rel_map returns List[Tuple[EntityNode, Relation, EntityNode]]
            entity_node = EntityNode(name=entity_name, label="")
            rel_map = self._store.get_rel_map([entity_node], depth=1, limit=50)
            neighbors = []
            for src, rel, tgt in rel_map:
                src_name = src.name if hasattr(src, 'name') else str(src)
                tgt_name = tgt.name if hasattr(tgt, 'name') else str(tgt)
                rel_label = rel.label if hasattr(rel, 'label') else str(rel)
                if relation_type and rel_label.upper() != relation_type.upper():
                    continue
                neighbors.append((tgt_name, rel_label, f"{entity_name}##{rel_label}##{tgt_name}"))
            return neighbors
        except Exception as e:
            logger.warning(f"[GraphRAG] 获取邻居失败: {e}")
            return []

    def get_triplets(self, entity_name: str, depth: int = 1) -> List[Dict[str, Any]]:
        """获取以某实体为中心的多跳三元组"""
        try:
            from llama_index.core.graph_stores.types import EntityNode
            entity_node = EntityNode(name=entity_name, label="")
            rel_map = self._store.get_rel_map([entity_node], depth=depth, limit=100)
            triplets = []

            # visited 使用 lowercase 用于去重比较
            visited: set = set()

            def traverse(current_key: str, current_depth: int):
                if current_depth > depth or current_key in visited:
                    return
                visited.add(current_key)

                for src, rel, tgt in rel_map:
                    src_n = src.name if hasattr(src, 'name') else str(src)
                    tgt_n = tgt.name if hasattr(tgt, 'name') else str(tgt)
                    rel_l = rel.label if hasattr(rel, 'label') else str(rel)
                    # 使用 lowercase 比较，但保留原始名称
                    if src_n.lower() == current_key.lower():
                        triplets.append({
                            "head": src_n,
                            "relation": rel_l,
                            "tail": tgt_n,
                            "head_id": src_n,
                            "tail_id": tgt_n
                        })
                        traverse(tgt_n.lower(), current_depth + 1)

            traverse(entity_name.lower(), 0)
            return triplets
        except Exception as e:
            logger.warning(f"[GraphRAG] 获取多跳三元组失败: {e}")
            return []

    def search_related_entities(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索相关实体（基于关键词匹配）"""
        results = []
        query_lower = query.lower()

        for name_lower, info in self._entity_info.items():
            score = 0.0
            if query_lower in name_lower:
                score = 1.0
            elif any(word in name_lower for word in query_lower.split()):
                score = 0.5

            if score > 0:
                results.append({
                    "entity": {
                        "name": info["name"],
                        "type": info.get("type", "UNKNOWN"),
                        "description": info.get("description", "")
                    },
                    "entity_id": info["name"],
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        # 使用缓存的实体信息计算，避免 get_rel_map([]) 的问题
        entity_types: Dict[str, int] = {}
        for info in self._entity_info.values():
            t = info.get("type", "UNKNOWN")
            entity_types[t] = entity_types.get(t, 0) + 1

        return {
            "entity_count": len(self._entity_info),
            "relation_count": self._relation_count,
            "index_size": len(self._entity_info),
            "entity_types": entity_types
        }

    def __len__(self) -> int:
        """返回实体数量"""
        return len(self._entity_info)

    def __contains__(self, item: str) -> bool:
        """检查实体是否存在（大小写不敏感）"""
        return item.lower() in self._entity_info

    def clear(self, delete_storage: bool = False):
        """清空图谱（仅清空缓存）"""
        self._entity_info.clear()
        self._relation_count = 0
        logger.info("[GraphRAG] 图谱缓存已清空")


class PersistentPropertyGraphStoreAdapter(SimplePropertyGraphStoreAdapter):
    """
    带持久化的适配器

    封装 LlamaIndex SimplePropertyGraphStore，提供磁盘持久化。
    """

    STORAGE_FILENAME = "knowledge_graph.json.gz"
    METADATA_FILENAME = "graph_metadata.json"

    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        save_interval: int = 100
    ):
        from llama_index.core.graph_stores import SimplePropertyGraphStore

        self.storage_path = Path(storage_path) if storage_path else _DEFAULT_GRAPH_DIR
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._dirty = False
        self._operation_count = 0

        # 创建底层存储
        store = SimplePropertyGraphStore()
        super().__init__(store)

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 尝试加载已有数据
        self._try_load()

    def _get_storage_file(self) -> Path:
        return self.storage_path / self.STORAGE_FILENAME

    def _get_metadata_file(self) -> Path:
        return self.storage_path / self.METADATA_FILENAME

    def _try_load(self) -> bool:
        """加载已保存的图谱"""
        storage_file = self._get_storage_file()
        metadata_file = self._get_metadata_file()

        if not storage_file.exists() or not metadata_file.exists():
            logger.info("[GraphRAG] 未发现已保存的知识图谱，将创建新图谱")
            return False

        try:
            import gzip
            import json
            from datetime import datetime
            from llama_index.core.graph_stores.types import EntityNode, Relation

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            with gzip.open(storage_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复三元组到 LlamaIndex store
            triplets = data.get("triplets", [])
            for triplet in triplets:
                head = triplet.get("head", "")
                relation = triplet.get("relation", "")
                tail = triplet.get("tail", "")
                if head and relation and tail:
                    try:
                        self._store.upsert_relations([Relation(label=relation, source_id=head, target_id=tail, properties={})])
                    except Exception:
                        pass

            # 恢复实体信息
            entities = data.get("entities", {})
            for ent in entities.values():
                name = ent.get("name", "")
                if name:
                    self._entity_info[name.lower()] = {
                        "name": name,
                        "type": ent.get("type", "UNKNOWN"),
                        "description": ent.get("description", "")
                    }

            self._relation_count = data.get("relation_count", 0)

            logger.info(
                f"✅ 知识图谱加载成功: "
                f"实体={len(self._entity_info)}, "
                f"关系={self._relation_count}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ 加载知识图谱失败: {e}，将创建新图谱")
            return False

    def save(self, force: bool = False) -> bool:
        """保存图谱到磁盘"""
        if not force and not self._dirty:
            logger.debug("图谱无变更，跳过保存")
            return True

        try:
            import gzip
            import json
            from datetime import datetime
            from llama_index.core.graph_stores.types import EntityNode

            storage_file = self._get_storage_file()
            metadata_file = self._get_metadata_file()

            # 从 LlamaIndex store 提取三元组（批量查询所有实体）
            triplets = []
            all_entities = list(self._entity_info.keys())

            # 批量处理，每批 10 个实体
            batch_size = 10
            for i in range(0, len(all_entities), batch_size):
                batch_keys = all_entities[i:i + batch_size]
                # 创建 EntityNode 列表
                batch_nodes = [EntityNode(name=name, label="") for name in batch_keys]
                rel_map = self._store.get_rel_map(batch_nodes, depth=1, limit=1000)
                # rel_map 是 List[Tuple[EntityNode, Relation, EntityNode]]
                for src, rel, tgt in rel_map:
                    src_name = src.name if hasattr(src, 'name') else str(src)
                    tgt_name = tgt.name if hasattr(tgt, 'name') else str(tgt)
                    rel_label = rel.label if hasattr(rel, 'label') else str(rel)
                    triplets.append({
                        "head": src_name,
                        "relation": rel_label,
                        "tail": tgt_name
                    })

            save_data = {
                "entities": {name: info for name, info in self._entity_info.items()},
                "triplets": triplets,
                "relation_count": self._relation_count,
                "version": "2.0"
            }

            metadata = {
                "saved_at": datetime.now().isoformat(),
                "entity_count": len(self._entity_info),
                "relation_count": self._relation_count
            }

            # 原子性保存
            temp_storage = storage_file.with_suffix('.tmp.gz')
            temp_metadata = metadata_file.with_suffix('.tmp')

            with gzip.open(temp_storage, 'wt', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            with open(temp_metadata, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            temp_storage.replace(storage_file)
            temp_metadata.replace(metadata_file)

            self._dirty = False
            self._operation_count = 0

            logger.info(f"💾 知识图谱已保存: 实体={metadata['entity_count']}, 关系={metadata['relation_count']}")
            return True

        except Exception as e:
            logger.error(f"❌ 保存知识图谱失败: {e}")
            return False

    def _mark_dirty(self):
        """标记为有未保存的变更"""
        self._dirty = True
        self._operation_count += 1
        if self.auto_save and self._operation_count >= self.save_interval:
            self.save()

    def add_entity(self, name: str, entity_type: str = "UNKNOWN", description: str = "", chunk_id: str = "") -> str:
        self._mark_dirty()
        return super().add_entity(name, entity_type, description, chunk_id)

    def add_relation(self, head: str, tail: str, relation: str, weight: float = 1.0, chunk_id: str = "") -> Optional[str]:
        self._mark_dirty()
        return super().add_relation(head, tail, relation, weight, chunk_id)

    def add_image_entity(self, figure_id: str, image_path: str, description: str = "", figure_type: str = "unknown", chunk_id: str = "") -> str:
        self._mark_dirty()
        return super().add_image_entity(figure_id, image_path, description, figure_type, chunk_id)

    def add_table_entity(self, table_id: str, description: str = "", chunk_id: str = "") -> str:
        self._mark_dirty()
        return super().add_table_entity(table_id, description, chunk_id)

    def clear(self, delete_storage: bool = False):
        super().clear()
        if delete_storage:
            try:
                storage_file = self._get_storage_file()
                metadata_file = self._get_metadata_file()
                if storage_file.exists():
                    storage_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
            except Exception as e:
                logger.error(f"删除图谱存储文件失败: {e}")


# ========== 向后兼容别名 ==========
MemoryGraphStore = PersistentPropertyGraphStoreAdapter


class GraphRAGEngine:
    """
    Graph RAG 引擎 - 扩展现有 HybridRAGEngine

    支持三种检索模式：
    - vector: 纯向量检索（委托给 base_engine）
    - graph: 纯图谱检索
    - hybrid: 向量 + 图谱混合检索
    """

    RETRIEVAL_MODES = ["vector", "graph", "hybrid"]

    def __init__(
        self,
        config: GraphRAGConfig,
        base_engine: Any,
        context: Any = None
    ):
        self.config = config
        self.base_engine = base_engine
        self.context = context
        self._graph_store: Optional[Any] = None
        self._index: Optional[Any] = None
        self._query_engine: Optional[Any] = None
        self._adapter: Optional[SimplePropertyGraphStoreAdapter] = None
        self._initialized = False

    def _get_llm(self):
        """获取 LLM 实例"""
        if self.base_engine is None:
            return None
        return getattr(self.base_engine, 'llm', None)

    async def initialize(self):
        """初始化图谱引擎"""
        if self._initialized:
            return

        if not self.config.enable_graph_rag:
            logger.info("Graph RAG 功能未启用")
            return

        try:
            if self.config.storage_type == "neo4j":
                # Neo4j 存储
                from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
                self._graph_store = Neo4jPropertyGraphStore(
                    url=self.config.neo4j_uri,
                    username=self.config.neo4j_user,
                    password=self.config.neo4j_password,
                    database="neo4j",
                    refresh_schema=True
                )
                logger.info(f"✅ Neo4j 图谱存储已连接: {self.config.neo4j_uri}")
                self._adapter = SimplePropertyGraphStoreAdapter(self._graph_store)
            else:
                self._adapter = PersistentPropertyGraphStoreAdapter()
                self._graph_store = self._adapter._store
                logger.info("✅ SimplePropertyGraphStore 已初始化")

            await self._init_index()

            logger.info(f"✅ Graph RAG 引擎已初始化 (存储类型: {self.config.storage_type})")
            logger.info(f"   - 最大三元组/Chunk: {self.config.max_triplets_per_chunk}")
            logger.info(f"   - 图谱检索TopK: {self.config.graph_retrieval_top_k}")
            logger.info(f"   - 混合检索Alpha: {self.config.hybrid_alpha}")

            self._initialized = True

        except ImportError as e:
            logger.error(f"❌ 缺少依赖: {e}")
            logger.info("请安装 llama-index: pip install llama-index")
        except Exception as e:
            logger.error(f"❌ Graph RAG 引擎初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _init_index(self):
        """初始化 LlamaIndex 索引"""
        try:
            from llama_index.core import PropertyGraphIndex
            from llama_index.core.retrievers import PGRetriever
            from llama_index.core.query_engine import RetrieverQueryEngine

            llm = self._get_llm()
            if llm is None:
                logger.warning("[GraphRAG] 未找到 LLM，图谱检索将使用默认配置")
                return

            self._index = PropertyGraphIndex.from_existing(
                graph_store=self._graph_store,
                llm=llm,
            )

            # 验证索引是否有效
            if self._index is None:
                logger.warning("[GraphRAG] 索引创建返回 None，检索功能可能受限")
            else:
                retriever = PGRetriever(self._index)
                self._query_engine = RetrieverQueryEngine.from_args(
                    self._index,
                    retriever=retriever,
                    llm=llm,
                )
                logger.info("✅ LlamaIndex PropertyGraphIndex 已创建")

        except ImportError as e:
            logger.warning(f"[GraphRAG] LlamaIndex 索引组件不可用（检索功能使用降级模式）: {e}")
        except Exception as e:
            logger.warning(f"[GraphRAG] 索引初始化失败（不影响图谱构建）: {e}")

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """搜索接口，支持三种模式"""
        if not self.config.enable_graph_rag:
            return {"type": "error", "message": "Graph RAG 功能未启用"}

        if mode not in self.RETRIEVAL_MODES:
            return {"type": "error", "message": f"不支持的检索模式: {mode}"}

        try:
            if mode == "vector":
                return await self._vector_search(query, top_k)
            elif mode == "graph":
                return await self._graph_search(query, top_k)
            else:
                return await self._hybrid_search(query, top_k)
        except Exception as e:
            logger.error(f"Graph RAG 搜索失败: {e}")
            return {"type": "error", "message": f"Graph RAG 搜索失败: {str(e)}"}

    async def _vector_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """纯向量检索"""
        if self.base_engine is None:
            return {"type": "error", "message": "基础引擎未初始化"}
        result = await self.base_engine.search(query, mode="retrieve")
        return result

    async def _graph_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """纯图谱检索"""
        if self._adapter is None:
            return {"type": "error", "message": "图谱适配器未初始化"}

        related_entities = self._adapter.search_related_entities(
            query,
            top_k=self.config.graph_retrieval_top_k
        )

        if not related_entities:
            return {
                "type": "graph",
                "answer": "",
                "sources": [],
                "entities": [],
                "triplets": [],
                "message": "未找到相关实体"
            }

        all_triplets = []
        all_entities = []

        for item in related_entities:
            entity_name = item["entity_id"]
            entity = item["entity"]

            all_entities.append({
                "entity_id": entity_name,
                "name": entity.get("name", ""),
                "type": entity.get("type", ""),
                "description": entity.get("description", ""),
                "score": item["score"]
            })

            triplets = self._adapter.get_triplets(entity_name, depth=1)
            all_triplets.extend(triplets)

        unique_triplets = []
        seen = set()
        for t in all_triplets:
            key = (t.get("head", ""), t.get("relation", ""), t.get("tail", ""))
            if key not in seen:
                seen.add(key)
                unique_triplets.append(t)

        if self._query_engine is not None:
            try:
                response = self._query_engine.query(query)
                graph_context = str(response)
            except Exception as e:
                logger.warning(f"[GraphRAG] LlamaIndex 查询失败: {e}")
                graph_context = self._build_graph_context(unique_triplets, all_entities)
        else:
            graph_context = self._build_graph_context(unique_triplets, all_entities)

        return {
            "type": "graph",
            "answer": graph_context,
            "sources": [],
            "entities": all_entities,
            "triplets": unique_triplets
        }

    async def _hybrid_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """混合检索：向量 + 图谱"""
        vector_task = self._vector_search(query, top_k)
        graph_task = self._graph_search(query, top_k)

        vector_result, graph_result = await asyncio.gather(
            vector_task,
            graph_task,
            return_exceptions=True
        )

        if isinstance(vector_result, Exception):
            logger.warning(f"向量检索失败，使用纯图谱检索: {vector_result}")
            return graph_result

        if isinstance(graph_result, Exception):
            logger.warning(f"图谱检索失败，使用纯向量检索: {graph_result}")
            return vector_result

        vr: Dict[str, Any] = vector_result
        gr: Dict[str, Any] = graph_result

        vector_sources = vr.get("sources", [])
        graph_entities = gr.get("entities", [])
        graph_triplets = gr.get("triplets", [])

        hybrid_answer = self._build_hybrid_context(
            vector_sources,
            graph_entities,
            graph_triplets,
            query
        )

        return {
            "type": "hybrid",
            "answer": hybrid_answer,
            "sources": vector_sources,
            "entities": graph_entities,
            "triplets": graph_triplets,
            "vector_results_count": len(vector_sources),
            "graph_entities_count": len(graph_entities)
        }

    def _build_graph_context(
        self,
        triplets: List[Dict[str, Any]],
        entities: List[Dict[str, Any]]
    ) -> str:
        """构建图谱上下文文本"""
        if not triplets:
            return "未在知识图谱中找到相关信息。"

        context = "【知识图谱检索结果】\n\n"

        if entities:
            context += "相关实体：\n"
            for e in entities[:5]:
                context += f"- {e['name']} ({e['type']})\n"
            context += "\n"

        context += "实体关系：\n"
        for t in triplets[:10]:
            context += f"- {t.get('head', '')} --[{t.get('relation', '')}]--> {t.get('tail', '')}\n"

        return context

    def _build_hybrid_context(
        self,
        vector_sources: List[Dict[str, Any]],
        graph_entities: List[Dict[str, Any]],
        graph_triplets: List[Dict[str, Any]],
        query: str
    ) -> str:
        """构建混合检索上下文"""
        parts = []

        if graph_entities or graph_triplets:
            graph_part = "【知识图谱】\n"
            if graph_entities:
                graph_part += "相关实体："
                for e in graph_entities[:3]:
                    graph_part += f"{e['name']}、"
                graph_part = graph_part.rstrip("、") + "\n"
            if graph_triplets:
                graph_part += "关系："
                for t in graph_triplets[:3]:
                    graph_part += f"{t.get('head', '')}--{t.get('relation', '')}-->{t.get('tail', '')}、"
                graph_part = graph_part.rstrip("、") + "\n"
            parts.append(graph_part)

        if vector_sources:
            vector_part = "【文档检索】\n"
            for i, src in enumerate(vector_sources[:3], 1):
                metadata = src.get("metadata", {})
                filename = metadata.get("file_name", "unknown")
                text = src.get("text", "")[:150]
                vector_part += f"{i}. [{filename}] {text}...\n"
            parts.append(vector_part)

        return "\n".join(parts) if parts else "未找到相关信息。"

    async def build_graph_from_nodes(self, nodes: List[Any]) -> Dict[str, Any]:
        """从文档节点构建知识图谱"""
        if not self.config.enable_graph_rag:
            return {"status": "skipped", "message": "Graph RAG 功能未启用"}

        try:
            try:
                from .graph_builder import MultimodalGraphBuilder
            except ImportError:
                from graph_builder import MultimodalGraphBuilder

            if self._adapter is None:
                await self.initialize()

            builder = MultimodalGraphBuilder(
                config=self.config,
                context=self.context
            )

            stats = await builder.build_from_nodes(nodes, self._adapter)

            # 处理完成后保存并清理内存
            if hasattr(self._adapter, 'save'):
                self._adapter.save(force=True)
            gc.collect()

            logger.info(f"✅ 知识图谱构建完成: {stats}")
            return {"status": "success", **stats}

        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    async def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if not self.config.enable_graph_rag:
            return {"enabled": False}

        if self._adapter is None:
            return {"enabled": True, "storage_type": self.config.storage_type, "entity_count": 0, "relation_count": 0}

        return {
            "enabled": True,
            "storage_type": self.config.storage_type,
            **self._adapter.get_stats()
        }

    async def clear_graph(self, delete_storage: bool = True) -> Dict[str, Any]:
        """清空图谱"""
        if not self.config.enable_graph_rag:
            return {"status": "skipped", "message": "Graph RAG 功能未启用"}

        if self._adapter is not None:
            # Neo4j 需要特殊处理：执行 Cypher 查询删除所有数据
            if self.config.storage_type == "neo4j" and self._graph_store is not None:
                try:
                    # 使用 client 执行 Cypher 删除所有节点和关系
                    with self._graph_store.client.session() as session:
                        session.run("MATCH (n) DETACH DELETE n")
                    logger.info("[GraphRAG] Neo4j 数据库已清空")
                except Exception as e:
                    logger.warning(f"[GraphRAG] 清空 Neo4j 数据库失败: {e}")

            self._adapter.clear(delete_storage=delete_storage)

            # 同时清理内存存储的持久化文件（如果存在）
            if delete_storage:
                storage_file = _DEFAULT_GRAPH_DIR / "knowledge_graph.json.gz"
                metadata_file = _DEFAULT_GRAPH_DIR / "graph_metadata.json"
                try:
                    if storage_file.exists():
                        storage_file.unlink()
                        logger.info(f"[GraphRAG] 已删除存储文件: {storage_file}")
                    if metadata_file.exists():
                        metadata_file.unlink()
                        logger.info(f"[GraphRAG] 已删除元数据文件: {metadata_file}")
                except Exception as e:
                    logger.warning(f"[GraphRAG] 删除存储文件失败: {e}")

        self._graph_store = None
        self._adapter = None
        self._index = None
        self._query_engine = None
        self._initialized = False

        return {"status": "success", "message": "图谱已清空"}


def create_graph_rag_engine(
    config: GraphRAGConfig,
    base_engine: Any,
    context: Any = None
) -> GraphRAGEngine:
    """创建 Graph RAG 引擎实例"""
    return GraphRAGEngine(config, base_engine, context)


async def build_graph_from_documents(
    documents: List[str],
    graph_store: Any,
    config: GraphRAGConfig,
    context: Any = None
) -> Dict[str, int]:
    """便捷函数：从文档列表构建图谱"""
    class SimpleNode:
        def __init__(self, text: str, metadata: Dict[str, Any]):
            self.text = text
            self.metadata = metadata

    nodes = [SimpleNode(doc, {"chunk_id": f"doc_{i}"}) for i, doc in enumerate(documents)]

    # 确保 graph_store 实现了所需接口
    if not hasattr(graph_store, 'add_entity') or not hasattr(graph_store, 'add_relation'):
        from llama_index.core.graph_stores import SimplePropertyGraphStore
        adapter = SimplePropertyGraphStoreAdapter(SimplePropertyGraphStore())
    else:
        adapter = graph_store

    builder = MultimodalGraphBuilder(config=config, context=context)
    return await builder.build_from_nodes(nodes, adapter)
