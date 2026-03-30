"""
Graph RAG Engine - 图谱增强检索引擎

支持三种检索模式：
1. vector - 纯向量检索
2. graph - 纯图谱检索
3. hybrid - 向量 + 图谱混合检索

存储后端支持：
- memory: Python 字典存储（适合小规模/测试，支持持久化）
- neo4j: Neo4j 图数据库（适合生产环境）
"""

import os
import json
import gzip
import hashlib
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from astrbot.api import logger

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from .rag_engine import RAGConfig

# Graph RAG 模块路径
_PLUGIN_DIR = Path(__file__).parent.resolve()
# 默认图谱存储目录
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


class MemoryGraphStore:
    """
    内存图谱存储（基于 Python 字典 + 磁盘持久化）

    数据结构：
    {
        "entities": {
            "entity_id": {
                "name": "实体名称",
                "type": "实体类型",
                "description": "描述",
                "chunk_ids": ["chunk_id1", "chunk_id2"]
            }
        },
        "relations": {
            "relation_id": {
                "head": "head_entity_id",
                "tail": "tail_entity_id",
                "relation": "关系类型",
                "weight": 1.0,
                "chunk_id": "source_chunk_id"
            }
        },
        "entity_index": {
            "entity_name": ["entity_id1", "entity_id2"]
        }
    }

    持久化功能：
    - 自动保存到磁盘（gzip 压缩 JSON）
    - 启动时自动加载
    - 支持增量更新
    - 保存前检查变更（避免不必要的写入）
    """

    # 存储文件名前缀
    STORAGE_FILENAME = "knowledge_graph.json.gz"
    METADATA_FILENAME = "graph_metadata.json"

    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        save_interval: int = 100  # 每 N 次操作自动保存
    ):
        """
        初始化内存图谱存储

        Args:
            storage_path: 存储路径（默认使用插件数据目录）
            auto_save: 是否自动保存
            save_interval: 自动保存的操作间隔
        """
        self.storage_path = Path(storage_path) if storage_path else _DEFAULT_GRAPH_DIR
        self.auto_save = auto_save
        self.save_interval = save_interval

        # 数据结构
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[str, Dict[str, Any]] = {}
        self.entity_index: Dict[str, List[str]] = {}  # name -> entity_ids
        self._next_entity_id = 1
        self._next_relation_id = 1

        # 状态追踪
        self._dirty = False  # 是否有过未保存的变更
        self._operation_count = 0  # 操作计数，用于触发自动保存
        self._last_save_time: Optional[datetime] = None

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 尝试加载已有数据
        self._try_load()

    def _get_storage_file(self) -> Path:
        """获取存储文件路径"""
        return self.storage_path / self.STORAGE_FILENAME

    def _get_metadata_file(self) -> Path:
        """获取元数据文件路径"""
        return self.storage_path / self.METADATA_FILENAME

    def _try_load(self) -> bool:
        """
        尝试加载已保存的图谱

        Returns:
            是否成功加载
        """
        storage_file = self._get_storage_file()
        metadata_file = self._get_metadata_file()

        if not storage_file.exists() or not metadata_file.exists():
            logger.info("📁 未发现已保存的知识图谱，将创建新图谱")
            return False

        try:
            # 加载元数据
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 加载图谱数据（gzip 压缩）
            with gzip.open(storage_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复数据
            self.entities = data.get("entities", {})
            self.relations = data.get("relations", {})
            self.entity_index = data.get("entity_index", {})
            self._next_entity_id = data.get("next_entity_id", 1)
            self._next_relation_id = data.get("next_relation_id", 1)

            # 验证数据一致性
            if not self._validate_data():
                logger.warning("⚠️ 图谱数据验证失败，将创建新图谱")
                self.clear()
                return False

            self._dirty = False
            self._last_save_time = datetime.fromisoformat(metadata.get("saved_at", datetime.now().isoformat()))

            logger.info(
                f"✅ 知识图谱加载成功: "
                f"实体={len(self.entities)}, "
                f"关系={len(self.relations)}, "
                f"保存时间={self._last_save_time}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ 加载知识图谱失败: {e}，将创建新图谱")
            self.clear()
            return False

    def _validate_data(self) -> bool:
        """
        验证数据一致性

        Returns:
            数据是否有效
        """
        if not isinstance(self.entities, dict):
            return False
        if not isinstance(self.relations, dict):
            return False
        if not isinstance(self.entity_index, dict):
            return False

        # 检查实体引用完整性
        for rel_id, rel_data in self.relations.items():
            if rel_data.get("head") not in self.entities:
                return False
            if rel_data.get("tail") not in self.entities:
                return False

        return True

    def save(self, force: bool = False) -> bool:
        """
        保存图谱到磁盘

        Args:
            force: 是否强制保存（忽略变更检查）

        Returns:
            是否保存成功
        """
        # 检查是否有变更
        if not force and not self._dirty:
            logger.debug("图谱无变更，跳过保存")
            return True

        try:
            storage_file = self._get_storage_file()
            metadata_file = self._get_metadata_file()

            # 构建保存数据
            save_data = {
                "entities": self.entities,
                "relations": self.relations,
                "entity_index": self.entity_index,
                "next_entity_id": self._next_entity_id,
                "next_relation_id": self._next_relation_id,
                "version": "1.0"
            }

            # 构建元数据
            metadata = {
                "saved_at": datetime.now().isoformat(),
                "entity_count": len(self.entities),
                "relation_count": len(self.relations),
                "checksum": self._compute_checksum(save_data)
            }

            # 原子性保存：先写临时文件，再重命名
            temp_storage = storage_file.with_suffix('.tmp.gz')
            temp_metadata = metadata_file.with_suffix('.tmp')

            # 保存数据（gzip 压缩）
            with gzip.open(temp_storage, 'wt', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            # 保存元数据
            with open(temp_metadata, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 原子替换
            temp_storage.replace(storage_file)
            temp_metadata.replace(metadata_file)

            self._dirty = False
            self._last_save_time = datetime.now()
            self._operation_count = 0

            logger.info(
                f"💾 知识图谱已保存: "
                f"实体={len(self.entities)}, "
                f"关系={len(self.relations)}, "
                f"路径={storage_file}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ 保存知识图谱失败: {e}")
            return False

    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        """
        计算数据的 MD5 校验和（用于变更检测）

        Args:
            data: 要计算校验和的数据

        Returns:
            MD5 校验和（16进制字符串）
        """
        # 按确定性顺序序列化
        def serialize(obj):
            if isinstance(obj, dict):
                return sorted((k, serialize(v)) for k, v in obj.items())
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            else:
                return str(obj)

        serialized = json.dumps(serialize(data), sort_keys=True)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()

    def _mark_dirty(self):
        """标记为有未保存的变更"""
        self._dirty = True
        self._operation_count += 1

        # 检查是否需要自动保存
        if self.auto_save and self._operation_count >= self.save_interval:
            self.save()

    @classmethod
    def load(cls, storage_path: str) -> "MemoryGraphStore":
        """
        从磁盘加载图谱（类方法）

        Args:
            storage_path: 存储路径

        Returns:
            加载后的 MemoryGraphStore 实例
        """
        instance = cls(storage_path=storage_path, auto_save=False)
        return instance

    def get_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息（包括持久化状态）"""
        base_stats = {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "index_size": len(self.entity_index),
            "storage_path": str(self.storage_path),
            "is_dirty": self._dirty,
            "last_save_time": self._last_save_time.isoformat() if self._last_save_time else None
        }
        return base_stats

    def clear(self, delete_storage: bool = False):
        """
        清空图谱

        Args:
            delete_storage: 是否同时删除存储文件
        """
        self.entities.clear()
        self.relations.clear()
        self.entity_index.clear()
        self._next_entity_id = 1
        self._next_relation_id = 1
        self._dirty = True

        # 如果指定删除存储文件
        if delete_storage:
            try:
                storage_file = self._get_storage_file()
                metadata_file = self._get_metadata_file()
                if storage_file.exists():
                    storage_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
                logger.info(f"🗑️ 图谱存储文件已删除: {self.storage_path}")
            except Exception as e:
                logger.error(f"删除图谱存储文件失败: {e}")

    def add_entity(
        self,
        name: str,
        entity_type: str = "UNKNOWN",
        description: str = "",
        chunk_id: str = ""
    ) -> str:
        """添加实体，返回实体ID"""
        # 检查是否已存在同名实体
        existing_ids = self.entity_index.get(name.lower(), [])
        if existing_ids:
            entity_id = existing_ids[0]
            # 更新实体的 chunk_ids
            if chunk_id and chunk_id not in self.entities[entity_id].get("chunk_ids", []):
                self.entities[entity_id].setdefault("chunk_ids", []).append(chunk_id)
                self._mark_dirty()  # 标记变更
            return entity_id

        # 创建新实体
        entity_id = f"entity_{self._next_entity_id}"
        self._next_entity_id += 1

        self.entities[entity_id] = {
            "name": name,
            "type": entity_type,
            "description": description,
            "chunk_ids": [chunk_id] if chunk_id else []
        }

        # 更新索引
        if name.lower() not in self.entity_index:
            self.entity_index[name.lower()] = []
        self.entity_index[name.lower()].append(entity_id)

        self._mark_dirty()  # 标记变更
        return entity_id

    def add_relation(
        self,
        head: str,
        tail: str,
        relation: str,
        weight: float = 1.0,
        chunk_id: str = ""
    ) -> Optional[str]:
        """添加关系，返回关系ID"""
        # 确保头尾实体存在
        if head.lower() not in self.entity_index or tail.lower() not in self.entity_index:
            return None

        head_ids = self.entity_index[head.lower()]
        tail_ids = self.entity_index[tail.lower()]

        if not head_ids or not tail_ids:
            return None

        # 创建关系
        relation_id = f"relation_{self._next_relation_id}"
        self._next_relation_id += 1

        self.relations[relation_id] = {
            "head": head_ids[0],
            "tail": tail_ids[0],
            "relation": relation,
            "weight": weight,
            "chunk_id": chunk_id
        }

        self._mark_dirty()  # 标记变更
        return relation_id

    def add_image_entity(
        self,
        figure_id: str,
        image_path: str,
        description: str = "",
        figure_type: str = "unknown",
        chunk_id: str = ""
    ) -> str:
        """
        添加图片实体

        Args:
            figure_id: 图片标识符（如 "Figure 2", "Table 1"）
            image_path: 图片文件路径
            description: 图片描述
            figure_type: 图片类型（chart/photo/diagram/graph/table/unknown）
            chunk_id: 来源 chunk ID

        Returns:
            实体ID
        """
        # 检查是否已存在同名实体
        existing_ids = self.entity_index.get(figure_id.lower(), [])
        if existing_ids:
            entity_id = existing_ids[0]
            # 更新图片信息
            self.entities[entity_id].update({
                "image_path": image_path,
                "description": description,
                "figure_type": figure_type
            })
            if chunk_id and chunk_id not in self.entities[entity_id].get("chunk_ids", []):
                self.entities[entity_id].setdefault("chunk_ids", []).append(chunk_id)
            self._mark_dirty()
            return entity_id

        # 创建新实体
        entity_id = f"figure_{self._next_entity_id}"
        self._next_entity_id += 1

        self.entities[entity_id] = {
            "name": figure_id,
            "type": "Figure",
            "image_path": image_path,
            "description": description,
            "figure_type": figure_type,
            "chunk_ids": [chunk_id] if chunk_id else []
        }

        # 更新索引
        if figure_id.lower() not in self.entity_index:
            self.entity_index[figure_id.lower()] = []
        self.entity_index[figure_id.lower()].append(entity_id)

        self._mark_dirty()
        return entity_id

    def add_table_entity(
        self,
        table_id: str,
        description: str = "",
        chunk_id: str = ""
    ) -> str:
        """
        添加表格实体

        Args:
            table_id: 表格标识符（如 "Table 1"）
            description: 表格描述
            chunk_id: 来源 chunk ID

        Returns:
            实体ID
        """
        # 检查是否已存在
        existing_ids = self.entity_index.get(table_id.lower(), [])
        if existing_ids:
            entity_id = existing_ids[0]
            if chunk_id and chunk_id not in self.entities[entity_id].get("chunk_ids", []):
                self.entities[entity_id].setdefault("chunk_ids", []).append(chunk_id)
            self._mark_dirty()
            return entity_id

        entity_id = f"table_{self._next_entity_id}"
        self._next_entity_id += 1

        self.entities[entity_id] = {
            "name": table_id,
            "type": "Table",
            "description": description,
            "chunk_ids": [chunk_id] if chunk_id else []
        }

        if table_id.lower() not in self.entity_index:
            self.entity_index[table_id.lower()] = []
        self.entity_index[table_id.lower()].append(entity_id)

        self._mark_dirty()
        return entity_id

    def remove_entity(self, entity_id: str) -> bool:
        """
        删除实体及其相关关系

        Args:
            entity_id: 实体ID

        Returns:
            是否删除成功
        """
        if entity_id not in self.entities:
            return False

        entity_name = self.entities[entity_id]["name"]

        # 删除相关关系
        related_relations = []
        for rel_id, rel_data in self.relations.items():
            if rel_data.get("head") == entity_id or rel_data.get("tail") == entity_id:
                related_relations.append(rel_id)

        for rel_id in related_relations:
            del self.relations[rel_id]

        # 从索引中移除
        name_lower = entity_name.lower()
        if name_lower in self.entity_index:
            self.entity_index[name_lower] = [eid for eid in self.entity_index[name_lower] if eid != entity_id]
            if not self.entity_index[name_lower]:
                del self.entity_index[name_lower]

        # 删除实体
        del self.entities[entity_id]

        self._mark_dirty()  # 标记变更
        return True

    def remove_relation(self, relation_id: str) -> bool:
        """
        删除关系

        Args:
            relation_id: 关系ID

        Returns:
            是否删除成功
        """
        if relation_id not in self.relations:
            return False

        del self.relations[relation_id]
        self._mark_dirty()  # 标记变更
        return True

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取实体"""
        return self.entities.get(entity_id)

    def get_entities_by_name(self, name: str) -> List[Dict[str, Any]]:
        """根据名称搜索实体"""
        entity_ids = self.entity_index.get(name.lower(), [])
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Tuple[str, str, str]]:
        """
        获取实体邻居

        Returns:
            List of (neighbor_entity_id, relation_type, relation_id)
        """
        neighbors = []

        for relation_id, rel_data in self.relations.items():
            if relation_type and rel_data["relation"] != relation_type:
                continue

            if direction in ("both", "out") and rel_data["head"] == entity_id:
                neighbors.append((rel_data["tail"], rel_data["relation"], relation_id))
            if direction in ("both", "in") and rel_data["tail"] == entity_id:
                neighbors.append((rel_data["head"], rel_data["relation"], relation_id))

        return neighbors

    def get_triplets(self, entity_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        获取以某实体为中心的多跳三元组

        Args:
            entity_id: 中心实体ID
            depth: 跳数

        Returns:
            List of triplets {"head": ..., "relation": ..., "tail": ...}
        """
        triplets = []
        visited = set()

        def dfs(current_id: str, current_depth: int, path: List):
            if current_depth > depth:
                return

            for neighbor_id, relation, rel_id in self.get_neighbors(current_id):
                key = (neighbor_id, relation, rel_id)
                if key in visited:
                    continue
                visited.add(key)

                head_entity = self.entities.get(current_id, {})
                tail_entity = self.entities.get(neighbor_id, {})

                if head_entity and tail_entity:
                    triplets.append({
                        "head": head_entity.get("name", ""),
                        "relation": relation,
                        "tail": tail_entity.get("name", ""),
                        "head_id": current_id,
                        "tail_id": neighbor_id
                    })

                path.append((neighbor_id, relation))
                dfs(neighbor_id, current_depth + 1, path)
                path.pop()

        dfs(entity_id, 0, [])
        return triplets

    def search_related_entities(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索相关实体"""
        results = []
        query_lower = query.lower()

        for entity_id, entity in self.entities.items():
            name_lower = entity["name"].lower()
            # 简单的词匹配
            score = 0.0
            if query_lower in name_lower:
                score = 1.0
            elif any(word in name_lower for word in query_lower.split()):
                score = 0.5

            if score > 0:
                results.append({
                    "entity": entity,
                    "entity_id": entity_id,
                    "score": score
                })

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class Neo4jGraphStore:
    """
    Neo4j 图谱存储（生产环境使用）

    使用继承模式封装 llama_index.graph_stores.neo4j.Neo4jGraphStore，
    提供与 MemoryGraphStore 相同的接口。

    需要安装依赖：pip install llama-index-graph-stores-neo4j

    Neo4j 数据模型：
    - 节点：label=Entity, 属性: id(实体名), type(实体类型), description, chunk_ids
    - 关系：type=关系名, 属性: weight, chunk_id
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "",
        database: str = "neo4j"
    ):
        """
        初始化 Neo4j 图谱存储

        Args:
            uri: Neo4j 连接 URI (bolt://localhost:7687)
            username: Neo4j 用户名
            password: Neo4j 密码
            database: 数据库名称 (默认 neo4j)
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._store = None  # llama_index Neo4jGraphStore 实例
        self._entity_id_map: Dict[str, str] = {}  # name -> internal_id
        self._next_entity_id = 1

    def _ensure_store(self):
        """确保底层存储已初始化"""
        if self._store is None:
            try:
                from llama_index.graph_stores.neo4j import Neo4jGraphStore
                self._store = Neo4jGraphStore(
                    url=self.uri,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                    refresh_schema=True
                )
                logger.info(f"✅ Neo4j 图谱存储已连接: {self.uri}")
            except ImportError:
                raise ImportError(
                    "Neo4j 支持需要安装 llama-index-graph-stores-neo4j: "
                    "pip install llama-index-graph-stores-neo4j"
                )

    def add_entity(
        self,
        name: str,
        entity_type: str = "UNKNOWN",
        description: str = "",
        chunk_id: str = ""
    ) -> str:
        """
        添加实体到图谱

        Args:
            name: 实体名称
            entity_type: 实体类型
            description: 实体描述
            chunk_id: 来源 chunk ID

        Returns:
            实体 ID
        """
        self._ensure_store()

        # 生成实体 ID (使用索引避免冲突)
        entity_id = f"entity_{self._next_entity_id}"
        self._next_entity_id += 1

        # 存储到 Neo4j：使用 upsert_triplet 创建孤立节点
        # 关系名为空表示只是创建实体节点
        safe_name = name.replace("'", "\\'")
        safe_desc = description.replace("'", "\\'") if description else ""

        # 使用 Cypher 创建节点并设置属性
        assert self._store is not None
        query = f"""
        MERGE (n:{{label}} {{id: $name}})
        ON CREATE SET n.type = $type, n.description = $desc, n.chunk_ids = [$chunk_id]
        ON MATCH SET n.description = $desc, n.chunk_ids = n.chunk_ids + $chunk_id
        RETURN n.id AS id
        """.format(label=self._store.node_label)

        self._store.query(query, {
            "name": name,
            "type": entity_type,
            "desc": safe_desc,
            "chunk_id": chunk_id
        })

        # 记录映射
        self._entity_id_map[name.lower()] = entity_id

        return entity_id

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

        Args:
            head: 头实体名称
            tail: 尾实体名称
            relation: 关系类型
            weight: 关系权重
            chunk_id: 来源 chunk ID

        Returns:
            关系 ID 或 None
        """
        self._ensure_store()

        # 先确保头尾实体存在
        if head.lower() not in self._entity_id_map:
            self.add_entity(head, chunk_id=chunk_id)
        if tail.lower() not in self._entity_id_map:
            self.add_entity(tail, chunk_id=chunk_id)

        # 使用 llama_index 的 upsert_triplet
        try:
            assert self._store is not None
            self._store.upsert_triplet(head, relation, tail)

            # 生成关系 ID
            relation_id = f"relation_{head}_{relation}_{tail}"
            return relation_id
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
            return None

    def add_image_entity(
        self,
        figure_id: str,
        image_path: str,
        description: str = "",
        figure_type: str = "unknown",
        chunk_id: str = ""
    ) -> str:
        """
        添加图片实体到图谱

        Args:
            figure_id: 图片标识符（如 "Figure 2", "Table 1"）
            image_path: 图片文件路径
            description: 图片描述
            figure_type: 图片类型（chart/photo/diagram/graph/table/unknown）
            chunk_id: 来源 chunk ID

        Returns:
            实体 ID
        """
        self._ensure_store()

        entity_id = f"figure_{self._next_entity_id}"
        self._next_entity_id += 1

        safe_figure_id = figure_id.replace("'", "\\'")
        safe_desc = description.replace("'", "\\'") if description else ""
        safe_path = image_path.replace("'", "\\'") if image_path else ""

        assert self._store is not None
        query = f"""
        MERGE (n:{{label}} {{id: $figure_id}})
        ON CREATE SET n.type = $type, n.description = $desc, n.image_path = $path,
                      n.figure_type = $figure_type, n.chunk_ids = [$chunk_id]
        ON MATCH SET n.description = $desc, n.image_path = $path,
                      n.figure_type = $figure_type, n.chunk_ids = n.chunk_ids + $chunk_id
        RETURN n.id AS id
        """.format(label=self._store.node_label)

        self._store.query(query, {
            "figure_id": figure_id,
            "type": "Figure",
            "desc": safe_desc,
            "path": safe_path,
            "figure_type": figure_type,
            "chunk_id": chunk_id
        })

        self._entity_id_map[figure_id.lower()] = entity_id
        return entity_id

    def add_table_entity(
        self,
        table_id: str,
        description: str = "",
        chunk_id: str = ""
    ) -> str:
        """
        添加表格实体到图谱

        Args:
            table_id: 表格标识符（如 "Table 1"）
            description: 表格描述
            chunk_id: 来源 chunk ID

        Returns:
            实体 ID
        """
        self._ensure_store()

        entity_id = f"table_{self._next_entity_id}"
        self._next_entity_id += 1

        safe_table_id = table_id.replace("'", "\\'")
        safe_desc = description.replace("'", "\\'") if description else ""

        assert self._store is not None
        query = f"""
        MERGE (n:{{label}} {{id: $table_id}})
        ON CREATE SET n.type = $type, n.description = $desc, n.chunk_ids = [$chunk_id]
        ON MATCH SET n.description = $desc, n.chunk_ids = n.chunk_ids + $chunk_id
        RETURN n.id AS id
        """.format(label=self._store.node_label)

        self._store.query(query, {
            "table_id": table_id,
            "type": "Table",
            "desc": safe_desc,
            "chunk_id": chunk_id
        })

        self._entity_id_map[table_id.lower()] = entity_id
        return entity_id

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        根据实体 ID 获取实体

        Args:
            entity_id: 实体 ID

        Returns:
            实体信息字典或 None
        """
        self._ensure_store()

        # 通过 Cypher 查询实体
        assert self._store is not None
        query = f"""
        MATCH (n:{{label}})
        WHERE n.id = $entity_id OR n.id CONTAINS $entity_id
        RETURN n.id AS id, n.type AS type, n.description AS description, n.chunk_ids AS chunk_ids
        LIMIT 1
        """.format(label=self._store.node_label)

        results = self._store.query(query, {"entity_id": entity_id})
        if results:
            record = results[0]
            return {
                "name": record.get("id", ""),
                "type": record.get("type", ""),
                "description": record.get("description", ""),
                "chunk_ids": record.get("chunk_ids", [])
            }
        return None

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Tuple[str, str, str]]:
        """
        获取实体的邻居节点和关系

        Args:
            entity_id: 实体 ID
            relation_type: 关系类型过滤（可选）
            direction: 方向 ("out", "in", "both")

        Returns:
            List of (neighbor_entity_id, relation_type, relation_id)
        """
        self._ensure_store()

        neighbors = []

        # 根据方向构建不同的查询
        if direction == "out":
            # 只查发出的关系
            query = f"""
            MATCH (n:{{label}})-[r]->(m:{{label}})
            WHERE n.id = $entity_id
            RETURN m.id AS neighbor, type(r) AS rel_type, 'out' AS direction
            """
        elif direction == "in":
            # 只查接收的关系
            query = f"""
            MATCH (n:{{label}})-[r]->(m:{{label}})
            WHERE m.id = $entity_id
            RETURN n.id AS neighbor, type(r) AS rel_type, 'in' AS direction
            """
        else:
            # 两个方向
            query = f"""
            MATCH (n:{{label}})-[r]->(m:{{label}})
            WHERE n.id = $entity_id OR m.id = $entity_id
            RETURN CASE WHEN n.id = $entity_id THEN m.id ELSE n.id END AS neighbor,
                   type(r) AS rel_type,
                   CASE WHEN n.id = $entity_id THEN 'out' ELSE 'in' END AS direction
            """

        assert self._store is not None
        query = query.format(label=self._store.node_label)
        results = self._store.query(query, {"entity_id": entity_id})

        for record in results:
            neighbor = record.get("neighbor", "")
            rel_type = record.get("rel_type", "")
            rel_direction = record.get("direction", "out")

            # 如果有关系类型过滤
            if relation_type and rel_type.upper() != relation_type.upper():
                continue

            # 生成 relation_id
            if rel_direction == "out":
                relation_id = f"rel_{entity_id}_{rel_type}_{neighbor}"
            else:
                relation_id = f"rel_{neighbor}_{rel_type}_{entity_id}"

            neighbors.append((neighbor, rel_type, relation_id))

        return neighbors

    def get_triplets(self, entity_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        获取以某实体为中心的多跳三元组

        Args:
            entity_id: 中心实体 ID
            depth: 跳数

        Returns:
            List of triplets {"head": ..., "relation": ..., "tail": ...}
        """
        self._ensure_store()

        triplets = []

        # 使用 get_rel_map 获取多跳关系
        try:
            assert self._store is not None
            rel_map = self._store.get_rel_map(
                subjs=[entity_id],
                depth=depth,
                limit=100
            )

            if entity_id in rel_map:
                flattened_rels = rel_map[entity_id]
                # flattened_rels 格式: [rel1, obj1, rel2, obj2, ...]
                # 需要按对处理
                current_head = entity_id
                i = 0
                while i < len(flattened_rels):
                    if i + 1 < len(flattened_rels):
                        rel = str(flattened_rels[i])
                        tail = str(flattened_rels[i + 1])
                        triplets.append({
                            "head": current_head,
                            "relation": rel,
                            "tail": tail,
                            "head_id": current_head,
                            "tail_id": tail
                        })
                        current_head = tail
                        i += 2
                    else:
                        break
        except Exception as e:
            logger.warning(f"获取多跳三元组失败: {e}")

        return triplets

    def search_related_entities(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        根据名称搜索相关实体

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相关实体列表
        """
        self._ensure_store()

        results = []

        # 使用 CONTAINS 进行模糊匹配
        assert self._store is not None
        cypher = f"""
        MATCH (n:{{label}})
        WHERE n.id CONTAINS $query OR toLower(n.id) CONTAINS toLower($query)
        RETURN n.id AS id, n.type AS type, n.description AS description, n.chunk_ids AS chunk_ids
        LIMIT $limit
        """.format(label=self._store.node_label)

        try:
            records = self._store.query(cypher, {
                "query": query,
                "limit": top_k
            })

            for record in records:
                entity_id = record.get("id", "")
                # 计算简单的匹配分数
                query_lower = query.lower()
                id_lower = entity_id.lower()
                if query_lower == id_lower:
                    score = 1.0
                elif query_lower in id_lower:
                    score = 0.8
                else:
                    score = 0.5

                results.append({
                    "entity": {
                        "name": entity_id,
                        "type": record.get("type", ""),
                        "description": record.get("description", ""),
                        "chunk_ids": record.get("chunk_ids", [])
                    },
                    "entity_id": entity_id,
                    "score": score
                })

        except Exception as e:
            logger.error(f"搜索实体失败: {e}")

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict[str, int]:
        """
        获取图谱统计信息

        Returns:
            统计字典
        """
        self._ensure_store()
        assert self._store is not None

        try:
            # 统计实体数量
            entity_query = f"""
            MATCH (n:{{label}})
            RETURN count(n) AS count
            """.format(label=self._store.node_label)

            # 统计关系数量
            rel_query = """
            MATCH ()-[r]->()
            RETURN count(r) AS count
            """

            entity_results = self._store.query(entity_query)
            rel_results = self._store.query(rel_query)

            entity_count = entity_results[0].get("count", 0) if entity_results else 0
            relation_count = rel_results[0].get("count", 0) if rel_results else 0

            return {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "index_size": len(self._entity_id_map)
            }
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return {"entity_count": 0, "relation_count": 0, "index_size": 0}

    def clear(self):
        """清空图谱"""
        self._ensure_store()
        assert self._store is not None

        try:
            # 删除所有节点（会级联删除关系）
            query = f"""
            MATCH (n:{{label}})
            DETACH DELETE n
            """.format(label=self._store.node_label)
            self._store.query(query)

            # 清空映射
            self._entity_id_map.clear()
            self._next_entity_id = 1

            logger.info("✅ Neo4j 图谱已清空")
        except Exception as e:
            logger.error(f"清空图谱失败: {e}")
            raise


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
        base_engine: Any,  # HybridRAGEngine
        context: Any = None
    ):
        self.config = config
        self.base_engine = base_engine
        self.context = context
        self._graph_store: Optional[Union[MemoryGraphStore, Neo4jGraphStore]] = None
        self._initialized = False

    def _get_graph_store(self) -> Union[MemoryGraphStore, Neo4jGraphStore]:
        """获取图谱存储实例"""
        if self._graph_store is None:
            if self.config.storage_type == "neo4j":
                self._graph_store = Neo4jGraphStore(
                    uri=self.config.neo4j_uri,
                    username=self.config.neo4j_user,
                    password=self.config.neo4j_password
                )
            else:
                self._graph_store = MemoryGraphStore()

        return self._graph_store

    async def initialize(self):
        """初始化图谱引擎"""
        if self._initialized:
            return

        if not self.config.enable_graph_rag:
            logger.info("Graph RAG 功能未启用")
            return

        logger.info(f"✅ Graph RAG 引擎已初始化 (存储类型: {self.config.storage_type})")
        logger.info(f"   - 最大三元组/Chunk: {self.config.max_triplets_per_chunk}")
        logger.info(f"   - 图谱检索TopK: {self.config.graph_retrieval_top_k}")
        logger.info(f"   - 混合检索Alpha: {self.config.hybrid_alpha}")

        self._initialized = True

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        搜索接口，支持三种模式

        Args:
            query: 查询文本
            mode: 检索模式 ("vector", "graph", "hybrid")
            top_k: 返回结果数量

        Returns:
            检索结果字典
        """
        if not self.config.enable_graph_rag:
            return {
                "type": "error",
                "message": "Graph RAG 功能未启用"
            }

        if mode not in self.RETRIEVAL_MODES:
            return {
                "type": "error",
                "message": f"不支持的检索模式: {mode}，支持的模式: {self.RETRIEVAL_MODES}"
            }

        try:
            if mode == "vector":
                return await self._vector_search(query, top_k)
            elif mode == "graph":
                return await self._graph_search(query, top_k)
            else:  # hybrid
                return await self._hybrid_search(query, top_k)
        except Exception as e:
            logger.error(f"Graph RAG 搜索失败: {e}")
            return {
                "type": "error",
                "message": f"Graph RAG 搜索失败: {str(e)}"
            }

    async def _vector_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """纯向量检索（委托给 base_engine）"""
        if self.base_engine is None:
            return {
                "type": "error",
                "message": "基础引擎未初始化"
            }

        result = await self.base_engine.search(query, mode="retrieve")
        return result

    async def _graph_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """纯图谱检索"""
        graph_store = self._get_graph_store()

        # 1. 搜索相关实体
        related_entities = graph_store.search_related_entities(
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

        # 2. 获取每个实体的多跳三元组
        all_triplets = []
        all_entities = []

        for item in related_entities:
            entity_id = item["entity_id"]
            entity = item["entity"]

            all_entities.append({
                "entity_id": entity_id,
                "name": entity.get("name", ""),
                "type": entity.get("type", ""),
                "description": entity.get("description", ""),
                "score": item["score"]
            })

            # 获取 1 跳三元组
            triplets = graph_store.get_triplets(entity_id, depth=1)
            all_triplets.extend(triplets)

        # 3. 去重并构建回答
        unique_triplets = []
        seen = set()
        for t in all_triplets:
            key = (t.get("head", ""), t.get("relation", ""), t.get("tail", ""))
            if key not in seen:
                seen.add(key)
                unique_triplets.append(t)

        # 4. 构建图谱上下文
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
        # 1. 并行执行向量检索和图谱检索
        vector_task = self._vector_search(query, top_k)
        graph_task = self._graph_search(query, top_k)

        vector_result, graph_result = await asyncio.gather(
            vector_task,
            graph_task,
            return_exceptions=True
        )

        # 处理异常
        if isinstance(vector_result, Exception):
            logger.warning(f"向量检索失败，使用纯图谱检索: {vector_result}")
            return graph_result  # type: ignore

        if isinstance(graph_result, Exception):
            logger.warning(f"图谱检索失败，使用纯向量检索: {graph_result}")
            return vector_result  # type: ignore

        # 3. RRF 融合（使用 alpha 加权）
        # 向量结果使用 similarity score
        # 图谱结果使用 entity score
        # 经过上述检查，两个结果都不是异常
        vr: Dict[str, Any] = vector_result  # type: ignore
        gr: Dict[str, Any] = graph_result  # type: ignore

        vector_sources = vr.get("sources", [])
        graph_entities = gr.get("entities", [])
        graph_triplets = gr.get("triplets", [])

        # 3. 构建混合回答
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

        # 实体信息
        if entities:
            context += "相关实体：\n"
            for e in entities[:5]:
                context += f"- {e['name']} ({e['type']})\n"
            context += "\n"

        # 关系信息
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

        # 图谱部分
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

        # 向量检索部分
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
        """
        从文档节点构建知识图谱

        Args:
            nodes: Node 列表，每个 Node 包含 text 和 metadata

        Returns:
            构建结果统计
        """
        if not self.config.enable_graph_rag:
            return {"status": "skipped", "message": "Graph RAG 功能未启用"}

        try:
            # 延迟导入避免循环依赖
            try:
                from .graph_builder import MultimodalGraphBuilder
            except ImportError:
                from graph_builder import MultimodalGraphBuilder

            # 创建多模态图谱构建器
            builder = MultimodalGraphBuilder(
                config=self.config,
                context=self.context
            )

            # 构建图谱
            graph_store = self._get_graph_store()
            stats = await builder.build_from_nodes(nodes, graph_store)

            # 如果是 MemoryGraphStore，保存到磁盘
            if isinstance(graph_store, MemoryGraphStore):
                graph_store.save(force=True)
                logger.info(f"💾 知识图谱已保存到磁盘")

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

        graph_store = self._get_graph_store()
        return {
            "enabled": True,
            "storage_type": self.config.storage_type,
            **graph_store.get_stats()
        }

    async def clear_graph(self, delete_storage: bool = True) -> Dict[str, Any]:
        """
        清空图谱

        Args:
            delete_storage: 是否删除存储文件（仅 MemoryGraphStore 有效）
        """
        if not self.config.enable_graph_rag:
            return {"status": "skipped", "message": "Graph RAG 功能未启用"}

        graph_store = self._get_graph_store()

        # 如果是 MemoryGraphStore，删除存储文件
        if delete_storage and isinstance(graph_store, MemoryGraphStore):
            graph_store.clear(delete_storage=True)
        else:
            graph_store.clear()

        return {"status": "success", "message": "图谱已清空"}


def create_graph_rag_engine(
    config: GraphRAGConfig,
    base_engine: Any,
    context: Any = None
) -> GraphRAGEngine:
    """
    创建 Graph RAG 引擎实例

    Args:
        config: Graph RAG 配置
        base_engine: 基础 RAG 引擎（HybridRAGEngine）
        context: AstrBot 上下文

    Returns:
        GraphRAGEngine 实例
    """
    engine = GraphRAGEngine(config, base_engine, context)
    return engine
