"""
Paper RAG 插件的 Milvus 管理器
完全参照 mnemosyne 插件的实现，避免与主进程冲突
"""

import os
import pathlib
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional, cast
from urllib.parse import urlparse
import asyncio

# 抑制底层库的 gRPC/absl 警告（必须在导入任何库之前设置）
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from pymilvus.exceptions import (
    CollectionNotExistException,
    MilvusException,
)

from astrbot.api import logger


class PaperMilvusManager:
    """
    论文插件的 Milvus 管理器

    特点：
    - 完全参照 mnemosyne 的实现
    - 使用唯一的连接别名
    - 延迟连接模式
    - 避免与主进程或其他插件冲突
    """

    def __init__(
        self,
        alias: str = "paperrag_v2",
        lite_path: Optional[str] = None,
        uri: Optional[str] = None,
        collection_name: str = "paper_embeddings",
        dim: int = 768,
        plugin_data_dir: Optional[str] = None,
        authentication: Optional[dict] = None,
        db_name: str = "default",
        **kwargs
    ):
        """
        初始化 Milvus 管理器

        Args:
            alias: 连接别名（必须唯一）
            lite_path: Milvus Lite 数据库文件路径
            uri: 标准 Milvus 连接 URI
            collection_name: 集合名称
            dim: 向量维度
            plugin_data_dir: 插件数据目录
            authentication: 认证信息 {"token": "..."} 或 {"user": "...", "password": "..."}
            db_name: 数据库名称（默认 "default"）
        """
        self.alias = alias
        self._collection_name = collection_name
        self._dim = dim
        self._plugin_data_dir = plugin_data_dir
        self.authentication = authentication or {}
        self.db_name = db_name or "default"

        # 连接配置
        self._lite_path = self._prepare_lite_path(lite_path) if lite_path else None
        self._uri = uri
        self._is_lite = self._lite_path is not None

        # 连接状态
        self._is_connected = False
        self._connection_info = {}
        self._collection = None

        # 确定连接模式
        self._configure_connection_mode()


    def _prepare_lite_path(self, path_input: str) -> str:
        """准备 Milvus Lite 路径"""
        path = Path(path_input)
        if not path.is_absolute():
            if self._plugin_data_dir:
                base = Path(self._plugin_data_dir).resolve()
                path = (base / path).resolve()
            else:
                path = path.resolve()

        # 如果是目录，附加默认文件名
        if path.is_dir() or (not path.exists() and not str(path).endswith('.db')):
            path = path / "milvus_papers.db"

        return str(path)

    def _configure_connection_mode(self):
        """配置连接模式"""
        if self._lite_path:
            self._configure_lite_explicit()
        elif self._uri:
            self._configure_uri()
        else:
            self._configure_lite_default()

    def _configure_lite_explicit(self):
        """配置使用显式指定的 Milvus Lite 路径"""
        self._is_lite = True
        logger.info(f"配置 Milvus Lite (别名: {self.alias}), 路径: '{self._lite_path}'")

        # 确保目录存在
        if self._lite_path:
            db_dir = os.path.dirname(self._lite_path)
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                    logger.info(f"为 Milvus Lite 创建了目录: '{db_dir}'")
                except OSError as e:
                    logger.error(f"无法为 Milvus Lite 创建目录 '{db_dir}': {e}")
                    raise

        # 使用处理后的完整文件路径作为 URI
        self._connection_info["uri"] = self._lite_path

    def _configure_lite_default(self):
        """配置使用默认的 Milvus Lite 路径"""
        self._is_lite = True

        if self._plugin_data_dir:
            default_path = self._prepare_lite_path(str(self._plugin_data_dir))
        else:
            default_path = "./data/milvus_papers.db"

        logger.warning(f"使用默认 Milvus Lite 路径: '{default_path}'")

        # 确保目录存在
        db_dir = os.path.dirname(default_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._connection_info["uri"] = default_path

    def _configure_uri(self):
        """配置使用标准网络 URI 连接"""
        self._is_lite = False
        logger.info(f"配置标准 Milvus (别名: {self.alias}), URI: '{self._uri}'")
        self._connection_info["uri"] = self._uri

    def connect(self) -> None:
        """建立到 Milvus 的连接"""
        if self._is_connected:
            logger.debug(f"已连接到 Milvus (别名: {self.alias})")
            return

        mode = "Milvus Lite" if self._is_lite else "Standard Milvus"

        try:
            logger.debug(f"尝试连接到 {mode} (别名: {self.alias})")

            # 构建连接参数
            connect_params = dict(self._connection_info)

            # 添加认证信息
            if self.authentication.get("token"):
                connect_params["token"] = self.authentication["token"]
                logger.debug(f"使用 token 认证")
            elif self.authentication.get("user"):
                connect_params["user"] = self.authentication["user"]
                if self.authentication.get("password"):
                    connect_params["password"] = self.authentication["password"]
                logger.debug(f"使用用户名/密码认证")

            # 添加数据库名称（如果不是默认）
            if self.db_name != "default":
                connect_params["db_name"] = self.db_name
                logger.debug(f"连接到数据库: {self.db_name}")

            connections.connect(
                alias=self.alias,
                **connect_params
            )

            self._is_connected = True
            logger.info(f"✅ 成功连接到 {mode} (别名: {self.alias})")

        except MilvusException as e:
            logger.error(f"连接 {mode} (别名: {self.alias}) 失败: {e}")
            self._is_connected = False
            raise
        except Exception as e:
            logger.error(f"连接时发生错误: {e}")
            self._is_connected = False
            raise

    def disconnect(self) -> None:
        """断开 Milvus 连接"""
        if not self._is_connected:
            return

        try:
            connections.disconnect(self.alias)
            self._is_connected = False
            logger.debug(f"已断开 Milvus 连接 (别名: {self.alias})")
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")
            self._is_connected = False

    @staticmethod
    async def _await_if_needed(result: Any) -> None:
        """
        等待异步操作完成（如果需要）

        兼容不同 pymilvus 版本的返回值类型：
        - 同步操作：返回 None
        - 协程：返回 coroutine
        - Future：返回 MutationFuture 或其他 Future 对象

        Args:
            result: pymilvus 操作的返回值
        """
        if asyncio.iscoroutine(result):
            await result
        elif result is not None and hasattr(result, 'done'):
            # 这是一个 Future (MutationFuture)
            try:
                result.result()
            except Exception as e:
                logger.warning(f"Future.result() 调用失败: {e}")

    async def _await_if_needed_with_result(self, result: Any) -> Any:
        """
        等待异步操作完成并返回结果（如果需要）

        Args:
            result: pymilvus 操作的返回值

        Returns:
            操作结果（如果有）
        """
        if asyncio.iscoroutine(result):
            return await result
        elif result is not None and hasattr(result, 'done'):
            try:
                return result.result()
            except Exception as e:
                logger.warning(f"Future.result() 调用失败: {e}")
                return None
        return result

    async def _ensure_collection(self):
        """确保集合已创建"""
        if self._collection is not None:
            return

        try:
            # 先确保已连接
            if not self._is_connected:
                self.connect()

            # 检查集合是否存在
            if utility.has_collection(self._collection_name, using=self.alias):
                logger.debug(f"集合 '{self._collection_name}' 已存在，加载")
                self._collection = Collection(self._collection_name, using=self.alias)
                collection = cast(Collection, self._collection)

                # 等待 load 完成（如果需要）
                load_result = collection.load()
                await self._await_if_needed(load_result)
            else:
                logger.debug(f"创建新集合 '{self._collection_name}'")

                # 定义 schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]

                schema = CollectionSchema(
                    fields=fields,
                    description=f"Paper embeddings collection (alias: {self.alias})"
                )

                self._collection = Collection(
                    name=self._collection_name,
                    schema=schema,
                    using=self.alias
                )

                # 创建索引
                is_lite = self._is_lite
                index_type = "AUTOINDEX" if is_lite else "HNSW"
                index_params: Dict[str, Any] = {
                    "index_type": index_type,
                    "metric_type": "COSINE"
                }

                # 如果是 HNSW，添加参数
                if not is_lite:
                    index_params["params"] = {"M": 8, "efConstruction": 64}

                # 等待索引创建完成（如果需要）
                collection = cast(Collection, self._collection)
                index_result = collection.create_index(
                    field_name="vector",
                    index_params=index_params
                )
                await self._await_if_needed(index_result)

                # 等待 load 完成（如果需要）
                load_result = collection.load()
                await self._await_if_needed(load_result)

                logger.info(f"✅ 集合 '{self._collection_name}' 创建成功 (索引: {index_type})")

        except Exception as e:
            logger.error(f"集合初始化失败: {e}")
            raise

    async def insert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        插入文档到 Milvus

        Args:
            documents: 文档列表，格式 [{"text": "...", "embedding": [...], "metadata": {...}}, ...]

        Returns:
            插入的文档数量
        """
        await self._ensure_collection()

        try:
            # 准备数据
            data = []
            for doc in documents:
                data.append({
                    "text": doc["text"],
                    "vector": doc["embedding"],
                    "metadata": doc.get("metadata", {})
                })

            # 插入数据
            collection = cast(Collection, self._collection)
            insert_result = collection.insert(data)

            # 等待插入完成（如果需要）
            await self._await_if_needed(insert_result)

            # 等待 flush 完成（如果需要）
            flush_result = collection.flush()
            await self._await_if_needed(flush_result)

            logger.debug(f"插入 {len(data)} 个文档到 Milvus")
            return len(data)

        except Exception as e:
            logger.error(f"插入文档失败: {e}")
            raise

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Args:
            query_embedding: 查询向量
            top_k: 返回前 K 个结果

        Returns:
            搜索结果列表
        """
        await self._ensure_collection()

        try:
            # 构建搜索参数
            # 对于 HNSW 索引，使用较大的 ef 参数提高搜索精度，减少重复
            search_params = {
                "metric_type": "COSINE",
                "params": {
                    "ef": max(top_k * 2, 64)  # ef >= top_k，提高搜索精度
                }
            }

            # 对于 AUTOINDEX（Milvus Lite），参数可能不同
            if self._is_lite:
                # Milvus Lite 的 AUTOINDEX 可能不支持 ef 参数
                search_params = {
                    "metric_type": "COSINE",
                    "params": {}
                }

            # 执行搜索
            collection = cast(Collection, self._collection)
            search_result = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"]
            )

            # 等待搜索完成（如果需要）
            results = await self._await_if_needed_with_result(search_result)

            if results is None:
                return []

            # 转换结果
            documents = []
            for hit in results[0]:
                documents.append({
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": float(hit.score)
                })

            return documents

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise

    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        列出所有文档

        Returns:
            文档列表（按文件分组）
        """
        await self._ensure_collection()

        try:
            collection = cast(Collection, self._collection)
            # 使用 query 获取所有数据
            # 注意：Milvus 要求空表达式必须指定 limit
            query_result = collection.query(
                expr="",  # 空表达式表示查询所有
                output_fields=["text", "metadata"],
                limit=16384  # Milvus 最大默认值，足够获取所有文档
            )

            # 等待查询完成（如果需要）
            results = await self._await_if_needed_with_result(query_result)

            if results is None:
                return []

            documents = []
            for hit in results:
                # Milvus query 返回的是字典
                if isinstance(hit, dict):
                    documents.append({
                        "text": hit.get("text"),
                        "metadata": hit.get("metadata", {})
                    })
                else:
                    # 兼容旧格式（如果有 .entity 属性）
                    documents.append({
                        "text": hit.entity.get("text"),
                        "metadata": hit.entity.get("metadata", {})
                    })

            # 按文件分组统计
            file_stats = {}
            for doc in documents:
                metadata = doc.get("metadata", {})
                file_name = metadata.get("file_name", "unknown")
                if file_name not in file_stats:
                    file_stats[file_name] = {
                        "file_name": file_name,
                        "chunk_count": 0,
                        "added_time": metadata.get("added_time", "unknown")
                    }
                file_stats[file_name]["chunk_count"] += 1

            # 转换为列表
            return list(file_stats.values())

        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            return []

    async def clear_collection(self) -> bool:
        """
        清空集合

        Returns:
            是否成功
        """
        try:
            await self._ensure_collection()

            # 删除集合
            drop_result = utility.drop_collection(self._collection_name, using=self.alias)

            # 等待删除完成（如果需要）
            await self._await_if_needed(drop_result)

            self._collection = None
            logger.info(f"✅ 集合 '{self._collection_name}' 已清空")
            return True

        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息字典
        """
        try:
            await self._ensure_collection()

            collection = cast(Collection, self._collection)
            # 获取集合统计信息
            stats_result = collection.load()
            await self._await_if_needed(stats_result)

            # 获取实体数量
            num_entities = collection.num_entities

            return {
                "total_entities": num_entities,
                "collection_name": self._collection_name,
                "dimension": self._dim,
                "is_lite": self._is_lite
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "total_entities": 0,
                "collection_name": self._collection_name,
                "dimension": self._dim,
                "is_lite": self._is_lite,
                "error": str(e)
            }
