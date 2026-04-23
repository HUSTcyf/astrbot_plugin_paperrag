"""
混合架构索引管理器 - 移除 BM25 版本

使用 Unsloth BGE-M3 的稀疏权重替代 BM25，不再需要：
- rank_bm25
- jieba

特性：
1. 直接使用pymilvus进行向量存储
2. 异步操作支持
3. 唯一连接别名，避免与主进程冲突
4. 支持 Milvus Lite 和标准网络 URI
5. 延迟连接模式
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, cast

# 抑制底层库的 gRPC/absl 警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from pymilvus.exceptions import MilvusException

from astrbot.api import logger


_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _PLUGIN_ROOT / "data"


class HybridIndexManager:
    """
    混合索引管理器（移除 BM25 版本）

    使用 Unsloth BGE-M3 的稀疏权重替代 BM25 关键词检索。
    保留原有的向量检索功能。
    """

    def __init__(
        self,
        milvus_uri: str = "./data/milvus_papers.db",
        collection_name: str = "paper_embeddings",
        embed_dim: int = 768,
        hybrid_search: bool = False,
        lite_path: Optional[str] = None,
        uri: Optional[str] = None,
        authentication: Optional[dict] = None,
        db_name: str = "default",
        alias: str = "hybrid_index"
    ):
        """
        初始化混合索引管理器

        Args:
            milvus_uri: Milvus Lite 数据库文件路径（已弃用，用 lite_path）
            collection_name: 集合名称
            embed_dim: embedding维度
            hybrid_search: 是否启用混合检索（现在由 BGE-M3 稀疏权重处理）
            lite_path: Milvus Lite 数据库文件路径
            uri: 标准 Milvus 连接 URI
            authentication: 认证信息 {"token": "..."} 或 {"user": "...", "password": "..."}
            db_name: 数据库名称
            alias: 连接别名（必须唯一）
        """
        # 向后兼容：milvus_uri 转为 lite_path
        if lite_path is None and milvus_uri:
            lite_path = milvus_uri

        self.alias = alias
        self._collection_name = collection_name
        self._dim = embed_dim
        self._hybrid_search = hybrid_search
        self.authentication = authentication or {}
        self.db_name = db_name or "default"

        # 连接配置
        self._lite_path = self._prepare_lite_path(lite_path) if lite_path else None
        self._uri = uri
        self._is_lite = self._lite_path is not None

        # 连接状态
        self._is_connected = False
        self._connection_info = {}
        self._collection: Optional[Collection] = None

        # 文档统计追踪
        self._doc_stats_file = None
        self._doc_stats: Dict[str, Dict[str, Any]] = {}
        self._file_name_has_pdf_suffix: Optional[bool] = None

        # 确定连接模式
        self._configure_connection_mode()

        # 初始化文档统计文件路径
        self._init_doc_stats()

        logger.info(f"✅ HybridIndexManager 初始化完成 (collection={collection_name}, dim={embed_dim}, alias={alias})")

    def _init_doc_stats(self):
        """初始化文档统计追踪文件"""
        if self._lite_path:
            db_dir = os.path.dirname(self._lite_path)
        elif self._uri:
            db_dir = os.path.dirname(self._uri) or "."
        else:
            db_dir = str(_DEFAULT_DATA_DIR)

        db_name = os.path.basename(self._lite_path) if self._lite_path else "milvus.db"

        if "qasper" in db_name.lower():
            if "text" in db_name.lower():
                doc_stats_filename = "qasper_doc_stats_text.json"
            elif "vision" in db_name.lower():
                doc_stats_filename = "qasper_doc_stats_vision.json"
            else:
                doc_stats_filename = "qasper_doc_stats.json"
        else:
            doc_stats_filename = "paper_doc_stats.json"

        self._doc_stats_file = os.path.join(db_dir, doc_stats_filename)
        self._load_doc_stats()
        logger.info(f"📊 文档统计文件: {self._doc_stats_file}")

    def _load_doc_stats(self):
        """从 JSON 文件加载文档统计"""
        if not self._doc_stats_file or not os.path.exists(self._doc_stats_file):
            self._doc_stats = {}
            return

        try:
            with open(self._doc_stats_file, 'r', encoding='utf-8') as f:
                self._doc_stats = json.load(f)
            logger.info(f"📊 已加载文档统计: {len(self._doc_stats)} 个文件")

            if self._doc_stats:
                first_file_name = next(iter(self._doc_stats.values())).get("file_name", "")
                self._file_name_has_pdf_suffix = first_file_name.lower().endswith(".pdf")
                logger.info(f"📊 file_name {'带' if self._file_name_has_pdf_suffix else '不带'} .pdf 后缀")
        except Exception as e:
            logger.warning(f"⚠️ 加载文档统计失败: {e}")
            self._doc_stats = {}

    def _save_doc_stats(self):
        """保存文档统计到 JSON 文件"""
        if not self._doc_stats_file:
            return

        try:
            db_dir = os.path.dirname(self._doc_stats_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            with open(self._doc_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self._doc_stats, f, ensure_ascii=False, indent=2)
            logger.debug(f"📊 已保存文档统计: {len(self._doc_stats)} 个文件")
        except Exception as e:
            logger.error(f"❌ 保存文档统计失败: {e}")

    def _update_doc_stats_on_insert(self, nodes: List[Any]):
        """插入数据后更新文档统计"""
        for node in nodes:
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            if isinstance(metadata, dict):
                file_name = metadata.get("file_name", "unknown")
                added_time = metadata.get("added_time", "")
                github_url = metadata.get("github_url")

                if file_name in self._doc_stats:
                    self._doc_stats[file_name]["chunk_count"] += 1
                    if github_url and not self._doc_stats[file_name].get("github_url"):
                        self._doc_stats[file_name]["github_url"] = github_url
                else:
                    doc_stat = {
                        "file_name": file_name,
                        "chunk_count": 1,
                        "added_time": added_time
                    }
                    if github_url:
                        doc_stat["github_url"] = github_url
                    self._doc_stats[file_name] = doc_stat

        self._save_doc_stats()

    def _update_doc_stats_on_delete(self, file_name: str) -> int:
        """删除文件后更新文档统计，返回删除的 chunk 数量"""
        deleted_count = 0

        if file_name in self._doc_stats:
            deleted_count = self._doc_stats[file_name]["chunk_count"]
            del self._doc_stats[file_name]
            self._save_doc_stats()
            logger.info(f"📊 已从统计中删除文件: {file_name} ({deleted_count} chunks)")

        return deleted_count

    def _clear_doc_stats(self):
        """清空文档统计"""
        self._doc_stats = {}
        self._save_doc_stats()
        logger.info("📊 已清空文档统计")

    def _prepare_lite_path(self, path_input: str) -> str:
        """准备 Milvus Lite 路径"""
        path = Path(path_input)
        if not path.is_absolute():
            path = path.resolve()

        if path.is_dir() or (not path.exists() and not str(path).endswith('.db')):
            path = path / "milvus_papers.db"

        return str(path)

    def _configure_connection_mode(self):
        """配置连接模式"""
        if self._lite_path:
            self._configure_lite()
        elif self._uri:
            self._configure_uri()
        else:
            self._configure_lite_default()

    def _configure_lite(self):
        """配置使用 Milvus Lite"""
        self._is_lite = True
        abs_path = os.path.abspath(self._lite_path) if self._lite_path else "None"
        logger.info(f"配置 Milvus Lite (别名: {self.alias}), 路径: '{abs_path}'")

        if self._lite_path:
            db_dir = os.path.dirname(self._lite_path)
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                    logger.info(f"为 Milvus Lite 创建了目录: '{db_dir}'")
                except OSError as e:
                    logger.error(f"无法为 Milvus Lite 创建目录 '{db_dir}': {e}")
                    raise

            self._prepare_lite_database_path(self._lite_path)

        self._connection_info["uri"] = self._lite_path

    def _configure_lite_default(self):
        """配置使用默认的 Milvus Lite 路径"""
        self._is_lite = True
        default_path = _DEFAULT_DATA_DIR / "milvus_papers.db"
        abs_path = str(default_path.resolve())

        logger.warning(f"使用默认 Milvus Lite 路径: '{abs_path}'")

        db_dir = default_path.parent
        if not db_dir.exists():
            os.makedirs(db_dir, exist_ok=True)

        self._prepare_lite_database_path(str(default_path))
        self._connection_info["uri"] = str(default_path)

    def _prepare_lite_database_path(self, db_path: str) -> None:
        """只准备 Milvus Lite 的目录与路径，不预创建数据库文件。"""
        path = Path(db_path).expanduser()
        db_dir = path.parent

        if not db_dir.exists():
            raise PermissionError(f"Milvus Lite 数据库目录不存在: {db_dir}")
        if not os.access(db_dir, os.W_OK | os.X_OK):
            raise PermissionError(f"Milvus Lite 数据库目录不可写: {db_dir}")

        if path.exists():
            if path.is_dir():
                raise PermissionError(f"Milvus Lite 数据库路径被目录占用: {path}")
            if path.stat().st_size == 0:
                logger.warning(f"Milvus Lite 检测到空数据库文件，删除后重建: {path}")
                try:
                    path.unlink()
                except OSError as e:
                    raise PermissionError(f"无法删除空数据库文件 {path}: {e}") from e
                return
            if not os.access(path, os.W_OK):
                raise PermissionError(f"Milvus Lite 数据库文件不可写: {path}")
            return

    def _configure_uri(self):
        """配置使用标准网络 URI 连接"""
        self._is_lite = False
        logger.info(f"配置标准 Milvus (别名: {self.alias}), URI: '{self._uri}'")
        self._connection_info["uri"] = self._uri

    def connect(self) -> None:
        """建立到 Milvus 的连接"""
        try:
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)
        except Exception:
            pass

        if self._is_connected:
            logger.debug(f"已连接到 Milvus (别名: {self.alias})")
            return

        mode = "Milvus Lite" if self._is_lite else "Standard Milvus"

        try:
            logger.debug(f"尝试连接到 {mode} (别名: {self.alias})")

            connect_params = dict(self._connection_info)

            if self.authentication.get("token"):
                connect_params["token"] = self.authentication["token"]
            elif self.authentication.get("user"):
                connect_params["user"] = self.authentication["user"]
                if self.authentication.get("password"):
                    connect_params["password"] = self.authentication["password"]

            if self.db_name != "default":
                connect_params["db_name"] = self.db_name

            connections.connect(alias=self.alias, **connect_params)

            if connections.has_connection(self.alias):
                self._is_connected = True
                logger.info(f"✅ 成功连接到 {mode} (别名: {self.alias})")
            else:
                self._is_connected = False
                raise Exception(f"连接验证失败: {self.alias}")

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

    async def _ensure_collection(self):
        """确保集合已创建"""
        if self._collection is not None:
            return

        try:
            if not self._is_connected:
                self.connect()
            else:
                if not connections.has_connection(self.alias):
                    self._is_connected = False
                    self.connect()

            if not connections.has_connection(self.alias):
                raise Exception(f"连接验证失败: {self.alias}")

            if utility.has_collection(self._collection_name, using=self.alias):
                logger.debug(f"集合 '{self._collection_name}' 已存在，加载")
                self._collection = Collection(self._collection_name, using=self.alias)
                collection = cast(Collection, self._collection)

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, collection.load)
            else:
                logger.debug(f"创建新集合 '{self._collection_name}'")

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

                is_lite = self._is_lite
                index_type = "AUTOINDEX" if is_lite else "HNSW"
                index_params: Dict[str, Any] = {
                    "index_type": index_type,
                    "metric_type": "COSINE"
                }

                if not is_lite:
                    index_params["params"] = {"M": 8, "efConstruction": 64}

                collection = cast(Collection, self._collection)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda c=collection, ip=index_params: c.create_index(field_name="vector", index_params=ip)
                )
                await loop.run_in_executor(None, collection.load)

                logger.info(f"✅ 集合 '{self._collection_name}' 创建成功 (索引: {index_type})")

        except Exception as e:
            if self._is_lite:
                logger.error(
                    f"集合初始化失败: {e} (Milvus Lite 路径: {self._connection_info.get('uri')})"
                )
            else:
                logger.error(f"集合初始化失败: {e}")
            raise

    async def insert_nodes(self, nodes: List[Any], embeddings: List[List[float]]) -> int:
        """
        插入Nodes到Milvus

        Args:
            nodes: Node列表
            embeddings: 对应的embedding列表

        Returns:
            插入的文档数量
        """
        await self._ensure_collection()

        if len(nodes) != len(embeddings):
            raise ValueError(f"Nodes数量({len(nodes)})与embeddings数量({len(embeddings)})不匹配")

        def make_serializable(obj):
            """递归转换对象为可 JSON 序列化的格式"""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                try:
                    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                        return [make_serializable(item) for item in obj]
                except Exception:
                    pass
                return str(obj)

        try:
            data = []

            for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
                metadata = node.metadata if hasattr(node, 'metadata') else {}

                if isinstance(metadata, dict):
                    metadata = make_serializable(metadata)
                    metadata_str = json.dumps(metadata, ensure_ascii=False)
                elif isinstance(metadata, str):
                    try:
                        json.loads(metadata)
                        metadata_str = metadata
                    except json.JSONDecodeError:
                        try:
                            metadata_str = metadata.replace("'", '"')
                            json.loads(metadata_str)
                        except Exception:
                            metadata_str = metadata
                else:
                    metadata = make_serializable(metadata)
                    metadata_str = json.dumps(metadata, ensure_ascii=False)

                data.append({
                    "vector": embedding,
                    "text": node.text if hasattr(node, 'text') else str(node),
                    "metadata": metadata_str
                })

            total_text_size = sum(len(d["text"]) for d in data)
            total_metadata_size = sum(len(d["metadata"]) for d in data)
            total_vector_size = sum(len(d["vector"]) * 4 for d in data)

            logger.info(f"📊 Chunks 统计: {len(data)} 个, "
                       f"文本 {total_text_size / 1024:.1f}KB, "
                       f"元数据 {total_metadata_size / 1024:.1f}KB, "
                       f"向量 {total_vector_size / 1024:.1f}KB")

            collection = cast(Collection, self._collection)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: collection.insert(data))
            await loop.run_in_executor(None, lambda: collection.flush())

            logger.info(f"✅ 插入 {len(data)} 个 nodes")

            self._update_doc_stats_on_insert(nodes)

            return len(data)

        except Exception as e:
            logger.error(f"插入文档失败: {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        await self._ensure_collection()

        try:
            search_params: Dict[str, Any] = {
                "metric_type": "COSINE",
                "params": {
                    "ef": max(top_k * 2, 64)
                }
            }

            if self._is_lite:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {}
                }

            collection = cast(Collection, self._collection)
            loop = asyncio.get_event_loop()
            raw_results = await loop.run_in_executor(
                None,
                lambda: collection.search(
                    data=[query_embedding],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["text", "metadata"]
                )
            )
            results: Any = cast(Any, raw_results)

            if results is None:
                return []

            documents = []
            results_first: Any = cast(Any, results[0])
            for hit in results_first:
                metadata = hit.entity.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        try:
                            metadata = json.loads(metadata.replace("'", '"'))
                        except Exception:
                            metadata = {}
                documents.append({
                    "text": hit.entity.get("text"),
                    "metadata": metadata or {},
                    "score": float(hit.score)
                })

            return documents

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise

    async def search_with_paper_filter(
        self,
        query_embedding: List[float],
        paper_ids: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        在指定论文范围内进行向量搜索

        Args:
            query_embedding: 查询向量
            paper_ids: 论文 ID 列表
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        await self._ensure_collection()

        try:
            if not paper_ids:
                return []

            paper_conditions = [f'metadata["file_name"] == "{pid}"' for pid in paper_ids]
            if len(paper_conditions) == 1:
                filter_expr = paper_conditions[0]
            else:
                filter_expr = " || ".join(paper_conditions)

            search_params: Dict[str, Any] = {
                "metric_type": "COSINE",
                "params": {}
            }

            collection = cast(Collection, self._collection)
            loop = asyncio.get_event_loop()

            raw_results = await loop.run_in_executor(
                None,
                lambda: collection.search(
                    data=[query_embedding],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=["text", "metadata"]
                )
            )

            results: Any = cast(Any, raw_results)
            if results is None or len(results) == 0:
                return []

            documents = []
            results_first: Any = cast(Any, results[0])
            for hit in results_first:
                metadata = hit.entity.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}

                documents.append({
                    "text": hit.entity.get("text"),
                    "metadata": metadata or {},
                    "score": float(hit.score),
                    "paper_id": metadata.get("file_name", "")
                })

            return documents

        except Exception as e:
            logger.error(f"带论文过滤的搜索失败: {e}")
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            await self._ensure_collection()

            collection = cast(Collection, self._collection)
            num_entities = collection.num_entities

            return {
                "status": "initialized",
                "collection_name": self._collection_name,
                "total_nodes": num_entities,
                "embed_dim": self._dim,
                "hybrid_search_enabled": self._hybrid_search,
                "is_lite": self._is_lite
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        从 Milvus 提取全量文本 chunks

        Returns:
            [{"text": str, "metadata": dict, "id": int}, ...]
        """
        try:
            await self._ensure_collection()
            collection = cast(Collection, self._collection)

            all_chunks = []
            loop = asyncio.get_event_loop()

            papers = await self.list_unique_documents()
            paper_names = [p.get("file_name", "") for p in papers if p.get("file_name")]

            logger.info(f"🔍 开始从 Milvus 提取全量 chunks ({len(paper_names)} 篇论文)...")

            for i, paper_name in enumerate(paper_names):
                if i % 20 == 0:
                    await self._ensure_collection()
                    collection = cast(Collection, self._collection)

                try:
                    query_name = paper_name
                    if self._file_name_has_pdf_suffix is not None:
                        if self._file_name_has_pdf_suffix and not query_name.lower().endswith('.pdf'):
                            query_name = query_name + ".pdf"
                        elif not self._file_name_has_pdf_suffix and query_name.lower().endswith('.pdf'):
                            query_name = query_name[:-4]

                    raw_results: Any = await loop.run_in_executor(
                        None,
                        lambda pn=query_name: collection.query(
                            expr=f'metadata["file_name"] == "{pn}"',
                            output_fields=["id", "text", "metadata"],
                        )
                    )
                    raw_results = cast(List[Dict[str, Any]], raw_results)

                    for row in raw_results:
                        chunk = {
                            "id": row.get("id"),
                            "text": row.get("text", ""),
                        }
                        meta = row.get("metadata", "{}")
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except Exception:
                                meta = {"raw": meta}
                        chunk["metadata"] = meta

                        if isinstance(meta, dict):
                            chunk["file_name"] = meta.get("file_name", "")
                            chunk["paper_id"] = meta.get("paper_id", chunk["file_name"])

                        all_chunks.append(chunk)

                except Exception as e:
                    logger.warning(f"  查询论文 {paper_name} 时出错: {e}")
                    continue

                if (i + 1) % 10 == 0:
                    logger.info(f"  已处理 {i + 1}/{len(paper_names)} 篇论文...")

            logger.info(f"✅ 共提取 {len(all_chunks)} 个 chunks")

            paper_counts: Dict[str, int] = {}
            for c in all_chunks:
                pid = c.get("paper_id", "unknown")
                paper_counts[pid] = paper_counts.get(pid, 0) + 1
            logger.info(f"📊 涉及 {len(paper_counts)} 篇论文")

            return all_chunks

        except Exception as e:
            logger.error(f"提取 chunks 失败: {e}")
            raise

    async def list_unique_documents(self) -> List[Dict[str, Any]]:
        """列出所有不同的文档"""
        if self._doc_stats:
            logger.info(f"📊 返回追踪的文档统计: {len(self._doc_stats)} 个文件")
            return list(self._doc_stats.values())

        logger.info("📊 追踪统计为空，正在检查数据库...")
        try:
            await self._ensure_collection()
            collection = cast(Collection, self._collection)
            total_entities = collection.num_entities

            if total_entities > 0:
                logger.warning("⚠️ 检测到数据库有数据但追踪统计为空")
                logger.warning("   建议：使用 /paper rebuild 重建索引以恢复完整统计")
        except Exception:
            pass

        return []

    async def delete_by_file_name(self, file_name: str) -> Dict[str, Any]:
        """
        根据文件名删除向量数据

        Args:
            file_name: 要删除的文件名

        Returns:
            删除结果
        """
        try:
            await self._ensure_collection()

            collection = cast(Collection, self._collection)

            file_name_escaped = file_name.replace('"', '\\"').replace('%', '\\%')
            expr = f'metadata["file_name"] like "%{file_name_escaped}%"'

            all_ids_to_delete = []
            BATCH_SIZE = 5000
            loop = asyncio.get_event_loop()

            while True:
                raw_results = await loop.run_in_executor(
                    None,
                    lambda: collection.query(
                        expr=expr,
                        output_fields=["id"],
                        limit=BATCH_SIZE
                    )
                )
                results: Any = cast(Any, raw_results)

                if not results:
                    break

                for hit in results:
                    entity_id = hit.get("id")
                    if entity_id is not None:
                        all_ids_to_delete.append(entity_id)

                if len(results) < BATCH_SIZE:
                    break

                if len(all_ids_to_delete) >= 100000:
                    logger.warning(f"⚠️ 文件 '{file_name}' 向量数量过大 ({len(all_ids_to_delete)})，已达安全限制")
                    break

            if not all_ids_to_delete:
                return {
                    "status": "success",
                    "deleted_count": 0,
                    "message": f"未找到文件 '{file_name}' 对应的向量数据"
                }

            DELETE_BATCH_SIZE = 1000
            total_deleted = 0

            for i in range(0, len(all_ids_to_delete), DELETE_BATCH_SIZE):
                batch_ids = all_ids_to_delete[i:i + DELETE_BATCH_SIZE]

                if len(batch_ids) == 1:
                    delete_expr = f"id == {batch_ids[0]}"
                else:
                    ids_str = ", ".join(str(id_) for id_ in batch_ids)
                    delete_expr = f"id in [{ids_str}]"

                await loop.run_in_executor(
                    None,
                    lambda de=delete_expr: collection.delete(de)
                )

                total_deleted += len(batch_ids)

            await loop.run_in_executor(None, lambda: collection.flush())

            logger.info(f"✅ 删除文件 '{file_name}': {total_deleted} 个向量")

            self._update_doc_stats_on_delete(file_name)

            return {
                "status": "success",
                "deleted_count": total_deleted,
                "message": f"已删除文件 '{file_name}' 的 {total_deleted} 个向量数据"
            }

        except Exception as e:
            logger.error(f"删除文件 '{file_name}' 失败: {e}")
            return {
                "status": "error",
                "deleted_count": 0,
                "message": f"删除失败: {e}"
            }

    async def clear(self) -> bool:
        """清空索引"""
        try:
            await self._ensure_collection()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: utility.drop_collection(
                self._collection_name, using=self.alias
            ))

            self._collection = None
            logger.info(f"✅ 集合 '{self._collection_name}' 已清空")

            self._clear_doc_stats()

            return True

        except Exception as e:
            logger.error(f"清空索引失败: {e}")
            return False

    async def get_all_references(self, allow_duplicates: bool = True) -> Dict[str, Any]:
        """
        从数据库中提取所有参考文献，统计论文名称出现频次

        Args:
            allow_duplicates: 是否允许同一篇论文对同一参考文献重复统计

        Returns:
            {"references": [...], "total_refs": int, "total_chunks": int}
        """
        try:
            all_chunks = await self.get_all_chunks()
            if not all_chunks:
                return {"references": [], "total_refs": 0, "total_chunks": 0}

            title_counter: Dict[str, Dict[str, Any]] = {}
            total_refs = 0

            if allow_duplicates:
                for chunk in all_chunks:
                    metadata = chunk.get("metadata", {})
                    if not isinstance(metadata, dict):
                        continue
                    references = metadata.get("cited_references", [])
                    if not references:
                        continue
                    for ref in references:
                        if not isinstance(ref, dict):
                            continue
                        title = ref.get("ref_title", "").strip()
                        if not title or len(title) < 5:
                            continue
                        title_norm = " ".join(title.lower().split())
                        if title_norm in title_counter:
                            title_counter[title_norm]["count"] += 1
                        else:
                            title_counter[title_norm] = {
                                "count": 1,
                                "raw_title": title,
                                "ref_authors": ref.get("ref_authors", ""),
                                "ref_year": ref.get("ref_year"),
                                "ref_doi": ref.get("ref_doi", "")
                            }
                        total_refs += 1
            else:
                title_to_papers: Dict[str, set] = {}
                for chunk in all_chunks:
                    metadata = chunk.get("metadata", {})
                    if not isinstance(metadata, dict):
                        continue
                    citing_paper = metadata.get("file_name", "") or metadata.get("paper_id", "")
                    references = metadata.get("cited_references", [])
                    if not references:
                        continue
                    for ref in references:
                        if not isinstance(ref, dict):
                            continue
                        title = ref.get("ref_title", "").strip()
                        if not title or len(title) < 5:
                            continue
                        title_norm = " ".join(title.lower().split())
                        if title_norm not in title_counter:
                            title_counter[title_norm] = {
                                "count": 0,
                                "raw_title": title,
                                "ref_authors": ref.get("ref_authors", ""),
                                "ref_year": ref.get("ref_year"),
                                "ref_doi": ref.get("ref_doi", "")
                            }
                            title_to_papers[title_norm] = set()
                        if citing_paper and citing_paper in title_to_papers[title_norm]:
                            continue
                        title_counter[title_norm]["count"] += 1
                        title_to_papers[title_norm].add(citing_paper)
                        total_refs += 1

            refs_list = []
            for title_norm, info in title_counter.items():
                refs_list.append({
                    "title": info["raw_title"],
                    "count": info["count"],
                    "authors": info["ref_authors"],
                    "year": info["ref_year"],
                    "doi": info["ref_doi"]
                })
            refs_list.sort(key=lambda x: x["count"], reverse=True)

            return {
                "references": refs_list,
                "total_refs": total_refs,
                "total_chunks": len(all_chunks)
            }
        except Exception as e:
            logger.error(f"提取参考文献统计失败: {e}")
            return {"references": [], "total_refs": 0, "total_chunks": 0, "error": str(e)}

    async def get_papers_with_zero_references(self) -> Dict[str, Any]:
        """
        获取参考文献数量为0的论文列表

        Returns:
            {"papers": [...], "total_papers": int, "total_zero_ref": int}
        """
        try:
            import json
            await self._ensure_collection()
            collection = cast(Collection, self._collection)
            loop = asyncio.get_event_loop()

            papers = await self.list_unique_documents()
            paper_names = [p.get("file_name", "") for p in papers if p.get("file_name")]

            logger.info(f"🔍 开始检查 {len(paper_names)} 篇论文的参考文献数量...")

            zero_ref_papers = []
            checked_count = 0

            for i, paper_name in enumerate(paper_names):
                if i % 20 == 0:
                    await self._ensure_collection()
                    collection = cast(Collection, self._collection)

                try:
                    query_name = paper_name
                    if self._file_name_has_pdf_suffix is not None:
                        if self._file_name_has_pdf_suffix and not query_name.lower().endswith('.pdf'):
                            query_name = query_name + ".pdf"
                        elif not self._file_name_has_pdf_suffix and query_name.lower().endswith('.pdf'):
                            query_name = query_name[:-4]

                    all_results: Any = await loop.run_in_executor(
                        None,
                        lambda pn=query_name: collection.query(
                            expr=f'metadata["file_name"] == "{pn}"',
                            output_fields=["metadata"],
                            limit=16384
                        )
                    )
                    all_results = cast(List[Dict[str, Any]], all_results)

                    if not all_results:
                        zero_ref_papers.append({"file_name": paper_name, "chunk_count": 0})
                        checked_count += 1
                        continue

                    has_any_ref = False
                    chunks_with_refs = 0
                    chunks_without_field = 0
                    for row in all_results:
                        meta = row.get("metadata", "{}")
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except json.JSONDecodeError:
                                meta = {}
                        if "cited_references" in meta:
                            cited_refs = meta["cited_references"]
                            if isinstance(cited_refs, list) and len(cited_refs) > 0:
                                has_any_ref = True
                                chunks_with_refs += 1
                        else:
                            chunks_without_field += 1

                    if not has_any_ref:
                        zero_ref_papers.append({
                            "file_name": paper_name,
                            "chunk_count": len(all_results)
                        })
                    checked_count += 1

                except Exception as e:
                    logger.warning(f"⚠️ 检查论文 {paper_name} 失败: {e}")
                    continue

            logger.info(f"📊 检查完成: {checked_count} 篇论文, {len(zero_ref_papers)} 篇无参考文献")
            return {
                "papers": zero_ref_papers,
                "total_papers": checked_count,
                "total_zero_ref": len(zero_ref_papers)
            }
        except Exception as e:
            logger.error(f"获取零引用论文失败: {e}")
            return {"papers": [], "total_papers": 0, "total_zero_ref": 0, "error": str(e)}

    def __del__(self):
        """析构函数，确保断开连接"""
        self.disconnect()
