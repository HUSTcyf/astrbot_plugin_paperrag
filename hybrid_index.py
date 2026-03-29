"""
混合架构索引管理器 - 完善版本

参照 mnemosyne/milvus_manager.py 实现，避免与llama-index集成问题

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


class HybridIndexManager:
    """
    混合索引管理器（完善版）

    参照 mnemosyne/milvus_manager.py 实现：
    - 使用唯一的连接别名（避免与主进程冲突）
    - 延迟连接模式
    - 异步操作支持
    - 支持 Milvus Lite 和标准网络 URI
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
            hybrid_search: 是否启用混合检索
            lite_path: Milvus Lite 数据库文件路径
            uri: 标准 Milvus 连接 URI
            authentication: 认证信息 {"token": "..."} 或 {"user": "...", "password": "..."}
            db_name: 数据库名称
            alias: 连接别名（必须唯一）
        """
        # 向后兼容：milvus_uri 转为 lite_path
        # 仅当 lite_path 未提供（None）时才使用 milvus_uri
        if lite_path is None and milvus_uri:
            lite_path = milvus_uri
        # 如果 lite_path 是空字符串，不使用 milvus_uri（让管理器使用插件目录下的默认路径）

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

        # 文档统计追踪（用于解决 Milvus Lite 大数据量查询限制）
        self._doc_stats_file = None  # JSON 文件路径
        # file_name -> {chunk_count, added_time, chunk_index_start, chunk_index_end, milvus_id_start, milvus_id_end}
        self._doc_stats: Dict[str, Dict[str, Any]] = {}

        # BM25 内存索引（延迟构建）
        self._bm25: Any = None  # BM25Okapi 实例
        self._bm25_corpus_texts: List[str] = []
        self._bm25_corpus_metadata: List[Any] = []

        # 确定连接模式
        self._configure_connection_mode()

        # 初始化文档统计文件路径
        self._init_doc_stats()

        logger.info(f"✅ HybridIndexManager 初始化完成 (collection={collection_name}, dim={embed_dim}, alias={alias})")

    def _init_doc_stats(self):
        """初始化文档统计追踪文件"""
        # 使用与 Milvus 数据库相同的目录
        if self._lite_path:
            db_dir = os.path.dirname(self._lite_path)
        elif self._uri:
            db_dir = os.path.dirname(self._uri) or "."
        else:
            plugin_dir = Path(__file__).parent
            db_dir = str(plugin_dir / "data")

        self._doc_stats_file = os.path.join(db_dir, "paper_doc_stats.json")
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
        except Exception as e:
            logger.warning(f"⚠️ 加载文档统计失败: {e}")
            self._doc_stats = {}

    def _save_doc_stats(self):
        """保存文档统计到 JSON 文件"""
        if not self._doc_stats_file:
            return

        try:
            # 确保目录存在
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

                if file_name in self._doc_stats:
                    self._doc_stats[file_name]["chunk_count"] += 1
                else:
                    self._doc_stats[file_name] = {
                        "file_name": file_name,
                        "chunk_count": 1,
                        "added_time": added_time
                    }

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

        # 如果是目录，附加默认文件名
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

        self._connection_info["uri"] = self._lite_path

    def _configure_lite_default(self):
        """配置使用默认的 Milvus Lite 路径（相对于插件目录）"""
        self._is_lite = True
        # 使用插件目录作为基准路径
        plugin_dir = Path(__file__).parent
        default_path = plugin_dir / "data" / "milvus_papers.db"
        abs_path = str(default_path.resolve())

        logger.warning(f"使用默认 Milvus Lite 路径: '{abs_path}'")

        # 确保目录存在
        db_dir = default_path.parent
        if not db_dir.exists():
            os.makedirs(db_dir, exist_ok=True)

        self._connection_info["uri"] = str(default_path)

    def _configure_uri(self):
        """配置使用标准网络 URI 连接"""
        self._is_lite = False
        logger.info(f"配置标准 Milvus (别名: {self.alias}), URI: '{self._uri}'")
        self._connection_info["uri"] = self._uri

    def connect(self) -> None:
        """建立到 Milvus 的连接"""
        # 先尝试断开旧连接（如果存在）
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

            # 构建连接参数
            connect_params = dict(self._connection_info)

            # 添加认证信息
            if self.authentication.get("token"):
                connect_params["token"] = self.authentication["token"]
                logger.debug("使用 token 认证")
            elif self.authentication.get("user"):
                connect_params["user"] = self.authentication["user"]
                if self.authentication.get("password"):
                    connect_params["password"] = self.authentication["password"]
                logger.debug("使用用户名/密码认证")

            # 添加数据库名称（如果不是默认）
            if self.db_name != "default":
                connect_params["db_name"] = self.db_name
                logger.debug(f"连接到数据库: {self.db_name}")

            connections.connect(
                alias=self.alias,
                **connect_params
            )

            # 验证连接是否真的建立了
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

    @staticmethod
    async def _await_if_needed(result: Any) -> None:
        """
        等待异步操作完成（如果需要）

        Args:
            result: pymilvus 操作的返回值
        """
        if asyncio.iscoroutine(result):
            await result
        elif result is not None and hasattr(result, 'done'):
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
            else:
                # 检查连接是否真的有效
                if not connections.has_connection(self.alias):
                    self._is_connected = False
                    self.connect()

            # 再次验证连接
            if not connections.has_connection(self.alias):
                raise Exception(f"连接验证失败: {self.alias}")

            # 检查集合是否存在
            if utility.has_collection(self._collection_name, using=self.alias):
                logger.debug(f"集合 '{self._collection_name}' 已存在，加载")
                self._collection = Collection(self._collection_name, using=self.alias)
                collection = cast(Collection, self._collection)

                # load 是阻塞操作，使用线程池避免阻塞事件循环
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, collection.load)  # type: ignore[misc]
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

                # 创建索引和加载是阻塞操作，使用线程池
                collection = cast(Collection, self._collection)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(  # type: ignore[misc]
                    None,
                    lambda c=collection, ip=index_params: c.create_index(field_name="vector", index_params=ip)
                )
                await loop.run_in_executor(None, collection.load)  # type: ignore[misc]

                logger.info(f"✅ 集合 '{self._collection_name}' 创建成功 (索引: {index_type})")

        except Exception as e:
            logger.error(f"集合初始化失败: {e}")
            raise

    async def insert_nodes(self, nodes: List[Any], embeddings: List[List[float]]) -> int:
        """
        插入Nodes到Milvus

        Args:
            nodes: Node列表（具有 text 和 metadata 属性）
            embeddings: 对应的embedding列表

        Returns:
            插入的文档数量
        """
        import json

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
                # 其他类型尝试转换为列表或字符串
                try:
                    # 检查是否有 __iter__ 属性
                    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                        return [make_serializable(item) for item in obj]
                except Exception:
                    pass
                # 最后转换为字符串
                return str(obj)

        try:
            # 准备数据
            data = []

            for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
                metadata = node.metadata if hasattr(node, 'metadata') else {}

                # 确保 metadata 是可 JSON 序列化的
                if isinstance(metadata, dict):
                    metadata = make_serializable(metadata)
                    metadata_str = json.dumps(metadata, ensure_ascii=False)
                elif isinstance(metadata, str):
                    # 已经是字符串，尝试解析确保它是有效 JSON
                    try:
                        json.loads(metadata)  # 验证是否是有效 JSON
                        metadata_str = metadata
                    except json.JSONDecodeError:
                        # 不是有效 JSON，可能是 Python repr 格式，转换
                        try:
                            # 替换单引号为双引号（简单处理）
                            metadata_str = metadata.replace("'", '"')
                            json.loads(metadata_str)  # 再次验证
                        except Exception:
                            metadata_str = metadata  # 使用原字符串
                else:
                    metadata = make_serializable(metadata)
                    metadata_str = json.dumps(metadata, ensure_ascii=False)

                data.append({
                    "vector": embedding,
                    "text": node.text if hasattr(node, 'text') else str(node),
                    "metadata": metadata_str
                })

            # 插入数据 - 使用线程池避免阻塞事件循环
            collection = cast(Collection, self._collection)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: collection.insert(data))  # type: ignore[misc]
            await loop.run_in_executor(None, lambda: collection.flush())  # type: ignore[misc]

            logger.info(f"✅ 插入 {len(data)} 个 nodes")

            # 更新文档统计追踪
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
            # 构建搜索参数
            # 对于 HNSW 索引，使用较大的 ef 参数提高搜索精度
            search_params: Dict[str, Any] = {
                "metric_type": "COSINE",
                "params": {
                    "ef": max(top_k * 2, 64)
                }
            }

            # 对于 AUTOINDEX（Milvus Lite），参数可能不同
            if self._is_lite:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {}
                }

            # 执行搜索 - 使用线程池避免阻塞事件循环
            collection = cast(Collection, self._collection)
            loop = asyncio.get_event_loop()
            # pymilvus 类型标注与运行时行为不符，使用 cast(Any, ...) 避免类型检查错误
            # type: ignore[misc] run_in_executor 类型标注与实际返回值不符
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
                return []

            # 转换结果
            import json
            documents = []
            results_first: Any = cast(Any, results[0])
            for hit in results_first:
                metadata = hit.entity.get("metadata", {})
                # 如果 metadata 是字符串，尝试反序列化
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        # 尝试替换单引号为双引号后解析
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

    async def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            await self._ensure_collection()

            collection = cast(Collection, self._collection)

            # 获取实体数量
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

    async def get_all_references(self) -> Dict[str, Any]:
        """
        从数据库中提取所有参考文献，统计论文名称出现频次

        Returns:
            Dict containing:
            - references: List of unique reference titles with frequency
            - total_refs: Total number of references extracted
            - total_chunks: Total chunks processed
        """
        try:
            all_chunks = await self.get_all_chunks()
            if not all_chunks:
                return {
                    "references": [],
                    "total_refs": 0,
                    "total_chunks": 0
                }

            # 统计论文标题出现频次
            title_counter: Dict[str, Dict[str, Any]] = {}
            total_refs = 0

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

                    # 提取论文标题
                    title = ref.get("ref_title", "").strip()
                    if not title or len(title) < 5:  # 跳过太短的标题
                        continue

                    # 标准化标题用于比较（转小写，去除多余空格）
                    title_normalized = " ".join(title.lower().split())

                    if title_normalized in title_counter:
                        title_counter[title_normalized]["count"] += 1
                        title_counter[title_normalized]["raw_title"] = title  # 保留原始大小写
                    else:
                        title_counter[title_normalized] = {
                            "count": 1,
                            "raw_title": title,
                            "ref_authors": ref.get("ref_authors", ""),
                            "ref_year": ref.get("ref_year"),
                            "ref_doi": ref.get("ref_doi", "")
                        }
                    total_refs += 1

            # 转换为列表并按频次排序
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

            logger.info(f"📚 提取参考文献统计: {len(refs_list)} 种不同论文, 共 {total_refs} 条引用")

            return {
                "references": refs_list,
                "total_refs": total_refs,
                "total_chunks": len(all_chunks)
            }

        except Exception as e:
            logger.error(f"提取参考文献统计失败: {e}")
            return {
                "references": [],
                "total_refs": 0,
                "total_chunks": 0,
                "error": str(e)
            }

    async def get_all_chunks(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        从 Milvus 提取全量文本 chunks（用于生成评测数据集）

        使用按论文分组查询的方式，避免 Milvus Lite 全表扫描限制。

        Args:
            batch_size: 每批提取的数量（此参数已保留但不再用于全表分页）

        Returns:
            [{"text": str, "metadata": dict, "id": int}, ...]
        """
        import json

        try:
            await self._ensure_collection()
            collection = cast(Collection, self._collection)

            all_chunks = []
            loop = asyncio.get_event_loop()

            # 获取所有论文列表（使用追踪的统计数据，避免全表扫描）
            papers = await self.list_unique_documents()
            paper_names = [p.get("file_name", "") for p in papers if p.get("file_name")]

            logger.info(f"🔍 开始从 Milvus 提取全量 chunks ({len(paper_names)} 篇论文)...")

            # 按论文逐个查询，避免全表扫描超出 Milvus Lite 限制
            # 每隔20篇论文重新确保连接有效，避免连接超时
            for i, paper_name in enumerate(paper_names):
                if i % 20 == 0:
                    await self._ensure_collection()
                    collection = cast(Collection, self._collection)

                try:
                    raw_results: Any = await loop.run_in_executor(
                        None,
                        lambda pn=paper_name: collection.query(
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
                        # 解析 metadata JSON 字符串
                        meta = row.get("metadata", "{}")
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except Exception:
                                meta = {"raw": meta}
                        chunk["metadata"] = meta

                        # 提取关键字段到顶层
                        if isinstance(meta, dict):
                            chunk["file_name"] = meta.get("file_name", "")
                            chunk["paper_id"] = meta.get("paper_id", chunk["file_name"])

                        all_chunks.append(chunk)

                except Exception as e:
                    logger.warning(f"  查询论文 {paper_name} 时出错: {e}")
                    continue

                # 进度显示
                if (i + 1) % 20 == 0:
                    logger.debug(f"  已处理 {i + 1}/{len(paper_names)} 篇论文...")

            logger.info(f"✅ 共提取 {len(all_chunks)} 个 chunks")

            # 按 paper_id 分组统计
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
        """列出所有不同的文档（按文件分组）- 使用追踪的统计数据"""
        # 直接返回追踪的文档统计，避免 Milvus Lite 大数据量查询限制
        if self._doc_stats:
            logger.info(f"📊 返回追踪的文档统计: {len(self._doc_stats)} 个文件")
            return list(self._doc_stats.values())

        # 如果追踪统计为空但数据库有数据，需要重建统计
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
            删除结果 {"status": "success/error", "deleted_count": int, "message": str}
        """
        import json

        try:
            await self._ensure_collection()

            collection = cast(Collection, self._collection)

            # 构建查询表达式：查找 metadata["file_name"] 中匹配的文件名
            # Milvus JSON 字段使用 metadata["field_name"] 语法访问
            expr = f'metadata["file_name"] like "%{file_name}%"'

            # 提取所有匹配的实体 ID（分批处理，避免超过限制）
            all_ids_to_delete = []
            BATCH_SIZE = 5000  # 每批处理的 ID 数量
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

                # 提取实体 ID
                for hit in results:
                    entity_id = hit.get("id")
                    if entity_id is not None:
                        all_ids_to_delete.append(entity_id)

                # 如果返回数量少于批次大小，说明已经查询完毕
                if len(results) < BATCH_SIZE:
                    break

                # 安全限制
                if len(all_ids_to_delete) >= 100000:
                    logger.warning(f"⚠️ 文件 '{file_name}' 向量数量过大 ({len(all_ids_to_delete)})，已达安全限制")
                    break

            if not all_ids_to_delete:
                return {
                    "status": "success",
                    "deleted_count": 0,
                    "message": f"未找到文件 '{file_name}' 对应的向量数据"
                }

            # 分批删除（Milvus 对表达式长度有限制）
            DELETE_BATCH_SIZE = 1000  # 每批删除的 ID 数量
            total_deleted = 0

            for i in range(0, len(all_ids_to_delete), DELETE_BATCH_SIZE):
                batch_ids = all_ids_to_delete[i:i + DELETE_BATCH_SIZE]

                # 构建删除表达式
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

            # 更新文档统计追踪
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

            # drop_collection 是阻塞操作，在线程池中执行
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: utility.drop_collection(  # type: ignore[misc]
                self._collection_name, using=self.alias
            ))

            self._collection = None
            logger.info(f"✅ 集合 '{self._collection_name}' 已清空")

            # 清空文档统计追踪
            self._clear_doc_stats()

            return True

        except Exception as e:
            logger.error(f"清空索引失败: {e}")
            return False

    # ============================================================================
    # BM25 检索器
    # ============================================================================

    async def bm25_search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        BM25 关键词检索（基于 rank_bm25）

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            BM25 得分列表 [{"text", "metadata", "score"}, ...]
        """
        import json

        # 延迟导入，避免未安装时阻塞
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 未安装，请运行: pip install rank-bm25")
            return []

        # 构建或刷新内存索引
        await self._ensure_bm25_index()

        if self._bm25 is None:
            return []

        try:
            # 分词并计算 BM25 分数
            tokenized_query = query.split()
            scores = self._bm25.get_scores(tokenized_query)

            # 构建 (index, score) 对，排序取 top_k
            scored = list(enumerate(scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            results = []
            seen_texts: set = set()
            for idx, score in scored[:top_k]:
                if idx >= len(self._bm25_corpus_texts):
                    continue
                text = self._bm25_corpus_texts[idx]
                # 去重：相同文本只保留第一个
                if text in seen_texts:
                    continue
                seen_texts.add(text)

                metadata = self._bm25_corpus_metadata[idx]
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        try:
                            metadata = json.loads(metadata.replace("'", '"'))
                        except Exception:
                            metadata = {}
                results.append({
                    "text": text,
                    "metadata": metadata or {},
                    "score": float(score)
                })

            logger.debug(f"BM25 检索: query='{query}', 返回 {len(results)} 条结果")
            return results

        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []

    async def _ensure_bm25_index(self):
        """延迟构建 BM25 内存索引"""
        if self._bm25 is not None:
            return

        try:
            # 从 Milvus 提取全量文本
            chunks = await self.get_all_chunks(batch_size=1000)

            if not chunks:
                logger.warning("⚠️ BM25 索引构建失败：Milvus 中无数据")
                return

            self._bm25_corpus_texts = [c["text"] for c in chunks]
            self._bm25_corpus_metadata = [c.get("metadata", {}) for c in chunks]

            # 构建 BM25 索引
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [text.split() for text in self._bm25_corpus_texts]
            self._bm25 = BM25Okapi(tokenized_corpus)

            logger.info(f"✅ BM25 索引构建完成: {len(self._bm25_corpus_texts)} 个 chunks")

        except Exception as e:
            logger.error(f"BM25 索引构建失败: {e}")
            self._bm25 = None

    def refresh_bm25_index(self):
        """强制刷新 BM25 索引（如论文增删后）"""
        self._bm25 = None
        self._bm25_corpus_texts = []
        self._bm25_corpus_metadata = []
        logger.info("🔄 BM25 索引已重置，下次检索时将重新构建")

    def __del__(self):
        """析构函数，确保断开连接"""
        self.disconnect()
