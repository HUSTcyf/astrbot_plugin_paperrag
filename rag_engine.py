"""
Paper RAG Plugin - 核心RAG引擎模块
实现本地论文库的检索增强生成功能
"""

import asyncio
import json
import os
import sys
import time
import warnings
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, cast
from dataclasses import dataclass
from datetime import datetime

from astrbot.api import logger
from astrbot.core.provider.provider import EmbeddingProvider  # type: ignore

# 抑制底层库的 gRPC/absl 警告（这些是正常的，不影响功能）
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # 只显示 gRPC 错误
os.environ['GLOG_minloglevel'] = '2'     # 抑制 Google 日志

try:
    from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema
    from pymilvus.client.types import DataType as MilvusDataType  # 使用别名避免冲突
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError(f"请先安装必要依赖: pip install pymilvus PyMuPDF")

# 导入语义分块模块
try:
    from .semantic_chunker import PDFParserAdvanced, SemanticChunker, Chunk
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError as e:
    SEMANTIC_CHUNKER_AVAILABLE = False
    # 定义类型占位符，避免类型检查错误
    PDFParserAdvanced = None  # type: ignore
    SemanticChunker = None  # type: ignore
    Chunk = None  # type: ignore
    logger.warning(f"⚠️ 语义分块模块不可用，将使用基础分块方式: {e}")

# 可选的多格式支持
try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    logger.warning("⚠️ python-docx 未安装，Word文档支持将被禁用")


@dataclass
class RAGConfig:
    """RAG配置类"""
    # Provider配置
    embedding_provider_id: str = ""
    llm_provider_id: str = ""

    # Milvus配置 - 支持 Lite 模式和远程服务器
    milvus_lite_path: str = "./data/milvus_papers.db"  # Lite 模式路径
    address: str = ""  # 远程 Milvus 服务器地址
    db_name: str = "default"  # 数据库名称
    authentication: Optional[dict] = None  # 认证信息 {token, user, password}
    collection_name: str = "paper_embeddings"

    # 检索配置
    embed_dim: int = 768
    top_k: int = 5
    similarity_cutoff: float = 0.3

    # 论文目录
    papers_dir: str = "./papers"

    # 语义分块配置
    chunk_size: int = 512  # 目标块大小
    chunk_overlap: int = 0  # 重叠大小（默认0，避免无限循环bug）
    min_chunk_size: int = 100  # 最小块大小
    use_semantic_chunking: bool = True  # 是否使用语义分块

    # 多模态配置
    enable_multimodal: bool = True  # 是否启用多模态提取

    def __post_init__(self):
        """初始化后处理"""
        if self.authentication is None:
            self.authentication = {}

    def validate(self) -> tuple[bool, str]:
        """验证配置"""
        if self.embed_dim % 64 != 0:
            return False, "嵌入维度必须是64的倍数"

        # 检查 Milvus 配置
        if not self.milvus_lite_path and not self.address:
            # 两个都为空，使用默认 Lite 路径
            self.milvus_lite_path = "./data/milvus_papers.db"

        return True, ""

    def get_connection_mode(self) -> str:
        """获取连接模式：'lite' 或 'remote'"""
        # milvus_lite_path 优先级更高
        if self.milvus_lite_path and self.milvus_lite_path.strip():
            return 'lite'
        elif self.address and self.address.strip():
            return 'remote'
        else:
            # 默认使用 Lite 模式
            return 'lite'

    def get_connection_uri(self) -> str:
        """获取连接 URI"""
        mode = self.get_connection_mode()
        if mode == 'lite':
            return self.milvus_lite_path
        else:
            return self.address


class EmbeddingProviderWrapper:
    """AstrBot Embedding Provider 包装类 - 仅支持 OpenAI 和 Gemini"""

    def __init__(self, provider):
        logger.info(f"🔍 [WRAPPER STEP 1] EmbeddingProviderWrapper.__init__() 开始")
        if not provider:
            raise ValueError("Embedding provider 不能为 None")
        logger.info(f"🔍 [WRAPPER STEP 2] 保存 provider 引用")
        self.provider = provider
        logger.info(f"✅ [WRAPPER STEP 3] EmbeddingProviderWrapper 初始化完成")

    async def embed(self, texts: str | List[str]) -> List[List[float]]:
        logger.info(f"🔍 [EMBED STEP 1] embed() 方法被调用，texts 类型: {type(texts)}")

        try:
            if isinstance(texts, str):
                texts = [texts]
            logger.info(f"🔍 [EMBED STEP 2] 文本数量: {len(texts)}")

            embeddings = None

            # 辅助函数：智能调用（同步/异步自适应）
            async def call_method(method, *args):
                if inspect.iscoroutinefunction(method):
                    return await method(*args)
                else:
                    return method(*args)

            if hasattr(self.provider, "embed_texts") and callable(self.provider.embed_texts):
                logger.info(f"🔍 [EMBED STEP 4] 使用 embed_texts() 方法")
                embeddings = await call_method(self.provider.embed_texts, texts)
                
            elif hasattr(self.provider, "get_embedding") and callable(self.provider.get_embedding):
                logger.info(f"🔍 [EMBED STEP 4] 使用 get_embedding() 方法")
                embeddings = [await call_method(self.provider.get_embedding, text) for text in texts]
                
            elif hasattr(self.provider, "embed") and callable(self.provider.embed):
                logger.info(f"🔍 [EMBED STEP 4] 使用 embed() 方法")
                embeddings = await call_method(self.provider.embed, texts)
            else:
                available_methods = [
                    attr for attr in ['embed_texts', 'get_embedding', 'embed']
                    if hasattr(self.provider, attr) and callable(getattr(self.provider, attr))
                ]
                raise Exception(f"Provider 没有支持的 embedding 方法。可用方法: {available_methods}")

            if not embeddings:
                raise Exception("Embedding provider 返回空结果")

            return embeddings

        except Exception as e:
            logger.error(f"❌ 获取 embedding 失败: {e}")
            raise Exception(f"获取 embedding 失败: {e}")

    async def get_text_embedding(self, text: str) -> List[float]:
        result = await self.embed([text])
        return result[0] if result and len(result) > 0 else []

    async def get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入"""
        return await self.get_text_embedding(query)


class MilvusStore:
    """Milvus 向量存储封装（使用旧式 API，避免阻塞）"""

    def __init__(self, uri: str, collection_name: str, dim: int, db_name: str = "default", authentication: Optional[dict] = None):
        self.uri = uri
        self.collection_name = collection_name
        self.dim = dim
        self.db_name = db_name
        self.authentication = authentication or {}
        self._alias = f"paper_rag_{collection_name}"  # 唯一的连接别名
        self._is_connected = False
        self._collection_initialized = False

    def _ensure_connected(self):
        """确保已连接到 Milvus（延迟连接）"""
        if self._is_connected:
            return

        try:
            # 准备连接参数
            connect_params = {}

            # Lite 模式：使用文件路径
            if self.uri.endswith(".db"):
                import os
                # 确保目录存在
                os.makedirs(os.path.dirname(self.uri) or ".", exist_ok=True)
                connect_params["uri"] = self.uri
                logger.info(f"🔍 连接模式: Milvus Lite ({self.uri})")
            else:
                # 远程模式
                connect_params["uri"] = self.uri
                logger.info(f"🔍 连接模式: 远程 Milvus ({self.uri})")

            # 添加认证信息
            if self.authentication.get("token"):
                connect_params["token"] = self.authentication["token"]
            elif self.authentication.get("user"):
                connect_params["user"] = self.authentication["user"]
                if self.authentication.get("password"):
                    connect_params["password"] = self.authentication["password"]

            # 添加数据库名称
            if self.db_name and self.db_name != "default":
                connect_params["db_name"] = self.db_name

            # 建立连接（使用旧式 API）
            logger.info(f"🔍 调用 connections.connect(alias={self._alias})...")
            connections.connect(
                alias=self._alias,
                **connect_params
            )
            self._is_connected = True
            logger.info(f"✅ Milvus 连接成功")

        except Exception as e:
            logger.error(f"❌ Milvus 连接失败: {e}")
            raise

    async def _ensure_collection(self):
        """确保集合已初始化（延迟初始化）"""
        if self._collection_initialized:
            return

        # 确保已连接（同步方法）
        self._ensure_connected()

        try:
            # 检查集合是否存在
            from pymilvus import utility
            has_collection = utility.has_collection(self.collection_name, using=self._alias)

            if not has_collection:
                logger.info(f"🔍 创建新集合: {self.collection_name}")

                # 创建字段
                fields = [
                    FieldSchema(name="id", dtype=MilvusDataType.INT64, is_primary=True, auto_id=True, description="primary_id"),  # type: ignore[arg-type]
                    FieldSchema(name="vector", dtype=MilvusDataType.FLOAT_VECTOR, dim=self.dim, description="vector"),  # type: ignore[arg-type]
                    FieldSchema(name="text", dtype=MilvusDataType.VARCHAR, max_length=65535, description="text"),  # type: ignore[arg-type]
                    FieldSchema(name="metadata", dtype=MilvusDataType.JSON, description="metadata")  # type: ignore[arg-type]
                ]

                # 创建 schema
                schema = CollectionSchema(fields=fields, description="paper embeddings collection")

                # 创建集合
                collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using=self._alias
                )

                # 创建索引（根据模式选择类型）
                # Lite 模式只支持 FLAT、IVF_FLAT 或 AUTOINDEX
                is_lite_mode = self.uri.endswith(".db")

                if is_lite_mode:
                    # Lite 模式：使用 AUTOINDEX（自动选择最佳索引）
                    index_params = {
                        "index_type": "AUTOINDEX",
                        "metric_type": "COSINE"
                    }
                    logger.info(f"🔍 使用 AUTOINDEX 索引 (Lite 模式)")
                else:
                    # 远程模式：可以使用 HNSW
                    index_params = {
                        "index_type": "HNSW",
                        "metric_type": "COSINE",
                        "params": {"M": 8, "efConstruction": 64}
                    }
                    logger.info(f"🔍 使用 HNSW 索引 (远程模式)")

                # create_index 可能返回协程或 Future
                index_result = collection.create_index(
                    field_name="vector",
                    index_params=index_params
                )

                # 处理不同类型的返回值
                if asyncio.iscoroutine(index_result):
                    await index_result
                elif index_result is not None and hasattr(index_result, 'done'):
                    index_result.result()  # type: ignore[arg-type]

                load_result = collection.load()

                # load() 可能返回 Future 对象，需要等待
                if load_result is not None and hasattr(load_result, 'done'):
                    load_result.result()  # type: ignore[arg-type]

                logger.info(f"✅ 集合创建完成: {self.collection_name}")

            self._collection_initialized = True

        except Exception as e:
            logger.error(f"❌ 集合初始化失败: {e}")
            raise

    @property
    def client(self):
        """兼容性属性，返回别名"""
        self._ensure_connected()
        return self._alias

    async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """批量添加文档"""
        if not documents:
            return 0

        # 确保集合已初始化
        await self._ensure_collection()

        try:
            from pymilvus import Collection

            # 1. 获取集合
            collection = Collection(self.collection_name, using=self._alias)

            # 2. 【关键】验证 embedding 数据类型（调试用）
            for i, doc in enumerate(documents):
                if asyncio.iscoroutine(doc["embedding"]):
                    raise RuntimeError(f"❌ documents[{i}]['embedding'] 是 coroutine！上游代码忘记 await")
                if not isinstance(doc["embedding"], (list, tuple)):
                    raise TypeError(f"❌ documents[{i}]['embedding'] 类型错误：{type(doc['embedding'])}")

            # 3. 准备数据
            data = [
                {
                    "vector": doc["embedding"],  # 此时应该是 List[float]
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {})
                }
                for doc in documents
            ]

            # 4. 插入数据（同步方法）
            insert_result = collection.insert(data)
            logger.info(f"✅ 插入成功：{insert_result.succ_count}/{len(data)}")

            # 5. 刷新（同步方法）
            collection.flush()

            return len(documents)

        except Exception as e:
            logger.error(f"❌ 添加文档失败：{e}")
            logger.error(f"🔍 调试信息：documents[0]['embedding'] 类型 = {type(documents[0]['embedding']) if documents else 'N/A'}")
            return 0

    async def search(self, query_vector: List[float], top_k: int = 5,
                     filter_expr: str = '') -> List[Dict[str, Any]]:
        """向量搜索"""
        # 确保集合已初始化
        await self._ensure_collection()

        try:
            from pymilvus import Collection

            # 获取集合
            collection = Collection(self.collection_name, using=self._alias)

            # 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                output_fields=["text", "metadata"]
            )

            # search() 可能返回协程或 SearchFuture
            if asyncio.iscoroutine(results):
                results = await results
            elif results is not None and hasattr(results, 'done'):
                results = results.result()  # type: ignore[arg-type]

            # 确保结果不为空
            if results is None:
                return []

            # 格式化结果
            formatted_results = []
            for result in results[0]:  # type: ignore[index]  # 第一个查询的结果
                formatted_results.append({
                    "text": result.entity.get("text", ""),
                    "metadata": result.entity.get("metadata", {}),
                    "score": result.distance
                })

            return formatted_results
        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return []

    async def list_papers(self) -> List[Dict[str, Any]]:
        """列出所有论文"""
        logger.info(f"🔍 [LIST PAPERS STEP 1] list_papers() 开始")

        # 确保集合已初始化
        await self._ensure_collection()
        logger.info(f"🔍 [LIST PAPERS STEP 2] 集合初始化完成")

        try:
            from pymilvus import Collection

            # 获取集合
            collection = Collection(self.collection_name, using=self._alias)

            logger.info(f"🔍 [LIST PAPERS STEP 3] 执行查询")

            # 查询所有文档（必须指定 limit，即使使用空表达式）
            results = collection.query(
                expr="",  # 无过滤条件
                output_fields=["metadata"],
                limit=16384  # 设置一个足够大的限制
            )

            # query() 可能返回协程或 QueryFuture
            if asyncio.iscoroutine(results):
                results = await results
            elif results is not None and hasattr(results, 'done'):
                results = results.result()  # type: ignore[arg-type]

            logger.info(f"🔍 [LIST PAPERS STEP 4] 查询完成，结果数: {len(results)}")

            # 按文件名聚合
            papers = {}
            for result in results:
                metadata = result.entity.get("metadata", {})
                filename = metadata.get("file_name", "unknown")

                if filename not in papers:
                    papers[filename] = {
                        "file_name": filename,
                        "chunk_count": 0,
                        "added_time": metadata.get("added_time", "")
                    }
                papers[filename]["chunk_count"] += 1

            logger.info(f"🔍 [LIST PAPERS STEP 5] 处理完成，返回 {len(papers)} 个文档")
            return list(papers.values())
        except Exception as e:
            logger.error(f"❌ 列出论文失败: {e}")
            import traceback
            logger.error(f"❌ 错误详情: {traceback.format_exc()}")
            return []

    async def clear(self):
        """清空集合"""
        try:
            from pymilvus import utility

            # 确保已连接
            self._ensure_connected()

            # 检查集合是否存在
            if utility.has_collection(self.collection_name, using=self._alias):
                # 删除集合（同步操作，不需要await）
                _ = utility.drop_collection(self.collection_name, using=self._alias)
                logger.info(f"✅ 已清空集合: {self.collection_name}")
                self._collection_initialized = False  # 重置标志，下次使用时重新创建
            else:
                logger.warning(f"⚠️ 集合不存在: {self.collection_name}")

        except Exception as e:
            logger.error(f"❌ 清空集合失败: {e}")
            raise


class PaperRAGEngine:
    """论文RAG引擎（完全异步初始化模式）"""

    def __init__(self, config: RAGConfig, context):
        self.config = config
        self.context = context

        # 完全延迟初始化：所有组件都延迟到第一次使用时
        self._store: MilvusStore | None = None
        self._store_initialized = False

        self._embedding_provider = None
        self._embedding_wrapper: EmbeddingProviderWrapper | None = None
        self._embedding_initialized = False

        self._llm_provider = None
        self._llm_initialized = False

        self._initialization_failed = False
        self._initialization_error = None

    def _ensure_store_initialized(self):
        """确保 Milvus 存储已初始化（延迟初始化）"""
        if self._store_initialized:
            return

        try:
            # 获取连接 URI
            uri = self.config.get_connection_uri()

            # Lite 模式需要确保目录存在
            if uri.endswith(".db"):
                Path(uri).parent.mkdir(parents=True, exist_ok=True)

            self._store = MilvusStore(
                uri=uri,
                collection_name=self.config.collection_name,
                dim=self.config.embed_dim,
                db_name=self.config.db_name,
                authentication=self.config.authentication
            )
            self._store_initialized = True
            logger.info(f"✅ Milvus 存储初始化成功 ({self.config.get_connection_mode()} 模式)")
        except Exception as e:
            logger.error(f"❌ Milvus 存储初始化失败: {e}")
            self._initialization_failed = True
            self._initialization_error = str(e)
            raise

    def _ensure_embedding_initialized(self):
        """确保 Embedding Provider 已初始化（延迟初始化）"""
        if self._embedding_initialized:
            return

        logger.info(f"🔍 [STEP 1] 开始初始化 Embedding Provider: {self.config.embedding_provider_id}")

        if not self.config.embedding_provider_id:
            raise Exception("embedding_provider_id 未配置")

        try:
            logger.info(f"🔍 [STEP 2] 获取 provider_manager")
            provider_manager = getattr(self.context, "provider_manager", None)
            if provider_manager is None:
                raise Exception("无法访问 context.provider_manager")

            logger.info(f"🔍 [STEP 3] 获取 inst_map")
            inst_map = getattr(provider_manager, "inst_map", None)
            if not isinstance(inst_map, dict):
                raise Exception("inst_map 不是 dict")

            logger.info(f"🔍 [STEP 4] 从 inst_map 获取 Provider")
            provider = inst_map.get(self.config.embedding_provider_id)
            if provider is None:
                raise Exception(f"Provider '{self.config.embedding_provider_id}' 不存在")

            logger.info(f"🔍 [STEP 5] 创建 EmbeddingProviderWrapper")
            # 包装 Provider
            self._embedding_provider = provider

            logger.info(f"🔍 [STEP 6] EmbeddingProviderWrapper.__init__() 调用")
            self._embedding_wrapper = EmbeddingProviderWrapper(provider)
            self._embedding_initialized = True

            logger.info(f"✅ [STEP 7] Embedding Provider 初始化完成")

        except Exception as e:
            logger.error(f"❌ Embedding Provider 初始化失败: {e}")
            self._initialization_failed = True
            self._initialization_error = str(e)
            raise

    def _ensure_llm_initialized(self):
        """确保 LLM Provider 已初始化（延迟初始化）"""
        if self._llm_initialized or not self.config.llm_provider_id:
            return

        try:
            provider_manager = getattr(self.context, "provider_manager", None)
            inst_map = getattr(provider_manager, "inst_map", None)
            if isinstance(inst_map, dict):
                provider = inst_map.get(self.config.llm_provider_id)
                if provider:
                    self._llm_provider = provider
                    self._llm_initialized = True
        except Exception as e:
            logger.warning(f"⚠️ LLM Provider 初始化失败: {e}")

    @property
    def store(self) -> MilvusStore:
        """获取 Milvus 存储（延迟初始化）"""
        logger.info(f"🔍 [PROPERTY] 访问 self.store property")
        logger.info(f"🔍 [PROPERTY] 调用 _ensure_store_initialized()")
        self._ensure_store_initialized()
        assert self._store is not None
        logger.info(f"✅ [PROPERTY] 返回 Milvus store")
        return self._store

    @property
    def embedding(self) -> EmbeddingProviderWrapper:
        """获取 Embedding Wrapper（延迟初始化）"""
        logger.info(f"🔍 [PROPERTY] 访问 self.embedding property")
        logger.info(f"🔍 [PROPERTY] 调用 _ensure_embedding_initialized()")
        self._ensure_embedding_initialized()
        assert self._embedding_wrapper is not None
        logger.info(f"✅ [PROPERTY] 返回 embedding wrapper")
        return self._embedding_wrapper

    @property
    def llm(self):
        """获取 LLM Provider（延迟初始化）"""
        self._ensure_llm_initialized()
        return self._llm_provider

    async def ingest_papers(self, file_paths: List[str]):
        """导入论文到知识库（异步生成器，yield进度更新）

        Args:
            file_paths: 文件路径列表

        Yields:
            进度更新字典: {
                "current": 当前文件索引,
                "total": 总文件数,
                "filename": 文件名,
                "status": 状态 (parsing/embedding/storing/done/skipped/error),
                "chunks": 片段数（仅done状态）,
                "error": 错误信息（仅error状态）
            }
        """
        if not file_paths:
            return

        start_time = time.time()
        total_chunks = 0
        skipped_count = 0
        error_count = 0

        for idx, file_path in enumerate(file_paths, 1):
            try:
                file_path_obj = Path(file_path)
                filename = str(file_path_obj.name)
                file_ext = str(file_path_obj.suffix.lower())

                logger.info(f"📄 [{idx}/{len(file_paths)}] 处理文件: {filename} ({file_ext})")

                # 报告开始解析
                yield {
                    "current": idx,
                    "total": len(file_paths),
                    "filename": filename,
                    "status": "parsing"
                }

                # 解析文档（自动检测格式）
                chunks = await self._parse_document(file_path)
                logger.info(f"📄 [{idx}/{len(file_paths)}] {filename}: 解析完成，得到 {len(chunks)} 个片段")

                if not chunks:
                    skipped_count += 1
                    logger.warning(f"⏭️  [{idx}/{len(file_paths)}] {filename}: 跳过（无文本内容）")
                    yield {
                        "current": idx,
                        "total": len(file_paths),
                        "filename": filename,
                        "status": "skipped"
                    }
                    continue

                # 报告开始向量化
                yield {
                    "current": idx,
                    "total": len(file_paths),
                    "filename": filename,
                    "status": "embedding"
                }

                # 生成嵌入（带限流）
                documents = []
                for i, chunk in enumerate(chunks):
                    if i % 5 == 0:  # 每5个chunk暂停一下
                        await asyncio.sleep(0.5)

                    embedding = await self.embedding.get_text_embedding(chunk)
                    documents.append({
                        "text": chunk,
                        "embedding": embedding,
                        "metadata": {
                            "file_name": filename,
                            "file_path": file_path,
                            "chunk_index": i,
                            "added_time": datetime.now().isoformat()
                        }
                    })

                # 报告开始存储
                yield {
                    "current": idx,
                    "total": len(file_paths),
                    "filename": filename,
                    "status": "storing"
                }

                # 批量添加到向量库
                count = await self.store.add_documents(documents)
                total_chunks += count

                # 报告完成
                yield {
                    "current": idx,
                    "total": len(file_paths),
                    "filename": filename,
                    "status": "done",
                    "chunks": count
                }

                logger.info(f"✅ 导入: {filename} ({count} 个片段)")

            except Exception as e:
                error_count += 1
                logger.error(f"❌ 导入失败 {file_path}: {e}")
                yield {
                    "current": idx,
                    "total": len(file_paths),
                    "filename": str(Path(file_path).name),
                    "status": "error",
                    "error": str(e)
                }

        # 返回最终统计
        elapsed_time = time.time() - start_time

        logger.info(f"📊 导入统计: 总文件 {len(file_paths)}, 成功处理 {len(file_paths) - skipped_count - error_count}, 跳过 {skipped_count}, 错误 {error_count}, 总片段 {total_chunks}")

        yield {
            "status": "complete",
            "files_count": len(file_paths),
            "skipped_count": skipped_count,
            "error_count": error_count,
            "chunks_count": total_chunks,
            "elapsed_time": elapsed_time
        }

    def _detect_file_type(self, file_path: str) -> str:
        """检测文件类型"""
        suffix = Path(file_path).suffix.lower()
        type_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.txt': 'text',
            '.md': 'text',
            '.markdown': 'text',
            '.html': 'html',
            '.htm': 'html',
            '.rtf': 'rich_text',
            '.odt': 'odt',
        }
        return type_map.get(suffix, 'unknown')

    async def _parse_pdf(self, file_path: str) -> List[str]:
        """解析PDF文件（使用语义感知分块）"""
        try:
            logger.info(f"🔍 [PDF PARSE] 开始解析: {str(Path(file_path).name)}")

            # 优先使用语义分块器
            if SEMANTIC_CHUNKER_AVAILABLE and self.config.use_semantic_chunking:
                logger.info(f"🔍 [PDF PARSE] 使用语义感知分块器")
                parser = PDFParserAdvanced(  # type: ignore
                    enable_multimodal=self.config.enable_multimodal
                )
                chunker = SemanticChunker(  # type: ignore
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap,
                    min_chunk_size=self.config.min_chunk_size,
                    max_chunk_size=self.config.chunk_size * 2  # 最大块为目标大小的2倍
                )

                # 使用高级解析和分块
                chunks_dict = parser.parse_and_chunk(file_path, chunker)
                logger.info(f"✅ [PDF PARSE] {str(Path(file_path).name)}: 语义分块生成 {len(chunks_dict)} 个片段")

                # 提取纯文本列表（保持兼容性）
                text_chunks = [c["text"] for c in chunks_dict]

                # 警告：分块数量过多
                if len(text_chunks) > 50:
                    logger.warning(f"⚠️ [PDF PARSE] {str(Path(file_path).name)}: 生成了 {len(text_chunks)} 个分块（超过50个），可能会影响检索性能")

                return text_chunks

            else:
                # 回退到基础 PyMuPDF 解析
                reason = "语义分块已禁用" if not self.config.use_semantic_chunking else "语义分块不可用"
                logger.warning(f"⚠️ [PDF PARSE] 使用基础 PyMuPDF 解析（{reason}）")
                doc: fitz.Document = fitz.open(file_path)  # type: ignore
                text_chunks: List[str] = []

                total_pages = len(doc)
                logger.info(f"🔍 [PDF PARSE] 总页数: {total_pages}")

                pages_with_text = 0
                total_text_length = 0

                for page_num, page in enumerate(doc, 1):  # type: ignore[arg-type]
                    text: str = page.get_text()  # type: ignore
                    text_length = len(text.strip())

                    if text_length > 0:
                        pages_with_text += 1
                        total_text_length += text_length

                        # 使用配置的分块大小
                        for i in range(0, len(text), self.config.chunk_size):
                            chunk: str = text[i:i+self.config.chunk_size]
                            if len(chunk.strip()) > self.config.min_chunk_size:  # 过滤太短的片段
                                text_chunks.append(chunk.strip())

                doc.close()

                logger.info(f"✅ [PDF PARSE] {str(Path(file_path).name)}: {pages_with_text}/{total_pages} 页有文本, 总字符 {total_text_length}, 生成 {len(text_chunks)} 个片段")

                if len(text_chunks) == 0:
                    logger.warning(f"⚠️ [PDF PARSE] {str(Path(file_path).name)}: 没有提取到任何文本（可能是扫描版PDF）")
                elif len(text_chunks) > 50:
                    logger.warning(f"⚠️ [PDF PARSE] {str(Path(file_path).name)}: 生成了 {len(text_chunks)} 个分块（超过50个），可能会影响检索性能")

                return text_chunks

        except Exception as e:
            logger.error(f"❌ PDF解析失败 {file_path}: {e}")
            import traceback
            logger.error(f"❌ 错误堆栈: {traceback.format_exc()}")
            return []

    async def _parse_docx(self, file_path: str) -> List[str]:
        """解析Word文档（DOCX）"""
        if not DOCX_SUPPORT:
            logger.error(f"❌ Word文档支持未启用，请安装 python-docx")
            return []

        try:
            from docx import Document as DocxDocument
            doc: DocxDocument = DocxDocument(file_path)  # type: ignore
            text_chunks = []
            full_text = []

            # 提取段落文本
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())

            # 提取表格文本
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        full_text.append(row_text)

            combined_text = "\n".join(full_text)

            # 分块处理
            chunk_size = 512
            for i in range(0, len(combined_text), chunk_size):
                chunk = combined_text[i:i+chunk_size]
                if len(chunk.strip()) > 50:
                    text_chunks.append(chunk.strip())

            if len(text_chunks) > 50:
                logger.warning(f"⚠️ [DOCX PARSE] {str(Path(file_path).name)}: 生成了 {len(text_chunks)} 个分块（超过50个），可能会影响检索性能")

            return text_chunks

        except Exception as e:
            logger.error(f"❌ Word文档解析失败 {file_path}: {e}")
            # 返回空列表
            return []

    async def _parse_text(self, file_path: str) -> List[str]:
        """解析纯文本文件（TXT, MD等）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            text_chunks = []
            chunk_size = 512
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if len(chunk.strip()) > 50:
                    text_chunks.append(chunk.strip())

            if len(text_chunks) > 50:
                logger.warning(f"⚠️ [TEXT PARSE] {str(Path(file_path).name)}: 生成了 {len(text_chunks)} 个分块（超过50个），可能会影响检索性能")

            return text_chunks

        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    text = f.read()

                text_chunks = []
                chunk_size = 512
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    if len(chunk.strip()) > 50:
                        text_chunks.append(chunk.strip())

                if len(text_chunks) > 50:
                    logger.warning(f"⚠️ [TEXT PARSE] {str(Path(file_path).name)}: 生成了 {len(text_chunks)} 个分块（超过50个），可能会影响检索性能")

                return text_chunks
            except Exception as e:
                logger.error(f"❌ 文本文件解析失败 {file_path}: {e}")
                return []
        except Exception as e:
            logger.error(f"❌ 文本文件解析失败 {file_path}: {e}")
            return []

    async def _parse_document(self, file_path: str) -> List[str]:
        """通用文档解析器 - 自动检测文件类型并选择合适的解析方法"""
        file_type = self._detect_file_type(file_path)

        if file_type == 'pdf':
            return await self._parse_pdf(file_path)
        elif file_type == 'docx':
            return await self._parse_docx(file_path)
        elif file_type in ('text', 'html', 'markdown'):
            return await self._parse_text(file_path)
        else:
            logger.error(f"❌ 不支持的文件类型: {file_type}")
            return []

    async def search(self, query: str, mode: str = "rag") -> Dict[str, Any]:
        """搜索论文"""
        # 生成查询嵌入
        query_embedding = await self.embedding.get_query_embedding(query)

        # 检索相关文档
        results = await self.store.search(
            query_vector=query_embedding,
            top_k=self.config.top_k
        )

        if mode == "retrieve":
            return {
                "type": "retrieve",
                "sources": results
            }

        # RAG模式：生成回答
        if not self.llm:
            return {
                "type": "error",
                "message": "LLM未配置，无法生成回答"
            }

        # 构建上下文
        context = "\n\n".join([
            f"[来源{i+1}] {r['metadata'].get('file_name', 'unknown')}\n{r['text']}"
            for i, r in enumerate(results)
        ])

        # 生成回答
        prompt = f"""基于以下论文内容回答问题：

{context}

问题：{query}

请提供详细的回答，并引用相关文献。"""

        try:
            # 使用 AstrBot LLM Provider
            response = await self.llm._call(prompt)
            answer = response.get_content() if hasattr(response, 'get_content') else str(response)

            return {
                "type": "rag",
                "answer": answer,
                "sources": results
            }
        except Exception as e:
            return {
                "type": "error",
                "message": f"生成回答失败: {e}"
            }

    async def list_papers(self) -> List[Dict[str, Any]]:
        """列出所有论文"""
        return await self.store.list_papers()
