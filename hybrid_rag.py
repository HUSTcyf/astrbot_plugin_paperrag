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
    3. 返回融合排序后的 top_k 结果
    """

    def __init__(
        self,
        index_manager: HybridIndexManager,
        embed_provider: Any,
        bm25_top_k: int = 20,
        alpha: float = 0.5,
        rrf_k: int = 60
    ):
        super().__init__(index_manager, embed_provider)
        self._bm25_top_k = bm25_top_k
        self._alpha = alpha
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, top_k: int = 5) -> QueryResult:
        """混合检索：BM25 + 向量 + RRF 融合"""
        try:
            # 1. 向量语义搜索
            query_embedding = await self._embed_provider.get_text_embedding(query)
            vector_results = await self._index_manager.search(
                query_embedding=query_embedding,
                top_k=top_k
            )

            # 2. BM25 关键词搜索
            bm25_results = await self._index_manager.bm25_search(
                query=query,
                top_k=self._bm25_top_k
            )

            # 3. RRF 融合
            fused = self._rrf_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                top_k=top_k
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

    def _rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) 分数融合

        RRF score = alpha / (rank + k) + (1-alpha) / (rank + k)

        Args:
            vector_results: 向量检索结果 [{"text", "metadata", "score"}, ...]
            bm25_results: BM25 检索结果 [{"text", "metadata", "score"}, ...]
            top_k: 返回结果数量
            alpha: 融合权重（0=纯BM25, 1=纯向量）

        Returns:
            融合排序后的结果列表
        """
        # 构建 text -> vector_rank 映射
        vector_rank: Dict[str, int] = {}
        for i, item in enumerate(vector_results):
            vector_rank[item["text"]] = i + 1

        # 构建 text -> bm25_rank 映射
        bm25_rank: Dict[str, int] = {}
        for i, item in enumerate(bm25_results):
            bm25_rank[item["text"]] = i + 1

        # 合并所有文本
        all_texts = set(vector_rank.keys()) | set(bm25_rank.keys())

        # 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        for text in all_texts:
            v_rank = vector_rank.get(text, len(vector_results) + 1)
            b_rank = bm25_rank.get(text, len(bm25_results) + 1)

            v_score = self._alpha / (v_rank + self._rrf_k)
            b_score = (1 - self._alpha) / (b_rank + self._rrf_k)
            rrf_scores[text] = v_score + b_score

        # 按 RRF 分数降序排列
        sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

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
        """确保LLM Provider已初始化"""
        if self._llm_initialized:
            assert self._llm_client is not None
            return self._llm_client

        # 优先使用配置的 provider_id
        provider_id = self.config.text_provider_id
        if not provider_id:
            # 回退到获取当前正在使用的 provider
            try:
                if self.context is not None:
                    self._llm_client = self.context.get_using_provider()
                    if self._llm_client:
                        logger.info("✅ 使用当前会话的 LLM Provider")
                        self._llm_initialized = True
                        return self._llm_client
            except Exception as e:
                logger.warning(f"⚠️ 获取当前Provider失败: {e}")

            # 如果 context 为空或获取 provider 失败，尝试使用 LlamaCpp 作为文本生成备选
            if LLAMA_CPP_VLM_AVAILABLE:
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
                        from .llama_cpp_vlm_provider import get_llama_cpp_vlm_provider
                    except ImportError:
                        from llama_cpp_vlm_provider import get_llama_cpp_vlm_provider

                    self._llm_client = get_llama_cpp_vlm_provider(
                        model_path=llama_model_path,
                        mmproj_path=llama_mmproj_path,
                        n_ctx=llama_n_ctx,
                        n_gpu_layers=llama_n_gpu_layers,
                        max_tokens=llama_max_tokens,
                        temperature=llama_temperature
                    )
                    logger.info("✅ 使用 LlamaCpp 本地模型进行文本生成")
                    self._llm_initialized = True
                    return self._llm_client
                except Exception as e:
                    logger.warning(f"⚠️ LlamaCpp 模型加载失败: {e}")

            # 如果都获取不到，报错
            raise ValueError(
                "未配置 text_provider_id、无法获取当前LLM Provider。"
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

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> QueryResult:
        """
        检索相关文档（llama-index风格接口）

        Args:
            query: 查询文本
            top_k: 返回数量

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

        try:
            # 执行向量检索
            query_result = await retriever.retrieve(query, top_k)

            # 如果启用了重排序且结果数量足够，进行重排序
            if reranker and len(query_result.nodes) >= 3:
                # 转换为字典格式
                results_dict = []
                for i, node in enumerate(query_result.nodes):
                    results_dict.append({
                        "text": node.text,
                        "metadata": node.metadata,
                        "score": query_result.scores[i] if i < len(query_result.scores) else 0.0
                    })

                # 执行重排序
                reranked_results = await reranker.rerank(
                    query=query,
                    results=results_dict,
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

                logger.info(f"✅ 重排序完成: {len(query_result.nodes)} → {len(nodes)} 个节点")
                return QueryResult(nodes=nodes, scores=scores)

            return query_result
        except Exception as e:
            logger.error(f"❌ 检索失败: {e}")
            return QueryResult(nodes=[], scores=[])

    async def search(
        self,
        query: str,
        mode: str = "rag",
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        搜索论文（支持多模态查询）

        Args:
            query: 查询文本
            mode: 模式
                - "rag": 检索 + RAG生成
                - "retrieve": 仅检索
            images: 图片路径列表（支持多模态查询）

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
                return await self._rag_query(query, images=images)

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return {
                "type": "error",
                "message": f"搜索失败: {e}"
            }

    async def _rag_query(
        self,
        query: str,
        images: Optional[List[str]] = None
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
            logger.debug(f"[VLM Debug] 原始images参数: {images}")
            logger.debug(f"[VLM Debug] 检索到的sources数量: {len(sources)}")
            for i, src in enumerate(sources):
                logger.debug(f"[VLM Debug] Source[{i}] metadata: {src.get('metadata', {})}")
                img_path = src.get("metadata", {}).get("image_path")
                if img_path:
                    logger.debug(f"[VLM Debug] Source[{i}] 有image_path: {img_path}")

            if images:
                # 用户上传了图片，直接使用VLM
                final_images = images
                logger.info(f"🖼️ 用户上传 {len(images)} 张图片，使用VLM模式")
                for i, img in enumerate(images):
                    logger.debug(f"[VLM Debug] 用户上传图片[{i}]: {img}")
            else:
                # 自动路由：根据查询和检索结果判断
                if self._should_use_vlm(query, sources):
                    # 从检索结果提取关联图片
                    final_images = self._extract_image_paths_from_sources(sources)
                    logger.debug(f"[VLM Debug] _extract_image_paths_from_sources返回: {final_images}")
                    if final_images:
                        logger.info(f"🖼️ 检索到 {len(final_images)} 张关联图片，使用VLM模式")
                        for i, img in enumerate(final_images):
                            exists = os.path.exists(img) if img else False
                            size = os.path.getsize(img) if exists else 0
                            logger.debug(f"[VLM Debug] 检索图片[{i}]: {img} (exists={exists}, size={size})")
                    else:
                        # 有视觉关键词但没有关联图片，回退到LLM
                        logger.info(f"📝 检测到视觉关键词但无关联图片，使用LLM模式")
                        final_images = None
                else:
                    final_images = None
                    logger.info(f"📝 纯文本查询，使用LLM模式")

            logger.debug(f"[VLM Debug] 最终final_images: {final_images}")

            # 生成答案
            answer = await self._generate_answer_with_llm(
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
        images: Optional[List[str]] = None
    ) -> str:
        """
        使用LLM Provider生成答案（支持多模态）

        Args:
            llm_provider: LLM Provider
            query: 查询文本
            sources: 检索到的源文档
            images: 图片路径列表（可选）
        """
        try:
            # 构建上下文
            context_parts = []
            for i, src in enumerate(sources):
                file_name = src['metadata'].get('file_name', 'unknown')
                text = src['text']
                context_parts.append(f"[来源{i+1}] {file_name}\n{text}")

            context = "\n\n".join(context_parts)

            # 构建提示
            text_prompt = f"""基于以下论文内容回答问题：

{context}

问题：{query}

请提供详细的回答，并引用相关文献。"""

            # 判断使用哪个Provider
            # 多模态：优先使用配置的 multimodal_provider_id
            # 文本：使用 text_provider_id 或当前provider
            use_multimodal = images is not None
            provider_to_use = llm_provider  # 默认使用文本provider

            logger.debug(f"[VLM Debug] ===== _generate_answer_with_llm 开始 =====")
            logger.debug(f"[VLM Debug] 输入images: {images}")
            logger.debug(f"[VLM Debug] use_multimodal: {use_multimodal}")
            logger.debug(f"[VLM Debug] multimodal_provider_id配置: {getattr(self.config, 'multimodal_provider_id', 'NOT_SET')}")

            if use_multimodal and self.config.multimodal_provider_id:
                try:
                    provider_manager = getattr(self.context, "provider_manager", None)
                    logger.debug(f"[VLM Debug] provider_manager: {provider_manager}")
                    if provider_manager:
                        inst_map = getattr(provider_manager, "inst_map", None)
                        logger.debug(f"[VLM Debug] inst_map: {inst_map}")
                        logger.debug(f"[VLM Debug] inst_map类型: {type(inst_map)}")
                        if isinstance(inst_map, dict):
                            vlm_provider = inst_map.get(self.config.multimodal_provider_id)
                            logger.debug(f"[VLM Debug] 从inst_map获取的vlm_provider: {vlm_provider}")
                            logger.debug(f"[VLM Debug] vlm_provider类型: {type(vlm_provider) if vlm_provider else None}")
                            if vlm_provider:
                                provider_to_use = vlm_provider
                                provider_name = getattr(vlm_provider, 'model_name', None) or getattr(vlm_provider, 'model', None) or 'unknown'
                                logger.info(f"🖼️ 使用VLM模式，多模态Provider: {self.config.multimodal_provider_id} (实际provider: {provider_name})")
                            else:
                                logger.warning(f"⚠️ 未找到多模态Provider: {self.config.multimodal_provider_id}，回退到文本模式")
                                logger.warning(f"[VLM Debug] inst_map的keys: {list(inst_map.keys()) if inst_map else []}")
                                use_multimodal = False
                except Exception as e:
                    logger.warning(f"⚠️ 获取多模态Provider失败: {e}，回退到文本模式")
                    use_multimodal = False
            elif use_multimodal:
                # multimodal_provider_id 未配置，使用 Llama.cpp VLM Provider
                if LLAMA_CPP_VLM_AVAILABLE:
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
                            from .llama_cpp_vlm_provider import get_llama_cpp_vlm_provider
                        except ImportError:
                            from llama_cpp_vlm_provider import get_llama_cpp_vlm_provider
                        provider_to_use = get_llama_cpp_vlm_provider(
                            model_path=llama_model_path,
                            mmproj_path=llama_mmproj_path,
                            n_ctx=llama_n_ctx,
                            n_gpu_layers=llama_n_gpu_layers,
                            max_tokens=llama_max_tokens,
                            temperature=llama_temperature
                        )
                        await provider_to_use.initialize()
                        logger.info(f"🖼️ 使用Llama.cpp VLM Provider: {llama_model_path}")
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
                logger.debug(f"[VLM Debug] 解析后的image_urls: {image_urls}")

                # 验证图片文件
                for i, img_url in enumerate(image_urls):
                    exists = os.path.exists(img_url)
                    size = os.path.getsize(img_url) if exists else 0
                    logger.debug(f"[VLM Debug] 图片[{i}] 路径: {img_url}")
                    logger.debug(f"[VLM Debug] 图片[{i}] 存在: {exists}, 大小: {size} bytes")
                    if not exists:
                        logger.warning(f"[VLM Debug] 图片[{i}] 文件不存在: {img_url}")

                # 检查prompt长度
                prompt_len = len(text_prompt)
                logger.debug(f"[VLM Debug] prompt长度: {prompt_len} 字符")
                logger.debug(f"[VLM Debug] prompt前200字符: {text_prompt[:200]}...")

                # 使用 text_chat 接口（AstrBot Provider 统一接口）
                # image_urls 支持本地文件路径或 URL
                start_time = time.time()
                if hasattr(provider_to_use, 'text_chat'):
                    try:
                        provider_class_name = type(provider_to_use).__name__
                        logger.info(f"[VLM Debug] 开始调用 {provider_class_name}.text_chat()")
                        logger.debug(f"[VLM Debug] provider类型: {provider_class_name}")
                        logger.debug(f"[VLM Debug] text_chat参数: prompt_len={prompt_len}, image_urls={image_urls}, temperature=0.7")

                        response = await provider_to_use.text_chat(
                            prompt=text_prompt,
                            image_urls=image_urls if image_urls else None,
                            contexts=[],
                            temperature=0.7
                        )

                        elapsed = time.time() - start_time
                        logger.info(f"[VLM Debug] text_chat完成，耗时: {elapsed:.2f}秒")
                        logger.debug(f"[VLM Debug] response类型: {type(response)}")
                        logger.debug(f"[VLM Debug] response内容: {response}")

                        answer = self._extract_answer_from_response(response)
                        logger.debug(f"[VLM Debug] 提取的answer: {answer[:200] if answer else 'None'}...")
                        logger.debug(f"[VLM Debug] ===== _generate_answer_with_llm 成功完成 =====")
                        return answer
                    except Exception as e:
                        elapsed = time.time() - start_time
                        logger.error(f"❌ VLM模式失败: {e}，耗时: {elapsed:.2f}秒")
                        logger.warning("[VLM Debug] VLM处理失败，尝试回退到LLM文本模式...")
                        import traceback
                        logger.debug(f"[VLM Debug] VLM异常详情: {traceback.format_exc()}")
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

    def _should_use_vlm(self, query: str, sources: List[Dict[str, Any]]) -> bool:
        """
        判断是否应该使用VLM（多模态模型）

        核心路由逻辑：
        1. 查询关键词检测（视觉相关词汇）
        2. 检索结果检测（是否关联了原图）
        3. 同时满足则使用VLM

        Args:
            query: 用户查询
            sources: 检索到的源文档

        Returns:
            True: 使用VLM，False: 使用纯文本LLM
        """
        # 检查查询是否包含视觉关键词
        query_has_visual = any(
            keyword in query.lower()
            for keyword in self.VISUAL_KEYWORDS
        )

        # 检查检索结果是否关联了图片
        sources_have_images = any(
            src.get("metadata", {}).get("image_path")
            for src in sources
        )

        # VLM条件：查询涉及视觉 OR 检索结果有图片
        use_vlm = query_has_visual or sources_have_images

        logger.debug(f"🔍 [VLM路由] query_has_visual={query_has_visual}, sources_have_images={sources_have_images} → use_vlm={use_vlm}")

        return use_vlm

    def _extract_image_paths_from_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """
        从检索结果中提取所有关联的图片路径

        Args:
            sources: 检索到的源文档

        Returns:
            图片路径列表（去重，最多返回3个）
        """
        image_paths = []
        seen = set()

        for src in sources:
            image_path = src.get("metadata", {}).get("image_path")
            if image_path and image_path not in seen:
                seen.add(image_path)
                image_paths.append(image_path)

            # 也检查顶层 metadata
            if isinstance(src.get("metadata"), dict):
                image_path = src["metadata"].get("image_path")
                if image_path and image_path not in seen:
                    seen.add(image_path)
                    image_paths.append(image_path)

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
