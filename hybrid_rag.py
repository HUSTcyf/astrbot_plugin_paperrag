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
import base64
from typing import List, Dict, Any, Optional, Union, cast

# 抑制底层库的 gRPC/absl 警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from astrbot.api import logger

# 导入混合架构组件
from .hybrid_parser import HybridPDFParser, Node
from .hybrid_index import HybridIndexManager

# 导入Embedding Provider
from .embedding_providers import (
    create_embedding_provider,
    OllamaEmbeddingProvider,
    AstrBotEmbeddingProvider
)

# 导入配置
from .rag_engine import RAGConfig


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


class HybridRAGEngine:
    """
    混合RAG引擎（多模态支持版）

    基于llama-index风格设计：
    - PDF解析（多模态）
    - Node结构存储
    - VectorRetriever检索器
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
        self._retriever: VectorRetriever = cast(VectorRetriever, None)

        # 初始化标志
        self._parser_initialized = False
        self._index_initialized = False
        self._embed_provider_initialized = False
        self._llm_initialized = False
        self._retriever_initialized = False

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
            self._embed_provider = create_embedding_provider(
                mode=self.config.embedding_mode,
                context=self.context,
                provider_id=self.config.embedding_provider_id,
                ollama_config=self.config.ollama_config,
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
            logger.debug(f"🔍 [DEBUG] _ensure_index_manager_initialized: mode='{mode}', milvus_lite_path='{self.config.milvus_lite_path}', address='{self.config.address}'")

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

    def _ensure_retriever_initialized(self) -> VectorRetriever:
        """确保检索器已初始化"""
        if self._retriever_initialized:
            assert self._retriever is not None
            return self._retriever

        # 确保依赖组件已初始化
        index_manager = self._ensure_index_manager_initialized()
        embed_provider = cast(Union[OllamaEmbeddingProvider, AstrBotEmbeddingProvider], self._embed_provider)

        if embed_provider is None:
            raise RuntimeError("Embed provider not initialized")

        self._retriever = VectorRetriever(
            index_manager=index_manager,
            embed_provider=embed_provider
        )
        self._retriever_initialized = True
        logger.info("✅ VectorRetriever初始化完成")
        assert self._retriever is not None
        return self._retriever

    async def _ensure_llm_initialized(self) -> Any:
        """确保GLM LLM已初始化"""
        if self._llm_initialized:
            assert self._llm_client is not None
            return self._llm_client

        if not self.config.glm_api_key:
            raise ValueError("GLM API Key未配置")

        try:
            from openai import AsyncOpenAI
            self._llm_client = AsyncOpenAI(
                api_key=self.config.glm_api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
            self._llm_initialized = True
            logger.info(f"✅ GLM LLM初始化完成: {self.config.glm_model}")
            assert self._llm_client is not None
            return self._llm_client
        except Exception as e:
            logger.error(f"❌ GLM LLM初始化失败: {e}")
            self._llm_initialized = True
            raise

    async def add_paper(self, file_path: str) -> Dict[str, Any]:
        """
        添加论文到知识库

        Args:
            file_path: PDF文件路径

        Returns:
            添加结果
        """
        try:
            # 确保组件已初始化
            parser = self._ensure_parser_initialized()
            embed_provider = await self._ensure_embed_provider_initialized()
            index_manager = self._ensure_index_manager_initialized()

            # 解析PDF并分块
            logger.info(f"📄 处理文件: {file_path}")
            nodes = parser.parse_and_split(file_path)

            if not nodes:
                return {
                    "status": "error",
                    "message": f"无法解析文件: {file_path}"
                }

            logger.info(f"📑 解析完成: {len(nodes)} 个节点")

            # 批量获取embeddings
            texts = [node.text for node in nodes]
            embeddings = await embed_provider.get_text_embeddings_batch(texts)

            # 插入到索引
            count = await index_manager.insert_nodes(nodes, embeddings)

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

        try:
            return await retriever.retrieve(query, top_k)
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
                    "results": results,
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

        Args:
            query: 查询文本
            images: 图片路径列表（可选）
        """
        try:
            # 检查LLM是否可用
            if not self.config.glm_api_key:
                return {
                    "type": "error",
                    "message": "GLM API Key未配置，无法生成回答。请配置glm_api_key或使用retrieve模式。"
                }

            # 确保LLM已初始化
            llm_client = await self._ensure_llm_initialized()

            # 执行检索
            query_result = await self.retrieve(query, self.config.top_k)

            if len(query_result) == 0 and not images:
                return {
                    "type": "error",
                    "message": "未找到相关文档"
                }

            # 转换为源文档格式
            sources = []
            for i, node in enumerate(query_result.nodes):
                sources.append({
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": query_result.scores[i] if i < len(query_result.scores) else 0.0
                })

            # 生成答案（支持多模态）
            answer = await self._generate_answer_with_llm(
                llm_client, query, sources, images=images
            )

            return {
                "type": "rag",
                "query": query,
                "answer": answer,
                "sources": sources,
                "images": images if images else None
            }

        except Exception as e:
            logger.error(f"❌ RAG生成失败: {e}")
            return {
                "type": "error",
                "message": f"RAG生成失败: {e}"
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

    async def _generate_answer_with_llm(
        self,
        llm_client: Any,
        query: str,
        sources: List[Dict[str, Any]],
        images: Optional[List[str]] = None
    ) -> str:
        """
        使用GLM LLM生成答案（支持多模态）

        Args:
            llm_client: LLM客户端
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

            # 多模态处理：构建消息内容
            if images:
                # 使用多模态模型 glm-4.6v-flash
                model = self.config.glm_multimodal_model
                content = []

                # 添加文本部分
                content.append({
                    "type": "text",
                    "text": text_prompt
                })

                # 添加图片部分
                for image_path in images:
                    base64_image = self._encode_image_to_base64(image_path)
                    if base64_image:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })

                messages = [
                    {"role": "user", "content": content}
                ]

                logger.info(f"🖼️ 使用多模态模式查询 (模型: {model}, 图片数: {len(images)})")
            else:
                # 使用文本模型
                model = self.config.glm_model
                messages = [
                    {"role": "user", "content": text_prompt}
                ]
                logger.info(f"📝 使用文本模式查询 (模型: {model})")

            # 调用GLM
            response = await llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            # 提取回答
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            logger.error(f"❌ LLM生成失败: {e}")
            return f"生成答案失败: {e}"

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
                self._retriever = cast(VectorRetriever, None)
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
