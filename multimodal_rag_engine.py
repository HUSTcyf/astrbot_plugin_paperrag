"""
多模态RAG引擎扩展
支持文本、图片、表格、公式的向量化和检索
"""

import asyncio
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
import hashlib

try:
    from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema
    from pymilvus.client.types import DataType as MilvusDataType
except ImportError:
    raise ImportError("请安装 pymilvus: pip install pymilvus")

from .multimodal_extractor import MultimodalPDFExtractor, ExtractedContent
from .vision_encoder import VisionEncoder, SigLIPMultiModalEncoder


@dataclass
class MultiModalChunk:
    """多模态文档块"""
    content_type: Literal["text", "image", "table", "formula"]  # 内容类型
    content: str  # 文本内容或描述
    file_name: str
    page_number: int
    chunk_index: int

    # 可选字段
    image_bytes: Optional[bytes] = None  # 图片数据（仅image类型）
    table_data: Optional[str] = None  # 表格数据（仅table类型）
    formula_latex: Optional[str] = None  # 公式LaTeX（仅formula类型）

    # 嵌入向量
    text_embedding: Optional[List[float]] = None
    vision_embedding: Optional[List[float]] = None

    # 元数据
    caption: Optional[str] = None  # 图注/表注
    context: Optional[str] = None  # 上下文
    bbox: Optional[tuple] = None  # 位置信息


class MultiModalIngestionEngine:
    """多模态文档摄入引擎（支持优雅降级）"""

    def __init__(self,
                 vision_encoder: Optional[VisionEncoder],
                 text_embedding_fn,  # AstrBot embedding function
                 extract_images: bool = True,
                 extract_tables: bool = True,
                 extract_formulas: bool = True):
        """
        初始化摄入引擎

        Args:
            vision_encoder: SigLIP视觉编码器（可以是None，将自动降级）
            text_embedding_fn: 文本嵌入函数（AstrBot provider）
            extract_images: 是否提取图片
            extract_tables: 是否提取表格
            extract_formulas: 是否提取公式
        """
        self.vision_encoder = vision_encoder
        self.text_embedding_fn = text_embedding_fn
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_formulas = extract_formulas

        # 创建提取器（会自动降级）
        self.extractor = MultimodalPDFExtractor(
            extract_images=extract_images,
            extract_tables=extract_tables,
            extract_formulas=extract_formulas,
            fallback_to_text=True  # 启用自动降级
        )

        # 检查视觉编码器可用性
        self.vision_available = vision_encoder is not None and getattr(vision_encoder, 'is_available', False)

        if self.vision_available:
            logger.info("✅ 视觉编码器已启用")
        else:
            logger.info("📝 视觉编码器不可用，将使用文本向量化模式")

    async def process_pdf(self, pdf_path: str) -> List[MultiModalChunk]:
        """
        处理PDF文件，提取并编码多模态内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            多模态块列表
        """
        logger.info(f"🔍 开始多模态处理: {pdf_path}")

        # 1. 提取多模态内容
        extracted = self.extractor.extract(pdf_path)

        logger.info(f"✅ 提取完成:")
        logger.info(f"   • 图片: {len(extracted.images)}")
        logger.info(f"   • 表格: {len(extracted.tables)}")
        logger.info(f"   • 公式: {len(extracted.formulas)}")

        chunks = []
        chunk_index = 0

        # 2. 处理图片
        if self.extract_images and extracted.images:
            for img in extracted.images:
                try:
                    # 生成视觉嵌入（如果可用）
                    vision_emb = None
                    if self.vision_available and img.image_bytes:
                        vision_emb = self.vision_encoder.encode_image(img.image_bytes)
                        if vision_emb is None:
                            logger.debug(f"视觉编码失败，使用纯文本: 页 {img.page_number}")

                    # 生成文本嵌入（图注+上下文）
                    text_content = f"{img.caption or ''} {img.context_before or ''} {img.context_after or ''}".strip()
                    text_emb = None
                    if text_content:
                        text_emb = await self.text_embedding_fn(text_content)

                    # 至少要有一种嵌入
                    if text_emb is None and vision_emb is None:
                        logger.debug(f"跳过图片（无内容）: 页 {img.page_number}")
                        continue

                    chunks.append(MultiModalChunk(
                        content_type="image",
                        content=text_content or f"Image on page {img.page_number}",
                        file_name=extracted.file_name,
                        page_number=img.page_number,
                        chunk_index=chunk_index,
                        image_bytes=img.image_bytes,
                        text_embedding=text_emb,
                        vision_embedding=vision_emb,
                        caption=img.caption,
                        context=f"{img.context_before or ''} {img.context_after or ''}".strip(),
                        bbox=img.bbox
                    ))
                    chunk_index += 1

                except Exception as e:
                    logger.warning(f"⚠️ 处理图片失败 (页 {img.page_number}): {e}")
                    continue

        # 3. 处理表格
        if self.extract_tables and extracted.tables:
            for table in extracted.tables:
                try:
                    # 表格转文本（Markdown格式）
                    table_text = table.markdown or table.csv or str(table.data)
                    text_content = f"{table.caption or ''} {table_text}".strip()

                    # 生成文本嵌入
                    text_emb = await self.text_embedding_fn(text_content)

                    if text_emb is None:
                        logger.debug(f"跳过表格（无嵌入）: 页 {table.page_number}")
                        continue

                    chunks.append(MultiModalChunk(
                        content_type="table",
                        content=text_content,
                        file_name=extracted.file_name,
                        page_number=table.page_number,
                        chunk_index=chunk_index,
                        table_data=table_text,
                        text_embedding=text_emb,
                        caption=table.caption,
                        context=table.context,
                        bbox=table.bbox
                    ))
                    chunk_index += 1

                except Exception as e:
                    logger.warning(f"⚠️ 处理表格失败 (页 {table.page_number}): {e}")
                    continue

        # 4. 处理公式
        if self.extract_formulas and extracted.formulas:
            for formula in extracted.formulas:
                try:
                    # 公式文本
                    text_content = f"Formula: {formula.text}"

                    # 生成文本嵌入
                    text_emb = await self.text_embedding_fn(text_content)

                    if text_emb is None:
                        logger.debug(f"跳过公式（无嵌入）: 页 {formula.page_number}")
                        continue

                    chunks.append(MultiModalChunk(
                        content_type="formula",
                        content=text_content,
                        file_name=extracted.file_name,
                        page_number=formula.page_number,
                        chunk_index=chunk_index,
                        formula_latex=formula.text,
                        text_embedding=text_emb,
                        caption=None,
                        context=None,
                        bbox=formula.bbox
                    ))
                    chunk_index += 1

                except Exception as e:
                    logger.warning(f"⚠️ 处理公式失败 (页 {formula.page_number}): {e}")
                    continue

        logger.info(f"✅ 多模态编码完成: {len(chunks)} 个块")
        return chunks


class MultiModalMilvusStore:
    """多模态向量存储（扩展原有 MilvusStore）"""

    def __init__(self,
                 uri: str,
                 collection_name: str,
                 text_dim: int,
                 vision_dim: int,
                 db_name: str = "default",
                 authentication: Optional[dict] = None):
        """
        初始化多模态存储

        Args:
            uri: Milvus URI
            collection_name: 集合名称
            text_dim: 文本嵌入维度
            vision_dim: 视觉嵌入维度
            db_name: 数据库名称
            authentication: 认证信息
        """
        self.uri = uri
        self.collection_name = collection_name
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.db_name = db_name or "default"
        self.authentication = authentication or {}
        self._alias = f"multimodal_{collection_name}"
        self._is_connected = False
        self._collection_initialized = False

    def _ensure_connected(self):
        """确保已连接"""
        if self._is_connected:
            return

        connect_params = {}

        if self.uri.endswith(".db"):
            import os
            os.makedirs(os.path.dirname(self.uri) or ".", exist_ok=True)
            connect_params["uri"] = self.uri
        else:
            connect_params["uri"] = self.uri

        if self.authentication.get("token"):
            connect_params["token"] = self.authentication["token"]

        if self.db_name != "default":
            connect_params["db_name"] = self.db_name

        connections.connect(alias=self._alias, **connect_params)
        self._is_connected = True

    async def _ensure_collection(self):
        """确保集合已初始化"""
        if self._collection_initialized:
            return

        # 确保已连接（同步方法）
        self._ensure_connected()

        has_collection = utility.has_collection(self.collection_name, using=self._alias)

        if not has_collection:
            # 创建多模态schema
            fields = [
                FieldSchema(name="id", dtype=MilvusDataType.INT64, is_primary=True, auto_id=True),  # type: ignore[arg-type]
                FieldSchema(name="text_vector", dtype=MilvusDataType.FLOAT_VECTOR, dim=self.text_dim),  # type: ignore[arg-type]
                FieldSchema(name="vision_vector", dtype=MilvusDataType.FLOAT_VECTOR, dim=self.vision_dim),  # type: ignore[arg-type]
                FieldSchema(name="content", dtype=MilvusDataType.VARCHAR, max_length=65535),  # type: ignore[arg-type]
                FieldSchema(name="content_type", dtype=MilvusDataType.VARCHAR, max_length=20),  # type: ignore[arg-type]
                FieldSchema(name="metadata", dtype=MilvusDataType.JSON),  # type: ignore[arg-type]
            ]

            schema = CollectionSchema(fields=fields, description="Multimodal document chunks")
            collection = Collection(name=self.collection_name, schema=schema, using=self._alias)

            # 创建索引
            is_lite = self.uri.endswith(".db")
            index_type = "AUTOINDEX" if is_lite else "HNSW"

            # create_index 可能返回协程或 Future
            text_index_result = collection.create_index(
                field_name="text_vector",
                index_params={"index_type": index_type, "metric_type": "COSINE"}
            )

            # 处理不同类型的返回值
            if asyncio.iscoroutine(text_index_result):
                await text_index_result
            elif text_index_result is not None and hasattr(text_index_result, 'done'):
                text_index_result.result()  # type: ignore[arg-type]

            vision_index_result = collection.create_index(
                field_name="vision_vector",
                index_params={"index_type": index_type, "metric_type": "COSINE"}
            )

            # 处理不同类型的返回值
            if asyncio.iscoroutine(vision_index_result):
                await vision_index_result
            elif vision_index_result is not None and hasattr(vision_index_result, 'done'):
                vision_index_result.result()  # type: ignore[arg-type]

            load_result = collection.load()

            # load() 可能返回 Future
            if load_result is not None and hasattr(load_result, 'done'):
                load_result.result()  # type: ignore[arg-type]

        self._collection_initialized = True

    async def add_chunks(self, chunks: List[MultiModalChunk]) -> int:
        """添加多模态块"""
        await self._ensure_collection()

        collection = Collection(self.collection_name, using=self._alias)

        data = []
        for chunk in chunks:
            # 只添加有嵌入向量的块
            if chunk.text_embedding is None and chunk.vision_embedding is None:
                continue

            data.append({
                "text_vector": chunk.text_embedding or [0.0] * self.text_dim,
                "vision_vector": chunk.vision_embedding or [0.0] * self.vision_dim,
                "content": chunk.content[:65535],
                "content_type": chunk.content_type,
                "metadata": {
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "caption": chunk.caption,
                    "bbox": chunk.bbox,
                    "added_time": datetime.now().isoformat()
                }
            })

        if data:
            insert_result = collection.insert(data)

            # insert() 可能返回协程或 MutationFuture
            if asyncio.iscoroutine(insert_result):
                insert_result = await insert_result
            elif insert_result is not None and hasattr(insert_result, 'done'):
                insert_result.result()  # type: ignore[arg-type]

            flush_result = collection.flush()

            # flush() 可能返回 Future
            if flush_result is not None and hasattr(flush_result, 'done'):
                flush_result.result()  # type: ignore[arg-type]

            return len(data)

        return 0

    async def hybrid_search(self,
                           query_text: str = None,
                           query_image_bytes: bytes = None,
                           text_weight: float = 0.5,
                           vision_weight: float = 0.5,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索（文本+图像）

        Args:
            query_text: 查询文本
            query_image_bytes: 查询图像
            text_weight: 文本权重
            vision_weight: 视觉权重
            top_k: 返回结果数

        Returns:
            检索结果
        """
        await self._ensure_collection()
        collection = Collection(self.collection_name, using=self._alias)

        results = []

        # 文本检索
        if query_text:
            # 这里需要调用 text embedding function
            # 暂时跳过
            pass

        # 视觉检索
        if query_image_bytes:
            # 这里需要调用 vision encoder
            # 暂时跳过
            pass

        return results
