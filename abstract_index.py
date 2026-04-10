"""
摘要索引管理器 - 基于摘要的两阶段检索

功能：
1. 为每篇论文提取摘要并生成向量
2. 存储在独立的 Milvus collection
3. 检索时先通过摘要匹配相关论文，再深入检索详情

两阶段检索流程：
  查询 → [阶段1] 摘要向量检索 → top-k 相关论文 → [阶段2] chunk 检索 → 最终结果
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, cast
from dataclasses import dataclass

from astrbot.api import logger

# 延迟导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pymilvus import Collection


# ============================================================================
# 本地 GGUF LLM 客户端（用于摘要提取）
# ============================================================================

class LocalGGUFClient:
    """使用本地 GGUF 模型提取摘要，优先复用已加载的 LlamaCppVLMProvider"""

    DEFAULT_MODEL_PATH = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    DEFAULT_MMproj_PATH = "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"

    def __init__(self, model_path: Optional[str] = None, mmproj_path: Optional[str] = None):
        self._model_path = model_path or self.DEFAULT_MODEL_PATH
        self._mmproj_path = mmproj_path or self.DEFAULT_MMproj_PATH
        self._llama: Optional[Any] = None
        self._is_loaded = False

    def _resolve_path(self, path: str) -> str:
        """解析模型路径（相对于插件目录）"""
        if os.path.isabs(path):
            return path
        plugin_dir = Path(__file__).parent.resolve()
        return str(plugin_dir / path)

    async def load(self) -> bool:
        """加载 GGUF 模型"""
        if self._is_loaded and self._llama is not None:
            return True

        model_path = self._resolve_path(self._model_path)
        mmproj_path = self._resolve_path(self._mmproj_path)

        if not os.path.exists(model_path) or not os.path.exists(mmproj_path):
            logger.warning(f"GGUF 模型文件不存在")
            return False

        try:
            from llama_cpp import Llama
            import concurrent.futures

            def _load():
                return Llama(
                    model_path=model_path,
                    mmproj=mmproj_path,
                    n_ctx=4096,
                    n_gpu_layers=99,
                    n_batch=32,
                    verbose=False,
                )

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self._llama = await loop.run_in_executor(executor, _load)
            self._is_loaded = True
            logger.info("✅ GGUF 模型加载成功")
            return True

        except Exception as e:
            logger.error(f"GGUF 模型加载失败: {e}")
            return False

    async def extract_title_and_abstract(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        使用 LLM 从论文开头提取标题和摘要

        Args:
            text: 论文开头的文本（直到 Introduction 之前）

        Returns:
            (标题, 摘要) 元组，任一失败返回 (None, None)
        """
        if not self._is_loaded or self._llama is None:
            success = await self.load()
            if not success:
                return None, None

        prompt = f"""从以下论文内容中提取标题和摘要。

要求：
1. 标题：返回论文的完整标题（通常在页面顶部），保持原文语言
2. 摘要：只返回摘要部分，完全保持原文语言（英文就返回英文，中文就返回中文），不要翻译，不要润色或修改
3. 如果内容明显不是论文，返回空标题和空摘要
4. 严格按照以下JSON格式返回，不要添加任何其他内容：
{{"title": "论文标题", "abstract": "摘要内容"}}

论文内容：
{text[:4000]}

JSON："""

        content: str = ""
        try:
            llama = self._llama
            if llama is None:
                return None, None
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: llama.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )
            )

            content = result["choices"][0]["message"]["content"].strip()

            # 解析 JSON
            # 尝试提取 JSON（可能包含在 markdown 代码块中）
            json_match = re.search(r'\{[^{}]*"title"[^{}]*"abstract"[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 尝试直接解析整个响应
                json_str = content

            data = json.loads(json_str)
            title = data.get("title", "").strip('"\n ')
            abstract = data.get("abstract", "").strip('"\n ')

            # 验证结果
            if len(abstract) < 30:
                logger.debug(f"摘要太短，可能是无效内容")
                return title if title else None, None

            return title, abstract

        except json.JSONDecodeError as e:
            logger.debug(f"JSON 解析失败: {e}, 尝试备用提取")
            # 备用：尝试用正则提取 LLM 响应中的 JSON
            return self._extract_title_abstract_fallback(content)
        except Exception as e:
            logger.warning(f"LLM 提取标题和摘要失败: {e}")
            return None, None

    def _extract_title_abstract_fallback(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """备用提取：当 JSON 解析失败时使用正则表达式提取"""
        try:
            import re
            # 尝试匹配 JSON 中的 title 和 abstract
            title_match = re.search(r'"title"\s*:\s*"([^"]+)"', text[:500])
            abstract_match = re.search(r'"abstract"\s*:\s*"([^"]+)"', text)

            title = title_match.group(1) if title_match else None
            abstract = abstract_match.group(1) if abstract_match else None

            if abstract and len(abstract) >= 30:
                return title, abstract
        except Exception:
            pass

        return None, None


# ============================================================================
# 摘要提取器
# ============================================================================

class AbstractExtractor:
    """从 PDF 中提取摘要"""

    # 摘要部分的关键词（按优先级）
    ABSTRACT_KEYWORDS = [
        r'^abstract\s*$',
        r'^摘要\s*$',
        r'^summary\s*$',
        r'^概述\s*$',
        r'^abstract.',
        r'^ABSTRACT\s*$',
    ]

    # 正文开始的标志（遇到这些说明摘要结束）
    INTRODUCTION_KEYWORDS = [
        r'^1\.?\s*introduction\s*$',
        r'^1\.?\s*引言\s*$',
        r'^introduction\s*$',
        r'^1\s+[A-Z]',
        r'^一、引言',
        r'^1\.?\s+[A-Z][a-z]+',  # "1. Background"
        r'^references\s*$',
        r'^参考文献\s*$',
    ]

    def __init__(self, parser=None):
        """
        Args:
            parser: 已有解析器（HybridPDFParser/DoclingExtractor），用于解析PDF
        """
        self._parser = parser

    async def extract_abstract_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        从 PDF 文件提取摘要

        Args:
            pdf_path: PDF 文件路径

        Returns:
            摘要文本，如果未找到返回 None
        """
        try:
            if self._parser:
                # 使用已有的解析器
                result = await self._parse_with_existing_parser(pdf_path)
            else:
                # 使用默认的 PyMuPDF 解析
                result = await self._parse_with_pymupdf(pdf_path)

            if result:
                abstract = self._extract_abstract_text(result)
                if abstract and len(abstract) > 50:  # 摘要太短可能是误识别
                    return abstract

            return None

        except Exception as e:
            logger.warning(f"提取摘要失败 {pdf_path}: {e}")
            return None

    async def _parse_with_existing_parser(self, pdf_path: str) -> Optional[str]:
        """使用已有解析器解析 PDF"""
        if self._parser is None:
            return None

        try:
            # 解析 PDF
            if hasattr(self._parser, 'parse_and_split'):
                nodes = await self._parser.parse_and_split(pdf_path, {}, None)
                # 合并所有文本用于摘要提取
                full_text = "\n".join(node.text for node in nodes if hasattr(node, 'text'))
                return full_text
            elif hasattr(self._parser, 'parse'):
                result = await self._parser.parse(pdf_path)
                if isinstance(result, dict):
                    return result.get('text', '')
                elif isinstance(result, str):
                    return result
            return None
        except Exception as e:
            logger.warning(f"解析器解析失败: {e}")
            return None

    async def _parse_with_pymupdf(self, pdf_path: str) -> Optional[str]:
        """使用 PyMuPDF 解析 PDF"""
        try:
            import pymupdf

            doc = pymupdf.open(pdf_path)
            full_text = ""

            for page_num in range(min(len(doc), 5)):  # 只看前5页
                page = doc[page_num]
                text = page.get_text()
                if text:
                    full_text += str(text) + "\n"

            doc.close()
            return full_text if full_text.strip() else None

        except Exception as e:
            logger.warning(f"PyMuPDF 解析失败: {e}")
            return None

    def _extract_abstract_text(self, full_text: str) -> Optional[str]:
        """
        从完整文本中提取摘要部分

        策略：
        1. 找到 "Abstract" 关键词所在行或段落开始
        2. 提取直到遇到 "Introduction" 或 "1. " 等正文开始标志
        """
        if not full_text:
            return None

        lines = full_text.split('\n')
        abstract_start = -1
        abstract_end = -1

        # 找到摘要开始位置
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 检查是否是摘要开始
            for pattern in self.ABSTRACT_KEYWORDS:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # 摘要可能在标题行，也可能在标题后的段落
                    if re.match(r'^abstract\s*$', line_stripped, re.IGNORECASE):
                        # 标题行，摘要从下一行开始
                        abstract_start = i + 1
                    else:
                        # 包含关键词的段落，尝试提取
                        abstract_start = i
                    break

            if abstract_start >= 0:
                break

        if abstract_start < 0:
            return None

        # 找到摘要结束位置
        for i in range(abstract_start + 1, len(lines)):
            line = lines[i].strip()

            # 跳过空行
            if not line:
                continue

            # 检查是否是正文开始
            for pattern in self.INTRODUCTION_KEYWORDS:
                if re.match(pattern, line, re.IGNORECASE):
                    abstract_end = i
                    break

            if abstract_end >= 0:
                break

        # 如果没找到结束符，取前几段
        if abstract_end < 0:
            # 摘要通常在前2-3段
            paragraph_count = 0
            for i in range(abstract_start, len(lines)):
                if lines[i].strip():
                    paragraph_count += 1
                if paragraph_count >= 3:  # 最多3段
                    abstract_end = i + 1
                    break

        if abstract_end < 0:
            abstract_end = len(lines)

        # 提取摘要文本
        abstract_lines = lines[abstract_start:abstract_end]
        abstract_text = '\n'.join(line.strip() for line in abstract_lines if line.strip())

        # 清理：移除可能的 "Abstract:" 前缀
        abstract_text = re.sub(r'^abstract:\s*', '', abstract_text, flags=re.IGNORECASE)
        abstract_text = abstract_text.strip()

        return abstract_text if abstract_text else None

    def extract_title_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        从 PDF 文件提取标题

        策略：
        1. 提取 PDF 元数据中的标题
        2. 如果没有，使用文件名前缀

        Args:
            pdf_path: PDF 文件路径

        Returns:
            标题文本，如果未找到返回 None
        """
        try:
            import pymupdf

            doc = pymupdf.open(pdf_path)

            # 尝试从元数据获取标题
            metadata = doc.metadata
            if metadata:
                title = metadata.get('title', '')
                if title and title.strip():
                    doc.close()
                    return title.strip()

            # 使用文件名前缀作为标题
            doc.close()
            filename = os.path.basename(pdf_path)
            title = os.path.splitext(filename)[0]
            # 移除常见后缀
            title = re.sub(r'[_-]?(v\d+)?(\.pdf)?$', '', title, flags=re.IGNORECASE)
            return title if title else None

        except Exception as e:
            logger.warning(f"提取标题失败 {pdf_path}: {e}")
            return None


# ============================================================================
# 摘要索引管理器
# ============================================================================

@dataclass
class PaperAbstract:
    """论文摘要数据结构"""
    paper_id: str           # 论文 ID（不含扩展名）
    file_name: str           # 完整文件名
    title: str = ""          # 论文标题
    abstract_text: str = ""  # 摘要文本
    vector: Optional[List[float]] = None  # 摘要向量
    metadata: Optional[Dict[str, Any]] = None  # 其他元数据

    def __post_init__(self):
        if self.vector is None:
            self.vector = []
        if self.metadata is None:
            self.metadata = {}


class AbstractIndexManager:
    """
    摘要索引管理器

    负责：
    1. 管理论文摘要的向量索引
    2. 提供基于摘要的快速论文筛选
    3. 支持两阶段检索
    """

    def __init__(
        self,
        milvus_uri: str = "./data/milvus_abstracts.db",
        collection_name: str = "paper_abstracts",
        embed_dim: int = 1024,
        embed_model = None,
        alias: str = "abstract_index"
    ):
        """
        Args:
            milvus_uri: Milvus Lite 数据库路径
            collection_name: collection 名称
            embed_dim: embedding 维度
            embed_model: embedding 模型实例（必须提供）
            alias: 连接别名
        """
        self.alias = alias
        self._collection_name = collection_name
        self._dim = embed_dim
        self._embed_model = embed_model
        self._llm_client = None
        self._is_connected = False
        self._collection = None

        # 解析路径
        if milvus_uri:
            self._db_path = milvus_uri if milvus_uri else "./data/milvus_abstracts.db"
        else:
            plugin_dir = Path(__file__).parent
            self._db_path = str(plugin_dir / "data" / "milvus_abstracts.db")

        # 确保目录存在
        db_dir = os.path.dirname(self._db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # 摘要数据缓存（paper_id -> PaperAbstract）
        self._abstract_cache: Dict[str, PaperAbstract] = {}

        # 文档统计（用于追踪已处理的论文）
        self._doc_stats_file = self._db_path.replace('.db', '_doc_stats.json')

        logger.info(f"✅ AbstractIndexManager 初始化完成 (collection={collection_name}, dim={embed_dim})")

    def set_embed_model(self, embed_model):
        """设置 embedding 模型"""
        self._embed_model = embed_model

    def set_llm_client(self, llm_client):
        """设置 LLM 客户端（用于摘要提取）"""
        self._llm_client = llm_client

    async def _extract_paper_beginning(self, pdf_path: str, max_chars: int = 3000) -> Optional[str]:
        """提取论文开头部分（用于 LLM 提取摘要）"""
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            full_text = ""

            for page_num in range(min(len(doc), 5)):
                page = doc[page_num]
                text = page.get_text()
                if text:
                    full_text += str(text) + "\n"
                if len(full_text) >= max_chars:
                    break

            doc.close()
            return full_text[:max_chars] if full_text.strip() else None

        except Exception as e:
            logger.warning(f"提取论文开头失败 {pdf_path}: {e}")
            return None

    async def initialize(self):
        """初始化连接和 collection"""
        await self._ensure_collection()

    async def _ensure_collection(self):
        """确保 collection 存在"""
        if self._collection is not None:
            return

        try:
            from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

            # 确保目录存在
            db_dir = os.path.dirname(self._db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            # 连接数据库（使用 Lite 模式）
            connections.connect(alias=self.alias, uri=self._db_path)

            # 检查 collection 是否存在
            if utility.has_collection(self._collection_name, using=self.alias):
                self._collection = Collection(self._collection_name, using=self.alias)
                self._collection.load()
                self._is_connected = True
                await self._load_doc_stats()
                return

            # 创建新的 collection
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="abstract_text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            ]
            schema = CollectionSchema(fields=fields, description="Paper abstracts collection")

            self._collection = Collection(
                name=self._collection_name,
                schema=schema,
                using=self.alias
            )

            # 创建索引
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._collection.create_index(  # type: ignore[union-attr]
                    field_name="vector",
                    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
                )
            )

            # 加载 collection
            await loop.run_in_executor(None, self._collection.load)

            self._is_connected = True
            logger.info(f"✅ 创建新 collection: {self._collection_name}")

        except Exception as e:
            logger.error(f"❌ 初始化 AbstractIndex 失败: {e}")
            raise

    async def _load_doc_stats(self):
        """加载文档统计"""
        if not self._doc_stats_file or not os.path.exists(self._doc_stats_file):
            self._abstract_cache = {}
            return

        try:
            with open(self._doc_stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 加载已处理的摘要数据
                self._abstract_cache = {
                    k: PaperAbstract(**v) for k, v in data.get('abstracts', {}).items()
                }
            logger.info(f"📊 已加载 {len(self._abstract_cache)} 篇论文摘要")
        except Exception as e:
            logger.warning(f"⚠️ 加载摘要统计失败: {e}")
            self._abstract_cache = {}

    def _save_doc_stats(self):
        """保存文档统计"""
        if not self._doc_stats_file:
            return

        try:
            db_dir = os.path.dirname(self._doc_stats_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            data = {
                'abstracts': {
                    k: {
                        'paper_id': v.paper_id,
                        'file_name': v.file_name,
                        'title': v.title,
                        'abstract_text': v.abstract_text,
                        'vector': v.vector,
                        'metadata': v.metadata
                    }
                    for k, v in self._abstract_cache.items()
                }
            }

            with open(self._doc_stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"⚠️ 保存摘要统计失败: {e}")

    async def index_paper(
        self,
        pdf_path: str,
        paper_id: str,
        file_name: str,
        title: str = "",
        abstract_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        为单篇论文建立摘要索引

        Args:
            pdf_path: PDF 文件路径（用于提取摘要和标题）
            paper_id: 论文 ID
            file_name: 文件名
            title: 论文标题（如果已提取）
            abstract_text: 如果已提取，传入摘要文本
            metadata: 额外元数据

        Returns:
            是否成功
        """
        await self._ensure_collection()

        try:
            extractor = AbstractExtractor()

            # LLM 同时提取标题和摘要（优先）
            if self._llm_client is not None and (not abstract_text or not title):
                paper_beginning = await self._extract_paper_beginning(pdf_path)
                if paper_beginning:
                    llm_title, llm_abstract = await self._llm_client.extract_title_and_abstract(paper_beginning)
                    if llm_abstract and len(llm_abstract) >= 30:
                        abstract_text = llm_abstract
                    if llm_title:
                        title = llm_title

            # 常规提取（回退）- 标题和摘要分别提取
            if not abstract_text or len(abstract_text) < 50:
                abstract_text = await extractor.extract_abstract_from_pdf(pdf_path)

            if not title:
                extracted_title = extractor.extract_title_from_pdf(pdf_path)
                if extracted_title:
                    title = extracted_title

            if not abstract_text or len(abstract_text) < 50:
                logger.warning(f"未找到有效摘要: {file_name}")
                return False

            if not title:
                title = file_name.replace('.pdf', '').replace('.PDF', '')

            # 生成向量（使用标题 + 摘要的组合）
            if self._embed_model is None:
                logger.error("❌ 未设置 embedding 模型")
                return False

            # 组合标题和摘要作为检索文本
            combined_text = f"{title}\n\n{abstract_text}"
            vector = await self._embed_text(combined_text)

            # 存储到 Milvus（使用 Collection API）
            loop = asyncio.get_event_loop()
            collection = self._collection
            if collection is None:
                logger.error("❌ Collection 未初始化")
                return False
            await loop.run_in_executor(
                None,
                lambda: collection.insert(data=[{
                    "paper_id": paper_id,
                    "file_name": file_name,
                    "title": title,
                    "abstract_text": abstract_text,
                    "vector": vector,
                }])
            )

            # 更新缓存
            self._abstract_cache[paper_id] = PaperAbstract(
                paper_id=paper_id,
                file_name=file_name,
                title=title,
                abstract_text=abstract_text,
                vector=vector,
                metadata=metadata or {}
            )
            self._save_doc_stats()

            logger.info(f"✅ 已索引摘要: {title} ({len(abstract_text)} chars)")
            return True

        except Exception as e:
            logger.error(f"❌ 索引摘要失败 {file_name}: {e}")
            return False

    async def index_papers_bulk(
        self,
        papers: List[Dict[str, str]],
        extractor: Optional[AbstractExtractor] = None,
        progress_callback=None
    ) -> Dict[str, int]:
        """
        批量索引论文摘要

        Args:
            papers: 论文列表 [{"pdf_path": ..., "paper_id": ..., "file_name": ...}, ...]
            extractor: 摘要提取器
            progress_callback: 进度回调函数 callback(current, total)

        Returns:
            {"success": count, "failed": count}
        """
        if extractor is None:
            extractor = AbstractExtractor()

        results = {"success": 0, "failed": 0}
        total = len(papers)

        for i, paper in enumerate(papers):
            try:
                success = await self.index_paper(
                    pdf_path=paper['pdf_path'],
                    paper_id=paper['paper_id'],
                    file_name=paper['file_name'],
                    metadata=cast(Dict[str, Any], paper.get('metadata')) if paper.get('metadata') is not None else None
                )

                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                logger.warning(f"⚠️ 索引失败 {paper.get('file_name')}: {e}")
                results["failed"] += 1

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    async def _embed_text(self, text: str) -> List[float]:
        """生成文本的 embedding 向量"""
        try:
            assert self._embed_model is not None
            if hasattr(self._embed_model, 'embed_text'):
                # 自定义包装的 embed_text 方法（可能同步）
                vector = self._embed_model.embed_text(text)
            elif hasattr(self._embed_model, 'get_text_embedding'):
                # OllamaEmbeddingProvider 等（异步方法，返回单向量）
                vector = await self._embed_model.get_text_embedding(text)
            elif hasattr(self._embed_model, 'embed'):
                # 其他异步接口（可能返回 List[List[float]]）
                result = await self._embed_model.embed(text)
                # embed() 可能返回嵌套列表 [[...]] 或直接是 [...]
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        vector = result[0]  # 取第一个向量的嵌套情况
                    else:
                        vector = result
                else:
                    vector = result
            elif callable(self._embed_model):
                # 直接是函数
                vector = self._embed_model(text)
            else:
                raise ValueError("Unknown embed_model type")

            # 确保是 list[float]
            if not isinstance(vector, list):
                vector = list(cast(Any, vector))

            return vector

        except Exception as e:
            logger.error(f"❌ 生成 embedding 失败: {e}")
            raise

    async def search_by_abstract(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        通过摘要向量检索相关论文（阶段1）

        Args:
            query: 查询文本
            top_k: 返回的相关论文数量

        Returns:
            [{"paper_id": ..., "file_name": ..., "abstract_text": ..., "score": ...}, ...]
        """
        await self._ensure_collection()

        try:
            # 生成查询向量
            query_vector = await self._embed_text(query)

            # 搜索
            client = self._collection
            if client is None:
                logger.error("❌ Collection 未初始化")
                return []
            search_params = {
                "metric_type": "COSINE",
                "params": {}
            }

            results = client.search(
                collection_name=self._collection_name,
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["paper_id", "file_name", "title", "abstract_text"]
            )

            # 转换结果
            papers = []
            results = cast(Any, results)
            for hit in results[0]:
                papers.append({
                    "paper_id": hit.entity.get("paper_id"),
                    "file_name": hit.entity.get("file_name"),
                    "title": hit.entity.get("title", ""),
                    "abstract_text": hit.entity.get("abstract_text"),
                    "score": float(hit.score)
                })

            return papers

        except Exception as e:
            logger.error(f"❌ 摘要检索失败: {e}")
            return []

    async def get_papers_by_ids(
        self,
        paper_ids: List[str]
    ) -> Dict[str, PaperAbstract]:
        """
        根据 paper_ids 获取摘要数据

        Args:
            paper_ids: 论文 ID 列表

        Returns:
            {paper_id: PaperAbstract, ...}
        """
        await self._ensure_collection()

        result = {}
        for pid in paper_ids:
            if pid in self._abstract_cache:
                result[pid] = self._abstract_cache[pid]

        return result

    async def get_all_abstracts(self) -> Dict[str, PaperAbstract]:
        """获取所有摘要数据"""
        await self._ensure_collection()
        return self._abstract_cache.copy()

    async def delete_paper(self, paper_id: str) -> bool:
        """删除论文摘要"""
        await self._ensure_collection()

        try:
            loop = asyncio.get_event_loop()
            collection = self._collection
            if collection is None:
                logger.error("❌ Collection 未初始化")
                return False
            await loop.run_in_executor(
                None,
                lambda: collection.delete(f'paper_id == "{paper_id}"')
            )

            if paper_id in self._abstract_cache:
                del self._abstract_cache[paper_id]
                self._save_doc_stats()

            logger.info(f"🗑️ 已删除摘要: {paper_id}")
            return True

        except Exception as e:
            logger.error(f"❌ 删除摘要失败 {paper_id}: {e}")
            return False

    def clear(self) -> bool:
        """清除摘要索引（删除 collection）"""
        try:
            from pymilvus import utility
            if utility.has_collection(self._collection_name, using=self.alias):
                utility.drop_collection(self._collection_name, using=self.alias)  # type: ignore[func-returns-value]
                logger.info("✅ 摘要索引已清除")
            self._collection = None
            self._is_connected = False
            self._abstract_cache = {}
            return True
        except Exception as e:
            logger.error(f"❌ 清除摘要索引失败: {e}")
            return False

    async def rebuild_index(
        self,
        papers: List[Dict[str, str]],
        extractor: Optional[AbstractExtractor] = None,
        force: bool = False,
        progress_callback=None
    ) -> Dict[str, int]:
        """
        重建所有摘要索引

        Args:
            papers: 论文列表
            extractor: 摘要提取器
            force: 是否强制重建（删除现有数据）
            progress_callback: 进度回调

        Returns:
            {"success": count, "failed": count}
        """
        if force:
            await self._ensure_collection()
            try:
                client = self._collection
                if client is None:
                    logger.warning("⚠️ Collection 未初始化")
                    return {"success": 0, "failed": 0}
                client.drop_collection(self._collection_name)  # type: ignore[attr-defined]
                self._collection = None
                self._abstract_cache = {}
                logger.info("🗑️ 已清空摘要索引")
            except Exception as e:
                logger.warning(f"⚠️ 清空索引失败: {e}")

        return await self.index_papers_bulk(papers, extractor, progress_callback)


# ============================================================================
# 两阶段检索器
# ============================================================================

class TwoStageRetriever:
    """
    两阶段检索器

    阶段1: 通过摘要快速筛选相关论文
    阶段2: 在筛选出的论文中进行详细检索
    """

    def __init__(
        self,
        abstract_index: AbstractIndexManager,
        chunk_index,  # HybridIndexManager
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            abstract_index: 摘要索引管理器
            chunk_index: chunk 索引管理器（HybridIndexManager）
            config: 配置
        """
        self._abstract_index = abstract_index
        self._chunk_index = chunk_index
        self._config = config or {}

        # 配置项
        self._abstract_top_k = self._config.get('abstract_top_k', 5)
        self._chunk_top_k = self._config.get('chunk_top_k', 10)
        self._hybrid_alpha = self._config.get('hybrid_alpha', 0.7)  # 向量权重

    async def retrieve(
        self,
        query: str,
        abstract_top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
        enable_two_stage: bool = True
    ) -> Dict[str, Any]:
        """
        两阶段检索

        Args:
            query: 查询文本
            abstract_top_k: 阶段1返回的论文数
            chunk_top_k: 阶段2返回的 chunk 数
            enable_two_stage: 是否启用两阶段检索

        Returns:
            {
                "type": "two_stage | direct",
                "papers": [...],  # 阶段1的论文结果
                "chunks": [...],   # 阶段2的 chunk 结果
                "query": query
            }
        """
        abstract_top_k = abstract_top_k if abstract_top_k is not None else self._abstract_top_k
        chunk_top_k = chunk_top_k if chunk_top_k is not None else self._chunk_top_k
        abstract_top_k = cast(int, abstract_top_k)
        chunk_top_k = cast(int, chunk_top_k)

        if not enable_two_stage:
            # 直接检索 chunks（原始策略）
            return await self._direct_retrieve(query, chunk_top_k)

        try:
            # ========== 阶段1: 摘要检索 ==========
            papers = await self._abstract_index.search_by_abstract(query, top_k=abstract_top_k)

            if not papers:
                logger.info("[TwoStage] 阶段1未找到相关论文，降级到直接检索")
                return await self._direct_retrieve(query, chunk_top_k)

            logger.info(f"[TwoStage] 阶段1: 找到 {len(papers)} 篇相关论文")
            for p in papers:
                logger.debug(f"  - {p['file_name']} (score={p['score']:.3f})")

            # ========== 阶段2: 在筛选论文中检索 chunks ==========
            paper_ids = [p['paper_id'] for p in papers]
            chunks = await self._chunk_index.search_with_paper_filter(
                query=query,
                paper_ids=paper_ids,
                top_k=chunk_top_k
            )

            return {
                "type": "two_stage",
                "papers": papers,
                "chunks": chunks,
                "query": query
            }

        except Exception as e:
            logger.error(f"[TwoStage] 两阶段检索失败: {e}")
            # 降级到直接检索
            return await self._direct_retrieve(query, chunk_top_k)

    async def _direct_retrieve(
        self,
        query: str,
        top_k: int
    ) -> Dict[str, Any]:
        """直接检索 chunks（原始策略）"""
        try:
            # 使用 HybridIndexManager 的搜索接口
            if hasattr(self._chunk_index, 'search'):
                results = await self._chunk_index.search(query, top_k=top_k)
            else:
                results = []

            return {
                "type": "direct",
                "papers": [],
                "chunks": results,
                "query": query
            }

        except Exception as e:
            logger.error(f"[TwoStage] 直接检索失败: {e}")
            return {
                "type": "direct",
                "papers": [],
                "chunks": [],
                "query": query
            }


# ============================================================================
# 便捷函数
# ============================================================================

async def create_abstract_index(config: Optional[Dict[str, Any]] = None) -> AbstractIndexManager:
    """
    创建摘要索引管理器

    Args:
        config: 配置 {
            "milvus_uri": "...",
            "collection_name": "...",
            "embed_dim": 1024,
            "embed_model": ...  # embedding 模型
        }
    """
    config = config or {}

    embed_dim = config.get('embed_dim', 1024)
    milvus_uri = config.get('milvus_uri', "./data/milvus_abstracts.db")
    collection_name = config.get('collection_name', "paper_abstracts")

    manager = AbstractIndexManager(
        milvus_uri=milvus_uri,
        collection_name=collection_name,
        embed_dim=embed_dim,
    )

    if config.get('embed_model'):
        manager.set_embed_model(config['embed_model'])

    await manager.initialize()
    return manager
