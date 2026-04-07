#!/usr/bin/env python3
"""
独立运行的摘要索引构建脚本

功能：
- 不依赖 AstrBot，独立运行
- 为 papers 目录下的 PDF 构建摘要索引
- 使用 Ollama 提供 embedding 服务

用法：
    cd /path/to/astrbot_plugin_paperrag
    python build_abstract_index.py --papers ./papers --ollama http://localhost:11434

依赖：
    pip install pymilvus httpx pymupdf
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# ============================================================================
# 简单日志
# ============================================================================

class SimpleLogger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _print(self, prefix: str, msg: str):
        """带刷新的打印"""
        print(f"{prefix} {msg}", flush=True)

    def info(self, msg: str):
        if self.verbose:
            self._print("ℹ️", msg)

    def warning(self, msg: str):
        self._print("⚠️", msg)

    def error(self, msg: str):
        self._print("❌", msg)

    def success(self, msg: str):
        self._print("✅", msg)

    def debug(self, msg: str):
        if self.verbose:
            self._print("  ", msg)

logger = SimpleLogger()


# ============================================================================
# 摘要提取器（复用 abstract_index.py 的逻辑）
# ============================================================================

class AbstractExtractor:
    """从 PDF 中提取摘要"""

    ABSTRACT_KEYWORDS = [
        r'^abstract\s*$',
        r'^摘要\s*$',
        r'^summary\s*$',
        r'^概述\s*$',
        r'^abstract.', 
        r'^abstract[—\-:]\s*$',  # Abstract:, Abstract—, Abstract-
        r'^ABSTRACT[—\-:]\s*$',  # ABSTRACT:, ABSTRACT—, ABSTRACT-
        r'^ABSTRACT\s*$',
    ]

    INTRODUCTION_KEYWORDS = [
        r'^1\.?\s*introduction\s*$',
        r'^1\.?\s*引言\s*$',
        r'^introduction\s*$',
        r'^1\s+[A-Z]',
        r'^一、引言',
        r'^1\.?\s+[A-Z][a-z]+',
        r'^references\s*$',
        r'^参考文献\s*$',
    ]

    def extract_abstract_from_pdf(self, pdf_path: str) -> Optional[str]:
        """从 PDF 文件提取摘要"""
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            full_text = ""

            for page_num in range(min(len(doc), 5)):
                page = doc[page_num]
                text = page.get_text()
                if text:
                    full_text += text + "\n"

            doc.close()
            if full_text.strip():
                return self._extract_abstract_text(full_text)
            return None

        except Exception as e:
            logger.warning(f"提取摘要失败 {pdf_path}: {e}")
            return None

    def _extract_abstract_text(self, full_text: str) -> Optional[str]:
        """从完整文本中提取摘要部分"""
        if not full_text:
            return None

        lines = full_text.split('\n')
        abstract_start = -1
        abstract_end = -1

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            for pattern in self.ABSTRACT_KEYWORDS:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    if re.match(r'^abstract\s*$', line_stripped, re.IGNORECASE):
                        abstract_start = i + 1
                    else:
                        abstract_start = i
                    break

            if abstract_start >= 0:
                break

        # 回退策略：如果没有找到 Abstract 关键词，尝试查找直接开始正文的格式
        if abstract_start < 0:
            # 尝试找到第一个不以数字开头的长段落（可能是摘要开始）
            abstract_opening_patterns = [
                r'^We\s+',  # We present, We propose, etc.
                r'^This\s+',  # This paper, This work, etc.
                r'^In\s+this\s+paper',
                r'^We\s+introduce',
                r'^This\s+work\s+',
                r'^We\s+describe',
                r'^We\s+develop',
                r'^We\s+show',
                r'^An\s+',
                r'^The\s+',  # The... (but not "The 1." or similar)
            ]
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # 跳过纯数字行（如页码 "1"）
                if re.match(r'^\d+$', line_stripped):
                    continue
                # 检查是否是摘要开头模式
                for pattern in abstract_opening_patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        abstract_start = i
                        break
                if abstract_start >= 0:
                    break

            if abstract_start < 0:
                preview = '\n'.join(lines[:100]) if lines else ''
                logger.warning(f"未找到 Abstract 关键词，前100行内容:\n{preview}")
                return None

        for i in range(abstract_start + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            for pattern in self.INTRODUCTION_KEYWORDS:
                if re.match(pattern, line, re.IGNORECASE):
                    abstract_end = i
                    break
            if abstract_end >= 0:
                break

        if abstract_end < 0:
            paragraph_count = 0
            for i in range(abstract_start, len(lines)):
                if lines[i].strip():
                    paragraph_count += 1
                if paragraph_count >= 3:
                    abstract_end = i + 1
                    break

        if abstract_end < 0:
            abstract_end = len(lines)

        abstract_lines = lines[abstract_start:abstract_end]
        abstract_text = '\n'.join(line.strip() for line in abstract_lines if line.strip())

        abstract_text = re.sub(r'^abstract:\s*', '', abstract_text, flags=re.IGNORECASE)
        abstract_text = abstract_text.strip()

        return abstract_text if abstract_text else None

    def extract_title_from_pdf(self, pdf_path: str) -> Optional[str]:
        """从 PDF 文件提取标题"""
        try:
            import pymupdf

            doc = pymupdf.open(pdf_path)
            metadata = doc.metadata
            if metadata:
                title = metadata.get('title', '')
                if title and title.strip():
                    doc.close()
                    return title.strip()

            doc.close()
            filename = os.path.basename(pdf_path)
            title = os.path.splitext(filename)[0]
            title = re.sub(r'[_-]?(v\d+)?(\.pdf)?$', '', title, flags=re.IGNORECASE)
            return title if title else None

        except Exception as e:
            logger.warning(f"提取标题失败 {pdf_path}: {e}")
            return None


# ============================================================================
# Ollama Embedding 客户端
# ============================================================================

class OllamaEmbeddingClient:
    """直接调用 Ollama API 获取 embeddings，支持连接复用和模型预热"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "bge-m3"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._embed_dim: Optional[int] = None
        self._client: Optional[httpx.AsyncClient] = None  # 复用 HTTP 客户端

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端（连接复用）"""
        if self._client is None:
            # 配置连接池，保持长连接
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                http2=True,
            )
        return self._client

    async def close(self):
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def is_model_loaded(self) -> bool:
        """检查模型是否已加载到内存"""
        try:
            client = await self._get_client()
            response = await client.get("/api/ps")
            response.raise_for_status()
            data = response.json()

            # 检查正在运行的模型中是否有当前模型
            models = data.get("models", [])
            for m in models:
                # 模型名称可能在 name 或 model 字段中
                model_name = m.get("name", "") or m.get("model", "")
                if model_name.startswith(self.model):
                    return True
            return False

        except Exception as e:
            logger.warning(f"检查模型状态失败: {e}")
            return False

    async def warmup(self) -> bool:
        """预热模型（如果未加载则加载到内存）"""
        if await self.is_model_loaded():
            logger.info(f"✅ 模型 {self.model} 已加载到内存")
            return True

        logger.info(f"🔄 模型 {self.model} 未加载，正在预热...")

        try:
            # 通过一次 embedding 调用预热模型
            client = await self._get_client()
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": "warming up"
                }
            )
            response.raise_for_status()

            if self._embed_dim is None:
                result = response.json()
                if "embedding" in result:
                    self._embed_dim = len(result["embedding"])

            logger.success(f"✅ 模型 {self.model} 已预热并加载到内存")
            return True

        except Exception as e:
            logger.error(f"模型预热失败: {e}")
            return False

    async def get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding"""
        client = await self._get_client()

        response = await client.post(
            "/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            }
        )
        response.raise_for_status()
        result = response.json()

        if "embedding" not in result:
            raise ValueError(f"Ollama 响应缺少 embedding 字段: {result}")

        embedding = result["embedding"]

        if self._embed_dim is None:
            self._embed_dim = len(embedding)
            logger.info(f"Ollama Embedding 向量维度: {self._embed_dim}")

        return embedding

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            return 1024  # BGE-M3 默认
        return self._embed_dim


# ============================================================================
# Milvus 摘要索引管理器
# ============================================================================

@dataclass
class PaperAbstract:
    paper_id: str
    file_name: str
    title: str = ""
    abstract_text: str = ""
    vector: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 本地 GGUF LLM 客户端（用于提取摘要）
# ============================================================================

class LocalGGUFClient:
    """使用本地 GGUF 模型提取摘要，优先复用已加载的 LlamaCppVLMProvider"""

    # 默认模型路径
    DEFAULT_MODEL_PATH = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    DEFAULT_MMproj_PATH = "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"

    def __init__(self, model_path: str = None, mmproj_path: str = None):
        self._model_path = model_path or self.DEFAULT_MODEL_PATH
        self._mmproj_path = mmproj_path or self.DEFAULT_MMproj_PATH
        self._llama: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._is_loaded = False

    def _resolve_path(self, path: str) -> str:
        """解析模型路径（相对于插件目录）"""
        if os.path.isabs(path):
            return path
        # 相对于插件目录
        plugin_dir = Path(__file__).parent.resolve()
        return str(plugin_dir / path)

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载（通过 LlamaCppVLMProvider 单例）"""
        try:
            # 尝试获取已缓存的 Provider
            from llama_cpp_vlm_provider import get_cached_llama_cpp_provider
            provider = get_cached_llama_cpp_provider()
            if provider is not None and provider._initialized and provider._llama is not None:
                logger.info(f"✅ 检测到已加载的 LlamaCppVLMProvider 模型: {provider.model_path}")
                self._llama = provider._llama
                self._is_loaded = True
                return True
            return False
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"检查 LlamaCppVLMProvider 失败: {e}")
            return False

    async def load(self) -> bool:
        """加载 GGUF 模型"""
        if self._is_loaded and self._llama is not None:
            logger.info("✅ GGUF 模型已在内存中，直接复用")
            return True

        model_path = self._resolve_path(self._model_path)
        mmproj_path = self._resolve_path(self._mmproj_path)

        # 检查文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
        if not os.path.exists(mmproj_path):
            logger.error(f"mmproj 文件不存在: {mmproj_path}")
            return False

        logger.info(f"🔄 正在加载 GGUF 模型: {model_path}")
        logger.info(f"   mmproj: {mmproj_path}")

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
            logger.success(f"✅ GGUF 模型加载成功")
            return True

        except Exception as e:
            logger.error(f"❌ GGUF 模型加载失败: {e}")
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

        try:
            llama = self._llama
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                lambda: llama.create_chat_completion(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
            )

            content = result["choices"][0]["message"]["content"].strip()

            # 解析 JSON
            import json
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

    async def close(self):
        """关闭模型（如果是我们自己加载的）"""
        # LlamaCppVLMProvider 单例不应在此关闭
        # 只有我们自己加载的模型才需要清理
        if self._llama is not None and not self.is_model_loaded():
            self._llama = None
            self._is_loaded = False
            logger.debug("本地 GGUF 模型已卸载")


class AbstractIndexManager:
    """摘要索引管理器"""

    def __init__(
        self,
        milvus_uri: str = "./data/milvus_abstracts.db",
        collection_name: str = "paper_abstracts",
        embed_dim: int = 1024,
        embed_client: OllamaEmbeddingClient = None,
        llm_client: LocalGGUFClient = None,
    ):
        self._db_path = milvus_uri
        self._collection_name = collection_name
        self._dim = embed_dim
        self._embed_client = embed_client
        self._llm_client = llm_client
        self._is_connected = False
        self._collection = None
        self._abstract_cache: Dict[str, PaperAbstract] = {}

        db_dir = os.path.dirname(self._db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._doc_stats_file = self._db_path.replace('.db', '_doc_stats.json')
        logger.info(f"AbstractIndexManager 初始化完成 (collection={collection_name}, dim={embed_dim})")

    def set_embed_client(self, embed_client: OllamaEmbeddingClient):
        self._embed_client = embed_client

    def set_llm_client(self, llm_client: LocalGGUFClient):
        self._llm_client = llm_client

    async def initialize(self):
        await self._ensure_collection()

    async def _ensure_collection(self):
        if self._collection is not None:
            return

        try:
            import asyncio
            from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
            from typing import cast

            db_dir = os.path.dirname(self._db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            connections.connect(alias="abstract_index", uri=self._db_path)

            if utility.has_collection(self._collection_name, using="abstract_index"):
                self._collection = cast(Collection, Collection(self._collection_name, using="abstract_index"))
                collection = cast(Collection, self._collection)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, collection.load)
                self._is_connected = True
                await self._load_doc_stats()
                return

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
                using="abstract_index"
            )
            collection = cast(Collection, self._collection)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: collection.create_index(
                    field_name="vector",
                    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
                )
            )
            await loop.run_in_executor(None, collection.load)

            self._is_connected = True
            logger.info(f"创建新 collection: {self._collection_name}")

        except Exception as e:
            logger.error(f"初始化 AbstractIndex 失败: {e}")
            raise

    async def _load_doc_stats(self):
        """加载文档统计"""
        if not self._doc_stats_file or not os.path.exists(self._doc_stats_file):
            self._abstract_cache = {}
            return

        try:
            with open(self._doc_stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._abstract_cache = {
                    k: PaperAbstract(**v) for k, v in data.get('abstracts', {}).items()
                }
            logger.info(f"已加载 {len(self._abstract_cache)} 篇论文摘要")
        except Exception as e:
            logger.warning(f"加载摘要统计失败: {e}")
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
            logger.warning(f"保存摘要统计失败: {e}")

    async def index_paper(
        self,
        pdf_path: str,
        paper_id: str,
        file_name: str,
        title: str = "",
        abstract_text: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """为单篇论文建立摘要索引

        提取策略：默认使用 LLM 同时提取标题和摘要，失败时使用常规提取
        """
        await self._ensure_collection()

        try:
            extractor = AbstractExtractor()

            # LLM 同时提取标题和摘要（优先）
            if self._llm_client is not None and (not abstract_text or not title):
                paper_beginning = self._extract_paper_beginning(pdf_path)
                if paper_beginning:
                    llm_title, llm_abstract = await self._llm_client.extract_title_and_abstract(paper_beginning)
                    if llm_abstract and len(llm_abstract) >= 30:
                        abstract_text = llm_abstract
                    if llm_title:
                        title = llm_title

            # 常规提取（回退）- 标题和摘要分别提取
            if not abstract_text or len(abstract_text) < 50:
                abstract_text = extractor.extract_abstract_from_pdf(pdf_path)

            if not title:
                title = extractor.extract_title_from_pdf(pdf_path)

            if not abstract_text or len(abstract_text) < 50:
                logger.warning(f"论文 {file_name} 未找到有效摘要，跳过")
                return False

            if not title:
                title = file_name.replace('.pdf', '').replace('.PDF', '')

            if self._embed_client is None:
                logger.error("未设置 embedding 客户端")
                return False

            # 组合标题和摘要作为检索文本
            combined_text = f"{title}\n\n{abstract_text}"
            vector = await self._embed_client.get_text_embedding(combined_text)

            # 存储到 Milvus（使用 Collection API，包装在 run_in_executor 中避免阻塞）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._collection.insert(data=[{
                    "paper_id": paper_id,
                    "file_name": file_name,
                    "title": title,
                    "abstract_text": abstract_text,
                    "vector": vector,
                }])
            )
            # 刷新确保数据持久化
            await loop.run_in_executor(None, lambda: self._collection.flush())

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

            logger.success(f"已索引摘要: {title} ({len(abstract_text)} chars)")
            return True

        except Exception as e:
            logger.error(f"索引摘要失败 {file_name}: {e}")
            return False

    def _extract_paper_beginning(self, pdf_path: str, max_chars: int = 3000) -> Optional[str]:
        """提取论文开头部分（用于 LLM 提取摘要）"""
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            full_text = ""

            # 提取前几页内容
            for page_num in range(min(len(doc), 5)):
                page = doc[page_num]
                text = page.get_text()
                if text:
                    full_text += text + "\n"
                if len(full_text) >= max_chars:
                    break

            doc.close()
            return full_text[:max_chars] if full_text.strip() else None

        except Exception as e:
            logger.warning(f"提取论文开头失败 {pdf_path}: {e}")
            return None

    async def get_all_abstracts(self) -> Dict[str, PaperAbstract]:
        """获取所有摘要数据"""
        await self._ensure_collection()
        return self._abstract_cache.copy()

    async def search_by_abstract(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """通过摘要向量检索相关论文"""
        await self._ensure_collection()

        try:
            query_vector = await self._embed_client.get_text_embedding(query)

            search_params = {
                "metric_type": "COSINE",
                "params": {}
            }

            # 使用 Collection API 搜索，包装在 run_in_executor 中避免阻塞
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["paper_id", "file_name", "title", "abstract_text"]
                )
            )

            papers = []
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
            logger.error(f"摘要检索失败: {e}")
            return []


# ============================================================================
# 主程序
# ============================================================================

async def build_abstract_index(
    papers_dir: str,
    ollama_url: str,
    ollama_model: str,
    embed_dim: int,
    skip: int = 0,
    force: bool = False,
    rebuild: bool = False,
):
    """构建摘要索引"""

    papers_path = Path(papers_dir)
    if not papers_path.exists():
        logger.error(f"论文目录不存在: {papers_dir}")
        return

    # 查找所有 PDF 文件
    pdf_files = []
    for ext in ['*.pdf', '*.PDF']:
        pdf_files.extend(papers_path.glob(ext))
    pdf_files = [f for f in pdf_files if not f.name.startswith('._')]

    if not pdf_files:
        logger.warning(f"在 {papers_dir} 中未找到 PDF 文件")
        return

    logger.info(f"找到 {len(pdf_files)} 篇 PDF 论文")
    logger.info(f"Ollama Embedding 地址: {ollama_url}")
    logger.info(f"Embedding 模型: {ollama_model}")

    # 初始化 embedding 客户端
    embed_client = OllamaEmbeddingClient(base_url=ollama_url, model=ollama_model)

    # 检查并预热 Embedding 模型
    try:
        # 先检查模型是否已加载
        if await embed_client.is_model_loaded():
            logger.success(f"✅ Embedding 模型 {ollama_model} 已加载到内存（快速模式）")
        else:
            logger.info(f"🔄 Embedding 模型 {ollama_model} 尚未加载，正在预热...")

        # 预热模型（确保加载到内存）
        warmup_success = await embed_client.warmup()
        if not warmup_success:
            raise Exception("模型预热失败")

        # 获取向量维度
        embed_dim = embed_client.embed_dim
        logger.info(f"Embedding 向量维度: {embed_dim}")

    except Exception as e:
        logger.error(f"Ollama Embedding 初始化失败: {e}")
        logger.error("请确保 Ollama 服务正在运行: ollama serve")
        await embed_client.close()
        return

    # 初始化本地 GGUF LLM 客户端（用于摘要提取）
    llm_client = LocalGGUFClient()
    if llm_client.is_model_loaded():
        logger.info("✅ GGUF LLM 模型已加载，直接复用")
    else:
        logger.info("🔄 GGUF LLM 未加载，将在需要时加载")
        await llm_client.load()

    # 初始化摘要索引管理器
    plugin_dir = Path(__file__).parent
    milvus_uri = str(plugin_dir / "data" / "milvus_abstracts.db")

    abstract_index = AbstractIndexManager(
        milvus_uri=milvus_uri,
        collection_name="paper_abstracts",
        embed_dim=embed_dim,
    )
    abstract_index.set_embed_client(embed_client)
    abstract_index.set_llm_client(llm_client)

    # 重建模式
    if rebuild:
        logger.info("🔄 清除旧数据...")
        from pymilvus import connections, utility
        try:
            connections.connect(alias="abstract_index", uri=milvus_uri)
            if utility.has_collection("paper_abstracts", using="abstract_index"):
                utility.drop_collection("paper_abstracts", using="abstract_index")
        except Exception as e:
            logger.warning(f"清除旧数据出错: {e}")
        doc_stats_path = milvus_uri.replace('.db', '_doc_stats.json')
        if os.path.exists(doc_stats_path):
            os.remove(doc_stats_path)

    await abstract_index.initialize()

    existing = await abstract_index.get_all_abstracts()
    processed_ids = set(existing.keys())
    logger.info(f"已存在 {len(processed_ids)} 篇摘要" if not rebuild else "重新构建摘要索引")

    results = {"success": 0, "failed": 0, "skipped": 0}
    start_time = time.time()

    for i, pdf_file in enumerate(pdf_files):
        paper_id = pdf_file.stem
        file_name = pdf_file.name

        if i < skip:
            continue
        if not force and paper_id in processed_ids:
            results["skipped"] += 1
            continue

        logger.info(f"[{i+1}/{len(pdf_files)}] {file_name}")

        try:
            success = await abstract_index.index_paper(
                pdf_path=str(pdf_file),
                paper_id=paper_id,
                file_name=file_name,
            )
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            logger.error(f"处理失败 {file_name}: {e}")
            results["failed"] += 1

    elapsed = time.time() - start_time
    await embed_client.close()

    print()
    logger.success("=" * 40)
    logger.success("摘要索引构建完成")
    print(f"  成功: {results['success']} | 失败: {results['failed']} | 跳过: {results['skipped']}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  位置: {milvus_uri}")


def main():
    parser = argparse.ArgumentParser(
        description="独立运行摘要索引构建脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python build_abstract_index.py --papers ./papers
  python build_abstract_index.py --papers ./papers --skip 30
  python build_abstract_index.py --papers ./papers --ollama http://localhost:11434 --model bge-m3
  python build_abstract_index.py --papers ./papers --force     # 强制重新处理已存在的论文
  python build_abstract_index.py --papers ./papers --rebuild    # 完全重建（删除旧数据库）

说明：
  - Embedding 使用 Ollama 服务（bge-m3 模型）
  - 摘要 LLM 提取使用本地 Qwen3.5-9B-GGUF 模型
  - 若 GGUF 模型已在 AstrBot 中加载，将直接复用
        """
    )

    parser.add_argument(
        '--papers', '-p',
        default='./papers',
        help='论文目录路径 (默认: ./papers)'
    )

    parser.add_argument(
        '--ollama', '-o',
        default='http://localhost:11434',
        help='Ollama 服务地址 (默认: http://localhost:11434)'
    )

    parser.add_argument(
        '--model', '-m',
        default='bge-m3',
        help='Embedding 模型 (默认: bge-m3)'
    )

    parser.add_argument(
        '--skip', '-s',
        type=int,
        default=0,
        help='跳过前 N 篇论文 (默认: 0)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制重新处理已存在的论文（不跳过）'
    )

    parser.add_argument(
        '--rebuild', '-r',
        action='store_true',
        help='完全重建：删除旧数据库，从头开始索引'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='安静模式（减少输出）'
    )

    args = parser.parse_args()

    global logger
    logger = SimpleLogger(verbose=not args.quiet)

    logger.info("=" * 50)
    logger.info("摘要索引构建脚本")
    logger.info("=" * 50)

    asyncio.run(build_abstract_index(
        papers_dir=args.papers,
        ollama_url=args.ollama,
        ollama_model=args.model,
        embed_dim=1024,
        skip=args.skip,
        force=args.force,
        rebuild=args.rebuild,
    ))


if __name__ == "__main__":
    main()
