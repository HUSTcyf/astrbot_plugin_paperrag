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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import httpx
from rapidfuzz import fuzz

try:
    from rag.paper_link_resolver import LinkResolution, PaperLinkResolver
except ImportError:
    from paper_link_resolver import LinkResolution, PaperLinkResolver

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


class OpenAlexAPIClient:
    """OpenAlex API 客户端 - 抗干扰标题模糊匹配"""

    REQUEST_DELAY = 0.6  # OpenAlex 免费限流：100次/分钟
    EMAIL = "astrbot@local"  # 附邮箱可提速至1000次/分钟

    def __init__(self):
        self._last_request_time = 0.0

    def _normalize(self, title: str) -> str:
        """基础清洗 + LaTeX 符号转换"""
        # 转换常见 LaTeX 符号为 ASCII
        title = title.replace('$π_0$', 'pi0').replace('$π_0.5$', 'pi05')
        title = title.replace('$π$', 'pi').replace('π', 'pi')
        # 去除 LaTeX 格式残留
        title = re.sub(r'\$([^$]+)\$', r'\1', title)
        title = re.sub(r'[^\w\s]', ' ', title).strip().lower()
        return title

    async def search_by_title(self, title: str, limit: int = 5) -> list:
        """使用 OpenAlex 搜索论文（自动抗干扰）"""
        elapsed = asyncio.get_event_loop().time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            await asyncio.sleep(self.REQUEST_DELAY - elapsed)

        try:
            import pyalex
            from pyalex import Works

            # 设置邮箱以提高限流（100->1000次/分钟）
            pyalex.config.email = self.EMAIL

            clean_q = self._normalize(title)

            # OpenAlex 搜索默认按相关性排序，标题权重最高
            works = Works().search(clean_q).get(per_page=limit)

            results = []
            for w in works:
                w = cast(dict, w)
                pub_title = w.get("title", "")
                if not pub_title:
                    continue
                arxiv_id = w.get("arxiv_id", "")
                results.append({
                    'title': pub_title,
                    'arxiv_id': arxiv_id,
                    'summary': w.get("abstract", "")[:200] if w.get("abstract") else "",
                    'doi': w.get("doi", ""),
                })

            self._last_request_time = asyncio.get_event_loop().time()
            return results

        except Exception as e:
            logger.debug(f"OpenAlex 搜索失败: {e}")
            return []

    def extract_arxiv_url(self, work: dict) -> Optional[str]:
        """从搜索结果提取 arxiv URL（优先用 arxiv_id，否则从 DOI 解析）"""
        arxiv_id = work.get('arxiv_id', '')
        if arxiv_id:
            return f"https://arxiv.org/abs/{arxiv_id}"
        # 从 DOI 解析（如 https://doi.org/10.48550/arxiv.2105.05233）
        doi = work.get('doi', '') or ''
        if 'arxiv.' in doi:
            arxiv_id = doi.split('arxiv.')[-1]
            return f"https://arxiv.org/abs/{arxiv_id}"
        return None

    def extract_doi_url(self, work: dict) -> Optional[str]:
        """从搜索结果提取 DOI 链接（会议/期刊版本）"""
        doi = work.get('doi', '') or ''
        if doi:
            return doi
        return None

    async def _arxiv_library_fallback(self, title: str) -> Tuple[Optional[str], Optional[str]]:
        """arXiv 库 fallback：直接用 arxiv 库搜索"""
        try:
            import arxiv

            loop = asyncio.get_event_loop()
            elapsed = loop.time() - self._last_request_time
            if elapsed < self.REQUEST_DELAY:
                await asyncio.sleep(self.REQUEST_DELAY - elapsed)

            client = arxiv.Client()
            # 使用标题搜索，在 executor 中执行避免阻塞
            search = arxiv.Search(query=title, max_results=5)
            results = await loop.run_in_executor(None, lambda: list(client.results(search)))

            if results:
                paper_id = results[0].entry_id.split('/')[-1]
                paper_id = re.sub(r'v\d+$', '', paper_id)
                self._last_request_time = loop.time()
                return f"https://arxiv.org/abs/{paper_id}", results[0].title

            return None, None

        except Exception as e:
            logger.debug(f"arXiv library fallback 失败: {e}")
            return None, None

    async def get_arxiv_by_title(self, title: str, threshold: float = 75) -> Tuple[Optional[str], Optional[str]]:
        """根据标题获取 arxiv 链接，使用 rapidfuzz 二次验证"""

        works = await self.search_by_title(title)
        best_work = None
        best_score = 0

        for w in works:
            pub_title = w.get('title', '')
            score = fuzz.token_set_ratio(self._normalize(title), self._normalize(pub_title))
            if score > best_score:
                best_score = score
                best_work = w

        if best_work is None or best_score < threshold:
            return None, None

        arxiv_url = self.extract_arxiv_url(best_work)
        if arxiv_url:
            return arxiv_url, None
        return None, None


def find_best_title_match(query_title: str, works: list, threshold: int = 80) -> tuple:
    """从候选结果中找到 rapidfuzz 相似度最高的（OpenAlex已做初步筛选）"""

    best_work = None
    best_score = 0

    def norm(t: str) -> str:
        """与 OpenAlexAPIClient._normalize 保持一致"""
        t = t.replace('$π_0$', 'pi0').replace('$π_0.5$', 'pi05')
        t = t.replace('$π$', 'pi').replace('π', 'pi')
        t = re.sub(r'\$([^$]+)\$', r'\1', t)
        t = re.sub(r'[^\w\s]', ' ', t).strip().lower()
        return t

    for work in works:
        result_title = work.get('title', '')
        score = fuzz.token_set_ratio(norm(query_title), norm(result_title))
        if score > best_score:
            best_score = score
            best_work = work

    if best_score < threshold:
        return None, 0
    return best_work, best_score / 100.0


# ============================================================================
# CORE API 客户端 - 查询论文的 arxiv 链接（作为 fallback）
# ============================================================================

class CoreAPIClient:
    """CORE API v3 客户端 - 用于搜索学术论文的 arxiv 链接"""

    BASE_URL = "https://api.core.ac.uk/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def search_by_title(self, title: str, limit: int = 3) -> list:
        """根据论文标题搜索论文（宽松匹配，去除标点符号）"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 预处理标题：去除多余标点，保留核心关键词
                # 去除冒号、括号内的内容（容易导致匹配失败）
                clean_title = re.sub(r'[:\(\[【【].*?[)\]】】]', '', title)
                # 去除多余空格
                clean_title = ' '.join(clean_title.split())
                # 去掉末尾的标点符号
                clean_title = clean_title.rstrip('.,;:')

                response = await client.post(
                    f"{self.BASE_URL}/search/works",
                    headers=self.headers,
                    json={"q": clean_title, "limit": limit}
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                if results:
                    return results
        except Exception as e:
            logger.warning(f"CORE API 搜索失败: {e}")
        return []

    def extract_arxiv_url(self, work: dict) -> Optional[str]:
        """从 work 记录提取 arxiv URL"""
        # 方法1：直接获取 arxivId 字段（CORE API 返回的驼峰字段名）
        arxiv_id = work.get("arxivId", "")
        if arxiv_id:
            # 去除版本号（如 2301.12345v1 -> 2301.12345）
            arxiv_id = re.sub(r'v\d+$', '', str(arxiv_id))
            return f"https://arxiv.org/abs/{arxiv_id}"

        # 方法2：扫描 sourceFulltextUrls
        urls = work.get("sourceFulltextUrls", []) or []
        for url in urls:
            if "arxiv.org" in str(url):
                match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', str(url))
                if match:
                    return f"https://arxiv.org/abs/{match.group(1)}"

        # 方法3：扫描 identifiers 数组
        identifiers = work.get("identifiers", []) or []
        for ident in identifiers:
            if isinstance(ident, dict) and ident.get("type") == "ARXIV_ID":
                arxiv_id = str(ident.get("identifier", ""))
                if arxiv_id:
                    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
                    return f"https://arxiv.org/abs/{arxiv_id}"

        return None

    def extract_github_url(self, work: dict) -> Optional[str]:
        """从 work 记录提取 GitHub URL"""
        # 优先从 sourceFulltextUrls 查找
        urls = work.get("sourceFulltextUrls", []) or []
        for url in urls:
            if url and "github.com" in str(url).lower():
                match = re.search(r'github\.com/[\w\-]+/[\w\-]+', str(url), re.IGNORECASE)
                if match:
                    return f"https://{match.group()}"

        # 降级：检查 downloadUrl
        download = work.get("downloadUrl", "") or ""
        if "github.com" in str(download).lower():
            match = re.search(r'github\.com/[\w\-]+/[\w\-]+', str(download), re.IGNORECASE)
            if match:
                return f"https://{match.group()}"
        return None

    async def get_arxiv_by_title(self, title: str, threshold: float = 0.6) -> Tuple[Optional[str], Optional[str]]:
        """根据标题获取 arxiv 和 GitHub 链接（选择相似度最高的结果，低于阈值则放弃）"""
        works = await self.search_by_title(title)
        best_work, best_score = find_best_title_match(title, works)

        if best_work is None or best_score < threshold:
            return None, None

        arxiv_url = self.extract_arxiv_url(best_work)
        if arxiv_url:
            github_url = self.extract_github_url(best_work)
            logger.debug(f"  → 标题匹配度: {best_score:.2%}")
            return arxiv_url, github_url
        return None, None


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
                    full_text += str(text) + "\n"

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
    # arxiv_url 和 github_url 存储在 metadata 中


# ============================================================================
# 本地 GGUF LLM 客户端（用于提取摘要）
# ============================================================================

class LocalGGUFClient:
    """使用本地 GGUF 模型提取摘要，优先复用已加载的 LlamaCppVLMProvider"""

    # 默认模型路径
    DEFAULT_MODEL_PATH = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    DEFAULT_MMproj_PATH = "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"

    def __init__(self, model_path: Optional[str] = None, mmproj_path: Optional[str] = None):
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
            from provider.llama_cpp_vlm import get_cached_llama_cpp_provider
            provider = get_cached_llama_cpp_provider()
            if provider is not None and provider._initialized:
                logger.info(f"✅ 检测到已加载的 LlamaCppVLMProvider 模型: {provider.model_path}")
                self._llama = provider.get_llama()
                self._is_loaded = True
                return True
            return False
        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"检查 LlamaCppVLMProvider 失败: {e}")
            return False

    async def load(self) -> bool:
        """加载 GGUF 模型（通过共享 provider）"""
        if self._is_loaded and self._llama is not None:
            logger.info("✅ GGUF 模型已在内存中，直接复用")
            return True

        try:
            from provider.llama_cpp_vlm import init_llama_cpp_vlm_provider

            provider = init_llama_cpp_vlm_provider(
                model_path=self._model_path,
                mmproj_path=self._mmproj_path,
                n_ctx=4096,
                n_gpu_layers=99,
            )
            await provider.initialize()
            self._llama = provider.get_llama()
            self._is_loaded = True
            logger.success(f"✅ GGUF 模型加载成功: {provider.model_path}")
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

        content: str = ""
        try:
            llama = self._llama
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                lambda: llama.create_chat_completion(  # type: ignore[union-attr]
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
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
        embed_client: Optional[OllamaEmbeddingClient] = None,
        llm_client: Optional[LocalGGUFClient] = None,
        core_api_key: Optional[str] = None,
        use_arxiv_api: bool = True,
    ):
        self._db_path = milvus_uri
        self._collection_name = collection_name
        self._dim = embed_dim
        self._embed_client = embed_client
        self._llm_client = llm_client
        self._core_api_key = core_api_key
        self._use_arxiv_api = use_arxiv_api
        self._core_client: Optional[CoreAPIClient] = None
        self._arxiv_client: Optional[OpenAlexAPIClient] = None
        self._link_resolver = PaperLinkResolver(
            core_api_key=self._core_api_key or "",
            enable_crossref=True,
            enable_openalex=True,
            enable_arxiv_library=self._use_arxiv_api,
            log_prefix="[BuildAbstract]",
        )
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
        abstract_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """为单篇论文建立摘要索引

        提取策略：优先使用本地 LLM 提取标题和摘要，规则提取只做兜底
        """
        await self._ensure_collection()

        try:
            extractor = AbstractExtractor()

            # LLM 同时提取标题和摘要（优先）
            if self._llm_client is not None and (not abstract_text or not title):
                paper_beginning = self._extract_paper_beginning(pdf_path)
                if paper_beginning:
                    llm_title, llm_abstract = await self._llm_client.extract_title_and_abstract(paper_beginning)
                    if llm_title:
                        title = llm_title
                    if llm_abstract and len(llm_abstract) >= 30:
                        abstract_text = llm_abstract

            # 常规提取（回退）- 标题和摘要分别提取
            if not title:
                title = extractor.extract_title_from_pdf(pdf_path) or ""

            if not abstract_text or len(abstract_text) < 50:
                abstract_text = extractor.extract_abstract_from_pdf(pdf_path)

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
                lambda: self._collection.insert(data=[{  # type: ignore[union-attr]
                    "paper_id": paper_id,
                    "file_name": file_name,
                    "title": title,
                    "abstract_text": abstract_text,
                    "vector": vector,
                }])
            )
            # 刷新确保数据持久化
            await loop.run_in_executor(None, lambda: self._collection.flush())  # type: ignore[union-attr]

            # 获取 arxiv / github / doi 链接
            existing_meta = self._abstract_cache.get(paper_id)
            resolution = None
            if not existing_meta or not existing_meta.metadata.get("arxiv_url"):
                resolution = await self._link_resolver.resolve_from_pdf(pdf_path, title_hint=title)

            # 将链接存入 metadata
            paper_metadata = dict(metadata) if metadata else {}
            if resolution is not None:
                if resolution.arxiv_url:
                    paper_metadata["arxiv_url"] = resolution.arxiv_url
                if resolution.github_url:
                    paper_metadata["github_url"] = resolution.github_url
                if resolution.doi_url:
                    paper_metadata["doi_url"] = resolution.doi_url
                if resolution.resolution_source:
                    paper_metadata["resolution_source"] = resolution.resolution_source
                if resolution.resolution_score:
                    paper_metadata["resolution_score"] = resolution.resolution_score
                if resolution.matched_title:
                    paper_metadata["matched_title"] = resolution.matched_title
                if resolution.matched_identifier:
                    paper_metadata["matched_identifier"] = resolution.matched_identifier

            # 更新缓存
            self._abstract_cache[paper_id] = PaperAbstract(
                paper_id=paper_id,
                file_name=file_name,
                title=title,
                abstract_text=abstract_text,
                vector=vector,
                metadata=paper_metadata,
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
                    full_text += str(text) + "\n"
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
            assert self._embed_client is not None
            query_vector = await self._embed_client.get_text_embedding(query)

            search_params = {
                "metric_type": "COSINE",
                "params": {}
            }

            # 使用 Collection API 搜索，包装在 run_in_executor 中避免阻塞
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._collection.search(  # type: ignore[union-attr]
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["paper_id", "file_name", "title", "abstract_text"]
                )
            )

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
            logger.error(f"摘要检索失败: {e}")
            return []


# ============================================================================
# 主程序
# ============================================================================

def _get_core_api_key_from_config() -> Optional[str]:
    """从配置文件读取 CORE API Key"""
    config_paths = [
        Path(__file__).parent.parent.parent / "config" / "astrbot_plugin_paperrag_config.json",
        Path.home() / "AstrBot" / "data" / "config" / "astrbot_plugin_paperrag_config.json",
    ]
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8-sig") as f:
                    config = json.load(f)
                key = config.get("core_api_key", "")
                if key:
                    return key
            except Exception:
                pass
    return None


def _get_freeapi_config() -> dict:
    """从配置文件读取 freeapi 配置"""
    config_paths = [
        Path(__file__).parent.parent.parent / "config" / "astrbot_plugin_paperrag_config.json",
        Path.home() / "AstrBot" / "data" / "config" / "astrbot_plugin_paperrag_config.json",
    ]
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8-sig") as f:
                    config = json.load(f)
                return {
                    "freeapi_url": config.get("freeapi_url", ""),
                    "freeapi_key": config.get("freeapi_key", ""),
                }
            except Exception:
                pass
    return {"freeapi_url": "", "freeapi_key": ""}


async def build_abstract_index(
    papers_dir: str,
    ollama_url: str,
    ollama_model: str,
    embed_dim: int,
    skip: int = 0,
    force: bool = False,
    rebuild: bool = False,
    core_api_key: Optional[str] = None,
    update_links_only: bool = False,
    use_arxiv_api: bool = True,
):
    """构建摘要索引"""

    # 优先使用传入的 key，否则从配置文件读取
    if not core_api_key:
        core_api_key = _get_core_api_key_from_config()

    # 初始化 pdf_files（可能为空列表）
    pdf_files = []

    # 仅更新链接模式不需要 papers 目录
    if not update_links_only:
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            logger.error(f"论文目录不存在: {papers_dir}")
            return

        # 查找所有 PDF 文件
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
        if await llm_client.load():
            logger.success("GGUF LLM 已就绪")
        else:
            logger.warning("GGUF LLM 加载失败，将使用规则提取作为兜底")

    # 初始化摘要索引管理器
    plugin_dir = Path(__file__).parent
    milvus_uri = str(plugin_dir / "data" / "milvus_abstracts.db")

    abstract_index = AbstractIndexManager(
        milvus_uri=milvus_uri,
        collection_name="paper_abstracts",
        embed_dim=embed_dim,
        core_api_key=core_api_key,
        use_arxiv_api=use_arxiv_api,
    )
    abstract_index.set_embed_client(embed_client)
    abstract_index.set_llm_client(llm_client)

    # 重建模式
    if rebuild:
        logger.info("🔄 清除旧数据...")
        from pymilvus import connections, utility
        try:
            connections.connect(alias="abstract_index", uri=milvus_uri)  # type: ignore[func-returns-value]
            if utility.has_collection("paper_abstracts", using="abstract_index"):
                utility.drop_collection("paper_abstracts", using="abstract_index")  # type: ignore[func-returns-value]
        except Exception as e:
            logger.warning(f"清除旧数据出错: {e}")
        doc_stats_path = milvus_uri.replace('.db', '_doc_stats.json')
        if os.path.exists(doc_stats_path):
            os.remove(doc_stats_path)

    await abstract_index.initialize()

    # 两种模式分离处理
    if update_links_only:
        # 仅更新链接模式（不提取摘要，不覆盖原数据）
        logger.info("🔗 仅更新论文链接模式")
        await embed_client.close()  # 不需要 embedding 服务
        await llm_client.close()

        existing = await abstract_index.get_all_abstracts()
        if not existing:
            logger.warning("没有找到已索引的论文，请先运行普通索引构建")
            return

        logger.info(f"找到 {len(existing)} 篇已索引的论文")

        # 初始化 OpenAlex API + arXiv 库 fallback
        arxiv_client = OpenAlexAPIClient()

        results = {"updated": 0, "skipped": 0, "failed": 0}
        # 详细记录每个论文的处理结果
        details = []  # list of dict: {paper_id, title, old_url, new_url, similarity, reason, api}
        start_time = time.time()

        for i, (paper_id, abstract) in enumerate(existing.items()):
            old_url = abstract.metadata.get("arxiv_url", "")
            detail = {
                "paper_id": paper_id,
                "title": abstract.title,
                "old_url": old_url,
                "new_url": "",
                "similarity": 0.0,
                "reason": "",
                "api": "",
            }

            # 如果已有 arxiv_url 且非 force 模式，跳过
            if abstract.metadata.get("arxiv_url") and not force:
                detail["reason"] = "已有链接，未强制更新"
                detail["new_url"] = old_url
                results["skipped"] += 1
                details.append(detail)
                continue

            logger.info(f"[{i+1}/{len(existing)}] 查询链接: {abstract.title[:50]}...")

            try:
                resolution = await abstract_index._link_resolver.resolve_by_title(abstract.title)
                arxiv_url = resolution.arxiv_url
                github_url = resolution.github_url
                doi_url = resolution.doi_url
                best_similarity = resolution.resolution_score / 100.0
                used_api = resolution.resolution_source or resolution.backend

                detail["similarity"] = best_similarity
                detail["api"] = used_api

                if arxiv_url or github_url or doi_url:
                    # 更新 metadata（保留原有数据）
                    metadata = dict(abstract.metadata)
                    if arxiv_url:
                        metadata["arxiv_url"] = arxiv_url
                    if doi_url:
                        metadata["doi_url"] = doi_url
                    if github_url:
                        metadata["github_url"] = github_url

                    abstract_index._abstract_cache[paper_id] = PaperAbstract(
                        paper_id=abstract.paper_id,
                        file_name=abstract.file_name,
                        title=abstract.title,
                        abstract_text=abstract.abstract_text,
                        vector=abstract.vector,
                        metadata=metadata,
                    )
                    abstract_index._save_doc_stats()
                    results["updated"] += 1
                    detail["new_url"] = arxiv_url or doi_url or github_url
                    detail["reason"] = "更新成功"
                elif best_similarity > 0:
                    # 找到匹配但无链接（可能是 OpenAlex 有结果但没有 DOI）
                    detail["reason"] = f"找到匹配但无链接 ({best_similarity:.1%})"
                    detail["new_url"] = old_url
                    results["skipped"] += 1
                else:
                    detail["reason"] = "未找到匹配结果"
                    detail["new_url"] = old_url
                    results["skipped"] += 1
            except Exception as e:
                logger.error(f"  → 查询失败: {e}")
                detail["reason"] = f"查询异常: {e}"
                results["failed"] += 1

            details.append(detail)

        elapsed = time.time() - start_time
        print()
        logger.success("=" * 60)
        logger.success("链接更新完成")
        print(f"  更新: {results['updated']} | 跳过: {results['skipped']} | 失败: {results['failed']}")
        print(f"  耗时: {elapsed:.1f}s")

        # 打印详细汇总
        print()
        print("=" * 60)
        print("📋 详细汇总")
        print("=" * 60)

        # 更新成功的
        updated_list = [d for d in details if d["new_url"] and d["reason"] == "更新成功"]
        if updated_list:
            print(f"\n✅ 更新成功 ({len(updated_list)} 篇):")
            for d in updated_list:
                print(f"  - {d['title'][:50]}...")
                print(f"    相似度: {d['similarity']:.1%} | API: {d['api']}")
                print(f"    新链接: {d['new_url']}")

        # 跳过的 - 进一步分类
        skipped_list = [d for d in details if d["reason"] and d["reason"] != "更新成功"]
        if skipped_list:
            # 按原因分类
            already_has_url_list = [d for d in skipped_list if "已有链接" in d["reason"]]
            no_match_list = [d for d in skipped_list if "未找到匹配" in d["reason"]]
            other_skipped = [d for d in skipped_list if d not in already_has_url_list and d not in no_match_list]

            if already_has_url_list:
                print(f"\n🔗 已有链接跳过 ({len(already_has_url_list)} 篇):")
                for d in already_has_url_list:
                    print(f"  - {d['title'][:50]}...")
                    print(f"    相似度: {d['similarity']:.1%}")
                    if d["old_url"]:
                        print(f"    原链接: {d['old_url']}")

            if no_match_list:
                print(f"\n🔍 未找到匹配 ({len(no_match_list)} 篇):")
                for d in no_match_list:
                    print(f"  - {d['title'][:50]}...")
                    print(f"    相似度: {d['similarity']:.1%}")

            if other_skipped:
                print(f"\n⏭️ 其他原因跳过 ({len(other_skipped)} 篇):")
                for d in other_skipped:
                    print(f"  - {d['title'][:50]}...")
                    print(f"    原因: {d['reason']} | 相似度: {d['similarity']:.1%}")

        # 失败的
        failed_list = [d for d in details if "异常" in d["reason"] or "失败" in d["reason"]]
        if failed_list:
            print(f"\n❌ 查询失败 ({len(failed_list)} 篇):")
            for d in failed_list:
                print(f"  - {d['title'][:50]}...")
                print(f"    原因: {d['reason']}")

    else:
        # 正常索引模式
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
  python build_abstract_index.py --papers ./papers --update-links-only  # 仅更新 arxiv/github 链接

说明：
  - Embedding 使用 Ollama 服务（bge-m3 模型）
  - 摘要 LLM 提取使用本地 Qwen3.5-9B-GGUF 模型
  - 若 GGUF 模型已在 AstrBot 中加载，将直接复用
  - --update-links-only 只更新 metadata（arxiv/github 链接），不覆盖原 title/abstract
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

    parser.add_argument(
        '--core-api-key',
        default=os.environ.get('CORE_API_KEY', ''),
        help='CORE API Key（用于查询 arxiv 链接）'
    )

    parser.add_argument(
        '--update-links-only', '-u',
        action='store_true',
        help='仅更新链接（不提取摘要，不覆盖原数据）'
    )

    parser.add_argument(
        '--no-arxiv-api',
        action='store_true',
        help='禁用 arXiv API 查询（当 arXiv 被限流时使用，仅依赖 CORE API）'
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
        core_api_key=args.core_api_key or None,
        update_links_only=args.update_links_only,
        use_arxiv_api=not args.no_arxiv_api,
    ))


if __name__ == "__main__":
    main()
