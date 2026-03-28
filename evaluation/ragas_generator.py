# -*- coding: utf-8 -*-
"""
Ragas 测试集生成器
从已构建的 Milvus 数据库中读取文本 chunks，生成问答对
"""

import asyncio
import json
import sys
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any, Type, Union
from dataclasses import dataclass, asdict, field
from dataclasses import dataclass as dc

# 确保能导入 astrbot.api
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from astrbot.api import logger
except Exception:
    import logging
    logger = logging.getLogger("ragas_generator")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.info = lambda x: print(x)
    logger.warning = lambda x: print(f"WARNING: {x}")
    logger.error = lambda x: print(f"ERROR: {x}")

# ============================================================================
# 懒加载 ragas（评估框架，可选安装）
# ============================================================================

RAGAS_AVAILABLE = True
TestsetGenerator = None
Document = None
BaseRagasLLM = None
BaseRagasEmbeddings = None

try:
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph
    from ragas.testset.synthesizers.generate import default_query_distribution
    from ragas.llms.base import BaseRagasLLM
    from ragas.embeddings.base import BaseRagasEmbeddings
    from llama_index.core import Document

    # 禁用 Ragas 遥测追踪（避免 SSL 证书过期错误）
    import os
    os.environ["RAGAS_DO_NOT_TRACK"] = "True"

except ImportError as e:
    RAGAS_AVAILABLE = False
    logger.warning(f"Ragas 评估框架未安装，部分功能不可用: {e}")


# ============================================================================
# 自定义 Ragas LLM 和 Embedding 包装器（适配 OpenAI 兼容接口）
# ============================================================================

try:
    from langchain_core.prompt_values import StringPromptValue
except ImportError:
    StringPromptValue = None

try:
    from ragas.llms.base import LLMResult, Generation
except ImportError:
    LLMResult = None
    Generation = None


class InvalidLLMResponseError(ValueError):
    """
    LLM 返回内容校验失败的专用异常。

    继承自 ValueError，确保被 ragas 的异常处理链路正确识别和报告。
    包含 .text 属性，适配 ragas 的 raise_first_exception 格式化逻辑。
    """

    def __init__(self, message: str, text: str = ""):
        super().__init__(message)
        self.text = text  # ragas raise_first_exception 会访问此属性


class OpenAICompatibleLLM(BaseRagasLLM if BaseRagasLLM else object):
    """
    适配 OpenAI 兼容接口的 Ragas LLM 包装器
    用于对接 ZhipuAI 等 OpenAI 兼容的 API

    正确实现 BaseRagasLLM 接口：
    - generate_text(prompt: PromptValue, ...) -> LLMResult
    - agenerate_text(prompt: PromptValue, ...) -> LLMResult
    - is_finished(response: LLMResult) -> bool
    - get_temperature(n: Optional[int] = None) -> float
    """

    # 全局并发限制信号量（类变量，所有实例共享）
    _semaphore: Optional[Any] = None
    _max_concurrent: int = 3  # 全局最大并发数

    @classmethod
    def set_max_concurrent(cls, n: int):
        """设置全局最大并发数"""
        cls._max_concurrent = n
        if cls._semaphore is not None:
            cls._semaphore.release()
            cls._semaphore = None

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_concurrent: int = 3,
        **kwargs
    ):
        if BaseRagasLLM:
            super().__init__()
        self.model = model
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._run_config = None
        self._max_concurrent = max_concurrent

    def set_run_config(self, run_config):
        self._run_config = run_config

    def get_temperature(self, n: Optional[int] = None) -> float:
        return self.temperature

    def is_finished(self, response: LLMResult) -> bool:
        """检查 LLM 调用是否完成"""
        if response is None:
            return False
        try:
            if response.generations and any(gen_list for gen_list in response.generations):
                return True
        except Exception:
            pass
        return False

    def _call_api(self, prompt_text: str, temperature: float, max_tokens: int, stop: Optional[list]):
        """同步调用 API（受全局 Semaphore 限制并发）"""
        import requests
        import threading

        # 同步路径也必须限速：获取或创建信号量
        # 注意：同步路径无法直接使用 asyncio.Semaphore，改用 threading.Semaphore
        if not hasattr(OpenAICompatibleLLM, '_sync_semaphore'):
            OpenAICompatibleLLM._sync_semaphore = threading.Semaphore(self._max_concurrent)
        sync_sem = OpenAICompatibleLLM._sync_semaphore

        url = f"{self.api_base}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt_text}],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if stop:
            data['stop'] = stop

        # 获取信号量后执行请求
        sync_sem.acquire()
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            raw_text = result['choices'][0]['message']['content'] or ""

            # 如果返回为空，返回一个有效的 StringIO JSON
            if not raw_text.strip():
                raw_text = '{"text": " "}'

            # 确保返回的是有效 JSON
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, str):
                    raw_text = json.dumps({"text": parsed})
                elif isinstance(parsed, dict) and "text" not in parsed:
                    raw_text = json.dumps({"text": json.dumps(parsed)})
            except json.JSONDecodeError:
                raw_text = json.dumps({"text": raw_text})

            return raw_text
        finally:
            sync_sem.release()

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[list] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """同步生成文本（实现 BaseRagasLLM 接口）"""
        prompt_text = self._get_prompt_text(prompt)
        text = self._call_api(prompt_text, temperature or self.temperature, self.max_tokens, stop)
        gen = Generation(text=text)
        return LLMResult(generations=[[gen]])

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[list] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """异步生成文本（实现 BaseRagasLLM 接口）"""
        import aiohttp
        import asyncio

        # 获取或创建信号量
        if OpenAICompatibleLLM._semaphore is None:
            OpenAICompatibleLLM._semaphore = asyncio.Semaphore(self._max_concurrent)

        prompt_text = self._get_prompt_text(prompt)
        url = f"{self.api_base}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt_text}],
            'temperature': temperature if temperature is not None else self.temperature,
            'max_tokens': self.max_tokens,
        }
        if stop:
            data['stop'] = stop

        async with OpenAICompatibleLLM._semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    result = await resp.json()

        raw_text = result['choices'][0]['message']['content']

        if not raw_text or not raw_text.strip():
            # LLM 返回空，跳过此任务
            raise ValueError("LLM returned empty content, skipping")

        # 确保返回的是有效 JSON，StringIO 需要 {"text": "..."} 格式
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, str):
                raw_text = json.dumps({"text": parsed})
            elif isinstance(parsed, dict) and "text" not in parsed:
                raw_text = json.dumps({"text": json.dumps(parsed)})
        except json.JSONDecodeError:
            raw_text = json.dumps({"text": raw_text})

        gen = Generation(text=raw_text)
        return LLMResult(generations=[[gen]])

    def _get_prompt_text(self, prompt: Any) -> str:
        """从 prompt 对象提取文本"""
        if StringPromptValue and isinstance(prompt, StringPromptValue):
            return prompt.to_string()
        if hasattr(prompt, 'to_string'):
            return prompt.to_string()
        return str(prompt)

    def _validate_json_response(self, text: str) -> str:
        """
        校验 LLM 返回内容是否为合法 JSON。

        策略：
        1. 直接解析
        2. 失败则去除 markdown code block 包装后重试
        3. 仍失败则记录日志并抛出 ValueError

        Returns:
            原始文本（校验通过后原样返回）

        Raises:
            InvalidLLMResponseError: 返回内容无法解析为 JSON（继承自 ValueError，携带 .text 属性供 ragas 异常处理使用）
        """
        if not text or not text.strip():
            raise InvalidLLMResponseError("LLM 返回内容为空", text=text)

        # 步骤 1：直接解析
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # 步骤 2：去除 markdown code block 包装
        import re
        stripped = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
        stripped = re.sub(r'\s*```$', '', stripped)
        try:
            json.loads(stripped)
            logger.warning(f"LLM 返回了带 markdown 包装的 JSON，已自动去除: {stripped[:80]}...")
            return stripped
        except json.JSONDecodeError:
            pass

        # 步骤 3：解析失败，记录日志并抛出
        preview = text.strip()[:200]
        logger.error(f"LLM 返回内容无法解析为 JSON: {preview}...")
        raise InvalidLLMResponseError(
            f"LLM 返回内容不是合法 JSON: {preview}",
            text=text.strip()
        )

    def __repr__(self):
        return f"OpenAICompatibleLLM(model={self.model})"


class OpenAICompatibleEmbeddings(BaseRagasEmbeddings if BaseRagasEmbeddings else object):
    """
    适配 OpenAI 兼容接口的 Ragas Embeddings 包装器
    用于对接 Ollama 等 OpenAI 兼容的 Embedding API

    特性：
    - 全局并发限制（Semaphore）
    - 同步/异步接口均受保护
    - aiohttp ClientSession 实例级复用
    """

    # 全局并发限制信号量（类变量，所有实例共享）
    _semaphore: Optional[Any] = None
    _max_concurrent: int = 3

    @classmethod
    def set_max_concurrent(cls, n: int):
        """设置全局最大并发数"""
        cls._max_concurrent = n
        if cls._semaphore is not None:
            cls._semaphore.release()
            cls._semaphore = None

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "ollama",
        max_concurrent: int = 3,
        batch_size: int = 50,
        **kwargs
    ):
        if BaseRagasEmbeddings:
            super().__init__()
        self.model = model
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self._max_concurrent = max_concurrent
        self._batch_size = batch_size
        # 复用 aiohttp session（实例级）
        self._session: Optional[Any] = None
        self._run_config = None

    @property
    def run_config(self):
        """Ragas 期望的 run_config 属性"""
        if self._run_config is None:
            from ragas.run_config import RunConfig
            self._run_config = RunConfig()
        return self._run_config

    def set_run_config(self, run_config):
        """设置 run_config（Ragas 接口）"""
        self._run_config = run_config

    def _get_sync_semaphore(self):
        """获取同步信号量（threading.Semaphore，所有实例共享）"""
        import threading
        if not hasattr(OpenAICompatibleEmbeddings, '_sync_semaphore'):
            OpenAICompatibleEmbeddings._sync_semaphore = threading.Semaphore(self._max_concurrent)
        return OpenAICompatibleEmbeddings._sync_semaphore

    def _post_embedding(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """通用的 embedding API 调用（受全局 Semaphore 限制并发）"""
        import requests
        import threading

        url = f"{self.api_base}/embeddings"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model,
            'input': texts,
        }

        sync_sem = self._get_sync_semaphore()
        sync_sem.acquire()
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
        finally:
            sync_sem.release()

        # 按 input 顺序返回 embeddings
        embeddings = result['data']
        embeddings.sort(key=lambda x: x['index'])
        return [e['embedding'] for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """同步获取单条文本 embedding"""
        return self._post_embedding(text)[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步批量获取文本 embeddings"""
        return self._post_embedding(texts)

    async def _get_session(self) -> Any:
        """获取或创建 aiohttp ClientSession（实例级复用）"""
        import aiohttp
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def aembed_query(self, text: str) -> List[float]:
        """异步获取单条文本 embedding"""
        return (await self.aembed_documents([text]))[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文本 embeddings（受全局 Semaphore 限制并发）"""
        import aiohttp

        # 获取或创建信号量
        if OpenAICompatibleEmbeddings._semaphore is None:
            OpenAICompatibleEmbeddings._semaphore = asyncio.Semaphore(self._max_concurrent)

        url = f"{self.api_base}/embeddings"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        async with OpenAICompatibleEmbeddings._semaphore:
            session = await self._get_session()
            data = {
                'model': self.model,
                'input': texts,
            }
            async with session.post(url, headers=headers, json=data) as resp:
                resp.raise_for_status()
                result = await resp.json()

        # 按 index 排序确保顺序
        embeddings = result['data']
        embeddings.sort(key=lambda x: x['index'])
        return [e['embedding'] for e in embeddings]

    async def close(self):
        """关闭 aiohttp session"""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def __repr__(self):
        return f"OpenAICompatibleEmbeddings(model={self.model})"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class EvalSample:
    """评测样本数据结构"""
    question: str
    answer: str
    contexts: List[str]
    evolution_type: str
    metadata: Dict[str, Any]
    episode_done: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalSample":
        return cls(**data)


# ============================================================================
# Milvus 数据加载器（按论文逐篇读取，存储于内存）
# ============================================================================

class MilvusDocumentLoader:
    """
    从 Milvus 数据库加载文档 chunks

    按论文逐篇读取并存储于内存，避免 Milvus Lite 偏移查询限制：
    1. 获取所有论文名称列表
    2. 逐篇查询该论文所有 chunks（含完整 text）
    3. 全部存入内存后进行采样
    """

    def __init__(
        self,
        milvus_lite_path: str = "./data/milvus_papers.db",
        collection_name: str = "paper_embeddings",
        embed_dim: int = 1024,
        alias: str = "paperrag_eval",
        milvus_query_batch_size: int = 50,
        paper_doc_stats_path: str = "./data/paper_doc_stats.json",
    ):
        """
        Args:
            milvus_lite_path: Milvus Lite 数据库文件路径
            collection_name: 集合名称
            embed_dim: embedding 维度
            alias: 连接别名
            milvus_query_batch_size: Milvus ID IN 查询批大小（默认50，影响查询次数和内存占用）
            paper_doc_stats_path: 论文统计信息 JSON 文件路径
        """
        self.milvus_lite_path = milvus_lite_path
        self.collection_name = collection_name
        self.embed_dim = embed_dim
        self.alias = alias
        self._connection = None
        self._collection = None
        # 内存中存储所有论文的 chunks
        self._all_chunks: Dict[str, List[Dict]] = {}
        self._milvus_query_batch_size = milvus_query_batch_size
        self._paper_doc_stats_path = paper_doc_stats_path

    def _ensure_connection(self):
        """确保已建立数据库连接"""
        if self._connection is not None:
            return

        from pymilvus import connections, Collection

        connections.connect(uri=self.milvus_lite_path)
        self._collection = Collection(self.collection_name)
        self._collection.load()
        self._connection = True
        print(f"✅ 已连接到 Milvus: {self.milvus_lite_path}")

    def _close(self):
        """关闭连接"""
        if self._connection is not None:
            from pymilvus import connections
            try:
                connections.disconnect(alias="default")
            except Exception:
                pass
            self._connection = None
            self._collection = None

    def _get_paper_names(self) -> List[str]:
        """从 paper_doc_stats.json 读取论文名称列表"""
        import os

        if not os.path.exists(self._paper_doc_stats_path):
            raise FileNotFoundError(
                f"论文统计文件不存在: {self._paper_doc_stats_path}\n"
                "请确认插件目录下 data/paper_doc_stats.json 文件存在"
            )

        with open(self._paper_doc_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        paper_names = list(stats.keys())
        print(f"   从 paper_doc_stats.json 读取到 {len(paper_names)} 篇论文")
        return paper_names

    def _load_paper_chunks(self, paper_name: str) -> List[Dict]:
        """
        加载指定论文的所有 chunks（含 text）

        利用 Milvus 表达式过滤 metadata["file_name"] 直接定位该论文所有 chunks，
        无需全表扫描。

        Args:
            paper_name: 论文文件名

        Returns:
            该论文所有 chunks 的完整数据
        """
        self._ensure_connection()

        # 步骤 1：通过表达式过滤直接获取该论文所有 chunk id（无扫描）
        try:
            results = self._collection.query(
                expr=f'metadata["file_name"] == "{paper_name}"',
                output_fields=["id", "metadata"],
            )
        except Exception as e:
            print(f"  查询 {paper_name} 失败: {str(e)[:60]}...")
            return []

        if not results:
            return []

        # 提取 id 和 chunk_idx，按 chunk_idx 排序
        chunks_meta = []
        for r in results:
            meta = r.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            chunks_meta.append({
                "id": r["id"],
                "chunk_idx": meta.get("chunk_index", 0),
                "metadata": meta,
            })
        chunks_meta.sort(key=lambda x: x["chunk_idx"])

        # 步骤 2：批量查询完整文本（id in [...]）
        chunk_ids = [c["id"] for c in chunks_meta]
        all_chunks_with_text = []

        for i in range(0, len(chunk_ids), self._milvus_query_batch_size):
            batch_ids = chunk_ids[i:i + self._milvus_query_batch_size]
            expr = f"id in {batch_ids}"
            try:
                text_results = self._collection.query(
                    expr=expr,
                    output_fields=["id", "text", "metadata"],
                )
                for tr in text_results:
                    meta = tr.get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {"raw": meta}
                    all_chunks_with_text.append({
                        "id": tr["id"],
                        "text": tr.get("text", ""),
                        "metadata": meta,
                    })
            except Exception as e:
                print(f"  查询文本失败: {str(e)[:60]}...")

        # 按 id 排序保持顺序
        all_chunks_with_text.sort(key=lambda x: x["id"])
        return all_chunks_with_text

    def load_all_papers(self) -> Dict[str, List[Dict]]:
        """
        加载所有论文到内存

        Returns:
            {paper_name: [chunks...], ...}
        """
        print(f"\n🔌 正在连接 Milvus 数据库...")
        print(f"   数据库路径: {self.milvus_lite_path}")
        print(f"   集合名称: {self.collection_name}")

        self._ensure_connection()

        try:
            # 获取所有论文名称
            print("\n📋 步骤 1: 获取论文列表...")
            paper_names = self._get_paper_names()
            print(f"   发现 {len(paper_names)} 篇论文")

            # 逐篇加载所有 chunks 到内存
            print("\n📥 步骤 2: 逐篇加载 chunks 到内存...")
            self._all_chunks = {}
            total_chunks = 0

            for i, fname in enumerate(paper_names):
                chunks = self._load_paper_chunks(fname)
                if chunks:
                    self._all_chunks[fname] = chunks
                    total_chunks += len(chunks)
                    print(f"   [{i+1}/{len(paper_names)}] {fname}: {len(chunks)} chunks (总计: {total_chunks})")

            print(f"\n✅ 加载完成，共 {len(self._all_chunks)} 篇论文，{total_chunks} 个 chunks")

            for fname, chunks in list(self._all_chunks.items())[:3]:
                print(f"   • {fname}: {len(chunks)} chunks")
            if len(self._all_chunks) > 3:
                print(f"   ... 还有 {len(self._all_chunks) - 3} 篇论文")

            return self._all_chunks

        finally:
            self._close()

    def load_documents(
        self,
        max_chunks: int = 200,
        sample_strategy: str = "uniform",
    ) -> List[Any]:
        """
        从内存中的论文数据加载文档 chunks

        Args:
            max_chunks: 最多加载的 chunk 数量
            sample_strategy: 采样策略
                - "uniform": 从每篇论文均匀采样
                - "first": 只用前 N 个 chunk（按 ID 顺序）

        Returns:
            LlamaIndex Document 对象列表
        """
        # 如果还没有加载，则先加载所有论文
        if not self._all_chunks:
            self.load_all_papers()

        if not self._all_chunks:
            print("❌ 未从数据库中找到任何 chunks")
            return []

        print(f"\n🎯 采样 {max_chunks} 个 chunks...")

        # 采样
        sampled_chunks = []
        if sample_strategy == "uniform" and len(self._all_chunks) > 0:
            chunks_per_paper = max(1, max_chunks // len(self._all_chunks))
            for fname, chunks in self._all_chunks.items():
                step = max(1, len(chunks) // chunks_per_paper)
                for i in range(0, len(chunks), step):
                    sampled_chunks.append(chunks[i])
                    if len(sampled_chunks) >= max_chunks:
                        break
                if len(sampled_chunks) >= max_chunks:
                    break
        else:
            # 直接从头取
            for fname in sorted(self._all_chunks.keys()):
                sampled_chunks.extend(self._all_chunks[fname])
                if len(sampled_chunks) >= max_chunks:
                    break

        sampled_chunks = sampled_chunks[:max_chunks]
        print(f"   采样了 {len(sampled_chunks)} 个 chunks")

        # 转换为 Document 对象
        print(f"\n🔄 转换为 Document 对象...")
        documents = []
        for chunk in sampled_chunks:
            meta = chunk.get("metadata", {})

            doc = Document(
                text=chunk.get("text", ""),
                metadata={
                    "chunk_id": chunk.get("id", 0),
                    "paper_id": meta.get("paper_id", ""),
                    "file_name": meta.get("file_name", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "source": "milvus",
                    **{k: v for k, v in meta.items()
                       if k not in ("file_name", "paper_id", "chunk_index")},
                }
            )
            documents.append(doc)

        print(f"   ✅ 成功加载 {len(documents)} 个 Document")
        return documents


# ============================================================================
# 测试集生成器
# ============================================================================

class RagasTestsetGenerator:
    """Ragas 测试集生成器 - 从 Milvus 数据库生成问答对"""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embed_base_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        embedding_mode: str = "api",
        ollama_base_url: str = "http://localhost:11434",
        ollama_embed_model: str = "bge-m3",
        language: str = "chinese",
        # Milvus 配置
        milvus_lite_path: str = "./data/milvus_papers.db",
        collection_name: str = "paper_embeddings",
        embed_dim: int = 1024,
        max_chunks: int = 200,
        max_concurrent: int = 3,
        paper_doc_stats_path: Optional[str] = None,
    ):
        """
        初始化生成器

        Args:
            llm_model: LLM 模型名称
            llm_base_url: API 基础 URL（智谱/DeepSeek 等）
            llm_api_key: API Key
            embedding_model: Embedding 模型
            embed_base_url: Embedding API URL
            embed_api_key: Embedding API Key
            embedding_mode: Embedding 模式 ("api" 或 "ollama")
            ollama_base_url: Ollama API 地址
            ollama_embed_model: Ollama Embedding 模型名称
            language: 生成语言（chinese/english）
            milvus_lite_path: Milvus Lite 数据库路径
            collection_name: Milvus 集合名称
            embed_dim: Embedding 维度
            max_chunks: 从数据库加载的最大 chunk 数量
            max_concurrent: LLM 最大并发数（默认 3，建议不超过 5）
            paper_doc_stats_path: 论文统计 JSON 文件路径（默认取插件 data/paper_doc_stats.json）
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "Ragas 评估框架未安装。请运行: pip install ragas datasets llama-index-core "
                "llama-index-llms-openai llama-index-embeddings-openai"
            )

        self.language = language
        self._llm = None
        self._embed_model = None
        self._max_chunks = max_chunks
        self._max_concurrent = max_concurrent

        # 论文统计文件路径（默认为插件 data 目录）
        if paper_doc_stats_path is None:
            paper_doc_stats_path = str(Path(__file__).parent.parent / "data" / "paper_doc_stats.json")

        # Milvus 加载器
        self._milvus_loader = MilvusDocumentLoader(
            milvus_lite_path=milvus_lite_path,
            collection_name=collection_name,
            embed_dim=embed_dim,
            alias=f"paperrag_eval_{id(self)}",
            paper_doc_stats_path=paper_doc_stats_path,
        )

        # 保存配置
        self._llm_config = {
            "model": llm_model,
            "base_url": llm_base_url,
            "api_key": llm_api_key,
        }
        self._embed_config = {
            "model": embedding_model,
            "base_url": embed_base_url,
            "api_key": embed_api_key,
            "mode": embedding_mode,
            "ollama_base_url": ollama_base_url,
            "ollama_embed_model": ollama_embed_model,
        }

        # 知识图谱（用于测试集生成）- 暂不预填充，让 ragas 自己管理
        self._knowledge_graph = None
        # 查询分布（延迟初始化）
        self._query_distribution = None

    def load_documents_from_milvus(self) -> List[Any]:
        """
        从 Milvus 数据库加载文档（同步方法）

        Returns:
            Document 对象列表
        """
        return self._milvus_loader.load_documents(
            max_chunks=self._max_chunks,
            sample_strategy="uniform",
        )

    def _get_llm(self):
        """获取 LLM 实例（延迟初始化）- 使用自定义 Ragas 兼容包装器"""
        if self._llm is None:
            if self._llm_config["base_url"]:
                print(f"🔧 正在初始化 LLM: {self._llm_config['model']} @ {self._llm_config['base_url']}")
                self._llm = OpenAICompatibleLLM(
                    model=self._llm_config["model"],
                    api_base=self._llm_config["base_url"],
                    api_key=self._llm_config["api_key"] or "sk-placeholder",
                    temperature=0.3,
                    max_concurrent=self._max_concurrent,
                )
                print(f"✅ LLM 初始化成功（最大并发: {self._max_concurrent}）")
            else:
                raise ValueError("base_url is required for LLM")
        return self._llm

    def _get_embed_model(self):
        """获取 Embedding 模型实例（延迟初始化）- 使用自定义 Ragas 兼容包装器"""
        if self._embed_model is None:
            embed_mode = self._embed_config.get("mode", "api")

            if embed_mode == "ollama":
                embed_api_base = f"{self._embed_config['ollama_base_url']}/v1"
                print(f"🔧 正在连接 Ollama embedding: {embed_api_base}/{self._embed_config['ollama_embed_model']}")
                self._embed_model = OpenAICompatibleEmbeddings(
                    model=self._embed_config["ollama_embed_model"],
                    api_base=embed_api_base,
                    api_key="ollama",  # Ollama 不需要真实 key
                    max_concurrent=self._max_concurrent,
                )
                print(f"✅ Ollama embedding 初始化成功")
            elif self._embed_config["base_url"]:
                print(f"🔧 正在使用 OpenAI 兼容 embedding: {self._embed_config['base_url']}/{self._embed_config['model']}")
                self._embed_model = OpenAICompatibleEmbeddings(
                    model=self._embed_config["model"],
                    api_base=self._embed_config["base_url"],
                    api_key=self._embed_config["api_key"] or "sk-placeholder",
                    max_concurrent=self._max_concurrent,
                )
            else:
                raise ValueError("embed_base_url or ollama mode is required for embedding")
        return self._embed_model

    async def generate_testset(
        self,
        documents: List[Any],
        test_size: int = 50,
        output_path: str = "results/testset.json",
        with_debugging_logs: bool = False,
        raise_exceptions: bool = False,
    ) -> List[EvalSample]:
        """
        自动生成测试集

        Args:
            documents: 论文文档列表（从 Milvus 加载的 Document 对象）
            test_size: 生成问题数量
            output_path: 输出文件路径
            with_debugging_logs: 是否输出调试日志
            raise_exceptions: 是否抛出异常

        Returns:
            测试样本列表
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas 未安装")

        if not documents:
            raise ValueError("没有可用的论文文档")

        print(f"\n{'='*60}")
        print(f"开始生成 {test_size} 个评测问题...")
        print(f"{'='*60}")

        # 初始化生成器
        generator = TestsetGenerator(
            llm=self._get_llm(),
            embedding_model=self._get_embed_model(),
            knowledge_graph=self._knowledge_graph,
        )

        # 获取查询分布
        if self._query_distribution is None:
            self._query_distribution = default_query_distribution(
                llm=self._get_llm(),
                kg=self._knowledge_graph,
            )

        # 生成测试集
        print("正在调用 LLM 生成问答对（可能需要几分钟）...")
        testset = generator.generate_with_llamaindex_docs(
            documents=documents,
            testset_size=test_size,
            query_distribution=self._query_distribution,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
        )

        # 转换为标准格式
        samples = []
        for sample in testset.samples:
            eval_sample_obj = sample.eval_sample
            # 兼容 SingleTurnSample 和 MultiTurnSample
            if hasattr(eval_sample_obj, 'user_input'):
                question = eval_sample_obj.user_input or ""
                answer = eval_sample_obj.reference or eval_sample_obj.response or ""
                contexts = eval_sample_obj.reference_contexts or []
            else:
                question = ""
                answer = ""
                contexts = []
            eval_sample = EvalSample(
                question=question,
                answer=answer,
                contexts=contexts,
                evolution_type=sample.synthesizer_name,
                metadata={},
                episode_done=True,
            )
            samples.append(eval_sample)

        # 保存结果
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in samples], f, ensure_ascii=False, indent=2)

        print(f"\n✅ 测试集已保存到: {output_path}")
        print(f"📊 问题类型分布: {self._count_types(samples)}")

        return samples

    def _count_types(self, samples: List[EvalSample]) -> Dict[str, int]:
        """统计问题类型分布"""
        counts: Dict[str, int] = {}
        for s in samples:
            etype = s.evolution_type
            counts[etype] = counts.get(etype, 0) + 1
        return counts

    def load_testset(self, path: str) -> List[EvalSample]:
        """加载已生成的测试集"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvalSample.from_dict(item) for item in data]

    def save_testset(self, samples: List[EvalSample], path: str) -> None:
        """保存测试集到文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in samples], f, ensure_ascii=False, indent=2)
