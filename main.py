"""
AstrBot Paper RAG Plugin
本地文档库RAG检索插件
支持基于Gemini Embedding + Milvus Lite的文档检索和问答

支持的文档格式:
- PDF (.pdf) - 使用PyMuPDF高效解析
- Word文档 (.docx, .doc) - 使用python-docx解析
- 纯文本 (.txt, .md) - 支持UTF-8和GBK编码
- HTML (.html, .htm) - 使用unstructured解析
- 其他格式 - 通过unstructured库自动解析（需安装）
"""

import asyncio
import gc
import json
import os
import re
import subprocess
import requests
from datetime import timedelta, datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, TYPE_CHECKING, List, cast

# 类型注解导入（仅在类型检查时导入，避免循环导入）
if TYPE_CHECKING:
    from .hybrid_rag import HybridRAGEngine

# 抑制底层库的 gRPC/absl 警告（必须在导入深度学习库之前设置）
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Image
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register

from .rag_engine import (
    create_rag_engine,
    RAGConfig
)


# ============================================================================
# 模块常量
# ============================================================================

SUPPORTED_DOC_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm']
"""支持的文档扩展名列表"""


def _is_hidden_file(file_path: Path) -> bool:
    """检测文件是否为 macOS 元数据文件（以 ._ 开头）"""
    return file_path.name.startswith("._")


# ============================================================================
# Neo4j 服务管理器
# ============================================================================

class Neo4jServiceManager:
    """
    Neo4j 原生服务管理：检查/启动 Neo4j 服务

    使用方式：
        manager = Neo4jServiceManager()
        await manager.ensure_neo4j_running()
    """

    def __init__(
        self,
        neo4j_config: Optional[dict] = None
    ):
        self.neo4j_config = neo4j_config or {
            "host": "localhost",
            "port": 7687,
            "http_port": 7474,
            "user": "neo4j",
            "password": "password",
            "neo4j_home": "/usr/local/var/neo4j",  # macOS Homebrew 默认
        }

    def _is_neo4j_available(self) -> bool:
        """检查 Neo4j 是否可用"""
        try:
            result = subprocess.run(
                ["neo4j", "status"],
                capture_output=True,
                text=True,
                check=False
            )
            return "running" in result.stdout.lower() or result.returncode == 0
        except FileNotFoundError:
            # neo4j 命令不存在，尝试其他方式检测
            pass

        # 尝试通过 bolt 端口检测
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("localhost", self.neo4j_config["port"]))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _is_homebrew_neo4j_installed(self) -> bool:
        """检查是否通过 Homebrew 安装了 Neo4j"""
        homebrew_paths = [
            "/usr/local/bin/neo4j",
            "/opt/homebrew/bin/neo4j",
            "/usr/local/var/neo4j",
            "/opt/homebrew/var/neo4j",
        ]
        for path in homebrew_paths:
            if os.path.exists(path) or os.path.exists(os.path.dirname(path)):
                return True
        return False

    def _get_neo4j_start_command(self) -> str:
        """获取 Neo4j 启动命令"""
        # Homebrew 安装
        if os.path.exists("/usr/local/bin/neo4j") or os.path.exists("/opt/homebrew/bin/neo4j"):
            return "neo4j start"
        # 直接安装
        return "sudo systemctl start neo4j"  # Linux systemd

    async def ensure_neo4j_running(self) -> bool:
        """
        确保 Neo4j 正在运行

        Returns:
            Neo4j 是否就绪
        """
        if self._is_neo4j_available():
            logger.info("[Neo4j] Neo4j 服务已运行")
            return True

        logger.info("[Neo4j] Neo4j 未运行，尝试启动...")

        if not self._is_homebrew_neo4j_installed():
            logger.warning("[Neo4j] 未检测到 Homebrew Neo4j 安装")
            logger.info("[Neo4j] 请运行以下命令安装 Neo4j:")
            logger.info("  brew install neo4j")
            logger.info("  brew services start neo4j")
            return False

        try:
            # 尝试启动 Neo4j
            cmd = self._get_neo4j_start_command()
            logger.info(f"[Neo4j] 执行: {cmd}")

            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                # 等待 Neo4j 启动
                await self._wait_for_neo4j_ready()
                return True
            else:
                logger.error(f"[Neo4j] 启动失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"[Neo4j] 启动异常: {e}")
            return False

    async def _wait_for_neo4j_ready(self, timeout: int = 60):
        """等待 Neo4j 就绪"""
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self._is_neo4j_available():
                logger.info("[Neo4j] ✅ Neo4j 服务已就绪")
                return
            await asyncio.sleep(2)
        logger.warning("[Neo4j] ⚠️ Neo4j 启动超时")

    def get_connection_info(self) -> dict:
        """获取 Neo4j 连接信息"""
        return {
            "uri": f"bolt://{self.neo4j_config['host']}:{self.neo4j_config['port']}",
            "user": self.neo4j_config["user"],
            "password": self.neo4j_config["password"],
            "http_port": self.neo4j_config["http_port"],
        }


# ============================================================================
# CORE API 客户端（替代 arXiv MCP）
# ============================================================================

class CoreAPIClient:
    """CORE API v3 客户端 - 用于搜索和下载开放获取论文"""

    BASE_URL = "https://api.core.ac.uk/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def search_by_title(self, title: str, year: Optional[int] = None, limit: int = 5) -> list:
        """根据论文标题搜索论文

        Args:
            title: 论文标题
            year: 可选的发表年份过滤
            limit: 返回结果数量

        Returns:
            匹配的论文列表
        """
        query_parts = [f'title:"{title}"']
        if year:
            query_parts.append(f"publishedDate:{year}")
        query = " AND ".join(query_parts)

        try:
            response = requests.post(
                f"{self.BASE_URL}/search/works",
                headers=self.headers,
                json={"q": query, "limit": min(limit, 100)},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            logger.error(f"CORE API搜索失败: {e}")
            return []

    def extract_arxiv_id(self, work: dict) -> Optional[str]:
        """从 work 记录中提取 arXiv ID"""
        urls = work.get("sourceFulltextUrls", []) or []
        for url in urls:
            if url and "arxiv.org" in url:
                match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url)
                if match:
                    return match.group(1)
        return None


@register(
    "paper_rag",
    "YourName",
    "本地文档库RAG检索插件 (支持PDF/Word/TXT/HTML, Gemini + Milvus Lite)",
    "1.7.2",
    "https://github.com/your/repo"
)
class PaperRAGPlugin(Star):
    """论文RAG检索插件"""

    def __init__(self, context: Context, config: dict = {}):
        super().__init__(context)
        self.config = config or {}
        self.context = context

        # 插件配置
        self.enabled = self.config.get("enabled", True)

        # 缓存
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl_seconds", 3600)
        self.cache_max_size = self.config.get("cache_max_entries", 100)
        self._response_cache = {}

        # Graph RAG 自动构建追踪
        self._papers_since_graph_build = 0

        # RAG引擎（懒加载）
        self._engine = None
        self._config_valid = False

        # 后台服务线程追踪
        self._grobid_thread = None
        self._neo4j_thread = None

        # 根据配置决定是否启动 Grobid 服务
        enable_grobid = self.config.get("enable_grobid", False)
        if enable_grobid:
            # 异步启动 Grobid 服务（用于参考文献解析）
            # 不阻塞插件初始化，服务在后台启动
            import threading
            self._grobid_thread = threading.Thread(target=self._ensure_grobid_running, daemon=True)
            self._grobid_thread.start()
            logger.info("📚 Document RAG Plugin initialized (支持PDF/Word/TXT/HTML, Grobid已启用)")
        else:
            logger.info("📚 Document RAG Plugin initialized (支持PDF/Word/TXT/HTML, Grobid未启用)")

        # 注册 LLM 可调用的论文搜索工具
        self._register_llm_tools()

        # 自动启动 Neo4j 服务
        if self.config.get("auto_start_neo4j", True):
            self._start_neo4j_service_async()

    def _start_neo4j_service_async(self):
        """异步启动 Neo4j 服务（不阻塞插件初始化）"""
        import threading

        def _start():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # 获取 Neo4j 配置
                graph_config = self.config.get("graph_rag", {})
                neo4j_config = {
                    "host": "localhost",
                    "port": 7687,
                    "http_port": 7474,
                    "user": graph_config.get("neo4j_user", "neo4j"),
                    "password": graph_config.get("neo4j_password", "password"),
                }

                manager = Neo4jServiceManager(neo4j_config=neo4j_config)
                success = loop.run_until_complete(manager.ensure_neo4j_running())

                if success:
                    conn_info = manager.get_connection_info()
                    logger.info(f"[Neo4j] ✅ 服务已就绪: {conn_info['uri']}")
                else:
                    logger.warning("[Neo4j] ⚠️ 服务未运行，请手动启动 Neo4j")

                loop.close()
            except Exception as e:
                logger.warning(f"[Neo4j] 自动检查服务失败: {e}")

        self._neo4j_thread = threading.Thread(target=_start, daemon=True)
        self._neo4j_thread.start()
        logger.info("[Neo4j] Neo4j 服务检查线程已启动（后台运行）")

    def _register_llm_tools(self):
        """注册 LLM 可调用的论文搜索工具"""
        if not self.config.get("enable_llm_tools", True):
            logger.info("📚 Paper RAG LLM工具已禁用")
            return

        async def search_papers_tool(event, query: str, top_k: int = 5) -> str:
            """搜索本地论文库并返回结果（RAG模式）"""
            engine = self._get_engine()
            if not engine:
                return "❌ RAG引擎未就绪，请检查配置文件"

            try:
                result = await engine.search(query, mode="rag")
                if result.get("type") == "error":
                    return f"❌ 搜索失败: {result.get('message', '未知错误')}"

                answer = result.get("answer", "")
                sources = result.get("sources", [])

                # 格式化输出
                output = f"💡 **搜索结果**\n\n{answer}\n\n" if answer else "📚 **检索结果**\n\n"

                for i, src in enumerate(sources[:top_k], 1):
                    metadata = src.get("metadata", {})
                    filename = metadata.get("file_name", "unknown")
                    text = src.get("text", "")[:200]
                    output += f"[{i}] **{filename}**\n{text}...\n\n"

                return output.strip() if output.strip() else "❌ 未找到相关文档"
            except Exception as e:
                logger.error(f"LLM工具搜索失败: {e}")
                return f"❌ 搜索异常: {e}"

        async def retrieve_papers_tool(event, query: str, top_k: int = 5) -> str:
            """仅检索论文片段，不生成回答"""
            engine = self._get_engine()
            if not engine:
                return "❌ RAG引擎未就绪，请检查配置文件"

            try:
                result = await engine.search(query, mode="retrieve")
                if result.get("type") == "error":
                    return f"❌ 检索失败: {result.get('message', '未知错误')}"

                sources = result.get("sources", [])
                if not sources:
                    return "📭 未找到相关文档"

                output = "📚 **检索结果**\n\n"
                for i, src in enumerate(sources[:top_k], 1):
                    metadata = src.get("metadata", {})
                    filename = metadata.get("file_name", "unknown")
                    score = src.get("score", 0.0)
                    text = src.get("text", "")[:300]
                    output += f"[{i}] **{filename}** (相似度: {score:.3f})\n{text}...\n\n"

                return output.strip()
            except Exception as e:
                logger.error(f"LLM工具检索失败: {e}")
                return f"❌ 检索异常: {e}"

        # 注册工具
        try:
            self.context.register_llm_tool(
                name="search_papers",
                func_args=[
                    {"type": "string", "name": "query", "description": "搜索查询关键词或问题"},
                    {"type": "integer", "name": "top_k", "description": "返回结果数量，默认5"},
                ],
                desc="搜索本地论文库，使用RAG生成答案。适用于回答关于论文内容的问题。",
                func_obj=search_papers_tool
            )

            self.context.register_llm_tool(
                name="retrieve_papers",
                func_args=[
                    {"type": "string", "name": "query", "description": "搜索查询关键词"},
                    {"type": "integer", "name": "top_k", "description": "返回结果数量，默认5"},
                ],
                desc="仅检索本地论文库中的相关片段，不生成回答。适用于需要直接查看原文的场景。",
                func_obj=retrieve_papers_tool
            )

            logger.info("✅ Paper RAG LLM工具已注册: search_papers, retrieve_papers")
        except Exception as e:
            logger.error(f"注册LLM工具失败: {e}")

    def _scan_documents(self, directory: str) -> List[Any]:
        """扫描目录中的支持文档文件"""
        from pathlib import Path
        papers_dir = directory or self.config.get("papers_dir", "./papers")
        doc_files = []
        for ext in SUPPORTED_DOC_EXTENSIONS:
            doc_files.extend(Path(papers_dir).glob(f"*{ext}"))
        for ext in SUPPORTED_DOC_EXTENSIONS:
            doc_files.extend(Path(papers_dir).glob(f"*{ext.upper()}"))
        return [f for f in doc_files if not _is_hidden_file(f)]

    def _create_graph_rag_config(self) -> "GraphRAGConfig":
        """创建 GraphRAGConfig（从配置中读取所有设置）"""
        from .graph_rag_engine import GraphRAGConfig
        graph_rag_config = self.config.get("graph_rag", {})
        multimodal_config = graph_rag_config.get("multimodal_extraction", {})
        return GraphRAGConfig(
            enable_graph_rag=True,
            storage_type=graph_rag_config.get("storage_type", "memory"),
            neo4j_uri=graph_rag_config.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=graph_rag_config.get("neo4j_user", "neo4j"),
            neo4j_password=graph_rag_config.get("neo4j_password", ""),
            max_triplets_per_chunk=graph_rag_config.get("max_triplets_per_chunk", 5),
            graph_retrieval_top_k=graph_rag_config.get("graph_retrieval_top_k", 5),
            hybrid_alpha=graph_rag_config.get("hybrid_alpha", 0.5),
            auto_build=graph_rag_config.get("auto_build", False),
            auto_build_threshold=graph_rag_config.get("auto_build_threshold", 10),
            multimodal_enabled=multimodal_config.get("enabled", True),
            max_images_per_chunk=multimodal_config.get("max_images_per_chunk", 1),
            extract_image_entities=multimodal_config.get("extract_image_entities", True),
        )

    def _get_engine(self) -> "Optional[HybridRAGEngine]":
        """获取RAG引擎（单例模式，带缓存）"""
        if self._engine is None and not self._config_valid:
            try:
                # 从插件配置创建RAG配置
                rag_config = RAGConfig(
                    embedding_mode=self.config.get("embedding_mode", "ollama"),
                    embedding_provider_id=self.config.get("embedding_provider_id", ""),
                    compress_provider_id=self.config.get("compress_provider_id", ""),
                    text_provider_id=self.config.get("text_provider_id", ""),
                    multimodal_provider_id=self.config.get("multimodal_provider_id", ""),
                    llama_vlm_model_path=self.config.get("llama_vlm_model_path", "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"),
                    llama_vlm_mmproj_path=self.config.get("llama_vlm_mmproj_path", "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"),
                    llama_vlm_max_tokens=self.config.get("llama_vlm_max_tokens", 2560),
                    llama_vlm_temperature=self.config.get("llama_vlm_temperature", 0.7),
                    llama_vlm_n_ctx=self.config.get("llama_vlm_n_ctx", 4096),
                    llama_vlm_n_gpu_layers=self.config.get("llama_vlm_n_gpu_layers", 99),
                    ollama_config=self.config.get("ollama", {}),
                    milvus_lite_path=self.config.get("milvus_lite_path", ""),
                    address=self.config.get("address", ""),
                    db_name=self.config.get("db_name", "default"),
                    authentication=self.config.get("authentication", {}),
                    collection_name=self.config.get("collection_name", "paper_embeddings"),
                    embed_dim=self.config.get("embed_dim", 768),
                    top_k=self.config.get("top_k", 5),
                    similarity_cutoff=self.config.get("similarity_cutoff", 0.3),
                    papers_dir=self.config.get("papers_dir", "./papers"),
                    chunk_size=self.config.get("chunk_size", 512),
                    chunk_overlap=self.config.get("chunk_overlap", 0),
                    min_chunk_size=self.config.get("min_chunk_size", 100),
                    use_semantic_chunking=self.config.get("use_semantic_chunking", True),
                    enable_multimodal=self.config.get("multimodal", {}).get("enabled", True),
                    figures_dir=self.config.get("figures_dir", ""),
                    enable_reranking=self.config.get("enable_reranking", False),
                    reranking_model=self.config.get("reranking_model", "BAAI/bge-reranker-v2-m3"),
                    reranking_device=self.config.get("reranking_device", "auto"),
                    reranking_adaptive=self.config.get("reranking_adaptive", True),
                    reranking_threshold=self.config.get("reranking_threshold", 0.0),
                    reranking_batch_size=self.config.get("reranking_batch_size", 32),
                    enable_llm_reference_parsing=self.config.get("enable_llm_reference_parsing", True),
                    freeapi_url=self.config.get("freeapi_url", ""),
                    freeapi_key=self.config.get("freeapi_key", ""),
                    # Graph RAG 配置
                    enable_graph_rag=self.config.get("enable_graph_rag", False),
                    graph_storage_type=self.config.get("graph_rag", {}).get("storage_type", "memory"),
                    graph_neo4j_uri=self.config.get("graph_rag", {}).get("neo4j_uri", "bolt://localhost:7687"),
                    graph_neo4j_user=self.config.get("graph_rag", {}).get("neo4j_user", "neo4j"),
                    graph_neo4j_password=self.config.get("graph_rag", {}).get("neo4j_password", ""),
                    graph_max_triplets_per_chunk=self.config.get("graph_rag", {}).get("max_triplets_per_chunk", 5),
                    graph_retrieval_top_k=self.config.get("graph_rag", {}).get("graph_retrieval_top_k", 5),
                    graph_hybrid_alpha=self.config.get("graph_rag", {}).get("hybrid_alpha", 0.5),
                    graph_auto_build=self.config.get("graph_rag", {}).get("auto_build", False),
                    graph_auto_build_threshold=self.config.get("graph_rag", {}).get("auto_build_threshold", 10)
                )

                # 验证配置
                valid, error_msg = rag_config.validate()
                if not valid:
                    logger.error(f"❌ RAG配置无效: {error_msg}")
                    self._config_valid = False
                    return None

                # 创建llama-index引擎（已移除旧版引擎支持）
                self._engine = create_rag_engine(rag_config, self.context)
                self._config_valid = True

            except Exception as e:
                logger.error(f"❌ RAG引擎初始化失败: {e}")
                self._config_valid = False
                return None

        return self._engine # type: ignore

    def _get_cache_key(self, query: str, mode: str, top_k: int) -> str:
        """生成缓存键"""
        return f"{query}|{mode}|{top_k}"

    def _get_cached_response(self, cache_key: str):
        """获取缓存的响应"""
        if not self.cache_enabled:
            return None

        import time
        if cache_key in self._response_cache:
            cached_data, timestamp = self._response_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"📦 使用缓存: {cache_key[:50]}...")
                return cached_data
            else:
                # 缓存过期，删除
                del self._response_cache[cache_key]

        return None

    def _set_cached_response(self, cache_key: str, response):
        """设置缓存"""
        if not self.cache_enabled:
            return

        import time
        # 如果缓存已满，删除最旧的条目
        if len(self._response_cache) >= self.cache_max_size:
            oldest_key = min(self._response_cache.keys(),
                           key=lambda k: self._response_cache[k][1])
            del self._response_cache[oldest_key]

        self._response_cache[cache_key] = (response, time.time())

    async def _maybe_trigger_graph_auto_build(self, papers_added: int = 1) -> bool:
        """
        检查是否需要自动构建知识图谱

        Args:
            papers_added: 本次添加的论文数量

        Returns:
            是否触发了自动构建
        """
        if not self.config.get("enable_graph_rag", False):
            return False

        auto_build = self.config.get("graph_rag", {}).get("auto_build", False)
        if not auto_build:
            return False

        self._papers_since_graph_build += papers_added
        threshold = self.config.get("graph_rag", {}).get("auto_build_threshold", 10)

        if self._papers_since_graph_build >= threshold:
            logger.info(f"📚 自动构建知识图谱（已添加 {self._papers_since_graph_build} 篇论文）")
            self._papers_since_graph_build = 0

            # 执行图谱构建
            try:
                engine = self._get_engine()
                if not engine:
                    return False

                # 触发构建（在后台异步执行，不阻塞命令响应）
                # 注意：这会重新解析所有文档
                import asyncio
                asyncio.create_task(self._run_graph_build_in_background(engine))
                return True
            except Exception as e:
                logger.error(f"自动构建知识图谱失败: {e}")
                return False

        return False

    async def _run_graph_build_in_background(self, engine):
        """后台运行图谱构建"""
        try:
            from .graph_rag_engine import GraphRAGEngine, GraphRAGConfig
        except Exception:
            from graph_rag_engine import GraphRAGEngine, GraphRAGConfig

        try:
            graph_config = self._create_graph_rag_config()

            graph_engine = GraphRAGEngine(graph_config, engine, self.context)
            await graph_engine.initialize()

            # 获取所有文档并构建图谱
            papers_dir = self.config.get("papers_dir", "./papers")
            doc_files = self._scan_documents(papers_dir)

            if not doc_files:
                return

            parser = engine._ensure_parser_initialized()
            all_nodes = []
            for doc_file in doc_files:
                try:
                    nodes = await parser.parse_and_split(str(doc_file), {}, None)
                    all_nodes.extend(nodes)
                except Exception:
                    pass

            await graph_engine.build_graph_from_nodes(all_nodes)
            logger.info("✅ 后台知识图谱构建完成")
        except Exception as e:
            logger.error(f"❌ 后台知识图谱构建失败: {e}")

    def _ensure_grobid_running(self):
        """
        确保 Grobid 服务正在运行
        首次使用需手动下载镜像，后续自动启动
        """
        import time

        grobid_url = os.environ.get("GROBID_URL", "http://localhost:8070")
        logger.info(f"🔍 检查 Grobid 服务状态 (URL: {grobid_url})...")

        # 步骤1：检查 Grobid 是否已可用
        logger.debug("📡 步骤1: 检查 Grobid 健康状态...")
        try:
            response = requests.get(f"{grobid_url}/api/health", timeout=3)
            if response.status_code == 200:
                logger.info("✅ Grobid 服务已就绪 (HTTP 200)")
                return
            else:
                logger.debug(f"⚠️ Grobid 返回状态码: {response.status_code}")
        except Exception as e:
            logger.debug(f"⚠️ Grobid 健康检查失败: {e}")

        # 步骤2：检查 Docker 是否可用
        logger.debug("🐳 步骤2: 检查 Docker 可用性...")
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=False
            )
            if result.returncode != 0:
                raise Exception("Docker 返回非零状态码")
            logger.debug("✅ Docker 可用")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error("❌ Docker 不可用")
            logger.warning(f"   原因: {e}")
            logger.info("💡 请手动启动 Grobid:")
            logger.info("   docker run --rm -p 8070:8070 grobid/grobid:0.8.2-full")
            return

        # 步骤3：检查 Docker 镜像是否已存在
        logger.debug("🔍 步骤3: 检查 Grobid 镜像是否存在...")
        try:
            result = subprocess.run(
                ["docker", "images", "-q", "grobid/grobid:0.8.2-full"],
                capture_output=True,
                text=True,
                timeout=10
            )
            image_exists = bool(result.stdout.strip())
            logger.debug(f"   镜像存在: {image_exists}")
        except Exception as e:
            logger.debug(f"⚠️ 检查镜像失败: {e}")
            image_exists = False

        # 步骤4：处理镜像状态
        if not image_exists:
            logger.warning("⚠️ Docker 镜像 'grobid/grobid:0.8.2-full' 不存在")
            logger.info("=" * 50)
            logger.info("📦 首次使用需要下载 Grobid 镜像 (约 2-3GB)")
            logger.info("   推荐手动下载以避免网络问题:")
            logger.info("   1. 打开终端执行:")
            logger.info("      docker pull grobid/grobid:0.8.2-full")
            logger.info("   2. 等待下载完成后，重启插件或重新加载 Paper RAG")
            logger.info("=" * 50)
            logger.info("💡 或者手动启动 Grobid:")
            logger.info("   docker run --rm -p 8070:8070 grobid/grobid:0.8.2-full")
            return

        # 步骤5：检查是否已有 Grobid 容器在运行
        logger.debug("🔍 步骤5: 检查运行中的 Grobid 容器...")
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "ancestor=grobid/grobid:0.8.2-full", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.stdout.strip():
                container_id = result.stdout.strip()
                logger.info(f"✅ 发现运行中的 Grobid 容器: {container_id}")

                # 验证容器是否真正响应
                logger.debug("📡 验证容器健康状态...")
                for i in range(5):
                    try:
                        response = requests.get(f"{grobid_url}/api/health", timeout=3)
                        if response.status_code == 200:
                            logger.info("✅ Grobid 服务已就绪")
                            return
                    except Exception:
                        pass
                    time.sleep(1)
                logger.warning("⚠️ 容器存在但服务未响应，继续等待...")

        except subprocess.TimeoutExpired:
            logger.warning("⚠️ 检查容器超时")
        except Exception as e:
            logger.debug(f"⚠️ 检查容器失败: {e}")

        # 步骤6：启动新 Grobid 容器
        logger.info("🚀 步骤6: 启动 Grobid 容器...")

        try:
            # 停止并删除旧容器（如果存在）
            subprocess.run(
                ["docker", "stop", "grobid_paperrag"],
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ["docker", "rm", "grobid_paperrag"],
                capture_output=True,
                timeout=30
            )
        except Exception:
            pass  # 容器可能不存在，忽略错误

        try:
            # 启动新容器
            proc = subprocess.Popen(
                [
                    "docker", "run",
                    "--rm",
                    "-d",
                    "--name", "grobid_paperrag",
                    "-p", "8070:8070",
                    "-m", "4g",
                    "grobid/grobid:0.8.2-full"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # 等待容器启动
            stdout, stderr = proc.communicate(timeout=30)
            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            if proc.returncode == 0 and stdout_str:
                container_id = stdout_str.strip()
                logger.info(f"✅ 容器启动成功 (ID: {container_id[:12]}...)")
            else:
                if stderr_str:
                    logger.debug(f"   stderr: {stderr_str[:200]}")
                logger.info("✅ 容器已在后台启动")

            # 步骤7：等待服务就绪
            logger.info("⏳ 步骤7: 等待 Grobid 服务就绪...")
            for i in range(30):  # 最多等待60秒
                elapsed = (i + 1) * 2
                try:
                    response = requests.get(f"{grobid_url}/api/health", timeout=3)
                    if response.status_code == 200:
                        logger.info(f"✅ Grobid 服务就绪! (耗时: {elapsed}s)")
                        return
                except Exception:
                    pass

                # 每10秒输出一次进度
                if i % 5 == 0:
                    logger.debug(f"   等待中... ({elapsed}s)")

            # 超时但服务可能在继续初始化
            logger.warning("⚠️ Grobid 启动超时 (60s)，服务可能在后台继续初始化")
            logger.info("   验证命令: curl http://localhost:8070/api/health")

        except subprocess.TimeoutExpired:
            logger.error("❌ 容器启动命令超时 (30s)")
            logger.info("💡 请手动启动 Grobid:")
            logger.info("   docker run --rm -p 8070:8070 grobid/grobid:0.8.2-full")
        except Exception as e:
            logger.error(f"❌ 启动 Grobid 失败: {e}")
            logger.info("💡 请手动启动 Grobid:")
            logger.info("   docker run --rm -p 8070:8070 grobid/grobid:0.8.2-full")

    # ==================== 命令系统 ====================

    @filter.command_group("paper")
    def paper_commands(self):
        """Paper RAG command group
        search       - Search documents and answer questions
        list         - List indexed documents
        add          - Add documents to knowledge base (PDF/Word/TXT supported)
        addf         - Add a single document to knowledge base
        delete       - Delete a specific paper from knowledge base
        clear        - Clear knowledge base
        rebuild      - Clear and re-add all documents
        refstats     - Show reference title frequency statistics (-1 for zero-ref papers)
        arxiv_add    - Search arxiv and download papers, then add to database (Admin)
        arxiv_refs   - Download highly-cited reference papers from arxiv (Admin)
        arxiv_sync   - Sync MCP downloaded papers to paperrag database (Admin)
        arxiv_cleanup- Clean up old versions of arxiv papers (Admin)
        graph_build  - Build knowledge graph from indexed documents
        graph_rebuild- Rebuild knowledge graph from scratch (clear + rebuild)
        graph_stats  - Show knowledge graph statistics
        graph_clear  - Clear knowledge graph (Admin)
        """
        pass

    @paper_commands.command("search")
    async def cmd_search(self, event: AstrMessageEvent,
                         query: str = '',
                         mode: str = "rag",
                         top_k: int = 5):
        """Search document library and answer questions

        Args:
            query: Search question
            mode: Mode (rag=retrieval augmented generation, retrieve=retrieval only, auto=intent routing when Graph RAG enabled)
            top_k: Number of results to return
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not query:
            yield event.plain_result("📚 Usage: /paper search [question]\nExample: /paper search What are the key innovations of 3D Gaussian Splatting")
            return

        # 获取引擎
        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪，请检查配置文件")
            return

        # 检查缓存
        cache_key = self._get_cache_key(query, mode, top_k)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            yield event.plain_result(cached_response)
            return

        # Send processing message
        yield event.plain_result(f"🔍 Searching document library...\nQuestion: {query}")

        try:
            # 意图识别与路由（当 mode="auto" 时）
            actual_mode = mode
            routing_info = ""
            if mode == "auto" and self.config.get("enable_graph_rag", False):
                try:
                    from .graph_rag_router import create_router, RetrievalMode
                except Exception:
                    from graph_rag_router import create_router, RetrievalMode

                router = create_router(context=self.context)
                route_result = router.route(query)
                actual_mode = route_result.mode.value
                routing_info = f"\n📊 意图识别: {route_result.thinking}"
                if route_result.entities:
                    routing_info += f"\n🔑 实体: {', '.join(route_result.entities)}"
                if route_result.query_refine != query:
                    routing_info += f"\n🔄 查询优化: {route_result.query_refine}"
                query = route_result.query_refine  # 使用优化后的查询
                logger.info(f"🔀 路由决策: {actual_mode} - {route_result.thinking}")
                yield event.plain_result(routing_info)
            elif mode == "auto":
                actual_mode = "rag"  # Graph RAG 未启用时默认回退到 RAG

            # Execute search
            response = await engine.search(query, mode=actual_mode)

            # 安全获取响应类型，避免 KeyError
            response_type = response.get("type", "unknown")

            # Format output
            if response_type == "retrieve":
                # Retrieve mode only
                output = self._format_retrieve_response(response.get("sources", []))
            elif response_type == "rag":
                # RAG mode
                output = self._format_rag_response(
                    response.get("answer", ""),
                    response.get("sources", [])
                )
            elif response_type == "error":
                output = f"❌ {response.get('message', 'Unknown error')}"
            else:
                output = f"❌ Unknown response type: {response_type}"

            # Cache response
            self._set_cached_response(cache_key, output)

            # Send result
            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            yield event.plain_result(f"❌ Search failed: {e}")

    @paper_commands.command("list")
    async def cmd_list(self, event: AstrMessageEvent):
        """List all documents in the library"""
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        engine = self._get_engine()

        if not engine:
            yield event.plain_result("❌ RAG engine is not ready, please check configuration")
            return

        try:
            papers = await engine.list_papers()

            if not papers:
                yield event.plain_result("📭 Document library is empty, please add documents first")
                return

            # Format output
            output = "📚 **Document Library**\n\n"
            for i, paper in enumerate(papers[:20], 1):  # Show max 20 papers
                output += f"{i}. ✅ **{paper['file_name']}**\n"
                output += f"   └─ Chunks: {paper['chunk_count']}\n"
                output += f"   └─ Added: {paper.get('added_time', 'unknown')}\n\n"

            if len(papers) > 20:
                output += f"...and {len(papers) - 20} more papers\n"

            output += f"\n📊 Total: {len(papers)} documents"

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"Failed to list papers: {e}")
            yield event.plain_result(f"❌ Failed to list papers: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("add")
    async def cmd_add(self, event: AstrMessageEvent, directory: str = ''):
        """Add documents to knowledge base (Admin)
        Supported formats: PDF, Word (.docx), TXT, MD, HTML, etc.

        Args:
            directory: Document directory path (optional, use configured path by default)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        papers_dir = directory or self.config.get("papers_dir", "./papers")

        # Check directory
        if not os.path.exists(papers_dir):
            yield event.plain_result(f"❌ Directory does not exist: {papers_dir}")
            return

        yield event.plain_result(f"🔍 Scanning directory: {papers_dir}")

        try:
            # Scan supported document files
            doc_files = self._scan_documents(papers_dir)

            if not doc_files:
                yield event.plain_result("📭 No supported document files found\nSupported formats: PDF, Word, TXT, Markdown, HTML")
                return

            yield event.plain_result(f"📄 Found {len(doc_files)} document files\n⏳ Starting import...")

            # Get engine
            engine = self._get_engine()
            if not engine:
                yield event.plain_result("❌ RAG engine is not ready")
                return

            # Import documents using new API
            import time
            start_time = time.time()

            total_files = len(doc_files)
            successful = 0
            failed = 0
            total_chunks = 0

            for idx, doc_file in enumerate(doc_files, 1):
                try:
                    file_path = str(doc_file)
                    file_name = doc_file.name

                    # Add single document
                    result = await engine.add_paper(file_path)

                    if result["status"] == "success":
                        chunks_added = result.get("chunks_added", 0)
                        total_chunks += chunks_added
                        successful += 1
                        yield event.plain_result(
                            f"✅ [{idx}/{total_files}] {file_name} - {chunks_added} chunks"
                        )

                        # 检查是否需要自动构建知识图谱
                        if successful == 1:  # 只在第一批成功时检查
                            auto_built = await self._maybe_trigger_graph_auto_build(successful)
                            if auto_built:
                                yield event.plain_result("📚 图谱自动构建已在后台触发")
                    else:
                        failed += 1
                        error_msg = result.get("message", "未知错误")
                        yield event.plain_result(
                            f"❌ [{idx}/{total_files}] {file_name} - {error_msg}"
                        )

                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to import {doc_file.name}: {e}")
                    yield event.plain_result(
                        f"❌ [{idx}/{total_files}] {doc_file.name} - {str(e)}"
                    )

            # Final summary
            elapsed_time = time.time() - start_time
            output = f"""✅ **Import Complete**

📊 Statistics:
  • Total files: {total_files}
  • Successfully processed: {successful}
  • Failed: {failed}
  • Chunks created: {total_chunks}
  • Time: {elapsed_time:.1f}s

💡 Tip: Use /paper search [question] to search documents"""

            if failed > 0:
                output += f"\n\n⚠️ {failed} files failed to process"

            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            yield event.plain_result(f"❌ Failed to add documents: {str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("addf")
    async def cmd_add_file(self, event: AstrMessageEvent, file_path: str = ''):
        """Add a single document to knowledge base (Admin)

        Args:
            file_path: Full path to the document file
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not file_path:
            yield event.plain_result("❌ Please provide file path\nUsage: /paper addf <file_path>")
            return

        file_path = file_path.strip()

        # Check file exists
        if not os.path.exists(file_path):
            yield event.plain_result(f"❌ File not found: {file_path}")
            return

        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            yield event.plain_result(f"❌ Not a file: {file_path}")
            return

        # Check supported format
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_DOC_EXTENSIONS:
            yield event.plain_result(f"❌ Unsupported format: {ext}\nSupported: {', '.join(SUPPORTED_DOC_EXTENSIONS)}")
            return

        # Get engine
        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        file_name = os.path.basename(file_path)
        yield event.plain_result(f"📄 Adding: {file_name}...")

        try:
            result = await engine.add_paper(file_path)

            if result.get("status") == "success":
                chunks_added = result.get("chunks_added", 0)
                yield event.plain_result(f"✅ {file_name}\n   └─ {chunks_added} chunks added")

                # 检查是否需要自动构建知识图谱
                auto_built = await self._maybe_trigger_graph_auto_build(1)
                if auto_built:
                    logger.info("📚 图谱自动构建已触发，将在后台运行")
            else:
                error_msg = result.get("message", "Unknown error")
                yield event.plain_result(f"❌ {file_name}\n   └─ {error_msg}")

        except Exception as e:
            logger.error(f"Failed to add {file_path}: {e}")
            yield event.plain_result(f"❌ {file_name}\n   └─ {str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("clear")
    async def cmd_clear(self, event: AstrMessageEvent, confirm: str = ''):
        """Clear document knowledge base (Admin)"""
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result("⚠️ Dangerous operation! Please use: /paper clear confirm")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        try:
            result = await engine.clear()
            self._response_cache.clear()
            if result.get("status") == "success":
                yield event.plain_result(f"✅ {result.get('message', 'Document library cleared')}")
            else:
                yield event.plain_result(f"❌ {result.get('message', 'Failed to clear document library')}")

        except Exception as e:
            logger.error(f"Failed to clear document library: {e}")
            yield event.plain_result(f"❌ Failed to clear document library: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("delete")
    async def cmd_delete(self, event: AstrMessageEvent, file_name: str = ''):
        """Delete a specific paper from knowledge base (Admin)

        Args:
            file_name: File name to delete (partial match supported)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not file_name:
            yield event.plain_result("❌ Please provide file name\nUsage: /paper delete <filename>\nExample: /paper delete transformer.pdf")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        try:
            result = await engine.delete_paper(file_name)

            if result.get("status") == "success":
                deleted_count = result.get("deleted_count", 0)
                yield event.plain_result(f"✅ {result.get('message', 'Paper deleted')}\n   └─ Deleted {deleted_count} vectors")
            else:
                yield event.plain_result(f"❌ {result.get('message', 'Failed to delete paper')}")

        except Exception as e:
            logger.error(f"Failed to delete paper: {e}")
            yield event.plain_result(f"❌ Failed to delete paper: {e}")

    @paper_commands.command("refstats")
    async def cmd_refstats(self, event: AstrMessageEvent, top_k: int = 20):
        """Show reference title frequency statistics

        Args:
            top_k: Number of top references to show (default: 20). Use -1 to list papers with zero references.
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        try:
            # 获取索引管理器
            index_manager = engine._ensure_index_manager_initialized()

            # top_k = -1 表示列出参考文献数量为0的论文
            if top_k == -1:
                yield event.plain_result("📊 正在查找无参考文献的论文...")

                result = await index_manager.get_papers_with_zero_references()

                if "error" in result:
                    yield event.plain_result(f"❌ 获取失败: {result['error']}")
                    return

                papers = result.get("papers", [])
                total_papers = result.get("total_papers", 0)
                total_zero_ref = result.get("total_zero_ref", 0)

                # total_papers == 0 表示未能成功获取论文列表
                if total_papers == 0:
                    yield event.plain_result("⚠️ 未能获取到论文列表，请检查索引是否初始化")
                    return

                if not papers:
                    yield event.plain_result("✅ 所有论文都已提取到参考文献")
                    return

                # 格式化输出
                output = f"📚 **无参考文献的论文** ({total_zero_ref}/{total_papers})\n\n"

                for i, paper in enumerate(papers, 1):
                    file_name = paper.get("file_name", "unknown")
                    chunk_count = paper.get("chunk_count", 0)

                    if len(file_name) > 70:
                        file_name_display = file_name[:67] + "..."
                    else:
                        file_name_display = file_name

                    output += f"{i:3d}. **{file_name_display}**\n"
                    output += f"      └─ chunks: {chunk_count}\n"

                yield event.plain_result(output.strip())
                return

            # 正常模式：显示高频引用论文统计
            yield event.plain_result("📊 正在统计参考文献...")

            stats = await index_manager.get_all_references()

            if "error" in stats:
                yield event.plain_result(f"❌ 获取统计失败: {stats['error']}")
                return

            references = stats.get("references", [])
            total_refs = stats.get("total_refs", 0)
            total_chunks = stats.get("total_chunks", 0)

            if not references:
                yield event.plain_result("📭 数据库中暂无参考文献信息\n💡 请先使用 /paper add 添加论文")
                return

            # 格式化输出
            output = f"📚 **参考文献统计**\n\n"
            output += f"📊 统计概览:\n"
            output += f"   • 涉及论文种类: {len(references)}\n"
            output += f"   • 引用总条次: {total_refs}\n"
            output += f"   • 处理文档块: {total_chunks}\n\n"

            output += f"🔝 **Top {min(top_k, len(references))} 高频引用论文**\n\n"

            for i, ref in enumerate(references[:top_k], 1):
                title = ref["title"]
                count = ref["count"]
                authors = ref.get("authors", "")
                year = ref.get("year", "N/A")

                # 截断过长标题
                if len(title) > 60:
                    title_display = title[:57] + "..."
                else:
                    title_display = title

                # 截断作者
                if authors and len(authors) > 40:
                    authors_display = authors[:37] + "..."
                else:
                    authors_display = authors

                output += f"{i:2d}. [{count:3d}次] **{title_display}**\n"
                if authors_display:
                    output += f"    └─ {authors_display}"
                    if year:
                        output += f" ({year})"
                    output += "\n"

            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"Failed to get refstats: {e}")
            yield event.plain_result(f"❌ 获取参考文献统计失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("reparse_zero_ref")
    async def cmd_reparse_zero_ref(self, event: AstrMessageEvent, confirm: str = ''):
        """Batch re-parse papers with zero references (Admin)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != 'confirm':
            yield event.plain_result("⚠️ This will re-parse all papers with zero references.\n"
                                   "This operation may take a long time.\n"
                                   "Usage: /paper reparse_zero_ref confirm")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        try:
            index_manager = engine._ensure_index_manager_initialized()

            # Step 1: Get papers with zero references
            yield event.plain_result("🔍 Step 1/4: Finding papers with zero references...")

            result = await index_manager.get_papers_with_zero_references()

            if "error" in result:
                yield event.plain_result(f"❌ 获取失败: {result['error']}")
                return

            papers = result.get("papers", [])
            total_zero_ref = result.get("total_zero_ref", 0)

            if not papers:
                yield event.plain_result("✅ All papers have extracted references")
                return

            yield event.plain_result(f"📊 Found {total_zero_ref} papers with zero references")

            # Step 2: Find file paths for each paper
            yield event.plain_result("🔍 Step 2/4: Locating paper files...")

            papers_dir = self.config.get("papers_dir", "./papers")
            papers_path = Path(papers_dir)

            if not papers_path.exists():
                yield event.plain_result(f"❌ Papers directory does not exist: {papers_dir}")
                return

            # Build a mapping of file_name to full path
            file_path_map: Dict[str, Path] = {}
            for ext in SUPPORTED_DOC_EXTENSIONS:
                for f in papers_path.glob(f"*{ext}"):
                    file_path_map[f.name] = f
                for f in papers_path.glob(f"*{ext.upper()}"):
                    file_path_map[f.name] = f

            # Also search subdirectories
            for ext in SUPPORTED_DOC_EXTENSIONS:
                for f in papers_path.rglob(f"*{ext}"):
                    file_path_map[f.name] = f
                for f in papers_path.rglob(f"*{ext.upper()}"):
                    file_path_map[f.name] = f

            # Match papers with their file paths
            papers_to_reparse = []
            not_found = []

            for paper in papers:
                file_name = paper.get("file_name", "")
                if file_name in file_path_map:
                    papers_to_reparse.append({
                        "file_name": file_name,
                        "file_path": str(file_path_map[file_name]),
                        "chunk_count": paper.get("chunk_count", 0)
                    })
                else:
                    not_found.append(file_name)

            if not_found:
                yield event.plain_result(f"⚠️ {len(not_found)} papers not found in {papers_dir}:")
                for fn in not_found[:5]:
                    yield event.plain_result(f"   - {fn}")
                if len(not_found) > 5:
                    yield event.plain_result(f"   ... and {len(not_found) - 5} more")

            if not papers_to_reparse:
                yield event.plain_result("❌ No paper files found for zero-ref papers")
                return

            yield event.plain_result(f"✅ Found {len(papers_to_reparse)} paper files")

            # Step 3: Delete from database
            yield event.plain_result("🔍 Step 3/4: Deleting from database and figures...")

            deleted_count = 0
            for paper in papers_to_reparse:
                file_name = paper["file_name"]
                file_path = paper.get("file_path")
                try:
                    result = await engine.delete_paper(file_name, file_path)
                    if result.get("status") == "success":
                        deleted_count += 1
                    else:
                        logger.warning(f"删除失败: {file_name} - {result.get('message')}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_name}: {e}")

                if deleted_count % 10 == 0:
                    yield event.plain_result(f"   Deleted {deleted_count}/{len(papers_to_reparse)}...")

            yield event.plain_result(f"✅ Deleted {deleted_count} papers from database")

            # Step 4: Re-parse and re-vectorize
            yield event.plain_result("🔍 Step 4/4: Re-parsing and vectorizing...")

            import time
            start_time = time.time()
            success_count = 0
            fail_count = 0
            total_chunks = 0

            for i, paper in enumerate(papers_to_reparse, 1):
                try:
                    result = await engine.add_paper(paper["file_path"])

                    if result.get("status") == "success":
                        chunks_added = result.get("chunks_added", 0)
                        total_chunks += chunks_added
                        success_count += 1
                    else:
                        fail_count += 1
                        logger.warning(f"Failed to re-parse {paper['file_name']}: {result.get('message')}")
                except Exception as e:
                    fail_count += 1
                    logger.error(f"Failed to re-parse {paper['file_name']}: {e}")

                # Progress update every 5 papers
                if i % 5 == 0 or i == len(papers_to_reparse):
                    elapsed = time.time() - start_time
                    yield event.plain_result(
                        f"   Progress: {i}/{len(papers_to_reparse)} "
                        f"(success: {success_count}, failed: {fail_count})"
                    )

            elapsed_time = time.time() - start_time

            output = f"""✅ **Reparse Complete**

📊 Statistics:
  • Total zero-ref papers: {total_zero_ref}
  • Files found: {len(papers_to_reparse)}
  • Successfully re-parsed: {success_count}
  • Failed: {fail_count}
  • Chunks created: {total_chunks}
  • Time: {elapsed_time:.1f}s

💡 Tip: Use /paper refstats -1 to check again"""

            if not_found:
                output += f"\n\n⚠️ {len(not_found)} papers not found in filesystem"

            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"Failed to reparse zero-ref papers: {e}")
            yield event.plain_result(f"❌ 操作失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("arxiv_add")
    async def cmd_arxiv_add(self, event: AstrMessageEvent, query: str = '', max_results: int = 5):
        """Search CORE API and download papers, then add to database (Admin)

        Args:
            query: Search query for papers (e.g., paper title, authors, keywords)
            max_results: Maximum number of papers to download (default: 5)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not query:
            yield event.plain_result("❌ Please provide search query\nUsage: /paper arxiv_add <query> [max_results]\nExample: /paper arxiv_add attention is all you need 3")
            return

        # Check CORE API key
        core_api_key = self.config.get("core_api_key", "")
        if not core_api_key:
            yield event.plain_result("❌ CORE API Key未配置\n请在插件配置中设置 core_api_key")
            return

        papers_dir = self.config.get("papers_dir", "./papers")

        # Ensure papers directory exists
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            papers_path.mkdir(parents=True, exist_ok=True)

        yield event.plain_result(f"🔍 在CORE搜索: \"{query}\"\n最大下载数量: {max_results}")

        try:
            # Step 1: Search CORE API
            yield event.plain_result("📡 正在搜索CORE...")
            core_client = CoreAPIClient(core_api_key)
            works = core_client.search_by_title(query, limit=max_results)

            if not works:
                yield event.plain_result("❌ 未找到相关论文")
                return

            yield event.plain_result(f"✅ 找到 {len(works)} 篇论文")

            # Step 2: Download each paper
            engine = self._get_engine()
            if not engine:
                yield event.plain_result("❌ RAG引擎未就绪")
                return

            successful = 0
            failed = 0
            skipped = 0

            for i, work in enumerate(works, 1):
                # 提取论文信息
                work_id = work.get('id', '')
                title = work.get('title', 'unknown')
                source_urls = work.get('sourceFulltextUrls', []) or []

                if not title:
                    logger.warning(f"⚠️ 论文信息缺少标题: {work}")
                    failed += 1
                    continue

                yield event.plain_result(f"\n📄 [{i}/{len(works)}] {title[:60]}...")

                # 提取下载URL（优先使用arXiv链接）
                pdf_url = None
                for url in source_urls:
                    if 'arxiv.org/pdf' in url:
                        pdf_url = url
                        break
                if not pdf_url and source_urls:
                    pdf_url = source_urls[0]

                if not pdf_url:
                    yield event.plain_result(f"   ⚠️ 无可下载链接")
                    failed += 1
                    continue

                # 确定文件名
                arxiv_id = core_client.extract_arxiv_id(work)
                if arxiv_id:
                    pdf_filename = f"{arxiv_id}.pdf"
                else:
                    safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                    pdf_filename = f"{work_id}_{safe_title}.pdf"

                pdf_path = papers_path / pdf_filename

                # 检查是否已存在
                if pdf_path.exists():
                    yield event.plain_result(f"   ⏭️ PDF已存在，跳过下载")
                    skipped += 1
                    result = await engine.add_paper(str(pdf_path))
                    if result.get("status") == "success":
                        successful += 1
                        yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                    continue

                # 下载PDF
                try:
                    yield event.plain_result(f"   📥 下载PDF: {pdf_url[:80]}...")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    }
                    pdf_response = requests.get(pdf_url, headers=headers, timeout=120, stream=True)

                    if pdf_response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        file_size = pdf_path.stat().st_size / (1024 * 1024)
                        yield event.plain_result(f"   ✅ 下载完成 ({file_size:.1f} MB)")

                        # Add to database
                        result = await engine.add_paper(str(pdf_path))
                        if result.get("status") == "success":
                            successful += 1
                            yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                        else:
                            yield event.plain_result(f"   ⚠️ 添加失败: {result.get('message', 'unknown')}")
                    else:
                        yield event.plain_result(f"   ❌ 下载失败: HTTP {pdf_response.status_code}")
                        failed += 1

                except Exception as e:
                    logger.error(f"下载论文失败: {e}")
                    yield event.plain_result(f"   ❌ 下载失败: {e}")
                    failed += 1

            # Summary
            output = f"""
📊 **CORE论文下载完成**

✅ 成功: {successful}
⏭️ 跳过: {skipped} (已存在)
❌ 失败: {failed}

📁 保存路径: {papers_dir}
💡 使用 /paper list 查看已添加的论文
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"arXiv操作失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ arXiv操作失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("arxiv_refs")
    async def cmd_arxiv_refs(self, event: AstrMessageEvent, top_k: int = 10, max_per_paper: int = 3):
        """Download highly-cited reference papers via CORE API and add to database (Admin)

        Args:
            top_k: Number of top-cited references to process (default: 10)
            max_per_paper: Maximum papers to download per reference (default: 3)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        # Check CORE API key
        core_api_key = self.config.get("core_api_key", "")
        if not core_api_key:
            yield event.plain_result("❌ CORE API Key未配置\n请在插件配置中设置 core_api_key")
            return

        papers_dir = self.config.get("papers_dir", "./papers")
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            papers_path.mkdir(parents=True, exist_ok=True)

        yield event.plain_result(f"📊 正在获取高频引用论文统计...")

        try:
            # Step 1: Get reference statistics
            index_manager = engine._ensure_index_manager_initialized()
            stats = await index_manager.get_all_references()

            if "error" in stats:
                yield event.plain_result(f"❌ 获取统计失败: {stats['error']}")
                return

            references = stats.get("references", [])
            if not references:
                yield event.plain_result("📭 数据库中暂无参考文献信息\n💡 请先使用 /paper add 添加论文")
                return

            # Get top-k references
            top_refs = references[:top_k]
            yield event.plain_result(f"📚 找到 {len(references)} 种参考文献，取前 {len(top_refs)} 个高频引用")

            # Step 2: Search and download each reference paper via CORE API
            successful = 0
            failed = 0
            skipped = 0
            total_downloaded = 0

            core_client = CoreAPIClient(core_api_key)

            for i, ref in enumerate(top_refs, 1):
                title = ref.get("title", "")
                year = ref.get("year", "")

                if not title:
                    continue

                yield event.plain_result(f"\n[{i}/{len(top_refs)}] 📝 {title[:60]}...")

                try:
                    # Search CORE API
                    yield event.plain_result(f"   🔍 搜索: {title[:60]}...")
                    works = core_client.search_by_title(title, year=int(year) if year else None, limit=max_per_paper)

                    if not works:
                        yield event.plain_result(f"   ⚠️ 未找到相关论文")
                        failed += 1
                        continue

                    # Download first (most relevant) result
                    work = works[0]
                    source_urls = work.get("sourceFulltextUrls", []) or []

                    # 提取下载URL（优先使用arXiv链接）
                    pdf_url = None
                    for url in source_urls:
                        if 'arxiv.org/pdf' in url:
                            pdf_url = url
                            break
                    if not pdf_url and source_urls:
                        pdf_url = source_urls[0]

                    if not pdf_url:
                        yield event.plain_result(f"   ⚠️ 无可下载链接")
                        failed += 1
                        continue

                    # 确定文件名
                    arxiv_id = core_client.extract_arxiv_id(work)
                    if arxiv_id:
                        pdf_filename = f"{arxiv_id}.pdf"
                    else:
                        work_id = work.get('id', 'unknown')
                        safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                        pdf_filename = f"{work_id}_{safe_title}.pdf"

                    pdf_path = papers_path / pdf_filename

                    if pdf_path.exists():
                        yield event.plain_result(f"   ⏭️ PDF已存在，跳过")
                        skipped += 1
                        result = await engine.add_paper(str(pdf_path))
                        if result.get("status") == "success":
                            successful += 1
                            yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                        continue

                    # Download PDF
                    yield event.plain_result(f"   📥 下载: {pdf_filename}")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    }
                    pdf_response = requests.get(pdf_url, headers=headers, timeout=120, stream=True)

                    if pdf_response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        file_size = pdf_path.stat().st_size / (1024 * 1024)
                        yield event.plain_result(f"   ✅ 下载完成 ({file_size:.1f} MB)")

                        # Add to database
                        result = await engine.add_paper(str(pdf_path))
                        if result.get("status") == "success":
                            successful += 1
                            total_downloaded += 1
                            yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                        else:
                            yield event.plain_result(f"   ⚠️ 添加失败: {result.get('message', 'unknown')}")
                    else:
                        yield event.plain_result(f"   ❌ 下载失败: HTTP {pdf_response.status_code}")
                        failed += 1

                except Exception as e:
                    logger.error(f"处理论文失败: {e}")
                    yield event.plain_result(f"   ❌ 错误: {e}")
                    failed += 1

            # Summary
            output = f"""
📊 **CORE高频引用论文下载完成**

✅ 成功: {successful}
⏭️ 跳过: {skipped}
❌ 失败: {failed}
📥 新增下载: {total_downloaded}

📁 保存路径: {papers_dir}
💡 使用 /paper list 查看所有论文
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"CORE批量下载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 操作失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("arxiv_sync")
    async def cmd_arxiv_sync(self, event: AstrMessageEvent, confirm: str = ''):
        """Sync arxiv MCP downloaded papers to paperrag database (Admin)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result("⚠️ 即将扫描 MCP 已下载的论文并添加到数据库\n使用 /paper arxiv_sync confirm 确认执行")
            return

        # Get MCP storage path from configuration（支持跨平台配置）
        mcp_storage_path = self.config.get("arxiv_mcp_storage_path", "/Volumes/ext/arxiv")

        if not os.path.exists(mcp_storage_path):
            yield event.plain_result(f"❌ MCP存储路径不存在: {mcp_storage_path}")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        yield event.plain_result(f"📁 扫描MCP存储路径: {mcp_storage_path}")

        try:
            # Scan for PDF files
            mcp_path = Path(mcp_storage_path)
            pdf_files = list(mcp_path.glob("*.pdf"))

            # Filter out macOS metadata files
            pdf_files = [f for f in pdf_files if not _is_hidden_file(f)]

            if not pdf_files:
                yield event.plain_result("📭 MCP目录中没有找到PDF文件")
                return

            yield event.plain_result(f"📄 找到 {len(pdf_files)} 个PDF文件")

            # Get paperrag papers directory for display
            papers_dir = self.config.get("papers_dir", "./papers")

            successful = 0
            failed = 0
            already_in_db = 0

            for i, pdf_file in enumerate(pdf_files, 1):
                yield event.plain_result(f"\n[{i}/{len(pdf_files)}] 📄 {pdf_file.name}")

                # Check if already exists in paperrag directory
                papers_path = Path(papers_dir)
                dest_path = papers_path / pdf_file.name

                if dest_path.exists():
                    yield event.plain_result(f"   ⏭️ 论文已存在于paperrag目录，跳过")
                    already_in_db += 1
                    continue

                # Copy file to paperrag directory
                try:
                    import shutil
                    shutil.copy2(pdf_file, dest_path)
                    file_size = dest_path.stat().st_size / (1024 * 1024)
                    yield event.plain_result(f"   📋 已复制 ({file_size:.1f} MB)")
                except Exception as e:
                    logger.error(f"复制文件失败: {e}")
                    yield event.plain_result(f"   ❌ 复制失败: {e}")
                    failed += 1
                    continue

                # Add to database
                try:
                    result = await engine.add_paper(str(dest_path))
                    if result.get("status") == "success":
                        successful += 1
                        yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                    else:
                        yield event.plain_result(f"   ⚠️ 添加失败: {result.get('message', 'unknown')}")
                        failed += 1
                except Exception as e:
                    logger.error(f"添加论文失败: {e}")
                    yield event.plain_result(f"   ❌ 添加失败: {e}")
                    failed += 1

            # Summary
            output = f"""
📊 **MCP论文同步完成**

✅ 成功: {successful}
⏭️ 跳过(已存在): {already_in_db}
❌ 失败: {failed}

📁 MCP路径: {mcp_storage_path}
📁 paperrag路径: {papers_dir}
💡 使用 /paper list 查看所有论文
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"MCP同步失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 同步失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("arxiv_cleanup")
    async def cmd_arxiv_cleanup(self, event: AstrMessageEvent, confirm: str = ''):
        """Clean up old versions of arxiv papers, keeping only latest versions (Admin)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result("⚠️ 即将清理arXiv论文旧版本，只保留最新版本\n使用 /paper arxiv_cleanup confirm 确认执行")
            return

        mcp_storage_path = self.config.get("arxiv_mcp_storage_path", "/Volumes/ext/arxiv")

        if not os.path.exists(mcp_storage_path):
            yield event.plain_result(f"❌ MCP存储路径不存在: {mcp_storage_path}")
            return

        yield event.plain_result(f"🧹 扫描MCP存储路径: {mcp_storage_path}")

        try:
            import re
            from collections import defaultdict

            mcp_path = Path(mcp_storage_path)

            # Find all PDF files (excluding macOS metadata files)
            pdf_files = [f for f in mcp_path.glob("*.pdf") if not _is_hidden_file(f)]

            if not pdf_files:
                yield event.plain_result("📭 MCP目录中没有找到PDF文件")
                return

            # Group papers by base ID (without version suffix)
            # e.g., 2603.11298.pdf and 2603.11298v2.pdf -> base_id = 2603.11298
            papers_by_base = defaultdict(list)

            for pdf_file in pdf_files:
                filename = pdf_file.name
                # Match arxiv ID pattern: YYMM.NNNNN or YYMM.NNNNNvX
                match = re.match(r'^(\d{4}\.\d+)(v\d+)?\.pdf$', filename, re.IGNORECASE)
                if match:
                    base_id = match.group(1)  # e.g., "2603.11298"
                    version_str = match.group(2)  # e.g., "v2" or None
                    version = int(version_str[1:]) if version_str else 1
                    papers_by_base[base_id].append({
                        'file': pdf_file,
                        'version': version,
                        'is_latest': False
                    })
                else:
                    logger.debug(f"无法识别的文件名: {filename}")

            # Find papers with multiple versions
            multi_version_papers = {k: v for k, v in papers_by_base.items() if len(v) > 1}

            if not multi_version_papers:
                yield event.plain_result("✅ 没有发现多版本论文，无需清理")
                return

            yield event.plain_result(f"📋 发现 {len(multi_version_papers)} 篇多版本论文")

            # Mark latest versions
            deleted_count = 0
            kept_count = 0

            for base_id, versions in multi_version_papers.items():
                # Sort by version descending
                versions.sort(key=lambda x: x['version'], reverse=True)

                # Mark latest as kept
                versions[0]['is_latest'] = True
                kept_count += 1

                # Delete old versions
                for v in versions[1:]:
                    old_file = v['file']
                    version = v['version']
                    try:
                        file_size = old_file.stat().st_size / (1024 * 1024)
                        old_file.unlink()
                        deleted_count += 1
                        yield event.plain_result(
                            f"   🗑️ 删除旧版本: {old_file.name} (v{version}, {file_size:.1f} MB)"
                        )
                    except Exception as e:
                        logger.error(f"删除文件失败: {old_file}: {e}")
                        yield event.plain_result(f"   ❌ 删除失败: {old_file.name}")

            # Also clean up macOS metadata files
            metadata_files = [f for f in mcp_path.glob("._*")]
            metadata_count = 0
            for meta_file in metadata_files:
                try:
                    meta_file.unlink()
                    metadata_count += 1
                except Exception as e:
                    logger.error(f"删除metadata文件失败: {meta_file}: {e}")

            output = f"""
📊 **arXiv论文版本清理完成**

📄 多版本论文: {len(multi_version_papers)} 篇
✅ 保留最新版本: {kept_count} 个
🗑️ 删除旧版本: {deleted_count} 个
📦 清理metadata: {metadata_count} 个

💡 建议：修改 MCP 配置添加 --max-version=1 参数（如果支持）
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"清理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 清理失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("rebuild")
    async def cmd_rebuild(self, event: AstrMessageEvent, directory: str = '', confirm: str = ''):
        """Clear and rebuild document knowledge base (Admin)

        Args:
            directory: Document directory path (optional, use configured path by default)
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result("⚠️ This will delete all existing embeddings and recreate them!\nUse: /paper rebuild <directory> confirm")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        # Use provided directory or fallback to configured path
        papers_dir = directory or self.config.get("papers_dir", "./papers")

        # Check directory
        if not os.path.exists(papers_dir):
            yield event.plain_result(f"❌ Directory does not exist: {papers_dir}")
            return

        yield event.plain_result("🔄 Step 1/5: Clearing knowledge base...")

        try:
            # Clear database
            result = await engine.clear()
            if result.get("status") != "success":
                yield event.plain_result(f"❌ Failed to clear: {result.get('message', 'Unknown error')}")
                return
            yield event.plain_result("✅ Step 1/5: Knowledge base cleared")

        except Exception as e:
            logger.error(f"Failed to clear document library: {e}")
            yield event.plain_result(f"❌ Failed to clear: {e}")
            return

        # Delete figures and tables folders
        yield event.plain_result("🔄 Step 2/5: Clearing figures...")
        plugin_dir = Path(__file__).parent
        figures_dir = plugin_dir / "data" / "figures"
        tables_dir = plugin_dir / "data" / "tables"

        for target_dir, name in [(figures_dir, "figures"), (tables_dir, "tables")]:
            if target_dir.exists() and target_dir.is_dir():
                try:
                    import shutil
                    shutil.rmtree(target_dir)
                    logger.info(f"✅ Deleted {name} folder: {target_dir}")
                    yield event.plain_result(f"✅ Step 2/5: {name.capitalize()} folder cleared")
                except Exception as e:
                    logger.warning(f"Failed to delete {name} folder: {e}")
                    yield event.plain_result(f"⚠️ Failed to delete {name}: {e}")
            else:
                yield event.plain_result(f"✅ Step 2/5: No {name} folder found, skipping")

        yield event.plain_result("🔄 Step 3/5: Scanning documents...")

        # Scan documents
        doc_files = self._scan_documents(papers_dir)

        if not doc_files:
            yield event.plain_result("📭 No supported documents found")
            return

        yield event.plain_result(f"📄 Step 3/5: Found {len(doc_files)} documents")

        # Re-add documents
        import time
        start_time = time.time()
        total_files = len(doc_files)
        successful = 0
        failed = 0
        total_chunks = 0

        yield event.plain_result("🔄 Step 4/5: Rebuilding embeddings... (this may take a while)")

        for idx, doc_file in enumerate(doc_files, 1):
            try:
                result = await engine.add_paper(str(doc_file))
                if result.get("status") == "success":
                    successful += 1
                    total_chunks += result.get("chunks_added", 0)
                else:
                    failed += 1
                    logger.warning(f"Failed to add {doc_file.name}: {result.get('message', 'Unknown error')}")

                # Progress update every 5 files or at the end
                if idx % 5 == 0 or idx == total_files:
                    elapsed = time.time() - start_time
                    yield event.plain_result(f"⏳ Progress: {idx}/{total_files} ({(idx/total_files*100):.1f}%) - {successful} added, {failed} failed")

            except Exception as e:
                failed += 1
                logger.error(f"Error adding {doc_file.name}: {e}")

        elapsed = time.time() - start_time
        yield event.plain_result(
            f"✅ Rebuild complete!\n"
            f"   📄 Documents: {successful}/{total_files} successful\n"
            f"   📊 Total chunks: {total_chunks}\n"
            f"   ⏱️ Time: {elapsed:.1f}s"
        )

        # Clear cache
        self._response_cache.clear()

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("rebuildf")
    async def cmd_rebuild_file(self, event: AstrMessageEvent, file_name: str = ''):
        """Rebuild a single paper in knowledge base (Admin)

        Args:
            file_name: File name to rebuild (partial match supported)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not file_name:
            yield event.plain_result("❌ Please provide file name\nUsage: /paper rebuildf <filename>\nExample: /paper rebuildf 2508.09977v2（survey）.pdf")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        # Find the paper file
        papers_dir = self.config.get("papers_dir", "./papers")
        paper_path = None

        # 安全验证：确保 papers_dir 是绝对路径且存在
        papers_dir_resolved = Path(papers_dir).resolve()

        for ext in ['', '.pdf', '.PDF', '.docx', '.txt', '.md']:
            candidate = os.path.join(papers_dir, file_name + ext) if ext else os.path.join(papers_dir, file_name)
            candidate_resolved = Path(candidate).resolve()

            # 安全检查：确保路径在 papers_dir 内（防止路径遍历）
            if not str(candidate_resolved).startswith(str(papers_dir_resolved)):
                continue

            if os.path.exists(candidate) and os.path.isfile(candidate):
                paper_path = candidate
                break

        # Try partial match in papers_dir
        if not paper_path:
            for p in Path(papers_dir).glob("*"):
                if file_name.lower() in p.name.lower():
                    p_resolved = p.resolve()
                    # 安全检查
                    if not str(p_resolved).startswith(str(papers_dir_resolved)):
                        continue
                    if p.is_file():
                        paper_path = str(p)
                        break

        if not paper_path:
            yield event.plain_result(f"❌ File not found: {file_name}")
            return

        # 使用找到的实际文件名（避免用户输入部分名称导致误删其他文件）
        actual_file_name = os.path.basename(paper_path)

        yield event.plain_result(f"🔄 Rebuilding: {actual_file_name}")

        try:
            # Step 1: Delete existing data
            yield event.plain_result("🔍 Step 1/2: Deleting existing data...")
            delete_result = await engine.delete_paper(actual_file_name)
            if delete_result.get("status") != "success":
                logger.warning(f"⚠️ 删除旧数据失败: {delete_result.get('message', 'Unknown error')}")

            # Step 2: Re-add the paper
            yield event.plain_result("🔨 Step 2/2: Re-parsing and indexing...")
            add_result = await engine.add_paper(paper_path)

            if add_result.get("status") == "success":
                chunks_added = add_result.get("chunks_added", 0)
                yield event.plain_result(
                    f"✅ Rebuild complete!\n"
                    f"   📄 File: {os.path.basename(paper_path)}\n"
                    f"   📊 Chunks: {chunks_added}"
                )
            else:
                yield event.plain_result(f"❌ Rebuild failed: {add_result.get('message', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Failed to rebuild paper: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ Failed to rebuild: {e}")

    @paper_commands.command("graph_build")
    async def cmd_graph_build(self, event: AstrMessageEvent, confirm: str = ''):
        """Build knowledge graph from indexed documents

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result(
                "⚠️ 即将从已索引的文档构建知识图谱\n"
                "注意：构建过程可能较慢（需要调用 LLM 抽取三元组）\n"
                "使用 /paper graph_build confirm 确认执行"
            )
            return

        # 检查 Graph RAG 是否启用
        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用\n请在插件配置中启用 enable_graph_rag")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        yield event.plain_result("🔨 正在构建知识图谱...\n⏳ 请稍候，这可能需要几分钟...")

        try:
            # 获取索引管理器
            index_manager = engine._ensure_index_manager_initialized()

            # 获取所有论文列表（用于逐篇加载chunks）
            yield event.plain_result("📖 正在从向量数据库读取论文列表...")

            try:
                papers = await index_manager.list_unique_documents()
            except Exception as e:
                yield event.plain_result(f"❌ 无法获取论文列表: {e}\n请确保已使用 /paper add 添加文档")
                return

            if not papers:
                yield event.plain_result("📭 向量数据库中未找到已索引的文档\n请先使用 /paper add 添加文档")
                return

            paper_names = [p.get("file_name", "") for p in papers if p.get("file_name")]
            yield event.plain_result(f"📚 找到 {len(paper_names)} 篇论文\n🔨 正在逐篇加载所有文档块...")

            # 导入必要的模块
            try:
                from .graph_rag_engine import GraphRAGEngine, GraphRAGConfig, MemoryGraphStore
            except Exception:
                from graph_rag_engine import GraphRAGEngine, GraphRAGConfig, MemoryGraphStore

            try:
                from .graph_builder import MultimodalGraphBuilder
            except Exception:
                from graph_builder import MultimodalGraphBuilder

            # ChunkNode 类用于适配 GraphBuilder
            class ChunkNode:
                """适配 GraphBuilder 的 Node 结构"""
                def __init__(self, chunk: Dict[str, Any]):
                    self.text = chunk.get("text", "")
                    self.metadata = chunk.get("metadata", {})

            import json

            yield event.plain_result(f"📑 开始逐篇构建知识图谱 ({len(paper_names)} 篇论文)...")

            # 创建 GraphRAGConfig（只创建一次）
            graph_config = self._create_graph_rag_config()

            # 根据配置决定存储类型
            if graph_config.storage_type == "neo4j":
                from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
                from .graph_rag_engine import SimplePropertyGraphStoreAdapter
                raw_store = Neo4jPropertyGraphStore(
                    url=graph_config.neo4j_uri,
                    username=graph_config.neo4j_user,
                    password=graph_config.neo4j_password,
                    database="neo4j",
                    refresh_schema=True
                )
                graph_store = SimplePropertyGraphStoreAdapter(raw_store)
                logger.info(f"[GraphRAG] 使用 Neo4j 存储: {graph_config.neo4j_uri}")
            else:
                graph_store = MemoryGraphStore()
                logger.info("[GraphRAG] 使用内存存储")

            builder = MultimodalGraphBuilder(config=graph_config, context=self.context)

            # 初始化 LLM（只初始化一次）
            await builder._ensure_llm_initialized()

            # 逐篇处理：每篇论文处理完后立即构建图谱，不累积所有 chunks
            total_stats = {
                "entities_added": 0,
                "text_triplets_added": 0,
                "image_entities_added": 0,
                "cross_modal_triplets_added": 0,
                "chunks_processed": 0,
                "chunks_with_images": 0,
                "chunks_failed": 0,
                "chunks_empty": 0
            }

            # 第二步：逐篇加载 chunks 并立即构建图谱
            await index_manager._ensure_collection()
            collection = index_manager._collection

            for i, paper_name in enumerate(paper_names):
                paper_name_escaped = paper_name.replace('"', '\\"')
                try:
                    # collection.query 是同步方法，需要用 run_in_executor 包装
                    _collection = cast(Any, collection)
                    raw_results = cast(
                        List[Dict[str, Any]],
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda pn=paper_name_escaped: _collection.query(
                                expr=f'metadata["file_name"] == "{pn}"',
                                output_fields=["id", "text", "metadata"],
                            )
                        )
                    )

                    if not raw_results:
                        continue

                    yield event.plain_result(f"📄 [{i+1}/{len(paper_names)}] {paper_name} ({len(raw_results)} chunks)")

                    # 解析该论文的 chunks
                    paper_chunks = []
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
                        paper_chunks.append(chunk)

                    # 立即为该论文创建节点并构建图谱
                    nodes = [ChunkNode(chunk) for chunk in paper_chunks]
                    stats = await builder.build_from_nodes(nodes, graph_store)

                    # 每篇论文处理完后保存并清理内存
                    if hasattr(graph_store, 'save'):
                        graph_store.save(force=True)
                    gc.collect()

                    # 累积统计
                    total_stats["entities_added"] += stats.get("entities_added", 0)
                    total_stats["text_triplets_added"] += stats.get("text_triplets_added", 0)
                    total_stats["image_entities_added"] += stats.get("image_entities_added", 0)
                    total_stats["cross_modal_triplets_added"] += stats.get("cross_modal_triplets_added", 0)
                    total_stats["chunks_with_images"] += stats.get("chunks_with_images", 0)
                    total_stats["chunks_failed"] += stats.get("chunks_failed", 0)
                    total_stats["chunks_empty"] += stats.get("chunks_empty", 0)
                    total_stats["chunks_processed"] += stats.get("chunks_processed", 0)

                except Exception as e:
                    logger.warning(f"处理论文 {paper_name} 失败: {e}")

                # 每隔20篇论文重新确保连接有效
                if (i + 1) % 20 == 0:
                    await index_manager._ensure_collection()
                    collection = index_manager._collection

                # 更新进度
                if (i + 1) % 5 == 0 or (i + 1) == len(paper_names):
                    yield event.plain_result(f"📥 构建进度: {i + 1}/{len(paper_names)} 篇论文...")

            # 保存图谱到磁盘
            graph_store.save(force=True)

            # 输出结果
            text_triplets = total_stats.get('text_triplets_added', 0)
            cross_triplets = total_stats.get('cross_modal_triplets_added', 0)
            total_triplets_val = text_triplets + cross_triplets
            output = f"""✅ **知识图谱构建完成**

📊 构建统计：
   • 处理论文：{len(paper_names)} 篇
   • 处理文档块：{total_stats.get('chunks_processed', 0)}
   • 添加实体：{total_stats.get('entities_added', 0)}
   • 文本三元组：{text_triplets}
   • 图片实体：{total_stats.get('image_entities_added', 0)}
   • 跨模态三元组：{cross_triplets}
   • 总三元组：{total_triplets_val}
   • 空块数：{total_stats.get('chunks_empty', 0)}
   • 失败块数：{total_stats.get('chunks_failed', 0)}

💾 图谱已自动保存到磁盘
💡 使用 /paper graph_stats 查看图谱详情"""
            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 构建失败: {e}")

    @paper_commands.command("graph_stats")
    async def cmd_graph_stats(self, event: AstrMessageEvent):
        """Show knowledge graph statistics"""
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用\n请在插件配置中启用 enable_graph_rag")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        try:
            from .graph_rag_engine import GraphRAGEngine, GraphRAGConfig
        except Exception:
            from graph_rag_engine import GraphRAGEngine, GraphRAGConfig

        try:
            graph_config = self._create_graph_rag_config()

            graph_engine = GraphRAGEngine(graph_config, engine, self.context)
            await graph_engine.initialize()

            stats = await graph_engine.get_graph_stats()

            storage_type = self.config.get("graph_rag", {}).get("storage_type", "memory")

            # 获取持久化状态
            is_dirty = stats.get('is_dirty', False)
            last_save = stats.get('last_save_time', None)
            storage_path = stats.get('storage_path', 'N/A')

            dirty_indicator = "⚠️ 有未保存的变更" if is_dirty else "✅ 已保存"

            output = f"""📊 **知识图谱统计**

存储类型：{storage_type}
   • 实体数量：{stats.get('entity_count', 0)}
   • 关系数量：{stats.get('relation_count', 0)}
   • 索引大小：{stats.get('index_size', 0)}

💾 持久化状态：
   • 状态：{dirty_indicator}
   • 存储路径：{storage_path}
   • 上次保存：{last_save if last_save else '从未保存'}

💡 使用 /paper graph_build confirm 构建图谱
💡 使用 /paper graph_rebuild confirm 重新构建图谱
💡 使用 /paper graph_clear confirm 清空图谱"""

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"获取图谱统计失败: {e}")
            yield event.plain_result(f"❌ 获取统计失败: {e}")

    @paper_commands.command("graph_rebuild")
    async def cmd_graph_rebuild(self, event: AstrMessageEvent, confirm: str = ''):
        """Rebuild knowledge graph from scratch (clear + rebuild)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result(
                "⚠️ 即将清空并重新构建知识图谱\n"
                "此操作不可恢复！\n"
                "使用 /paper graph_rebuild confirm 确认执行"
            )
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用\n请在插件配置中启用 enable_graph_rag")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        try:
            from .graph_rag_engine import GraphRAGEngine, GraphRAGConfig, MemoryGraphStore
        except Exception:
            from graph_rag_engine import GraphRAGEngine, GraphRAGConfig, MemoryGraphStore

        # 步骤1: 清空现有图谱
        yield event.plain_result("🗑️ 正在清空现有知识图谱...")

        try:
            graph_config = self._create_graph_rag_config()

            graph_engine = GraphRAGEngine(graph_config, engine, self.context)
            await graph_engine.initialize()

            clear_result = await graph_engine.clear_graph()
            if clear_result.get("status") == "success":
                yield event.plain_result("✅ 现有图谱已清空")
            else:
                logger.warning(f"清空图谱返回: {clear_result}")
        except Exception as e:
            logger.error(f"清空图谱失败: {e}")
            yield event.plain_result(f"⚠️ 清空图谱时出现错误，继续重建: {e}")

        # 步骤2: 重新构建图谱（复用 graph_build 的逻辑）
        yield event.plain_result("\n🔨 正在重新构建知识图谱...\n⏳ 请稍候，这可能需要几分钟...")

        try:
            index_manager = engine._ensure_index_manager_initialized()

            yield event.plain_result("📖 正在从向量数据库读取论文列表...")

            try:
                papers = await index_manager.list_unique_documents()
            except Exception as e:
                yield event.plain_result(f"❌ 无法获取论文列表: {e}\n请确保已使用 /paper add 添加文档")
                return

            if not papers:
                yield event.plain_result("📭 向量数据库中未找到已索引的文档\n请先使用 /paper add 添加文档")
                return

            paper_names = [p.get("file_name", "") for p in papers if p.get("file_name")]
            yield event.plain_result(f"📚 找到 {len(paper_names)} 篇论文\n🔨 正在逐篇加载所有文档块...")

            try:
                from .graph_builder import MultimodalGraphBuilder
            except Exception:
                from graph_builder import MultimodalGraphBuilder

            class ChunkNode:
                """适配 GraphBuilder 的 Node 结构"""
                def __init__(self, chunk: Dict[str, Any]):
                    self.text = chunk.get("text", "")
                    self.metadata = chunk.get("metadata", {})

            yield event.plain_result(f"📑 开始逐篇构建知识图谱 ({len(paper_names)} 篇论文)...")

            graph_config = self._create_graph_rag_config()

            # 根据配置决定存储类型
            if graph_config.storage_type == "neo4j":
                from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
                from .graph_rag_engine import SimplePropertyGraphStoreAdapter
                raw_store = Neo4jPropertyGraphStore(
                    url=graph_config.neo4j_uri,
                    username=graph_config.neo4j_user,
                    password=graph_config.neo4j_password,
                    database="neo4j",
                    refresh_schema=True
                )
                graph_store = SimplePropertyGraphStoreAdapter(raw_store)
                logger.info(f"[GraphRAG] 使用 Neo4j 存储: {graph_config.neo4j_uri}")
            else:
                graph_store = MemoryGraphStore()
                logger.info("[GraphRAG] 使用内存存储")

            builder = MultimodalGraphBuilder(config=graph_config, context=self.context)
            await builder._ensure_llm_initialized()

            total_stats = {
                "entities_added": 0,
                "text_triplets_added": 0,
                "image_entities_added": 0,
                "cross_modal_triplets_added": 0,
                "chunks_processed": 0,
                "chunks_with_images": 0,
                "chunks_failed": 0,
                "chunks_empty": 0
            }

            await index_manager._ensure_collection()
            collection = index_manager._collection

            for i, paper_name in enumerate(paper_names):
                paper_name_escaped = paper_name.replace('"', '\\"')
                try:
                    _collection = cast(Any, collection)
                    raw_results = cast(
                        List[Dict[str, Any]],
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda pn=paper_name_escaped: _collection.query(
                                expr=f'metadata["file_name"] == "{pn}"',
                                output_fields=["id", "text", "metadata"],
                            )
                        )
                    )

                    chunks = []
                    for r in raw_results:
                        text = r.get("text", "")
                        if text and len(text) >= 50:
                            chunks.append(ChunkNode(r))

                    if not chunks:
                        continue

                    # 使用批量构建（每篇论文一个批次）
                    result = await builder.build_from_nodes(chunks, graph_store)

                    total_stats["entities_added"] += result.get("entities_added", 0)
                    total_stats["text_triplets_added"] += result.get("text_triplets_added", 0)
                    total_stats["image_entities_added"] += result.get("image_entities_added", 0)
                    total_stats["cross_modal_triplets_added"] += result.get("cross_modal_triplets_added", 0)
                    total_stats["chunks_processed"] += result.get("chunks_processed", 0)
                    total_stats["chunks_with_images"] += result.get("chunks_with_images", 0)
                    total_stats["chunks_failed"] += result.get("chunks_failed", 0)
                    total_stats["chunks_empty"] += result.get("chunks_empty", 0)

                    if (i + 1) % 5 == 0 or i == len(paper_names) - 1:
                        yield event.plain_result(
                            f"📊 进度: {i + 1}/{len(paper_names)} 篇论文\n"
                            f"   本批次: 实体+{result.get('entities_added', 0)}, "
                            f"三元组+{result.get('text_triplets_added', 0)}"
                        )

                except Exception as e:
                    logger.error(f"处理论文 {paper_name} 失败: {e}")
                    total_stats["chunks_failed"] += 1
                    continue

            # 保存图谱
            if graph_config.storage_type == "memory":
                graph_engine._graph_store = graph_store
            else:
                await graph_engine._save_to_neo4j(graph_store)

            # 更新内存图谱引用
            if hasattr(graph_engine, '_graph_store'):
                graph_engine._graph_store = graph_store

            output = f"""🎉 **知识图谱重建完成！**

📊 **构建统计**
   • 实体数量：{total_stats['entities_added']}
   • 文本三元组：{total_stats['text_triplets_added']}
   • 图片实体：{total_stats['image_entities_added']}
   • 跨模态三元组：{total_stats['cross_modal_triplets_added']}
   • 处理块数：{total_stats['chunks_processed']}
   • 失败块数：{total_stats['chunks_failed']}

💾 图谱已自动保存到磁盘
💡 使用 /paper graph_stats 查看图谱详情"""

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"重建知识图谱失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 重建失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_clear")
    async def cmd_graph_clear(self, event: AstrMessageEvent, confirm: str = ''):
        """Clear knowledge graph (Admin)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用")
            return

        if confirm != "confirm":
            yield event.plain_result("⚠️ 即将清空知识图谱\n此操作不可恢复！\n使用 /paper graph_clear confirm 确认执行")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        try:
            from .graph_rag_engine import GraphRAGEngine, GraphRAGConfig
        except Exception:
            from graph_rag_engine import GraphRAGEngine, GraphRAGConfig

        try:
            graph_config = self._create_graph_rag_config()

            graph_engine = GraphRAGEngine(graph_config, engine, self.context)
            await graph_engine.initialize()

            result = await graph_engine.clear_graph()

            if result.get("status") == "success":
                yield event.plain_result("✅ 知识图谱已清空")
            else:
                yield event.plain_result(f"❌ 清空失败: {result.get('message', '未知错误')}")

        except Exception as e:
            logger.error(f"清空图谱失败: {e}")
            yield event.plain_result(f"❌ 清空失败: {e}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_backup")
    async def cmd_graph_backup(self, event: AstrMessageEvent, mode: str = 'online'):
        """Backup Neo4j knowledge graph (Admin)

        Args:
            mode: 'online' (Cypher export, no downtime) or 'offline' (file copy, requires stop)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用")
            return

        from .graph_rag_engine import GraphRAGConfig
        graph_config = self._create_graph_rag_config()

        if graph_config.storage_type != "neo4j":
            yield event.plain_result("❌ 只有 Neo4j 存储模式支持备份")
            return

        yield event.plain_result(f"🔄 开始备份图谱 (模式: {mode})...")

        try:
            if mode == 'offline':
                result = await self._offline_backup(graph_config)
            else:
                result = await self._online_backup(graph_config)

            if result["status"] == "success":
                backup_file = result.get("backup_file", "unknown")
                size = result.get("size", 0)
                nodes = result.get("nodes", 0)
                rels = result.get("relations", 0)

                output = f"""✅ **图谱备份完成！**

📦 **备份文件**: `{backup_file}`
📊 **数据统计**:
   • 节点数: {nodes}
   • 关系数: {rels}
   • 文件大小: {size}

💡 使用 `/paper graph_restore {backup_file}` 恢复备份"""

                if mode == 'offline':
                    output += "\n⚠️ 离线备份已完成，Neo4j 服务已恢复"

                yield event.plain_result(output)
            else:
                yield event.plain_result(f"❌ 备份失败: {result.get('message', '未知错误')}")

        except Exception as e:
            logger.error(f"备份图谱失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 备份失败: {e}")

    async def _online_backup(self, graph_config: "GraphRAGConfig") -> dict:
        """在线备份：使用 Cypher 导出为 JSON"""
        import json
        import gzip
        from pathlib import Path
        from datetime import datetime

        try:
            from neo4j import GraphDatabase
        except ImportError:
            return {"status": "error", "message": "请安装 neo4j 驱动: pip install neo4j"}

        # 连接到 Neo4j
        driver = GraphDatabase.driver(
            graph_config.neo4j_uri,
            auth=(graph_config.neo4j_user, graph_config.neo4j_password)
        )

        # 获取插件目录（用于相对路径）
        plugin_dir = Path(__file__).parent
        backup_dir = plugin_dir / "data" / "graph_store"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"neo4j_backup_{timestamp}.json.gz"

        try:
            with driver.session() as session:
                # 导出节点
                nodes_data = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels,
                           properties(n) as props,
                           elementId(n) as id
                """).data()

                # 导出关系
                rels_data = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN type(r) as rel_type,
                           properties(r) as props,
                           elementId(startNode(r)) as start_id,
                           elementId(endNode(r)) as end_id
                """).data()

            nodes_count = len(nodes_data)
            rels_count = len(rels_data)

            backup = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "mode": "online",
                "nodes": nodes_data,
                "relationships": rels_data,
                "node_count": nodes_count,
                "rel_count": rels_count
            }

            # 压缩写入
            with gzip.open(backup_file, 'wt', encoding='utf-8') as f:
                json.dump(backup, f, indent=2, ensure_ascii=False)

            size = backup_file.stat().st_size
            size_str = self._format_size(size)

            logger.info(f"[GraphRAG] 在线备份完成: {backup_file}, {nodes_count} 节点, {rels_count} 关系")

            return {
                "status": "success",
                "backup_file": str(backup_file.relative_to(plugin_dir)),
                "size": size_str,
                "nodes": nodes_count,
                "relations": rels_count
            }

        finally:
            driver.close()

    async def _offline_backup(self, graph_config: "GraphRAGConfig") -> dict:
        """离线备份：复制 Neo4j 数据目录"""
        import shutil
        import subprocess
        from pathlib import Path
        from datetime import datetime

        # Neo4j 数据目录
        neo4j_data_dir = Path("/opt/homebrew/var/neo4j/data")

        if not neo4j_data_dir.exists():
            return {"status": "error", "message": f"Neo4j 数据目录不存在: {neo4j_data_dir}"}

        # 获取插件目录（用于相对路径）
        plugin_dir = Path(__file__).parent
        backup_dir = plugin_dir / "data" / "graph_store"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = backup_dir / f"neo4j_backup_{timestamp}"
        backup_subdir.mkdir(parents=True, exist_ok=True)

        try:
            # 检查 neo4j 命令是否可用
            neo4j_bin = shutil.which("neo4j")
            if not neo4j_bin:
                return {"status": "error", "message": "neo4j 命令未找到，请确保 Neo4j 已安装"}

            # 停止 Neo4j
            logger.info("[GraphRAG] 停止 Neo4j 服务...")
            stop_result = subprocess.run(["neo4j", "stop"], capture_output=True, text=True)
            if stop_result.returncode != 0:
                logger.warning(f"[GraphRAG] neo4j stop 返回: {stop_result.stdout} {stop_result.stderr}")

            # 等待服务完全停止
            import time
            time.sleep(3)

            # 复制数据目录
            source_db = neo4j_data_dir / "databases" / "neo4j"
            dest_db = backup_subdir / "databases" / "neo4j"

            if source_db.exists():
                shutil.copytree(source_db, dest_db)
            else:
                return {"status": "error", "message": f"数据库目录不存在: {source_db}"}

            # 复制事务日志（可选）
            source_tx = neo4j_data_dir / "transactions" / "neo4j"
            dest_tx = backup_subdir / "transactions" / "neo4j"
            if source_tx.exists():
                shutil.copytree(source_tx, dest_tx)

            # 重启 Neo4j
            logger.info("[GraphRAG] 重启 Neo4j 服务...")
            start_result = subprocess.run(["neo4j", "start"], capture_output=True, text=True)
            if start_result.returncode != 0:
                logger.warning(f"[GraphRAG] neo4j start 返回: {start_result.stdout} {start_result.stderr}")

            # 等待 Neo4j 启动
            time.sleep(5)

            # 计算备份大小
            total_size = sum(f.stat().st_size for f in backup_subdir.rglob('*') if f.is_file())
            size_str = self._format_size(total_size)

            # 节点数估计（通过文件数）
            node_files = list((backup_subdir / "databases" / "neo4j").rglob("*.db"))[:5]
            nodes_est = "多个"

            logger.info(f"[GraphRAG] 离线备份完成: {backup_subdir}")

            return {
                "status": "success",
                "backup_file": str(backup_subdir.relative_to(plugin_dir)),
                "size": size_str,
                "nodes": nodes_est,
                "relations": "多个"
            }

        except Exception as e:
            logger.error(f"[GraphRAG] 离线备份失败: {e}")
            # 尝试重启 Neo4j
            try:
                subprocess.run(["neo4j", "start"], capture_output=True)
            except:
                pass
            return {"status": "error", "message": str(e)}

    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_restore")
    async def cmd_graph_restore(self, event: AstrMessageEvent, backup_file: str = ''):
        """Restore Neo4j knowledge graph from backup (Admin)

        Args:
            backup_file: 备份文件名（从 data/graph_store 目录）
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用")
            return

        if not backup_file:
            # 列出可用备份
            from pathlib import Path
            plugin_dir = Path(__file__).parent
            backup_dir = plugin_dir / "data" / "graph_store"

            backups = sorted(backup_dir.glob("neo4j_backup_*"), reverse=True)
            if not backups:
                yield event.plain_result("❌ 未找到任何备份文件")
                return

            msg = "📦 **可用备份列表**:\n\n"
            for i, b in enumerate(backups[:10], 1):
                size = b.stat().st_size
                size_str = self._format_size(size)
                mtime = datetime.fromtimestamp(b.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                msg += f"{i}. `{b.name}`\n   大小: {size_str}, 修改: {mtime}\n\n"

            msg += "💡 使用 `/paper graph_restore <文件名>` 恢复备份"
            yield event.plain_result(msg)
            return

        from .graph_rag_engine import GraphRAGConfig
        graph_config = self._create_graph_rag_config()

        if graph_config.storage_type != "neo4j":
            yield event.plary_result("❌ 只有 Neo4j 存储模式支持恢复")
            return

        yield event.plain_result(f"🔄 正在恢复备份: {backup_file}...")

        try:
            result = await self._restore_backup(backup_file, graph_config)

            if result["status"] == "success":
                yield event.plain_result(f"""✅ **备份恢复完成！**

📦 **已恢复**: `{backup_file}`
📊 **数据**: {result.get('nodes', '?')} 节点, {result.get('relations', '?')} 关系

⚠️ 如果使用离线备份，恢复后需要重启 Neo4j 服务""")
            else:
                yield event.plain_result(f"❌ 恢复失败: {result.get('message', '未知错误')}")

        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            yield event.plain_result(f"❌ 恢复失败: {e}")

    async def _restore_backup(self, backup_file: str, graph_config: "GraphRAGConfig") -> dict:
        """从备份恢复"""
        import json
        import gzip
        from pathlib import Path

        try:
            from neo4j import GraphDatabase
        except ImportError:
            return {"status": "error", "message": "请安装 neo4j 驱动: pip install neo4j"}

        plugin_dir = Path(__file__).parent
        backup_path = plugin_dir / "data" / "graph_store" / backup_file

        if not backup_path.exists():
            return {"status": "error", "message": f"备份文件不存在: {backup_path}"}

        driver = GraphDatabase.driver(
            graph_config.neo4j_uri,
            auth=(graph_config.neo4j_user, graph_config.neo4j_password)
        )

        try:
            # 根据文件扩展名判断格式
            if str(backup_path).endswith('.gz'):
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                    backup = json.load(f)
            else:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup = json.load(f)

            nodes = backup.get("nodes", [])
            rels = backup.get("relationships", [])

            with driver.session() as session:
                # 清空现有数据
                session.run("MATCH (n) DETACH DELETE n")

                # 恢复节点
                for node in nodes:
                    labels = node.get("labels", [])
                    props = node.get("props", {})
                    label_str = ":" + ":".join(labels) if labels else ""

                    # 过滤不支持的属性类型
                    clean_props = {}
                    for k, v in props.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            clean_props[k] = v

                    if clean_props:
                        props_str = "{" + ", ".join(
                            f"k: {repr(v)}" for k, v in clean_props.items()
                        ) + "}"
                        session.run(f"CREATE (n{label_str} {props_str})")
                    else:
                        session.run(f"CREATE (n{label_str})")

                # 恢复关系
                for rel in rels:
                    rel_type = rel.get("rel_type", "REL")
                    start_id = rel.get("start_id")
                    end_id = rel.get("end_id")
                    props = rel.get("props", {})

                    if start_id and end_id:
                        clean_props = {}
                        for k, v in props.items():
                            if isinstance(v, (str, int, float, bool)) or v is None:
                                clean_props[k] = v

                        if clean_props:
                            props_str = "{" + ", ".join(
                                f"k: {repr(v)}" for k, v in clean_props.items()
                            ) + "}"
                            session.run(f"""
                                MATCH (a), (b)
                                WHERE elementId(a) = $start_id AND elementId(b) = $end_id
                                CREATE (a)-[r:{rel_type} {props_str}]->(b)
                            """, start_id=start_id, end_id=end_id)
                        else:
                            session.run(f"""
                                MATCH (a), (b)
                                WHERE elementId(a) = $start_id AND elementId(b) = $end_id
                                CREATE (a)-[r:{rel_type}]->(b)
                            """, start_id=start_id, end_id=end_id)

            logger.info(f"[GraphRAG] 备份恢复完成: {len(nodes)} 节点, {len(rels)} 关系")

            return {
                "status": "success",
                "nodes": len(nodes),
                "relations": len(rels)
            }

        finally:
            driver.close()

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_backup_list")
    async def cmd_graph_backup_list(self, event: AstrMessageEvent):
        """List available graph backups (Admin)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用")
            return

        from pathlib import Path
        from datetime import datetime

        plugin_dir = Path(__file__).parent
        backup_dir = plugin_dir / "data" / "graph_store"

        # 查找两种备份格式
        json_backups = list(backup_dir.glob("neo4j_backup_*.json.gz"))
        dir_backups = list(backup_dir.glob("neo4j_backup_*/"))

        all_backups = []
        for b in json_backups:
            all_backups.append({
                "name": b.name,
                "path": b,
                "size": b.stat().st_size,
                "mtime": b.stat().st_mtime,
                "type": "online (JSON.gz)"
            })
        for b in dir_backups:
            total_size = sum(f.stat().st_size for f in b.rglob("*") if f.is_file())
            all_backups.append({
                "name": b.name,
                "path": b,
                "size": total_size,
                "mtime": b.stat().st_mtime,
                "type": "offline (directory)"
            })

        if not all_backups:
            yield event.plain_result("❌ 未找到任何备份文件\n\n💡 使用 `/paper graph_backup` 创建备份")
            return

        # 按时间排序
        all_backups.sort(key=lambda x: x["mtime"], reverse=True)

        msg = "📦 **图谱备份列表**:\n\n"
        for i, b in enumerate(all_backups[:10], 1):
            size_str = self._format_size(b["size"])
            mtime = datetime.fromtimestamp(b["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
            msg += f"{i}. `{b['name']}`\n"
            msg += f"   类型: {b['type']}, 大小: {size_str}\n"
            msg += f"   时间: {mtime}\n\n"

        msg += "💡 使用 `/paper graph_restore <文件名>` 恢复备份\n"
        msg += "💡 使用 `/paper graph_backup` 创建新备份"

        yield event.plain_result(msg)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_link")
    async def cmd_graph_link(self, event: AstrMessageEvent, action: str = 'status'):
        """Manage Neo4j data symlink (Admin)

        Args:
            action: 'create' | 'remove' | 'status' (default: status)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        from pathlib import Path
        import os

        # Neo4j 原始数据目录
        neo4j_db_path = Path("/opt/homebrew/var/neo4j/data/databases/neo4j")

        # 插件内的链接目标路径
        plugin_dir = Path(__file__).parent
        graph_store_dir = plugin_dir / "data" / "graph_store"
        graph_store_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = graph_store_dir / "neo4j_data"

        if action == 'status':
            # 检查符号链接状态
            if symlink_path.is_symlink():
                target = os.readlink(symlink_path)
                exists = symlink_path.exists()
                real_path = symlink_path.resolve()

                status = "✅ 正常" if exists else "⚠️ 链接断裂（目标不存在）"

                msg = f"""📊 **Neo4j 符号链接状态**

🔗 链接路径: `data/graph_store/neo4j_data`
📍 目标路径: `{target}`
📁 解析路径: `{real_path}`
📦 状态: {status}

💡 可用操作:
  `/paper graph_link create` - 创建/重建链接
  `/paper graph_link remove` - 删除链接"""
            elif symlink_path.exists():
                status = "⚠️ 存在同名文件/目录（非符号链接）"
                msg = f"""📊 **Neo4j 符号链接状态**

⚠️  `data/graph_store/neo4j_data` 已存在但不是符号链接
   状态: {status}

💡 请先删除或备份后再创建链接:
  `/paper graph_link remove`"""
            else:
                msg = f"""📊 **Neo4j 符号链接状态**

❌ 符号链接未创建

💡 创建链接:
  `/paper graph_link create`"""

            yield event.plain_result(msg)

        elif action == 'create':
            if not neo4j_db_path.exists():
                yield event.plain_result(f"❌ Neo4j 数据目录不存在: `{neo4j_db_path}`\n\n请确保 Neo4j 已安装并初始化")
                return

            # 如果已存在符号链接，先删除
            if symlink_path.is_symlink():
                symlink_path.unlink()
                logger.info(f"[GraphRAG] 已删除旧符号链接")
            elif symlink_path.exists():
                yield event.plain_result(f"⚠️ `data/graph_store/neo4j_data` 已存在且不是符号链接\n请先手动删除后再试")
                return

            try:
                # 创建符号链接（使用相对路径）
                # 从 graph_store 目录到 neo4j 数据目录的相对路径
                rel_path = os.path.relpath(neo4j_db_path, graph_store_dir)
                os.symlink(rel_path, symlink_path)

                yield event.plain_result(f"""✅ **符号链接创建成功！**

🔗 链接: `data/graph_store/neo4j_data`
📍 指向: `{rel_path}` (相对路径)

💡 现在可以直接在 `data/graph_store/neo4j_data` 访问 Neo4j 数据

⚠️ 注意: 删除此链接不会影响原始数据，但重建需要此命令""")

            except Exception as e:
                yield event.plain_result(f"❌ 创建符号链接失败: {e}")

        elif action == 'remove':
            if not symlink_path.exists():
                yield event.plain_result("❌ 符号链接不存在，无需删除")
                return

            if not symlink_path.is_symlink():
                yield event.plain_result("⚠️ `data/graph_store/neo4j_data` 不是符号链接，无法使用此命令删除\n请手动删除")
                return

            try:
                symlink_path.unlink()
                yield event.plain_result("✅ 符号链接已删除\n\n⚠️ 删除链接不影响原始 Neo4j 数据\n💡 使用 `/paper graph_link create` 重新创建链接")
            except Exception as e:
                yield event.plain_result(f"❌ 删除符号链接失败: {e}")

        else:
            yield event.plain_result(f"❌ 未知操作: {action}\n\n可用操作: `status` | `create` | `remove`")

    # ==================== 创意生成命令 ====================

    @filter.command_group("idea")
    def idea_commands(self):
        """研究创意生成命令组
        explore     - 探索研究想法（完整流程）
        analyze     - 分析研究主题
        search      - 多源知识检索
        generate    - 基于知识上下文生成想法
        """
        pass

    @idea_commands.command("explore")
    async def cmd_idea_explore(self, event: AstrMessageEvent,
                                topic: str = '',
                                depth: str = "standard",
                                num_ideas: int = 3):
        """
        探索研究想法（完整流程）

        Args:
            topic: 研究主题描述
            depth: 分析深度 (quick/standard/deep)
            num_ideas: 生成想法数量
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not topic:
            yield event.plain_result("📚 Usage: /idea explore <研究主题>\nExample: /idea explore 大语言模型在医学诊断中的应用")
            return

        yield event.plain_result(f"🔍 正在分析研究主题...\n主题: {topic}")

        try:
            # 导入创意引擎
            from .idea_engine import IdeaEngine

            # 获取RAG引擎
            rag_engine = self._get_engine()

            # 创建创意引擎
            idea_engine = IdeaEngine(context=self.context, rag_engine=rag_engine)

            # 1. 分析主题
            yield event.plain_result("📊 正在分析研究领域...")
            analysis = await idea_engine.analyze_topic(topic, depth)

            if not analysis:
                yield event.plain_result("❌ 主题分析失败")
                return

            analysis_output = f"""**📊 主题分析结果**

**领域**: {analysis.domain}

**关键词**: {', '.join(analysis.keywords[:8])}

**探索角度**: {', '.join(analysis.exploration_angles)}

**摘要**: {analysis.summary}
"""
            yield event.plain_result(analysis_output)

            # 2. 检索知识
            yield event.plain_result("🌐 正在检索网络资源 + 📚 本地论文库...")
            all_queries = analysis.search_queries + analysis.local_rag_queries
            knowledge = await idea_engine.search_knowledge(all_queries, local_rag_top_k=3, web_top_k=5)

            stats_output = f"""
✅ 检索完成
- 网络资源: {knowledge['stats']['web_count']} 条
- 本地论文: {knowledge['stats']['local_count']} 条
"""
            yield event.plain_result(stats_output)

            # 3. 生成想法
            yield event.plain_result("💡 正在生成研究想法...")
            ideas = await idea_engine.generate_ideas(
                knowledge_context=knowledge['fused_context'],
                research_domain=analysis.domain,
                num_ideas=num_ideas
            )

            if not ideas:
                yield event.plain_result("❌ 想法生成失败")
                return

            # 格式化输出
            ideas_output = f"**💡 研究想法 ({len(ideas)}个)**\n\n"

            for i, idea in enumerate(ideas, 1):
                feasibility_bar = "★" * int(idea.feasibility * 5) + "☆" * (5 - int(idea.feasibility * 5))

                ideas_output += f"""---
**[{i}] {idea.title}**

**📝 描述**: {idea.description[:300]}...

**✨ 创新点**: {idea.novelty[:150]}

**🔧 方法论**: {idea.methodology[:150]}

**⚠️ 挑战**: {', '.join(idea.potential_challenges[:2])}

**📈 可行性**: {feasibility_bar} ({idea.feasibility:.0%})
"""

            ideas_output += "\n---\n💡 回复 /idea generate <想法序号> 可获取更详细的提案大纲"

            yield event.plain_result(ideas_output)

        except Exception as e:
            logger.error(f"创意探索失败: {e}")
            yield event.plain_result(f"❌ 创意探索失败: {e}")

    @idea_commands.command("analyze")
    async def cmd_idea_analyze(self, event: AstrMessageEvent,
                               topic: str = '',
                               depth: str = "standard"):
        """
        分析研究主题

        Args:
            topic: 研究主题描述
            depth: 分析深度 (quick/standard/deep)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not topic:
            yield event.plain_result("📚 Usage: /idea analyze <研究主题>")
            return

        yield event.plain_result(f"🔍 分析主题: {topic}")

        try:
            from .idea_engine import IdeaEngine
            rag_engine = self._get_engine()
            idea_engine = IdeaEngine(context=self.context, rag_engine=rag_engine)

            analysis = await idea_engine.analyze_topic(topic, depth)

            if not analysis:
                yield event.plain_result("❌ 主题分析失败")
                return

            output = f"""**📊 主题分析结果**

**研究领域**: {analysis.domain}

**核心关键词**:
{chr(10).join(f"- {k}" for k in analysis.keywords[:8])}

**搜索查询**:
{chr(10).join(f"- {q}" for q in analysis.search_queries[:5])}

**本地检索词**:
{chr(10).join(f"- {q}" for q in analysis.local_rag_queries[:3])}

**探索角度**:
{', '.join(analysis.exploration_angles)}

**摘要**: {analysis.summary}
"""
            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"主题分析失败: {e}")
            yield event.plain_result(f"❌ 分析失败: {e}")

    @idea_commands.command("search")
    async def cmd_idea_search(self, event: AstrMessageEvent,
                              queries: str = '',
                              local_k: int = 5,
                              web_k: int = 10):
        """
        多源知识检索

        Args:
            queries: 逗号分隔的搜索查询
            local_k: 本地RAG召回数
            web_k: 网络搜索召回数
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not queries:
            yield event.plain_result("📚 Usage: /idea search <查询1>, <查询2>, ...\nExample: /idea search LLM medical diagnosis, GPT-4 clinical")
            return

        query_list = [q.strip() for q in queries.split(",") if q.strip()]

        yield event.plain_result(f"🔍 执行多源检索: {query_list}")

        try:
            from .idea_engine import IdeaEngine
            rag_engine = self._get_engine()
            idea_engine = IdeaEngine(context=self.context, rag_engine=rag_engine)

            knowledge = await idea_engine.search_knowledge(query_list, local_rag_top_k=local_k, web_top_k=web_k)

            output = f"""**✅ 检索完成**

**统计**:
- 网络资源: {knowledge['stats']['web_count']} 条
- 本地论文: {knowledge['stats']['local_count']} 条

---
**📚 本地论文相关片段**:
"""

            for i, r in enumerate(knowledge['local_results'][:10], 1):
                output += f"\n{i}. **{r['paper']}** (p.{r['page']})"
                output += f"\n   {r['text'][:150]}..."

            if knowledge['web_results']:
                output += "\n\n---\n**🌐 网络资源**:\n"
                for i, r in enumerate(knowledge['web_results'][:10], 1):
                    output += f"\n{i}. **{r['title']}**"
                    output += f"\n   {r['snippet'][:150]}..."

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"检索失败: {e}")
            yield event.plain_result(f"❌ 检索失败: {e}")

    @idea_commands.command("generate")
    async def cmd_idea_generate(self, event: AstrMessageEvent,
                                context: str = '',
                                domain: str = "",
                                num: int = 3,
                                focus: str = "all"):
        """
        基于知识上下文生成研究想法

        Args:
            context: 知识上下文（可直接粘贴检索结果）
            domain: 研究领域
            num: 生成数量
            focus: 侧重点 (novelty/feasibility/impact/all)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not context:
            yield event.plain_result("📚 Usage: /idea generate <知识上下文>\n建议先使用 /idea search 检索相关知识，然后将结果作为上下文传入")
            return

        yield event.plain_result(f"💡 正在生成 {num} 个研究想法...")

        try:
            from .idea_engine import IdeaEngine
            rag_engine = self._get_engine()
            idea_engine = IdeaEngine(context=self.context, rag_engine=rag_engine)

            ideas = await idea_engine.generate_ideas(
                knowledge_context=context,
                research_domain=domain,
                num_ideas=num,
                idea_focus=focus
            )

            if not ideas:
                yield event.plain_result("❌ 想法生成失败")
                return

            output = f"**💡 研究想法 ({len(ideas)}个)**\n\n"

            for i, idea in enumerate(ideas, 1):
                feasibility_bar = "★" * int(idea.feasibility * 5) + "☆" * (5 - int(idea.feasibility * 5))

                output += f"""---
**[{i}] {idea.title}**

**📝 描述**: {idea.description[:300]}

**✨ 创新点**: {idea.novelty[:200]}

**🔧 方法论**: {idea.methodology[:200]}

**⚠️ 挑战**: {', '.join(idea.potential_challenges[:3])}

**📈 可行性**: {feasibility_bar} ({idea.feasibility:.0%})
"""

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"想法生成失败: {e}")
            yield event.plain_result(f"❌ 生成失败: {e}")

    def _format_retrieve_response(self, sources: list) -> str:
        """Format retrieval results"""
        output = "📚 **Document Search Results**\n\n"

        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            filename = metadata.get("file_name", "unknown")
            score = source.get("score", 0.0)
            text = source.get("text", "")[:200]

            output += f"[{i}] **{filename}** (similarity: {score:.3f})\n"
            output += f"{text}...\n\n"

        return output.strip()

    def _format_rag_response(self, answer: str, sources: list) -> str:
        """Format RAG response"""
        output = f"💡 **Answer**\n\n{answer}\n\n"
        output += "📚 **References**\n\n"

        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            filename = metadata.get("file_name", "unknown")
            chunk_index = metadata.get("chunk_index", 0)
            text = source.get("text", "")[:150]

            output += f"[{i}] **{filename}** (chunk #{chunk_index})\n"
            output += f"> {text}...\n\n"

        return output.strip()

    # ==================== Lifecycle Management ====================

    async def terminate(self):
        """Called when plugin is unloaded"""
        logger.info("📚 Document RAG Plugin is unloading...")

        # 注销 LLM 工具
        if self.config.get("enable_llm_tools", True):
            try:
                self.context.unregister_llm_tool("search_papers")
                self.context.unregister_llm_tool("retrieve_papers")
                logger.info("✅ Paper RAG LLM工具已注销")
            except Exception as e:
                logger.warning(f"注销LLM工具时出现警告: {e}")

        # Clear resources
        self._response_cache.clear()

        # 清理 Llama.cpp VLM Provider
        try:
            from .llama_cpp_vlm_provider import reset_llama_cpp_vlm_provider
            reset_llama_cpp_vlm_provider()
            logger.info("[Llama.cpp-VLM] Provider 已清理")

            # 强制垃圾回收，尝试避免退出时的 Metal assert 错误
            import gc
            gc.collect()
            logger.info("[Llama.cpp-VLM] 垃圾回收完成")
        except Exception as e:
            logger.warning(f"[Llama.cpp-VLM] 清理时出现警告: {e}")

        # Note: No need to explicitly close Milvus connection
        # MilvusClient manages connection automatically

        await super().terminate()
