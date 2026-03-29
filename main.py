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
import json
import os
import re
import subprocess
import requests
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

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


@register(
    "paper_rag",
    "YourName",
    "本地文档库RAG检索插件 (支持PDF/Word/TXT/HTML, Gemini + Milvus Lite)",
    "1.5.0",
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

        # RAG引擎（懒加载）
        self._engine = None
        self._config_valid = False

        # 根据配置决定是否启动 Grobid 服务
        enable_grobid = self.config.get("enable_grobid", False)
        if enable_grobid:
            # 异步启动 Grobid 服务（用于参考文献解析）
            # 不阻塞插件初始化，服务在后台启动
            import threading
            grobid_thread = threading.Thread(target=self._ensure_grobid_running, daemon=True)
            grobid_thread.start()
            logger.info("📚 Document RAG Plugin initialized (支持PDF/Word/TXT/HTML, Grobid已启用)")
        else:
            logger.info("📚 Document RAG Plugin initialized (支持PDF/Word/TXT/HTML, Grobid未启用)")

        # 注册 LLM 可调用的论文搜索工具
        self._register_llm_tools()

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
                    reranking_batch_size=self.config.get("reranking_batch_size", 32)
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
        refstats     - Show reference title frequency statistics
        arxiv_add    - Search arxiv and download papers, then add to database (Admin)
        arxiv_refs   - Download highly-cited reference papers from arxiv (Admin)
        arxiv_sync   - Sync MCP downloaded papers to paperrag database (Admin)
        arxiv_cleanup- Clean up old versions of arxiv papers (Admin)
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
            mode: Mode (rag=retrieval augmented generation, retrieve=retrieval only)
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
            # Execute search
            response = await engine.search(query, mode=mode)

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
            supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm']
            doc_files = []
            for ext in supported_extensions:
                doc_files.extend(Path(papers_dir).glob(f"*{ext}"))
            # Also support uppercase extensions
            for ext in supported_extensions:
                doc_files.extend(Path(papers_dir).glob(f"*{ext.upper()}"))

            # Filter out macOS metadata files (starting with "._")
            doc_files = [f for f in doc_files if not f.name.startswith("._")]

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
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm']
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in supported_extensions:
            yield event.plain_result(f"❌ Unsupported format: {ext}\nSupported: {', '.join(supported_extensions)}")
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
            top_k: Number of top references to show (default: 20)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        try:
            yield event.plain_result("📊 正在统计参考文献...")

            # 获取索引管理器
            index_manager = engine._ensure_index_manager_initialized()
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
    @paper_commands.command("arxiv_add")
    async def cmd_arxiv_add(self, event: AstrMessageEvent, query: str = '', max_results: int = 5):
        """Search arxiv and download papers, then add to database (Admin)

        Args:
            query: Search query for arxiv (e.g., paper title, authors, keywords)
            max_results: Maximum number of papers to download (default: 5)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not query:
            yield event.plain_result("❌ Please provide search query\nUsage: /paper arxiv_add <query> [max_results]\nExample: /paper arxiv_add attention is all you need 3")
            return

        # Get MCP tool manager
        try:
            mcp_manager = self.context.get_llm_tool_manager()
            mcp_clients = mcp_manager.mcp_client_dict
        except Exception as e:
            logger.error(f"Failed to get MCP manager: {e}")
            yield event.plain_result(f"❌ 无法获取MCP工具管理器: {e}")
            return

        # Check if arxiv MCP is available
        if "arxiv" not in mcp_clients:
            yield event.plain_result("❌ arxiv MCP服务器未连接\n请检查 mcp_server.json 配置")
            return

        arxiv_client = mcp_clients["arxiv"]
        papers_dir = self.config.get("papers_dir", "./papers")

        # Ensure papers directory exists
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            papers_path.mkdir(parents=True, exist_ok=True)

        yield event.plain_result(f"🔍 在arXiv搜索: \"{query}\"\n最大下载数量: {max_results}")

        try:
            # Step 1: Search arxiv
            yield event.plain_result("📡 正在搜索arXiv...")

            search_result = await arxiv_client.call_tool_with_reconnect(
                tool_name="search_arxiv",
                arguments={"query": query, "max_results": max_results},
                read_timeout_seconds=timedelta(seconds=60)
            )

            if not search_result or not hasattr(search_result, 'content'):
                yield event.plain_result("❌ arXiv搜索失败: 无返回结果")
                return

            # Parse search results
            papers_info = []
            for content_block in search_result.content:
                text = getattr(content_block, 'text', None)
                if text:
                    try:
                        paper_data = json.loads(text)
                        if isinstance(paper_data, dict):
                            papers_info.append(paper_data)
                        elif isinstance(paper_data, list):
                            papers_info.extend(paper_data)
                    except json.JSONDecodeError:
                        logger.debug(f"arXiv返回非JSON内容: {text[:200]}...")
                        continue

            if not papers_info:
                yield event.plain_result("❌ 未找到相关论文")
                return

            yield event.plain_result(f"✅ 找到 {len(papers_info)} 篇论文")

            # Step 2: Download each paper
            engine = self._get_engine()
            if not engine:
                yield event.plain_result("❌ RAG引擎未就绪")
                return

            successful = 0
            failed = 0
            skipped = 0

            for i, paper in enumerate(papers_info, 1):
                # 提取论文信息
                paper_id = paper.get('id', paper.get('paper_id', ''))
                title = paper.get('title', 'unknown')
                arxiv_url = paper.get('url', paper.get('arxiv_url', ''))

                if not paper_id and not arxiv_url:
                    logger.warning(f"⚠️ 论文信息缺少ID或URL: {paper}")
                    failed += 1
                    continue

                yield event.plain_result(f"\n📄 [{i}/{len(papers_info)}] {title[:60]}...")

                # 确定文件名
                if paper_id:
                    pdf_filename = f"{paper_id}.pdf"
                else:
                    # 从URL提取arXiv ID
                    match = re.search(r'(\d+\.\d+)', arxiv_url)
                    if match:
                        paper_id = match.group(1)
                        pdf_filename = f"{paper_id}.pdf"
                    else:
                        # 使用标题生成文件名
                        safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                        pdf_filename = f"{safe_title}.pdf"

                pdf_path = papers_path / pdf_filename

                # 检查是否已存在
                if pdf_path.exists():
                    yield event.plain_result(f"   ⏭️ PDF已存在，跳过下载")
                    skipped += 1
                    # 仍然添加到数据库
                    result = await engine.add_paper(str(pdf_path))
                    if result.get("status") == "success":
                        successful += 1
                        yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                    else:
                        yield event.plain_result(f"   ⚠️ 添加失败: {result.get('message', 'unknown')}")
                    continue

                # 下载PDF
                try:
                    # 使用arXiv API下载PDF
                    if not paper_id:
                        # 从URL提取ID
                        match = re.search(r'(\d+\.\d+)', arxiv_url)
                        if match:
                            paper_id = match.group(1)

                    if paper_id:
                        # arXiv PDF下载URL格式
                        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                        yield event.plain_result(f"   📥 下载PDF: {pdf_url}")

                        # 下载文件
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
                        else:
                            yield event.plain_result(f"   ❌ 下载失败: HTTP {pdf_response.status_code}")
                            failed += 1
                            continue
                    else:
                        yield event.plain_result(f"   ❌ 无法确定arXiv ID")
                        failed += 1
                        continue

                except Exception as e:
                    logger.error(f"下载论文失败: {e}")
                    yield event.plain_result(f"   ❌ 下载失败: {e}")
                    failed += 1
                    continue

                # Step 3: Add to database
                try:
                    result = await engine.add_paper(str(pdf_path))
                    if result.get("status") == "success":
                        successful += 1
                        yield event.plain_result(f"   ✅ 已添加到数据库 (chunks: {result.get('chunks_added', 0)})")
                    else:
                        yield event.plain_result(f"   ⚠️ 添加失败: {result.get('message', 'unknown')}")
                except Exception as e:
                    logger.error(f"添加论文失败: {e}")
                    yield event.plain_result(f"   ❌ 添加失败: {e}")
                    failed += 1

            # Summary
            output = f"""
📊 **arXiv论文下载完成**

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
        """Download highly-cited reference papers from arxiv and add to database (Admin)

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

        # Get MCP tool manager
        try:
            mcp_manager = self.context.get_llm_tool_manager()
            mcp_clients = mcp_manager.mcp_client_dict
        except Exception as e:
            logger.error(f"Failed to get MCP manager: {e}")
            yield event.plain_result(f"❌ 无法获取MCP工具管理器: {e}")
            return

        if "arxiv" not in mcp_clients:
            yield event.plain_result("❌ arxiv MCP服务器未连接\n请检查 mcp_server.json 配置")
            return

        arxiv_client = mcp_clients["arxiv"]
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

            # Step 2: Search and download each reference paper from arxiv
            successful = 0
            failed = 0
            skipped = 0
            total_downloaded = 0

            import re

            for i, ref in enumerate(top_refs, 1):
                title = ref.get("title", "")
                authors = ref.get("authors", "")
                year = ref.get("year", "")

                if not title:
                    continue

                yield event.plain_result(f"\n[{i}/{len(top_refs)}] 📝 {title[:60]}...")

                # Build search query from title and authors
                search_parts = [title]
                if authors:
                    # Extract first author surname
                    first_author = authors.split(',')[0].strip()
                    search_parts.append(first_author)
                if year:
                    search_parts.append(str(year))

                search_query = " ".join(search_parts)
                yield event.plain_result(f"   🔍 搜索: {search_query[:80]}...")

                try:
                    # Search arxiv
                    search_result = await arxiv_client.call_tool_with_reconnect(
                        tool_name="search_arxiv",
                        arguments={"query": search_query, "max_results": max_per_paper},
                        read_timeout_seconds=timedelta(seconds=60)
                    )

                    if not search_result or not hasattr(search_result, 'content'):
                        yield event.plain_result(f"   ⚠️ 搜索无结果")
                        failed += 1
                        continue

                    # Parse results
                    papers_info = []
                    for content_block in search_result.content:
                        text = getattr(content_block, 'text', None)
                        if text:
                            try:
                                paper_data = json.loads(text)
                                if isinstance(paper_data, dict):
                                    papers_info.append(paper_data)
                                elif isinstance(paper_data, list):
                                    papers_info.extend(paper_data)
                            except json.JSONDecodeError:
                                continue

                    if not papers_info:
                        yield event.plain_result(f"   ⚠️ 未找到相关论文")
                        failed += 1
                        continue

                    # Download first (most relevant) result
                    paper = papers_info[0]
                    paper_id = paper.get('id', paper.get('paper_id', ''))

                    if not paper_id:
                        # Try to extract from URL
                        arxiv_url = paper.get('url', paper.get('arxiv_url', ''))
                        match = re.search(r'(\d+\.\d+)', arxiv_url)
                        if match:
                            paper_id = match.group(1)

                    if not paper_id:
                        yield event.plain_result(f"   ⚠️ 无法确定arXiv ID")
                        failed += 1
                        continue

                    pdf_filename = f"{paper_id}.pdf"
                    pdf_path = papers_path / pdf_filename

                    if pdf_path.exists():
                        yield event.plain_result(f"   ⏭️ PDF已存在，跳过")
                        skipped += 1
                        # Still add to database
                        result = await engine.add_paper(str(pdf_path))
                        if result.get("status") == "success":
                            successful += 1
                            yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                        continue

                    # Download PDF
                    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
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
📊 **高频引用论文下载完成**

✅ 成功: {successful}
⏭️ 跳过: {skipped}
❌ 失败: {failed}
📥 新增下载: {total_downloaded}

📁 保存路径: {papers_dir}
💡 使用 /paper list 查看所有论文
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"arXiv批量下载失败: {e}")
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

        # Get MCP storage path from configuration
        mcp_storage_path = "/Volumes/ext/arxiv"

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
            pdf_files = [f for f in pdf_files if not f.name.startswith("._")]

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

        mcp_storage_path = "/Volumes/ext/arxiv"

        if not os.path.exists(mcp_storage_path):
            yield event.plain_result(f"❌ MCP存储路径不存在: {mcp_storage_path}")
            return

        yield event.plain_result(f"🧹 扫描MCP存储路径: {mcp_storage_path}")

        try:
            import re
            from collections import defaultdict

            mcp_path = Path(mcp_storage_path)

            # Find all PDF files (excluding macOS metadata files)
            pdf_files = [f for f in mcp_path.glob("*.pdf") if not f.name.startswith("._")]

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

        yield event.plain_result("🔄 Step 1/4: Clearing knowledge base...")

        try:
            # Clear database
            result = await engine.clear()
            if result.get("status") != "success":
                yield event.plain_result(f"❌ Failed to clear: {result.get('message', 'Unknown error')}")
                return
            yield event.plain_result("✅ Step 1/4: Knowledge base cleared")

        except Exception as e:
            logger.error(f"Failed to clear document library: {e}")
            yield event.plain_result(f"❌ Failed to clear: {e}")
            return

        # Delete figures folder
        yield event.plain_result("🔄 Step 2/4: Clearing figures...")
        figures_dir = self.config.get("figures_dir", "")
        if not figures_dir:
            # Default figures directory
            figures_dir = Path(__file__).parent / "data" / "figures"
        else:
            figures_dir = Path(figures_dir)

        if figures_dir.exists() and figures_dir.is_dir():
            try:
                import shutil
                shutil.rmtree(figures_dir)
                logger.info(f"✅ Deleted figures folder: {figures_dir}")
                yield event.plain_result(f"✅ Step 2/4: Figures folder cleared")
            except Exception as e:
                logger.warning(f"Failed to delete figures folder: {e}")
                yield event.plain_result(f"⚠️ Failed to delete figures: {e}")
        else:
            yield event.plain_result("✅ Step 2/4: No figures folder found, skipping")

        yield event.plain_result("🔄 Step 3/4: Scanning documents...")

        # Scan documents
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm']
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(Path(papers_dir).glob(f"*{ext}"))
        for ext in supported_extensions:
            doc_files.extend(Path(papers_dir).glob(f"*{ext.upper()}"))

        doc_files = [f for f in doc_files if not f.name.startswith("._")]

        if not doc_files:
            yield event.plain_result("📭 No supported documents found")
            return

        yield event.plain_result(f"📄 Step 3/4: Found {len(doc_files)} documents")

        # Re-add documents
        import time
        start_time = time.time()
        total_files = len(doc_files)
        successful = 0
        failed = 0
        total_chunks = 0

        yield event.plain_result("🔄 Step 4/4: Rebuilding embeddings... (this may take a while)")

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
