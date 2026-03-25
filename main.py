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
import subprocess
import requests
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
    "2.0.0",
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

    def _get_engine(self) -> "Optional[HybridRAGEngine]":
        """获取RAG引擎（单例模式，带缓存）"""
        if self._engine is None and not self._config_valid:
            try:
                # 从插件配置创建RAG配置
                rag_config = RAGConfig(
                    embedding_mode=self.config.get("embedding_mode", "ollama"),
                    embedding_provider_id=self.config.get("embedding_provider_id", ""),
                    glm_api_key=self.config.get("glm_api_key", ""),
                    glm_model=self.config.get("glm_model", "glm-4.7-flash"),
                    glm_multimodal_model=self.config.get("glm_multimodal_model", "glm-4.6v-flash"),
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
        clear        - Clear knowledge base
        rebuild      - Clear and re-add all documents
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

            # Clear related cache
            self._response_cache.clear()

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            yield event.plain_result(f"❌ Failed to add documents: {e}")

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

        yield event.plain_result("🔄 Step 1/3: Clearing knowledge base...")

        try:
            # Clear database
            result = await engine.clear()
            if result.get("status") != "success":
                yield event.plain_result(f"❌ Failed to clear: {result.get('message', 'Unknown error')}")
                return
            yield event.plain_result("✅ Step 1/3: Knowledge base cleared")

        except Exception as e:
            logger.error(f"Failed to clear document library: {e}")
            yield event.plain_result(f"❌ Failed to clear: {e}")
            return

        yield event.plain_result("🔄 Step 2/3: Scanning documents...")

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

        yield event.plain_result(f"📄 Step 2/3: Found {len(doc_files)} documents")

        # Re-add documents
        import time
        start_time = time.time()
        total_files = len(doc_files)
        successful = 0
        failed = 0
        total_chunks = 0

        yield event.plain_result("🔄 Step 3/3: Rebuilding embeddings... (this may take a while)")

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

        # Clear resources
        self._response_cache.clear()

        # Note: No need to explicitly close Milvus connection
        # MilvusClient manages connection automatically

        await super().terminate()
