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
from pathlib import Path
from typing import Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Image
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register

from .rag_engine import (
    PaperRAGEngine,
    RAGConfig
)


@register(
    "paper_rag",
    "YourName",
    "本地文档库RAG检索插件 (支持PDF/Word/TXT/HTML, Gemini + Milvus Lite)",
    "1.0.0",
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

        logger.info("📚 Document RAG Plugin initialized (支持PDF/Word/TXT/HTML)")

    def _get_engine(self) -> Optional[PaperRAGEngine]:
        """获取RAG引擎（单例模式，带缓存）"""
        if self._engine is None and not self._config_valid:
            try:
                # 从插件配置创建RAG配置
                rag_config = RAGConfig(
                    embedding_provider_id=self.config.get("embedding_provider_id", ""),
                    llm_provider_id=self.config.get("llm_provider_id", ""),
                    milvus_lite_path=self.config.get("milvus_lite_path", "./data/milvus_papers.db"),
                    address=self.config.get("address", ""),
                    db_name=self.config.get("db_name", "default"),
                    authentication=self.config.get("authentication", {}),
                    collection_name=self.config.get("collection_name", "paper_embeddings"),
                    embed_dim=self.config.get("embed_dim", 768),
                    top_k=self.config.get("top_k", 5),
                    similarity_cutoff=self.config.get("similarity_cutoff", 0.3),
                    papers_dir=self.config.get("papers_dir", "./papers"),
                    # 语义分块配置
                    chunk_size=self.config.get("chunk_size", 512),
                    chunk_overlap=self.config.get("chunk_overlap", 50),
                    min_chunk_size=self.config.get("min_chunk_size", 100),
                    use_semantic_chunking=self.config.get("use_semantic_chunking", True),
                    # 多模态配置
                    enable_multimodal=self.config.get("multimodal", {}).get("enabled", True)
                )

                # 验证配置
                valid, error_msg = rag_config.validate()
                if not valid:
                    logger.error(f"❌ RAG配置无效: {error_msg}")
                    self._config_valid = False
                    return None

                # 初始化引擎（传递context以获取Provider）
                self._engine = PaperRAGEngine(rag_config, self.context)
                self._config_valid = True

            except Exception as e:
                logger.error(f"❌ RAG引擎初始化失败: {e}")
                self._config_valid = False
                return None

        return self._engine

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

    # ==================== 命令系统 ====================

    @filter.command_group("paper")
    def paper_commands(self):
        """Paper RAG command group
        search       - Search documents and answer questions
        list         - List indexed documents
        add          - Add documents to knowledge base (PDF/Word/TXT supported)
        clear        - Clear knowledge base
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

            # Format output
            if response["type"] == "retrieve":
                # Retrieve mode only
                output = self._format_retrieve_response(response["sources"])
            elif response["type"] == "rag":
                # RAG mode
                output = self._format_rag_response(
                    response["answer"],
                    response["sources"]
                )
            elif response["type"] == "error":
                output = f"❌ {response['message']}"
            else:
                output = "❌ Unknown response type"

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

            if not doc_files:
                yield event.plain_result("📭 No supported document files found\nSupported formats: PDF, Word, TXT, Markdown, HTML")
                return

            yield event.plain_result(f"📄 Found {len(doc_files)} document files\n⏳ Starting import...")

            # Get engine
            engine = self._get_engine()
            if not engine:
                yield event.plain_result("❌ RAG engine is not ready")
                return

            # Import documents and stream progress
            # Convert to list to ensure correct type
            file_paths = [str(f) for f in doc_files]
            async for progress in engine.ingest_papers(file_paths):
                if progress["status"] == "parsing":
                    yield event.plain_result(
                        f"📖 [{progress['current']}/{progress['total']}] "
                        f"Parsing: {progress['filename']}"
                    )
                elif progress["status"] == "embedding":
                    yield event.plain_result(
                        f"🔢 [{progress['current']}/{progress['total']}] "
                        f"Vectorizing: {progress['filename']}"
                    )
                elif progress["status"] == "storing":
                    yield event.plain_result(
                        f"💾 [{progress['current']}/{progress['total']}] "
                        f"Storing: {progress['filename']}"
                    )
                elif progress["status"] == "done":
                    yield event.plain_result(
                        f"✅ [{progress['current']}/{progress['total']}] "
                        f"{progress['filename']} - {progress['chunks']} chunks"
                    )
                elif progress["status"] == "skipped":
                    yield event.plain_result(
                        f"⏭️  [{progress['current']}/{progress['total']}] "
                        f"Skipped: {progress['filename']}"
                    )
                elif progress["status"] == "error":
                    yield event.plain_result(
                        f"❌ [{progress['current']}/{progress['total']}] "
                        f"Failed: {progress['filename']} - {progress['error']}"
                    )
                elif progress["status"] == "complete":
                    # Final summary
                    files_count = progress['files_count']
                    chunks_count = progress['chunks_count']
                    skipped_count = progress.get('skipped_count', 0)
                    error_count = progress.get('error_count', 0)
                    elapsed_time = progress['elapsed_time']

                    output = f"""✅ **Import Complete**

📊 Statistics:
  • Total files: {files_count}
  • Successfully processed: {files_count - skipped_count - error_count}
  • Skipped (no text): {skipped_count}
  • Errors: {error_count}
  • Chunks created: {chunks_count}
  • Time: {elapsed_time:.1f}s

💡 Tip: Use /paper search [question] to search documents"""

                    if skipped_count > 0:
                        output += f"\n\n⚠️ {skipped_count} files had no extractable text (may be scanned PDFs)"

                    if error_count > 0:
                        output += f"\n\n❌ {error_count} files failed to process"

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
            await engine.store.clear()
            self._response_cache.clear()

            yield event.plain_result("✅ Document library cleared")

        except Exception as e:
            logger.error(f"Failed to clear document library: {e}")
            yield event.plain_result(f"❌ Failed to clear document library: {e}")

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
