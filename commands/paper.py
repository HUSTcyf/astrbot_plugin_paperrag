"""Paper domain commands for PaperRAG."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.message_event_result import MessageChain

from .retrieval_helpers import RetrievalHelpersMixin

try:
    from ..plugin_common import SUPPORTED_DOC_EXTENSIONS
except ImportError:
    from plugin_common import SUPPORTED_DOC_EXTENSIONS

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


class PaperCommandsMixin(RetrievalHelpersMixin):
    async def _extract_missing_abstract_text(self, file_path: str) -> Optional[str]:
        """Extract an abstract for zero-abstract repair without mutating indexes."""
        try:
            from ..rag.abstract_index import AbstractExtractor
        except ImportError:
            from rag.abstract_index import AbstractExtractor

        extractor = AbstractExtractor()
        return await extractor.extract_abstract_from_pdf(file_path)

    def _build_file_path_map(self, papers_path: Path) -> dict[str, Path]:
        """Build a file_name -> full path mapping from papers directory."""
        file_path_map: dict[str, Path] = {}

        for ext in SUPPORTED_DOC_EXTENSIONS:
            for f in papers_path.glob(f"*{ext}"):
                file_path_map[f.name] = f
            for f in papers_path.glob(f"*{ext.upper()}"):
                file_path_map[f.name] = f

        for ext in SUPPORTED_DOC_EXTENSIONS:
            for f in papers_path.rglob(f"*{ext}"):
                file_path_map[f.name] = f
            for f in papers_path.rglob(f"*{ext.upper()}"):
                file_path_map[f.name] = f

        return file_path_map

    def _match_papers_to_files(
        self,
        papers: list[dict],
        papers_dir: str,
    ) -> tuple[list[dict], list[str]]:
        """Locate filesystem paths for a batch of paper records."""
        papers_path = Path(papers_dir)
        file_path_map = self._build_file_path_map(papers_path)

        matched: list[dict] = []
        not_found: list[str] = []

        for paper in papers:
            file_name = paper.get("file_name", "")
            if file_name in file_path_map:
                matched.append({
                    **paper,
                    "file_path": str(file_path_map[file_name]),
                })
            else:
                not_found.append(file_name)

        return matched, not_found

    def _get_papers_with_zero_abstracts(self) -> dict:
        """Compare paper stats and abstract stats, returning PDF papers without abstracts."""
        paper_stats_path = _PLUGIN_DIR / "data" / "paper_doc_stats.json"
        abstract_stats_path = _PLUGIN_DIR / "data" / "milvus_abstracts_doc_stats.json"

        if not paper_stats_path.exists():
            return {"error": f"未找到论文统计文件: {paper_stats_path}"}

        try:
            with open(paper_stats_path, "r", encoding="utf-8") as f:
                paper_stats = json.load(f)
        except Exception as e:
            return {"error": f"读取论文统计失败: {e}"}

        if not isinstance(paper_stats, dict):
            return {"error": "paper_doc_stats.json 格式无效"}

        abstracts: dict = {}
        if abstract_stats_path.exists():
            try:
                with open(abstract_stats_path, "r", encoding="utf-8") as f:
                    abstract_stats = json.load(f)
                if isinstance(abstract_stats, dict):
                    loaded_abstracts = abstract_stats.get("abstracts", {})
                    if isinstance(loaded_abstracts, dict):
                        abstracts = loaded_abstracts
            except Exception as e:
                logger.warning(f"读取摘要统计失败，跳过: {e}")

        papers: list[dict] = []
        total_papers = 0

        for file_name, stats in paper_stats.items():
            if not isinstance(file_name, str) or not file_name.lower().endswith(".pdf"):
                continue

            total_papers += 1
            paper_id = os.path.splitext(file_name)[0]
            abstract_entry = abstracts.get(paper_id)
            abstract_text = ""
            extracted_chars = None

            if isinstance(abstract_entry, dict):
                abstract_text = str(abstract_entry.get("abstract_text") or "").strip()
                metadata = abstract_entry.get("metadata", {})
                if isinstance(metadata, dict):
                    extracted_chars = metadata.get("extracted_abstract_chars")

            if abstract_entry and abstract_text and extracted_chars != 0:
                continue

            reason = "missing_entry"
            if abstract_entry and not abstract_text:
                reason = "empty_abstract"
            elif abstract_entry and extracted_chars == 0:
                reason = "zero_chars"

            paper_record = stats if isinstance(stats, dict) else {}
            papers.append({
                "paper_id": paper_id,
                "file_name": file_name,
                "chunk_count": paper_record.get("chunk_count", 0),
                "added_time": paper_record.get("added_time", ""),
                "reason": reason,
            })

        return {
            "papers": papers,
            "total_papers": total_papers,
            "total_zero_abstract": len(papers),
            "total_with_abstract": total_papers - len(papers),
        }


    async def _paper_search(self, event: AstrMessageEvent,
                         query: str = '',
                         top_k: int = 5):
        """Search document library and answer questions

        Args:
            query: Search question
            top_k: Number of results to return
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not query:
            yield event.plain_result("📚 Usage: /paper search [question]\nExample: /paper search What are the key innovations of 3D Gaussian Splatting")
            return

        mode = "rag"  # 内部路由模式（rag/retrieve/auto）

        # Agentic RAG 模式（由配置开关控制）
        if self.config.get("enable_agentic_rag", False):
            async for result in self._agentic_rag(event, query=query, top_k=top_k):
                yield result
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
            # 学术论文意图检查 - 非论文问题直接由LLM回答
            if mode in ("rag", "auto"):
                is_academic, intent_reason = await self._check_academic_intent(query)
                if not is_academic:
                    logger.info(f"[PaperRAG] 非学术问题，跳过检索: {intent_reason}")
                    yield event.plain_result(f"💬 这不是学术论文相关问题，直接由LLM回答...\n\n")
                    # 由LLM直接回答（不经过RAG）
                    llm_response = await self._llm_direct_answer(query)
                    if llm_response:
                        yield event.plain_result(llm_response)
                    else:
                        yield event.plain_result("抱歉，我无法回答这个问题。")
                    return
                yield event.plain_result(f"✅ 意图确认: {intent_reason}")

            # 意图识别与路由（当 mode="auto" 时）
            actual_mode = mode
            routing_info = ""
            if mode == "auto" and self.config.get("enable_graph_rag", False):
                try:
                    from ..graphrag.graph_rag_router import create_router, RetrievalMode
                except ImportError:
                    from graphrag.graph_rag_router import create_router, RetrievalMode

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

            # Execute search - all modes now go through unified HybridRetriever pipeline
            # Graph retrieval is handled internally as RRF 4th channel when enabled
            response = await engine.search(query, top_k=top_k, mode=actual_mode)
            sources = self._query_result_to_sources(response)
            if sources:
                sources = await self._resolve_sources_arxiv(sources)

            # ALL modes: VLM normalizes chunk texts to remove [Page N], Figure/Table noise, etc.
            sources = await self._compact_chunk_texts_with_vlm(sources)

            # Format output - all modes go through unified RAG pipeline
            if actual_mode == "retrieve":
                output = self._format_retrieve_response(sources)
            else:
                answer = await self._generate_rag_answer(query, sources)
                output = self._format_rag_response(answer, sources)

            # Cache response
            self._set_cached_response(cache_key, output)

            # Extract associated images from sources
            images = self._extract_images_from_sources(sources)

            # Send result
            yield event.plain_result(output)

            # Send images with captions
            if images:
                for img in images:
                    try:
                        chain = MessageChain()
                        chain.message(img['caption'])
                        chain.file_image(img["path"])
                        await event.send(chain)
                    except Exception as e:
                        logger.warning(f"[PaperRAG] 发送图片失败 {img['path']}: {e}")
                        yield event.plain_result(f"[图片] {img['caption']}: {img['path']}")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            yield event.plain_result(f"❌ Search failed: {e}")


    async def _agentic_rag(self, event: AstrMessageEvent, query: str = '', top_k: int = 5):
        """Agentic RAG complex query (multi-hop reasoning / comparison / citation tracing)

        Args:
            query: Search question
            top_k: Number of results to return (default: 5)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not query:
            yield event.plain_result(
                "📚 Usage: /paper arag [question]\n"
                "Example: /paper arag 比较 ViT 和 CNN 的差异\n"
                "Example: /paper arag 这篇论文引用了哪些方法\n"
                "Example: /paper arag attention mechanism 的原理"
            )
            return

        yield event.plain_result(f"🧠 Agentic RAG 查询中...\n问题: {query}")

        try:
            try:
                from ..agentic_rag import run_agentic_rag
            except ImportError:
                from agentic_rag import run_agentic_rag

            final_answer = await run_agentic_rag(query, self.context, top_k=top_k, config=self.config)

            if not final_answer or not final_answer.strip():
                yield event.plain_result("⚠️ 未能生成回答，请检查知识库是否有相关文档")
                return

            yield event.plain_result(final_answer)

        except Exception as e:
            logger.error(f"Agentic RAG failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ Agentic RAG 执行失败: {e}")

    async def _agentic_rag_tool(self, query: str, top_k: int = 5) -> str:
        """LLM Tool 版本：同步返回字符串，供 context.register_llm_tool 使用。

        Args:
            query: 查询字符串
            top_k: 召回数

        Returns:
            final_answer 字符串
        """
        class _FakeEvent:
            """伪造事件对象，仅用于触发 _agentic_rag 内部逻辑。"""
            def plain_result(self, text: str) -> str:
                return text

        fake_event = _FakeEvent()
        results: list[str] = []
        async for result in self._agentic_rag(fake_event, query=query, top_k=top_k):
            if isinstance(result, str):
                results.append(result)
        return "\n".join(results) if results else ""

    async def _react_rag(self, event: AstrMessageEvent, query: str = '', top_k: int = 5):
        """Tool-Using Agent (ReAct 模式) 复杂查询

        Args:
            query: Search question
            top_k: Number of results to return (default: 5)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not query:
            yield event.plain_result(
                "🤖 Usage: /paper react [question]\n"
                "Example: /paper react 比较 ViT 和 CNN 的差异\n"
                "Example: /paper react attention mechanism 的原理"
            )
            return

        yield event.plain_result(f"🤖 ReAct Agent 查询中...\n问题: {query}")

        try:
            try:
                from ..agentic_rag import run_react_rag
            except ImportError:
                from agentic_rag import run_react_rag

            final_answer = await run_react_rag(query, self.context, top_k=top_k, config=self.config)

            if not final_answer or not final_answer.strip():
                yield event.plain_result("⚠️ 未能生成回答，请检查知识库是否有相关文档")
                return

            yield event.plain_result(final_answer)

        except Exception as e:
            logger.error(f"ReAct Agent failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ ReAct Agent 执行失败: {e}")

    async def _react_rag_tool(self, query: str, top_k: int = 5) -> str:
        """LLM Tool 版本：ReAct Agent，供 context.register_llm_tool 使用。"""
        class _FakeEvent:
            def plain_result(self, text: str) -> str:
                return text

        fake_event = _FakeEvent()
        results: list[str] = []
        async for result in self._react_rag(fake_event, query=query, top_k=top_k):
            if isinstance(result, str):
                results.append(result)
        return "\n".join(results) if results else ""


    async def _paper_list(self, event: AstrMessageEvent):
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


    async def _paper_add(self, event: AstrMessageEvent, directory: str = ''):
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


    async def _paper_addf(self, event: AstrMessageEvent, file_path: str = ''):
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


    async def _paper_clear(self, event: AstrMessageEvent, confirm: str = ''):
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


    async def _paper_delete(self, event: AstrMessageEvent, file_name: str = ''):
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


    async def _paper_refstats(self, event: AstrMessageEvent, top_k: int = 20, dedup: int = 0):
        """Show reference title frequency statistics

        Args:
            top_k: Number of top references to show (default: 20). Use -1 to list papers with zero references.
            dedup: If 1, count each citation at most once per paper (removes in-paper duplicates).
                   If 0 (default), count every citation occurrence.
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

            stats = await index_manager.get_all_references(allow_duplicates=(dedup == 0))

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
            dedup_note = "（去重）" if dedup == 1 else ""
            output = f"📚 **参考文献统计** {dedup_note}\n\n"
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


    async def _paper_abstractstats(self, event: AstrMessageEvent, top_k: int = 20):
        """Show abstract extraction statistics.

        Args:
            top_k: Use -1 to list PDF papers with no extracted abstract.
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        try:
            result = self._get_papers_with_zero_abstracts()

            if "error" in result:
                yield event.plain_result(f"❌ 获取摘要统计失败: {result['error']}")
                return

            papers = result.get("papers", [])
            total_papers = result.get("total_papers", 0)
            total_zero_abstract = result.get("total_zero_abstract", 0)
            total_with_abstract = result.get("total_with_abstract", 0)

            if total_papers == 0:
                yield event.plain_result("⚠️ 未找到任何已索引的 PDF 论文")
                return

            if top_k == -1:
                yield event.plain_result("📊 正在查找无摘要的论文...")

                if not papers:
                    yield event.plain_result("✅ 所有 PDF 论文都已成功提取摘要")
                    return

                output = f"📚 **无摘要的论文** ({total_zero_abstract}/{total_papers})\n\n"

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

            output = "📚 **摘要提取统计**\n\n"
            output += "📊 统计概览:\n"
            output += f"   • PDF论文总数: {total_papers}\n"
            output += f"   • 已提取摘要: {total_with_abstract}\n"
            output += f"   • 未提取摘要: {total_zero_abstract}\n\n"
            output += "💡 使用 `/paper abstractstats -1` 列出无摘要论文\n"
            output += "💡 使用 `/paper reparse_zero_abstract confirm` 批量重新提取摘要"

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"Failed to get abstractstats: {e}")
            yield event.plain_result(f"❌ 获取摘要统计失败: {e}")


    async def _paper_reparse_zero_ref(self, event: AstrMessageEvent, confirm: str = ''):
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

            papers_to_reparse, not_found = self._match_papers_to_files(papers, papers_dir)

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
                    result = await engine.delete_paper(file_name, file_path or "")
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


    async def _paper_reparse_zero_abstract(self, event: AstrMessageEvent, confirm: str = ''):
        """Batch re-extract abstracts for papers without abstracts (Admin).

        Args:
            confirm: Must be 'confirm' to proceed.
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != 'confirm':
            yield event.plain_result(
                "⚠️ This will re-extract abstracts for all PDF papers without abstracts.\n"
                "This operation may take a long time.\n"
                "Usage: /paper reparse_zero_abstract confirm"
            )
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        try:
            yield event.plain_result("🔍 Step 1/3: Finding papers with zero abstracts...")
            result = self._get_papers_with_zero_abstracts()

            if "error" in result:
                yield event.plain_result(f"❌ 获取失败: {result['error']}")
                return

            papers = result.get("papers", [])
            total_zero_abstract = result.get("total_zero_abstract", 0)

            if not papers:
                yield event.plain_result("✅ All PDF papers already have abstracts")
                return

            yield event.plain_result(f"📊 Found {total_zero_abstract} papers with zero abstracts")

            yield event.plain_result("🔍 Step 2/3: Locating paper files...")
            papers_dir = self.config.get("papers_dir", "./papers")
            papers_path = Path(papers_dir)
            if not papers_path.exists():
                yield event.plain_result(f"❌ Papers directory does not exist: {papers_dir}")
                return

            papers_to_reparse, not_found = self._match_papers_to_files(papers, papers_dir)

            if not_found:
                yield event.plain_result(f"⚠️ {len(not_found)} papers not found in {papers_dir}:")
                for fn in not_found[:5]:
                    yield event.plain_result(f"   - {fn}")
                if len(not_found) > 5:
                    yield event.plain_result(f"   ... and {len(not_found) - 5} more")

            if not papers_to_reparse:
                yield event.plain_result("❌ No paper files found for zero-abstract papers")
                return

            yield event.plain_result(f"✅ Found {len(papers_to_reparse)} paper files")

            yield event.plain_result("🔍 Step 3/3: Re-extracting missing abstracts without deleting existing paper stats...")
            abstract_manager = await engine._ensure_abstract_manager_initialized()
            if abstract_manager is None:
                yield event.plain_result("❌ Abstract index manager is not ready")
                return

            import time
            start_time = time.time()
            success_count = 0
            fail_count = 0

            for i, paper in enumerate(papers_to_reparse, 1):
                paper_id = paper["paper_id"]
                file_name = paper["file_name"]
                file_path = paper["file_path"]

                try:
                    abstract_text = await self._extract_missing_abstract_text(file_path)
                    if not abstract_text or len(abstract_text) < 50:
                        fail_count += 1
                        logger.warning(f"Failed to re-extract abstract for {file_name}")
                    else:
                        delete_vectors = getattr(abstract_manager, "delete_paper_vectors_only", None)
                        if delete_vectors is not None:
                            vectors_cleaned = bool(await delete_vectors(paper_id))
                            if not vectors_cleaned:
                                logger.warning(
                                    f"旧摘要向量未确认清理或本来不存在，继续写入新摘要: {file_name}"
                                )

                        ok = await abstract_manager.index_paper(
                            pdf_path=file_path,
                            paper_id=paper_id,
                            file_name=file_name,
                            abstract_text=abstract_text,
                            metadata={
                                "abstract_source": "pdf_text_reparse",
                                "extracted_abstract_chars": len(abstract_text),
                            },
                        )
                        if ok:
                            success_count += 1
                        else:
                            fail_count += 1
                            logger.warning(f"Failed to re-extract abstract for {file_name}")
                except Exception as e:
                    fail_count += 1
                    logger.error(f"Failed to re-extract abstract for {file_name}: {e}")

                if i % 5 == 0 or i == len(papers_to_reparse):
                    yield event.plain_result(
                        f"   Progress: {i}/{len(papers_to_reparse)} "
                        f"(success: {success_count}, failed: {fail_count})"
                    )

            elapsed_time = time.time() - start_time
            output = f"""✅ **Abstract Reparse Complete**

📊 Statistics:
  • Total zero-abstract papers: {total_zero_abstract}
  • Files found: {len(papers_to_reparse)}
  • Successfully re-extracted: {success_count}
  • Failed: {fail_count}
  • Time: {elapsed_time:.1f}s

💡 Tip: Use /paper abstractstats -1 to check again"""

            if not_found:
                output += f"\n\n⚠️ {len(not_found)} papers not found in filesystem"

            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"Failed to reparse zero-abstract papers: {e}")
            yield event.plain_result(f"❌ 操作失败: {e}")


    async def _paper_rebuild(self, event: AstrMessageEvent, directory: str = '', confirm: str = ''):
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

            # 同时清除摘要索引
            try:
                try:
                    from ..rag.abstract_index import AbstractIndexManager
                except ImportError:
                    from rag.abstract_index import AbstractIndexManager
                plugin_dir = _PLUGIN_DIR
                embed_dim = self.config.get("embed_dim", 768)
                milvus_uri = str(plugin_dir / "data" / "milvus_abstracts.db")
                abstract_index = AbstractIndexManager(
                    milvus_uri=milvus_uri,
                    embed_dim=embed_dim,
                    core_api_key=self.config.get("core_api_key", ""),
                    use_arxiv_api=self.config.get("use_arxiv_api", True),
                )
                await abstract_index.initialize()
                abstract_index.clear()
                yield event.plain_result("✅ Step 1/5b: Abstract index cleared")
            except Exception as e:
                logger.warning(f"清除摘要索引失败: {e}")

        except Exception as e:
            logger.error(f"Failed to clear document library: {e}")
            yield event.plain_result(f"❌ Failed to clear: {e}")
            return

        # Delete figures and tables folders
        yield event.plain_result("🔄 Step 2/5: Clearing figures...")
        plugin_dir = _PLUGIN_DIR
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


    async def _paper_rebuildf(self, event: AstrMessageEvent, file_name: str = ''):
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
