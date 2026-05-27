"""Paper domain commands for PaperRAG."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.message_event_result import MessageChain

from .retrieval_helpers import RetrievalHelpersMixin
import time

try:
    from ..plugin_common import SUPPORTED_DOC_EXTENSIONS
except ImportError:
    from plugin_common import SUPPORTED_DOC_EXTENSIONS


def _llm_display_name(llm_config: dict) -> str:
    """Extract a human-readable LLM name from config (provider or freeapi)."""
    provider = llm_config.get("provider")
    if provider is not None:
        model = getattr(provider, 'model', None) or getattr(provider, 'model_name', None)
        if model:
            return model
        provider_id = getattr(provider, 'provider_id', None) or getattr(provider, 'provider_name', None)
        if provider_id:
            return provider_id
        return "provider"
    return llm_config.get("model", "unknown")

_PLUGIN_DIR = Path(__file__).resolve().parent.parent

# Claude Code 权限相关 stderr 关键字（需要用户手动授权时出现）
_PERMISSION_KEYWORDS = re.compile(
    r"permission|approval|authorization|not allowed|denied|requires? (human|user|manual)",
    re.IGNORECASE,
)

# code_execute 危险命令模式（模块级常量，避免每次调用重新创建）
_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-rf\s+/($|\*)", "rm -rf /"),
    (r"curl\s+.*\|\s*(ba)?sh", "curl ... | sh"),
    (r"wget\s+.*\|\s*(ba)?sh", "wget ... | sh"),
    (r"git\s+push\s+(-f|--force)", "git push --force"),
    (r"\bsudo\b", "sudo"),
    (r"chmod\s+777", "chmod 777"),
    (r">\s*/dev/(sd[a-z]|nvme\d+n\d+|xvd[a-z]|hd[a-z]|mmcblk\d+)", "> /dev/disk (磁盘覆写)"),
]


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

                # 自动知识提取到 wiki（非阻塞，失败不影响主流程）
                if self.config.get("auto_extract_knowledge", False):
                    asyncio.create_task(
                        self._extract_knowledge_to_wiki(query, answer, sources)
                    )

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

    async def _extract_knowledge_to_wiki(
        self, query: str, answer: str, sources: list[dict]
    ) -> None:
        """Fire-and-forget: extract verifiable knowledge to wiki after Q&A."""
        try:
            from agentic_rag.knowledge_extractor import run_knowledge_extraction
            result = await run_knowledge_extraction(
                query=query,
                answer=answer,
                sources=sources,
                context=self.context,
                config=self.config,
            )
            logger.info(
                f"[PaperRAG] Knowledge extraction: status={result.get('status')}, "
                f"pages_written={result.get('pages_written', 0)}"
            )
            if result.get("status") == "rejected":
                logger.info(
                    f"[PaperRAG] Knowledge extraction rejected: {result.get('reason')}"
                )
        except Exception as e:
            logger.warning(f"[PaperRAG] Knowledge extraction failed (non-fatal): {e}")

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

        yield event.plain_result("🔄 Running Agentic RAG query...")
        logger.info(f"[PaperRAG] Agentic RAG query: {query}")

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

    async def _agentic_rag_tool(self, event: AstrMessageEvent, query: str, top_k: int = 5) -> str:
        """LLM Tool wrapper: consumes the async generator and returns the final answer text.

        AstrBot's call_local_llm_tool passes the real event as first positional arg.
        We pass it directly to _agentic_rag. Because this is a coroutine (not an
        async generator), the yields from _agentic_rag are consumed internally and
        never reach the framework for user delivery.  Only the last meaningful yield
        (final answer or error) is returned to the LLM.

        Args:
            event: AstrMessageEvent (injected by framework)
            query: query string
            top_k: number of results to retrieve

        Returns:
            Final answer text (or error message) as a plain string for the LLM.
        """
        results: list[str] = []
        async for result in self._agentic_rag(event, query=query, top_k=top_k):
            text = result.get_plain_text() if hasattr(result, "get_plain_text") else str(result)
            if text.strip():
                results.append(text.strip())
        return results[-1] if results else ""

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

        yield event.plain_result("🤖 Running ReAct Agent query...")
        logger.info(f"[PaperRAG] ReAct Agent query: {query}")

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

    async def _react_rag_tool(self, event: AstrMessageEvent, query: str, top_k: int = 5) -> str:
        """LLM Tool wrapper: consumes the React Agent async generator, returns final text.

        Args:
            event: AstrMessageEvent (injected by framework)
            query: query string
            top_k: number of results to retrieve

        Returns:
            Final answer text (or error message) as a plain string for the LLM.
        """
        results: list[str] = []
        async for result in self._react_rag(event, query=query, top_k=top_k):
            text = result.get_plain_text() if hasattr(result, "get_plain_text") else str(result)
            if text.strip():
                results.append(text.strip())
        return results[-1] if results else ""

    async def _paper_search_tool(self, event: AstrMessageEvent, query: str, top_k: int = 5) -> str:
        """LLM Tool wrapper: 基础 RAG 检索 + LLM 生成回答，返回纯文本。

        比 paper_arag/paper_react 更轻量，适合简单的论文内容检索。
        内部调用 engine.search → 文本清洗 → LLM 生成回答。

        Args:
            event: AstrMessageEvent (injected by framework)
            query: query string
            top_k: number of results to retrieve

        Returns:
            Generated answer text, or error message.
        """
        engine = self._get_engine()
        if not engine:
            return "RAG 引擎未就绪"

        try:
            response = await engine.search(query, top_k=top_k, mode="rag")
        except Exception as e:
            logger.error(f"[paper_search] 检索失败: {e}")
            return f"检索失败: {e}"

        sources = self._query_result_to_sources(response)
        if not sources:
            return "未找到相关论文片段"

        try:
            sources = await self._compact_chunk_texts_with_vlm(sources)
            answer = await self._generate_rag_answer(query, sources)
        except ValueError as e:
            logger.error(f"[paper_search] 上下文窗口不足: {e}")
            return f"检索到 {len(sources)} 条结果，但上下文窗口不足，无法生成回答。请尝试缩小查询范围。"
        except Exception as e:
            logger.error(f"[paper_search] 回答生成失败: {e}")
            return f"回答生成失败: {e}"

        return answer if answer else "未能生成回答"

    @staticmethod
    def _validate_code_execute_task(task: str) -> str | None:
        """输入校验：危险模式检测。合法返回 None，非法返回错误信息。"""
        for pattern, label in _DANGEROUS_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                logger.error(f"[code_execute] 危险模式拒绝: {label}")
                return f"任务包含潜在危险操作 ({label})，已被拒绝。请移除危险命令后重试。"
        return None

    async def _code_execute_tool(self, event: AstrMessageEvent, task: str, timeout: int = 300) -> str:
        """LLM Tool wrapper: 启动 claude -p 子进程执行编程任务，同步返回结果。

        Agent 应先调用 paper_search/paper_arag/paper_react 检索相关知识，
        整合上下文后形成完整任务再调用此工具。

        Args:
            event: AstrMessageEvent (injected by framework)
            task: 完整的编程任务描述，需包含所有必要上下文和指令
            timeout: 最大执行秒数，默认300

        Returns:
            Claude Code 的输出文本
        """
        error = self._validate_code_execute_task(task)
        if error:
            return error

        logger.debug(f"[code_execute] 完整任务: {task}")

        work_dir = str(_PLUGIN_DIR)
        cmd = [
            "claude", "-p", "--", task,
            "--output-format", "text",
            "--allowedTools",
            "Read,Write(astrbot_plugin_paperrag/**),Edit(astrbot_plugin_paperrag/**),Bash(git:*,python:*,pytest:*,pip:*),Grep,Glob",
        ]

        logger.info(f"[code_execute] 执行: {task[:100]}...")
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except FileNotFoundError:
            logger.error("[code_execute] claude 命令未找到（未安装或不在 PATH 中）")
            return "Claude Code 未安装或不在 PATH 中，请联系管理员安装。"
        except asyncio.TimeoutError:
            logger.error(f"[code_execute] 超时 ({timeout}s): {task[:100]}...")
            if process is not None:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
            return f"Claude Code 超时 ({timeout}s)，请缩小任务范围或增加超时。"

        output = stdout.decode("utf-8", errors="replace").strip()
        err_output = stderr.decode("utf-8", errors="replace").strip()

        # 权限检测：无论 returncode 是否为 0，stderr 中出现权限关键字都应提示
        perm_in_output = _PERMISSION_KEYWORDS.search(output)
        perm_in_stderr = _PERMISSION_KEYWORDS.search(err_output)
        if perm_in_output or perm_in_stderr:
            logger.warning(f"[code_execute] 需要用户手动授权: {task[:100]}...")
            return (
                f"Claude Code 执行此任务需要额外权限（如 Bash 执行或网络访问）。\n\n"
                f"请先在服务器上审查以下任务是否安全，确认无误后手动执行：\n\n"
                f"```bash\ncd '{work_dir}'\n"
                f"claude --dangerously-skip-permissions -p \"$(cat <<'EOF'\n{task}\nEOF\n)\"\n```\n\n"
                f"注意：--dangerously-skip-permissions 将绕过所有权限检查，仅在审查确认安全后使用。"
            )

        if process.returncode != 0:
            logger.error(f"[code_execute] 非零退出 (rc={process.returncode}): stderr={err_output[:200]}")
            if not output:
                return f"Claude Code 退出码 {process.returncode}: {err_output[:500]}"

        if err_output:
            logger.warning(f"[code_execute] stderr: {err_output[:200]}")

        logger.info(f"[code_execute] 完成 ({len(output)} chars)")
        return output if output else "(no output)"


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
                error_msg = result.get("error") or result.get("message", "Unknown error")
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

            # top_k = -1 表示列出每篇论文的参考文献解析成功/失败统计
            if top_k == -1:
                yield event.plain_result("📊 正在分析参考文献解析质量...")

                doc_stats_path = _PLUGIN_DIR / "data" / "paper_doc_stats.json"
                if not doc_stats_path.exists():
                    yield event.plain_result("📭 paper_doc_stats.json 不存在，请先添加论文")
                    return

                try:
                    with open(doc_stats_path, "r", encoding="utf-8") as f:
                        all_stats = json.load(f)
                except Exception as e:
                    yield event.plain_result(f"❌ 读取 paper_doc_stats.json 失败: {e}")
                    return

                if not isinstance(all_stats, dict):
                    yield event.plain_result("❌ paper_doc_stats.json 格式无效")
                    return

                # Build per-paper stats:
                #   linked: has DOI or arXiv URL (true success)
                #   title_only: has title but no link (LLM OK, link resolution failed)
                #   no_title: empty title (LLM extraction failed)
                paper_rows: list[dict] = []
                unparsed_papers: list[str] = []
                total_refs_all = 0
                total_linked_all = 0
                total_title_only_all = 0
                total_no_title_all = 0

                for file_name, stats in all_stats.items():
                    refs = stats.get("references", {})
                    if not isinstance(refs, dict) or not refs:
                        # Paper added to knowledge base but references never parsed
                        unparsed_papers.append(file_name)
                        continue

                    total = len(refs)
                    linked = 0
                    title_only = 0
                    no_title = 0
                    for r in refs.values():
                        if not isinstance(r, dict):
                            continue
                        has_link = bool(r.get("ref_doi") or r.get("ref_arxiv_url"))
                        has_title = bool(r.get("ref_title", "").strip())
                        if has_link:
                            linked += 1
                        elif has_title:
                            title_only += 1
                        else:
                            no_title += 1

                    total_refs_all += total
                    total_linked_all += linked
                    total_title_only_all += title_only
                    total_no_title_all += no_title

                    paper_rows.append({
                        "file_name": file_name,
                        "total": total,
                        "linked": linked,
                        "title_only": title_only,
                        "no_title": no_title,
                        "unresolved": title_only + no_title,
                    })

                if not paper_rows and not unparsed_papers:
                    yield event.plain_result("📭 没有已解析参考文献的论文")
                    return

                # Sort: most unresolved first, then by file name
                paper_rows.sort(key=lambda r: (-r["unresolved"], r["file_name"]))

                total_papers = len(paper_rows) + len(unparsed_papers)
                total_unresolved = total_title_only_all + total_no_title_all
                output = "📚 **参考文献解析质量报告**\n\n"
                output += f"📊 总计: {total_refs_all} 条参考文献\n"
                output += f"   • 已链接 (DOI/arXiv): {total_linked_all}"
                if total_refs_all > 0:
                    output += f" ({total_linked_all/total_refs_all:.1%})"
                output += f"\n   • 仅有标题无链接: {total_title_only_all}"
                output += f"\n   • 标题为空 (LLM 失败): {total_no_title_all}"
                output += f"\n📄 论文总数: {total_papers}"
                output += f" (已解析: {len(paper_rows)}, 未解析: {len(unparsed_papers)})\n\n"

                output += "📋 **逐篇统计** (未解决数降序):\n\n"

                for i, row in enumerate(paper_rows, 1):
                    file_name = row["file_name"]
                    display_name = file_name if len(file_name) <= 60 else file_name[:57] + "..."

                    if row["unresolved"] == 0:
                        status_icon = "✅"
                    elif row["linked"] > 0:
                        status_icon = "⚠️"
                    else:
                        status_icon = "❌"

                    output += f"{i:3d}. {status_icon} **{display_name}**\n"
                    parts = [f"{row['total']} 条"]
                    if row["linked"]:
                        parts.append(f"{row['linked']} 已链接")
                    if row["title_only"]:
                        parts.append(f"{row['title_only']} 仅标题")
                    if row["no_title"]:
                        parts.append(f"{row['no_title']} 空标题")
                    output += f"      └─ {', '.join(parts)}"
                    if row["total"] > 0:
                        output += f" (链接率 {row['linked']/row['total']:.0%})"
                    output += "\n"

                if unparsed_papers:
                    output += f"\n📭 **未解析参考文献的论文 ({len(unparsed_papers)} 篇):**\n\n"
                    for i, name in enumerate(unparsed_papers, 1):
                        display_name = name if len(name) <= 60 else name[:57] + "..."
                        output += f"  {i}. {display_name}\n"
                        output += f"     💡 使用 /paper reparseref {name} 重新解析\n"

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
            total_papers = stats.get("total_papers", 0)

            if not references:
                yield event.plain_result("📭 数据库中暂无参考文献信息\n💡 请先使用 /paper add 添加论文")
                return

            # 格式化输出
            output = f"📚 **参考文献统计**（每篇论文同一引用只计一次）\n\n"
            output += f"📊 统计概览:\n"
            output += f"   • 被引用文献种类: {len(references)}\n"
            output += f"   • 跨论文引用总次数: {total_refs}\n"
            output += f"   • 论文总数: {total_papers}\n\n"

            output += f"🔝 **Top {min(top_k, len(references))} 高频引用论文**\n\n"

            for i, ref in enumerate(references[:top_k], 1):
                title = ref["title"]
                count = ref["count"]
                authors = ref.get("authors", "")
                year = ref.get("year", "N/A")
                doi = ref.get("doi", "")

                if doi:
                    doi_url = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
                    title_line = f"{i:2d}. [{count:3d}次] [{title}]({doi_url})\n"
                else:
                    title_line = f"{i:2d}. [{count:3d}次] **{title}**\n"
                output += title_line

                meta_parts = []
                if authors:
                    meta_parts.append(authors)
                if year:
                    meta_parts.append(str(year))
                if meta_parts:
                    output += f"    └─ {' · '.join(meta_parts)}\n"

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
        """Batch re-parse references for papers with zero/missing references (Admin)

        Uses lightweight path: PyMuPDF text extraction + LLM reference parsing.
        Does NOT delete or rebuild chunks/embeddings — references only.

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != 'confirm':
            yield event.plain_result("⚠️ This will re-parse references for all papers with zero refs.\n"
                                   "This operation may take a long time.\n"
                                   "Usage: /paper reparse_zero_ref confirm")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        try:
            # Step 1: Resolve LLM config upfront
            yield event.plain_result("🔍 Step 1/4: Resolving LLM config...")
            llm_config = await engine._resolve_llm_config()
            if not llm_config:
                yield event.plain_result(
                    "❌ LLM reference parsing is not configured. "
                    "Enable enable_llm_reference_parsing or set freeapi_url/freeapi_key in plugin config."
                )
                return
            yield event.plain_result(f"🤖 Using LLM: {_llm_display_name(llm_config)}")

            # Step 2: Get papers with zero references
            yield event.plain_result("🔍 Step 2/4: Finding papers with zero references...")

            index_manager = engine._ensure_index_manager_initialized()
            result = await index_manager.get_papers_with_zero_references()

            if "error" in result:
                yield event.plain_result(f"❌ 获取失败: {result['error']}")
                return

            papers = result.get("papers", [])
            total_zero_ref = result.get("total_zero_ref", 0)

            if not papers:
                yield event.plain_result("✅ All papers have references with valid titles")
                return

            yield event.plain_result(f"📊 Found {total_zero_ref} papers with zero/invalid references")

            # Step 3: Locate paper files
            yield event.plain_result("🔍 Step 3/4: Locating paper files...")

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

            # Step 4: Lightweight reference re-parsing
            yield event.plain_result("🔍 Step 4/4: Re-parsing references (lightweight, no re-index)...")

            from rag.reference_processor import process_references_with_llm
            import fitz

            start_time = time.time()
            success_count = 0
            fail_count = 0
            synced_chunks = 0
            sync_errors = 0
            index_manager = engine._ensure_index_manager_initialized()

            for i, paper in enumerate(papers_to_reparse, 1):
                file_path = paper["file_path"]
                file_name = paper["file_name"]

                try:
                    with fitz.open(file_path) as doc:
                        raw_text = "".join(str(page.get_text()) for page in doc)

                    if not raw_text.strip():
                        fail_count += 1
                        logger.warning(f"[reparse_zero_ref] No extractable text: {file_name}")
                        continue

                    refs, _ = await process_references_with_llm(
                        file_path, [], raw_text, llm_config,
                        enable_fallback_search=True,
                    )
                    if refs:
                        success_count += 1
                        # Sync chunk-level cited_ref_ids in Milvus
                        sync_result = await index_manager.sync_cited_ref_ids_for_paper(
                            file_name, refs
                        )
                        if sync_result.get("error"):
                            sync_errors += 1
                            logger.warning(
                                f"[reparse_zero_ref] cited_ref_ids sync failed for "
                                f"{file_name}: {sync_result['error']}"
                            )
                        else:
                            synced_chunks += sync_result.get("synced", 0)
                    else:
                        fail_count += 1
                        logger.warning(f"[reparse_zero_ref] No references found: {file_name}")
                except Exception as e:
                    fail_count += 1
                    logger.error(f"[reparse_zero_ref] Failed {file_name}: {e}")

                if i % 5 == 0 or i == len(papers_to_reparse):
                    yield event.plain_result(
                        f"   Progress: {i}/{len(papers_to_reparse)} "
                        f"(success: {success_count}, failed: {fail_count}, "
                        f"chunks synced: {synced_chunks})"
                    )

            elapsed_time = time.time() - start_time

            output = f"""✅ **Reference Reparse Complete**

📊 Statistics:
  • Total zero-ref papers: {total_zero_ref}
  • Files found: {len(papers_to_reparse)}
  • Successfully re-parsed: {success_count}
  • Failed: {fail_count}
  • Chunk cited_ref_ids synced: {synced_chunks}{' (⚠️ ' + str(sync_errors) + ' sync failures)' if sync_errors else ''}
  • Time: {elapsed_time:.1f}s

💡 Tip: Use /paper refstats -1 to verify results"""

            if not_found:
                output += f"\n\n⚠️ {len(not_found)} papers not found in filesystem"

            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"Failed to reparse zero-ref papers: {e}")
            yield event.plain_result(f"❌ 操作失败: {e}")


    async def _paper_repair_refs(self, event: AstrMessageEvent, confirm: str = ''):
        """Auto-classify and repair papers with unlinked references (Admin)

        Reads paper_doc_stats.json and splits papers into two strategies:

        Strategy A — full_reparse: Papers with any no_title refs (LLM extraction failed)
            or completely unparsed. Runs full pipeline: PyMuPDF text extraction + LLM
            reference parsing + link resolution.

        Strategy B — link_only: Papers where ALL unlinked refs have valid titles. Only
            link resolution failed. Lightweight repair: reloads refs from JSON and
            re-runs PaperLinkResolver enrichment (no LLM extraction, no PyMuPDF).

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != 'confirm':
            yield event.plain_result(
                "⚠️ This will repair all papers with unlinked references.\n"
                "Papers with empty titles → full reparse (LLM)\n"
                "Papers with only missing links → lightweight link repair\n"
                "Usage: /paper repair_refs confirm"
            )
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        try:
            # Step 1: Resolve LLM config
            yield event.plain_result("🔍 Step 1/3: Resolving LLM config...")
            llm_config = await engine._resolve_llm_config()
            if not llm_config:
                yield event.plain_result(
                    "❌ LLM reference parsing is not configured. "
                    "Enable enable_llm_reference_parsing or set freeapi_url/freeapi_key in plugin config."
                )
                return
            yield event.plain_result(f"🤖 Using LLM: {_llm_display_name(llm_config)}")

            # Step 2: Auto-classify papers
            yield event.plain_result("🔍 Step 2/3: Analyzing paper_doc_stats.json...")

            index_manager = engine._ensure_index_manager_initialized()
            classification = await index_manager.classify_papers_for_repair()

            if classification.get("error"):
                yield event.plain_result(f"❌ Analysis failed: {classification['error']}")
                return

            full_reparse = classification.get("full_reparse", [])
            link_only = classification.get("link_only", [])
            total_papers = classification.get("total_papers", 0)

            if not full_reparse and not link_only:
                yield event.plain_result("✅ All papers have fully-linked references. Nothing to repair.")
                return

            # Show classification summary
            total_full_unlinked = sum(p["title_only"] + p["no_title"] for p in full_reparse)
            total_link_unlinked = sum(p["title_only"] for p in link_only)
            full_no_title = sum(p["no_title"] for p in full_reparse)

            yield event.plain_result(
                f"📊 **Auto-Classification** (out of {total_papers} papers):\n\n"
                f"🔴 **Full Reparse** ({len(full_reparse)} papers):\n"
                f"   • {total_full_unlinked} unlinked refs ({full_no_title} with empty titles)\n"
                f"   • Needs: LLM extraction + link resolution\n\n"
                f"🔗 **Link-Only Repair** ({len(link_only)} papers):\n"
                f"   • {total_link_unlinked} unlinked refs (all have valid titles)\n"
                f"   • Needs: link re-resolution only (no LLM extraction)\n"
            )

            papers_dir = self.config.get("papers_dir", "./papers")
            papers_path = Path(papers_dir)
            if not papers_path.exists():
                yield event.plain_result(f"❌ Papers directory does not exist: {papers_dir}")
                return

            start_time = time.time()

            # ---- Strategy B: Link-only repair (runs first — fast, no LLM) ----
            link_success = 0
            link_fail = 0
            link_newly_linked = 0
            link_synced_chunks = 0
            link_sync_errors = 0

            if link_only:
                yield event.plain_result(
                    f"\n🔗 **Phase 1: Link-Only Repair** ({len(link_only)} papers)..."
                )

                from rag.reference_processor import repair_links_for_paper

                link_matched, link_not_found = self._match_papers_to_files(link_only, papers_dir)
                if link_not_found:
                    yield event.plain_result(
                        f"   ⚠️ {len(link_not_found)} link-only papers not found in filesystem, skipped"
                    )

                for i, paper in enumerate(link_matched, 1):
                    file_name = paper["file_name"]
                    try:
                        result = await repair_links_for_paper(
                            file_name, llm_config, enable_fallback_search=True
                        )
                        if result.get("error"):
                            link_fail += 1
                            logger.warning(f"[repair_refs:link] {file_name}: {result['error']}")
                        else:
                            link_success += 1
                            link_newly_linked += result.get("newly_linked", 0)
                    except Exception as e:
                        link_fail += 1
                        logger.error(f"[repair_refs:link] Failed {file_name}: {e}")

                    if i % 10 == 0 or i == len(link_matched):
                        yield event.plain_result(
                            f"   Link repair: {i}/{len(link_matched)} "
                            f"(success: {link_success}, failed: {link_fail}, "
                            f"newly linked: {link_newly_linked})"
                        )

            # ---- Strategy A: Full reparse (runs second — heavy, uses LLM) ----
            full_success = 0
            full_fail = 0
            full_synced_chunks = 0
            full_sync_errors = 0
            all_not_found: list[str] = []

            if full_reparse:
                yield event.plain_result(
                    f"\n🔴 **Phase 2: Full Reparse** ({len(full_reparse)} papers)..."
                )

                full_matched, full_not_found = self._match_papers_to_files(full_reparse, papers_dir)
                all_not_found = full_not_found

                if full_not_found:
                    yield event.plain_result(
                        f"   ⚠️ {len(full_not_found)} full-reparse papers not found:"
                    )
                    for fn in full_not_found[:5]:
                        yield event.plain_result(f"      - {fn}")
                    if len(full_not_found) > 5:
                        yield event.plain_result(f"      ... and {len(full_not_found) - 5} more")

                if not full_matched:
                    yield event.plain_result("   ❌ No full-reparse paper files found")
                else:
                    from rag.reference_processor import process_references_with_llm
                    import fitz

                    for i, paper in enumerate(full_matched, 1):
                        file_path = paper["file_path"]
                        file_name = paper["file_name"]

                        try:
                            with fitz.open(file_path) as doc:
                                raw_text = "".join(str(page.get_text()) for page in doc)

                            if not raw_text.strip():
                                full_fail += 1
                                logger.warning(f"[repair_refs:full] No extractable text: {file_name}")
                                continue

                            refs, _ = await process_references_with_llm(
                                file_path, [], raw_text, llm_config,
                                enable_fallback_search=True,
                            )
                            if refs:
                                full_success += 1
                                sync_result = await index_manager.sync_cited_ref_ids_for_paper(
                                    file_name, refs
                                )
                                if sync_result.get("error"):
                                    full_sync_errors += 1
                                    logger.warning(
                                        f"[repair_refs:full] cited_ref_ids sync failed for "
                                        f"{file_name}: {sync_result['error']}"
                                    )
                                else:
                                    full_synced_chunks += sync_result.get("synced", 0)
                            else:
                                full_fail += 1
                                logger.warning(f"[repair_refs:full] No references found: {file_name}")
                        except Exception as e:
                            full_fail += 1
                            logger.error(f"[repair_refs:full] Failed {file_name}: {e}")

                        if i % 5 == 0 or i == len(full_matched):
                            yield event.plain_result(
                                f"   Full reparse: {i}/{len(full_matched)} "
                                f"(success: {full_success}, failed: {full_fail}, "
                                f"chunks synced: {full_synced_chunks})"
                            )

            elapsed_time = time.time() - start_time

            # ---- Final report ----
            total_success = link_success + full_success
            total_fail = link_fail + full_fail
            total_synced = link_synced_chunks + full_synced_chunks
            total_sync_errs = link_sync_errors + full_sync_errors

            output = f"""✅ **Reference Repair Complete**

📊 **Results** ({elapsed_time:.1f}s):

🔗 Link-Only Repair:
  • Papers: {link_success} success, {link_fail} failed
  • Newly linked refs: {link_newly_linked}

🔴 Full Reparse:
  • Papers: {full_success} success, {full_fail} failed
  • Chunk cited_ref_ids synced: {full_synced_chunks}{' (⚠️ ' + str(full_sync_errors) + ' sync failures)' if full_sync_errors else ''}

📊 Total: {total_success} success, {total_fail} failed
💡 Tip: Use /paper refstats -1 to verify results"""

            if all_not_found:
                output += f"\n\n⚠️ {len(all_not_found)} full-reparse papers not found in filesystem"

            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"Failed to repair references: {e}")
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

            # 清除 paper_doc_stats.json
            doc_stats_path = _PLUGIN_DIR / "data" / "paper_doc_stats.json"
            if doc_stats_path.exists():
                try:
                    doc_stats_path.unlink()
                    yield event.plain_result("✅ Step 1/5c: Document stats cleared")
                except Exception as e:
                    logger.warning(f"清除 paper_doc_stats.json 失败: {e}")

        except Exception as e:
            logger.error(f"Failed to clear document library: {e}")
            yield event.plain_result(f"❌ Failed to clear: {e}")
            return

        # Delete figures, tables, and captions folders
        yield event.plain_result("🔄 Step 2/5: Clearing figures, tables, and captions...")
        plugin_dir = _PLUGIN_DIR
        figures_dir = plugin_dir / "data" / "figures"
        tables_dir = plugin_dir / "data" / "tables"
        captions_dir = plugin_dir / "data" / "captions"

        for target_dir, name in [(figures_dir, "figures"), (tables_dir, "tables"), (captions_dir, "captions")]:
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
                    logger.warning(f"Failed to add {doc_file.name}: {result.get('error') or result.get('message', 'Unknown error')}")

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
                yield event.plain_result(f"❌ Rebuild failed: {add_result.get('error') or add_result.get('message', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Failed to rebuild paper: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ Failed to rebuild: {e}")

    async def _paper_reparseref(self, event: AstrMessageEvent, file_name: str = ''):
        """Re-parse references for a single paper without full index rebuild (Admin)

        Extracts raw text from PDF via PyMuPDF (fast, no re-chunking/embedding),
        then calls process_references_with_llm directly to re-parse references.
        Results are saved to data/paper_doc_stats.json.
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not file_name:
            yield event.plain_result(
                "❌ Usage: /paper reparseref <filename>\n"
                "Example: /paper reparseref 2508.09977v2.pdf"
            )
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        # Locate the PDF file (same logic as rebuildf)
        papers_dir = self.config.get("papers_dir", "./papers")
        papers_dir_resolved = Path(papers_dir).resolve()
        paper_path = None

        for ext in ['', '.pdf', '.PDF']:
            candidate = os.path.join(papers_dir, file_name + ext) if ext else os.path.join(papers_dir, file_name)
            candidate_resolved = Path(candidate).resolve()
            if not str(candidate_resolved).startswith(str(papers_dir_resolved)):
                continue
            if os.path.exists(candidate) and os.path.isfile(candidate):
                paper_path = candidate
                break

        if not paper_path:
            for p in Path(papers_dir).glob("*"):
                if file_name.lower() in p.name.lower():
                    p_resolved = p.resolve()
                    if not str(p_resolved).startswith(str(papers_dir_resolved)):
                        continue
                    if p.is_file():
                        paper_path = str(p)
                        break

        if not paper_path:
            yield event.plain_result(f"❌ File not found: {file_name}")
            return

        actual_file_name = os.path.basename(paper_path)
        yield event.plain_result(f"📝 Re-parsing references for: {actual_file_name}")

        # Extract raw text using PyMuPDF
        try:
            import fitz
            with fitz.open(paper_path) as doc:
                raw_text = "".join(str(page.get_text()) for page in doc)
        except Exception as e:
            logger.error(f"[reparseref] PDF text extraction failed: {e}")
            yield event.plain_result(f"❌ Failed to extract text from PDF: {e}")
            return

        if not raw_text.strip():
            yield event.plain_result(f"❌ No extractable text in {actual_file_name} (scanned PDF?)")
            return

        yield event.plain_result(f"📄 Extracted {len(raw_text)} chars of raw text")

        # Resolve LLM config via engine's shared method (freeapi → provider)
        llm_config = await engine._resolve_llm_config()
        if not llm_config:
            yield event.plain_result(
                "❌ LLM reference parsing is not configured. "
                "Enable enable_llm_reference_parsing or set freeapi_url/freeapi_key in plugin config."
            )
            return

        yield event.plain_result(f"🤖 Using LLM: {_llm_display_name(llm_config)}")

        # Run reference parsing followed by chunk-level cited_ref_ids sync
        try:
            from rag.reference_processor import process_references_with_llm
            yield event.plain_result("⏳ Parsing references with LLM (may take a few minutes)...")
            references, _chunks = await process_references_with_llm(
                paper_path, [], raw_text, llm_config,
                enable_fallback_search=True,
            )
        except Exception as e:
            logger.error(f"[reparseref] Reference parsing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ Reference parsing failed: {e}")
            return

        if references:
            # Sync chunk-level cited_ref_ids in Milvus to match new references
            index_manager = engine._ensure_index_manager_initialized()
            sync_result = await index_manager.sync_cited_ref_ids_for_paper(
                actual_file_name, references
            )
            sync_msg = ""
            if sync_result.get("error"):
                sync_msg = (
                    f"\n   ⚠️ cited_ref_ids sync failed: {sync_result['error']}"
                )
            elif sync_result.get("synced", 0) > 0:
                sync_msg = (
                    f"\n   🔄 Synced cited_ref_ids: {sync_result['synced']} chunks updated"
                    f" ({sync_result.get('unchanged', 0)} unchanged)"
                )
            elif sync_result.get("total_chunks", 0) > 0:
                sync_msg = (
                    f"\n   ✅ cited_ref_ids up-to-date"
                    f" ({sync_result['total_chunks']} chunks)"
                )
            # else: no chunks in Milvus (paper not indexed yet), no message

            yield event.plain_result(
                f"✅ References re-parsed successfully!\n"
                f"   📄 File: {actual_file_name}\n"
                f"   📚 References found: {len(references)}"
                f"{sync_msg}"
            )
        else:
            yield event.plain_result(
                f"⚠️ No references found. The LLM may have timed out "
                f"or {actual_file_name} has no recognizable reference section."
            )
