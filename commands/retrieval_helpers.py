"""Shared retrieval/presentation helpers used by paper commands."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List

from astrbot.api import logger

from .base import PluginCoreBase, _PLUGIN_DIR
from astrbot.api.event import AstrMessageEvent
from rag.token_utils import count_tokens


class RetrievalHelpersMixin(PluginCoreBase):
    """Helpers for search tools, source enrichment, and result presentation."""

    async def _search_papers_tool_impl(self, query: str | None, top_k: int = 5) -> str:
        """搜索本地论文库并返回结果（RAG模式）"""

        if isinstance(query, AstrMessageEvent):
            logger.warning("[PaperRAG] search_papers_tool 收到 event 对象，跳过执行")
            return "❌ 此工具不支持此调用方式"

        if not isinstance(query, str) or not query.strip():
            logger.warning(f"[PaperRAG] search_papers_tool 收到无效 query: {type(query)}")
            return "❌ 查询参数无效"

        if len(query.strip()) < 3:
            return "❌ 查询内容太短"

        engine = self._get_engine()
        if not engine:
            return "❌ RAG引擎未就绪，请检查配置文件"

        try:
            result = await engine.search(query, mode="rag")

            nodes = result.nodes if hasattr(result, "nodes") else []
            scores = result.scores if hasattr(result, "scores") else []

            output = "📚 **检索结果**\n\n"
            for i, (node, score) in enumerate(zip(nodes[:top_k], scores[:top_k]), 1):
                metadata = getattr(node, "metadata", {})
                filename = metadata.get("file_name", "unknown")
                text = getattr(node, "text", "")[:200]
                output += f"[{i}] **{filename}** (score={score:.3f})\n{text}...\n\n"

            return output.strip() if output.strip() else "❌ 未找到相关文档"
        except Exception as e:
            logger.error(f"LLM工具搜索失败: {e}")
            return f"❌ 搜索异常: {e}"

    async def _retrieve_papers_tool_impl(self, query: str | None, top_k: int = 5) -> str:
        """仅检索论文片段，不生成回答"""

        if isinstance(query, AstrMessageEvent):
            logger.warning("[PaperRAG] retrieve_papers_tool 收到 event 对象，跳过执行")
            return "❌ 此工具不支持此调用方式"

        if not isinstance(query, str) or not query.strip():
            logger.warning(f"[PaperRAG] retrieve_papers_tool 收到无效 query: {type(query)}")
            return "❌ 查询参数无效"

        if len(query.strip()) < 2:
            return "❌ 查询内容太短"

        engine = self._get_engine()
        if not engine:
            return "❌ RAG引擎未就绪，请检查配置文件"

        try:
            result = await engine.search(query, mode="retrieve")

            nodes = result.nodes if hasattr(result, "nodes") else []
            scores = result.scores if hasattr(result, "scores") else []
            if not nodes:
                return "📭 未找到相关文档"

            output = "📚 **检索结果**\n\n"
            for i, (node, score) in enumerate(zip(nodes[:top_k], scores[:top_k]), 1):
                metadata = getattr(node, "metadata", {})
                filename = metadata.get("file_name", "unknown")
                text = getattr(node, "text", "")[:300]
                output += f"[{i}] **{filename}** (相似度: {score:.3f})\n{text}...\n\n"

            return output.strip()
        except Exception as e:
            logger.error(f"LLM工具检索失败: {e}")
            return f"❌ 检索异常: {e}"

    def _register_llm_tools(self):
        """注册 LLM 可调用的论文搜索工具"""
        if not self.config.get("enable_llm_tools", True):
            logger.info("📚 Paper RAG LLM工具已禁用")
            return

        async def search_tool(event, query: str | None, top_k: int = 5):
            return await self._search_papers_tool_impl(query, top_k)

        async def retrieve_tool(event, query: str | None, top_k: int = 5):
            return await self._retrieve_papers_tool_impl(query, top_k)

        try:
            self.context.register_llm_tool(
                name="search_papers",
                func_args=[
                    {"type": "string", "name": "query", "description": "搜索查询关键词或问题"},
                    {"type": "integer", "name": "top_k", "description": "返回结果数量，默认5"},
                ],
                desc="【严格使用条件】仅当用户明确提到需要搜索本地论文库、查找论文内容、查询已索引的文档时才能调用。例如：'搜索本地论文'、'查找相关论文'、'查询某篇论文的内容'等明确提及本地论文的场景。如果用户只是询问一般性问题而未提及本地论文，禁止调用此工具。",
                func_obj=search_tool,
            )

            self.context.register_llm_tool(
                name="retrieve_papers",
                func_args=[
                    {"type": "string", "name": "query", "description": "搜索查询关键词"},
                    {"type": "integer", "name": "top_k", "description": "返回结果数量，默认5"},
                ],
                desc="【严格使用条件】仅当用户明确提到需要检索本地已索引论文的原文片段、查看论文原文内容时才能调用。例如：'查看相关论文原文'、'检索论文片段'等明确要求查看本地论文内容的场景。如果用户未明确提及本地论文，禁止调用此工具。",
                func_obj=retrieve_tool,
            )

            logger.info("✅ Paper RAG LLM工具已注册: search_papers, retrieve_papers")
        except Exception as e:
            logger.error(f"注册LLM工具失败: {e}")

    async def _resolve_source_arxiv(self, source: dict) -> dict:
        metadata = source.get("metadata", {})
        filename = metadata.get("file_name", "unknown")

        paper_title = await self._get_paper_title_from_abstract_index(filename)

        if not paper_title:
            paper_title = filename
            if paper_title.endswith(".pdf"):
                paper_title = paper_title[:-4]
            elif paper_title.endswith(".txt"):
                paper_title = paper_title[:-4]

        arxiv_url = ""
        github_url = ""
        try:
            doc_stats_file = _PLUGIN_DIR / "data" / "milvus_abstracts_doc_stats.json"
            if doc_stats_file.exists():
                with open(doc_stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                abstracts = data.get("abstracts", {})
                paper_id = filename[:-4] if filename.endswith(".pdf") else filename
                if paper_id in abstracts:
                    meta = abstracts[paper_id].get("metadata", {})
                    arxiv_url = meta.get("arxiv_url", "")
                    github_url = meta.get("github_url", "")
        except Exception:
            pass

        source = dict(source)
        source["display_name"] = paper_title
        source["arxiv_url"] = arxiv_url
        source["github_url"] = github_url
        logger.debug(f"[PaperRAG] 引用: {filename} -> {paper_title} @ {arxiv_url}")
        return source

    async def _get_paper_title_from_abstract_index(self, filename: str) -> str:
        try:
            doc_stats_file = _PLUGIN_DIR / "data" / "milvus_abstracts_doc_stats.json"
            if not doc_stats_file.exists():
                return ""

            with open(doc_stats_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            paper_id = filename[:-4] if filename.endswith(".pdf") else filename
            abstracts = data.get("abstracts", {})
            if paper_id in abstracts:
                title = abstracts[paper_id].get("title", "")
                if title:
                    logger.debug(f"[PaperRAG] 从abstract index找到标题: {filename} -> {title}")
                    return title

            for abstract_data in abstracts.values():
                if abstract_data.get("file_name", "") == filename:
                    title = abstract_data.get("title", "")
                    if title:
                        logger.debug(f"[PaperRAG] 从abstract index找到标题: {filename} -> {title}")
                        return title
        except Exception as e:
            logger.debug(f"[PaperRAG] 获取论文标题失败: {e}")

        return ""

    async def _resolve_sources_arxiv(self, sources: list) -> list:
        return await asyncio.gather(*[self._resolve_source_arxiv(s) for s in sources])

    def _query_result_to_sources(self, result) -> list:
        nodes = getattr(result, "nodes", [])
        scores = getattr(result, "scores", [1.0] * len(nodes))
        sources = []
        for node, score in zip(nodes, scores):
            metadata = getattr(node, "metadata", {})
            text = getattr(node, "text", "")
            sources.append({
                "text": text,
                "metadata": metadata,
                "score": score,
            })
        return sources

    async def _compact_chunk_texts_with_vlm(self, sources: list) -> list:
        if not sources:
            return sources

        # 简单文本清洗：只删除 [Page X] 标签
        page_pattern = re.compile(r'\[Page\s*\d+(?:-\d+)?\]', re.IGNORECASE)
        table_fig_pattern = re.compile(r'^(?:TABLE|Figure|Fig\.)\s*\d+:\s*$', re.MULTILINE)

        for s in sources:
            text = s.get("text", "")
            # 删除 [Page N] 标签
            text = page_pattern.sub('', text)
            # 删除单独的 TABLE N: / Figure N: 行
            text = table_fig_pattern.sub('', text)
            s["text"] = text

        return sources

    # prompt 模板（{context_text} 是 chunks 拼接结果的占位符，
    # token 预算用 {context_text}=="" 计算固定开销）
    _PROMPT_TEMPLATE = (
        "你是一个严谨的论文阅读助手。请只基于下面给出的本地论文片段回答用户问题。\n\n"
        "要求：\n"
        "1. 如果证据不足，请明确说明\"不足以从当前检索片段得出结论\"，不要编造。\n"
        "2. 回答要直接、结构清晰，优先总结方法、结论、实验或对比关系。\n"
        "3. 需要引用证据时使用 [1]、[2] 这样的编号，对应下方论文片段。\n"
        "4. 不要输出与论文片段无关的泛泛解释。\n"
        "5. **噪声过滤**：部分片段可能包含图表的坐标轴刻度值（如连续的数字 0.0、0.2、0.4、2000、4000、6000 或百分比 0.75、0.50、0.25）、图表标签（如 PSNR、RMSE、Iteration）、散落的短数字序列。这些是图片文字提取的残留，不是论文正文。回答时自动忽略这些噪声片段，不要引用它们，也不要因它们的存在而影响答案质量。\n"
        "6. **具体性优先**：优先提取论文中的具体数据（数值、指标、对比结果）、方法名称、数据集名称，而非泛泛解释概念。\n"
        "7. **简洁回答**：回答控制在 3-5 句话以内，避免冗余扩展。如果用户问具体问题，直接给出具体答案。\n"
        "\n用户问题：\n{query}\n\n检索到的论文片段：\n{context_text}\n\n请给出答案："
    )

    def _build_rag_answer_prompt(self, query: str, sources: list) -> str:
        model_ctx_window = self.config.get("llama_vlm_n_ctx")

        if isinstance(model_ctx_window, int) and model_ctx_window > 0:
            output_max = self.config.get("text_llm_max_tokens", 2048)

            # 固定开销：模板中除 {context_text} 以外的所有文本
            base_prompt = self._PROMPT_TEMPLATE.format(query=query, context_text="")
            base_tokens = count_tokens(base_prompt)
            available = model_ctx_window - output_max - base_tokens

            if available <= 0:
                raise ValueError(
                    f"上下文窗口不足以容纳 prompt 固定开销："
                    f"context_window={model_ctx_window}, "
                    f"output_max_tokens={output_max}, "
                    f"base_tokens={base_tokens}, "
                    f"available={available}。请增大 llama_vlm_n_ctx 或减小 text_llm_max_tokens。"
                )

            # 逐块装入，精确计数块文本 + 分隔符
            blocks: list[str] = []
            used = 0
            for s in sources:
                text = s.get("text", "")
                if not text:
                    continue
                metadata = s.get("metadata", {}) or {}
                display_name = s.get("display_name") or metadata.get("file_name", "unknown")
                chunk_index = metadata.get("chunk_index", 0)
                score = s.get("score", 0.0)
                block = f"[{len(blocks) + 1}] {display_name} (chunk #{chunk_index}, score={score:.3f})\n{text}"

                # 分隔符 + 块文本（第一个块前无分隔符）
                sep = "" if not blocks else "\n\n"
                t = count_tokens(sep + block)

                if not blocks and t > available:
                    raise ValueError(
                        f"第一个 chunk 就需要 {t} tokens，但可用于 chunks 的空间只有 {available}。"
                        f"请增大 llama_vlm_n_ctx 或调小 chunk 大小。"
                    )
                if used + t > available:
                    break

                blocks.append(block)
                used += t

            if len(blocks) < len(sources):
                logger.info(
                    f"[TokenBudget] {len(sources)} → {len(blocks)} chunks "
                    f"({used}/{available} tokens)"
                )

            return self._PROMPT_TEMPLATE.format(query=query, context_text="\n\n".join(blocks))

        # 无 context_window 配置：保持向后兼容，全量装入
        context_blocks = []
        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {}) or {}
            display_name = source.get("display_name") or metadata.get("file_name", "unknown")
            chunk_index = metadata.get("chunk_index", 0)
            score = source.get("score", 0.0)
            text = source.get("text", "")
            context_blocks.append(
                f"[{i}] {display_name} (chunk #{chunk_index}, score={score:.3f})\n{text}"
            )

        return self._PROMPT_TEMPLATE.format(query=query, context_text="\n\n".join(context_blocks))

    async def _generate_rag_answer(self, query: str, sources: list) -> str:
        """Generate a grounded RAG answer from retrieved sources."""
        if not sources:
            return "未找到可用于回答的本地论文片段。"

        prompt = self._build_rag_answer_prompt(query, sources)
        try:
            from provider.llm_utils import call_llm
            answer = await call_llm(prompt, self.context, self.config, temperature=0.2)
            return answer or "LLM 未返回有效回答。"
        except RuntimeError as e:
            return f"已完成检索，但{e}"
        except Exception as e:
            logger.error(f"[PaperRAG] RAG回答生成失败: {e}")
            return f"回答生成失败: {e}"

    def _format_retrieve_response(self, sources: list) -> str:
        output = "📚 **Document Search Results**\n\n"

        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            filename = metadata.get("file_name", "unknown")
            score = source.get("score", 0.0)
            text = source.get("text", "")[:200]
            display_name = source.get("display_name", filename)

            output += f"[{i}] **{display_name}** (similarity: {score:.3f})\n"
            output += f"{text}...\n\n"

        return output.strip()

    def _format_rag_response(self, answer: str, sources: list) -> str:
        output = f"💡 **Answer**\n\n{answer}\n\n"
        output += "📚 **References**\n\n"

        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            filename = metadata.get("file_name", "unknown")
            chunk_index = metadata.get("chunk_index", 0)
            text = source.get("text", "")[:150]
            display_name = source.get("display_name", filename)
            arxiv_url = source.get("arxiv_url", "")

            if arxiv_url:
                ref_text = f"[{display_name}]({arxiv_url})"
            else:
                ref_text = f"**{display_name}**"

            output += f"[{i}] {ref_text} (chunk #{chunk_index})\n"
            output += f"> {text}...\n\n"

        return output.strip()

    def _extract_images_from_sources(self, sources: list, max_images: int = 5) -> List[Dict[str, Any]]:
        seen = set()
        images = []
        for source in sources:
            metadata = source.get("metadata", {}) or {}
            image_path = metadata.get("image_path")
            if not image_path or not os.path.exists(image_path):
                continue
            if image_path in seen:
                continue
            seen.add(image_path)
            caption = metadata.get("image_caption") or os.path.basename(image_path)
            paper = metadata.get("file_name", "unknown")
            images.append({"path": image_path, "caption": caption, "paper": paper})
            if len(images) >= max_images:
                break
        return images
