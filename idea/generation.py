"""
知识融合、创意生成与周报生成
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from astrbot.api import logger

from .datatypes import ResearchIdea, TopicAnalysis

_IDEA_SCHEMA_PATH = Path(__file__).parent / "idea_schema.gbnf"

from .utils import (
    format_ideas_as_markdown,
    fuse_knowledge,
    fuse_knowledge_context,
    load_paper_urls,
)
from provider.llm_utils import (
    parse_json_response,
    extract_text_from_response,
)
from rag.token_utils import count_tokens
from .vm import IdeaEngineVM
from .websearch import IdeaEngineWebSearch
import uuid as uuid_module


class IdeaEngineGeneration(IdeaEngineVM, IdeaEngineWebSearch):
    """知识融合、创意生成与周报生成。继承链：... → IdeaEngineVM → IdeaEngineGeneration"""

    def _fuse_knowledge(
        self,
        local_results: List[Dict]
    ) -> str:
        """将本地知识融合为统一上下文（简化版：只处理local_results）"""
        return fuse_knowledge(local_results)

    def _fuse_knowledge_context(
        self,
        local_results: List[Dict],
        web_results: List[Dict]
    ) -> str:
        """将 local 和 web 结果融合为文本上下文"""
        return fuse_knowledge_context(local_results, web_results)

    def _build_verified_reference_index(self, local_results: List[Dict]) -> str:
        """从 paper_doc_stats.json 查找 chunk 引用的参考文献详情，构建权威引用索引。

        chunk 只存储 cited_ref_ids，完整引用信息（标题、作者、DOI 等）
        在 data/paper_doc_stats.json 中，由 _save_paper_doc_stats 写入。
        """
        from rag.reference_processor import _load_paper_doc_stats

        all_stats = _load_paper_doc_stats()
        if not all_stats:
            logger.debug("_build_verified_reference_index: paper_doc_stats.json 为空")
            return ""

        seen: set[str] = set()
        refs: List[Dict] = []
        for r in local_results:
            metadata = r.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    continue
            cited_ref_ids = metadata.get("cited_ref_ids") or []
            if not cited_ref_ids:
                continue
            file_name = metadata.get("file_name", "")
            paper_key = file_name
            paper_info = all_stats.get(paper_key, {})
            paper_refs = paper_info.get("references", {})
            for ref_id in cited_ref_ids:
                dedup_key = f"{paper_key}:{ref_id}"
                if ref_id and dedup_key not in seen and ref_id in paper_refs:
                    seen.add(dedup_key)
                    refs.append(paper_refs[ref_id])

        if not refs:
            logger.debug(f"_build_verified_reference_index: 无已验证参考文献 (检查了 {len(local_results)} 条 local_results)")
            return ""

        logger.info(f"_build_verified_reference_index: 构建 {len(refs)} 条已验证参考文献")

        lines = ["## 已验证的参考文献索引（权威来源）："]
        for ref in refs:
            title = ref.get("ref_title", "")
            authors = ref.get("ref_authors", "")
            year = ref.get("ref_year", "")
            venue = ref.get("ref_venue", "")
            doi = ref.get("ref_doi", "")
            arxiv_url = ref.get("ref_arxiv_url", "")
            github_url = ref.get("ref_github_url", "")

            first_author = (authors or "").split(",")[0].strip()
            label = f"{first_author} et al., {year}" if first_author and year else ""

            url = f"https://doi.org/{doi}" if doi else arxiv_url or github_url or ""

            parts = [f"- "]
            if label:
                parts.append(f"**{label}**: ")
            parts.append(title)
            if venue:
                parts.append(f" ({venue})")
            if url:
                parts.append(f" [{url}]")
            lines.append("".join(parts))

        return "\n".join(lines)

    def _get_llm_provider(self):
        """获取LLM provider（统一 4 步解析链）。"""
        if not self.context:
            logger.debug("[IdeaEngine] _get_llm_provider: context 为 None")
            return None
        try:
            from provider.llm_utils import get_llm_provider
            return get_llm_provider(self.context, getattr(self, '_plugin_config', None))
        except ImportError:
            logger.warning("[IdeaEngine] 无法导入 get_llm_provider，使用内联回退")
            return None

    def _compute_vlm_max_tokens(self, vlm_provider, prompt: str, max_cap: int = 4096) -> int:
        """Dynamically compute output token budget from VLM n_ctx minus prompt tokens.

        Args:
            max_cap: upper bound for output tokens (default 4096, use 8192 for long-form generation).
        """
        n_ctx = getattr(vlm_provider, 'n_ctx', 16384)
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens > n_ctx - max_cap - 512:
            logger.warning(
                f"[IdeaEngine] Prompt ({prompt_tokens} tokens) nearly fills "
                f"context window ({n_ctx}); output may be truncated or fail"
            )
        return max(1024, min(n_ctx - prompt_tokens - 512, max_cap))

    def _load_paper_urls(self) -> Dict[str, Any]:
        """从 milvus_abstracts_doc_stats.json 加载论文完整信息"""
        return load_paper_urls()

    async def generate_ideas(
        self,
        knowledge_context: str,
        research_domain: str = "",
        num_ideas: int = 3,
        idea_focus: str = "all",
        topic: str = ""
    ) -> List[ResearchIdea]:
        """基于知识上下文生成研究想法"""
        logger.info(f"[IdeaEngine] 生成{num_ideas}个研究想法，topic={topic}")

        vlm_provider = await self._get_vlm_provider_async()
        if vlm_provider:
            logger.info("[IdeaEngine] 使用本地VLM生成ideas")
        else:
            logger.warning("[IdeaEngine] 本地VLM不可用，将使用云端LLM")

        focus_instruction = {
            "novelty": "特别强调创新性和独特贡献",
            "feasibility": "特别强调技术可行性和实现路径",
            "impact": "特别强调潜在影响力和应用价值",
            "all": "综合考虑创新性、可行性和影响力"
        }.get(idea_focus, "")

        prompt = f"""基于以下收集的知识上下文（包含相关论文的摘要和主要贡献），针对用户的研究主题，生成{num_ideas}个研究想法。

**用户研究主题：{topic}**

收集的知识（请仔细阅读，这些是与主题相关的参考资料）：
{knowledge_context[:8000]}

{focus_instruction}

**分析现有工作的痛点**：
从上述论文中分析当前领域的主要问题和挑战：
1. 哪些问题还没有被很好地解决？
2. 现有方法的局限性是什么？（精度、速度、泛化能力、计算成本等）
3. 哪些场景或应用仍然困难？

**重要约束**：
- 想法必须与「{topic}」紧密相关
- 每个想法都要能追溯到参考资料中的具体内容
- 不要生成与主题无关的通用性想法
- **必须参考论文摘要的表述风格**，明确说明解决了什么问题

请为每个想法返回以下JSON格式的信息：
{{
    "ideas": [
        {{
            "title": "想法标题（参考摘要风格，明确研究问题）",
            "description": "详细描述该想法针对的具体问题，以及初步的解决思路",
            "novelty": "创新点：明确说明该想法解决了现有工作中的什么问题/痛点",
            "methodology": "方法论建议：具体的技术路线",
            "potential_challenges": ["挑战1", "挑战2"],
            "related_work": ["相关工作1"],
            "feasibility": 0.0到1.0之间的浮点数,
            "inspiration_sources": ["灵感来源1"]
        }},
        ...
    ]
}}

请严格按照JSON格式返回，只返回JSON，不要包含其他文字。"""

        try:
            response_text = ""
            if vlm_provider:
                response = await vlm_provider.text_chat(
                    prompt=prompt,
                    grammar=str(_IDEA_SCHEMA_PATH),
                )
                if hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, dict):
                    response_text = response.get("content", "") or response.get("text", "")
                else:
                    response_text = str(response)
            else:
                from provider.llm_utils import call_llm
                config = getattr(self, '_plugin_config', None)
                response_text = await call_llm(prompt, self.context, config)

            result = self._parse_json_response(response_text)

            if result and "ideas" in result:
                ideas = []
                for item in result["ideas"][:num_ideas]:
                    ideas.append(ResearchIdea(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        novelty=item.get("novelty", ""),
                        methodology=item.get("methodology", ""),
                        potential_challenges=item.get("potential_challenges", []),
                        related_work=item.get("related_work", []),
                        feasibility=item.get("feasibility", 0.5),
                        inspiration_sources=item.get("inspiration_sources", [])
                    ))
                return ideas
            else:
                logger.warning(f"[IdeaEngine] JSON解析失败，response: {response_text}")
                return []
        except Exception as e:
            logger.error(f"[IdeaEngine] 创意生成失败: {e}")
        return []

    async def add_ideas_to_topic(
        self,
        topic: str,
        num_ideas: int = 3,
        idea_focus: str = "all"
    ) -> Tuple[List[ResearchIdea], Dict[str, Any]]:
        """为已有 topic 追加新想法（复用现有 context）"""
        context_data = self._load_context(topic)
        if not context_data:
            raise ValueError(f"Topic '{topic}' 不存在，请先运行 /idea gen <topic> 生成想法")

        knowledge = {
            "local_results": context_data.get("local_results", []),
            "web_results": context_data.get("web_results", []),
            "fused_context": fuse_knowledge_context(
                context_data.get("local_results", []),
                context_data.get("web_results", [])
            )
        }

        ideas = await self.generate_ideas(
            knowledge_context=knowledge.get("fused_context", ""),
            research_domain=context_data.get("domain", ""),
            num_ideas=num_ideas,
            idea_focus=idea_focus
        )

        self._save_ideas_append(ideas, topic, knowledge)
        return ideas, knowledge

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """从文本中解析 JSON（支持 ````json 包裹）"""
        return parse_json_response(text)

    def _format_ideas_as_markdown(self, ideas: List[ResearchIdea], topic: str) -> str:
        """将想法列表格式化为 Markdown 文本"""
        if not ideas:
            return f"## {topic}\n\n（暂无想法）"

        lines = [f"# {topic}\n"]
        for i, idea in enumerate(ideas, 1):
            lines.append(f"## 想法 {i}: {idea.title}\n")
            lines.append(f"**可行性评分**: {idea.feasibility:.1%}\n")
            if idea.description:
                lines.append(f"**描述**: {idea.description}\n")
            if idea.novelty:
                lines.append(f"**创新点**: {idea.novelty}\n")
            if idea.methodology:
                lines.append(f"**方法**: {idea.methodology}\n")
            if idea.potential_challenges:
                lines.append(f"**挑战**: {'; '.join(idea.potential_challenges)}\n")
            if idea.related_work:
                lines.append(f"**相关工作**: {', '.join(idea.related_work)}\n")
            lines.append("---\n")
        return "\n".join(lines)

    def _extract_text_from_response(self, response) -> str:
        """从 LLM 响应中提取文本"""
        return extract_text_from_response(response)

    def find_topic_by_folder(self, folder_name: str) -> Optional[str]:
        """根据 folder_name 查找 topic 名称"""
        index = self._get_topic_index()
        return index.get(folder_name)

    def convert_to_research_ideas(self, ideas_list: List[Dict[str, Any]]) -> List[ResearchIdea]:
        """将想法字典列表转换为 ResearchIdea 对象列表"""
        ideas = []
        for item in ideas_list:
            idea_data = item.get("idea", item)
            ideas.append(ResearchIdea(
                title=idea_data.get("title", ""),
                description=idea_data.get("description", ""),
                novelty=idea_data.get("novelty", ""),
                methodology=idea_data.get("methodology", ""),
                potential_challenges=idea_data.get("potential_challenges", []),
                related_work=idea_data.get("related_work", []),
                feasibility=idea_data.get("feasibility", 0.5),
                inspiration_sources=idea_data.get("inspiration_sources", [])
            ))
        return ideas

    async def regenerate_all(
        self,
        folder_hash: str,
        num_ideas: int = 3,
        idea_focus: str = "all"
    ) -> Tuple[List[ResearchIdea], str, Dict[str, Any]]:
        """根据 folder hash 重新生成所有 ideas 以及初始周报"""
        context_data = self._load_context(folder_hash)
        if not context_data:
            raise ValueError(f"Folder hash '{folder_hash}' 不存在或无 context.json")

        topic = context_data.get("topic", folder_hash)

        knowledge = {
            "local_results": context_data.get("local_results", []),
            "web_results": context_data.get("web_results", []),
            "fused_context": self._fuse_knowledge_context(
                context_data.get("local_results", []),
                context_data.get("web_results", [])
            )
        }

        ideas = await self.generate_ideas(
            knowledge_context=knowledge.get("fused_context", ""),
            research_domain=context_data.get("domain", ""),
            num_ideas=num_ideas,
            idea_focus=idea_focus
        )

        if not ideas:
            raise ValueError("Ideas 重新生成失败")

        initial_draft = await self._generate_initial_draft_vlm(ideas, topic, knowledge)
        self._regenerate_ideas_save(ideas, topic, knowledge, initial_draft)

        return ideas, initial_draft, knowledge

    def _regenerate_ideas_save(
        self,
        ideas: List[ResearchIdea],
        topic: str,
        knowledge: Dict[str, Any],
        initial_draft: str
    ) -> None:
        """重新生成后保存 ideas 到文件"""

        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        self._save_context(topic, knowledge)

        for f in folder.glob("*.json"):
            if f.name != "context.json":
                f.unlink()

        draft_file = folder / "initial_draft.md"
        with open(draft_file, "w", encoding="utf-8") as f:
            f.write(initial_draft)

        for idea in ideas:
            idea_uuid = str(uuid_module.uuid4())[:8]
            idea_data = {
                "id": idea_uuid,
                "topic": topic,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "idea": {
                    "title": idea.title,
                    "description": idea.description,
                    "novelty": idea.novelty,
                    "methodology": idea.methodology,
                    "potential_challenges": idea.potential_challenges,
                    "related_work": idea.related_work,
                    "feasibility": idea.feasibility,
                    "inspiration_sources": idea.inspiration_sources
                }
            }
            file_path = folder / f"{idea_uuid}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(idea_data, f, ensure_ascii=False, indent=2)

        index = self._get_topic_index()
        index[folder.name] = topic
        self._save_topic_index(index)

    async def analyze_topic(self, topic: str, depth: str = "standard") -> Optional["TopicAnalysis"]:
        """
        分析研究主题，生成搜索策略（简化版：使用VLM）

        Args:
            topic: 研究话题
            depth: 分析深度 (quick/standard/deep)

        Returns:
            TopicAnalysis: 结构化的主题分析
        """
        logger.info(f"[IdeaEngine] 分析主题: {topic}, 深度: {depth}")

        # 优先使用VLM分析
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            logger.warning("[IdeaEngine] VLM不可用，使用简单topic分析")
            return TopicAnalysis(
                domain="",
                keywords=[topic],
                search_queries=[topic],
                local_rag_queries=[topic],
                exploration_angles=[topic],
                summary=topic
            )

        prompt = f"""分析以下研究主题，生成结构化的信息收集计划：

研究主题：{topic}

请分析并返回以下JSON格式的信息：

{{
    "domain": "研究领域",
    "keywords": ["关键词1", "关键词2", ...],
    "search_queries": ["查询1", "查询2", ...],
    "local_rag_queries": ["本地检索词1", "本地检索词2", ...],
    "exploration_angles": ["角度1", "角度2", ...],
    "summary": "主题摘要"
}}

请严格按照JSON格式返回，不要包含其他文字。"""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                temperature=0.1,
                max_tokens=self._compute_vlm_max_tokens(vlm_provider, prompt)
            )

            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, dict):
                response_text = response.get("content", "") or response.get("text", "")
            else:
                response_text = str(response)

            result = self._parse_json_response(response_text)

            if result:
                return TopicAnalysis(
                    domain=result.get("domain", ""),
                    keywords=result.get("keywords", []),
                    search_queries=result.get("search_queries", [topic]),
                    local_rag_queries=result.get("local_rag_queries", [topic]),
                    exploration_angles=result.get("exploration_angles", []),
                    summary=result.get("summary", topic)
                )
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM分析失败: {e}，使用简单分析")

        # Fallback: 简单分析
        logger.warning("[IdeaEngine] VLM 不可用，返回空 domain 的 TopicAnalysis")
        return TopicAnalysis(
            domain="",
            keywords=[topic],
            search_queries=[topic],
            local_rag_queries=[topic],
            exploration_angles=[topic],
            summary=topic
        )

    async def search_knowledge(
        self,
        queries: List[str],
        local_rag_top_k: int = 5,
        web_top_k: int = 0
    ) -> Dict[str, Any]:
        """
        多源知识检索（支持本地RAG + 网络搜索）

        Args:
            queries: 搜索查询列表
            local_rag_top_k: 本地RAG召回数
            web_top_k: 网络搜索召回数

        Returns:
            Dict包含 local_results, web_results, fused_context
        """
        logger.info(f"[IdeaEngine] search_knowledge: 查询数={len(queries)}, local_k={local_rag_top_k}, web_k={web_top_k}")

        local_results = []
        web_results = []

        # 1. 本地RAG搜索
        if self._rag_engine and local_rag_top_k > 0:
            try:
                for query in queries[:5]:  # 限制查询数
                    result = await self._rag_engine.search(query, mode="retrieve")
                    # result is QueryResult with .nodes (list of Node) and .scores (list of float)
                    nodes = result.nodes if hasattr(result, 'nodes') else []
                    scores = result.scores if hasattr(result, 'scores') else [0.0] * len(nodes)
                    logger.info(f"[IdeaEngine] search_knowledge: query='{query[:50]}...' 返回 nodes 数量: {len(nodes)}")
                    for i, node in enumerate(nodes[:local_rag_top_k]):
                        src_metadata = node.metadata if hasattr(node, 'metadata') else {}
                        if isinstance(src_metadata, str):
                            try:
                                src_metadata = json.loads(src_metadata)
                            except Exception:
                                src_metadata = {}
                        score = scores[i] if i < len(scores) else 0.0
                        local_results.append({
                            "text": node.text if hasattr(node, 'text') else str(node),
                            "paper": src_metadata.get("file_name", "Unknown"),
                            "score": score,
                            "metadata": {
                                "file_name": src_metadata.get("file_name", "Unknown"),
                                "chunk_index": src_metadata.get("chunk_index", i),
                                "image_path": src_metadata.get("image_path"),
                                "image_caption": src_metadata.get("image_caption"),
                                "table_csv_path": src_metadata.get("table_csv_path"),
                                "table_png_path": src_metadata.get("table_png_path"),
                                "table_caption": src_metadata.get("table_caption"),
                                "cited_ref_ids": src_metadata.get("cited_ref_ids", []),
                            }
                        })
                logger.info(f"[IdeaEngine] 本地RAG检索完成，找到 {len(local_results)} 条结果")
            except Exception as e:
                logger.error(f"[IdeaEngine] 本地RAG搜索失败: {e}")

        # 2. 网络搜索（通过Bright Data MCP）
        bright_data_ok = self._check_bright_data_config()
        logger.info(f"[IdeaEngine] 网络搜索条件检查: web_top_k={web_top_k}, bright_data_ok={bright_data_ok}")
        if web_top_k > 0 and bright_data_ok:
            try:
                logger.info(f"[IdeaEngine] 开始网络搜索，查询数: {len(queries)}")
                web_results = cast(List[Dict[str, Any]], await self._search_web(queries, web_top_k))
                logger.info(f"[IdeaEngine] 网络搜索完成，找到 {len(web_results)} 条结果")
            except Exception as e:
                logger.error(f"[IdeaEngine] 网络搜索失败: {e}")

        # 3. 融合上下文
        fused_context = self._fuse_knowledge_context(local_results, web_results)

        logger.info(f"[IdeaEngine] search_knowledge 返回: local_results={len(local_results)}, web_results={len(web_results)}")
        return {
            "local_results": local_results,
            "web_results": web_results,
            "fused_context": fused_context,
            "stats": {
                "web_count": len(web_results),
                "local_count": len(local_results)
            }
        }

    async def _generate_initial_draft_vlm(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        使用 VLM 生成初始周报草稿

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            knowledge: 知识检索结果

        Returns:
            初始周报草稿字符串
        """
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            logger.warning("[IdeaEngine] VLM 不可用，使用简单格式化")
            return self._format_ideas_as_markdown(ideas, topic)

        # 构建引用上下文（使用过滤后的图片）
        citations_context = ""
        media_instructions = ""

        # 加载论文URL映射
        paper_urls = self._load_paper_urls()
        logger.info(f"[IdeaEngine] [DEBUG] 已加载论文URL映射，数量: {len(paper_urls)}")

        verified_index = ""

        if knowledge:
            local_results = knowledge.get("local_results", [])
            if local_results:
                # 调用图表过滤
                filtered_images = await self._filter_figures_by_relevance(local_results)
                logger.info(f"[IdeaEngine] 图表预过滤完成，保留 {len(filtered_images)} 张相关图片")

                # DEBUG: 打印过滤后的图片信息
                for i, img in enumerate(filtered_images, 1):
                    img_path = img.get('image_path', '')
                    img_caption = img.get('image_caption', '')
                    logger.info(f"[IdeaEngine] [DEBUG] 图片{i}: path={img_path}, caption={img_caption}")

                # 用 chunk 上下文丰富图表 caption（LLM 纯文本）
                if filtered_images and vlm_provider:
                    temp_figure_infos = []
                    for img in filtered_images:
                        img_path = img.get('image_path', '')
                        if img_path:
                            temp_figure_infos.append({
                                "path": img_path,
                                "caption": img.get('image_caption', '') or Path(img_path).name,
                                "type": "table" if "Table" in Path(img_path).name else "fig",
                            })
                    temp_figure_infos = await self._enrich_figure_captions(
                        temp_figure_infos, local_results, vlm_provider
                    )
                    # 用丰富后的 caption 更新 filtered_images
                    enriched_map = {fi['path']: fi['caption'] for fi in temp_figure_infos}
                    for img in filtered_images:
                        p = img.get('image_path', '')
                        if p in enriched_map:
                            img['image_caption'] = enriched_map[p]

                # 构建本地论文引用上下文（包含正文内容和URL）
                citations_context += "## 本地论文引用：\n"
                papers: Dict[str, List] = {}
                for r in local_results:
                    paper = r.get("paper", "Unknown")
                    if paper not in papers:
                        papers[paper] = []
                    papers[paper].append(r)
                for paper, chunks in papers.items():
                    # 从 milvus_abstracts_doc_stats.json 查找 URL（URL 在 metadata 中）
                    paper_key = paper
                    if paper.endswith('.pdf'):
                        paper_key = paper[:-4]
                    paper_info = paper_urls.get(paper_key, {})
                    metadata = paper_info.get('metadata', {})
                    paper_url = metadata.get('arxiv_url', '') or metadata.get('doi_url', '')
                    if paper_url:
                        citations_context += f"### [{paper}]({paper_url})\n"
                        logger.info(f"[IdeaEngine] [DEBUG] 找到论文URL: {paper} -> {paper_url}")
                    else:
                        citations_context += f"### {paper}\n"
                        logger.info(f"[IdeaEngine] [DEBUG] 未找到论文URL: {paper} (key: {paper_key})")
                    for chunk in chunks[:5]:
                        text = chunk.get("text", "")
                        if text:
                            citations_context += f"- {text[:300]}\n"
                    citations_context += "\n"

                # 追加已验证的参考文献索引
                verified_index = self._build_verified_reference_index(local_results)
                if verified_index:
                    citations_context += verified_index + "\n\n"

                # 添加图片信息（真实路径直接列出，caption 在上，路径在下）
                caption_cache: Dict[str, Dict[str, Any]] = {}  # paper_folder -> {by_filename, by_number}
                media_lines: List[str] = ["\n## 可用图片（必须使用这些真实路径，不要生成新路径）：\n"]
                no_caption_images: List[Dict[str, int | str]] = []  # 供 VLM fallback

                for i, img in enumerate(filtered_images, 1):
                    img_path = img.get('image_path', '')
                    img_filename = Path(img_path).name
                    paper_folder = Path(img_path).parent.name
                    if paper_folder not in caption_cache:
                        caption_cache[paper_folder] = self._load_figure_captions(img_path)
                    caps = caption_cache[paper_folder]
                    by_filename = caps.get("by_filename", {})
                    real_caption = by_filename.get(img_filename, '')
                    if real_caption:
                        img_caption = real_caption
                        logger.info(f"[IdeaEngine] [DEBUG] 图片{i} 使用真实caption: {img_filename} -> {real_caption[:50]}...")
                    else:
                        # Try enriched caption from prior _enrich_figure_captions call
                        enriched = img.get('image_caption', '')
                        if enriched and enriched != img_filename and len(enriched) > 10:
                            img_caption = enriched
                            logger.info(f"[IdeaEngine] [DEBUG] 图片{i} 使用LLM丰富caption: {img_filename} -> {enriched[:50]}...")
                        else:
                            img_caption = img_filename
                            logger.warning(f"[IdeaEngine] [DEBUG] 图片{i} 无真实caption: {img_filename}")
                            no_caption_images.append({"index": i, "path": img_path, "filename": img_filename})
                    media_lines.append(f"图 {i}：{img_caption}\n{img_path}\n")

                # VLM fallback：批量为无 caption 的图片生成描述
                if no_caption_images and vlm_provider:
                    vlm_descriptions = await self._vlm_describe_images_batch(vlm_provider, no_caption_images)
                    desc_map: Dict[int, str] = {int(desc["index"]): str(desc.get("caption", "")) for desc in vlm_descriptions if "index" in desc}
                    for li, line in enumerate(media_lines):
                        for idx, vlm_cap in desc_map.items():
                            if line.startswith(f"图 {idx}："):
                                # 替换 caption 部分，保留路径
                                parts = line.split('\n', 1)
                                if len(parts) == 2:
                                    media_lines[li] = f"图 {idx}：{vlm_cap}\n{parts[1]}"
                                logger.info(f"[IdeaEngine] [DEBUG] VLM 补充 caption: 图 {idx} -> {vlm_cap[:50]}...")
                                break

                media_instructions = ''.join(media_lines)

            # 添加网络搜索引用
            web_results = knowledge.get("web_results", [])
            if web_results:
                citations_context += "\n## 网络搜索引用：\n"
                for i, r in enumerate(web_results[:5], 1):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "")[:200]
                    if url:
                        citations_context += f"- [{title}]({url})\n"
                    else:
                        citations_context += f"- {title}\n"
                    if snippet:
                        citations_context += f"  摘要: {snippet}...\n"
                citations_context += "\n"

        ideas_summary = self._format_ideas_as_markdown(ideas, topic)

        prompt = f"""基于以下研究想法和参考资料，生成一个详细完整的组会周报。

研究主题：{topic}

研究想法：
{ideas_summary}

参考资料（RAG检索到的chunk，包含丰富信息，请充分利用）：
{citations_context}
{media_instructions}

请生成一个详细完整的组会周报，包含以下章节，每个章节都要有详细展开：
1. 背景动机：详细说明问题的背景、重要性、现有方法的不足（5-8句）
2. 相关工作：详细综述相关方法和论文，引用论文的具体贡献（5-8句）

3. 方法论：详细描述方法细节、工作流程、技术路线（5-10句）
4. 创新点：明确列出2-3个具体创新点，并解释为什么这些创新有效（5-8句）
5. 实验benchmark：详细说明实验设置、数据集、对比方法、评价指标（5-8句）
6. 挑战与解决方案：每个挑战都要详细说明原因和对应的具体解决方案（5-8句）
7. 下一步计划：具体的下一步研究方向和可行的改进思路（3-5句）
8. 参考文献：列出所有引用的论文和网页资源，**严格格式**：
`1. [论文全名](URL)`

**章节标题格式（必须严格遵守）**：每个章节必须使用如下精确标题，不得自行修改或添加额外修饰：
- `## 1. 背景动机`
- `## 2. 相关工作`
- `## 3. 方法论`
- `## 4. 创新点`
- `## 5. 实验Benchmark`
- `## 6. 挑战与解决方案`
- `## 7. 下一步计划`
- `## 8. 参考文献`
- 数字序号列表，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL（如 `URL (URL)` ）、禁止纯文本URL、禁止不使用markdown链接

**重要**：
1. 参考资料中包含丰富的细节信息，请充分利用这些信息生成详细内容，不要简略！
2. **图表引用格式（重要，严格遵守）**：
   - **禁止在正文中使用任何图片语法**：`!(..)`、`` ![](..) ``、`[图片:...](...)` 一律禁止
   - **禁止在正文中编造具体的图/表编号**（如"如图1所示"、"如图2所示"、"如图3所示"），因为你还不知道哪些图表实际存在
   - 如需引用图表，只使用泛指描述（如"相关实验结果如图所示"、"方法流程如图表所示"），不指定编号
   - **参考文献章节中禁止出现任何图片路径**
3. **不要引用不相关的图片**，如果内容与某张图片无关，不要引用
4. **引用网络资源**：在相关工作章节中，如果某些方法或观点来自网络搜索结果，请使用 `[标题](URL)` 格式引用
5. **参考文献必须完整**：在"参考文献"章节中，**严格格式** `1. [**论文全名**](URL)` 列出所有本地论文和网络资源，**禁止裸URL或括号重复URL**

**强制引用规则（必须严格遵守）**：
- **所有论文引用必须使用上述"已验证的参考文献索引"中的标题和链接**
- 该索引是经过 LLM+arXiv 校验的权威来源，论文名、作者、年份均为正确值
- "本地论文引用"部分的正文可能包含 PDF 提取噪声，其论文名称不可直接使用
- 正文中提到某篇论文时，查找索引中对应的条目，使用索引中的标题和 DOI 链接
- 引用格式：在方法名/论文名**首次出现的位置**直接替换为 markdown 链接，如 `[PanoGS](url)`、`[FLARE](url)`
- **禁止**在句子末尾追加论文全名！错误示例：`PanoGS 提出了xxx。PanoGS: Gaussian-based Panoptic Segmentation...` ← 这种格式绝对不允许。全名只允许出现在参考文献章节
"""

        try:
            logger.info("[IdeaEngine] 使用 VLM 生成详细初始周报草稿...")
            max_tokens_vlm = self._compute_vlm_max_tokens(vlm_provider, prompt, max_cap=8192)
            logger.info(
                f"[IdeaEngine] VLM 草稿 max_tokens={max_tokens_vlm} "
                f"(n_ctx={getattr(vlm_provider, 'n_ctx', 16384)}, prompt={count_tokens(prompt)} tokens)"
            )
            draft = await self._vlm_chat_with_progress(
                vlm_provider,
                prompt=prompt,
                temperature=0.7,
                max_tokens=max_tokens_vlm,
                task_name="VLM生成初始周报草稿"
            )

            # ===== Plan B 两阶段润色 =====
            # 仅在有引用上下文时执行两阶段润色
            if citations_context and len(citations_context) > 50:
                # --- 步骤1：生成核心记忆 ---
                core_memory = ""
                try:
                    logger.info(f"[IdeaEngine] Plan B 步骤1：生成核心记忆，引用: {len(citations_context)} 字符")
                    memory_prompt = f"""请对以下学术引用资料生成一段简洁的"核心观点记忆"（不超过800字），用于后续润色组会周报。

要求：
- 保留每个论文的：论文名、核心方法/技术路线、关键贡献/结论
- 去掉冗余的实验细节和重复信息
- 用简洁的要点列表组织，每条不超过2句
- 输出格式：直接输出压缩后的核心观点，不要加任何前缀说明
- **注意**：引用资料末尾的"已验证的参考文献索引"是经过校验的权威来源，其论文名、作者、年份、链接均为正确值，压缩时务必完整保留这些信息

引用资料：
{citations_context}

核心观点记忆："""
                    mem_max_tokens = self._compute_vlm_max_tokens(vlm_provider, memory_prompt)
                    memory_response = await vlm_provider.text_chat(
                        prompt=memory_prompt,
                        contexts=[],
                        temperature=0.2,
                        max_tokens=mem_max_tokens
                    )
                    core_memory = extract_text_from_response(memory_response) or ""
                    logger.info(f"[IdeaEngine] Plan B 核心记忆生成完成，长度: {len(core_memory)}")
                except Exception as e:
                    logger.warning(f"[IdeaEngine] Plan B 核心记忆生成失败: {e}")
                    core_memory = citations_context[:2000]

                # --- 步骤2：用核心记忆 + 草稿润色 ---
                if core_memory:
                    try:
                        verified_section = f"\n\n## 已验证的参考文献索引（权威来源，必须使用这些标题和链接）：\n{verified_index}" if verified_index else ""
                        polish_prompt = f"""你是一个学术助手，负责对以下组会周报草稿进行润色和完善。

参考资料（核心记忆）：
{core_memory}{verified_section}

原始草稿：
{draft}

**重要指令**：
- 在原文基础上适当扩展：每个简短的要点/列表项扩展为1-2句连贯段落
- 保持原文的整体结构和章节顺序，只做润色和扩展，不打乱框架
- 充分利用核心记忆中的信息，但不要直接复制，要融会贯通

**强制引用规则（必须严格遵守）**：
- **所有论文引用必须使用上述"已验证的参考文献索引"中的标题和链接**
- 该索引是经过 LLM+arXiv 校验的权威来源，论文名、作者、年份、DOI 均为正确值
- "本地论文引用"正文可能含 PDF 噪声，其论文名称不可直接使用
- 正文提到某篇论文时，必须在索引中查找对应条目，使用索引中的标题和 DOI 链接

格式要求：
- 包含章节：背景动机、相关工作、方法论、创新点、实验benchmark、挑战与解决方案、下一步计划、参考文献
- **扩展原则**：将简短的要点列表扩展为连贯段落，但不能变成全新的内容
- **列表格式**：创新点和挑战与解决方案部分使用数字序号列表（如"1. 挑战一：xxx"）

**正文引用格式（重要）**：
- 正文中的引用：使用论文简称加markdown链接，如 [FLARE](https://arxiv.org/abs/2502.12138)、[NoPoSplat](https://arxiv.org/abs/2505.23716)
- **禁止在正文中使用论文全名或裸URL**
**参考文献格式（重要，严格遵守）**：
- 放在最后一个章节
- 每行一条，**严格格式**：`1. [论文全名](URL)`
- 数字序号列表，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL

**图表引用格式（重要，严格遵守）**：
- **禁止在正文中使用任何图片语法**
- **禁止在正文中引用具体图号**（如"如图1所示"），图表将由系统追加到文档末尾
- **参考文献章节中禁止出现任何图片路径**
- **不要生成"论文图表"章节**，图表由系统自动追加到末尾

请直接输出润色后的内容："""
                        logger.info(f"[IdeaEngine] Plan B 步骤2：润色草稿")
                        polish_provider = self.context.get_using_provider() if hasattr(self.context, 'get_using_provider') else None
                        if polish_provider:
                            response = await polish_provider.text_chat(
                                prompt=polish_prompt,
                                contexts=[],
                                temperature=0.3,
                            )
                            polished = extract_text_from_response(response)
                        else:
                            logger.warning("[IdeaEngine] Plan B 步骤2 无 polish provider，保持原内容")
                            polished = ""
                        if polished and len(polished) > 100:
                            draft = polished
                            logger.info(f"[IdeaEngine] Plan B 润色完成，长度: {len(polished)}")
                        else:
                            logger.warning(f"[IdeaEngine] Plan B 润色结果过短，保持原内容")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] Plan B 润色失败: {e}，保持原内容")
            else:
                # 无引用上下文时，直接润色草稿
                logger.info("[IdeaEngine] 无引用上下文，直接润色草稿")
                try:
                    simplify_prompt = f"""你是一个学术助手，负责对以下组会周报草稿进行润色和完善。

原始草稿：
{draft}

**重要指令**：
- 在原文基础上适当扩展：每个简短的要点/列表项扩展为1-2句连贯段落
- 保持原文的整体结构和章节顺序，只做润色和扩展，不打乱框架

格式要求：
- 包含章节：背景动机、相关工作、方法论、创新点、实验benchmark、挑战与解决方案、下一步计划、参考文献
- **扩展原则**：将简短的要点列表扩展为连贯段落，但不能变成全新的内容

**正文引用格式（重要）**：
- 使用论文简称加markdown链接，如 [FLARE](https://arxiv.org/abs/2502.12138)
- **禁止在正文中使用论文全名或裸URL**

**参考文献格式（重要，严格遵守）**：
- 每行一条，**严格格式**：`1. [论文全名](URL)`

**图表引用格式（重要，严格遵守）**：
- **禁止在正文中使用任何图片语法**
- **不要生成"论文图表"章节**

请直接输出润色后的内容："""
                    polish_provider = self.context.get_using_provider() if hasattr(self.context, 'get_using_provider') else None
                    if polish_provider:
                        response = await polish_provider.text_chat(
                            prompt=simplify_prompt,
                            contexts=[],
                            temperature=0.3,
                        )
                        polished = extract_text_from_response(response)
                    else:
                        logger.warning("[IdeaEngine] 直接润色无 polish provider，保持原内容")
                        polished = ""
                    if polished and len(polished) > 100:
                        draft = polished
                        logger.info(f"[IdeaEngine] 直接润色完成，长度: {len(polished)}")
                    else:
                        logger.warning(f"[IdeaEngine] 润色结果过短，保持原内容")
                except Exception as e:
                    logger.warning(f"[IdeaEngine] 直接润色失败: {e}，保持原内容")

            return draft
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 生成失败: {e}，使用简单格式化")
            return ideas_summary

    async def to_feishu_markdown(
        self,
        ideas: List[ResearchIdea],
        topic: str = "",
        include_sources: bool = True,
        initial_draft: str = ""
    ) -> str:
        """
        将研究想法格式化为飞书文档兼容的Markdown格式（带VLM润色）

        流程：
        1. 如果有 initial_draft，使用它作为内容；否则从 ideas 生成本地格式化草稿
        2. VLM 润色内容（结构、格式、语言）
        3. 返回飞书兼容的Markdown格式

        格式规范：
        - 标题层级：# 一级 > ## 二级 > ### 三级
        - 列表格式：使用 - 或 1. ，保持一致性
        - 图片引用：使用 [图X] 格式
        - 公式格式：使用 $公式$ 行内公式
        - 飞书兼容：不使用复杂表格语法

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            include_sources: 是否包含灵感来源
            initial_draft: 预生成的周报草稿（来自 _generate_initial_draft_vlm）

        Returns:
            str: 飞书兼容的Markdown格式内容
        """
        if not ideas and not initial_draft:
            return ""

        # Step 1: 确定要润色的内容
        if initial_draft:
            # 使用预生成的草稿
            content_to_polish = initial_draft
        else:
            # 从 ideas 生成本地格式化草稿
            markdown_parts = [f"# {topic or '研究想法'}\n" if topic else "# 研究想法\n"]

            for i, idea in enumerate(ideas, 1):
                feasibility_bar = "★" * int(idea.feasibility * 5) + "☆" * (5 - int(idea.feasibility * 5))

                markdown_parts.append(f"## {i}. {idea.title}\n")
                markdown_parts.append(f"**可行性**: {feasibility_bar} ({idea.feasibility:.0%})\n")
                markdown_parts.append(f"\n### 描述\n{idea.description}\n")
                markdown_parts.append(f"\n### 创新点\n{idea.novelty}\n")
                markdown_parts.append(f"\n### 方法论\n{idea.methodology}\n")

                if idea.potential_challenges:
                    markdown_parts.append("\n### 潜在挑战\n")
                    for challenge in idea.potential_challenges:
                        markdown_parts.append(f"- {challenge}\n")

                if idea.related_work:
                    markdown_parts.append("\n### 相关工作\n")
                    for work in idea.related_work:
                        markdown_parts.append(f"- {work}\n")

                if include_sources and idea.inspiration_sources:
                    markdown_parts.append("\n### 灵感来源\n")
                    for source in idea.inspiration_sources:
                        markdown_parts.append(f"- {source}\n")

                markdown_parts.append("\n---\n")

            content_to_polish = "".join(markdown_parts)

        # Step 2: VLM 润色
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            logger.warning("[IdeaEngine] VLM不可用，返回未润色版本")
            return content_to_polish

        polish_prompt = f"""你是一个学术写作润色专家。请对以下研究想法内容进行润色，使其更加专业、流畅、符合学术规范。

要求：
1. 保持原有结构和关键信息
2. 优化语言表达，使其更加专业和准确
3. 改善句子结构，避免冗余
4. 确保格式规范（标题层级、列表符号等）
5. 输出必须是有效的Markdown格式

待润色的内容：
{content_to_polish}

请直接输出润色后的Markdown内容，不要包含其他解释或说明。"""

        try:
            polish_max_tokens = self._compute_vlm_max_tokens(vlm_provider, polish_prompt)
            polished_response = await vlm_provider.text_chat(
                prompt=polish_prompt,
                temperature=0.3,
                max_tokens=polish_max_tokens
            )

            if hasattr(polished_response, 'content'):
                polished = polished_response.content.strip()
            elif isinstance(polished_response, dict):
                polished = polished_response.get("content", "") or polished_response.get("text", "")
            else:
                polished = str(polished_response)

            if polished and len(polished) > len(content_to_polish) * 0.5:
                logger.info(f"[IdeaEngine] VLM润色完成，原始长度 {len(content_to_polish)}，润色后 {len(polished)}")
                return polished
            else:
                logger.warning("[IdeaEngine] VLM润色结果异常，返回未润色版本")
                return content_to_polish

        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM润色失败: {e}，返回未润色版本")
            return content_to_polish
