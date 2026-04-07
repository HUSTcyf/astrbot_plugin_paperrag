"""
研究创意生成引擎

整合Bright Data网络搜索 + 本地Paper RAG + LLM生成
"""

import json
import re
import asyncio
import os
import base64
import httpx
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import logger
from astrbot.core.agent.run_context import ContextWrapper


class CoreAPIClient:
    """CORE API v3 客户端 - 用于搜索和解析学术论文的 arxiv 链接"""

    BASE_URL = "https://api.core.ac.uk/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def search_by_title(self, title: str, year: Optional[int] = None, limit: int = 5) -> list:
        """根据论文标题搜索论文（异步）- 支持精确和模糊匹配，带重试"""
        # 优先尝试精确匹配
        query_parts = [f'title:"{title}"']
        if year:
            query_parts.append(f"publishedDate:{year}")
        query = " AND ".join(query_parts)

        results = await self._do_search_with_retry(query, limit)
        if results:
            return results

        # 如果精确匹配失败，尝试模糊匹配（从标题提取关键词）
        # 移除常见词缀，获取核心关键词
        clean_title = re.sub(r'\s*(:| - |\.|,)\s*', ' ', title)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()

        # 提取前5个单词作为搜索词
        words = clean_title.split()[:6]
        if len(words) >= 2:
            fuzzy_query = ' '.join(words)
            logger.info(f"[CoreAPI] 精确匹配失败，尝试模糊搜索: {fuzzy_query}")
            return await self._do_search_with_retry(fuzzy_query, limit)

        return []

    async def _do_search_with_retry(self, query: str, limit: int, max_retries: int = 2) -> list:
        """执行 CORE API 搜索，带重试机制"""
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.BASE_URL}/search/works",
                        headers=self.headers,
                        json={"q": query, "limit": min(limit, 100)}
                    )
                    response.raise_for_status()
                    return response.json().get("results", [])
            except httpx.HTTPStatusError as e:
                logger.error(f"CORE API HTTP错误: {e.response.status_code} - {e.response.text[:200] if e.response.text else 'empty'}")
                return []
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"CORE API 连接失败，重试 ({attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(1)  # 等待1秒后重试
                else:
                    logger.error(f"CORE API 连接失败: {type(e).__name__}: {e}")
                    return []
            except Exception as e:
                logger.error(f"CORE API搜索失败: {type(e).__name__}: {e}")
                return []
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

    async def get_arxiv_link(self, paper_title: str) -> Tuple[Optional[str], Optional[str]]:
        """
        根据论文标题获取 arxiv 链接和 GitHub 链接（异步）

        Returns:
            Tuple[arxiv_url, github_url]: (arxiv链接, GitHub链接)
        """
        works = await self.search_by_title(paper_title, limit=3)
        if not works:
            return None, None

        # 优先找有 arxiv 的结果
        for work in works:
            arxiv_id = self.extract_arxiv_id(work)
            if arxiv_id:
                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

                # 尝试找 GitHub 链接
                github_url = None
                download_url = work.get("downloadUrl", "") or ""
                if "github.com" in download_url.lower():
                    github_url = download_url
                else:
                    # 搜索结果的 description 或其他字段可能包含 GitHub
                    desc = work.get("description", "") or ""
                    if "github.com" in desc.lower():
                        github_match = re.search(r'github\.com/[\w\-]+/[\w\-]+', desc, re.IGNORECASE)
                        if github_match:
                            github_url = f"https://{github_match.group()}"

                return arxiv_url, github_url

        return None, None

    def _extract_arxiv_from_works(self, works: list) -> Tuple[Optional[str], Optional[str]]:
        """从 CORE API 搜索结果中提取 arxiv 和 github 链接"""
        for work in works:
            arxiv_id = self.extract_arxiv_id(work)
            if arxiv_id:
                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

                # 尝试找 GitHub 链接
                github_url = None
                download_url = work.get("downloadUrl", "") or ""
                if "github.com" in download_url.lower():
                    github_url = download_url
                else:
                    desc = work.get("description", "") or ""
                    if "github.com" in desc.lower():
                        github_match = re.search(r'github\.com/[\w\-]+/[\w\-]+', desc, re.IGNORECASE)
                        if github_match:
                            github_url = f"https://{github_match.group()}"

                return arxiv_url, github_url
        return None, None


@dataclass
class ResearchIdea:
    """研究想法"""
    title: str
    description: str
    novelty: str
    methodology: str
    potential_challenges: List[str]
    related_work: List[str]
    feasibility: float
    inspiration_sources: List[str]


@dataclass
class TopicAnalysis:
    """主题分析结果"""
    domain: str
    keywords: List[str]
    search_queries: List[str]
    local_rag_queries: List[str]
    exploration_angles: List[str]
    summary: str


class IdeaEngine:
    """
    研究创意生成引擎

    使用流程：
    1. analyze_topic - 分析研究主题
    2. search_knowledge - 收集知识（网络+本地）
    3. generate_ideas - 生成研究想法
    """

    def __init__(self, context, rag_engine=None):
        """
        初始化创意引擎

        Args:
            context: AstrBot上下文（用于LLM调用）
            rag_engine: RAG引擎实例
        """
        self.context = context
        self._rag_engine = rag_engine
        self._bright_data_available = True

    def _get_llm_provider(self):
        """获取LLM provider"""
        if not self.context:
            return None
        # 尝试获取当前正在使用的provider
        provider = getattr(self.context, 'get_using_provider', None)
        if provider:
            return provider()
        # fallback: 尝试通过provider_manager获取
        provider_manager = getattr(self.context, 'provider_manager', None)
        if provider_manager:
            inst_map = getattr(provider_manager, 'inst_map', None)
            if isinstance(inst_map, dict) and inst_map:
                return list(inst_map.values())[0]
        return None

    def _get_feishu_tool(self):
        """获取飞书MCP工具"""
        if not self.context:
            logger.error("[IdeaEngine] context 为 None")
            return None
        provider_manager = getattr(self.context, 'provider_manager', None)
        if not provider_manager:
            logger.error("[IdeaEngine] provider_manager 为 None")
            return None
        llm_tools = getattr(provider_manager, 'llm_tools', None)
        if not llm_tools:
            logger.error("[IdeaEngine] llm_tools 为 None")
            return None

        func_list = getattr(llm_tools, 'func_list', [])
        logger.info(f"[IdeaEngine] func_list 长度: {len(func_list)}")
        logger.info(f"[IdeaEngine] func_list 工具名: {[t.name for t in func_list]}")

        # 打印每个工具的详细信息
        for i, tool in enumerate(func_list):
            logger.info(f"[IdeaEngine] 工具[{i}]: name={tool.name}, description={tool.description[:50] if tool.description else 'None'}...")
            # 检查是否是 MCPTool
            if hasattr(tool, 'mcp_server_name'):
                logger.info(f"[IdeaEngine]   -> MCP工具, server={tool.mcp_server_name}")
            if hasattr(tool, 'parameters'):
                logger.info(f"[IdeaEngine]   -> parameters={tool.parameters}")

        # 查找 feishu 相关的工具
        for tool in func_list:
            if 'feishu' in tool.name.lower():
                logger.info(f"[IdeaEngine] 找到飞书工具: {tool.name}")
                # 检查工具类型
                if hasattr(tool, 'mcp_server_name'):
                    logger.info(f"[IdeaEngine] 这是 MCP 工具, server={tool.mcp_server_name}")
                return tool
        return None

    def _extract_text_from_response(self, response) -> str:
        """从 LLM 响应中提取文本"""
        # 方法1：检查 result_chain（AstrBot 格式）
        if hasattr(response, 'result_chain'):
            chain = getattr(response.result_chain, 'chain', None)
            if chain and len(chain) > 0:
                first = chain[0]
                if hasattr(first, 'get_text'):
                    return first.get_text()
                elif hasattr(first, 'text'):
                    return first.text
        # 方法2：检查 content 属性（LlamaCpp 格式）
        if hasattr(response, 'content'):
            return response.content
        # 方法3：dict 格式
        if isinstance(response, dict):
            return response.get("content", "") or response.get("text", "")
        # 方法4：字符串格式
        return str(response)

    async def _pre_resolve_arxiv_links(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        预解析本地论文的 arxiv 链接，避免在 prompt 中重复查询

        Args:
            knowledge: 知识检索结果

        Returns:
            Dict: 增强了 arxiv_links 和 github_links 的 knowledge
        """
        if not knowledge:
            return knowledge

        local_results = knowledge.get("local_results", [])
        if not local_results:
            return knowledge

        core_api_key = self._get_core_api_key()
        if not core_api_key:
            logger.warning("[IdeaEngine] CORE API Key 未配置，无法预解析 arxiv 链接")
            return knowledge

        # 克隆 knowledge 避免修改原始数据
        enriched = dict(knowledge)
        enriched["arxiv_links"] = {}
        enriched["github_links"] = {}

        core_client = CoreAPIClient(core_api_key)

        for i, result in enumerate(local_results[:10]):
            paper = result.get("paper", "")
            if not paper or paper in enriched["arxiv_links"]:
                continue

            arxiv_url, github_url = await core_client.get_arxiv_link(paper)
            if arxiv_url:
                enriched["arxiv_links"][paper] = arxiv_url
                enriched["github_links"][paper] = github_url
                logger.info(f"[IdeaEngine] 预解析: {paper[:30]} -> {arxiv_url}")

        return enriched

    async def _polish_content_for_feishu(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any], str]:
        """
        使用 LLM 将研究想法润色为组会周报学术风格

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            knowledge: 知识检索结果（包含 web_results, local_results）

        Returns:
            Tuple[str, Dict, str]: (润色后的 Markdown 内容（带引用）, 提取的媒体资源, LLM生成的标题)
        """
        provider = self._get_llm_provider()
        if not provider:
            logger.warning("[IdeaEngine] 无法获取 LLM Provider，使用原始格式")
            return self._format_ideas_as_markdown(ideas, topic), {"images": [], "tables": []}, topic

        # 构建原始内容摘要
        ideas_summary = self._format_ideas_as_markdown(ideas, topic)

        # 构建引用上下文（包含媒体资源）
        citations_context, extracted_media = self._build_citations_context(knowledge)

        # 构建媒体说明
        media_instructions = ""
        if extracted_media["images"]:
            media_instructions += f"\n\n**可用的图片资源：**\n"
            for img in extracted_media["images"]:
                media_instructions += f"- {img['index']}: {img['caption']} (来源: {img['source_paper']}, 页码: {img['source_page']})\n"
        if extracted_media["tables"]:
            media_instructions += f"\n**可用的表格资源：**\n"
            for tbl in extracted_media["tables"]:
                media_instructions += f"- {tbl['index']}: {tbl['caption']} (来源: {tbl['source_paper']}, 页码: {tbl['source_page']})\n"
                if tbl['csv_content'] and tbl['csv_content'] != "(无法读取)":
                    media_instructions += f"  表格内容预览:\n```\n{tbl['csv_content'][:200]}...\n```\n"

        # 第一步：让 LLM 生成一个合适的文档标题
        title_prompt = f"""给定以下研究主题，请为飞书文档生成一个简洁、有意义、学术风格的标题。

研究主题：{topic}

要求：
1. 标题应该反映研究的核心内容，不要直接使用原始问题
2. 标题长度适中（5-15个字）
3. 可以包含 emoji 作为装饰
4. 直接输出标题，不要加任何说明

例如：
- 如果主题是"大模型在代码生成中的应用"，可以生成："🚀 代码生成新范式：大模型赋能编程"
- 如果主题是"多模态大模型研究"，可以生成："🔍 多模态大模型研究进展"

请直接输出标题："""

        try:
            title_response = await provider.text_chat(
                prompt=title_prompt,
                contexts=[],
                temperature=0.7,
                max_tokens=256
            )
            generated_title = self._extract_text_from_response(title_response)
            generated_title = generated_title.strip() if generated_title else topic
            logger.info(f"[IdeaEngine] LLM 生成的标题: {generated_title}")
        except Exception as e:
            logger.warning(f"[IdeaEngine] 生成标题失败: {e}，使用原始主题")
            generated_title = topic

        # 第二步：润色文档内容
        polish_prompt = f"""你是一个专业的学术助手，擅长撰写组会周报风格的技术文档。

请将以下研究想法整理成规范的组会周报格式，要求：

## 文档结构（必须包含以下所有章节）

1. **背景与动机**：阐述研究问题的背景和重要性
2. **相关工作**：列出与本研究想法相关的已发表工作，直接使用论文标题作为引用
3. **方法论**：详细描述 proposed 方法
4. **创新点**：明确列出主要贡献
5. **实验设置与 Benchmark**：列出该领域公认的 benchmark 数据集/任务
6. **潜在挑战与解决方案**：列出可能遇到的挑战及应对方案
7. **下一步研究计划**：拟定 3-5 个具体的后续研究步骤

## 格式要求
1. 语言简洁专业，学术性强
2. 使用 Markdown 格式输出
3. **引用格式**：直接使用论文标题作为引用（例如："...根据论文《Deformable Radial Kernel Splatting》的研究..."），在输出最终内容时，将论文名称替换为完整的 arxiv 链接（例如：`[Deformable Radial Kernel Splatting](https://arxiv.org/abs/2412.11752)`）
4. **重要**：如果引用内容中包含图片或表格，可以使用以下标记在适当位置插入：
   - 插入图片: `<!-- INSERT_IMAGE:本地图-1 -->`
   - 插入表格: `<!-- INSERT_TABLE:本地表-1 -->`
   例如: `如图 <!-- INSERT_IMAGE:本地图-1 --> 所示`，`见表 <!-- INSERT_TABLE:本地表-1 -->`
5. 实验部分，列出该领域常用的 Benchmark，如 GLUE、SuperGLUE、SQuAD、COCO 等公认的评估基准

## 参考资料

{citations_context}
{media_instructions}

## 原始研究想法

{ideas_summary}

请直接输出润色后的完整内容（含引用和图片/表格标记），不要添加额外说明。"""

        try:
            logger.info("[IdeaEngine] 开始润色内容（含引用、相关工作、Benchmark 和研究计划）...")
            response = await provider.text_chat(
                prompt=polish_prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=16384
            )
            polished = self._extract_text_from_response(response)
            logger.info(f"[IdeaEngine] 内容润色完成，长度: {len(polished) if polished else 0}")
            content = polished if polished and polished.strip() else ideas_summary

            return content, extracted_media, generated_title
        except Exception as e:
            logger.error(f"[IdeaEngine] 润色失败: {e}，使用原始格式")
            return ideas_summary, {"images": [], "tables": []}, topic

    def _build_citations_context(self, knowledge: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        构建引用上下文，包含本地检索和网络搜索的结果

        Args:
            knowledge: 知识检索结果

        Returns:
            Tuple[str, Dict]: (格式化的引用上下文字符串, 提取的媒体资源字典)
                媒体资源格式: {
                    "images": [{"index": "本地图-1", "path": str, "base64": str, "caption": str}, ...],
                    "tables": [{"index": "本地表-1", "csv_path": str, "png_path": str, "caption": str, "csv_content": str}, ...]
                }
        """
        if not knowledge:
            return "（无可用引用来源）", {"images": [], "tables": []}

        parts = []
        extracted_media = {"images": [], "tables": []}
        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])

        # 本地检索引用
        local_image_idx = 0
        local_table_idx = 0
        if local_results:
            parts.append("## 本地论文检索引用：\n")
            for i, result in enumerate(local_results[:10], 1):  # 最多10条
                paper = result.get("paper", "Unknown")
                page = result.get("page", "N/A")
                text = result.get("text", "")[:300]
                score = result.get("score", 0.0)
                metadata = result.get("metadata", {})
                file_name = metadata.get("file_name", "")

                # 从 filename 提取 arxiv ID
                arxiv_id = ""
                if file_name:
                    import re
                    match = re.match(r'^(\d{4}\.\d{4,})', file_name)
                    if match:
                        arxiv_id = match.group(1)

                # 检查是否有图片
                image_path = metadata.get("image_path")

                # 检查是否有图片
                image_path = metadata.get("image_path")
                if image_path and os.path.exists(image_path):
                    local_image_idx += 1
                    img_index = f"本地图-{local_image_idx}"
                    img_caption = metadata.get("image_caption", f"图 {local_image_idx}")
                    # 读取图片并转为 base64
                    try:
                        with open(image_path, "rb") as f:
                            img_base64 = base64.b64encode(f.read()).decode("utf-8")
                        extracted_media["images"].append({
                            "index": img_index,
                            "path": image_path,
                            "base64": img_base64,
                            "caption": img_caption,
                            "source_paper": paper,
                            "source_page": page
                        })
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] 读取图片失败 {image_path}: {e}")
                        img_base64 = None

                    # 构建引用（包含论文名称和完整 arxiv 链接，供 LLM 替换）
                    if arxiv_id:
                        ref_str = f"{paper} (https://arxiv.org/abs/{arxiv_id})"
                    else:
                        ref_str = paper
                    parts.append(f"- {ref_str} (页码: {page}, 相关度: {score:.3f}, 图片: {img_index})\n")
                    if img_base64:
                        parts.append(f"  - 图片说明: {img_caption}\n")
                else:
                    # 构建引用（包含论文名称和完整 arxiv 链接，供 LLM 替换）
                    if arxiv_id:
                        ref_str = f"{paper} (https://arxiv.org/abs/{arxiv_id})"
                    else:
                        ref_str = paper
                    parts.append(f"- {ref_str} (页码: {page}, 相关度: {score:.3f})\n")

                # 检查是否有表格
                table_csv_path = metadata.get("table_csv_path")
                table_png_path = metadata.get("table_png_path")
                table_caption = metadata.get("table_caption", "")

                if table_csv_path or table_png_path:
                    local_table_idx += 1
                    tbl_index = f"本地表-{local_table_idx}"
                    csv_content = ""
                    if table_csv_path and os.path.exists(table_csv_path):
                        try:
                            with open(table_csv_path, "r", encoding="utf-8") as f:
                                csv_content = f.read()[:500]  # 限制内容长度
                            extracted_media["tables"].append({
                                "index": tbl_index,
                                "csv_path": table_csv_path,
                                "png_path": table_png_path,
                                "caption": table_caption,
                                "csv_content": csv_content,
                                "source_paper": paper,
                                "source_page": page
                            })
                        except Exception as e:
                            logger.warning(f"[IdeaEngine] 读取表格失败 {table_csv_path}: {e}")
                            csv_content = "(无法读取)"

                    if not any(t["index"] == tbl_index for t in extracted_media["tables"]):
                        extracted_media["tables"].append({
                            "index": tbl_index,
                            "csv_path": table_csv_path or "",
                            "png_path": table_png_path or "",
                            "caption": table_caption,
                            "csv_content": csv_content,
                            "source_paper": paper,
                            "source_page": page
                        })
                    parts.append(f"    └─ 包含表格: {tbl_index} - {table_caption}\n")

                parts.append(f"    摘要: {text}...\n\n")

        # 网络搜索引用（直接使用 Markdown 链接格式）
        if web_results:
            parts.append(f"## 网络搜索引用：\n")
            for i, result in enumerate(web_results[:10], 1):  # 最多10条
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                snippet = result.get("snippet", "")[:300]
                # 直接使用 Markdown 链接格式
                link_str = f"[{title}]({url})" if url else title
                parts.append(f"- {link_str}\n")
                parts.append(f"  - 摘要: {snippet}...\n\n")

        if not parts:
            return "（无可用引用来源）", {"images": [], "tables": []}

        return "\n".join(parts), extracted_media

    def _create_media_blocks(self, extracted_media: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        从提取的媒体资源创建飞书块（图片/表格）

        注意：图片需要两阶段处理 - 先创建空图片块，再上传绑定
        表格需要使用 create_feishu_table 工具

        Args:
            extracted_media: 从本地检索中提取的媒体资源

        Returns:
            Tuple[List[Dict], List[Dict]]: (飞书文本/图片块列表, 待上传图片列表)
                待上传图片格式: [{"block_index": int, "path": str, "base64": str, "caption": str}, ...]
        """
        blocks = []
        pending_images = []  # 待上传的图片列表

        images = extracted_media.get("images", [])
        tables = extracted_media.get("tables", [])

        if not images and not tables:
            return blocks, pending_images

        # 添加附录标题
        blocks.append({
            "blockType": "heading",
            "options": {
                "heading": {
                    "level": 2,
                    "content": "附录：参考图片与表格"
                }
            }
        })

        # 添加图片说明文本块
        for img in images:
            caption = f"图: {img.get('caption', '')} (来源: {img.get('source_paper', '')}, 页码: {img.get('source_page', '')})"
            blocks.append({
                "blockType": "text",
                "options": {
                    "text": {
                        "textStyles": [
                            {"text": caption, "style": {"italic": True}}
                        ]
                    }
                }
            })
            # 创建空图片块（后续需要通过 upload_and_bind_image_to_block 上传）
            blocks.append({
                "blockType": "image",
                "options": {
                    "image": {}
                }
            })
            # 记录待上传图片
            if img.get("base64"):
                pending_images.append({
                    "block_index": len(blocks) - 1,  # 图片块在 blocks 中的索引
                    "path": img.get("path", ""),
                    "base64": img.get("base64", ""),
                    "caption": img.get("caption", "")
                })
            logger.info(f"[IdeaEngine] 创建图片块占位符: {img.get('index')}")

        # 表格暂不处理（需要使用 create_feishu_table 工具）
        if tables:
            blocks.append({
                "blockType": "text",
                "options": {
                    "text": {
                        "textStyles": [
                            {"text": f"（包含 {len(tables)} 个表格，需手动查看原文档）", "style": {"italic": True}}
                        ]
                    }
                }
            })
            for tbl in tables:
                logger.info(f"[IdeaEngine] 表格暂不支持: {tbl.get('index')}")

        return blocks, pending_images

    def _remove_media_markers(self, content: str) -> str:
        """移除内容中的媒体标记"""
        content = re.sub(r'<!--\s*INSERT_IMAGE:[^>]+-->\s*', '', content)
        content = re.sub(r'<!--\s*INSERT_TABLE:[^>]+-->\s*', '', content)
        return content

    def _format_ideas_as_markdown(self, ideas: List["ResearchIdea"], topic: str) -> str:
        """将研究想法格式化为 Markdown（备用方案）"""
        output = f"# {topic}\n\n"
        for i, idea in enumerate(ideas, 1):
            output += f"## [{i}] {idea.title}\n\n"
            output += f"** novelty: ** {idea.novelty}\n\n"
            output += f"** methodology: ** {idea.methodology}\n\n"
            if idea.potential_challenges:
                output += f"** challenges: ** {', '.join(idea.potential_challenges)}\n\n"
            output += "---\n\n"
        return output

    async def analyze_topic(self, topic: str, depth: str = "standard") -> TopicAnalysis:
        """
        分析研究主题，生成搜索策略

        Args:
            topic: 研究话题
            depth: 分析深度 (quick/standard/deep)

        Returns:
            TopicAnalysis: 结构化的主题分析
        """
        logger.info(f"[IdeaEngine] 分析主题: {topic}, 深度: {depth}")

        depth_config = {
            "quick": {"num_queries": 3, "num_angles": 2},
            "standard": {"num_queries": 5, "num_angles": 4},
            "deep": {"num_queries": 10, "num_angles": 6}
        }
        config = depth_config.get(depth, depth_config["standard"])

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

        provider = self._get_llm_provider()
        if not provider:
            logger.error("[IdeaEngine] LLM provider未初始化")
            return None

        try:
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.1,
                max_tokens=2048
            )

            response_text = ""
            # 方法1：检查 result_chain（AstrBot 格式）
            if hasattr(response, 'result_chain'):
                chain = getattr(response.result_chain, 'chain', None)
                if chain and len(chain) > 0:
                    first = chain[0]
                    if hasattr(first, 'get_text'):
                        response_text = first.get_text()
                    elif hasattr(first, 'text'):
                        response_text = first.text
            # 方法2：检查 content 属性（LlamaCpp 格式）
            elif hasattr(response, 'content'):
                response_text = response.content
            # 方法3：dict 格式
            elif isinstance(response, dict):
                response_text = response.get("content", "") or response.get("text", "")
            # 方法4：字符串格式
            else:
                response_text = str(response)
            result = self._parse_json_response(response_text)

            if result:
                return TopicAnalysis(
                    domain=result.get("domain", ""),
                    keywords=result.get("keywords", []),
                    search_queries=result.get("search_queries", []),
                    local_rag_queries=result.get("local_rag_queries", []),
                    exploration_angles=result.get("exploration_angles", []),
                    summary=result.get("summary", "")
                )

        except Exception as e:
            logger.error(f"[IdeaEngine] 主题分析失败: {e}")

        return None

    async def search_knowledge(
        self,
        queries: List[str],
        local_rag_top_k: int = 5,
        web_top_k: int = 10
    ) -> Dict[str, Any]:
        """
        多源知识检索

        Args:
            queries: 搜索查询列表
            local_rag_top_k: 本地RAG召回数
            web_top_k: 网络搜索召回数

        Returns:
            Dict包含 web_results, local_results, fused_context
        """
        logger.info(f"[IdeaEngine] 检索知识，查询数: {len(queries)}")

        web_results = []
        local_results = []

        # 1. 本地RAG搜索
        if self._rag_engine and local_rag_top_k > 0:
            try:
                for query in queries[:5]:  # 限制查询数
                    result = await self._rag_engine.search(query, mode="retrieve")
                    sources = result.get("sources", [])
                    for src in sources[:local_rag_top_k]:
                        local_results.append({
                            "text": src.get("text", "")[:500],
                            "paper": src.get("metadata", {}).get("file_name", "Unknown"),
                            "page": str(src.get("metadata", {}).get("page", "")),
                            "score": src.get("score", 0.0)
                        })
            except Exception as e:
                logger.error(f"[IdeaEngine] 本地RAG搜索失败: {e}")

        # 2. 网络搜索（通过Bright Data MCP）
        if self._bright_data_available:
            try:
                web_results = await self._search_web(queries, web_top_k)
            except Exception as e:
                logger.error(f"[IdeaEngine] 网络搜索失败: {e}")

        # 3. 知识融合
        fused_context = self._fuse_knowledge(web_results, local_results)

        return {
            "web_results": web_results,
            "local_results": local_results,
            "fused_context": fused_context,
            "stats": {
                "web_count": len(web_results),
                "local_count": len(local_results)
            }
        }

    async def _search_web(self, queries: List[str], top_k: int) -> List[Dict]:
        """通过网络搜索获取信息（通过Bright Data MCP）"""
        results = []

        try:
            # API Token - 从 mcp_server.json 读取
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            try:
                with open(mcp_config_path, "r", encoding="utf-8") as f:
                    mcp_config = json.load(f)
                api_token = mcp_config.get("mcpServers", {}).get("BrightData", {}).get("env", {}).get("API_TOKEN", "")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"无法从 {mcp_config_path} 读取 BrightData API Token: {e}")

            if not api_token:
                raise ValueError("BrightData API Token 未配置")

            # 启动Bright Data MCP服务器
            env = {**os.environ, "API_TOKEN": api_token}

            proc = await asyncio.create_subprocess_exec(
                "npx", "@brightdata/mcp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            for query in queries[:5]:
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "search_engine",
                        "arguments": {
                            "query": query,
                            "num_results": top_k,
                            "source": "web"
                        }
                    }
                }

                # MCP协议要求每行一个JSON-RPC消息
                request_str = json.dumps(rpc_request) + "\n"

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=request_str.encode()),
                        timeout=60
                    )

                    if stdout:
                        response = json.loads(stdout.decode())
                        content = response.get("result", {}).get("content", [])

                        # 解析搜索结果
                        if content and len(content) > 0:
                            text = content[0].get("text", "")
                            if text:
                                try:
                                    data = json.loads(text)
                                    organic = data.get("organic", [])
                                    for item in organic:
                                        results.append({
                                            "title": item.get("title", ""),
                                            "url": item.get("link", ""),
                                            "snippet": item.get("description", "")
                                        })
                                except json.JSONDecodeError:
                                    pass

                except asyncio.TimeoutError:
                    logger.warning(f"[IdeaEngine] 查询超时: {query}")
                    continue

            # 关闭进程
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

        except Exception as e:
            logger.error(f"[IdeaEngine] Bright Data调用失败: {e}")

        return results

    async def generate_ideas(
        self,
        knowledge_context: str,
        research_domain: str = "",
        num_ideas: int = 3,
        idea_focus: str = "all"
    ) -> List[ResearchIdea]:
        """
        基于知识上下文生成研究想法

        Args:
            knowledge_context: 融合后的知识上下文
            research_domain: 研究领域
            num_ideas: 生成想法数量
            idea_focus: 侧重点 (novelty/feasibility/impact/all)

        Returns:
            List[ResearchIdea]: 研究想法列表
        """
        logger.info(f"[IdeaEngine] 生成{num_ideas}个研究想法")

        focus_instruction = {
            "novelty": "特别强调创新性和独特贡献",
            "feasibility": "特别强调技术可行性和实现路径",
            "impact": "特别强调潜在影响力和应用价值",
            "all": "综合考虑创新性、可行性和影响力"
        }.get(idea_focus, "")

        prompt = f"""基于以下收集的知识上下文，生成{num_ideas}个研究想法。

研究领域：{research_domain or "通用研究领域"}

收集的知识：
{knowledge_context[:8000]}

{focus_instruction}

请为每个想法返回以下JSON格式的信息：

{{
    "ideas": [
        {{
            "title": "想法标题",
            "description": "详细描述",
            "novelty": "创新点",
            "methodology": "方法论建议",
            "potential_challenges": ["挑战1", "挑战2"],
            "related_work": ["相关工作1", "相关工作2"],
            "feasibility": 0.8,
            "inspiration_sources": ["灵感来源1", "灵感来源2"]
        }},
        ...
    ],
    "analysis_summary": "对现有工作的分析总结"
}}

请严格按照JSON格式返回，只返回JSON，不要包含其他文字。"""

        provider = self._get_llm_provider()
        if not provider:
            logger.error("[IdeaEngine] LLM provider未初始化")
            return []

        try:
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.7,
                max_tokens=4096
            )

            response_text = ""
            # 方法1：检查 result_chain（AstrBot 格式）
            if hasattr(response, 'result_chain'):
                chain = getattr(response.result_chain, 'chain', None)
                if chain and len(chain) > 0:
                    first = chain[0]
                    if hasattr(first, 'get_text'):
                        response_text = first.get_text()
                    elif hasattr(first, 'text'):
                        response_text = first.text
            # 方法2：检查 content 属性（LlamaCpp 格式）
            elif hasattr(response, 'content'):
                response_text = response.content
            # 方法3：dict 格式
            elif isinstance(response, dict):
                response_text = response.get("content", "") or response.get("text", "")
            # 方法4：字符串格式
            else:
                response_text = str(response)
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

        except Exception as e:
            logger.error(f"[IdeaEngine] 创意生成失败: {e}")

        return []

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """解析LLM返回的JSON响应"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON块
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # 尝试提取JSON对象
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"[IdeaEngine] JSON解析失败: {text[:200]}")
        return None

    def _fuse_knowledge(
        self,
        web_results: List[Dict],
        local_results: List[Dict]
    ) -> str:
        """将多源知识融合为统一上下文"""
        parts = ["# 收集到的相关知识\n"]

        # 网络资源
        if web_results:
            parts.append("## 网络资源\n")
            for i, r in enumerate(web_results[:10], 1):
                parts.append(f"{i}. **{r.get('title', '')}**")
                parts.append(f"   {r.get('snippet', '')}")
                parts.append("")

        # 本地论文
        if local_results:
            parts.append("## 本地论文库\n")
            papers = {}
            for r in local_results:
                paper = r.get("paper", "Unknown")
                if paper not in papers:
                    papers[paper] = []
                papers[paper].append(r)

            for paper, chunks in list(papers.items())[:5]:
                parts.append(f"### {paper}")
                for chunk in chunks[:2]:
                    text = chunk.get("text", "")[:300]
                    if text:
                        parts.append(f"- {text}...")
                parts.append("")

        return "\n".join(parts)

    async def to_feishu_markdown(
        self,
        ideas: List[ResearchIdea],
        topic: str = "",
        include_sources: bool = True
    ) -> str:
        """
        将研究想法格式化为飞书文档兼容的Markdown格式

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            include_sources: 是否包含灵感来源

        Returns:
            str: 飞书兼容的Markdown格式内容
        """
        if not ideas:
            return ""

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

        return "".join(markdown_parts)

    async def create_feishu_document(
        self,
        ideas: List[ResearchIdea],
        topic: str = "",
        folder_token: str = "",
        knowledge: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        创建飞书文档并写入研究想法

        流程：
        1. LLM 生成文档标题（不使用原始问题作为标题）
        2. LLM 润色内容（组会周报学术风格，含引用、相关工作、Benchmark、研究计划）
        3. 通过 CORE API 将本地引用解析为 arxiv 链接
        4. 直接调用 AstrBot 已连接的 feishu MCP 工具
        5. 返回润色内容和文档链接

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            folder_token: 飞书文件夹Token（可选）
            knowledge: 知识检索结果（包含 web_results, local_results）

        Returns:
            Dict包含 document_id, url, polished_content, error 等信息
        """
        try:
            # 0. 预解析本地论文的 arxiv 链接（避免在 prompt 中重复查询）
            knowledge = await self._pre_resolve_arxiv_links(knowledge)

            # 1. LLM 润色内容（组会周报学术风格，含引用、媒体标记、相关工作、Benchmark 和研究计划）
            # 返回值：polished_content, extracted_media, generated_title
            polished_content, extracted_media, generated_title = await self._polish_content_for_feishu(ideas, topic, knowledge)
            if not polished_content:
                return {"error": "内容为空", "polished_content": ""}

            # 2. 备用解析（处理 LLM 偶尔使用错误格式的情况）
            polished_content = await self._resolve_references(polished_content, knowledge)

            # 3. 移除媒体标记，保留纯文本内容用于飞书文档
            cleaned_content = self._remove_media_markers(polished_content)

            # 4. 生成飞书块格式（不含媒体）
            blocks = self._markdown_to_feishu_blocks(cleaned_content)

            # 5. 生成媒体块（图片/表格），附加在文档末尾
            media_blocks, pending_images = self._create_media_blocks(extracted_media)
            blocks.extend(media_blocks)

            # 6. 获取飞书工具
            feishu_tool = self._get_feishu_tool()
            if not feishu_tool:
                return {
                    "error": "未找到飞书 MCP 工具，请确认飞书 MCP 已配置并启用",
                    "polished_content": cleaned_content  # 同时返回清理后的内容供用户审阅
                }

            # 7. 直接调用 feishu 工具创建文档
            try:
                # 创建 ContextWrapper
                ctx_wrapper = ContextWrapper(context=self.context)

                # 根据是否提供 folder_token 选择模式
                if not folder_token:
                    logger.error("[IdeaEngine] 未提供 folder_token，无法创建文档")
                    return {
                        "error": "创建飞书文档需要提供 folder_token（飞书文件夹链接中的 token）\n\n"
                                 "获取方式：\n"
                                 "1. 打开飞书文档所在的文件夹\n"
                                 "2. 点击文件夹右上角的「···」\n"
                                 "3. 选择「复制链接」\n"
                                 "4. 链接格式: https://xxx.feishu.cn/drive/folder/xxxxx\n"
                                 "5. 链接最后一部分就是 folder_token（如 FWK2fMleClICfodlHHWc4Mygnhb）\n"
                                 "6. 使用方式: /idea tofeishu <主题> <folder_token>\n\n"
                                 "例如: /idea tofeishu 我的研究想法 FWK2fMleClICfodlHHWc4Mygnhb",
                        "polished_content": cleaned_content
                    }

                # 使用 LLM 生成的标题创建文档，而非使用原始 topic
                logger.info(f"[IdeaEngine] 使用 LLM 生成的标题创建文档: {generated_title}")
                logger.info(f"[IdeaEngine] folder_token: {folder_token}")
                create_result = await feishu_tool.call(
                    ctx_wrapper,
                    title=generated_title,
                    folderToken=folder_token
                )
                logger.info(f"[IdeaEngine] feishu 创建文档结果类型: {type(create_result)}")
                logger.info(f"[IdeaEngine] feishu 创建文档结果: {repr(create_result)[:500]}")

                # 解析结果
                result_text = ""
                doc_info = {}

                # 检查是否是 MCP CallToolResult 格式
                if hasattr(create_result, 'content') and create_result.content:
                    try:
                        first_content = create_result.content[0]
                        result_text = getattr(first_content, 'text', None) or str(first_content)
                        logger.info(f"[IdeaEngine] result_text: {result_text[:500]}")
                        if result_text and result_text.strip():
                            doc_info = json.loads(result_text)
                        else:
                            logger.warning("[IdeaEngine] result_text 为空")
                    except json.JSONDecodeError as je:
                        # JSON 解析失败，检查 result_text 是否包含实际错误信息
                        if result_text and ("失败" in result_text or "错误" in result_text or "error" in result_text.lower()):
                            logger.error(f"[IdeaEngine] 飞书返回错误: {result_text[:500]}")
                            return {
                                "error": result_text.strip(),
                                "polished_content": polished_content
                            }
                        logger.error(f"[IdeaEngine] JSON 解析失败: {je}, result_text: {result_text[:200]}")
                        return {
                            "error": f"解析结果失败: {je}",
                            "polished_content": polished_content
                        }
                elif hasattr(create_result, 'isError') and create_result.isError:
                    # MCP 返回错误
                    error_text = str(create_result)
                    logger.error(f"[IdeaEngine] feishu 工具返回错误: {error_text}")
                    return {
                        "error": f"飞书工具错误: {error_text}",
                        "polished_content": polished_content
                    }
                else:
                    # 其他格式
                    result_text = str(create_result)
                    logger.warning(f"[IdeaEngine] 未知结果格式: {result_text[:200]}")
                    try:
                        doc_info = json.loads(result_text) if result_text.strip() else {}
                    except json.JSONDecodeError:
                        doc_info = {}

                document_id = (
                    doc_info.get("document", {}).get("document_id")
                    or doc_info.get("document_id")
                    or doc_info.get("objToken")
                    or doc_info.get("obj_token")
                )
                if not document_id:
                    return {
                        "error": f"文档创建失败: {result_text or str(create_result)}",
                        "polished_content": polished_content
                    }

                logger.info(f"[IdeaEngine] 文档创建成功: {document_id}")

            except Exception as e:
                logger.error(f"[IdeaEngine] 调用 feishu 工具失败: {e}")
                import traceback
                logger.error(f"[IdeaEngine] 详细错误: {traceback.format_exc()}")
                return {
                    "error": f"飞书工具调用失败: {e}",
                    "polished_content": polished_content
                }

            # 8. 获取文档的根块 ID
            root_block_id = "0"
            try:
                get_blocks_tool = None
                provider_manager = getattr(self.context, 'provider_manager', None)
                if provider_manager:
                    llm_tools = getattr(provider_manager, 'llm_tools', None)
                    if llm_tools:
                        func_list = getattr(llm_tools, 'func_list', [])
                        for tool in func_list:
                            if tool.name == 'get_feishu_document_blocks':
                                get_blocks_tool = tool
                                break

                if get_blocks_tool:
                    blocks_info_result = await get_blocks_tool.call(
                        ctx_wrapper,
                        documentId=document_id
                    )
                    logger.info(f"[IdeaEngine] 获取块信息结果: {repr(blocks_info_result)[:500]}")

                    # 解析获取块信息结果，找到根块 ID
                    if hasattr(blocks_info_result, 'content') and blocks_info_result.content:
                        result_text = getattr(blocks_info_result.content[0], 'text', None)
                        if result_text:
                            try:
                                blocks_data = json.loads(result_text)
                                logger.info(f"[IdeaEngine] 块数据解析成功: {str(blocks_data)[:500]}")
                                # 检查是否是列表格式
                                if isinstance(blocks_data, list):
                                    if len(blocks_data) > 0:
                                        first_item = blocks_data[0]
                                        root_block_id = first_item.get('block_id') if isinstance(first_item, dict) else None
                                        logger.info(f"[IdeaEngine] 从列表获取根块 ID: {root_block_id}")
                                elif isinstance(blocks_data, dict):
                                    items = blocks_data.get('data', {}).get('items', []) or blocks_data.get('items', [])
                                    if items and len(items) > 0:
                                        root_block_id = items[0].get('block_id', '0')
                                        logger.info(f"[IdeaEngine] 从字典获取根块 ID: {root_block_id}")
                            except json.JSONDecodeError as e:
                                logger.warning(f"[IdeaEngine] 解析块数据失败: {e}")

                if not root_block_id:
                    root_block_id = "0"
                logger.info(f"[IdeaEngine] 最终使用的根块 ID: {root_block_id}")

            except Exception as e:
                logger.warning(f"[IdeaEngine] 获取根块 ID 失败: {e}，使用默认值 0")

            # 9. 添加内容块（分阶段处理，参照 test_feishu）
            logger.info(f"[IdeaEngine] 准备添加内容块")
            blocks_created = 0
            images_uploaded = 0
            tables_created = 0

            try:
                # 获取工具
                provider_manager = getattr(self.context, 'provider_manager', None)
                add_blocks_tool = None
                upload_image_tool = None
                create_table_tool = None

                if provider_manager:
                    llm_tools = getattr(provider_manager, 'llm_tools', None)
                    if llm_tools:
                        func_list = getattr(llm_tools, 'func_list', [])
                        for tool in func_list:
                            if tool.name == 'batch_create_feishu_blocks':
                                add_blocks_tool = tool
                            elif tool.name == 'upload_and_bind_image_to_block':
                                upload_image_tool = tool
                            elif tool.name == 'create_feishu_table':
                                create_table_tool = tool

                logger.info(f"[IdeaEngine] 工具: add_blocks={add_blocks_tool is not None}, "
                           f"upload_image={upload_image_tool is not None}, create_table={create_table_tool is not None}")

                # 9.1 添加文本块（不含图片的纯文本内容）
                if add_blocks_tool and blocks:
                    logger.info(f"[IdeaEngine] 添加文本块，数量: {len(blocks)}")
                    blocks_result = await add_blocks_tool.call(
                        ctx_wrapper,
                        documentId=document_id,
                        parentBlockId=root_block_id,
                        index=0,
                        blocks=blocks
                    )
                    logger.info(f"[IdeaEngine] 添加文本块结果: {repr(blocks_result)[:500]}")

                    # 检查错误
                    if hasattr(blocks_result, 'isError') and blocks_result.isError:
                        error_text = ""
                        if hasattr(blocks_result, 'content') and blocks_result.content:
                            error_text = getattr(blocks_result.content[0], 'text', str(blocks_result))
                        logger.error(f"[IdeaEngine] 添加文本块失败: {error_text[:200] if error_text else str(blocks_result)[:200]}")
                    else:
                        # 解析成功添加的块数
                        if hasattr(blocks_result, 'content') and blocks_result.content:
                            result_text = getattr(blocks_result.content[0], 'text', None)
                            if result_text:
                                try:
                                    result_data = json.loads(result_text)
                                    blocks_created = result_data.get("totalBlocksCreated", 0)
                                    logger.info(f"[IdeaEngine] 成功添加 {blocks_created} 个文本块")
                                except json.JSONDecodeError:
                                    logger.warning(f"[IdeaEngine] 解析文本块结果失败")

                # 9.2 处理图片（两阶段：创建空图片块 → 上传绑定实际图片）
                images = extracted_media.get("images", [])
                if images and upload_image_tool:
                    logger.info(f"[IdeaEngine] 开始处理 {len(images)} 张图片")
                    current_index = len(blocks)  # 从文本块之后开始

                    for i, img in enumerate(images):
                        img_path = img.get("path", "")
                        if not img_path or not os.path.exists(img_path):
                            logger.warning(f"[IdeaEngine] 图片不存在: {img_path}")
                            continue

                        # 创建空图片块
                        image_blocks = [
                            {
                                "blockType": "image",
                                "options": {
                                    "image": {}
                                }
                            }
                        ]

                        img_result = await add_blocks_tool.call(
                            ctx_wrapper,
                            documentId=document_id,
                            parentBlockId=root_block_id,
                            index=current_index,
                            blocks=image_blocks
                        )
                        logger.info(f"[IdeaEngine] 创建图片块[{i}]结果: {repr(img_result)[:500]}")

                        # 检查错误
                        if hasattr(img_result, 'isError') and img_result.isError:
                            logger.error(f"[IdeaEngine] 创建图片块[{i}]失败")
                            continue

                        # 从结果中提取图片块 ID
                        image_block_id = None
                        try:
                            if hasattr(img_result, 'content') and img_result.content:
                                result_text = getattr(img_result.content[0], 'text', None)
                                if result_text:
                                    result_data = json.loads(result_text)
                                    image_info = result_data.get('imageBlocksInfo', {})
                                    if image_info:
                                        block_ids = image_info.get('blockIds', [])
                                        if block_ids:
                                            image_block_id = block_ids[0]
                                            logger.info(f"[IdeaEngine] 图片块[{i}] ID: {image_block_id}")
                        except Exception as e:
                            logger.error(f"[IdeaEngine] 解析图片块ID失败: {e}")

                        # 上传并绑定实际图片
                        if image_block_id:
                            with open(img_path, 'rb') as f:
                                img_base64 = base64.b64encode(f.read()).decode('utf-8')

                            upload_result = await upload_image_tool.call(
                                ctx_wrapper,
                                documentId=document_id,
                                images=[{
                                    "blockId": image_block_id,
                                    "imagePathOrUrl": img_path
                                }]
                            )
                            logger.info(f"[IdeaEngine] 上传图片[{i}]结果: {repr(upload_result)[:500]}")

                            if hasattr(upload_result, 'isError') and upload_result.isError:
                                logger.error(f"[IdeaEngine] 上传图片[{i}]失败")
                            else:
                                images_uploaded += 1
                                logger.info(f"[IdeaEngine] 图片[{i}]上传成功")

                        current_index += 1

                # 9.3 处理表格
                tables = extracted_media.get("tables", [])
                if tables and create_table_tool:
                    logger.info(f"[IdeaEngine] 开始处理 {len(tables)} 个表格")
                    current_index = len(blocks) + len(images)  # 从文本块和图片块之后开始

                    for i, tbl in enumerate(tables):
                        csv_path = tbl.get("csv_path", "")
                        caption = tbl.get("caption", f"表格 {i+1}")

                        # 读取 CSV 内容
                        cells_data = []
                        if csv_path and os.path.exists(csv_path):
                            try:
                                with open(csv_path, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()

                                # 解析 CSV 构建表格
                                for row_idx, line in enumerate(lines[:10]):  # 最多10行
                                    cols = line.strip().split(',')
                                    for col_idx, cell_text in enumerate(cols[:10]):  # 最多10列
                                        cells_data.append({
                                            "coordinate": {"row": row_idx, "column": col_idx},
                                            "content": {
                                                "blockType": "text",
                                                "options": {
                                                    "text": {
                                                        "textStyles": [
                                                            {"text": cell_text.strip(), "style": {"bold": row_idx == 0}}
                                                        ]
                                                    }
                                                }
                                            }
                                        })
                            except Exception as e:
                                logger.error(f"[IdeaEngine] 读取CSV失败: {e}")

                        if cells_data:
                            table_config = {
                                "columnSize": max([c["coordinate"]["column"] for c in cells_data]) + 1,
                                "rowSize": max([c["coordinate"]["row"] for c in cells_data]) + 1,
                                "cells": cells_data
                            }

                            table_result = await create_table_tool.call(
                                ctx_wrapper,
                                documentId=document_id,
                                parentBlockId=root_block_id,
                                index=current_index,
                                tableConfig=table_config
                            )
                            logger.info(f"[IdeaEngine] 创建表格[{i}]结果: {repr(table_result)[:500]}")

                            if hasattr(table_result, 'isError') and table_result.isError:
                                logger.error(f"[IdeaEngine] 创建表格[{i}]失败")
                            else:
                                tables_created += 1
                                logger.info(f"[IdeaEngine] 表格[{i}]创建成功")

                            current_index += 1

                logger.info(f"[IdeaEngine] 块添加完成: 文本块={blocks_created}, 图片={images_uploaded}, 表格={tables_created}")

            except Exception as e:
                logger.error(f"[IdeaEngine] 添加内容块失败: {e}")
                import traceback
                logger.error(f"[IdeaEngine] 详细错误: {traceback.format_exc()}")
                # 文档已创建，块添加失败不影响最终结果
                blocks_created = 0

            return {
                "document_id": document_id,
                "url": f"https://feishu.cn/docx/{document_id}",
                "title": generated_title,  # 使用 LLM 生成的标题
                "blocks_created": blocks_created,
                "images_uploaded": images_uploaded,
                "tables_created": tables_created,
                "polished_content": cleaned_content,  # 返回清理后的内容（无媒体标记）供用户审阅
                "media_count": {
                    "images": len(extracted_media.get("images", [])),
                    "tables": len(extracted_media.get("tables", []))
                }
            }

        except Exception as e:
            logger.error(f"[IdeaEngine] 创建飞书文档失败: {e}")
            import traceback
            logger.error(f"[IdeaEngine] 详细错误: {traceback.format_exc()}")
            return {"error": str(e)}

    async def _resolve_references(
        self,
        content: str,
        knowledge: Dict[str, Any] = None
    ) -> str:
        """
        将本地论文引用解析为 arxiv 链接，网络引用解析为带链接的格式

        Args:
            content: 包含引用标记的文档内容
            knowledge: 知识检索结果

        Returns:
            str: 将引用标记替换为带链接的格式
        """
        if not knowledge:
            return content

        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])

        if not local_results and not web_results:
            return content

        # 1. 处理本地引用（通过 CORE API 获取 arxiv 链接）
        paper_to_arxiv = {}
        paper_to_github = {}

        if local_results:
            core_api_key = self._get_core_api_key()
            if core_api_key:
                core_client = CoreAPIClient(core_api_key)
                for result in local_results[:10]:  # 最多处理10条
                    paper = result.get("paper", "")
                    if not paper or paper in paper_to_arxiv:
                        continue

                    arxiv_url, github_url = await core_client.get_arxiv_link(paper)
                    if arxiv_url:
                        paper_to_arxiv[paper] = arxiv_url
                        paper_to_github[paper] = github_url
                        logger.info(f"[IdeaEngine] 解析论文到 arxiv: {paper[:50]} -> {arxiv_url}")
            else:
                logger.warning("[IdeaEngine] CORE API Key 未配置，本地引用将只显示论文标题")

        # 2. 处理网络引用（直接使用 URL）
        web_refs = {}
        for i, result in enumerate(web_results[:10], 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            if url:
                web_refs[i] = (title, url)

        # 3. 解析并替换引用
        # 3.1 先处理连续的本地引用（如 [本地-1][本地-2]）
        def replace_local_refs(match):
            """处理连续的本地引用"""
            refs = match.group(0)  # 整个匹配的字符串，如 "[本地-1][本地-2]"
            indices = re.findall(r'\[本地-(\d+)\]', refs)

            resolved = []
            for idx_str in indices:
                idx = int(idx_str)
                if idx <= len(local_results):
                    paper = local_results[idx - 1].get("paper", "")
                    if paper in paper_to_arxiv:
                        arxiv = paper_to_arxiv[paper]
                        github = paper_to_github.get(paper)
                        if github:
                            resolved.append(f"[{paper}]({arxiv}) - [GitHub]({github})")
                        else:
                            resolved.append(f"[{paper}]({arxiv})")
                    elif paper:
                        # 没有 arxiv 链接，只显示论文标题
                        resolved.append(paper)
                    else:
                        resolved.append(f"[本地-{idx}]")
                else:
                    resolved.append(f"[本地-{idx}]")

            if len(resolved) == 1:
                return resolved[0]
            elif len(resolved) == 2:
                return f"{resolved[0]} 和 {resolved[1]}"
            else:
                return "、".join(resolved[:-1]) + f" 和 {resolved[-1]}"

        # 3.2 处理连续的网络引用（如 [网络-1][网络-2][网络-3]）
        def replace_web_refs(match):
            """处理连续的网络引用"""
            refs = match.group(0)  # 整个匹配的字符串
            indices = re.findall(r'\[网络-(\d+)\]', refs)

            resolved = []
            for idx_str in indices:
                idx = int(idx_str)
                if idx in web_refs:
                    title, url = web_refs[idx]
                    resolved.append(f"[{title}]({url})")
                else:
                    resolved.append(f"[网络-{idx}]")

            if len(resolved) == 1:
                return resolved[0]
            elif len(resolved) == 2:
                return f"{resolved[0]} 和 {resolved[1]}"
            else:
                return "、".join(resolved[:-1]) + f" 和 {resolved[-1]}"

        # 3.3 处理单个本地引用（不在连续引用中的）
        def replace_single_local_ref(match):
            idx = int(match.group(1))
            if idx <= len(local_results):
                paper = local_results[idx - 1].get("paper", "")
                if paper in paper_to_arxiv:
                    arxiv = paper_to_arxiv[paper]
                    github = paper_to_github.get(paper)
                    if github:
                        return f"[{paper}]({arxiv}) - [GitHub]({github})"
                    else:
                        return f"[{paper}]({arxiv})"
                elif paper:
                    return paper
            return match.group(0)

        # 3.4 处理单个网络引用（不在连续引用中的）
        def replace_single_web_ref(match):
            idx = int(match.group(1))
            if idx in web_refs:
                title, url = web_refs[idx]
                return f"[{title}]({url})"
            return match.group(0)

        # 执行替换（按优先级：连续引用 → 单个引用）
        # 注意：正则匹配时会优先匹配更长的模式

        # 先处理连续引用（多个连续的 [本地-N] 或 [网络-N]）
        content = re.sub(r'(\[本地-\d+\])+', replace_local_refs, content)
        content = re.sub(r'(\[网络-\d+\])+', replace_web_refs, content)

        # 再处理单个引用
        content = re.sub(r'\[本地-(\d+)\]', replace_single_local_ref, content)
        content = re.sub(r'\[网络-(\d+)\]', replace_single_web_ref, content)

        return content

    def _get_core_api_key(self) -> Optional[str]:
        """从配置文件获取 CORE API Key"""
        try:
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)
            # 尝试从多个位置获取 CORE API key
            return (
                mcp_config.get("mcpServers", {}).get("CORE", {}).get("env", {}).get("API_TOKEN") or
                mcp_config.get("core_api_key")
            )
        except Exception as e:
            logger.error(f"[IdeaEngine] 读取 CORE API 配置失败: {e}")
            return None

    def _markdown_to_feishu_blocks(self, markdown_text: str) -> List[Dict]:
        """将Markdown文本转换为飞书块格式"""
        blocks = []
        lines = markdown_text.split("\n")

        for line in lines:
            line = line.rstrip()

            # 一级标题 # xxx
            if line.startswith("# ") and not line.startswith("## "):
                content = self._strip_markdown_style(line[2:].strip())
                blocks.append({
                    "blockType": "heading",
                    "options": {
                        "heading": {
                            "level": 1,
                            "content": content
                        }
                    }
                })
            # 二级标题 ## xxx
            elif line.startswith("## ") and not line.startswith("### "):
                content = self._strip_markdown_style(line[3:].strip())
                blocks.append({
                    "blockType": "heading",
                    "options": {
                        "heading": {
                            "level": 2,
                            "content": content
                        }
                    }
                })
            # 三级标题 ### xxx
            elif line.startswith("### "):
                content = self._strip_markdown_style(line[4:].strip())
                blocks.append({
                    "blockType": "heading",
                    "options": {
                        "heading": {
                            "level": 3,
                            "content": content
                        }
                    }
                })
            # 分割线 ---
            elif line.startswith("---"):
                blocks.append({
                    "blockType": "text",
                    "options": {
                        "text": {
                            "textStyles": [
                                {"text": "─────────────────────────────────", "style": {}}
                            ]
                        }
                    }
                })
            # 无序列表 - xxx 或 * xxx
            # 飞书 API 支持 markdown 渲染，保留原始内容
            elif line.startswith("- ") or line.startswith("* "):
                content = line[2:].strip()
                blocks.append({
                    "blockType": "list",
                    "options": {
                        "list": {
                            "content": content,
                            "isOrdered": False
                        }
                    }
                })
            # 有序列表 1. xxx 或 1) xxx
            # 飞书 API 支持 markdown 渲染，保留原始内容
            elif re.match(r'^\d+[\.\)]\s', line):
                match = re.match(r'^(\d+[\.\)])\s+(.*)$', line)
                if match:
                    content = match.group(2).strip()
                    blocks.append({
                        "blockType": "list",
                        "options": {
                            "list": {
                                "content": content,
                                "isOrdered": True
                            }
                        }
                    })
            # 空行
            elif line.strip() == "":
                pass
            # 普通文本（使用 textStyles 处理行内样式）
            else:
                text_content = line.strip()
                if text_content:
                    blocks.append({
                        "blockType": "text",
                        "options": {
                            "text": {
                                "textStyles": self._parse_inline_styles(text_content)
                            }
                        }
                    })

        return blocks

    def _strip_markdown_style(self, text: str) -> str:
        """移除 Markdown 样式标记，保留纯文本"""
        # 按优先级匹配：先处理长标记，再处理短标记
        # 移除 ***加粗斜体*** → 加粗斜体
        text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)
        # 移除 **加粗** → 加粗
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        # 移除 *斜体* → 斜体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        return text

    def _parse_inline_styles(self, text: str) -> List[Dict[str, Any]]:
        """
        解析行内 Markdown 样式（加粗、斜体），返回飞书 textStyles 格式

        支持：
        - **加粗** → bold: true
        - *斜体* → italic: true
        - ***加粗斜体*** → bold: true, italic: true
        """
        if not text:
            return [{"text": "", "style": {}}]

        styles = []
        # 匹配顺序：***加粗斜体*** > **加粗** > *斜体*
        pattern = r'(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*)'

        last_end = 0
        for match in re.finditer(pattern, text):
            # 添加匹配之前的普通文本
            if match.start() > last_end:
                plain_text = text[last_end:match.start()]
                if plain_text:
                    styles.append({"text": plain_text, "style": {}})

            matched_text = match.group(0)
            inner_text = matched_text[2:-2]  # 去掉前后标记

            if matched_text.startswith('***') and matched_text.endswith('***'):
                # 加粗斜体
                styles.append({"text": inner_text, "style": {"bold": True, "italic": True}})
            elif matched_text.startswith('**') and matched_text.endswith('**'):
                # 加粗
                styles.append({"text": inner_text, "style": {"bold": True}})
            elif matched_text.startswith('*') and matched_text.endswith('*'):
                # 斜体
                styles.append({"text": inner_text, "style": {"italic": True}})

            last_end = match.end()

        # 添加最后剩余的文本
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining:
                styles.append({"text": remaining, "style": {}})

        # 如果没有匹配到任何样式，返回原始文本
        if not styles:
            return [{"text": text, "style": {}}]

        return styles

    async def _call_feishu_mcp_create_doc(
        self,
        title: str,
        folder_token: str = ""
    ) -> Dict[str, Any]:
        """调用 feishu-mcp 创建文档"""
        try:
            # 从 mcp_server.json 读取配置
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            logger.info(f"[IdeaEngine] 读取 MCP 配置: {mcp_config_path}")
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)

            feishu_config = mcp_config.get("mcpServers", {}).get("feishu", {})
            env_vars = feishu_config.get("env", {})
            logger.info(f"[IdeaEngine] feishu env keys: {list(env_vars.keys())}")

            # 构建 MCP 请求
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "create_feishu_document",
                    "arguments": {
                        "title": title,
                        "folderToken": folder_token
                    }
                }
            }
            logger.info(f"[IdeaEngine] MCP 请求: {json.dumps(mcp_request)[:200]}")

            # 调用 feishu-mcp
            logger.info("[IdeaEngine] 启动 feishu-mcp 进程...")
            proc = await asyncio.create_subprocess_exec(
                "npx", "-y", "feishu-mcp@latest", "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **env_vars}
            )
            logger.info(f"[IdeaEngine] feishu-mcp PID: {proc.pid}")

            # 发送请求
            request_data = json.dumps(mcp_request).encode()
            logger.info(f"[IdeaEngine] 发送请求 ({len(request_data)} bytes)...")
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=request_data),
                timeout=120
            )

            logger.info(f"[IdeaEngine] feishu-mcp 返回: stdout={len(stdout)} bytes, stderr={len(stderr)} bytes")

            if stderr:
                stderr_text = stderr.decode(errors="replace")
                logger.warning(f"[IdeaEngine] feishu-mcp stderr: {stderr_text[:500]}")

            if stdout:
                stdout_text = stdout.decode(errors="replace")
                logger.info(f"[IdeaEngine] feishu-mcp stdout: {stdout_text[:1000]}")
                try:
                    response = json.loads(stdout_text)
                    logger.info(f"[IdeaEngine] 解析响应成功: {str(response)[:200]}")
                    result = response.get("result", {}).get("content", [{}])[0].get("text", "{}")
                    logger.info(f"[IdeaEngine] feishu-mcp result: {result[:500]}")
                    doc_info = json.loads(result)
                    document_id = doc_info.get("document_id") or doc_info.get("objToken") or doc_info.get("obj_token")
                    if document_id:
                        logger.info(f"[IdeaEngine] 文档创建成功: {document_id}")
                        return {"document_id": document_id}
                    return {"error": f"创建文档失败: {result}"}
                except json.JSONDecodeError as e:
                    logger.error(f"[IdeaEngine] JSON解析失败: {e}, stdout: {stdout_text[:500]}")
                    return {"error": f"JSON解析失败: {e}"}

            # 无 stdout 时打印更多信息
            logger.error(f"[IdeaEngine] feishu-mcp 无 stdout 输出, proc.returncode={proc.returncode}")
            if stderr:
                logger.error(f"[IdeaEngine] stderr: {stderr.decode(errors='replace')[:500]}")
            return {"error": f"feishu-mcp 无响应 (returncode={proc.returncode})"}

        except asyncio.TimeoutError:
            logger.error("[IdeaEngine] feishu-mcp 调用超时")
            return {"error": "feishu-mcp 调用超时"}
        except Exception as e:
            import traceback
            logger.error(f"[IdeaEngine] 调用 feishu-mcp 失败: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

    async def _call_feishu_mcp_add_blocks(
        self,
        document_id: str,
        blocks: List[Dict]
    ) -> Dict[str, Any]:
        """调用 feishu-mcp 添加内容块"""
        try:
            # 从 mcp_server.json 读取配置
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)

            feishu_config = mcp_config.get("mcpServers", {}).get("feishu", {})
            env_vars = feishu_config.get("env", {})

            # 构建 MCP 请求
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "batch_create_feishu_blocks",
                    "arguments": {
                        "documentId": document_id,
                        "parentBlockId": "0",
                        "index": 0,
                        "blocks": blocks
                    }
                }
            }

            # 调用 feishu-mcp
            proc = await asyncio.create_subprocess_exec(
                "npx", "-y", "feishu-mcp@latest", "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **env_vars}
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps(mcp_request).encode()),
                timeout=120
            )

            if stderr:
                stderr_text = stderr.decode()
                logger.warning(f"[IdeaEngine] feishu-mcp blocks stderr: {stderr_text}")

            if stdout:
                response = json.loads(stdout.decode())
                result = response.get("result", {}).get("content", [{}])[0].get("text", "{}")
                result_data = json.loads(result)
                if result_data.get("totalBlocksCreated", 0) > 0:
                    return {"success": True, "blocks_created": result_data.get("totalBlocksCreated")}
                return {"error": result_data}

            return {"error": "feishu-mcp 无响应"}

        except json.JSONDecodeError as e:
            logger.error(f"[IdeaEngine] 解析 feishu-mcp 响应失败: {e}")
            return {"error": f"解析响应失败: {e}"}
        except asyncio.TimeoutError:
            return {"error": "feishu-mcp 调用超时"}
        except Exception as e:
            logger.error(f"[IdeaEngine] 调用 feishu-mcp 失败: {e}")
            return {"error": str(e)}
