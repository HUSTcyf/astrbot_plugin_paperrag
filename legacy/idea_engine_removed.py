"""
研究创意生成引擎 - 已移除的代码

本文件包含从简化版 idea_engine.py 中移除的所有方法。
保留作为参考，以便将来需要时恢复功能。
"""

import hashlib
import json
import re
import mistune
import asyncio
import os
import base64
from typing import Dict, Any, List, Optional, Tuple, cast
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from astrbot.api import logger
from astrbot.core.agent.message import AssistantMessageSegment, ToolCallMessageSegment
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.provider.entities import ToolCallsResult
from astrbot.core.astr_main_agent_resources import TOOL_CALL_PROMPT


# ==================== 已移除的方法 ====================

def _check_bright_data_config(self) -> bool:
    """检查 Bright Data MCP 是否已配置"""
    try:
        mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
        if not mcp_config_path.exists():
            logger.warning("[IdeaEngine] mcp_server.json 不存在，Bright Data 搜索将不可用")
            return False
        with open(mcp_config_path, "r", encoding="utf-8") as f:
            mcp_config = json.load(f)
        api_token = mcp_config.get("mcpServers", {}).get("BrightData", {}).get("env", {}).get("API_TOKEN", "")
        if not api_token:
            logger.warning("[IdeaEngine] Bright Data API Token 未配置，网络搜索将不可用")
            return False
        return True
    except Exception as e:
        logger.warning(f"[IdeaEngine] 检查 Bright Data 配置失败: {e}")
        return False


def delete_ideas_by_uuids(self, uuids: List[str]) -> Tuple[List[str], Optional[str]]:
    """
    根据 UUID 列表删除想法文件

    通过扫描 topic_index 定位 UUID 所在文件夹

    Args:
        uuids: UUID 列表

    Returns:
        Tuple[List[已删除的UUID], 所属topic]
    """
    ideas_dir = self._get_ideas_dir()
    index = self._get_topic_index()
    deleted = []
    found_topic = None

    for folder_name, topic in index.items():
        folder = ideas_dir / folder_name
        if not folder.exists():
            continue
        for uid in uuids:
            file_path = folder / f"{uid}.json"
            if file_path.exists():
                file_path.unlink()
                deleted.append(uid)
                if found_topic is None:
                    found_topic = topic
                logger.info(f"[IdeaEngine] 已删除想法: {file_path}")

    return deleted, found_topic


async def add_ideas_to_topic(
    self,
    topic: str,
    num_ideas: int = 3,
    idea_focus: str = "all"
) -> Tuple[List["ResearchIdea"], Dict[str, Any]]:
    """
    为已有 topic 追加新想法（复用现有 context）

    Args:
        topic: 已有 topic
        num_ideas: 追加想法数量
        idea_focus: 想法聚焦方向

    Returns:
        Tuple[新生成的想法列表, 现有knowledge dict]
    """
    # 加载现有 context
    context_data = self._load_context(topic)
    if not context_data:
        raise ValueError(f"Topic '{topic}' 不存在，请先运行 /idea {topic}")

    # 重建 knowledge dict（格式需与 search_knowledge 一致）
    knowledge = {
        "local_results": context_data.get("local_results", []),
        "web_results": context_data.get("web_results", []),
        "fused_context": self._fuse_knowledge_context(
            context_data.get("local_results", []),
            context_data.get("web_results", [])
        )
    }

    # 生成新想法
    ideas = await self.generate_ideas(
        knowledge_context=knowledge.get("fused_context", ""),
        research_domain=context_data.get("domain", ""),
        num_ideas=num_ideas,
        idea_focus=idea_focus
    )

    # 保存新想法（追加到 topic 文件夹）
    self._save_ideas_append(ideas, topic, knowledge)

    return ideas, knowledge


def _save_ideas_append(
    self,
    ideas: List["ResearchIdea"],
    topic: str,
    knowledge: Dict[str, Any]
) -> List[Tuple[str, Path]]:
    """追加保存想法到已有 topic 文件夹（不覆盖已有想法）"""
    import uuid as uuid_module

    folder = self._topic_folder(topic)
    folder.mkdir(parents=True, exist_ok=True)

    # 更新 topic 索引（folder_name → topic）
    index = self._get_topic_index()
    index[folder.name] = topic
    self._save_topic_index(index)

    results = []
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
        results.append((idea_uuid, file_path))
        logger.info(f"[IdeaEngine] 追加想法已保存: {file_path}")

    return results


async def regenerate_all(
    self,
    folder_hash: str,
    num_ideas: int = 3,
    idea_focus: str = "all"
) -> Tuple[List["ResearchIdea"], str, Dict[str, Any]]:
    """
    根据 folder hash 重新生成所有 ideas 以及初始周报

    Args:
        folder_hash: topic 的 folder hash（16位 MD5）
        num_ideas: 生成想法数量
        idea_focus: 想法聚焦方向

    Returns:
        Tuple[新生成的想法列表, 初始周报草稿, knowledge dict]
    """
    context_data = self._load_context(folder_hash)
    if not context_data:
        raise ValueError(f"Folder hash '{folder_hash}' 不存在或无 context.json")

    topic = context_data.get("topic", folder_hash)

    # 重建 knowledge dict
    knowledge = {
        "local_results": context_data.get("local_results", []),
        "web_results": context_data.get("web_results", []),
        "fused_context": self._fuse_knowledge_context(
            context_data.get("local_results", []),
            context_data.get("web_results", [])
        )
    }

    # 1. 重新生成所有 ideas（使用 VLM）
    ideas = await self.generate_ideas(
        knowledge_context=knowledge.get("fused_context", ""),
        research_domain=context_data.get("domain", ""),
        num_ideas=num_ideas,
        idea_focus=idea_focus
    )

    if not ideas:
        raise ValueError("Ideas 重新生成失败")

    # 2. 生成初始周报草稿（使用 VLM）
    initial_draft = await self._generate_initial_draft_vlm(ideas, topic, knowledge)

    # 3. 保存新 ideas 到文件（覆盖原有）
    self._regenerate_ideas_save(ideas, topic, knowledge, initial_draft)

    logger.info(f"[IdeaEngine] 已重新生成 {len(ideas)} 个 ideas 和初始周报草稿")
    return ideas, initial_draft, knowledge


def _regenerate_ideas_save(
    self,
    ideas: List["ResearchIdea"],
    topic: str,
    knowledge: Dict[str, Any],
    initial_draft: str = ""
) -> List[Tuple[str, Path]]:
    """覆盖保存想法到 topic 文件夹（删除旧 ideas，保存新 ideas）"""
    import uuid as uuid_module

    folder = self._topic_folder(topic)
    folder.mkdir(parents=True, exist_ok=True)

    # 删除旧的 idea 文件（保留 context.json）
    for f in folder.glob("*.json"):
        if f.name != "context.json":
            f.unlink()
            logger.info(f"[IdeaEngine] 删除旧 idea 文件: {f}")

    # 更新 context（保持原有 context 不变，只更新 local_results 和 web_results）
    existing_context = self._load_context(topic)
    if existing_context:
        knowledge_to_save = {
            "topic": topic,
            "domain": existing_context.get("domain", ""),
            "local_results": knowledge.get("local_results", []),
            "web_results": knowledge.get("web_results", [])
        }
    else:
        knowledge_to_save = {
            "topic": topic,
            "local_results": knowledge.get("local_results", []),
            "web_results": knowledge.get("web_results", [])
        }
    self._save_context(topic, knowledge_to_save)

    # 保存新 ideas
    results = []
    for idea in ideas:
        idea_uuid = str(uuid_module.uuid4())[:8]
        idea_data = {
            "id": idea_uuid,
            "topic": topic,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "regenerated": True,
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
        results.append((idea_uuid, file_path))
        logger.info(f"[IdeaEngine] 新 idea 已保存: {file_path}")

    # 更新 topic 索引
    index = self._get_topic_index()
    index[folder.name] = topic
    self._save_topic_index(index)

    # 保存初始周报草稿（如果有）
    if initial_draft:
        draft_path = folder / "initial_draft.md"
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(initial_draft)
        logger.info(f"[IdeaEngine] 初始周报草稿已保存: {draft_path}")

    return results


def _fuse_knowledge_context(
    self,
    local_results: List[Dict],
    web_results: List[Dict]
) -> str:
    """将 local 和 web 结果融合为文本上下文"""
    parts = []
    # 本地论文 - 使用正确的字段名：paper, text
    for r in local_results:
        paper = r.get("paper", "")
        text = r.get("text", "")
        if paper or text:
            parts.append(f"[本地文档] {paper}\n{text}")
    # 网页 - 使用正确的字段名：title, snippet, url
    for r in web_results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        if title or snippet:
            parts.append(f"[网页] {title}\n{snippet}\n{url}")
    return "\n\n".join(parts)


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
    logger.debug(f"[IdeaEngine] func_list 长度: {len(func_list)}")

    # 查找 feishu 相关的工具
    for tool in func_list:
        if 'feishu' in tool.name.lower():
            logger.info(f"[IdeaEngine] 找到飞书工具: {tool.name}")
            return tool
    return None


async def _pre_resolve_arxiv_links(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
    """
    预解析本地论文的 arxiv 链接，从 milvus_abstracts_doc_stats.json 获取

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

    # 直接从本地 JSON 文件读取 arxiv 链接
    doc_stats_path = Path(__file__).parent / "data" / "milvus_abstracts_doc_stats.json"
    if not doc_stats_path.exists():
        logger.warning(f"[IdeaEngine] 未找到 {doc_stats_path}")
        return knowledge

    try:
        with open(doc_stats_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        abstracts = data.get("abstracts", {})
    except Exception as e:
        logger.warning(f"[IdeaEngine] 读取 doc_stats 失败: {e}")
        return knowledge

    # 克隆 knowledge 避免修改原始数据
    enriched = dict(knowledge)
    enriched["arxiv_links"] = {}
    enriched["github_links"] = {}

    for result in local_results[:10]:
        paper = result.get("paper", "")
        if not paper or paper in enriched["arxiv_links"]:
            continue

        # 查找匹配的 paper_id
        arxiv_url = None
        github_url = None

        # 直接匹配
        if paper in abstracts:
            meta = abstracts[paper].get("metadata", {})
            arxiv_url = meta.get("arxiv_url") or None
            github_url = meta.get("github_url") or None
        else:
            # 尝试模糊匹配（去掉 .pdf 后缀）
            paper_clean = paper
            if paper_clean.endswith(".pdf"):
                paper_clean = paper_clean[:-4]
            if paper_clean in abstracts:
                meta = abstracts[paper_clean].get("metadata", {})
                arxiv_url = meta.get("arxiv_url") or None
                github_url = meta.get("github_url") or None

        if arxiv_url:
            enriched["arxiv_links"][paper] = arxiv_url
            enriched["github_links"][paper] = github_url
            logger.info(f"[IdeaEngine] 预解析: {paper[:30]} -> {arxiv_url}")

    return enriched


async def _polish_content_for_feishu(
    self,
    ideas: List["ResearchIdea"],
    topic: str,
    knowledge: Optional[Dict[str, Any]] = None
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
    # ... (full implementation removed)


def _build_citations_context(self, knowledge: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    构建引用上下文，包含本地检索和网络搜索的结果

    Args:
        knowledge: 知识检索结果

    Returns:
        Tuple[str, Dict]: (格式化的引用上下文字符串, 提取的媒体资源字典)
    """
    # ... (full implementation removed)


def _is_simple_caption(self, caption: str) -> bool:
    """检查 caption 是否只是简单的编号而没有实际描述"""
    if not caption:
        return True
    # 匹配简单的 Figure/Table/图 编号格式
    simple_pattern = re.match(
        r'^(Figure|Fig\.|Fig|Table|表|图)\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)?)$',
        caption.strip(),
        re.IGNORECASE
    )
    return simple_pattern is not None


async def _enhance_media_captions(
    self,
    extracted_media: Dict[str, Any],
    knowledge: Optional[Dict[str, Any]] = None,
    enable_qwen: bool = True
) -> Dict[str, Any]:
    """
    增强媒体 caption：
    1. 对于简单编号的 caption（如 "Figure 4"），调用本地 Qwen 模型根据图片生成描述
    2. 调用 AstrBot LLM 根据上下文润色描述

    Args:
        extracted_media: 提取的媒体资源
        knowledge: 知识检索结果（包含相关 chunks）
        enable_qwen: 是否启用 Qwen 图像分析（需要本地部署 Qwen-VL）

    Returns:
        Dict: 增强后的媒体资源
    """
    # ... (full implementation removed)


async def _generate_image_summary_with_qwen(
    self,
    image_path: str,
    context_text: str
) -> Optional[dict]:
    """
    调用本地 LlamaCppVLMProvider (Qwen3.5-9B-GGUF) 分析图片并生成摘要

    策略：优先提取图片中的文字（OCR），若无文字则进行视觉分析

    Args:
        image_path: 图片路径
        context_text: 相关上下文文本（chunk 片段）

    Returns:
        dict: {"summary": str, "is_relevant": bool}，失败返回 None
    """
    # ... (full implementation removed)


async def _final_image_decision(
    self,
    qwen_result: dict,
    context_text: str,
    original_caption: str,
    caption_type: str = "图"
) -> Optional[str]:
    """
    调用云端 LLM 综合本地 VLM 的摘要和判断，做最终决定

    Args:
        qwen_result: 本地 VLM 返回的结果 {"summary": str, "is_relevant": bool}
        context_text: 上下文文本
        original_caption: 原始 caption
        caption_type: "图" 或 "表"

    Returns:
        str: 润色后的 caption 或 [SKIP]
    """
    # ... (full implementation removed)


async def _polish_caption_with_llm(
    self,
    description: str,
    context_text: str,
    caption_type: str = "图"
) -> Optional[str]:
    """
    调用 AstrBot LLM 润色 caption

    Args:
        description: 原始描述
        context_text: 上下文文本
        caption_type: "图" 或 "表"

    Returns:
        str: 润色后的 caption
    """
    # ... (full implementation removed)


async def _audit_media_relevance(
    self,
    content: str,
    extracted_media: Dict[str, Any],
    knowledge: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    使用本地 VLM 按章节审阅文档中插入的图片是否与内容相关，不相关则删除。

    流程：
    1. 按 Markdown 章节（## xxx）分块，每章作为一个审阅单元
    2. 每章收集所有 <!-- INSERT_IMAGE:本地图-N --> 标记，批量传给 VLM
    3. VLM 判断哪些图片与该章节内容不相关/有事实错误
    4. 根据 VLM 决策删除对应标记（不对文本进行任何润色或精简）
    5. 返回修改后的内容和更新后的 extracted_media

    Args:
        content: 包含 INSERT_IMAGE 标记的文档内容
        extracted_media: 提取的媒体资源
        knowledge: 知识检索结果（用于获取图片路径等信息）

    Returns:
        Tuple[str, Dict]: (审阅后的内容, 更新后的 extracted_media)
    """
    # ... (full implementation removed)


async def _vlm_audit_section(
    self,
    vlm_provider,
    section_title: str,
    section_text: str,
    images: List[Tuple[str, str, str]]
) -> List[str]:
    """
    调用本地 VLM 审阅某个章节中的所有图片，返回需要删除的标记列表。

    策略：不修改文本、不润色内容，只判断哪些图片与章节内容不相关/有事实错误。

    Args:
        vlm_provider: VLM 提供者
        section_title: 章节标题（如 "## 背景动机"）
        section_text: 章节正文内容（不包含标题行）
        images: List[(index, image_path, caption)]，该章节所有图片

    Returns:
        List[str]: 需要删除的标记列表，如 ["本地图-1", "本地图-3"]
    """
    # ... (full implementation removed)


async def _cleanup_content_for_feishu(self, content: str) -> str:
    """
    使用 LLM 清理内容中因媒体引用被跳过而导致的断句问题

    Args:
        content: 原始内容（可能包含空引用如"如图 所示"）

    Returns:
        str: 清理后的内容
    """
    # ... (full implementation removed)


def debug_media_captions(self, knowledge: Dict[str, Any]) -> str:
    """
    调试函数：统计图片/表格的 caption 提取情况

    Args:
        knowledge: 知识检索结果

    Returns:
        str: 统计报告
    """
    # ... (full implementation removed)


def _create_media_blocks(self, extracted_media: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从提取的媒体资源创建飞书块（图片/表格）

    注意：图片需要两阶段处理 - 先创建空图片块，再上传绑定
    表格需要使用 create_feishu_table 工具

    Args:
        extracted_media: 从本地检索中提取的媒体资源

    Returns:
        Tuple[List[Dict], List[Dict]]: (飞书文本/图片块列表, 待上传图片列表)
    """
    # ... (full implementation removed)


def _replace_media_markers(
    self,
    content: str,
    extracted_media: Dict[str, Any]
) -> str:
    """
    将媒体标记替换为带 caption 的引用格式

    Args:
        content: 包含 <!-- INSERT_IMAGE:本地图-X --> 或 <!-- INSERT_TABLE:本地表-X --> 标记的内容
        extracted_media: 提取的媒体资源，包含 images 和 tables 列表

    Returns:
        str: 标记被替换为引用格式，如 "图1: caption文本" 或 "表1: caption文本"
    """
    # ... (full implementation removed)


async def analyze_topic(self, topic: str, depth: str = "standard") -> Optional[TopicAnalysis]:
    """
    分析研究主题，生成搜索策略

    Args:
        topic: 研究话题
        depth: 分析深度 (quick/standard/deep)

    Returns:
        TopicAnalysis: 结构化的主题分析
    """
    # ... (full implementation removed)


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
    # ... (full implementation removed)


def _calculate_text_similarity(self, text1: str, text2: str) -> float:
    """
    简单的文字相似度计算（基于关键词重叠）

    Args:
        text1: 图片提取文字
        text2: chunks 文本

    Returns:
        float: 相似度 0.0 ~ 1.0
    """
    # ... (full implementation removed)


def _write_search_to_temp_file(self, web_results: List[Dict]) -> str:
    """
    将网络搜索结果写入临时文件，供 LLM 读取

    Args:
        web_results: 搜索结果列表

    Returns:
        str: 临时文件路径
    """
    # ... (full implementation removed)


async def _call_brightdata_mcp_tool(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: int = 120
) -> Dict[str, Any]:
    """
    通用 Bright Data MCP 工具调用方法

    支持的工具:
    - search_engine: 搜索引擎搜索
    - search_engine_batch: 批量搜索引擎搜索
    - scrape_as_markdown: 抓取单个页面为 Markdown
    - scrape_batch: 批量抓取页面为 Markdown
    - discover: AI 驱动的智能搜索

    Args:
        tool_name: 工具名称
        arguments: 工具参数
        timeout: 超时时间（秒）

    Returns:
        Dict 包含工具执行结果
    """
    # ... (full implementation removed)


async def _search_web(self, queries: List[str], top_k: int) -> List[Dict]:
    """通过网络搜索获取信息（通过Bright Data MCP）"""
    # ... (full implementation removed)


async def _scrape_as_markdown(self, url: str) -> Dict[str, Any]:
    """
    抓取单个页面为 Markdown

    Args:
        url: 网页 URL

    Returns:
        Dict 包含 success, markdown 内容或 error
    """
    # ... (full implementation removed)


async def _scrape_batch_markdown(self, urls: List[str]) -> Dict[str, Any]:
    """
    批量抓取页面为 Markdown

    Args:
        urls: URL 列表（最多5个）

    Returns:
        Dict 包含 success, results 列表或 error
    """
    # ... (full implementation removed)


async def _discover_search(
    self,
    query: str,
    intent: str = "",
    country: str = "US",
    num_results: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    AI 驱动的智能搜索

    Args:
        query: 搜索查询
        intent: 搜索意图描述
        country: 国家代码
        num_results: 返回结果数量

    Returns:
        Dict 包含 success, results 列表或 error
    """
    # ... (full implementation removed)


async def _search_engine_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    批量搜索引擎搜索

    Args:
        queries: 查询列表，每个包含 query, engine, geo_location 等

    Returns:
        Dict 包含 success, results 列表或 error
    """
    # ... (full implementation removed)


async def _call_arxiv_mcp_tool(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: int = 60
) -> Dict[str, Any]:
    """
    通过 func_list 调用 arxiv MCP 工具

    Args:
        tool_name: arxiv 工具名称（如 search_papers, read_paper）
        arguments: 工具参数
        timeout: 超时时间

    Returns:
        Dict 包含工具执行结果
    """
    # ... (full implementation removed)


async def _parse_and_execute_brightdata_tools(
    self,
    text: str,
    max_iterations: int = 3
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    解析 LLM 输出中的 Bright Data 工具调用标签并执行（优化版：单次遍历）

    支持的标签格式:
    - <search>query</search>
    - <discover>query|intent|country</discover>
    - <scrape>url</scrape>
    - <batch_search>["q1","q2"]</batch_search>
    - <scrape_batch>["url1","url2"]</scrape_batch>
    """
    # ... (full implementation removed)


def _format_search_result(self, result: Dict[str, Any]) -> str:
    """格式化搜索结果"""
    # ... (full implementation removed)


def _format_discover_result(self, result: Dict[str, Any]) -> str:
    """格式化 discover 结果"""
    # ... (full implementation removed)


def _format_scrape_result(self, result: Dict[str, Any]) -> str:
    """格式化抓取结果"""
    # ... (full implementation removed)


def _format_arxiv_result(self, result: Dict[str, Any]) -> str:
    """格式化 arxiv 搜索结果"""
    # ... (full implementation removed)


def _format_arxiv_paper_result(self, result: Dict[str, Any]) -> str:
    """格式化 arxiv 论文详情"""
    # ... (full implementation removed)


# ==================== 原生 Agent 方案实现 ====================

async def _polish_with_native_agent(
    self,
    polish_prompt: str,
    ideas_summary: str = "",
    contexts: list[dict] | None = None,
    max_iterations: int = 25,
) -> str:
    """
    使用原生 Agent 方案进行内容润色（干净的上下文）

    - system_prompt=None，避免注入 AstrBot 人格/配置
    - contexts 只包含必要的参考资料
    - 通过 func_tool 传递工具集，框架自动处理 tool_calls 循环
    """
    # ... (full implementation removed)


async def _handle_tool_calls_loop(
    self,
    provider,
    initial_response,
    tool_set,
    max_iterations: int = 25,
    polish_prompt: str = "",
    ideas_summary: str = "",
):
    """处理原生 tool_calls 循环 - 模仿旧方案：执行所有工具调用后做一次整合"""
    # ... (full implementation removed)


def _build_tool_calls_result(
    self,
    tool_names: list[str],
    tool_args: list[dict],
    tool_ids: list[str],
    tool_results: list[tuple[str, str, str]],
) -> ToolCallsResult:
    """构造 ToolCallsResult 用于回传给 LLM"""
    # ... (full implementation removed)


async def _execute_llm_tool(self, tool_name: str, args: dict) -> str:
    """执行单个 LLM 工具调用"""
    # ... (full implementation removed)


async def test_brightdata_mcp(self, query: str) -> Dict[str, Any]:
    """
    测试 Bright Data MCP 学术搜索功能

    此方法直接调用 Bright Data MCP 进行搜索测试。

    Args:
        query: 搜索查询词

    Returns:
        Dict包含搜索结果或错误信息
    """
    # ... (full implementation removed)


async def test_feishu_markdown_formats(
    self,
    folder_token: str = ""
) -> Dict[str, Any]:
    """
    测试飞书文档的 Markdown 格式渲染（使用 mistune v3 插件）

    测试内容：
    - 一级/二级/三级标题
    - 加粗、斜体、加粗斜体
    - 行内代码
    - 删除线
    - 链接 [文本](url)
    - 无序列表、有序列表
    - 分割线
    - 图表引用 [图X]、[表X]
    - LaTeX 公式 $公式$
    - 混合内容

    Args:
        folder_token: 飞书文件夹 Token

    Returns:
        Dict包含测试结果
    """
    # ... (full implementation removed)


async def create_feishu_document(
    self,
    ideas: List[ResearchIdea],
    topic: str = "",
    folder_token: str = "",
    knowledge: Optional[Dict[str, Any]] = None,
    table_format: str = "png"
) -> Dict[str, Any]:
    """
    创建飞书文档并写入研究想法

    流程：
    1. LLM 生成文档标题（不使用原始问题作为标题）
    2. LLM 润色内容（组会周报学术风格，含引用、相关工作、Benchmark、研究计划）
    3. 直接调用 AstrBot 已连接的 feishu MCP 工具
    4. 返回润色内容和文档链接

    Args:
        ideas: 研究想法列表
        topic: 研究主题
        folder_token: 飞书文件夹Token（可选）
        knowledge: 知识检索结果（包含 web_results, local_results）
        table_format: 表格插入格式，可选 "csv"、"md"、"png"(默认)、"auto"

    Returns:
        Dict包含 document_id, url, polished_content, error 等信息
    """
    # ... (full implementation removed)


async def _resolve_references(
    self,
    content: str,
    knowledge: Optional[Dict[str, Any]] = None
) -> str:
    """
    将本地论文引用解析为 arxiv 链接，网络引用解析为带链接的格式

    Args:
        content: 包含引用标记的文档内容
        knowledge: 知识检索结果

    Returns:
        str: 将引用标记替换为带链接的格式
    """
    # ... (full implementation removed)


def _markdown_to_feishu_blocks(self, markdown_text: str) -> List[Dict]:
    """将Markdown文本转换为飞书块格式"""
    # ... (full implementation removed)
