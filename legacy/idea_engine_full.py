"""
研究创意生成引擎

整合Bright Data网络搜索 + 本地Paper RAG + LLM生成
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
        # 检查 Bright Data MCP 是否配置
        self._bright_data_available = self._check_bright_data_config()

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

    def _get_ideas_dir(self) -> Path:
        """获取想法存储根目录，不存在则创建"""
        ideas_dir = Path(__file__).parent.parent.parent / "plugin_data" / "astrbot_plugin_paperrag" / "ideas"
        ideas_dir.mkdir(parents=True, exist_ok=True)
        return ideas_dir

    def _topic_folder(self, topic: str) -> Path:
        """获取 topic 对应的文件夹路径（使用 MD5 哈希，跨进程稳定）"""
        return self._get_ideas_dir() / self._topic_hash(topic)

    def _topic_hash(self, topic: str) -> str:
        """计算 topic 对应的 folder hash（MD5 hex 前16位）"""
        return hashlib.md5(topic.encode()).hexdigest()[:16]

    def _get_topic_index(self) -> Dict[str, str]:
        """获取 folder_name → topic 的索引"""
        index_file = self._get_ideas_dir() / "topic_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    logger.warning("[IdeaEngine] topic_index.json 格式错误（非 dict）")
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def find_topic_by_folder(self, folder_name: str) -> Optional[str]:
        """根据 folder_name 查找对应的 topic"""
        return self._get_topic_index().get(folder_name)

    def _save_topic_index(self, index: Dict[str, str]) -> None:
        """保存 topic → folder_name 索引"""
        index_file = self._get_ideas_dir() / "topic_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def list_all_topics(self) -> List[Dict[str, Any]]:
        """
        列出所有已保存的 topic 及其元信息

        Returns:
            List[Dict]: [{"topic": str, "folder": str, "idea_count": int, "created_at": str}, ...]
        """
        index = self._get_topic_index()
        ideas_dir = self._get_ideas_dir()
        result = []

        for folder_name, topic in index.items():
            folder = ideas_dir / folder_name
            if not folder.exists():
                continue

            # 统计 idea 数量（排除 context.json）
            idea_files = [f for f in folder.glob("*.json") if f.name != "context.json"]
            created_at = ""
            if (folder / "context.json").exists():
                try:
                    with open(folder / "context.json", "r", encoding="utf-8") as f:
                        ctx = json.load(f)
                        if isinstance(ctx, dict):
                            created_at = ctx.get("created_at", "")
                except (json.JSONDecodeError, IOError):
                    pass

            result.append({
                "topic": topic,
                "folder": folder_name,
                "idea_count": len(idea_files),
                "created_at": created_at
            })

        return result

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
        # 加载现有 context
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
            # 保留原有 domain 等信息
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

        # ========== 第一阶段图表过滤 ==========
        # 在构建引用上下文之前先过滤，只保留与 ideas 相关的图片
        filtered_media = {"images": [], "tables": []}
        if knowledge:
            local_results = knowledge.get("local_results", [])
            if local_results:
                # 调用第二阶段图表过滤
                filtered_images = await self._filter_figures_by_relevance(local_results)
                logger.info(f"[IdeaEngine] 图表预过滤完成，保留 {len(filtered_images)}/{len(local_results)} 张相关图片")

                # 构建简化版 extracted_media（不含 base64，VLM 只需 caption 列表）
                for i, img in enumerate(filtered_images, 1):
                    filtered_media["images"].append({
                        "index": f"本地图-{i}",
                        "path": img.get("image_path", ""),
                        "caption": img.get("image_caption", ""),
                        "source_paper": img.get("paper", ""),
                        "source_page": img.get("page", "")
                    })
                # 表格去重（保留有 caption 的表格，按 csv_path 去重）
                table_set = set()
                for r in local_results:
                    metadata = r.get("metadata", {})
                    tbl_caption = metadata.get("table_caption", "")
                    tbl_csv = metadata.get("table_csv_path", "")
                    if tbl_caption and tbl_csv and tbl_csv not in table_set:
                        table_set.add(tbl_csv)
                        filtered_media["tables"].append({
                            "index": f"本地表-{len(filtered_media['tables']) + 1}",
                            "csv_path": tbl_csv,
                            "caption": tbl_caption,
                            "source_paper": r.get("paper", ""),
                            "source_page": r.get("page", "")
                        })

        # 构建引用上下文（使用过滤后的图片）
        citations_context = ""
        if filtered_media["images"] or filtered_media["tables"]:
            citations_context = "## 可引用的图片资源：\n"
            for img in filtered_media["images"]:
                citations_context += f"- {img['index']}: {img['caption']} (来源: {img['source_paper']}, 页码: {img['source_page']})\n"
            if filtered_media["tables"]:
                citations_context += "\n## 可引用的表格资源：\n"
                for tbl in filtered_media["tables"]:
                    citations_context += f"- {tbl['index']}: {tbl['caption']} (来源: {tbl['source_paper']}, 页码: {tbl['source_page']})\n"

        # 构建媒体说明（供 VLM 在生成内容时引用）
        # 注意：这里只传递 caption 列表，不需要 base64（base64 只在上传飞书时才需要）
        media_instructions = ""
        if filtered_media["images"]:
            media_instructions += f"\n\n**可引用的图片资源：**\n"
            for img in filtered_media["images"]:
                media_instructions += f"- {img['index']}: {img['caption']} (来源: {img['source_paper']}, 页码: {img['source_page']})\n"
        if filtered_media["tables"]:
            media_instructions += f"\n**可引用的表格资源：**\n"
            for tbl in filtered_media["tables"]:
                media_instructions += f"- {tbl['index']}: {tbl['caption']} (来源: {tbl['source_paper']}, 页码: {tbl['source_page']})\n"

        ideas_summary = self._format_ideas_as_markdown(ideas, topic)

        prompt = f"""基于以下研究想法和参考资料，生成一个详细完整的组会周报。

研究主题：{topic}

研究想法：
{ideas_summary}

参考资料（RAG检索到的chunk，包含丰富信息，请充分利用）：
{citations_context}
{media_instructions}

请生成一个详细完整的组会周报，包含以下章节，每个章节都要有详细展开：
- 背景动机：详细说明问题的背景、重要性、现有方法的不足（5-8句）
- 相关工作：详细综述相关方法和论文，引用论文的具体贡献（5-8句）
- 方法论：详细描述方法细节、工作流程、技术路线（5-10句）
- 创新点：明确列出2-3个具体创新点，并解释为什么这些创新有效（5-8句）
- 实验benchmark：详细说明实验设置、数据集、对比方法、评价指标（5-8句）
- 挑战与解决方案：每个挑战都要详细说明原因和对应的具体解决方案（5-8句）
- 下一步计划：具体的下一步研究方向和可行的改进思路（3-5句）

**重要**：参考资料中包含丰富的细节信息，请充分利用这些信息生成详细内容，不要简略！
"""

        try:
            logger.info("[IdeaEngine] 使用 VLM 生成详细初始周报草稿...")
            draft = await self._vlm_chat_with_progress(
                vlm_provider,
                prompt=prompt,
                temperature=0.7,
                max_tokens=4096,
                task_name="VLM生成初始周报草稿"
            )
            return draft
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 生成失败: {e}，使用简单格式化")
            return ideas_summary

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

    def _get_context_path(self, topic: str) -> Path:
        """获取 topic 文件夹下的 context.json 路径"""
        return self._topic_folder(topic) / "context.json"

    def _save_context(self, topic: str, knowledge: Dict[str, Any]) -> None:
        """保存共享 context 到 topic 文件夹"""
        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)
        ctx_data = {
            "topic": topic,
            "local_results": knowledge.get("local_results", []),
            "web_results": knowledge.get("web_results", [])
        }
        logger.info(f"[IdeaEngine] _save_context: web_results保存数量={len(ctx_data['web_results'])}, local_results保存数量={len(ctx_data['local_results'])}")
        with open(self._get_context_path(topic), "w", encoding="utf-8") as f:
            json.dump(ctx_data, f, ensure_ascii=False, indent=2)

    def _load_context(self, topic: str) -> Optional[Dict[str, Any]]:
        """加载共享 context（topic 可能是原始名称或 folder hash）"""
        # 如果 topic 本身是合法的 folder 名，直接使用；否则计算 hash
        folder = self._get_ideas_dir() / topic
        if not folder.exists():
            folder = self._topic_folder(topic)
        ctx_path = folder / "context.json"
        if not ctx_path.exists():
            return None
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"[IdeaEngine] _load_context: web_results={len(data.get('web_results', []))}, local_results={len(data.get('local_results', []))}")
                    return data
                logger.warning(f"[IdeaEngine] context.json 格式错误（非 dict 类型）: {type(data)}")
                return None
        except (json.JSONDecodeError, IOError):
            return None

    def save_ideas_to_file(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Dict[str, Any]
    ) -> List[Tuple[str, Path]]:
        """
        将多个想法及上下文保存到 topic 文件夹

        目录结构:
        ideas/
          topic_index.json
          <hash(topic)>/
            context.json          # 共享 context
            <uuid1>.json        # 单个 idea
            <uuid2>.json

        Args:
            ideas: 研究想法列表
            topic: 原始 topic
            knowledge: 知识检索结果

        Returns:
            List[Tuple[str, Path]]: [(uuid, 文件路径), ...]
        """
        import uuid as uuid_module

        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        # 保存共享 context
        self._save_context(topic, knowledge)

        # 保存每个 idea 到 topic 文件夹
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
            logger.info(f"[IdeaEngine] 想法已保存: {file_path}")

        # 更新 topic 索引（folder_name → topic）
        index = self._get_topic_index()
        index[folder.name] = topic
        self._save_topic_index(index)

        return results

    def load_ideas_by_uuids(
        self,
        uuids: List[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        根据 UUID 列表加载想法，同时加载共享 context

        通过扫描 topic_index.json 定位 UUID 所在的文件夹，无需 topic 参数

        Args:
            uuids: UUID 列表，如 ["a1b2c3d4", "e5f6g7h8"]

        Returns:
            Tuple[List[想法dict], context dict]
        """
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()
        loaded = []
        found_topic = None

        for folder_name, topic in index.items():
            folder = ideas_dir / folder_name
            if not folder.exists():
                continue
            for uid in uuids:
                file_path = folder / f"{uid}.json"
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            loaded.append(data)
                            if found_topic is None:
                                found_topic = topic
                        else:
                            logger.warning(f"[IdeaEngine] 想法文件格式错误（非 dict）: {uid}")
                    except (json.JSONDecodeError, IOError) as e:
                        logger.error(f"[IdeaEngine] 读取想法文件失败 {uid}: {e}")

        context = self._load_context(found_topic) if found_topic else None
        return loaded, context

    def load_ideas_by_topic(
        self, folder_hash: str
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        根据 folder hash 加载该 topic 下所有想法

        Args:
            folder_hash: folder 名称（MD5 hash）

        Returns:
            Tuple[List[想法dict], context dict]
        """
        folder = self._get_ideas_dir() / folder_hash
        if not folder.exists():
            logger.warning(f"[IdeaEngine] 未找到 folder_hash={folder_hash} 的文件夹")
            return [], None

        loaded = []
        for file_path in folder.glob("*.json"):
            if file_path.name == "context.json":
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    loaded.append(data)
                else:
                    logger.warning(f"[IdeaEngine] 想法文件格式错误（非 dict）: {file_path.name}")
            except (json.JSONDecodeError, IOError):
                logger.warning(f"[IdeaEngine] 跳过损坏的想法文件: {file_path.name}")

        context = self._load_context(folder_hash)
        return loaded, context

    def load_initial_draft(self, folder_hash: str) -> Optional[str]:
        """加载已保存的初始周报草稿

        Args:
            folder_hash: folder 名称（MD5 hash）

        Returns:
            草稿内容，如果不存在则返回 None
        """
        folder = self._get_ideas_dir() / folder_hash
        draft_path = folder / "initial_draft.md"
        if not draft_path.exists():
            logger.info(f"[IdeaEngine] 未找到已保存的初始草稿: {draft_path}")
            return None
        try:
            with open(draft_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"[IdeaEngine] 加载已保存的初始草稿，长度: {len(content)}")
            return content
        except Exception as e:
            logger.warning(f"[IdeaEngine] 读取初始草稿失败: {e}")
            return None

    def convert_to_research_ideas(
        self, ideas_list: List[Dict[str, Any]]
    ) -> List["ResearchIdea"]:
        """
        将想法数据列表转换回 ResearchIdea 对象列表

        Args:
            ideas_list: load_ideas_by_uuids 返回的想法列表

        Returns:
            List[ResearchIdea]
        """
        research_ideas = []
        for data in ideas_list:
            if not isinstance(data, dict):
                logger.warning(f"[IdeaEngine] 跳过无效想法数据: {type(data)}")
                continue
            item = data.get("idea", {})
            research_ideas.append(ResearchIdea(
                title=item.get("title", ""),
                description=item.get("description", ""),
                novelty=item.get("novelty", ""),
                methodology=item.get("methodology", ""),
                potential_challenges=item.get("potential_challenges", []),
                related_work=item.get("related_work", []),
                feasibility=item.get("feasibility", 0.5),
                inspiration_sources=item.get("inspiration_sources", [])
            ))
        return research_ideas

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

    def _get_vlm_provider(self):
        """获取本地VLM provider（LlamaCppVLMProvider）"""
        try:
            from .llama_cpp_vlm_provider import (
                get_llama_cpp_vlm_provider,
            )
        except ImportError as e:
            logger.warning(f"[IdeaEngine] 无法导入 LlamaCppVLMProvider: {e}")
            return None

        try:
            # 使用原生的单例获取函数
            vlm_provider = get_llama_cpp_vlm_provider()
            return vlm_provider
        except Exception as e:
            logger.warning(f"[IdeaEngine] 获取 VLM Provider 失败: {e}")
            return None

    async def _get_vlm_provider_async(self):
        """异步获取并初始化本地VLM provider"""
        vlm_provider = self._get_vlm_provider()
        if vlm_provider is None:
            return None

        # 如果未初始化，等待初始化完成
        if not vlm_provider._initialized:
            logger.info("[IdeaEngine] VLM Provider 未初始化，等待初始化...")
            await vlm_provider.initialize()

        return vlm_provider

    async def _vlm_chat_with_progress(self, vlm_provider, prompt: str, temperature: float, max_tokens: int, task_name: str = "VLM生成") -> str:
        """
        带进度提示的VLM调用，在推理过程中每10秒输出一次状态

        Args:
            vlm_provider: VLM provider
            prompt: 提示词
            temperature: 温度
            max_tokens: 最大token数
            task_name: 任务名称（用于日志）

        Returns:
            生成的文本内容
        """
        import asyncio
        import time

        logger.info(f"[IdeaEngine] {task_name}开始，prompt长度: {len(prompt)}")

        async def progress_logger():
            """后台定时输出进度日志"""
            elapsed = 0
            while True:
                await asyncio.sleep(10)
                elapsed += 10
                logger.info(f"[IdeaEngine] {task_name}进行中，已耗时{elapsed}秒...")

        # 启动进度日志任务
        progress_task = asyncio.create_task(progress_logger())

        try:
            # 执行VLM推理
            response = await vlm_provider.text_chat(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 取消进度日志任务
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

            if hasattr(response, 'content'):
                result = response.content
            elif isinstance(response, dict):
                result = response.get("content", "") or response.get("text", "")
            else:
                result = str(response)

            logger.info(f"[IdeaEngine] {task_name}完成，生成{len(result)}字符")
            return result

        except asyncio.CancelledError:
            # 如果VLM任务被取消，也取消进度日志
            progress_task.cancel()
            raise

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
        provider = self._get_llm_provider()
        if not provider:
            logger.warning("[IdeaEngine] 无法获取 LLM Provider，使用原始格式")
            return self._format_ideas_as_markdown(ideas, topic), {"images": [], "tables": []}, topic

        # 构建原始内容摘要
        ideas_summary = self._format_ideas_as_markdown(ideas, topic)

        # ========== 预过滤：只保留与 ideas 相关的图片 ==========
        # 注意：这里与 _generate_initial_draft_vlm 保持一致，确保 VLM 草稿和 LLM 润色使用同样的图片
        filtered_media = {"images": [], "tables": []}
        citations_context = ""
        if knowledge:
            local_results = knowledge.get("local_results", [])
            web_results = knowledge.get("web_results", [])
            if local_results:
                # 先添加本地 chunk 文本引用
                citations_context += "## 本地论文引用：\n"
                papers: Dict[str, List] = {}
                for r in local_results:
                    paper = r.get("paper", "Unknown")
                    if paper not in papers:
                        papers[paper] = []
                    papers[paper].append(r)

                for paper, chunks in papers.items():
                    citations_context += f"### {paper}\n"
                    for chunk in chunks[:5]:
                        text = chunk.get("text", "")
                        if text:
                            citations_context += f"- {text}\n"
                    citations_context += "\n"

                # 再添加网页引用
                if web_results:
                    citations_context += "## 网络资源引用：\n"
                    for i, r in enumerate(web_results, 1):
                        title = r.get("title", "")
                        url = r.get("url", "")
                        snippet = r.get("snippet", "")
                        if url:
                            citations_context += f"- [{title}]({url})\n"
                        else:
                            citations_context += f"- {title}\n"
                        if snippet:
                            citations_context += f"  摘要: {snippet}\n"
                    citations_context += "\n"

                # 调用第二阶段图表过滤
                filtered_images = await self._filter_figures_by_relevance(local_results)
                logger.info(f"[IdeaEngine] LLM润色前图表过滤，保留 {len(filtered_images)}/{len(local_results)} 张相关图片")

                # 构建过滤后的 extracted_media（包含 base64，供后续飞书上传使用）
                local_image_idx = 0
                for img in filtered_images:
                    image_path = img.get("image_path", "")
                    if not image_path or not os.path.exists(image_path):
                        logger.warning(f"[IdeaEngine] 过滤后的图片文件不存在，跳过: {image_path}")
                        continue
                    local_image_idx += 1
                    img_index = f"本地图-{local_image_idx}"
                    img_caption = img.get("image_caption", f"图 {local_image_idx}")

                    # 读取 base64
                    img_base64 = None
                    try:
                        with open(image_path, "rb") as f:
                            img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] 读取图片失败 {image_path}: {e}")

                    filtered_media["images"].append({
                        "index": img_index,
                        "path": image_path,
                        "base64": img_base64,
                        "caption": img_caption,
                        "source_paper": img.get("paper", ""),
                        "source_page": img.get("page", "")
                    })

                    # 构建引用文本
                    citations_context += f"- {img_caption} (来源: {img.get('paper', '')}, 页码: {img.get('page', '')}, 相关度: {img.get('image_score', 0):.3f})\n"

                # 也对表格做简单过滤
                table_set = set()
                for r in local_results:
                    metadata = r.get("metadata", {})
                    tbl_caption = metadata.get("table_caption", "")
                    tbl_csv = metadata.get("table_csv_path", "")
                    if tbl_caption and tbl_csv and tbl_csv not in table_set:
                        table_set.add(tbl_csv)
                        filtered_media["tables"].append({
                            "index": f"本地表-{len(filtered_media['tables']) + 1}",
                            "csv_path": tbl_csv,
                            "caption": tbl_caption,
                            "source_paper": r.get("paper", ""),
                            "source_page": r.get("page", "")
                        })
                        citations_context += f"- {tbl_caption} (来源: {r.get('paper', '')}, 页码: {r.get('page', '')})\n"

        # 如果没有过滤出任何媒体，使用空的 extracted_media（让后续 LLM 不要插入图片引用）
        extracted_media = filtered_media if (filtered_media["images"] or filtered_media["tables"]) else {"images": [], "tables": []}

        # 增强媒体 caption（对简单编号的 caption 调用 Qwen + LLM 生成描述）
        extracted_media = await self._enhance_media_captions(extracted_media, knowledge)

        # 将网络搜索结果写入临时文件，供 LLM 读取完整内容
        web_search_file = ""
        web_results = knowledge.get("web_results", []) if knowledge else []
        web_results_content = ""

        logger.info(f"[IdeaEngine] 润色阶段 web_results 数量: {len(web_results)}")

        if web_results:
            web_search_file = self._write_search_to_temp_file(web_results)
            # 读取搜索结果文件内容，格式化后加入 prompt
            try:
                with open(web_search_file, "r", encoding="utf-8") as f:
                    web_results_content = f.read()
                logger.info(f"[IdeaEngine] 已读取搜索结果文件，内容长度: {len(web_results_content)}")
            except Exception as e:
                logger.warning(f"[IdeaEngine] 读取搜索结果文件失败: {e}")
                web_results_content = ""

        # 构建媒体说明（排除被标记跳过的图片）
        # 包含 VLM 提取的图片描述，供 LLM 判断相关性
        media_instructions = ""
        valid_images = [img for img in extracted_media["images"] if not img.get("_skip")]
        valid_tables = [tbl for tbl in extracted_media.get("tables", []) if not tbl.get("_skip")]
        if valid_images:
            media_instructions += f"\n\n**可用的图片资源（VLM提取的文字描述）：**\n"
            for img in valid_images:
                page_str = f"，页码: {img['source_page']}" if img.get('source_page') else ""
                img_desc = img.get('image_description', img.get('caption', ''))
                media_instructions += f"- {img['index']}: {img_desc} (来源: {img['source_paper']}{page_str})\n"
        if valid_tables:
            media_instructions += f"\n**可用的表格资源：**\n"
            for tbl in valid_tables:
                page_str = f"，页码: {tbl['source_page']}" if tbl.get('source_page') else ""
                media_instructions += f"- {tbl['index']}: {tbl['caption']} (来源: {tbl['source_paper']}{page_str})\n"
                if tbl['csv_content'] and tbl['csv_content'] != "(无法读取)":
                    media_instructions += f"  表格内容预览:\n```\n{tbl['csv_content'][:200]}...\n```\n"

        # 重建 citations_context（排除被 _enhance_media_captions 跳过的图片）
        # 注意：原始 citations_context 在 _enhance_media_captions 之前构建，可能包含已被跳过的图片
        citations_context = ""
        for img in valid_images:
            citations_context += f"- {img.get('caption', '')} (来源: {img.get('source_paper', '')}, 页码: {img.get('source_page', '')}, 相关度: {img.get('image_score', 0):.3f})\n"
        for tbl in valid_tables:
            citations_context += f"- {tbl.get('caption', '')} (来源: {tbl.get('source_paper', '')}, 页码: {tbl.get('source_page', '')})\n"

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

        # 第二步：检查是否有已保存的初始周报草稿
        # 先尝试加载已保存的草稿（通过计算folder_hash）
        folder_hash = self._topic_hash(topic)
        saved_draft = self.load_initial_draft(folder_hash)

        if saved_draft:
            logger.info(f"[IdeaEngine] 找到已保存的初始草稿，长度: {len(saved_draft)}，直接使用")
            initial_draft = saved_draft
        else:
            # 没有已保存的草稿，需要重新生成
            vlm_provider = await self._get_vlm_provider_async()

            # VLM生成初始版本的prompt
            vlm_draft_prompt = f"""基于以下研究想法和参考资料，生成一个详细完整的组会周报。

研究主题：{topic}

研究想法：
{ideas_summary}

参考资料（RAG检索到的chunk，包含丰富信息，请充分利用）：
{citations_context}
{media_instructions}

请生成一个详细完整的组会周报，包含以下章节，每个章节都要有详细展开：
- 背景动机：详细说明问题的背景、重要性、现有方法的不足（5-8句）
- 相关工作：详细综述相关方法和论文，引用论文的具体贡献（5-8句）
- 方法论：详细描述方法细节、工作流程、技术路线（5-10句）
- 创新点：使用数字列表格式，如"1. 创新一：xxx"、"2. 创新二：xxx"，每个创新点说明其内容和效果
- 实验benchmark：详细说明实验设置、数据集、对比方法、评价指标（5-8句）
- 挑战与解决方案：使用数字列表格式，如"1. 挑战一：xxx。原因：xxx。解决方案：xxx"，结构清晰
- 下一步计划：具体的下一步研究方向和可行的改进思路（3-5句）

**重要**：
1. 参考资料中包含丰富的细节信息，请充分利用这些信息生成详细内容，不要简略！
2. **列表格式必须使用数字序号**（如"1."、"2."），**禁止使用"•"或"-"等无序列表符号**！
3. **每个列表项占一行**，项内部不要换行

请直接输出周报内容：
"""

            initial_draft = ""
            try:
                if vlm_provider:
                    logger.info("[IdeaEngine] 未找到已保存草稿，使用VLM生成详细初始周报版本...")
                    initial_draft = await self._vlm_chat_with_progress(
                        vlm_provider,
                        prompt=vlm_draft_prompt,
                        temperature=0.7,
                        max_tokens=4096,
                        task_name="VLM生成周报草稿"
                    )
                else:
                    logger.info("[IdeaEngine] VLM不可用，跳过初始版本生成")
                    initial_draft = ideas_summary
            except Exception as e:
                logger.warning(f"[IdeaEngine] VLM生成初始版本失败: {e}，使用ideas摘要")
                initial_draft = ideas_summary

        # 第三步：让LLM只对初始版本进行润色
        # 网络搜索结果内容（实际读取文件内容）
        web_file_instruction = ""
        if web_results_content:
            web_file_instruction = f"""
网络搜索结果：
{web_results_content}

请在润色时充分利用上述搜索结果，补充相关背景和技术细节。
"""
        elif web_search_file:
            web_file_instruction = f"""
网络搜索结果已保存至：{web_search_file}
如果你需要了解更多网络资源信息，可以使用工具搜索相关资料。
"""

        polish_prompt = f"""你是一个学术助手，负责对以下组会周报草稿进行润色和完善。

{web_file_instruction}

参考资料：
{citations_context}
{media_instructions}

原始草稿：
{initial_draft}

**重要指令**：
- **只润色，不重新生成**：不要完全重写，只在原有内容基础上进行润色、补充和完善
- 如果某些章节内容过于简单，可以适当扩展
- **可以使用工具搜索补充资料**，但不要完全替换原有内容
- 保持原文的结构和思路，只做细节优化
- **图片相关性判断**：根据上方参考资料中的图片描述（VLM提取的文字），判断每张图片是否与当前周报内容真正相关。如果图片与周报内容无关或重复，不要引用该图片。

格式要求：
- 包含章节：背景动机、相关工作、方法论、创新点、实验benchmark、挑战与解决方案、下一步计划
- **每个章节都要有详细的展开论述**，不能只是简短的要点列表
- **列表格式**：创新点和挑战与解决方案部分使用数字序号列表（如"1. 挑战一：xxx"），每个列表项包含：标题、原因/内容、解决方案/详细说明，结构清晰
- **列表项格式**：使用"1. xxx"、"2. xxx"格式，每个列表项占一行，项内部不要换行
- 引用格式：[论文名](url)，不用加粗或代码块
- **图表引用格式**：图片/表格引用必须嵌入在句子中间，**禁止单独成段**，只有真正相关的图片才引用
- **段落格式**：段落内容要完整连贯，不要把完整句子拆成列表项

请直接输出润色后的内容："""

        logger.info(f"[IdeaEngine] ====== LLM润色开始 ======")
        logger.info(f"[IdeaEngine] citations_context长度: {len(citations_context)}, media_instructions长度: {len(media_instructions)}")
        logger.info(f"[IdeaEngine] web_file_instruction长度: {len(web_file_instruction)}")
        logger.info(f"[IdeaEngine] polish_prompt总长度: {len(polish_prompt)}")
        logger.info(f"[IdeaEngine] initial_draft长度: {len(initial_draft) if initial_draft else 0}")

        try:
            # LLM润色
            response = await provider.text_chat(
                prompt=polish_prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=32768
            )
            polished = self._extract_text_from_response(response)
            logger.info(f"[IdeaEngine] LLM润色完成，原始输出长度: {len(polished) if polished else 0}")
            if polished:
                logger.info(f"[IdeaEngine] LLM润色输出前500字符: {polished[:500]}")

            if polished and len(polished) > 100:
                logger.info("[IdeaEngine] 润色成功")
                content = polished
            else:
                logger.info("[IdeaEngine] 润色结果过短，使用VLM初始版本")
                content = initial_draft if initial_draft and initial_draft.strip() else ideas_summary

            logger.info(f"[IdeaEngine] 最终周报长度: {len(content)}")
            logger.info(f"[IdeaEngine] ====== LLM润色结束 ======")
            return content, extracted_media, generated_title
        except Exception as e:
            logger.error(f"[IdeaEngine] LLM润色失败: {e}，使用VLM初始版本")
            return initial_draft if initial_draft and initial_draft.strip() else ideas_summary, {"images": [], "tables": []}, topic

    def _build_citations_context(self, knowledge: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
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

        parts: List[str] = []
        extracted_media: Dict[str, Any] = {"images": [], "tables": []}
        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])

        # 本地检索引用
        local_image_idx = 0
        local_table_idx = 0
        if local_results:
            parts.append("## 本地论文检索引用：\n")
            for i, result in enumerate(local_results[:8], 1):  # 最多8条
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
                img_index = None
                img_base64 = None
                img_caption = None
                if image_path:
                    if os.path.exists(image_path):
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
                    else:
                        logger.warning(f"[IdeaEngine] 图片元数据存在但文件缺失: {image_path}")

                # 构建引用（无论图片是否存在都添加引用）
                if arxiv_id:
                    ref_str = f"{paper} (https://arxiv.org/abs/{arxiv_id})"
                else:
                    ref_str = paper
                if img_index:
                    parts.append(f"- {ref_str} (页码: {page}, 相关度: {score:.3f}, 图片: {img_index})\n")
                    if img_base64:
                        parts.append(f"  - 图片说明: {img_caption}\n")
                else:
                    parts.append(f"- {ref_str} (页码: {page}, 相关度: {score:.3f})\n")

                # 检查是否有表格
                table_csv_path = metadata.get("table_csv_path")
                table_png_path = metadata.get("table_png_path")
                table_caption = metadata.get("table_caption", "")

                # 尝试推断 md_path 和 png_path（与 csv 同目录，同名文件）
                table_md_path = ""
                if table_csv_path:
                    table_md_path = table_csv_path.replace(".csv", ".md")
                # 如果 png_path 为空，尝试从 csv_path 推断
                if not table_png_path and table_csv_path:
                    table_png_path = table_csv_path.replace(".csv", ".png")
                    if not os.path.exists(table_png_path):
                        table_png_path = ""  # 推断的路径也不存在，则置空

                if table_csv_path or table_png_path:
                    local_table_idx += 1
                    tbl_index = f"本地表-{local_table_idx}"
                    csv_content = ""
                    if table_csv_path:
                        if not os.path.exists(table_csv_path):
                            logger.warning(f"[IdeaEngine] 表格CSV元数据存在但文件缺失: {table_csv_path}")
                        else:
                            try:
                                with open(table_csv_path, "r", encoding="utf-8") as f:
                                    csv_content = f.read()[:500]  # 限制内容长度
                                extracted_media["tables"].append({
                                    "index": tbl_index,
                                    "csv_path": table_csv_path,
                                    "png_path": table_png_path,
                                    "md_path": table_md_path if os.path.exists(table_md_path) else "",
                                    "caption": table_caption,
                                    "csv_content": csv_content,
                                    "source_paper": paper,
                                    "source_page": page
                                })
                            except Exception as e:
                                logger.warning(f"[IdeaEngine] 读取表格失败 {table_csv_path}: {e}")
                                csv_content = "(无法读取)"
                    if table_png_path and not os.path.exists(table_png_path):
                        logger.warning(f"[IdeaEngine] 表格PNG元数据存在但文件缺失: {table_png_path}")

                    if not any(t["index"] == tbl_index for t in extracted_media["tables"]):
                        extracted_media["tables"].append({
                            "index": tbl_index,
                            "csv_path": table_csv_path or "",
                            "png_path": table_png_path or "",
                            "md_path": table_md_path if os.path.exists(table_md_path) else "",
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
            for i, result in enumerate(web_results[:5], 1):  # 最多5条
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
        if not extracted_media:
            return extracted_media

        local_results = knowledge.get("local_results", []) if knowledge else []

        # 构建 chunk 上下文映射（paper -> chunks）
        chunk_contexts: Dict[str, List[str]] = {}
        for result in local_results:
            paper = result.get("metadata", {}).get("file_name", "")
            text = result.get("text", "")
            if paper and text:
                if paper not in chunk_contexts:
                    chunk_contexts[paper] = []
                chunk_contexts[paper].append(text[:500])  # 限制每个 chunk 长度

        # 增强图片 captions
        for img in extracted_media.get("images", []):
            caption = img.get("caption", "")
            if self._is_simple_caption(caption) and enable_qwen:
                paper = img.get("source_paper", "")
                image_path = img.get("path", "")

                if image_path and os.path.exists(image_path):
                    # 获取相关 chunks
                    related_chunks = chunk_contexts.get(paper, [])[:5]  # 最多5个相关 chunk
                    context_text = "\n".join(related_chunks)

                    # 调用 Qwen 分析图片（生成摘要和相关性判断）
                    qwen_result = await self._generate_image_summary_with_qwen(
                        image_path, context_text
                    )

                    if qwen_result:
                        # 将 VLM 的摘要和相关性判断发给云端 LLM 做最终决定
                        final_decision = await self._final_image_decision(
                            qwen_result=qwen_result,
                            context_text=context_text,
                            original_caption=caption,
                            caption_type="图"
                        )
                        if final_decision and final_decision.strip() == "[SKIP]":
                            logger.info(f"[IdeaEngine] 云端 LLM 认为图片无关，跳过: index={img.get('index')}")
                            img["_skip"] = True  # 标记跳过
                        elif final_decision and final_decision.strip():
                            logger.info(f"[IdeaEngine] 图片 caption 增强: '{caption}' -> '{final_decision}'")
                            img["caption"] = final_decision
                    else:
                        # VLM 失败时保留原 caption
                        logger.info(f"[IdeaEngine] VLM 分析失败，保留原 caption: index={img.get('index')}")

        # 增强表格 captions
        for tbl in extracted_media.get("tables", []):
            caption = tbl.get("caption", "")
            if self._is_simple_caption(caption) and enable_qwen:
                paper = tbl.get("source_paper", "")
                csv_path = tbl.get("csv_path", "")
                csv_content = tbl.get("csv_content", "")

                if csv_content:
                    # 1. 使用 CSV 内容 + chunks 作为上下文
                    related_chunks = chunk_contexts.get(paper, [])[:3]
                    context_text = f"表格内容:\n{csv_content[:300]}\n\n相关上下文:\n" + "\n".join(related_chunks)

                    # 2. 调用 LLM 直接润色（表格不需要 Qwen 分析）
                    polished = await self._polish_caption_with_llm(
                        caption, context_text, caption_type="表"
                    )

                    if polished and polished.strip() == "[SKIP]":
                        logger.info(f"[IdeaEngine] LLM 认为表格描述无意义，跳过: index={tbl.get('index')}")
                        tbl["_skip"] = True
                    elif polished:
                        logger.info(f"[IdeaEngine] 表格 caption 增强: '{caption}' -> '{polished}'")
                        tbl["caption"] = polished

        return extracted_media

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
        try:
            # 导入 LlamaCppVLMProvider 相关函数
            try:
                from .llama_cpp_vlm_provider import (
                    get_llama_cpp_vlm_provider,
                    get_cached_llama_cpp_provider,
                    init_llama_cpp_vlm_provider,
                    LlamaCppVLMProvider
                )
            except ImportError as e:
                logger.warning(f"[IdeaEngine] 无法导入 LlamaCppVLMProvider: {e}")
                return None

            # 优先复用已初始化的单例
            vlm_provider = get_cached_llama_cpp_provider()
            if vlm_provider is None:
                logger.info("[IdeaEngine] LlamaCppVLMProvider 未初始化，尝试初始化...")
                # 使用默认路径（llama_cpp_vlm_provider.py 内部会自动下载模型）
                model_dir = os.path.join(os.path.dirname(__file__), "models", "Qwen3.5-9B-GGUF")
                model_path = os.path.join(model_dir, "Qwen3.5-9B-UD-Q4_K_XL.gguf")
                mmproj_path = os.path.join(model_dir, "mmproj-BF16.gguf")

                vlm_provider = init_llama_cpp_vlm_provider(
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    n_ctx=4096,
                    n_gpu_layers=99,
                    max_tokens=512,
                    temperature=0.3
                )
                await vlm_provider.initialize()

            # ========== 策略1：优先提取文字 ==========
            text_extract_prompt = """请只提取这张图片中的所有文字内容，不要进行任何视觉描述。

要求：
1. 按原文顺序输出所有文字
2. 保留原文的换行和段落结构
3. 忽略图片中的图表、图像等非文字内容
4. 如果没有文字，输出"（图片中无文字）"

输出格式：
TEXT: <提取的文字内容>"""

            text_response = await vlm_provider.text_chat(
                prompt=text_extract_prompt,
                image_urls=[image_path],
                temperature=0.3,
                max_tokens=512
            )

            extracted_text = ""
            if text_response and hasattr(text_response, 'content'):
                content = text_response.content.strip()
                # 解析 TEXT: 格式
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('TEXT:'):
                        extracted_text = line[5:].strip()
                        break

            # 判断提取的文字是否有意义（排除"无文字"提示）
            has_meaningful_text = (
                extracted_text
                and extracted_text != "（图片中无文字）"
                and len(extracted_text) >= 10
            )

            if has_meaningful_text:
                logger.info(f"[IdeaEngine] Qwen 文字提取成功: {extracted_text[:60]}...")
                return {
                    "summary": extracted_text,
                    "is_relevant": True  # 有文字的图片默认认为相关
                }

            # ========== 策略2：无文字时进行视觉分析 ==========
            logger.info(f"[IdeaEngine] 图片无文字，进行视觉分析: {image_path}")

            visual_prompt = f"""请分析这张图片。

上下文内容（图片所在论文中的相关段落）：
{context_text[:1000] if context_text else '无'}

请仔细观察图片的视觉内容，判断：
1. 图片的主题和类型（散点图、特征图、热力图、渲染图、流程图、实物照片等）
2. 图片是否与上面的上下文内容相关

然后按以下格式输出（只输出这两行，不要加任何前缀或解释）：
SUMMARY: <图片内容的详细摘要>
RELEVANT: <yes 或 no>

判断标准：
- 如果图片内容与上下文主题相关（例如上下文讨论3DGS，图片也是3DGS相关的图），RELEVANT: yes
- 如果图片与上下文完全无关（例如上下文讨论某个方法的效果，图片却是完全不相关的特征图/渲染图），RELEVANT: no"""

            visual_response = await vlm_provider.text_chat(
                prompt=visual_prompt,
                image_urls=[image_path],
                temperature=0.3,
                max_tokens=512
            )

            if visual_response and hasattr(visual_response, 'content'):
                content = visual_response.content.strip()
                if content:
                    logger.info(f"[IdeaEngine] Qwen 视觉分析成功: {content[:80]}...")
                    # 解析输出
                    summary = ""
                    is_relevant = None
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('SUMMARY:'):
                            summary = line[8:].strip()
                        elif line.startswith('RELEVANT:'):
                            relevant_str = line[9:].strip().lower()
                            is_relevant = relevant_str in ('yes', 'y', 'true')
                    if summary:
                        return {"summary": summary, "is_relevant": is_relevant}

            return None

        except Exception as e:
            logger.warning(f"[IdeaEngine] Qwen 图像分析异常: {e}")
            import traceback
            logger.warning(f"[IdeaEngine] 详细错误: {traceback.format_exc()}")
            return None

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
        provider = self._get_llm_provider()
        if not provider:
            return None

        vlm_summary = qwen_result.get("summary", "")
        vlm_is_relevant = qwen_result.get("is_relevant")

        media_type = "图片" if caption_type == "图" else "表格"
        prompt = f"""给定一个{media_type}的本地模型分析结果和上下文内容，请做最终决定。

本地模型（VLM）分析：
- 摘要：{vlm_summary}
- VLM 判断相关性：{'相关' if vlm_is_relevant else '不相关' if vlm_is_relevant is False else '不确定'}

原始 caption：{original_caption}

上下文（图片所在论文中的相关段落）：
{context_text[:1500] if context_text else '无'}

要求：
1. 综合 VLM 的分析、原始 caption 和上下文内容，做最终判断
2. 如果{media_type}明显与上下文不相关、模糊无法理解、或明显无意义，输出 [SKIP]
3. 如果{media_type}有价值，生成一个简洁的 caption（不超过 100 字符），突出核心信息
4. 只需要输出 caption 或 [SKIP]，不要加任何前缀或解释
"""

        try:
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=128
            )
            result = self._extract_text_from_response(response)
            if result and result.strip():
                return result.strip()
        except Exception as e:
            logger.warning(f"[IdeaEngine] 云端 LLM 决策失败: {e}")
        return None

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
        provider = self._get_llm_provider()
        if not provider:
            return description

        media_type = "图片" if caption_type == "图" else "表格"
        prompt = f"""给定以下{media_type}描述和上下文内容，请生成一个简洁的 caption。

描述：
{description}

上下文：
{context_text[:1500] if context_text else '无'}

要求：
1. 首先判断描述是否与上下文内容相关、是否有意义
2. 如果描述明显不相关、模糊、无法理解，或与上下文内容不符，直接输出 [SKIP]
3. 如果描述有意义，则生成一个简洁的 caption，突出{media_type}的核心信息
4. 只需要输出描述内容，不要包含"图X"、"表X"等编号前缀
5. 不要超过 100 个字符
6. 直接输出 caption 或 [SKIP]，不要加任何前缀或解释
"""
        try:
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=64
            )
            result = self._extract_text_from_response(response)
            if result:
                return result.strip()
        except Exception as e:
            logger.warning(f"[IdeaEngine] LLM caption 润色失败: {e}")

        return description

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
        # 检查是否有图片标记需要审阅
        image_markers_in_content = re.findall(r'<!--\s*INSERT_IMAGE:本地图-(\d+)\s*-->', content)
        if not image_markers_in_content:
            return content, extracted_media

        # 检查 VLM 是否可用
        try:
            from .llama_cpp_vlm_provider import get_cached_llama_cpp_provider
            vlm_provider = get_cached_llama_cpp_provider()
            if vlm_provider is None:
                logger.info("[IdeaEngine] VLM Provider 未初始化，跳过图片审阅")
                return content, extracted_media
        except ImportError:
            logger.warning("[IdeaEngine] 无法导入 LlamaCppVLMProvider，跳过图片审阅")
            return content, extracted_media

        images = extracted_media.get("images", [])
        # 构建 index -> image 的映射
        image_map: Dict[str, Dict] = {}
        for img in images:
            index = img.get("index", "")
            if index:
                image_map[index] = img

        # 解析图片路径（带 fallback）
        def resolve_image_path(img: Dict) -> str:
            path = img.get("path", "")
            if path and os.path.exists(path):
                return path
            # fallback: 从 knowledge 中查找
            if knowledge:
                source_paper = img.get("source_paper", "")
                for result in knowledge.get("local_results", []):
                    metadata = result.get("metadata", {})
                    if (metadata.get("image_path") and
                            source_paper == metadata.get("file_name")):
                        fallback = metadata.get("image_path", "")
                        if fallback and os.path.exists(fallback):
                            return fallback
            return path if path else ""

        # ========== 第一步：按 ## 章节分块 ==========
        # 保留原始标题行位置，用于重建
        lines = content.split('\n')
        sections: List[Dict[str, Any]] = []
        current_section = {"title": "", "lines": [], "start_line": 0}

        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            # 检测章节标题（1-3 级标题：# xxx / ## xxx / ### xxx）
            if re.match(r'^#{1,3}\s+\S', stripped):
                # 保存上一个章节
                if current_section["lines"] or current_section["title"]:
                    sections.append(current_section)
                # 开始新章节
                current_section = {"title": stripped, "lines": [], "start_line": line_idx}
            else:
                current_section["lines"].append(line)

        # 保存最后一个章节
        if current_section["lines"] or current_section["title"]:
            sections.append(current_section)

        # 如果没有章节标题，整个内容作为一个章节
        if len(sections) == 1 and not sections[0]["title"]:
            sections[0]["title"] = "(正文)"

        # ========== 第二步：每章节批量审阅图片 ==========
        all_remove_markers: List[str] = []  # 所有需要删除的标记

        for sec in sections:
            sec_title = sec["title"]
            sec_text = '\n'.join(sec["lines"])

            # 提取本章所有图片标记
            sec_image_markers = re.findall(r'<!--\s*INSERT_IMAGE:本地图-(\d+)\s*-->', sec_text)
            if not sec_image_markers:
                continue

            # 准备本章所有图片的路径和 caption
            sec_images: List[Tuple[str, str, str]] = []  # (index, path, caption)
            for img_num in sec_image_markers:
                index = f"本地图-{img_num}"
                img = image_map.get(index)
                if not img:
                    logger.warning(f"[IdeaEngine] 找不到图片 {index}，将删除其标记")
                    all_remove_markers.append(index)
                    continue
                path = resolve_image_path(img)
                if not path or not os.path.exists(path):
                    logger.warning(f"[IdeaEngine] 图片路径不存在: {path}，将删除 {index} 的标记")
                    all_remove_markers.append(index)
                    continue
                caption = img.get("caption", "")
                sec_images.append((index, path, caption))

            if not sec_images:
                continue

            # 批量调用 VLM 审阅本章节所有图片
            try:
                remove_list = await self._vlm_audit_section(
                    vlm_provider, sec_title, sec_text, sec_images
                )
                all_remove_markers.extend(remove_list)
                logger.info(f"[IdeaEngine] 章节审阅完成 [{sec_title}]，删除 {len(remove_list)} 个标记: {remove_list}")
            except Exception as e:
                logger.warning(f"[IdeaEngine] 章节审阅失败 [{sec_title}]: {e}")

        # ========== 第三步：根据决策删除标记 ==========
        new_content = content
        removed_indices: set = set()
        for index in all_remove_markers:
            # 去重
            if index in removed_indices:
                continue
            # 删除标记
            pattern = rf'<!--\s*INSERT_IMAGE:{re.escape(index)}\s*-->'
            new_content = re.sub(pattern, '', new_content)
            removed_indices.add(index)
            # 标记图片为跳过
            if index in image_map:
                image_map[index]["_skip"] = True

        # ========== 第四步：重建 extracted_media（更新 _skip 状态） ==========
        updated_images = []
        seen_indices = set()
        for img in images:
            idx = img.get("index", "")
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            if idx in image_map:
                updated_images.append(image_map[idx])
            else:
                updated_images.append(img)

        extracted_media["images"] = updated_images

        # 清理由删除引起的孤立引导词（如 "如图" 后面紧跟删除的标记，导致变成 "如图 "）
        new_content = re.sub(r'(\[如图\]|\[如图\s*\])', '', new_content)
        new_content = re.sub(r'(\[表\]|\[表\s*\])', '', new_content)
        new_content = re.sub(r' +', ' ', new_content)

        logger.info(f"[IdeaEngine] 图片审阅完成，共删除 {len(removed_indices)} 个标记")
        return new_content, extracted_media

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
        # 构建图片列表描述
        image_descriptions = []
        for idx, path, caption in images:
            # 从路径中提取文件名作为简短描述
            filename = os.path.basename(path)
            image_descriptions.append(f"  - {idx}: 文件={filename}, caption={caption}")

        images_list_str = '\n'.join(image_descriptions)

        prompt = f"""你是一个学术图片审核助手。你的任务是根据正文内容判断哪些图片不应该出现在这个章节中，并修正正文中的事实错误。

**重要原则：只修改有事实错误的内容，其他部分保持原样不动：不要修改、删除或添加任何标题（##）、列表、代码块等结构标记，不要重新组织句子，不要删减内容。只判断哪些图片应当被删除。**

章节标题：{section_title}

章节正文：
{section_text}

该章节中插入的图片：
{images_list_str}

判断标准：
1. **相关性**：图片内容是否与该章节的主题相关。如果图片与章节讨论的内容完全无关（如：章节讲的是方法A，但图片展示的是方法B的结果），应删除。
2. **事实一致性**：正文对图片的引用是否属实。如果正文说"如图X所示..."但图片内容与正文描述明显不符，应删除该图片。
3. **上下文匹配**：图片是否是该章节论证的一部分。如果图片只是提供了无关的补充信息而非支撑该章节的核心论点，应删除。

操作规则：
- **只删除图片标记，不要修改正文任何内容**
- 如果图片与章节内容高度相关且引用准确，保留
- 如果无法确定，倾向于保留

请按以下 JSON 格式输出（只输出 JSON，不要任何其他文字）：
{{
  "reasoning": "简要说明判断理由",
  "remove": ["本地图-1", "本地图-3"],
  "modified": [{{"loc": "位置", "orig": "原句", "corr": "修正后", "reason": "原因"}}]
}}"""

        try:
            # 批量传入所有图片（VLM 可以同时看多张图）
            image_paths = [path for _, path, _ in images]

            response = await vlm_provider.text_chat(
                prompt=prompt,
                image_urls=image_paths,
                temperature=0.1,
                max_tokens=512
            )

            if not (response and hasattr(response, 'content')):
                return []

            result_text = response.content.strip()

            # 解析 JSON
            import json
            # 去掉 markdown 代码块标记（如 ```json ... ``` 或 ``` ... ```）
            cleaned_text = re.sub(r'```(?:json)?\s*', '', result_text.strip())
            cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
            try:
                # 尝试直接解析
                decision = json.loads(cleaned_text)
            except json.JSONDecodeError:
                # 尝试从清洗后的文本中提取 JSON 部分
                json_match = re.search(r'\{.+\}', cleaned_text, re.DOTALL)
                if json_match:
                    try:
                        decision = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        logger.warning(f"[IdeaEngine] VLM 审阅 JSON 解析失败: {result_text[:200]}")
                        return []
                else:
                    logger.warning(f"[IdeaEngine] VLM 审阅结果无法解析: {result_text[:200]}")
                    return []

            remove_list = decision.get("remove", [])
            reasoning = decision.get("reasoning", "")
            modified = decision.get("modified", [])
            if modified:
                for m in modified:
                    logger.info(f"[IdeaEngine] 🔍 修正 [{m.get('loc','')}]: 「{m.get('orig','')}」→「{m.get('corr','')}」 原因: {m.get('reason','')}")
            logger.info(f"[IdeaEngine] VLM 审阅理由: {reasoning[:200]}")

            # 验证返回的列表只包含有效索引
            valid_indices = {idx for idx, _, _ in images}
            validated = []
            for item in remove_list:
                if item in valid_indices:
                    validated.append(item)
                else:
                    logger.warning(f"[IdeaEngine] VLM 返回了无效的图片编号: {item}")

            return validated

        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 章节审阅异常: {e}")
            return []

    async def _cleanup_content_for_feishu(self, content: str) -> str:
        """
        使用 LLM 清理内容中因媒体引用被跳过而导致的断句问题

        Args:
            content: 原始内容（可能包含空引用如"如图 所示"）

        Returns:
            str: 清理后的内容
        """
        provider = self._get_llm_provider()
        if not provider:
            return content

        cleanup_prompt = f"""请检查以下文本，修复因图片/表格引用被移除而导致的断句问题。

问题示例：
- "如图 所示"（图片被跳过，引用为空）
- "表 可见"（表格被跳过，引用为空）
- "如图 描述了..."（引用为空，只剩引导词）

修复要求：
1. 如果一句话中的图片/表格引用为空导致语句不通顺，重新组织该句或删除该引用
2. **重要**：保持原文中的所有 markdown 格式和引用标记不变，特别是：
   - `## xxx` 等标题标记必须原样保留，**不要删除或替换**
   - `[图X]`、`[表X]` 格式的引用（如 `[表2]`）必须原样保留，不要修改
   - `**加粗**`、`*斜体*`、`***加粗斜体***` 等格式必须原样保留
   - `[论文标题](url)` 格式的链接必须原样保留，**不要用代码块包裹**
   - **`$公式$` 和 `$$公式$$` 必须原样保留，不要修改其中的任何字符，不要重新组织包含公式的句子**
   - 不要修改括号内的内容
3. 只输出修复后的文本，不要添加任何说明

待修复文本：
---
{content}
---

修复后的文本："""

        # 检查是否需要清理（避免无谓的 LLM 调用改变内容）
        needs_cleanup = any(pattern in content for pattern in ['如图', '表', '图'])
        if not needs_cleanup:
            return content

        # 如果内容包含 LaTeX 公式，跳过 LLM 清理（LLM 可能误改公式格式）
        if '$' in content:
            logger.info("[IdeaEngine] 跳过 LLM 内容清理（内容含 LaTeX 公式）")
            return content

        try:
            response = await provider.text_chat(
                prompt=cleanup_prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=4096
            )
            result = self._extract_text_from_response(response)
            if result:
                result = result.strip()
                logger.info(f"[IdeaEngine] LLM 内容清理完成，长度: {len(result)}")
                return result
        except Exception as e:
            logger.warning(f"[IdeaEngine] LLM 内容清理失败: {e}")

        return content

    def debug_media_captions(self, knowledge: Dict[str, Any]) -> str:
        """
        调试函数：统计图片/表格的 caption 提取情况

        Args:
            knowledge: 知识检索结果

        Returns:
            str: 统计报告
        """
        local_results = knowledge.get("local_results", [])

        images_with_desc = 0
        images_without_desc = 0
        tables_with_desc = 0
        tables_without_desc = 0

        report_lines = ["\n📊 **媒体 Caption 提取统计**\n"]

        for result in local_results:
            metadata = result.get("metadata", {})
            paper = metadata.get("file_name", "Unknown")

            # 检查图片 caption
            img_caption = metadata.get("image_caption", "")
            img_path = metadata.get("image_path", "")
            if img_path:
                if img_caption and len(img_caption) > 15:  # 有实际描述（超过15字符）
                    images_with_desc += 1
                else:
                    images_without_desc += 1
                    caption_display = img_caption if img_caption else "(无caption)"
                    report_lines.append(f"  ❌ 图片缺失描述: {paper} | caption: {caption_display}")

            # 检查表格 caption
            tbl_caption = metadata.get("table_caption", "")
            tbl_path = metadata.get("table_csv_path", "") or metadata.get("table_png_path", "")
            if tbl_path:
                if tbl_caption and len(tbl_caption) > 10:  # 有实际描述
                    tables_with_desc += 1
                else:
                    tables_without_desc += 1
                    caption_display = tbl_caption if tbl_caption else "(无caption)"
                    report_lines.append(f"  ❌ 表格缺失描述: {paper} | caption: {caption_display}")

        total_images = images_with_desc + images_without_desc
        total_tables = tables_with_desc + tables_without_desc

        report_lines.insert(1, f"  📷 图片: {images_with_desc}/{total_images} 有详细描述")
        report_lines.insert(2, f"  📋 表格: {tables_with_desc}/{total_tables} 有详细描述")

        if images_without_desc == 0 and tables_without_desc == 0:
            report_lines.append("  ✅ 所有媒体都有完整的 caption 描述！")
        else:
            missing = images_without_desc + tables_without_desc
            report_lines.append(f"  ⚠️ 共有 {missing} 个媒体缺失详细描述")

        return "\n".join(report_lines)

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
        blocks: List[Dict[str, Any]] = []
        pending_images: List[Dict[str, Any]] = []  # 待上传的图片列表

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

        # 添加图片说明文本块（图片块由 9.2 节的图片处理循环单独创建，避免重复）
        for img in images:
            # 跳过被标记为不匹配的图片
            if img.get("_skip"):
                logger.info(f"[IdeaEngine] 跳过不匹配的图片: index={img.get('index')}")
                continue

            # 提取图片编号
            img_index = img.get("index", "")
            match = re.search(r'-(\d+)$', img_index)
            num = match.group(1) if match else img_index
            caption_text = img.get('caption', '')
            source = img.get('source_paper', '')
            page = img.get('source_page', '')

            # 格式：图1: <caption> (来源: xxx, 页码: xx)
            # 前缀"图1:"加粗，后面内容正常
            blocks.append({
                "blockType": "text",
                "options": {
                    "text": {
                        "textStyles": [
                            {"text": f"图{num}: ", "style": {"bold": True}},
                            {"text": f"{caption_text} (来源: {source}, 页码: {page})", "style": {}}
                        ]
                    }
                }
            })
            # 注意：不要在这里创建图片块！图片块由 9.2 节的图片处理循环通过 batch_create_feishu_blocks 单独创建
            # 记录待上传图片信息（供 9.2 节使用）
            if img.get("base64") or img.get("path"):
                pending_images.append({
                    "path": img.get("path", ""),
                    "base64": img.get("base64", ""),
                    "caption": img.get("caption", "")
                })
            logger.info(f"[IdeaEngine] 记录图片信息: {img.get('index')}, path={img.get('path', '')}")

        # 表格在 create_feishu_document 中单独处理（根据 table_format 选择 csv/png/md）
        # 这里只需记录表格信息即可（跳过标记为 _skip 的表格）
        for tbl in tables:
            if tbl.get("_skip"):
                logger.info(f"[IdeaEngine] 跳过不匹配的表格: index={tbl.get('index')}")
                continue
            logger.info(f"[IdeaEngine] 记录表格信息: {tbl.get('index')}, caption={tbl.get('caption', '')}")

        return blocks, pending_images

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
        if not extracted_media:
            return content

        images = extracted_media.get("images", [])
        tables = extracted_media.get("tables", [])

        # 构建 index -> caption 的映射，同时收集被跳过的图片
        image_refs: Dict[str, str] = {}
        skipped_image_indices: set = set()  # 被跳过的图片 index 集合
        for img in images:
            # 跳过被标记为不匹配的图片
            if img.get("_skip"):
                index = img.get("index", "")
                if index:
                    skipped_image_indices.add(index)
                continue
            index = img.get("index", "")
            caption = img.get("caption", "")
            if index and caption:
                # 提取序号（如 "本地图-1" -> "1"）
                match = re.search(r'-(\d+)$', index)
                if match:
                    num = match.group(1)
                    # 主 body 引用用 [图X] 格式（_parse_inline_styles 会转换为加粗）
                    image_refs[index] = f"[图{num}]"

        table_refs: Dict[str, str] = {}
        skipped_table_indices: set = set()  # 被跳过的表格 index 集合
        for tbl in tables:
            # 跳过被标记为不匹配的表格
            if tbl.get("_skip"):
                index = tbl.get("index", "")
                if index:
                    skipped_table_indices.add(index)
                continue
            index = tbl.get("index", "")
            caption = tbl.get("caption", "")
            if index and caption:
                match = re.search(r'-(\d+)$', index)
                if match:
                    num = match.group(1)
                    # 主 body 引用用 [表X] 格式（_parse_inline_styles 会转换为加粗）
                    table_refs[index] = f"[表{num}]"

        def replace_image_marker(match):
            marker = match.group(0)
            # 提取标记中的 index（如 "本地图-1"）
            idx_match = re.search(r'本地图-(\d+)', marker)
            if idx_match:
                full_key = f"本地图-{idx_match.group(1)}"
                # 如果图片被标记为跳过，不显示引用
                if full_key in skipped_image_indices:
                    return ""
                if full_key in image_refs:
                    return image_refs[full_key]
            # 回退：只保留标记内的序号
            if idx_match:
                return f"[图{idx_match.group(1)}]"
            return ""

        def replace_table_marker(match):
            marker = match.group(0)
            idx_match = re.search(r'本地表-(\d+)', marker)
            if idx_match:
                full_key = f"本地表-{idx_match.group(1)}"
                # 如果表格被标记为跳过，不显示引用
                if full_key in skipped_table_indices:
                    return ""
                if full_key in table_refs:
                    return table_refs[full_key]
            if idx_match:
                return f"[表{idx_match.group(1)}]"
            return ""

        # 替换图片标记
        content = re.sub(r'<!--\s*INSERT_IMAGE:[^>]+-->', replace_image_marker, content)
        # 替换表格标记
        content = re.sub(r'<!--\s*INSERT_TABLE:[^>]+-->', replace_table_marker, content)

        # 清理冗余描述（如"见了"、"如图"等引导词后面只剩下标点的情况）
        content = re.sub(r'[见了如图见下图见上表见表]\s*\[\]', '', content)  # 移除非预期的空引用
        content = re.sub(r'\[\]\s*', '', content)  # 移除空引用
        content = re.sub(r' +', ' ', content)  # 只替换多余空格，保留换行符
        return content.strip()

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

    async def analyze_topic(self, topic: str, depth: str = "standard") -> Optional[TopicAnalysis]:
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
                        src_metadata = src.get("metadata", {})
                        local_results.append({
                            "text": src.get("text", ""),
                            "paper": src_metadata.get("file_name", "Unknown"),
                            "page": str(src_metadata.get("page", "")),
                            "score": src.get("score", 0.0),
                            "embedding": src.get("embedding", None),  # 存储 embedding 用于相似度计算
                            "metadata": {
                                "file_name": src_metadata.get("file_name", "Unknown"),
                                "page": str(src_metadata.get("page", "")),
                                # 媒体资源字段（供 _build_citations_context 提取图片/表格）
                                "image_path": src_metadata.get("image_path"),
                                "image_caption": src_metadata.get("image_caption"),
                                "table_csv_path": src_metadata.get("table_csv_path"),
                                "table_png_path": src_metadata.get("table_png_path"),
                                "table_caption": src_metadata.get("table_caption"),
                            }
                        })
            except Exception as e:
                logger.error(f"[IdeaEngine] 本地RAG搜索失败: {e}")

        # 2. 网络搜索 - 已移除，搜索仅在 LLM 润色阶段由 LLM 自行决定

        # 3. 知识融合
        fused_context = self._fuse_knowledge(web_results, local_results)

        logger.info(f"[IdeaEngine] 知识检索完成：local={len(local_results)}条, web={len(web_results)}条, fused_context长度={len(fused_context)}")

        return {
            "web_results": web_results,
            "local_results": local_results,
            "fused_context": fused_context,
            "stats": {
                "web_count": len(web_results),
                "local_count": len(local_results)
            }
        }

    async def _filter_figures_by_relevance(
        self,
        local_results: List[Dict[str, Any]],
        relevance_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        第一阶段图表预过滤：只使用召回 chunks 自带的图片，VLM 提取描述

        筛选策略：
        1. 从 local_results 提取每条 chunk 自带的 image_path
        2. VLM 提取图片文字描述
        3. 返回所有图片及其描述，由后续润色阶段 LLM 根据整体上下文判断是否保留

        Args:
            local_results: RAG 检索结果列表
            relevance_threshold: 相关性阈值，默认 0.5

        Returns:
            List[Dict]: 包含图片路径和 VLM 描述的列表，供润色阶段 LLM 判断
        """
        logger.info(f"[IdeaEngine] 第一阶段图表预过滤开始，输入 {len(local_results)} 条结果")

        # Step 1: 获取 VLM provider
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            logger.warning("[IdeaEngine] VLM 不可用，跳过图表过滤，返回空列表")
            return []

        # Step 2: 收集所有 chunks 自带的图片
        chunk_images: List[Dict[str, Any]] = []
        for result in local_results:
            metadata = result.get("metadata", {})
            image_path = metadata.get("image_path", "")
            if not image_path:
                continue

            # 获取关联的 chunk 文本
            chunk_text = result.get("text", "")
            paper = result.get("paper", metadata.get("file_name", "Unknown"))

            # 提取页码
            page_match = re.search(r'(\d+)-Figure', Path(image_path).name)
            page_num = page_match.group(1) if page_match else metadata.get("page", "")

            chunk_images.append({
                "image_path": image_path,
                "chunk_text": chunk_text,
                "paper": paper,
                "page": page_num,
                "result": result
            })

        if not chunk_images:
            logger.warning("[IdeaEngine] 没有找到 chunk 自带的图片，跳过图表过滤")
            return []

        logger.info(f"[IdeaEngine] 找到 {len(chunk_images)} 张 chunk 自带的图片")

        # Step 3: VLM 提取每张图片的文字描述
        filtered_images = []
        for idx, img_info in enumerate(chunk_images, 1):
            image_path = img_info["image_path"]
            chunk_text = img_info["chunk_text"]
            paper = img_info["paper"]

            # VLM 提取图片中的文字
            img_description = await self._extract_text_from_image(vlm_provider, image_path)
            logger.info(f"[IdeaEngine] 图片描述提取 [{idx}/{len(chunk_images)}]: {Path(image_path).name} -> {img_description[:100] if img_description else '(无文字)'}...")

            # 返回所有图片，描述作为上下文传给润色阶段 LLM
            filtered_images.append({
                "image_path": image_path,
                "image_caption": img_description or Path(image_path).name,  # VLM 提取的描述
                "image_description": img_description,  # 保留原始描述供 LLM 判断
                "image_score": 0.5,  # 初始分数，润色阶段 LLM 会重新评估
                "text_score": 0.5,
                "caption_richness": 1.0 if img_description else 0.3,
                "paper": paper,
                "page": img_info["page"],
                "text": chunk_text,
                "result": img_info["result"]
            })

        logger.info(f"[IdeaEngine] 第一阶段图表预过滤完成，共 {len(chunk_images)} 张图片，描述已提取")
        return filtered_images

    async def _extract_text_from_image(self, vlm_provider, image_path: str) -> str:
        """
        使用 VLM 从图片中提取文字描述

        Args:
            vlm_provider: VLM provider
            image_path: 图片路径

        Returns:
            str: 图片中的文字描述
        """
        prompt = """请仔细阅读这张学术图片，提取图片中所有可见的文字内容，包括：
1. 图表标题和副标题
2. 坐标轴标签和刻度
3. 图例说明
4. 公式和符号
5. 表格内容
6. 任何其他可见文字

请直接输出提取的文字，不要解释。如果图片中没有文字或文字无法辨认，请输出"无文字"。"""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                image_urls=[image_path],
                temperature=0.1,
                max_tokens=512
            )

            if response and hasattr(response, 'content'):
                return response.content.strip()
            return ""
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 提取文字失败: {image_path}, {e}")
            return ""

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        简单的文字相似度计算（基于关键词重叠）

        Args:
            text1: 图片提取文字
            text2: chunks 文本

        Returns:
            float: 相似度 0.0 ~ 1.0
        """
        if not text1 or not text2 or text1 == "无文字":
            return 0.0

        # 简单分词
        words1 = set(re.findall(r'[\w]+', text1.lower()))
        words2 = set(re.findall(r'[\w]+', text2.lower()))

        # 移除停用词
        stopwords = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "with", "without", "from", "by"}
        words1 = words1 - stopwords
        words2 = words2 - stopwords

        if not words1:
            return 0.0

        # 计算 Jaccard 相似度
        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _write_search_to_temp_file(self, web_results: List[Dict]) -> str:
        """
        将网络搜索结果写入临时文件，供 LLM 读取

        Args:
            web_results: 搜索结果列表

        Returns:
            str: 临时文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"/tmp/paperrag_search_{timestamp}.json"

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(web_results, f, ensure_ascii=False, indent=2)
            logger.info(f"[IdeaEngine] 搜索结果已写入临时文件: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"[IdeaEngine] 写入搜索结果到临时文件失败: {e}")
            return ""

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
        try:
            # API Token - 从 mcp_server.json 读取
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            try:
                with open(mcp_config_path, "r", encoding="utf-8") as f:
                    mcp_config = json.load(f)
                api_token = mcp_config.get("mcpServers", {}).get("BrightData", {}).get("env", {}).get("API_TOKEN", "")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                return {"success": False, "error": f"无法读取配置: {e}"}

            if not api_token:
                return {"success": False, "error": "BrightData API Token 未配置"}

            # 启动 Bright Data MCP 服务器
            env = {**os.environ, "API_TOKEN": api_token}
            proc = await asyncio.create_subprocess_exec(
                "npx", "@brightdata/mcp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # 构建请求
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }

            request_str = json.dumps(rpc_request) + "\n"
            logger.info(f"[IdeaEngine] Bright Data MCP 调用: {tool_name}, 参数: {json.dumps(arguments)[:200]}")

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=request_str.encode()),
                    timeout=timeout
                )

                # 关闭进程
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (ProcessLookupError, asyncio.TimeoutError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass

                if stderr:
                    stderr_text = stderr.decode()
                    if stderr_text and "Error" in stderr_text:
                        logger.warning(f"[IdeaEngine] Bright Data stderr: {stderr_text[:200]}")

                if stdout:
                    stdout_text = stdout.decode().strip()
                    # 尝试解析第一行有效的 JSON（可能有日志等额外输出）
                    response = None
                    for line in stdout_text.split('\n'):
                        line = line.strip()
                        if line and line.startswith('{'):
                            try:
                                response = json.loads(line)
                                break
                            except json.JSONDecodeError:
                                continue
                    if response is None:
                        # 尝试整体解析（某些情况下是单行）
                        try:
                            response = json.loads(stdout_text)
                        except json.JSONDecodeError as e:
                            logger.warning(f"[IdeaEngine] JSON 解析失败: {e}, 内容: {stdout_text[:200]}")
                            return {"success": False, "error": f"JSON 解析失败: {e}"}
                    content = response.get("result", {}).get("content", [])

                    if content and len(content) > 0:
                        text = content[0].get("text", "")
                        if text:
                            # 尝试解析为 JSON
                            try:
                                data = json.loads(text)
                                return {"success": True, "data": data}
                            except json.JSONDecodeError:
                                # 返回原始文本（如 Markdown）
                                return {"success": True, "data": text}

                    return {"success": True, "data": None}

            except asyncio.TimeoutError:
                logger.warning(f"[IdeaEngine] Bright Data MCP 调用超时: {tool_name}")
                return {"success": False, "error": "调用超时"}

        except Exception as e:
            logger.error(f"[IdeaEngine] Bright Data MCP 调用失败: {e}")
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "未知错误"}

    async def _search_web(self, queries: List[str], top_k: int) -> List[Dict]:
        """通过网络搜索获取信息（通过Bright Data MCP）"""
        results = []

        try:
            for query in queries[:5]:
                result = await self._call_brightdata_mcp_tool(
                    tool_name="search_engine",
                    arguments={
                        "query": query,
                        "num_results": top_k,
                        "source": "web"
                    }
                )

                if result.get("success"):
                    data = result.get("data", {})
                    if isinstance(data, dict):
                        organic = data.get("organic", [])
                        for item in organic:
                            results.append({
                                "title": item.get("title", ""),
                                "url": item.get("link", ""),
                                "snippet": item.get("description", "")
                            })
                else:
                    logger.warning(f"[IdeaEngine] 搜索失败: {query}, 错误: {result.get('error')}")

        except Exception as e:
            logger.error(f"[IdeaEngine] Bright Data搜索失败: {e}")

        return results

    async def _scrape_as_markdown(self, url: str) -> Dict[str, Any]:
        """
        抓取单个页面为 Markdown

        Args:
            url: 网页 URL

        Returns:
            Dict 包含 success, markdown 内容或 error
        """
        result = await self._call_brightdata_mcp_tool(
            tool_name="scrape_as_markdown",
            arguments={"url": url}
        )

        if result.get("success"):
            return {
                "success": True,
                "markdown": result.get("data", ""),
                "url": url
            }
        return {
            "success": False,
            "error": result.get("error", "抓取失败")
        }

    async def _scrape_batch_markdown(self, urls: List[str]) -> Dict[str, Any]:
        """
        批量抓取页面为 Markdown

        Args:
            urls: URL 列表（最多5个）

        Returns:
            Dict 包含 success, results 列表或 error
        """
        urls = urls[:5]  # 最多5个
        result = await self._call_brightdata_mcp_tool(
            tool_name="scrape_batch",
            arguments={"urls": urls}
        )

        if result.get("success"):
            data = result.get("data", "")
            return {
                "success": True,
                "results": data,  # 可能是字符串或结构化数据
                "urls": urls
            }
        return {
            "success": False,
            "error": result.get("error", "批量抓取失败")
        }

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
        arguments = {
            "query": query,
            "num_results": num_results,
            "country": country
        }
        if intent:
            arguments["intent"] = intent
        if kwargs.get("language"):
            arguments["language"] = kwargs["language"]
        if kwargs.get("start_date"):
            arguments["start_date"] = kwargs["start_date"]
        if kwargs.get("end_date"):
            arguments["end_date"] = kwargs["end_date"]

        result = await self._call_brightdata_mcp_tool(
            tool_name="discover",
            arguments=arguments
        )

        if result.get("success"):
            data = result.get("data", {})
            # discover 返回的是 scored_results 格式
            if isinstance(data, dict):
                results = data.get("results", []) or data.get("scored_results", [])
                return {
                    "success": True,
                    "results": results,
                    "query": query
                }
            return {"success": True, "results": [], "query": query}
        return {
            "success": False,
            "error": result.get("error", "智能搜索失败")
        }

    async def _search_engine_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量搜索引擎搜索

        Args:
            queries: 查询列表，每个包含 query, engine, geo_location 等

        Returns:
            Dict 包含 success, results 列表或 error
        """
        queries = queries[:5]  # 最多5个
        result = await self._call_brightdata_mcp_tool(
            tool_name="search_engine_batch",
            arguments={"queries": queries}
        )

        if result.get("success"):
            data = result.get("data", {})
            return {
                "success": True,
                "results": data,
                "queries": queries
            }
        return {
            "success": False,
            "error": result.get("error", "批量搜索失败")
        }

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
        try:
            provider_manager = getattr(self.context, 'provider_manager', None)
            if not provider_manager:
                return {"success": False, "error": "provider_manager 不可用"}

            llm_tools = getattr(provider_manager, 'llm_tools', None)
            if not llm_tools:
                return {"success": False, "error": "llm_tools 不可用"}

            func_list = getattr(llm_tools, 'func_list', [])
            target_tool = None

            # 查找 arxiv 工具（支持别名映射）
            alias_map = {
                "search_arxiv": "search_papers",
                "get_paper": "read_paper",
                "get_abstract": "get_abstract",
                "semantic_search": "semantic_search",
                "download_paper": "download_paper",
            }
            actual_name = alias_map.get(tool_name, tool_name)

            for tool in func_list:
                if hasattr(tool, 'name') and tool.name == actual_name:
                    target_tool = tool
                    break

            if not target_tool:
                arxiv_tools = [t.name for t in func_list if hasattr(t, 'name')]
                logger.warning(f"[IdeaEngine] 未找到 arxiv 工具 '{actual_name}'，可用: {arxiv_tools}")
                return {"success": False, "error": f"未找到 arxiv 工具: {actual_name}"}

            # 检查工具类型
            is_mcp = hasattr(target_tool, 'mcp_server_name')
            logger.info(f"[IdeaEngine] 工具类型: {'MCP' if is_mcp else 'Native'}, mcp_server: {getattr(target_tool, 'mcp_server_name', 'N/A')}")
            logger.info(f"[IdeaEngine] 工具属性: {[a for a in dir(target_tool) if not a.startswith('_')]}")

            logger.info(f"[IdeaEngine] 调用 arxiv 工具: {target_tool.name}, 参数: {arguments}")

            # 创建 ctx_wrapper
            ctx_wrapper = ContextWrapper(context=self.context)
            is_mcp = hasattr(target_tool, 'mcp_server_name')

            result = None
            error_msg = None

            # MCP 工具调用方式
            if is_mcp:
                try:
                    result = await target_tool.call(ctx_wrapper, **arguments)
                except Exception as e:
                    error_msg = f"MCP调用失败: {e}"
            else:
                # Native 工具 - 使用 handler
                handler = getattr(target_tool, 'handler', None)
                if handler:
                    try:
                        # handler 签名是 handler(event, query, top_k)
                        # event 需要作为第一个参数传入
                        result = await handler(ctx_wrapper, **arguments)
                    except Exception as e:
                        error_msg = f"Native handler调用失败: {e}"
                else:
                    error_msg = "Native工具无handler"

            if error_msg and result is None:
                return {"success": False, "error": error_msg}

            # 解析结果
            assert result is not None
            if hasattr(result, 'content') and result.content:
                text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                return {"success": True, "data": text}
            elif hasattr(result, 'text'):
                return {"success": True, "data": result.text}
            else:
                return {"success": True, "data": str(result)}

        except Exception as e:
            logger.error(f"[IdeaEngine] arxiv MCP 工具调用失败: {e}")
            return {"success": False, "error": str(e)}

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
        import re

        # 预编译 regex（避免每次迭代重新编译）
        _PATTERNS = (
            ("search", re.compile(r'<search>([^<]+)</search>')),
            ("discover", re.compile(r'<discover>([^<]+)</discover>')),
            ("scrape", re.compile(r'<scrape>([^<]+)</scrape>')),
            ("batch_search", re.compile(r'<batch_search>(\[[^\]]+\])</batch_search>')),
            ("scrape_batch", re.compile(r'<scrape_batch>(\[[^\]]+\])</scrape_batch>')),
        )

        all_results: List[Dict[str, Any]] = []
        iteration = 0
        current_text = text

        while iteration < max_iterations:
            iteration += 1
            found_any = False

            # 单次遍历处理所有标签类型
            for tool_type, pattern in _PATTERNS:
                for match in pattern.finditer(current_text):
                    found_any = True
                    content = match.group(1).strip()

                    if tool_type == "search":
                        logger.info(f"[IdeaEngine] 执行 search: {content}")
                        result = await self._call_brightdata_mcp_tool(
                            "search_engine", {"query": content, "num_results": 5, "source": "web"}
                        )
                        formatted = self._format_search_result(result)
                        all_results.append({"type": "search", "query": content, "result": formatted})
                        current_text = current_text.replace(match.group(0), f"\n[搜索:{content}]\n{formatted}\n", 1)

                    elif tool_type == "discover":
                        parts = content.split("|")
                        query, intent, country = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else ""), (parts[2].strip() if len(parts) > 2 else "US")
                        logger.info(f"[IdeaEngine] 执行 discover: {query}")
                        result = await self._discover_search(query=query, intent=intent, country=country[:2] if len(country) == 2 else "US", num_results=5)
                        formatted = self._format_discover_result(result)
                        all_results.append({"type": "discover", "query": query, "result": formatted})
                        current_text = current_text.replace(match.group(0), f"\n[AI搜索:{query}]\n{formatted}\n", 1)

                    elif tool_type == "scrape":
                        logger.info(f"[IdeaEngine] 执行 scrape: {content}")
                        result = await self._scrape_as_markdown(content)
                        formatted = self._format_scrape_result(result)
                        all_results.append({"type": "scrape", "url": content, "result": formatted})
                        current_text = current_text.replace(match.group(0), f"\n[网页内容]\n{formatted[:1000]}\n", 1)

                    elif tool_type == "batch_search":
                        try:
                            queries = json.loads(content)
                            if isinstance(queries, list) and len(queries) <= 5:
                                logger.info(f"[IdeaEngine] 执行 batch_search: {queries}")
                                result = await self._search_engine_batch([{"query": q, "engine": "google"} for q in queries])
                                all_results.append({"type": "batch_search", "queries": queries, "result": "批量搜索完成"})
                                current_text = current_text.replace(match.group(0), "\n[批量搜索完成]\n", 1)
                        except json.JSONDecodeError:
                            pass

                    elif tool_type == "scrape_batch":
                        try:
                            urls = json.loads(content)
                            if isinstance(urls, list) and len(urls) <= 5:
                                logger.info(f"[IdeaEngine] 执行 scrape_batch: {urls}")
                                result = await self._scrape_batch_markdown(urls)
                                all_results.append({"type": "scrape_batch", "urls": urls, "result": "批量抓取完成"})
                                current_text = current_text.replace(match.group(0), "\n[批量抓取完成]\n", 1)
                        except json.JSONDecodeError:
                            pass

                    elif tool_type == "arxiv_search":
                        logger.info(f"[IdeaEngine] 执行 arxiv_search: {content}")
                        result = await self._call_arxiv_mcp_tool("search_papers", {"query": content, "max_results": 5})
                        formatted = self._format_arxiv_result(result)
                        all_results.append({"type": "arxiv_search", "query": content, "result": formatted})
                        current_text = current_text.replace(match.group(0), f"\n[arXiv搜索:{content}]\n{formatted}\n", 1)

                    elif tool_type == "arxiv_paper":
                        logger.info(f"[IdeaEngine] 执行 arxiv_paper: {content}")
                        result = await self._call_arxiv_mcp_tool("read_paper", {"paper_id": content})
                        formatted = self._format_arxiv_paper_result(result)
                        all_results.append({"type": "arxiv_paper", "arxiv_id": content, "result": formatted})
                        current_text = current_text.replace(match.group(0), f"\n[arXiv论文:{content}]\n{formatted}\n", 1)

            if not found_any:
                break

        return current_text, all_results

    def _format_search_result(self, result: Dict[str, Any]) -> str:
        """格式化搜索结果"""
        if not result.get("success"):
            return f"搜索失败: {result.get('error', '未知错误')}"
        data = result.get("data", {})
        if isinstance(data, dict):
            organic = data.get("organic", [])
            if organic:
                return "\n".join([f"- [{i.get('title','')}]({i.get('link','')})" for i in organic[:3]])
        return "无结果"

    def _format_discover_result(self, result: Dict[str, Any]) -> str:
        """格式化 discover 结果"""
        if not result.get("success"):
            return f"AI搜索失败: {result.get('error', '未知错误')}"
        results_list = result.get("results", [])
        if results_list:
            return "\n".join([f"- [{i.get('title','')}]({i.get('url', i.get('link',''))})" for i in results_list[:3]])
        return "无结果"

    def _format_scrape_result(self, result: Dict[str, Any]) -> str:
        """格式化抓取结果"""
        if not result.get("success"):
            return f"抓取失败: {result.get('error', '未知错误')}"
        # scrape_as_markdown 返回的是 data 字段（不是 markdown）
        markdown = result.get("data") or ""
        if isinstance(markdown, dict):
            # 有时返回的是 JSON 对象
            return str(markdown)
        if not markdown:
            return "空内容"
        return (markdown[:500] + "...") if len(markdown) > 500 else markdown

    def _format_arxiv_result(self, result: Dict[str, Any]) -> str:
        """格式化 arxiv 搜索结果"""
        if not result.get("success"):
            return f"arXiv搜索失败: {result.get('error', '未知错误')}"
        data = result.get("data", "")
        if not data:
            return "无结果"
        # 尝试解析 JSON
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
            if isinstance(parsed, list):
                return "\n".join([f"- [{p.get('title', p.get('name', 'N/A'))}]({p.get('url', p.get('arxiv_url', ''))})" for p in parsed[:3]])
            elif isinstance(parsed, dict):
                papers = parsed.get("papers", []) or parsed.get("results", [])
                return "\n".join([f"- [{p.get('title', 'N/A')}]({p.get('url', p.get('arxiv_url', ''))})" for p in papers[:3]])
        except (json.JSONDecodeError, TypeError):
            pass
        # 如果是原始文本
        if isinstance(data, str) and len(data) > 10:
            return data[:500] + "..." if len(data) > 500 else data
        return "无结果"

    def _format_arxiv_paper_result(self, result: Dict[str, Any]) -> str:
        """格式化 arxiv 论文详情"""
        if not result.get("success"):
            return f"arXiv论文获取失败: {result.get('error', '未知错误')}"
        data = result.get("data", "")
        if not data:
            return "无内容"
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
            if isinstance(parsed, dict):
                title = parsed.get("title", "")
                authors = parsed.get("authors", "")
                abstract = parsed.get("abstract", "")[:300]
                url = parsed.get("url", parsed.get("arxiv_url", ""))
                return f"**{title}**\n作者: {authors}\n\n摘要: {abstract}...\n\n链接: {url}"
        except (json.JSONDecodeError, TypeError):
            pass
        if isinstance(data, str):
            return data[:500] + "..." if len(data) > 500 else data
        return "无内容"

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
        provider = self._get_llm_provider()
        if not provider:
            raise RuntimeError("[IdeaEngine] 无法获取 LLM provider")

        # 获取工具集（只包含 arxiv 和 Bright Data MCP 工具，减少 token 消耗）
        provider_manager = getattr(self.context, 'provider_manager', None)
        tool_set = None
        if provider_manager:
            llm_tools = getattr(provider_manager, 'llm_tools', None)
            if llm_tools:
                full_tool_set = llm_tools.get_full_tool_set()
                # 只保留 arxiv 和 Bright Data 相关工具
                allowed_tools = {
                    'search_papers', 'download_paper', 'list_papers', 'read_paper',
                    'get_abstract', 'semantic_search', 'reindex', 'citation_graph',
                    'watch_topic', 'check_alerts', 'search_engine', 'scrape_as_markdown',
                    'search_engine_batch', 'scrape_batch', 'discover'
                }
                filtered_tools = [t for t in full_tool_set.tools if t.name in allowed_tools]
                if filtered_tools:
                    from astrbot.core.agent.tool import ToolSet
                    tool_set = ToolSet(filtered_tools)
                    logger.info(f"[IdeaEngine] 过滤后工具集: {[t.name for t in tool_set.tools]}")
                else:
                    tool_set = full_tool_set
                    logger.warning("[IdeaEngine] 未找到匹配的 MCP 工具，使用完整工具集")

        if not tool_set:
            raise RuntimeError("[IdeaEngine] 无法获取工具集")

        # 干净的上下文：只包含参考资料，无 AstrBot 人格/配置
        clean_contexts = contexts or []

        # 第一次调用 - 先不使用工具，根据原始 prompt 生成初始内容
        # 这确保 LLM 先遵循格式要求生成内容，而不是急着调用工具
        logger.info("[IdeaEngine] 原生 Agent 第1次调用（生成初始内容，不使用工具）...")

        response = await provider.text_chat(
            prompt=polish_prompt,  # 直接传 prompt
            contexts=clean_contexts,  # 参考资料作为额外上下文
            temperature=0.3,
            max_tokens=32768,  # 允许生成长内容
        )

        # 处理工具调用循环
        final_response = await self._handle_tool_calls_loop(
            provider, response, tool_set, max_iterations, polish_prompt, ideas_summary
        )

        return final_response.completion_text

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
        current_response = initial_response
        iteration = 0
        tool_results_data = []  # [(tc_id, name, result_str)]

        while iteration < max_iterations:
            tool_names = getattr(current_response, 'tools_call_name', []) or []
            tool_args = getattr(current_response, 'tools_call_args', []) or []
            tool_ids = getattr(current_response, 'tools_call_ids', []) or []

            if not tool_names:
                logger.info(f"[IdeaEngine] 工具调用循环结束（共 {iteration} 次）")
                break

            logger.info(f"[IdeaEngine] 第 {iteration + 1} 次工具调用: {tool_names}")

            # 执行工具
            for i, (name, args) in enumerate(zip(tool_names, tool_args)):
                tc_id = tool_ids[i] if i < len(tool_ids) else f"call_{i}"
                try:
                    result = await self._execute_llm_tool(name, args)
                    tool_results_data.append((tc_id, name, str(result) if result else ""))
                    logger.info(f"[IdeaEngine] 工具 {name} 执行成功")
                except Exception as e:
                    logger.error(f"[IdeaEngine] 工具 {name} 执行失败: {e}")
                    tool_results_data.append((tc_id, name, f"执行失败: {e}"))

            # 构造 tool_calls_result
            tool_calls_result = self._build_tool_calls_result(
                tool_names, tool_args, tool_ids, tool_results_data
            )

            # 继续获取响应
            current_response = await provider.text_chat(
                prompt="请继续调用工具（如果还需要更多资料）或直接输出最终内容。",
                contexts=[],
                system_prompt=TOOL_CALL_PROMPT,
                func_tool=tool_set,
                tool_calls_result=tool_calls_result,
            )

            iteration += 1

        if iteration >= max_iterations:
            logger.warning(f"[IdeaEngine] 达到最大迭代次数 {max_iterations}")

        # 旧方案的关键：执行完工具后，用一个整合 prompt 明确传入原始 ideas + 当前内容 + 工具结果
        if tool_results_data:
            logger.info(f"[IdeaEngine] 执行了 {len(tool_results_data)} 个工具调用，进行内容整合...")
            results_summary = "\n".join([
                f"[{name}] {str(result)[:200]}" for _, name, result in tool_results_data
            ])
            current_content = current_response.completion_text if current_response else ""

            prompt_len = len(ideas_summary) + len(current_content) + len(results_summary)
            logger.info(f"[IdeaEngine] 整合 prompt 各部分长度: ideas_summary={len(ideas_summary)}, current_content={len(current_content)}, results={len(results_summary)}, 总计={prompt_len}")

            integration_prompt = f"""你之前已经根据原始研究想法撰写了组会周报，并调用工具获取了更多资料。

原始研究想法：
{ideas_summary}

已撰写的当前内容（可能不完整）：
{current_content if current_content else '(暂无内容)'}

工具调用结果：
{results_summary}

请将工具获取的信息整合到当前内容中，补充"相关工作"章节的详细内容。
要求：
1. 保持已有内容不变，只在相应章节补充新信息
2. 使用[论文名](url)格式添加引用
3. 不要重复已有的内容
4. 保持格式要求（图表引用嵌入句子中间等）

请直接输出整合后的完整内容："""

            logger.info("[IdeaEngine] 开始调用整合 LLM...")
            integration_response = await provider.text_chat(
                prompt=integration_prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=16384,
            )
            logger.info("[IdeaEngine] 整合 LLM 调用完成")
            if integration_response and integration_response.completion_text:
                current_response = integration_response
                logger.info(f"[IdeaEngine] 内容整合完成，最终长度: {len(integration_response.completion_text)}")

        return current_response

    def _build_tool_calls_result(
        self,
        tool_names: list[str],
        tool_args: list[dict],
        tool_ids: list[str],
        tool_results: list[tuple[str, str, str]],
    ) -> ToolCallsResult:
        """构造 ToolCallsResult 用于回传给 LLM"""
        # 构造 tool_calls_info (assistant message)
        tool_calls_list = []
        for i, (name, args) in enumerate(zip(tool_names, tool_args)):
            tc_id = tool_ids[i] if i < len(tool_ids) else f"call_{i}"
            tool_calls_list.append({
                "id": tc_id,
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else (args if isinstance(args, str) else json.dumps(args))
                }
            })

        tool_calls_info = AssistantMessageSegment(
            role="assistant",
            content="",
            tool_calls=tool_calls_list
        )

        # 构造 tool_calls_result (tool results)
        tool_result_segments = []
        for tc_id, name, result in tool_results:
            tool_result_segments.append(ToolCallMessageSegment(
                role="tool",
                tool_call_id=tc_id,
                content=str(result) if result else "执行完成",
            ))

        return ToolCallsResult(
            tool_calls_info=tool_calls_info,
            tool_calls_result=tool_result_segments,
        )

    async def _execute_llm_tool(self, tool_name: str, args: dict) -> str:
        """执行单个 LLM 工具调用"""
        # 查找工具
        provider_manager = getattr(self.context, 'provider_manager', None)
        llm_tools = getattr(provider_manager, 'llm_tools', None)
        func_tool = llm_tools.get_func(tool_name) if llm_tools else None

        if not func_tool:
            return f"工具 {tool_name} 不存在"

        try:
            # MCP 工具使用 call() 方法，普通工具使用 handler
            if func_tool.handler:
                result = await func_tool.handler(None, **args)
            else:
                # MCP 工具（如 arxiv、Bright Data）使用 call() 方法
                # 需要创建 ContextWrapper
                from astrbot.core.agent.run_context import ContextWrapper
                ctx = ContextWrapper(context=None, tool_call_timeout=60)
                call_result = await func_tool.call(ctx, **args)
                # call_result 是 mcp.types.CallToolResult，提取文本内容
                if hasattr(call_result, 'content') and call_result.content:
                    first_content = call_result.content[0]
                    if hasattr(first_content, 'text'):
                        result = first_content.text
                    else:
                        result = str(first_content)
                else:
                    result = str(call_result)
            return str(result) if result else "执行完成"
        except Exception as e:
            logger.error(f"[IdeaEngine] 工具执行异常: {tool_name}, {e}")
            return f"执行异常: {e}"

    async def generate_ideas(
        self,
        knowledge_context: str,
        research_domain: str = "",
        num_ideas: int = 3,
        idea_focus: str = "all",
        topic: str = ""
    ) -> List[ResearchIdea]:
        """
        基于知识上下文生成研究想法

        Args:
            knowledge_context: 融合后的知识上下文
            research_domain: 研究领域
            num_ideas: 生成想法数量
            idea_focus: 侧重点 (novelty/feasibility/impact/all)
            topic: 用户原始研究主题/问题

        Returns:
            List[ResearchIdea]: 研究想法列表
        """
        logger.info(f"[IdeaEngine] 生成{num_ideas}个研究想法，topic={topic}")

        # 优先使用本地VLM生成ideas
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

        prompt = f"""基于以下收集的知识上下文，针对用户的研究主题，生成{num_ideas}个研究想法。

**用户研究主题：{topic}**

收集的知识（请仔细阅读，这些是与主题相关的参考资料）：
{knowledge_context[:6000]}

{focus_instruction}

**重要约束**：
- 想法必须与「{topic}」紧密相关，不能偏离主题
- 如果收集的知识中有与主题不相关的内容，请忽略
- 每个想法都要能追溯到上述参考资料中的具体内容
- 不要生成与主题无关的通用性想法

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
            "feasibility": 0.0到1.0之间的浮点数,
            "inspiration_sources": ["灵感来源1", "灵感来源2"]
        }},
        ...
    ],
    "analysis_summary": "对现有工作的分析总结"
}}

请严格按照JSON格式返回，只返回JSON，不要包含其他文字。"""

        logger.info(f"[IdeaEngine] 生成ideas的prompt长度: {len(prompt)}")

        try:
            if vlm_provider:
                # 使用本地VLM生成
                logger.info("[IdeaEngine] 调用VLM text_chat...")
                response = await vlm_provider.text_chat(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=2048
                )

                logger.info(f"[IdeaEngine] VLM响应类型: {type(response)}")
                logger.info(f"[IdeaEngine] VLM响应属性: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")

                # 提取响应文本
                response_text = ""
                if hasattr(response, 'content'):
                    response_text = response.content
                    logger.info(f"[IdeaEngine] 从response.content提取，长度: {len(response_text)}")
                elif isinstance(response, dict):
                    response_text = response.get("content", "") or response.get("text", "")
                    logger.info(f"[IdeaEngine] 从dict提取，长度: {len(response_text)}")
                else:
                    response_text = str(response)
                    logger.info(f"[IdeaEngine] 强制转str，长度: {len(response_text)}")

                logger.info(f"[IdeaEngine] VLM原始响应前200字符: {response_text[:200]}")
            else:
                # Fallback: 使用云端LLM
                logger.info("[IdeaEngine] 使用云端LLM生成ideas")
                provider = self._get_llm_provider()
                if not provider:
                    logger.error("[IdeaEngine] 云端LLM provider也未初始化")
                    return []

                response = await provider.text_chat(
                    prompt=prompt,
                    contexts=[],
                    temperature=0.7,
                    max_tokens=4096
                )

                if hasattr(response, 'result_chain'):
                    chain = getattr(response.result_chain, 'chain', None)
                    if chain and len(chain) > 0:
                        first = chain[0]
                        if hasattr(first, 'get_text'):
                            response_text = first.get_text()
                        elif hasattr(first, 'text'):
                            response_text = first.text
                elif hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, dict):
                    response_text = response.get("content", "") or response.get("text", "")
                else:
                    response_text = str(response)

                logger.info(f"[IdeaEngine] 云端LLM响应长度: {len(response_text)}")

            logger.info(f"[IdeaEngine] 最终响应长度: {len(response_text)}")

            result = self._parse_json_response(response_text)

            if result and "ideas" in result:
                logger.info(f"[IdeaEngine] JSON解析成功，ideas数量: {len(result['ideas'])}")
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
                logger.warning(f"[IdeaEngine] JSON解析失败或无ideas，response前100字符: {response_text[:100]}")
                return []

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
            for i, r in enumerate(web_results, 1):
                parts.append(f"{i}. **{r.get('title', '')}**")
                parts.append(f"   {r.get('snippet', '')}")
                parts.append("")

        # 本地论文
        if local_results:
            parts.append("## 本地论文库\n")
            papers: Dict[str, List[Dict[str, Any]]] = {}
            for r in local_results:
                paper = r.get("paper", "Unknown")
                if paper not in papers:
                    papers[paper] = []
                papers[paper].append(r)

            for paper, chunks in papers.items():
                parts.append(f"### {paper}")
                for chunk in chunks[:5]:
                    text = chunk.get("text", "")
                    if text:
                        parts.append(f"- {text}")
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

    async def test_brightdata_mcp(self, query: str) -> Dict[str, Any]:
        """
        测试 Bright Data MCP 学术搜索功能

        此方法直接调用 Bright Data MCP 进行搜索测试。

        Args:
            query: 搜索查询词

        Returns:
            Dict包含搜索结果或错误信息
        """
        try:
            logger.info(f"[IdeaEngine] 测试 Bright Data MCP 搜索: {query}")

            # 直接调用 Bright Data MCP 搜索
            results = await self._search_web(queries=[query], top_k=10)
            if results:
                return {
                    "success": True,
                    "results": results
                }
            else:
                return {
                    "success": False,
                    "error": "未找到结果或搜索失败"
                }

        except Exception as e:
            logger.error(f"[IdeaEngine] Bright Data MCP 测试失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

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
        test_markdown = """# 一级标题测试

## 二级标题测试

### 三级标题测试

这是一段普通文本。

**这是加粗文本**

*这是斜体文本*

***这是加粗斜体文本***

`行内代码文本`

~~这是删除线文本~~

[链接示例](https://example.com)

- 无序列表项 1
- 无序列表项 2
- 无序列表项 3

1. 有序列表项 1
2. 有序列表项 2
3. 有序列表项 3

---

混合测试：**加粗** 和 *斜体* 和 `代码` 和 ~~删除线~~ 在同一行。

## 图表引用测试

如图[图1]所示，这是图片引用测试。

请参见[表3]的数据。

[图2]和[表1]同时出现在同一行。

## 公式测试

这是$E=mc^2$公式测试。

在$\\alpha$和$\\beta$之间的公式测试。

## 组合测试

如图[图1]所示，$E=mc^2$是重要的公式。根据[表1]的数据显示，$\\alpha$和$\\beta$的关系见[图5]。
"""

        # 1. 获取飞书工具并创建文档
        feishu_tool = self._get_feishu_tool()
        if not feishu_tool:
            return {"success": False, "error": "未找到飞书 MCP 工具"}

        if not folder_token:
            return {"success": False, "error": "需要提供 folder_token"}

        ctx_wrapper = ContextWrapper(context=self.context)

        # 创建文档
        create_result = await feishu_tool.call(
            ctx_wrapper,
            title="Markdown格式测试",
            folderToken=folder_token
        )

        # 解析 document_id
        document_id = None
        if hasattr(create_result, 'content') and create_result.content:
            result_text = getattr(create_result.content[0], 'text', None)
            if result_text:
                try:
                    doc_info = json.loads(result_text)
                    document_id = doc_info.get("document", {}).get("document_id") or doc_info.get("document_id")
                except json.JSONDecodeError:
                    pass

        if not document_id:
            return {"success": False, "error": f"创建文档失败: {repr(create_result)[:200]}"}

        # 2. 转换 Markdown 为飞书块
        blocks = self._markdown_to_feishu_blocks(test_markdown)
        logger.info(f"[Test] 生成 {len(blocks)} 个飞书块")

        # 3. 添加内容块（复用 create_feishu_document 的逻辑）
        provider_manager = getattr(self.context, 'provider_manager', None)
        add_blocks_tool = None
        if provider_manager:
            llm_tools = getattr(provider_manager, 'llm_tools', None)
            if llm_tools:
                for tool in getattr(llm_tools, 'func_list', []):
                    if tool.name == 'batch_create_feishu_blocks':
                        add_blocks_tool = tool
                        break

        if not add_blocks_tool:
            return {"success": False, "error": "未找到 batch_create_feishu_blocks 工具"}

        # 3.1 获取文档的根块 ID（复用 create_feishu_document 的逻辑）
        root_block_id = "0"
        try:
            get_blocks_tool = None
            if provider_manager:
                llm_tools = getattr(provider_manager, 'llm_tools', None)
                if llm_tools:
                    for tool in getattr(llm_tools, 'func_list', []):
                        if tool.name == 'get_feishu_document_blocks':
                            get_blocks_tool = tool
                            break

            if get_blocks_tool:
                blocks_info_result = await get_blocks_tool.call(
                    ctx_wrapper,
                    documentId=document_id
                )
                logger.info(f"[Test] 获取块信息结果: {repr(blocks_info_result)[:500]}")

                if hasattr(blocks_info_result, 'content') and blocks_info_result.content:
                    result_text = getattr(blocks_info_result.content[0], 'text', None)
                    if result_text:
                        try:
                            blocks_data = json.loads(result_text)
                            if isinstance(blocks_data, list):
                                if len(blocks_data) > 0:
                                    first_item = blocks_data[0]
                                    root_block_id = first_item.get('block_id', '0') if isinstance(first_item, dict) else '0'
                            elif isinstance(blocks_data, dict):
                                items = blocks_data.get('data', {}).get('items', []) or blocks_data.get('items', [])
                                if items and len(items) > 0:
                                    root_block_id = items[0].get('block_id', '0')
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.warning(f"[Test] 获取根块 ID 失败: {e}，使用默认值 0")

        logger.info(f"[Test] 最终使用的根块 ID: {root_block_id}")

        # 3.2 添加块
        blocks_result = await add_blocks_tool.call(
            ctx_wrapper,
            documentId=document_id,
            parentBlockId=root_block_id,
            index=0,
            blocks=blocks
        )

        # 4. 检查结果
        blocks_created = 0
        if hasattr(blocks_result, 'isError') and blocks_result.isError:
            error_text = ""
            if hasattr(blocks_result, 'content') and blocks_result.content:
                error_text = getattr(blocks_result.content[0], 'text', str(blocks_result))
            return {"success": False, "error": f"添加块失败: {error_text[:500]}"}
        elif hasattr(blocks_result, 'content') and blocks_result.content:
            result_text = getattr(blocks_result.content[0], 'text', None)
            if result_text:
                try:
                    result_data = json.loads(result_text)
                    blocks_created = result_data.get("totalBlocksCreated", 0)
                except json.JSONDecodeError:
                    return {"success": False, "error": f"解析结果失败: {result_text[:200]}"}

        return {
            "success": True,
            "document_id": document_id,
            "url": f"https://feishu.cn/docx/{document_id}",
            "blocks_created": blocks_created,
            "blocks_count": len(blocks),
            "test_content": test_markdown
        }

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
        try:
            # 0. 预解析本地论文的 arxiv 链接
            if knowledge is not None:
                knowledge = await self._pre_resolve_arxiv_links(knowledge)

            # 1. LLM 润色内容（组会周报学术风格，含引用、媒体标记、相关工作、Benchmark 和研究计划）
            # 返回值：polished_content, extracted_media, generated_title
            polished_content, extracted_media, generated_title = await self._polish_content_for_feishu(ideas, topic, knowledge)
            if not polished_content:
                return {"error": "内容为空", "polished_content": ""}

            # 2. 备用解析（处理 LLM 偶尔使用错误格式的情况）
            polished_content = await self._resolve_references(polished_content, knowledge)

            # 3. 使用本地 VLM 审阅图片与内容的相关性（在标记替换之前进行，依赖原始 INSERT_IMAGE 标记）
            audited_content, extracted_media = await self._audit_media_relevance(
                polished_content, extracted_media, knowledge
            )

            # 3.1 将媒体标记替换为带 caption 的引用格式
            cleaned_content = self._replace_media_markers(audited_content, extracted_media)

            # 3.2 使用 LLM 清理因媒体引用被跳过而导致的断句
            cleaned_content = await self._cleanup_content_for_feishu(cleaned_content)

            # 4. 生成飞书块格式（不含媒体）
            logger.info(f"[IdeaEngine] [DEBUG] 生成飞书块前的内容（前500字）:\n{cleaned_content[:500]}")
            blocks = self._markdown_to_feishu_blocks(cleaned_content)
            logger.info(f"[IdeaEngine] [DEBUG] 生成的块数量: {len(blocks)}，前3块: {blocks[:3]}")

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
                        result_text = cast(Optional[str], getattr(blocks_info_result.content[0], 'text', None))
                        if result_text:
                            try:
                                blocks_data = json.loads(result_text)
                                logger.info(f"[IdeaEngine] 块数据解析成功: {str(blocks_data)[:500]}")
                                # 检查是否是列表格式
                                if isinstance(blocks_data, list):
                                    if len(blocks_data) > 0:
                                        first_item = blocks_data[0]
                                        root_block_id = first_item.get('block_id', '0') if isinstance(first_item, dict) else '0'
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
                            result_text = cast(Optional[str], getattr(blocks_result.content[0], 'text', None))
                            if result_text:
                                try:
                                    result_data = json.loads(result_text)
                                    blocks_created = result_data.get("totalBlocksCreated", 0)
                                    logger.info(f"[IdeaEngine] 成功添加 {blocks_created} 个文本块")
                                except json.JSONDecodeError:
                                    logger.warning(f"[IdeaEngine] 解析文本块结果失败")

                # 9.2 处理图片（三阶段：创建空图片块 → 上传绑定 → 创建居中图表说明）
                images = extracted_media.get("images", [])
                if images and upload_image_tool:
                    logger.info(f"[IdeaEngine] 开始处理 {len(images)} 张图片")
                    current_index = len(blocks)  # 从文本块之后开始

                    for i, img in enumerate(images):
                        # 跳过被标记为不匹配的图片
                        if img.get("_skip"):
                            logger.info(f"[IdeaEngine] 跳过图片: index={img.get('index')}")
                            continue

                        img_path = img.get("path", "")
                        if not img_path or not os.path.exists(img_path):
                            logger.warning(f"[IdeaEngine] 图片不存在: {img_path}")
                            continue

                        caption = img.get("caption", "")
                        # 构建图片序号（如 "本地图-1" -> "图1"）
                        idx_match = re.search(r'-(\d+)$', img.get("index", ""))
                        fig_num = idx_match.group(1) if idx_match else str(i + 1)
                        # 如果 caption 只是简单编号，用 图1 格式；否则用 图1: caption 格式（加粗前缀）
                        simple_pattern = re.match(
                            r'^(Figure|Fig\.|Fig|Table|表|图)\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)?)$',
                            caption.strip() if caption else "",
                            re.IGNORECASE
                        )
                        if simple_pattern:
                            # 简单编号：加粗显示
                            caption_block = {
                                "blockType": "text",
                                "options": {
                                    "text": {
                                        "textStyles": [
                                            {"text": f"图{fig_num}", "style": {"bold": True}}
                                        ],
                                        "align": 2  # 2 = 居中
                                    }
                                }
                            }
                        elif caption:
                            # 有描述：前缀加粗
                            caption_block = {
                                "blockType": "text",
                                "options": {
                                    "text": {
                                        "textStyles": [
                                            {"text": f"图{fig_num}: ", "style": {"bold": True}},
                                            {"text": caption, "style": {}}
                                        ],
                                        "align": 2  # 2 = 居中
                                    }
                                }
                            }
                        else:
                            caption_block = {
                                "blockType": "text",
                                "options": {
                                    "text": {
                                        "textStyles": [
                                            {"text": f"图{fig_num}", "style": {"bold": True}}
                                        ],
                                        "align": 2  # 2 = 居中
                                    }
                                }
                            }

                        # 创建空图片块 + 居中图表说明文本块（一次性创建）
                        image_and_caption_blocks = [
                            {
                                "blockType": "image",
                                "align": 2,  # 2 = 居中
                                "options": {
                                    "image": {}
                                }
                            },
                            caption_block
                        ]

                        if add_blocks_tool is None:
                            logger.warning(f"[IdeaEngine] add_blocks_tool 不可用，跳过图片块[{i}]")
                            continue

                        img_result = await add_blocks_tool.call(
                            ctx_wrapper,
                            documentId=document_id,
                            parentBlockId=root_block_id,
                            index=current_index,
                            blocks=image_and_caption_blocks
                        )
                        logger.info(f"[IdeaEngine] 创建图片块[{i}]结果: {repr(img_result)[:500]}")

                        # 检查错误
                        if hasattr(img_result, 'isError') and img_result.isError:
                            logger.error(f"[IdeaEngine] 创建图片块[{i}]失败")
                            continue

                        # 从结果中提取图片块 ID
                        image_block_id: Optional[str] = None
                        try:
                            if hasattr(img_result, 'content') and img_result.content:
                                result_text = cast(Optional[str], getattr(img_result.content[0], 'text', None))
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

                        current_index += 2  # 图片块 + 居中图表说明文本块

                # 9.3 处理表格（创建表格后添加居中图表说明）
                tables = extracted_media.get("tables", [])
                if tables:
                    logger.info(f"[IdeaEngine] 开始处理 {len(tables)} 个表格 (格式: {table_format})")
                    # 计算实际处理的图片数量（排除跳过的）
                    processed_image_count = sum(1 for img in images if not img.get("_skip"))
                    current_index = len(blocks) + processed_image_count * 2  # 从文本块和图片块之后开始

                    for i, tbl in enumerate(tables):
                        # 跳过被标记为不匹配的表格
                        if tbl.get("_skip"):
                            logger.info(f"[IdeaEngine] 跳过表格: index={tbl.get('index')}")
                            continue

                        csv_path = tbl.get("csv_path", "")
                        png_path = tbl.get("png_path", "")
                        md_path = tbl.get("md_path", "")
                        caption = tbl.get("caption", f"表格 {i+1}")

                        if table_format == "csv":
                            # CSV 格式：使用 create_feishu_table 工具
                            if not create_table_tool:
                                logger.warning(f"[IdeaEngine] create_feishu_table 工具不可用，跳过表格 {i}")
                                continue
                            if not csv_path or not os.path.exists(csv_path):
                                logger.warning(f"[IdeaEngine] 表格[{i}] CSV路径为空或文件不存在")
                                continue

                            cells_data: List[Dict[str, Any]] = []
                            try:
                                with open(csv_path, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()

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
                                continue

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
                                if hasattr(table_result, 'isError') and table_result.isError:
                                    error_text = ""
                                    if hasattr(table_result, 'content') and table_result.content:
                                        error_text = getattr(table_result.content[0], 'text', str(table_result))
                                    logger.error(f"[IdeaEngine] 创建表格[{i}]失败: {error_text[:300] if error_text else 'unknown error'}")
                                else:
                                    tables_created += 1
                                    logger.info(f"[IdeaEngine] 表格[{i}] (CSV) 创建成功")
                                    # 创建居中图表说明文本块
                                    if add_blocks_tool:
                                        idx_match = re.search(r'-(\d+)$', tbl.get("index", ""))
                                        tbl_num = idx_match.group(1) if idx_match else str(i + 1)
                                        # 如果 caption 只是简单编号，用 表1 格式；否则用 表1: caption 格式（表1加粗）
                                        simple_pattern = re.match(
                                            r'^(Figure|Fig\.|Fig|Table|表|图)\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)?)$',
                                            caption.strip() if caption else "",
                                            re.IGNORECASE
                                        )
                                        if simple_pattern:
                                            caption_text = f"表{tbl_num}"
                                            caption_block = [{
                                                "blockType": "text",
                                                "options": {
                                                    "text": {
                                                        "textStyles": [
                                                            {"text": caption_text, "style": {"bold": True}}
                                                        ],
                                                        "align": 2  # 居中
                                                    }
                                                }
                                            }]
                                        elif caption:
                                            caption_block = [{
                                                "blockType": "text",
                                                "options": {
                                                    "text": {
                                                        "textStyles": [
                                                            {"text": f"表{tbl_num}: ", "style": {"bold": True}},
                                                            {"text": caption, "style": {}}
                                                        ],
                                                        "align": 2  # 居中
                                                    }
                                                }
                                            }]
                                        else:
                                            caption_text = f"表{tbl_num}"
                                            caption_block = [{
                                                "blockType": "text",
                                                "options": {
                                                    "text": {
                                                        "textStyles": [
                                                            {"text": caption_text, "style": {"bold": True}}
                                                        ],
                                                        "align": 2  # 居中
                                                    }
                                                }
                                            }]
                                        await add_blocks_tool.call(
                                            ctx_wrapper,
                                            documentId=document_id,
                                            parentBlockId=root_block_id,
                                            index=current_index + 1,
                                            blocks=caption_block
                                        )
                                current_index += 2  # 表格块 + 居中图表说明文本块

                        elif table_format == "md":
                            # MD 格式：作为文本块插入
                            md_content = ""
                            if md_path and os.path.exists(md_path):
                                try:
                                    with open(md_path, 'r', encoding='utf-8') as f:
                                        md_content = f.read()
                                except Exception as e:
                                    logger.error(f"[IdeaEngine] 读取MD失败: {e}")
                            elif csv_path and os.path.exists(csv_path):
                                try:
                                    with open(csv_path, 'r', encoding='utf-8') as f:
                                        md_content = f"```csv\n{f.read()}\n```"
                                except Exception as e:
                                    logger.error(f"[IdeaEngine] 读取CSV失败: {e}")

                            if md_content:
                                blocks.append({
                                    "blockType": "text",
                                    "options": {
                                        "text": {
                                            "textStyles": [
                                                {"text": f"{caption}\n{md_content}", "style": {"code": True}}
                                            ]
                                        }
                                    }
                                })
                                tables_created += 1
                                logger.info(f"[IdeaEngine] 表格[{i}] (MD) 添加为文本块")
                                current_index += 1

                        elif table_format == "png":
                            # PNG 格式：作为图片插入
                            if not png_path or not os.path.exists(png_path):
                                # PNG 不存在时，尝试用 Markdown 格式代替
                                logger.warning(f"[IdeaEngine] 表格[{i}] PNG不存在(png_path={png_path})，尝试使用Markdown格式")
                                if md_path and os.path.exists(md_path):
                                    try:
                                        with open(md_path, 'r', encoding='utf-8') as f:
                                            md_content = f.read()[:500]
                                        md_block = [{
                                            "blockType": "text",
                                            "options": {
                                                "text": {
                                                    "textStyles": [{"text": f"📋 {caption}\n```\n{md_content}\n```", "style": {"inline_code": False}}],
                                                    "align": 1  # 左对齐
                                                }
                                            }
                                        }]
                                        if add_blocks_tool:
                                            await add_blocks_tool.call(
                                                ctx_wrapper,
                                                documentId=document_id,
                                                parentBlockId=root_block_id,
                                                index=current_index,
                                                blocks=md_block
                                            )
                                            tables_created += 1
                                            logger.info(f"[IdeaEngine] 表格[{i}] (MD替代) 添加成功")
                                            current_index += 1
                                            continue
                                    except Exception as e:
                                        logger.error(f"[IdeaEngine] 表格[{i}] Markdown替代失败: {e}")
                                logger.warning(f"[IdeaEngine] 表格[{i}] PNG和Markdown都不存在，跳过")
                                continue

                            # 先创建空图片块（居中）
                            image_blocks = [{
                                "blockType": "image",
                                "align": 2,  # 2 = 居中
                                "options": {"image": {}}
                            }]

                            if add_blocks_tool is None:
                                logger.warning(f"[IdeaEngine] add_blocks_tool 不可用，跳过表格图片[{i}]")
                                continue

                            img_result = await add_blocks_tool.call(
                                ctx_wrapper,
                                documentId=document_id,
                                parentBlockId=root_block_id,
                                index=current_index,
                                blocks=image_blocks
                            )
                            image_block_id: Optional[str] = None
                            try:
                                if hasattr(img_result, 'content') and img_result.content:
                                    result_text = cast(Optional[str], getattr(img_result.content[0], 'text', None))
                                    if result_text:
                                        result_data = json.loads(result_text)
                                        image_info = result_data.get('imageBlocksInfo', {})
                                        if image_info:
                                            block_ids = image_info.get('blockIds', [])
                                            if block_ids:
                                                image_block_id = block_ids[0]
                            except Exception as e:
                                logger.error(f"[IdeaEngine] 解析图片块ID失败: {e}")

                            if image_block_id and upload_image_tool is not None:
                                upload_result = await upload_image_tool.call(
                                    ctx_wrapper,
                                    documentId=document_id,
                                    images=[{
                                        "blockId": image_block_id,
                                        "imagePathOrUrl": png_path
                                    }]
                                )
                                if hasattr(upload_result, 'isError') and upload_result.isError:
                                    logger.error(f"[IdeaEngine] 上传表格图片[{i}]失败")
                                else:
                                    tables_created += 1
                                    logger.info(f"[IdeaEngine] 表格[{i}] (PNG) 创建成功")
                                    # 创建居中图表说明文本块
                                    if add_blocks_tool:
                                        idx_match = re.search(r'-(\d+)$', tbl.get("index", ""))
                                        tbl_num = idx_match.group(1) if idx_match else str(i + 1)
                                        # 如果 caption 只是简单编号，用 表1 格式；否则用 表1: caption 格式（表1加粗）
                                        simple_pattern = re.match(
                                            r'^(Figure|Fig\.|Fig|Table|表|图)\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)?)$',
                                            caption.strip() if caption else "",
                                            re.IGNORECASE
                                        )
                                        if simple_pattern:
                                            caption_text = f"表{tbl_num}"
                                            caption_block = [{
                                                "blockType": "text",
                                                "options": {
                                                    "text": {
                                                        "textStyles": [
                                                            {"text": caption_text, "style": {"bold": True}}
                                                        ],
                                                        "align": 2  # 居中
                                                    }
                                                }
                                            }]
                                        elif caption:
                                            caption_block = [{
                                                "blockType": "text",
                                                "options": {
                                                    "text": {
                                                        "textStyles": [
                                                            {"text": f"表{tbl_num}: ", "style": {"bold": True}},
                                                            {"text": caption, "style": {}}
                                                        ],
                                                        "align": 2  # 居中
                                                    }
                                                }
                                            }]
                                        else:
                                            caption_text = f"表{tbl_num}"
                                            caption_block = [{
                                                "blockType": "text",
                                                "options": {
                                                    "text": {
                                                        "textStyles": [
                                                            {"text": caption_text, "style": {"bold": True}}
                                                        ],
                                                        "align": 2  # 居中
                                                    }
                                                }
                                            }]
                                        await add_blocks_tool.call(
                                            ctx_wrapper,
                                            documentId=document_id,
                                            parentBlockId=root_block_id,
                                            index=current_index + 1,
                                            blocks=caption_block
                                        )
                            current_index += 2  # 图片块 + 居中图表说明文本块

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
        if not knowledge:
            return content

        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])

        if not local_results and not web_results:
            return content

        # 1. 处理本地引用（从本地 JSON 文件获取 arxiv 链接）
        paper_to_arxiv = {}
        paper_to_github = {}

        if local_results:
            doc_stats_path = Path(__file__).parent / "data" / "milvus_abstracts_doc_stats.json"
            if doc_stats_path.exists():
                try:
                    with open(doc_stats_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    abstracts = data.get("abstracts", {})

                    for result in local_results:
                        paper = result.get("paper", "")
                        if not paper or paper in paper_to_arxiv:
                            continue

                        # 查找匹配的 paper_id
                        if paper in abstracts:
                            meta = abstracts[paper].get("metadata", {})
                        else:
                            paper_clean = paper[:-4] if paper.endswith(".pdf") else paper
                            meta = abstracts.get(paper_clean, {}).get("metadata", {})

                        arxiv_url = meta.get("arxiv_url") or None
                        github_url = meta.get("github_url") or None

                        if arxiv_url:
                            paper_to_arxiv[paper] = arxiv_url
                            paper_to_github[paper] = github_url
                            logger.info(f"[IdeaEngine] 解析论文到 arxiv: {paper[:50]} -> {arxiv_url}")
                except Exception as e:
                    logger.warning(f"[IdeaEngine] 读取 doc_stats 失败: {e}")

        # 2. 处理网络引用（直接使用 URL）
        web_refs = {}
        for i, result in enumerate(web_results, 1):
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
            elif line.startswith("- ") or line.startswith("* "):
                raw_content = line[2:].strip()
                if raw_content:
                    blocks.append({
                        "blockType": "list",
                        "options": {
                            "list": {
                                "content": raw_content,
                                "isOrdered": False
                            }
                        }
                    })
            # 有序列表 1. xxx 或 1) xxx
            elif re.match(r'^\d+[\.\)]\s', line):
                match = re.match(r'^(\d+[\.\)])\s+(.*)$', line)
                if match:
                    raw_content = match.group(2).strip()
                    if raw_content:
                        blocks.append({
                            "blockType": "list",
                            "options": {
                                "list": {
                                    "content": raw_content,
                                    "isOrdered": True
                                }
                            }
                        })
            # 空行
            elif line.strip() == "":
                pass
            # 普通文本（直接使用 textStyles 处理行内样式，由 _parse_inline_styles 自动处理样式标记）
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
        """移除 Markdown 样式标记，保留纯文本（仅用于标题等纯文本块）"""
        # 按优先级匹配：先处理长标记，再处理短标记
        # 注意：仅用于标题等本身不支持行内样式的块，
        # 列表项和普通段落使用 _parse_inline_styles 保留行内样式
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        return text

    def _strip_outer_markdown_style(self, text: str) -> str:
        """
        移除整行文本的外层 Markdown 样式标记（当整行都是样式文本时）
        例如：***加粗斜体文本*** → 加粗斜体文本
              **加粗文本** → 加粗文本
              *斜体文本* → 斜体文本
              `代码文本` → 代码文本
        但保留行内样式：混合测试：**加粗** 和 *斜体* → 混合测试：**加粗** 和 *斜体*
        """
        # 检查是否是整行样式（从头到尾都是样式标记包裹的内容）
        # 使用非贪婪匹配 .+? 来避免 [^*] 无法匹配某些字符的问题
        if re.match(r'^(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)$', text):
            # 整行都是样式，移除外层标记
            return self._strip_markdown_style(text)
        return text

    def _create_feishu_markdown(self) -> mistune.Markdown:
        """
        创建带自定义插件的 mistune Markdown：
        - [图X]/[表X] 引用 → <strong>[图X]</strong> / <strong>[表X]</strong>
        - $公式$ → <em class="latex">公式</em>

        使用 mistune v3 插件 API：
        - md.inline.register(name, pattern, func, before='...') 注册解析规则
        - md.renderer.register(name, func) 注册渲染函数
        """
        # ====== 图/表引用解析函数 ======
        def parse_fig_ref(md, m, state):
            """解析 [图X] 或 [表X] 引用"""
            text = m.group(0)  # e.g., '[图1]' or '[表3]'
            state.append_token({'type': 'fig_ref', 'raw': text})
            return m.end()

        def render_fig_ref(renderer, text):
            """渲染图/表引用为加粗格式"""
            return f'<strong>{text}</strong>'

        # ====== LaTeX 公式解析函数 ======
        def parse_latex(md, m, state):
            """解析 $公式$"""
            latex_match = m.group('latex')
            if latex_match:
                # latex_match includes the $ signs, e.g., '$E=mc^2$'
                formula = latex_match[1:-1]  # Strip $ signs
                state.append_token({'type': 'latex', 'raw': formula})
            return m.end()

        def render_latex(renderer, text):
            """渲染 LaTeX 公式为 <eq>formula</eq> 格式（飞书 equation 元素）"""
            return f'<eq>{text}</eq>'

        # 创建带自定义插件的 Markdown
        md = mistune.create_markdown(plugins=['strikethrough'])
        if md is None:
            raise RuntimeError("mistune.create_markdown() returned None")

        # 注册 [图X]/[表X] 规则（在 link 之前，避免 [text](url) 干扰）
        md.inline.register('fig_ref', r'\[(图|表)(\d+)\]', parse_fig_ref, before='link')  # type: ignore
        md.renderer.register('fig_ref', render_fig_ref)  # type: ignore

        # 注册 $公式$ 规则（在 emphasis 之前）
        md.inline.register('latex', r'\$([^$\n]+?)\$', parse_latex, before='emphasis')  # type: ignore
        md.renderer.register('latex', render_latex)  # type: ignore

        return md

    def _parse_inline_styles(self, text: str) -> List[Dict[str, Any]]:
        """
        使用 mistune + html.parser 解析 Markdown 文本，返回飞书 textStyles 格式

        支持：
        - **加粗** → bold: true
        - *斜体* → italic: true
        - ***加粗斜体*** → bold: true, italic: true
        - `行内代码` → inline_code: true
        - [文本](链接) → 链接文本 + (url)
        - [图X]、[表X] → 加粗图表引用
        - $公式$ / $$公式$$ → equation 元素

        如果 mistune 解析失败，回退到纯文本（不做任何样式处理）。
        """
        if not text:
            return [{"text": "", "style": {}}]

        try:
            md = self._create_feishu_markdown()
            html = md(text)
            result = self._parse_html_with_html_parser(cast(str, html))
            if result and any(item.get("text") or item.get("equation") for item in result):
                return result
        except Exception as e:
            logger.warning(f"[IdeaEngine] mistune 解析失败: {e}")

        return [{"text": text, "style": {}}]

    def _parse_html_with_html_parser(self, html: str) -> List[Dict[str, Any]]:
        """使用 Python 内置 html.parser 解析 HTML"""
        from html.parser import HTMLParser
        from html import unescape

        class FeishuHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = []
                self.current_text = ""
                self.styles = {}
                self.link_url = None
                self._in_eq = False  # 是否在 <eq> 标签内
                self._eq_text = ""   # 公式内容

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs) if attrs else {}

                # 先输出当前累积的文本（如果有）
                if self.current_text and tag not in ('br',):
                    self.result.append({
                        "text": unescape(self.current_text),
                        "style": dict(self.styles)
                    })
                    self.current_text = ""

                if tag == 'strong':
                    self.styles['bold'] = True
                elif tag == 'em':
                    self.styles['italic'] = True
                elif tag == 'code':
                    self.styles['inline_code'] = True
                elif tag in ('del', 's', 'strike'):
                    self.styles['strikethrough'] = True
                elif tag == 'a':
                    self.link_url = attrs_dict.get('href')
                    # 链接文本需要加粗
                    self.styles['bold'] = True
                elif tag == 'br':
                    self.result.append({"text": "\n", "style": {}})
                elif tag == 'eq':
                    # 公式标签：开始收集公式内容
                    self._in_eq = True
                    self._eq_text = ""

            def handle_endtag(self, tag):
                # 如果是在公式标签内，输出公式元素
                if tag == 'eq' and self._in_eq:
                    self.result.append({"equation": self._eq_text, "style": {}})
                    self._in_eq = False
                    self._eq_text = ""
                    return

                # 先输出当前累积的文本
                if self.current_text:
                    self.result.append({
                        "text": unescape(self.current_text),
                        "style": dict(self.styles)
                    })
                    self.current_text = ""

                if tag == 'a':
                    if self.link_url:
                        self.result.append({"text": f" ({self.link_url})", "style": {}})
                        self.link_url = None
                        # 移除链接添加的 bold 样式
                        self.styles.pop('bold', None)
                elif tag in ('strong', 'em', 'code', 'del', 's', 'strike'):
                    key = {'strong': 'bold', 'em': 'italic', 'code': 'inline_code',
                           'del': 'strikethrough', 's': 'strikethrough', 'strike': 'strikethrough'}.get(tag)
                    if key:
                        self.styles.pop(key, None)
                elif tag == 'p':
                    pass

            def handle_data(self, data):
                if self._in_eq:
                    self._eq_text += data
                else:
                    self.current_text += data

            def handle_entityref(self, name):
                if self._in_eq:
                    self._eq_text += unescape(f'&{name};')
                else:
                    self.current_text += unescape(f'&{name};')

            def handle_charref(self, name):
                if self._in_eq:
                    self._eq_text += unescape(f'&#{name};')
                else:
                    self.current_text += unescape(f'&#{name};')

        # 移除 <p> 标签和末尾空白
        html = html.replace('<p>', '').replace('</p>', '').strip()

        parser = FeishuHTMLParser()
        parser.feed(html)

        # 保存剩余文本
        if parser.current_text:
            parser.result.append({
                "text": unescape(parser.current_text),
                "style": dict(parser.styles)
            })

        # 合并相邻的相同 style 文本
        merged = []
        for item in parser.result:
            if merged and merged[-1].get('text') and item.get('text') and merged[-1].get('style') == item.get('style'):
                merged[-1]['text'] += item['text']
            else:
                merged.append(item)

        return merged

    def _strip_html_tags_simple(self, text: str) -> str:
        """
        移除 HTML 标签（不使用正则表达式）
        简单处理：只移除 <p> 和 </p> 标签
        """
        result = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == '<':
                # 检查是否是 <p 或 </p
                if text[i:i+2] == '<p' or text[i:i+3] == '</p':
                    # 找到对应的 >
                    j = text.find('>', i)
                    if j != -1:
                        i = j + 1
                        continue
            result.append(text[i])
            i += 1
        return ''.join(result)

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
            with open(mcp_config_path, "r", encoding="utf-8-sig") as f:
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
