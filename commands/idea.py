"""Idea generation commands for PaperRAG."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .base import PluginCoreBase

_PLUGIN_DIR = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    from ..idea import IdeaEngine


def _create_idea_engine(context, rag_engine):
    try:
        from ..idea import IdeaEngine
    except ImportError:
        from idea import IdeaEngine

    return IdeaEngine(context=context, rag_engine=rag_engine)


class IdeaCommandsMixin(PluginCoreBase):
    async def _idea_gen(self, event: AstrMessageEvent,
                        topic: str = ""):
        """
        生成研究想法并保存到文件（第一阶段）

        使用方式:
        /idea gen <研究主题>
        Example: /idea gen 稀疏3DGS开放词汇统一重建

        流程：分析主题 → 检索知识 → 生成想法 → 保存到文件
        保存后可手动编辑打磨，之后用 /idea tofeishu 创建飞书文档
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not topic:
            yield event.plain_result("📚 Usage: /idea gen <研究主题>\nExample: /idea gen 稀疏3DGS开放词汇统一重建")
            return

        yield event.plain_result(f"💡 正在分析研究主题...\n主题: {topic}")

        try:
            rag_engine = self._get_engine()
            idea_engine = _create_idea_engine(self.context, rag_engine)

            # 1. 分析主题
            yield event.plain_result("📊 正在分析研究主题...")
            analysis = await idea_engine.analyze_topic(topic, depth="standard")
            if not analysis:
                yield event.plain_result("❌ 主题分析失败")
                return

            # 2. 收集知识
            yield event.plain_result("📚 正在检索知识...")
            all_queries = (analysis.search_queries + analysis.local_rag_queries)[:5]
            knowledge = await idea_engine.search_knowledge(all_queries, local_rag_top_k=25, web_top_k=25)

            # 3. 生成研究想法
            yield event.plain_result("💡 正在生成研究想法...")
            ideas = await idea_engine.generate_ideas(
                knowledge_context=knowledge.get("fused_context", ""),
                research_domain=analysis.domain,
                num_ideas=3,
                idea_focus="all",
                topic=topic
            )

            if not ideas:
                yield event.plain_result("❌ 想法生成失败")
                return

            # 4. 保存到文件（每个 idea 独立文件）
            saved = idea_engine.save_ideas_to_file(
                ideas=ideas,
                topic=topic,
                knowledge=knowledge
            )

            # 5. 返回想法摘要
            output = f"**💡 研究想法已生成并保存**\n\n"
            output += f"📁 文件数: {len(saved)} 个\n\n"
            output += f"**🔑 UUID 列表**\n"
            for uid, path in saved:
                output += f"`{uid}`\n"

            output += f"\n**想法摘要 ({len(ideas)}个)**\n\n"

            for i, idea in enumerate(ideas, 1):
                output += f"""---
**[{i}] {idea.title}**

**✨ 创新点**: {idea.novelty[:150]}...
**🔧 方法论**: {idea.methodology[:150]}...
**⚠️ 挑战**: {', '.join(idea.potential_challenges[:2])}
"""

            output += "\n---\n\n"
            output += "💡 如需调整想法，可编辑上述 JSON 文件中的 `ideas` 数组。\n"
            output += "📄 确认后，使用 `/idea tofeishu <研究主题> <folder_token>` 创建飞书文档。"

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"[IdeaEngine] 想法生成失败: {e}")
            yield event.plain_result(f"❌ 生成失败: {e}")
    async def _idea_list(self, event: AstrMessageEvent):
        """
        列出所有已保存的 topic 及其想法数量

        使用方式:
        /idea list
        """
        try:
            rag_engine = self._get_engine()
            if not rag_engine:
                yield event.plain_result("❌ RAG引擎未初始化")
                return

            idea_engine = _create_idea_engine(self.context, rag_engine)
            topics = idea_engine.list_all_topics()

            if not topics:
                yield event.plain_result("📭 暂无已保存的 topic，运行 /idea <主题> 生成想法")
                return

            lines = ["**📚 已保存的 Topics：**\n"]
            for i, t in enumerate(topics, 1):
                lines.append(f"{i}. **{t['topic']}**")
                lines.append(f"   📁 `{t['folder']}` · 💡 {t['idea_count']} 个想法 · {t['created_at']}")
            lines.append("")
            lines.append("使用 `/idea show <topic>` 查看详情")
            lines.append("使用 `/idea tofeishu <topic>` 创建飞书文档")

            yield event.plain_result("\n".join(lines))

        except Exception as e:
            logger.error(f"[IdeaEngine] 列出 topic 失败: {e}")
            yield event.plain_result(f"❌ 列出失败: {e}")
    async def _idea_show(self, event: AstrMessageEvent, identifier: str = ""):
        """
        显示单个 topic 下的所有想法（支持 topic 名称或 folder hash）

        使用方式:
        /idea show <topic>
        /idea show <folder_hash>
        Example: /idea show 稀疏3DGS开放词汇统一重建
        Example: /idea show -4500404867533322446
        """
        try:
            if not identifier:
                yield event.plain_result("📚 Usage: /idea show <topic>\nExample: /idea show 稀疏3DGS")
                return

            rag_engine = self._get_engine()
            if not rag_engine:
                yield event.plain_result("❌ RAG引擎未初始化")
                return

            idea_engine = _create_idea_engine(self.context, rag_engine)

            # identifier 可能是 folder hash 或 topic 名称，统一解析为 folder hash
            ideas_dir = idea_engine._get_ideas_dir()
            folder_hash = identifier if (ideas_dir / identifier).exists() else idea_engine._topic_hash(identifier)
            real_topic = idea_engine.find_topic_by_folder(folder_hash)
            ideas_list, context_data = idea_engine.load_ideas_by_topic(folder_hash)

            if not ideas_list:
                yield event.plain_result(f"❌ 未找到 folder hash={folder_hash} 的想法")
                return

            display_name = real_topic if real_topic else identifier
            lines = [f"**💡 Topic: {display_name}**（共 {len(ideas_list)} 个想法）\n"]
            for i, idea_data in enumerate(ideas_list, 1):
                idea = idea_data.get("idea", {})
                title = idea.get("title", "无标题")
                idea_id = idea_data.get("id", "?")
                novelty = idea.get("novelty", "")
                feasibility = idea.get("feasibility", "")
                lines.append(f"**{i}. {title}** `[{idea_id}]`")
                if isinstance(novelty, str) and novelty:
                    lines.append(f"   🎯 创新点: {novelty[:80]}{'...' if len(novelty) > 80 else ''}")
                if isinstance(feasibility, (int, float)) and feasibility:
                    lines.append(f"   ✅ 可行性: {feasibility}")
                lines.append("")

            lines.append("使用 `/idea tofeishu <uuid1,uuid2>` 创建飞书文档")
            lines.append("使用 `/idea tofeishu <topic>` 创建全部想法的飞书文档")

            yield event.plain_result("\n".join(lines))

        except Exception as e:
            logger.error(f"[IdeaEngine] 显示想法失败: {e}")
            yield event.plain_result(f"❌ 显示失败: {e}")
    async def _idea_add(self, event: AstrMessageEvent,
                           topic: str = "",
                           num_ideas: int = 3):
        """
        为已有 topic 追加新想法（复用现有知识上下文）

        使用方式:
        /idea add <topic> [数量]
        Example: /idea add 稀疏3DGS 2
        """
        try:
            if not topic:
                yield event.plain_result("📚 Usage: /idea add <topic> [数量]\nExample: /idea add 稀疏3DGS 2")
                return

            rag_engine = self._get_engine()
            if not rag_engine:
                yield event.plain_result("❌ RAG引擎未初始化")
                return

            idea_engine = _create_idea_engine(self.context, rag_engine)

            yield event.plain_result(f"💡 正在为 topic「{topic}」追加 {num_ideas} 个想法...\n⏳ 复用现有知识上下文生成新想法")

            ideas, knowledge = await idea_engine.add_ideas_to_topic(
                topic=topic,
                num_ideas=num_ideas,
                idea_focus="all"
            )

            if not ideas:
                yield event.plain_result("❌ 想法生成失败")
                return

            # 获取刚保存的 UUID（最后 num_ideas 个）
            all_ideas, _ = idea_engine.load_ideas_by_topic(idea_engine._topic_hash(topic))
            new_uuids = [a.get("id") for a in all_ideas[-num_ideas:]]

            output = f"**✅ 已为「{topic}」追加 {len(ideas)} 个新想法**\n\n"
            output += f"**🔑 新增 UUID 列表**\n"
            for uid in new_uuids:
                output += f"`{uid}`\n"

            output += f"\n**新想法摘要**\n\n"
            for i, idea in enumerate(ideas, 1):
                output += f"""---
**[{i}] {idea.title}**

**✨ 创新点**: {idea.novelty[:150]}...
**🔧 方法论**: {idea.methodology[:150]}...
"""

            output += "\n---\n\n"
            output += f"💡 当前 topic 共有 {len(all_ideas)} 个想法。\n"
            output += f"📄 使用 `/idea tofeishu <topic>` 创建飞书文档。"

            yield event.plain_result(output)

        except ValueError as e:
            yield event.plain_result(f"❌ {e}")
        except Exception as e:
            logger.error(f"[IdeaEngine] 追加想法失败: {e}")
            yield event.plain_result(f"❌ 追加失败: {e}")
    async def _idea_del(self, event: AstrMessageEvent,
                           ids: str = ""):
        """
        删除指定 UUID 的想法

        使用方式:
        /idea del <uuid1,uuid2,...>
        Example: /idea del a1b2c3d4,e5f6g7h8
        """
        try:
            if not ids:
                yield event.plain_result("📚 Usage: /idea del <uuid1,uuid2,...>\nExample: /idea del a1b2c3d4,e5f6g7h8")
                return

            uuids = [u.strip() for u in ids.split(",")]
            rag_engine = self._get_engine()
            if not rag_engine:
                yield event.plain_result("❌ RAG引擎未初始化")
                return

            idea_engine = _create_idea_engine(self.context, rag_engine)

            deleted, topic = idea_engine.delete_ideas_by_uuids(uuids)

            if not deleted:
                yield event.plain_result(f"❌ 未找到匹配 UUID 的想法: {uuids}")
                return

            output = f"**🗑️ 已删除 {len(deleted)} 个想法**\n"
            for uid in deleted:
                output += f"  - `{uid}`\n"
            if topic:
                remaining, _ = idea_engine.load_ideas_by_topic(idea_engine._topic_hash(topic))
                output += f"\n💡 topic「{topic}」剩余 {len(remaining)} 个想法"

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"[IdeaEngine] 删除想法失败: {e}")
            yield event.plain_result(f"❌ 删除失败: {e}")
    async def _idea_delete(self, event: AstrMessageEvent,
                              topic_or_hash: str = ""):
        """
        完全删除指定 topic（包括 folder 本身）

        使用方式:
        /idea delete <topic名称或folder_hash>
        Example: /idea delete 稀疏3DGS开放词汇统一重建
        Example: /idea delete 8a160941c48c813c
        """
        try:
            if not topic_or_hash:
                yield event.plain_result("📚 Usage: /idea delete <topic名称或folder_hash>\nExample: /idea delete 稀疏3DGS开放词汇\nExample: /idea delete 8a160941c48c813c")
                return

            rag_engine = self._get_engine()
            if not rag_engine:
                yield event.plain_result("❌ RAG引擎未初始化")
                return

            idea_engine = _create_idea_engine(self.context, rag_engine)

            success, topic, folder_hash = idea_engine.delete_topic_by_hash(topic_or_hash)

            if not success:
                yield event.plain_result(f"❌ 未找到 topic「{topic_or_hash}」")
                return

            yield event.plain_result(f"**🗑️ 已完全删除 topic「{topic}」**\n\nfolder: `{folder_hash}`")

        except Exception as e:
            logger.error(f"[IdeaEngine] 删除 topic 失败: {e}")
            yield event.plain_result(f"❌ 删除失败: {e}")
    async def _idea_clear(self, event: AstrMessageEvent,
                              topic: str = ""):
        """
        清空指定 topic 下所有想法（保留 folder）

        使用方式:
        /idea clear <topic>
        Example: /idea clear 稀疏3DGS开放词汇统一重建
        """
        try:
            if not topic:
                yield event.plain_result("📚 Usage: /idea clear <topic>\nExample: /idea clear 稀疏3DGS开放词汇统一重建")
                return

            rag_engine = self._get_engine()
            if not rag_engine:
                yield event.plain_result("❌ RAG引擎未初始化")
                return

            idea_engine = _create_idea_engine(self.context, rag_engine)

            deleted_count, actual_topic = idea_engine.clear_ideas_by_topic(topic)

            if deleted_count == 0:
                yield event.plain_result(f"❌ 未找到 topic「{topic}」下的想法")
                return

            yield event.plain_result(f"**🗑️ 已清空 topic「{actual_topic}」的所有想法**\n\n已删除 {deleted_count} 个想法文件")

        except Exception as e:
            logger.error(f"[IdeaEngine] 清空想法失败: {e}")
            yield event.plain_result(f"❌ 清空失败: {e}")
    async def _idea_explore(self, event: AstrMessageEvent,
                                topic: str = '',
                                depth: str = "standard",
                                num_ideas: int = 3):
        """
        探索研究想法（完整流程）

        Args:
            topic: 研究主题描述
            depth: 分析深度 (quick/standard/deep)
            num_ideas: 生成想法数量
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not topic:
            yield event.plain_result("📚 Usage: /idea explore <研究主题>\nExample: /idea explore 大语言模型在医学诊断中的应用")
            return

        # Agentic workflow 模式（config 开关控制）
        if self.config.get("enable_agentic_ideas", False):
            yield event.plain_result(f"🧠 Agentic Idea 生成中...\n主题: {topic}")
            try:
                from idea import run_agentic_ideas
                rag_engine = self._get_engine()
                result = await run_agentic_ideas(
                    topic=topic,
                    context=self.context,
                    depth=depth,
                    num_ideas=num_ideas,
                    rag_engine=rag_engine,
                    config=self.config,
                )
                # 流式输出各阶段
                for step in result.get("steps", []):
                    yield event.plain_result(step)
                # 输出最终结果
                final = result.get("final_output", "")
                if final:
                    yield event.plain_result(final)
                saved = result.get("saved_paths", [])
                if saved:
                    yield event.plain_result(f"\n✅ 已保存 {len(saved)} 个想法到本地")
                else:
                    yield event.plain_result("⚠️ 想法未保存（请检查知识库是否有相关文档）")
            except Exception as e:
                logger.error(f"Agentic idea failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                yield event.plain_result(f"❌ Agentic Idea 生成失败: {e}")
            return

        # ===== 原有流程（enable_agentic_ideas=False） =====

        yield event.plain_result(f"🔍 正在分析研究主题...\n主题: {topic}")

        try:
            # 获取RAG引擎
            rag_engine = self._get_engine()

            # 创建创意引擎
            idea_engine = _create_idea_engine(self.context, rag_engine)

            # 1. 分析主题
            yield event.plain_result("📊 正在分析研究领域...")
            analysis = await idea_engine.analyze_topic(topic, depth)

            if not analysis:
                yield event.plain_result("❌ 主题分析失败")
                return

            analysis_output = f"""**📊 主题分析结果**

**领域**: {analysis.domain}

**关键词**: {', '.join(analysis.keywords[:8])}

**探索角度**: {', '.join(analysis.exploration_angles)}

**摘要**: {analysis.summary}
"""
            yield event.plain_result(analysis_output)

            # 2. 检索知识
            yield event.plain_result("🌐 正在检索网络资源 + 📚 本地论文库...")
            all_queries = analysis.search_queries + analysis.local_rag_queries
            knowledge = await idea_engine.search_knowledge(all_queries, local_rag_top_k=25, web_top_k=25)

            stats_output = f"""
✅ 检索完成
- 网络资源: {knowledge['stats']['web_count']} 条
- 本地论文: {knowledge['stats']['local_count']} 条
"""
            yield event.plain_result(stats_output)

            # 3. 生成想法
            yield event.plain_result("💡 正在生成研究想法...")
            ideas = await idea_engine.generate_ideas(
                knowledge_context=knowledge['fused_context'],
                research_domain=analysis.domain,
                num_ideas=num_ideas,
                topic=topic
            )

            if not ideas:
                yield event.plain_result("❌ 想法生成失败")
                return

            # 格式化输出
            ideas_output = f"**💡 研究想法 ({len(ideas)}个)**\n\n"

            for i, idea in enumerate(ideas, 1):
                feasibility_bar = "★" * int(idea.feasibility * 5) + "☆" * (5 - int(idea.feasibility * 5))

                ideas_output += f"""---
**[{i}] {idea.title}**

**📝 描述**: {idea.description[:300]}...

**✨ 创新点**: {idea.novelty[:150]}

**🔧 方法论**: {idea.methodology[:150]}

**⚠️ 挑战**: {', '.join(idea.potential_challenges[:2])}

**📈 可行性**: {feasibility_bar} ({idea.feasibility:.0%})
"""

            ideas_output += "\n---\n💡 回复 /idea generate <想法序号> 可获取更详细的提案大纲"

            yield event.plain_result(ideas_output)

        except Exception as e:
            logger.error(f"创意探索失败: {e}")
            yield event.plain_result(f"❌ 创意探索失败: {e}")
    async def _idea_analyze(self, event: AstrMessageEvent,
                               topic: str = '',
                               depth: str = "standard"):
        """
        分析研究主题

        Args:
            topic: 研究主题描述
            depth: 分析深度 (quick/standard/deep)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not topic:
            yield event.plain_result("📚 Usage: /idea analyze <研究主题>")
            return

        yield event.plain_result(f"🔍 分析主题: {topic}")

        try:
            rag_engine = self._get_engine()
            idea_engine = _create_idea_engine(self.context, rag_engine)

            analysis = await idea_engine.analyze_topic(topic, depth)

            if not analysis:
                yield event.plain_result("❌ 主题分析失败")
                return

            output = f"""**📊 主题分析结果**

**研究领域**: {analysis.domain}

**核心关键词**:
{chr(10).join(f"- {k}" for k in analysis.keywords[:8])}

**搜索查询**:
{chr(10).join(f"- {q}" for q in analysis.search_queries[:5])}

**本地检索词**:
{chr(10).join(f"- {q}" for q in analysis.local_rag_queries[:3])}

**探索角度**:
{', '.join(analysis.exploration_angles)}

**摘要**: {analysis.summary}
"""
            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"主题分析失败: {e}")
            yield event.plain_result(f"❌ 分析失败: {e}")
    async def _idea_search(self, event: AstrMessageEvent,
                              queries: str = '',
                              local_k: int = 5,
                              web_k: int = 10):
        """
        多源知识检索

        Args:
            queries: 逗号分隔的搜索查询
            local_k: 本地RAG召回数
            web_k: 网络搜索召回数
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not queries:
            yield event.plain_result("📚 Usage: /idea search <查询1>, <查询2>, ...\nExample: /idea search LLM medical diagnosis, GPT-4 clinical")
            return

        query_list = [q.strip() for q in queries.split(",") if q.strip()]

        yield event.plain_result(f"🔍 执行多源检索: {query_list}")

        try:
            rag_engine = self._get_engine()
            idea_engine = _create_idea_engine(self.context, rag_engine)

            knowledge = await idea_engine.search_knowledge(query_list, local_rag_top_k=local_k, web_top_k=web_k)

            output = f"""**✅ 检索完成**

**统计**:
- 网络资源: {knowledge['stats']['web_count']} 条
- 本地论文: {knowledge['stats']['local_count']} 条

---
**📚 本地论文相关片段**:
"""

            for i, r in enumerate(knowledge['local_results'][:10], 1):
                output += f"\n{i}. **{r['paper']}** (p.{r['page']})"
                output += f"\n   {r['text'][:150]}..."

            if knowledge['web_results']:
                output += "\n\n---\n**🌐 网络资源**:\n"
                for i, r in enumerate(knowledge['web_results'][:10], 1):
                    output += f"\n{i}. **{r['title']}**"
                    output += f"\n   {r['snippet'][:150]}..."

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"检索失败: {e}")
            yield event.plain_result(f"❌ 检索失败: {e}")
    async def _idea_generate(self, event: AstrMessageEvent,
                                context: str = '',
                                domain: str = "",
                                num: int = 3,
                                focus: str = "all"):
        """
        基于知识上下文生成研究想法

        Args:
            context: 知识上下文（可直接粘贴检索结果）
            domain: 研究领域
            num: 生成数量
            focus: 侧重点 (novelty/feasibility/impact/all)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not context:
            yield event.plain_result("📚 Usage: /idea generate <知识上下文>\n建议先使用 /idea search 检索相关知识，然后将结果作为上下文传入")
            return

        yield event.plain_result(f"💡 正在生成 {num} 个研究想法...")

        try:
            rag_engine = self._get_engine()
            idea_engine = _create_idea_engine(self.context, rag_engine)

            ideas = await idea_engine.generate_ideas(
                knowledge_context=context,
                research_domain=domain,
                num_ideas=num,
                idea_focus=focus,
                topic=""
            )

            if not ideas:
                yield event.plain_result("❌ 想法生成失败")
                return

            output = f"**💡 研究想法 ({len(ideas)}个)**\n\n"

            for i, idea in enumerate(ideas, 1):
                feasibility_bar = "★" * int(idea.feasibility * 5) + "☆" * (5 - int(idea.feasibility * 5))

                output += f"""---
**[{i}] {idea.title}**

**📝 描述**: {idea.description[:300]}

**✨ 创新点**: {idea.novelty[:200]}

**🔧 方法论**: {idea.methodology[:200]}

**⚠️ 挑战**: {', '.join(idea.potential_challenges[:3])}

**📈 可行性**: {feasibility_bar} ({idea.feasibility:.0%})
"""

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"想法生成失败: {e}")
            yield event.plain_result(f"❌ 生成失败: {e}")
    async def _idea_tofeishu(self, event: AstrMessageEvent,
                                  ids: str = "",
                                  folder_token: str = "",
                                  refresh: str = "auto"):
        """
        将研究想法导出为飞书文档（第二阶段）

        使用方式:
        /idea tofeishu <topic> [folder_token] [refresh]      # 按 topic 加载全部
        /idea tofeishu <uuid1,uuid2,...> [folder_token] [refresh]  # 按 UUID 加载指定想法
        /idea tofeishu <topic> [folder_token] refresh       # 强制重新检索

        Examples:
        /idea tofeishu 稀疏3DGS开放词汇统一重建
        /idea tofeishu 稀疏3DGS开放词汇统一重建 <folder_token>
        /idea tofeishu a1b2c3d4,e5f6g7h8 <folder_token>           # 加载指定 UUID
        /idea tofeishu 稀疏3DGS <folder_token> refresh           # 强制重新检索

        加载优先级：ids 含逗号 → UUID 精确加载；否则 → topic 加载该 topic 下全部
        refresh: auto(默认)使用已有草稿 / refresh=refresh 强制重新检索
        表格格式: 由插件配置 feishu_table_format 指定（png/csv/md）
        """
        try:
            # 转换 refresh 字符串为布尔值
            table_format = self.config.get("feishu_table_format", "png")
            enable_paper_banana = self.config.get("enable_paper_banana", False)
            refresh_flag = refresh.lower() == "refresh" if refresh else False
            logger.info(f"[IdeaEngine] tofeishu refresh={repr(refresh)}, refresh_flag={refresh_flag}, table_format={repr(table_format)}, enable_paper_banana={enable_paper_banana}")

            if not ids:
                yield event.plain_result("📚 Usage: /idea tofeishu <topic> [folder_token]\n       /idea tofeishu <uuid1,uuid2,...> [folder_token]\nExample: /idea tofeishu 稀疏3DGS a1b2c3d4,e5f6g7h8")
                return

            if not folder_token:
                yield event.plain_result("""❌ 创建飞书文档需要提供 folder_token

获取方式：
1. 打开飞书文档所在的文件夹
2. 点击文件夹右上角的「···」
3. 选择「复制链接」
4. 链接格式: https://xxx.feishu.cn/drive/folder/xxxxx
5. 链接最后一部分就是 folder_token（如 FWK2fMleClICfodlHHWc4Mygnhb）
6. 使用方式: /idea tofeishu <主题> <folder_token>

例如: /idea tofeishu 我的研究想法 FWK2fMleClICfodlHHWc4Mygnhb""")
                return

            # 获取 RAG 引擎
            rag_engine = self._get_engine()
            if not rag_engine:
                yield event.plain_result("❌ RAG引擎未初始化")
                return

            # 初始化 IdeaEngine
            idea_engine = _create_idea_engine(self.context, rag_engine)

            # 判断是 UUID 列表还是 topic
            if "," in ids:
                # 按 UUID 精确加载
                uuids = [u.strip() for u in ids.split(",")]
                yield event.plain_result(f"📂 按 UUID 加载想法: {uuids}")
                ideas_list, context_data = idea_engine.load_ideas_by_uuids(uuids)
                if not ideas_list:
                    yield event.plain_result(f"❌ 未找到指定 UUID 的想法: {uuids}")
                    return
                ideas = idea_engine.convert_to_research_ideas(ideas_list)
                knowledge = context_data or {}
                topic = context_data.get("topic", ids) if context_data else ids
                folder_hash = context_data.get("_folder_hash") if context_data else None
                # UUID 加载时忽略 refresh（无重新检索需求）
                refresh_flag = False
            else:
                # 按 topic 加载（支持 folder hash 直接查找）
                # 优先检查 ids 是否已是合法 folder hash
                ideas_dir = idea_engine._get_ideas_dir()
                folder_hash = ids if (ideas_dir / ids).exists() else idea_engine._topic_hash(ids)
                # 如果 ids 是 folder hash，从 context 中读取真实 topic 名称
                topic = idea_engine.find_topic_by_folder(folder_hash) or ids
                if refresh_flag:
                    # refresh 只重新生成草稿，不重新检索 ideas 和 knowledge
                    yield event.plain_result(f"🔄 重新生成草稿: {topic}")
                    ideas_list, context_data = idea_engine.load_ideas_by_topic(folder_hash)
                    if not ideas_list:
                        yield event.plain_result(f"❌ 未找到 folder hash={folder_hash} 的想法")
                        return
                    topic = context_data.get("topic", topic) if context_data else topic
                    ideas = idea_engine.convert_to_research_ideas(ideas_list)
                    knowledge = context_data or {}
                else:
                    ideas_list, context_data = idea_engine.load_ideas_by_topic(folder_hash)
                    if not ideas_list:
                        yield event.plain_result(f"❌ 未找到 folder hash={folder_hash} 的想法")
                        return
                    topic = context_data.get("topic", topic) if context_data else topic
                    ideas = idea_engine.convert_to_research_ideas(ideas_list)
                    knowledge = context_data or {}

            if not ideas:
                yield event.plain_result("❌ 想法为空")
                return

            # 加载已有的 initial_draft.md（仅当 refresh_flag=False 且有 folder_hash 时）
            initial_draft = ""
            if not refresh_flag:
                if folder_hash:
                    draft_file = idea_engine._get_ideas_dir() / folder_hash / "initial_draft.md"
                    if draft_file.exists():
                        try:
                            with open(draft_file, "r", encoding="utf-8") as f:
                                initial_draft = f.read()
                            logger.info(f"[IdeaEngine] 已加载已有草稿: {draft_file}, 长度: {len(initial_draft)}")
                        except OSError as e:
                            logger.warning(f"[IdeaEngine] 读取草稿失败 [OSError]: {e}")
                            yield event.plain_result(f"⚠️ 读取已有草稿失败 [OSError]: {e}，将重新生成")
                            initial_draft = ""
                        except UnicodeDecodeError as e:
                            logger.warning(f"[IdeaEngine] 读取草稿失败 [UnicodeDecodeError]: {e}")
                            yield event.plain_result(f"⚠️ 读取已有草稿失败 [UnicodeDecodeError]: {e}，将重新生成")
                            initial_draft = ""
                        except Exception as e:
                            logger.error(f"[IdeaEngine] 读取草稿失败 [Unexpected]: {type(e).__name__}: {e}")
                            yield event.plain_result(f"⚠️ 读取已有草稿失败 [{type(e).__name__}]: {e}，将重新生成")
                            initial_draft = ""
                    # else: 草稿不存在，正常，会生成新的
                else:
                    logger.debug("[IdeaEngine] 无 folder_hash（UUID路径），跳过草稿加载，将生成新草稿")

            # 4. 创建飞书文档（传入知识检索结果以生成引用）
            yield event.plain_result("📄 正在创建飞书文档...")
            try:
                result = await idea_engine.create_feishu_document(
                    ideas=ideas,
                    topic=topic,
                    folder_token=folder_token,
                    knowledge=knowledge,
                    table_format=table_format,
                    initial_draft=initial_draft,
                    enable_paper_banana=enable_paper_banana
                )
            except Exception as e:
                logger.error(f"[IdeaEngine] create_feishu_document 异常 [{type(e).__name__}]: {e}")
                import traceback
                logger.error(traceback.format_exc())
                yield event.plain_result(f"❌ 创建飞书文档异常 [{type(e).__name__}]: {e}")
                return

            # 如果生成了新草稿且refresh_flag=True，保存草稿
            if refresh_flag and result.get("polished_content"):
                if folder_hash:
                    draft_file = idea_engine._get_ideas_dir() / folder_hash / "initial_draft.md"
                    draft_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with open(draft_file, "w", encoding="utf-8") as f:
                            f.write(result["polished_content"])
                        logger.info(f"[IdeaEngine] 已保存新草稿: {draft_file}")
                    except OSError as e:
                        logger.warning(f"[IdeaEngine] 保存草稿失败 [OSError]: {e}")
                        yield event.plain_result(f"⚠️ 保存草稿失败 [OSError]: {e}")
                    except Exception as e:
                        logger.error(f"[IdeaEngine] 保存草稿失败 [{type(e).__name__}]: {e}")
                        yield event.plain_result(f"⚠️ 保存草稿失败 [{type(e).__name__}]: {e}")
                else:
                    logger.warning("[IdeaEngine] 无法保存草稿：UUID路径缺少 folder_hash")
                    yield event.plain_result("⚠️ 草稿已生成但无法保存（缺少 folder_hash）")

            if not result:
                yield event.plain_result("❌ 创建飞书文档失败: 未知错误")
                return

            if result.get("error"):
                polished = result.get("polished_content", "")
                if polished:
                    # 即使创建失败，也返回润色内容供用户审阅
                    output = f"""❌ 飞书文档创建失败，但以下是润色后的内容供审阅：

---

{polished}

---

❌ 错误信息: {result.get('error')}

💡 你可以复制上方内容手动创建飞书文档"""
                    yield event.plain_result(output)
                else:
                    yield event.plain_result(f"❌ 创建飞书文档失败: {result.get('error')}")
                return

            # 成功
            url = result.get("url", "")
            blocks_created = result.get("blocks_created", 0)
            polished = result.get("polished_content", "")
            media_count = result.get("media_count", {})
            images_count = media_count.get("images", 0)
            tables_count = media_count.get("tables", 0)

            media_info = ""
            if images_count > 0 or tables_count > 0:
                media_info = f"\n🖼️ **图片数**: {images_count}\n📊 **表格数**: {tables_count}"

            output = f"""✅ **飞书文档创建成功！**

📄 **文档标题**: {topic}
📊 **生成想法数**: {len(ideas)}
📝 **创建块数**: {blocks_created}{media_info}
🔗 **文档链接**: {url}

---

**📋 文档内容预览（供审阅）：**

{polished}

---

💡 回复 /idea explore {topic} 可查看详细想法分析"""
            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"飞书文档创建失败: {e}")
            yield event.plain_result(f"❌ 创建失败: {e}")
    async def _idea_testblocks(self, event: AstrMessageEvent, folder_token: str = ""):
        """
        测试 Markdown 转飞书块的转换逻辑并实际创建飞书文档

        使用方式:
        /idea testblocks <folder_token>
        Example: /idea testblocks <your_folder_token>
        """
        try:
            rag_engine = self._get_engine()
            idea_engine = _create_idea_engine(self.context, rag_engine)

            yield event.plain_result("正在创建测试飞书文档（列表样式+图片+引用）...")
            result = await idea_engine.test_feishu_markdown_formats(folder_token=folder_token)

            if result.get("success"):
                url = result.get("url", "")
                blocks = result.get("blocks_created", 0)
                images = result.get("image_count", 0)
                lists = result.get("list_styles_updated", 0)
                yield event.plain_result(f"测试文档创建成功\n链接: {url}\n块数: {blocks}\n图片: {images}\n列表样式更新: {lists}")
            else:
                yield event.plain_result(f"❌ 测试失败: {result.get('error', '未知错误')}")

        except Exception as e:
            import traceback
            logger.error(f"testblocks 失败: {e}\n{traceback.format_exc()}")
            yield event.plain_result(f"❌ 测试失败: {e}")
            yield event.plain_result(f"❌ 测试失败: {e}")
    async def _idea_regen(self, event: AstrMessageEvent,
                               folder_hash: str = "",
                               refresh: str = "auto",
                               num: int = 3,
                               focus: str = "all"):
        """
        根据 folder hash 重新生成所有 ideas 以及初始周报

        使用方式:
        /idea regen <folder_hash> [refresh] [num] [focus]
        Example: /idea regen a1b2c3d4e5f6g7h8
        Example: /idea regen a1b2c3d4e5f6g7h8 5 novelty
        Example: /idea regen a1b2c3d4e5f6g7h8 refresh  # 强制重新检索知识

        Args:
            folder_hash: topic 的 folder hash（16位 MD5）
            refresh: auto(默认)使用缓存context / refresh=refresh 强制重新检索
            num: 生成想法数量（默认3）
            focus: 想法聚焦方向 (novelty/feasibility/impact/all)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not folder_hash:
            yield event.plain_result("""📚 Usage: /idea regen <folder_hash> [refresh] [num] [focus]

根据 folder hash 重新生成所有 ideas 和初始周报草稿。
refresh 参数强制重新检索本地RAG和网络搜索知识。

Examples:
  /idea regen a1b2c3d4e5f6g7h8
  /idea regen a1b2c3d4e5f6g7h8 5
  /idea regen a1b2c3d4e5f6g7h8 3 novelty
  /idea regen a1b2c3d4e5f6g7h8 refresh  # 强制重新检索

使用 /idea list 查看所有 topic 及其 folder hash。
""")
            return

        try:
            # 转换 refresh 字符串为布尔值
            logger.info(f"[IdeaEngine] regen refresh={repr(refresh)}, type={type(refresh).__name__}")
            refresh_bool = refresh.lower() == "refresh" if refresh else False
            logger.info(f"[IdeaEngine] refresh_bool={refresh_bool}")

            rag_engine = self._get_engine()
            if not rag_engine:
                yield event.plain_result("❌ RAG引擎未初始化")
                return

            idea_engine = _create_idea_engine(self.context, rag_engine)

            # 检查 folder 是否存在
            ideas_dir = idea_engine._get_ideas_dir()
            folder = ideas_dir / folder_hash
            if not folder.exists():
                yield event.plain_result(f"❌ Folder hash 不存在: {folder_hash}")
                return

            # 获取 topic 名称
            topic = idea_engine.find_topic_by_folder(folder_hash) or folder_hash

            if refresh_bool:
                yield event.plain_result(f"🔄 强制重新检索知识并重新生成...\nfolder_hash: {folder_hash}\nnum: {num}, focus: {focus}")

                # 重新检索知识
                analysis = await idea_engine.analyze_topic(topic, depth="standard")
                if not analysis:
                    yield event.plain_result("❌ 主题分析失败")
                    return

                all_queries = (analysis.search_queries + analysis.local_rag_queries)[:5]
                knowledge = await idea_engine.search_knowledge(all_queries, local_rag_top_k=25, web_top_k=25)
                yield event.plain_result(f"📚 重新检索完成: {len(knowledge.get('local_results', []))} 条本地 + {len(knowledge.get('web_results', []))} 条网络")

                # 加载现有 ideas
                ideas_list, _ = idea_engine.load_ideas_by_topic(folder_hash)
                ideas = idea_engine.convert_to_research_ideas(ideas_list)

                # 重新生成所有 ideas（使用 VLM）
                ideas = await idea_engine.generate_ideas(
                    knowledge_context=knowledge.get("fused_context", ""),
                    research_domain=analysis.domain,
                    num_ideas=num,
                    idea_focus=focus,
                    topic=topic
                )

                if not ideas:
                    yield event.plain_result("❌ 想法重新生成失败")
                    return

                # 生成初始周报草稿
                initial_draft = await idea_engine._generate_initial_draft_vlm(ideas, topic, knowledge)

                # 保存
                idea_engine.save_ideas_to_file(ideas, topic, knowledge)

            else:
                yield event.plain_result(f"🔄 正在重新生成 ideas 和初始周报...\nfolder_hash: {folder_hash}\nnum: {num}, focus: {focus}")

                # 调用 regenerate_all（使用缓存的 context）
                ideas, initial_draft, knowledge = await idea_engine.regenerate_all(
                    folder_hash=folder_hash,
                    num_ideas=num,
                    idea_focus=focus
                )

            if not ideas:
                yield event.plain_result("❌ 重新生成失败")
                return

            # 输出结果
            feasibility_bar = "★" * int(ideas[0].feasibility * 5) + "☆" * (5 - int(ideas[0].feasibility * 5))

            output = f"""✅ **重新生成完成！**

📁 **Folder Hash**: `{folder_hash}`
📊 **新生成想法数**: {len(ideas)}
🔍 **聚焦方向**: {focus}
{'🔄 **已强制重新检索知识**' if refresh_bool else ''}

---

**💡 新生成的想法 ({len(ideas)}个)**:

"""

            for i, idea in enumerate(ideas, 1):
                output += f"""**[{i}] {idea.title}**

**✨ 创新点**: {idea.novelty[:150]}...
**🔧 方法论**: {idea.methodology[:150]}...
**⚠️ 挑战**: {', '.join(idea.potential_challenges[:2])}
**📈 可行性**: {feasibility_bar} ({idea.feasibility:.0%})

---
"""

            output += f"""

---

**📝 初始周报草稿预览**：

{initial_draft[:2000]}{'...' if len(initial_draft) > 2000 else ''}

---

💡 使用 `/idea tofeishu {folder_hash} <folder_token>` 创建飞书文档。
"""
            yield event.plain_result(output)

        except ValueError as e:
            yield event.plain_result(f"❌ {e}")
        except Exception as e:
            logger.error(f"Idea 重新生成失败: {e}")
            yield event.plain_result(f"❌ 重新生成失败: {e}")
