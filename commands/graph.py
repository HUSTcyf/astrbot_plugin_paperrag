"""Graph domain commands for PaperRAG."""

from __future__ import annotations

import asyncio
import json
import gc
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast


from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .base import PluginCoreBase

_PLUGIN_DIR = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    from ..graphrag.graph_rag_engine import GraphRAGConfig
    from ..rag.hybrid_rag import HybridRAGEngine


class ChunkNode:
    """适配 GraphBuilder 的 Node 结构"""
    def __init__(self, chunk: dict[str, Any]):
        self.text = chunk.get("text", "")
        self.metadata = chunk.get("metadata", {})


class GraphCommandsMixin(PluginCoreBase):
    _CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def _is_safe_cypher_identifier(self, value: Any) -> bool:
        return isinstance(value, str) and bool(self._CYPHER_IDENTIFIER_RE.match(value))

    def _escape_cypher_identifier(self, value: str) -> str:
        """Escape a Cypher identifier — label or relationship type — to prevent injection.

        Replaces embedded backticks with a Unicode escape sequence (U+200B zero-width space)
        so that outer backticks remain unambiguous. Also replaces newlines and null bytes
        which Neo4j treats as invalid in identifiers.
        """
        if not isinstance(value, str):
            return "UNKNOWN"
        # Remove embedded backticks by replacing with zero-width space
        escaped = value.replace("`", "​")
        # Strip control characters that Neo4j rejects in identifiers
        escaped = "".join(c for c in escaped if c not in "\x00\n\r")
        return escaped if escaped else "UNKNOWN"

    def _format_node_labels(self, labels: Any) -> str:
        """Format node labels with proper Cypher escaping (backtick for all)."""
        if not isinstance(labels, list):
            return ""
        if not labels:
            return ""
        escaped = [self._escape_cypher_identifier(l) for l in labels if l and self._is_safe_cypher_identifier(l)]
        return "".join(f":`{l}`" for l in escaped) if escaped else ""

    def _format_relationship_type(self, rel_type: Any) -> str:
        """Format relationship type with Cypher backtick escaping."""
        if not self._is_safe_cypher_identifier(rel_type):
            return "`REL`"
        return f"`{rel_type}`"

    def _clean_neo4j_props(self, props: Any) -> Dict[str, Any]:
        if not isinstance(props, dict):
            return {}
        clean_props: Dict[str, Any] = {}
        for key, value in props.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean_props[key] = value
        return clean_props

    def _run_dynamic_cypher(self, session: Any, query: str, **params: Any) -> Any:
        """Run Cypher assembled from pre-validated identifiers."""
        return session.run(cast(Any, query), **params)

    async def _run_graph_build_in_background(self, engine):
        """后台运行图谱构建"""
        try:
            from ..graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig
        except ImportError:
            from graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig

        try:
            graph_config = self._create_graph_rag_config()

            graph_engine = GraphRAGEngine(graph_config, engine, self.context)
            await graph_engine.initialize()

            # 获取所有文档并构建图谱
            papers_dir = self.config.get("papers_dir", "./papers")
            doc_files = self._scan_documents(papers_dir)

            if not doc_files:
                return

            parser = engine._ensure_parser_initialized()
            all_nodes = []
            for doc_file in doc_files:
                try:
                    nodes = await parser.parse_and_split(str(doc_file), {}, None)
                    all_nodes.extend(nodes)
                except Exception:
                    pass

            await graph_engine.build_graph_from_nodes(all_nodes)
            logger.info("✅ 后台知识图谱构建完成")
        except Exception as e:
            logger.error(f"❌ 后台知识图谱构建失败: {e}")


    async def _paper_graph_build(self, event: AstrMessageEvent, confirm: str = '', skip: str = ''):
        """Build knowledge graph from indexed documents

        Args:
            confirm: Must be 'confirm' to proceed
            skip: Number of papers to skip (e.g., '30' to skip first 30 papers)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        # 检查点文件路径
        plugin_dir = _PLUGIN_DIR
        checkpoint_file = plugin_dir / "data" / "graph_build_checkpoint.json"

        if confirm != "confirm":
            skip_hint = ""
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        ckpt = json.load(f)
                        sc = ckpt.get("skip_count", 0)
                        if sc > 0:
                            skip_hint = f"\n⚠️ 检测到检查点（已处理 {sc} 篇），将自动从第 {sc} 篇后继续"
                except Exception:
                    pass
            yield event.plain_result(
                "⚠️ 即将从已索引的文档构建知识图谱\n"
                "注意：构建过程在后台运行，可随时中断（检查点自动保存）\n"
                "使用 /paper graph_build confirm [N] 确认执行\n"
                "例如：/paper graph_build confirm 30 表示跳过前30篇，从第31篇开始"
                + skip_hint
            )
            return

        # 检查 Graph RAG 是否启用
        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用\n请在插件配置中启用 enable_graph_rag")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        # 解析 skip 参数
        skip_count = 0
        if skip:
            skip_str = str(skip).strip()
            # 移除 skip= 前缀（如果存在）
            if '=' in skip_str:
                skip_str = skip_str.split('=', 1)[1].strip()
            # 移除非数字字符
            skip_str = ''.join(c for c in skip_str if c.isdigit())
            if skip_str:
                skip_count = int(skip_str)
            else:
                yield event.plain_result(f"❌ skip 参数无效: {skip}，请使用数字")
                return

        # 立即返回，后台任务开始构建
        yield event.plain_result(f"🔨 知识图谱构建已在后台启动...\n📋 查看进度：检查 AstrBot 控制台日志\n💾 每篇论文处理完自动保存检查点")

        # 启动后台任务
        asyncio.create_task(self._graph_build_background_task(event, engine, skip_count))


    async def _graph_build_background_task(self, event: AstrMessageEvent, engine, skip_count: int = 0):
        """后台运行图谱构建任务（支持检查点和进度日志）"""
        from pathlib import Path

        plugin_dir = _PLUGIN_DIR

        def send_msg(text: str):
            """发送消息到用户（仅通过日志）"""
            logger.info(f"[GraphBuild] {text}")

        try:
            # 获取索引管理器
            index_manager = engine._ensure_index_manager_initialized()

            send_msg("📖 正在从向量数据库读取论文列表...")

            try:
                papers = await index_manager.list_unique_documents()
            except Exception as e:
                send_msg(f"❌ 无法获取论文列表: {e}\n请确保已使用 /paper add 添加文档")
                return

            if not papers:
                send_msg("📭 向量数据库中未找到已索引的文档\n请先使用 /paper add 添加文档")
                return

            paper_names = [p.get("file_name", "") for p in papers if p.get("file_name")]
            send_msg(f"📚 找到 {len(paper_names)} 篇论文，开始逐篇构建...")

            # 导入必要的模块
            try:
                from ..graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig
            except ImportError:
                from graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig

            try:
                from ..graphrag.graph_builder import MultimodalGraphBuilder
            except ImportError:
                from graphrag.graph_builder import MultimodalGraphBuilder

            import json as _json_json

            send_msg(f"📑 开始逐篇构建知识图谱 ({len(paper_names)} 篇论文)...")

            # 创建 GraphRAGConfig（只创建一次）
            graph_config = self._create_graph_rag_config()

            # Neo4j 存储
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

            try:
                from ..graphrag.graph_rag_engine import SimplePropertyGraphStoreAdapter
            except ImportError:
                from graphrag.graph_rag_engine import SimplePropertyGraphStoreAdapter
            raw_store = Neo4jPropertyGraphStore(
                url=graph_config.neo4j_uri,
                username=graph_config.neo4j_user,
                password=graph_config.neo4j_password,
                database="neo4j",
                refresh_schema=True
            )
            graph_store = SimplePropertyGraphStoreAdapter(raw_store)
            logger.info(f"[GraphRAG] 使用 Neo4j 存储: {graph_config.neo4j_uri}")

            builder = MultimodalGraphBuilder(config=graph_config, context=self.context)

            # 初始化 LLM（只初始化一次）
            await builder._ensure_llm_initialized()

            # 逐篇处理：每篇论文处理完后立即构建图谱，不累积所有 chunks
            total_stats = {
                "entities_added": 0,
                "text_triplets_added": 0,
                "image_entities_added": 0,
                "cross_modal_triplets_added": 0,
                "chunks_processed": 0,
                "chunks_with_images": 0,
                "chunks_failed": 0,
                "chunks_empty": 0
            }

            # 逐篇加载 chunks 并立即构建图谱
            await index_manager._ensure_collection()
            collection = index_manager._collection

            # 检查点机制：优先使用传入的 skip_count，否则尝试读取检查点文件
            import json as _json
            checkpoint_file = plugin_dir / "data" / "graph_build_checkpoint.json"
            if skip_count == 0 and checkpoint_file.exists():
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        ckpt = _json.load(f)
                        skip_count = ckpt.get("skip_count", 0)
                        if skip_count > 0:
                            send_msg(f"🔄 检测到检查点，将跳过前 {skip_count} 篇已处理的论文")
                except Exception:
                    pass
            elif skip_count > 0:
                send_msg(f"⏭️ 使用 skip 参数，跳过前 {skip_count} 篇论文")

            for i, paper_name in enumerate(paper_names):
                # 跳过已处理的论文
                if i < skip_count:
                    continue

                paper_name_escaped = paper_name.replace('"', '\\"')
                try:
                    # collection.query 是同步方法，需要用 run_in_executor 包装
                    _collection = cast(Any, collection)
                    raw_results = cast(
                        List[Dict[str, Any]],
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            cast(Any, lambda pn=paper_name_escaped: _collection.query(
                                expr=f'metadata["file_name"] == "{pn}"',
                                output_fields=["id", "text", "metadata"],
                            ))
                        )
                    )

                    if not raw_results:
                        continue

                    send_msg(f"📄 [{i+1}/{len(paper_names)}] {paper_name} ({len(raw_results)} chunks)")

                    # 解析该论文的 chunks
                    paper_chunks = []
                    for row in raw_results:
                        chunk = {
                            "id": row.get("id"),
                            "text": row.get("text", ""),
                        }
                        meta = row.get("metadata", "{}")
                        if isinstance(meta, str):
                            try:
                                meta = _json_json.loads(meta)
                            except Exception:
                                meta = {"raw": meta}
                        chunk["metadata"] = meta
                        paper_chunks.append(chunk)

                    # 立即为该论文创建节点并构建图谱
                    nodes = [ChunkNode(chunk) for chunk in paper_chunks]
                    stats = await builder.build_from_nodes(nodes, graph_store)

                    # 每篇论文处理完后清理内存
                    gc.collect()

                    # 累积统计
                    total_stats["entities_added"] += stats.get("entities_added", 0)
                    total_stats["text_triplets_added"] += stats.get("text_triplets_added", 0)
                    total_stats["image_entities_added"] += stats.get("image_entities_added", 0)
                    total_stats["cross_modal_triplets_added"] += stats.get("cross_modal_triplets_added", 0)
                    total_stats["chunks_with_images"] += stats.get("chunks_with_images", 0)
                    total_stats["chunks_failed"] += stats.get("chunks_failed", 0)
                    total_stats["chunks_empty"] += stats.get("chunks_empty", 0)
                    total_stats["chunks_processed"] += stats.get("chunks_processed", 0)

                    # 每篇论文后保存检查点（保证最多丢失1篇）
                    processed_count = i + 1
                    try:
                        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                        temp_file = checkpoint_file.with_suffix('.tmp')
                        with open(temp_file, "w", encoding="utf-8") as f:
                            _json.dump({
                                "skip_count": processed_count,
                                "saved_at": datetime.now().isoformat()
                            }, f, ensure_ascii=False)
                        temp_file.replace(checkpoint_file)
                        logger.debug(f"💾 检查点已保存: {processed_count} 篇论文")
                    except Exception as e:
                        logger.warning(f"⚠️ 保存检查点失败: {e}")

                    # 每篇论文都更新进度
                    send_msg(f"📥 进度: {i + 1}/{len(paper_names)} 篇论文...")

                except Exception as e:
                    logger.warning(f"处理论文 {paper_name} 失败: {e}")
                    # 发生异常时保存检查点
                    try:
                        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                        temp_file = checkpoint_file.with_suffix('.tmp')
                        with open(temp_file, "w", encoding="utf-8") as f:
                            _json.dump({
                                "skip_count": i,  # 从当前论文重试
                                "saved_at": datetime.now().isoformat(),
                                "last_paper": paper_name
                            }, f, ensure_ascii=False)
                        temp_file.replace(checkpoint_file)
                        send_msg(f"⚠️ 处理论文 {paper_name} 失败，已保存检查点")
                    except Exception:
                        pass

                # 每隔20篇论文重新确保连接有效
                if (i + 1) % 20 == 0:
                    await index_manager._ensure_collection()
                    collection = index_manager._collection

            # 构建成功，删除检查点文件
            if checkpoint_file.exists():
                try:
                    checkpoint_file.unlink()
                    logger.debug("💾 检查点文件已清除")
                except Exception:
                    pass

            # 输出最终结果
            text_triplets = total_stats.get('text_triplets_added', 0)
            cross_triplets = total_stats.get('cross_modal_triplets_added', 0)
            total_triplets_val = text_triplets + cross_triplets
            output = f"""✅ **知识图谱构建完成**

📊 构建统计：
   • 处理论文：{len(paper_names)} 篇
   • 处理文档块：{total_stats.get('chunks_processed', 0)}
   • 添加实体：{total_stats.get('entities_added', 0)}
   • 文本三元组：{text_triplets}
   • 图片实体：{total_stats.get('image_entities_added', 0)}
   • 跨模态三元组：{cross_triplets}
   • 总三元组：{total_triplets_val}
   • 空块数：{total_stats.get('chunks_empty', 0)}
   • 失败块数：{total_stats.get('chunks_failed', 0)}

💡 使用 /paper graph_stats 查看图谱详情"""
            send_msg(output)

        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            send_msg(f"❌ 构建失败: {e}")


    async def _paper_graph_stats(self, event: AstrMessageEvent):
        """Show knowledge graph statistics"""
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用\n请在插件配置中启用 enable_graph_rag")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        try:
            from ..graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig
        except ImportError:
            from graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig

        try:
            graph_config = self._create_graph_rag_config()

            graph_engine = GraphRAGEngine(graph_config, engine, self.context)
            await graph_engine.initialize()

            stats = await graph_engine.get_graph_stats()

            storage_type = self.config.get("graph_rag", {}).get("storage_type", "neo4j")

            output = f"""📊 **知识图谱统计**

存储类型：{storage_type}
   • 实体数量：{stats.get('entity_count', 0)}
   • 关系数量：{stats.get('relation_count', 0)}
   • 索引大小：{stats.get('index_size', 0)}

💡 使用 /paper graph_build confirm 构建图谱
💡 使用 /paper graph_rebuild confirm 重新构建图谱
💡 使用 /paper graph_clear confirm 清空图谱"""

            yield event.plain_result(output)

        except Exception as e:
            logger.error(f"获取图谱统计失败: {e}")
            yield event.plain_result(f"❌ 获取统计失败: {e}")


    async def _paper_graph_rebuild(self, event: AstrMessageEvent, confirm: str = ''):
        """Rebuild knowledge graph from scratch (clear + rebuild)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result(
                "⚠️ 即将清空并重新构建知识图谱\n"
                "执行前会自动备份当前图谱\n"
                "使用 /paper graph_rebuild confirm 确认执行"
            )
            return

        # 并发保护：防止多个重建任务同时运行
        if getattr(self, '_is_rebuilding', False):
            yield event.plain_result("⚠️ 图谱重建任务已在运行中，请等待完成后重试")
            return
        self._is_rebuilding = True

        try:
            if not self.config.get("enable_graph_rag", False):
                yield event.plain_result("❌ Graph RAG 功能未启用\n请在插件配置中启用 enable_graph_rag")
                return

            engine = self._get_engine()
            if not engine:
                yield event.plain_result("❌ RAG引擎未就绪")
                return

            try:
                from ..graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig
            except ImportError:
                from graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig

            graph_config = self._create_graph_rag_config()

            # 步骤0: 检查图谱是否为空，非空才备份
            node_count = -1
            driver = None
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    graph_config.neo4j_uri,
                    auth=(graph_config.neo4j_user, graph_config.neo4j_password)
                )
                driver.verify_connectivity()
                with driver.session() as session:
                    record = session.run("MATCH (n) RETURN count(n) AS cnt").single()
                    node_count = record["cnt"] if record else 0
            except Exception as e:
                logger.warning(f"[GraphRebuild] Neo4j 未就绪: {e}")
                node_count = -1
            finally:
                if driver:
                    driver.close()

            if node_count == 0:
                yield event.plain_result("📭 当前图谱为空，跳过备份")
            elif node_count > 0:
                try:
                    dump_result = await self._online_backup(graph_config)
                    if dump_result.get("status") == "success":
                        dump_file = dump_result.get("backup_file", "online backup")
                        dump_size = dump_result.get("size", "unknown")
                        dump_nodes = dump_result.get("nodes", "?")
                        dump_rels = dump_result.get("relations", "?")
                        yield event.plain_result(f"✅ 备份完成: {dump_file} ({dump_size}), {dump_nodes} 节点, {dump_rels} 关系")
                    else:
                        logger.error(f"[GraphRebuild] 备份失败: {dump_result.get('message', '未知错误')}")
                        yield event.plain_result(f"❌ 备份失败: {dump_result.get('message', '未知错误')}，中止重建以保护数据")
                        return
                except Exception as e:
                    logger.error(f"[GraphRebuild] 备份失败（中止重建）: {e}")
                    yield event.plain_result(f"❌ 备份异常: {e}，中止重建以保护数据")
                    return
            # node_count == -1 时静默跳过备份，继续执行

            # 步骤1: 清空现有图谱
            yield event.plain_result("🗑️ 正在清空现有知识图谱...")

            graph_engine = GraphRAGEngine(graph_config, engine, self.context)
            await graph_engine.initialize()

            clear_result = await graph_engine.clear_graph()
            if clear_result.get("status") == "success":
                yield event.plain_result("✅ 现有图谱已清空")
            else:
                logger.error(f"清空图谱失败: {clear_result}")
                yield event.plain_result(f"❌ 清空图谱失败，中止重建以保护数据: {clear_result.get('message', '未知错误')}")
                return

            # 步骤2: 删除检查点，从头构建图谱
            checkpoint_file = _PLUGIN_DIR / "data" / "graph_build_checkpoint.json"
            if checkpoint_file.exists():
                try:
                    checkpoint_file.unlink()
                except Exception:
                    pass

            yield event.plain_result("🔨 知识图谱重建已在后台启动...\n📋 查看进度：检查 AstrBot 控制台日志")

            asyncio.create_task(self._graph_build_background_task(event, engine, skip_count=0))
        finally:
            self._is_rebuilding = False


    async def _paper_graph_clear(self, event: AstrMessageEvent, confirm: str = ''):
        """Clear knowledge graph (Admin)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用")
            return

        if confirm != "confirm":
            yield event.plain_result("⚠️ 即将清空知识图谱\n此操作不可恢复！\n使用 /paper graph_clear confirm 确认执行")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        try:
            from ..graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig
        except ImportError:
            from graphrag.graph_rag_engine import GraphRAGEngine, GraphRAGConfig

        try:
            graph_config = self._create_graph_rag_config()

            graph_engine = GraphRAGEngine(graph_config, engine, self.context)
            await graph_engine.initialize()

            result = await graph_engine.clear_graph()

            if result.get("status") == "success":
                yield event.plain_result("✅ 知识图谱已清空")
            else:
                yield event.plain_result(f"❌ 清空失败: {result.get('message', '未知错误')}")

        except Exception as e:
            logger.error(f"清空图谱失败: {e}")
            yield event.plain_result(f"❌ 清空失败: {e}")


    async def _paper_graph_backup(self, event: AstrMessageEvent, mode: str = 'online'):
        """Backup Neo4j knowledge graph (Admin)

        Args:
            mode: 'online' (Cypher export, no downtime) or 'dump' (neo4j-admin dump, requires stop)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用")
            return

        try:
            from ..graphrag.graph_rag_engine import GraphRAGConfig
        except ImportError:
            from graphrag.graph_rag_engine import GraphRAGConfig
        graph_config = self._create_graph_rag_config()

        yield event.plain_result(f"🔄 开始备份图谱 (模式: {mode})...")

        try:
            if mode == 'dump':
                logger.warning("[GraphRAG] dump 模式已禁用，改用在线备份")
                result = await self._online_backup(graph_config)
            elif mode == 'offline':
                result = await self._offline_backup(graph_config)
            else:
                result = await self._online_backup(graph_config)

            if result["status"] == "success":
                backup_file = result.get("backup_file", result.get("dump_file", "unknown"))
                size = result.get("size", 0)
                nodes = result.get("nodes", "?")
                rels = result.get("relations", "?")

                output = f"""✅ **图谱备份完成！**

📦 **备份文件**: `{backup_file}`
📊 **数据统计**:
   • 节点数: {nodes}
   • 关系数: {rels}
   • 文件大小: {size}"""

                if mode == 'offline':
                    output += "\n\n⚠️ 离线目录备份需手动恢复，当前 `/paper graph_restore` 仅支持在线 JSON 备份"
                elif mode == 'dump':
                    output += "\n\n💡 使用 `neo4j-admin database load neo4j --from=neo4j.dump` 恢复"
                else:
                    output += f"\n\n💡 使用 `/paper graph_restore {backup_file}` 恢复备份"

                yield event.plain_result(output)
            else:
                yield event.plain_result(f"❌ 备份失败: {result.get('message', '未知错误')}")

        except Exception as e:
            logger.error(f"备份图谱失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 备份失败: {e}")

    async def _online_backup(self, graph_config: "GraphRAGConfig") -> dict:
        """在线备份：使用 Cypher 导出为 JSON"""
        import json
        import gzip
        from pathlib import Path
        from datetime import datetime

        try:
            from neo4j import GraphDatabase
        except ImportError:
            return {"status": "error", "message": "请安装 neo4j 驱动: pip install neo4j"}

        # 连接到 Neo4j
        driver = GraphDatabase.driver(
            graph_config.neo4j_uri,
            auth=(graph_config.neo4j_user, graph_config.neo4j_password)
        )

        # 获取插件目录（用于相对路径）
        plugin_dir = _PLUGIN_DIR
        backup_dir = plugin_dir / "data" / "graph_store"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"neo4j_backup_{timestamp}.json.gz"

        try:
            with driver.session() as session:
                tx = session.begin_transaction()
                try:
                    # 在同一事务中导出节点和关系，确保一致性
                    nodes_data = tx.run("""
                        MATCH (n)
                        RETURN labels(n) as labels,
                               properties(n) as props,
                               elementId(n) as id
                    """).data()

                    rels_data = tx.run("""
                        MATCH (a)-[r]->(b)
                        RETURN type(r) as rel_type,
                               properties(r) as props,
                               elementId(startNode(r)) as start_id,
                               elementId(endNode(r)) as end_id
                    """).data()
                    tx.commit()
                except Exception:
                    tx.rollback()
                    raise

            nodes_count = len(nodes_data)
            rels_count = len(rels_data)

            backup = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "mode": "online",
                "nodes": nodes_data,
                "relationships": rels_data,
                "node_count": nodes_count,
                "rel_count": rels_count
            }

            # 压缩写入
            with gzip.open(backup_file, 'wt', encoding='utf-8') as f:
                json.dump(backup, f, indent=2, ensure_ascii=False)

            size = backup_file.stat().st_size
            size_str = self._format_size(size)

            logger.info(f"[GraphRAG] 在线备份完成: {backup_file}, {nodes_count} 节点, {rels_count} 关系")

            return {
                "status": "success",
                "backup_file": str(backup_file.relative_to(plugin_dir)),
                "size": size_str,
                "nodes": nodes_count,
                "relations": rels_count
            }

        finally:
            driver.close()




    async def _offline_backup(self, graph_config: "GraphRAGConfig") -> dict:
        """离线备份：复制 Neo4j 数据目录"""
        import shutil
        import subprocess
        from pathlib import Path
        from datetime import datetime

        neo4j_data_dir = Path("/opt/homebrew/var/neo4j/data")
        if not neo4j_data_dir.exists():
            return {"status": "error", "message": f"Neo4j 数据目录不存在: {neo4j_data_dir}"}

        plugin_dir = _PLUGIN_DIR
        backup_dir = plugin_dir / "data" / "graph_store"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = backup_dir / f"neo4j_backup_{timestamp}"
        backup_subdir.mkdir(parents=True, exist_ok=True)

        try:
            neo4j_bin = shutil.which("neo4j")
            if not neo4j_bin:
                return {"status": "error", "message": "neo4j 命令未找到"}

            logger.info("[GraphRAG] 停止 Neo4j 服务...")
            stop_result = subprocess.run(["neo4j", "stop"], capture_output=True, text=True)
            if stop_result.returncode != 0:
                logger.warning(f"[GraphRAG] neo4j stop: {stop_result.stdout} {stop_result.stderr}")

            import time
            time.sleep(3)

            source_db = neo4j_data_dir / "databases" / "neo4j"
            dest_db = backup_subdir / "databases" / "neo4j"
            if source_db.exists():
                shutil.copytree(source_db, dest_db)
            else:
                return {"status": "error", "message": f"数据库目录不存在: {source_db}"}

            source_tx = neo4j_data_dir / "transactions" / "neo4j"
            dest_tx = backup_subdir / "transactions" / "neo4j"
            if source_tx.exists():
                shutil.copytree(source_tx, dest_tx)

            logger.info("[GraphRAG] 重启 Neo4j 服务...")
            subprocess.run(["neo4j", "start"], capture_output=True)
            time.sleep(5)

            total_size = sum(f.stat().st_size for f in backup_subdir.rglob('*') if f.is_file())
            size_str = self._format_size(total_size)

            logger.info(f"[GraphRAG] 离线备份完成: {backup_subdir}")
            return {
                "status": "success",
                "backup_file": str(backup_subdir.relative_to(plugin_dir)),
                "size": size_str,
                "nodes": "多个",
                "relations": "多个"
            }
        except Exception as e:
            logger.error(f"[GraphRAG] 离线备份失败: {e}")
            try:
                subprocess.run(["neo4j", "start"], capture_output=True)
            except Exception:
                pass
            return {"status": "error", "message": str(e)}


    def _format_size(self, size: float) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


    async def _paper_graph_restore(self, event: AstrMessageEvent, backup_file: str = ''):
        """Restore Neo4j knowledge graph from backup (Admin)

        Args:
            backup_file: 备份文件名（从 data/graph_store 目录）
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用")
            return

        if not backup_file:
            async for result in self._paper_graph_backup_list(event):
                yield result
            return

        try:
            from ..graphrag.graph_rag_engine import GraphRAGConfig
        except ImportError:
            from graphrag.graph_rag_engine import GraphRAGConfig
        graph_config = self._create_graph_rag_config()

        yield event.plain_result(f"🔄 正在恢复备份: {backup_file}...")

        try:
            result = await self._restore_backup(backup_file, graph_config)

            if result["status"] == "success":
                yield event.plain_result(f"""✅ **备份恢复完成！**

📦 **已恢复**: `{backup_file}`
📊 **数据**: {result.get('nodes', '?')} 节点, {result.get('relations', '?')} 关系

⚠️ 如果使用离线备份，恢复后需要重启 Neo4j 服务""")
            else:
                yield event.plain_result(f"❌ 恢复失败: {result.get('message', '未知错误')}")

        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            yield event.plain_result(f"❌ 恢复失败: {e}")


    async def _restore_backup(self, backup_file: str, graph_config: "GraphRAGConfig") -> dict:
        """从备份恢复"""
        import json
        import gzip
        from pathlib import Path

        try:
            from neo4j import GraphDatabase
        except ImportError:
            return {"status": "error", "message": "请安装 neo4j 驱动: pip install neo4j"}

        plugin_dir = _PLUGIN_DIR
        # 防止路径遍历：只取文件名部分
        safe_name = Path(backup_file).name
        backup_path = plugin_dir / "data" / "graph_store" / safe_name

        if not backup_path.exists():
            return {"status": "error", "message": f"备份文件不存在: {backup_path}"}

        if backup_path.is_dir():
            return {
                "status": "error",
                "message": "离线目录备份不支持通过 /paper graph_restore 自动恢复，请手动停止 Neo4j 后恢复数据目录。当前命令仅支持在线 JSON 备份（.json 或 .json.gz）。",
            }

        if not (backup_path.name.endswith(".json") or backup_path.name.endswith(".json.gz")):
            return {
                "status": "error",
                "message": "不支持的备份格式，当前命令仅支持在线 JSON 备份（.json 或 .json.gz）。",
            }

        driver = GraphDatabase.driver(
            graph_config.neo4j_uri,
            auth=(graph_config.neo4j_user, graph_config.neo4j_password)
        )

        try:
            # 根据文件扩展名判断格式
            if str(backup_path).endswith('.gz'):
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                    backup = json.load(f)
            else:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup = json.load(f)

            # Validate backup structure before destructive operations
            nodes = backup.get("nodes", [])
            rels = backup.get("relationships", [])
            for n in nodes:
                if "id" not in n or "labels" not in n:
                    raise ValueError(f"Invalid backup node structure: {n}")

            with driver.session(database="neo4j") as session:
                # Use transaction for atomic restore with rollback on failure
                tx = session.begin_transaction()
                try:
                    # 清空现有数据（在事务内）
                    tx.run("MATCH (n) DETACH DELETE n")

                    # 恢复节点
                    for node in nodes:
                        labels = node.get("labels", [])
                        props = node.get("props", {})
                        backup_id = node.get("id")
                        label_str = self._format_node_labels(labels)
                        clean_props = self._clean_neo4j_props(props)
                        if backup_id:
                            clean_props["__backup_id"] = backup_id
                        tx.run(
                            f"CREATE (n{label_str}) SET n = $props",  # type: ignore[arg-type]
                            props=clean_props,
                        )

                    # 恢复关系
                    for rel in rels:
                        rel_type = rel.get("rel_type", "REL")
                        start_id = rel.get("start_id")
                        end_id = rel.get("end_id")
                        props = rel.get("props", {})
                        if start_id and end_id:
                            clean_props = self._clean_neo4j_props(props)
                            rel_type_str = self._format_relationship_type(rel_type)
                            tx.run(
                                f"MATCH (a {{__backup_id: $start_id}}), (b {{__backup_id: $end_id}}) "
                                f"CREATE (a)-[r:{rel_type_str}]->(b) SET r = $props",  # type: ignore[arg-type]
                                start_id=start_id,
                                end_id=end_id,
                                props=clean_props,
                            )

                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    raise RuntimeError(f"Restore failed, rolled back: {e}") from e

            # 清理 backup_id 属性（事务外）
            session.run("MATCH (n) REMOVE n.__backup_id")

            logger.info(f"[GraphRAG] 备份恢复完成: {len(nodes)} 节点, {len(rels)} 关系")

            return {
                "status": "success",
                "nodes": len(nodes),
                "relations": len(rels)
            }

        finally:
            driver.close()


    async def _paper_graph_backup_list(self, event: AstrMessageEvent):
        """List available graph backups (Admin)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not self.config.get("enable_graph_rag", False):
            yield event.plain_result("❌ Graph RAG 功能未启用")
            return

        from pathlib import Path
        from datetime import datetime

        plugin_dir = _PLUGIN_DIR
        backup_dir = plugin_dir / "data" / "graph_store"

        all_backups: List[Dict[str, Any]] = []

        # neo4j.dump
        dump_file = backup_dir / "neo4j.dump"
        if dump_file.exists():
            all_backups.append({
                "name": dump_file.name,
                "path": dump_file,
                "size": dump_file.stat().st_size,
                "mtime": dump_file.stat().st_mtime,
                "type": "neo4j-admin dump"
            })

        # 在线 JSON 备份
        for b in backup_dir.glob("neo4j_backup_*.json.gz"):
            all_backups.append({
                "name": b.name,
                "path": b,
                "size": b.stat().st_size,
                "mtime": b.stat().st_mtime,
                "type": "online (JSON.gz)"
            })

        if not all_backups:
            yield event.plain_result("❌ 未找到任何备份文件\n\n💡 使用 `/paper graph_backup` 创建备份")
            return

        # 按时间排序
        cast(List[Dict[str, Any]], all_backups).sort(key=lambda x: x["mtime"], reverse=True)

        msg = "📦 **图谱备份列表**:\n\n"
        for i, b_item in enumerate(cast(List[Dict[str, Any]], all_backups)[:10], 1):
            size_str = self._format_size(b_item["size"])
            mtime = datetime.fromtimestamp(b_item["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
            msg += f"{i}. `{b_item['name']}`\n"
            msg += f"   类型: {b_item['type']}, 大小: {size_str}\n"
            msg += f"   时间: {mtime}\n\n"

        msg += "💡 使用 `/paper graph_restore <文件名>` 恢复在线 JSON 备份\n"
        msg += "💡 neo4j.dump 需使用 `neo4j-admin database load` 恢复\n"
        msg += "💡 使用 `/paper graph_backup` 创建新备份"

        yield event.plain_result(msg)


    async def _paper_graph_link(self, event: AstrMessageEvent, action: str = 'status'):
        """Manage Neo4j data symlink (Admin)

        Args:
            action: 'create' | 'remove' | 'status' (default: status)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        from pathlib import Path
        import os

        # Neo4j 原始数据目录
        neo4j_db_path = Path("/opt/homebrew/var/neo4j/data/databases/neo4j")

        # 插件内的链接目标路径
        plugin_dir = _PLUGIN_DIR
        graph_store_dir = plugin_dir / "data" / "graph_store"
        graph_store_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = graph_store_dir / "neo4j_data"

        if action == 'status':
            # 检查符号链接状态
            if symlink_path.is_symlink():
                target = os.readlink(symlink_path)
                exists = symlink_path.exists()
                real_path = symlink_path.resolve()

                status = "✅ 正常" if exists else "⚠️ 链接断裂（目标不存在）"

                msg = f"""📊 **Neo4j 符号链接状态**

🔗 链接路径: `data/graph_store/neo4j_data`
📍 目标路径: `{target}`
📁 解析路径: `{real_path}`
📦 状态: {status}

💡 可用操作:
  `/paper graph_link create` - 创建/重建链接
  `/paper graph_link remove` - 删除链接"""
            elif symlink_path.exists():
                status = "⚠️ 存在同名文件/目录（非符号链接）"
                msg = f"""📊 **Neo4j 符号链接状态**

⚠️  `data/graph_store/neo4j_data` 已存在但不是符号链接
   状态: {status}

💡 请先删除或备份后再创建链接:
  `/paper graph_link remove`"""
            else:
                msg = f"""📊 **Neo4j 符号链接状态**

❌ 符号链接未创建

💡 创建链接:
  `/paper graph_link create`"""

            yield event.plain_result(msg)

        elif action == 'create':
            if not neo4j_db_path.exists():
                yield event.plain_result(f"❌ Neo4j 数据目录不存在: `{neo4j_db_path}`\n\n请确保 Neo4j 已安装并初始化")
                return

            # 如果已存在符号链接，先删除
            if symlink_path.is_symlink():
                symlink_path.unlink()
                logger.info(f"[GraphRAG] 已删除旧符号链接")
            elif symlink_path.exists():
                yield event.plain_result(f"⚠️ `data/graph_store/neo4j_data` 已存在且不是符号链接\n请先手动删除后再试")
                return

            try:
                # 创建符号链接（使用相对路径）
                # 从 graph_store 目录到 neo4j 数据目录的相对路径
                rel_path = os.path.relpath(neo4j_db_path, graph_store_dir)
                os.symlink(rel_path, symlink_path)

                yield event.plain_result(f"""✅ **符号链接创建成功！**

🔗 链接: `data/graph_store/neo4j_data`
📍 指向: `{rel_path}` (相对路径)

💡 现在可以直接在 `data/graph_store/neo4j_data` 访问 Neo4j 数据

⚠️ 注意: 删除此链接不会影响原始数据，但重建需要此命令""")

            except Exception as e:
                yield event.plain_result(f"❌ 创建符号链接失败: {e}")

        elif action == 'remove':
            if not symlink_path.exists():
                yield event.plain_result("❌ 符号链接不存在，无需删除")
                return

            if not symlink_path.is_symlink():
                yield event.plain_result("⚠️ `data/graph_store/neo4j_data` 不是符号链接，无法使用此命令删除\n请手动删除")
                return

            try:
                symlink_path.unlink()
                yield event.plain_result("✅ 符号链接已删除\n\n⚠️ 删除链接不影响原始 Neo4j 数据\n💡 使用 `/paper graph_link create` 重新创建链接")
            except Exception as e:
                yield event.plain_result(f"❌ 删除符号链接失败: {e}")

        else:
            yield event.plain_result(f"❌ 未知操作: {action}\n\n可用操作: `status` | `create` | `remove`")
