"""
Graph RAG Engine - 图谱增强检索引擎

存储后端：
- neo4j: Neo4jPropertyGraphStore（唯一支持的存储后端）

图谱检索作为 RRF 第四通道融入 HybridRetriever，不再独立运行。
"""

import asyncio
import gc
import shutil
import subprocess
import time
import traceback
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import logger

if TYPE_CHECKING:
    from ..rag.rag_engine import RAGConfig

try:
    from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import LLMSynonymRetriever
except Exception as e:
    logger.warning(f"[GraphRAG] LLMSynonymRetriever 导入失败: {e}")
    logger.warning(traceback.format_exc())
    LLMSynonymRetriever = None


# Case-insensitive wrapper for LLMSynonymRetriever.
# Default LLMSynonymRetriever capitalizes synonyms ("PSNR"→"Psnr"),
# which then fails to match entity IDs stored in original case.
# This subclass emits all case variants for robust matching.
if LLMSynonymRetriever is not None:
    class _CaseInsensitiveSynonymRetriever(LLMSynonymRetriever):
        def _parse_llm_output(self, output: str) -> list:
            if self._output_parsing_fn:
                return self._output_parsing_fn(output)
            raw = [x.strip() for x in output.strip().split("^") if x.strip()]
            expanded = set()
            for m in raw:
                expanded.add(m)
                expanded.add(m.capitalize())
                expanded.add(m.upper())
                expanded.add(m.lower())
            return list(expanded)
else:
    _CaseInsensitiveSynonymRetriever = None  # type: ignore[assignment, misc]


@dataclass
class GraphRAGConfig:
    """Graph RAG 配置类"""
    enable_graph_rag: bool = False
    storage_type: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    max_triplets_per_chunk: int = 5
    graph_retrieval_top_k: int = 5
    graph_rrf_weight: float = 0.2  # 图谱在 RRF 融合中的权重
    auto_build: bool = False  # 是否自动构建图谱
    auto_build_threshold: int = 10  # 自动构建阈值
    # 多模态配置
    multimodal_enabled: bool = True  # 是否启用多模态图谱抽取
    max_images_per_chunk: int = 1  # 每个chunk最多处理图片数
    extract_image_entities: bool = True  # 是否提取图片为实体

    @classmethod
    def from_rag_config(cls, config: "RAGConfig") -> "GraphRAGConfig":
        """从 RAGConfig 创建 GraphRAGConfig"""
        return cls(
            enable_graph_rag=getattr(config, 'enable_graph_rag', False),
            storage_type=getattr(config, 'graph_storage_type', 'neo4j'),
            neo4j_uri=getattr(config, 'graph_neo4j_uri', 'bolt://localhost:7687'),
            neo4j_user=getattr(config, 'graph_neo4j_user', 'neo4j'),
            neo4j_password=getattr(config, 'graph_neo4j_password', ''),
            max_triplets_per_chunk=getattr(config, 'graph_max_triplets_per_chunk', 5),
            graph_retrieval_top_k=getattr(config, 'graph_retrieval_top_k', 5),
            graph_rrf_weight=getattr(config, 'graph_rrf_weight', None) or getattr(config, 'hybrid_alpha', 0.2),
            auto_build=getattr(config, 'graph_auto_build', False),
            auto_build_threshold=getattr(config, 'graph_auto_build_threshold', 10),
            multimodal_enabled=getattr(config, 'graph_multimodal_enabled', True),
            max_images_per_chunk=getattr(config, 'graph_max_images_per_chunk', 1),
            extract_image_entities=getattr(config, 'graph_extract_image_entities', True),
        )


class SimplePropertyGraphStoreAdapter:

    def __init__(self, graph_store: Any):
        self._store = graph_store
        self._entity_info: Dict[str, Dict[str, Any]] = {}
        self._relation_count = 0

    @property
    def _driver(self):
        """获取 Neo4j driver"""
        return getattr(self._store, 'client', None) or getattr(self._store, '_driver', None)

    def add_entity(
        self,
        name: str,
        entity_type: str = "UNKNOWN",
        description: str = "",
        chunk_id: str = ""
    ) -> Any:
        """添加实体到图谱（如果实体已存在则不覆盖）"""
        try:
            if name.lower() not in self._entity_info:
                driver = self._driver
                if driver:
                    # 正确的转义：先转义反斜杠，再转义单引号
                    def escape_cypher(s: str) -> str:
                        if not s:
                            return ""
                        return s.replace("\\", "\\\\").replace("'", "\\'")

                    escaped_name = escape_cypher(name)
                    escaped_desc = escape_cypher(description)
                    escaped_chunk_id = escape_cypher(chunk_id) if chunk_id else ""
                    # Cypher label 需要用反引号包裹
                    escaped_type = entity_type.replace("`", "``")

                    if escaped_name:
                        cypher = f"""MERGE (n:`{escaped_type}` {{name: '{escaped_name}'}}) SET n.description = '{escaped_desc}'"""
                        if escaped_chunk_id:
                            cypher += f", n.chunk_id = '{escaped_chunk_id}'"
                        cypher += " RETURN n"
                        with driver.session(database="neo4j") as session:
                            session.run(cypher)

                self._entity_info[name.lower()] = {
                    "name": name, "type": entity_type, "description": description, "chunk_id": chunk_id
                }
            return name
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加实体失败: {e}")
            return None

    def add_relation(
        self,
        head: str,
        tail: str,
        relation: str,
        relation_description: str = "",
        weight: float = 1.0,
        chunk_id: str = ""
    ) -> Optional[str]:
        """Add a relation to the graph.

        Args:
            head: Source entity name.
            tail: Target entity name.
            relation: Edge label (closed-set predicate or cross-modal free-text).
            relation_description: Free-text human-readable description (stored as property).
            weight: Confidence score.
            chunk_id: Source chunk identifier.
        """
        try:
            driver = self._driver
            if driver:
                escaped_head = head.replace("\\", "\\\\").replace("'", "\\'")
                escaped_tail = tail.replace("\\", "\\\\").replace("'", "\\'")
                escaped_rel = relation.replace("`", "``").replace("'", "\\'")
                escaped_desc = relation_description.replace("\\", "\\\\").replace("'", "\\'")
                escaped_chunk_id = chunk_id.replace("\\", "\\\\").replace("'", "\\'") if chunk_id else ""
                set_parts = []
                if escaped_desc:
                    set_parts.append(f"r.description = '{escaped_desc}'")
                if escaped_chunk_id:
                    set_parts.append(f"r.chunk_id = '{escaped_chunk_id}'")
                set_parts.append(f"r.weight = {float(weight)}")
                set_clause = " SET " + ", ".join(set_parts) if set_parts else ""
                with driver.session(database="neo4j") as session:
                    session.run(
                        f"MERGE (a {{name: '{escaped_head}'}}) "
                        f"MERGE (b {{name: '{escaped_tail}'}}) "
                        f"MERGE (a)-[r:`{escaped_rel}`]->(b)"
                        f"{set_clause}"
                    )

            if head.lower() not in self._entity_info:
                self._entity_info[head.lower()] = {"name": head, "type": "UNKNOWN", "description": "", "chunk_id": chunk_id}
            if tail.lower() not in self._entity_info:
                self._entity_info[tail.lower()] = {"name": tail, "type": "UNKNOWN", "description": "", "chunk_id": chunk_id}

            self._relation_count += 1
            return f"{head}##{relation}##{tail}"
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加关系失败: {e}")
            logger.warning(traceback.format_exc())
            return None

    def add_image_entity(
        self,
        figure_id: str,
        image_path: str,
        description: str = "",
        figure_type: str = "unknown",
        chunk_id: str = ""
    ) -> str:
        """添加图片实体（幂等：已存在则跳过）"""
        try:
            if figure_id.lower() in self._entity_info:
                return figure_id
            driver = self._driver
            if driver:
                esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")  # noqa: E731
                fig_label = f"Figure_{figure_type}".replace("`", "``")
                set_parts = [
                    f"n.description = '{esc(description)}'",
                    f"n.image_path = '{esc(image_path)}'",
                    f"n.figure_type = '{esc(figure_type)}'",
                ]
                if chunk_id:
                    set_parts.append(f"n.chunk_id = '{esc(chunk_id)}'")
                with driver.session(database="neo4j") as session:
                    session.run(
                        f"MERGE (n:`{fig_label}` {{name: '{esc(figure_id)}'}}) "
                        f"SET {', '.join(set_parts)}"
                    )
            self._entity_info[figure_id.lower()] = {
                "name": figure_id,
                "type": f"Figure:{figure_type}",
                "description": description,
                "image_path": image_path,
                "chunk_id": chunk_id,
            }
            return figure_id
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加图片实体失败: {e}")
            logger.warning(traceback.format_exc())
            return figure_id

    def add_media_link(
        self,
        chunk_id: str,
        media_path: str,
        media_type: str = "image",
        caption: str = "",
    ):
        """Deterministic Chunk→Media edge from metadata. Survives VLM failure."""
        key = f"__media__{chunk_id.lower()}##{media_path.lower()}"
        if key in self._entity_info:
            return
        driver = self._driver
        try:
            if driver:
                esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
                with driver.session(database="neo4j") as session:
                    session.run(
                        f"MERGE (c:Chunk {{id: '{esc(chunk_id)}'}}) "
                        f"MERGE (m:Media {{path: '{esc(media_path)}'}}) "
                        f"SET m.type = '{esc(media_type)}', m.caption = '{esc(caption)}' "
                        f"MERGE (c)-[r:HAS_MEDIA]->(m)"
                    )
            self._entity_info[key] = {"name": key, "type": "MediaLink", "description": ""}
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加媒体链接失败: {e}")

    def add_table_entity(
        self,
        table_id: str,
        description: str = "",
        chunk_id: str = ""
    ) -> str:
        """添加表格实体（幂等：已存在则跳过）"""
        try:
            if table_id.lower() in self._entity_info:
                return table_id
            driver = self._driver
            if driver:
                esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")  # noqa: E731
                set_parts = [f"n.description = '{esc(description)}'"]
                if chunk_id:
                    set_parts.append(f"n.chunk_id = '{esc(chunk_id)}'")
                with driver.session(database="neo4j") as session:
                    session.run(
                        f"MERGE (n:Table {{name: '{esc(table_id)}'}}) "
                        f"SET {', '.join(set_parts)}"
                    )
            self._entity_info[table_id.lower()] = {
                "name": table_id,
                "type": "Table",
                "description": description,
                "chunk_id": chunk_id,
            }
            return table_id
        except Exception as e:
            logger.warning(f"[GraphRAG] 添加表格实体失败: {e}")
            logger.warning(traceback.format_exc())
            return table_id

    def get_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        # 使用缓存的实体信息计算，避免 get_rel_map([]) 的问题
        entity_types: Dict[str, int] = {}
        for info in self._entity_info.values():
            t = info.get("type", "UNKNOWN")
            entity_types[t] = entity_types.get(t, 0) + 1

        return {
            "entity_count": len(self._entity_info),
            "relation_count": self._relation_count,
            "index_size": len(self._entity_info),
            "entity_types": entity_types
        }

    def __len__(self) -> int:
        """返回实体数量"""
        return len(self._entity_info)

    def __contains__(self, item: str) -> bool:
        """检查实体是否存在（大小写不敏感）"""
        return item.lower() in self._entity_info

    def clear(self, delete_storage: bool = False):
        """清空图谱（仅清空缓存）"""
        self._entity_info.clear()
        self._relation_count = 0
        logger.info("[GraphRAG] 图谱缓存已清空")


class GraphRAGEngine:
    """
    Graph RAG 引擎 - 扩展现有 HybridRAGEngine

    支持三种检索模式：
    - vector: 纯向量检索（委托给 base_engine）
    - graph: 纯图谱检索
    - hybrid: 向量 + 图谱混合检索
    """

    RETRIEVAL_MODES = ["vector", "graph", "graph_local", "graph_global", "hybrid"]

    def __init__(
        self,
        config: GraphRAGConfig,
        base_engine: Any,
        context: Any = None
    ):
        self.config = config
        self.base_engine = base_engine
        self.context = context
        self._graph_store: Optional[Any] = None
        self._index: Optional[Any] = None
        self._query_engine: Optional[Any] = None
        self._adapter: Optional[Any] = None
        self._initialized = False
        self._health_status: str = "not_initialized"

    async def _get_llm(self):
        """从 AstrBot Provider 创建 LlamaIndex 兼容的 LLM，优先使用本地VLM"""
        try:
            from llama_index.llms.openai import OpenAI

            # 优先使用本地VLM（与 HybridRAGEngine 保持一致）
            try:
                try:
                    from idea.llama_cpp_vlm_provider import get_llama_cpp_vlm_provider
                except ImportError:
                    from ..idea.llama_cpp_vlm_provider import get_llama_cpp_vlm_provider
                vlm_provider = get_llama_cpp_vlm_provider()
                if vlm_provider and not getattr(vlm_provider, '_initialized', False):
                    logger.info("[GraphRAG] 本地VLM未初始化，尝试初始化...")
                    await vlm_provider.initialize()
                if vlm_provider and getattr(vlm_provider, '_initialized', False):
                    logger.info("[GraphRAG] 使用本地VLM Provider")
                    model = getattr(vlm_provider, 'model_name', '') or 'local-vlm'
                    api_base = getattr(vlm_provider, 'api_base', '') or 'http://localhost:8080/v1'
                    api_key = getattr(vlm_provider, 'api_key', 'dummy') or 'dummy'
                    return OpenAI(model=model, api_key=api_key, base_url=api_base)
            except Exception as e:
                logger.debug(f"[GraphRAG] 本地VLM不可用: {e}")

            # Fall back to cloud provider
            if self.context is None:
                return None
            provider = self.context.get_using_provider()
            if provider is None:
                return None
            model = getattr(provider, 'model_name', '') or getattr(provider, 'provider_config', {}).get('model', '')
            if not model:
                return None
            api_key = getattr(provider, 'chosen_api_key', None)
            if not api_key:
                try:
                    api_key = provider.get_current_key()
                except Exception:
                    pass
            if not api_key:
                return None
            api_base = getattr(provider, 'provider_config', {}).get('api_base', '')
            kwargs = {"model": model, "api_key": api_key}
            if api_base:
                kwargs["api_base"] = api_base
            logger.info(f"[GraphRAG] 创建 LlamaIndex LLM: model={model}")
            return OpenAI(**kwargs)
        except Exception as e:
            logger.warning(f"[GraphRAG] 创建 LlamaIndex LLM 失败: {e}")
            return None

    async def _ensure_neo4j_running(self) -> None:
        """Verify Neo4j is reachable at the configured URI; if not, start it via CLI and poll.

        Raises:
            RuntimeError: if Neo4j cannot be started or does not become reachable.
        """
        from neo4j import GraphDatabase

        uri = self.config.neo4j_uri
        user = self.config.neo4j_user
        password = self.config.neo4j_password

        # Fast path: already reachable
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=5)
            driver.verify_connectivity()
            driver.close()
            logger.info(f"[GraphRAG] Neo4j already reachable at {uri}")
            return
        except Exception:
            pass  # fall through to start attempt

        # Slow path: start via CLI
        neo4j_bin = shutil.which("neo4j")
        if not neo4j_bin:
            raise RuntimeError(
                "[GraphRAG] neo4j command not found in PATH — cannot auto-start. "
                "Please ensure Neo4j is installed and 'neo4j' is on your PATH, "
                "or start Neo4j manually before enabling Graph RAG."
            )

        logger.warning(f"[GraphRAG] Neo4j not reachable at {uri}, attempting neo4j start...")
        result = subprocess.run([neo4j_bin, "start"], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"[GraphRAG] neo4j start failed: {result.stderr}")

        # Poll until reachable (max 30s)
        for attempt in range(15):
            time.sleep(2)
            try:
                driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=5)
                driver.verify_connectivity()
                driver.close()
                logger.info(f"[GraphRAG] Neo4j started and verified reachable at {uri}")
                return
            except Exception:
                continue

        raise RuntimeError(
            f"[GraphRAG] Neo4j did not become reachable after start at {uri}. "
            "Check 'neo4j-admin dump' permissions and database integrity."
        )

    async def initialize(self):
        """初始化图谱引擎"""
        if self._initialized:
            return

        if not self.config.enable_graph_rag:
            logger.info("Graph RAG 功能未启用")
            return

        try:
            # Ensure Neo4j is running (auto-start or verify connectivity)
            await self._ensure_neo4j_running()

            # Neo4j 存储（唯一支持的存储后端）
            from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
            self._graph_store = Neo4jPropertyGraphStore(
                username=self.config.neo4j_user,
                password=self.config.neo4j_password,
                url=self.config.neo4j_uri,
                database="neo4j",
                refresh_schema=True
            )
            logger.info(f"✅ Neo4j 图谱存储已连接: {self.config.neo4j_uri}")
            self._adapter = SimplePropertyGraphStoreAdapter(self._graph_store)

            await self._init_index()

            # 仅在 index 真正创建成功时才标记初始化完成
            if self._index is not None:
                logger.info(f"✅ Graph RAG 引擎已初始化 (存储类型: {self.config.storage_type})")
                logger.info(f"   - 最大三元组/Chunk: {self.config.max_triplets_per_chunk}")
                logger.info(f"   - 图谱检索TopK: {self.config.graph_retrieval_top_k}")
                logger.info(f"   - 图谱RRF权重: {self.config.graph_rrf_weight}")
                self._initialized = True
                self._health_status = "healthy"
            else:
                logger.warning("[GraphRAG] 图谱索引未创建成功，图谱检索将不可用")
                self._health_status = "index_unavailable"
                # 不设置 _initialized，允许下次调用重试（LLM/Neo4j 可能稍后可用）

        except ImportError as e:
            logger.error(f"❌ 缺少依赖: {e}")
            logger.info("请安装 llama-index: pip install llama-index")
            self._health_status = "missing_dependency"
            # 不设置 _initialized，允许重试
        except Exception as e:
            logger.error(f"❌ Graph RAG 引擎初始化失败: {e}")
            logger.error(traceback.format_exc())
            self._health_status = f"failed: {e}"
            # 不设置 _initialized，允许下次调用重试

    async def _init_index(self):
        """初始化 LlamaIndex 索引和 query engine"""
        try:
            from llama_index.core import PropertyGraphIndex

            llm = await self._get_llm()
            if llm is None:
                logger.warning("[GraphRAG] 未找到 LLM，图谱检索将不可用")
                return

            if self._graph_store is None:
                logger.warning("[GraphRAG] 图谱存储未初始化，跳过索引创建")
                return

            # embed_kg_nodes=False 禁用向量检索，只使用 LLMSynonymRetriever + TextToCypherRetriever
            self._index = PropertyGraphIndex.from_existing(
                property_graph_store=self._graph_store,
                llm=llm,
                embed_model=None,
                embed_kg_nodes=False,
            )

            if self._index is None:
                logger.warning("[GraphRAG] 索引创建返回 None，检索功能可能受限")
                return

            if LLMSynonymRetriever is None:
                logger.warning("[GraphRAG] LLMSynonymRetriever 不可用，图谱检索功能受限")
                return

            top_k = self.config.graph_retrieval_top_k
            sub_retrievers: list[Any] = [
                _CaseInsensitiveSynonymRetriever(
                    graph_store=self._graph_store,
                    include_text=True,
                    llm=llm,
                    limit=top_k,
                ),
            ]

            # TextToCypherRetriever: translates natural language to Cypher for
            # complex multi-hop queries that synonym matching cannot handle.
            try:
                from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import TextToCypherRetriever
                sub_retrievers.append(TextToCypherRetriever(
                    graph_store=self._graph_store,
                    include_text=True,
                    llm=llm,
                ))
            except ImportError:
                logger.warning("[GraphRAG] TextToCypherRetriever 不可用，仅使用同义检索")
            except Exception as e:
                logger.warning(f"[GraphRAG] TextToCypherRetriever 创建失败: {e}")

            retriever = self._index.as_retriever(
                sub_retrievers=sub_retrievers,
                include_text=True,
            )
            # RetrieverQueryEngine needs a valid OpenAI-compatible LLM model name for
            # response synthesis. 'local-vlm' is not in OpenAI's model list, so skip it.
            # We only use get_retriever() which returns the retriever directly.
            self._query_engine = None

            retriever_names = [type(r).__name__ for r in sub_retrievers]
            logger.info(f"✅ Graph RAG 检索器已创建: {retriever_names}, limit={top_k}")

        except ImportError as e:
            logger.warning(f"[GraphRAG] LlamaIndex 索引组件不可用: {e}")
        except Exception as e:
            logger.warning(f"[GraphRAG] 索引初始化失败（不影响图谱构建）: {e}")
            logger.warning(traceback.format_exc())

    async def get_retriever(self):
        """返回 PGRetriever，供 HybridRetriever 作为第四通道使用"""
        if not self._initialized:
            await self.initialize()
        if self._health_status != "healthy":
            logger.warning(f"[GraphRAG] 引擎未就绪，状态: {self._health_status}")
            return None
        if self._graph_store is None or self._index is None:
            return None
        if LLMSynonymRetriever is None:
            return None

        llm = await self._get_llm()
        if llm is None:
            return None

        top_k = self.config.graph_retrieval_top_k
        sub_retrievers: list[Any] = [
            _CaseInsensitiveSynonymRetriever(
                graph_store=self._graph_store,
                include_text=True,
                llm=llm,
                limit=top_k,
            ),
        ]
        try:
            from llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher import TextToCypherRetriever
            sub_retrievers.append(TextToCypherRetriever(
                graph_store=self._graph_store,
                include_text=True,
                llm=llm,
            ))
        except Exception:
            pass

        return self._index.as_retriever(sub_retrievers=sub_retrievers, include_text=True)

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """搜索接口，支持三种模式"""
        if not self.config.enable_graph_rag:
            return {"type": "error", "message": "Graph RAG 功能未启用"}

        if mode not in self.RETRIEVAL_MODES:
            return {"type": "error", "message": f"不支持的检索模式: {mode}"}

        try:
            if mode == "vector":
                return await self._vector_search(query, top_k)
            elif mode in ("graph", "graph_local", "graph_global"):
                return await self._graph_search(query, top_k)
            else:
                return await self._hybrid_search(query, top_k)
        except Exception as e:
            logger.error(f"Graph RAG 搜索失败: {e}")
            return {"type": "error", "message": f"Graph RAG 搜索失败: {str(e)}"}

    async def _vector_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """纯向量检索"""
        if self.base_engine is None:
            return {"type": "error", "message": "基础引擎未初始化"}
        result = await self.base_engine.search(query, mode="retrieve")
        return result

    async def _graph_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """图谱检索 — 委托 LlamaIndex 官方 query engine"""
        if self._query_engine is None:
            return {"type": "error", "message": "图谱查询引擎未初始化（缺少LLM）"}

        try:
            response = await asyncio.to_thread(self._query_engine.query, query)
            sources = []
            for n in getattr(response, "source_nodes", []):
                sources.append({
                    "text": str(getattr(n, "text", "")),
                    "metadata": getattr(n, "metadata", {}),
                    "score": getattr(n, "score", None) or 0,
                })
            return {
                "type": "graph",
                "answer": str(response),
                "sources": sources,
                "entities": [],
                "triplets": [],
            }
        except Exception as e:
            logger.error(f"[GraphRAG] 图谱查询失败: {e}")
            return {"type": "error", "message": f"图谱查询失败: {str(e)}"}

    async def _hybrid_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """混合检索 — 图谱答案 + 向量库补充来源"""
        graph_result = await self._graph_search(query, top_k)

        if graph_result.get("type") == "error":
            graph_result["type"] = "hybrid"
            return graph_result

        # Supplement with vector DB sources so references have real text
        vector_sources = []
        if self.base_engine is not None:
            try:
                vector_result = await self.base_engine.search(query, mode="retrieve")
                nodes = getattr(vector_result, "nodes", [])
                scores = getattr(vector_result, "scores", [1.0] * len(nodes))
                for node, score in zip(nodes[:top_k], scores[:top_k]):
                    vector_sources.append({
                        "text": getattr(node, "text", ""),
                        "metadata": getattr(node, "metadata", {}),
                        "score": score,
                    })
            except Exception as e:
                logger.warning(f"[GraphRAG] 向量补充检索失败: {e}")

        return {
            "type": "hybrid",
            "answer": graph_result.get("answer", ""),
            "sources": graph_result.get("sources", []) + vector_sources,
            "entities": graph_result.get("entities", []),
            "triplets": graph_result.get("triplets", []),
        }

    async def build_graph_from_nodes(self, nodes: List[Any]) -> Dict[str, Any]:
        """从文档节点构建知识图谱"""
        if not self.config.enable_graph_rag:
            return {"status": "skipped", "message": "Graph RAG 功能未启用"}

        try:
            try:
                from .graph_builder import MultimodalGraphBuilder  # type: ignore[import]
            except ImportError:
                from .graph_builder import MultimodalGraphBuilder

            if self._adapter is None:
                await self.initialize()

            builder = MultimodalGraphBuilder(
                config=self.config,
                context=self.context
            )

            stats = await builder.build_from_nodes(nodes, self._adapter)

            # 处理完成后清理内存
            gc.collect()

            logger.info(f"✅ 知识图谱构建完成: {stats}")
            return {"status": "success", **stats}

        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    async def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if not self.config.enable_graph_rag:
            return {"enabled": False}

        if self._adapter is None:
            logger.warning("[GraphRAG] get_graph_stats: _adapter 为 None，可能引擎初始化不完整")
            return {"enabled": True, "storage_type": self.config.storage_type, "entity_count": 0, "relation_count": 0}

        if self._adapter is not None:
            driver = self._adapter._driver
            if driver is not None:
                try:
                    def _query_neo4j():
                        with driver.session(database="neo4j") as session:
                            node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
                            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
                        return node_count, rel_count
                    node_count, rel_count = await asyncio.to_thread(_query_neo4j)
                    logger.info(f"[GraphRAG] Neo4j 统计查询成功: {node_count} 节点, {rel_count} 关系")
                    return {
                        "enabled": True,
                        "storage_type": self.config.storage_type,
                        "entity_count": node_count,
                        "relation_count": rel_count,
                        "index_size": node_count,
                    }
                except Exception as e:
                    logger.warning(f"[GraphRAG] Neo4j 统计查询失败，回退到缓存: {e}")
            else:
                logger.warning("[GraphRAG] get_graph_stats: _graph_store 存在但无 client/_driver 属性")

        return {
            "enabled": True,
            "storage_type": self.config.storage_type,
            **(self._adapter.get_stats() if self._adapter else {}),
        }

    async def clear_graph(self) -> Dict[str, Any]:
        """清空图谱"""
        if not self.config.enable_graph_rag:
            return {"status": "skipped", "message": "Graph RAG 功能未启用"}

        if self._adapter is not None:
            # Neo4j: 执行 Cypher 删除所有节点和关系
            driver = self._adapter._driver
            if driver is not None:
                try:
                    def _clear_neo4j():
                        with driver.session(database="neo4j") as session:
                            session.run("MATCH (n) DETACH DELETE n")
                    await asyncio.to_thread(_clear_neo4j)
                    logger.info("[GraphRAG] Neo4j 数据库已清空")
                except Exception as e:
                    logger.warning(f"[GraphRAG] 清空 Neo4j 数据库失败: {e}")
            else:
                logger.warning("[GraphRAG] clear_graph: 无可用 driver，跳过 Neo4j 清空")

            self._adapter.clear()

        self._graph_store = None
        self._adapter = None
        self._index = None
        self._query_engine = None
        self._initialized = False

        return {"status": "success", "message": "图谱已清空"}


def create_graph_rag_engine(
    config: GraphRAGConfig,
    base_engine: Any,
    context: Any = None
) -> GraphRAGEngine:
    """创建 Graph RAG 引擎实例"""
    return GraphRAGEngine(config, base_engine, context)


async def build_graph_from_documents(
    documents: List[str],
    graph_store: Any,
    config: GraphRAGConfig,
    context: Any = None
) -> Dict[str, int]:
    """便捷函数：从文档列表构建图谱"""
    from .graph_builder import MultimodalGraphBuilder
    class SimpleNode:
        def __init__(self, text: str, metadata: Dict[str, Any]):
            self.text = text
            self.metadata = metadata

    nodes = [SimpleNode(doc, {"chunk_id": f"doc_{i}"}) for i, doc in enumerate(documents)]

    # 确保 graph_store 实现了所需接口
    if not hasattr(graph_store, 'add_entity') or not hasattr(graph_store, 'add_relation'):
        from llama_index.core.graph_stores import SimplePropertyGraphStore
        adapter = SimplePropertyGraphStoreAdapter(SimplePropertyGraphStore())
    else:
        adapter = graph_store

    builder = MultimodalGraphBuilder(config=config, context=context)
    return await builder.build_from_nodes(nodes, adapter)
