"""
Graph RAG Engine - 图谱增强检索引擎

存储后端：
- neo4j: Neo4jPropertyGraphStore（唯一支持的存储后端）

图谱检索作为 RRF 第四通道融入 HybridRetriever，不再独立运行。
"""

from __future__ import annotations

import asyncio
import gc
import re
import shutil
import subprocess
import time
import traceback
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

from neo4j import GraphDatabase

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
    auto_build: bool = False  # 是否自动构建图谱
    auto_build_threshold: int = 10  # 自动构建阈值
    # 多模态配置
    multimodal_enabled: bool = True  # 是否启用多模态图谱抽取
    max_images_per_chunk: int = 1  # 每个chunk最多处理图片数
    extract_image_entities: bool = True  # 是否提取图片为实体
    # LLM 配置
    llm_n_ctx: int = 16384  # 模型上下文窗口大小
    llm_max_tokens: int = 8192  # 最大生成 token 数

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
            auto_build=getattr(config, 'graph_auto_build', False),
            auto_build_threshold=getattr(config, 'graph_auto_build_threshold', 10),
            multimodal_enabled=getattr(config, 'graph_multimodal_enabled', True),
            max_images_per_chunk=getattr(config, 'graph_max_images_per_chunk', 1),
            extract_image_entities=getattr(config, 'graph_extract_image_entities', True),
            llm_n_ctx=getattr(config, 'graph_llm_n_ctx', 16384),
            llm_max_tokens=getattr(config, 'graph_llm_max_tokens', 8192),
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


_CYPHER_PROMPT = """\
You are a Neo4j Cypher query generator for an academic paper knowledge graph.

## Schema
Node labels: Method, Model, Task, Dataset, Metric, Component, Limitation, Application, Baseline
Node properties: name (string), description (string)

Relationship semantics (Source → Target):
- ADDRESSES: Method/Model → Task it solves
- PROPOSES: Method → new Component/technique it introduces
- USES_COMPONENT: Method → Component it relies on
- EVALUATED_ON: Method → Dataset used for evaluation
- ACHIEVES: Method → Metric result
- COMPARES_WITH: Method → Baseline compared to
- OUTPERFORMS: Method → Baseline it outperforms
- LIMITED_BY: Method → Limitation
- APPLIES_TO: Method → Application domain
- EXTENDS: Method → prior Method/Model it builds upon
- TRAINS_ON: Model → Dataset used for training
- IMPLEMENTS: Model → Code repository
- REQUIRES: Method → hardware/resource requirement

## Rules
- **CRITICAL**: ALWAYS bind relationships to r with `-[r:TYPE]->`, NEVER use `-[:TYPE]->` — \
the RETURN clause needs `type(r)` so `r` must be bound in every MATCH
- Return exactly: coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type
- Use CONTAINS for fuzzy name matching (names may include suffixes like "dataset")
- Do NOT filter on description — it is often empty
- Use OR between alternatives, not AND, to maximize recall
- For indirect connections, use MATCH ... WHERE ... WITH ... MATCH ... (2-hop query)
- LIMIT 30
- Output ONLY the Cypher query, no explanation, no backticks

## Examples
Q: What methods are evaluated on Mip-NeRF360?
MATCH (h)-[r:EVALUATED_ON]->(t) WHERE t.name CONTAINS 'Mip-NeRF360' \
RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: What are the limitations of Gaussian Splatting?
MATCH (h)-[r:LIMITED_BY]->(t:Limitation) WHERE h.name CONTAINS 'Gaussian' OR h.name CONTAINS '3DGS' \
RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: How does InstantSplat achieve sparse-view reconstruction?
MATCH (h)-[r:ADDRESSES|USES_COMPONENT|ACHIEVES]->(t) WHERE h.name CONTAINS 'InstantSplat' \
RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: Compare 3DGS with NeRF methods
MATCH (h)-[r:COMPARES_WITH|OUTPERFORMS]->(t) WHERE h.name CONTAINS '3DGS' OR h.name CONTAINS 'Gaussian' \
OR t.name CONTAINS 'NeRF' \
RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: What components are used by sparse-view reconstruction methods?  (2-hop)
MATCH (h)-[r1:ADDRESSES]->(task) WHERE task.name CONTAINS 'sparse-view' WITH h \
MATCH (h)-[r:USES_COMPONENT]->(t) \
RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: What is the relationship between MASt3R and DUSt3R?
MATCH (h)-[r]->(t) WHERE h.name CONTAINS 'MASt3R' OR h.name CONTAINS 'DUSt3R' \
OR t.name CONTAINS 'MASt3R' OR t.name CONTAINS 'DUSt3R' \
RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: What methods address novel view synthesis?
MATCH (h)-[r:ADDRESSES]->(t:Task) WHERE t.name CONTAINS 'novel view' \
RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: What hardware do methods require?
MATCH (h)-[r:REQUIRES]->(t) WHERE t.name CONTAINS 'GPU' OR t.name CONTAINS 'memory' \
RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, \
type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

## Query
{query}
"""


_TEXT_TO_CYPHER_TEMPLATE = """\
## Cypher Query Generation for Academic Knowledge Graph

Generate a Cypher statement to query the academic paper knowledge graph.

### Schema
{schema}

Node labels and their semantics:
- Method: algorithm, optimization method, training technique
- Model: named architecture (BERT, GPT, Transformer, ResNet)
- Task: research problem (classification, QA, generation)
- Dataset: benchmark (GLUE, ImageNet, COCO)
- Metric: evaluation measure (accuracy, F1, BLEU)
- Component: layer, module, sub-architecture
- Limitation: weakness, constraint, failure mode
- Application: real-world use case, domain
- Baseline: previous method, compared system

Relationship semantics (h → t):
- ADDRESSES: Method/Model → Task
- PROPOSES: Method → new Component
- USES_COMPONENT: Method → Component
- EVALUATED_ON: Method → Dataset
- ACHIEVES: Method → Metric
- COMPARES_WITH: Method → Baseline
- OUTPERFORMS: Method → Baseline
- LIMITED_BY: Method → Limitation
- APPLIES_TO: Method → Application
- EXTENDS: Method → prior Method
- TRAINS_ON: Model → Dataset
- IMPLEMENTS: Model → repository
- REQUIRES: Method → resource
- ABLATES_ON: Method → Component

### Rules
1. **CRITICAL**: Bind EVERY relationship with a variable — in multi-hop `(a)-[r1:X]->(b)-[r2:Y]->(c)`, both `r1` and `r2` must be bound. NEVER use anonymous `-[:TYPE]->`.
2. **CRITICAL**: CONTAINS is a WHERE-only operator. NEVER use it inside node property patterns `{{name: ...}}`. In node patterns, ONLY use exact equality: `{{name: 'exact value'}}`. Put all CONTAINS filters in a WHERE clause after MATCH.
3. Cypher query MUST start with MATCH, CALL, MERGE, or RETURN — never start with WHERE
4. Node patterns must use exact name matching only. For fuzzy matching, write: `MATCH (n:Label) WHERE n.name CONTAINS 'keyword' OR n.name CONTAINS 'keyword2'`
5. Do NOT filter on description — it is often empty
6. Use OR between alternatives, not AND, to maximize recall
7. LIMIT 30

### Required RETURN Format
Return exactly (replace `h`, `r`, `t` with your bound node/relationship variables):

RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type

### Examples

Q: What methods are evaluated on Mip-NeRF360? (single-hop, WHERE on tail)
MATCH (h)-[r:EVALUATED_ON]->(t) WHERE t.name CONTAINS 'Mip-NeRF360' RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: How does InstantSplat achieve sparse-view reconstruction? (single-hop, WHERE on head)
MATCH (h)-[r:ADDRESSES|USES_COMPONENT]->(t) WHERE h.name CONTAINS 'InstantSplat' RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: Compare 3DGS with NeRF methods (single-hop, WHERE on both sides)
MATCH (h)-[r:COMPARES_WITH|OUTPERFORMS]->(t) WHERE h.name CONTAINS '3DGS' OR t.name CONTAINS 'NeRF' RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, type(r) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

Q: How does MASt3R use Transformer for 3D tasks? (multi-hop with WHERE on tail)
MATCH (h:Model {{name: 'MASt3R'}})-[r1:USES_COMPONENT]->(c:Component {{name: 'Transformer'}})-[r2:USES_COMPONENT|ADDRESSES]->(t:Task) WHERE t.name CONTAINS '3D' RETURN coalesce(h.name,'') AS head, labels(h)[0] AS head_type, type(r2) AS relation, coalesce(t.name,'') AS tail, labels(t)[0] AS tail_type LIMIT 30

### WRONG — NEVER do this:
MATCH ... (t:Task {{name: CONTAINS 'keyword'}})   ← CONTAINS in node pattern — WILL CAUSE SYNTAX ERROR
MATCH ... -[:USES_COMPONENT]->(t)                  ← anonymous relationship — WILL CAUSE Cypher error

### Question
{question}

Output ONLY the Cypher statement, no explanations, no backticks.
"""


def _parse_cypher_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """将 Cypher 返回记录解析为 entities + triplets。"""
    entity_set: dict[str, str] = {}
    triplets: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for r in records:
        head = str(r.get("head") or "")
        tail = str(r.get("tail") or "")
        relation = str(r.get("relation") or "")
        if not head or not tail:
            continue
        entity_set.setdefault(head, str(r.get("head_type", "")))
        entity_set.setdefault(tail, str(r.get("tail_type", "")))
        key = (head, relation, tail)
        if key not in seen:
            seen.add(key)
            triplets.append({
                "head": head, "relation": relation,
                "tail": tail, "description": "",
            })
    entities = [{"name": n, "type": t} for n, t in entity_set.items()]
    return entities, triplets


_VALID_CYPHER_STARTS: frozenset[str] = frozenset({
    "MATCH", "CALL", "CREATE", "MERGE", "RETURN", "UNWIND",
    "WITH", "OPTIONAL", "EXPLAIN", "PROFILE", "SHOW", "DROP",
    "LOAD", "FOREACH", "USE", "REMOVE", "SET", "DETACH",
})


def _make_cypher_validator(graph_store: Any):
    """Return a ``cypher_validator`` callable that validates LLM-generated Cypher.

    Two-stage check:
    1. Fast: first token must be a legal Cypher clause keyword.
    2. Definitive: run ``EXPLAIN <query>`` through Neo4j's own parser.
       Neo4j's parser is the gold standard — it rejects any syntactically
       invalid Cypher before execution, so this catches 100% of syntax
       errors (clause ordering, expression validity, etc.).

    The returned function matches the ``cypher_validator`` callback
    signature expected by ``TextToCypherRetriever``: ``(str) -> str``.
    """
    def _validate(cypher_query: str) -> str:
        stripped = cypher_query.strip()
        if not stripped:
            raise ValueError("TextToCypherRetriever 返回空 Cypher")

        # Stage 1 — fast first-token check (no network round-trip)
        first_token = stripped.split(maxsplit=1)[0].upper()
        if first_token not in _VALID_CYPHER_STARTS:
            raise ValueError(
                f"TextToCypherRetriever 生成了非法的 Cypher 语句，"
                f"首 keyword={first_token}，完整内容:\n{stripped[:500]}"
            )

        # Stage 2 — definitive Neo4j EXPLAIN parse check
        try:
            graph_store.structured_query(f"EXPLAIN {stripped}")
        except Exception as e:
            raise ValueError(
                f"TextToCypherRetriever 生成的 Cypher 未通过 Neo4j EXPLAIN 校验: "
                f"{e}\n完整内容:\n{stripped[:500]}"
            ) from e

        return cypher_query

    return _validate


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

    async def _get_llm(self, prefer_cloud: bool = False):
        """从 AstrBot Provider 创建 LlamaIndex 兼容的 LLM。

        Args:
            prefer_cloud: 为 True 时跳过本地 VLM，直接使用云端 Provider（适合 Cypher 生成）。
        """
        try:
            from provider.llm_utils import get_llama_index_llm
            llm = await get_llama_index_llm(self.context, prefer_cloud=prefer_cloud)
            return llm
        except Exception as e:
            logger.warning(f"[GraphRAG] 创建 LlamaIndex LLM 失败: {e}")
            return None

    async def _ensure_neo4j_running(self) -> None:
        """Verify Neo4j is reachable at the configured URI; if not, start it via CLI and poll.

        Raises:
            RuntimeError: if Neo4j cannot be started or does not become reachable.
        """
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
        except Exception as e:
            logger.debug(f"[GraphRAG] Fast-path Neo4j connectivity check failed, falling through to start attempt: {e}")

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

        if result.returncode != 0 and "already running" in result.stderr:
            logger.warning(f"[GraphRAG] Neo4j already running (pid detected), verifying actual connectivity...")
            try:
                driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=5)
                driver.verify_connectivity()
                # 额外验证：执行一个简单查询确保数据库真的可用
                with driver.session() as session:
                    result = session.run("RETURN 1")
                    result.consume()
                driver.close()
                logger.info(f"[GraphRAG] Neo4j already running and database verified at {uri}")
                return
            except Exception as e:
                logger.error(f"[GraphRAG] Neo4j reported running but verification failed: {e}")
                return

        if result.returncode != 0:
            # 如果是"already running"错误，前面已处理过连通性，这里不再重复抛异常
            if not ("already running" in result.stderr):
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
                    cypher_validator=_make_cypher_validator(self._graph_store),
                    text_to_cypher_template=_TEXT_TO_CYPHER_TEMPLATE,
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
                cypher_validator=_make_cypher_validator(self._graph_store),
                text_to_cypher_template=_TEXT_TO_CYPHER_TEMPLATE,
            ))
        except Exception as e:
            logger.warning(f"[GraphRAG] TextToCypherRetriever 创建失败，仅使用同义检索: {e}")

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
        """图谱检索 — Text-to-Cypher 子图提取 + 可选 query engine 回答"""
        entities, triplets = await self._fetch_subgraph(query)

        answer = ""
        sources = []
        if self._query_engine is not None:
            try:
                response = await asyncio.to_thread(self._query_engine.query, query)
                answer = str(response)
                for n in getattr(response, "source_nodes", []):
                    sources.append({
                        "text": str(getattr(n, "text", "")),
                        "metadata": getattr(n, "metadata", {}),
                        "score": getattr(n, "score", None) or 0,
                    })
            except Exception as e:
                logger.warning(f"[GraphRAG] query engine 调用失败: {e}")

        return {
            "type": "graph",
            "answer": answer,
            "sources": sources,
            "entities": entities,
            "triplets": triplets,
        }

    async def _fetch_subgraph(self, query: str) -> tuple[list[dict], list[dict]]:
        """从 Neo4j 提取与查询相关的实体和三元组（Text-to-Cypher + 关键词 fallback）。"""
        driver = self._adapter._driver if self._adapter else None
        if driver is None:
            return [], []

        # Text-to-Cypher: LLM 生成 Cypher 查询
        try:
            llm = await self._get_llm(prefer_cloud=True)
            if llm is not None:
                prompt = _CYPHER_PROMPT.format(query=query)
                response = await asyncio.to_thread(llm.complete, prompt)
                cypher = response.text.strip()
                if cypher.startswith("```"):
                    first_nl = cypher.find("\n")
                    cypher = cypher[first_nl + 1:] if first_nl != -1 else cypher[3:]
                    if cypher.endswith("```"):
                        cypher = cypher[:-3]
                    cypher = cypher.strip()

                def _run(tx):
                    result = tx.run(cypher)
                    return [dict(r) for r in result]

                with driver.session() as session:
                    records = session.execute_read(_run)

                if records:
                    entities, triplets = _parse_cypher_records(records)
                    if triplets:
                        logger.info(
                            f"[GraphRAG] Text-to-Cypher: {len(entities)} 实体, "
                            f"{len(triplets)} 三元组"
                        )
                        return entities, triplets
        except Exception as e:
            logger.warning(f"[GraphRAG] Text-to-Cypher 失败，回退关键词: {e}")

        # Fallback: 关键词匹配
        return await self._fetch_subgraph_keywords(query)

    async def _fetch_subgraph_keywords(self, query: str) -> tuple[list[dict], list[dict]]:
        """关键词匹配提取子图（fallback）。"""
        entities: list[dict] = []
        triplets: list[dict] = []

        driver = self._adapter._driver if self._adapter else None
        if driver is None:
            return entities, triplets

        _STOPWORDS = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "has", "have", "had", "can", "could", "will",
            "would", "should", "may", "might", "shall", "must", "need",
            "what", "which", "who", "whom", "how", "when", "where", "why",
            "this", "that", "these", "those", "it", "its", "he", "she", "they",
            "we", "you", "i", "me", "my", "our", "your", "his", "her", "their",
            "of", "in", "on", "at", "to", "for", "with", "from", "by", "about",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "and", "or", "not", "no", "nor", "but", "so", "if", "then",
            "than", "too", "very", "just", "also", "only", "such", "each", "all",
            "any", "both", "few", "more", "most", "other", "some", "up", "out",
            "method", "methods", "approach", "approaches", "model", "models",
            "dataset", "datasets", "result", "results", "compare", "comparison",
            "limitation", "limitations", "achieve", "achieves", "achieved",
            "evaluate", "evaluated", "evaluation", "use", "used", "using",
            "propose", "proposes", "proposed", "based", "performance",
            "novel", "new", "different", "improve", "improves", "improved",
            "work", "works", "paper", "study", "task", "tasks",
            "component", "components", "technique", "techniques",
            "better", "best", "state", "art", "recent", "efficient",
            "effective", "show", "shown", "describe", "described",
            "explain", "tell", "give", "list", "find", "found",
        })

        tokens = re.findall(r'[A-Za-z][A-Za-z0-9_\-]*', query)
        tokens += [w for w in re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_]{2,}', query)
                   if w not in tokens]

        raw = list(dict.fromkeys(
            w for w in tokens
            if w.lower() not in _STOPWORDS and len(w) >= 2
        ))
        filtered = []
        for w in raw:
            if not any(w != o and w in o for o in raw):
                filtered.append(w)
        keywords = filtered[:5]

        if not keywords:
            return entities, triplets

        try:
            def _run(tx, kw_list):
                ent_result = tx.run("""
                    MATCH (n)
                    WHERE any(kw IN $kw_list WHERE coalesce(n.name, n.id, '') CONTAINS kw)
                    RETURN labels(n)[0] AS type, coalesce(n.name, n.id, '') AS name
                    LIMIT 20
                """, kw_list=kw_list)
                ents = [{"name": r["name"], "type": r["type"] or ""} for r in ent_result]

                trip_result = tx.run("""
                    MATCH (h)-[r]->(t)
                    WHERE any(kw IN $kw_list WHERE
                        coalesce(h.name, h.id, '') CONTAINS kw
                        OR coalesce(t.name, t.id, '') CONTAINS kw)
                    RETURN coalesce(h.name, h.id, '') AS head,
                           type(r) AS relation,
                           coalesce(t.name, t.id, '') AS tail,
                           coalesce(r.description, '') AS description
                    LIMIT 30
                """, kw_list=kw_list)
                trips = []
                seen = set()
                for r in trip_result:
                    key = (r["head"], r["relation"], r["tail"])
                    if key not in seen:
                        seen.add(key)
                        trips.append({
                            "head": r["head"],
                            "relation": r["relation"],
                            "tail": r["tail"],
                            "description": r["description"] or "",
                        })
                return ents, trips

            with driver.session() as session:
                entities, triplets = session.execute_read(_run, keywords)

            logger.info(
                f"[GraphRAG] 关键词子图提取: {len(entities)} 实体, "
                f"{len(triplets)} 三元组 (keywords={keywords})"
            )
        except Exception as e:
            logger.warning(f"[GraphRAG] 关键词子图提取失败: {e}")

        return entities, triplets

    async def _hybrid_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """混合检索 — 图谱子图 + 向量库补充来源"""
        graph_result = await self._graph_search(query, top_k)

        # 补充向量检索来源
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
