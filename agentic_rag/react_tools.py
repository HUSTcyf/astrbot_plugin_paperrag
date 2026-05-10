"""
ReAct Tool Executor — 工具注册与执行。
"""

from __future__ import annotations

from astrbot.api import logger
from .engine_utils import get_engine, get_graph_engine

TOOL_DEFINITIONS = {
    "vector_search": "vector_search(query) - 在论文数据库中进行向量检索，返回相关论文片段",
    "graph_search": "graph_search(query) - 搜索知识图谱，获取实体关系和结构化知识",
    "list_documents": "list_documents() - 列出所有已收录的论文及其基本信息（文件名、chunk数）",
    "graph_stats": "graph_stats() - 获取知识图谱的统计信息（实体数、关系数等）",
    "get_paper_info": "get_paper_info(文件名) - 获取指定论文的详细信息（chunk数、摘要、DOI等）",
    "reference_stats": "reference_stats(top_k) - 获取参考文献引用统计，top_k 为返回的高频引用数量（默认10，-1 列出无参考文献的论文）",
    "abstract_stats": "abstract_stats(top_k) - 获取摘要提取统计，top_k 为返回数量（-1 列出无摘要的论文）",
}


async def _tool_vector_search(
    query: str, context, config, top_k: int
) -> tuple[str, list[dict]]:
    """向量检索工具。返回 (observation_text, retrieved_nodes)。"""

    engine = get_engine(context, config)
    if engine is None:
        return "错误：无法初始化 RAG 引擎", []

    try:
        result = await engine.search(query, mode="retrieve", top_k=top_k)
    except Exception as e:
        logger.error(f"[react_tools] vector_search 失败: {e}")
        return f"检索失败: {e}", []

    nodes: list[dict] = []
    if hasattr(result, "nodes"):
        scores = getattr(result, "scores", [])
        for i, node in enumerate(result.nodes):
            score = scores[i] if i < len(scores) else 0.0
            nodes.append({
                "text": getattr(node, "text", str(node)),
                "score": score,
                "metadata": getattr(node, "metadata", {}),
            })
    elif isinstance(result, list):
        nodes = result
    elif isinstance(result, dict):
        raw = result.get("nodes") or result.get("results") or []
        for item in raw:
            if isinstance(item, dict):
                nodes.append(item)
            elif hasattr(item, "text"):
                nodes.append({
                    "text": getattr(item, "text", ""),
                    "score": getattr(item, "score", 1.0),
                    "metadata": getattr(item, "metadata", {}),
                })

    if not nodes:
        return "未找到相关论文片段。", []

    display_nodes = nodes[:5]
    parts = [f"找到 {len(nodes)} 个相关片段："]
    for i, node in enumerate(display_nodes, 1):
        text = node.get("text", "")[:200]
        score = node.get("score", 0.0)
        meta = node.get("metadata", {})
        filename = meta.get("file_name", meta.get("source_path", "unknown"))
        parts.append(f"[{i}] ({filename}, score={score:.3f})\n{text}")

    return "\n\n".join(parts), nodes


async def _tool_graph_search(
    query: str, context, config, top_k: int
) -> tuple[str, list[dict], list[dict], list[dict]]:
    """图谱检索工具。返回 (observation_text, source_nodes, entities, relations)。"""

    try:
        graph_engine = await get_graph_engine(context, config)
    except Exception as e:
        logger.error(f"[react_tools] graph_engine 初始化失败: {e}")
        return f"图谱引擎初始化失败: {e}", [], [], []

    if graph_engine is None:
        return "知识图谱未初始化或不可用。", [], [], []

    try:
        result = await graph_engine.search(query, mode="hybrid", top_k=top_k)
    except Exception as e:
        logger.error(f"[react_tools] graph_search 失败: {e}")
        return f"图谱检索失败: {e}", [], [], []

    if not isinstance(result, dict):
        return "图谱返回格式异常。", [], [], []

    entities = result.get("entities") or []
    triplets = result.get("triplets") or []
    sources = result.get("sources") or []
    answer = result.get("answer", "")

    parts = []
    if answer:
        parts.append(f"图谱回答: {answer}")
    if entities:
        parts.append("实体:")
        for e in entities[:5]:
            name = e.get("name", "")
            etype = e.get("type", "")
            parts.append(f"  - {name} ({etype})")
    if triplets:
        parts.append("关系:")
        for t in triplets[:5]:
            head = t.get("head", "")
            rel = t.get("relation", "")
            tail = t.get("tail", "")
            parts.append(f"  - {head} → [{rel}] → {tail}")
    if sources:
        parts.append(f"相关文本片段: {len(sources)} 条")

    if not parts:
        return "图谱中未找到相关信息。", [], [], []

    source_nodes: list[dict] = []
    for s in sources[:top_k]:
        if isinstance(s, dict):
            source_nodes.append({
                "text": s.get("text", ""),
                "score": s.get("score", 0.0),
                "metadata": s.get("metadata", {}),
                "source": "graph",
            })
        elif hasattr(s, "text"):
            source_nodes.append({
                "text": getattr(s, "text", ""),
                "score": getattr(s, "score", 1.0),
                "metadata": getattr(s, "metadata", {}),
                "source": "graph",
            })

    relations: list[dict] = []
    for t in triplets:
        if isinstance(t, dict):
            relations.append({
                "head": t.get("head", ""),
                "relation": t.get("relation", ""),
                "tail": t.get("tail", ""),
                "description": t.get("description", ""),
            })

    return "\n".join(parts), source_nodes, entities, relations


async def _tool_list_documents(context, config) -> tuple[str, list[dict]]:
    """列出所有已收录论文。"""

    engine = get_engine(context, config)
    if engine is None:
        return "错误：无法初始化 RAG 引擎", []

    try:
        papers = await engine.list_papers()
    except Exception as e:
        logger.error(f"[react_tools] list_documents 失败: {e}")
        return f"获取文档列表失败: {e}", []

    if not papers:
        return "文档库为空，尚未添加任何论文。", []

    parts = [f"已收录 {len(papers)} 篇论文："]
    for i, p in enumerate(papers[:20], 1):
        parts.append(
            f"[{i}] {p['file_name']} "
            f"(chunks: {p.get('chunk_count', 0)}, "
            f"added: {p.get('added_time', 'unknown')})"
        )
    if len(papers) > 20:
        parts.append(f"...还有 {len(papers) - 20} 篇")
    return "\n".join(parts), []


async def _tool_graph_stats(context, config) -> str:
    """获取知识图谱统计信息。"""

    try:
        graph_engine = await get_graph_engine(context, config)
    except Exception as e:
        logger.error(f"[react_tools] graph_stats 初始化失败: {e}")
        return f"图谱引擎初始化失败: {e}"

    if graph_engine is None:
        return "知识图谱未初始化或未启用（请检查 enable_graph_rag 配置）。"

    try:
        stats = await graph_engine.get_graph_stats()
    except Exception as e:
        logger.error(f"[react_tools] graph_stats 查询失败: {e}")
        return f"图谱统计查询失败: {e}"

    if not stats.get("enabled", False):
        return "知识图谱功能未启用。"

    return (
        f"知识图谱统计：\n"
        f"- 存储类型: {stats.get('storage_type', 'unknown')}\n"
        f"- 实体数量: {stats.get('entity_count', 0)}\n"
        f"- 关系数量: {stats.get('relation_count', 0)}"
    )


async def _tool_get_paper_info(filename: str, context, config) -> str:
    """获取指定论文的详细信息。"""

    engine = get_engine(context, config)
    if engine is None:
        return "错误：无法初始化 RAG 引擎"

    try:
        papers = await engine.list_papers()
    except Exception as e:
        logger.error(f"[react_tools] get_paper_info 获取论文列表失败: {e}")
        return f"获取论文列表失败: {e}"

    if not papers:
        return f"文档库为空，未找到论文: {filename}"

    matched = None
    for p in papers:
        if p.get("file_name", "") == filename:
            matched = p
            break
    if matched is None:
        for p in papers:
            if filename.lower() in p.get("file_name", "").lower():
                matched = p
                break

    if matched is None:
        return f"未找到论文: {filename}（可用 list_documents 查看所有论文）"

    parts = [
        f"论文: {matched['file_name']}",
        f"- Chunks: {matched.get('chunk_count', 0)}",
        f"- 添加时间: {matched.get('added_time', 'unknown')}",
    ]
    if matched.get("github_url"):
        parts.append(f"- GitHub: {matched['github_url']}")

    # 尝试获取摘要信息（可选增强，失败不影响主流程）
    try:
        abstract_mgr = await engine._ensure_abstract_manager_initialized()
    except Exception as e:
        logger.warning(f"[react_tools] get_paper_info 摘要管理器初始化跳过: {e}")
        abstract_mgr = None

    if abstract_mgr is not None:
        try:
            paper_id = matched["file_name"].rsplit(".", 1)[0] if "." in matched["file_name"] else matched["file_name"]
            abstracts = await abstract_mgr.get_papers_by_ids([paper_id])
            if paper_id in abstracts:
                ab = abstracts[paper_id]
                if ab.title:
                    parts.insert(1, f"- 标题: {ab.title}")
                if ab.abstract_text:
                    abstract_preview = ab.abstract_text[:300]
                    parts.append(f"- 摘要: {abstract_preview}...")
        except Exception as e:
            logger.warning(f"[react_tools] get_paper_info 摘要查询跳过: {e}")

    return "\n".join(parts)


async def _tool_reference_stats(top_k_str: str, context, config) -> str:
    """获取参考文献引用统计。top_k_str: "10" 或 "-1"（列出零引用论文）。"""

    engine = get_engine(context, config)
    if engine is None:
        return "错误：无法初始化 RAG 引擎"

    try:
        top_k = int(top_k_str.strip()) if top_k_str.strip() else 10
    except ValueError:
        top_k = 10

    try:
        index_manager = engine._ensure_index_manager_initialized()
    except Exception as e:
        logger.error(f"[react_tools] reference_stats 索引管理器初始化失败: {e}")
        return f"索引管理器初始化失败: {e}"

    if top_k == -1:
        try:
            result = await index_manager.get_papers_with_zero_references()
        except Exception as e:
            logger.error(f"[react_tools] reference_stats(-1) 失败: {e}")
            return f"获取零引用论文失败: {e}"

        if "error" in result:
            return f"获取失败: {result['error']}"

        papers = result.get("papers", [])
        total_papers = result.get("total_papers", 0)
        total_zero_ref = result.get("total_zero_ref", 0)

        if total_papers == 0:
            return "未能获取到论文列表，请检查索引是否初始化"

        if not papers:
            return "所有论文都已提取到参考文献"

        parts = [f"无参考文献的论文 ({total_zero_ref}/{total_papers})："]
        for i, p in enumerate(papers[:20], 1):
            parts.append(f"[{i}] {p.get('file_name', 'unknown')} (chunks: {p.get('chunk_count', 0)})")
        if len(papers) > 20:
            parts.append(f"...还有 {len(papers) - 20} 篇")
        return "\n".join(parts)

    try:
        stats = await index_manager.get_all_references(allow_duplicates=True)
    except Exception as e:
        logger.error(f"[react_tools] reference_stats 获取引用统计失败: {e}")
        return f"获取引用统计失败: {e}"

    if "error" in stats:
        return f"获取统计失败: {stats['error']}"

    references = stats.get("references", [])
    total_refs = stats.get("total_refs", 0)
    total_chunks = stats.get("total_chunks", 0)

    if not references:
        return "数据库中暂无参考文献信息"

    parts = [
        f"参考文献统计：涉及 {len(references)} 种论文，引用总条次 {total_refs}，处理文档块 {total_chunks}",
        f"Top {min(top_k, len(references))} 高频引用：",
    ]
    for i, ref in enumerate(references[:top_k], 1):
        title = ref["title"][:80]
        count = ref["count"]
        authors = ref.get("authors", "")
        year = ref.get("year", "")
        author_str = f" ({authors[:40]}, {year})" if authors else ""
        parts.append(f"[{i}] [{count}次] {title}{author_str}")

    return "\n".join(parts)


async def _tool_abstract_stats(top_k_str: str, context, config) -> str:
    """获取摘要提取统计。top_k_str: 任意值 或 "-1"（列出零摘要论文）。"""

    engine = get_engine(context, config)
    if engine is None:
        return "错误：无法初始化 RAG 引擎"

    try:
        top_k = int(top_k_str.strip()) if top_k_str.strip() else 0
    except ValueError:
        top_k = 0

    # 获取论文统计和摘要统计
    try:
        index_manager = engine._ensure_index_manager_initialized()
        doc_stats = getattr(index_manager, '_doc_stats', None)
        if doc_stats is None:
            logger.warning("[react_tools] abstract_stats: _doc_stats 属性不存在，可能 index_manager 接口已变更")
            doc_stats = {}
    except Exception as e:
        logger.error(f"[react_tools] abstract_stats 获取论文统计失败: {e}")
        return f"获取论文统计失败: {e}"

    # 尝试获取摘要缓存
    abstracts: dict = {}
    try:
        abstract_mgr = await engine._ensure_abstract_manager_initialized()
    except Exception as e:
        logger.warning(f"[react_tools] abstract_stats 摘要管理器初始化跳过: {e}")
        abstract_mgr = None

    if abstract_mgr is not None:
        try:
            abstracts = await abstract_mgr.get_all_abstracts()
        except Exception as e:
            logger.warning(f"[react_tools] abstract_stats 摘要查询跳过: {e}")

    pdf_papers = {k: v for k, v in doc_stats.items() if isinstance(k, str) and k.lower().endswith(".pdf")}
    total_papers = len(pdf_papers)

    if total_papers == 0:
        return "未找到任何已索引的 PDF 论文"

    zero_abstract: list[dict] = []
    for file_name, stats in pdf_papers.items():
        paper_id = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
        if paper_id not in abstracts or not getattr(abstracts.get(paper_id), 'abstract_text', '').strip():
            zero_abstract.append({
                "file_name": file_name,
                "chunk_count": stats.get("chunk_count", 0) if isinstance(stats, dict) else 0,
            })

    total_with = total_papers - len(zero_abstract)

    if top_k == -1:
        if not zero_abstract:
            return "所有 PDF 论文都已成功提取摘要"
        parts = [f"无摘要的论文 ({len(zero_abstract)}/{total_papers})："]
        for i, p in enumerate(zero_abstract[:20], 1):
            parts.append(f"[{i}] {p['file_name']} (chunks: {p['chunk_count']})")
        if len(zero_abstract) > 20:
            parts.append(f"...还有 {len(zero_abstract) - 20} 篇")
        return "\n".join(parts)

    return (
        f"摘要提取统计：\n"
        f"- PDF论文总数: {total_papers}\n"
        f"- 已提取摘要: {total_with}\n"
        f"- 未提取摘要: {len(zero_abstract)}\n"
        f"（使用 abstract_stats(-1) 列出无摘要的论文）"
    )


async def react_tool_executor_node(state: dict) -> dict:
    """执行 pending 的工具调用，追加 observation 到 scratchpad。"""
    action = state.get("_pending_action")
    if not action:
        return {"steps": ["tool_executor: NO_ACTION"]}

    tool_name = action.get("tool", "")
    tool_args = action.get("args", "")
    context = state.get("_context")
    config = state.get("_config")
    top_k = state.get("top_k", 5)

    scratchpad = state.get("scratchpad", "")

    if tool_name == "vector_search":
        observation, nodes = await _tool_vector_search(tool_args, context, config, top_k)
        scratchpad += f"\nOBSERVATION:\n{observation}\n"

        return {
            "scratchpad": scratchpad,
            "retrieved_nodes": nodes,
            "_pending_action": None,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": [f"tool_executor: vector_search OK ({len(nodes)} nodes)"],
        }

    if tool_name == "graph_search":
        observation, source_nodes, entities, relations = await _tool_graph_search(
            tool_args, context, config, top_k
        )
        scratchpad += f"\nOBSERVATION:\n{observation}\n"

        return {
            "scratchpad": scratchpad,
            "retrieved_nodes": source_nodes,
            "graph_entities": entities,
            "graph_relations": relations,
            "_pending_action": None,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": [f"tool_executor: graph_search OK (ents={len(entities)}, rels={len(relations)})"],
        }

    if tool_name == "list_documents":
        observation, _ = await _tool_list_documents(context, config)
        scratchpad += f"\nOBSERVATION:\n{observation}\n"
        return {
            "scratchpad": scratchpad,
            "_pending_action": None,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": [f"tool_executor: list_documents OK"],
        }

    if tool_name == "graph_stats":
        observation = await _tool_graph_stats(context, config)
        scratchpad += f"\nOBSERVATION:\n{observation}\n"
        return {
            "scratchpad": scratchpad,
            "_pending_action": None,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": [f"tool_executor: graph_stats OK"],
        }

    if tool_name == "get_paper_info":
        observation = await _tool_get_paper_info(tool_args, context, config)
        scratchpad += f"\nOBSERVATION:\n{observation}\n"
        return {
            "scratchpad": scratchpad,
            "_pending_action": None,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": [f"tool_executor: get_paper_info OK"],
        }

    if tool_name == "reference_stats":
        observation = await _tool_reference_stats(tool_args, context, config)
        scratchpad += f"\nOBSERVATION:\n{observation}\n"
        return {
            "scratchpad": scratchpad,
            "_pending_action": None,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": [f"tool_executor: reference_stats OK"],
        }

    if tool_name == "abstract_stats":
        observation = await _tool_abstract_stats(tool_args, context, config)
        scratchpad += f"\nOBSERVATION:\n{observation}\n"
        return {
            "scratchpad": scratchpad,
            "_pending_action": None,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "steps": [f"tool_executor: abstract_stats OK"],
        }

    scratchpad += f"\nOBSERVATION:\n错误：未知工具 '{tool_name}'。可用工具: vector_search, graph_search, list_documents, graph_stats, get_paper_info, reference_stats, abstract_stats\n"
    return {
        "scratchpad": scratchpad,
        "_pending_action": None,
        "steps": [f"tool_executor: UNKNOWN_TOOL ({tool_name})"],
    }
