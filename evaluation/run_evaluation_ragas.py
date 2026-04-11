# -*- coding: utf-8 -*-
"""
Ragas 自动化评估主入口
从 Milvus 数据库读取 chunks → 生成测试集 → RAG 评估 → 报告生成

流程:
  1. 从 Milvus 提取全量 chunk 文本
  2. 按论文分组，构建 llama-index Document
  3. 调用 Ragas TestsetGenerator 生成问答对
  4. 使用 HybridRAGEngine 执行 RAG 查询
  5. 调用 Ragas Evaluator 计算 6 大指标
  6. 生成 HTML + Markdown 报告

用法:
  # 完整流程（提取文本 -> 生成测试集 -> 评估 -> 报告）
  python -m evaluation.run_evaluation_ragas --step all

  # 从 Milvus 提取全量文本（调试用）
  python -m evaluation.run_evaluation_ragas --step extract

  # 仅生成测试集（需已有 milvus_chunks.json）
  python -m evaluation.run_evaluation_ragas --step generate

  # 完整流程使用已有 chunks 文件（避免重复从数据库读取）
  python -m evaluation.run_evaluation_ragas --step all --use-existing-chunks

环境变量:
  EVAL_LLM_API_KEY 评估用 LLM API Key
"""

import asyncio
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 确保 evaluation 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrbot.api import logger


# 多模态文档生成时，发送给 LLM 的文档数量倍数（test_size * MULTIMODAL_DOC_MULTIPLIER）
MULTIMODAL_DOC_MULTIPLIER = 2


# ============================================================================
# 步骤 1: 从 Milvus 提取全量文本
# ============================================================================

def create_index_manager() -> Any:
    """创建 HybridIndexManager 实例（复用现有配置）"""
    plugin_dir = Path(__file__).parent.parent
    config_path = plugin_dir / "data" / "cmd_config.json"

    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    rag_config = config.get("rag_config", {})
    milvus_config = rag_config.get("milvus", {})

    from hybrid_index import HybridIndexManager

    return HybridIndexManager(
        collection_name=rag_config.get("collection_name", "paper_embeddings"),
        embed_dim=rag_config.get("embed_dim", 1024),
        milvus_uri=str(plugin_dir / "data" / "milvus_papers.db"),
        uri=milvus_config.get("address", ""),
        db_name=milvus_config.get("db_name", "default"),
        authentication=milvus_config.get("authentication"),
        hybrid_search=False,
    )


async def extract_chunks_from_milvus(
    output_path: str = "results/milvus_chunks.json",
) -> List[Dict[str, Any]]:
    """
    从 Milvus 提取全量 chunk 文本（按论文逐篇加载）

    Returns:
        [{"text": str, "metadata": dict, "paper_id": str}, ...]
    """
    print(f"\n{'='*60}")
    print("📤 步骤 1/4: 从 Milvus 提取全量文本（按论文逐篇加载）")
    print("=" * 60)

    index_manager = create_index_manager()
    chunks = await index_manager.get_all_chunks()

    # 保存到文件（原子写入）
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, output_path)

    print(f"✅ 提取完成: {len(chunks)} chunks -> {output_path}")

    # 按论文统计
    paper_counts: Dict[str, int] = {}
    for c in chunks:
        pid = c.get("paper_id", "unknown")
        paper_counts[pid] = paper_counts.get(pid, 0) + 1

    print(f"📊 论文数: {len(paper_counts)}")
    print(f"📊 Chunk 分布（前10）:")
    for pid, cnt in sorted(paper_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {pid}: {cnt} chunks")

    return chunks


# ============================================================================
# 步骤 2: 将 chunks 构建为 llama-index Document（支持多模态）
# ============================================================================

def _read_table_csv_as_text(csv_path: str, max_rows: int = 50) -> str:
    """
    读取 CSV 表格文件并转换为可读的文本格式

    Args:
        csv_path: CSV 文件路径
        max_rows: 最大读取行数（避免表格过大）

    Returns:
        表格的文本表示
    """
    from pathlib import Path
    import csv
    import io

    try:
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            return ""

        content = csv_path_obj.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            return ""

        # 限制行数
        header = rows[0] if rows else []
        data_rows = rows[1:max_rows]
        remaining = len(rows) - 1 - max_rows

        # 构建文本表示
        lines = []
        lines.append("| " + " | ".join(str(c) for c in header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in data_rows:
            cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
            lines.append("| " + " | ".join(cells) + " |")

        result = "\n".join(lines)
        if remaining > 0:
            result += f"\n... (还有 {remaining} 行)"

        return result
    except Exception as e:
        return f"[Table: {csv_path} (read error)]"


def _get_multimodal_context(node_metadata: Dict[str, Any], figures_dir: Path) -> str:
    """
    从 chunk metadata 中提取图片/表格上下文信息，转换为文本描述

    Args:
        node_metadata: chunk 的 metadata
        figures_dir: figures 目录路径

    Returns:
        包含图片/表格描述的文本
    """
    context_parts = []

    # 处理图片
    image_path = node_metadata.get("image_path", "")
    if image_path:
        image_caption = node_metadata.get("image_caption", "")
        if image_caption:
            context_parts.append(f"[IMAGE: {image_caption}]\nFile: {image_path}")
        else:
            context_parts.append(f"[IMAGE]\nFile: {image_path}")

    # 处理多图片
    all_images = node_metadata.get("all_images", [])
    for img_info in all_images:
        if isinstance(img_info, dict):
            img_path = img_info.get("path", "")
            img_caption = img_info.get("caption", "")
            if img_path and img_path != image_path:  # 避免重复
                if img_caption:
                    context_parts.append(f"[IMAGE: {img_caption}]\nFile: {img_path}")
                else:
                    context_parts.append(f"[IMAGE]\nFile: {img_path}")

    # 处理表格
    table_path = node_metadata.get("table_path", "")
    if table_path:
        table_caption = node_metadata.get("table_caption", "")
        table_text = _read_table_csv_as_text(table_path)
        if table_text:
            caption_prefix = f"[TABLE: {table_caption}]\n" if table_caption else "[TABLE]\n"
            context_parts.append(f"{caption_prefix}File: {table_path}\n{table_text}")
        else:
            context_parts.append(f"[TABLE: {table_caption}]\nFile: {table_path}")

    return "\n\n".join(context_parts) if context_parts else ""


def chunks_to_documents(
    chunks: List[Dict[str, Any]],
    min_chunks_per_paper: int = 5,
    include_multimodal: bool = True,
    figures_dir: str = "",
) -> List[Any]:
    """
    将 Milvus chunk 列表转换为 llama-index Document 列表

    Args:
        chunks: extract_chunks_from_milvus 返回的 chunk 列表
        min_chunks_per_paper: 最少 chunk 数才生成 Document（过滤太少内容的论文）
        include_multimodal: 是否在文本中包含图片/表格描述
        figures_dir: figures 目录路径（为空则使用插件默认路径）

    Returns:
        llama-index Document 列表
    """
    from collections import defaultdict
    from llama_index.core import Document as LIDocument
    from pathlib import Path

    # 确定 figures 目录
    if not figures_dir:
        plugin_dir = Path(__file__).parent.parent
        figures_dir = str(plugin_dir / "data" / "figures")

    papers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        pid = chunk.get("paper_id", "unknown")
        if pid and pid != "unknown":
            papers[pid].append(chunk)

    documents = []

    for paper_id, paper_chunks in papers.items():
        if len(paper_chunks) < min_chunks_per_paper:
            continue

        # 按 chunk id 排序，保证顺序
        paper_chunks.sort(key=lambda x: x.get("id", 0))

        # 构建带多模态信息的文档
        combined_parts = []
        has_multimodal = False
        multimodal_count = 0
        all_image_paths = []
        all_table_paths = []

        for c in paper_chunks:
            text = c.get("text", "")
            if not text:
                continue

            # 如果启用多模态且 metadata 中有图片/表格信息，追加描述
            if include_multimodal:
                metadata = c.get("metadata", {})
                if isinstance(metadata, dict):
                    multimodal_ctx = _get_multimodal_context(metadata, Path(figures_dir))
                    if multimodal_ctx:
                        has_multimodal = True
                        multimodal_count += 1
                        text = text + "\n\n" + multimodal_ctx
                    # 收集路径信息
                    if metadata.get("image_path"):
                        all_image_paths.append(metadata["image_path"])
                    if metadata.get("table_path"):
                        all_table_paths.append(metadata["table_path"])

            combined_parts.append(text)

        combined_text = "\n\n".join(combined_parts)

        if not combined_text.strip():
            continue

        doc = LIDocument(
            text=combined_text,
            metadata={
                "paper_id": paper_id,
                "chunk_count": len(paper_chunks),
                "source": "milvus",
                "has_multimodal": has_multimodal,
                "multimodal_count": multimodal_count,
                "image_paths": all_image_paths,
                "table_paths": all_table_paths,
            }
        )
        documents.append(doc)

    print(f"📄 构建 {len(documents)} 篇论文 Document")
    if include_multimodal:
        multimodal_docs = sum(1 for d in documents if d.metadata.get("has_multimodal"))
        print(f"   其中 {multimodal_docs} 篇包含图片/表格")
    return documents


# ============================================================================
# 步骤 2.5: 多模态测试集生成（仅生成与图表关联的问答对）
# ============================================================================

def extract_multimodal_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从 chunks 中提取与图片/表格关联的 chunks

    检查以下字段：
    - metadata.image_path: 直接的图片路径
    - metadata.table_path: 直接的表格路径
    - metadata.all_images: 图片列表
    - metadata.multimodal_data.images: 多模态数据中的图片
    - metadata.multimodal_data.tables: 多模态数据中的表格

    Args:
        chunks: 所有 chunks 列表

    Returns:
        包含图片/表格信息的 chunks 列表
    """
    multimodal_chunks = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        # 检查是否有图片或表格
        has_image = bool(metadata.get("image_path"))
        has_table = bool(metadata.get("table_path"))

        # 也检查 all_images 字段
        all_images = metadata.get("all_images", [])
        if not has_image and isinstance(all_images, list) and len(all_images) > 0:
            has_image = True

        # 检查 multimodal_data 中的 images 和 tables
        multimodal_data = metadata.get("multimodal_data", {})
        if isinstance(multimodal_data, dict):
            # multimodal_data.images 是一个列表
            mm_images = multimodal_data.get("images", [])
            if not has_image and isinstance(mm_images, list) and len(mm_images) > 0:
                has_image = True

            # multimodal_data.tables 是一个列表
            mm_tables = multimodal_data.get("tables", [])
            if not has_table and isinstance(mm_tables, list) and len(mm_tables) > 0:
                has_table = True

        if has_image or has_table:
            # 添加标记以便后续处理
            chunk = dict(chunk)  # 复制避免修改原数据
            chunk["_multimodal_type"] = []
            if has_image:
                chunk["_multimodal_type"].append("image")
            if has_table:
                chunk["_multimodal_type"].append("table")
            multimodal_chunks.append(chunk)

    print(f"🖼️ 找到 {len(multimodal_chunks)} 个与图片/表格关联的 chunks")
    return multimodal_chunks


def extract_multimodal_chunks_with_context(
    chunks: List[Dict[str, Any]],
    context_before: int = 1,
    context_after: int = 1,
    max_context_chunks_per_paper: int = 10,
) -> List[Dict[str, Any]]:
    """
    提取多模态 chunks 并附带同论文的上下文 chunks

    对于每个多模态 chunk，会添加：
    1. 同论文中相邻的普通 chunks（前后各 N 个）
    2. 同论文中的随机采样普通 chunks（最多 M 个）

    Args:
        chunks: 所有 chunks 列表
        context_before: 每个多模态 chunk 前面取几个普通 chunks
        context_after: 每个多模态 chunk 后面取几个普通 chunks
        max_context_chunks_per_paper: 每个论文最多添加的上下文 chunks 总数

    Returns:
        (多模态 chunks, 上下文 chunks) 元组
    """
    from collections import defaultdict
    import random

    # 按 paper_id 分组
    chunks_by_paper = defaultdict(list)
    for i, chunk in enumerate(chunks):
        paper_id = chunk.get("metadata", {}).get("paper_id", "") or chunk.get("paper_id", "")
        if paper_id:
            chunks_by_paper[paper_id].append((i, chunk))

    # 找出多模态 chunks 和普通 chunks
    multimodal_indices = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        has_image = bool(metadata.get("image_path") or metadata.get("all_images"))
        has_table = bool(metadata.get("table_path"))
        mm_data = metadata.get("multimodal_data", {})
        if isinstance(mm_data, dict):
            if mm_data.get("images") or mm_data.get("tables"):
                has_image = True

        if has_image or has_table:
            # 找到这个 chunk 在原始列表中的索引
            for i, c in enumerate(chunks):
                if c.get("id") == chunk.get("id"):
                    multimodal_indices.add(i)
                    break

    # 收集上下文 chunks
    context_chunks = []
    context_indices = set()

    for paper_id, paper_chunks in chunks_by_paper.items():
        paper_chunk_indices = [idx for idx, _ in paper_chunks]
        multimodal_in_paper = [idx for idx, _ in paper_chunks if idx in multimodal_indices]

        # 对于每个多模态 chunk，添加相邻的普通 chunks
        for mm_idx in multimodal_in_paper:
            # 前面的上下文
            for offset in range(1, context_before + 1):
                ctx_idx = mm_idx - offset
                if ctx_idx >= 0 and ctx_idx not in multimodal_indices and ctx_idx not in context_indices:
                    context_indices.add(ctx_idx)

            # 后面的上下文
            for offset in range(1, context_after + 1):
                ctx_idx = mm_idx + offset
                if ctx_idx < len(chunks) and ctx_idx not in multimodal_indices and ctx_idx not in context_indices:
                    context_indices.add(ctx_idx)

        # 如果上下文还不够，随机采样一些普通 chunks
        existing_count = len(context_indices)
        if existing_count < max_context_chunks_per_paper:
            non_multimodal = [idx for idx, _ in paper_chunks
                            if idx not in multimodal_indices and idx not in context_indices]
            needed = max_context_chunks_per_paper - existing_count
            # 按论文内顺序采样，保持均匀分布
            non_multimodal.sort()
            step = max(1, len(non_multimodal) // (needed * 2))
            sampled = non_multimodal[::step][:needed]
            for idx in sampled:
                context_indices.add(idx)

    # 构建返回列表
    multimodal_result = [chunks[i] for i in range(len(chunks)) if i in multimodal_indices]
    context_result = [chunks[i] for i in range(len(chunks)) if i in context_indices]

    print(f"🖼️ 多模态 chunks: {len(multimodal_result)}")
    print(f"📝 上下文 chunks: {len(context_result)}")

    return multimodal_result, context_result


def build_multimodal_documents(
    multimodal_chunks: List[Dict[str, Any]],
    figures_dir: str = "",
) -> List[Any]:
    """
    将多模态 chunks 构建为独立的 Document（每个 chunk 一个 Document）

    每个 Document 包含：
    - 原始文本
    - 图片路径和标题（如果有）
    - 表格内容和路径（如果有）

    Args:
        multimodal_chunks: extract_multimodal_chunks 返回的 chunks
        figures_dir: figures 目录路径

    Returns:
        多模态 Document 列表
    """
    from llama_index.core import Document as LIDocument

    if not figures_dir:
        plugin_dir = Path(__file__).parent.parent
        figures_dir = str(plugin_dir / "data" / "figures")

    documents = []
    for chunk in multimodal_chunks:
        metadata = chunk.get("metadata", {})
        text = chunk.get("text", "")
        multimodal_types = chunk.get("_multimodal_type", [])

        # 构建多模态描述（带格式指令前缀）
        parts = [MULTIMODAL_DOC_PREFIX, text]

        # 处理图片
        image_path = metadata.get("image_path", "")
        if image_path:
            image_caption = metadata.get("image_caption", "")
            if image_caption:
                parts.append(f"[IMAGE: {image_caption}]\nFile: {image_path}")
            else:
                parts.append(f"[IMAGE]\nFile: {image_path}")

        # 处理其他图片
        all_images = metadata.get("all_images", [])
        for img_info in all_images:
            if isinstance(img_info, dict):
                img_path = img_info.get("path", "")
                img_caption = img_info.get("caption", "")
                if img_path and img_path != image_path:
                    if img_caption:
                        parts.append(f"[IMAGE: {img_caption}]\nFile: {img_path}")
                    else:
                        parts.append(f"[IMAGE]\nFile: {img_path}")

        # 处理表格
        table_path = metadata.get("table_path", "")
        processed_table_paths = set()  # 避免重复处理

        if table_path:
            table_caption = metadata.get("table_caption", "")
            table_text = _read_table_csv_as_text(table_path)
            if table_caption:
                parts.append(f"[TABLE: {table_caption}]\nFile: {table_path}")
            else:
                parts.append(f"[TABLE]\nFile: {table_path}")
            if table_text:
                parts.append(table_text)
            processed_table_paths.add(table_path)

        # 也处理 multimodal_data.tables 中的表格（包含 markdown 内容）
        multimodal_data = metadata.get("multimodal_data", {})
        if isinstance(multimodal_data, dict):
            mm_tables = multimodal_data.get("tables", [])
            if isinstance(mm_tables, list):
                for table_info in mm_tables:
                    if isinstance(table_info, dict):
                        # multimodal_data.tables 中的表格没有 path 字段，直接使用 markdown 内容
                        tbl_markdown = table_info.get("markdown", "")
                        tbl_caption = table_info.get("caption", "")
                        page_num = table_info.get("page_number", "")
                        tbl_idx = table_info.get("table_index", "")

                        if tbl_markdown:
                            # 生成表格标识符（用于去重）
                            tbl_identifier = f"p{page_num}_t{tbl_idx}"
                            if tbl_identifier not in processed_table_paths:
                                if tbl_caption:
                                    parts.append(f"[TABLE {tbl_identifier}: {tbl_caption}]")
                                else:
                                    parts.append(f"[TABLE {tbl_identifier}]")
                                parts.append(tbl_markdown)
                                processed_table_paths.add(tbl_identifier)

        combined_text = "\n\n".join(parts)

        # 嵌入 chunk_id 用于后续匹配（不会影响 Ragas 生成）
        chunk_id = chunk.get("id", "")
        if chunk_id:
            combined_text = f"[CHUNK_ID:{chunk_id}]\n\n{combined_text}"

        doc = LIDocument(
            text=combined_text,
            metadata={
                "paper_id": chunk.get("paper_id", ""),
                "chunk_id": chunk_id,
                "source": "multimodal",
                "has_multimodal": True,  # 标记为多模态文档
                "multimodal_types": multimodal_types,
                "image_path": image_path,
                "table_path": table_path,
                "image_caption": metadata.get("image_caption", ""),
                "table_caption": metadata.get("table_caption", ""),
            }
        )
        documents.append(doc)

    print(f"📄 构建 {len(documents)} 个多模态 Document")
    return documents


MULTIMODAL_QUESTION_PROMPT = """你是一个学术论文问答对生成专家。请根据以下论文片段生成 {n} 个问答对。

要求：
1. 问题必须涉及图片或表格的具体内容
2. 答案应直接来自图片标题、表格内容或图片中的数据
3. 问题应该多样化，包括：描述图片内容、从表格中提取数据、比较表格数据、解释图表趋势等
4. 【重要】答案中的论文引用必须使用标准 Markdown 链接格式，如 [论文名](url)，禁止用代码块包裹

论文片段：
{context}

请生成 {n} 个问答对，格式为 JSON：
{{
  "question": "问题内容",
  "answer": "答案内容"
}}
"""

# 多模态文档前缀指令
MULTIMODAL_DOC_PREFIX = """【重要格式要求】
- 答案中的论文引用必须使用标准 Markdown 链接格式，如 [论文名](url)
- 禁止将引用放在代码块中
- 例如：正确格式是 [NoPoSplat](https://arxiv.org/abs/2502.12138)

"""


async def generate_multimodal_testset(
    chunks: List[Dict[str, Any]],
    test_size: int = 20,
    output_path: str = "results/testset.json",
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    llm_api_key: str = "",
    max_rpm: int = 96,
    context_before: int = 1,
    context_after: int = 1,
) -> List[Any]:
    """
    生成多模态问答对（专门针对图片和表格）

    会提取多模态 chunks 并附带同论文的上下文 chunks，使生成的问答能基于更完整的语境。

    Args:
        chunks: 所有 chunks 列表
        test_size: 生成问题数量
        output_path: 输出路径（会将结果 append 到已有文件）
        llm_model: LLM 模型名称
        llm_base_url: API 基础 URL
        llm_api_key: API Key
        max_rpm: RPM 限制
        context_before: 每个多模态 chunk 前面取几个普通 chunks 作为上下文
        context_after: 每个多模态 chunk 后面取几个普通 chunks 作为上下文

    Returns:
        生成的多模态样本列表
    """
    print(f"\n{'='*60}")
    print(f"🖼️ 步骤: 生成 {test_size} 个多模态评测问题（图片/表格）")
    print("=" * 60)

    # 提取多模态 chunks 和上下文 chunks
    multimodal_chunks, context_chunks = extract_multimodal_chunks_with_context(
        chunks,
        context_before=context_before,
        context_after=context_after,
    )

    if not multimodal_chunks:
        print("⚠️ 没有找到与图片/表格关联的 chunks")
        return []

    # 构建多模态 documents
    multimodal_docs = build_multimodal_documents(multimodal_chunks)

    if not multimodal_docs:
        print("⚠️ 没有成功构建多模态文档")
        return []

    # 构建上下文 documents（普通文本）
    from llama_index.core import Document as LIDocument
    context_docs = []
    for chunk in context_chunks:
        text = chunk.get("text", "")
        if text:
            doc = LIDocument(
                text=text,
                metadata={
                    "paper_id": chunk.get("paper_id", "") or chunk.get("metadata", {}).get("paper_id", ""),
                    "chunk_id": chunk.get("id", ""),
                    "source": "context",
                }
            )
            context_docs.append(doc)

    # 合并多模态文档和上下文文档
    all_docs = multimodal_docs + context_docs

    # 使用 Ragas 生成测试集
    from .ragas_generator import RagasTestsetGenerator

    generator = RagasTestsetGenerator(
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model="text-embedding-v3",
        embed_base_url=llm_base_url,
        embed_api_key=llm_api_key,
        embedding_mode="api",
        ollama_base_url="http://localhost:11434",
        ollama_embed_model="bge-m3",
        max_rpm=max_rpm,
    )

    # 只使用有多模态内容的文档
    multimodal_docs = [doc for doc in all_docs if doc.metadata.get("has_multimodal", False)]
    print(f"📊 多模态文档数量: {len(multimodal_docs)}")

    # 使用足够的文档（多模态 + 上下文）
    trimmed_docs = multimodal_docs[:min(test_size * MULTIMODAL_DOC_MULTIPLIER, len(multimodal_docs))]

    # 使用临时文件避免覆盖主文件（generate_testset 会直接写入文件）
    import tempfile
    import uuid
    temp_output = str(Path(tempfile.gettempdir()) / f"ragas_temp_{uuid.uuid4().hex}.json")

    print(f"正在调用 LLM 生成 {test_size} 个多模态问答对...")
    samples = await generator.generate_testset(
        documents=trimmed_docs,
        test_size=test_size,
        output_path=temp_output,
    )

    # 构建 chunk_id -> metadata 的映射（从原始 chunks）
    # 注意：chunk.get("id") 返回的是 int，需要转为 string 用于匹配
    chunk_id_to_metadata = {}
    for chunk in multimodal_chunks:
        chunk_id = str(chunk.get("id", ""))
        if chunk_id:
            metadata = chunk.get("metadata", {})
            chunk_id_to_metadata[chunk_id] = {
                "image_path": metadata.get("image_path", ""),
                "table_path": metadata.get("table_path", ""),
                "image_caption": metadata.get("image_caption", ""),
                "table_caption": metadata.get("table_caption", ""),
            }

    # 从 contexts[0] 解析 chunk_id 并找回 metadata
    import re
    from .ragas_generator import EvalSample
    multimodal_samples = []
    matched_count = 0
    unmatched_count = 0

    for sample in samples:
        contexts = sample.contexts if hasattr(sample, 'contexts') else []
        source_text = contexts[0] if contexts else ""

        # 解析 chunk_id
        match = re.search(r'\[CHUNK_ID:([^\]]+)\]', source_text)
        chunk_id = match.group(1) if match else ""
        metadata = chunk_id_to_metadata.get(chunk_id, {})

        is_multimodal = True  # 所有样本都是多模态的
        if chunk_id and metadata:
            matched_count += 1
        else:
            unmatched_count += 1

        sample.metadata = {
            "is_multimodal": is_multimodal,
            "multimodal_types": ["image", "table"],
            "image_path": metadata.get("image_path", ""),
            "table_path": metadata.get("table_path", ""),
            "image_caption": metadata.get("image_caption", ""),
            "table_caption": metadata.get("table_caption", ""),
            "is_context": False,
        }
        multimodal_samples.append(sample)

    print(f"📊 Metadata 匹配: {matched_count} 成功, {unmatched_count} 未匹配")

    # Append 到现有 testset.json
    existing_samples = []
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        try:
            with open(output_path_obj, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_samples = existing_data
                print(f"📖 读取已有测试集: {len(existing_samples)} 个样本")
            else:
                print(f"⚠️ 测试集格式错误（非列表）: {type(existing_data)}")
        except Exception as e:
            print(f"⚠️ 读取现有 testset 失败: {e}")
    else:
        print(f"📝 测试集文件不存在，将创建新文件")

    # 合并样本（带去重）
    new_sample_dicts = [s.to_dict() for s in multimodal_samples]

    # 去重：基于问题文本的 MD5 hash 去重，避免长文本截断导致的问题
    def get_question_key(sample: dict) -> str:
        q = sample.get("question", "")
        return hashlib.md5(q.encode()).hexdigest()[:16]

    seen_questions = set()
    deduped_existing = []
    for sample in existing_samples:
        q_key = get_question_key(sample)
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            deduped_existing.append(sample)

    deduped_new = []
    for sample in new_sample_dicts:
        q_key = get_question_key(sample)
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            deduped_new.append(sample)

    all_samples = deduped_existing + deduped_new

    print(f"🔄 合并（去重后）: {len(deduped_existing)} 已有 + {len(deduped_new)} 新增 = {len(all_samples)} 总计")
    if len(existing_samples) != len(deduped_existing) or len(new_sample_dicts) != len(deduped_new):
        print(f"   ⚠️ 去除 {len(existing_samples) - len(deduped_existing) + len(new_sample_dicts) - len(deduped_new)} 个重复样本")

    # 路径安全验证
    try:
        output_path_obj = Path(output_path).resolve()
        expected_dir = Path(__file__).parent.parent.joinpath("results").resolve()
        if not str(output_path_obj).startswith(str(expected_dir)):
            raise ValueError(f"Output path must be within {expected_dir}")
    except Exception as e:
        print(f"⚠️ 路径验证失败: {e}")
        raise

    # 原子写入：先写临时文件，再 rename
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    import os
    temp_final = str(output_path_obj) + ".tmp"
    with open(temp_final, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    os.replace(temp_final, output_path_obj)

    print(f"✅ 多模态测试集已生成并追加到: {output_path}")
    print(f"   新增 {len(multimodal_samples)} 个多模态样本")
    print(f"   总计 {len(all_samples)} 个样本")

    # 清理临时文件
    try:
        Path(temp_output).unlink()
    except Exception:
        pass

    return multimodal_samples


# ============================================================================
# 步骤 3: 生成测试集
# ============================================================================

async def generate_testset_from_documents(
    documents: List[Any],
    test_size: int = 50,
    output_path: str = "results/testset.json",
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    llm_api_key: str = "",
    embedding_model: str = "text-embedding-v3",
    embed_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    embed_api_key: str = "",
    embedding_mode: str = "api",
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "bge-m3",
    max_rpm: int = 96,
) -> List[Any]:
    """使用 Ragas 生成测试集"""
    print(f"\n{'='*60}")
    print(f"📝 步骤 2/4: 生成 {test_size} 个评测问题")
    print("=" * 60)

    from .ragas_generator import RagasTestsetGenerator

    generator = RagasTestsetGenerator(
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embedding_mode=embedding_mode,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
        max_rpm=max_rpm,
    )

    samples = await generator.generate_testset(
        documents=documents,
        test_size=test_size,
        output_path=output_path,
    )

    print(f"✅ 测试集生成完成: {len(samples)} 个样本 -> {output_path}")
    return samples


# ============================================================================
# 步骤 4: 执行评估
# ============================================================================

async def run_evaluation(
    query_engine: Any,
    testset_path: str,
    output_path: str = "results/evaluation_results.csv",
    max_concurrent: int = 5,
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    llm_api_key: str = "",
    embedding_model: str = "text-embedding-v3",
    embed_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    embed_api_key: str = "",
    embedding_mode: str = "ollama",
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "bge-m3",
    eval_embedding_mode: str = "api",
) -> Any:
    """执行 Ragas 评估"""
    print(f"\n{'='*60}")
    print("📊 步骤 3/4: 执行 Ragas 评估")
    print("=" * 60)

    from .ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embedding_mode=eval_embedding_mode,  # 评估用独立的 embedding 模式
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
    )

    results = await evaluator.evaluate(
        query_engine=query_engine,
        testset_path=testset_path,
        output_path=output_path,
        max_concurrent=max_concurrent,
    )

    print(f"✅ 评估完成 -> {output_path}")
    return results


async def run_evaluation_from_raw_answers(
    raw_answers_path: str,
    output_path: str = "results/evaluation_results.csv",
    max_concurrent: int = 5,
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    llm_api_key: str = "",
    embedding_model: str = "text-embedding-v3",
    embed_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    embed_api_key: str = "",
    embedding_mode: str = "api",
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "bge-m3",
) -> Any:
    """从已有 raw_answers.json 执行 Ragas 评估（跳过 RAG 推理）"""
    print(f"\n{'='*60}")
    print("📊 步骤 3/4: 执行 Ragas 评估（跳过 RAG 推理）")
    print("=" * 60)

    from .ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embedding_mode=embedding_mode,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
    )

    results = await evaluator.evaluate_from_raw_answers(
        raw_answers_path=raw_answers_path,
        output_path=output_path,
        max_concurrent=max_concurrent,
    )

    print(f"✅ 评估完成 -> {output_path}")
    return results


# ============================================================================
# 步骤 5: 生成报告
# ============================================================================

def generate_reports(
    results_path: str,
    output_dir: str = "results",
    plugin_version: str = "1.0.0",
    paper_name: str = "AstrBot Paper RAG (Milvus)",
) -> Dict[str, str]:
    """生成 HTML + Markdown 报告"""
    print(f"\n{'='*60}")
    print("📋 步骤 4/4: 生成评测报告")
    print("=" * 60)

    from .report_generator import ReportGenerator

    reporter = ReportGenerator(results_path)
    html_path = reporter.generate_html_report(
        str(Path(output_dir) / "evaluation_report.html"),
        plugin_version=plugin_version,
        paper_name=paper_name,
    )
    md_path = reporter.generate_markdown_report(
        str(Path(output_dir) / "evaluation_report.md"),
        plugin_version=plugin_version,
        paper_name=paper_name,
    )

    print(f"✅ 报告生成完成:")
    print(f"   HTML: {html_path}")
    print(f"   Markdown: {md_path}")

    return {"html": html_path, "markdown": md_path}


# ============================================================================
# 完整流程
# ============================================================================

async def run_full_pipeline(
    output_dir: str = "results",
    test_size: int = 50,
    max_concurrent: int = 5,
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    llm_api_key: str = "",
    embedding_model: str = "text-embedding-v3",
    embed_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    embed_api_key: str = "",
    embedding_mode: str = "api",
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "bge-m3",
    max_rpm: int = 96,
    plugin_version: str = "1.0.0",
    eval_llm_model: str = "gpt-4o-mini",
    eval_llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    use_existing_chunks: bool = False,
    existing_chunks_path: str = "results/milvus_chunks.json",
    eval_embedding_mode: str = "api",
) -> dict:
    """
    完整评测流程

    Args:
        use_existing_chunks: 是否使用已提取的 chunks 文件（避免重复从 Milvus 读取）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========== 步骤 1: 提取文本 ==========
    chunks_path = existing_chunks_path if use_existing_chunks else str(output_path / "milvus_chunks.json")

    if use_existing_chunks and Path(chunks_path).exists():
        print(f"\n{'='*60}")
        print("📤 步骤 1/4: 加载已有 chunks 文件")
        print("=" * 60)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✅ 加载 {len(chunks)} chunks from {chunks_path}")
    else:
        chunks = await extract_chunks_from_milvus(
            output_path=chunks_path,
        )

    if not chunks:
        return {"success": False, "error": "No chunks extracted from Milvus"}

    # ========== 步骤 2: 构建 Document + 生成测试集 ==========
    documents = chunks_to_documents(chunks, min_chunks_per_paper=5, include_multimodal=True)

    if not documents:
        return {"success": False, "error": "No valid documents created"}

    testset_path = str(output_path / "testset.json")
    samples = await generate_testset_from_documents(
        documents=documents,
        test_size=test_size,
        output_path=testset_path,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embedding_mode=embedding_mode,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
        max_rpm=max_rpm,
    )

    # ========== 步骤 3: 创建 RAG 查询引擎 ==========
    print(f"\n{'='*60}")
    print("🔧 步骤 3/4: 初始化 HybridRAG 引擎")
    print("=" * 60)

    # 使用插件现有配置创建引擎
    from rag_engine import create_rag_engine, RAGConfig

    plugin_dir = Path(__file__).parent.parent
    config_path = plugin_dir.parent.parent / "config" / "astrbot_plugin_paperrag_config.json"

    with open(config_path, "r", encoding="utf-8-sig") as f:
        rag_cfg = json.load(f)

    config = RAGConfig(
        embedding_mode=rag_cfg.get("embedding_mode", "ollama"),
        embedding_provider_id=rag_cfg.get("embedding_provider_id", ""),
        compress_provider_id=rag_cfg.get("compress_provider_id", ""),
        text_provider_id=rag_cfg.get("text_provider_id", ""),
        ollama_config=rag_cfg.get("ollama_config", {}),
        milvus_lite_path=str(plugin_dir / "data" / "milvus_papers.db"),
        address=rag_cfg.get("milvus", {}).get("address", ""),
        db_name=rag_cfg.get("milvus", {}).get("db_name", "default"),
        collection_name=rag_cfg.get("collection_name", "paper_embeddings"),
        embed_dim=rag_cfg.get("embed_dim", 1024),
        top_k=rag_cfg.get("top_k", 5),
        similarity_cutoff=rag_cfg.get("similarity_cutoff", 0.3),
        chunk_size=rag_cfg.get("chunk_size", 512),
        min_chunk_size=rag_cfg.get("min_chunk_size", 100),
        use_semantic_chunking=rag_cfg.get("use_semantic_chunking", True),
        enable_reranking=rag_cfg.get("enable_reranking", False),
    )

    # context 需要从 AstrBot 传入，这里用 None（引擎会跳过LLM初始化用于检索模式）
    class FakeContext:
        def __init__(self):
            self.provider_manager = None

    fake_context = FakeContext()

    engine = create_rag_engine(config, fake_context)
    print("✅ HybridRAG 引擎创建成功")

    # ========== 步骤 4: 执行评估 ==========
    results_path = str(output_path / "evaluation_results.csv")
    await run_evaluation(
        query_engine=engine,
        testset_path=testset_path,
        output_path=results_path,
        max_concurrent=max_concurrent,
        llm_model=eval_llm_model,
        llm_base_url=eval_llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
        embedding_mode=eval_embedding_mode,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
    )

    # ========== 步骤 5: 生成报告 ==========
    reports = generate_reports(
        results_path=results_path,
        output_dir=str(output_path),
        plugin_version=plugin_version,
        paper_name="AstrBot Paper RAG (Milvus DB)",
    )

    # ========== 完成 ==========
    print(f"\n{'='*60}")
    print("🎉 评测完成！")
    print(f"{'='*60}")
    print(f"📁 结果目录: {output_dir}/")
    print(f"   • Chunks: {chunks_path}")
    print(f"   • 测试集: {testset_path}")
    print(f"   • 评估结果: {results_path}")
    print(f"   • HTML报告: {reports['html']}")
    print(f"   • Markdown报告: {reports['markdown']}")
    print(f"{'='*60}")

    return {
        "success": True,
        "chunks": chunks_path,
        "testset": testset_path,
        "results": results_path,
        "reports": reports,
    }


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ragas 自动化评测工具 - 从 Milvus 数据库生成测试集并评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（提取文本 -> 生成测试集 -> 评估 -> 报告）
  python -m evaluation.run_evaluation_ragas --step all

  # 从 Milvus 提取全量文本（调试用）
  python -m evaluation.run_evaluation_ragas --step extract

  # 仅生成测试集（需已有 milvus_chunks.json）
  python -m evaluation.run_evaluation_ragas --step generate

  # 使用已有 chunks 文件（避免重复从数据库读取）
  python -m evaluation.run_evaluation_ragas --step all --use-existing-chunks

  # 指定测试集大小
  python -m evaluation.run_evaluation_ragas --step all --test-size 100

环境变量:
  EVAL_LLM_API_KEY 评估用 LLM API Key
        """
    )

    # 步骤参数
    parser.add_argument(
        "--step",
        choices=["all", "extract", "generate", "evaluate", "multimodal"],
        default="all",
        help="执行步骤: all=完整流程, extract=仅提取文本, generate=仅生成测试集, evaluate=需提供引擎, multimodal=仅生成多模态测试集"
    )

    # 输出
    parser.add_argument("--output-dir", default="results", help="输出目录")

    # 测试集配置
    parser.add_argument("--test-size", type=int, default=50, help="生成测试问题数量")
    parser.add_argument("--multimodal-test-size", type=int, default=20, help="多模态测试问题数量")
    parser.add_argument("--multimodal-context-before", type=int, default=1, help="多模态 chunk 前面的上下文 chunks 数量")
    parser.add_argument("--multimodal-context-after", type=int, default=1, help="多模态 chunk 后面的上下文 chunks 数量")

    # LLM 配置（用于生成测试集）
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM 模型名称")
    parser.add_argument("--llm-base-url", default="https://open.bigmodel.cn/api/paas/v4", help="LLM API 基础 URL")
    parser.add_argument("--llm-api-key", default="", help="LLM API Key（可使用环境变量）")

    # Eval LLM 配置（用于评估指标计算）
    parser.add_argument("--eval-llm-model", default="gpt-4o-mini", help="评估用 LLM 模型名称")
    parser.add_argument("--eval-llm-base-url", default="https://open.bigmodel.cn/api/paas/v4", help="评估用 LLM API 基础 URL")
    parser.add_argument("--eval-llm-api-key", default="", help="评估用 LLM API Key")

    # Embedding 配置
    parser.add_argument("--embedding-model", default="text-embedding-v3", help="Embedding 模型名称")
    parser.add_argument("--embed-base-url", default="https://open.bigmodel.cn/api/paas/v4", help="Embedding API 基础 URL")
    parser.add_argument("--embed-api-key", default="", help="Embedding API Key")
    parser.add_argument(
        "--embedding-mode",
        choices=["api", "ollama"],
        default="ollama",
        help="Embedding 模式: api=使用远程API, ollama=使用本地 ollama (默认: ollama)"
    )
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama API 基础 URL")
    parser.add_argument("--ollama-embed-model", default="bge-m3", help="Ollama Embedding 模型名称")

    # 评估用 Embedding 配置（默认使用 API 以保证兼容性）
    parser.add_argument(
        "--eval-embedding-mode",
        choices=["api", "ollama"],
        default="ollama",
        help="评估指标用 Embedding 模式: api=使用远程API, ollama=使用本地 ollama (默认)"
    )

    # 评测参数
    parser.add_argument("--max-concurrent", type=int, default=5, help="最大并发数")
    parser.add_argument("--max-rpm", type=int, default=96, help="RPM 限制（默认96）")

    # 报告配置
    parser.add_argument("--plugin-version", default="1.0.0", help="插件版本")
    parser.add_argument("--paper-name", default="AstrBot Paper RAG", help="论文/系统名称")

    # 已有数据
    parser.add_argument("--use-existing-chunks", action="store_true", help="使用已有 chunks 文件（避免重复从 Milvus 读取）")
    parser.add_argument("--existing-chunks-path", default="results/milvus_chunks.json", help="已有 chunks 文件路径")
    parser.add_argument("--skip-rag", action="store_true", help="跳过 RAG 推理，直接从 raw_answers.json 读取已有结果进行评估")
    parser.add_argument("--raw-answers-path", default="results/raw_answers.json", help="raw_answers.json 路径（用于 --skip-rag）")

    args = parser.parse_args()

    # 优先级: 显式参数 > 环境变量 > 插件配置
    llm_api_key = args.llm_api_key or os.getenv("EVAL_LLM_API_KEY", "")
    embed_api_key = args.embed_api_key or os.getenv("EVAL_LLM_API_KEY", "")

    # 尝试从插件配置读取 freeapi 设置（当 API Key 未显式提供时）
    plugin_config_path = Path(__file__).parent.parent.parent.parent / "config" / "astrbot_plugin_paperrag_config.json"
    embed_base_url = args.embed_base_url  # 默认使用命令行参数
    if plugin_config_path.exists():
        with open(plugin_config_path, "r", encoding="utf-8-sig") as f:
            plugin_config = json.load(f)
        config_freeapi_key = plugin_config.get("freeapi_key", "")
        config_freeapi_url = plugin_config.get("freeapi_url", "")
        if config_freeapi_key and not llm_api_key:
            llm_api_key = config_freeapi_key
            print(f"✅ 已从插件配置加载 freeapi key")
        if config_freeapi_url and args.llm_base_url == "https://open.bigmodel.cn/api/paas/v4":
            llm_base_url = config_freeapi_url + "/v1/"
            # freeapi 同时用于 LLM 和 Embedding
            embed_base_url = config_freeapi_url + "/v1/"
            print(f"✅ 已从插件配置加载 freeapi: {llm_base_url}")
        else:
            llm_base_url = args.llm_base_url
    else:
        llm_base_url = args.llm_base_url

    print(f"\n📊 配置信息:")
    print(f"   步骤: {args.step}")
    print(f"   LLM 模型: {args.llm_model}")
    print(f"   LLM API URL: {llm_base_url}")
    print(f"   Embedding 模型: {args.embedding_model}")
    print(f"   Embedding API URL: {embed_base_url}")
    print(f"   Embedding 模式: {args.embedding_mode}")
    if args.embedding_mode == "ollama":
        print(f"   Ollama 地址: {args.ollama_base_url}")
        print(f"   Ollama Embed 模型: {args.ollama_embed_model}")

    if not llm_api_key:
        print("⚠️ 警告: 未提供 API Key（设置 EVAL_LLM_API_KEY 环境变量或使用 --llm-api-key）")

    # ========== 根据步骤执行 ==========

    # 设置 RPM 限制
    from .ragas_generator import OpenAICompatibleLLM
    OpenAICompatibleLLM.set_max_rpm(args.max_rpm)
    print(f"   RPM 限制: {args.max_rpm}")
    if args.step == "all":
        asyncio.run(run_full_pipeline(
            output_dir=args.output_dir,
            test_size=args.test_size,
            max_concurrent=args.max_concurrent,
            llm_model=args.llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=args.embedding_model,
            embed_base_url=embed_base_url,
            embed_api_key=embed_api_key,
            embedding_mode=args.embedding_mode,
            ollama_base_url=args.ollama_base_url,
            ollama_embed_model=args.ollama_embed_model,
            max_rpm=args.max_rpm,
            plugin_version=args.plugin_version,
            eval_llm_model=args.eval_llm_model,
            eval_llm_base_url=args.eval_llm_base_url,
            use_existing_chunks=args.use_existing_chunks,
            existing_chunks_path=args.existing_chunks_path,
            eval_embedding_mode=args.eval_embedding_mode,
        ))

    elif args.step == "extract":
        path = args.existing_chunks_path
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        chunks = asyncio.run(extract_chunks_from_milvus(
            output_path=path,
        ))
        print(f"\n✅ 提取 {len(chunks)} chunks 完成")

    elif args.step == "generate":
        chunks_path = args.existing_chunks_path
        if not Path(chunks_path).exists():
            print(f"❌ 文件不存在: {chunks_path}")
            print("请先运行: python -m evaluation.run_evaluation_ragas --step extract")
            return
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        documents = chunks_to_documents(chunks, include_multimodal=True)
        asyncio.run(generate_testset_from_documents(
            documents=documents,
            test_size=args.test_size,
            llm_model=args.llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=args.embedding_model,
            embed_base_url=embed_base_url,
            embed_api_key=embed_api_key,
            embedding_mode=args.embedding_mode,
            ollama_base_url=args.ollama_base_url,
            ollama_embed_model=args.ollama_embed_model,
            max_rpm=args.max_rpm,
        ))

    elif args.step == "evaluate":
        # ========== 仅执行评估（使用已有测试集）==========
        results_path = str(Path(args.output_dir) / "evaluation_results.csv")

        if args.skip_rag:
            # 跳过 RAG 推理，直接从 raw_answers.json 读取评估
            raw_answers_path = args.raw_answers_path
            if not Path(raw_answers_path).exists():
                print(f"❌ raw_answers.json 不存在: {raw_answers_path}")
                print("请先运行带 RAG 推理的评估命令，生成该文件")
                return
            # 读取实际问题数量
            with open(raw_answers_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            actual_count = len(raw_data)
            print(f"✅ 跳过 RAG 推理，从已有结果评估")
            print(f"   raw_answers: {raw_answers_path}")
            print(f"   测试问题数量: {actual_count} (来自 raw_answers.json)")
            asyncio.run(run_evaluation_from_raw_answers(
                raw_answers_path=raw_answers_path,
                output_path=results_path,
                max_concurrent=args.max_concurrent,
                llm_model=args.eval_llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedding_model=args.embedding_model,
                embed_base_url=embed_base_url,
                embed_api_key=embed_api_key,
                embedding_mode=args.embedding_mode,
                ollama_base_url=args.ollama_base_url,
                ollama_embed_model=args.ollama_embed_model,
            ))
        else:
            # 正常流程：RAG 推理 + 评估
            testset_path = str(Path(args.output_dir) / "testset.json")

            if not Path(testset_path).exists():
                print(f"❌ 测试集不存在: {testset_path}")
                print("请先运行: python -m evaluation.run_evaluation_ragas --step all --use-existing-chunks")
                return

            print(f"✅ 使用已有测试集: {testset_path}")
            print(f"   测试问题数量: {args.test_size}")

            # 创建 RAG 引擎
            from rag_engine import create_rag_engine, RAGConfig

            plugin_dir = Path(__file__).parent.parent
            config_path = plugin_dir.parent.parent / "config" / "astrbot_plugin_paperrag_config.json"

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8-sig") as f:
                    rag_cfg = json.load(f)
            else:
                rag_cfg = {}

            # 当 embedding_mode 为 ollama 时，确保 ollama_config 不为空
            ollama_cfg = rag_cfg.get("ollama_config") or {}
            if args.embedding_mode == "ollama" and not ollama_cfg:
                ollama_cfg = {
                    "model": args.ollama_embed_model,
                    "base_url": args.ollama_base_url,
                }

            config = RAGConfig(
                embedding_mode=args.embedding_mode,
                embedding_provider_id=rag_cfg.get("embedding_provider_id", ""),
                compress_provider_id=rag_cfg.get("compress_provider_id", ""),
                text_provider_id=rag_cfg.get("text_provider_id", ""),
                ollama_config=ollama_cfg,
                milvus_lite_path=str(plugin_dir / "data" / "milvus_papers.db"),
                address=rag_cfg.get("milvus", {}).get("address", ""),
                db_name=rag_cfg.get("milvus", {}).get("db_name", "default"),
                collection_name=rag_cfg.get("collection_name", "paper_embeddings"),
                embed_dim=rag_cfg.get("embed_dim", 1024),
                top_k=rag_cfg.get("top_k", 5),
                similarity_cutoff=rag_cfg.get("similarity_cutoff", 0.3),
                chunk_size=rag_cfg.get("chunk_size", 512),
                min_chunk_size=rag_cfg.get("min_chunk_size", 100),
                use_semantic_chunking=rag_cfg.get("use_semantic_chunking", True),
                enable_reranking=rag_cfg.get("enable_reranking", False),
            )

            # Fake context for engine
            class FakeContext:
                def __init__(self):
                    self.provider_manager = None

            fake_context = FakeContext()

            print("🔧 初始化 HybridRAG 引擎...")
            engine = create_rag_engine(config, fake_context)
            print("✅ 引擎创建成功")

            # 执行评估
            asyncio.run(run_evaluation(
                query_engine=engine,
                testset_path=testset_path,
                output_path=results_path,
                max_concurrent=args.max_concurrent,
                llm_model=args.eval_llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedding_model=args.embedding_model,
                embed_base_url=embed_base_url,
                embed_api_key=embed_api_key,
                eval_embedding_mode=args.eval_embedding_mode,
            ))

        # 生成报告
        generate_reports(
            results_path=results_path,
            output_dir=args.output_dir,
            plugin_version=args.plugin_version,
            paper_name=args.paper_name,
        )

        print(f"\n🎉 评估完成！结果: {results_path}")

    elif args.step == "multimodal":
        # ========== 仅生成多模态测试集 ==========
        chunks_path = args.existing_chunks_path
        if args.use_existing_chunks and Path(chunks_path).exists():
            print(f"\n{'='*60}")
            print("📤 加载已有 chunks 文件")
            print("=" * 60)
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            print(f"✅ 加载 {len(chunks)} chunks from {chunks_path}")
        else:
            chunks = asyncio.run(extract_chunks_from_milvus(output_path=chunks_path))

        if not chunks:
            print("❌ 没有可用的 chunks")
            return

        asyncio.run(generate_multimodal_testset(
            chunks=chunks,
            test_size=args.multimodal_test_size,
            output_path=str(Path(args.output_dir) / "testset.json"),
            llm_model=args.llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            max_rpm=args.max_rpm,
            context_before=args.multimodal_context_before,
            context_after=args.multimodal_context_after,
        ))
        print(f"\n🎉 多模态测试集生成完成！")


if __name__ == "__main__":
    main()
