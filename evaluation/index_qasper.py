#!/usr/bin/env python3
"""
Qasper 论文索引脚本

使用插件的虚拟环境运行:
    ../.venv/bin/python index_qasper.py

或激活虚拟环境后运行:
    source ../.venv/bin/activate
    python index_qasper.py

将 Qasper 数据集中的论文文本索引到 Milvus 数据库

注意：Qasper 数据集不包含 PDF 文件，只包含提取的论文文本。
本脚本直接使用数据集中的 full_text 进行索引。

用法:
    python index_qasper.py                    # 索引所有论文到 data/milvus_qasper.db
    python index_qasper.py --split train       # 仅索引训练集
    python index_qasper.py --reinit             # 重新初始化数据库
    python index_qasper.py --milvus-qasper-path ./data/milvus_qasper.db  # 指定数据库路径
    python index_qasper.py --qasper-doc-stats ./data/qasper_doc_stats.json  # 指定统计文件路径
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加插件目录到路径
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 数据集路径
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets" / "data" / "qasper"


def load_qasper_data(data_dir: Path, split: str = "all") -> dict:
    """
    加载 Qasper 数据集

    Args:
        data_dir: 数据目录 (包含 train.jsonl, validation.jsonl, test.jsonl)
        split: "all", "train", "validation", "test"

    Returns:
        {paper_id: paper_data} 字典
    """
    result = {}

    splits_to_load = []
    if split == "all":
        splits_to_load = ["train", "validation", "test"]
    else:
        splits_to_load = [split]

    for s in splits_to_load:
        jsonl_file = data_dir / f"{s}.jsonl"
        if not jsonl_file.exists():
            print(f"⚠️  文件不存在: {jsonl_file}")
            continue

        print(f"加载 {s} 集: {jsonl_file}")
        count = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    paper = json.loads(line)
                    paper_id = paper.get("id", f"paper_{count}")
                    result[paper_id] = paper
                    count += 1

        print(f"  加载了 {count} 篇论文")

    return result


def prepare_figure_caption_nodes(paper: dict, paper_id: str) -> List[dict]:
    """
    从论文中提取图表 caption 作为独立节点

    Args:
        paper: 论文数据
        paper_id: 论文 ID

    Returns:
        Node 列表，每个 caption 一个 Node
    """
    nodes = []
    metadata = {
        "paper_id": paper_id,
        "file_name": paper_id,
        "paper_title": paper.get("title", ""),
    }

    figures_and_tables = paper.get("figures_and_tables", [])
    for fig_idx, fig in enumerate(figures_and_tables):
        caption = fig.get("caption", "")
        if not caption or len(caption.strip()) < 20:
            continue

        file_name = fig.get("file", "")
        node_metadata = {
            **metadata,
            "section_name": "figures_and_tables",
            "paragraph_index": fig_idx,
            "node_type": "figure_caption",
            "figure_file": file_name,
        }

        # 区分 Figure 和 Table
        if file_name.lower().startswith(("fig", "figure")):
            node_text = f"Figure: {caption}"
        else:
            node_text = f"Table: {caption}"

        nodes.append({
            "text": node_text,
            "metadata": node_metadata
        })

    return nodes


def prepare_nodes_from_paper(paper: dict, paper_id: str) -> tuple:
    """
    将论文转换为 Node 列表

    Args:
        paper: 论文数据
        paper_id: 论文 ID

    Returns:
        (nodes, chunk_count) 元组
        - nodes: Node 列表，每个段落一个 Node
        - chunk_count: 该论文的 chunk 数量
    """
    nodes = []

    # 论文元数据
    # Qasper 使用 paper_id 作为 file_name（兼容 ragas_generator 的 MilvusDocumentLoader）
    metadata = {
        "paper_id": paper_id,
        "file_name": paper_id,  # 兼容 file_name 字段用于查询
        "paper_title": paper.get("title", ""),  # Qasper 使用 title 而不是 paper_title
        "abstract": paper.get("abstract", ""),
        "source_split": paper.get("source_split", ""),
        "doi": paper.get("doi"),
        "arxiv_id": paper.get("arxiv_id"),
    }

    # 处理 full_text
    full_text = paper.get("full_text", [])
    chunk_count = 0
    if isinstance(full_text, list):
        for section in full_text:
            section_name = section.get("section_name", "") if isinstance(section, dict) else ""
            paragraphs = section.get("paragraphs", []) if isinstance(section, dict) else []

            for para_idx, para_text in enumerate(paragraphs):
                if not para_text or not isinstance(para_text, str):
                    continue

                # 跳过太短的段落
                if len(para_text.strip()) < 50:
                    continue

                node_metadata = {
                    **metadata,
                    "section_name": section_name,
                    "paragraph_index": para_idx,
                    "node_type": "paragraph",
                }

                nodes.append({
                    "text": para_text.strip(),
                    "metadata": node_metadata
                })
                chunk_count += 1
    else:
        # full_text 是字符串（某些格式）
        if full_text and isinstance(full_text, str):
            nodes.append({
                "text": full_text.strip(),
                "metadata": metadata
            })
            chunk_count += 1

    return nodes, chunk_count


async def initialize_milvus(
    config: dict,
    collection_name: str,
    milvus_lite_path: Optional[str] = None
):
    """初始化 Milvus 数据库（清除旧数据）"""
    from milvus_manager import PaperMilvusManager

    # 使用覆盖路径或配置中的路径
    effective_lite_path = milvus_lite_path if milvus_lite_path else config.get("milvus_lite_path", "")

    # 注意：Qasper 使用 bge-m3 模型，固定 1024 维
    embed_dim = 1024

    milvus_config = {
        "milvus_lite_path": effective_lite_path,
        "address": config.get("address", ""),
        "db_name": config.get("db_name", "default"),
        "authentication": config.get("authentication", {}),
        "collection_name": collection_name,
        "embed_dim": embed_dim,
    }

    manager = PaperMilvusManager(
        collection_name=collection_name,
        dim=embed_dim,
        lite_path=milvus_config["milvus_lite_path"],
        address=milvus_config["address"],
        db_name=milvus_config["db_name"],
        authentication=milvus_config["authentication"],
    )

    # 清除旧集合
    print("清除旧集合...")
    await manager.clear_collection()
    print("集合已清除")

    # 重新创建
    print("创建新集合...")
    await manager._ensure_collection()
    print("集合已创建")

    return manager


async def index_papers(
    config: dict,
    papers_data: dict,
    collection_name: str = "paper_embeddings",
    milvus_lite_path: Optional[str] = None,
    include_figures: bool = False,
) -> tuple:
    """
    将论文索引到 Milvus

    Args:
        config: 插件配置
        papers_data: {paper_id: paper_data}
        collection_name: 集合名称
        milvus_lite_path: 可选的 Milvus Lite 路径覆盖
        include_figures: 是否将图表 captions 加入索引

    Returns:
        (total_indexed, paper_stats) 元组
        - total_indexed: 索引的段落数量
        - paper_stats: {paper_id: {"file_name": ..., "chunk_count": ..., "paper_title": ...}, ...}
    """
    from embedding_providers import create_embedding_provider, OllamaEmbeddingProvider, OllamaEmbeddingConfig
    from milvus_manager import PaperMilvusManager

    # 使用覆盖路径或配置中的路径
    effective_lite_path = milvus_lite_path if milvus_lite_path else config.get("milvus_lite_path", "")

    # 注意：Qasper 使用 bge-m3 模型，固定 1024 维
    # 不使用配置文件中可能不正确的 embed_dim 值
    embed_dim = 1024

    milvus_config = {
        "milvus_lite_path": effective_lite_path,
        "address": config.get("address", ""),
        "db_name": config.get("db_name", "default"),
        "authentication": config.get("authentication", {}),
        "collection_name": collection_name,
        "embed_dim": embed_dim,
    }

    # 初始化 Milvus
    milvus_manager = PaperMilvusManager(
        collection_name=collection_name,
        dim=embed_dim,
        lite_path=milvus_config["milvus_lite_path"],
        address=milvus_config["address"],
        db_name=milvus_config["db_name"],
        authentication=milvus_config["authentication"],
    )

    await milvus_manager._ensure_collection()

    # 初始化 Embedding Provider
    embedding_mode = config.get("embedding_mode", "ollama")

    if embedding_mode == "ollama":
        ollama_config = config.get("ollama", {})
        embed_config = OllamaEmbeddingConfig(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            model=ollama_config.get("model", "bge-m3"),
            timeout=ollama_config.get("timeout", 120.0),
            batch_size=ollama_config.get("batch_size", 10),
        )
        embed_provider = OllamaEmbeddingProvider(config=embed_config)
    else:
        # API 模式
        embed_provider = create_embedding_provider(
            mode="astrbot",
            provider_id=config.get("embedding_provider_id", ""),
        )

    # 准备所有 nodes 并统计每篇论文的 chunk 数量
    print("\n准备节点...")
    all_nodes = []
    paper_stats: Dict[str, Dict] = {}
    total_figure_captions = 0

    for paper_id, paper in papers_data.items():
        nodes, chunk_count = prepare_nodes_from_paper(paper, paper_id)

        # 可选：添加图表 caption 节点
        fig_caption_nodes = []
        if include_figures:
            fig_caption_nodes = prepare_figure_caption_nodes(paper, paper_id)
            nodes.extend(fig_caption_nodes)
            total_figure_captions += len(fig_caption_nodes)

        all_nodes.extend(nodes)
        # Qasper 使用 paper_id 作为 file_name
        paper_title = paper.get("title", paper_id)
        paper_stats[paper_id] = {
            "file_name": paper_id,  # 使用 paper_id 作为 file_name
            "chunk_count": chunk_count,
            "figure_caption_count": len(fig_caption_nodes),
            "paper_title": paper_title,
            "source_split": paper.get("source_split", ""),
        }

    print(f"共 {len(all_nodes)} 个节点 (含 {total_figure_captions} 个图表 captions)")

    # 分批索引
    batch_size = config.get("batch_size", 10)
    total_indexed = 0

    print(f"\n开始索引 (每批 {batch_size})...")

    for i in range(0, len(all_nodes), batch_size):
        batch = all_nodes[i:i + batch_size]

        # 提取文本用于 embedding
        texts = [node["text"] for node in batch]

        # 获取 embeddings
        try:
            if embedding_mode == "ollama":
                embeddings = await embed_provider.get_text_embeddings_batch(texts)
            else:
                embeddings = await embed_provider.get_text_embeddings_batch(texts)

            # 准备插入文档
            documents = []
            for node_dict, embedding in zip(batch, embeddings):
                documents.append({
                    "text": node_dict["text"],
                    "metadata": node_dict["metadata"],
                    "embedding": embedding
                })

            # 插入 Milvus
            await milvus_manager.insert_documents(documents)
            total_indexed += len(batch)

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(all_nodes):
                print(f"  已索引 {total_indexed}/{len(all_nodes)} 段落")

        except Exception as e:
            print(f"  ⚠️  批次 {i // batch_size} 索引失败: {e}")
            continue

    print(f"\n✅ 索引完成！共索引 {total_indexed} 个段落")
    return total_indexed, paper_stats


async def main_async(args):
    """异步主函数"""
    import datetime

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR

    # 加载 Qasper 数据
    print(f"\n加载 Qasper 数据集: {data_dir}")
    papers_data = load_qasper_data(data_dir, split=args.split)

    if not papers_data:
        print("❌ 没有加载到任何论文数据")
        return

    total_papers = len(papers_data)
    total_paragraphs = sum(
        sum(1 for s in p.get("full_text", []) if isinstance(s, dict) for _ in s.get("paragraphs", []) if _)
        for p in papers_data.values()
    )

    print(f"\n数据集统计:")
    print(f"  论文数: {total_papers}")
    print(f"  段落数: ~{total_paragraphs}")

    collection_name = "paper_embeddings"

    # 根据是否包含图表 captions 选择数据库路径
    include_figures = args.include_figures_captions
    if include_figures:
        default_qasper_path = str(SCRIPT_DIR / "data" / "milvus_qasper_vision.db")
        default_stats_path = str(SCRIPT_DIR / "data" / "qasper_doc_stats_vision.json")
        print("\n[INFO] 将把图表 captions 加入索引 (vision 模式)")
    else:
        default_qasper_path = str(SCRIPT_DIR / "data" / "milvus_qasper_text.db")
        default_stats_path = str(SCRIPT_DIR / "data" / "qasper_doc_stats_text.json")
        print("\n[INFO] text-only 模式")

    milvus_qasper_path = args.milvus_qasper_path if args.milvus_qasper_path else default_qasper_path
    print(f"\n📦 Milvus Qasper 数据库路径: {milvus_qasper_path}")

    # 硬编码配置，不读取配置文件
    plugin_config = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "bge-m3",
            "timeout": 120.0,
            "batch_size": 10,
        },
    }

    if args.reinit:
        # 重新初始化数据库
        print(f"\n重新初始化 Qasper 数据库: {collection_name}")
        await initialize_milvus(plugin_config, collection_name, milvus_lite_path=milvus_qasper_path)

    # 索引论文
    print(f"\n索引论文到集合: {collection_name}")
    total_indexed, paper_stats = await index_papers(
        plugin_config,
        papers_data,
        collection_name=collection_name,
        milvus_lite_path=milvus_qasper_path,
        include_figures=include_figures
    )

    # 保存 qasper_doc_stats.json
    stats_path = args.qasper_doc_stats if args.qasper_doc_stats else default_stats_path

    # 转换为与 paper_doc_stats.json 兼容的格式
    doc_stats = {}
    added_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for paper_id, stats in paper_stats.items():
        doc_stats[paper_id] = {
            "file_name": stats["file_name"],
            "chunk_count": stats["chunk_count"],
            "figure_caption_count": stats.get("figure_caption_count", 0),
            "added_time": added_time,
            "paper_title": stats.get("paper_title", ""),
            "source_split": stats.get("source_split", ""),
        }

    # 写入文件
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(doc_stats, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 论文统计已保存到: {stats_path}")

    # 统计图表 caption 总数
    total_fig_captions = sum(s.get("figure_caption_count", 0) for s in paper_stats.values())

    print("\n" + "=" * 60)
    print("索引完成!")
    print("=" * 60)
    print(f"  论文数: {total_papers}")
    print(f"  段落数: {total_indexed - total_fig_captions}")
    print(f"  图表 captions: {total_fig_captions}")
    print(f"  集合名: {collection_name}")
    print(f"  Milvus 数据库: {milvus_qasper_path}")
    print(f"  统计文件: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Qasper 论文索引脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python index_qasper.py                                    # 索引所有论文到 data/milvus_qasper.db
  python index_qasper.py --split train                     # 仅索引训练集
  python index_qasper.py --reinit                          # 重新初始化数据库
  python index_qasper.py --include-figures-captions        # 索引时包含图表 captions
  python index_qasper.py --milvus-qasper-path /path/to.db  # 指定 Milvus 数据库路径
  python index_qasper.py --qasper-doc-stats /path/to.json  # 指定统计文件路径
        """
    )

    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "validation", "test"],
                        help="数据集划分 (默认: all)")
    parser.add_argument("--reinit", action="store_true",
                        help="重新初始化数据库（清除旧数据）")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录路径")
    parser.add_argument("--milvus-qasper-path", type=str, default=None,
                        help="Qasper 专用 Milvus 数据库路径 (默认: ./data/milvus_qasper.db)")
    parser.add_argument("--qasper-doc-stats", type=str, default=None,
                        help="Qasper 论文统计 JSON 文件路径 (默认: ./data/qasper_doc_stats.json)")
    parser.add_argument("--include-figures-captions", action="store_true",
                        help="将图表 captions 加入索引（用于对比实验）")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
