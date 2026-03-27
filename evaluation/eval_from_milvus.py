# -*- coding: utf-8 -*-
"""
从现有 Milvus 数据库生成评测数据集 + 执行评测

流程:
  1. 从 Milvus 提取全量 chunk 文本
  2. 按论文分组，构建 llama-index Document
  3. 调用 Ragas TestsetGenerator 生成问答对
  4. 使用 HybridRAGEngine 执行 RAG 查询
  5. 调用 Ragas Evaluator 计算 6 大指标
  6. 生成 HTML + Markdown 报告

用法:
    # 完整流程
    python -m evaluation.eval_from_milvus --step all

    # 仅从 Milvus 提取文本（调试用）
    python -m evaluation.eval_from_milvus --step extract

    # 仅生成测试集（需要已有提取的文本）
    python -m evaluation.eval_from_milvus --step generate

    # 仅执行评估（需要已有测试集和 RAG 引擎）
    python -m evaluation.eval_from_milvus --step evaluate
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 确保 evaluation 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from astrbot.api import logger

# ============================================================================
# 步骤 1: 从 Milvus 提取全量文本
# ============================================================================

def create_index_manager() -> Any:
    """创建 HybridIndexManager 实例（复用现有配置）"""
    import json
    from pathlib import Path

    # 读取插件配置
    plugin_dir = Path(__file__).parent.parent
    config_path = plugin_dir / "data" / "cmd_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 提取 RAG 配置
    rag_config = config.get("rag_config", {})
    milvus_config = rag_config.get("milvus", {})
    embedding_config = rag_config.get("embedding", {})

    from .hybrid_index import HybridIndexManager

    return HybridIndexManager(
        collection_name=rag_config.get("collection_name", "paper_embeddings"),
        dim=rag_config.get("embed_dim", 1024),
        milvus_lite_path=str(plugin_dir / "data" / "milvus_papers.db"),
        address=milvus_config.get("address", ""),
        db_name=milvus_config.get("db_name", "default"),
        authentication=milvus_config.get("authentication"),
        hybrid_search=False,
    )


async def extract_chunks_from_milvus(
    output_path: str = "results/milvus_chunks.json",
    batch_size: int = 1000,
) -> List[Dict[str, Any]]:
    """
    从 Milvus 提取全量 chunk 文本

    Returns:
        [{"text": str, "metadata": dict, "paper_id": str}, ...]
    """
    print(f"\n{'='*60}")
    print("📤 步骤 1/4: 从 Milvus 提取全量文本")
    print("=" * 60)

    index_manager = create_index_manager()
    chunks = await index_manager.get_all_chunks(batch_size=batch_size)

    # 保存到文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

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
# 步骤 2: 将 chunks 构建为 llama-index Document
# ============================================================================

def chunks_to_documents(
    chunks: List[Dict[str, Any]],
    min_chunks_per_paper: int = 5,
) -> List[Any]:
    """
    将 Milvus chunk 列表转换为 llama-index Document 列表

    Args:
        chunks: extract_chunks_from_milvus 返回的 chunk 列表
        min_chunks_per_paper: 最少 chunk 数才生成 Document（过滤太少内容的论文）

    Returns:
        llama-index Document 列表
    """
    # 按 paper_id 分组
    from collections import defaultdict

    papers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        pid = chunk.get("paper_id", "unknown")
        if pid and pid != "unknown":
            papers[pid].append(chunk)

    # 构建 Document
    documents = []
    from llama_index.core import Document as LIDocument

    for paper_id, paper_chunks in papers.items():
        if len(paper_chunks) < min_chunks_per_paper:
            continue

        # 按 chunk id 排序，保证顺序
        paper_chunks.sort(key=lambda x: x.get("id", 0))

        # 合并同一论文的所有 chunk（用双换行分隔）
        combined_text = "\n\n".join(c.get("text", "") for c in paper_chunks if c.get("text"))

        if not combined_text.strip():
            continue

        doc = LIDocument(
            text=combined_text,
            metadata={
                "paper_id": paper_id,
                "chunk_count": len(paper_chunks),
                "source": "milvus",
            }
        )
        documents.append(doc)

    print(f"📄 构建 {len(documents)} 篇论文 Document")
    return documents


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
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
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
    )

    samples = await generator.generate_testset(
        documents=documents,
        test_size=test_size,
        output_path=output_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
    )

    results = await evaluator.evaluate(
        query_engine=query_engine,
        testset_path=testset_path,
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
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    plugin_version: str = "1.0.0",
    milvus_batch_size: int = 1000,
    eval_llm_model: str = "gpt-4o-mini",
    eval_llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    eval_llm_api_key: str = "",
    use_existing_chunks: bool = False,
    existing_chunks_path: str = "results/milvus_chunks.json",
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
            batch_size=milvus_batch_size,
        )

    if not chunks:
        return {"success": False, "error": "No chunks extracted from Milvus"}

    # ========== 步骤 2: 构建 Document + 生成测试集 ==========
    documents = chunks_to_documents(chunks, min_chunks_per_paper=5)

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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # ========== 步骤 3: 创建 RAG 查询引擎 ==========
    print(f"\n{'='*60}")
    print("🔧 步骤 3/4: 初始化 HybridRAG 引擎")
    print("=" * 60)

    # 使用插件现有配置创建引擎
    from .rag_engine import create_rag_engine, RAGConfig
    import json
    from pathlib import Path

    plugin_dir = Path(__file__).parent.parent
    config_path = plugin_dir / "data" / "cmd_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        plugin_cfg = json.load(f)

    rag_cfg = plugin_cfg.get("rag_config", {})

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
        pass

    fake_context = FakeContext()
    fake_context.provider_manager = None

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
        llm_api_key=eval_llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=embed_base_url,
        embed_api_key=embed_api_key,
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
        description="从 Milvus 数据库生成评测数据集并执行 Ragas 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（提取文本 -> 生成测试集 -> 评估 -> 报告）
  python -m evaluation.eval_from_milvus --step all --test-size 50

  # 使用已有的 milvus_chunks.json（避免重复从数据库读取）
  python -m evaluation.eval_from_milvus --step all --use-existing-chunks

  # 仅提取文本
  python -m evaluation.eval_from_milvus --step extract

  # 仅生成测试集（需已有 milvus_chunks.json）
  python -m evaluation.eval_from_milvus --step generate --test-size 100

环境变量:
  ZHIPU_API_KEY    智谱 API Key
  EVAL_LLM_API_KEY 评估用 LLM API Key（优先级高于 ZHIPU_API_KEY）
        """
    )

    parser.add_argument("--step", choices=["all", "extract", "generate", "evaluate"], default="all")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--milvus-batch-size", type=int, default=1000)

    # LLM 配置（用于生成测试集）
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--llm-api-key", default="")

    # Eval LLM 配置（用于评估指标计算）
    parser.add_argument("--eval-llm-model", default="gpt-4o-mini")
    parser.add_argument("--eval-llm-base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--eval-llm-api-key", default="")

    # Embedding 配置
    parser.add_argument("--embedding-model", default="text-embedding-v3")
    parser.add_argument("--embed-base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--embed-api-key", default="")

    # 报告配置
    parser.add_argument("--plugin-version", default="1.0.0")
    parser.add_argument("--paper-name", default="AstrBot Paper RAG")

    # 已有数据
    parser.add_argument("--use-existing-chunks", action="store_true")
    parser.add_argument("--existing-chunks-path", default="results/milvus_chunks.json")

    args = parser.parse_args()

    # 优先级: 显式参数 > 环境变量
    llm_api_key = args.llm_api_key or os.getenv("ZHIPU_API_KEY", "")
    eval_llm_api_key = args.eval_llm_api_key or os.getenv("EVAL_LLM_API_KEY", "") or llm_api_key
    embed_api_key = args.embed_api_key or os.getenv("ZHIPU_API_KEY", "")

    if not llm_api_key:
        print("⚠️ 警告: 未提供 API Key（设置 ZHIPU_API_KEY 环境变量）")

    if args.step == "all":
        asyncio.run(run_full_pipeline(
            output_dir=args.output_dir,
            test_size=args.test_size,
            max_concurrent=args.max_concurrent,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=args.embedding_model,
            embed_base_url=args.embed_base_url,
            embed_api_key=embed_api_key,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            plugin_version=args.plugin_version,
            milvus_batch_size=args.milvus_batch_size,
            eval_llm_model=args.eval_llm_model,
            eval_llm_base_url=args.eval_llm_base_url,
            eval_llm_api_key=eval_llm_api_key,
            use_existing_chunks=args.use_existing_chunks,
            existing_chunks_path=args.existing_chunks_path,
        ))

    elif args.step == "extract":
        path = args.existing_chunks_path
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)
        chunks = asyncio.run(extract_chunks_from_milvus(
            output_path=path,
            batch_size=args.milvus_batch_size,
        ))
        print(f"\n✅ 提取 {len(chunks)} chunks 完成")

    elif args.step == "generate":
        chunks_path = args.existing_chunks_path
        if not Path(chunks_path).exists():
            print(f"❌ 文件不存在: {chunks_path}")
            print("请先运行: python -m evaluation.eval_from_milvus --step extract")
            return
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        documents = chunks_to_documents(chunks)
        asyncio.run(generate_testset_from_documents(
            documents=documents,
            test_size=args.test_size,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=args.embedding_model,
            embed_base_url=args.embed_base_url,
            embed_api_key=embed_api_key,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        ))

    elif args.step == "evaluate":
        print("⚠️ evaluate 步骤需要 RAG 引擎，请通过 Python API 调用 run_full_pipeline")


if __name__ == "__main__":
    main()
