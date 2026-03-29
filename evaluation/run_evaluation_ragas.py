# -*- coding: utf-8 -*-
"""
Ragas 自动化评估主入口
从 Milvus 数据库读取 chunks → 生成测试集 → RAG 评估 → 报告生成

用法:
    # 生成测试集
    python -m evaluation.run_evaluation --step generate --test-size 50

    # 执行评估
    python -m evaluation.run_evaluation --step evaluate --testset-path results/testset.json

    # 完整流程
    python -m evaluation.run_evaluation --step all --test-size 50
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Any

print("DEBUG: 1. imports start", flush=True)

# 确保 evaluation 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# 主入口函数
# ============================================================================

async def run_full_evaluation(
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    llm_api_key: str = "",
    embedding_model: str = "text-embedding-v3",
    embed_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    embed_api_key: str = "",
    embedding_mode: str = "api",
    ollama_base_url: str = "http://localhost:11434",
    ollama_embed_model: str = "bge-m3",
    test_size: int = 50,
    output_dir: str = "results",
    max_concurrent: int = 5,
    plugin_version: str = "1.0.0",
    paper_name: str = "AstrBot Paper RAG",
    query_engine: Optional[Any] = None,
    # Milvus 配置
    milvus_lite_path: str = "./data/milvus_papers.db",
    collection_name: str = "paper_embeddings",
    embed_dim: int = 1024,
) -> dict:
    """
    执行完整评测流程

    Args:
        llm_model: LLM 模型名称
        llm_base_url: LLM API 地址
        llm_api_key: LLM API Key
        embedding_model: Embedding 模型
        embed_base_url: Embedding API 地址
        embed_api_key: Embedding API Key
        test_size: 测试问题数量
        output_dir: 输出目录
        max_concurrent: 最大并发数
        plugin_version: 插件版本
        paper_name: 论文/系统名称
        query_engine: 可选，RAG 查询引擎（不提供则创建默认引擎）
        milvus_lite_path: Milvus Lite 数据库路径
        collection_name: Milvus 集合名称
        embed_dim: Embedding 维度

    Returns:
        包含各步骤结果的字典
    """
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========== 步骤 1: 生成测试集 ==========
    print("\n" + "=" * 60)
    print("📝 步骤 1/3: 自动生成测试集")
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
        milvus_lite_path=milvus_lite_path,
        collection_name=collection_name,
        embed_dim=embed_dim,
        max_chunks=10,
        max_concurrent=max_concurrent,
    )

    documents = generator.load_documents_from_milvus()
    if not documents:
        print("❌ 未从 Milvus 数据库中找到文档")
        return {"success": False, "error": "No documents found in Milvus"}

    testset_path = str(output_path / "testset.json")
    samples = await generator.generate_testset(
        documents=documents,
        test_size=test_size,
        output_path=testset_path,
    )
    results["testset"] = {"path": testset_path, "count": len(samples)}
    print(f"✅ 测试集生成完成: {len(samples)} 个样本")

    # ========== 步骤 2: 执行评估 ==========
    print("\n" + "=" * 60)
    print("📊 步骤 2/3: 执行 Ragas 评估")
    print("=" * 60)

    from .ragas_evaluator import RagasEvaluator

    # 获取或创建查询引擎
    if query_engine is None:
        print("⚠️ 未提供 query_engine，使用默认 llama-index 引擎（仅用于测试）")
        from llama_index.core import VectorStoreIndex, Document
        # 使用已有文档创建简单引擎
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        print("✅ 已创建默认查询引擎")

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

    results_path = str(output_path / "evaluation_results.csv")
    await evaluator.evaluate(
        query_engine=query_engine,
        testset_path=testset_path,
        output_path=results_path,
        max_concurrent=max_concurrent,
    )
    results["evaluation"] = {"path": results_path}
    print("✅ Ragas 评估完成")

    # ========== 步骤 3: 生成报告 ==========
    print("\n" + "=" * 60)
    print("📋 步骤 3/3: 生成评测报告")
    print("=" * 60)

    from .report_generator import ReportGenerator

    reporter = ReportGenerator(results_path)
    html_path = reporter.generate_html_report(
        str(output_path / "evaluation_report.html"),
        plugin_version=plugin_version,
        paper_name=paper_name,
    )
    md_path = reporter.generate_markdown_report(
        str(output_path / "evaluation_report.md"),
        plugin_version=plugin_version,
        paper_name=paper_name,
    )
    results["reports"] = {"html": html_path, "markdown": md_path}
    print("✅ 报告生成完成")

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("✅ 评测完成！")
    print(f"{'='*60}")
    print(f"📁 结果目录: {output_dir}/")
    print(f"   • 测试集: {testset_path}")
    print(f"   • 评估结果: {results_path}")
    print(f"   • HTML 报告: {html_path}")
    print(f"   • Markdown 报告: {md_path}")
    print("=" * 60)

    results["success"] = True
    return results


async def run_generate_only(
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    embedding_model: str,
    embed_base_url: str,
    embed_api_key: str,
    embedding_mode: str,
    ollama_base_url: str,
    ollama_embed_model: str,
    test_size: int,
    output_dir: str,
    milvus_lite_path: str,
    collection_name: str,
    embed_dim: int,
    max_concurrent: int = 3,
) -> dict:
    """仅生成测试集"""
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
        milvus_lite_path=milvus_lite_path,
        collection_name=collection_name,
        embed_dim=embed_dim,
        max_chunks=10,
        max_concurrent=max_concurrent,
    )

    documents = generator.load_documents_from_milvus()
    if not documents:
        return {"success": False, "error": "No documents found in Milvus"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    testset_path = str(output_path / "testset.json")

    samples = await generator.generate_testset(
        documents=documents,
        test_size=test_size,
        output_path=testset_path,
    )

    return {"success": True, "path": testset_path, "count": len(samples)}


async def run_evaluate_only(
    query_engine: Any,
    testset_path: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    embedding_model: str,
    embed_base_url: str,
    embed_api_key: str,
    embedding_mode: str,
    ollama_base_url: str,
    ollama_embed_model: str,
    output_dir: str,
    max_concurrent: int,
) -> dict:
    """仅执行评估"""
    from .ragas_evaluator import RagasEvaluator
    from .report_generator import ReportGenerator

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
        max_concurrent=max_concurrent,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_path = str(output_path / "evaluation_results.csv")

    await evaluator.evaluate(
        query_engine=query_engine,
        testset_path=testset_path,
        output_path=results_path,
        max_concurrent=max_concurrent,
    )

    # 自动生成报告
    reporter = ReportGenerator(results_path)
    html_path = reporter.generate_html_report(str(output_path / "evaluation_report.html"))
    md_path = reporter.generate_markdown_report(str(output_path / "evaluation_report.md"))

    return {"success": True, "results_path": results_path, "html": html_path, "md": md_path}


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    print("DEBUG: 4. main() start", flush=True)

    # 默认值：使用插件 data 目录下的 Milvus 数据库
    default_milvus_path = str(Path(__file__).parent.parent / "data" / "milvus_papers.db")

    parser = argparse.ArgumentParser(
        description="Ragas 自动化评测工具 - 从 Milvus 数据库生成测试集并评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅生成测试集（使用默认数据库路径）
  python -m evaluation.run_evaluation --step generate --test-size 100

  # 指定数据库路径
  python -m evaluation.run_evaluation --step generate --milvus-lite-path ./data/milvus_papers.db --test-size 100

  # 完整流程（生成测试集 + 评估 + 报告）
  python -m evaluation.run_evaluation --step all --test-size 50

  # 仅执行评估（需要已有测试集）
  python -m evaluation.run_evaluation --step evaluate --testset-path results/testset.json

环境变量:
  ZHIPU_API_KEY    智谱 API Key（可替代 --llm-api-key）
        """
    )

    # 步骤参数
    parser.add_argument(
        "--step",
        choices=["all", "generate", "evaluate"],
        default="all",
        help="执行步骤: all=完整流程, generate=仅生成测试集, evaluate=仅执行评估"
    )

    # Milvus 数据库配置
    parser.add_argument(
        "--milvus-lite-path",
        default=default_milvus_path,
        help=f"Milvus Lite 数据库文件路径 (默认: {default_milvus_path})"
    )
    parser.add_argument(
        "--collection-name",
        default="paper_embeddings",
        help="Milvus 集合名称 (默认: paper_embeddings)"
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=1024,
        help="Embedding 维度 (默认: 1024, 与 bge-m3 一致)"
    )

    # 测试集路径
    parser.add_argument("--testset-path", default="results/testset.json", help="测试集文件路径（evaluate 步骤需要）")

    # LLM 配置
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM 模型名称")
    parser.add_argument("--llm-base-url", default="https://open.bigmodel.cn/api/paas/v4", help="LLM API 基础 URL")
    parser.add_argument("--llm-api-key", default="", help="LLM API Key（可使用环境变量）")

    # Embedding 配置
    parser.add_argument("--embedding-model", default="text-embedding-v3", help="Embedding 模型名称")
    parser.add_argument("--embed-base-url", default="https://open.bigmodel.cn/api/paas/v4", help="Embedding API 基础 URL")
    parser.add_argument("--embed-api-key", default="", help="Embedding API Key（可使用环境变量）")
    parser.add_argument(
        "--embedding-mode",
        choices=["api", "ollama"],
        default="ollama",
        help="Embedding 模式: api=使用远程API, ollama=使用本地 ollama (默认: ollama)"
    )
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama API 基础 URL")
    parser.add_argument("--ollama-embed-model", default="bge-m3", help="Ollama Embedding 模型名称")

    # 评测参数
    parser.add_argument("--test-size", type=int, default=50, help="生成测试问题数量")
    parser.add_argument("--max-concurrent", type=int, default=5, help="最大并发数")

    # 输出
    parser.add_argument("--output-dir", default="results", help="输出目录")

    # 报告
    parser.add_argument("--plugin-version", default="1.0.0", help="插件版本")
    parser.add_argument("--paper-name", default="AstrBot Paper RAG", help="论文/系统名称")

    args = parser.parse_args()
    print("DEBUG: 5. args parsed", args, flush=True)

    # 从环境变量获取 API Key
    llm_api_key = args.llm_api_key or os.getenv("ZHIPU_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")
    embed_api_key = args.embed_api_key or os.getenv("ZHIPU_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")

    # 直接设置默认 LLM 配置（使用 freeapi.json）
    freeapi_config_path = Path(__file__).parent / "freeapi.json"
    if not freeapi_config_path.exists():
        raise FileNotFoundError(f"freeapi.json 配置文件不存在: {freeapi_config_path}")
    import json as json_lib
    with open(freeapi_config_path, "r") as f:
        freeapi_config = json_lib.load(f)
    llm_api_key = freeapi_config.get("API_KEY", "")
    # base_url 需要添加 /v1/ 后缀
    llm_base_url = freeapi_config.get("API_URL", "") + "/v1/"
    llm_model = "gpt-4o-mini"
    print(f"✅ 已加载 freeapi.json 配置: {llm_base_url} ({llm_model})")
    embedding_model = "bge-m3"
    embedding_mode = args.embedding_mode
    ollama_base_url = args.ollama_base_url
    ollama_embed_model = args.ollama_embed_model
    milvus_lite_path = args.milvus_lite_path
    collection_name = args.collection_name
    embed_dim = args.embed_dim

    print(f"\n📊 配置信息:")
    print(f"   Milvus 数据库: {milvus_lite_path}")
    print(f"   集合名称: {collection_name}")
    print(f"   Embedding 维度: {embed_dim}")
    print(f"   LLM 模型: {llm_model}")
    print(f"   Embedding 模式: {embedding_mode}")
    if embedding_mode == "ollama":
        print(f"   Ollama 地址: {ollama_base_url}")
        print(f"   Ollama Embed 模型: {ollama_embed_model}")

    if not llm_api_key:
        print("⚠️ 警告: 未提供 API Key，部分功能可能受限")

    # 根据步骤执行
    if args.step == "all":
        asyncio.run(run_full_evaluation(
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=embedding_model,
            embed_base_url=args.embed_base_url,
            embed_api_key=embed_api_key,
            embedding_mode=embedding_mode,
            ollama_base_url=ollama_base_url,
            ollama_embed_model=ollama_embed_model,
            test_size=args.test_size,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
            plugin_version=args.plugin_version,
            paper_name=args.paper_name,
            milvus_lite_path=milvus_lite_path,
            collection_name=collection_name,
            embed_dim=embed_dim,
        ))

    elif args.step == "generate":
        try:
            result = asyncio.run(run_generate_only(
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedding_model=embedding_model,
                embed_base_url=args.embed_base_url,
                embed_api_key=embed_api_key,
                embedding_mode=embedding_mode,
                ollama_base_url=ollama_base_url,
                ollama_embed_model=ollama_embed_model,
                test_size=args.test_size,
                output_dir=args.output_dir,
                milvus_lite_path=milvus_lite_path,
                collection_name=collection_name,
                embed_dim=embed_dim,
                max_concurrent=args.max_concurrent,
            ))
            if result.get("success"):
                print(f"\n✅ 测试集生成完成: {result['path']} ({result['count']} 个样本)")
            else:
                print(f"\n❌ 生成失败: {result.get('error')}")
        except Exception as e:
            print(f"\n❌ 执行出错: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    elif args.step == "evaluate":
        print("⚠️ evaluate 步骤需要提供 query_engine，请通过 Python API 调用")
        print("   from evaluation.run_evaluation import run_evaluate_only")
        print("   await run_evaluate_only(query_engine=your_engine, ...)")


if __name__ == "__main__":
    main()
