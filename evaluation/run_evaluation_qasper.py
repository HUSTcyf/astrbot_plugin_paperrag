#!/usr/bin/env python3
"""
Qasper 数据集评估脚本

支持两种评估模式：
1. 官方 Qasper 评估 - 使用当前 RAG 系统生成 predictions 并完成评估
2. RAGAS 评估 - 从 Milvus Qasper 数据库生成测试集并评估

使用插件的虚拟环境运行:
    ../.venv/bin/python run_evaluation_qasper.py --all

或激活虚拟环境后运行:
    source ../.venv/bin/activate
    python run_evaluation_qasper.py --all

用法:
    # 官方 Qasper 评估
    python run_evaluation_qasper.py --all                         # 生成 predictions 并评估
    python run_evaluation_qasper.py --all --force_english          # 生成英文 predictions 并评估
    python run_evaluation_qasper.py --generate                    # 仅生成 predictions
    python run_evaluation_qasper.py --generate --force_english     # 生成英文 predictions
    python run_evaluation_qasper.py --evaluate                    # 仅运行评估

    # RAGAS 评估（使用 Milvus Qasper 数据库）
    python run_evaluation_qasper.py --ragas --all                 # 完整 RAGAS 评估流程
    python run_evaluation_qasper.py --ragas --generate            # 仅生成测试集
    python run_evaluation_qasper.py --ragas --evaluate            # 仅执行评估

    python run_evaluation_qasper.py --config /path/to/config.json  # 指定配置文件
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加插件目录到路径
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 默认配置路径
# 插件目录: .../data/plugins/astrbot_plugin_paperrag/
# 配置文件: /Users/chenyifeng/AstrBot/data/config/astrbot_plugin_paperrag_config.json
DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent.parent / "config" / "astrbot_plugin_paperrag_config.json"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "cache"
DEFAULT_EVAL_OUTPUT = SCRIPT_DIR / "evaluation_output"

# RAGAS 专用路径
DEFAULT_MILVUS_QASPER_PATH = SCRIPT_DIR / "data" / "milvus_qasper.db"
DEFAULT_QASPER_DOC_STATS_PATH = SCRIPT_DIR / "data" / "qasper_doc_stats.json"
DEFAULT_RAGAS_OUTPUT = SCRIPT_DIR / "results_qasper"


def load_config(config_path: Path) -> dict:
    """加载插件配置"""
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_test_data(test_file: Path) -> dict:
    """加载测试集"""
    if not test_file.exists():
        print(f"❌ 测试集文件不存在: {test_file}")
        print("请确保已将 qasper-test-v0.3.json 放在正确位置")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        return json.load(f)


async def initialize_rag_engine(config: dict, milvus_lite_path: str = ''):
    """初始化 RAG 引擎

    Args:
        config: 插件配置
        milvus_lite_path: 可选的 Milvus 数据库路径覆盖，用于 Qasper 评估
    """
    from rag_engine import RAGConfig, create_rag_engine

    # 使用 Qasper 专用数据库路径（如果提供）
    effective_milvus_path = milvus_lite_path if len(milvus_lite_path) > 0 else config.get("milvus_lite_path", "")

    # 创建 RAG 配置
    rag_config = RAGConfig(
        embedding_mode=config.get("embedding_mode", "ollama"),
        embedding_provider_id=config.get("embedding_provider_id", ""),
        compress_provider_id=config.get("compress_provider_id", ""),
        text_provider_id=config.get("text_provider_id", ""),
        multimodal_provider_id=config.get("multimodal_provider_id", ""),
        llama_vlm_model_path=config.get("llama_vlm_model_path", "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"),
        llama_vlm_mmproj_path=config.get("llama_vlm_mmproj_path", "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"),
        llama_vlm_max_tokens=config.get("llama_vlm_max_tokens", 2560),
        llama_vlm_temperature=config.get("llama_vlm_temperature", 0.7),
        llama_vlm_n_ctx=config.get("llama_vlm_n_ctx", 4096),
        llama_vlm_n_gpu_layers=config.get("llama_vlm_n_gpu_layers", 99),
        ollama_config=config.get("ollama", {}),
        milvus_lite_path=effective_milvus_path,
        address=config.get("address", ""),
        db_name=config.get("db_name", "default"),
        authentication=config.get("authentication", {}),
        collection_name=config.get("collection_name", "paper_embeddings"),
        embed_dim=config.get("embed_dim", 1024),  # Qasper 使用 bge-m3 模型，固定 1024 维
        top_k=config.get("top_k", 5),
        similarity_cutoff=config.get("similarity_cutoff", 0.3),
        papers_dir=config.get("papers_dir", "./papers"),
        chunk_size=config.get("chunk_size", 512),
        chunk_overlap=config.get("chunk_overlap", 0),
        min_chunk_size=config.get("min_chunk_size", 100),
        use_semantic_chunking=config.get("use_semantic_chunking", True),
        enable_multimodal=config.get("multimodal", {}).get("enabled", True),
        figures_dir=config.get("figures_dir", ""),
        enable_reranking=config.get("enable_reranking", False),
        reranking_model=config.get("reranking_model", "BAAI/bge-reranker-v2-m3"),
        reranking_device=config.get("reranking_device", "auto"),
        reranking_adaptive=config.get("reranking_adaptive", True),
        reranking_threshold=config.get("reranking_threshold", 0.0),
        reranking_batch_size=config.get("reranking_batch_size", 32),
        enable_bm25=config.get("enable_bm25", False),
        bm25_top_k=config.get("bm25_top_k", 20),
        hybrid_alpha=config.get("hybrid_alpha", 0.5),
        hybrid_rrf_k=config.get("hybrid_rrf_k", 60),
    )

    # 验证配置
    valid, error_msg = rag_config.validate()
    if not valid:
        print(f"❌ RAG配置无效: {error_msg}")
        sys.exit(1)

    # 创建引擎 (使用 None 作为 context，因为评估脚本不需要 AstrBot 上下文)
    engine = create_rag_engine(rag_config, context=None)

    return engine


def load_existing_predictions(output_file: Path) -> dict:
    """
    加载已存在的预测文件，返回 {question_id: prediction} 的字典

    Args:
        output_file: predictions.jsonl 文件路径

    Returns:
        包含已存在预测的字典，key为question_id
    """
    existing = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pred = json.loads(line.strip())
                    qid = pred.get("question_id", "")
                    if qid:
                        existing[qid] = pred
                except json.JSONDecodeError:
                    continue
    return existing


def backup_predictions(output_file: Path) -> Optional[Path]:
    """
    备份已存在的预测文件

    Args:
        output_file: 预测文件路径

    Returns:
        备份文件路径，如果无需备份则返回None
    """
    if not output_file.exists():
        return None

    import shutil
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = output_file.parent / f"predictions_backup_{timestamp}.jsonl"

    # 只备份有内容的文件
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if len(lines) > 0:
            shutil.copy2(output_file, backup_file)
            return backup_file
    return None


def save_predictions(predictions: dict, output_file: Path):
    """
    保存预测到JSONL文件

    Args:
        predictions: {question_id: prediction} 字典
        output_file: 输出文件路径
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions.values():
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


async def generate_predictions(
    engine,
    test_data: dict,
    output_file: Path,
    batch_size: int = 10,
    delay_between_batches: float = 1.0,
    resume: bool = True,
    force_english: bool = False
) -> int:
    """
    生成预测文件

    Args:
        engine: RAG 引擎实例
        test_data: 测试数据
        output_file: 输出文件路径
        batch_size: 每批处理的问题数
        delay_between_batches: 批次间的延迟（秒）
        resume: 是否启用断点续传（跳过已有答案的问题，默认True）
        force_english: 强制使用英文回答（默认False）

    Returns:
        新生成的预测数量
    """
    print("\n" + "=" * 60)
    print("生成 Predictions")
    print("=" * 60)

    # 断点续传：加载已存在的预测
    existing_predictions = {}
    questions_to_process = []
    total_questions = 0
    backup_file = None

    if resume:
        # 先备份原始文件
        backup_file = backup_predictions(output_file)
        if backup_file:
            print(f"📂 已备份原始文件: {backup_file}")

        existing_predictions = load_existing_predictions(output_file)
        print(f"📂 已加载 {len(existing_predictions)} 条已有预测")

    # 统计问题数并构建待处理列表
    for paper_id, paper_data in test_data.items():
        total_questions += len(paper_data.get("qas", []))

    print(f"总问题数: {total_questions}")

    # 构建待处理问题列表（跳过已有有效答案的）
    for paper_id, paper_data in test_data.items():
        paper_title = paper_data.get("title", "Unknown")
        qas = paper_data.get("qas", [])

        for qa in qas:
            question_id = qa.get("question_id", "")
            question = qa.get("question", "")

            if not question:
                continue

            # 断点续传：跳过已有有效答案的问题
            if resume and question_id in existing_predictions:
                existing_answer = existing_predictions[question_id].get("predicted_answer", "")
                if existing_answer and existing_answer.strip():
                    continue  # 已有有效答案，跳过

            questions_to_process.append({
                "question_id": question_id,
                "question": question,
                "paper_title": paper_title
            })

    print(f"📝 本次需处理问题数: {len(questions_to_process)}")
    if len(questions_to_process) == 0:
        print("✅ 所有问题都已完成，无需处理")
        print(f"   预测文件: {output_file}")
        return 0

    processed = 0
    for qa_info in questions_to_process:
        question_id = qa_info["question_id"]
        question = qa_info["question"]

        # 调用 RAG 引擎获取答案
        evidence = []
        try:
            result = await engine.search(question, mode="rag", force_english=force_english)

            if result.get("type") == "rag":
                answer = result.get("answer", "")
                # 提取 evidence：从检索结果中获取来源文本
                sources = result.get("sources", [])
                for src in sources:
                    evidence.append(src.get("text", ""))
            elif result.get("type") == "error":
                answer = ""
                print(f"  ⚠️  [{question_id}] 错误: {result.get('message', 'Unknown error')}")
            else:
                answer = ""

        except Exception as e:
            answer = ""
            print(f"  ⚠️  [{question_id}] 异常: {e}")

        existing_predictions[question_id] = {
            "question_id": question_id,
            "predicted_answer": answer,
            "predicted_evidence": evidence
        }

        processed += 1

        # 进度显示
        if processed % 50 == 0:
            print(f"  已处理 {processed}/{len(questions_to_process)} 个问题...")

        # 批次延迟，避免请求过快
        if processed % batch_size == 0:
            await asyncio.sleep(delay_between_batches)
            # 定期保存，防止中断丢失数据
            save_predictions(existing_predictions, output_file)

    # 最终保存
    save_predictions(existing_predictions, output_file)

    print(f"\n✅ 本次新生成 {processed} 条预测")
    print(f"   总计已有预测: {len(existing_predictions)} 条")
    print(f"   保存到: {output_file}")

    return processed


def run_evaluator(predictions_file: Path, gold_file: Path, output_dir: Path, text_evidence_only: bool = False):
    """
    运行官方评估脚本

    Args:
        predictions_file: predictions.jsonl 文件路径
        gold_file: gold 标准文件路径 (qasper-test-v0.3.json)
        output_dir: 输出目录
        text_evidence_only: 是否仅使用文本证据
    """
    print("\n" + "=" * 60)
    print("运行评估")
    print("=" * 60)

    if not predictions_file.exists():
        print(f"❌ Predictions 文件不存在: {predictions_file}")
        print("请先运行: python run_evaluation.py --generate")
        sys.exit(1)

    if not gold_file.exists():
        print(f"❌ Gold 文件不存在: {gold_file}")
        print("请先运行: python qasper_downloader.py")
        sys.exit(1)

    # 使用官方评估脚本
    evaluator_script = SCRIPT_DIR / "datasets" / "qasper_evaluator.py"
    if not evaluator_script.exists():
        print(f"❌ 评估脚本不存在: {evaluator_script}")
        sys.exit(1)

    # 构建命令
    cmd = [
        sys.executable,
        str(evaluator_script),
        "--predictions", str(predictions_file),
        "--gold", str(gold_file),
    ]

    if text_evidence_only:
        cmd.append("--text_evidence_only")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "evaluation_results.json"

    print(f"\n运行评估命令:")
    print(f"  {' '.join(cmd)}")
    print(f"\n评估结果将保存到: {output_file}")

    # 执行评估
    import subprocess
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print("\n评估结果:")
        print(result.stdout)

        # 保存结果
        try:
            eval_result = json.loads(result.stdout)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 评估结果已保存到: {output_file}")
        except json.JSONDecodeError:
            print(f"\n⚠️ 无法解析评估结果为 JSON")

    if result.stderr:
        print("\n错误信息:")
        print(result.stderr)

    return result.returncode == 0


# ============================================================================
# RAGAS 评估函数（Qasper 版本）
# ============================================================================

async def run_ragas_full_evaluation(
    config: dict,
    test_size: int = 50,
    output_dir: str = '',
    max_concurrent: int = 5,
) -> dict:
    """
    执行 RAGAS 完整评测流程（Qasper 版本）

    Args:
        config: 插件配置
        test_size: 测试问题数量
        output_dir: 输出目录
        max_concurrent: 最大并发数

    Returns:
        包含各步骤结果的字典
    """
    from ragas_generator import RagasTestsetGenerator
    from ragas_evaluator import RagasEvaluator
    from report_generator import ReportGenerator

    results = {}

    # RAGAS 专用路径
    milvus_qasper_path = str(DEFAULT_MILVUS_QASPER_PATH)
    qasper_stats_path = str(DEFAULT_QASPER_DOC_STATS_PATH)

    if len(output_dir) == 0:
        output_dir = str(DEFAULT_RAGAS_OUTPUT)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 从插件配置读取 freeapi 设置
    llm_api_key = config.get("freeapi_key", "")
    llm_base_url = config.get("freeapi_url", "") + "/v1/"
    llm_model = "gpt-4o-mini"
    embedding_model = "bge-m3"

    print(f"\n📊 RAGAS Qasper 配置信息:")
    print(f"   Milvus Qasper 数据库: {milvus_qasper_path}")
    print(f"   Qasper 统计文件: {qasper_stats_path}")
    print(f"   LLM 模型: {llm_model}")
    if not llm_api_key:
        print(f"   ⚠️ 警告: freeapi_key 未配置，RAGAS 评估可能失败")

    # ========== 步骤 1: 生成测试集 ==========
    print("\n" + "=" * 60)
    print("📝 步骤 1/3: 自动生成测试集 (Qasper)")
    print("=" * 60)

    generator = RagasTestsetGenerator(
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=llm_base_url,
        embed_api_key=llm_api_key,
        embedding_mode="ollama",
        ollama_base_url=config.get("ollama", {}).get("base_url", "http://localhost:11434"),
        ollama_embed_model=config.get("ollama", {}).get("model", "bge-m3"),
        milvus_lite_path=milvus_qasper_path,
        collection_name=config.get("collection_name", "paper_embeddings"),
        embed_dim=config.get("embed_dim", 1024),
        max_chunks=10,
        max_concurrent=max_concurrent,
        paper_doc_stats_path=qasper_stats_path,
    )

    documents = generator.load_documents_from_milvus()
    if not documents:
        print("❌ 未从 Milvus Qasper 数据库中找到文档")
        return {"success": False, "error": "No documents found in Milvus Qasper"}

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

    # 使用已有文档创建简单引擎
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    print("✅ 已创建默认查询引擎")

    evaluator = RagasEvaluator(
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=llm_base_url,
        embed_api_key=llm_api_key,
        embedding_mode="ollama",
        ollama_base_url=config.get("ollama", {}).get("base_url", "http://localhost:11434"),
        ollama_embed_model=config.get("ollama", {}).get("model", "bge-m3"),
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

    reporter = ReportGenerator(results_path)
    html_path = reporter.generate_html_report(
        str(output_path / "evaluation_report.html"),
        plugin_version="1.0.0",
        paper_name="Qasper Paper RAG",
    )
    md_path = reporter.generate_markdown_report(
        str(output_path / "evaluation_report.md"),
        plugin_version="1.0.0",
        paper_name="Qasper Paper RAG",
    )
    results["reports"] = {"html": html_path, "markdown": md_path}
    print("✅ 报告生成完成")

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("✅ Qasper RAGAS 评测完成！")
    print(f"{'='*60}")
    print(f"📁 结果目录: {output_dir}/")
    print(f"   • 测试集: {testset_path}")
    print(f"   • 评估结果: {results_path}")
    print(f"   • HTML 报告: {html_path}")
    print(f"   • Markdown 报告: {md_path}")
    print("=" * 60)

    results["success"] = True
    return results


async def run_ragas_generate_only(
    config: dict,
    test_size: int = 50,
    output_dir: str = '',
    max_concurrent: int = 3,
) -> dict:
    """仅生成测试集（RAGAS Qasper 版本）"""
    from ragas_generator import RagasTestsetGenerator

    milvus_qasper_path = str(DEFAULT_MILVUS_QASPER_PATH)
    qasper_stats_path = str(DEFAULT_QASPER_DOC_STATS_PATH)

    if len(output_dir) == 0:
        output_dir = str(DEFAULT_RAGAS_OUTPUT)

    # 从插件配置读取 freeapi 设置
    llm_api_key = config.get("freeapi_key", "")
    llm_base_url = config.get("freeapi_url", "") + "/v1/"
    llm_model = "gpt-4o-mini"
    embedding_model = "bge-m3"

    generator = RagasTestsetGenerator(
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        embedding_model=embedding_model,
        embed_base_url=llm_base_url,
        embed_api_key=llm_api_key,
        embedding_mode="ollama",
        ollama_base_url=config.get("ollama", {}).get("base_url", "http://localhost:11434"),
        ollama_embed_model=config.get("ollama", {}).get("model", "bge-m3"),
        milvus_lite_path=milvus_qasper_path,
        collection_name=config.get("collection_name", "paper_embeddings"),
        embed_dim=config.get("embed_dim", 1024),
        max_chunks=10,
        max_concurrent=max_concurrent,
        paper_doc_stats_path=qasper_stats_path,
    )

    documents = generator.load_documents_from_milvus()
    if not documents:
        return {"success": False, "error": "No documents found in Milvus Qasper"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    testset_path = str(output_path / "testset.json")

    samples = await generator.generate_testset(
        documents=documents,
        test_size=test_size,
        output_path=testset_path,
    )

    return {"success": True, "path": testset_path, "count": len(samples)}


async def main_async(args):
    """异步主函数"""
    # 解析路径
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = DEFAULT_CONFIG_PATH

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    output_dir = Path(args.output) if args.output else DEFAULT_EVAL_OUTPUT

    # 加载配置
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)

    # ========== RAGAS 评估模式 ==========
    if args.ragas:
        print("\n" + "=" * 60)
        print("🎯 RAGAS 评估模式 (Qasper)")
        print("=" * 60)

        ragas_output_dir = args.ragas_output or str(DEFAULT_RAGAS_OUTPUT)

        if args.ragas_generate or args.ragas_all:
            result = await run_ragas_generate_only(
                config=config,
                test_size=args.test_size,
                output_dir=ragas_output_dir,
                max_concurrent=args.max_concurrent,
            )
            if result.get("success"):
                print(f"\n✅ 测试集生成完成: {result['path']} ({result['count']} 个样本)")
            else:
                print(f"\n❌ 生成失败: {result.get('error')}")
            return

        if args.ragas_evaluate or args.ragas_all:
            print("⚠️ RAGAS evaluate 步骤需要提供 query_engine，请通过 Python API 调用")
            print("   from evaluation.run_evaluation_qasper import run_ragas_full_evaluation")
            print("   await run_ragas_full_evaluation(config=config, test_size=50, output_dir='results_qasper')")
            return

        if not args.ragas_generate and not args.ragas_evaluate and not args.ragas_all:
            # 交互式 RAGAS 模式
            print("\n请选择 RAGAS 操作:")
            print("  1. 生成测试集")
            print("  2. 完整流程 (生成 + 评估 + 报告)")
            print("  0. 退出")

            choice = input("\n请输入选项 (0-2): ").strip()

            if choice == "1":
                result = await run_ragas_generate_only(
                    config=config,
                    test_size=args.test_size,
                    output_dir=ragas_output_dir,
                    max_concurrent=args.max_concurrent,
                )
                if result.get("success"):
                    print(f"\n✅ 测试集生成完成: {result['path']} ({result['count']} 个样本)")
                else:
                    print(f"\n❌ 生成失败: {result.get('error')}")
            elif choice == "2":
                await run_ragas_full_evaluation(
                    config=config,
                    test_size=args.test_size,
                    output_dir=ragas_output_dir,
                    max_concurrent=args.max_concurrent,
                )
            else:
                print("已退出")
            return

    # ========== 官方 Qasper 评估模式 ==========
    # 使用 Qasper 专用 Milvus 数据库
    milvus_qasper_path = str(DEFAULT_MILVUS_QASPER_PATH)

    # 加载测试数据
    test_data_path = SCRIPT_DIR / "datasets" / "qasper-test-v0.3.json"
    print(f"加载测试数据: {test_data_path}")
    test_data = load_test_data(test_data_path)

    # 统计
    total_papers = len(test_data)
    total_questions = sum(len(p.get("qas", [])) for p in test_data.values())
    print(f"\n数据集统计:")
    print(f"  论文数: {total_papers}")
    print(f"  问题数: {total_questions}")
    print(f"  Milvus Qasper 数据库: {milvus_qasper_path}")

    predictions_file = output_dir / "predictions.jsonl"
    gold_file = SCRIPT_DIR / "datasets" / "qasper-test-v0.3.json"

    if args.generate or args.all:
        # 初始化 RAG 引擎（使用 Qasper 专用数据库）
        print("\n初始化 RAG 引擎...")
        print("=" * 60)

        engine = await initialize_rag_engine(config, milvus_lite_path=milvus_qasper_path)

        # 生成预测
        await generate_predictions(
            engine,
            test_data,
            predictions_file,
            batch_size=args.batch_size,
            delay_between_batches=args.delay,
            resume=not args.no_resume,
            force_english=args.force_english
        )

    if args.evaluate or args.all:
        # 运行评估
        run_evaluator(predictions_file, gold_file, output_dir, args.text_evidence_only)

    if not args.generate and not args.evaluate and not args.all:
        # 交互式模式
        print("\n请选择操作:")
        print("  1. 生成 predictions")
        print("  2. 运行评估")
        print("  3. 全部执行 (生成 + 评估)")
        print("  0. 退出")

        choice = input("\n请输入选项 (0-3): ").strip()

        if choice == "1":
            engine = await initialize_rag_engine(config, milvus_lite_path=milvus_qasper_path)
            await generate_predictions(engine, test_data, predictions_file, resume=True, force_english=args.force_english)
        elif choice == "2":
            run_evaluator(predictions_file, gold_file, output_dir, args.text_evidence_only)
        elif choice == "3":
            engine = await initialize_rag_engine(config, milvus_lite_path=milvus_qasper_path)
            await generate_predictions(engine, test_data, predictions_file, resume=True, force_english=args.force_english)
            run_evaluator(predictions_file, gold_file, output_dir, args.text_evidence_only)
        else:
            print("已退出")


def main():
    parser = argparse.ArgumentParser(
        description="Qasper 数据集评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 官方 Qasper 评估
  python run_evaluation_qasper.py --all                         # 生成 predictions 并评估（支持断点续传）
  python run_evaluation_qasper.py --all --force_english         # 生成英文 predictions 并评估
  python run_evaluation_qasper.py --generate                    # 仅生成 predictions（支持断点续传）
  python run_evaluation_qasper.py --generate --force_english     # 生成英文 predictions
  python run_evaluation_qasper.py --generate --no_resume        # 禁用断点续传，重新生成所有预测
  python run_evaluation_qasper.py --evaluate                    # 仅运行评估

  # RAGAS 评估（使用 Milvus Qasper 数据库）
  python run_evaluation_qasper.py --ragas --all                 # 完整 RAGAS 评估流程
  python run_evaluation_qasper.py --ragas --generate            # 仅生成测试集
  python run_evaluation_qasper.py --ragas --evaluate            # 仅执行评估

  python run_evaluation_qasper.py --config /path/to/config.json  # 指定配置文件
  python run_evaluation_qasper.py --data_dir /path/to/data    # 指定数据目录
  python run_evaluation_qasper.py --output /path/to/output    # 指定输出目录
        """
    )

    # 官方 Qasper 评估参数
    parser.add_argument("--generate", action="store_true", help="仅生成 predictions (官方评估模式)")
    parser.add_argument("--evaluate", action="store_true", help="仅运行评估 (官方评估模式)")
    parser.add_argument("--all", action="store_true", help="生成 predictions 并运行评估 (官方评估模式)")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--output", type=str, default=None, help="输出目录路径 (官方评估模式)")
    parser.add_argument("--batch_size", type=int, default=10, help="每批处理的问题数 (默认: 10)")
    parser.add_argument("--delay", type=float, default=1.0, help="批次间延迟秒数 (默认: 1.0)")
    parser.add_argument("--text_evidence_only", action="store_true",
                        help="仅使用文本证据 (忽略表格/图片证据)")
    parser.add_argument("--no_resume", action="store_true",
                        help="禁用断点续传，重新生成所有预测（默认启用断点续传）")
    parser.add_argument("--force_english", action="store_true",
                        help="强制使用英文回答（用于英文数据集评估，默认关闭）")

    # RAGAS 评估参数
    parser.add_argument("--ragas", action="store_true", help="使用 RAGAS 评估模式 (Qasper)")
    parser.add_argument("--ragas-generate", action="store_true", help="仅生成测试集 (RAGAS 模式)")
    parser.add_argument("--ragas-evaluate", action="store_true", help="仅执行评估 (RAGAS 模式，需要 query_engine)")
    parser.add_argument("--ragas-all", action="store_true", help="完整 RAGAS 评估流程")
    parser.add_argument("--ragas-output", type=str, default=None, help="RAGAS 输出目录 (默认: results_qasper)")
    parser.add_argument("--test-size", type=int, default=50, help="RAGAS 生成测试问题数量 (默认: 50)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="RAGAS 最大并发数 (默认: 5)")

    args = parser.parse_args()

    # 运行异步主函数
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
