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
    python run_evaluation_qasper.py --generate                    # 仅生成 predictions
    python run_evaluation_qasper.py --generate --limit 10         # 仅处理前10个问题（快速评估）
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


def extract_evidence_spans(sources: List[Dict[str, Any]], query: str, answer: str = "", max_chars: int = 500) -> List[str]:
    """
    方法2：从检索到的chunks中找到与答案最匹配的精确文本作为evidence

    思路：
    1. 先解析答案，提取关键信息（数字、名词短语等）
    2. 在chunks中精确匹配这些关键信息
    3. 返回包含答案的精确文本片段（而非完整chunk）

    Args:
        sources: 检索到的源文档列表
        query: 查询文本
        answer: 生成的答案
        max_chars: 最大总字符数

    Returns:
        精确的evidence spans列表
    """
    import re

    if not answer or not sources:
        # 如果没有答案，返回完整chunk（fallback）
        return [src.get("text", "")[:500] for src in sources[:3]]

    # 1. 从答案中提取关键信息
    answer_lower = answer.lower()

    # 提取数字（对于factual问题最重要）
    answer_numbers = re.findall(r'[\d,.]+', answer)

    # 提取名词短语（4个字母以上的词）
    answer_nouns = set(re.findall(r'\b[a-z]{4,}\b', answer_lower))

    # 提取被引用的术语（如 TABREF, FIGREF 等）
    answer_refs = re.findall(r'(TABREF|FIGREF|REF)\d+', answer, re.IGNORECASE)

    evidence_spans = []

    for src in sources:
        text = src.get("text", "")
        if not text:
            continue

        # 2. 分割句子
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 3. 对每个句子评分，找出包含答案的句子
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()

            # 计算匹配分数
            score = 0

            # 数字匹配（权重最高）
            for num in answer_numbers:
                if num in sent_lower:
                    score += 10
                    # 检查数字周围的上下文
                    num_idx = sent_lower.find(num)
                    context_start = max(0, num_idx - 30)
                    context_end = min(len(sent_lower), num_idx + len(num) + 30)
                    # 如果数字在有意义的上下文中（不是页码等）
                    context = sent_lower[context_start:context_end]
                    if any(w in context for w in ['=', ':', '%', 'accuracy', 'precision', 'recall', 'f1', 'score', 'result', 'dataset', 'model']):
                        score += 5

            # 名词匹配
            for noun in answer_nouns:
                if noun in sent_lower:
                    score += 2

            # 引用匹配
            for ref in answer_refs:
                if ref.lower() in sent_lower.lower():
                    score += 8

            # 答案完整出现在句子中（最高权重）
            if answer_lower[:50] in sent_lower:
                score += 20

            if score > 0:
                scored_sentences.append((score, sent))

        # 按分数排序
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # 4. 选取高分句子直到达到max_chars
        current_len = sum(len(e) for e in evidence_spans)
        for score, sent in scored_sentences:
            if current_len + len(sent) <= max_chars:
                evidence_spans.append(sent)
                current_len += len(sent)
            if current_len >= max_chars:
                break

    # 5. 如果没有找到匹配的句子，fallback到包含关键词的句子
    if not evidence_spans:
        for src in sources[:3]:
            text = src.get("text", "")
            if text:
                # 找第一句包含答案关键词的
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sent in sentences:
                    sent_lower = sent.lower()
                    if any(noun in sent_lower for noun in list(answer_nouns)[:5]):
                        evidence_spans.append(sent)
                        break
                if evidence_spans:
                    break

    # 6. 去重并限制数量
    seen = set()
    unique_spans = []
    for span in evidence_spans:
        # 用前50字符作为key去重
        span_key = re.sub(r'\s+', ' ', span[:50]).strip()
        if span_key and span_key not in seen:
            seen.add(span_key)
            unique_spans.append(span)

    return unique_spans[:5]


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
        papers_dir="./papers",  # Qasper 评估时 BM25 已禁用，此参数不影响
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
        enable_bm25=True,  # Qasper 评估启用 BM25 混合检索
        bm25_top_k=config.get("bm25_top_k", 50),
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
    limit: int = 0
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
        limit: 限制处理的问题数，0表示不限制（默认0）

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

            # 如果设置了 limit，达到数量后停止构建列表
            if limit > 0 and len(questions_to_process) >= limit:
                break

        if limit > 0 and len(questions_to_process) >= limit:
            break

    print(f"📝 本次需处理问题数: {len(questions_to_process)}" + (f" (已限制前 {limit} 个)" if limit > 0 else ""))
    if len(questions_to_process) == 0:
        print("✅ 所有问题都已完成，无需处理")
        print(f"   预测文件: {output_file}")
        return 0

    processed = 0
    for qa_info in questions_to_process:
        question_id = qa_info["question_id"]
        question = qa_info["question"]

        # 调用 RAG 引擎获取答案（使用Qasper专用方法生成简短答案）
        evidence = []
        try:
            result = await engine.search_for_qasper(question, mode="rag")

            if result.get("type") == "rag":
                answer = result.get("answer", "")
                unanswerable = result.get("unanswerable", False)
                # 提取 evidence：从检索结果中提取精确的evidence spans
                sources = result.get("sources", [])
                if not unanswerable:
                    evidence = extract_evidence_spans(sources, question, answer, max_chars=500)
            elif result.get("type") == "error":
                answer = ""
                unanswerable = False
                print(f"  ⚠️  [{question_id}] 错误: {result.get('message', 'Unknown error')}")
            else:
                answer = ""
                unanswerable = False

        except Exception as e:
            answer = ""
            unanswerable = False
            print(f"  ⚠️  [{question_id}] 异常: {e}")

        existing_predictions[question_id] = {
            "question_id": question_id,
            "predicted_answer": answer,
            "predicted_evidence": evidence,
            "unanswerable": unanswerable
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


async def generate_predictions_llm_only(
    engine,
    test_data: dict,
    output_file: Path,
    batch_size: int = 10,
    delay_between_batches: float = 1.0,
    resume: bool = True,
    limit: int = 0
) -> int:
    """
    生成纯LLM基线预测（不进行检索，直接使用LLM回答）

    Args:
        engine: RAG 引擎实例
        test_data: 测试数据
        output_file: 输出文件路径
        batch_size: 每批处理的问题数
        delay_between_batches: 批次间的延迟（秒）
        resume: 是否启用断点续传
        limit: 限制处理的问题数，0表示不限制

    Returns:
        新生成的预测数量
    """
    print("\n" + "=" * 60)
    print("生成 Predictions (纯LLM基线 - 无检索)")
    print("=" * 60)

    # 断点续传：加载已存在的预测
    existing_predictions = {}
    questions_to_process = []
    total_questions = 0
    backup_file = None

    if resume:
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
                    continue

            questions_to_process.append({
                "question_id": question_id,
                "question": question,
                "paper_title": paper_title,
                "paper_id": paper_id,
                "figures_and_tables": paper_data.get("figures_and_tables", [])
            })

            if limit > 0 and len(questions_to_process) >= limit:
                break

        if limit > 0 and len(questions_to_process) >= limit:
            break

    print(f"📝 本次需处理问题数: {len(questions_to_process)}" + (f" (已限制前 {limit} 个)" if limit > 0 else ""))
    if len(questions_to_process) == 0:
        print("✅ 所有问题都已完成，无需处理")
        print(f"   预测文件: {output_file}")
        return 0

    # 视觉关键词（用于检测是否需要VLM模式，与hybrid_rag.py保持一致）
    VISUAL_KEYWORDS = [
        "图", "figure", "chart", "plot", "graph",
        "表格", "table", "公式", "formula", "equation",
        "架构", "architecture", "diagram", "示意图",
        "图像", "picture", "photo", "image",
        "多少", "数值", "数据", "哪个大", "哪个小",
        "颜色", "曲线", "峰值", "趋势"
    ]

    # Qasper图片目录
    figures_base_dir = SCRIPT_DIR / "datasets" / "test_figures_and_tables"

    def question_has_visual_keyword(query: str) -> bool:
        """检测问题是否包含视觉关键词"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in VISUAL_KEYWORDS)

    def build_paper_image_paths(paper_id: str, figures_and_tables: list) -> list:
        """根据论文ID和图表列表构建完整的图片路径"""
        image_paths = []
        for fig in figures_and_tables:
            if isinstance(fig, dict) and "file" in fig:
                fig_path = figures_base_dir / paper_id / fig["file"]
                if fig_path.exists():
                    image_paths.append(str(fig_path.resolve()))
        return image_paths

    processed = 0
    for qa_info in questions_to_process:
        question_id = qa_info["question_id"]
        question = qa_info["question"]
        paper_id = qa_info.get("paper_id", "")
        figures_and_tables = qa_info.get("figures_and_tables", [])

        # VLM路由：若问题涉及视觉内容且论文有图片，则使用VLM模式
        images = None
        if question_has_visual_keyword(question) and figures_and_tables:
            images = build_paper_image_paths(paper_id, figures_and_tables)
            if images:
                print(f"  🖼️  [{question_id}] VLM模式: 加载{len(images)}张图片")

        # 调用 RAG 引擎获取答案（纯LLM模式，无检索）
        evidence = []
        try:
            result = await engine.search_for_qasper_llm_only(question, images=images)

            if result.get("type") == "llm_only":
                answer = result.get("answer", "")
                # 纯LLM模式没有evidence
                sources = result.get("sources", [])
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
            "predicted_evidence": evidence,
            "unanswerable": False  # 纯LLM模式不判断不可回答
        }

        processed += 1

        if processed % 50 == 0:
            print(f"  已处理 {processed}/{len(questions_to_process)} 个问题...")

        if processed % batch_size == 0:
            await asyncio.sleep(delay_between_batches)
            save_predictions(existing_predictions, output_file)

    save_predictions(existing_predictions, output_file)

    print(f"\n✅ 本次新生成 {processed} 条预测")
    print(f"   总计已有预测: {len(existing_predictions)} 条")
    print(f"   保存到: {output_file}")

    return processed


def run_evaluator(predictions_file: Path, gold_file: Path, output_dir: Path, text_evidence_only: bool = False, use_bert_f1: bool = False):
    """
    运行官方评估脚本

    Args:
        predictions_file: predictions.jsonl 文件路径
        gold_file: gold 标准文件路径 (qasper-test-v0.3.json)
        output_dir: 输出目录
        text_evidence_only: 是否仅使用文本证据
        use_bert_f1: 是否使用BERTScore F1评估（语义评估，更适合QASPER长文档）
    """
    print("\n" + "=" * 60)
    print("运行评估")
    if use_bert_f1:
        print("  [BERTScore F1 模式 - 语义评估]")
    else:
        print("  [Token F1 模式 - 词汇重叠评估]")
    print("=" * 60)

    if not predictions_file.exists():
        print(f"❌ Predictions 文件不存在: {predictions_file}")
        print("请先运行: python run_evaluation.py --generate")
        sys.exit(1)

    if not gold_file.exists():
        print(f"❌ Gold 文件不存在: {gold_file}")
        print("请先运行: python qasper_downloader.py")
        sys.exit(1)

    # 加载预测文件
    predictions = {}
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pred = json.loads(line)
                qid = pred.get("question_id", "")
                if qid:
                    predictions[qid] = pred

    num_predictions = len(predictions)
    print(f"📊 预测数量: {num_predictions}")

    # 使用官方评估脚本
    evaluator_script = SCRIPT_DIR / "datasets" / "qasper_evaluator.py"
    if not evaluator_script.exists():
        print(f"❌ 评估脚本不存在: {evaluator_script}")
        sys.exit(1)

    # 检查是否是子集评估（预测数量 < gold总数）
    # 如果是，创建临时子集 gold 文件
    import tempfile
    gold_data = None
    temp_gold_file = None
    predicted_question_ids = set(predictions.keys())

    with open(gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    total_gold_questions = sum(len(paper["qas"]) for paper in gold_data.values())
    is_subset_evaluation = num_predictions < total_gold_questions

    if is_subset_evaluation:
        print(f"⚠️ 检测到子集评估: {num_predictions} < {total_gold_questions} (总数)")
        print(f"   将只评估已有预测的问题，missing predictions 将为 0")

        # 创建子集 gold 数据
        gold_subset = {}
        for paper_id, paper_data in gold_data.items():
            subset_qas = []
            for qa in paper_data.get("qas", []):
                if qa.get("question_id") in predicted_question_ids:
                    subset_qas.append(qa)
            if subset_qas:
                subset_paper = dict(paper_data)
                subset_paper["qas"] = subset_qas
                gold_subset[paper_id] = subset_paper

        # 保存临时文件
        temp_gold_file = output_dir / "gold_subset.json"
        with open(temp_gold_file, "w", encoding="utf-8") as f:
            json.dump(gold_subset, f, ensure_ascii=False)
        print(f"   临时子集 gold 文件: {temp_gold_file}")

        # 使用子集 gold 文件进行评估
        effective_gold_file = temp_gold_file
    else:
        effective_gold_file = gold_file

    # 构建命令
    cmd = [
        sys.executable,
        str(evaluator_script),
        "--predictions", str(predictions_file),
        "--gold", str(effective_gold_file),
    ]

    if text_evidence_only:
        cmd.append("--text_evidence_only")

    if use_bert_f1:
        cmd.append("--bert-score")

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

    # 清理临时文件
    if temp_gold_file and temp_gold_file.exists():
        temp_gold_file.unlink()
        print(f"\n🗑️ 已清理临时文件: {temp_gold_file}")

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

    # RAGAS 专用路径（使用默认路径，RAGAS 模式不支持通过命令行选择数据库）
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
    milvus_qasper_path = args.milvus_qasper_path if args.milvus_qasper_path else str(DEFAULT_MILVUS_QASPER_PATH)

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

    # 确定预测文件路径
    if args.predictions_file:
        predictions_file = Path(args.predictions_file)
    else:
        predictions_file = output_dir / "predictions.jsonl"
    gold_file = SCRIPT_DIR / "datasets" / "qasper-test-v0.3.json"

    if args.generate or args.all:
        # 初始化 RAG 引擎（使用 Qasper 专用数据库）
        print("\n初始化 RAG 引擎...")
        print("=" * 60)

        engine = await initialize_rag_engine(config, milvus_lite_path=milvus_qasper_path)

        # 生成预测
        if args.llm_only:
            # 纯LLM基线模式（无检索）
            predictions_file = output_dir / "predictions_llm_only.jsonl"
            print("⚠️  使用纯LLM基线模式（无检索）")
            await generate_predictions_llm_only(
                engine,
                test_data,
                predictions_file,
                batch_size=args.batch_size,
                delay_between_batches=args.delay,
                resume=not args.no_resume,
                limit=args.limit
            )
        else:
            await generate_predictions(
                engine,
                test_data,
                predictions_file,
                batch_size=args.batch_size,
                delay_between_batches=args.delay,
                resume=not args.no_resume,
                limit=args.limit
            )

    if args.evaluate or args.all:
        # 运行评估
        run_evaluator(predictions_file, gold_file, output_dir, args.text_evidence_only, args.bert_score)

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
            await generate_predictions(engine, test_data, predictions_file, resume=True)
        elif choice == "2":
            run_evaluator(predictions_file, gold_file, output_dir, args.text_evidence_only, args.bert_score)
        elif choice == "3":
            engine = await initialize_rag_engine(config, milvus_lite_path=milvus_qasper_path)
            await generate_predictions(engine, test_data, predictions_file, resume=True)
            run_evaluator(predictions_file, gold_file, output_dir, args.text_evidence_only, args.bert_score)
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
  python run_evaluation_qasper.py --generate                    # 仅生成 predictions（支持断点续传）
  python run_evaluation_qasper.py --generate --no_resume        # 禁用断点续传，重新生成所有预测
  python run_evaluation_qasper.py --generate --limit 10         # 仅处理前10个问题（快速评估）
  python run_evaluation_qasper.py --all --limit 20              # 评估前20个问题
  python run_evaluation_qasper.py --evaluate                    # 仅运行评估

  # RAGAS 评估（使用 Milvus Qasper 数据库）
  python run_evaluation_qasper.py --ragas --all                 # 完整 RAGAS 评估流程
  python run_evaluation_qasper.py --ragas --generate            # 仅生成测试集
  python run_evaluation_qasper.py --ragas --evaluate            # 仅执行评估

  python run_evaluation_qasper.py --config /path/to/config.json  # 指定配置文件
  python run_evaluation_qasper.py --data_dir /path/to/data    # 指定数据目录
  python run_evaluation_qasper.py --output /path/to/output    # 指定输出目录
  python run_evaluation_qasper.py --milvus-qasper-path ./data/milvus_qasper_vision.db  # 指定 Milvus 数据库（vision 模式）
  python run_evaluation_qasper.py --milvus-qasper-path ./data/milvus_qasper_text.db    # 指定 Milvus 数据库（text-only 模式）
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
    parser.add_argument("--bert_score", action="store_true",
                        help="使用BERTScore F1进行评估（语义评估，更适合QASPER长文档自由形式答案）")
    parser.add_argument("--no_resume", action="store_true",
                        help="禁用断点续传，重新生成所有预测（默认启用断点续传）")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理的问题数量，0表示不限制（默认0，用于快速评估）")
    parser.add_argument("--llm_only", action="store_true",
                        help="纯LLM基线模式：不进行检索，直接使用LLM回答（用于基线对比）")
    parser.add_argument("--predictions_file", type=str, default=None,
                        help="指定预测文件路径（用于评估时选择不同基线的预测结果）")

    # RAGAS 评估参数
    parser.add_argument("--ragas", action="store_true", help="使用 RAGAS 评估模式 (Qasper)")
    parser.add_argument("--ragas-generate", action="store_true", help="仅生成测试集 (RAGAS 模式)")
    parser.add_argument("--ragas-evaluate", action="store_true", help="仅执行评估 (RAGAS 模式，需要 query_engine)")
    parser.add_argument("--ragas-all", action="store_true", help="完整 RAGAS 评估流程")
    parser.add_argument("--ragas-output", type=str, default=None, help="RAGAS 输出目录 (默认: results_qasper)")
    parser.add_argument("--test-size", type=int, default=50, help="RAGAS 生成测试问题数量 (默认: 50)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="RAGAS 最大并发数 (默认: 5)")
    parser.add_argument("--milvus-qasper-path", type=str, default=None,
                        help="Qasper 专用 Milvus 数据库路径 (默认: milvus_qasper.db)")

    args = parser.parse_args()

    # 运行异步主函数
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
