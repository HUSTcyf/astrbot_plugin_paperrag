# -*- coding: utf-8 -*-
"""
基于 Ragas 的 RAG 评估器
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, cast
from dataclasses import dataclass, asdict

import pandas as pd
from ragas import evaluate, RunConfig, EvaluationDataset
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    AnswerCorrectness,
)
from datasets import Dataset

# 禁用 Ragas 遥测追踪（避免 SSL 证书过期错误）
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "True"

from astrbot.api import logger

# 导入自定义 Ragas 兼容包装器（与 ragas_generator 共用）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragas_generator import OpenAICompatibleLLM, OpenAICompatibleEmbeddings, EvalSample


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class EvaluationResult:
    """评估结果"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str

    # Ragas 指标
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_relevancy: float
    answer_correctness: float

    # 元数据
    latency_ms: float
    question_type: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        return cls(**data)


# ============================================================================
# RAG 查询引擎包装器（兼容 HybridRAGEngine）
# ============================================================================

class RAGQueryWrapper:
    """
    RAG 查询引擎包装器
    统一 HybridRAGEngine 和标准 llama-index QueryEngine 的接口
    """

    def __init__(self, query_engine: Any):
        """
        Args:
            query_engine: HybridRAGEngine 实例或 llama-index QueryEngine 实例
        """
        self._engine = query_engine

        # 检测引擎类型
        self._is_hybrid = hasattr(query_engine, "search")
        self._is_llama = hasattr(query_engine, "aquery") or hasattr(query_engine, "_query")

        logger.info(f"RAG 引擎类型: {'HybridRAGEngine' if self._is_hybrid else 'llama-index QueryEngine'}")

    async def aquery(self, query: str) -> Dict[str, Any]:
        """
        执行异步查询

        Returns:
            包含 response 和 source_nodes 的字典
        """
        if self._is_hybrid:
            # HybridRAGEngine
            result = await self._engine.search(query, mode="rag")
            if result.get("type") == "error":
                raise ValueError(result.get("message", "Unknown error"))
            return result
        else:
            # llama-index QueryEngine
            if hasattr(self._engine, "aquery"):
                response = await self._engine.aquery(query)
            else:
                response = self._engine.query(query)

            # 转换为统一格式
            source_nodes = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    source_nodes.append(node)
            elif hasattr(response, "nodes"):
                source_nodes = response.nodes

            return {
                "response": getattr(response, "response", str(response)),
                "source_nodes": source_nodes,
            }


# ============================================================================
# Ragas 评估器
# ============================================================================

class RagasEvaluator:
    """Ragas 评估器"""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embed_base_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        embedding_mode: str = "api",
        ollama_base_url: str = "http://localhost:11434",
        ollama_embed_model: str = "bge-m3",
        max_concurrent: int = 3,
    ):
        """
        初始化评估器

        Args:
            llm_model: LLM 模型名称
            llm_base_url: API 基础 URL
            llm_api_key: API Key
            embedding_model: Embedding 模型
            embed_base_url: Embedding API URL
            embed_api_key: Embedding API Key
            embedding_mode: Embedding 模式 ("api" 或 "ollama")
            ollama_base_url: Ollama API 地址
            ollama_embed_model: Ollama Embedding 模型名称
        """
        self._llm = None
        self._embed_model = None
        self._max_concurrent = max_concurrent

        self._llm_config = {
            "model": llm_model,
            "base_url": llm_base_url,
            "api_key": llm_api_key,
        }
        self._embed_config = {
            "model": embedding_model,
            "base_url": embed_base_url,
            "api_key": embed_api_key,
            "mode": embedding_mode,
            "ollama_base_url": ollama_base_url,
            "ollama_embed_model": ollama_embed_model,
        }

    def _get_llm(self):
        """获取 LLM 实例（延迟初始化）- 使用自定义 Ragas 兼容包装器"""
        if self._llm is None:
            if self._llm_config["base_url"]:
                self._llm = OpenAICompatibleLLM(
                    model=self._llm_config["model"],
                    api_base=self._llm_config["base_url"],
                    api_key=self._llm_config["api_key"] or "sk-placeholder",
                    temperature=0,
                    max_concurrent=self._max_concurrent,
                )
            else:
                raise ValueError("base_url is required for LLM")
        return self._llm

    def _get_embed_model(self):
        """获取 Embedding 模型实例（延迟初始化）- 使用自定义 Ragas 兼容包装器"""
        if self._embed_model is None:
            embed_mode = self._embed_config.get("mode", "api")

            if embed_mode == "ollama":
                embed_api_base = f"{self._embed_config['ollama_base_url']}/v1"
                self._embed_model = OpenAICompatibleEmbeddings(
                    model=self._embed_config["ollama_embed_model"],
                    api_base=embed_api_base,
                    api_key="ollama",
                    max_concurrent=self._max_concurrent,
                )
            elif self._embed_config["base_url"]:
                self._embed_model = OpenAICompatibleEmbeddings(
                    model=self._embed_config["model"],
                    api_base=self._embed_config["base_url"],
                    api_key=self._embed_config["api_key"] or "sk-placeholder",
                    max_concurrent=self._max_concurrent,
                )
            else:
                raise ValueError("embed_base_url or ollama mode is required for embedding")
        return self._embed_model

    def _get_ragas_metrics(self):
        """获取 Ragas 指标列表"""
        llm = cast(InstructorBaseRagasLLM, self._get_llm())
        embeddings = cast(BaseRagasEmbedding, self._get_embed_model())
        return [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
            ContextRelevance(llm=llm),
            AnswerCorrectness(llm=llm),
        ]

    async def evaluate(
        self,
        query_engine: Any,
        testset_path: str,
        output_path: str = "results/evaluation_results.csv",
        max_concurrent: int = 5,
    ) -> pd.DataFrame:
        """
        执行评估

        Args:
            query_engine: RAG 查询引擎（HybridRAGEngine 或 llama-index QueryEngine）
            testset_path: 测试集路径
            output_path: 结果输出路径
            max_concurrent: 最大并发数

        Returns:
            评估结果 DataFrame
        """
        print(f"\n{'='*60}")
        print("开始 Ragas 评估...")
        print(f"{'='*60}")

        # 加载测试集
        from .ragas_generator import RagasTestsetGenerator
        generator = RagasTestsetGenerator(
            llm_model=self._llm_config["model"],
            llm_base_url=self._llm_config["base_url"],
            llm_api_key=self._llm_config["api_key"],
            embedding_model=self._embed_config["model"],
            embed_base_url=self._embed_config["base_url"],
            embed_api_key=self._embed_config["api_key"],
        )
        samples = generator.load_testset(testset_path)
        print(f"加载测试集: {len(samples)} 个样本")

        # 包装查询引擎
        rag_wrapper = RAGQueryWrapper(query_engine)

        # 准备数据
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        latencies = []
        question_types = []

        # 并发查询
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_sample(sample: EvalSample, idx: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self._process_single_sample(rag_wrapper, sample, idx)

        print(f"执行 RAG 查询（最大并发: {max_concurrent}）...")
        tasks = [process_sample(s, i) for i, s in enumerate(samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"样本 {i} 处理失败: {r}")
                continue
            if r is None or not isinstance(r, dict):
                continue

            questions.append(r["question"])
            answers.append(r["answer"])
            contexts_list.append(r["contexts"])
            ground_truths.append([r["ground_truth"]])  # Ragas 要求列表格式
            latencies.append(r["latency_ms"])
            question_types.append(r.get("question_type", "unknown"))
            success_count += 1

        print(f"成功处理 {success_count}/{len(samples)} 个样本")

        if success_count == 0:
            raise ValueError("没有成功处理任何样本")

        # 构建 Ragas 数据集
        print("构建 Ragas 数据集...")
        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        })

        # 执行 Ragas 评估
        print("运行 Ragas 评估（计算 6 大指标）...")
        run_config = RunConfig(max_workers=max_concurrent, timeout=180)

        evaluation_result = evaluate(
            dataset=ragas_dataset,
            metrics=self._get_ragas_metrics(),
            llm=self._get_llm(),
            embeddings=self._get_embed_model(),
            run_config=run_config,
        )

        # 转换为 DataFrame
        scores_df = cast(EvaluationDataset, evaluation_result).to_pandas()

        # 添加元数据列
        scores_df["latency_ms"] = latencies
        scores_df["question_type"] = question_types

        # 保存结果
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 评估结果已保存到: {output_path}")

        self._print_summary(scores_df)

        return scores_df

    async def _process_single_sample(
        self,
        rag_wrapper: RAGQueryWrapper,
        sample: EvalSample,
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """处理单个样本"""
        start = time.time()

        try:
            result = await rag_wrapper.aquery(sample.question)
            latency = (time.time() - start) * 1000

            # 提取上下文文本
            contexts = []
            source_nodes = result.get("source_nodes", [])
            for node in source_nodes:
                if hasattr(node, "text"):
                    contexts.append(node.text)
                elif isinstance(node, dict):
                    contexts.append(node.get("text", ""))
                else:
                    contexts.append(str(node))

            return {
                "question": sample.question,
                "answer": result.get("response", ""),
                "contexts": contexts,
                "ground_truth": sample.answer,
                "latency_ms": latency,
                "question_type": sample.evolution_type,
            }

        except Exception as e:
            logger.error(f"样本 {idx} 处理失败: {e}")
            return None

    def _print_summary(self, df: pd.DataFrame) -> None:
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("📊 Ragas 评估摘要")
        print("=" * 60)

        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_relevancy",
            "answer_correctness",
        ]

        for metric in metrics:
            if metric in df.columns:
                avg = df[metric].mean()
                std = df[metric].std()
                # 处理 NaN
                if pd.isna(avg):
                    print(f"{metric:25s}: N/A (部分样本评估失败)")
                else:
                    print(f"{metric:25s}: {avg:.3f} ± {std:.3f}")

        print("=" * 60)
        print(f"总样本数: {len(df)}")

        if "latency_ms" in df.columns:
            avg_latency = df["latency_ms"].mean()
            if not pd.isna(avg_latency):
                print(f"平均延迟: {avg_latency:.0f}ms")

        print("=" * 60)


# ============================================================================
# 使用示例
# ============================================================================

async def main():
    """使用示例"""
    from llama_index.core import VectorStoreIndex, Document

    # 初始化评估器（使用 freeapi）
    evaluator = RagasEvaluator(
        llm_model="gpt-4o-mini",
        llm_base_url="https://free.v36.cm/v1/",
        llm_api_key="your-api-key",
        embedding_model="text-embedding-3-small",
        embed_base_url="https://free.v36.cm/v1/",
        embed_api_key="your-api-key",
    )

    # 创建测试查询引擎
    documents = [Document(text="测试文档内容")]
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # 执行评估
    results = await evaluator.evaluate(
        query_engine=query_engine,
        testset_path="results/testset.json",
        output_path="results/evaluation_results.csv",
        max_concurrent=5,
    )


if __name__ == "__main__":
    asyncio.run(main())
