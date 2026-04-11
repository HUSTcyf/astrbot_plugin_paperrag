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
from ragas.llms.base import InstructorBaseRagasLLM, llm_factory
from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
# 使用内部 `_` 模块的 metric 类，避免 collections 模块的 class identity 问题
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._nv_metrics import ContextRelevance
from ragas.metrics._answer_correctness import AnswerCorrectness
from datasets import Dataset

# 禁用 Ragas 遥测追踪（避免 SSL 证书过期错误）
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "True"

from astrbot.api import logger

# 导入 EvalSample（用于测试集加载）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragas_generator import EvalSample


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

    async def aquery(self, query: str, force_english: bool = False) -> Dict[str, Any]:
        """
        执行异步查询

        Returns:
            包含 response 和 source_nodes 的字典
        """
        if self._is_hybrid:
            # HybridRAGEngine
            result = await self._engine.search(query, mode="rag", force_english=force_english)
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
        """获取 LLM 实例（延迟初始化）- 使用 llm_factory 创建 InstructorLLM"""
        if self._llm is None:
            if self._llm_config["base_url"]:
                # 使用 ragas 0.4.3 新接口 llm_factory 创建 InstructorLLM
                # 这解决了 collection metrics 要求的 InstructorLLM 接口
                from openai import OpenAI
                client = OpenAI(
                    base_url=self._llm_config["base_url"].rstrip('/'),
                    api_key=self._llm_config["api_key"] or "sk-placeholder",
                )
                self._llm = llm_factory(
                    model=self._llm_config["model"],
                    provider="openai",
                    client=client,
                    temperature=0,
                )
                print(f"✅ LLM (InstructorLLM) 初始化成功: {self._llm_config['model']} @ {self._llm_config['base_url']}")
            else:
                raise ValueError("base_url is required for LLM")
        return self._llm

    def _get_embed_model(self):
        """获取 Embedding 模型实例（延迟初始化）- 使用 embedding_factory 创建 modern embeddings"""
        if self._embed_model is None:
            embed_mode = self._embed_config.get("mode", "api")

            if embed_mode == "ollama":
                embed_api_base = f"{self._embed_config['ollama_base_url']}/v1"
                # 使用 ragas 0.4.3 新接口 embedding_factory 创建 modern embeddings
                # 这解决了 AnswerRelevancy 等指标要求的 modern embeddings 接口
                from openai import OpenAI
                client = OpenAI(
                    base_url=embed_api_base,
                    api_key="ollama",
                )
                self._embed_model = embedding_factory(
                    provider="openai",
                    model=self._embed_config["ollama_embed_model"],
                    client=client,
                    interface="modern",
                )
                print(f"✅ Embedding (modern) 初始化成功: {self._embed_config['ollama_embed_model']} @ {embed_api_base}")
            elif self._embed_config["base_url"]:
                from openai import OpenAI
                client = OpenAI(
                    base_url=self._embed_config["base_url"].rstrip('/'),
                    api_key=self._embed_config["api_key"] or "sk-placeholder",
                )
                self._embed_model = embedding_factory(
                    provider="openai",
                    model=self._embed_config["model"],
                    client=client,
                    interface="modern",
                )
                print(f"✅ Embedding (modern) 初始化成功: {self._embed_config['model']} @ {self._embed_config['base_url']}")
            else:
                raise ValueError("embed_base_url or ollama mode is required for embedding")
        return self._embed_model

    def _get_embed_model_with_legacy(self):
        """获取 Embedding 模型，添加 embed_query/aembed_query 兼容方法

        OpenAIEmbeddings 使用 embed_text/aembed_text 现代接口，
        但 ragas 内部指标仍调用 embed_query/aembed_query。
        此方法返回添加了兼容方法的包装器。

        注意：同步客户端不支持 aembed_text()，因此 async 方法通过
        run_in_executor 调用同步 embed_text() 来避免阻塞事件循环。

        重要：此类必须支持 copy.copy() 和 pickle，否则在 ragas executor
        复制指标时会触发无限递归（__getattr__ -> __reduce_ex__ 循环）。
        """
        import copy

        embed = self._get_embed_model()

        # 如果已经有 embed_query，直接返回
        if hasattr(embed, 'embed_query'):
            return embed

        # 否则包装添加兼容方法
        class EmbeddingWithLegacy:
            """给 modern OpenAIEmbeddings 添加 embed_query/aembed_query 兼容方法"""

            def __init__(self, inner):
                # 使用 object.__setattr__ 避免描述符冲突
                object.__setattr__(self, '_inner', inner)

            # 显式定义所有需要的方法（避免依赖 __getattr__）

            def embed_query(self, text: str):
                return self._inner.embed_text(text)

            def embed_documents(self, texts):
                return self._inner.embed_texts(texts)

            async def aembed_query(self, text: str):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self._inner.embed_text(text)
                )

            async def aembed_documents(self, texts):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self._inner.embed_texts(texts)
                )

            def embed_text(self, text: str):
                return self._inner.embed_text(text)

            def embed_texts(self, texts):
                return self._inner.embed_texts(texts)

            async def aembed_text(self, text: str):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self._inner.embed_text(text)
                )

            async def aembed_texts(self, texts):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self._inner.embed_texts(texts)
                )

            # 支持 copy.copy（使用 __reduce__ 而非 __getattr__ 避免无限递归）
            def __reduce__(self):
                return (
                    self.__class__,
                    (self._inner,),
                )

            # 透传所有其他属性
            def __getattr__(self, name):
                return getattr(self._inner, name)

        return EmbeddingWithLegacy(embed)

    def _get_ragas_metrics(self):
        """获取 Ragas 指标列表"""
        llm = cast(InstructorBaseRagasLLM, self._get_llm())
        # 使用带 legacy 兼容方法的包装器，避免 embed_query/aembed_query 缺失错误
        embeddings = cast(BaseRagasEmbedding, self._get_embed_model_with_legacy())
        return [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
            ContextRelevance(llm=llm),
            AnswerCorrectness(llm=llm, embeddings=embeddings),
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
        has_multimodal_count = 0
        used_images_all: list = []

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"样本 {i} 处理失败: {r}")
                continue
            if r is None or not isinstance(r, dict):
                continue

            questions.append(r["question"])
            answers.append(r["answer"])
            contexts_list.append(r["contexts"])
            ground_truths.append(r["ground_truth"])  # 字符串
            latencies.append(r["latency_ms"])
            question_types.append(r.get("question_type", "unknown"))

            # 多模态统计
            if r.get("has_multimodal"):
                has_multimodal_count += 1
            used_images = r.get("used_images") or []
            used_images_all.extend(used_images)

            success_count += 1

        print(f"成功处理 {success_count}/{len(samples)} 个样本")
        if has_multimodal_count > 0:
            print(f"🖼️ 其中 {has_multimodal_count} 个样本涉及图片/表格")

        if success_count == 0:
            raise ValueError("没有成功处理任何样本")

        # 保存原始回答到 JSON
        raw_results_path = Path(output_path).parent / "raw_answers.json"
        raw_data = []
        for i in range(len(questions)):
            # 获取当前样本的多模态信息
            r = results[i] if i < len(results) and isinstance(results[i], dict) else {}
            raw_data.append({
                "question": questions[i],
                "answer": answers[i],
                "contexts": contexts_list[i],
                "ground_truth": ground_truths[i] if ground_truths[i] else "",
                "latency_ms": latencies[i],
                "question_type": question_types[i],
                "has_multimodal": r.get("has_multimodal", False),
                "used_images": r.get("used_images") or [],
            })
        Path(raw_results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(raw_results_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 原始回答已保存到: {raw_results_path}")

        # 构建 Ragas 数据集
        print("构建 Ragas 数据集...")
        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "reference": ground_truths,
        })

        # 执行 Ragas 评估
        print("运行 Ragas 评估（计算 6 大指标）...")
        run_config = RunConfig(max_workers=max_concurrent, timeout=180)

        evaluation_result = evaluate(
            dataset=ragas_dataset,
            metrics=self._get_ragas_metrics(),
            llm=self._get_llm(),
            embeddings=self._get_embed_model_with_legacy(),
            run_config=run_config,
        )

        # 转换为 DataFrame
        scores_df = cast(EvaluationDataset, evaluation_result).to_pandas()

        # 添加元数据列
        scores_df["latency_ms"] = latencies
        scores_df["question_type"] = question_types

        # 添加多模态信息
        multimodal_flags = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                multimodal_flags.append(r.get("has_multimodal", False))
            else:
                multimodal_flags.append(False)
        scores_df["has_multimodal"] = multimodal_flags

        # 保存结果
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 评估结果已保存到: {output_path}")

        self._print_summary(scores_df)

        return scores_df

    async def evaluate_from_raw_answers(
        self,
        raw_answers_path: str,
        output_path: str = "results/evaluation_results.csv",
        max_concurrent: int = 5,
    ) -> pd.DataFrame:
        """
        从已有的 raw_answers.json 读取结果，直接运行 Ragas 评估（跳过 RAG 推理）

        Args:
            raw_answers_path: raw_answers.json 文件路径
            output_path: 结果输出路径
            max_concurrent: 最大并发数

        Returns:
            评估结果 DataFrame
        """
        print(f"\n{'='*60}")
        print("从已有 raw_answers.json 运行 Ragas 评估（跳过 RAG 推理）...")
        print(f"{'='*60}")

        # 读取 raw_answers.json
        with open(raw_answers_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        print(f"加载 {len(raw_data)} 个已有结果")

        questions = [d["question"] for d in raw_data]
        answers = [d["answer"] for d in raw_data]
        contexts_list = [d["contexts"] for d in raw_data]
        ground_truths = [d.get("ground_truth", "") for d in raw_data]
        latencies = [d.get("latency_ms", 0) for d in raw_data]
        question_types = [d.get("question_type", "unknown") for d in raw_data]
        has_multimodal_list = [d.get("has_multimodal", False) for d in raw_data]

        # 多模态统计
        multimodal_count = sum(1 for h in has_multimodal_list if h)
        if multimodal_count > 0:
            print(f"🖼️ 其中 {multimodal_count} 个样本涉及图片/表格")

        # 构建 Ragas 数据集
        print("构建 Ragas 数据集...")
        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "reference": ground_truths,
        })

        # 执行 Ragas 评估
        print("运行 Ragas 评估（计算 6 大指标）...")
        run_config = RunConfig(max_workers=max_concurrent, timeout=180)

        evaluation_result = evaluate(
            dataset=ragas_dataset,
            metrics=self._get_ragas_metrics(),
            llm=self._get_llm(),
            embeddings=self._get_embed_model_with_legacy(),
            run_config=run_config,
        )

        # 转换为 DataFrame
        scores_df = cast(EvaluationDataset, evaluation_result).to_pandas()

        # 添加元数据列
        scores_df["latency_ms"] = latencies
        scores_df["question_type"] = question_types
        scores_df["has_multimodal"] = has_multimodal_list

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
        """处理单个样本（支持多模态）"""
        start = time.time()

        try:
            result = await rag_wrapper.aquery(sample.question, force_english=True)
            latency = (time.time() - start) * 1000

            # 使用测试集原始标记，而不是从检索结果推断
            has_multimodal = sample.metadata.get("is_multimodal", False)

            # 提取上下文文本（包含多模态信息）
            contexts = []
            source_nodes = result.get("sources", [])

            for node in source_nodes:
                if hasattr(node, "text"):
                    text = node.text
                    node_metadata = getattr(node, "metadata", {}) if hasattr(node, "metadata") else {}
                elif isinstance(node, dict):
                    text = node.get("text", "")
                    node_metadata = node.get("metadata", {})
                else:
                    text = str(node)
                    node_metadata = {}

                # 从检索结果中提取多模态信息（不影响 has_multimodal 判定）
                if isinstance(node_metadata, dict):
                    image_path = node_metadata.get("image_path", "")
                    table_path = node_metadata.get("table_path", "")

                    # 如果 RAG 引擎没有自动添加图片信息，手动添加
                    if image_path and "[IMAGE" not in text:
                        image_caption = node_metadata.get("image_caption", "")
                        if image_caption:
                            text = text + f"\n\n[IMAGE: {image_caption}]"

                    # 如果有表格信息但文本中没有，手动添加
                    if table_path and "[TABLE" not in text:
                        table_caption = node_metadata.get("table_caption", "")
                        if table_caption:
                            text = text + f"\n\n[TABLE: {table_caption}]"

                contexts.append(text)

            # 获取使用的图片列表（从检索结果中提取，与原始问题类型无关）
            used_images = result.get("images", [])
            if not used_images:
                # 从 sources 中提取图片路径
                used_images = []
                for node in source_nodes:
                    if hasattr(node, "metadata"):
                        img = node.metadata.get("image_path", "")
                        if img and img not in used_images:
                            used_images.append(img)

            return {
                "question": sample.question,
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": sample.answer,
                "latency_ms": latency,
                "question_type": sample.evolution_type,
                "has_multimodal": has_multimodal,
                "used_images": used_images,
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

        # 多模态统计
        if "has_multimodal" in df.columns:
            multimodal_count = df["has_multimodal"].sum()
            if multimodal_count > 0:
                print(f"🖼️ 涉及图片/表格: {multimodal_count} 个样本 ({multimodal_count/len(df)*100:.1f}%)")

                # 按多模态分组显示指标
                multimodal_df = df[df["has_multimodal"] == True]
                text_only_df = df[df["has_multimodal"] == False]

                if len(multimodal_df) > 0:
                    print(f"\n  📊 多模态样本指标:")
                    for metric in metrics:
                        if metric in multimodal_df.columns:
                            avg = multimodal_df[metric].mean()
                            if not pd.isna(avg):
                                print(f"     {metric:25s}: {avg:.3f}")

                if len(text_only_df) > 0:
                    print(f"\n  📊 仅文本样本指标:")
                    for metric in metrics:
                        if metric in text_only_df.columns:
                            avg = text_only_df[metric].mean()
                            if not pd.isna(avg):
                                print(f"     {metric:25s}: {avg:.3f}")

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
