# -*- coding: utf-8 -*-
"""
基于 Ragas 的 RAG 评估器
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, cast

import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig, EvaluationDataset
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._nv_metrics import ContextRelevance
from ragas.metrics._answer_correctness import AnswerCorrectness
# 禁用 Ragas 遥测追踪（避免 SSL 证书过期错误）
os.environ["RAGAS_DO_NOT_TRACK"] = "True"

from astrbot.api import logger

# MiniMax 专属请求字段合并（openai SDK 兼容）。双重 import 模式（CLAUDE.md 约定）。
try:
    from .minimax_compat import apply_llm_request_fields
except ImportError:
    from minimax_compat import apply_llm_request_fields  # type: ignore


class _LLMWithN:
    """自定义 LLM wrapper，替代 InstructorLLM 以支持 n>1。

    问题：llm_factory() 创建的 InstructorLLM 在 RAGAS 的 generate_multiple() 中
    走专门的分支（line 253-273），该分支忽略 n 参数，始终只返回 1 个 generation。

    方案：不使用 InstructorLLM，直接用 OpenAI client + response_format=json_object 生成
    JSON 输出。因为此类不是 InstructorBaseRagasLLM，RAGAS 的 generate_multiple() 会走
    BaseRagasLLM 分支（line 274-283），该分支原生支持 n 参数。

    每次 generate() 调用发送 n 次独立的 API 请求（确保任何 API 提供商都兼容），
    返回 LLMResult 供 RAGAS 的 RagasOutputParser 解析为 Pydantic model。
    """

    def __init__(self, client, model: str, temperature: float = 0, max_tokens: int = 16384):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.agenerate_text: Any = None  # 由 _make_agenerate_text 在 _get_llm() 中设置
        # MiniMax 思考模式需禁用；标准端点（智谱等）发纯净请求。详见 minimax_compat.py。
        self._endpoint_base_url = str(getattr(client, "base_url", ""))

    async def generate(self, prompt_value, n=1, temperature=None, stop=None, callbacks=None):
        """被 RAGAS generate_multiple() 的 BaseRagasLLM 分支调用。"""
        from langchain_core.outputs import Generation, LLMResult

        kwargs = dict(
            model=self._model,
            messages=[{"role": "user", "content": prompt_value.text}],
            max_tokens=self._max_tokens,
            temperature=temperature if temperature is not None else self._temperature,
            n=n,
        )
        # MiniMax 专属字段（thinking 走 extra_body / response_format 顶层）；标准端点为空 dict
        apply_llm_request_fields(kwargs, self._endpoint_base_url)
        if stop:
            kwargs["stop"] = stop

        try:
            r = self._client.chat.completions.create(**kwargs)
        except Exception:
            # 兜底：部分端点（MiniMax-M3 等）连 n>1 的请求本身都 400 拒绝
            # （"model does not support n > 1"），降级为 n 次独立的 n=1 请求，
            # 保持 generate_multiple 需要的 n 条 generation 语义不变。
            single_kwargs = {k: v for k, v in kwargs.items() if k != "n"}
            gens = []
            for _ in range(n):
                r = self._client.chat.completions.create(**single_kwargs)
                gens.append(Generation(text=r.choices[0].message.content or ""))
            return LLMResult(generations=[gens])

        gens = [Generation(text=choice.message.content or "") for choice in r.choices]
        # 兜底：部分 API 代理不支持 n 参数，仅返回 1 个 choice
        while len(gens) < n:
            fallback_kwargs = {k: v for k, v in kwargs.items() if k != "n"}
            r = self._client.chat.completions.create(**fallback_kwargs)
            gens.append(Generation(text=r.choices[0].message.content or ""))
        return LLMResult(generations=[gens])


def _get_git_info() -> dict:
    """获取当前代码仓库的 git 信息，用于结果溯源"""
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.PIPE
        ).strip()
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, text=True, stderr=subprocess.PIPE
        ).strip()
        commit_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%ai", "HEAD"],
            cwd=repo_root, text=True, stderr=subprocess.PIPE
        ).strip()
        is_dirty = bool(subprocess.check_output(
            ["git", "diff", "--stat", "HEAD"],
            cwd=repo_root, text=True, stderr=subprocess.PIPE
        ).strip())
        return {
            "commit": commit_hash,
            "commit_date": commit_date,
            "dirty": is_dirty,
        }
    except FileNotFoundError:
        logger.warning("git 未安装或不在 PATH 中，无法获取代码溯源信息")
    except subprocess.CalledProcessError as e:
        logger.warning(f"git 命令执行失败（可能不在 git 仓库中），无法获取代码溯源信息: {e.stderr.strip() if e.stderr else e}")
    return {"commit": "unknown", "commit_date": "unknown", "dirty": False}


def load_raw_answers(path: str) -> tuple[list[dict], dict]:
    """加载 raw_answers.json（兼容新旧格式）。

    Returns:
        (results_list, metadata_dict) — 旧格式时 metadata 为空 dict
    """
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        return loaded, {}
    if isinstance(loaded, dict) and "results" in loaded:
        return loaded["results"], loaded.get("_metadata", {})
    raise ValueError(f"无法识别的 raw_answers.json 格式: {type(loaded).__name__}")


sys.path.insert(0, str(Path(__file__).parent))
from ragas_generator import EvalSample, UnslothEmbeddingsWrapper, OpenAICompatibleEmbeddings

# call_llm import（确保 plugin root 在 sys.path 中）
_plugin_root = str(Path(__file__).parent.parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)
from provider.llm_utils import call_llm  # noqa: E402


# ============================================================================
# RAG 查询引擎包装器（兼容 HybridRAGEngine）
# ============================================================================

class RAGQueryWrapper:
    """HybridRAGEngine 查询包装器，统一检索 + 答案生成接口"""

    def __init__(
        self,
        query_engine: Any,
        answer_top_k: int = 5,
        context=None,
        config: Optional[dict] = None,
        mode: str = "rag",
    ):
        """
        Args:
            query_engine: HybridRAGEngine 实例
            answer_top_k: 用 top-K 个检索 chunk 生成答案（默认5）
            context: AstrBot Context（用于 call_llm() 调用系统 LLM）
            config: 插件配置字典
            mode: 评估模式 "rag" | "agentic"
        """
        self._engine = query_engine
        self._answer_top_k = answer_top_k
        self._context = context
        self._config = config
        self._mode = mode

        if mode == "agentic":
            # agentic 模式不需要 engine.search()，直接调 LangGraph workflow
            return

        if not hasattr(query_engine, "search"):
            raise TypeError(f"query_engine 必须为 HybridRAGEngine（需有 search 方法），实际: {type(query_engine)}")

    @staticmethod
    def _node_to_dict(node, score=None) -> dict:
        """将检索节点统一转为 dict 格式。"""
        if isinstance(node, dict):
            return {
                "text": node.get("text", ""),
                "metadata": node.get("metadata", {}),
                "score": score if score is not None else node.get("score", 0.0),
            }
        return {
            "text": getattr(node, "text", ""),
            "metadata": getattr(node, "metadata", {}),
            "score": score if score is not None else getattr(node, "score", 0.0),
        }

    async def aquery(self, query: str) -> Dict[str, Any]:
        """执行异步查询，返回 sources、answer、images 字典"""
        if self._mode == "agentic":
            return await self._agentic_query(query)

        result = await self._engine.search(query)

        nodes = getattr(result, "nodes", [])
        scores = getattr(result, "scores", [1.0] * len(nodes))
        source_nodes = [
            self._node_to_dict(node, scores[i] if i < len(scores) else 0.0)
            for i, node in enumerate(nodes)
        ]

        answer = ""
        answer_chunks = source_nodes[:self._answer_top_k]
        answer_texts = [n["text"] for n in answer_chunks if n["text"]]
        if answer_texts and self._context is not None:
            context_block = "\n\n".join(answer_texts)
            prompt = (
                "You are answering a research question using excerpts from academic papers. "
                "Use the provided excerpts to give the most accurate and complete answer possible.\n\n"
                "Rules:\n"
                "- Answer in 4-8 sentences. Include ALL specific numbers, metrics, method names, "
                "and dataset names found in the excerpts that are relevant to the question.\n"
                "- If the exact term from the question does not appear in the excerpts, identify "
                "the most closely related information and explain what the excerpts do say.\n"
                "- Stay grounded in the excerpts; do not fabricate facts.\n\n"
                f"Paper excerpts:\n{context_block}\n\n"
                f"Question: {query}\n\nAnswer:"
            )
            answer = await self._generate_answer(prompt, fallback=context_block)

        return {
            "response": answer,
            "answer": answer,
            "sources": source_nodes,
            "images": [],
        }

    async def _generate_answer(self, prompt: str, fallback: str = "") -> str:
        """通过 call_llm() 调用系统 LLM 生成答案。"""

        if self._context is not None:
            try:
                answer = await call_llm(
                    prompt, self._context, self._config,
                    temperature=0.3, max_tokens=2048,
                )
                if answer and answer.strip():
                    return answer
                logger.warning("[RAGQueryWrapper] 系统 LLM 返回空答案")
            except Exception as e:
                logger.error(f"[RAGQueryWrapper] 系统 LLM 调用失败: {type(e).__name__}: {e}")

        # call_llm() 失败时返回原始 context 文本，确保评估仍可继续
        if fallback:
            logger.warning(f"[RAGQueryWrapper] 使用 fallback 文本 ({len(fallback)} chars)")
            return fallback
        return ""

    async def _agentic_query(self, query: str) -> Dict[str, Any]:
        """Agentic RAG 查询：直接调用 LangGraph workflow，提取 final_answer + retrieved_nodes。"""
        from agentic_rag.workflow import compile_workflow

        app = compile_workflow()
        initial_state = {
            "query": query,
            "_context": self._context,
            "_config": self._config,
            "top_k": self._answer_top_k,
            "steps": [],
        }
        result = await app.ainvoke(initial_state)

        final_answer = result.get("final_answer", "")
        retrieved_nodes = result.get("retrieved_nodes", [])
        if not isinstance(retrieved_nodes, list):
            retrieved_nodes = []

        sources = [self._node_to_dict(node) for node in retrieved_nodes]

        return {
            "response": final_answer,
            "answer": final_answer,
            "sources": sources,
            "images": [],
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
        max_concurrent: int = 3,
        answer_top_k: int = 5,
        llm_max_tokens: int = 16384,
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
            embedding_mode: Embedding 模式 ("api")
            llm_max_tokens: LLM max_tokens（默认 16384，推理模型需更高值容纳 reasoning tokens）
        """
        self._llm = None
        self._embed_model = None
        self._max_concurrent = max_concurrent
        self._answer_top_k = answer_top_k

        self._llm_config = {
            "model": llm_model,
            "base_url": llm_base_url,
            "api_key": llm_api_key,
            "max_tokens": llm_max_tokens,
        }
        self._embed_config = {
            "model": embedding_model,
            "base_url": embed_base_url,
            "api_key": embed_api_key,
            "mode": embedding_mode,
        }

    @staticmethod
    def _create_openai_client(base_url: str, api_key: str):
        from openai import OpenAI
        import httpx
        return OpenAI(
            base_url=base_url.rstrip('/'),
            api_key=api_key or "sk-placeholder",
            max_retries=5,
            timeout=httpx.Timeout(300.0, connect=30.0),
        )

    def _get_llm(self):
        """获取 LLM 实例（延迟初始化）"""
        if self._llm is None:
            if self._llm_config["base_url"]:
                client = self._create_openai_client(
                    self._llm_config["base_url"], self._llm_config["api_key"])
                self._llm = _LLMWithN(
                    client=client,
                    model=self._llm_config["model"],
                    temperature=0,
                    max_tokens=self._llm_config.get("max_tokens", 16384),
                )
                # 兼容旧 API 指标（_nv_metrics 调用 agenerate_text）
                self._llm.agenerate_text = self._make_agenerate_text(
                    client, self._llm_config["model"],
                    self._llm_config.get("max_tokens", 16384))
                print(f"✅ LLM (_LLMWithN, supports n>1) 初始化成功: {self._llm_config['model']} @ {self._llm_config['base_url']}")
            else:
                raise ValueError("base_url is required for LLM")
        return self._llm

    @staticmethod
    def _make_agenerate_text(client, model: str, max_tokens: int):
        """给 InstructorLLM 添加 agenerate_text 方法（直接调 OpenAI API）。"""
        from langchain_core.outputs import LLMResult, Generation
        async def _agenerate_text(prompt, n=1, temperature=None, stop=None, callbacks=None):
            text = getattr(prompt, 'text', str(prompt))
            kwargs: dict = dict(model=model, messages=[dict(role="user", content=text)],
                                max_tokens=max_tokens, temperature=temperature or 0)
            # MiniMax 专属字段（thinking 走 extra_body / response_format 顶层）；标准端点为空 dict
            apply_llm_request_fields(kwargs, str(getattr(client, "base_url", "")))
            if stop:
                kwargs["stop"] = stop
            gens = []
            for _ in range(n):
                r = client.chat.completions.create(**kwargs)
                gens.append(Generation(text=r.choices[0].message.content or ""))
            return LLMResult(generations=[gens])
        return _agenerate_text

    def _get_embed_model(self):
        """获取 Embedding 模型实例（延迟初始化）

        使用自定义 OpenAICompatibleEmbeddings（复用 ragas_generator 的实现）：
        - MiniMax /v1/embeddings 为非标准格式（texts/type=query/embo-01/vectors），
          embedding_factory 的 OpenAI 客户端（input/data 格式）在该端点不可用
        - 自带并发限制和 3 次指数退避重试
        """
        if self._embed_model is None:
            if self._embed_config.get("mode") == "unsloth":
                print("🔧 正在初始化本地 BGE-M3 embedding (UnslothEmbeddingsWrapper)...")
                self._embed_model = UnslothEmbeddingsWrapper(
                    model_path=self._embed_config.get("model_path"),
                    device=self._embed_config.get("device", ""),
                )
                print("✅ 本地 BGE-M3 embedding 初始化成功")
            elif self._embed_config["base_url"]:
                self._embed_model = OpenAICompatibleEmbeddings(
                    model=self._embed_config["model"],
                    api_base=self._embed_config["base_url"],
                    api_key=self._embed_config["api_key"] or "sk-placeholder",
                )
                print(f"✅ Embedding 初始化成功: {self._embed_config['model']} @ {self._embed_config['base_url']}")
            else:
                raise ValueError("embed_base_url is required for embedding")
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
        llm = cast(BaseRagasLLM, self._get_llm())
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

    def _evaluate_and_save(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[str],
        latencies: list,
        question_types: list[str],
        has_multimodal_list: list[bool],
        output_path: str,
        max_concurrent: int,
    ) -> pd.DataFrame:
        """构建 Ragas 数据集、执行评估（含 NaN 重试）、添加元数据、保存结果。

        evaluate() 和 evaluate_from_raw_answers() 共享的核心评估逻辑。
        """
        print("构建 Ragas 数据集...")
        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "reference": ground_truths,
        })

        print("运行 Ragas 评估（计算 6 大指标，串行执行防止 429）...")
        run_config = RunConfig(max_workers=1, timeout=180)
        evaluation_result = evaluate(
            dataset=ragas_dataset,
            metrics=self._get_ragas_metrics(),
            llm=cast(BaseRagasLLM, self._get_llm()),
            embeddings=cast(BaseRagasEmbedding, self._get_embed_model_with_legacy()),
            run_config=run_config,
        )
        scores_df = cast(EvaluationDataset, evaluation_result).to_pandas()

        # 对缺失值（NaN）重试一次
        nan_mask = scores_df.isna().any(axis=1)
        if bool(nan_mask.any()):
            nan_count = int(nan_mask.sum())
            nan_indices = [i for i in range(len(nan_mask)) if bool(nan_mask.iloc[i])]
            print(f"\n⚠️ {nan_count} 个样本存在 NaN 指标值，正在重试...")
            try:
                full_dict = cast(dict, ragas_dataset.to_dict())
                retry_dict = {k: [v[i] for i in nan_indices] for k, v in full_dict.items()}
                retry_dataset = Dataset.from_dict(retry_dict)
                retry_result = evaluate(
                    dataset=retry_dataset,
                    metrics=self._get_ragas_metrics(),
                    llm=cast(BaseRagasLLM, self._get_llm()),
                    embeddings=cast(BaseRagasEmbedding, self._get_embed_model_with_legacy()),
                    run_config=RunConfig(max_workers=1, timeout=180),
                )
                retry_df = cast(EvaluationDataset, retry_result).to_pandas()
                for col in retry_df.columns:
                    if col not in scores_df.columns:
                        continue
                    for local_i, global_i in enumerate(nan_indices):
                        val = scores_df.at[global_i, col]
                        if isinstance(val, float) and pd.isna(val):
                            scores_df.at[global_i, col] = retry_df.at[local_i, col]
                filled = int(sum(1 for i in nan_indices if not bool(scores_df.iloc[i].isna().any())))
                print(f"✅ 重试完成: {filled}/{nan_count} 个样本已填充")
            except Exception as e:
                logger.warning(f"NaN 重试失败: {e}")

        # 添加元数据列并保存
        scores_df["latency_ms"] = latencies
        scores_df["question_type"] = question_types
        scores_df["has_multimodal"] = has_multimodal_list

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 评估结果已保存到: {output_path}")

        self._print_summary(scores_df)
        return scores_df

    async def evaluate(
        self,
        query_engine: Any,
        testset_path: str,
        output_path: str = "results/evaluation_results.csv",
        max_concurrent: int = 5,
        context=None,
        config: Optional[dict] = None,
        mode: str = "rag",
    ) -> pd.DataFrame:
        """
        执行评估

        Args:
            query_engine: RAG 查询引擎（HybridRAGEngine 或 llama-index QueryEngine）
            testset_path: 测试集路径
            output_path: 结果输出路径
            max_concurrent: 最大并发数
            context: AstrBot Context（用于系统 LLM 答案生成）
            config: 插件配置字典
            mode: 评估模式 "rag" | "agentic"

        Returns:
            评估结果 DataFrame
        """
        mode_label = "Agentic RAG" if mode == "agentic" else "RAG"
        print(f"\n{'='*60}")
        print(f"开始 Ragas 评估 [{mode_label}]...")
        print(f"{'='*60}")

        # 加载测试集（直接反序列化，避免无用的 RagasTestsetGenerator 实例化）
        with open(testset_path, "r", encoding="utf-8") as f:
            _testset_data = json.load(f)
        samples = [EvalSample.from_dict(item) for item in _testset_data]
        print(f"加载测试集: {len(samples)} 个样本")

        # 包装查询引擎（answer 生成通过 call_llm() 使用系统 LLM provider）
        rag_wrapper = RAGQueryWrapper(
            query_engine,
            answer_top_k=self._answer_top_k,
            context=context,
            config=config,
            mode=mode,
        )

        # =====================================================================
        # 增量保存：每完成一个样本就写入 raw_answers.json，确保中途崩溃不丢失已计算结果
        # =====================================================================
        _raw_name = f"raw_answers_{mode}.json" if mode != "rag" else "raw_answers.json"
        raw_results_path = Path(output_path).parent / _raw_name
        Path(raw_results_path).parent.mkdir(parents=True, exist_ok=True)

        # 先写入初始状态（含元数据和空结果列表）
        if raw_results_path.exists():
            logger.warning(f"{raw_results_path} 已存在，将被覆盖")
            print(f"⚠️  {raw_results_path} 已存在，将被覆盖")

        git_info = _get_git_info()
        base_payload = {
            "_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_info["commit"],
                "git_commit_date": git_info["commit_date"],
                "git_dirty": git_info["dirty"],
                "total_samples": len(samples),
                "success_count": 0,
                "mode": mode,
                "llm_model": self._llm_config["model"],
                "llm_base_url": self._llm_config["base_url"],
                "embedding_model": self._embed_config["model"],
                "embed_base_url": self._embed_config["base_url"],
                "answer_top_k": self._answer_top_k,
                "max_concurrent": max_concurrent,
            },
            "results": [],
        }
        with open(raw_results_path, "w", encoding="utf-8") as f:
            json.dump(base_payload, f, ensure_ascii=False, indent=2)

        commit_label = git_info["commit"][:8] if git_info["commit"] != "unknown" else "unknown"
        print(f"📝 增量保存到: {raw_results_path}  (commit: {commit_label})")

        semaphore = asyncio.Semaphore(max_concurrent)
        total_samples = len(samples)
        results_ordered: list[Optional[dict]] = [None] * total_samples
        results_count = 0
        write_lock = asyncio.Lock()

        async def process_and_save(sample: EvalSample, idx: int) -> None:
            """处理单个样本并立即增量保存到 raw_answers.json"""
            nonlocal results_count
            try:
                async with semaphore:
                    result = await self._process_single_sample(rag_wrapper, sample, idx)
            except Exception as e:
                logger.error(f"样本 {idx} 处理失败: {e}")
                return
            if result is None or not isinstance(result, dict):
                return

            async with write_lock:
                results_ordered[idx] = result
                results_count += 1
                _n = results_count

                # 按原始顺序构建显示列表
                ordered = [
                    {
                        "question": r["question"],
                        "answer": r["answer"],
                        "contexts": r["contexts"],
                        "ground_truth": r.get("ground_truth", ""),
                        "latency_ms": r["latency_ms"],
                        "question_type": r.get("question_type", "unknown"),
                        "has_multimodal": r.get("has_multimodal", False),
                        "used_images": r.get("used_images") or [],
                    }
                    for r in results_ordered if r is not None
                ]

                base_payload["_metadata"]["success_count"] = _n
                base_payload["results"] = ordered
                try:
                    with open(raw_results_path, "w", encoding="utf-8") as f:
                        json.dump(base_payload, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"写入 raw_answers.json 失败 (已处理 {_n}/{total_samples}): {e}")

            print(f"  [{_n}/{total_samples}] 已保存到 {raw_results_path.name}")

        print(f"执行 RAG 查询（最大并发: {max_concurrent}）...")
        tasks = [asyncio.create_task(process_and_save(s, i)) for i, s in enumerate(samples)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # 收集结果用于 Ragas 评估
        success_count = results_count

        if success_count == 0:
            raise ValueError("没有成功处理任何样本")

        print(f"成功处理 {success_count}/{len(samples)} 个样本")
        print(f"✅ 原始回答已保存到: {raw_results_path}")

        # 从 results_ordered 按序重建列表，供后续 Ragas 评估使用
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        latencies = []
        question_types = []
        has_multimodal_count = 0
        used_images_all: list = []

        for r in results_ordered:
            if r is None:
                continue
            questions.append(r["question"])
            answers.append(r["answer"])
            contexts_list.append(r["contexts"])
            ground_truths.append(r.get("ground_truth", ""))
            latencies.append(r["latency_ms"])
            question_types.append(r.get("question_type", "unknown"))
            if r.get("has_multimodal"):
                has_multimodal_count += 1
            used_images = r.get("used_images") or []
            used_images_all.extend(used_images)

        if has_multimodal_count > 0:
            print(f"🖼️ 其中 {has_multimodal_count} 个样本涉及图片/表格")

        multimodal_flags = [r.get("has_multimodal", False) for r in results_ordered if r is not None]
        return self._evaluate_and_save(
            questions, answers, contexts_list, ground_truths,
            latencies, question_types, multimodal_flags,
            output_path, max_concurrent,
        )

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

        # 读取 raw_answers.json（兼容新旧格式）
        raw_data, meta = load_raw_answers(raw_answers_path)
        if not meta:
            print("⚠️ 旧格式 raw_answers.json（无元数据），建议重新运行 RAG 推理以生成含溯源信息的版本")
        else:
            commit_short = meta.get("git_commit", "unknown")[:8]
            print(f"📋 元数据: commit={commit_short}, dirty={meta.get('git_dirty')}, "
                  f"generated_at={meta.get('generated_at', 'unknown')}")

        print(f"加载 {len(raw_data)} 个已有结果")

        has_multimodal_list = [d.get("has_multimodal", False) for d in raw_data]
        multimodal_count = sum(has_multimodal_list)
        if multimodal_count > 0:
            print(f"🖼️ 其中 {multimodal_count} 个样本涉及图片/表格")

        return self._evaluate_and_save(
            [d["question"] for d in raw_data],
            [d["answer"] for d in raw_data],
            [d["contexts"] for d in raw_data],
            [d.get("ground_truth", "") for d in raw_data],
            [d.get("latency_ms", 0) for d in raw_data],
            [d.get("question_type", "unknown") for d in raw_data],
            has_multimodal_list,
            output_path, max_concurrent,
        )

    async def _process_single_sample(
        self,
        rag_wrapper: RAGQueryWrapper,
        sample: EvalSample,
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """处理单个样本（支持多模态）"""
        start = time.time()

        try:
            result = await rag_wrapper.aquery(sample.question)
            latency = (time.time() - start) * 1000

            # 使用测试集原始标记，而不是从检索结果推断
            has_multimodal = sample.metadata.get("is_multimodal", False)

            # 提取上下文文本（包含多模态信息）
            contexts = []
            source_nodes = result.get("sources", [])

            for node in source_nodes:
                if isinstance(node, dict):
                    text = node.get("text", "")
                    node_metadata = node.get("metadata", {})
                elif hasattr(node, "text"):
                    text = node.text
                    node_metadata = getattr(node, "metadata", {}) if hasattr(node, "metadata") else {}
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

            # 从检索结果中提取图片路径（RAGQueryWrapper.aquery 始终返回 images:[]）
            used_images = []
            for node in source_nodes:
                if isinstance(node, dict):
                    img = node.get("metadata", {}).get("image_path", "")
                elif hasattr(node, "metadata"):
                    img = node.metadata.get("image_path", "")
                else:
                    continue
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

    _METRIC_NAMES = [
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "context_relevancy", "answer_correctness",
    ]

    def _print_summary(self, df: pd.DataFrame) -> None:
        """打印评估摘要"""
        print(f"\n{'='*60}\n📊 Ragas 评估摘要\n{'='*60}")

        for m in self._METRIC_NAMES:
            if m in df.columns:
                avg = df[m].mean()
                if pd.isna(avg):
                    print(f"{m:25s}: N/A (部分样本评估失败)")
                else:
                    print(f"{m:25s}: {avg:.3f} ± {df[m].std():.3f}")

        print(f"{'='*60}\n总样本数: {len(df)}")

        if "latency_ms" in df.columns:
            avg_latency = df["latency_ms"].mean()
            if not pd.isna(avg_latency):
                print(f"平均延迟: {avg_latency:.0f}ms")

        if "has_multimodal" in df.columns:
            mm_count = int(df["has_multimodal"].sum())
            if mm_count > 0:
                print(f"🖼️ 涉及图片/表格: {mm_count} 个样本 ({mm_count/len(df)*100:.1f}%)")
                for label, mask in [("多模态", df["has_multimodal"] == True), ("仅文本", df["has_multimodal"] == False)]:
                    sub = df[mask]
                    if len(sub) > 0:
                        print(f"\n  📊 {label}样本指标:")
                        for m in self._METRIC_NAMES:
                            if m in sub.columns:
                                avg = sub[m].mean()
                                if not pd.isna(avg):
                                    print(f"     {m:25s}: {avg:.3f}")

        print("=" * 60)


