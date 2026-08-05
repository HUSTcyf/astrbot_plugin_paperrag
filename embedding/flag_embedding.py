"""
FlagEmbedding BGE-M3 Provider for PaperRAG

使用 FlagEmbedding (BGEM3FlagModel) 本地加载 BGE-M3 模型，获取：
1. 稠密向量 (dense vector) - 用于 Milvus 向量检索
2. 稀疏权重 (sparse/lexical weight) - 用于关键词匹配检索
3. 多向量序列 (colbert vector) - 用于 ColBERT 式 late-interaction reranking

优势：跨平台兼容（Linux CUDA / Mac MPS / CPU），API 简洁，一行 encode() 返回全部。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from astrbot.api import logger
# FlagEmbedding 是可选的：unsloth/astrbot 模式下不依赖该包。
# 缺包时模块仍可导入，仅在显式选择 flag 模式初始化时才会报错。
try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError as _flag_import_error:
    FLAG_EMBEDDING_AVAILABLE = False


class FlagEmbeddingModel:
    """
    FlagEmbedding BGE-M3 模型封装

    提供与 UnslothEmbeddingModel 相同的接口：
    - get_dense_embedding()
    - get_sparse_weight()
    - get_multi_vector()
    - get_query_sparse_vs_doc_dense()
    """

    _lock = asyncio.Lock()

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        max_seq_length: int = 512,
        use_fp16: bool = True,
    ):
        model_dir = Path(__file__).parent.parent.resolve() / "models" / "bge-m3"
        self.model_path = model_path or str(model_dir)
        self.device = device
        self.max_seq_length = max_seq_length
        self.use_fp16 = use_fp16

        self.model: Any = None
        self._initialized = False
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._embedding_dim

    @property
    def tokenizer(self):
        if self.model is None:
            return None
        return self.model.tokenizer

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._load_model)
            self._initialized = True

    def _load_model(self) -> None:
        if not FLAG_EMBEDDING_AVAILABLE:
            raise RuntimeError(
                "FlagEmbedding 包未安装，无法使用 flag 嵌入模式。"
                "请执行 pip install FlagEmbedding，或改用 unsloth / astrbot 嵌入模式。"
                f"原始导入错误: {_flag_import_error}"
            )

        logger.info(f"[FlagEmbedding] 加载模型: {self.model_path}")
        self.model = BGEM3FlagModel(
            self.model_path,
            use_fp16=self.use_fp16,
            device=self.device,
        )
        self._embedding_dim = self.model.model.config.hidden_size
        logger.info(f"[FlagEmbedding] 加载完成, dim={self._embedding_dim}, device={self.device}")

    def get_dense_embedding(self, texts: Union[str, List[str]]) -> List[List[float]]:
        assert self.model is not None
        if isinstance(texts, str):
            texts = [texts]
        output = self.model.encode(texts, batch_size=min(len(texts), 16), max_length=self.max_seq_length)
        dense = output["dense_vecs"]
        if isinstance(dense, np.ndarray):
            dense = dense.astype(np.float32)
        else:
            dense = dense.cpu().float().numpy()
        return dense.tolist()

    def get_sparse_weight(
        self, text: str, query_embedding: Optional[List[float]] = None
    ) -> Dict[int, float]:
        """
        获取 BGE-M3 的词法稀疏权重（lexical weights）。

        BGE-M3 的 encode() 返回 lexical_weights: Dict[str, float]，
        我们将其转换为 Dict[int, float]（token_id → weight）。
        """
        assert self.model is not None

        output = self.model.encode(
            [text], batch_size=1, max_length=self.max_seq_length,
            return_dense=False, return_sparse=True, return_colbert_vecs=False,
        )
        lexical = output.get("lexical_weights")
        if not lexical:
            return {}

        # lexical_weights is List[Dict[str, float]]
        weights = lexical[0] if isinstance(lexical, list) else lexical

        # Map token string → token_id via model tokenizer
        tokenizer = self.model.tokenizer
        result: Dict[int, float] = {}
        for token_str, weight in weights.items():
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid != tokenizer.unk_token_id:
                result[tid] = weight
        return result

    def get_multi_vector(self, text: str) -> List[List[float]]:
        """获取 ColBERT 多向量序列。"""
        assert self.model is not None
        output = self.model.encode(
            [text], batch_size=1, max_length=self.max_seq_length,
            return_dense=False, return_sparse=False, return_colbert_vecs=True,
        )
        colbert = output["colbert_vecs"]
        if isinstance(colbert, list) and len(colbert) > 0:
            vecs = colbert[0]
            if isinstance(vecs, np.ndarray):
                return vecs.astype(np.float32).tolist()
            return vecs.cpu().float().numpy().tolist()
        return []

    def get_query_sparse_vs_doc_dense(self, query: str, doc_text: str) -> float:
        """计算 query 词法权重 × doc 词法权重的内积得分。"""
        assert self.model is not None

        output = self.model.encode(
            [query, doc_text], batch_size=2, max_length=self.max_seq_length,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        lexical = output.get("lexical_weights")
        if not lexical or len(lexical) < 2:
            # 降级：用 dense cosine 相似度
            dv = output["dense_vecs"]
            if isinstance(dv, np.ndarray):
                dv = dv.astype(np.float32)
            else:
                dv = dv.cpu().float().numpy()
            qv, docv = dv[0], dv[1]
            q_norm = np.linalg.norm(qv) + 1e-8
            d_norm = np.linalg.norm(docv) + 1e-8
            return float(np.dot(qv / q_norm, docv / d_norm))

        q_weights: Dict[str, float] = lexical[0] if isinstance(lexical, list) else lexical  # type: ignore[arg-type]
        d_weights: Dict[str, float] = lexical[1] if isinstance(lexical, list) else lexical  # type: ignore[arg-type]

        score = 0.0
        for token_str, qw in q_weights.items():
            dw = d_weights.get(token_str, 0.0)
            if dw > 0:
                score += qw * dw
        return score

    def colbert_rerank(
        self,
        query_text: str,
        doc_texts: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """ColBERT 式 MaxSim reranking（与 UnslothEmbeddingModel 接口一致）。"""
        assert self.model is not None

        q_vecs = self.get_multi_vector(query_text)
        if not q_vecs:
            return [(i, 0.0) for i in range(len(doc_texts))]

        q_tensor = torch.tensor(q_vecs, dtype=torch.float32)

        scores = []
        for i, doc_text in enumerate(doc_texts):
            d_vecs = self.get_multi_vector(doc_text)
            if not d_vecs:
                scores.append(0.0)
                continue
            d_tensor = torch.tensor(d_vecs, dtype=torch.float32)
            sim = torch.matmul(q_tensor, d_tensor.T)
            scores.append(sim.max(dim=1).values.sum().item())

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(idx, scores[idx]) for idx in ranked[:top_k]]

    async def encode_async(self, texts: Union[str, List[str]]) -> List[List[float]]:
        return await asyncio.to_thread(self.get_dense_embedding, texts)


# ============================================================================
# 全局单例
# ============================================================================

_flag_model_instance: Optional[FlagEmbeddingModel] = None


def get_flag_model(
    model_path: str = "",
    device: str = "cpu",
    max_seq_length: int = 512,
    use_fp16: bool = True,
) -> FlagEmbeddingModel:
    global _flag_model_instance
    if _flag_model_instance is None:
        _flag_model_instance = FlagEmbeddingModel(
            model_path=model_path,
            device=device,
            max_seq_length=max_seq_length,
            use_fp16=use_fp16,
        )
    return _flag_model_instance


async def init_flag_model(
    model_path: str = "",
    device: str = "cpu",
    max_seq_length: int = 512,
    use_fp16: bool = True,
) -> FlagEmbeddingModel:
    model = get_flag_model(model_path, device, max_seq_length, use_fp16)
    await model.initialize()
    return model


def reset_flag_model() -> None:
    global _flag_model_instance
    _flag_model_instance = None
    logger.info("[FlagEmbedding] 单例已重置")
