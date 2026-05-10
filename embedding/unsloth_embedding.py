"""
Unsloth Embedding Provider for PaperRAG

使用 Unsloth FastSentenceTransformer 本地加载 BGE-M3 模型，获取：
1. 稠密向量 (dense vector) - 用于 Milvus 向量检索
2. 稀疏权重 (sparse weight) - 替代 BM25 关键词检索
3. 多向量序列 (multi-vector) - 用于 ColBERT 式 late-interaction reranking

参照 BGE_M3.ipynb 实现，模型下载到 ./models/bge-m3/
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np

from astrbot.api import logger
import platform
import concurrent.futures
from transformers import AutoTokenizer, AutoModel

# ============================================================================
# ABSPEC 公式实现
# ============================================================================

def compute_abscore_sparse(
    hidden_states: torch.Tensor,
    query_embedding: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> Dict[int, float]:
    """
    使用 ABSPEC 公式计算稀疏权重

    ABSPEC (Attention-Based Sparse Representation for Lexical Matching):
    w_i = ||h_i|| * exp(h_i · e_query / T) / Σ exp(h_j · e_query / T)

    其中：
    - h_i 是第 i 个 token 的 hidden state
    - e_query 是查询的稠密向量（mean pooling 后的 query embedding）
    - T 是温度参数（temperature）
    - ||h_i|| 是 hidden state 的 L2 范数

    这个公式结合了语义相似度（exp 部分）和词汇重要性（||h_i|| 部分），
    能够识别查询中的关键词及其重要性。

    Args:
        hidden_states: (num_tokens, hidden_dim) token hidden states
        query_embedding: (hidden_dim,) 查询的稠密向量
        temperature: 温度参数，控制分布平滑度

    Returns:
        Dict[int, float]: token_id -> weight 稀疏权重字典
    """
    if hidden_states is None or len(hidden_states) == 0:
        return {}

    # 计算每个 token 的 L2 范数
    norms = torch.norm(hidden_states, p=2, dim=-1)  # (num_tokens,)

    if query_embedding is not None:
        # 计算注意力分数：h_i · e_query
        # 确保维度匹配
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # (1, hidden_dim)

        attention_scores = torch.matmul(hidden_states, query_embedding.T).squeeze(-1)  # (num_tokens,)
        attention_scores = attention_scores / temperature

        # 归一化：exp(h_i · e_query) / Σ exp(h_j · e_query)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (num_tokens,)

        # 最终权重：||h_i|| * attention_weight
        sparse_weights = norms * attention_weights
    else:
        # 如果没有 query embedding，直接使用 L2 范数归一化
        sparse_weights = norms / (norms.sum() + 1e-8)

    # 转换为 Python dict（token_id -> weight）
    sparse_dict: Dict[int, float] = {}
    for i, w in enumerate(sparse_weights.tolist()):
        if w > 1e-6:  # 过滤掉太小的权重
            sparse_dict[i] = w

    return sparse_dict


def compute_multi_vector(
    hidden_states: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> List[List[float]]:
    """
    提取多向量序列（用于 ColBERT 式 late-interaction reranking）

    ColBERT 风格：保留每个 token 的向量表示，通过 max-sim 计算相关性

    Args:
        hidden_states: (num_tokens, hidden_dim) token hidden states
        mask: (num_tokens,) mask 数组，1=有效token，0=padding

    Returns:
        List[List[float]]: 每个有效 token 的向量列表
    """
    if hidden_states is None or len(hidden_states) == 0:
        return []

    # 移除 [CLS] 和 [SEP]（通常是第一个和最后一个）
    # BGE-M3 使用 [CLS] token 做句子级 embedding
    if len(hidden_states) > 2:
        hidden_states = hidden_states[1:-1]  # 移除首尾
        if mask is not None:
            mask = mask[1:-1]

    # 应用 mask（如果提供）
    if mask is not None:
        valid_indices = mask.bool()
        hidden_states = hidden_states[valid_indices]

    # L2 归一化（ColBERT 风格）
    norms = torch.norm(hidden_states, p=2, dim=-1, keepdim=True)
    normalized = hidden_states / (norms + 1e-8)

    # 转换为 Python list
    return normalized.cpu().tolist()


# ============================================================================
# UnslothEmbeddingModel 单例
# ============================================================================

class UnslothEmbeddingModel:
    """
    Unsloth BGE-M3 Embedding 模型

    提供三种输出：
    1. 稠密向量 - model.encode() / mean pooling
    2. 稀疏权重 - ABSPEC 公式
    3. 多向量序列 - per-token hidden states
    """

    _instance: Optional["UnslothEmbeddingModel"] = None
    _lock = asyncio.Lock()

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        max_seq_length: int = 512,
    ):
        """
        Args:
            model_path: 模型路径（如 ./models/bge-m3）
            device: 运行设备 ("mps", "cuda", "cpu")
            max_seq_length: 最大序列长度
        """
        self.model_path = model_path
        self.device = device
        self.max_seq_length = max_seq_length

        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._embedding_dim: Optional[int] = None
        # Apple Silicon 使用 transformers 直连，不走 Unsloth（Unsloth 不支持 MPS）
        self._using_transformers_direct: bool = False
        # BGE-M3 专用投影层
        self._sparse_linear: Optional[Dict[str, torch.Tensor]] = None
        self._colbert_linear: Optional[Dict[str, torch.Tensor]] = None

    def _get_transformer_model(self):
        """
        获取底层 transformer 模型

        FastSentenceTransformer: self.model.model 是 XLMRobertaModel
        AutoModel: self.model 直接是 transformer 模型
        """
        assert self.model is not None
        if self._using_transformers_direct:
            return self.model
        else:
            return self.model.model

    def _get_plugin_models_dir(self) -> Path:
        """获取插件的 models 目录"""
        # __file__ = .../astrbot_plugin_paperrag/embedding/unsloth_embedding.py
        # .parent = .../astrbot_plugin_paperrag/embedding/
        # .parent.parent = .../astrbot_plugin_paperrag/
        return Path(__file__).parent.parent.resolve() / "models"

    def _ensure_model_downloaded(self) -> str:
        """确保模型已下载，不存在则自动下载"""
        model_dir = self._get_plugin_models_dir() / "bge-m3"
        model_path = str(model_dir)

        # 检查必要文件
        config_file = model_dir / "config.json"
        tokenizer_file = model_dir / "tokenizer.json"

        if not config_file.exists() or not tokenizer_file.exists():
            logger.info(f"[UnslothEmbedding] 模型不存在，正在下载...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="unsloth/bge-m3",
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
            logger.info(f"[UnslothEmbedding] 模型下载完成: {model_path}")

        return model_path

    def _is_apple_silicon(self) -> bool:
        """检测是否在 Apple Silicon 上运行"""
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def _load_model(self) -> None:
        """加载模型（在独立线程中执行）"""

        def _load():

            # 确保模型已下载
            actual_model_path = self._ensure_model_downloaded()

            logger.info(f"[UnslothEmbedding] 加载模型: {actual_model_path}")
            logger.info(f"[UnslothEmbedding] 设备: {self.device}, max_seq_length: {self.max_seq_length}")

            # Apple Silicon: 使用 transformers 直连（Unsloth 不支持 MPS）
            # NVIDIA/AMD/Intel: 使用 Unsloth（性能更好）
            if self._is_apple_silicon():
                logger.info("[UnslothEmbedding] Apple Silicon 检测到，使用 transformers + MPS")
                self._using_transformers_direct = True
                tokenizer = AutoTokenizer.from_pretrained(actual_model_path, local_files_only=True)
                model = AutoModel.from_pretrained(actual_model_path, local_files_only=True)
            else:
                from unsloth import FastSentenceTransformer

                self._using_transformers_direct = False
                model = FastSentenceTransformer.from_pretrained(
                    model_name=actual_model_path,
                    max_seq_length=self.max_seq_length,
                )
                tokenizer = model.tokenizer

            # 设置设备
            if self.device == "mps" and torch.backends.mps.is_available():
                model = model.to("mps")
                logger.info("[UnslothEmbedding] 使用 MPS (Apple Silicon)")
            elif self.device == "cuda" and torch.cuda.is_available():
                model = model.to("cuda")
                logger.info("[UnslothEmbedding] 使用 CUDA")
            else:
                logger.info("[UnslothEmbedding] 使用 CPU")

            # 获取 embedding 维度
            sample_input = ["test"]
            with torch.no_grad():
                if self._using_transformers_direct:
                    inputs = tokenizer(
                        sample_input,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_seq_length,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs["attention_mask"]
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    # L2 归一化
                    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
                    embeddings = embeddings / (norms + 1e-8)
                else:
                    embeddings = model.encode(sample_input)
            self._embedding_dim = embeddings.shape[1]
            logger.info(f"[UnslothEmbedding] Embedding 维度: {self._embedding_dim}")

            # 加载 BGE-M3 专用投影层
            model_dir = self._ensure_model_downloaded()
            sparse_state = torch.load(Path(model_dir) / "sparse_linear.pt", weights_only=False)
            colbert_state = torch.load(Path(model_dir) / "colbert_linear.pt", weights_only=False)
            self._sparse_linear = {k: v.to(model.device) for k, v in sparse_state.items()}
            self._colbert_linear = {k: v.to(model.device) for k, v in colbert_state.items()}
            logger.info("[UnslothEmbedding] 投影层已加载 (sparse_linear, colbert_linear)")

            return model, tokenizer

        # 在线程池中加载模型
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_load)
            self.model, self.tokenizer = future.result()

        self._initialized = True
        logger.info("[UnslothEmbedding] 模型加载完成")

    async def initialize(self) -> None:
        """异步初始化模型"""
        if self._initialized:
            return

        async with self._lock:
            # 双重检查
            if self._initialized:
                return

            logger.info("[UnslothEmbedding] 开始初始化...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            logger.info(f"[UnslothEmbedding] 初始化完成，维度: {self._embedding_dim}")

    def get_dense_embedding(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        获取稠密向量（用于 Milvus 检索）

        Args:
            texts: 单个文本或文本列表

        Returns:
            List[List[float]]: 归一化的稠密向量列表
        """
        if not self._initialized:
            raise RuntimeError("模型未初始化，请先调用 initialize()")

        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            assert self.model is not None
            if self._using_transformers_direct:
                # AutoModel: 手动 mean pooling + L2 归一化
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
                attention_mask = inputs["attention_mask"]

                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
                embeddings = sum_embeddings / sum_mask
                # L2 归一化
                norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
                embeddings = embeddings / (norms + 1e-8)
            else:
                # FastSentenceTransformer: 直接用 encode
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

        return embeddings.cpu().tolist()

    def get_sparse_weight(self, text: str, query_embedding: Optional[List[float]] = None) -> Dict[int, float]:
        """
        获取稀疏权重（用于关键词检索，替代 BM25）

        ABSPEC 公式: w_i = ||h_i|| * exp(h_i · e_query / T) / Σ exp(h_j · e_query / T)

        Args:
            text: 文本字符串
            query_embedding: 可选的查询向量，用于计算注意力权重

        Returns:
            Dict[int, float]: token_id -> weight 稀疏权重
        """
        if not self._initialized:
            raise RuntimeError("模型未初始化，请先调用 initialize()")

        # Tokenize
        assert self.tokenizer is not None
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            return_attention_mask=True,
        )

        assert self.model is not None
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # 获取 token 序列
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Forward 获取 hidden states
        with torch.no_grad():
            # 获取模型输出
            transformer = self._get_transformer_model()
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # 获取最后一层 hidden states
            # hidden_states 形状: (batch, seq_len, hidden_dim)
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs.last_hidden_state

            # 移除 batch 维度
            hidden_states = hidden_states[0]  # (seq_len, hidden_dim)

        # 使用 sparse_linear 投影层计算每个 token 的稀疏分数
        # sparse_linear: [1, 1024] @ [1024] + [1] → [seq_len]
        assert self._sparse_linear is not None
        w = self._sparse_linear["weight"].to(hidden_states.dtype)
        b = self._sparse_linear["bias"].to(hidden_states.dtype)
        sparse_scores = torch.matmul(hidden_states, w.T).squeeze(-1)
        sparse_scores = sparse_scores + b.squeeze(-1)
        # sparse_scores: (seq_len,)

        if query_embedding is not None:
            # ABSPEC 注意力加权：结合语义相似度和词汇重要性
            query_emb = torch.tensor(
                query_embedding,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            # 注意力分数：h_i · e_query
            attention_scores = torch.matmul(hidden_states, query_emb).squeeze(-1)
            attention_scores = attention_scores / 1.0  # temperature=1.0
            attention_weights = torch.softmax(attention_scores, dim=-1)
            # 最终权重：sparse_score * attention_weight
            final_weights = sparse_scores * attention_weights
        else:
            final_weights = sparse_scores

        # 过滤掉特殊 token（[CLS], [SEP], [PAD] 等）
        special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}
        filtered_weights: Dict[int, float] = {}
        for token_id, weight in enumerate(final_weights.tolist()):
            if token_id < len(tokens):
                token = tokens[token_id]
                if token not in special_tokens:
                    filtered_weights[token_id] = float(weight)

        return filtered_weights

    def get_sparse_embedding(self, text: str, query_embedding: Optional[List[float]] = None) -> Dict[int, float]:
        """
        获取稀疏权重的别名（兼容接口）

        Args:
            text: 文本字符串
            query_embedding: 可选的查询向量

        Returns:
            Dict[int, float]: token_id -> weight
        """
        return self.get_sparse_weight(text, query_embedding)

    def get_multi_vector(self, text: str) -> List[List[float]]:
        """
        获取多向量序列（用于 ColBERT 式 reranking）

        Args:
            text: 文本字符串

        Returns:
            List[List[float]]: 每个 token 的归一化向量
        """
        if not self._initialized:
            raise RuntimeError("模型未初始化，请先调用 initialize()")

        # Tokenize
        assert self.tokenizer is not None
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            return_attention_mask=True,
        )

        assert self.model is not None
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Forward 获取 hidden states
        with torch.no_grad():
            transformer = self._get_transformer_model()
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs.last_hidden_state

            # 移除 batch 维度
            hidden_states = hidden_states[0]  # (seq_len, hidden_dim)

            # 转换为 mask tensor
            mask = attention_mask[0]

        # 使用 colbert_linear 投影层得到 ColBERT 风格的多向量
        # colbert_linear: [1024, 1024] @ [1024, 1] + [1024] → [1024, 1]
        assert self._colbert_linear is not None
        colbert_vecs = torch.matmul(
            hidden_states,
            self._colbert_linear["weight"].to(hidden_states.dtype)
        ) + self._colbert_linear["bias"].to(hidden_states.dtype)
        # colbert_vecs: (seq_len, hidden_dim)

        # 应用 attention mask 并 L2 归一化（ColBERT 风格）
        mask_expanded = mask.unsqueeze(-1).expand(colbert_vecs.size()).float()
        colbert_vecs = colbert_vecs * mask_expanded
        colbert_norms = torch.norm(colbert_vecs, p=2, dim=-1, keepdim=True)
        colbert_vecs = colbert_vecs / (colbert_norms + 1e-8)

        # 移除 [CLS] 和 [SEP] token（保留中间的实际内容 token）
        # mask 是原始 1D attention mask tensor，shape: (seq_len,)
        mask_1d = mask
        if len(colbert_vecs) > 2:
            colbert_vecs = colbert_vecs[1:-1]
            mask_1d = mask_1d[1:-1]

        # 只保留有效 token（mask_1d 仍为 1D）
        valid_mask = mask_1d.bool()
        colbert_vecs = colbert_vecs[valid_mask]

        return colbert_vecs.cpu().tolist()

    def get_query_sparse_vs_doc_dense(
        self,
        query_text: str,
        doc_text: str,
    ) -> float:
        """
        计算查询稀疏权重与文档稠密向量的相关性分数

        用于 Sparse Retrieval：将查询的稀疏表示与文档的稠密向量进行匹配

        公式: score = Σ max_i(w_query[i] * sim(h_i_doc, e_query))

        其中：
        - w_query[i] 是查询第 i 个 token 的 ABSPEC 权重
        - h_i_doc 是文档第 i 个 token 的 hidden state
        - e_query 是查询的 mean pooling 向量

        Args:
            query_text: 查询文本
            doc_text: 文档文本

        Returns:
            float: 相关性分数
        """
        # 获取查询的稀疏权重和稠密向量
        query_embedding = self.get_dense_embedding([query_text])[0]
        query_sparse = self.get_sparse_weight(query_text, query_embedding)

        if not query_sparse:
            return 0.0

        # Tokenize 文档
        assert self.tokenizer is not None
        inputs = self.tokenizer(
            doc_text,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            return_attention_mask=True,
        )

        assert self.model is not None
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Forward
        with torch.no_grad():
            transformer = self._get_transformer_model()
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs.last_hidden_state

            hidden_states = hidden_states[0]  # (seq_len, hidden_dim)

        # 将查询向量转换为 tensor
        query_emb_tensor = torch.tensor(query_embedding, dtype=hidden_states.dtype, device=hidden_states.device)

        # 计算每个 token 与查询向量的相似度
        # sim(h_i, e_query) = h_i · e_query / (||h_i|| * ||e_query||)
        norms_doc = torch.norm(hidden_states, p=2, dim=-1)
        norms_query = torch.norm(query_emb_tensor, p=2) + 1e-8

        # 归一化文档向量
        normalized_doc = hidden_states / norms_doc.unsqueeze(-1)
        normalized_query = query_emb_tensor / norms_query

        # 计算相似度 (num_tokens,)
        similarities = torch.matmul(normalized_doc, normalized_query)

        # 加权求和
        score = 0.0
        for token_id, weight in query_sparse.items():
            if token_id < len(similarities):
                score += weight * similarities[token_id].item()

        return score

    def colbert_rerank(
        self,
        query_text: str,
        doc_texts: List[str],
        top_k: int = 5,
    ) -> List[tuple[int, float]]:
        """
        ColBERT 式 reranking

        ColBERT 风格 late-interaction reranking：
        1. 查询和文档都编码为多向量序列
        2. 计算 max-sim: max_j(sim(q_i, d_j))
        3. 累加所有查询 token 的 max-sim 分数

        Args:
            query_text: 查询文本
            doc_texts: 文档文本列表
            top_k: 返回 top-k 结果

        Returns:
            List[tuple[int, float]]: (doc_index, score) 按分数降序排列
        """
        if not self._initialized:
            raise RuntimeError("模型未初始化，请先调用 initialize()")

        # 获取查询的多向量
        query_vectors = self.get_multi_vector(query_text)
        if not query_vectors:
            return [(i, 0.0) for i in range(len(doc_texts))]

        query_tensor = torch.tensor(query_vectors, dtype=torch.float32)

        # 处理每个文档
        scores = []
        for i, doc_text in enumerate(doc_texts):
            doc_vectors = self.get_multi_vector(doc_text)
            if not doc_vectors:
                scores.append(0.0)
                continue

            doc_tensor = torch.tensor(doc_vectors, dtype=torch.float32)

            # 计算 max-sim
            # query_tensor: (num_query_tokens, dim)
            # doc_tensor: (num_doc_tokens, dim)
            # sim_matrix: (num_query_tokens, num_doc_tokens)
            sim_matrix = torch.matmul(query_tensor, doc_tensor.T)

            # max-sim: 对每个查询 token 取最大值，然后求和
            max_sim = sim_matrix.max(dim=1).values.sum().item()
            scores.append(max_sim)

        # 按分数降序排列
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        return [(idx, scores[idx]) for idx in sorted_indices[:top_k]]

    @property
    def embedding_dim(self) -> int:
        """获取 embedding 维度"""
        if self._embedding_dim is None:
            # 尝试从模型推断
            return 1024  # BGE-M3 默认 1024
        return self._embedding_dim

    async def encode_async(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """异步编码接口"""
        return self.get_dense_embedding(texts)


# ============================================================================
# 单例管理
# ============================================================================

_embedding_instance: Optional[UnslothEmbeddingModel] = None


def get_embedding_model(
    model_path: Optional[str] = None,
    device: str = "mps",
    max_seq_length: int = 512,
) -> UnslothEmbeddingModel:
    """
    获取 UnslothEmbeddingModel 单例

    Args:
        model_path: 模型路径，默认使用 ./models/bge-m3
        device: 运行设备 ("mps", "cuda", "cpu")
        max_seq_length: 最大序列长度

    Returns:
        UnslothEmbeddingModel 单例
    """
    global _embedding_instance

    if _embedding_instance is None:
        if model_path is None:
            plugin_dir = Path(__file__).parent.parent
            model_path = str(plugin_dir / "models" / "bge-m3")

        _embedding_instance = UnslothEmbeddingModel(
            model_path=model_path,
            device=device,
            max_seq_length=max_seq_length,
        )

    return _embedding_instance


async def init_embedding_model(
    model_path: Optional[str] = None,
    device: str = "mps",
    max_seq_length: int = 512,
) -> UnslothEmbeddingModel:
    """
    初始化并获取 UnslothEmbeddingModel 单例

    Args:
        model_path: 模型路径
        device: 运行设备
        max_seq_length: 最大序列长度

    Returns:
        初始化完成的 UnslothEmbeddingModel
    """
    model = get_embedding_model(model_path, device, max_seq_length)
    await model.initialize()
    return model


def reset_embedding_model() -> None:
    """重置单例（用于重新加载模型）"""
    global _embedding_instance
    _embedding_instance = None
    logger.info("[UnslothEmbedding] 单例已重置")


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=== Unsloth BGE-M3 Embedding 测试 ===\n")

        # 初始化模型
        print("1. 初始化模型...")
        model = await init_embedding_model()
        print(f"   Embedding 维度: {model.embedding_dim}\n")

        # 测试稠密向量
        print("2. 测试稠密向量...")
        texts = ["Hello, world!", "This is a test."]
        dense = model.get_dense_embedding(texts)
        print(f"   输入: {len(texts)} 个文本")
        print(f"   输出: {len(dense)} 个向量, 维度: {len(dense[0])}\n")

        # 测试稀疏权重
        print("3. 测试稀疏权重...")
        query = "What is attention mechanism?"
        sparse = model.get_sparse_weight(query)
        print(f"   查询: '{query}'")
        print(f"   稀疏权重: {len(sparse)} 个 token 有权重")
        print(f"   前5个: {dict(list(sparse.items())[:5])}\n")

        # 测试多向量
        print("4. 测试多向量...")
        doc = "Attention mechanism is a key component of transformer models."
        multi_vec = model.get_multi_vector(doc)
        print(f"   文档: '{doc}'")
        print(f"   多向量数量: {len(multi_vec)}")
        print(f"   向量维度: {len(multi_vec[0]) if multi_vec else 0}\n")

        # 测试 ColBERT reranking
        print("5. 测试 ColBERT reranking...")
        query = "What is attention?"
        docs = [
            "Attention is a mechanism used in neural networks.",
            "The weather is nice today.",
            "Transformers use self-attention to process sequences.",
        ]
        results = model.colbert_rerank(query, docs, top_k=3)
        print(f"   查询: '{query}'")
        for idx, score in results:
            print(f"   [{score:.4f}] {docs[idx][:50]}...\n")

        print("=== 测试完成 ===")

    asyncio.run(test())
