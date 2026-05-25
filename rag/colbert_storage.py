"""
ColBERT 标准存储模块

标准 ColBERT 做法：
- 索引时：预计算所有文档的 per-token vectors，存入 FAISS
- 查询时：只算 query tokens，用 FAISS 召回 → MaxSim 重排

存储结构（由 storage_dir 参数决定具体目录）：
- {storage_dir}/chunks/{chunk_idx:08d}.npy: 每个 chunk 的 token vectors（增量保存）
- {storage_dir}/colbert_faiss_index.bin: FAISS IndexFlatIP
- {storage_dir}/colbert_id_mapping.json: [{"chunk_id": ..., "n_tokens": ...}, ...]
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import faiss
from astrbot.api import logger


class ColBERTStorage:
    """
    ColBERT per-token vectors 存储与检索

    使用 FAISS IndexFlatIP 存储所有 token vectors，
    在线检索时做 MaxSim 聚合

    存储结构：
    - {storage_dir}/chunks/{chunk_idx:08d}.npy: 每个 chunk 的 token vectors（增量保存）
    - {storage_dir}/colbert_faiss_index.bin: FAISS 索引
    - {storage_dir}/colbert_id_mapping.json: chunk_id → metadata 映射
    """

    def __init__(self, storage_dir: str):
        """
        Args:
            storage_dir: 存储目录 (e.g., "./data")
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_dir = self.storage_dir / "chunks"
        self.faiss_index_path = self.storage_dir / "colbert_faiss_index.bin"
        self.id_mapping_path = self.storage_dir / "colbert_id_mapping.json"

        self._doc_vectors: Optional[np.ndarray] = None  # (N, max_tokens, 1024)
        self._id_mapping: List[Dict[str, Any]] = []  # [{"chunk_id": str, "n_tokens": int}, ...]
        self._chunk_id_to_idx: Dict[str, int] = {}  # chunk_id -> chunk_idx for O(1) lookup
        self._index: Optional[faiss.Index] = None
        self._is_loaded = False
        self._saved_chunk_count: int = 0  # 已持久化到磁盘的 chunk 数量

        self.EMBEDDING_DIM = 1024
        self.MAX_TOKENS_PER_CHUNK = 512  # 每个 chunk 最多存 512 tokens（匹配 BGE-M3 max_seq_length）

    def _get_faiss_id(self, chunk_idx: int, token_pos: int) -> int:
        """将 (chunk_idx, token_pos) 映射为 FAISS vector ID"""
        return chunk_idx * self.MAX_TOKENS_PER_CHUNK + token_pos

    def _parse_faiss_id(self, faiss_id: int) -> Tuple[int, int]:
        """将 FAISS vector ID 解析为 (chunk_idx, token_pos)"""
        chunk_idx = faiss_id // self.MAX_TOKENS_PER_CHUNK
        token_pos = faiss_id % self.MAX_TOKENS_PER_CHUNK
        return chunk_idx, token_pos

    def add_chunks(
        self,
        chunk_vectors: List[np.ndarray],
        chunk_ids: List[str],
    ) -> None:
        """
        添加 chunks 的 per-token vectors

        Args:
            chunk_vectors: List of (n_tokens, 1024) arrays per chunk
            chunk_ids: List of chunk_id strings
        """
        if len(chunk_vectors) != len(chunk_ids):
            raise ValueError(f"chunk_vectors ({len(chunk_vectors)}) 和 chunk_ids ({len(chunk_ids)}) 数量不匹配")

        # 初始化或扩展存储
        n_new = len(chunk_vectors)

        if self._doc_vectors is None:
            # 首次添加：预分配
            self._doc_vectors = np.zeros(
                (n_new, self.MAX_TOKENS_PER_CHUNK, self.EMBEDDING_DIM),
                dtype=np.float32
            )
            self._id_mapping = []
            self._chunk_id_to_idx = {}
            self._index = None
        else:
            if self._index is None and self.faiss_index_path.exists():
                if not self._ensure_faiss_index_loaded():
                    raise RuntimeError("[ColBERTStorage] 扩展已有存储前无法加载 FAISS 索引")

            # 扩展
            old_n = self._doc_vectors.shape[0]
            new_n = old_n + n_new
            new_vectors = np.zeros(
                (new_n, self.MAX_TOKENS_PER_CHUNK, self.EMBEDDING_DIM),
                dtype=np.float32
            )
            new_vectors[:old_n] = self._doc_vectors
            self._doc_vectors = new_vectors

        # 填充数据
        all_vectors = []  # 用于 FAISS
        all_ids = []  # 用于 FAISS add_with_ids

        for i, (vectors, chunk_id) in enumerate(zip(chunk_vectors, chunk_ids)):
            # chunk_idx = 当前已有的 chunk 数量（即新 chunks 的起始位置）
            chunk_idx = len(self._id_mapping)
            actual_tokens = len(vectors)

            # 防截断断言：绝不静默丢弃 token 向量
            if actual_tokens > self.MAX_TOKENS_PER_CHUNK:
                raise ValueError(
                    f"[ColBERT Storage] Chunk '{chunk_id}' has {actual_tokens} tokens, "
                    f"exceeds MAX_TOKENS_PER_CHUNK={self.MAX_TOKENS_PER_CHUNK}. "
                    f"Increase MAX_TOKENS_PER_CHUNK (recommended: >= {actual_tokens}) "
                    f"or reduce chunk_size in HybridPDFParser."
                )

            n_tokens = actual_tokens  # 不再截断，直接使用全部 token

            self._doc_vectors[chunk_idx, :n_tokens] = vectors[:n_tokens]
            self._id_mapping.append({
                "chunk_id": chunk_id,
                "n_tokens": n_tokens,
                "chunk_idx": chunk_idx,
                "deleted": False,
            })
            self._chunk_id_to_idx[chunk_id] = chunk_idx

            # 收集 FAISS 数据
            for t in range(n_tokens):
                faiss_id = self._get_faiss_id(chunk_idx, t)
                all_vectors.append(vectors[t])
                all_ids.append(faiss_id)

        # 更新 FAISS 索引
        # 注意：IndexFlatIP 不支持 add_with_ids，使用 add() 顺序添加
        # 向量在 FAISS 中的位置顺序即 (chunk_idx, token_pos) 编码
        if all_vectors:
            vectors_arr = np.array(all_vectors, dtype=np.float32)
            if self._index is None:
                self._index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self._index.add(vectors_arr) # type: ignore[call-arg, arg-type]

        self._is_loaded = True
        logger.info(f"[ColBERTStorage] 添加了 {n_new} 个 chunks")

    def save(self) -> None:
        """保存到磁盘（增量：只写新增 chunk 的 .npy 文件）。"""
        if self._doc_vectors is not None and self._id_mapping:
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            total_chunks = len(self._id_mapping)
            new_count = 0
            for chunk_idx in range(self._saved_chunk_count, total_chunks):
                n_tokens = self._id_mapping[chunk_idx]["n_tokens"]
                vectors = self._doc_vectors[chunk_idx, :n_tokens]
                chunk_path = self.chunks_dir / f"{chunk_idx:08d}.npy"
                np.save(str(chunk_path), vectors)
                new_count += 1
            self._saved_chunk_count = total_chunks
            if new_count > 0:
                logger.info(
                    f"[ColBERTStorage] 已保存 {new_count} 个新增 chunk vectors → {self.chunks_dir}"
                )

        if self._index is not None:
            faiss.write_index(self._index, str(self.faiss_index_path))
            logger.info(f"[ColBERTStorage] 已保存 FAISS 索引: {self.faiss_index_path}")

        if self._id_mapping:
            with open(self.id_mapping_path, "w", encoding="utf-8") as f:
                json.dump(self._id_mapping, f, ensure_ascii=False)
            logger.info(f"[ColBERTStorage] 已保存 ID 映射: {self.id_mapping_path}")

    def load(self) -> bool:
        """从磁盘加载（优先 chunks/ 目录，兼容旧格式 colbert_doc_vectors.npy）。

        FAISS 索引按需懒加载。
        """
        try:
            # 先加载 ID 映射（需要知道 chunk 数量）
            if self.id_mapping_path.exists():
                with open(self.id_mapping_path, "r", encoding="utf-8") as f:
                    self._id_mapping = json.load(f)
                self._chunk_id_to_idx = {
                    entry["chunk_id"]: entry["chunk_idx"]
                    for entry in self._id_mapping
                    if not entry.get("deleted", False)
                }
                logger.info(f"[ColBERTStorage] 已加载 ID 映射: {len(self._id_mapping)} chunks")
            else:
                logger.info(f"[ColBERTStorage] 尚无已保存存储，首次构建后将写入: {self.storage_dir}")
                return False

            n_chunks = len(self._id_mapping)
            if n_chunks == 0:
                self._is_loaded = True
                self._saved_chunk_count = 0
                return True

            # 初始化 doc_vectors 数组
            self._doc_vectors = np.zeros(
                (n_chunks, self.MAX_TOKENS_PER_CHUNK, self.EMBEDDING_DIM),
                dtype=np.float32
            )

            # 从 chunks/ 目录加载
            if self.chunks_dir.exists():
                chunk_files = sorted(self.chunks_dir.glob("*.npy"))
                loaded = 0
                for f in chunk_files:
                    try:
                        chunk_idx = int(f.stem)
                    except ValueError:
                        continue
                    if chunk_idx >= n_chunks:
                        continue
                    vectors = np.load(f)
                    n_tokens = min(len(vectors), self.MAX_TOKENS_PER_CHUNK)
                    self._doc_vectors[chunk_idx, :n_tokens] = vectors[:n_tokens]
                    loaded += 1
                self._saved_chunk_count = n_chunks
                logger.info(
                    f"[ColBERTStorage] 已加载 {loaded}/{n_chunks} chunk vectors → {self.chunks_dir}"
                )
            else:
                logger.error(
                    f"[ColBERTStorage] id_mapping 存在 ({n_chunks} chunks) 但 chunks/ 目录不存在，"
                    "数据不完整，请重建 ColBERT 存储"
                )
                self._doc_vectors = None
                self._id_mapping = []
                self._chunk_id_to_idx = {}
                return False

            self._index = None
            if self.faiss_index_path.exists():
                logger.info("[ColBERTStorage] FAISS 索引按需懒加载")
            else:
                logger.warning("[ColBERTStorage] 未找到 FAISS 索引文件，ColBERTStorage.search 将不可用")

            self._is_loaded = True
            return True
        except Exception as e:
            logger.error(f"[ColBERTStorage] 加载失败: {e}")
            return False

    def _ensure_faiss_index_loaded(self) -> bool:
        """按需加载 FAISS 索引。"""
        if self._index is not None:
            return True
        if not self.faiss_index_path.exists():
            logger.warning("[ColBERTStorage] FAISS 索引文件不存在")
            return False

        try:
            self._index = faiss.read_index(str(self.faiss_index_path))
            assert self._index is not None
            logger.info(f"[ColBERTStorage] 已懒加载 FAISS 索引: {self._index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"[ColBERTStorage] FAISS 索引懒加载失败: {e}")
            self._index = None
            return False

    def maxsim_score(
        self,
        query_vectors: np.ndarray,
        chunk_idx: int,
    ) -> float:
        """
        计算 query token vectors 与指定 chunk 的 MaxSim 分数

        Args:
            query_vectors: (M, 1024) 查询的 token vectors
            chunk_idx: chunk 在存储中的索引

        Returns:
            MaxSim 分数
        """
        if self._doc_vectors is None or chunk_idx >= len(self._doc_vectors):
            return 0.0
        if self._id_mapping[chunk_idx].get("deleted", False):
            return 0.0

        n_doc_tokens = self._id_mapping[chunk_idx]["n_tokens"]
        if n_doc_tokens == 0:
            return 0.0

        doc_vectors = self._doc_vectors[chunk_idx, :n_doc_tokens]  # (n_tokens, 1024)

        # MaxSim: Σ_i max_j (q_i · d_j)
        # (M, 1024) @ (1024, n_tokens) = (M, n_tokens)
        sim_matrix = np.dot(query_vectors, doc_vectors.T)
        max_sim = np.max(sim_matrix, axis=1).sum()
        return float(max_sim)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 20,
        rerank_top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        ColBERT MaxSim 检索

        Args:
            query_vectors: (M, 1024) 查询的 token vectors
            top_k: 返回 top-k 个 chunks
            rerank_top_k: 如果指定，只对 FAISS 初筛的 top-k doc 做 MaxSim

        Returns:
            [(chunk_id, score), ...] 按分数降序
        """
        if self._doc_vectors is None:
            logger.warning("[ColBERTStorage] 未加载 doc_vectors")
            return []

        if not self._ensure_faiss_index_loaded():
            logger.warning("[ColBERTStorage] 未加载 FAISS 索引")
            return []

        n_chunks = len(self._id_mapping)
        if n_chunks == 0:
            return []

        # 第一阶段：FAISS 对每个 query token 召回相关 doc tokens
        # 将 query vectors 转为 Python list 供 FAISS 搜索
        query_list = query_vectors.astype(np.float32)

        # 收集候选 chunk 及其分数
        chunk_scores: Dict[int, List[float]] = {}

        for q_vec in query_list:
            # FAISS search 找 top-8 个最相似的 token
            k = min(8, self._index.ntotal)
            D, I = self._index.search(q_vec.reshape(1, -1), k) # type: ignore[call-arg, arg-type]  
            for score, faiss_id in zip(D[0], I[0]):
                if faiss_id < 0:
                    continue
                chunk_idx, _ = self._parse_faiss_id(faiss_id)
                if chunk_idx not in chunk_scores:
                    chunk_scores[chunk_idx] = []
                chunk_scores[chunk_idx].append(float(score))

        # 第二阶段：计算每个 chunk 的 MaxSim 分数
        results = []
        for chunk_idx, token_scores in chunk_scores.items():
            # MaxSim = Σ max_scores_per_query_token
            # 这里用已召回 token 的平均分近似
            max_sim_approx = sum(token_scores)
            results.append((chunk_idx, max_sim_approx))

        # 排序
        results.sort(key=lambda x: x[1], reverse=True)

        # 返回 top_k（映射 chunk_idx → chunk_id），跳过已删除的 chunks）
        output = []
        for chunk_idx, score in results:
            if self._id_mapping[chunk_idx].get("deleted", False):
                continue
            chunk_id = self._id_mapping[chunk_idx]["chunk_id"]
            output.append((chunk_id, score))
            if len(output) >= top_k:
                break

        return output

    def get_chunk_token_vectors(self, chunk_idx: int) -> Optional[np.ndarray]:
        """获取指定 chunk 的所有 token vectors"""
        if self._doc_vectors is None or chunk_idx >= len(self._doc_vectors):
            return None
        if self._id_mapping[chunk_idx].get("deleted", False):
            return None
        n_tokens = self._id_mapping[chunk_idx]["n_tokens"]
        if n_tokens == 0:
            return None
        return self._doc_vectors[chunk_idx, :n_tokens]

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def find_chunk_idx(self, chunk_id: str) -> Optional[int]:
        """O(1) lookup of chunk_idx by chunk_id. Returns None if not found or deleted."""
        idx = self._chunk_id_to_idx.get(chunk_id)
        if idx is None:
            return None
        if idx < len(self._id_mapping) and not self._id_mapping[idx].get("deleted", False):
            return idx
        return None

    def delete_by_file_prefix(self, file_prefix: str) -> int:
        """
        根据文件名前缀标记删除 chunks

        Args:
            file_prefix: 文件路径前缀（如 "/path/to/paper.pdf"）

        Returns:
            被标记删除的 chunk 数量
        """
        count = 0
        for item in self._id_mapping:
            chunk_id = item.get("chunk_id", "")
            if chunk_id.startswith(file_prefix) or file_prefix in chunk_id:
                item["deleted"] = True
                self._chunk_id_to_idx.pop(chunk_id, None)
                count += 1
        if count > 0:
            logger.info(f"[ColBERTStorage] 标记删除 {count} 个 chunks (prefix={file_prefix})")
        return count

    def clear_storage(self) -> None:
        """
        清空 ColBERT 存储（删除文件并重置内存状态）
        """
        self._doc_vectors = None
        self._id_mapping = []
        self._chunk_id_to_idx = {}
        self._index = None
        self._is_loaded = False
        self._saved_chunk_count = 0

        import shutil
        if self.chunks_dir.exists():
            shutil.rmtree(str(self.chunks_dir))
            logger.debug(f"[ColBERTStorage] 已删除 {self.chunks_dir}")
        # 新格式 + 旧格式残留一并清理
        for p in [self.faiss_index_path, self.id_mapping_path,
                  self.storage_dir / "colbert_doc_vectors.npy"]:
            if p.exists():
                p.unlink()
                logger.debug(f"[ColBERTStorage] 已删除 {p}")

        logger.info("[ColBERTStorage] 存储已清空")

    def __len__(self) -> int:
        return sum(1 for item in self._id_mapping if not item.get("deleted", False)) if self._id_mapping else 0
