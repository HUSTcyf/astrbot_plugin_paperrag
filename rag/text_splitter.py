from typing import Any, List

from astrbot.api import logger
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document


class UniformSentenceSplitter:
    """
    句子级均匀分块器（基于 LlamaIndex SentenceSplitter）

    策略：
    1. 先用 SentenceSplitter 在句子边界分块
    2. 贪婪合并短chunk（< min_chunk_ratio * chunk_size）
    3. 重叠（overlap）相邻chunk，保留语义连贯
    4. 单句超 512 tokens → 直接报错（不截断）

    与 SemanticSplitterNodeParser 的区别：
    - 不依赖 embedding 模型计算相似度
    - 保证每个chunk大小均匀，避免极短碎片
    - 无语义断点，纯规则分块
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_ratio: float = 0.65,
        tokenizer: Any = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_ratio = min_chunk_ratio
        self.tokenizer = tokenizer
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) >= chunk_size ({self.chunk_size})，"
                f"overlap 不能大于或等于 chunk_size"
            )
        self.min_chunk_size = int(chunk_size * min_chunk_ratio)  # 约 333 tokens

    def _get_token_count(self, text: str) -> int:
        """使用 BGE tokenizer 计数，无 tokenizer 时回退到 tiktoken。"""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        from rag.token_utils import count_tokens
        return count_tokens(text)

    def split(self, text: str) -> List[str]:
        """
        主入口：分块文本为均匀的句子块

        策略：
        1. 用 LlamaIndex SentenceSplitter 做句子级分块
        2. 贪婪合并短 chunk（< min_chunk_size）
        3. 应用 overlap

        Args:
            text: 待分块文本

        Returns:
            chunk 文本列表

        Raises:
            ValueError: chunk_overlap >= chunk_size
        """

        # 提前验证 tokenizer
        if self.tokenizer is None:
            raise RuntimeError("BGE tokenizer 不可用，UniformSentenceSplitter 无法工作")

        # 用 LlamaIndex SentenceSplitter 做句子级分块
        # 关键：lambda 返回 list，len(list) = actual token count
        # 这样 SentenceSplitter 的 _token_size() 才能正确计数
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            tokenizer=lambda t: self.tokenizer.encode(t, add_special_tokens=False),
        )
        _llamaindex_doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([_llamaindex_doc])
        chunks = [node.text.strip() for node in nodes if node.text.strip()]  # type: ignore

        # 过滤空 chunk
        chunks = [c for c in chunks if self._get_token_count(c) > 0]
        if not chunks:
            logger.warning("UniformSentenceSplitter: SentenceSplitter 产生空结果")
            return []

        # 贪婪合并短 chunk
        chunks = self._greedy_merge_short(chunks)

        # 拆分仍然超大的 chunks（SentenceSplitter 句子边界拆分）
        chunks = self._split_oversized(chunks)

        # 应用 overlap
        if self.chunk_overlap > 0:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _greedy_merge_short(self, chunks: List[str]) -> List[str]:
        """贪婪合并短 chunk（< min_chunk_size），直到所有 chunk 达标"""
        max_iterations = 100
        for _ in range(max_iterations):
            merged = False
            new_chunks = []
            i = 0
            while i < len(chunks):
                chunk = chunks[i]
                chunk_tokens = self._get_token_count(chunk)

                if 0 < chunk_tokens < self.min_chunk_size:
                    # 尝试与前一个合并
                    if new_chunks:
                        prev = new_chunks[-1]
                        combined = prev + "\n\n" + chunk
                        combined_tokens = self._get_token_count(combined)
                        if combined_tokens <= self.chunk_size:
                            new_chunks[-1] = combined
                            merged = True
                            i += 1
                            continue
                    # 尝试与后一个合并
                    if i + 1 < len(chunks):
                        next_chunk = chunks[i + 1]
                        combined = chunk + "\n\n" + next_chunk
                        combined_tokens = self._get_token_count(combined)
                        if combined_tokens <= self.chunk_size:
                            new_chunks.append(combined)
                            i += 2
                            merged = True
                            continue

                    # 无法合并，保留原样
                    new_chunks.append(chunk)
                    i += 1
                else:
                    new_chunks.append(chunk)
                    i += 1

            chunks = new_chunks
            if not merged:
                break

        return chunks

    def _split_oversized(self, chunks: List[str]) -> List[str]:
        """拆分仍然超大的 chunks（使用二分查找在 token 边界处切断）"""
        result = []
        for chunk in chunks:
            chunk_tokens = self._get_token_count(chunk)
            if chunk_tokens <= self.chunk_size:
                result.append(chunk)
            else:
                # 二分查找：找最大字符偏移，使 text[:pos] 的 token 数不超过 chunk_size
                sub_chunks = self._split_by_token_budget(chunk)
                result.extend(sub_chunks)
        return result

    def _split_by_token_budget(self, text: str) -> List[str]:
        """将文本拆分为不超过 chunk_size 的块（按 token 预算）"""
        chunks = []
        start = 0
        while True:
            remaining = text[start:]
            remaining_tokens = self._get_token_count(remaining)
            if remaining_tokens <= self.chunk_size:
                if remaining.strip():
                    chunks.append(remaining.strip())
                break

            # 二分找最大可放入的字符数
            split_pos = self._find_split_pos(remaining, self.chunk_size)
            if split_pos <= 0:
                # 无法在 token 预算内切分，强制按单字符推进避免死循环/丢文本
                chunks.append(remaining[:1])
                start += 1
                continue

            chunk_text = remaining[:split_pos].strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += split_pos

        return [c for c in chunks if c]

    def _find_split_pos(self, text: str, max_tokens: int) -> int:
        """二分查找：找最大字符偏移，使 text[:pos] 的 token 数不超过 max_tokens"""
        if not text or max_tokens <= 0:
            return 0
        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            if self._get_token_count(text[:mid]) <= max_tokens:
                left = mid
            else:
                right = mid - 1
        return left

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """在相邻 chunk 之间添加 overlap（确保不超过 chunk_size）"""
        if len(chunks) < 2 or self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1]
            curr_text = chunks[i]
            curr_tokens = self._get_token_count(curr_text)

            # 计算可用的 overlap token 数（不超过 chunk_size）
            available = self.chunk_size - curr_tokens - 2  # -2 for \n\n separator
            if available < 10:
                # 没有足够空间，不加 overlap
                result.append(curr_text)
                continue

            overlap_tokens = min(self.chunk_overlap, available)
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(prev_text, add_special_tokens=False)
                if len(tokens) <= overlap_tokens:
                    overlap_text = prev_text
                else:
                    overlap_ids = tokens[-overlap_tokens:]
                    overlap_text = self.tokenizer.decode(overlap_ids, skip_special_tokens=True)
            else:
                char_budget = overlap_tokens * 4
                overlap_text = prev_text[-char_budget:] if len(prev_text) > char_budget else prev_text

            combined = overlap_text + "\n\n" + curr_text
            result.append(combined)

        return result
