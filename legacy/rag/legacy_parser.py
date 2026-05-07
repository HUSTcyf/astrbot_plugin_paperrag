"""
Legacy 自定义语义分块解析器

本模块保留原有手工实现的语义分块算法，仅作为向后兼容。
建议使用 LlamaIndex SemanticSplitterNodeParser（更稳定、更高效）。
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# 抑制底层库的 gRPC/absl 警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from astrbot.api import logger

# 复用 Node 类
try:
    from ...rag.hybrid_parser import Node
except ImportError:
    from rag.hybrid_parser import Node


@dataclass
class LegacyPDFParser:
    """
    自定义语义分块解析器（Legacy）

    策略：
    1. 优先按段落分割（段落是语义完整的单元）
    2. 如果段落太大，按句子分割
    3. 如果句子太长，按子句分割（逗号等）
    4. 使用 overlap 保持相邻块之间的语义连贯
    """

    # 中断符号优先级（越靠前越优先作为断点）
    SENTENCE_DELIMITERS = [
        '\n\n',      # 段落分隔符（最高优先级）
        '。',        # 句号
        '！',        # 感叹号
        '？',        # 问号
        '；',        # 分号
        '，',        # 逗号（最低优先级）
        '. ',        # 英文句号
        '! ',        # 英文感叹号
        '? ',        # 英文问号
        '; ',        # 英文分号
        ', ',        # 英文逗号
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._tokenizer = None

    def _get_tokenizer(self):
        """懒加载 BGE-M3 tokenizer"""
        if self._tokenizer is None:
            try:
                from ...embedding.flag_embedding import get_flag_model
                flag_model = get_flag_model()
                if flag_model._initialized and flag_model.tokenizer is not None:
                    self._tokenizer = flag_model.tokenizer
                    return self._tokenizer
            except Exception:
                pass

            try:
                from ...embedding.unsloth_embedding import get_embedding_model
                model_instance = get_embedding_model()
                if model_instance is not None and model_instance.tokenizer is not None:
                    self._tokenizer = model_instance.tokenizer
                    return self._tokenizer
            except Exception:
                pass

            try:
                from transformers import AutoTokenizer
                from pathlib import Path
                plugin_root = Path(__file__).parent.parent.parent
                model_dir = plugin_root / "models" / "bge-m3"
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(model_dir), local_files_only=True
                )
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def _get_token_count(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return len(text) // 4
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    # ==================== 语义分块核心方法 ====================

    def _semantic_chunk(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        start_chunk_index: int = 0
    ) -> List[Node]:
        nodes = []
        paragraphs = self._split_by_paragraphs(text)
        if not paragraphs:
            return nodes

        chunk_index = start_chunk_index
        i = 0

        while i < len(paragraphs):
            para = paragraphs[i]
            para_size = self._get_token_count(para)

            if para_size > self.chunk_size:
                sub_chunks = self._split_large_paragraph(para)
                for sub_chunk in sub_chunks:
                    nodes.append(Node(
                        text=sub_chunk,
                        metadata=self._build_lightweight_metadata(base_metadata, chunk_index)
                    ))
                    chunk_index += 1
            else:
                current_parts = []
                current_size = 0

                while i < len(paragraphs):
                    next_para = paragraphs[i]
                    next_size = self._get_token_count(next_para)
                    sep_size = 2 if current_parts else 0

                    if current_size + sep_size + next_size <= self.chunk_size:
                        current_parts.append(next_para)
                        current_size += sep_size + next_size
                        i += 1
                    else:
                        break

                if current_size < self.min_chunk_size and i < len(paragraphs):
                    while i < len(paragraphs) and current_size < self.min_chunk_size:
                        next_para = paragraphs[i]
                        next_size = self._get_token_count(next_para)
                        sep_size = 2 if current_parts else 0
                        current_parts.append(next_para)
                        current_size += sep_size + next_size
                        i += 1

                chunk_text = self._join_parts(current_parts)
                if nodes and self.chunk_overlap > 0:
                    chunk_text = self._apply_overlap(chunk_text, nodes)

                nodes.append(Node(
                    text=chunk_text,
                    metadata=self._build_lightweight_metadata(base_metadata, chunk_index)
                ))
                chunk_index += 1
                continue

            i += 1

        nodes = self._post_process_chunks(nodes)
        return nodes

    def _post_process_chunks(self, nodes: List[Node]) -> List[Node]:
        if not nodes:
            return nodes

        too_short_threshold = max(self.min_chunk_size // 2, 30)
        result = []
        i = 0

        while i < len(nodes):
            node = nodes[i]
            text_len = self._get_token_count(node.text)

            if text_len < too_short_threshold:
                if result:
                    prev_node = result[-1]
                    merged_text = prev_node.text + "\n\n" + node.text
                    result[-1] = Node(
                        text=merged_text,
                        metadata={**prev_node.metadata}
                    )
                elif i + 1 < len(nodes):
                    next_node = nodes[i + 1]
                    merged_text = node.text + "\n\n" + next_node.text
                    nodes[i + 1] = Node(
                        text=merged_text,
                        metadata={**next_node.metadata}
                    )
                    i += 2
                    continue
                else:
                    result.append(node)
                i += 1
                continue

            if text_len > self.chunk_size:
                sub_chunks = self._split_long_text(node.text)
                for sub_text in sub_chunks:
                    result.append(Node(
                        text=sub_text,
                        metadata={**node.metadata}
                    ))
                i += 1
                continue

            result.append(node)
            i += 1

        for idx, node in enumerate(result):
            node.metadata["chunk_index"] = idx

        return result

    def _split_long_text(self, text: str) -> List[str]:
        chunks = []
        start = 0

        while True:
            total_tokens = self._get_token_count(text[start:])
            if total_tokens <= self.chunk_size:
                remaining = text[start:].strip()
                if remaining:
                    chunks.append(remaining)
                break

            target_tokens = self.chunk_size
            sample_text = text[start:]
            left, right = 0, len(sample_text)
            while left < right:
                mid = (left + right + 1) // 2
                t = self._get_token_count(sample_text[:mid])
                if t <= target_tokens:
                    left = mid
                else:
                    right = mid - 1
            split_pos = left

            if split_pos <= 0:
                break

            chunk_text = sample_text[:split_pos].strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += split_pos

            if start >= len(text):
                break

        return [c for c in chunks if c]

    def _find_token_boundary(self, text: str) -> int:
        for delimiter in ['。', '！', '？', '. ', '! ', '? ']:
            pos = text.rfind(delimiter)
            if pos > len(text) // 4:
                return pos + len(delimiter)

        for delimiter in ['，', ', ', '；', '; ']:
            pos = text.rfind(delimiter)
            if pos > len(text) // 4:
                return pos + len(delimiter)

        space_pos = text.rfind(' ')
        if space_pos > len(text) // 4:
            return space_pos + 1

        return -1

    def _split_by_paragraphs(self, text: str) -> List[str]:
        parts = text.split('\n\n')
        return [p.strip() for p in parts if p.strip()]

    def _split_large_paragraph(self, para: str) -> List[str]:
        sentences = self._split_by_sentences(para)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self._get_token_count(sentence)

            if sentence_tokens > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                sub_clauses = self._split_by_clauses(sentence)
                for clause in sub_clauses:
                    clause = clause.strip()
                    if not clause:
                        continue

                    clause_tokens = self._get_token_count(clause)
                    if clause_tokens > self.chunk_size:
                        target = self.chunk_size
                        left, right = 0, len(clause)
                        while left < right:
                            mid = (left + right + 1) // 2
                            t = self._get_token_count(clause[:mid])
                            if t <= target:
                                left = mid
                            else:
                                right = mid - 1
                        clipped = clause[:left]
                        if chunks:
                            chunks[-1] += " " + clipped
                            if self._get_token_count(chunks[-1]) > self.chunk_size:
                                merged = chunks.pop()
                                sub = self._split_long_text(merged)
                                chunks.extend(sub)
                        else:
                            chunks.append(clipped)
                    else:
                        chunks.append(clause)

            elif current_tokens + sentence_tokens > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens

            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        pattern = r'(?<=[。！？.!?])\s*'
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]

    def _split_by_clauses(self, text: str) -> List[str]:
        pattern = r'(?<=[，,；;|])\s*'
        parts = re.split(pattern, text)
        return parts if parts else [text]

    def _join_parts(self, parts: List[str]) -> str:
        return "\n\n".join(parts)

    def _get_overlap_text(self, parts: List[str]) -> str:
        if not parts or self.chunk_overlap <= 0:
            return ""
        overlap_parts = []
        current_size = 0
        overlap_char_budget = self.chunk_overlap * 4
        for part in reversed(parts):
            if current_size >= self.chunk_overlap:
                break
            overlap_parts.insert(0, part)
            current_size += len(part) // 4 + 1
        return "\n\n".join(overlap_parts)

    def _apply_overlap(self, chunk_text: str, existing_nodes: List[Node]) -> str:
        if self.chunk_overlap <= 0 or not existing_nodes:
            return chunk_text

        prev_node = existing_nodes[-1]
        prev_text = prev_node.text
        overlap_char_len = self.chunk_overlap * 4

        if len(prev_text) <= overlap_char_len:
            overlap_text = prev_text
        else:
            overlap_text = prev_text[-overlap_char_len:]

        overlap_text = self._clean_overlap(overlap_text)
        current_tokens = self._get_token_count(chunk_text)
        max_overlap_tokens = self.chunk_size - current_tokens

        if max_overlap_tokens <= 0:
            return chunk_text

        if overlap_text:
            return overlap_text + "\n\n" + chunk_text

        return chunk_text

    def _clean_overlap(self, overlap_text: str) -> str:
        if not overlap_text:
            return ""

        for delimiter in self.SENTENCE_DELIMITERS:
            idx = overlap_text.rfind(delimiter)
            if idx > 0:
                return overlap_text[idx + len(delimiter):].strip()

        for delimiter in ['，', ', ', '， ', '、', '; ', '； ']:
            idx = overlap_text.rfind(delimiter)
            if idx > len(overlap_text) // 2:
                return overlap_text[idx + len(delimiter):].strip()

        if ' ' in overlap_text:
            idx = overlap_text.rfind(' ')
            if idx > len(overlap_text) // 2:
                return overlap_text[idx + 1:].strip()

        return ""

    def _build_lightweight_metadata(
        self,
        base_metadata: Dict[str, Any],
        chunk_index: int = 0
    ) -> Dict[str, Any]:
        lightweight_fields = [
            "file_name", "file_path", "parser", "total_pages",
            "images_count", "tables_count", "formulas_count",
            "added_time", "extracted_title", "extracted_abstract"
        ]
        metadata = {"chunk_index": chunk_index}
        for key in lightweight_fields:
            if key in base_metadata:
                metadata[key] = base_metadata[key]
        return metadata
