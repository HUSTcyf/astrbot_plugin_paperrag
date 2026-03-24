"""
混合架构解析器 - 语义分块版本

策略：
1. 使用PDFParserAdvanced提取多模态内容（表格/公式/图片）
2. 语义分块（保持语句/段落完整）
3. 支持块重叠（保持语义连贯）
4. 保留学术论文的元数据（页码/章节/引用）
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 抑制底层库的 gRPC/absl 警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from astrbot.api import logger

# 导入自定义PDF解析器
from .semantic_chunker import PDFParserAdvanced


@dataclass
class Node:
    """简化的Node类（替代llama-index的BaseNode）"""
    text: str
    metadata: Dict[str, Any]


class HybridPDFParser:
    """
    混合PDF解析器（语义分块版本）

    结合：
    - 自定义多模态提取（PDFParserAdvanced）
    - 语义分块（保持语句/段落完整）
    - 块重叠（保持语义连贯）
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
        enable_multimodal: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        初始化混合解析器

        Args:
            enable_multimodal: 是否启用多模态提取
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
            min_chunk_size: 最小块大小（避免太小）
        """
        self.enable_multimodal = enable_multimodal
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # 初始化自定义PDF解析器
        self.pdf_parser = PDFParserAdvanced(
            enable_multimodal=enable_multimodal
        )

        logger.info(f"✅ HybridPDFParser初始化完成 (chunk_size={chunk_size}, overlap={chunk_overlap}, min_size={min_chunk_size})")

    def parse_pdf_to_documents(self, pdf_path: str) -> List[Node]:
        """
        解析PDF为Documents

        Args:
            pdf_path: PDF文件路径

        Returns:
            Node列表
        """
        try:
            filename = Path(pdf_path).name

            # 使用自定义解析器提取多模态内容
            logger.debug(f"🔍 解析PDF: {filename}")
            text, metadata = self.pdf_parser.parse_pdf(pdf_path)

            # 构建增强的文本（包含多模态占位符）
            enhanced_text = self._build_enhanced_text(text, metadata)

            # 创建单个Node（整篇论文）
            document = Node(
                text=enhanced_text,
                metadata={
                    "file_name": filename,
                    "file_path": str(pdf_path),
                    "total_pages": metadata.get("total_pages", 0),
                    "parser": "HybridPDFParser",
                    "images_count": metadata.get("images_count", 0),
                    "tables_count": metadata.get("tables_count", 0),
                    "formulas_count": metadata.get("formulas_count", 0),
                    "multimodal_data": metadata.get("multimodal_data", {})
                }
            )

            # 打印多模态内容统计
            images_count = metadata.get("images_count", 0)
            tables_count = metadata.get("tables_count", 0)
            formulas_count = metadata.get("formulas_count", 0)
            multimodal_data = metadata.get("multimodal_data", {})

            multimodal_parts = []
            if images_count > 0:
                multimodal_parts.append(f"图片{images_count}个")
            if tables_count > 0:
                multimodal_parts.append(f"表格{tables_count}个")
            if formulas_count > 0:
                multimodal_parts.append(f"公式{formulas_count}个")

            multimodal_info = ", ".join(multimodal_parts) if multimodal_parts else "无"
            logger.info(f"📊 PDF解析完成: {filename} ({len(enhanced_text)} chars) - 提取到{multimodal_info}")

            return [document]

        except Exception as e:
            logger.error(f"❌ PDF解析失败 {pdf_path}: {e}")
            return []

    def _build_enhanced_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """构建增强的文本（跳过表格，只包含文字与公式）"""
        multimodal_data = metadata.get("multimodal_data", {})

        if not multimodal_data:
            return text

        text_parts = [text]
        # 添加表格                                                                                                                                                                     
        # tables = multimodal_data.get("tables", [])                                                                                                                                     
        # if tables:                                                                                                                                                                     
        #     text_parts.append("\n\n=== Tables ===\n")                                                                                                                                  
        # for table in tables:                                                                                                                                                       
        #     caption = table.get("caption", "")                                                                                                                                     
        #     markdown = table.get("markdown", "")                                                                                                                                   
        #     page_num = table.get("page_number", 0)                                                                                                                                 
        #     text_parts.append(f"\n[Table on page {page_num}: {caption}]\n{markdown}\n")

        # 添加公式（跳过表格，不对其进行向量化）
        formulas = multimodal_data.get("formulas", [])
        if formulas:
            text_parts.append("\n\n=== Formulas ===\n")
            for formula in formulas:
                formula_text = formula.get("text", "")
                page_num = formula.get("page_number", 0)
                text_parts.append(f"\n[Formula on page {page_num}]\n$$ {formula_text} $$\n")

        return "\n".join(text_parts)

    def parse_and_split(self, pdf_path: str) -> List[Node]:
        """
        解析PDF并分块为Nodes

        Args:
            pdf_path: PDF文件路径

        Returns:
            Node列表
        """
        try:
            # 解析为Documents
            documents = self.parse_pdf_to_documents(pdf_path)

            if not documents:
                logger.warning(f"⚠️ 无法解析PDF: {pdf_path}")
                return []

            # 语义分块
            logger.debug(f"🔄 语义分块处理...")
            all_nodes = []
            chunk_index = 0

            for doc in documents:
                nodes = self._semantic_chunk(doc.text, doc.metadata, chunk_index)
                all_nodes.extend(nodes)
                chunk_index += len(nodes)

            logger.debug(f"✅ 语义分块完成: {len(all_nodes)} 个nodes")
            return all_nodes

        except Exception as e:
            logger.error(f"❌ PDF分块失败 {pdf_path}: {e}")
            return []

    def _semantic_chunk(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        start_chunk_index: int = 0
    ) -> List[Node]:
        """
        语义分块算法

        策略：
        1. 优先按段落分割（段落是语义完整的单元）
        2. 如果段落太大，按句子分割
        3. 如果句子太长，按子句分割（逗号等）
        4. 使用 overlap 保持相邻块之间的语义连贯

        Args:
            text: 待分块文本
            base_metadata: 基础元数据
            start_chunk_index: 起始块索引

        Returns:
            Node列表
        """
        nodes = []

        # 第一步：按段落分割
        paragraphs = self._split_by_paragraphs(text)

        if not paragraphs:
            return nodes

        # 第二步：将段落逐个处理并分割
        chunk_index = start_chunk_index
        i = 0

        while i < len(paragraphs):
            para = paragraphs[i]
            para_size = len(para)

            # 如果单个段落就超过 chunk_size，需要进一步分割
            if para_size > self.chunk_size:
                # 分割大段落
                sub_chunks = self._split_large_paragraph(para)
                for sub_chunk in sub_chunks:
                    nodes.append(Node(
                        text=sub_chunk,
                        metadata={**base_metadata, "chunk_index": chunk_index}
                    ))
                    chunk_index += 1
            else:
                # 尝试将多个小段落组合成一个块
                current_parts = []
                current_size = 0
                last_end_idx = i  # 记录这个块使用了哪些段落

                # 收集段落直到超过 chunk_size
                while i < len(paragraphs):
                    next_para = paragraphs[i]
                    next_size = len(next_para)
                    sep_size = 2 if current_parts else 0  # \n\n 分隔符

                    if current_size + sep_size + next_size <= self.chunk_size:
                        current_parts.append(next_para)
                        current_size += sep_size + next_size
                        i += 1
                    else:
                        break

                # 如果收集的段落太少（小于min_chunk_size），尝试合并更多
                if current_size < self.min_chunk_size and i < len(paragraphs):
                    # 添加更多段落直到达到最小大小
                    while i < len(paragraphs) and current_size < self.min_chunk_size:
                        next_para = paragraphs[i]
                        next_size = len(next_para)
                        sep_size = 2 if current_parts else 0

                        current_parts.append(next_para)
                        current_size += sep_size + next_size
                        i += 1

                # 应用 overlap
                chunk_text = self._join_parts(current_parts)
                if nodes and self.chunk_overlap > 0:
                    chunk_text = self._apply_overlap(chunk_text, nodes)

                nodes.append(Node(
                    text=chunk_text,
                    metadata={**base_metadata, "chunk_index": chunk_index}
                ))
                chunk_index += 1

                # 由于上面的while循环已经增加了i，这里不需要再增加
                continue

            i += 1

        return nodes

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 按双换行分割（段落分隔）
        parts = text.split('\n\n')
        paragraphs = []

        for part in parts:
            stripped = part.strip()
            if stripped:
                paragraphs.append(stripped)

        return paragraphs

    def _split_large_paragraph(self, para: str) -> List[str]:
        """
        分割过大的段落

        按句子分割，如果句子还是太大，按子句分割

        Args:
            para: 段落文本

        Returns:
            子块列表
        """
        # 按句子分割（中文和英文）
        sentences = self._split_by_sentences(para)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果单个句子就超过 chunk_size
            if len(sentence) > self.chunk_size:
                # 先保存当前累积的块
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 进一步按子句分割
                sub_clauses = self._split_by_clauses(sentence)
                for clause in sub_clauses:
                    clause = clause.strip()
                    if not clause:
                        continue

                    if len(clause) > self.chunk_size:
                        # 仍然太大，直接截断并添加到上一个块的末尾
                        if chunks:
                            chunks[-1] += " " + clause[:self.chunk_size]
                    else:
                        chunks.append(clause)

            elif len(current_chunk) + len(sentence) > self.chunk_size:
                # 当前块加上句子会超过限制，保存当前块并开始新块
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

            else:
                # 正常情况
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # 保存最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本（支持中英文）"""
        sentences = []

        # 使用正则表达式按句子结束符分割
        # 匹配：。！？. ! ? 和其后的空格
        pattern = r'(?<=[。！？.!?])\s*'
        parts = re.split(pattern, text)

        for part in parts:
            stripped = part.strip()
            if stripped:
                sentences.append(stripped)

        return sentences

    def _split_by_clauses(self, text: str) -> List[str]:
        """按子句分割（逗号、分号等）"""
        clauses = []

        # 按逗号、分号分割（保留分隔符）
        pattern = r'(?<=[，,；;])\s*'
        parts = re.split(pattern, text)

        for part in parts:
            stripped = part.strip()
            if stripped:
                clauses.append(stripped)

        return clauses if clauses else [text]

    def _join_parts(self, parts: List[str]) -> str:
        """将段落列表合并为文本"""
        return "\n\n".join(parts)

    def _get_overlap_text(self, parts: List[str]) -> str:
        """
        获取 overlap 部分

        从parts的末尾部分获取，使其与下一个块有语义连贯

        Args:
            parts: 段落列表

        Returns:
            overlap 文本
        """
        if not parts or self.chunk_overlap <= 0:
            return ""

        # 从后向前收集，直到达到 overlap 大小
        overlap_parts = []
        current_size = 0

        for part in reversed(parts):
            if current_size >= self.chunk_overlap:
                break
            overlap_parts.insert(0, part)
            current_size += len(part) + 2  # 加上分隔符大小

        return "\n\n".join(overlap_parts)

    def _apply_overlap(self, chunk_text: str, existing_nodes: List[Node]) -> str:
        """
        将当前块与前一个块应用 overlap

        Args:
            chunk_text: 当前块文本
            existing_nodes: 已有的节点列表

        Returns:
            应用 overlap 后的文本
        """
        if self.chunk_overlap <= 0 or not existing_nodes:
            return chunk_text

        # 获取前一个块的末尾部分
        prev_node = existing_nodes[-1]
        prev_text = prev_node.text

        # 提取前一个块的末尾 overlap 部分
        if len(prev_text) <= self.chunk_overlap:
            overlap_text = prev_text
        else:
            overlap_text = prev_text[-self.chunk_overlap:]

        # 清理 overlap 文本（确保不会在句子中间断开）
        overlap_text = self._clean_overlap(overlap_text)

        # 如果overlap太大，截断它以确保最终块不超过chunk_size
        max_overlap = self.chunk_size - len(chunk_text)
        if max_overlap < 0:
            # chunk_text 已经超过 chunk_size，不需要 overlap
            return chunk_text

        if len(overlap_text) > max_overlap:
            overlap_text = overlap_text[-max_overlap:]

        # 将 overlap 添加到当前块的开头
        if overlap_text:
            return overlap_text + "\n\n" + chunk_text

        return chunk_text

    def _clean_overlap(self, overlap_text: str) -> str:
        """
        清理 overlap 文本，确保不在句子中间断开

        Args:
            overlap_text: overlap 文本

        Returns:
            清理后的 overlap 文本
        """
        if not overlap_text:
            return ""

        # 查找最后一个完整的句子结束位置
        for delimiter in self.SENTENCE_DELIMITERS:
            # 反向查找最后一个分隔符
            idx = overlap_text.rfind(delimiter)
            if idx > 0:
                # 返回最后一个完整句子之后的部分
                return overlap_text[idx + len(delimiter):].strip()

        # 如果没有找到完整的句子结束符，尝试找到一个合适的断点
        # 查找最后一个逗号或逗号+空格
        for delimiter in ['，', ', ', '， ', '、', '; ', '； ']:
            idx = overlap_text.rfind(delimiter)
            if idx > len(overlap_text) // 2:  # 确保不是在太短的位置断开
                return overlap_text[idx + len(delimiter):].strip()

        # 如果还是找不到，在 chunk_overlap 范围内找一个空格断开
        if ' ' in overlap_text:
            idx = overlap_text.rfind(' ')
            if idx > len(overlap_text) // 2:
                return overlap_text[idx + 1:].strip()

        # 无法干净地断开，返回空（不使用 overlap）
        return ""

    # ==================== 向后兼容的简单分块方法 ====================

    def _split_text(self, text: str, base_metadata: Dict[str, Any], start_chunk_index: int = 0) -> List[Node]:
        """
        简单的文本分块（向后兼容）

        DEPRECATED: 请使用 _semantic_chunk 方法

        Args:
            text: 待分块文本
            base_metadata: 基础元数据
            start_chunk_index: 起始块索引

        Returns:
            Node列表
        """
        return self._semantic_chunk(text, base_metadata, start_chunk_index)
