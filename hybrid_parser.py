"""
混合架构解析器 - 语义分块版本

策略：
1. 使用PDFParserAdvanced提取多模态内容（表格/公式/图片）
2. 语义分块（保持语句/段落完整）
3. 支持块重叠（保持语义连贯）
4. 保留学术论文的元数据（页码/章节/引用）
5. 识别并关联参考文献（使用 Grobid 解析）
6. 图片存储与关联（通过图片引用关联到文本块）
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# 抑制底层库的 gRPC/absl 警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import fitz  # PyMuPDF

from astrbot.api import logger

# 导入自定义PDF解析器（兼容直接运行和包运行）
try:
    from .multimodal_extractor import PDFParserAdvanced
except ImportError:
    from multimodal_extractor import PDFParserAdvanced

# 导入引用处理器（兼容直接运行和包运行）
try:
    from .reference_processor import (
        ReferenceExtractor,
        CitationLinker,
        process_references_and_citations,
        process_references_and_citations_grobid,
        process_references_with_llm,
        Reference
    )
except ImportError:
    from reference_processor import (
        ReferenceExtractor,
        CitationLinker,
        process_references_and_citations,
        process_references_and_citations_grobid,
        process_references_with_llm,
        Reference
    )


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
    - 图片存储与关联（通过图片引用将图片关联到文本块，供VLM使用）
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
        min_chunk_size: int = 100,
        figures_dir: str = "",
        llm_config: Dict[str, Any] = {},
        arxiv_client: Any = None
    ):
        """
        初始化混合解析器

        Args:
            enable_multimodal: 是否启用多模态提取
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
            min_chunk_size: 最小块大小（避免太小）
            figures_dir: 图片存储目录（空则使用 papers/figures）
            llm_config: LLM 配置字典（可选），包含 model、api_base、api_key
            arxiv_client: arXiv MCP 客户端（可选），用于查询论文详情
        """
        self.enable_multimodal = enable_multimodal
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.figures_dir = figures_dir
        self.llm_config = llm_config
        self.arxiv_client = arxiv_client

        # 初始化自定义PDF解析器
        self.pdf_parser = PDFParserAdvanced(
            enable_multimodal=enable_multimodal
        )

        # 图片索引计数器（用于生成唯一文件名）
        self._image_counter = 0

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
                    "multimodal_data": metadata.get("multimodal_data", {}),
                    "raw_text": text,  # 保存原始文本用于参考文献提取
                    "added_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    async def parse_and_split(
        self,
        pdf_path: str,
        llm_config: Dict[str, Any] = {},
        arxiv_client: Any = None
    ) -> List[Node]:
        """
        解析PDF并分块为Nodes

        Args:
            pdf_path: PDF文件路径
            llm_config: LLM 配置字典（可选），包含 model、api_base、api_key
            arxiv_client: arXiv MCP 客户端（可选，会覆盖初始化时设置的 client）

        Returns:
            Node列表
        """
        try:
            # 解析为Documents
            documents = self.parse_pdf_to_documents(pdf_path)

            if not documents:
                logger.warning(f"⚠️ 无法解析PDF: {pdf_path}")
                return []

            # 提取并保存图片（获取图片路径映射）
            image_paths = {}
            image_pages = {}
            if documents and self.enable_multimodal:
                multimodal_data = documents[0].metadata.get("multimodal_data", {})
                image_paths, image_pages = self._extract_and_save_images(pdf_path, multimodal_data)

            # 语义分块
            logger.debug(f"🔄 语义分块处理...")
            all_nodes = []
            chunk_index = 0

            # 保存原始文本用于引用提取（不使用增强文本）
            # 增强文本会跳过表格等内容，可能导致参考文献部分不完整
            raw_text = documents[0].metadata.get("raw_text", documents[0].text)
            logger.info(f"📝 原始文本总长度: {len(raw_text)} 字符, {len(raw_text.split(chr(10)))} 行")
            # 调试：查找 References 在原始文本中的位置
            lines = raw_text.split('\n')
            for doc in documents:
                nodes = self._semantic_chunk(doc.text, doc.metadata, chunk_index)
                all_nodes.extend(nodes)
                chunk_index += len(nodes)

            # 通过图片引用将图片关联到文本块
            if image_paths:
                all_nodes = self._associate_images_with_chunks(all_nodes, image_paths, image_pages)

            # 引用处理
            # 优先使用 LLM-based 解析（如果提供了 LLM config）
            # 否则使用正则表达式解析
            effective_llm_config = llm_config or self.llm_config
            effective_arxiv_client = arxiv_client or self.arxiv_client

            if effective_llm_config:
                logger.debug(f"🔄 LLM引用处理...")
                try:
                    references, all_nodes = await process_references_with_llm(
                        pdf_path, all_nodes, raw_text,
                        llm_config=effective_llm_config,
                        arxiv_client=effective_arxiv_client
                    )
                except Exception as e:
                    logger.warning(f"⚠️ LLM引用处理失败，回退到正则解析: {e}")
                    use_grobid = False
                    references, all_nodes = process_references_and_citations_grobid(
                        pdf_path, all_nodes, raw_text, use_grobid=use_grobid
                    )
            else:
                logger.debug(f"🔄 正则引用处理...")
                use_grobid = False
                references, all_nodes = process_references_and_citations_grobid(
                    pdf_path, all_nodes, raw_text, use_grobid=use_grobid
                )

            if references and documents:
                # 将参考文献信息添加到文档元数据
                first_doc = documents[0]
                first_doc.metadata['references'] = [ref.to_dict() for ref in references]
                logger.info(f"📚 识别到 {len(references)} 条参考文献")

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

        # 第三步：后处理 - 合并极短chunk，拆分极长chunk
        nodes = self._post_process_chunks(nodes)

        return nodes

    def _post_process_chunks(self, nodes: List[Node]) -> List[Node]:
        """
        后处理：合并极短chunk，拆分极长chunk

        Args:
            nodes: 分块后的节点列表

        Returns:
            处理后的节点列表
        """
        if not nodes:
            return nodes

        # 极短阈值: 小于 min_chunk_size 的一半就认为是异常短（需合并）
        too_short_threshold = max(self.min_chunk_size // 2, 50)
        # 极长阈值: 超过 chunk_size * 10 就认为需要拆分
        too_long_threshold = self.chunk_size * 10

        result = []

        i = 0
        while i < len(nodes):
            node = nodes[i]
            text_len = len(node.text)

            # 极短chunk：合并到前一个chunk（如果存在），否则合并到下一个chunk
            if text_len < too_short_threshold:
                if result:
                    # 合并到前一个chunk
                    prev_node = result[-1]
                    merged_text = prev_node.text + "\n\n" + node.text
                    result[-1] = Node(
                        text=merged_text,
                        metadata={**prev_node.metadata}
                    )
                elif i + 1 < len(nodes):
                    # 没有前一个chunk，合并到下一个chunk
                    next_node = nodes[i + 1]
                    merged_text = node.text + "\n\n" + next_node.text
                    nodes[i + 1] = Node(
                        text=merged_text,
                        metadata={**next_node.metadata}
                    )
                    # 跳过下一个节点（已合并）
                    i += 2
                    continue
                else:
                    # 只有一个节点且太短，直接保留
                    result.append(node)
                i += 1
                continue

            # 极长chunk：拆分为固定大小的子块
            if text_len > too_long_threshold:
                sub_chunks = self._split_long_text(node.text)
                for sub_text in sub_chunks:
                    result.append(Node(
                        text=sub_text,
                        metadata={**node.metadata}
                    ))
                i += 1
                continue

            # 正常chunk
            result.append(node)
            i += 1

        # 重建 chunk_index
        for idx, node in enumerate(result):
            node.metadata["chunk_index"] = idx

        return result

    def _split_long_text(self, text: str) -> List[str]:
        """
        将超长文本拆分为固定大小的块（不保留语义，简单分割）

        用于处理PDF解析异常导致的超长段落

        Args:
            text: 超长文本

        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunks.append(text[start:])
                break

            # 尝试在句子边界断开
            chunk_text = text[start:end]
            split_pos = -1

            # 从后向前找句子边界
            for delimiter in ['。', '！', '？', '. ', '! ', '? ']:
                pos = chunk_text.rfind(delimiter)
                if pos > self.chunk_size // 4:  # 确保不在太靠前的位置
                    split_pos = pos + len(delimiter)
                    break

            # 如果找不到好的断点，尝试找逗号
            if split_pos == -1:
                for delimiter in ['，', ', ', '；', '; ']:
                    pos = chunk_text.rfind(delimiter)
                    if pos > self.chunk_size // 4:
                        split_pos = pos + len(delimiter)
                        break

            # 如果还找不到，在空格处断开
            if split_pos == -1:
                space_pos = chunk_text.rfind(' ')
                if space_pos > self.chunk_size // 4:
                    split_pos = space_pos + 1
                else:
                    split_pos = end

            chunks.append(chunk_text[:split_pos].strip())
            start += split_pos

        return [c for c in chunks if c]

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

    # ==================== 图片存储与关联（VLM支持） ====================

    def _get_figures_dir(self, pdf_path: str) -> str:
        """
        获取图片存储目录

        Args:
            pdf_path: PDF文件路径

        Returns:
            图片存储目录路径
        """
        if self.figures_dir:
            return self.figures_dir

        # 默认使用插件目录下的 data/figures
        plugin_dir = Path(__file__).parent
        return str(plugin_dir / "data" / "figures")

    def _save_image_to_disk(
        self,
        image: Any,
        pdf_path: str,
        page_num: int,
        image_index: Any
    ) -> Optional[str]:
        """
        保存图片到磁盘

        将提取的图片保存到磁盘，返回图片路径。
        这个路径将存储在 Milvus 中，供 VLM 后续加载。

        Args:
            image: PIL.Image 对象
            pdf_path: PDF文件路径
            page_num: 页码
            image_index: 图片索引

        Returns:
            图片文件路径，失败返回 None
        """
        try:
            from PIL import Image
            import io

            figures_dir = self._get_figures_dir(pdf_path)

            # 确保目录存在
            figures_path = Path(figures_dir)
            figures_path.mkdir(parents=True, exist_ok=True)

            # 生成唯一文件名：{pdf_name}_p{page}_i{index}.png
            pdf_name = Path(pdf_path).stem
            filename = f"{pdf_name}_p{page_num}_i{image_index}.png"
            image_path = figures_path / filename

            # 保存为 PNG（原图保存，查询时再transform）
            image.save(str(image_path), format="PNG", optimize=True)

            # 获取文件大小
            file_size = image_path.stat().st_size / 1024  # KB

            logger.info(f"🖼️ 保存图片: {filename} (页{page_num}, {image.size[0]}x{image.size[1]}, {file_size:.1f}KB) → {image_path}")

            return str(image_path)

        except Exception as e:
            logger.warning(f"⚠️ 保存图片失败: {e}")
            return None

    def _extract_and_save_images(self, pdf_path: str, multimodal_data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        提取并保存图片，返回图片路径映射

        核心：提取图片保存到磁盘，供 VLM 后续加载。

        改进：基于图注和位置聚类，提取完整大图
        - 将相同图注（Figure 1, Figure 2）的图片按位置聚类
        - 计算覆盖所有子图的外接矩形
        - 从 PDF 页面直接裁剪外接矩形区域，得到完整大图

        Args:
            pdf_path: PDF文件路径
            multimodal_data: 多模态数据（包含 images 列表）

        Returns:
            Tuple[Dict[str, str], Dict[str, int]]: (图注->图片路径, 图注->页码) 的映射
        """
        image_paths: Dict[str, str] = {}
        image_pages: Dict[str, int] = {}

        images = multimodal_data.get("images", [])
        if not images:
            return image_paths, image_pages

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.warning(f"⚠️ 无法打开PDF提取图片: {pdf_path}, {e}")
            return image_paths, image_pages

        try:
            # 第一步：按图注分组图片
            caption_groups = self._group_images_by_caption(images)

            # 第二步：对每组图片，提取完整大图
            for caption, img_list in caption_groups.items():
                if not img_list:
                    continue

                try:
                    page_num = img_list[0].get("page_number", 0)
                    if page_num == 0:
                        continue

                    # 获取该页面的所有图片 bbox
                    page: fitz.Page = doc[page_num - 1]  # type: ignore[assignment]
                    image_list = page.get_images(full=True)

                    # 构建该组图片的 bbox 列表
                    group_bboxes = []
                    for img_info in img_list:
                        img_idx = img_info.get("image_index", 0)
                        if img_idx < len(image_list):
                            # 获取图片在页面上的位置
                            img_rects = page.get_image_rects(image_list[img_idx][0])
                            if img_rects:
                                group_bboxes.append(img_rects[0])

                    if not group_bboxes:
                        continue

                    # 计算外接矩形
                    merged_rect = self._merge_bboxes(group_bboxes)

                    # 从 PDF 页面裁剪外接矩形区域
                    clipped_image = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=merged_rect)
                    pil_image = self._pixmap_to_pil_image(clipped_image)

                    if pil_image:
                        # 保存完整大图（使用图注作为索引）
                        figure_idx = self._extract_figure_number(caption)
                        image_path = self._save_image_to_disk(
                            pil_image, pdf_path, page_num, figure_idx
                        )
                        if image_path:
                            # 每个图注只保存一次
                            if caption not in image_paths:
                                image_paths[caption] = image_path
                                image_pages[caption] = page_num

                except Exception as e:
                    logger.debug(f"⚠️ 处理图片组 {caption} 失败: {e}")
                    continue

        finally:
            doc.close()

        if image_paths:
            figures_dir = self._get_figures_dir(pdf_path)
            logger.info(f"🖼️ 提取并保存 {len(image_paths)} 张图片至 {figures_dir}")
            for caption, path in image_paths.items():
                logger.debug(f"   {caption} (页{image_pages[caption]}) → {Path(path).name}")

        return image_paths, image_pages

    def _group_images_by_caption(self, images: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        将图片按图注分组

        规则：
        1. 有 caption 的图片：相同 caption 为一组
        2. 无 caption 的图片：分配到距离最近的 captioned 组

        Args:
            images: 图片信息列表

        Returns:
            Dict[str, List[Dict]]: 图注 -> 图片列表 的映射
        """
        groups = {}  # {caption: [img_info, ...]}
        no_caption_images = []

        # 第一步：将所有图片按 caption 分组
        for img_info in images:
            caption = img_info.get("caption")
            if caption:
                # 有 caption 的图片
                if caption not in groups:
                    groups[caption] = []
                groups[caption].append(img_info)
            else:
                # 无 caption 的图片暂存
                no_caption_images.append(img_info)

        # 第二步：无 caption 的图片分配到最近的组
        for img in no_caption_images:
            img_page = img.get("page_number", 0)
            if img_page == 0:
                # 无法确定页码，加入第一个组或创建 no_caption 组
                if groups:
                    groups[list(groups.keys())[0]].append(img)
                else:
                    groups["no_caption"] = [img]
                continue

            # 找距离最近的组
            best_caption = None
            min_distance = float('inf')

            for caption, img_list in groups.items():
                if not img_list:
                    continue
                # 使用该组第一张图片的页码作为组页码
                group_page = img_list[0].get("page_number", 0)
                distance = abs(group_page - img_page)
                if distance < min_distance:
                    min_distance = distance
                    best_caption = caption

            # 加入最近的组
            if best_caption:
                groups[best_caption].append(img)
            else:
                # 没有 captioned 组，创建 no_caption 组
                if "no_caption" not in groups:
                    groups["no_caption"] = []
                groups["no_caption"].append(img)

        return groups

    def _merge_bboxes(self, bboxes: List) -> "fitz.Rect":
        """
        计算多个 bbox 的外接矩形

        Args:
            bboxes: bbox 列表 [(x0, y0, x1, y1), ...]

        Returns:
            覆盖所有 bbox 的外接矩形
        """
        if not bboxes:
            return fitz.Rect(0, 0, 0, 0)

        x0 = min(b.x0 for b in bboxes)
        y0 = min(b.y0 for b in bboxes)
        x1 = max(b.x1 for b in bboxes)
        y1 = max(b.y1 for b in bboxes)

        return fitz.Rect(x0, y0, x1, y1)

    def _pixmap_to_pil_image(self, pixmap) -> Optional[Any]:
        """
        将 PyMuPDF Pixmap 转换为 PIL Image

        Args:
            pixmap: PyMuPDF Pixmap 对象

        Returns:
            PIL Image 对象
        """
        try:
            from PIL import Image
            import io

            # Pixmap 转换为 PNG bytes
            png_data = pixmap.tobytes("png")
            return Image.open(io.BytesIO(png_data))
        except Exception as e:
            logger.debug(f"⚠️ Pixmap 转 PIL Image 失败: {e}")
            return None

    def _associate_images_with_chunks(
        self,
        nodes: List[Node],
        image_paths: Dict[str, str],
        image_pages: Dict[str, int] = {}
    ) -> List[Node]:
        """
        将图片路径关联到包含对应引用的 chunk

        关联逻辑：通过识别正文中的图片引用（Figure 1, Fig. 2, 图1等）
        将图片关联到包含对应引用的文本块。

        Args:
            nodes: 分块后的节点列表
            image_paths: 图注 -> 图片路径 的映射
            image_pages: 图注 -> 页码 的映射

        Returns:
            添加了 image_path 信息的节点列表
        """
        if not image_paths:
            return nodes

        # 构建图片引用映射：figure编号 -> (图片路径, 图注)
        figure_refs: Dict[str, Tuple[str, str]] = {}
        for caption, path in image_paths.items():
            figure_num = self._extract_figure_number(caption)
            if figure_num:
                figure_refs[figure_num] = (path, caption)

        if not figure_refs:
            logger.debug("⚠️ 未找到图片引用编号")
            return nodes

        # 预编译图片引用正则表达式
        import re
        figure_patterns = [
            re.compile(r'(?:Figure|Fig\.?|图)\s*(\d+[a-zA-Z]?)', re.IGNORECASE),
            re.compile(r'(?:Figure|Fig\.?|图)\s*(\d+[a-zA-Z]?)\s*[:.\)]', re.IGNORECASE),
        ]

        # 关联图片到 chunk
        associated_count = 0
        for node in nodes:
            node.metadata = dict(node.metadata)  # 复制避免修改原数据

            found_images = []  # 可能一个 chunk 引用多张图片

            for pattern in figure_patterns:
                for match in pattern.finditer(node.text):
                    fig_num = match.group(1)
                    if fig_num in figure_refs:
                        path, caption = figure_refs[fig_num]
                        # 去重
                        if path not in [img[0] for img in found_images]:
                            found_images.append((path, caption, fig_num))

            if found_images:
                # 关联第一张图片（主要图片）
                node.metadata["image_path"] = found_images[0][0]
                node.metadata["image_caption"] = found_images[0][1]
                node.metadata["image_figure_num"] = found_images[0][2]
                node.metadata["has_image"] = True

                # 如果有多张图片，保存所有图片
                if len(found_images) > 1:
                    node.metadata["all_images"] = [
                        {"path": img[0], "caption": img[1], "figure_num": img[2]}
                        for img in found_images
                    ]
                associated_count += 1
            else:
                node.metadata["has_image"] = False

        logger.info(f"🔗 通过图片引用关联 {associated_count}/{len(nodes)} 个文本块到图片")
        return nodes

    def _extract_figure_number(self, caption: str) -> Optional[str]:
        """
        从图注中提取图片编号

        Args:
            caption: 图注文本，如 "Figure 1: Quantitative Results..."

        Returns:
            图片编号，如 "1" 或 "1a"，None 表示未找到
        """
        import re
        # 匹配 Figure 1, Fig. 1, 图 1 等格式
        patterns = [
            r'(?:Figure|Fig\.?)\s*(\d+[a-zA-Z]?)',
            r'(?:图)\s*(\d+[a-zA-Z]?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_page_number_from_text(self, text: str) -> int:
        """从文本中提取页码"""
        import re
        # 匹配 [Page X] 格式
        match = re.search(r'\[Page\s+(\d+)\]', text)
        if match:
            return int(match.group(1))

        # 匹配第X页
        match = re.search(r'第\s*(\d+)\s*页', text)
        if match:
            return int(match.group(1))

        return 0
