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
        CitationLinker,
        process_references_with_llm,
        Reference
    )
except ImportError:
    from reference_processor import (
        CitationLinker,
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
            result = self.pdf_parser.parse_pdf(pdf_path)

            # 处理 Docling Multimodal 返回的 3-tuple
            if len(result) == 3:
                text, raw_text, metadata = result  # Docling Multimodal
            else:
                text, metadata = result  # 普通 PyMuPDF
                raw_text = text

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
                    "raw_text": raw_text,  # 保存原始文本用于参考文献提取
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
            table_paths = {}
            table_pages = {}
            formula_refs = {}
            formula_pages = {}
            if documents and self.enable_multimodal:
                multimodal_data = documents[0].metadata.get("multimodal_data", {})
                # 使用 PDF 文件名作为 paper_id（用于按论文分类存储）
                paper_id = Path(pdf_path).stem
                image_paths, image_pages = self._extract_and_save_images(pdf_path, multimodal_data, paper_id)
                table_paths, table_pages = self._extract_and_save_tables(pdf_path, multimodal_data, paper_id)
                formula_refs, formula_pages = self._extract_formula_refs(multimodal_data)

            # 语义分块
            logger.debug(f"🔄 语义分块处理...")
            all_nodes = []
            chunk_index = 0

            # 保存原始文本用于引用提取（不使用增强文本）
            # 增强文本会跳过表格等内容，可能导致参考文献部分不完整
            raw_text = documents[0].metadata.get("raw_text", documents[0].text)
            logger.info(f"📝 原始文本总长度: {len(raw_text)} 字符, {len(raw_text.split(chr(10)))} 行")

            # 提取 GitHub 链接作为元数据
            import re
            github_pattern = re.compile(r'https?://github\.com/[a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+')
            github_links = github_pattern.findall(raw_text)
            if github_links:
                # 取第一个（最早出现）作为论文对应的 GitHub 仓库
                documents[0].metadata['github_url'] = github_links[0]
                logger.info(f"📝 提取到 GitHub 链接: {github_links[0]}")

            lines = raw_text.split('\n')
            for doc in documents:
                nodes = self._semantic_chunk(doc.text, doc.metadata, chunk_index)
                all_nodes.extend(nodes)
                chunk_index += len(nodes)

            # 通过图片引用将图片关联到文本块
            if image_paths:
                all_nodes = self._associate_images_with_chunks(all_nodes, image_paths, image_pages)

            # 通过表格引用将表格关联到文本块
            if table_paths:
                all_nodes = self._associate_tables_with_chunks(all_nodes, table_paths, table_pages)

            # 通过公式引用将公式关联到文本块
            if formula_refs:
                all_nodes = self._associate_formulas_with_chunks(all_nodes, formula_refs, formula_pages)

            # 引用处理 - 仅使用 LLM-based 解析
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
                    logger.warning(f"⚠️ LLM引用处理失败: {e}")
                    references = []
            else:
                logger.debug(f"🔄 未配置 LLM，跳过引用处理")
                references = []

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

    def _build_lightweight_metadata(
        self,
        base_metadata: Dict[str, Any],
        chunk_index: int = 0
    ) -> Dict[str, Any]:
        """
        构建轻量级 chunk metadata，避免大数据字段导致 Milvus 消息超限

        排除以下大字段:
        - raw_text: 整篇论文原文（可能数百KB）
        - multimodal_data: 包含所有表格markdown等（可能很大）
        """
        # 保留的轻量级字段
        lightweight_fields = [
            "file_name", "file_path", "parser", "total_pages",
            "images_count", "tables_count", "formulas_count",
            "added_time"
        ]
        metadata = {"chunk_index": chunk_index}
        for key in lightweight_fields:
            if key in base_metadata:
                metadata[key] = base_metadata[key]
        return metadata

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
                        metadata=self._build_lightweight_metadata(base_metadata, chunk_index)
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
                    metadata=self._build_lightweight_metadata(base_metadata, chunk_index)
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

    def _get_figures_dir(self, pdf_path: str, paper_id: str = None) -> str:
        """
        获取图片存储目录

        Args:
            pdf_path: PDF文件路径
            paper_id: 论文ID（用于按论文分类存储）

        Returns:
            图片存储目录路径
        """
        if self.figures_dir:
            base_dir = self.figures_dir
        else:
            # 默认使用插件目录下的 data/figures
            plugin_dir = Path(__file__).parent
            base_dir = str(plugin_dir / "data" / "figures")

        # 按论文分类存储
        if paper_id:
            return str(Path(base_dir) / paper_id)
        else:
            # 降级：使用 PDF 文件名作为目录名
            pdf_stem = Path(pdf_path).stem
            return str(Path(base_dir) / pdf_stem)

    def _save_image_to_disk(
        self,
        image: Any,
        pdf_path: str,
        page_num: int,
        figure_id: str,  # 格式: "3-Table1-1" 或 "5-Figure1-2"
        paper_id: str = None  # 论文ID，用于按论文分类存储
    ) -> Optional[str]:
        """
        保存图片到磁盘

        将提取的图片保存到磁盘，返回图片路径。
        这个路径将存储在 Milvus 中，供 VLM 后续加载。

        Args:
            image: PIL.Image 对象
            pdf_path: PDF文件路径
            page_num: 页码
            figure_id: 图片标识符，格式如 "3-Table1-1"
            paper_id: 论文ID，用于按论文分类存储

        Returns:
            图片文件路径，失败返回 None
        """
        try:
            from PIL import Image
            import io

            figures_dir = self._get_figures_dir(pdf_path, paper_id)

            # 确保目录存在
            figures_path = Path(figures_dir)
            figures_path.mkdir(parents=True, exist_ok=True)

            # 生成文件名：{figure_id}.png（对齐 Qasper 格式）
            # 例如: 3-Table1-1.png, 5-Figure1-2.png
            filename = f"{figure_id}.png"
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

    def _extract_and_save_images(self, pdf_path: str, multimodal_data: Dict[str, Any], paper_id: str = None) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        提取并保存图片，返回图片路径映射

        策略（优先使用 docling 已保存的图片，避免重复提取）：
        1. docling 已保存图片（saved_path 存在）→ 复制到 paper_id 目录
        2. 无 saved_path（如 MultimodalPDFExtractor fallback）→ PyMuPDF 重新提取

        Args:
            pdf_path: PDF文件路径
            multimodal_data: 多模态数据（包含 images 列表）
            paper_id: 论文ID（用于按论文分类存储）

        Returns:
            Tuple[Dict[str, str], Dict[str, int]]: (图注->图片路径, 图注->页码) 的映射
        """
        image_paths: Dict[str, str] = {}
        image_pages: Dict[str, int] = {}

        images = multimodal_data.get("images", [])
        if not images:
            return image_paths, image_pages

        # 直接使用 docling flat 路径（已包含页码，不会冲突）
        # 图片路径: data/figures/{page}-Figure{idx}.png
        for idx, img_info in enumerate(images):
            saved_path = img_info.get("saved_path")
            caption = img_info.get("caption") or ""
            page_num = img_info.get("page_number", 0)

            if saved_path and Path(saved_path).exists():
                # 使用 (caption, page_num, idx) 作为复合键，避免 caption 重复导致覆盖
                key = (caption, page_num, idx)
                if key not in image_paths:
                    image_paths[key] = saved_path
                    image_pages[key] = page_num
                    logger.debug(f"🖼️ [docling] 图片路径: {saved_path}")

        if image_paths:
            logger.info(f"🖼️ 使用 docling 图片 {len(image_paths)} 张")

        return image_paths, image_pages

    def _extract_and_save_tables(
        self,
        pdf_path: str,
        multimodal_data: Dict[str, Any],
        paper_id: str = None
    ) -> Tuple[Dict[Tuple[str, int, int], Tuple[str, str, str]], Dict[Tuple[str, int, int], int]]:
        """
        提取并保存表格（CSV/PNG），返回表格路径映射

        策略（优先使用 docling 已保存的表格）：
        1. docling 已保存（saved_csv_path/saved_png_path 存在）→ 复制到 paper_id 目录
        2. 无 saved_path → 跳过（表格由 docling 原始保存）

        Args:
            pdf_path: PDF文件路径
            multimodal_data: 多模态数据（包含 tables 列表）
            paper_id: 论文ID（用于按论文分类存储）

        Returns:
            Tuple[Dict, Dict]: (表注->(csv_path, png_path, caption), 表注->页码) 的映射
        """
        table_paths: Dict[str, Tuple[str, str, str]] = {}  # caption -> (csv_path, png_path, caption)
        table_pages: Dict[str, int] = {}

        tables = multimodal_data.get("tables", [])
        if not tables:
            return table_paths, table_pages

        # 直接使用 docling flat 路径（已包含页码，不会冲突）
        # 表格路径: data/tables/{page}-Table{idx}.csv/.png
        for idx, table_info in enumerate(tables):
            saved_csv = table_info.get("saved_csv_path")
            saved_png = table_info.get("saved_png_path")
            caption = table_info.get("caption") or ""
            page_num = table_info.get("page_number", 0)

            # 使用 (caption, page_num, idx) 作为复合键，避免 caption 重复导致覆盖
            key = (caption, page_num, idx)
            if key not in table_paths:
                table_paths[key] = (saved_csv or "", saved_png or "", caption)
                table_pages[key] = page_num
                if saved_csv:
                    logger.debug(f"📊 [docling] 表格 CSV: {saved_csv}")
                if saved_png:
                    logger.debug(f"📊 [docling] 表格 PNG: {saved_png}")

        if table_paths:
            logger.info(f"📊 使用 docling 表格 {len(table_paths)} 个")

        return table_paths, table_pages

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

        关联逻辑：
        1. 通过识别正文中的图片引用（Figure 1, Fig. 2, 图1等）
        2. 结合页码信息进行匹配（同一页或相邻页的图片更可能相关）

        Args:
            nodes: 分块后的节点列表
            image_paths: 图注 -> 图片路径 的映射
            image_pages: 图注 -> 页码 的映射

        Returns:
            添加了 image_path 信息的节点列表
        """
        if not image_paths:
            return nodes

        # 构建图片引用映射：figure编号 -> (图片路径, 图注, 页码)
        figure_refs: Dict[str, Tuple[str, str, int]] = {}
        for key, path in image_paths.items():
            caption_str, page_num, idx = key  # key is (caption_str, page_num, idx)
            figure_num = self._extract_figure_number(caption_str)
            if figure_num:
                figure_refs[figure_num] = (path, caption_str, page_num)

        if not figure_refs:
            logger.debug("⚠️ 未找到图片引用编号")
            return nodes

        # 预编译图片引用正则表达式
        # 支持: Figure 1, Figure 1a, Figure A1, Figure S1, Figure 1-1, Figures 1 and 2, Fig.1 等
        import re
        figure_patterns = [
            # 基础模式: Figure 1, Fig. 1, 图 1
            re.compile(r'(?:Figure|Fig\.?|图)\s*([A-Za-z]?\d+[A-Za-z]?(?:-?\d+)?)', re.IGNORECASE),
            # 带标点的: Figure 1: 或 Figure 1)
            re.compile(r'(?:Figure|Fig\.?|图)\s*([A-Za-z]?\d+[A-Za-z]?(?:-?\d+)?)\s*[:.\)]', re.IGNORECASE),
            # 复数形式: Figures 1 and 2, Figs. 1-3
            re.compile(r'(?:Figures|Figs\.?)\s*(\d+(?:\s*(?:,|and|and\s+)?\s*\d+)*)', re.IGNORECASE),
            # 无空格形式: Fig.1
            re.compile(r'(?:Fig\.?)\s*(\d+)', re.IGNORECASE),
        ]

        # 关联图片到 chunk
        associated_count = 0
        for node in nodes:
            node.metadata = dict(node.metadata)  # 复制避免修改原数据

            # 从chunk文本中提取页码
            chunk_page = self._extract_page_number_from_text(node.text)

            found_images = []  # 可能一个 chunk 引用多张图片

            for pattern in figure_patterns:
                for match in pattern.finditer(node.text):
                    fig_num = match.group(1)
                    if fig_num in figure_refs:
                        path, caption, fig_page = figure_refs[fig_num]
                        # 检查页码相近（同一页或相邻2页内）
                        page_match = (chunk_page > 0 and fig_page > 0 and abs(chunk_page - fig_page) <= 2)
                        # 去重
                        if path not in [img[0] for img in found_images]:
                            found_images.append((path, caption, fig_num, page_match))

            # 按匹配质量排序（页码匹配的优先）
            found_images.sort(key=lambda x: x[3], reverse=True)

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

    def _associate_tables_with_chunks(
        self,
        nodes: List[Node],
        table_paths: Dict[str, Tuple[str, str, str]],
        table_pages: Dict[str, int] = {}
    ) -> List[Node]:
        """
        将表格路径关联到包含对应引用的 chunk

        关联逻辑：
        1. 通过识别正文中的表格引用（Table 1, 表 2 等）
        2. 结合页码信息进行匹配（同一页或相邻页的表格更可能相关）

        Args:
            nodes: 分块后的节点列表
            table_paths: 表注 -> (csv_path, png_path, caption) 的映射
            table_pages: 表注 -> 页码 的映射

        Returns:
            添加了 table_path 信息的节点列表
        """
        if not table_paths:
            return nodes

        # 构建表格引用映射：table编号 -> (csv_path, png_path, caption, 页码)
        table_refs: Dict[str, Tuple[str, str, str, int]] = {}
        for key, paths in table_paths.items():
            caption_str, page_num, idx = key  # key is (caption_str, page_num, idx)
            table_num = self._extract_table_number(caption_str)
            if table_num:
                table_refs[table_num] = (paths[0], paths[1], caption_str, page_num)

        if not table_refs:
            logger.debug("⚠️ 未找到表格引用编号")
            return nodes

        # 预编译表格引用正则表达式
        # 支持: Table 1, Table 1a, Table A1, Tables 1 and 2 等
        import re
        table_patterns = [
            # 基础模式: Table 1, 表 1
            re.compile(r'(?:Table|表)\s*([A-Za-z]?\d+[A-Za-z]?(?:-?\d+)?)', re.IGNORECASE),
            # 带标点的: Table 1: 或 Table 1)
            re.compile(r'(?:Table|表)\s*([A-Za-z]?\d+[A-Za-z]?(?:-?\d+)?)\s*[:.\)]', re.IGNORECASE),
            # 复数形式: Tables 1 and 2
            re.compile(r'(?:Tables)\s*(\d+(?:\s*(?:,|and|and\s+)?\s*\d+)*)', re.IGNORECASE),
        ]

        # 关联表格到 chunk
        associated_count = 0
        for node in nodes:
            node.metadata = dict(node.metadata)  # 复制避免修改原数据

            # 从chunk文本中提取页码
            chunk_page = self._extract_page_number_from_text(node.text)

            found_tables = []  # 可能一个 chunk 引用多个表格

            for pattern in table_patterns:
                for match in pattern.finditer(node.text):
                    tbl_num = match.group(1)
                    if tbl_num in table_refs:
                        csv_path, png_path, caption, tbl_page = table_refs[tbl_num]
                        # 检查页码相近（同一页或相邻2页内）
                        page_match = (chunk_page > 0 and tbl_page > 0 and abs(chunk_page - tbl_page) <= 2)
                        # 去重
                        if csv_path not in [tbl[0] for tbl in found_tables]:
                            found_tables.append((csv_path, png_path, caption, tbl_num, page_match))

            # 按匹配质量排序（页码匹配的优先）
            found_tables.sort(key=lambda x: x[4], reverse=True)

            if found_tables:
                # 关联第一个表格（主要表格）
                node.metadata["table_csv_path"] = found_tables[0][0]
                node.metadata["table_png_path"] = found_tables[0][1]
                node.metadata["table_caption"] = found_tables[0][2]
                node.metadata["table_num"] = found_tables[0][3]
                node.metadata["has_table"] = True

                # 如果有多个表格，保存所有表格
                if len(found_tables) > 1:
                    node.metadata["all_tables"] = [
                        {"csv_path": tbl[0], "png_path": tbl[1], "caption": tbl[2], "table_num": tbl[3]}
                        for tbl in found_tables
                    ]
                associated_count += 1
            else:
                node.metadata["has_table"] = False

        logger.info(f"🔗 通过表格引用关联 {associated_count}/{len(nodes)} 个文本块到表格")
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
        # 匹配 Figure 1, Fig. 1, 图 1, Figure A1, Figure S1 等格式
        patterns = [
            r'(?:Figure|Fig\.?)\s*([A-Za-z]?\d+[A-Za-z]?(?:-?\d+)?)',
            r'(?:图)\s*([A-Za-z]?\d+[A-Za-z]?(?:-?\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_table_number(self, caption: str) -> Optional[str]:
        """
        从表注中提取表格编号

        Args:
            caption: 表注文本，如 "Table 1: Performance Results..."

        Returns:
            表格编号，如 "1" 或 "1a"，None 表示未找到
        """
        import re
        # 匹配 Table 1, Table 1a, Table A1 等格式
        patterns = [
            r'(?:Table|表)\s*([A-Za-z]?\d+[A-Za-z]?(?:-?\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_formula_refs(
        self,
        multimodal_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Tuple[str, int, str]], Dict[str, int]]:
        """
        提取公式引用信息

        Args:
            multimodal_data: 多模态数据（包含 formulas 列表）

        Returns:
            Tuple[Dict, Dict]: (公式编号 -> (formula_text, page_num, formula_id), 公式编号 -> 页码) 的映射
        """
        formula_refs: Dict[str, Tuple[str, int, str]] = {}  # formula_num -> (text, page, id)
        formula_pages: Dict[str, int] = {}

        formulas = multimodal_data.get("formulas", [])
        for frm in formulas:
            formula_text = frm.get("text", "")
            formula_index = frm.get("formula_index", 0)
            page_num = frm.get("page_number", 0)

            # 使用 "Eq-{index}" 或 "Equation-{index}" 作为编号
            formula_refs[f"Eq-{formula_index}"] = (formula_text, page_num, f"Eq-{formula_index}")
            formula_refs[f"Equation-{formula_index}"] = (formula_text, page_num, f"Eq-{formula_index}")
            # 也支持纯数字编号
            formula_refs[str(formula_index)] = (formula_text, page_num, f"Eq-{formula_index}")

        return formula_refs, formula_pages

    def _associate_formulas_with_chunks(
        self,
        nodes: List[Node],
        formula_refs: Dict[str, Tuple[str, int, str]],
        formula_pages: Dict[str, int] = {}
    ) -> List[Node]:
        """
        将公式关联到包含对应引用的 chunk

        关联逻辑：通过识别正文中的公式引用（Equation (1), 式(1), Eq. 1, (1) 等）
        将公式内容关联到包含对应引用的文本块。

        Args:
            nodes: 分块后的节点列表
            formula_refs: 公式编号 -> (formula_text, page_num, formula_id) 的映射
            formula_pages: 公式编号 -> 页码 的映射

        Returns:
            添加了 formula_info 信息的节点列表
        """
        if not formula_refs:
            return nodes

        # 预编译公式引用正则表达式
        # 支持: Equation (1), Eq. 1, 式(1), (1) 等
        import re
        formula_patterns = [
            re.compile(r'(?:Equation|式)\s*\((\d+)\)', re.IGNORECASE),
            re.compile(r'(?:Eq\.?)\s*(\d+)', re.IGNORECASE),
            re.compile(r'\((\d+)\)(?!\s*[,\.])', re.IGNORECASE),  # 孤立的 (1) 引用
        ]

        # 关联公式到 chunk
        associated_count = 0
        for node in nodes:
            node.metadata = dict(node.metadata)  # 复制避免修改原数据

            found_formulas = []  # 可能一个 chunk 引用多个公式

            for pattern in formula_patterns:
                for match in pattern.finditer(node.text):
                    formula_num = match.group(1)
                    if formula_num in formula_refs:
                        formula_text, page_num, formula_id = formula_refs[formula_num]
                        # 去重
                        if not any(f[2] == formula_id for f in found_formulas):
                            found_formulas.append((formula_text, page_num, formula_id))
                            associated_count += 1

            if found_formulas:
                # 只关联第一个公式（通常是最相关的）
                first_formula = found_formulas[0]
                node.metadata["formula_text"] = first_formula[0]
                node.metadata["formula_page"] = first_formula[1]
                node.metadata["formula_id"] = first_formula[2]
                # 如果有多个公式，也保存列表
                if len(found_formulas) > 1:
                    node.metadata["all_formulas"] = [
                        {"text": f[0], "page": f[1], "id": f[2]} for f in found_formulas
                    ]

        logger.info(f"🔗 通过公式引用关联 {associated_count}/{len(nodes)} 个文本块到公式")
        return nodes

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
