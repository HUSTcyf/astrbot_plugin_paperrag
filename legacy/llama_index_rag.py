"""
混合架构解析器 - 简化版本

策略：
1. 使用PDFParserAdvanced提取多模态内容（表格/公式/图片）
2. 简单的文本分块（避免llama-index依赖）
3. 保留学术论文的元数据（页码/章节/引用）
"""

import os
from pathlib import Path
from typing import List, Dict, Any
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
    混合PDF解析器（简化版）

    结合：
    - 自定义多模态提取（PDFParserAdvanced）
    - 简单的文本分块
    """

    def __init__(
        self,
        enable_multimodal: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        初始化混合解析器

        Args:
            enable_multimodal: 是否启用多模态提取
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠大小
        """
        self.enable_multimodal = enable_multimodal
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化自定义PDF解析器
        self.pdf_parser = PDFParserAdvanced(
            enable_multimodal=enable_multimodal
        )

        logger.info(f"✅ HybridPDFParser初始化完成 (chunk_size={chunk_size}, overlap={chunk_overlap})")

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

            logger.debug(f"✅ PDF解析完成: {filename} ({len(enhanced_text)} chars)")
            return [document]

        except Exception as e:
            logger.error(f"❌ PDF解析失败 {pdf_path}: {e}")
            return []

    def _build_enhanced_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """构建增强的文本（包含多模态占位符）"""
        multimodal_data = metadata.get("multimodal_data", {})

        if not multimodal_data:
            return text

        text_parts = [text]

        # 添加表格
        tables = multimodal_data.get("tables", [])
        if tables:
            text_parts.append("\n\n=== Tables ===\n")
            for table in tables:
                caption = table.get("caption", "")
                markdown = table.get("markdown", "")
                page_num = table.get("page_number", 0)
                text_parts.append(f"\n[Table on page {page_num}: {caption}]\n{markdown}\n")

        # 添加公式
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

            # 简单分块
            logger.debug(f"🔄 分块处理...")
            all_nodes = []
            chunk_index = 0

            for doc in documents:
                nodes = self._split_text(doc.text, doc.metadata, chunk_index)
                all_nodes.extend(nodes)
                chunk_index += len(nodes)

            logger.debug(f"✅ 分块完成: {len(all_nodes)} 个nodes")
            return all_nodes

        except Exception as e:
            logger.error(f"❌ PDF分块失败 {pdf_path}: {e}")
            return []

    def _split_text(self, text: str, base_metadata: Dict[str, Any], start_chunk_index: int = 0) -> List[Node]:
        """简单的文本分块"""
        nodes = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        chunk_index = start_chunk_index

        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # 保存当前chunk
                if current_chunk.strip():
                    nodes.append(Node(
                        text=current_chunk.strip(),
                        metadata={
                            **base_metadata,
                            "chunk_index": chunk_index
                        }
                    ))
                    chunk_index += 1

                # 开始新chunk
                current_chunk = para + "\n\n"

        # 保存最后一个chunk
        if current_chunk.strip():
            nodes.append(Node(
                text=current_chunk.strip(),
                metadata={
                    **base_metadata,
                    "chunk_index": chunk_index
                }
            ))

        return nodes
