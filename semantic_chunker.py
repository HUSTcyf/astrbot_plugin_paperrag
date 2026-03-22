"""
语义感知文档分块模块
支持基于标题层级、段落边界的智能分块策略
集成多模态内容提取（图片、表格、公式）
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# 多模态提取（可选）
MULTIMODAL_AVAILABLE = False
try:
    from multimodal_extractor import (
        MultimodalPDFExtractor,
        ExtractedImage,
        ExtractedTable,
        ExtractedFormula
    )
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    print(e)
    MULTIMODAL_AVAILABLE = False


@dataclass
class Chunk:
    """文本块"""
    content: str
    metadata: Dict[str, Any]
    chunk_type: str  # 'title', 'paragraph', 'table', 'formula', 'code', 'image'

    # 多模态内容（可选）
    image_data: Optional[bytes] = None  # 图片数据
    table_data: Optional[str] = None  # 表格数据
    formula_latex: Optional[str] = None  # 公式LaTeX


class SemanticChunker:
    """语义感知分块器"""

    def __init__(self,
                 chunk_size: int = 512,
                 overlap: int = 0,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1024):
        """
        初始化语义分块器

        Args:
            chunk_size: 目标块大小（字符数）
            overlap: 重叠大小（字符数）
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_by_semantic(self, text: str, filename: str = "") -> List[Chunk]:
        """
        基于语义边界进行分块（标题+段落）

        策略：
        1. 按段落分割（双换行）
        2. 识别标题（# 标记）
        3. 合并小段直到达到目标大小
        4. 添加重叠

        Args:
            text: 提取的文本
            filename: 文件名

        Returns:
            文档块列表
        """
        chunks: List[Chunk] = []

        if not text or not text.strip():
            return chunks

        lines = text.split('\n')
        current_section = []
        current_type = "paragraph"

        for line in lines:
            stripped = line.strip()

            # 检测标题（Markdown格式）
            if stripped.startswith('#'):
                # 保存之前积累的内容
                if current_section:
                    merged_chunk = self._merge_with_overlap('\n'.join(current_section))
                    if merged_chunk:
                        chunks.append(Chunk(
                            content=merged_chunk,
                            metadata={"chunk_index": len(chunks), "file_name": filename},
                            chunk_type="paragraph"
                        ))

                current_section = [stripped]
                current_type = "title"
                continue

            # 检测代码块
            if stripped.startswith('```') or stripped.startswith('~~~'):
                current_type = "code"
                current_section.append(stripped)
                continue

            # 检测表格
            if '|' in stripped and line.count('|') >= 2:
                current_type = "table"
                current_section.append(stripped)
                continue

            # 检测公式（LaTeX）
            if any(pattern in stripped for pattern in [r'$$', r'\[', r'\(', r'\begin']):
                current_type = "formula"
                current_section.append(stripped)
                continue

            # 普通段落
            if stripped:
                current_section.append(stripped)

        # 处理最后一段
        if current_section:
            merged_chunk = self._merge_with_overlap('\n'.join(current_section))
            if merged_chunk:
                chunks.append(Chunk(
                    content=merged_chunk,
                    metadata={"chunk_index": len(chunks), "file_name": filename},
                    chunk_type=current_type
                ))

        return chunks

    def chunk_by_sliding_window(self, text: str, filename: str = "") -> List[Chunk]:
        """
        滑动窗口分块（带重叠）

        适合长文本的密集检索

        Args:
            text: 提取的文本
            filename: 文件名

        Returns:
            文档块列表
        """
        chunks: List[Chunk] = []

        if not text or not text.strip():
            return chunks

        # 预处理：规范化换行
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # 找到最近的段落边界（避免在单词中间切断）
            if end < len(text):
                # 向前查找最近的换行符
                boundary = text.rfind('\n', start, end)
                if boundary > start + self.min_chunk_size:
                    end = boundary + 1

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={"chunk_index": chunk_index, "file_name": filename},
                    chunk_type="paragraph"
                ))
                chunk_index += 1

            # 移动窗口（带重叠）
            start = end - self.overlap
            if start < 0:
                start = 0
            elif start >= len(text):
                break

        return chunks

    def _merge_with_overlap(self, text: str) -> str:
        """添加重叠到文本块"""
        # 简化实现：当前不添加重叠
        # 可以改进为：从上一块的末尾获取 overlap 字符
        return text

    def chunk_document_smart(self, text: str, filename: str = "") -> List[Chunk]:
        """
        智能文档分块（混合策略）

        策略：
        - 短文本（<2000字符）：语义分块
        - 长文本：滑动窗口分块

        Args:
            text: 提取的文本
            filename: 文件名

        Returns:
            文档块列表
        """
        if len(text) < 2000:
            # 短文本：语义分块
            return self.chunk_by_semantic(text, filename)
        else:
            # 长文本：滑动窗口分块
            return self.chunk_by_sliding_window(text, filename)


class PDFParserAdvanced:
    """高级 PDF 解析器（使用 PyMuPDF，集成多模态提取）"""

    def __init__(self,
                 use_unstructured: bool = False,
                 enable_multimodal: bool = True):
        """
        初始化PDF解析器

        Args:
            use_unstructured: 保留参数以兼容旧接口，但不再使用
            enable_multimodal: 是否启用多模态提取（图片、表格、公式）
        """
        if not fitz:
            logger.warning("PyMuPDF 不可用，PDF解析功能将被禁用")
            self.available = False
            self.enable_multimodal = False
            return

        self.available = True
        self.enable_multimodal = enable_multimodal and MULTIMODAL_AVAILABLE

        # 初始化多模态提取器
        if self.enable_multimodal:
            try:
                self.multimodal_extractor = MultimodalPDFExtractor(
                    extract_images=True,
                    extract_tables=True,
                    extract_formulas=True,
                    fallback_to_text=True
                )
                logger.info("✅ 多模态提取器已启用")
            except Exception as e:
                logger.warning(f"⚠️ 多模态提取器初始化失败: {e}")
                self.enable_multimodal = False

    def parse_pdf(self, pdf_path: str) -> tuple[str, Dict[str, Any]]:
        """
        解析PDF文件

        Args:
            pdf_path: PDF文件路径

        Returns:
            (提取的文本, 元数据字典)
        """
        filename = Path(pdf_path).name

        if not self.available:
            raise ImportError("PyMuPDF 不可用，请安装: pip install PyMuPDF")

        try:
            # 如果启用多模态，使用多模态提取
            if self.enable_multimodal:
                return self._parse_with_multimodal(pdf_path)
            else:
                return self._parse_with_pymupdf(pdf_path)

        except Exception as e:
            raise Exception(f"PDF解析失败 {filename}: {e}")

    def _parse_with_multimodal(self, pdf_path: str) -> tuple[str, Dict[str, Any]]:
        """使用多模态提取器解析"""
        extracted = self.multimodal_extractor.extract(pdf_path)

        # 构建文本（包含图片、表格、公式的占位符）
        text_parts = []

        # 添加原始文本
        if extracted.text:
            text_parts.append(extracted.text)

        # 添加图片占位符
        if extracted.images:
            for img in extracted.images:
                caption = f" [{img.caption or f'Figure on page {img.page_number}'}]"

        # 添加表格（Markdown格式）
        if extracted.tables:
            for table in extracted.tables:
                if table.markdown:
                    text_parts.append(f"\n{table.markdown}\n")

        # 添加公式
        if extracted.formulas:
            for formula in extracted.formulas:
                text_parts.append(f"\n$$ {formula.text} $$\n")

        # 合并文本
        full_text = '\n'.join(text_parts)

        # 构建元数据
        metadata = {
            "file_name": extracted.file_name,
            "total_pages": extracted.text.count('[Page ') if extracted.text else 0,
            "parser": "PyMuPDF-Multimodal",
            "images_count": len(extracted.images),
            "tables_count": len(extracted.tables),
            "formulas_count": len(extracted.formulas),
            "multimodal_data": {
                "images": [
                    {
                        "page_number": img.page_number,
                        "image_index": img.image_index,
                        "caption": img.caption,
                        "bbox": img.bbox,
                        "has_image_bytes": img.image_bytes is not None
                    }
                    for img in extracted.images
                ],
                "tables": [
                    {
                        "page_number": table.page_number,
                        "table_index": table.table_index,
                        "caption": table.caption,
                        "rows": len(table.data) if table.data else 0,
                        "markdown": table.markdown
                    }
                    for table in extracted.tables
                ],
                "formulas": [
                    {
                        "page_number": formula.page_number,
                        "formula_index": formula.formula_index,
                        "text": formula.text[:100],  # 只保留前100字符
                        "type": formula.type
                    }
                    for formula in extracted.formulas
                ]
            }
        }

        return full_text, metadata

    def _parse_with_pymupdf(self, pdf_path: str) -> tuple[str, Dict[str, Any]]:
        """使用 PyMuPDF 解析（结构化提取）"""
        doc = fitz.open(pdf_path)
        text_parts = []
        metadata = {
            "file_name": Path(pdf_path).name,
            "total_pages": len(doc),
            "parser": "PyMuPDF"
        }

        # 按页提取，保留页码信息
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                # 添加页码标记
                text_parts.append(f"\n[Page {page_num}]\n{text}")

            # 提取图片信息（仅统计）
            image_list = page.get_images()
            if image_list:
                metadata["total_images"] = metadata.get("total_images", 0) + len(image_list)

        doc.close()

        full_text = '\n'.join(text_parts)
        return full_text, metadata

    def parse_and_chunk(self, pdf_path: str, chunker: SemanticChunker) -> List[Dict[str, Any]]:
        """
        解析PDF并智能分块

        Args:
            pdf_path: PDF文件路径
            chunker: 语义分块器

        Returns:
            文档块列表（可直接用于向量化）

        策略：
        - 图片/表格/公式：保留完整，不分块
        - 文本：使用语义分块
        - 可以将完整的多模态内容与文本chunk拼接
        """
        # 解析PDF
        text, metadata = self.parse_pdf(pdf_path)

        # 提取多模态数据
        multimodal_data = metadata.get("multimodal_data", {})

        # 智能分块（仅对文本）
        chunks = chunker.chunk_document_smart(text, metadata["file_name"])

        # 格式化为向量化格式
        vector_chunks = []

        for chunk in chunks:
            chunk_dict = {
                "text": chunk.content,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "chunk_type": chunk.chunk_type
                }
            }

            # 添加多模态内容（不分块，完整保留）
            if chunk.image_data:
                chunk_dict["image_data"] = chunk.image_data

            if chunk.table_data:
                chunk_dict["table_data"] = chunk.table_data

            if chunk.formula_latex:
                chunk_dict["formula_latex"] = chunk.formula_latex

            vector_chunks.append(chunk_dict)

        # 添加独立的图片块（完整保留）
        if multimodal_data.get("images"):
            for img_info in multimodal_data["images"]:
                vector_chunks.append({
                    "text": f"[Image: {img_info.get('caption', 'Figure')}]",
                    "metadata": {
                        **metadata,
                        "chunk_type": "image",
                        "page_number": img_info["page_number"],
                        "image_index": img_info["image_index"],
                        "caption": img_info.get("caption"),
                        "bbox": img_info.get("bbox")
                    }
                })

        # 添加独立的表格块（完整保留）
        if multimodal_data.get("tables"):
            for table_info in multimodal_data["tables"]:
                vector_chunks.append({
                    "text": table_info.get("markdown", table_info.get("caption", "Table")),
                    "metadata": {
                        **metadata,
                        "chunk_type": "table",
                        "page_number": table_info["page_number"],
                        "table_index": table_info["table_index"],
                        "caption": table_info.get("caption"),
                        "is_table": True
                    },
                    "table_data": table_info.get("markdown")
                })

        # 添加独立的公式块（完整保留）
        if multimodal_data.get("formulas"):
            for formula_info in multimodal_data["formulas"]:
                vector_chunks.append({
                    "text": f"$$ {formula_info['text']} $$",
                    "metadata": {
                        **metadata,
                        "chunk_type": "formula",
                        "page_number": formula_info["page_number"],
                        "formula_type": formula_info["type"]
                    },
                    "formula_latex": formula_info["text"]
                })

        return vector_chunks
