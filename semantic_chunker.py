"""
PDF解析器模块 - 专为学术论文优化的多模态提取

特性：
- 使用PyMuPDF高效解析PDF
- 提取表格（转换为Markdown）
- 提取公式（保留LaTeX格式）
- 提取图片（保留元数据）
- 保留页码和章节结构

注意：分块功能已移至llama-index NodeParser
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

import fitz  # PyMuPDF

# 多模态提取（可选）
from .multimodal_extractor import (
    MultimodalPDFExtractor,
    ExtractedImage,
    ExtractedTable,
    ExtractedFormula
)
MULTIMODAL_AVAILABLE = True


@dataclass
class Chunk:
    """文本块（已弃用，保留用于向后兼容）"""
    content: str
    metadata: Dict[str, Any]
    chunk_type: str  # 'title', 'paragraph', 'table', 'formula', 'code', 'image'

    # 多模态内容（可选）
    image_data: Optional[bytes] = None  # 图片数据
    table_data: Optional[str] = None  # 表格数据
    formula_latex: Optional[str] = None  # 公式LaTeX


class PDFParserAdvanced:
    """高级 PDF 解析器（使用 PyMuPDF，集成多模态提取）"""

    def __init__(self, enable_multimodal: bool = True):
        """
        初始化PDF解析器

        Args:
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
                assert MultimodalPDFExtractor is not None
                self.multimodal_extractor = MultimodalPDFExtractor(  # type: ignore
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
        filename = str(Path(pdf_path).name)

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
        assert MultimodalPDFExtractor is not None  # 确保多模态提取器可用
        assert self.multimodal_extractor is not None
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
        assert fitz is not None  # 确保fitz可用
        doc: fitz.Document = fitz.open(pdf_path)  # type: ignore
        text_parts = []
        metadata = {
            "file_name": str(Path(pdf_path).name),
            "total_pages": len(doc),
            "parser": "PyMuPDF"
        }

        # 按页提取，保留页码信息
        for page_num, page in enumerate(doc, 1):  # type: ignore[arg-type]
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
