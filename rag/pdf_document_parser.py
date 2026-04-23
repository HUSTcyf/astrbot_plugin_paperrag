from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from astrbot.api import logger

try:
    from .parser_types import Node
except ImportError:
    from parser_types import Node


class PDFDocumentParserMixin:
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

            # parse_pdf 返回 2-tuple: (text, metadata)
            text, metadata = result
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
