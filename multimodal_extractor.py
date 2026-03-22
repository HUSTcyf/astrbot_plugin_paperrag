"""
多模态PDF内容提取器
支持提取图片、表格、公式等结构化内容，保留位置和上下文信息
支持优雅降级：如果依赖不可用，自动禁用相应功能
"""

import io
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 条件导入（优雅降级）
from PIL import Image
PIL_AVAILABLE = True

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logger.error("❌ PyMuPDF 不可用，PDF提取功能将被禁用")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError as e:
    PDFPLUMBER_AVAILABLE = False
    PDFPLUMBER_IMPORT_ERROR = str(e)
    logger.warning(f"⚠️ pdfplumber 不可用: {e}")


@dataclass
class ExtractedImage:
    """提取的图片"""
    page_number: int
    image_index: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    image: Optional[Image.Image] = None
    image_bytes: Optional[bytes] = None
    base64_data: Optional[str] = None
    caption: Optional[str] = None  # 图注（Figure X）
    context_before: Optional[str] = None  # 前文
    context_after: Optional[str] = None  # 后文


@dataclass
class ExtractedTable:
    """提取的表格"""
    page_number: int
    table_index: int
    bbox: Tuple[float, float, float, float]
    html: Optional[str] = None  # HTML格式
    markdown: Optional[str] = None  # Markdown格式
    csv: Optional[str] = None  # CSV格式
    data: Optional[List[List[str]]] = None  # 原始数据
    caption: Optional[str] = None  # 表注（Table X）
    context: Optional[str] = None  # 上下文


@dataclass
class ExtractedFormula:
    """提取的公式"""
    page_number: int
    formula_index: int
    text: str  # LaTeX或文本表示
    bbox: Optional[Tuple[float, float, float, float]] = None
    type: str = "unknown"  # latex, inline, display


@dataclass
class ExtractedContent:
    """提取的多模态内容"""
    file_name: str
    images: List[ExtractedImage]
    tables: List[ExtractedTable]
    formulas: List[ExtractedFormula]
    text: str  # 完整文本


class MultimodalPDFExtractor:
    """多模态PDF内容提取器（支持优雅降级）"""

    def __init__(self,
                 extract_images: bool = True,
                 extract_tables: bool = True,
                 extract_formulas: bool = True,
                 image_max_size: Tuple[int, int] = (2048, 2048),
                 fallback_to_text: bool = True):
        """
        初始化提取器

        Args:
            extract_images: 是否提取图片
            extract_tables: 是否提取表格
            extract_formulas: 是否提取公式
            image_max_size: 图片最大尺寸（宽，高）
            fallback_to_text: 如果依赖不可用是否自动降级
        """
        # 检查核心依赖
        if not fitz:
            if fallback_to_text:
                logger.warning("⚠️ PyMuPDF 不可用，多模态提取将被禁用")
                self.available = False
            else:
                raise ImportError("请安装 PyMuPDF: pip install PyMuPDF")
        else:
            self.available = True

        # 根据依赖可用性调整功能
        self.extract_images = extract_images and PIL_AVAILABLE
        self.extract_tables = extract_tables and PDFPLUMBER_AVAILABLE
        self.extract_formulas = extract_formulas
        self.image_max_size = image_max_size

        # 记录降级情况
        if extract_images and not PIL_AVAILABLE:
            logger.warning("⚠️ 图片提取被禁用: Pillow 不可用")
        if extract_tables and not PDFPLUMBER_AVAILABLE:
            logger.warning("⚠️ 表格提取被禁用: pdfplumber 不可用")

        # 报告可用功能
        available_features = []
        if self.available:
            if self.extract_images:
                available_features.append("图片")
            if self.extract_tables:
                available_features.append("表格")
            if self.extract_formulas:
                available_features.append("公式")

        if available_features:
            logger.info(f"✅ 多模态提取器启用: {', '.join(available_features)}")
        else:
            logger.info("📝 多模态提取器回退到纯文本模式")

    def extract(self, pdf_path: str) -> ExtractedContent:
        """
        提取PDF的多模态内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            ExtractedContent: 提取的内容
        """
        if not self.available:
            # 回退：只返回空内容和文件名
            pdf_path = Path(pdf_path)
            logger.warning(f"⚠️ 提取器不可用，返回纯文本模式: {pdf_path.name}")
            return ExtractedContent(
                file_name=pdf_path.name,
                images=[],
                tables=[],
                formulas=[],
                text=""
            )

        pdf_path = Path(pdf_path)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"❌ 打开PDF失败 {pdf_path.name}: {e}")
            return ExtractedContent(
                file_name=pdf_path.name,
                images=[],
                tables=[],
                formulas=[],
                text=""
            )

        images = []
        tables = []
        formulas = []
        full_text_parts = []

        try:
            for page_num, page in enumerate(doc, 1):
                try:
                    # 提取文本
                    text = page.get_text()
                    full_text_parts.append(text)

                    # 提取图片
                    if self.extract_images:
                        try:
                            page_images = self._extract_images_from_page(page, page_num, text)
                            images.extend(page_images)
                        except Exception as e:
                            logger.debug(f"⚠️ 页 {page_num} 图片提取失败: {e}")

                    # 提取表格
                    if self.extract_tables:
                        try:
                            page_tables = self._extract_tables_from_page(str(pdf_path), page_num)
                            tables.extend(page_tables)
                        except Exception as e:
                            logger.debug(f"⚠️ 页 {page_num} 表格提取失败: {e}")

                    # 提取公式
                    if self.extract_formulas:
                        try:
                            page_formulas = self._extract_formulas_from_text(text, page_num)
                            formulas.extend(page_formulas)
                        except Exception as e:
                            logger.debug(f"⚠️ 页 {page_num} 公式提取失败: {e}")

                except Exception as e:
                    logger.warning(f"⚠️ 处理页 {page_num} 失败: {e}")
                    continue

            return ExtractedContent(
                file_name=pdf_path.name,
                images=images,
                tables=tables,
                formulas=formulas,
                text='\n'.join(full_text_parts)
            )

        finally:
            try:
                doc.close()
            except:
                pass

    def _extract_images_from_page(self, page, page_num: int, page_text: str, minwh: int = 50) -> List[ExtractedImage]:
        """从页面提取图片"""
        if not PIL_AVAILABLE:
            return []

        images = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)

                # 解码图片
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                # 调整大小（保留宽高比）
                image.thumbnail(self.image_max_size, Image.Resampling.LANCZOS)

                # 获取图片位置
                try:
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        bbox = img_rects[0]
                    else:
                        bbox = (0, 0, 0, 0)
                except:
                    bbox = (0, 0, 0, 0)
                x1, y1, x2, y2 = bbox
                if (x2-x1) < minwh or (y2-y1) < minwh:
                    continue

                # 查找图注（Figure X）
                caption = self._find_figure_caption(page_text, img_index)

                # 转换为base64（可选）
                base64_data = None
                # buffered = io.BytesIO()
                # image.save(buffered, format="PNG", optimize=True)
                # base64_data = base64.b64encode(buffered.getvalue()).decode()

                images.append(ExtractedImage(
                    page_number=page_num,
                    image_index=img_index,
                    bbox=bbox,
                    image=image,
                    image_bytes=image_bytes,
                    base64_data=base64_data,
                    caption=caption,
                    context_before=self._get_context_before(page_text, bbox),
                    context_after=self._get_context_after(page_text, bbox)
                ))

            except Exception as e:
                logger.debug(f"⚠️ 提取图片失败 (页 {page_num}, 图 {img_index}): {e}")
                continue

        return images

    def _extract_tables_from_page(self, pdf_path: str, page_num: int) -> List[ExtractedTable]:
        """从页面提取表格（使用 pdfplumber）"""
        tables = []

        if not PDFPLUMBER_AVAILABLE:
            return tables

        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num - 1]

                for table_index, table in enumerate(page.extract_tables()):
                    if not table or len(table) == 0:
                        continue

                    # 转换为不同格式
                    html = self._table_to_html(table)
                    markdown = self._table_to_markdown(table)
                    csv_data = self._table_to_csv(table)

                    tables.append(ExtractedTable(
                        page_number=page_num,
                        table_index=table_index,
                        bbox=(0, 0, 0, 0),  # pdfplumber 不提供精确bbox
                        html=html,
                        markdown=markdown,
                        csv=csv_data,
                        data=table
                    ))

        except Exception as e:
            logger.debug(f"⚠️ 提取表格失败 (页 {page_num}): {e}")

        return tables

    def _extract_formulas_from_text(self, text: str, page_num: int) -> List[ExtractedFormula]:
        """从文本中提取LaTeX公式"""
        formulas = []

        try:
            # 匹配 $$...$$ (display math)
            display_math = re.finditer(r'\$\$([^$]+)\$\$', text)
            for idx, match in enumerate(display_math):
                formulas.append(ExtractedFormula(
                    page_number=page_num,
                    formula_index=idx,
                    text=match.group(1).strip(),
                    type='display'
                ))

            # 匹配 \(...\) 或 $...$ (inline math)
            inline_math = re.finditer(r'\$\$?([^$]+)\$\$?|\[([^\]]+)\]', text)
            offset = len(formulas)
            for idx, match in enumerate(inline_math):
                formulas.append(ExtractedFormula(
                    page_number=page_num,
                    formula_index=offset + idx,
                    text=match.group(1) or match.group(2),
                    type='inline'
                ))

            # 匹配 \begin{equation}...\end{equation}
            equations = re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}', text, re.DOTALL)
            offset = len(formulas)
            for idx, match in enumerate(equations):
                formulas.append(ExtractedFormula(
                    page_number=page_num,
                    formula_index=offset + idx,
                    text=match.group(1).strip(),
                    type='latex'
                ))

        except Exception as e:
            logger.debug(f"⚠️ 公式提取失败 (页 {page_num}): {e}")

        return formulas

    def _find_figure_caption(self, text: str, image_index: int) -> Optional[str]:
        """查找图注（Figure X）"""
        # 匹配 "Figure 1:", "Fig. 1", "图1" 等
        patterns = [
            r'(?:Figure|Fig\.)\s*(\d+)[:.]\s*([^\n]+)',
            r'图\s*(\d+)[:.]\s*([^\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"Figure {match.group(1)}: {match.group(2).strip()}"

        return None

    def _get_context_before(self, text: str, bbox: Tuple[float, float, float, float], max_chars: int = 200) -> Optional[str]:
        """获取图片前的文本上下文"""
        # 简化实现：返回文本开头
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    def _get_context_after(self, text: str, bbox: Tuple[float, float, float, float], max_chars: int = 200) -> Optional[str]:
        """获取图片后的文本上下文"""
        # 简化实现：返回文本末尾
        if len(text) > max_chars:
            return "..." + text[-max_chars:]
        return text

    def _table_to_html(self, table: List[List[str]]) -> str:
        """将表格转换为HTML"""
        if not table or len(table) == 0:
            return ""

        html = ["<table>"]
        for i, row in enumerate(table):
            tag = "th" if i == 0 else "td"
            html.append(f"<tr>{''.join(f'<{tag}>{cell or ""}</{tag}>' for cell in row)}</tr>")
        html.append("</table>")
        return '\n'.join(html)

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """将表格转换为Markdown"""
        if not table or len(table) == 0:
            return ""

        lines = []
        for i, row in enumerate(table):
            line = "| " + " | ".join(str(cell or "") for cell in row) + " |"
            lines.append(line)
            if i == 0:
                # 添加分隔线
                lines.append("|" + "|".join(["---"] * len(row)) + "|")
        return '\n'.join(lines)

    def _table_to_csv(self, table: List[List[str]]) -> str:
        """将表格转换为CSV"""
        if not table or len(table) == 0:
            return ""

        lines = []
        for row in table:
            line = ",".join(str(cell or "") for cell in row)
            lines.append(line)
        return '\n'.join(lines)
