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

# 必需依赖
from PIL import Image
import fitz  # PyMuPDF

# Docling 预导入（在 AstrBot 事件循环中避免延迟导入崩溃）
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import PictureItem, TableItem, FormulaItem
    _DOCLING_PREIMPORTED = True
except ImportError:
    _DOCLING_PREIMPORTED = False

# 多模态可用标志
MULTIMODAL_AVAILABLE = True

# 可选依赖（pdfplumber）
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError as e:
    PDFPLUMBER_AVAILABLE = False
    PDFPLUMBER_IMPORT_ERROR = str(e)
    logger.warning(f"⚠️ pdfplumber 不可用，表格提取功能将被禁用: {e}")

# 预编译的行号正则表达式模式（用于检测页边距行号）
_LINE_NUMBER_PATTERNS = [
    re.compile(r'^\s*\d+\s*$', re.IGNORECASE),            # 纯数字: "1", "  2  "
    re.compile(r'^\s*\d+\.\s*$', re.IGNORECASE),          # "1.", "2."
    re.compile(r'^\s*\d+\)\s*$', re.IGNORECASE),          # "1)", "2)"
    re.compile(r'^\s*\[\d+\]\s*$', re.IGNORECASE),       # "[1]", "[2]"
    re.compile(r'^\s*\(\d+\)\s*$', re.IGNORECASE),       # "(1)", "(2)"
    re.compile(r'^\s*\d+[a-zA-Z]?\s*$', re.IGNORECASE),  # "1a", "2b"
]


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
    saved_path: Optional[str] = None  # 文件系统路径（docling 保存后填充）


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
    saved_csv_path: Optional[str] = None  # CSV 文件路径
    saved_png_path: Optional[str] = None  # PNG 图片路径


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
                 fallback_to_text: bool = True,
                 nms_iou_threshold: float = 0.5,
                 enable_nms: bool = True):
        """
        初始化提取器

        Args:
            extract_images: 是否提取图片
            extract_tables: 是否提取表格
            extract_formulas: 是否提取公式
            image_max_size: 图片最大尺寸（宽，高）
            fallback_to_text: 如果依赖不可用是否自动降级
            nms_iou_threshold: NMS IoU 阈值（0-1），默认0.5
            enable_nms: 是否启用NMS图片去重，默认True
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
        self.extract_images = extract_images
        self.extract_tables = extract_tables and PDFPLUMBER_AVAILABLE
        self.extract_formulas = extract_formulas
        self.image_max_size = image_max_size

        # NMS 配置
        self.nms_iou_threshold = nms_iou_threshold
        self.enable_nms = enable_nms

        # 记录降级情况
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

    def _extract_text_without_line_numbers(self, page: fitz.Page) -> str:
        """
        从页面提取文本，同时过滤掉两侧的行号

        行号特征：
        1. 位置：在页面左侧（x0很小）或右侧（x1接近页面宽度）
        2. 格式：纯数字、数字+点、数字+括号，如 "1", "2.", "1)"
        3. 字体：通常比正文小

        Args:
            page: PyMuPDF页面对象

        Returns:
            过滤后的文本
        """
        # 获取页面尺寸
        page_width = page.rect.width

        # 行号过滤阈值：页宽的百分比
        left_margin_threshold = page_width * 0.08   # 左侧8%以内认为是行号
        right_margin_threshold = page_width * 0.92    # 右侧92%以外认为是行号

        # 使用预编译的行号正则表达式
        # 使用 "dict" 模式获取带位置信息的文本块
        page_dict = page.get_text("dict")

        text_lines = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # type 0 = text block
                continue

            for line in block.get("lines", []):
                bbox = line.get("bbox", [])
                if not bbox:
                    continue

                x0, y0, x1, y1 = bbox

                # 检查是否是边缘位置（可能是行号）
                is_left_edge = x0 < left_margin_threshold
                is_right_edge = x1 > right_margin_threshold

                # 提取行文本
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")

                if not line_text.strip():
                    continue

                # 判断是否是行号
                is_line_number = False
                if (is_left_edge or is_right_edge):
                    for pattern in _LINE_NUMBER_PATTERNS:
                        if pattern.match(line_text.strip()):
                            is_line_number = True
                            break

                # 如果不是行号，添加到结果
                if not is_line_number:
                    text_lines.append(line_text)

        return '\n'.join(text_lines)

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
            pdf_path_obj = Path(pdf_path)
            logger.warning(f"⚠️ 提取器不可用，返回纯文本模式: {pdf_path_obj.name}")
            return ExtractedContent(
                file_name=str(pdf_path_obj.name),
                images=[],
                tables=[],
                formulas=[],
                text=""
            )

        pdf_path_obj = Path(pdf_path)
        try:
            assert fitz is not None
            doc = fitz.open(pdf_path_obj)
        except Exception as e:
            logger.error(f"❌ 打开PDF失败 {pdf_path_obj.name}: {e}")
            return ExtractedContent(
                file_name=str(pdf_path_obj.name),
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
            for page_num, page in enumerate(doc, 1):  # type: ignore[arg-type]
                try:
                    # 提取文本（过滤行号）
                    text = self._extract_text_without_line_numbers(page)
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
                            page_tables = self._extract_tables_from_page(str(pdf_path), page_num, text)
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
                file_name=str(pdf_path_obj.name),
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

    def _extract_images_from_page(self, page, page_num: int, page_text: str, minwh: int = 10) -> List[ExtractedImage]:
        """从页面提取图片

        Args:
            page: PyMuPDF页面对象
            page_num: 页码
            page_text: 页面文本
            minwh: 最小图片宽高（像素），默认10
        """
        """从页面提取图片"""

        images = []
        image_list = page.get_images(full=True)

        # 🔧 修复：添加有效图片计数器，用于正确匹配图注
        valid_image_count = 0

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]

                base_image = page.parent.extract_image(xref)

                # 解码图片
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                # 🔧 修复1: 在调整大小前先检查原始图片尺寸
                orig_width, orig_height = image.size
                if orig_width < minwh or orig_height < minwh:
                    continue  # 🔧 关键：跳过小图时不增加valid_image_count

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

                # 查找图注（Figure X）
                # 🔧 关键修复：使用valid_image_count而不是img_index，确保图注正确匹配
                caption = self._find_figure_caption(page_text, valid_image_count, bbox)

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

                # 🔧 有效图片计数器+1（只有在成功添加图片后才增加）
                valid_image_count += 1

            except Exception as e:
                logger.debug(f"⚠️ 提取图片失败 (页 {page_num}, 图 {img_index}): {e}")
                continue

        # 🔧 修复3: 使用改进的去重策略（NMS + 内容去重）
        if self.enable_nms:
            images = self._deduplicate_images(images, page_num)

        return images

    def _deduplicate_images(self, images: List[ExtractedImage], page_num: int, keep_no_cap=False) -> List[ExtractedImage]:
        """
        使用多级策略过滤重复图片

        策略：
        0. 基于图片尺寸去重：相同尺寸的图片认为是重复的
        1. ⚠️ 跳过基于图注去重：让所有相同图注的图片保留，用于计算完整大图
        2. NMS位置去重：IoU > 阈值的重叠图片，保留面积大的

        Args:
            images: 提取的图片列表
            page_num: 页码（用于日志）

        Returns:
            过滤后的图片列表
        """
        if not images:
            return images

        original_count = len(images)
        current_images = images

        # ⚠️ 跳过基于图片尺寸去重：相同尺寸可能是不同的子图
        # ⚠️ 跳过基于图注的去重：保留所有相同图注的图片，后续在 hybrid_parser 中合并为完整大图


        # NMS位置去重（仅对有效bbox的图片）- 保留
        valid_bbox_images = [img for img in current_images if img.bbox != (0, 0, 0, 0)]
        no_bbox_images = [img for img in current_images if img.bbox == (0, 0, 0, 0)]

        # 如果没有有效bbox的图片，直接返回
        if not valid_bbox_images:
            return current_images

        # 按面积排序（从大到小），确保优先处理大图
        image_areas = []
        for idx, img in enumerate(valid_bbox_images):
            x1, y1, x2, y2 = img.bbox
            width = x2 - x1 if x2 > x1 else 0
            height = y2 - y1 if y2 > y1 else 0
            area = width * height
            image_areas.append((idx, area, img))

        # 按面积降序排序
        image_areas.sort(key=lambda x: x[1], reverse=True)

        # 🔧 修复NMS逻辑：使用remove_indices而不是keep_indices
        remove_indices = set()

        for i, (idx_i, area_i, img_i) in enumerate(image_areas):
            if idx_i in remove_indices:
                continue

            # 检查与后续图片的重叠
            for j in range(i + 1, len(image_areas)):
                idx_j, area_j, img_j = image_areas[j]

                if idx_j in remove_indices:
                    continue

                # 计算当前图片与后续图片的 IoU
                iou = self._calculate_iou(img_i.bbox, img_j.bbox)

                if iou > self.nms_iou_threshold:
                    # 图片重叠，标记移除面积较小的（后续图片）
                    remove_indices.add(idx_j)
                    logger.debug(f"🔍 [NMS 页 {page_num}] 图片 {idx_j} 与 {idx_i} 重叠 (IoU={iou:.2f})，移除较小的")

        # 返回未被移除的图片
        nms_filtered_images = [valid_bbox_images[idx] for idx in range(len(valid_bbox_images)) if idx not in remove_indices]

        # 添加没有bbox信息的图片
        final_images = nms_filtered_images + no_bbox_images

        # 🔧 安全机制：如果NMS去重后图片数量为0，禁用NMS去重
        if len(final_images) == 0 and len(current_images) > 0:
            logger.warning(f"⚠️ [NMS 页 {page_num}] 过滤后图片数量为0，禁用NMS去重")
            return current_images

        if len(current_images) > len(final_images):
            logger.info(f"🔍 [NMS 页 {page_num}] 位置重叠过滤: {len(current_images)} → {len(final_images)} 张")

        return final_images

    def _calculate_iou(self, bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float]) -> float:
        """
        计算两个 Bbox 的 IoU (Intersection over Union)

        Args:
            bbox1: 第一个边界框 (x1, y1, x2, y2)
            bbox2: 第二个边界框 (x1, y1, x2, y2)

        Returns:
            IoU 值 (0-1 之间)
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 计算交集区域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # 如果没有交集
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        # 交集面积
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集面积
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area

        # 避免除以零
        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _extract_tables_from_page(self, pdf_path: str, page_num: int, page_text: str = "") -> List[ExtractedTable]:
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

                    # 清理表格数据：将 None 替换为空字符串
                    cleaned_table = [
                        [str(cell) if cell is not None else "" for cell in row]
                        for row in table
                    ]

                    # 提取表注
                    table_caption = self._find_table_caption(page_text, table_index, (0, 0, 0, 0))

                    # 转换为不同格式
                    html = self._table_to_html(cleaned_table)
                    markdown = self._table_to_markdown(cleaned_table)
                    csv_data = self._table_to_csv(cleaned_table)

                    tables.append(ExtractedTable(
                        page_number=page_num,
                        table_index=table_index,
                        bbox=(0, 0, 0, 0),  # pdfplumber 不提供精确bbox
                        html=html,
                        markdown=markdown,
                        csv=csv_data,
                        data=cleaned_table,  # 使用清理后的数据
                        caption=table_caption
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

    def _extract_figure_number(self, caption: Optional[str]) -> Optional[str]:
        """
        从图注中提取图注编号（如 "Figure 1", "Figure A1", "Figure S1"）

        Args:
            caption: 图注文本

        Returns:
            图注编号，如 "1"、"A1"、"S1-1"，如果无法提取则返回 None
        """
        if not caption:
            return None

        # 🔧 修复：支持所有Figure开头的格式，不限于数字
        # 匹配 "Figure 1:", "Figure A1:", "Figure S1:", "Fig. 1a", "图1" 等
        patterns = [
            r'(?:Figure|Fig\.)\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)',  # Figure 1, Figure A1, Figure S1-1
            r'图\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)',                # 图1, 图A1
        ]

        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1)  # 返回纯编号，如 "1"、"A1"、"S1-1"

        return None

    def _find_figure_caption(self, text: str, image_index: int, bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Optional[str]:
        """
        查找图注（Figure X）

        Args:
            text: 页面文本
            image_index: 图片索引
            bbox: 图片在页面上的位置 (x1, y1, x2, y2)

        Returns:
            图注文本，如果找不到则返回 None
        """
        # 🔧 修复：匹配所有Figure开头的格式，包括字母和数字
        # 支持: Figure 1, Figure A1, Figure S1-2, Fig. 1a, 图A1, 图1-2
        patterns = [
            r'(?:Figure|Fig\.)\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)[:.]\s*([^\n]+)',
            r'图\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)[:.]\s*([^\n]+)',
        ]

        # 🔧 修复：查找所有匹配，根据图片位置选择最合适的图注
        all_matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                all_matches.append((match, match.group(1), match.group(2).strip()))

        if not all_matches:
            return None

        # 如果只有一个匹配，直接返回
        if len(all_matches) == 1:
            match, num, desc = all_matches[0]
            return f"Figure {num}: {desc}"

        # 🔧 如果有多个匹配，根据image_index选择
        # 策略：第i张图片对应第i个图注
        if image_index < len(all_matches):
            match, num, desc = all_matches[image_index]
            return f"Figure {num}: {desc}"
        else:
            # 如果图片数量超过图注数量，返回 None（让调用方处理无图注情况）
            return None

    def _find_table_caption(self, text: str, table_index: int, bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Optional[str]:
        """
        查找表注（Table X）

        Args:
            text: 页面文本
            table_index: 表格索引
            bbox: 表格在页面上的位置 (x1, y1, x2, y2)

        Returns:
            表注文本，如果找不到则返回 None
        """
        # 匹配 Table 开头的格式
        # 支持: Table 1, Table 2, Table A1, 表1, 表 1 等
        patterns = [
            r'(?:Table|表)\s*(\d+[A-Za-z]?(?:-[A-Za-z0-9]+)*)[:.]\s*([^\n]+)',
        ]

        # 查找所有匹配
        all_matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                all_matches.append((match, match.group(1), match.group(2).strip()))

        if not all_matches:
            return None

        # 如果只有一个匹配，直接返回
        if len(all_matches) == 1:
            match, num, desc = all_matches[0]
            return f"Table {num}: {desc}"

        # 如果有多个匹配，根据 table_index 选择
        if table_index < len(all_matches):
            match, num, desc = all_matches[table_index]
            return f"Table {num}: {desc}"
        else:
            # 如果表格数量超过表注数量，返回 None
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
            row_html = "".join(f"<{tag}>{cell if cell else ''}</{tag}>" for cell in row)
            html.append(f"<tr>{row_html}</tr>")
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

    def get_figures_and_tables(self, extracted: ExtractedContent) -> List[Dict[str, str]]:
        """
        生成 Qasper 格式的 figures_and_tables 列表

        Args:
            extracted: extract() 方法返回的 ExtractedContent 对象

        Returns:
            List[Dict]: [{"file": "3-Table1-1.png", "caption": "..."}, ...]
        """
        result = []

        # 处理图片
        for img in extracted.images:
            if img.caption:
                # 直接根据 caption 前缀判断是 Figure 还是 Table
                caption_upper = img.caption.upper()
                if caption_upper.startswith("FIGURE") or caption_upper.startswith("FIG."):
                    figure_type = "Figure"
                elif caption_upper.startswith("TABLE"):
                    figure_type = "Table"
                else:
                    figure_type = "Figure"  # 默认当作 Figure

                figure_num = self._extract_figure_number(img.caption) or str(img.image_index)
                # 生成文件名格式: {page}-{type}{num}-{variant}.png
                # 例如: 3-Table1-1.png, 5-Figure1-1.png
                result.append({
                    "file": f"{img.page_number}-{figure_type}{figure_num}-{img.image_index}.png",
                    "caption": img.caption
                })

        # 处理表格（如果提取了 caption）
        for table in extracted.tables:
            if table.caption:
                table_num = self._extract_table_number(table.caption) or str(table.table_index)
                result.append({
                    "file": f"{table.page_number}-Table{table_num}-{table.table_index}.png",
                    "caption": table.caption
                })

        return result

    def _extract_table_number(self, caption: Optional[str]) -> Optional[str]:
        """
        从表注中提取表格编号（如 "Table 1", "Table A1"）

        Args:
            caption: 表注文本

        Returns:
            表格编号，如 "1"、"A1"，如果无法提取则返回 None
        """
        if not caption:
            return None

        # 匹配 "Table 1:", "Table A1:", "表1" 等
        patterns = [
            r'(?:Table|表)\s*(\d+[A-Za-z]?(?:-[A-Za-z0-9]+)*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return match.group(1)

        return None


# ============================================================================
# DoclingExtractor - 基于 docling 的 PDF 多模态提取器
# ============================================================================


# 模块级标志：确保全局配置只执行一次
_GLOBAL_DOCLING_CONFIGURED = False


def _configure_docling_globals() -> None:
    """
    配置 docling 全局设置（在进程级别只执行一次）

    这个函数配置：
    1. HuggingFace 模型缓存目录
    2. PyTorch 设备选项（避免 MPS 崩溃）
    3. docling settings

    必须在任何 docling/transformers/torch 导入之前调用，或在首次使用时调用。
    """
    global _GLOBAL_DOCLING_CONFIGURED
    if _GLOBAL_DOCLING_CONFIGURED:
        return

    import os
    from pathlib import Path

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. 设置 HF_HOME，使 huggingface_hub 缓存到本地目录
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(models_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(models_dir))

    # 2. 强制 CPU 模式（避免 Apple Silicon MPS 崩溃）
    # 注意：这些设置需要在 torch 导入之前或首次使用时设置
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 3. 设置 docling settings
    from docling.datamodel.settings import settings
    settings.cache_dir = str(models_dir)
    settings.artifacts_path = str(models_dir)

    _GLOBAL_DOCLING_CONFIGURED = True
    logger.info(f"🔧 [DoclingExtractor] 全局配置完成: cache_dir={settings.cache_dir}")


class DoclingExtractor:
    """基于 docling 的 PDF 多模态提取器，支持图片和表格提取"""

    # 本地模型目录（插件目录下）
    _LOCAL_MODELS_DIR = Path(__file__).parent / "models"

    def __init__(self, fallback_extractor: Optional["MultimodalPDFExtractor"] = None):
        """
        初始化 DoclingExtractor

        Args:
            fallback_extractor: 降级使用的提取器（MultimodalPDFExtractor）
        """
        self.fallback = fallback_extractor
        self._available = self._check_available()
        self._plugin_dir = Path(__file__).parent
        self._figures_base = self._plugin_dir / "data" / "figures"
        self._tables_base = self._plugin_dir / "data" / "tables"

        # 配置本地模型目录（全局配置，只在首次执行）
        if self._available:
            _configure_docling_globals()

        if self._available:
            self._figures_base.mkdir(parents=True, exist_ok=True)
            self._tables_base.mkdir(parents=True, exist_ok=True)

        # 调试日志：检查模型状态
        if self._available:
            self._log_model_status(self._LOCAL_MODELS_DIR)

    def _log_model_status(self, models_dir: Path) -> None:
        """输出模型加载状态调试信息"""
        logger.info(f"🔧 [DoclingExtractor] 本地模型目录: {models_dir}")

        # 检查各模型是否存在（使用正确的 repo_cache_folder 名称）
        expected_models = [
            ("布局模型", "docling-project--docling-models"),
            ("公式模型", "docling-project--CodeFormulaV2"),
            ("图片分类器", "docling-project--DocumentFigureClassifier-v2.5"),
            ("RapidOCR", "RapidOcr"),
        ]

        all_found = True
        for name, folder in expected_models:
            model_path = models_dir / folder
            if model_path.exists():
                # 尝试估算大小
                try:
                    size = sum(
                        f.stat().st_size
                        for f in model_path.rglob("*")
                        if f.is_file()
                    )
                    size_mb = size / (1024 * 1024)
                    logger.info(f"   ✅ {name}: {size_mb:.1f} MB")
                except Exception:
                    logger.info(f"   ✅ {name}: 已存在")
            else:
                logger.warning(f"   ⚠️ {name}: 未找到 ({folder})")
                all_found = False

        if all_found:
            logger.info(f"✅ [DoclingExtractor] 所有模型已就绪，将从本地加载")
        else:
            logger.warning(f"⚠️ [DoclingExtractor] 部分模型缺失，将自动从 HuggingFace 下载")

    def _check_available(self) -> bool:
        """检测 docling 是否可用"""
        # 使用预导入的模块标志
        return _DOCLING_PREIMPORTED

    def extract(self, pdf_path: str) -> ExtractedContent:
        """
        使用 docling 提取 PDF 多模态内容

        Args:
            pdf_path: PDF 文件路径

        Returns:
            ExtractedContent: 提取的内容（图片、表格、文本）
        """
        if not self._available:
            if self.fallback:
                logger.warning(f"[DoclingExtractor] docling 不可用，使用 PyMuPDF fallback")
                return self.fallback.extract(pdf_path)
            return ExtractedContent(file_name=Path(pdf_path).name, images=[], tables=[], formulas=[], text="")

        # 确保全局配置已执行（可能在 __init__ 时 docling 未导入）
        _configure_docling_globals()

        try:
            return self._extract_with_docling(pdf_path)
        except (Exception, KeyboardInterrupt) as e:
            logger.warning(f"[DoclingExtractor] docling extraction failed, falling back: {e}")
            if self.fallback:
                return self.fallback.extract(pdf_path)
            return ExtractedContent(file_name=Path(pdf_path).name, images=[], tables=[], formulas=[], text="")

    def _extract_with_docling(self, pdf_path: str) -> ExtractedContent:
        """
        使用 docling 提取 PDF 内容（支持图片、表格、公式）

        由于 AstrBot 环境下 docling convert() 会触发 segfault，
        使用独立进程执行以隔离运行环境。
        """
        import subprocess
        import sys
        import tempfile
        import json
        import os

        pdf_path = Path(pdf_path)
        paper_id = pdf_path.stem

        figures_dir = self._figures_base / paper_id
        tables_dir = self._tables_base / paper_id
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        plugin_dir = Path(__file__).parent

        # 构建子进程脚本
        script = f'''
import sys
import os
import io
import json
from pathlib import Path

models_dir = Path("{plugin_dir}") / "models"
os.environ["HF_HOME"] = str(models_dir)
os.environ["TRANSFORMERS_CACHE"] = str(models_dir)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem, FormulaItem

def _table_data_to_csv(data):
    lines = []
    for row in data:
        # 将每个元素转为字符串，处理嵌套列表
        cells = []
        for cell in row:
            if isinstance(cell, list):
                cell = str(cell)
            cells.append(str(cell) if cell is not None else "")
        line = ",".join(cells)
        lines.append(line)
    return "\\n".join(lines)

def _csv_to_markdown(data):
    """将 CSV 数据转换为 Markdown 表格格式"""
    if not data:
        return ""
    lines = []
    row_count = 0
    for i, row in enumerate(data):
        row_count += 1
        cells = []
        for cell in row:
            if isinstance(cell, list):
                cell = str(cell)
            cell_str = str(cell) if cell is not None else ""
            # 转义管道符和换行符（f-string中需要双写反斜杠）
            cell_str = cell_str.replace("|", "\\\\|").replace("\\n", " ")
            cells.append(cell_str)
        lines.append("| " + " | ".join(cells) + " |")
        # 添加表头分隔符
        if i == 0:
            separator = "| " + " | ".join(["---"] * len(cells)) + " |"
            lines.append(separator)
    if row_count == 0:
        return ""
    return "\\n".join(lines)

pdf_path = Path("{pdf_path}")
paper_id = "{paper_id}"
figures_dir = Path("{figures_dir}")
tables_dir = Path("{tables_dir}")

pipeline_options = PdfPipelineOptions(
    generate_picture_images=True,
    generate_page_images=False,
    do_table_structure=True,
    do_ocr=True,
    do_formula_enrichment=False,
    images_scale=2.0,
)

converter = DocumentConverter(
    format_options={{
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }}
)

result = converter.convert(pdf_path)

images = []
tables = []
formulas = []
text_parts = []
figure_counters = {{}}
table_counters = {{}}
formula_counters = {{}}

for element, _level in result.document.iterate_items():
    if isinstance(element, PictureItem):
        page_no = element.prov[0].page_no
        figure_idx = figure_counters.get(page_no, 0) + 1
        figure_counters[page_no] = figure_idx
        if element.image is None:
            continue
        pil_image = element.image.pil_image
        filename = f"{{page_no}}-Figure{{figure_idx}}.png"
        save_path = figures_dir / filename
        pil_image.save(save_path, format="PNG")
        images.append({{
            "page_number": page_no,
            "image_index": figure_idx,
            "bbox": [0, 0, 0, 0],
            "caption": f"Figure {{figure_idx}}",
            "saved_path": str(save_path),
        }})
    elif isinstance(element, TableItem):
        page_no = element.prov[0].page_no
        table_idx = table_counters.get(page_no, 0) + 1
        table_counters[page_no] = table_idx
        table_csv = _table_data_to_csv(element.data)
        csv_filename = f"{{page_no}}-Table{{table_idx}}.csv"
        png_filename = f"{{page_no}}-Table{{table_idx}}.png"
        csv_path = tables_dir / csv_filename
        png_path = tables_dir / png_filename
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(table_csv)
        saved_png_path = None
        if element.image is not None:
            pil_image = element.image.pil_image
            pil_image.save(png_path, format="PNG")
            saved_png_path = str(png_path)
        table_markdown = _csv_to_markdown(element.data)
        tables.append({{
            "page_number": page_no,
            "table_index": table_idx,
            "bbox": [0, 0, 0, 0],
            "csv": table_csv,
            "markdown": table_markdown,
            "caption": f"Table {{table_idx}}",
            "saved_csv_path": str(csv_path),
            "saved_png_path": saved_png_path,
        }})
    elif isinstance(element, FormulaItem):
        page_no = element.prov[0].page_no if element.prov else 1
        formula_idx = formula_counters.get(page_no, 0) + 1
        formula_counters[page_no] = formula_idx
        latex_text = element.text or ""
        formulas.append({{
            "page_number": page_no,
            "formula_index": formula_idx,
            "text": latex_text,
            "bbox": [0, 0, 0, 0],
            "type": "display",
        }})

if result.document.texts:
    text_parts.extend([t.text for t in result.document.texts])

result_json = json.dumps({{
    "file_name": pdf_path.name,
    "images": images,
    "tables": tables,
    "formulas": formulas,
    "text": "\\\\n".join(text_parts),
}})

print(result_json)
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=str(plugin_dir),
            )

            if result.returncode != 0:
                raise RuntimeError(f"docling extraction failed: {result.stderr}")

            output = result.stdout.strip()
            if not output:
                raise RuntimeError("docling extraction returned empty result")

            data = json.loads(output)

            # 构建 ExtractedContent
            images_out = [
                ExtractedImage(
                    page_number=img["page_number"],
                    image_index=img["image_index"],
                    bbox=tuple(img["bbox"]),
                    caption=img.get("caption"),
                    saved_path=img.get("saved_path"),
                )
                for img in data.get("images", [])
            ]

            tables_out = [
                ExtractedTable(
                    page_number=tbl["page_number"],
                    table_index=tbl["table_index"],
                    bbox=tuple(tbl["bbox"]),
                    csv=tbl.get("csv"),
                    markdown=tbl.get("markdown"),
                    caption=tbl.get("caption"),
                    saved_csv_path=tbl.get("saved_csv_path"),
                    saved_png_path=tbl.get("saved_png_path"),
                )
                for tbl in data.get("tables", [])
            ]

            formulas_out = [
                ExtractedFormula(
                    page_number=frm["page_number"],
                    formula_index=frm["formula_index"],
                    text=frm["text"],
                    bbox=tuple(frm["bbox"]) if frm.get("bbox") else None,
                    type=frm.get("type", "unknown"),
                )
                for frm in data.get("formulas", [])
            ]

            logger.info(f"[DoclingExtractor] 提取完成: 图片={len(images_out)}, 表格={len(tables_out)}, 公式={len(formulas_out)}")

            return ExtractedContent(
                file_name=data.get("file_name", ""),
                images=images_out,
                tables=tables_out,
                formulas=formulas_out,
                text=data.get("text", ""),
            )

        finally:
            os.unlink(script_path)

    def _pil_to_bytes(self, pil_image: Image.Image) -> bytes:
        """将 PIL Image 转换为 bytes"""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return buffered.getvalue()

    def _table_data_to_csv(self, data: List[List[str]]) -> str:
        """将表格数据转换为 CSV 格式"""
        lines = []
        for row in data:
            line = ",".join(f'"{cell}"' if ',' in cell else cell for cell in row)
            lines.append(line)
        return "\n".join(lines)

    def _extract_figure_caption(self, element, index: int) -> Optional[str]:
        """从 PictureItem 提取图注"""
        return f"Figure {index}"

    def _extract_table_caption(self, element, index: int) -> Optional[str]:
        """从 TableItem 提取表注"""
        return f"Table {index}"


# ============================================================================
# PDFParserAdvanced - 高级PDF解析器，集成多模态提取和分块
# ============================================================================

@dataclass
class Chunk:
    """文本块（已弃用，保留用于向后兼容）"""
    content: str
    metadata: Dict[str, Any]
    chunk_type: str = ""  # 'title', 'paragraph', 'table', 'formula', 'code', 'image'

    # 多模态内容（可选）
    image_data: Optional[bytes] = None
    table_data: Optional[str] = None
    formula_latex: Optional[str] = None


def _build_extracted_content_from_json(data: Dict[str, Any]) -> ExtractedContent:
    """从子进程返回的 JSON 数据构建 ExtractedContent"""
    images = []
    for img in data.get("images", []):
        images.append(ExtractedImage(
            page_number=img["page_number"],
            image_index=img["image_index"],
            bbox=tuple(img["bbox"]),
            caption=img.get("caption"),
            saved_path=img.get("saved_path"),
        ))

    tables = []
    for tbl in data.get("tables", []):
        tables.append(ExtractedTable(
            page_number=tbl["page_number"],
            table_index=tbl["table_index"],
            bbox=tuple(tbl["bbox"]),
            csv=tbl.get("csv"),
            caption=tbl.get("caption"),
            saved_csv_path=tbl.get("saved_csv_path"),
            saved_png_path=tbl.get("saved_png_path"),
        ))

    formulas = []
    for frm in data.get("formulas", []):
        formulas.append(ExtractedFormula(
            page_number=frm["page_number"],
            formula_index=frm["formula_index"],
            text=frm["text"],
            bbox=tuple(frm["bbox"]) if frm.get("bbox") else None,
            type=frm.get("type", "unknown"),
        ))

    return ExtractedContent(
        file_name=data.get("file_name", ""),
        images=images,
        tables=tables,
        formulas=formulas,
        text=data.get("text", ""),
    )


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
                self.multimodal_extractor = MultimodalPDFExtractor(
                    extract_images=True,
                    extract_tables=True,
                    extract_formulas=True,
                    fallback_to_text=True
                )
                self.docling_extractor = DoclingExtractor(fallback_extractor=self.multimodal_extractor)
                logger.info("✅ DoclingExtractor 已启用（默认），MultimodalPDFExtractor 作为 fallback")
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
            if self.enable_multimodal:
                return self._parse_with_multimodal(pdf_path)
            else:
                return self._parse_with_pymupdf(pdf_path)

        except Exception as e:
            raise Exception(f"PDF解析失败 {filename}: {e}")

    def _parse_with_multimodal(self, pdf_path: str) -> tuple[str, Dict[str, Any]]:
        """使用 DoclingExtractor 提取图片/表格/公式 + PyMuPDF 提取完整文本"""
        # 1. DoclingExtractor 提取多模态内容（图片、表格、公式）
        extracted = self.docling_extractor.extract(pdf_path)

        # 2. PyMuPDF 提取完整文本（用于 chunks 和参考文献解析）
        pymupdf_text, _ = self._parse_with_pymupdf(pdf_path)

        # 3. 构建增强文本（PyMuPDF 文本 + Docling 提取的表格/公式）
        text_parts = []

        # 添加 PyMuPDF 提取的文本
        if pymupdf_text:
            text_parts.append(pymupdf_text)

        # 添加 Docling 表格 markdown
        if extracted.tables:
            for table in extracted.tables:
                if table.markdown:
                    text_parts.append(f"\n{table.markdown}\n")

        # 添加 Docling 公式
        if extracted.formulas:
            for formula in extracted.formulas:
                text_parts.append(f"\n$$ {formula.text} $$\n")

        full_text = '\n'.join(text_parts)

        metadata = {
            "file_name": extracted.file_name,
            "total_pages": extracted.text.count('[Page ') if extracted.text else 0,
            "parser": "Docling-Multimodal",
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
                        "has_image_bytes": img.image_bytes is not None,
                        "saved_path": img.saved_path
                    }
                    for img in extracted.images
                ],
                "tables": [
                    {
                        "page_number": table.page_number,
                        "table_index": table.table_index,
                        "caption": table.caption,
                        "rows": len(table.data) if table.data else 0,
                        "markdown": table.markdown,
                        "saved_csv_path": table.saved_csv_path,
                        "saved_png_path": table.saved_png_path
                    }
                    for table in extracted.tables
                ],
                "formulas": [
                    {
                        "page_number": formula.page_number,
                        "formula_index": formula.formula_index,
                        "text": formula.text[:100],
                        "type": formula.type
                    }
                    for formula in extracted.formulas
                ]
            }
        }

        # 返回: (增强文本用于chunks, PyMuPDF原文用于参考文献解析, metadata)
        return full_text, pymupdf_text, metadata

    def _parse_with_pymupdf(self, pdf_path: str) -> tuple[str, Dict[str, Any]]:
        """使用 PyMuPDF 解析（结构化提取，过滤行号）"""
        doc = fitz.open(pdf_path)
        text_parts = []
        metadata = {
            "file_name": str(Path(pdf_path).name),
            "total_pages": len(doc),
            "parser": "PyMuPDF"
        }

        for page_num, page in enumerate(doc, 1):
            page_text = self._extract_text_without_line_numbers(page)
            if page_text.strip():
                text_parts.append(f"\n[Page {page_num}]\n{page_text}")

            image_list = page.get_images()
            if image_list:
                metadata["total_images"] = metadata.get("total_images", 0) + len(image_list)

        doc.close()

        full_text = '\n'.join(text_parts)
        return full_text, metadata

    def _extract_text_without_line_numbers(self, page: fitz.Page) -> str:
        """
        从页面提取文本，同时过滤掉两侧的行号
        """
        page_width = page.rect.width
        left_margin_threshold = page_width * 0.08
        right_margin_threshold = page_width * 0.92

        page_dict = page.get_text("dict")

        text_lines = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                bbox = line.get("bbox", [])
                if not bbox:
                    continue

                x0, y0, x1, y1 = bbox
                is_left_edge = x0 < left_margin_threshold
                is_right_edge = x1 > right_margin_threshold

                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")

                if not line_text.strip():
                    continue

                is_line_number = False
                if (is_left_edge or is_right_edge):
                    for pattern in _LINE_NUMBER_PATTERNS:
                        if pattern.match(line_text.strip()):
                            is_line_number = True
                            break

                if not is_line_number:
                    text_lines.append(line_text)

        return '\n'.join(text_lines)
