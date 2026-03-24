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

# 可选依赖（pdfplumber）
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError as e:
    PDFPLUMBER_AVAILABLE = False
    PDFPLUMBER_IMPORT_ERROR = str(e)
    logger.warning(f"⚠️ pdfplumber 不可用，表格提取功能将被禁用: {e}")


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
        # 🔧 DEBUG: 显示输入图片信息
        logger.warning(f"🔍 [DEBUG 页 {page_num}] _deduplicate_images 输入: {len(images)} 张图片")
        for i, img in enumerate(images):
            caption_key = self._extract_figure_number(img.caption)
            size = f"{img.image.size[0]}x{img.image.size[1]}" if img.image else "N/A"
            # logger.warning(f"🔍 [DEBUG 页 {page_num}]   图片[{i}]: caption='{img.caption}', key='{caption_key}', size={size}")

        if not images:
            return images

        original_count = len(images)
        current_images = images

        # ⚠️ 跳过基于图片尺寸去重：相同尺寸可能是不同的子图
        # ⚠️ 跳过基于图注的去重：保留所有相同图注的图片，后续在 hybrid_parser 中合并为完整大图

        logger.warning(f"🔍 [DEBUG 页 {page_num}] 保留所有 {len(current_images)} 张图片（尺寸去重和图注去重已跳过，将在后续合并为完整大图）")

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

                    # 清理表格数据：将 None 替换为空字符串
                    cleaned_table = [
                        [str(cell) if cell is not None else "" for cell in row]
                        for row in table
                    ]

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
                        data=cleaned_table  # 使用清理后的数据
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
            图注编号，如 "Figure 1"，如果无法提取则返回 None
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
                return f"Figure {match.group(1)}"

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
            # 如果图片数量超过图注数量，使用最后一个图注
            match, num, desc = all_matches[-1]
            return f"Figure {num}: {desc}"

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
