#!/usr/bin/env python3
"""
修复 /data/tables 目录下所有损坏的 CSV 文件。

损坏原因：旧代码将 TableData 对象的 repr() 字符串（如 "[TableCell(bbox=...)]"）
直接写入 CSV，而不是提取单元格的实际文本。

修复方案：使用 docling 从 PDF 重新提取表格

使用方法：
    python repair_table_csvs.py [--csv-only] [--force] [--limit N]
    python repair_table_csvs.py --single <csv_file>  # 单文件测试
    python repair_table_csvs.py --dry-run            # 仅预览不保存
"""

import os
import re
import sys
import io
import shutil
import warnings
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# 过滤 docling deprecated 警告 (warnings 模块)
warnings.filterwarnings("ignore", category=DeprecationWarning,
                      message=r".*TableItem\.export_to.*")
warnings.filterwarnings("ignore", category=DeprecationWarning,
                      message=r".*Usage of.*")

# 过滤 docling logging 警告
logging.getLogger("docling").setLevel(logging.ERROR)
logging.getLogger("docling_core").setLevel(logging.ERROR)

# 添加插件路径
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

TABLES_DIR = plugin_dir / "data" / "tables"
PAPERS_DIR = Path("/Volumes/ext/Master/papers")
REPAIRED_DIR = "_repaired"


def is_valid_csv(content: str) -> bool:
    """检查 CSV 内容是否是有效的表格数据（而非 repr 损坏）"""
    if not content or not content.strip():
        return False
    lines = content.strip().split("\n")
    if len(lines) < 2:
        return False
    for line in lines[:3]:
        if "TableCell(" in line or "BoundingBox(" in line or "[TableCell" in line:
            return False
    return True


# 模块级别 DocumentConverter 单例（避免每次创建占用大量内存）
_converter = None


def _get_converter():
    """获取或创建 DocumentConverter 单例"""
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()
    return _converter


def extract_table_from_pdf(pdf_path: str, page_no: int, table_idx: int) -> Optional[tuple]:
    """
    使用 docling 从 PDF 中重新提取指定页面的表格

    Returns:
        (csv_content: str, markdown_content: str, png_path: str) or None
    """
    try:
        from docling_core.types.doc.document import TableItem
        from PIL import Image

        converter = _get_converter()
        result = converter.convert(pdf_path)

        try:
            table_counters = {}
            for element, _level in result.document.iterate_items():
                if isinstance(element, TableItem):
                    page_no_elem = element.prov[0].page_no
                    idx = table_counters.get(page_no_elem, 0) + 1
                    table_counters[page_no_elem] = idx

                    if page_no_elem == page_no and idx == table_idx:
                        csv_content = table_to_csv(element)
                        md_content = table_to_markdown(element)

                        # 从 PDF 裁剪表格区域生成 PNG
                        png_data = table_to_png(element, pdf_path)

                        return csv_content, md_content, png_data
            return None
        finally:
            del result
    except Exception as e:
        print(f"    [ERROR] docling 提取失败: {e}")
        return None


def table_to_png(table_item, pdf_path: str) -> Optional[bytes]:
    """从 PDF 中裁剪表格区域生成 PNG"""
    try:
        import fitz  # PyMuPDF

        bbox = table_item.prov[0].bbox
        page_no = table_item.prov[0].page_no

        doc = fitz.open(pdf_path)
        pdf_page = doc[page_no - 1]  # 页码从1开始
        page_height = pdf_page.rect.height

        # BOTTOMLEFT: l=左边, t=下边, r=右边, b=上边
        # 转换到 PyMuPDF TOPLEFT: 需要翻转 Y 坐标
        left = bbox.l
        bottom = bbox.t
        right = bbox.r
        top = bbox.b

        # PyMuPDF Rect: x0, y0, x1, y1 (y0 < y1)
        x0 = left
        y0 = page_height - top
        x1 = right
        y1 = page_height - bottom

        # 确保坐标有效
        if y0 >= y1:
            y0, y1 = y1, y0

        clip_rect = fitz.Rect(x0, y0, x1, y1)
        mat = fitz.Matrix(2, 2)  # 2x 缩放
        pix = pdf_page.get_pixmap(matrix=mat, clip=clip_rect)
        img_data = pix.tobytes('png')

        doc.close()
        return img_data
    except Exception:
        return None


def table_to_csv(table_item) -> str:
    """将 TableItem 转换为 CSV 格式"""
    try:
        df = table_item.export_to_dataframe(doc=None)
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=True, encoding='utf-8')
        return buf.getvalue()
    except Exception:
        pass

    # 回退：手动遍历 grid
    if hasattr(table_item, 'grid'):
        rows = table_item.grid
    else:
        return ""

    lines = []
    for row in rows:
        cells = []
        for cell in row:
            if hasattr(cell, '_get_text'):
                cell_text = cell._get_text(doc=None) or ""
            elif isinstance(cell, list):
                cell_text = str(cell)
            else:
                cell_text = str(cell) if cell is not None else ""
            cell_text = cell_text.replace('"', '""')
            cells.append(cell_text)
        line = ",".join(f'"{c}"' if ',' in c or '"' in c else c for c in cells)
        lines.append(line)
    return "\n".join(lines)


def table_to_markdown(table_item) -> str:
    """将 TableItem 转换为 Markdown 格式"""
    try:
        return table_item.export_to_markdown(doc=None)
    except Exception:
        pass

    try:
        df = table_item.export_to_dataframe(doc=None)
        cols = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            cells = [str(v).replace("|", "\\|").replace("\n", " ") for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    except Exception:
        pass

    # 回退：手动遍历 grid
    if hasattr(table_item, 'grid'):
        rows = table_item.grid
    else:
        return ""

    if not rows:
        return ""
    lines = []
    for i, row in enumerate(rows):
        cells = []
        for cell in row:
            if hasattr(cell, '_get_text'):
                cell_str = cell._get_text(doc=None) or ""
            elif isinstance(cell, list):
                cell_str = str(cell)
            else:
                cell_str = str(cell) if cell is not None else ""
            cell_str = cell_str.replace("|", "\\|").replace("\n", " ")
            cells.append(cell_str)
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            separator = "| " + " | ".join(["---"] * len(cells)) + " |"
            lines.append(separator)
    return "\n".join(lines)


def parse_original_csv(csv_path: Path) -> Optional[tuple]:
    """从文件路径解析原始 CSV（已废弃，使用 parse_original_csv_content）"""
    content = csv_path.read_text(encoding="utf-8")
    return parse_original_csv_content(content)


def parse_original_csv_content(content: str) -> Optional[tuple]:
    """
    从 content 字符串中解析表格数据（使用 eval 解析 TableCell repr 字符串）

    Returns:
        (table_data: list[list[str]], num_rows: int, num_cols: int) or None
    """
    # 检查是否是损坏的 CSV
    if not ("TableCell(" in content or "table_cells" in content):
        return None

    # 提取 [TableCell(...), ...] 部分
    match = re.search(r'(\[TableCell\(.*\)\])', content, re.DOTALL)
    if not match:
        return None

    cells_repr = match.group(1)
    cells_repr_clean = re.sub(r'\s+', ' ', cells_repr)

    # 定义类型以便 eval
    code = '''
class CoordOrigin:
    TOPLEFT = None
class BoundingBox:
    def __init__(self, l, t, r, b, coord_origin):
        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.coord_origin = coord_origin
class TableCell:
    def __init__(self, bbox, row_span, col_span, start_row_offset_idx, end_row_offset_idx, start_col_offset_idx, end_col_offset_idx, text, column_header, row_header, row_section, fillable):
        self.bbox = bbox
        self.row_span = row_span
        self.col_span = col_span
        self.start_row_offset_idx = start_row_offset_idx
        self.end_row_offset_idx = end_row_offset_idx
        self.start_col_offset_idx = start_col_offset_idx
        self.end_col_offset_idx = end_col_offset_idx
        self.text = text
        self.column_header = column_header
        self.row_header = row_header
        self.row_section = row_section
        self.fillable = fillable
'''
    namespace = {}
    exec(code, namespace)

    # 修复 coord_origin 表示法
    cells_repr_fixed = cells_repr_clean.replace("<CoordOrigin.TOPLEFT: 'TOPLEFT'>", "CoordOrigin.TOPLEFT")

    try:
        cells = eval(cells_repr_fixed, namespace)
    except Exception:
        return None

    if not cells:
        return None

    # 从 TableCell 对象提取数据
    cells_data = []
    for cell in cells:
        cells_data.append({
            'text': cell.text,
            'row': cell.start_row_offset_idx,
            'col': cell.start_col_offset_idx,
            'row_span': cell.row_span,
            'col_span': cell.col_span
        })

    num_rows = max(c['row'] + c['row_span'] for c in cells_data)
    num_cols = max(c['col'] + c['col_span'] for c in cells_data)

    # 构建表格矩阵
    table = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells_data:
        r, c = cell['row'], cell['col']
        if r < num_rows and c < num_cols:
            table[r][c] = cell['text']

    return (table, num_rows, num_cols)


def normalize_csv_content(csv_content: str) -> set:
    """
    将 CSV 内容规范化为可比较的集合
    忽略空白和大小写差异
    """
    lines = csv_content.strip().split('\n')
    cells = set()
    for line in lines:
        # 分割并清理每个单元格
        parts = line.split(',')
        for part in parts:
            cleaned = part.strip().strip('"').lower()
            if cleaned:
                cells.add(cleaned)
    return cells


def compare_table_data(original_table: list, repaired_csv: str, num_rows: int, num_cols: int) -> tuple:
    """
    比较原始表格数据与修复后的 CSV 内容
    使用更宽松的匹配逻辑：只要主体数据匹配即可

    Returns:
        (match: bool, matched_count: int, total_count: int, detail: str)
    """
    # 解析修复后的 CSV
    repaired_lines = repaired_csv.strip().split('\n')
    repaired_table = []
    for line in repaired_lines:
        cells = []
        in_quote = False
        current = ""
        for char in line:
            if char == '"':
                in_quote = not in_quote
            elif char == ',' and not in_quote:
                cells.append(current.strip().strip('"'))
                current = ""
            else:
                current += char
        cells.append(current.strip().strip('"'))
        repaired_table.append(cells)

    # 提取原始表格中的所有非空文本值
    original_values = set()
    for row in original_table:
        for cell in row:
            val = cell.strip().lower()
            if val:
                original_values.add(val)

    # 提取修复后表格中的所有非空文本值
    repaired_values = set()
    for row in repaired_table:
        for cell in row:
            val = cell.strip().lower()
            if val:
                repaired_values.add(val)

    # 计算数值匹配（允许小误差）
    numeric_matched = 0
    numeric_total = 0
    original_numeric = set()
    repaired_numeric = set()

    for val in original_values:
        try:
            float(val)
            original_numeric.add(val)
        except ValueError:
            pass

    for val in repaired_values:
        try:
            f = float(val)
            repaired_numeric.add(val)
            # 检查是否与原始数值匹配（允许0.01误差）
            for orig_val in original_numeric:
                try:
                    if abs(f - float(orig_val)) < 0.01:
                        numeric_matched += 1
                        break
                except ValueError:
                    pass
            numeric_total += 1
        except ValueError:
            pass

    # 计算文本匹配（忽略大小写和空白）
    text_matched = 0
    for val in repaired_values:
        if val in original_values:
            text_matched += 1
        else:
            # 检查是否数值匹配（已计入）
            try:
                f = float(val)
                for orig_val in original_values:
                    try:
                        if abs(f - float(orig_val)) < 0.01:
                            text_matched += 1
                            break
                    except ValueError:
                        pass
            except ValueError:
                pass

    total = len(repaired_values)
    match_rate = (text_matched + numeric_matched) / total if total > 0 else 0

    # 主体数据匹配：80%以上匹配即通过
    detail = f"匹配率: {text_matched + numeric_matched}/{total} ({match_rate:.1%})"
    if original_values != repaired_values:
        diff = original_values - repaired_values
        if diff:
            samples = list(diff)[:3]
            detail += f", 原始独有: {samples}"

    return (match_rate >= 0.8, text_matched + numeric_matched, total, detail)


def repair_single_csv(csv_path: Path, pdf_path: Optional[str] = None,
                      force: bool = False, dry_run: bool = False,
                      bypass_compare: bool = False) -> tuple:
    """
    修复单个 CSV 文件

    Returns:
        (success: bool, method: str, csv_content: str, md_content: str)
    """
    content = csv_path.read_text(encoding="utf-8")
    csv_is_valid = is_valid_csv(content)

    # 正常CSV且非force模式：跳过
    if csv_is_valid and not force:
        return (True, "skipped", "", "", "")

    if not pdf_path:
        return (False, "", "", "", "")

    # 从文件名解析页码和表格索引
    name_parts = csv_path.stem
    parts = name_parts.split("-Table")
    if len(parts) != 2:
        return (False, "", "", "", "")

    try:
        page_no = int(parts[0])
        table_idx = int(parts[1])
    except ValueError:
        return (False, "", "", "", "")

    # 解析原始 CSV 数据（如果损坏）
    original_table = None
    if not is_valid_csv(content):
        original_table = parse_original_csv_content(content)

    # 使用 docling 从 PDF 重新提取
    result = extract_table_from_pdf(pdf_path, page_no, table_idx)
    if result:
        csv_content, md_content, png_data = result

        # 验证与原始数据匹配（仅对损坏的CSV，且未跳过对比）
        if original_table and not bypass_compare:
            orig_table, num_rows, num_cols = original_table
            match, matched, total, detail = compare_table_data(orig_table, csv_content, num_rows, num_cols)
            if not match:
                return (False, f"mismatch({detail})", "", "", "")

        return (True, "pdf_reextract", csv_content, md_content, png_data)

    return (False, "", "", "", "")


def repair_tables(force: bool = False, limit: Optional[int] = None,
                  single_file: Optional[str] = None, dry_run: bool = False,
                  inplace: bool = False, bypass_compare: bool = False):
    """扫描并修复所有损坏的 CSV 文件"""
    if not TABLES_DIR.exists():
        print(f"[ERROR] 目录不存在: {TABLES_DIR}")
        return

    paper_dirs = sorted([d for d in TABLES_DIR.iterdir() if d.is_dir()])
    print(f"找到 {len(paper_dirs)} 个论文目录")
    print(f"模式: force={force}, dry_run={dry_run}")
    print()

    # 收集所有需要处理的 CSV 文件
    all_csv_files = []
    for paper_dir in paper_dirs:
        paper_name = paper_dir.name

        # 查找对应的 PDF
        pdf_path = None
        for ext in [".pdf", ".PDF"]:
            candidate = PAPERS_DIR / f"{paper_name}{ext}"
            if candidate.exists():
                pdf_path = str(candidate)
                break

        csv_files = sorted(paper_dir.glob("*.csv"))
        # 跳过 _repaired 目录
        csv_files = [f for f in csv_files if "_repaired" not in f.parent.name]

        if single_file:
            csv_files = [f for f in csv_files if str(f) == single_file]
            if not csv_files:
                continue

        if not csv_files:
            continue

        for csv_file in csv_files:
            all_csv_files.append({
                'csv_file': csv_file,
                'pdf_path': pdf_path,
                'paper_name': paper_name
            })

    # 使用 tqdm 显示进度
    total_repaired = 0
    total_skipped = 0
    total_errors = 0
    results = []

    print(f"共 {len(all_csv_files)} 个 CSV 文件待处理\n")

    # 创建进度条
    pbar = tqdm(all_csv_files, desc="修复进度", unit="file")

    for item in pbar:
        csv_file = item['csv_file']
        pdf_path = item['pdf_path']

        content = csv_file.read_text(encoding="utf-8")
        is_valid = is_valid_csv(content)

        if is_valid and not force:
            total_skipped += 1
            pbar.set_postfix_str(f"跳过: {csv_file.name}")
            continue

        if not pdf_path:
            total_errors += 1
            pbar.set_postfix_str(f"无PDF: {csv_file.name}")
            continue

        success, method, csv_content, md_content, png_data = repair_single_csv(
            csv_file, pdf_path, force, dry_run, bypass_compare
        )

        if success and method == "skipped":
            total_skipped += 1
            pbar.set_postfix_str(f"跳过: {csv_file.name}")
        elif success:
            if dry_run:
                total_repaired += 1
            else:
                if inplace:
                    # 直接覆盖原文件
                    output_csv = csv_file
                    output_md = csv_file.parent / f"{csv_file.stem}.md"
                    output_png = csv_file.parent / f"{csv_file.stem}.png"
                else:
                    # 保存到 _repaired 目录
                    repaired_dir = csv_file.parent / REPAIRED_DIR
                    repaired_dir.mkdir(exist_ok=True)
                    output_csv = repaired_dir / csv_file.name
                    output_md = repaired_dir / f"{csv_file.stem}.md"
                    output_png = repaired_dir / f"{csv_file.stem}.png"

                output_csv.write_text(csv_content, encoding="utf-8")
                output_md.write_text(md_content, encoding="utf-8")
                if png_data:
                    output_png.write_bytes(png_data)

                total_repaired += 1
                results.append({
                    'csv': csv_file.name,
                    'paper': item['paper_name'],
                    'method': method,
                    'output': str(output_csv)
                })
            pbar.set_postfix_str(f"✓ 修复: {csv_file.name} ({method})")
        else:
            total_errors += 1
            pbar.set_postfix_str(f"✗ 失败: {csv_file.name} ({method})")

        if limit and total_repaired >= limit:
            pbar.close()
            break

    # 如果指定了 single_file 但没找到，报错
    if single_file and total_repaired == 0 and total_skipped == 0:
        print(f"[ERROR] 文件不存在: {single_file}")
        return

    print(f"\n{'='*50}")
    print(f"修复完成")
    print(f"  正常/跳过: {total_skipped}")
    print(f"  已修复: {total_repaired}")
    print(f"  失败: {total_errors}")
    if dry_run:
        print(f"  (预览模式，未实际保存)")
    print(f"{'='*50}")

    if results:
        print("\n修复的文件:")
        for r in results[:10]:
            print(f"  {r['csv']} ({r['method']}) -> {r['output']}")
        if len(results) > 10:
            print(f"  ... 还有 {len(results) - 10} 个")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="修复损坏的表格 CSV 文件（使用 docling 重新提取）")
    parser.add_argument("--force", action="store_true",
                        help="强制重建所有文件，包括已正常的")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理文件数量")
    parser.add_argument("--single", type=str, default=None,
                        help="仅处理单个 CSV 文件（完整路径）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览，不保存文件")
    parser.add_argument("--inplace", action="store_true",
                        help="直接覆盖原文件（而非保存到 _repaired 目录）")
    parser.add_argument("--bypass-compare", action="store_true",
                        help="跳过与原始CSV内容对比检查（用于内容已损坏的文件）")

    args = parser.parse_args()

    print("=== CSV 表格修复工具 (docling 版本) ===")
    print(f"表格目录: {TABLES_DIR}")
    print(f"论文目录: {PAPERS_DIR}")
    print(f"输出模式: {'覆盖原文件' if args.inplace else '_repaired 目录'}")
    print()

    repair_tables(
        force=args.force,
        limit=args.limit,
        single_file=args.single,
        dry_run=args.dry_run,
        inplace=args.inplace,
        bypass_compare=args.bypass_compare
    )
