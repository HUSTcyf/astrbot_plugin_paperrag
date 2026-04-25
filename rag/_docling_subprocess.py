#!/usr/bin/env python3
"""
Docling PDF 提取子进程脚本
由 DoclingExtractor._extract_with_docling 调用，运行在独立进程中以隔离环境。

用法:
    python _docling_subprocess.py <pdf_path> <paper_id> <figures_dir> <tables_dir> <models_dir>
"""
import sys
import os
import io
import json
import traceback
from pathlib import Path

# 设置模型路径（插件目录下）
if len(sys.argv) > 5:
    MODELS_DIR = Path(sys.argv[5])
else:
    MODELS_DIR = Path(__file__).parent.parent / "models"
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import PictureItem, TableItem, FormulaItem
import fitz  # PyMuPDF


def _table_data_to_csv(data):
    """将表格数据转换为 CSV 格式字符串"""
    if hasattr(data, 'export_to_dataframe'):
        try:
            df = data.export_to_dataframe(doc=None)
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=True, encoding='utf-8')
            return buf.getvalue()
        except Exception:
            pass

    if hasattr(data, 'grid'):
        rows = data.grid
    else:
        rows = data

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
        escaped_cells = ['"' + c + '"' if ',' in c or '"' in c else c for c in cells]
        line = ",".join(escaped_cells)
        lines.append(line)
    return "\n".join(lines)


def _csv_to_markdown(data):
    """将表格数据转换为 Markdown 表格格式"""
    if hasattr(data, 'export_to_markdown'):
        try:
            return data.export_to_markdown(doc=None)
        except Exception:
            pass

    if hasattr(data, 'export_to_dataframe'):
        try:
            df = data.export_to_dataframe(doc=None)
            cols = list(df.columns)
            lines = []
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, row in df.iterrows():
                cells = [str(v).replace("|", chr(124)).replace(chr(10), " ") for v in row]
                lines.append("| " + " | ".join(cells) + " |")
            return "\n".join(lines)
        except Exception:
            pass

    if hasattr(data, 'grid'):
        rows = data.grid
    else:
        rows = data

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
            cell_str = cell_str.replace("|", chr(124)).replace(chr(10), " ")
            cells.append(cell_str)
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            separator = "| " + " | ".join(["---"] * len(cells)) + " |"
            lines.append(separator)
    return "\n".join(lines)


def _table_to_png_bytes(table_item, pdf_path: str):
    """优先使用 docling 自带图片；没有时按 bbox 从 PDF 裁切表格 PNG。"""
    try:
        # 1) 先用 docling 自带的表格图片
        if getattr(table_item, "image", None) is not None:
            pil_image = table_item.image.pil_image
            if pil_image:
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                return buf.getvalue()

        # 2) 再按 bbox 回裁 PDF
        if not getattr(table_item, "prov", None):
            return None
        prov = table_item.prov[0]
        bbox = getattr(prov, "bbox", None)
        page_no = getattr(prov, "page_no", None)
        if bbox is None or page_no is None:
            return None

        doc = fitz.open(pdf_path)
        try:
            pdf_page = doc[page_no - 1]
            page_height = pdf_page.rect.height

            left = bbox.l
            bottom = bbox.t
            right = bbox.r
            top = bbox.b

            x0 = left
            y0 = page_height - top
            x1 = right
            y1 = page_height - bottom

            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            clip_rect = fitz.Rect(x0, y0, x1, y1)
            if clip_rect.is_empty or clip_rect.width <= 0 or clip_rect.height <= 0:
                return None

            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip_rect, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None


def main():
    if len(sys.argv) < 5:
        print(json.dumps({"error": "Usage: python _docling_subprocess.py <pdf_path> <paper_id> <figures_dir> <tables_dir>"}), file=sys.stdout)
        sys.exit(1)

    pdf_path_arg = sys.argv[1]
    paper_id_arg = sys.argv[2]
    figures_dir_arg = Path(sys.argv[3])
    tables_dir_arg = Path(sys.argv[4])
    figures_dir_arg.mkdir(parents=True, exist_ok=True)
    tables_dir_arg.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions(
        generate_picture_images=True,
        generate_page_images=False,
        do_table_structure=True,
        do_ocr=True,
        do_formula_enrichment=False,
        do_code_enrichment=False,  # 禁用代码 enrichment，避免复杂 PDF 触发内部 SyntaxError
        images_scale=2.0,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    try:
        result = converter.convert(pdf_path_arg)
    except Exception as e:
        traceback.print_exc()
        print(json.dumps({
            "images": [],
            "tables": [],
            "formulas": [],
            "error": str(e),
        }), file=sys.stdout)
        sys.exit(1)

    images = []
    tables = []
    formulas = []
    text_parts = []
    figure_counters = {}
    table_counters = {}
    formula_counters = {}

    current_page = 0

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            page_no = element.prov[0].page_no
            figure_idx = figure_counters.get(page_no, 0) + 1
            figure_counters[page_no] = figure_idx
            if element.image is None:
                continue
            pil_image = element.image.pil_image
            assert pil_image is not None
            filename = f"{page_no}-Figure{figure_idx}.png"
            save_path = figures_dir_arg / filename
            pil_image.save(save_path, format="PNG")
            images.append({
                "page_number": page_no,
                "image_index": figure_idx,
                "bbox": [0, 0, 0, 0],
                "caption": f"Figure {figure_idx}",
                "saved_path": str(save_path),
            })
        elif isinstance(element, TableItem):
            page_no = element.prov[0].page_no
            table_idx = table_counters.get(page_no, 0) + 1
            table_counters[page_no] = table_idx
            table_csv = _table_data_to_csv(element)
            table_markdown = _csv_to_markdown(element)
            csv_filename = f"{page_no}-Table{table_idx}.csv"
            md_filename = f"{page_no}-Table{table_idx}.md"
            png_filename = f"{page_no}-Table{table_idx}.png"
            csv_path = tables_dir_arg / csv_filename
            md_path = tables_dir_arg / md_filename
            png_path = tables_dir_arg / png_filename
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(table_csv)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(table_markdown)
            saved_png_path = None
            png_data = _table_to_png_bytes(element, pdf_path_arg)
            if png_data:
                png_path.write_bytes(png_data)
                saved_png_path = str(png_path)
            tables.append({
                "page_number": page_no,
                "table_index": table_idx,
                "bbox": [0, 0, 0, 0],
                "csv": table_csv,
                "markdown": table_markdown,
                "caption": f"Table {table_idx}",
                "saved_csv_path": str(csv_path),
                "saved_md_path": str(md_path),
                "saved_png_path": saved_png_path,
            })
        elif isinstance(element, FormulaItem):
            page_no = element.prov[0].page_no if element.prov else 1
            formula_idx = formula_counters.get(page_no, 0) + 1
            formula_counters[page_no] = formula_idx
            latex_text = element.text or ""
            formulas.append({
                "page_number": page_no,
                "formula_index": formula_idx,
                "text": latex_text,
                "bbox": [0, 0, 0, 0],
                "type": "display",
            })
        else:
            # 文本项（段落、标题、列表等）— 排除表格
            if hasattr(element, 'text') and element.text and element.text.strip():
                if hasattr(element, 'prov') and element.prov:
                    page_no = element.prov[0].page_no
                    if page_no != current_page:
                        text_parts.append(f"\n[Page {page_no}]")
                        current_page = page_no
                text_parts.append(element.text)

    # 不再使用 result.document.texts（可能包含表格文本）
    # 文本已从 iterate_items 循环中收集（排除了 TableItem）

    result_json = json.dumps({
        "file_name": Path(pdf_path_arg).name,
        "images": images,
        "tables": tables,
        "formulas": formulas,
        "text": "\n".join(text_parts),
    })

    print(result_json)


if __name__ == "__main__":
    main()
