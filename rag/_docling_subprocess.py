#!/usr/bin/env python3
"""
Docling PDF 提取子进程脚本
由 DoclingExtractor._extract_with_docling 调用，运行在独立进程中以隔离环境。

用法:
    python _docling_subprocess.py <pdf_path> <paper_id> <figures_dir> <tables_dir> <models_dir> [captions_dir]
"""
import sys
import os
import io
import json
import re
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


def _extract_logical_num(caption_text: str, prefix: str) -> str | None:
    """从真实图注/表注中提取逻辑编号。

    例: "Figure 3: System Overview" + prefix="Figure" → "3"
        "Table 2: Results"           + prefix="Table"  → "2"

    返回 None 表示无法提取。
    """
    if not caption_text or not caption_text.strip():
        return None
    # 匹配 "Figure 3", "Fig. 3", "Fig 3", "Figure 3:", "Figure 3a", "图 1", "表 2" 等
    _zh = {"Figure": "图", "Table": "表"}
    zh_char = _zh.get(prefix, "")
    patterns = [
        rf'{prefix}\s*([A-Za-z]?\d+[A-Za-z]?)',
        rf'{prefix[:3]}\.?\s*([A-Za-z]?\d+[A-Za-z]?)',  # "Fig. 3" / "Fig 3" / "Tab. 1"
    ]
    if zh_char:
        patterns.append(rf'{zh_char}\s*([A-Za-z]?\d+[A-Za-z]?)')
    for pat in patterns:
        m = re.search(pat, caption_text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _cell_to_text(cell) -> str:
    """从 docling 表格单元格提取纯文本。"""
    if hasattr(cell, '_get_text'):
        return cell._get_text(doc=None) or ""
    if isinstance(cell, list):
        return str(cell)
    return str(cell) if cell is not None else ""


def _table_data_to_csv(data):
    """将表格数据转换为 CSV 格式字符串"""
    if hasattr(data, 'export_to_dataframe'):
        try:
            df = data.export_to_dataframe(doc=None)
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=True)
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
            cell_text = _cell_to_text(cell)
            cell_text = cell_text.replace("\n", " ").replace("\r", "").replace('"', '""')
            cells.append(cell_text)
        escaped_cells = ['"' + c + '"' if ',' in c or '"' in c or '\n' in c else c for c in cells]
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
                cells = [str(v).replace("|", "\\|").replace("\n", " ") for v in row]
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
            cell_str = _cell_to_text(cell)
            cell_str = cell_str.replace("|", "\\|").replace("\n", " ")
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

            # bbox uses PDF coordinate system (origin at bottom-left).
            # PyMuPDF uses screen coordinates (origin at top-left).
            # Convert: screen_y = page_height - pdf_y
            x0 = bbox.l
            y0 = page_height - bbox.b
            x1 = bbox.r
            y1 = page_height - bbox.t

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
        print(json.dumps({"error": "Usage: python _docling_subprocess.py <pdf_path> <paper_id> <figures_dir> <tables_dir> [models_dir] [captions_dir]"}), file=sys.stdout)
        sys.exit(1)

    pdf_path_arg = sys.argv[1]
    paper_id_arg = sys.argv[2]
    figures_dir_arg = Path(sys.argv[3])
    tables_dir_arg = Path(sys.argv[4])
    captions_dir_arg = Path(sys.argv[6]) if len(sys.argv) > 6 else None
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
            if element.image is None:
                continue

            # 使用 docling 原生 caption 解析获取真实图注文本
            try:
                real_caption = element.caption_text(result.document) or ""
            except Exception:
                real_caption = ""

            # 从真实图注提取逻辑编号，失败则标记为 unknown
            logical_num = _extract_logical_num(real_caption, "Figure") if real_caption else ""
            if logical_num:
                # 使用论文真实编号作为文件名
                label = f"Figure{logical_num}"
                # 处理同编号冲突：同一逻辑编号 + 同一页码 → 加后缀
                # （通常不会出现，但处理 figure 跨页或补充材料的情况）
                global_key = f"{page_no}-{label}"
                collision = figure_counters.get(global_key, 0) + 1
                figure_counters[global_key] = collision
                if collision > 1:
                    label = f"Figure{logical_num}_v{collision}"
            else:
                # 无有效 caption：全局 unknown 计数器
                unknown_idx = figure_counters.get("_unknown", 0) + 1
                figure_counters["_unknown"] = unknown_idx
                label = f"unknown_{unknown_idx}"
                logical_num = ""

            caption = real_caption or label

            filename = f"{page_no}-{label}.png"
            save_path = figures_dir_arg / filename
            pil_image = element.image.pil_image
            assert pil_image is not None
            pil_image.save(save_path, format="PNG")
            images.append({
                "page_number": page_no,
                "image_index": 0,  # 保留字段；有效图表以 logical_num+saved_path 标识，unknown 以 saved_path 标识
                "logical_num": logical_num,
                "bbox": [0, 0, 0, 0],
                "caption": caption,
                "saved_path": str(save_path),
            })
        elif isinstance(element, TableItem):
            page_no = element.prov[0].page_no

            # 使用 docling 原生 caption 解析获取真实表注文本
            try:
                real_caption = element.caption_text(result.document) or ""
            except Exception:
                real_caption = ""

            # 从真实表注提取逻辑编号，失败则标记为 unknown
            logical_num = _extract_logical_num(real_caption, "Table") if real_caption else ""
            if logical_num:
                label = f"Table{logical_num}"
                global_key = f"{page_no}-{label}"
                collision = table_counters.get(global_key, 0) + 1
                table_counters[global_key] = collision
                if collision > 1:
                    label = f"Table{logical_num}_v{collision}"
            else:
                # 无有效 caption：全局 unknown 计数器
                unknown_idx = table_counters.get("_unknown", 0) + 1
                table_counters["_unknown"] = unknown_idx
                label = f"unknown_{unknown_idx}"
                logical_num = ""

            caption = real_caption or label

            table_csv = _table_data_to_csv(element)
            table_markdown = _csv_to_markdown(element)
            csv_filename = f"{page_no}-{label}.csv"
            md_filename = f"{page_no}-{label}.md"
            png_filename = f"{page_no}-{label}.png"
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
                "table_index": 0,
                "logical_num": logical_num,
                "bbox": [0, 0, 0, 0],
                "csv": table_csv,
                "markdown": table_markdown,
                "caption": caption,
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

    # Save captions JSON for downstream consumers (idea engine, etc.)
    if captions_dir_arg is not None:
        try:
            captions_dir_arg.mkdir(parents=True, exist_ok=True)
            captions: dict = {}
            for img in images:
                filename = Path(img["saved_path"]).name
                key = filename.rsplit(".", 1)[0]  # e.g. "1-Figure3"
                captions[key] = {
                    "caption": img["caption"],
                    "filename": filename,
                    "page": img["page_number"],
                    "number": img.get("logical_num") or "",
                }
            for tbl in tables:
                if tbl.get("saved_png_path"):
                    filename = Path(tbl["saved_png_path"]).name
                    key = filename.rsplit(".", 1)[0]
                    captions[key] = {
                        "caption": tbl["caption"],
                        "filename": filename,
                        "page": tbl["page_number"],
                        "number": tbl.get("logical_num") or "",
                        "type": "table",
                    }
            captions_path = captions_dir_arg / f"{paper_id_arg}.json"
            with open(captions_path, "w", encoding="utf-8") as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Non-fatal: captions save failure should not block the pipeline
            print(f"Warning: failed to save captions JSON: {e}", file=sys.stderr)

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
