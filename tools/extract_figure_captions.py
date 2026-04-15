"""
批量从 PDF 提取 Figure 和 Table Caption 脚本

用法:
    cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag
    source .venv/bin/activate
    python extract_figure_captions.py <pdf_dir>

示例:
    python extract_figure_captions.py /Volumes/ext/Master/papers
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# ---------------------------------------------------------------------------
# Docling 提取
# ---------------------------------------------------------------------------

def extract_from_pdf(pdf_path: str) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]]]:
    """
    从 PDF 提取所有图片和表格的 caption。

    Returns:
        (page_figures, page_tables):
            page_figures: Dict[page_no, List[figure_data]]
            page_tables:  Dict[page_no, List[table_data]]
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print("❌ docling 未安装，请运行: pip install docling")
        return {}, {}

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    export = result.document.export_to_dict()

    pictures = export.get("pictures", [])
    tables = export.get("tables", [])
    texts = export.get("texts", [])
    pages = export.get("pages", {})

    def get_page_height(page_no: int) -> float:
        return pages.get(str(page_no), {}).get("size", {}).get("height", 0)

    def extract_figure_caption(text: str) -> Tuple[Optional[str], Optional[str]]:
        """从文本中提取 'Figure X. caption' 或 'Fig X. caption' 格式"""
        match = re.match(r"^(Fig(?:ure)?\.?\s+\d+[\.:]\s*)(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            prefix = match.group(1).strip()
            suffix = match.group(2).strip()
            fig_match = re.search(r'(\d+)', prefix)
            if fig_match:
                return f"Figure {fig_match.group(1)}. {suffix}", fig_match.group(1)
        return None, None

    def extract_table_caption(text: str) -> Tuple[Optional[str], Optional[str]]:
        """从文本中提取 'Table X. caption' 格式"""
        match = re.match(r"^(Table\s+\d+[\.:]\s*)(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            prefix = match.group(1).strip()
            suffix = match.group(2).strip()
            tab_match = re.search(r'(\d+)', prefix)
            if tab_match:
                return f"Table {tab_match.group(1)}. {suffix}", tab_match.group(1)
        return None, None

    def extract_number(text: str, prefix_pattern: str) -> Optional[str]:
        match = re.search(prefix_pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    # 构建 page -> texts 索引
    page_texts: Dict[str, List[dict]] = {}
    for i, t in enumerate(texts):
        prov = t.get("prov", [])
        if not prov:
            continue
        page = prov[0].get("page_no")
        if page:
            key = str(page)
            if key not in page_texts:
                page_texts[key] = []
            page_texts[key].append({
                "idx": i,
                "text": t.get("text", ""),
                "bbox": prov[0].get("bbox", {}),
            })

    def find_caption(page_str: str, bottom_pdf: float, extractor_fn) -> Tuple[Optional[str], Optional[str]]:
        """在图片/表格上下文中找 caption：caption 可能在图上方或下方，取最近者"""
        if page_str not in page_texts:
            return None, None
        best_caption: Optional[str] = None
        best_num: Optional[str] = None
        best_dist = float("inf")
        for pt in page_texts[page_str]:
            text_bbox = pt.get("bbox")
            if not text_bbox:
                continue
            caption, fig_num = extractor_fn(pt["text"])
            if not caption:
                continue
            text_t = text_bbox.get("t", 0)
            text_b = text_bbox.get("b", 0)
            # docling y 坐标从下往上：caption 可能在图上方（text_b > bottom_pdf）或下方（text_t < bottom_pdf）
            if text_b > bottom_pdf:
                dist = text_b - bottom_pdf
            elif text_t < bottom_pdf:
                dist = bottom_pdf - text_t
            else:
                continue  # 重叠
            if dist < best_dist:
                best_dist = dist
                best_caption = caption
                best_num = fig_num
        return best_caption, best_num

    # 提取 figures（优先用 docling captions 字段，fallback 到 proximity 匹配）
    page_figures: Dict[int, List[dict]] = {}
    for i, pic in enumerate(pictures):
        page_no = pic["prov"][0]["page_no"]
        pic_bbox = pic["prov"][0].get("bbox", {})
        pic_bottom_pdf = pic_bbox.get("b", 0)
        page_str = str(page_no)

        caption_refs = pic.get("captions", [])
        if caption_refs:
            ref = caption_refs[0]
            idx = int(ref["$ref"].split("/")[-1])
            raw_caption = texts[idx].get("text", "") if idx < len(texts) else ""
            parsed, fig_num = extract_figure_caption(raw_caption)
            if not parsed:
                parsed = raw_caption.strip()
                fig_num_match = re.search(r'Fig(?:ure)?\s*(\d+)', raw_caption, re.IGNORECASE)
                if fig_num_match:
                    fig_num = fig_num_match.group(1)
        else:
            parsed = None
            fig_num = ""

        if parsed:
            page_figures.setdefault(page_no, []).append({
                "docling_index": i + 1,
                "bottom_pdf": pic_bottom_pdf,
                "caption": parsed,
                "number": fig_num,
            })
        else:
            caption, fig_num_fb = find_caption(page_str, pic_bottom_pdf, extract_figure_caption)
            page_figures.setdefault(page_no, []).append({
                "docling_index": i + 1,
                "bottom_pdf": pic_bottom_pdf,
                "caption": caption or "",
                "number": fig_num_fb or "",
            })

    # 提取 tables（优先用 docling captions 字段，fallback 到 proximity 匹配）
    page_tables: Dict[int, List[dict]] = {}
    for i, tab in enumerate(tables):
        page_no = tab["prov"][0]["page_no"]
        tab_bbox = tab["prov"][0].get("bbox", {})
        tab_bottom_pdf = tab_bbox.get("b", 0)
        page_str = str(page_no)

        # 优先用 docling 的 captions 字段
        caption_refs = tab.get("captions", [])
        if caption_refs:
            ref = caption_refs[0]
            idx = int(ref["$ref"].split("/")[-1])
            raw_caption = texts[idx].get("text", "") if idx < len(texts) else ""
            parsed, tab_num = extract_table_caption(raw_caption)
            if not parsed:
                parsed = raw_caption.strip()
                tab_num_match = re.search(r'Table\s*(\d+)', raw_caption, re.IGNORECASE)
                if tab_num_match:
                    tab_num = tab_num_match.group(1)
        else:
            parsed = None
            tab_num = ""

        if parsed:
            page_tables.setdefault(page_no, []).append({
                "docling_index": i + 1,
                "bottom_pdf": tab_bottom_pdf,
                "caption": parsed,
                "number": tab_num,
            })
        else:
            # fallback 到 proximity 匹配
            caption, tab_num_fallback = find_caption(page_str, tab_bottom_pdf, extract_table_caption)
            page_tables.setdefault(page_no, []).append({
                "docling_index": i + 1,
                "bottom_pdf": tab_bottom_pdf,
                "caption": caption or "",
                "number": tab_num_fallback or "",
            })

    return page_figures, page_tables


# ---------------------------------------------------------------------------
# 文件名解析
# ---------------------------------------------------------------------------

def parse_figure_filename(fname: str) -> Optional[dict]:
    """解析 figure 文件名，如 '14-Figure1.png' -> {page, number, filename}"""
    match = re.match(r"(\d+)-Figure(\d+)\.png", fname, re.IGNORECASE)
    if match:
        return {"page": int(match.group(1)), "number": match.group(2), "filename": fname}
    return None


def parse_table_filename(fname: str) -> Optional[dict]:
    """解析 table 文件名，如 '5-Table1.png' -> {page, number, filename}"""
    match = re.match(r"(\d+)-Table(\d+)\.png", fname, re.IGNORECASE)
    if match:
        return {"page": int(match.group(1)), "number": match.group(2), "filename": fname}
    return None


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------

def match_items(
    page_items: Dict[int, List[dict]],
    media_dir: Path,
    prefix: str,
    parse_fn,
) -> Dict[str, dict]:
    """将 docling 提取的 caption 与实际文件配对（greedy 最近邻）"""
    # 加载实际文件
    actual_files: Dict[int, List[dict]] = {}
    for fname in sorted(media_dir.glob("*.png")):
        parsed = parse_fn(fname.name)
        if not parsed:
            continue
        page = parsed["page"]
        actual_files.setdefault(page, []).append(parsed)

    result: Dict[str, dict] = {}
    all_pages = sorted(set(list(page_items.keys()) + list(actual_files.keys())))

    for page in all_pages:
        items = page_items.get(page, [])
        files = actual_files.get(page, [])

        used_files = set()
        used_items = set()

        # 按 bottom_pdf 排序，贪心匹配
        for item in sorted(items, key=lambda x: x["bottom_pdf"]):
            if not item["caption"] or item["docling_index"] in used_items:
                continue

            best_file = None
            best_dist = float("inf")
            item_num = int(item["number"]) if item["number"] else 0

            for f in sorted(files, key=lambda x: int(x["number"]) if x["number"] else 0):
                if f["filename"] in used_files:
                    continue
                f_num = int(f["number"]) if f["number"] else 0
                dist = abs(item_num - f_num)
                if dist < best_dist:
                    best_dist = dist
                    best_file = f

            if best_file:
                key = f"{page}-{prefix}{item['docling_index']}"
                result[key] = {
                    "caption": item["caption"],
                    "filename": best_file["filename"],
                    "page": page,
                    "number": item["number"],
                }
                used_files.add(best_file["filename"])
                used_items.add(item["docling_index"])

        # 无 caption 的文件也要记录
        for f in sorted(files, key=lambda x: int(x["number"]) if x["number"] else 0):
            if f["filename"] not in used_files:
                key = f"{page}-{prefix}{f['number']}"
                if key not in result:
                    result[key] = {
                        "caption": "",
                        "filename": f["filename"],
                        "page": page,
                        "number": f["number"],
                    }

    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_paper(
    pdf_path: str,
    figures_base_dir: str,
    tables_base_dir: str,
    output_dir: Path,
) -> Tuple[int, int, str]:
    """处理单篇论文，提取 figures 和 tables 的 caption"""
    paper_name = Path(pdf_path).stem
    figures_dir = Path(figures_base_dir) / paper_name
    tables_dir = Path(tables_base_dir) / paper_name

    # docling 提取
    page_figures, page_tables = extract_from_pdf(pdf_path)

    # 匹配
    matched_figures = {}
    matched_tables = {}
    if figures_dir.exists():
        matched_figures = match_items(page_figures, figures_dir, "Figure", parse_figure_filename)
    if tables_dir.exists():
        matched_tables = match_items(page_tables, tables_dir, "Table", parse_table_filename)

    # 合并
    matched = {**matched_figures, **matched_tables}

    # 输出
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{paper_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)

    total = len(matched)
    success = sum(1 for v in matched.values() if v["caption"])
    return success, total, paper_name


def main():
    parser = argparse.ArgumentParser(description="批量提取 PDF 中 figure 和 table 的 caption")
    parser.add_argument("pdf_dir", help="PDF 文件所在目录")
    parser.add_argument("--output", "-o", default="data/captions",
                        help="caption JSON 输出目录 (默认: data/figures_caption)")
    parser.add_argument("--figures-base", "-f", default="data/figures",
                        help="图片目录 (默认: data/figures)")
    parser.add_argument("--tables-base", "-t", default="data/tables",
                        help="表格目录 (默认: data/tables)")
    parser.add_argument("--dry-run", action="store_true", help="仅扫描不处理")
    parser.add_argument("--skip", default="", help="跳过的 PDF 名称（逗号分隔，部分匹配）")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output)
    figures_base = args.figures_base
    tables_base = args.tables_base
    skip_names = [s.strip() for s in args.skip.split(",") if s.strip()]

    if not pdf_dir.exists():
        print(f"❌ PDF 目录不存在: {pdf_dir}")
        return 1

    pdf_files = sorted([f for f in pdf_dir.glob("*.pdf") if not f.name.startswith("._")])
    if not pdf_files:
        print(f"❌ 目录中没有 PDF 文件: {pdf_dir}")
        return 1

    print(f"📂 找到 {len(pdf_files)} 个 PDF 文件")
    print(f"📁 caption 输出到: {output_dir}")
    print(f"🖼️  figures: {figures_base}")
    print(f"📊 tables: {tables_base}")
    if skip_names:
        print(f"⏭️  跳过: {skip_names}")
    print()

    if args.dry_run:
        for pdf in pdf_files:
            skip = any(s in pdf.name for s in skip_names)
            mark = "⏭️ [SKIP]" if skip else "  [TODO]"
            print(f"  {mark} {pdf.name}")
        return 0

    total_success = 0
    total_count = 0
    errors: List[str] = []
    skipped: List[str] = []

    for pdf in pdf_files:
        if any(s in pdf.name for s in skip_names):
            skipped.append(pdf.name)
            continue

        print(f"🔄 处理: {pdf.name}")
        try:
            success, count, name = process_paper(
                str(pdf), figures_base, tables_base, output_dir
            )
            total_success += success
            total_count += count
            fig_count = sum(1 for k, v in {**{}}.items() if "Figure" in k)
            print(f"  ✅ {name}: {success}/{count} captions")
        except Exception as e:
            import traceback
            errors.append(f"{pdf.name}: {e}")
            print(f"  ❌ {pdf.name}: {e}")
            traceback.print_exc()

    print()
    print(f"📊 总计: {total_success}/{total_count} captions ({len(skipped)} 跳过, {len(errors)} 失败)")
    if errors:
        print(f"\n❌ 失败列表:")
        for e in errors:
            print(f"  - {e}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
