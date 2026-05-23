"""
Markdown 处理与图片工具方法
"""

import base64
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote
from pathlib import Path
from astrbot.api import logger

from .utils import strip_markdown_style, parse_inline_styles
from .ideas import IdeaEngineIdeas


class IdeaEngineMarkdown(IdeaEngineIdeas):
    """Markdown 处理与图片工具方法。继承链：... → IdeaEngineIdeas → IdeaEngineMarkdown"""

    def _markdown_to_feishu_blocks(self, markdown_text: str) -> List[Dict]:
        """将 Markdown 文本转换为飞书块格式

        对于含行内图片的段落，将图片前后的文本合并为一个文本块，
        图片紧跟在文本块后面（与飞书文档阅读体验一致）。
        """
        blocks = []
        lines = markdown_text.split("\n")

        for line in lines:
            line = line.rstrip()

            if line.startswith("# ") and not line.startswith("## "):
                content = strip_markdown_style(line[2:].strip())
                blocks.append({
                    "blockType": "heading",
                    "options": {"heading": {"level": 1, "content": content}}
                })
            elif line.startswith("## ") and not line.startswith("### "):
                content = strip_markdown_style(line[3:].strip())
                blocks.append({
                    "blockType": "heading",
                    "options": {"heading": {"level": 2, "content": content}}
                })
            elif line.startswith("### "):
                content = strip_markdown_style(line[4:].strip())
                blocks.append({
                    "blockType": "heading",
                    "options": {"heading": {"level": 3, "content": content}}
                })
            elif line.startswith("---"):
                blocks.append({
                    "blockType": "text",
                    "options": {"text": {"textStyles": [{"text": "─────────────────────────────────", "style": {}}]}}
                })
            elif line.startswith("- ") or line.startswith("* "):
                raw_content = line[2:].strip()
                if raw_content:
                    blocks.append({
                        "blockType": "list",
                        "options": {"list": {"content": raw_content, "isOrdered": False}},
                        "_textStyles": parse_inline_styles(raw_content)
                    })
            elif re.match(r'^\d+[\.\)]\s', line):
                match = re.match(r'^(\d+[\.\)])\s+(.*)$', line)
                if match:
                    raw_content = match.group(2).strip()
                    if raw_content:
                        blocks.append({
                            "blockType": "list",
                            "options": {"list": {"content": raw_content, "isOrdered": True}},
                            "_textStyles": parse_inline_styles(raw_content)
                        })
            elif line.strip() == "":
                pass
            else:
                text_content = line.strip()
                if text_content:
                    segments = self._extract_inline_images(text_content)
                    text_parts = []
                    image_blocks = []
                    for seg in segments:
                        if seg["type"] == "text":
                            text_parts.append(seg["content"])
                        elif seg["type"] == "image":
                            img_block = self._make_image_block(seg["path"], seg["caption"])
                            if img_block is not None:
                                image_blocks.append(img_block)
                    merged_text = "".join(text_parts)
                    if merged_text.strip():
                        blocks.append({
                            "blockType": "text",
                            "options": {"text": {"textStyles": parse_inline_styles(merged_text)}}
                        })
                    blocks.extend(image_blocks)

        return blocks

    def _find_methodology_end_index(self, blocks: List[Dict]) -> int:
        """找到方法论章节的结束位置（下一个同级/更高级标题之前），返回插入索引。
        如果找不到方法论章节，返回 blocks 末尾。"""
        method_start = -1
        method_level = 3
        for i, b in enumerate(blocks):
            if b.get("blockType") == "heading":
                opts = b.get("options", {}).get("heading", {})
                content = opts.get("content", "")
                level = opts.get("level", 0)
                if level <= 3 and re.search(r'方法|method', content, re.IGNORECASE):
                    method_start = i
                    method_level = level
                    break

        if method_start < 0:
            return len(blocks)

        for i in range(method_start + 1, len(blocks)):
            b = blocks[i]
            if b.get("blockType") == "heading":
                opts = b.get("options", {}).get("heading", {})
                level = opts.get("level", 0)
                if level <= method_level:
                    return i
        return len(blocks)

    def _extract_path_from_paren(self, line: str) -> str | None:
        """从一行中提取 (path) 格式的路径，支持路径中含括号如 2502.12138v4(nopo)，
        支持 ASCII ) 和全角 ） 闭合括号，扩展名大小写不敏感。"""
        for ext in ['png', 'jpg', 'jpeg', 'webp', 'gif', 'PNG', 'JPG', 'JPEG', 'WEBP', 'GIF']:
            for end in [f'.{ext})', f'.{ext}）']:
                idx = line.rfind(end)
                if idx > 0:
                    paren_count = 1
                    j = idx - 1
                    while j >= 0 and paren_count > 0:
                        ch = line[j]
                        if ch == ')' or ch == '）':
                            paren_count += 1
                        elif ch == '(' or ch == '（':
                            paren_count -= 1
                        j -= 1
                    if paren_count == 0 and j >= 0:
                        path = line[j+2:idx+len(end)-1]
                        if path.startswith('/'):
                            return path
        return None

    def _extract_markdown_image_from_text(self, text: str) -> list[tuple[str, str]]:
        """
        使用平衡括号计数从文本中提取所有 markdown 图片格式的 (path, caption)。
        支持路径中包含括号如 2502.12138v4(nopo)，也支持中文括号 ） 结尾。

        Returns:
            [(path, caption), ...]
        """
        results = []
        _EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'PNG', 'JPG', 'JPEG', 'WEBP', 'GIF']

        logger.debug(f"[_extract_markdown_image_from_text] 输入文本长度: {len(text)}")
        if len(text) < 500:
            logger.debug(f"[_extract_markdown_image_from_text] 文本内容: {text}")

        for ext in _EXTENSIONS:
            end_markers = [f'.{ext})', f'.{ext}）']
            start = 0
            while True:
                idx = -1
                found_marker = ''
                for em in end_markers:
                    t = text.find(em, start)
                    if t >= 0 and (idx < 0 or t < idx):
                        idx = t
                        found_marker = em
                if idx < 0:
                    break

                paren_count = 1
                j = idx - 1
                found_paren = False
                while j >= 0 and paren_count > 0:
                    if text[j] in (')', '）'):
                        paren_count += 1
                    elif text[j] in ('(', '（'):
                        paren_count -= 1
                        if paren_count == 0:
                            found_paren = True
                            break
                    j -= 1

                if found_paren and j >= 2 and text[j-1] == ']':
                    # Must be ![...](path), not [...](url).  rfind alone is not
                    # enough: an earlier ![ could be matched for a later link,
                    # so we also verify that the closing bracket paired with
                    # this ![ is exactly the ] at j-1.
                    caption_start = text.rfind('![', 0, j-1)
                    closing = text.find(']', caption_start + 2) if caption_start >= 0 else -1
                    if caption_start >= 0 and closing == j - 1:
                        path = text[j+1:idx+len(ext)+2]
                        path = path[:-1]
                        path = unquote(path)
                        caption = text[caption_start+2:text.find(']', caption_start)]
                        if path.startswith('/') and len(path) > 5:
                            results.append((path, caption))
                            start = idx + len(found_marker)
                            continue
                start = idx + 1

        logger.info(f"[_extract_markdown_image_from_text] 共提取到 {len(results)} 张图片: {results}")
        return results

    def _extract_path_from_line(self, line: str) -> tuple[str | None, str | None]:
        """
        从一行中提取路径和说明文字。
        支持格式：
        1. (path) 格式
        2. 路径：/path 或 路径 /path 格式
        3. ![caption](path) 格式
        返回：(路径, 说明文字) 或 (None, None)
        """
        _EXT = r'(?:png|jpg|jpeg|webp|gif|PNG|JPG|JPEG|WEBP|GIF)'

        path = self._extract_path_from_paren(line)
        if path:
            return path, line

        path_match = re.search(r'路径[：:\s]+([/][^\s]+\.(?:' + _EXT + r')[)）]?', line)
        if path_match:
            return path_match.group(1).rstrip(')）'), line

        md_images = self._extract_markdown_image_from_text(line)
        if md_images:
            path, caption = md_images[0]
            return path, caption

        return None, None

    def _normalize_figure_references(self, markdown_text: str) -> str:
        """
        前处理：
        1. 将论文图表章节中的裸路径转换为标准 markdown 图片语法
        2. 正文图片保持不动（已由 _append_figure_section 嵌入正确位置）
        3. 如果没有论文图表章节，直接返回
        """
        logger.info(f"[IdeaEngine] normalize 输入长度: {len(markdown_text)}")
        markdown_text = unquote(markdown_text)

        logger.debug(f"[IdeaEngine] normalize 输入: {markdown_text}")

        figure_match = re.search(r'##.*论文图表', markdown_text)

        if figure_match:
            before = markdown_text[:figure_match.start()]
            section = markdown_text[figure_match.start():]

            logger.info(f"[IdeaEngine] ✅ 找到论文图表章节: 位置={figure_match.start()}, 正文长度={len(before)}, 章节长度={len(section)}")
            logger.debug(f"[IdeaEngine] 论文图表章节: {section}")

            # Do NOT remove body images — _append_figure_section already placed them correctly.
            # Only normalize the figure section: convert bare paths to markdown images.
            fig_images = self._extract_markdown_image_from_text(section)
            logger.info(f"[IdeaEngine] 论文图表章节中找到 {len(fig_images)} 张图片: {fig_images}")

            lines = section.split('\n')
            result = []
            found_images = []
            pending_path = None

            for line in lines:
                stripped = line.strip()

                if stripped.startswith('/') and re.search(r'\.(?:png|jpg|jpeg|webp|gif|PNG|JPG|JPEG|WEBP|GIF)$', stripped):
                    pending_path = stripped
                    continue

                if pending_path is not None:
                    fig_cap_match = re.match(r'^图\s*\d+\s*(.+)$', stripped)
                    if fig_cap_match:
                        caption = fig_cap_match.group(1).strip()
                        result.append(f'![{caption}]({pending_path})')
                        found_images.append((caption, pending_path))
                        logger.info(f"[IdeaEngine] 章节转换图片: caption='{caption}', path='{pending_path}'")
                        pending_path = None
                        continue
                    result.append(pending_path)
                    pending_path = None

                result.append(line)

            logger.info(f"[IdeaEngine] normalize 完成: 章节转换图片={len(found_images)}")
            logger.debug(f"[IdeaEngine] normalize 输出末尾200字符: {(before + chr(10).join(result))[-200:]}")
            return before + '\n'.join(result)

        else:
            logger.info("[IdeaEngine] 未找到论文图表章节，图片已由 _append_figure_section 嵌入，跳过")
            return markdown_text

    def _append_figure_section(self, text: str, knowledge: Optional[Dict[str, Any]] = None) -> str:
        """将论文图表嵌入相关工作章节（插入到该章节末尾）。"""
        if not knowledge:
            return text

        local_results = knowledge.get("local_results", [])
        if not local_results:
            return text

        caption_cache: Dict[str, Dict[str, str]] = {}
        figure_entries: List[tuple[str, str, str]] = []

        for r in local_results:
            metadata = r.get('metadata', {})
            paper = r.get('paper', 'Unknown')

            img_path = metadata.get('image_path', '')
            img_caption = metadata.get('image_caption', '')
            if img_path and os.path.exists(img_path):
                img_filename = Path(img_path).name
                paper_folder = Path(img_path).parent.name
                if paper_folder not in caption_cache:
                    caption_cache[paper_folder] = self._load_figure_captions(img_path)
                fname_to_caption = caption_cache[paper_folder]
                real_caption = fname_to_caption.get(img_filename) or img_caption or img_filename
                figure_entries.append((real_caption, img_path, "fig"))

            table_png = metadata.get('table_png_path', '')
            if table_png and os.path.exists(table_png):
                table_caption = metadata.get('table_caption', '') or Path(table_png).stem
                figure_entries.append((table_caption, table_png, "table"))

        if not figure_entries:
            return text

        fig_md_parts = []
        fig_idx = 0
        tbl_idx = 0
        for caption, path, label_type in figure_entries:
            if label_type == "table":
                tbl_idx += 1
                fig_md_parts.append(f"![表 {tbl_idx}：{caption}]({path})")
            else:
                fig_idx += 1
                fig_md_parts.append(f"![图 {fig_idx}：{caption}]({path})")
        fig_md = "\n\n".join(fig_md_parts) + "\n"

        # Insert after 相关工作 / Related Work section
        related_match = re.search(
            r'(##\s*(?:相关工作|Related\s*Work|相关工作与进展|背景与相关工作).*?)(?=##\s|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if related_match:
            insert_pos = related_match.end()
            return text[:insert_pos] + "\n" + fig_md + text[insert_pos:]
        # Fallback: find any heading containing 实验/方法/创新 and insert before it
        fallback = re.search(r'(?=##\s*(?:实验|方法|创新|Method|Experiment))', text)
        if fallback:
            return text[:fallback.start()] + "\n" + fig_md + text[fallback.start():]
        # Last resort: append
        return text + "\n" + fig_md

    def _replace_placeholder_paths_by_caption(self, text: str, local_results: List[Dict]) -> str:
        """
        用 caption 匹配将草稿中的占位符图片路径替换为真实路径。
        1. 从 local_results 构建真实图片路径 + caption 列表
        2. 提取草稿论文图表章节中的 placeholder 图片
        3. 用 caption 关键词匹配找到真实路径并替换
        """
        fig_match = re.search(r'(##\s*论文图表.*?)(?=##\s|\Z)', text, re.DOTALL)
        if not fig_match:
            return text

        caption_cache: Dict[str, Dict[str, str]] = {}
        real_images: List[tuple[str, str]] = []
        for r in local_results:
            img_path = r.get('metadata', {}).get('image_path', '')
            if not img_path:
                continue
            img_filename = Path(img_path).name
            paper_folder = Path(img_path).parent.name
            if paper_folder not in caption_cache:
                caption_cache[paper_folder] = self._load_figure_captions(img_path)
            fname_to_caption = caption_cache[paper_folder]
            real_caption = fname_to_caption.get(img_filename, img_filename)
            real_images.append((img_path, real_caption))

        if not real_images:
            return text

        placeholder_pattern = re.findall(r'(!\[(.*?)\]\()(/[^\)]+\.(?:png|jpg|jpeg|webp|gif))\)', fig_match.group(1))
        if not placeholder_pattern:
            return text

        result_text = text
        for full_match, placeholder_caption, placeholder_path in placeholder_pattern:
            if placeholder_path.startswith('/Users'):
                continue
            best_match_path = None
            best_score = 0
            for real_path, real_caption in real_images:
                if not os.path.exists(real_path):
                    continue
                score = sum(1 for w in placeholder_caption if w in real_caption)
                if score > best_score:
                    best_score = score
                    best_match_path = real_path
            if best_match_path and best_score > 0:
                real_md = f'![{placeholder_caption}]({best_match_path})'
                result_text = result_text.replace(full_match, real_md, 1)
                logger.info(f"[IdeaEngine] 路径替换: {placeholder_caption} -> {best_match_path}")
            else:
                logger.warning(f"[IdeaEngine] 未找到匹配的真实路径 for: {placeholder_caption}")

        return result_text

    def _convert_paren_paths_to_markdown(self, text: str) -> str:
        """
        将裸括号路径 (/abs/path/img.ext) 转为标准 markdown 图片格式 ![image](path)。
        使用平衡括号计数，支持路径中包含括号如 2502.12138v4(nopo)。
        """
        logger.debug(f"[_convert_paren_paths_to_markdown] 输入长度: {len(text)}")

        _EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'PNG', 'JPG', 'JPEG', 'WEBP', 'GIF']

        result = []
        i = 0
        converted_count = 0
        while i < len(text):
            if text[i] == '(' and i + 1 < len(text) and text[i + 1] == '/':
                found_ext = None
                for ext in _EXTENSIONS:
                    end_marker = f'.{ext})'
                    pos = text.find(end_marker, i + 1)
                    if pos >= 0:
                        if found_ext is None or pos < found_ext[0]:
                            found_ext = (pos, len(ext), ext)

                if found_ext:
                    ext_pos, ext_len, ext = found_ext
                    start = i + 1
                    inner = text[start:ext_pos + ext_len + 1]
                    paren_count = 1
                    for ch in inner:
                        if ch == '(':
                            paren_count += 1
                        elif ch == ')':
                            paren_count -= 1

                    if paren_count == 1:
                        path = text[start:ext_pos + ext_len + 1]
                        path = path[:-1]
                        if i > 0 and text[i - 1] == ']':
                            result.append(text[i])
                            i += 1
                            continue
                        result.append(f'![image]({path})')
                        converted_count += 1
                        logger.info(f"[_convert_paren_paths_to_markdown] ✅ 转换裸路径为图片: {path}")
                        i = ext_pos + ext_len + 1
                        continue

            result.append(text[i])
            i += 1

        logger.info(f"[_convert_paren_paths_to_markdown] 完成: 转换了 {converted_count} 个裸路径为图片格式")
        return ''.join(result)

    def _extract_inline_images(self, text: str) -> List[Dict[str, str]]:
        """
        从文本中提取行内图片引用，返回分段列表。
        支持三种格式：
          1. 标准 markdown 图片: ![caption](path)
          2. 图片引用格式: [图片: caption](path) 或 [图片:caption](path)
          3. 中文括号格式: （详见：/path/to/file.png）
          4. 裸路径格式（自动转换）: caption (/abs/path/to/file.png)
        Returns:
            [{"type": "text"|"image", "content"|"path"|"caption": str}, ...]
        """
        logger.debug(f"[_extract_inline_images] 输入长度: {len(text)}")
        if len(text) < 300:
            logger.debug(f"[_extract_inline_images] 输入文本: {text}")

        text_after_preprocess = self._convert_paren_paths_to_markdown(text)
        logger.debug(f"[_extract_inline_images] 预处理后文本长度: {len(text_after_preprocess)}")
        if text_after_preprocess != text:
            logger.info(f"[_extract_inline_images] ⚠️ 预处理有变化，长度变化: {len(text)} -> {len(text_after_preprocess)}")
            if len(text_after_preprocess) < 500:
                logger.debug(f"[_extract_inline_images] 预处理后文本: {text_after_preprocess}")
        text = text_after_preprocess

        _EXT = r'png|jpg|jpeg|webp|gif'
        text = re.sub(
            r'(图\s*\d+[：:]\s*)(.+?)\s*\[(.+?\.(?:' + _EXT + r'))\]',
            r'![\1\2](\3)',
            text
        )

        text = re.sub(
            r'(本地图-\d+[：:]\s*)(.+?)\s*\[(.+?\.(?:' + _EXT + r'))\]',
            r'![\1\2](\3)',
            text
        )

        _IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.gif')

        segments = []
        stripped = text.strip()
        if stripped.startswith('![') and any(ext in stripped.lower() for ext in _IMG_EXTS):
            last_paren = stripped.rfind(')')
            open_bracket = stripped.rfind('](', 0, last_paren)
            if open_bracket > 0:
                caption = stripped[2:open_bracket]
                path = stripped[open_bracket+2:last_paren]
                # Try raw path first (may contain URL-encoded chars on disk),
                # then fall back to decoded path.
                if os.path.exists(path):
                    pass
                else:
                    decoded = unquote(path)
                    if decoded != path and os.path.exists(decoded):
                        path = decoded
                    else:
                        logger.warning(f"[_extract_inline_images] 图片不存在: {path}")
                        path = ""
                if path:
                    segments.append({"type": "image", "path": path, "caption": caption})
        if not segments:
            segments.append({"type": "text", "content": text})
        return segments

    def _ensure_png(self, img_path: str) -> str:
        """webp/其他非PNG格式转为PNG，返回可用图片路径"""
        if not img_path:
            return img_path
        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as pil_img:
                if pil_img.format == 'PNG' and pil_img.mode in ('RGB', 'RGBA'):
                    return img_path
                png_path = img_path.rsplit('.', 1)[0] + '_converted.png'
                pil_img.convert('RGBA' if pil_img.mode == 'RGBA' else 'RGB').save(png_path, 'PNG')
                logger.info(f"[IdeaEngine] 图片已转为PNG: {png_path}")
                return png_path
        except Exception as e:
            logger.warning(f"[IdeaEngine] 图片格式转换失败: {e}")
        return img_path

    def _make_image_block(self, image_path: str, caption: str = "") -> Optional[Dict]:
        """根据本地图片路径构造飞书图片块。文件不存在或读取失败时返回 None。"""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"[IdeaEngine] 图片文件不存在: {image_path}")
                return None
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            return {
                "blockType": "image",
                "options": {
                    "image": {
                        "base64": img_base64,
                        "caption": caption,
                        "image_path": image_path
                    }
                }
            }
        except Exception as e:
            logger.warning(f"[IdeaEngine] 读取图片失败 {image_path}: {e}")
            return None
