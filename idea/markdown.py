"""
Markdown 处理与图片工具方法
"""

import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote
from pathlib import Path
from astrbot.api import logger

from .ideas import IdeaEngineIdeas

# Canonical section structure.  Order matters — used for anchor placement.
# (canonical_heading, keyword_for_normalization)
CANONICAL_SECTIONS = [
    ("## 1. 背景动机", "背景动机"),
    ("## 2. 相关工作", "相关工作"),
    ("## 3. 方法论", "方法论"),
    ("## 4. 创新点", "创新点"),
    ("## 5. 实验Benchmark", "实验Benchmark"),
    ("## 6. 挑战与解决方案", "挑战与解决方案"),
    ("## 7. 下一步计划", "下一步计划"),
    ("## 8. 参考文献", "参考文献"),
]

# Map keyword → canonical heading for fast lookup
_SECTION_KEYWORD_TO_HEADING = {kw: h for h, kw in CANONICAL_SECTIONS}


class IdeaEngineMarkdown(IdeaEngineIdeas):
    """Markdown 处理与图片工具方法。继承链：... → IdeaEngineIdeas → IdeaEngineMarkdown"""

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

    @staticmethod
    def _normalize_section_headings(text: str) -> str:
        """逐行扫描，将含已知章节名的标题行替换为标准 ``## N. 章节名``。

        归一化空白后比较，容错 LLM 在中文与英文之间插入空格（如
        "实验 Benchmark" → "实验Benchmark"）。仅处理以 ``#`` 开头的行，
        避免正文中提及章节名导致的误替换。
        """
        lines = text.split('\n')
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                stripped_nospace = stripped.replace(' ', '').replace('\t', '')
                for kw, canonical in _SECTION_KEYWORD_TO_HEADING.items():
                    if kw in stripped_nospace:
                        result.append(canonical)
                        break
                else:
                    result.append(line)
            else:
                result.append(line)
        return '\n'.join(result)

    @staticmethod
    def _find_section_anchor(text: str, heading: str, next_heading: str | None) -> str | None:
        """用 ``str.find()`` 精确定位两个章节标题之间的最后一句话作为锚点。

        Args:
            text: 规范化后的文档文本
            heading: 当前章节的标准标题，如 ``"## 2. 相关工作"``
            next_heading: 下一章节的标准标题，如 ``"## 3. 方法论"``。
                          为 None 时取文档末尾。

        Returns:
            锚点文本（最后一句），或 None
        """
        start = text.find(heading)
        if start == -1:
            return None

        body_start = start + len(heading)
        if next_heading is not None:
            end = text.find(next_heading, body_start)
            if end != -1:
                body = text[body_start:end]
            else:
                # Next heading not found — take rest of text
                body = text[body_start:]
        else:
            body = text[body_start:]

        # 取最后一段非空文字的最后 60 个字符
        content_lines = [l.strip() for l in body.split('\n') if l.strip() and not l.strip().startswith('#')]
        if not content_lines:
            # 章节无正文，用标题本身作锚点
            anchor = heading.lstrip('#').strip()
            logger.info(f"[IdeaEngine] ⚠️ 锚点回退（章节无正文）: {anchor[:60]!r}")
            return anchor

        last_line = content_lines[-1]
        anchor = last_line[-60:].lstrip().rstrip('，；：！？, ;:!?')
        if len(anchor) >= 5:
            logger.info(f"[IdeaEngine] ✅ 锚点: {heading} 末尾 → {anchor[:60]!r}")
            return anchor

        logger.warning(f"[IdeaEngine] ❌ 锚点文本过短: {last_line!r}")
        return None

    @staticmethod
    def _find_figure_anchors(text: str) -> dict:
        """返回引用图表和方法论图表的插入锚点。

        用 ``str.find()`` 精确定位章节边界，不依赖正则或模糊匹配。
        调用前应先用 ``_normalize_section_headings`` 规范化章节标题。

        Returns:
            {"related_work": str | None, "methodology": str | None}
        """
        return {
            "related_work": IdeaEngineMarkdown._find_section_anchor(
                text, "## 2. 相关工作", "## 3. 方法论"
            ),
            "methodology": IdeaEngineMarkdown._find_section_anchor(
                text, "## 3. 方法论", "## 4. 创新点"
            ),
        }

    def _append_figure_section(self, text: str, knowledge: Optional[Dict[str, Any]] = None) -> tuple[list[dict], dict]:
        """收集图表信息并找到插入锚点。

        Returns:
            (figure_infos, anchors) 其中 anchors = {"related_work": str|None, "methodology": str|None}
        """
        empty_anchors = {"related_work": None, "methodology": None}

        if not knowledge:
            return [], empty_anchors

        local_results = knowledge.get("local_results", [])
        if not local_results:
            return [], empty_anchors

        caption_cache: Dict[str, Dict[str, Any]] = {}
        figure_infos: list[dict] = []

        for r in local_results:
            metadata = r.get('metadata', {})

            img_path = metadata.get('image_path', '')
            img_caption = metadata.get('image_caption', '')
            img_figure_num = str(metadata.get('image_figure_num', '') or '')
            if img_path:
                if os.path.exists(img_path):
                    img_filename = Path(img_path).name
                    paper_folder = Path(img_path).parent.name
                    if paper_folder not in caption_cache:
                        caption_cache[paper_folder] = self._load_figure_captions(img_path)
                    caps = caption_cache[paper_folder]
                    by_filename = caps.get("by_filename", {})
                    by_number = caps.get("by_number", {})
                    # Multi-strategy: by_filename → by_number → metadata → fallback
                    real_caption = by_filename.get(img_filename, "")
                    if not real_caption and img_figure_num:
                        real_caption = by_number.get(f"fig:{img_figure_num}", "")
                    if not real_caption:
                        real_caption = img_caption or img_filename
                    figure_infos.append({
                        "path": img_path,
                        "caption": real_caption,
                        "type": "fig",
                        "figure_num": img_figure_num,
                    })
                else:
                    logger.warning(f"[IdeaEngine] 图片路径不存在，跳过: {img_path}")

            table_png = metadata.get('table_png_path', '')
            if table_png:
                if os.path.exists(table_png):
                    table_filename = Path(table_png).name
                    table_num = str(metadata.get('table_num', '') or '')
                    paper_folder = Path(table_png).parent.name
                    if paper_folder not in caption_cache:
                        caption_cache[paper_folder] = self._load_figure_captions(table_png)
                    caps = caption_cache[paper_folder]
                    by_filename = caps.get("by_filename", {})
                    by_number = caps.get("by_number", {})
                    # Multi-strategy: by_filename → by_number → metadata → fallback
                    table_caption = by_filename.get(table_filename, "")
                    if not table_caption and table_num:
                        table_caption = by_number.get(f"table:{table_num}", "")
                    if not table_caption:
                        table_caption = metadata.get('table_caption', '') or Path(table_png).stem
                    figure_infos.append({
                        "path": table_png,
                        "caption": table_caption,
                        "type": "table",
                        "table_num": table_num,
                    })
                else:
                    logger.warning(f"[IdeaEngine] 表格图片路径不存在，跳过: {table_png}")

        if not figure_infos:
            logger.info("[IdeaEngine] 所有图表路径均不存在于磁盘")
            return [], empty_anchors

        # Number figures and tables
        fig_idx = 0
        tbl_idx = 0
        for fi in figure_infos:
            if fi["type"] == "table":
                tbl_idx += 1
                fi["caption"] = f"表 {tbl_idx}：{fi['caption']}"
            else:
                fig_idx += 1
                fi["caption"] = f"图 {fig_idx}：{fi['caption']}"
        logger.info(f"[IdeaEngine] 有效图表: {fig_idx} 张图, {tbl_idx} 张表 (共 {len(figure_infos)} 个)")

        # 规范化章节标题后用 str.find() 找锚点
        normalized = IdeaEngineMarkdown._normalize_section_headings(text)
        anchors = IdeaEngineMarkdown._find_figure_anchors(normalized)
        return figure_infos, anchors

    async def _enrich_figure_captions(
        self,
        figure_infos: list[dict],
        local_results: list[dict],
        llm_provider,
    ) -> list[dict]:
        """用 chunk 上下文通过 LLM 丰富图表 caption。

        figure_infos 的 caption 已含编号前缀（"图 1：Table 1"），
        此方法剥离前缀 → LLM 生成 → 重新加上前缀。LLM 失败时保留原 caption。
        """
        if not figure_infos or not local_results or not llm_provider:
            return figure_infos

        # 1. 构建 paper_folder → accumulated chunk text
        paper_chunks: dict[str, str] = {}
        for r in local_results:
            metadata = r.get('metadata', {})
            # 从 image_path 或 table_png_path 推导 paper folder
            for key in ('image_path', 'table_png_path'):
                p = metadata.get(key, '')
                if p:
                    paper_folder = Path(p).parent.name
                    break
            else:
                continue
            text = r.get('text', '')
            if not text:
                continue
            if paper_folder not in paper_chunks:
                paper_chunks[paper_folder] = ''
            paper_chunks[paper_folder] += text + '\n'

        # 1b. 构建 paper_folder → by_number（从 captions JSON）
        paper_by_number: dict[str, dict[str, str]] = {}
        for fi in figure_infos:
            fi_path = fi.get('path', '')
            if not fi_path:
                continue
            paper_folder = Path(fi_path).parent.name
            if paper_folder not in paper_by_number:
                caps = self._load_figure_captions(fi_path)
                paper_by_number[paper_folder] = caps.get("by_number", {})

        # 2. 逐个图表请求 LLM 生成 caption
        for fi in figure_infos:
            fi_path = fi.get('path', '')
            if not fi_path:
                continue

            paper_folder = Path(fi_path).parent.name
            chunk_text = paper_chunks.get(paper_folder, '')
            if not chunk_text:
                logger.info(f"[IdeaEngine] caption 跳过（无 chunk 上下文）: {Path(fi_path).name}")
                continue

            # 剥离编号前缀 "图 N：" / "表 N："
            prefix = ''
            numbering_match = re.match(r'^((?:图|表)\s*\d+[：:])', fi['caption'])
            if numbering_match:
                prefix = numbering_match.group(1)
                raw_caption = fi['caption'][numbering_match.end():].strip()
            else:
                raw_caption = fi['caption']

            filename = Path(fi_path).name

            # 区分两种场景：
            #   有实质原始 caption → 以原始 caption 为准，chunk 仅作上下文参考
            #   原始 caption 为空/仅编号 → 先尝试 by_number，再谨慎从 chunk 推断
            has_substantive_caption = bool(raw_caption and len(raw_caption) > 20)
            if not has_substantive_caption:
                # 尝试 by_number 查找（从 captions JSON 的 logical number）
                fi_type = fi.get('type', 'fig')
                logical_num = fi.get('figure_num', '') if fi_type == 'fig' else fi.get('table_num', '')
                by_number = paper_by_number.get(paper_folder, {})
                if logical_num:
                    typed_key = f"fig:{logical_num}" if fi_type == 'fig' else f"table:{logical_num}"
                    if typed_key in by_number:
                        raw_caption = by_number[typed_key]
                        has_substantive_caption = True
                        logger.info(f"[IdeaEngine] caption by_number 命中: {filename} "
                                    f"key={typed_key} → {raw_caption[:50]}...")

            if has_substantive_caption:
                prompt = f"""请将以下英文学术图表描述改写为简洁的中文描述（1-2句话，不超过80字）。

原始描述（以此为准，不得偏离原意）：
{raw_caption}

论文上下文（仅供参考，帮助理解背景）：
{chunk_text}

要求：
1. 严格以原始描述为准，保留其核心信息
2. 论文上下文仅用于帮助理解术语和背景，不得改变原始描述的含义
3. 将英文改写为中文，精简到1-2句话
4. 直接输出中文描述，不要加"如图"、"该图展示"等引导语
5. 不要输出"图 X"、"表 X"、"Figure X"、"Table X"等编号，直接描述内容"""
            else:
                prompt = f"""以下图表缺少原始描述，请根据论文上下文推断其内容，给出简洁的中文描述（1-2句话，不超过80字）。

图表文件：{filename}

论文上下文：
{chunk_text}

要求：
1. 从上下文找出与该图表最相关的信息，推断图表内容
2. 若上下文提到相邻编号的图表（如 Figure 3、Table 2），可据此推断该图表在论文中的位置和作用
3. 仅基于上下文已有信息，不编造未提及的内容
4. 直接输出中文描述，不要加"如图"、"该图展示"等引导语
5. 不要输出"图 X"、"表 X"、"Figure X"、"Table X"等编号，直接描述内容"""

            try:
                response = await llm_provider.text_chat(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=128,
                )
                new_caption = ''
                if hasattr(response, 'content'):
                    new_caption = response.content
                elif isinstance(response, dict):
                    new_caption = response.get('content', '') or response.get('text', '')
                else:
                    new_caption = str(response)

                new_caption = new_caption.strip().strip('"').strip("'").strip()
                # Strip redundant figure/table numbering from LLM output
                # (the prefix "图 N：" / "表 N：" is already applied above)
                new_caption = re.sub(
                    r'^(?:图|表|Figure|Fig\.?|Table)\s*[A-Za-z]?\d+[A-Za-z]?(?:[：:\s]+|$)',
                    '', new_caption
                ).strip()
                if new_caption:
                    old = fi['caption']
                    fi['caption'] = f"{prefix}{new_caption}" if prefix else new_caption
                    logger.info(f"[IdeaEngine] caption 已丰富: "
                                f"{old[:40]}... → {fi['caption'][:60]}...")
                else:
                    logger.warning(f"[IdeaEngine] LLM 返回空 caption，保留原值: {filename}")
            except Exception as e:
                logger.warning(f"[IdeaEngine] caption 丰富失败 ({filename}): {e}")

        return figure_infos

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

        caption_cache: Dict[str, Dict[str, Any]] = {}
        real_images: List[tuple[str, str]] = []
        for r in local_results:
            img_path = r.get('metadata', {}).get('image_path', '')
            if not img_path:
                continue
            img_filename = Path(img_path).name
            paper_folder = Path(img_path).parent.name
            if paper_folder not in caption_cache:
                caption_cache[paper_folder] = self._load_figure_captions(img_path)
            caps = caption_cache[paper_folder]
            by_filename = caps.get("by_filename", {})
            real_caption = by_filename.get(img_filename, img_filename)
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
