"""
研究创意生成引擎（简化版）

核心流程：VLM生成ideas → VLM生成周报草稿 → 格式化输出

关键约束：
- 不调用云端LLM润色
- 不调用网络搜索
- 不做复杂的媒体caption增强
"""

import hashlib
import json
import re
import os
import shutil
import asyncio
import mistune
import base64
import concurrent.futures
import tempfile
from urllib.parse import unquote
from typing import Dict, Any, List, Optional, Tuple, cast
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from astrbot.api import logger
from astrbot.core.astr_main_agent_resources import TOOL_CALL_PROMPT


@dataclass
class ResearchIdea:
    """研究想法"""
    title: str
    description: str
    novelty: str
    methodology: str
    potential_challenges: List[str]
    related_work: List[str]
    feasibility: float
    inspiration_sources: List[str]


@dataclass
class TopicAnalysis:
    """主题分析结果"""
    domain: str
    keywords: List[str]
    search_queries: List[str]
    local_rag_queries: List[str]
    exploration_angles: List[str]
    summary: str


class IdeaEngine:
    """
    研究创意生成引擎（简化版）

    使用流程：
    1. generate_ideas - 基于本地RAG结果生成研究想法
    2. _generate_initial_draft_vlm - VLM生成周报草稿
    3. to_feishu_markdown - 格式化输出
    """

    def __init__(self, context, rag_engine=None):
        """
        初始化创意引擎

        Args:
            context: AstrBot上下文（用于LLM/VLM调用）
            rag_engine: RAG引擎实例
        """
        self.context = context
        self._rag_engine = rag_engine

    def _check_bright_data_config(self) -> bool:
        """检查 Bright Data MCP 是否已配置"""
        try:
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            if not mcp_config_path.exists():
                logger.warning("[IdeaEngine] mcp_server.json 不存在，Bright Data 搜索将不可用")
                return False
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)
            api_token = mcp_config.get("mcpServers", {}).get("BrightData", {}).get("env", {}).get("API_TOKEN", "")
            if not api_token:
                logger.warning("[IdeaEngine] Bright Data API Token 未配置，网络搜索将不可用")
                return False
            return True
        except Exception as e:
            logger.warning(f"[IdeaEngine] 检查 Bright Data 配置失败: {e}")
            return False

    def _get_feishu_tool(self):
        """获取飞书MCP工具"""
        if not self.context:
            logger.error("[IdeaEngine] context 为 None")
            return None
        provider_manager = getattr(self.context, 'provider_manager', None)
        if not provider_manager:
            logger.error("[IdeaEngine] provider_manager 为 None")
            return None
        llm_tools = getattr(provider_manager, 'llm_tools', None)
        if not llm_tools:
            logger.error("[IdeaEngine] llm_tools 为 None")
            return None

        func_list = getattr(llm_tools, 'func_list', [])
        logger.debug(f"[IdeaEngine] func_list 长度: {len(func_list)}")

        # 查找 feishu 相关的工具
        for tool in func_list:
            if 'feishu' in tool.name.lower():
                logger.info(f"[IdeaEngine] 找到飞书工具: {tool.name}")
                return tool
        return None

    def _strip_markdown_style(self, text: str) -> str:
        """移除 Markdown 样式标记，保留纯文本（仅用于标题等纯文本块）"""
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        return text

    def _strip_outer_markdown_style(self, text: str) -> str:
        """移除整行文本的外层 Markdown 样式标记（当整行都是样式文本时）"""
        if re.match(r'^(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)$', text):
            return self._strip_markdown_style(text)
        return text

    def _create_feishu_markdown(self) -> mistune.Markdown:
        """创建带自定义插件的 mistune Markdown（LaTeX + 图表引用）"""

        def parse_fig_ref(md, m, state):
            text = m.group(0)
            state.append_token({'type': 'fig_ref', 'raw': text})
            return m.end()

        def render_fig_ref(renderer, text):
            return f'<strong>{text}</strong>'

        def parse_latex(md, m, state):
            latex_match = m.group('latex')
            if latex_match:
                formula = latex_match[1:-1]
                state.append_token({'type': 'latex', 'raw': formula})
            return m.end()

        def render_latex(renderer, text):
            return f'<eq>{text}</eq>'

        md = mistune.create_markdown(plugins=['strikethrough'])
        assert md.renderer is not None
        md.inline.register('fig_ref', r'\[(图|表)(\d+)\]', parse_fig_ref, before='link')
        md.renderer.register('fig_ref', render_fig_ref)
        md.inline.register('latex', r'\$([^$\n]+?)\$', parse_latex, before='emphasis')
        md.renderer.register('latex', render_latex)
        return md

    def _parse_html_with_html_parser(self, html: str) -> List[Dict[str, Any]]:
        """使用 Python 内置 html.parser 解析 HTML"""
        from html.parser import HTMLParser
        from html import unescape

        class FeishuHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = []
                self.current_text = ""
                self.styles = {}
                self.link_url = None
                self._in_eq = False
                self._eq_text = ""

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs) if attrs else {}
                if self.current_text and tag not in ('br', 'img'):
                    self.result.append({
                        "text": unescape(self.current_text),
                        "style": dict(self.styles)
                    })
                    self.current_text = ""

                if tag == 'strong':
                    self.styles['bold'] = True
                elif tag == 'em':
                    self.styles['italic'] = True
                elif tag == 'code':
                    self.styles['inline_code'] = True
                elif tag in ('del', 's', 'strike'):
                    self.styles['strikethrough'] = True
                elif tag == 'a':
                    self.link_url = attrs_dict.get('href')
                    self.styles['bold'] = True
                elif tag == 'br':
                    self.result.append({"text": "\n", "style": {}})
                elif tag == 'img':
                    # 处理 markdown 图片：先输出前面的文本，再处理图片
                    if self.current_text:
                        self.result.append({"text": unescape(self.current_text), "style": dict(self.styles)})
                        self.current_text = ""
                    src = attrs_dict.get('src', '')
                    alt = attrs_dict.get('alt', '图片')
                    if src:
                        self.result.append({"text": f"[图片: {alt}]({src})", "style": {}})
                elif tag == 'eq':
                    self._in_eq = True
                    self._eq_text = ""

            def handle_endtag(self, tag):
                if tag == 'eq' and self._in_eq:
                    self.result.append({"equation": self._eq_text, "style": {}})
                    self._in_eq = False
                    self._eq_text = ""
                    return

                if self.current_text:
                    self.result.append({
                        "text": unescape(self.current_text),
                        "style": dict(self.styles)
                    })
                    self.current_text = ""

                if tag == 'a':
                    if self.link_url:
                        self.result.append({"text": f" ({self.link_url})", "style": {}})
                        self.link_url = None
                        self.styles.pop('bold', None)
                elif tag in ('strong', 'em', 'code', 'del', 's', 'strike'):
                    key = {'strong': 'bold', 'em': 'italic', 'code': 'inline_code',
                           'del': 'strikethrough', 's': 'strikethrough', 'strike': 'strikethrough'}.get(tag)
                    if key:
                        self.styles.pop(key, None)

            def handle_data(self, data):
                if self._in_eq:
                    self._eq_text += data
                else:
                    self.current_text += data

            def handle_entityref(self, name):
                if self._in_eq:
                    self._eq_text += unescape(f'&{name};')
                else:
                    self.current_text += unescape(f'&{name};')

            def handle_charref(self, name):
                if self._in_eq:
                    self._eq_text += unescape(f'&#{name};')
                else:
                    self.current_text += unescape(f'&#{name};')

        html = html.replace('<p>', '').replace('</p>', '').strip()
        parser = FeishuHTMLParser()
        parser.feed(html)
        if parser.current_text:
            parser.result.append({
                "text": unescape(parser.current_text),
                "style": dict(parser.styles)
            })
        merged = []
        for item in parser.result:
            if merged and merged[-1].get('text') and item.get('text') and merged[-1].get('style') == item.get('style'):
                merged[-1]['text'] += item['text']
            else:
                merged.append(item)
        return merged

    def _parse_inline_styles(self, text: str) -> List[Dict[str, Any]]:
        """使用 mistune + html.parser 解析 Markdown 文本，返回飞书 textStyles 格式"""
        if not text:
            return [{"text": "", "style": {}}]
        try:
            md = self._create_feishu_markdown()
            html = md(text)
            result = self._parse_html_with_html_parser(cast(str, html))
            if result and any(item.get("text") or item.get("equation") for item in result):
                return result
        except Exception as e:
            logger.warning(f"[IdeaEngine] mistune 解析失败: {e}")
        return [{"text": text, "style": {}}]

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
                content = self._strip_markdown_style(line[2:].strip())
                blocks.append({
                    "blockType": "heading",
                    "options": {"heading": {"level": 1, "content": content}}
                })
            elif line.startswith("## ") and not line.startswith("### "):
                content = self._strip_markdown_style(line[3:].strip())
                blocks.append({
                    "blockType": "heading",
                    "options": {"heading": {"level": 2, "content": content}}
                })
            elif line.startswith("### "):
                content = self._strip_markdown_style(line[4:].strip())
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
                        "_textStyles": self._parse_inline_styles(raw_content)
                    })
            elif re.match(r'^\d+[\.\)]\s', line):
                match = re.match(r'^(\d+[\.\)])\s+(.*)$', line)
                if match:
                    raw_content = match.group(2).strip()
                    if raw_content:
                        blocks.append({
                            "blockType": "list",
                            "options": {"list": {"content": raw_content, "isOrdered": True}},
                            "_textStyles": self._parse_inline_styles(raw_content)
                        })
            elif line.strip() == "":
                pass
            else:
                text_content = line.strip()
                if text_content:
                    segments = self._extract_inline_images(text_content)
                    # 合并所有文本段为一个文本块
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
                            "options": {"text": {"textStyles": self._parse_inline_styles(merged_text)}}
                        })
                    # 图片块紧跟在文本块后面
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
            end_markers = [f'.{ext})', f'.{ext}）']  # 英文和中文括号
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
                    if text[j] == ')':
                        paren_count += 1
                    elif text[j] == '(':
                        paren_count -= 1
                        if paren_count == 0:
                            found_paren = True
                            break
                    j -= 1

                if found_paren and j >= 2:
                    if text[j-1] == ']' and text[j-2] == '!':
                        # slice includes .ext) so we strip last char
                        path = text[j+1:idx+len(ext)+2]
                        path = path[:-1]
                        path = unquote(path)
                        caption_start = text.rfind('![', 0, j-2)
                        caption = text[caption_start+2:text.find(']', caption_start)] if caption_start >= 0 else ''
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

        # 格式1: (path) 格式（支持 ASCII ) 和全角 ））
        path = self._extract_path_from_paren(line)
        if path:
            return path, line

        # 格式2: 路径：/path 或 路径 /path 格式（支持结尾为 ） 或 )）
        path_match = re.search(r'路径[：:\s]+([/][^\s]+\.(?:' + _EXT + r')[)）]?', line)
        if path_match:
            return path_match.group(1).rstrip(')）'), line

        # 格式3: 标准 markdown 图片（使用平衡括号计数，支持路径中含括号）
        md_images = self._extract_markdown_image_from_text(line)
        if md_images:
            path, caption = md_images[0]
            return path, caption

        return None, None

    def _normalize_figure_references(self, markdown_text: str) -> str:
        """
        前处理：
        1. 将论文图表章节中的各种图片格式转换为标准 markdown 图片语法
        2. 将正文中的 markdown 图片语法转换为纯文字引用
        3. 如果没有论文图表章节，则在全文查找图片并移到末尾
        """
        logger.info(f"[IdeaEngine] normalize 输入长度: {len(markdown_text)}")
        markdown_text = unquote(markdown_text)  # 全局URL解码

        # 打印前200字符用于调试
        logger.debug(f"[IdeaEngine] normalize 输入前200字符: {markdown_text[:200]}")

        figure_match = re.search(r'##.*论文图表', markdown_text)

        if figure_match:
            # 有论文图表章节，按原有逻辑处理
            before = markdown_text[:figure_match.start()]
            section = markdown_text[figure_match.start():]

            logger.info(f"[IdeaEngine] ✅ 找到论文图表章节: 位置={figure_match.start()}, 正文长度={len(before)}, 章节长度={len(section)}")
            logger.debug(f"[IdeaEngine] 论文图表章节前100字符: {section[:100]}")

            # 使用平衡括号计数提取正文中的 markdown 图片
            body_images = self._extract_markdown_image_from_text(before)
            logger.info(f"[IdeaEngine] 正文中找到 {len(body_images)} 张 markdown 图片")

            # 将正文中的 markdown 图片转为纯文字引用
            body_images_removed = before
            for path, caption in body_images:
                # 替换 ![caption](path) 为 caption (path)
                md_pattern = re.escape(f'![{caption}]({path})')
                body_images_removed = re.sub(md_pattern, f'{caption} ({path})', body_images_removed)
                logger.debug(f"[IdeaEngine] 移除正文图片: {caption} ({path})")

            # 处理论文图表章节中的路径格式（使用平衡括号计数）
            fig_images = self._extract_markdown_image_from_text(section)
            logger.info(f"[IdeaEngine] 论文图表章节中找到 {len(fig_images)} 张图片: {fig_images}")

            # 处理论文图表章节：解析多行格式，提取图片并过滤原始文本行
            # 支持格式：
            #   ## 论文图表
            #   图 1：caption
            #   路径：/path/to/fig.PNG）
            #   说明：描述文字
            # 输出：## 论文图表\n![caption](path)\n（仅保留标题和图片，原始说明行全部过滤）
            lines = section.split('\n')
            result = []
            found_images = []
            pending_path = None  # 等待与下一行 caption 合并

            for line in lines:
                stripped = line.strip()

                # 行1：裸路径 /path/to/fig.png
                if stripped.startswith('/') and re.search(r'\.(?:png|jpg|jpeg|webp|gif|PNG|JPG|JPEG|WEBP|GIF)$', stripped):
                    pending_path = stripped
                    continue

                # 行2：图 X caption
                if pending_path is not None:
                    fig_cap_match = re.match(r'^图\s*\d+\s*(.+)$', stripped)
                    if fig_cap_match:
                        caption = fig_cap_match.group(1).strip()
                        result.append(f'![{caption}]({pending_path})')
                        found_images.append((caption, pending_path))
                        logger.info(f"[IdeaEngine] 章节转换图片: caption='{caption}', path='{pending_path}'")
                        pending_path = None
                        continue
                    # 不是 caption 行，path 降级为普通文本
                    result.append(pending_path)
                    pending_path = None

                # 保留其他行（## 标题、空行等）
                result.append(line)

            logger.info(f"[IdeaEngine] normalize 完成: 正文移除图片={len(body_images)}, 章节转换图片={len(found_images)}")
            logger.debug(f"[IdeaEngine] normalize 输出末尾200字符: {(body_images_removed + chr(10).join(result))[-200:]}")
            return body_images_removed + '\n'.join(result)

        else:
            # 没有论文图表章节！需要在全文查找图片并移到末尾
            logger.warning("[IdeaEngine] 未找到论文图表章节，将在全文查找图片并移到末尾")

            # 使用平衡括号计数找出所有 markdown 图片
            images = self._extract_markdown_image_from_text(markdown_text)


            if not images:
                logger.info("[IdeaEngine] 全篇无 markdown 图片，跳过处理")
                return markdown_text

            logger.info(f"[IdeaEngine] 全篇找到 {len(images)} 张图片: {images}")

            # 将正文中的所有 markdown 图片转为纯文字
            text_only = markdown_text
            for path, caption in images:
                md_pattern = re.escape(f'![{caption}]({path})')
                text_only = re.sub(md_pattern, f'{caption} ({path})', text_only)

            # 在末尾添加论文图表章节
            fig_section = "\n\n## 论文图表\n"
            for i, (path, caption) in enumerate(images, 1):
                if not caption:
                    caption = os.path.splitext(os.path.basename(path))[0]
                fig_section += f"\n![{caption}]({path})\n"
                logger.info(f"[IdeaEngine] 添加图片到末尾: caption='{caption}', path='{path}'")

            logger.info(f"[IdeaEngine] normalize 完成: 移动 {len(images)} 张图片到末尾")
            return text_only + fig_section

    def _append_figure_section(self, text: str, knowledge: Optional[Dict[str, Any]] = None) -> str:
        """
        在参考文献章节之后追加论文图表章节。
        用真实图片路径和 caption，不依赖LLM生成。
        """
        if not knowledge:
            logger.warning("[IdeaEngine] [_append_figure_section] knowledge 为空，跳过")
            return text

        local_results = knowledge.get("local_results", [])
        logger.info(f"[IdeaEngine] [_append_figure_section] local_results 数量: {len(local_results)}")
        if not local_results:
            return text

        # 获取所有图片（跳过 relevance 过滤，末尾 append 不需要）
        caption_cache: Dict[str, Dict[str, str]] = {}
        figure_entries: List[tuple[str, str]] = []  # (caption, path)

        for r in local_results:
            metadata = r.get('metadata', {})
            img_path = metadata.get('image_path', '')
            img_caption = metadata.get('image_caption', '')
            if not img_path or not os.path.exists(img_path):
                continue
            img_filename = Path(img_path).name
            paper_folder = Path(img_path).parent.name
            if paper_folder not in caption_cache:
                caption_cache[paper_folder] = self._load_figure_captions(img_path)
            fname_to_caption = caption_cache[paper_folder]
            real_caption = fname_to_caption.get(img_filename) or img_caption or img_filename
            figure_entries.append((real_caption, img_path))

        if not figure_entries:
            logger.warning(f"[IdeaEngine] [_append_figure_section] figure_entries 为空（可能被 os.path.exists 跳过），共检查 {len(local_results)} 条 local_results")
            return text

        logger.info(f"[IdeaEngine] [_append_figure_section] 成功构建 {len(figure_entries)} 个图表条目")
        fig_section = "\n## 9. 论文图表\n"
        for i, (caption, path) in enumerate(figure_entries, 1):
            fig_section += f"![图 {i}：{caption}]({path})\n\n"

        # 追加到参考文献之后
        ref_match = re.search(r'(##\s*参考文献.*?)(?=##\s|\Z)', text, re.DOTALL)
        if ref_match:
            insert_pos = ref_match.end()
            text = text[:insert_pos] + fig_section + text[insert_pos:]
            logger.info(f"[IdeaEngine] 论文图表章节已追加，共 {len(figure_entries)} 张图片")
        else:
            text += fig_section
            logger.info(f"[IdeaEngine] 未找到参考文献章节，直接在末尾追加论文图表，共 {len(figure_entries)} 张图片")

        return text

    def _replace_placeholder_paths_by_caption(self, text: str, local_results: List[Dict]) -> str:
        """
        用 caption 匹配将草稿中的占位符图片路径替换为真实路径。
        1. 从 local_results 构建真实图片路径 + caption 列表
        2. 提取草稿论文图表章节中的 placeholder 图片
        3. 用 caption 关键词匹配找到真实路径并替换
        """
        import re
        # 找论文图表章节
        fig_match = re.search(r'(##\s*论文图表.*?)(?=##\s|\Z)', text, re.DOTALL)
        if not fig_match:
            return text

        # 构建真实图片列表：路径 -> caption
        caption_cache: Dict[str, Dict[str, str]] = {}
        real_images: List[tuple[str, str]] = []  # (path, caption)
        for r in local_results:
            img_path = r.get('image_path', '')
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

        # 提取章节中所有 placeholder 图片（路径不以 /Users 开头）
        placeholder_pattern = re.findall(r'(!\[(.*?)\]\()(/[^\)]+\.(?:png|jpg|jpeg|webp|gif))\)', fig_match.group(1))
        if not placeholder_pattern:
            return text

        result_text = text
        for full_match, placeholder_caption, placeholder_path in placeholder_pattern:
            if placeholder_path.startswith('/Users'):
                continue  # 已是真实路径，跳过
            # 用 caption 关键词匹配找真实图片
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

        # 查找所有可能的图片路径，使用平衡括号计数
        result = []
        i = 0
        converted_count = 0
        while i < len(text):
            # 检查是否是 (/
            if text[i] == '(' and i + 1 < len(text) and text[i + 1] == '/':
                logger.debug(f"[_convert_paren_paths_to_markdown] 发现可能的路径开始于 i={i}")
                # 找到了可能的图片路径开始
                # 查找 .{ext}) 的位置
                found_ext = None
                ext_end = -1
                for ext in _EXTENSIONS:
                    end_marker = f'.{ext})'
                    pos = text.find(end_marker, i + 1)
                    if pos >= 0:
                        if found_ext is None or pos < found_ext[0]:
                            found_ext = (pos, len(ext), ext)

                if found_ext:
                    ext_pos, ext_len, ext = found_ext
                    logger.debug(f"[_convert_paren_paths_to_markdown] 找到扩展名 {ext} at ext_pos={ext_pos}")
                    # 使用平衡括号计数找到匹配的 (
                    start = i + 1
                    # 检查这个 ( 和 .ext) 之间的括号是否平衡
                    inner = text[start:ext_pos + ext_len + 1]
                    paren_count = 1
                    for ch in inner:
                        if ch == '(':
                            paren_count += 1
                        elif ch == ')':
                            paren_count -= 1

                    logger.debug(f"[_convert_paren_paths_to_markdown] 括号平衡检查: paren_count={paren_count}, inner={inner[:50]}...")

                    if paren_count == 1:
                        # 这是一个有效的路径格式
                        path = text[start:ext_pos + ext_len + 1]  # includes .ext)
                        path = path[:-1]  # remove trailing )
                        logger.debug(f"[_convert_paren_paths_to_markdown] 有效路径: path={path}, i={i}, text[i-1]={repr(text[i-1] if i > 0 else '')}")
                        # 检查前面是否是 ] (说明已经有 ![...](...) 格式)
                        if i > 0 and text[i - 1] == ']':
                            # 已经有 caption，这是标准 markdown 图片，跳过
                            logger.debug(f"[_convert_paren_paths_to_markdown] 前面是 ]，跳过（已有caption）")
                            result.append(text[i])
                            i += 1
                            continue
                        # 转换为 markdown 图片格式
                        result.append(f'![image]({path})')
                        converted_count += 1
                        logger.info(f"[_convert_paren_paths_to_markdown] ✅ 转换裸路径为图片: {path}")
                        i = ext_pos + ext_len + 1
                        continue
                    else:
                        logger.debug(f"[_convert_paren_paths_to_markdown] 括号不平衡，跳过")

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

        # 预处理：将裸括号路径 (/abs/path/img.ext) 转为标准 markdown 格式
        # 使用平衡括号计数，支持路径中含括号如 2502.12138v4(nopo)
        text_after_preprocess = self._convert_paren_paths_to_markdown(text)
        logger.debug(f"[_extract_inline_images] 预处理后文本长度: {len(text_after_preprocess)}")
        if text_after_preprocess != text:
            logger.info(f"[_extract_inline_images] ⚠️ 预处理有变化，长度变化: {len(text)} -> {len(text_after_preprocess)}")
            if len(text_after_preprocess) < 500:
                logger.debug(f"[_extract_inline_images] 预处理后文本: {text_after_preprocess}")
        text = text_after_preprocess

        # 预处理：处理 "图 X：caption [path]" 格式（无 ! 前缀的括号路径）
        # 如 "图 1：方法流程 [/path/xxx.png]" 或 "图1：方法流程 [path]" 转为 "![图1：方法流程](path)"
        _EXT = r'png|jpg|jpeg|webp|gif'
        text = re.sub(
            r'(图\s*\d+[：:]\s*)(.+?)\s*\[(.+?\.(?:' + _EXT + r'))\]',
            r'![\1\2](\3)',
            text
        )

        # 预处理：处理 "本地图-N: caption [path]" 格式（来自 media_lines 的格式）
        # 如 "本地图-1: 方法流程描述 [/path/fig.png]" 转为 "![本地图-1: 方法流程描述](path)"
        text = re.sub(
            r'(本地图-\d+[：:]\s*)(.+?)\s*\[(.+?\.(?:' + _EXT + r'))\]',
            r'![\1\2](\3)',
            text
        )

        segments = []
        stripped = text.strip()
        # 简单直接：找最后一个 ) 和对应的 (
        if stripped.startswith('![') and '.png' in stripped.lower():
            last_paren = stripped.rfind(')')
            open_bracket = stripped.rfind('](', 0, last_paren)
            if open_bracket > 0:
                caption = stripped[2:open_bracket]
                path = stripped[open_bracket+2:last_paren]
                path = unquote(path)
                if os.path.exists(path):
                    segments.append({"type": "image", "path": path, "caption": caption})
                else:
                    logger.warning(f"[_extract_inline_images] 图片不存在: {path}")
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

    def _build_citations_context(self, knowledge: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        构建引用上下文，包含本地检索和网络搜索的结果

        Args:
            knowledge: 知识检索结果

        Returns:
            Tuple[str, Dict]: (格式化的引用上下文字符串, 提取的媒体资源字典)
                媒体资源格式: {
                    "images": [{"index": "本地图-1", "path": str, "base64": str, "caption": str}, ...],
                    "tables": [{"index": "本地表-1", "csv_path": str, "png_path": str, "caption": str, "csv_content": str}, ...]
                }
        """
        if not knowledge:
            return "（无可用引用来源）", {"images": [], "tables": []}

        parts: List[str] = []
        extracted_media: Dict[str, Any] = {"images": [], "tables": []}
        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])

        # 本地检索引用
        local_image_idx = 0
        local_table_idx = 0
        if local_results:
            parts.append("## 本地论文检索引用：\n")
            for i, result in enumerate(local_results[:8], 1):  # 最多8条
                paper = result.get("paper", "Unknown")
                page = result.get("page", "N/A")
                text = result.get("text", "")[:300]
                score = result.get("score", 0.0)
                metadata = result.get("metadata", {})
                file_name = metadata.get("file_name", "")

                # 从 filename 提取 arxiv ID
                arxiv_id = ""
                if file_name:
                    import re
                    match = re.match(r'^(\d{4}\.\d{4,})', file_name)
                    if match:
                        arxiv_id = match.group(1)

                # 检查是否有图片
                image_path = metadata.get("image_path")
                img_index = None
                img_base64 = None
                img_caption = None
                if image_path:
                    if os.path.exists(image_path):
                        local_image_idx += 1
                        img_index = f"本地图-{local_image_idx}"
                        img_caption = metadata.get("image_caption", f"图 {local_image_idx}")
                        # 读取图片并转为 base64
                        try:
                            import base64
                            with open(image_path, "rb") as f:
                                img_base64 = base64.b64encode(f.read()).decode("utf-8")
                            extracted_media["images"].append({
                                "index": img_index,
                                "path": image_path,
                                "base64": img_base64,
                                "caption": img_caption,
                                "source_paper": paper,
                                "source_page": page
                            })
                        except Exception as e:
                            logger.warning(f"[IdeaEngine] 读取图片失败 {image_path}: {e}")
                            img_base64 = None
                    else:
                        logger.warning(f"[IdeaEngine] 图片元数据存在但文件缺失: {image_path}")

                # 构建引用（无论图片是否存在都添加引用）
                if arxiv_id:
                    ref_str = f"{paper} (https://arxiv.org/abs/{arxiv_id})"
                else:
                    ref_str = paper
                if img_index:
                    parts.append(f"- {ref_str} (页码: {page}, 相关度: {score:.3f}, 图片: {img_index})\n")
                    if img_base64:
                        parts.append(f"  - 图片说明: {img_caption}\n")
                else:
                    parts.append(f"- {ref_str} (页码: {page}, 相关度: {score:.3f})\n")

                # 检查是否有表格
                table_csv_path = metadata.get("table_csv_path")
                table_png_path = metadata.get("table_png_path")
                table_caption = metadata.get("table_caption", "")

                # 尝试推断 md_path 和 png_path（与 csv 同目录，同名文件）
                table_md_path = ""
                if table_csv_path:
                    table_md_path = table_csv_path.replace(".csv", ".md")
                # 如果 png_path 为空，尝试从 csv_path 推断
                if not table_png_path and table_csv_path:
                    table_png_path = table_csv_path.replace(".csv", ".png")
                    if not os.path.exists(table_png_path):
                        table_png_path = ""  # 推断的路径也不存在，则置空

                if table_csv_path or table_png_path:
                    local_table_idx += 1
                    tbl_index = f"本地表-{local_table_idx}"
                    csv_content = ""
                    if table_csv_path:
                        if not os.path.exists(table_csv_path):
                            logger.warning(f"[IdeaEngine] 表格CSV元数据存在但文件缺失: {table_csv_path}")
                        else:
                            try:
                                with open(table_csv_path, "r", encoding="utf-8") as f:
                                    csv_content = f.read()[:500]  # 限制内容长度
                                extracted_media["tables"].append({
                                    "index": tbl_index,
                                    "csv_path": table_csv_path,
                                    "png_path": table_png_path,
                                    "md_path": table_md_path if os.path.exists(table_md_path) else "",
                                    "caption": table_caption,
                                    "csv_content": csv_content,
                                    "source_paper": paper,
                                    "source_page": page
                                })
                            except Exception as e:
                                logger.warning(f"[IdeaEngine] 读取表格失败 {table_csv_path}: {e}")
                                csv_content = "(无法读取)"
                    if table_png_path and not os.path.exists(table_png_path):
                        logger.warning(f"[IdeaEngine] 表格PNG元数据存在但文件缺失: {table_png_path}")

                    if not any(t["index"] == tbl_index for t in extracted_media["tables"]):
                        extracted_media["tables"].append({
                            "index": tbl_index,
                            "csv_path": table_csv_path or "",
                            "png_path": table_png_path or "",
                            "md_path": table_md_path if os.path.exists(table_md_path) else "",
                            "caption": table_caption,
                            "csv_content": csv_content,
                            "source_paper": paper,
                            "source_page": page
                        })
                    parts.append(f"    └─ 包含表格: {tbl_index} - {table_caption}\n")

                parts.append(f"    摘要: {text}...\n\n")

        # 网络搜索引用（直接使用 Markdown 链接格式）
        if web_results:
            parts.append(f"## 网络搜索引用：\n")
            for i, result in enumerate(web_results[:5], 1):  # 最多5条
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                snippet = result.get("snippet", "")[:300]
                # 直接使用 Markdown 链接格式
                link_str = f"[{title}]({url})" if url else title
                parts.append(f"- {link_str}\n")
                parts.append(f"  - 摘要: {snippet}...\n\n")

        if not parts:
            return "（无可用引用来源）", {"images": [], "tables": []}

        return "\n".join(parts), extracted_media

    def _is_simple_caption(self, caption: str) -> bool:
        """检查 caption 是否只是简单的编号而没有实际描述"""
        if not caption:
            return True
        # 简单检查：只包含 "图 X" 或 "Figure X" 且没有实际描述
        import re
        patterns = [
            r'^图\s*\d+$',
            r'^Figure\s*\d+$',
            r'^fig\.\s*\d+$',
            r'^\d+$',
        ]
        for pattern in patterns:
            if re.match(pattern, caption.strip(), re.IGNORECASE):
                return True
        return False

    async def _enhance_media_captions(self, extracted_media: Dict, knowledge: Dict) -> Dict:
        """增强媒体 caption（简化版直接返回）"""
        return extracted_media

    async def test_feishu_markdown_formats(self, folder_token: str = "") -> Dict[str, Any]:
        """
        测试用：列表样式 + 图片插入 + 引用链接
        """
        from astrbot.core.agent.run_context import ContextWrapper
        ctx_wrapper = ContextWrapper(context=self.context)

        provider_manager = getattr(self.context, 'provider_manager', None)
        if not provider_manager:
            return {"success": False, "error": "provider_manager 不可用"}

        llm_tools = getattr(provider_manager, 'llm_tools', None)

        # 收集工具
        feishu_tool = add_blocks_tool = upload_image_tool = update_text_tool = get_blocks_tool = None
        if llm_tools:
            for tool in getattr(llm_tools, 'func_list', []):
                if tool.name == 'create_feishu_document':
                    feishu_tool = tool
                elif tool.name == 'batch_create_feishu_blocks':
                    add_blocks_tool = tool
                elif tool.name == 'upload_and_bind_image_to_block':
                    upload_image_tool = tool
                elif tool.name == 'batch_update_feishu_block_text':
                    update_text_tool = tool
                elif tool.name == 'get_feishu_document_blocks':
                    get_blocks_tool = tool

        if not feishu_tool or not add_blocks_tool:
            return {"success": False, "error": "缺少必要工具"}

        # 从 initial_draft.md 读取内容测试
        draft_path = "/Users/chenyifeng/AstrBot/data/plugin_data/astrbot_plugin_paperrag/ideas/8a160941c48c813c/initial_draft.md"
        try:
            with open(draft_path, "r", encoding="utf-8") as f:
                test_markdown = f.read()
            test_markdown = unquote(test_markdown)  # URL解码
            logger.info(f"[Test] 读取测试文档: {draft_path}, 长度={len(test_markdown)}")
        except Exception as e:
            logger.error(f"[Test] 读取文件失败: {e}")
            return {"success": False, "error": f"读取文件失败: {e}"}

        # 转换为块
        all_blocks = self._markdown_to_feishu_blocks(test_markdown)
        image_count = sum(1 for b in all_blocks if b.get("blockType") == "image")
        list_count = sum(1 for b in all_blocks if b.get("blockType") == "list")
        list_with_styles = sum(1 for b in all_blocks if b.get("blockType") == "list" and b.get("_textStyles"))
        logger.info(f"[Test] 转换 {len(all_blocks)} 个块: {image_count} 图片, {list_count} 列表(其中 {list_with_styles} 个含样式)")

        # 创建飞书文档
        create_result = await feishu_tool.call(ctx_wrapper, title="[测试] 列表样式+图片+引用", folderToken=folder_token or "")

        doc_info = {}
        if hasattr(create_result, 'content') and create_result.content:
            result_text = getattr(create_result.content[0], 'text', None) or str(create_result.content[0])
            try:
                doc_info = json.loads(result_text)
            except json.JSONDecodeError:
                pass

        document_id = (
            doc_info.get("document", {}).get("document_id")
            or doc_info.get("document_id")
            or doc_info.get("objToken")
            or doc_info.get("obj_token")
        )
        if not document_id:
            return {"success": False, "error": f"文档创建失败: {create_result}"}

        # 插入块（交错：文本批量，图片逐张两步上传）
        images_uploaded = 0
        current_index = 0
        text_batch: list = []
        batch_start_index = 0
        # 记录哪些 all_blocks 索引对应到列表块（需要后续更新样式）
        # (原始文本内容, _textStyles)
        list_items_to_update: list[tuple[str, dict]] = []

        async def flush_batch():
            nonlocal text_batch, batch_start_index
            if not text_batch:
                return
            result = await add_blocks_tool.call(
                ctx_wrapper, documentId=document_id,
                parentBlockId=document_id, index=batch_start_index, blocks=text_batch
            )
            if hasattr(result, 'isError') and result.isError:
                raw_text = ""
                if hasattr(result, 'content') and result.content:
                    raw_text = getattr(result.content[0], 'text', '') or str(result.content[0])
                logger.error(f"[Test] 文本块插入失败: {raw_text[:300]}")
            else:
                logger.info(f"[Test] 插入 {len(text_batch)} 个块 (index={batch_start_index})")
            text_batch = []

        for b in all_blocks:
            if b.get("blockType") == "image":
                await flush_batch()

                opts = b.get("options", {}).get("image", {})
                img_path = opts.get("image_path", "")
                img_base64 = opts.get("base64", "")
                if not img_path and img_base64:
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp.write(base64.b64decode(img_base64))
                    tmp.close()
                    img_path = tmp.name

                if img_path and os.path.exists(img_path):
                    img_path = self._ensure_png(img_path)
                    img_caption = opts.get("caption", "")
                    logger.info(f"[Test] caption='{img_caption}'")
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(img_path) as pil_img:
                            orig_w, orig_h = pil_img.size
                        img_width, img_height = orig_w, orig_h
                        logger.info(f"[Test] 图片: {img_width}x{img_height}")
                    except Exception as e:
                        img_width, img_height = 768, 768
                    img_result = await add_blocks_tool.call(
                        ctx_wrapper, documentId=document_id,
                        parentBlockId=document_id, index=current_index,
                        blocks=[{"blockType": "image", "options": {"image": {"width": img_width, "height": img_height}}}]
                    )
                    image_block_id = None
                    try:
                        if hasattr(img_result, 'content') and img_result.content:
                            r_text = getattr(img_result.content[0], 'text', None) or str(img_result.content[0])
                            r_data = json.loads(r_text)
                            image_info = r_data.get('imageBlocksInfo', {})
                            if isinstance(image_info, dict):
                                block_ids = image_info.get('blockIds', [])
                                if block_ids:
                                    image_block_id = block_ids[0]
                    except Exception as e:
                        logger.error(f"[Test] 解析图片块ID失败: {e}")

                    if image_block_id and upload_image_tool:
                        upload_res = await upload_image_tool.call(
                            ctx_wrapper, documentId=document_id,
                            images=[{"blockId": image_block_id, "imagePathOrUrl": img_path}]
                        )
                        if upload_res and not getattr(upload_res, 'isError', True):
                            images_uploaded += 1
                            logger.info(f"[Test] 图片上传成功，添加caption: '{img_caption}'")
                            # 追加 caption 文本块（同级，不是子块）
                            if img_caption and add_blocks_tool:
                                caption_block = [{
                                    "blockType": "text",
                                    "options": {
                                        "text": {
                                            "textStyles": [{"text": img_caption, "style": {"bold": True, "text_color": 7}}],
                                            "align": 2
                                        }
                                    }
                                }]
                                cap_res = await add_blocks_tool.call(
                                    ctx_wrapper, documentId=document_id,
                                    parentBlockId=document_id,
                                    index=current_index + 1,
                                    blocks=caption_block
                                )
                                if hasattr(cap_res, 'isError') and cap_res.isError:
                                    err = getattr(cap_res.content[0], 'text', str(cap_res))[:200] if hasattr(cap_res, 'content') and cap_res.content else str(cap_res)
                                    logger.error(f"[Test] caption块追加失败: {err}")
                                else:
                                    logger.info(f"[Test] caption块追加成功")
                                    current_index += 1  # caption占一个block位置

                current_index += 1
                batch_start_index = current_index
            else:
                # 记录带样式的列表块（使用原始 content 作为匹配键）
                if b.get("blockType") == "list" and b.get("_textStyles"):
                    list_content = b.get("options", {}).get("list", {}).get("content", "")
                    list_items_to_update.append((list_content, b.get("_textStyles") or {}))
                text_batch.append(b)
                current_index += 1

        await flush_batch()

        # 通过 get_feishu_document_blocks 获取所有块的 ID，按文本内容匹配列表块
        updated_lists = 0
        if list_items_to_update and update_text_tool and get_blocks_tool:
            try:
                blocks_result = await get_blocks_tool.call(ctx_wrapper, documentId=document_id)
                blocks_text = ""
                if hasattr(blocks_result, 'content') and blocks_result.content:
                    blocks_text = getattr(blocks_result.content[0], 'text', '') or str(blocks_result.content[0])

                # 解析 JSON：get_feishu_document_blocks 返回 JSON 数组，后面追加了特殊块提示文本
                # 使用 json.JSONDecoder().raw_decode() 自动忽略尾部内容（找到第一个完整 JSON 数组）
                all_doc_blocks = []
                try:
                    if blocks_text:
                        decoder = json.JSONDecoder()
                        all_doc_blocks, end_pos = decoder.raw_decode(blocks_text)
                        logger.info(f"[Test] JSON 解析成功，{len(all_doc_blocks)} 个块，忽略尾部 {len(blocks_text) - end_pos} 字符")
                except Exception as e:
                    logger.warning(f"[Test] JSON 解析失败: {e}")
                logger.info(f"[Test] 文档共有 {len(all_doc_blocks)} 个块")

                # 匹配：按文本内容找到列表块（空白符归一化后比较）
                def _normalize_text(t: str) -> str:
                    """归一化空白符：将多个连续空白符合并为一个，去除首尾空白"""
                    import re
                    return re.sub(r'\s+', ' ', t).strip()

                updates = []
                matched_block_ids = set()  # 防止重复匹配同一块
                for list_text, text_styles in list_items_to_update:
                    norm_list_text = _normalize_text(list_text)
                    for block in all_doc_blocks:
                        block_id = block.get("block_id", "")
                        if block_id in matched_block_ids:
                            continue
                        block_type = block.get("block_type", 0)
                        # block_type 12=bullet, 13=ordered
                        if block_type not in (12, 13):
                            continue
                        # 从 block 中提取文本内容
                        block_data = block.get("bullet") or block.get("ordered") or {}
                        elements = block_data.get("elements", [])
                        block_text = ""
                        for elem in elements:
                            tr = elem.get("text_run", {})
                            if tr.get("content"):
                                block_text += tr["content"]
                        if _normalize_text(block_text) == norm_list_text:
                            block_id = block.get("block_id", "")
                            matched_block_ids.add(block_id)
                            logger.info(f"[Test] 匹配到列表块: block_id={block_id}, text={block_text[:50]}")
                            # 构建 textElements
                            text_elements = []
                            for ts in text_styles:
                                if ts.get("equation"):
                                    text_elements.append({"equation": ts["equation"], "style": ts.get("style", {})})
                                else:
                                    text_elements.append({"text": ts.get("text", ""), "style": ts.get("style", {})})
                            updates.append({"blockId": block_id, "textElements": text_elements})
                            break

                if updates:
                    logger.info(f"[Test] 更新 {len(updates)} 个列表块样式")
                    upd_result = await update_text_tool.call(
                        ctx_wrapper, documentId=document_id, updates=updates
                    )
                    if hasattr(upd_result, 'isError') and upd_result.isError:
                        err = ""
                        if hasattr(upd_result, 'content') and upd_result.content:
                            err = getattr(upd_result.content[0], 'text', '') or str(upd_result.content[0])
                        logger.error(f"[Test] 列表样式更新失败: {err[:300]}")
                    else:
                        updated_lists = len(updates)
                        logger.info(f"[Test] 列表样式更新成功 ({updated_lists} 个)")
            except Exception as e:
                logger.error(f"[Test] 获取或更新块样式失败: {e}")

        url = f"https://feishu.cn/docx/{document_id}"
        return {
            "success": True,
            "document_id": document_id,
            "url": url,
            "blocks_created": len(all_blocks),
            "image_count": images_uploaded,
            "list_styles_updated": updated_lists,
        }

    async def test_paperbanana_image(self, folder_token: str = "") -> Dict[str, Any]:
        """测试：插入 PaperBanana webp 图片到飞书文档，对比不同尺寸"""
        import glob
        from astrbot.core.agent.run_context import ContextWrapper

        ctx_wrapper = ContextWrapper(self.context)
        provider_manager = getattr(self.context, 'provider_manager', None)
        if not provider_manager:
            return {"success": False, "error": "provider_manager 不可用"}

        llm_tools = getattr(provider_manager, 'llm_tools', None)
        feishu_tool = add_blocks_tool = upload_image_tool = None
        if llm_tools:
            for tool in getattr(llm_tools, 'func_list', []):
                if tool.name == 'create_feishu_document':
                    feishu_tool = tool
                elif tool.name == 'batch_create_feishu_blocks':
                    add_blocks_tool = tool
                elif tool.name == 'upload_and_bind_image_to_block':
                    upload_image_tool = tool

        if not feishu_tool or not add_blocks_tool:
            return {"success": False, "error": "缺少必要工具"}

        # 找最新的 PaperBanana webp 图片
        webp_files = sorted(glob.glob("/var/folders/vh/z2w2jz1x4yb3rfvdzs0nlpjr0000gn/T/*.webp"), key=os.path.getmtime, reverse=True)
        webp_files += sorted(glob.glob("/var/folders/vh/z2w2jz1x4yb3rfvdzs0nlpjr0000gn/T/*.png"), key=os.path.getmtime, reverse=True)
        if not webp_files:
            return {"success": False, "error": "未找到 PaperBanana 图片，请先生成一张"}
        img_path = webp_files[0]

        # webp → PNG 转换（飞书可能对 webp 支持不好）
        from PIL import Image as PILImage
        if img_path.lower().endswith('.webp') or PILImage.open(img_path).format == 'WEBP':
            png_path = img_path.rsplit('.', 1)[0] + '_converted.png'
            PILImage.open(img_path).convert('RGBA' if PILImage.open(img_path).mode == 'RGBA' else 'RGB').save(png_path, 'PNG')
            img_path = png_path
            logger.info(f"[TestPB] WebP 已转为 PNG: {img_path}")

        orig_w, orig_h = PILImage.open(img_path).size
        logger.info(f"[TestPB] 使用图片: {img_path}, 尺寸: {orig_w}x{orig_h}")

        # 创建文档
        create_result = await feishu_tool.call(ctx_wrapper, title="[TestPB] PaperBanana 图片尺寸测试", folderToken=folder_token or "")
        document_id = None
        try:
            if hasattr(create_result, 'content') and create_result.content:
                r_text = getattr(create_result.content[0], 'text', '') or str(create_result.content[0])
                r_data = json.loads(r_text)
                document_id = r_data.get('document', {}).get('document_id', '')
        except Exception as e:
            logger.error(f"[TestPB] 解析文档ID失败: {e}")
        if not document_id:
            return {"success": False, "error": "创建文档失败"}

        # 插入 3 张：原始、5x、50x
        sizes = [
            ("原始", orig_w, orig_h),
            ("5x", orig_w * 5, orig_h * 5),
            ("50x", orig_w * 50, orig_h * 50),
        ]

        for idx, (label, w, h) in enumerate(sizes):
            # 创建空图片块
            img_result = await add_blocks_tool.call(
                ctx_wrapper, documentId=document_id,
                parentBlockId=document_id, index=idx,
                blocks=[{"blockType": "image", "options": {"image": {"width": w, "height": h}}}]
            )
            image_block_id = None
            try:
                if hasattr(img_result, 'content') and img_result.content:
                    r_text = getattr(img_result.content[0], 'text', None) or str(img_result.content[0])
                    r_data = json.loads(r_text)
                    block_ids = r_data.get('imageBlocksInfo', {}).get('blockIds', [])
                    if block_ids:
                        image_block_id = block_ids[0]
            except Exception as e:
                logger.error(f"[TestPB] 解析块ID失败: {e}")

            if image_block_id and upload_image_tool:
                await upload_image_tool.call(
                    ctx_wrapper, documentId=document_id,
                    images=[{"blockId": image_block_id, "imagePathOrUrl": img_path}]
                )
                logger.info(f"[TestPB] 已上传 {label}: {w}x{h}")

        url = f"https://feishu.cn/docx/{document_id}"
        info = f"原始尺寸: {orig_w}x{orig_h}\n对比: 原始 / 2倍 / 10倍\n图片: {os.path.basename(img_path)}"
        return {"success": True, "document_id": document_id, "url": url, "info": info}

    async def create_feishu_document(
        self,
        ideas: List,
        topic: str = "",
        folder_token: str = "",
        knowledge: Optional[Dict[str, Any]] = None,
        table_format: str = "png",
        initial_draft: str = "",
        enable_paper_banana: bool = False
    ) -> Dict[str, Any]:
        """
        创建飞书文档并写入研究想法

        流程：
        1. 使用已有草稿或生成完整周报草稿（VLM）
        2. 提取方法论部分，用本地 VLM 转述为 PaperBanana 图表格式（可选）
        3. 调用 PaperBanana 生成方法图（可选）
        4. 将周报内容和图片插入飞书文档
        """
        try:
            # 1. 使用已有草稿或生成周报草稿
            if initial_draft:
                weekly_report = initial_draft
                logger.info("[IdeaEngine] 使用已有草稿，长度: %d", len(weekly_report))
            else:
                logger.info("[IdeaEngine] 生成完整周报草稿...")
                weekly_report = await self._generate_initial_draft_vlm(ideas, topic, knowledge)
            if not weekly_report:
                return {"error": "周报草稿生成失败", "polished_content": ""}

            # 后处理1：用 caption 匹配替换占位符图片路径为真实路径
            if knowledge:
                local_results = knowledge.get("local_results", [])
                if local_results:
                    weekly_report = self._replace_placeholder_paths_by_caption(weekly_report, local_results)
                    logger.info(f"[IdeaEngine] caption路径替换完成")

            # 2. LLM润色（两阶段：先用本地模型对引用生摘要，再用摘要+草稿润色）
            citations_context = ""
            if knowledge:
                local_results = knowledge.get("local_results", [])
                web_results = knowledge.get("web_results", [])
                if local_results:
                    citations_context += "## 本地论文引用：\n"
                    papers: Dict[str, List] = {}
                    for r in local_results:
                        paper = r.get("paper", "Unknown")
                        if paper not in papers:
                            papers[paper] = []
                        papers[paper].append(r)
                    for paper, chunks in papers.items():
                        citations_context += f"### {paper}\n"
                        for chunk in chunks[:5]:
                            text = chunk.get("text", "")
                            if text:
                                citations_context += f"- {text}\n"
                        citations_context += "\n"
                if web_results:
                    citations_context += "## 网络资源引用：\n"
                    for i, r in enumerate(web_results[:10], 1):
                        title = r.get("title", "")
                        url = r.get("url", "")
                        snippet = r.get("snippet", "")
                        if url:
                            citations_context += f"- [{title}]({url})\n"
                        else:
                            citations_context += f"- {title}\n"
                        if snippet:
                            citations_context += f"  摘要: {snippet}\n"
                    citations_context += "\n"

            # Plan B: 分步处理引用——步骤1生成核心记忆，步骤2用核心记忆润色
            if citations_context and len(citations_context) > 50:
                llm_provider = await self._get_vlm_provider_async()
                if llm_provider:
                    # --- 步骤1：生成核心记忆 ---
                    core_memory = ""
                    try:
                        logger.info(f"[IdeaEngine] 步骤1：生成核心记忆，引用: {len(citations_context)} 字符")
                        memory_prompt = f"""请对以下学术引用资料生成一段简洁的"核心观点记忆"（不超过800字），用于后续润色组会周报。

要求：
- 保留每个论文的：论文名、核心方法/技术路线、关键贡献/结论
- 去掉冗余的实验细节和重复信息
- 用简洁的要点列表组织，每条不超过2句
- 输出格式：直接输出压缩后的核心观点，不要加任何前缀说明

引用资料：
{citations_context}

核心观点记忆："""
                        memory_response = await llm_provider.text_chat(
                            prompt=memory_prompt,
                            contexts=[],
                            temperature=0.2,
                            max_tokens=2048
                        )
                        core_memory = self._extract_text_from_response(memory_response) or ""
                        logger.info(f"[IdeaEngine] 核心记忆生成完成，长度: {len(core_memory)}")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] 核心记忆生成失败: {e}，使用原始引用摘要")
                        core_memory = citations_context[:2000]

                    # --- 步骤2：用核心记忆 + 草稿润色 ---
                    polish_prompt = f"""你是一个学术助手，负责对以下组会周报草稿进行润色和完善。

参考资料（核心记忆）：
{core_memory}

原始草稿：
{weekly_report}

**重要指令**：
- 在原文基础上适当扩展：每个简短的要点/列表项扩展为1-2句连贯段落
- 保持原文的整体结构和章节顺序，只做润色和扩展，不打乱框架
- 充分利用核心记忆中的信息，但不要直接复制，要融会贯通

格式要求：
- 包含章节：背景动机、相关工作、方法论、创新点、实验benchmark、挑战与解决方案、下一步计划、参考文献、论文图表
- **扩展原则**：将简短的要点列表扩展为连贯段落，但不能变成全新的内容
- **列表格式**：创新点和挑战与解决方案部分使用数字序号列表（如"1. 挑战一：xxx"）

**正文引用格式（重要）**：
- 正文中的引用：使用论文简称加markdown链接，如 [FLARE](https://arxiv.org/abs/2502.12138)、[NoPoSplat](https://arxiv.org/abs/2505.23716)
- **禁止在正文中使用论文全名或裸URL**
- **正文及正文中所有涉及引用的地方（论文简称如FLARE、方法名称、引用标记如[4][5]等）一律加粗**，不得有的加粗有的不加粗

**参考文献格式（重要，严格遵守）**：
- 放在最后一个章节
- 每行一条，**严格格式**：`1. [**论文全名**](URL)`
- 数字序号列表，全名加粗，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL（如 `URL (URL)` ）、禁止纯文本URL

**图表引用格式（重要，严格遵守）**：
- **禁止在正文中使用任何图片语法**：`!(..)`、`` ![](..) ``、`[图片:...](...)` 一律禁止
- 正文引用图片时只用文字描述，如"如图1所示"、"如图2所示"
- **参考文献章节中禁止出现任何图片路径**，参考文献中的方法图引用一律改为纯文字描述（如"NoPoSplat 方法流程图"），不得出现 /Users/ 等路径
- **不要生成"论文图表"章节**，该章节会在后续流程中自动添加

请直接输出润色后的内容："""

                    try:
                        polish_total = len(core_memory) + len(weekly_report)
                        logger.info(f"[IdeaEngine] 步骤2：润色草稿，核心记忆: {len(core_memory)}, 草稿: {len(weekly_report)}, 总prompt: ~{polish_total}")
                        response = await llm_provider.text_chat(
                            prompt=polish_prompt,
                            contexts=[],
                            temperature=0.3,
                            max_tokens=32768
                        )
                        polished = self._extract_text_from_response(response)
                        if polished and len(polished) > 100:
                            weekly_report = polished
                            logger.info(f"[IdeaEngine] Plan B 润色完成，长度: {len(polished)}")
                        else:
                            logger.warning(f"[IdeaEngine] 润色结果过短，保持原内容")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] Plan B 润色失败: {e}，保持原内容")
                else:
                    logger.info("[IdeaEngine] 无LLM provider，跳过润色")
            else:
                # 无引用上下文时，直接润色草稿
                logger.info("[IdeaEngine] 无引用上下文，直接润色草稿")
                llm_provider = await self._get_vlm_provider_async()
                if llm_provider:
                    simplify_prompt = f"""你是一个学术助手，负责对以下组会周报草稿进行润色和完善。

原始草稿：
{weekly_report}

**重要指令**：
- 在原文基础上适当扩展：每个简短的要点/列表项扩展为1-2句连贯段落
- 保持原文的整体结构和章节顺序，只做润色和扩展，不打乱框架

格式要求：
- 包含章节：背景动机、相关工作、方法论、创新点、实验benchmark、挑战与解决方案、下一步计划、参考文献
- **扩展原则**：将简短的要点列表扩展为连贯段落，但不能变成全新的内容
- **列表格式**：创新点和挑战与解决方案部分使用数字序号列表（如"1. 挑战一：xxx"）

**正文引用格式（重要）**：
- 正文中的引用：使用论文简称加markdown链接，如 [FLARE](https://arxiv.org/abs/2502.12138)、[NoPoSplat](https://arxiv.org/abs/2505.23716)
- **禁止在正文中使用论文全名或裸URL**
- **正文及正文中所有涉及引用的地方（论文简称如FLARE、方法名称、引用标记如[4][5]等）一律加粗**，不得有的加粗有的不加粗

**参考文献格式（重要，严格遵守）**：
- 放在最后一个章节
- 每行一条，**严格格式**：`1. [**论文全名**](URL)`
- 数字序号列表，全名加粗，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL（如 `URL (URL)` ）、禁止纯文本URL

**图表引用格式（重要，严格遵守）**：
- **禁止在正文中使用任何图片语法**：`!(..)`、`` ![](..) ``、`[图片:...](...)` 一律禁止
- 正文引用图片时只用文字描述，如"如图1所示"、"如图2所示"
- **参考文献章节中禁止出现任何图片路径**，参考文献中的方法图引用一律改为纯文字描述（如"NoPoSplat 方法流程图"），不得出现 /Users/ 等路径
- **不要生成"论文图表"章节**，该章节会在后续流程中自动添加

请直接输出润色后的内容："""

                    try:
                        logger.info(f"[IdeaEngine] 直接润色草稿，原始长度: {len(weekly_report)}")
                        response = await llm_provider.text_chat(
                            prompt=simplify_prompt,
                            contexts=[],
                            temperature=0.3,
                            max_tokens=32768
                        )
                        polished = self._extract_text_from_response(response)
                        if polished and len(polished) > 100:
                            weekly_report = polished
                            logger.info(f"[IdeaEngine] 直接润色完成，长度: {len(polished)}")
                        else:
                            logger.warning(f"[IdeaEngine] 润色结果过短，保持原内容")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] 润色失败: {e}，保持原内容")
                else:
                    logger.info("[IdeaEngine] 无LLM provider，跳过润色")

            # 手动追加论文图表章节（用真实路径，不依赖LLM生成）
            weekly_report = self._append_figure_section(weekly_report, knowledge)

            # --- PaperBanana 方法图生成（可选）---
            figure_blocks = []
            if enable_paper_banana:
                # 3. 提取方法论章节内容
                methodology_text = self._extract_methodology_section(weekly_report)
                logger.info(f"[IdeaEngine] 方法论章节长度: {len(methodology_text)}")
                if len(methodology_text) < 50:
                    logger.warning("[IdeaEngine] 方法论章节过短，PaperBanana 图表可能质量不佳")

                # 4. 尝试从 captions 目录加载 caption，若无则用 VLM 生成
                paper_caption = self._load_caption_for_paper(topic)
                if not paper_caption:
                    paper_caption = await self._generate_caption_with_vlm(topic, methodology_text)
                logger.info(f"[IdeaEngine] PaperBanana caption: {paper_caption[:50] if paper_caption else 'None'}...")

                # 5. 用本地 VLM 将方法论转述为 PaperBanana 图表格式
                paperbanana_format_text = ""
                if methodology_text:
                    paperbanana_format_text = await self._refactor_for_paperbanana(methodology_text, topic)
                    logger.info(f"[IdeaEngine] PaperBanana 格式转述完成，长度: {len(paperbanana_format_text)}")

                # 6. 调用 PaperBanana 生成方法图
                if paperbanana_format_text:
                    logger.info("[IdeaEngine] 正在生成方法图（PaperBanana）...")
                    figure_blocks = await self._generate_method_figures_with_paperbanana_from_text(
                        paperbanana_format_text, topic, caption=paper_caption
                    )
                    logger.info(f"[IdeaEngine] PaperBanana 生成完成，共 {len(figure_blocks)} 张方法图")
            else:
                logger.info("[IdeaEngine] PaperBanana 未启用，跳过方法图生成")

            # 6. 使用本地 VLM 生成简洁标题
            generated_title = topic
            llm_provider = await self._get_vlm_provider_async()
            if llm_provider:
                title_prompt = f"""给定以下研究主题，请为飞书文档生成一个简洁、有意义、学术风格的标题。

研究主题：{topic}

要求：
1. 标题应该反映研究的核心内容，不要直接使用原始问题
2. 标题长度适中（5-15个字）
3. 可以包含 emoji 作为装饰
4. 直接输出标题，不要加任何说明

例如：
- 如果主题是"大模型在代码生成中的应用"，可以生成："🚀 代码生成新范式：大模型赋能编程"
- 如果主题是"多模态大模型研究"，可以生成："🔍 多模态大模型研究进展"

请直接输出标题："""
                try:
                    title_response = await llm_provider.text_chat(
                        prompt=title_prompt,
                        contexts=[],
                        temperature=0.7,
                        max_tokens=256
                    )
                    generated_title = self._extract_text_from_response(title_response)
                    generated_title = generated_title.strip() if generated_title else topic
                    logger.info(f"[IdeaEngine] LLM生成标题: {generated_title}")
                except Exception as e:
                    logger.warning(f"[IdeaEngine] 生成标题失败: {e}，使用原始主题")
                    generated_title = topic

            # 5. 获取飞书工具
            feishu_tool = self._get_feishu_tool()
            if not feishu_tool:
                return {"error": "未找到飞书 MCP 工具，请确认飞书 MCP 已配置并启用", "polished_content": weekly_report}

            from astrbot.core.agent.run_context import ContextWrapper
            ctx_wrapper = ContextWrapper(context=self.context)

            # 6. 保存草稿（提前保存，防止飞书API失败丢失）
            try:
                folder_hash = self._topic_hash(topic)
                draft_file = self._get_ideas_dir() / folder_hash / "initial_draft.md"
                draft_file.parent.mkdir(parents=True, exist_ok=True)
                with open(draft_file, "w", encoding="utf-8") as f:
                    f.write(weekly_report)
                logger.info(f"[IdeaEngine] 草稿已提前保存: {draft_file}")
            except Exception as e:
                logger.warning(f"[IdeaEngine] 提前保存草稿失败: {e}")

            # 7. 创建文档
            logger.info(f"[IdeaEngine] 创建飞书文档: {generated_title}, folder_token: {folder_token}")
            create_result = await feishu_tool.call(ctx_wrapper, title=generated_title, folderToken=folder_token)

            # 7. 解析 document_id
            doc_info = {}
            if hasattr(create_result, 'content') and create_result.content:
                result_text = getattr(create_result.content[0], 'text', None) or str(create_result.content[0])
                try:
                    doc_info = json.loads(result_text)
                except json.JSONDecodeError:
                    pass

            document_id = (
                doc_info.get("document", {}).get("document_id")
                or doc_info.get("document_id")
                or doc_info.get("objToken")
                or doc_info.get("obj_token")
            )
            if not document_id:
                return {"error": f"文档创建失败: {create_result}", "polished_content": weekly_report}

            logger.info(f"[IdeaEngine] 文档创建成功: {document_id}")

            # 8. 获取根块 ID 并插入内容（使用 document_id 作为 parentBlockId 用于根级操作）
            root_block_id = document_id  # 根据 Feishu API，根级操作使用 document_id 作为 parentBlockId

            # 9. 将周报内容转换为飞书块格式（含行内样式：粗体、斜体、LaTeX公式等）
            provider_manager = getattr(self.context, 'provider_manager', None)
            all_blocks = []
            if weekly_report:
                logger.info(f"[IdeaEngine] 周报内容长度: {len(weekly_report)}, 转换块数量: {len(weekly_report.split(chr(10)))}")
                # 调试日志：显示原始内容末尾（检查是否截断）
                logger.debug(f"[IdeaEngine] 原始内容末尾200字符: '''{weekly_report[-200:]}'''")
                # 前处理：将各种图表格式统一转为标准 markdown 图片语法
                weekly_report = self._normalize_figure_references(weekly_report)
                logger.debug(f"[IdeaEngine] normalize后内容末尾200字符: '''{weekly_report[-200:]}'''")
                # 统计图表引用数量（图1、图2等，用于验证）
                figure_refs = re.findall(r'图\s*\d+', weekly_report)
                logger.info(f"[IdeaEngine] 图表引用数量: {len(figure_refs)}，引用: {figure_refs}")
                # 查找论文图表章节位置
                fig_sec = re.search(r'##.*论文图表', weekly_report)
                if fig_sec:
                    logger.debug(f"[IdeaEngine] 论文图表章节位置: {fig_sec.start()}, 内容: '''{weekly_report[fig_sec.start():fig_sec.start()+100]}'''")
                else:
                    logger.warning("[IdeaEngine] 未找到论文图表章节（标题或有序列表）")

                # 关键修复：在调用 _markdown_to_feishu_blocks 之前，
                # 先把正文（论文图表章节之前）中的 (path) 格式替换为 [path]，
                # 防止 _extract_inline_images 的预处理贪婪正则错误匹配
                if fig_sec:
                    body_text = weekly_report[:fig_sec.start()]
                    fig_section = weekly_report[fig_sec.start():]
                    # 使用平衡括号计数函数替换正文中的 (path) 为 [path]
                    body_text_safe = self._convert_paren_paths_to_markdown(body_text)
                    # 把转换后的 ![image](path) 转为 [image](path) 即图片引用格式（后面不再处理）
                    body_text_safe = re.sub(r'!\[image\]\(([/][^)]+\.(?:png|jpg|jpeg|webp|gif))\)', r'[\1]', body_text_safe)
                    weekly_report = body_text_safe + fig_section
                    logger.info(f"[IdeaEngine] 正文路径替换: 替换前长度={len(body_text)}, 替换后长度={len(body_text_safe)}")
                    logger.debug(f"[IdeaEngine] 正文路径替换后末尾200字符: '''{body_text_safe[-200:]}'''")

                all_blocks = self._markdown_to_feishu_blocks(weekly_report)
                # 统计图片块数量
                image_block_count = sum(1 for b in all_blocks if b.get("blockType") == "image")
                logger.info(f"[IdeaEngine] 转换后的块数量: {len(all_blocks)}，其中图片块: {image_block_count}")
                if figure_refs and image_block_count != len(figure_refs):
                    logger.warning(f"[IdeaEngine] ⚠️ 图表引用数({len(figure_refs)})与图片块数({image_block_count})不匹配")
                # 图片块详情
                for i, b in enumerate(all_blocks):
                    if b.get("blockType") == "image":
                        opts = b.get("options", {}).get("image", {})
                        logger.info(f"[IdeaEngine] 图片块[{i}]: path='{opts.get('image_path', 'N/A')}', caption='{opts.get('caption', 'N/A')}'")

            # 将 PaperBanana 生成的方法图插入到方法论章节末尾
            if figure_blocks:
                method_insert_idx = self._find_methodology_end_index(all_blocks)
                logger.info(f"[IdeaEngine] 将 {len(figure_blocks)} 张方法图插入到索引 {method_insert_idx}")
                for i, fb in enumerate(figure_blocks):
                    all_blocks.insert(method_insert_idx + i, fb)

            if all_blocks:
                add_blocks_tool = None
                upload_image_tool = None
                update_text_tool = None
                get_blocks_tool = None
                if provider_manager:
                    llm_tools = getattr(provider_manager, 'llm_tools', None)
                    if llm_tools:
                        for tool in getattr(llm_tools, 'func_list', []):
                            if tool.name == 'batch_create_feishu_blocks':
                                add_blocks_tool = tool
                            elif tool.name == 'upload_and_bind_image_to_block':
                                upload_image_tool = tool
                            elif tool.name == 'batch_update_feishu_block_text':
                                update_text_tool = tool
                            elif tool.name == 'get_feishu_document_blocks':
                                get_blocks_tool = tool

                if add_blocks_tool:
                    # 交错插入：按原始顺序遍历，文本块批量插入，图片块逐张两步上传
                    images_uploaded = 0
                    current_index = 0
                    text_batch: list = []
                    batch_start_index = 0
                    # 记录列表块: (原始文本内容, _textStyles)
                    list_items_to_update: list[tuple[str, dict]] = []

                    async def _flush_text_batch_async():
                        """异步刷出累积的文本块"""
                        nonlocal text_batch, batch_start_index, get_blocks_tool
                        if not text_batch:
                            return
                        CHUNK_SIZE = 20
                        for chunk_start in range(0, len(text_batch), CHUNK_SIZE):
                            chunk = text_batch[chunk_start:chunk_start + CHUNK_SIZE]
                            chunk_index = batch_start_index + chunk_start
                            result = await add_blocks_tool.call(
                                ctx_wrapper,
                                documentId=document_id,
                                parentBlockId=root_block_id,
                                index=chunk_index,
                                blocks=chunk
                            )
                            if hasattr(result, 'isError') and result.isError:
                                err = getattr(result.content[0], 'text', str(result))[:300] if hasattr(result, 'content') and result.content else str(result)
                                logger.error(f"[IdeaEngine] 文本块插入失败 (chunk {chunk_start}, index={chunk_index}): {err}")
                            else:
                                logger.info(f"[IdeaEngine] 插入 {len(chunk)} 个文本块 (index={chunk_index})")
                        text_batch = []

                    for b in all_blocks:
                        if b.get("blockType") == "image":
                            # 先刷出累积的文本块
                            await _flush_text_batch_async()

                            # 单张图片：创建空块 → 提取 block_id → 上传
                            opts = b.get("options", {}).get("image", {})
                            img_path = opts.get("image_path", "")
                            img_base64 = opts.get("base64", "")
                            if not img_path and img_base64:
                                import tempfile
                                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                                tmp.write(base64.b64decode(img_base64))
                                tmp.close()
                                img_path = tmp.name

                            if img_path and os.path.exists(img_path):
                                img_path = self._ensure_png(img_path)
                                img_width = opts.get("width")
                                img_height = opts.get("height")
                                if not img_width or not img_height:
                                    try:
                                        from PIL import Image as PILImage
                                        with PILImage.open(img_path) as pil_img:
                                            orig_w, orig_h = pil_img.size
                                        img_width, img_height = orig_w, orig_h
                                        logger.info(f"[IdeaEngine] 图片尺寸: {img_path} → {img_width}x{img_height}")
                                    except Exception as e:
                                        img_width, img_height = 768, 768
                                else:
                                    logger.info(f"[IdeaEngine] 图片尺寸 [来自block]: {img_width}x{img_height}")

                                img_result = await add_blocks_tool.call(
                                    ctx_wrapper,
                                    documentId=document_id,
                                    parentBlockId=root_block_id,
                                    index=current_index,
                                    blocks=[{"blockType": "image", "options": {"image": {"width": img_width, "height": img_height}}}]
                                )
                                image_block_id = None
                                try:
                                    if hasattr(img_result, 'content') and img_result.content:
                                        r_text = getattr(img_result.content[0], 'text', None) or str(img_result.content[0])
                                        r_data = json.loads(r_text)
                                        image_info = r_data.get('imageBlocksInfo', {})
                                        if isinstance(image_info, dict):
                                            block_ids = image_info.get('blockIds', [])
                                            if block_ids:
                                                image_block_id = block_ids[0]
                                except Exception as e:
                                    logger.error(f"[IdeaEngine] 解析图片块ID失败: {e}")

                                if image_block_id and upload_image_tool:
                                    upload_res = await upload_image_tool.call(
                                        ctx_wrapper,
                                        documentId=document_id,
                                        images=[{"blockId": image_block_id, "imagePathOrUrl": img_path}]
                                    )
                                    if upload_res and not getattr(upload_res, 'isError', True):
                                        images_uploaded += 1
                                        # 给图片块追加 caption 子块
                                        img_caption = opts.get("caption", "")
                                        if img_caption and add_blocks_tool:
                                            caption_block = [{
                                                "blockType": "text",
                                                "options": {
                                                    "text": {
                                                        "textStyles": [{"text": img_caption, "style": {"bold": True, "text_color": 7}}],
                                                        "align": 2
                                                    }
                                                }
                                            }]
                                            await add_blocks_tool.call(
                                                ctx_wrapper, documentId=document_id,
                                                parentBlockId=document_id,
                                                index=current_index + 1,
                                                blocks=caption_block
                                            )
                                    else:
                                        err_msg = ""
                                        if hasattr(upload_res, 'content') and upload_res.content:
                                            err_msg = getattr(upload_res.content[0], 'text', str(upload_res))[:200]
                                        logger.error(f"[IdeaEngine] 图片上传失败: {err_msg}")
                                else:
                                    if not image_block_id:
                                        logger.error(f"[IdeaEngine] 未获取到图片块ID，跳过上传")
                                    elif not upload_image_tool:
                                        logger.error(f"[IdeaEngine] upload_image_tool 不可用")

                            current_index += 1
                            batch_start_index = current_index
                        else:
                            # 文本块/列表块：累积到 batch，同时记录需要更新样式的列表块
                            if b.get("blockType") == "list" and b.get("_textStyles"):
                                list_content = b.get("options", {}).get("list", {}).get("content", "")
                                list_items_to_update.append((list_content, b.get("_textStyles") or {}))
                            text_batch.append(b)
                            current_index += 1

                    # 刷出剩余的文本块
                    await _flush_text_batch_async()

                    # 通过 get_feishu_document_blocks 获取块 ID，再更新列表样式
                    if list_items_to_update and update_text_tool and get_blocks_tool:
                        try:
                            blocks_result = await get_blocks_tool.call(ctx_wrapper, documentId=document_id)
                            blocks_text = ""
                            if hasattr(blocks_result, 'content') and blocks_result.content:
                                blocks_text = getattr(blocks_result.content[0], 'text', '') or str(blocks_result.content[0])
                            all_doc_blocks = []
                            try:
                                if blocks_text:
                                    decoder = json.JSONDecoder()
                                    all_doc_blocks, end_pos = decoder.raw_decode(blocks_text)
                                    logger.info(f"[IdeaEngine] JSON 解析成功，{len(all_doc_blocks)} 个块，忽略尾部 {len(blocks_text) - end_pos} 字符")
                            except Exception as e:
                                logger.warning(f"[IdeaEngine] JSON 解析失败: {e}")
                            logger.info(f"[IdeaEngine] 获取到 {len(all_doc_blocks)} 个文档块，准备更新 {len(list_items_to_update)} 个列表样式")

                            # 按文本内容匹配列表块
                            # 匹配前定义归一化函数和已匹配块集合
                            def _normalize_text(t: str) -> str:
                                import re
                                return re.sub(r'\s+', ' ', t).strip()

                            updates = []
                            matched_block_ids = set()
                            for list_text, text_styles in list_items_to_update:
                                norm_list_text = _normalize_text(list_text)
                                for block in all_doc_blocks:
                                    block_id = block.get("block_id", "")
                                    if block_id in matched_block_ids:
                                        continue
                                    block_type = block.get("block_type", 0)
                                    if block_type not in (12, 13):
                                        continue
                                    block_data = block.get("bullet") or block.get("ordered") or {}
                                    elements = block_data.get("elements", [])
                                    block_text = ""
                                    for elem in elements:
                                        tr = elem.get("text_run", {})
                                        if tr.get("content"):
                                            block_text += tr["content"]
                                    if _normalize_text(block_text) == norm_list_text:
                                        matched_block_ids.add(block_id)
                                        text_elements = []
                                        for ts in text_styles:
                                            if ts.get("equation"):
                                                text_elements.append({"equation": ts["equation"], "style": ts.get("style", {})})
                                            else:
                                                text_elements.append({"text": ts.get("text", ""), "style": ts.get("style", {})})
                                        updates.append({"blockId": block_id, "textElements": text_elements})
                                        logger.info(f"[IdeaEngine] 匹配列表块: block_id={block_id}, text={block_text[:30]}")
                                        break

                            if updates:
                                logger.info(f"[IdeaEngine] 更新 {len(updates)} 个列表块样式")
                                for i in range(0, len(updates), 50):
                                    batch = updates[i:i + 50]
                                    upd_result = await update_text_tool.call(
                                        ctx_wrapper,
                                        documentId=document_id,
                                        updates=batch
                                    )
                                    if hasattr(upd_result, 'isError') and upd_result.isError:
                                        err = getattr(upd_result.content[0], 'text', str(upd_result))[:300] if hasattr(upd_result, 'content') and upd_result.content else str(upd_result)
                                        logger.error(f"[IdeaEngine] 列表样式更新失败: {err}")
                                    else:
                                        logger.info(f"[IdeaEngine] 列表样式更新成功 ({len(batch)} 个块)")
                        except Exception as e:
                            logger.error(f"[IdeaEngine] 获取或更新块样式失败: {e}")

                    logger.info(f"[IdeaEngine] 文档写入完成: {images_uploaded} 张图片已上传, 总块数: {len(all_blocks)}")
                else:
                    logger.warning("[IdeaEngine] 未找到 batch_create_feishu_blocks 工具")
            else:
                logger.warning("[IdeaEngine] all_blocks 为空，跳过块插入")

            url = f"https://feishu.cn/docx/{document_id}"
            return {
                "success": True,
                "document_id": document_id,
                "url": url,
                "blocks_created": len(all_blocks),
                "polished_content": weekly_report
            }

        except Exception as e:
            logger.error(f"[IdeaEngine] 飞书文档创建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e), "polished_content": ""}

    def _extract_methodology_section(self, text: str) -> str:
        """从周报文本中提取方法论章节内容"""
        lines = text.split("\n")
        in_methodology = False
        methodology_lines = []
        for line in lines:
            # 匹配方法论相关的二级或三级标题
            stripped = line.strip()
            if re.match(r'^#{2,3}\s*(方法论|方法|methodology|Methodology|Method)', stripped, re.IGNORECASE):
                in_methodology = True
                continue
            elif in_methodology and stripped.startswith("#"):
                # 遇到下一个标题，停止
                if re.match(r'^#{1,3}\s', stripped):
                    break
            if in_methodology:
                methodology_lines.append(line)
        if not methodology_lines:
            # 回退：提取前50%的内容作为方法描述
            mid = len(lines) // 2
            methodology_lines = lines[mid:]
        return "\n".join(methodology_lines).strip()

    def _load_caption_for_paper(self, topic: str) -> Optional[str]:
        """
        从 data/captions/ 目录加载论文 caption。

        尝试匹配 topic 对应的 caption 文件，返回第一个 figure 的 caption。

        Args:
            topic: 研究主题（通常包含论文简称）

        Returns:
            Caption 字符串，或 None（未找到时）
        """
        captions_dir = Path(__file__).parent / "data" / "captions"
        if not captions_dir.exists():
            logger.debug(f"[IdeaEngine] Captions 目录不存在: {captions_dir}")
            return None

        # 尝试从 topic 中提取论文 ID（常见格式如 arXiv ID、论文简称）
        # 例如: "2406.02058v2(OpenGaussian)" 或 "OpenGaussian"
        topic_clean = topic.strip()

        # 尝试直接匹配目录中的文件
        for caption_file in captions_dir.glob("*.json"):
            # 检查文件名是否包含 topic 关键词
            filename_lower = caption_file.stem.lower()
            topic_lower = topic_clean.lower()
            # 去掉版本号后缀进行匹配
            topic_base = re.sub(r'v\d+$', '', topic_lower)

            if topic_base in filename_lower or filename_lower.startswith(topic_base.replace(' ', '')):
                try:
                    with open(caption_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # 取第一个 figure 的 caption
                    if data and isinstance(data, dict):
                        first_key = next(iter(data.keys()), None)
                        if first_key and "caption" in data[first_key]:
                            caption = data[first_key]["caption"]
                            logger.info(f"[IdeaEngine] 从 {caption_file.name} 加载 caption: {caption[:50]}...")
                            return caption
                except Exception as e:
                    logger.warning(f"[IdeaEngine] 加载 caption 文件失败 {caption_file}: {e}")
        return None

    async def _generate_caption_with_vlm(self, topic: str, methodology_text: str) -> Optional[str]:
        """用 VLM 根据方法论生成 caption"""
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            return None

        prompt = f"""给定以下研究主题和方法论描述，请为该论文的方法流程图生成一个简洁的中文名称作为 caption。

研究主题：{topic}

方法论摘要：
{methodology_text[:500]}

要求：
1. 用中文，起一个简洁的方法名称（5-15字），如"方法流程"、"前馈式稀疏重建框架"等
2. 不要直接照搬研究主题，要根据方法论的核心步骤和特点来命名
3. 直接输出名称，不要加任何前缀说明

Caption："""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=128
            )
            caption = self._extract_text_from_response(response)
            return caption.strip() if caption else None
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM caption 生成失败: {e}")
            return None

    async def _refactor_for_paperbanana(self, methodology_text: str, topic: str) -> str:
        """用本地 VLM 将方法论文本转述为 PaperBanana 学术图表格式"""
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            logger.warning("[IdeaEngine] 本地 VLM 不可用，跳过 PaperBanana 格式转述")
            return methodology_text

        prompt = f"""将以下方法论内容转述为适合生成科研图表的详细描述文本。

要求：
1. 保持学术严谨风格，包含每个模块/步骤的具体描述
2. 使用 Markdown 格式，以 ## Methodology: [主题] 开头
3. 使用 ### 二级标题划分不同模块（如 Retriever Agent、Planner Agent 等）
4. 在描述中加入 LaTeX 数学公式（$...$）来精确表达算法过程
5. 在每个模块描述后给出该模块在图表中的可视化表示形式（如：框图、流程箭头、表格等）
6. 保持原文中的关键术语和技术细节不变

主题：{topic}

方法论原文：
{methodology_text}

请直接输出转述后的 Markdown 文本，不要添加任何说明："""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.3,
                max_tokens=4096
            )
            result = self._extract_text_from_response(response)
            return result.strip() if result else methodology_text
        except Exception as e:
            logger.warning(f"[IdeaEngine] PaperBanana 格式转述失败: {e}")
            return methodology_text

    async def _call_brightdata_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        通用 Bright Data MCP 工具调用方法

        支持的工具:
        - search_engine: 搜索引擎搜索
        - search_engine_batch: 批量搜索引擎搜索
        - scrape_as_markdown: 抓取单个页面为 Markdown
        - scrape_batch: 批量抓取页面为 Markdown
        - discover: AI 驱动的智能搜索

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            timeout: 超时时间（秒）

        Returns:
            Dict 包含工具执行结果
        """
        try:
            # API Token - 从 mcp_server.json 读取
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            try:
                with open(mcp_config_path, "r", encoding="utf-8") as f:
                    mcp_config = json.load(f)
                api_token = mcp_config.get("mcpServers", {}).get("BrightData", {}).get("env", {}).get("API_TOKEN", "")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                return {"success": False, "error": f"无法读取配置: {e}"}

            if not api_token:
                return {"success": False, "error": "BrightData API Token 未配置"}

            # 启动 Bright Data MCP 服务器
            env = {**os.environ, "API_TOKEN": api_token}
            proc = await asyncio.create_subprocess_exec(
                "npx", "@brightdata/mcp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # 构建请求
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }

            request_str = json.dumps(rpc_request) + "\n"
            logger.info(f"[IdeaEngine] Bright Data MCP 调用: {tool_name}, 参数: {json.dumps(arguments)[:200]}")

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=request_str.encode()),
                    timeout=timeout
                )

                # 关闭进程
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (ProcessLookupError, asyncio.TimeoutError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass

                if stderr:
                    stderr_text = stderr.decode()
                    logger.info(f"[IdeaEngine] Bright Data stderr: {stderr_text[:500]}")

                if stdout:
                    stdout_text = stdout.decode().strip()
                    # discover API 返回流式 JSON-RPC 响应（进度通知 + 最终结果）
                    # 需要遍历所有行找到包含 "result" 的最终响应
                    response = None
                    for line in stdout_text.split('\n'):
                        line = line.strip()
                        if line and line.startswith('{'):
                            try:
                                parsed = json.loads(line)
                                # 跳过 progress 通知，找到包含 result 的最终响应
                                if "result" in parsed:
                                    response = parsed
                                    break
                            except json.JSONDecodeError:
                                continue
                    if response is None:
                        # 尝试整体解析（某些情况下是单行）
                        try:
                            response = json.loads(stdout_text)
                        except json.JSONDecodeError as e:
                            logger.warning(f"[IdeaEngine] JSON 解析失败: {e}, 内容: {stdout_text[:200]}")
                            return {"success": False, "error": f"JSON 解析失败: {e}"}
                    content = response.get("result", {}).get("content", [])
                    logger.info(f"[IdeaEngine] Bright Data MCP 原始响应: response_keys={list(response.keys()) if response else None}, content长度={len(content) if content else 0}")

                    if content and len(content) > 0:
                        text = content[0].get("text", "")
                        logger.info(f"[IdeaEngine] Bright Data MCP text长度={len(text) if text else 0}, text前200: {text[:200] if text else 'empty'}")
                        if text:
                            # 尝试解析为 JSON
                            try:
                                data = json.loads(text)
                                logger.info(f"[IdeaEngine] Bright Data MCP 解析成功, data_keys={list(data.keys()) if isinstance(data, dict) else 'list'}")
                                return {"success": True, "data": data}
                            except json.JSONDecodeError as e:
                                # 返回原始文本（如 Markdown）
                                logger.info(f"[IdeaEngine] Bright Data MCP text非JSON，返回原文")
                                return {"success": True, "data": text}

                    logger.warning(f"[IdeaEngine] Bright Data MCP 无有效content或text为空")
                    return {"success": True, "data": None}

            except asyncio.TimeoutError:
                logger.warning(f"[IdeaEngine] Bright Data MCP 调用超时: {tool_name}")
                return {"success": False, "error": "调用超时"}

        except Exception as e:
            logger.error(f"[IdeaEngine] Bright Data MCP 调用失败: {e}")
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "未知错误"}

    async def _search_web(self, queries: List[str], top_k: int) -> List[Dict]:
        """通过网络搜索获取信息（通过Bright Data MCP discover API）"""
        logger.info(f"[IdeaEngine] _search_web 被调用, queries数量={len(queries)}, top_k={top_k}")
        results = []

        # discover API 可能会返回 400，需要重试
        max_retries = 3
        retry_delay = 5  # 秒

        for attempt in range(max_retries):
            try:
                for i, query in enumerate(queries[:5]):
                    logger.info(f"[IdeaEngine] _search_web: discover 查询 {i+1}: {query[:50]}...")
                    result = await self._call_brightdata_mcp_tool(
                        tool_name="discover",
                        arguments={
                            "query": query,
                            "intent": "Find academic research papers and technical articles",
                            "country": "US",
                            "num_results": min(top_k, 10)
                        },
                        timeout=60  # discover API 轮询可能需要几秒
                    )

                    if result.get("success"):
                        data = result.get("data", {})
                        logger.info(f"[IdeaEngine] _search_web: data类型={type(data).__name__}")
                        if isinstance(data, list) and len(data) > 0:
                            logger.info(f"[IdeaEngine] _search_web: discover结果数量={len(data)}")
                            for item in data:
                                results.append({
                                    "title": item.get("title", ""),
                                    "url": item.get("link", ""),
                                    "snippet": item.get("description", "")
                                })
                            break  # 成功获取结果，跳出查询循环
                        elif isinstance(data, str) and "failed" in data.lower():
                            logger.warning(f"[IdeaEngine] _search_web: discover失败 (尝试 {attempt+1}): {data[:100]}")
                        elif isinstance(data, dict):
                            organic = data.get("organic", [])
                            logger.info(f"[IdeaEngine] _search_web: organic数量={len(organic)}")
                            for item in organic:
                                results.append({
                                    "title": item.get("title", ""),
                                    "url": item.get("link", ""),
                                    "snippet": item.get("description", "")
                                })
                            if organic:
                                break  # 成功获取结果
                    else:
                        logger.warning(f"[IdeaEngine] _search_web: discover调用失败 (尝试 {attempt+1}): {result.get('error')}")

                # 如果成功获取结果，跳出重试循环
                if results:
                    break

                # 没有结果，等待后重试
                if attempt < max_retries - 1:
                    logger.info(f"[IdeaEngine] _search_web: 未获取到结果，{retry_delay}秒后重试...")
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                logger.error(f"[IdeaEngine] _search_web 重试 {attempt+1} 异常: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

        if not results:
            logger.warning(f"[IdeaEngine] _search_web: 所有尝试均未获取到网络结果")

        return results

    async def _search_engine_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量搜索引擎搜索

        Args:
            queries: 查询列表，每个包含 query, engine, geo_location 等

        Returns:
            Dict 包含 success, results 列表或 error
        """
        queries = queries[:5]  # 最多5个
        result = await self._call_brightdata_mcp_tool(
            tool_name="search_engine_batch",
            arguments={"queries": queries}
        )

        if result.get("success"):
            data = result.get("data", {})
            return {
                "success": True,
                "results": data,
                "queries": queries
            }
        return {
            "success": False,
            "error": result.get("error", "批量搜索失败")
        }

    async def _discover_search(
        self,
        query: str,
        intent: str = "",
        country: str = "US",
        num_results: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        AI 驱动的智能搜索

        Args:
            query: 搜索查询
            intent: 搜索意图描述
            country: 国家代码
            num_results: 结果数量

        Returns:
            Dict 包含搜索结果
        """
        result = await self._call_brightdata_mcp_tool(
            tool_name="discover",
            arguments={
                "query": query,
                "intent": intent,
                "country": country,
                "num_results": num_results
            }
        )

        if result.get("success"):
            data = result.get("data", {})
            if isinstance(data, dict):
                return {
                    "success": True,
                    "results": data.get("results", []),
                    "intent": data.get("intent", intent)
                }
            elif isinstance(data, list):
                return {
                    "success": True,
                    "results": data,
                    "intent": intent
                }
        return {
            "success": False,
            "error": result.get("error", "搜索失败")
        }

    async def _scrape_as_markdown(self, url: str) -> Dict[str, Any]:
        """抓取单个页面为 Markdown"""
        result = await self._call_brightdata_mcp_tool(
            tool_name="scrape_as_markdown",
            arguments={"url": url}
        )

        if result.get("success"):
            return {
                "success": True,
                "markdown": result.get("data", ""),
                "url": url
            }
        return {
            "success": False,
            "error": result.get("error", "抓取失败")
        }

    async def _scrape_batch_markdown(self, urls: List[str]) -> Dict[str, Any]:
        """
        批量抓取页面为 Markdown

        Args:
            urls: URL 列表（最多5个）

        Returns:
            Dict 包含 success, results 列表或 error
        """
        urls = urls[:5]  # 最多5个
        result = await self._call_brightdata_mcp_tool(
            tool_name="scrape_batch",
            arguments={"urls": urls}
        )

        if result.get("success"):
            data = result.get("data", "")
            return {
                "success": True,
                "results": data,
                "urls": urls
            }
        return {
            "success": False,
            "error": result.get("error", "批量抓取失败")
        }

    async def test_brightdata_mcp(self, query: str) -> Dict[str, Any]:
        """测试 Bright Data MCP 搜索功能"""
        result = await self._call_brightdata_mcp_tool(
            tool_name="search_engine",
            arguments={
                "query": query,
                "num_results": 5,
                "source": "web"
            }
        )
        if result.get("success"):
            data = result.get("data", {})
            organic = data.get("organic", []) if isinstance(data, dict) else []
            return {
                "success": True,
                "results": organic
            }
        return result

    def _get_ideas_dir(self) -> Path:
        """获取想法存储根目录，不存在则创建"""
        # 向上三级: idea_engine.py -> plugin -> plugins -> data
        ideas_dir = Path(__file__).parent.parent.parent / "plugin_data" / "astrbot_plugin_paperrag" / "ideas"
        ideas_dir.mkdir(parents=True, exist_ok=True)
        return ideas_dir

    def _load_figure_captions(self, image_path: str) -> Dict[str, str]:
        """
        从 captions JSON 加载指定图片的 caption。

        Args:
            image_path: 图片完整路径，如 /.../data/figures/2502.12138v4(nopo)/14-Figure1.png

        Returns:
            Dict: key = 图片文件名 (如 "14-Figure1.png"), value = caption 文本
                空 dict 表示文件不存在或解析失败
        """
        import json
        path = Path(image_path)
        if not path.exists():
            return {}
        # 从路径提取 paper_name: .../figures/{paper_name}/{N-FigureM}.png
        figures_dir = path.parent
        paper_name = figures_dir.name  # 如 "2502.12138v4(nopo)"
        caption_file = figures_dir.parent.parent / "captions" / f"{paper_name}.json"
        if not caption_file.exists():
            return {}
        try:
            with open(caption_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # data: { "14-Figure5": {"caption": "...", "filename": "14-Figure1.png", ...}, ... }
            # 建立 filename -> caption 的反向索引
            fname_to_caption = {}
            for v in data.values():
                fname = v.get("filename", "")
                caption = v.get("caption", "")
                if fname and caption:
                    fname_to_caption[fname] = caption
            return fname_to_caption
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[IdeaEngine] 读取 caption 文件失败: {caption_file} ({e})")
            return {}

    def _topic_folder(self, topic: str) -> Path:
        """获取 topic 对应的文件夹路径（使用 MD5 哈希，跨进程稳定）"""
        return self._get_ideas_dir() / self._topic_hash(topic)

    def _topic_hash(self, topic: str) -> str:
        """计算 topic 对应的 folder hash（MD5 hex 前16位）"""
        return hashlib.md5(topic.encode()).hexdigest()[:16]

    def _get_topic_index(self) -> Dict[str, str]:
        """获取 folder_name → topic 的索引"""
        index_file = self._get_ideas_dir() / "topic_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    logger.warning("[IdeaEngine] topic_index.json 格式错误（非 dict）")
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_topic_index(self, index: Dict[str, str]) -> None:
        """保存 topic → folder_name 索引"""
        index_file = self._get_ideas_dir() / "topic_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def list_all_topics(self) -> List[Dict[str, Any]]:
        """
        列出所有已保存的 topic 及其元信息

        Returns:
            List[Dict]: [{"topic": str, "folder": str, "idea_count": int, "created_at": str}, ...]
        """
        index = self._get_topic_index()
        ideas_dir = self._get_ideas_dir()
        result = []

        for folder_name, topic in index.items():
            folder = ideas_dir / folder_name
            if not folder.exists():
                continue

            # 统计 idea 数量（排除 context.json）
            idea_files = [f for f in folder.glob("*.json") if f.name != "context.json"]
            created_at = ""
            if (folder / "context.json").exists():
                try:
                    with open(folder / "context.json", "r", encoding="utf-8") as f:
                        ctx = json.load(f)
                        if isinstance(ctx, dict):
                            created_at = ctx.get("created_at", "")
                except (json.JSONDecodeError, IOError):
                    pass

            result.append({
                "topic": topic,
                "folder": folder_name,
                "idea_count": len(idea_files),
                "created_at": created_at
            })

        return result

    def _topic_folder_by_hash(self, folder_hash: str) -> Path:
        """根据 folder hash 获取文件夹路径"""
        return self._get_ideas_dir() / folder_hash

    def _get_context_path(self, topic: str) -> Path:
        """获取 topic 文件夹下的 context.json 路径"""
        return self._topic_folder(topic) / "context.json"

    def _save_context(self, topic: str, knowledge: Dict[str, Any]) -> None:
        """保存共享 context 到 topic 文件夹"""
        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        # 详细记录每条 local_result 的 text 长度
        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])
        logger.info(f"[IdeaEngine] _save_context 开始:")
        logger.info(f"  - folder: {folder}")
        logger.info(f"  - local_results 数量: {len(local_results)}")
        logger.info(f"  - web_results 数量: {len(web_results)}")
        for i, lr in enumerate(local_results):
            text = lr.get("text", "")
            paper = lr.get("paper", "?")[:30]
            metadata = lr.get("metadata", {})
            img_path = metadata.get("image_path", "")
            table_path = metadata.get("table_csv_path", "")
            logger.info(f"    local_result[{i}]: paper={paper}, text_len={len(text)}, img={bool(img_path)}, table={bool(table_path)}")

        ctx_data = {
            "topic": topic,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "local_results": local_results,
            "web_results": web_results
        }
        logger.info(f"[IdeaEngine] _save_context: 准备写入 {len(ctx_data['local_results'])} 条 local_results")
        with open(self._get_context_path(topic), "w", encoding="utf-8") as f:
            json.dump(ctx_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[IdeaEngine] _save_context: 写入完成，文件大小={self._get_context_path(topic).stat().st_size} bytes")

    def _load_context(self, topic: str) -> Optional[Dict[str, Any]]:
        """加载共享 context（topic 可能是原始名称或 folder hash）"""
        # 如果 topic 本身是合法的 folder 名，直接使用；否则计算 hash
        folder = self._get_ideas_dir() / topic
        if not folder.exists():
            folder = self._topic_folder(topic)
        ctx_path = folder / "context.json"
        if not ctx_path.exists():
            return None
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"[IdeaEngine] _load_context: local_results={len(data.get('local_results', []))}")
                    return data
                logger.warning(f"[IdeaEngine] context.json 格式错误（非 dict 类型）: {type(data)}")
                return None
        except (json.JSONDecodeError, IOError):
            return None

    def _get_vlm_provider(self):
        """获取本地VLM provider（LlamaCppVLMProvider）"""
        try:
            from .llama_cpp_vlm_provider import (
                get_llama_cpp_vlm_provider,
            )
        except ImportError as e:
            logger.warning(f"[IdeaEngine] 无法导入 LlamaCppVLMProvider: {e}")
            return None

        try:
            vlm_provider = get_llama_cpp_vlm_provider()
            return vlm_provider
        except Exception as e:
            logger.warning(f"[IdeaEngine] 获取 VLM Provider 失败: {e}")
            return None

    async def _get_vlm_provider_async(self):
        """异步获取并初始化本地VLM provider"""
        vlm_provider = self._get_vlm_provider()
        if vlm_provider is None:
            return None

        # 如果未初始化，等待初始化完成
        if not vlm_provider._initialized:
            logger.info("[IdeaEngine] VLM Provider 未初始化，等待初始化...")
            await vlm_provider.initialize()

        return vlm_provider

    async def _vlm_chat_with_progress(self, vlm_provider, prompt: str, temperature: float, max_tokens: int, task_name: str = "VLM生成") -> str:
        """
        带进度提示的VLM调用，在推理过程中每10秒输出一次状态

        Args:
            vlm_provider: VLM provider
            prompt: 提示词
            temperature: 温度
            max_tokens: 最大token数
            task_name: 任务名称（用于日志）

        Returns:
            生成的文本内容
        """
        logger.info(f"[IdeaEngine] {task_name}开始，prompt长度: {len(prompt)}")

        async def progress_logger():
            """后台定时输出进度日志"""
            elapsed = 0
            while True:
                await asyncio.sleep(10)
                elapsed += 10
                logger.info(f"[IdeaEngine] {task_name}进行中，已耗时{elapsed}秒...")

        # 启动进度日志任务
        progress_task = asyncio.create_task(progress_logger())

        try:
            # 执行VLM推理
            response = await vlm_provider.text_chat(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 取消进度日志任务
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

            if hasattr(response, 'content'):
                result = response.content
            elif isinstance(response, dict):
                result = response.get("content", "") or response.get("text", "")
            else:
                result = str(response)

            logger.info(f"[IdeaEngine] {task_name}完成，生成{len(result)}字符")
            return result

        except asyncio.CancelledError:
            # 如果VLM任务被取消，也取消进度日志
            progress_task.cancel()
            raise

    async def _vlm_describe_images_batch(
        self,
        vlm_provider,
        images: List[Dict[str, int | str]],
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> List[Dict[str, Any]]:
        """
        批量为图片生成文字描述（VLM fallback）。

        Args:
            vlm_provider: VLM provider
            images: [{"index": int, "path": str, "filename": str}, ...]
            temperature: 温度
            max_tokens: 每张图最大 token 数

        Returns:
            [{"index": int, "caption": str}, ...]
        """
        if not images:
            return []

        # 构建批量提示词
        image_list = []
        for img in images:
            image_list.append(f"本地图-{img['index']}: {img['filename']}\n  路径: {img['path']}")

        images_section = "\n".join(image_list)
        prompt = f"""你是一个学术图片描述助手。请为以下学术图片生成简短准确的描述文字（1-2句话）。

要求：
1. 直接描述图片内容，不要添加"如图所示"等引导语
2. 使用英文逗号分隔的主要信息描述
3. 不要超过50个字

图片列表：
{images_section}

请按以下JSON格式输出（每张图一行，不要有其他内容）：
{{"index": 1, "caption": "描述文字"}}
{{"index": 2, "caption": "描述文字"}}
"""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                temperature=temperature,
                max_tokens=512 * len(images),
            )
            raw = ""
            if hasattr(response, 'content'):
                raw = response.content
            elif isinstance(response, dict):
                raw = response.get("content", "") or response.get("text", "")
            else:
                raw = str(response)

            # 解析 JSON 行
            results: List[Dict[str, int | str]] = []
            for line in raw.strip().split('\n'):
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    obj = json.loads(line)
                    if "index" in obj and "caption" in obj:
                        results.append({"index": int(obj["index"]), "caption": obj["caption"]})
                except (json.JSONDecodeError, ValueError):
                    continue

            logger.info(f"[IdeaEngine] VLM 生成了 {len(results)} 个图片描述")
            return results

        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 图片描述失败: {e}")
            return []

    async def _filter_figures_by_relevance(
        self,
        local_results: List[Dict[str, Any]],
        relevance_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        图表预过滤：召回 chunk 关联论文中的所有图/表，用 rerank 选取最相关的。

        策略：
        1. 从 local_results 找出关联的论文及其图片类型（Figure/Table）
        2. 对每篇论文，加载其 caption JSON 中所有同类型的图表
        3. 用 rerank 模型对所有图表与 chunk 文本进行重排序
        4. 选取排名靠前的图表返回

        Args:
            local_results: RAG 检索结果列表
            relevance_threshold: 相关性阈值

        Returns:
            List[Dict]: 包含图片路径、真实 caption、rerank 分数的列表
        """
        logger.info(f"[IdeaEngine] 图表预过滤开始，输入 {len(local_results)} 条结果")

        # Step 1: 找出哪些论文关联了哪些类型的图表（Figure/Table）
        paper_figure_types: Dict[str, set] = {}  # paper -> {"Figure", "Table"}
        paper_chunk_texts: Dict[str, str] = {}   # paper -> 拼接的 chunk 文本
        for result in local_results:
            metadata = result.get("metadata", {})
            image_path = metadata.get("image_path", "")
            if not image_path:
                continue
            fname = Path(image_path).name
            # 判断是 Figure 还是 Table
            if fname.startswith(("Figure", "figure")):
                img_type = "Figure"
            elif fname.startswith(("Table", "table")):
                img_type = "Table"
            else:
                continue
            paper = result.get("paper", metadata.get("file_name", "Unknown"))
            if paper.endswith('.pdf'):
                paper = paper[:-4]
            if paper not in paper_figure_types:
                paper_figure_types[paper] = set()
                paper_chunk_texts[paper] = ""
            paper_figure_types[paper].add(img_type)
            chunk_text = result.get("text", "")
            if chunk_text and paper_chunk_texts[paper]:
                paper_chunk_texts[paper] += "\n"
            paper_chunk_texts[paper] += chunk_text

        if not paper_figure_types:
            logger.warning("[IdeaEngine] 没有找到关联的图表，跳过图表过滤")
            return []

        logger.info(f"[IdeaEngine] 找到 {len(paper_figure_types)} 篇关联论文: {list(paper_figure_types.keys())}")

        # Step 2: 加载每篇论文的所有相关类型图表
        all_candidates: List[Dict[str, Any]] = []  # 候选图表列表
        for paper, img_types in paper_figure_types.items():
            # 加载该论文的 caption JSON（自动处理 paper name 与文件名不完全匹配的情况）
            captions_data = self._load_captions_by_paper(paper)
            if not captions_data:
                logger.warning(f"[IdeaEngine] 论文无 caption 数据: {paper}")
                continue
            chunk_text = paper_chunk_texts.get(paper, "")

            # 查找实际文件夹（处理 "2502.12138v4" -> "2502.12138v4(nopo)" 这样的情况）
            media_base = Path(__file__).parent / "data"
            figure_base = media_base / "figures"
            table_base = media_base / "tables"
            actual_folder: Optional[Path] = None
            for base in [figure_base, table_base]:
                if not base.exists():
                    continue
                # 前缀匹配：paper="2502.12138v4" 匹配 folder="2502.12138v4(nopo)"
                for folder in base.iterdir():
                    if folder.is_dir() and folder.name.startswith(paper):
                        actual_folder = folder
                        break
                if actual_folder:
                    break

            for key, info in captions_data.items():
                caption = info.get("caption", "")
                filename = info.get("filename", "")
                page = info.get("page", "")
                # 判断类型
                is_figure = any(t == "Figure" for t in img_types) and "Figure" in key
                is_table = any(t == "Table" for t in img_types) and "Table" in key
                if not (is_figure or is_table):
                    continue
                # 合成完整路径
                img_dir = actual_folder if actual_folder else (figure_base if is_figure else table_base) / paper
                full_path = str(img_dir / filename) if filename else ""
                if not full_path or not Path(full_path).exists():
                    continue
                all_candidates.append({
                    "image_path": full_path,
                    "image_caption": caption or filename,
                    "paper": paper,
                    "page": page,
                    "chunk_text": chunk_text,
                    "result": None,
                    "caption": caption,
                    "filename": filename,
                })

        if not all_candidates:
            logger.warning("[IdeaEngine] 没有找到候选图表")
            return []

        logger.info(f"[IdeaEngine] 共有 {len(all_candidates)} 个候选图表，开始 rerank...")

        # Step 3: 用 rerank 对所有候选图表进行重排序
        query = "相关研究内容：" + "\n".join(paper_chunk_texts.values())
        candidates_for_rerank = [
            {"text": c.get("caption", "") or c.get("filename", ""), "metadata": c, "score": 0.5}
            for c in all_candidates
        ]

        try:
            from llama_index_reranker import rerank_results
            reranked = await rerank_results(
                results=candidates_for_rerank,
                query=query,
                top_k=min(10, len(candidates_for_rerank))
            )
            logger.info(f"[IdeaEngine] rerank 完成，{len(reranked)} 个候选")
        except Exception as e:
            logger.warning(f"[IdeaEngine] rerank 失败: {e}，使用原始顺序")
            reranked = [{"metadata": c, "score": 0.5} for c in all_candidates]

        # Step 4: 整理返回格式（应用 relevance_threshold 过滤）
        filtered_images = []
        for item in reranked:
            score = item.get("score", 0.5)
            if score < relevance_threshold:
                continue
            c = item.get("metadata", {})
            filtered_images.append({
                "image_path": c.get("image_path", ""),
                "image_caption": c.get("image_caption", ""),
                "image_description": c.get("caption", ""),
                "image_score": score,
                "text_score": 0.5,
                "caption_richness": 1.0 if c.get("caption") else 0.3,
                "paper": c.get("paper", ""),
                "page": c.get("page", ""),
                "text": c.get("chunk_text", ""),
                "result": c.get("result"),
            })

        logger.info(f"[IdeaEngine] 图表预过滤完成（threshold={relevance_threshold}），返回 {len(filtered_images)} 张图片")
        return filtered_images

    def _load_captions_by_paper(self, paper_name: str) -> Dict[str, Any]:
        """加载指定论文的所有图表 caption 信息（自动查找匹配的 JSON 文件）。"""
        import json
        captions_dir = Path(__file__).parent / "data" / "captions"
        if not captions_dir.exists():
            logger.warning(f"[IdeaEngine] caption 目录不存在: {captions_dir}")
            return {}

        # 精确匹配优先，否则用前缀匹配（处理 paper name 与文件夹名不完全一致的情况）
        exact_path = captions_dir / f"{paper_name}.json"
        if exact_path.exists():
            caption_file = exact_path
        else:
            # 查找前缀匹配的文件（如 "2502.12138v4" 可能对应 "2502.12138v4(nopo).json"）
            matches = [f for f in captions_dir.iterdir() if f.name.startswith(paper_name) and f.suffix == ".json"]
            if matches:
                caption_file = matches[0]
                logger.debug(f"[IdeaEngine] caption 文件前缀匹配: {paper_name} -> {caption_file.name}")
            else:
                logger.warning(f"[IdeaEngine] 未找到 caption 文件: {paper_name}")
                return {}

        try:
            with open(caption_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"[IdeaEngine] caption JSON 解析失败: {caption_file}, error: {e}")
            return {}
        except OSError as e:
            logger.warning(f"[IdeaEngine] caption 文件读取失败: {caption_file}, error: {e}")
            return {}

    async def _extract_text_from_image(self, vlm_provider, image_path: str) -> str:
        """
        使用 VLM 从图片中提取文字描述并判断图片类型

        Args:
            vlm_provider: VLM provider
            image_path: 图片路径

        Returns:
            str: 图片中的文字描述和类型判断
        """
        prompt = """请仔细阅读这张学术图片，按以下步骤处理：

**第一步：判断图片类型**
这是学术科研场景的图片，请先判断图片属于以下哪种类型：
- 表格（Table）：包含行列数据的表格
- 架构图（Architecture）：网络结构、系统架构、模型框架
- 方法图（Method）：算法流程、技术路线、步骤说明
- 统计分析图（Statistics）：柱状图、折线图、散点图、饼图等

**第二步：提取文字**
提取图片中所有可见的文字内容，包括：
1. 图表标题和副标题
2. 坐标轴标签和刻度
3. 图例说明
4. 公式和符号
5. 表格内容
6. 任何其他可见文字

**输出格式**：
如果图片有文字：
[图片类型] 提取的文字内容

如果图片无文字：
[图片类型] 无文字 - 简要描述图片内容（1-2句话）

请直接输出，不要解释。"""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                image_urls=[image_path],
                temperature=0.1,
                max_tokens=512
            )

            if response and hasattr(response, 'content'):
                return response.content.strip()
            return ""
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 提取文字失败: {image_path}, {e}")
            return ""

    def _fuse_knowledge(
        self,
        local_results: List[Dict]
    ) -> str:
        """将本地知识融合为统一上下文（简化版：只处理local_results）"""
        parts = ["# 收集到的相关知识\n"]

        # 本地论文
        if local_results:
            parts.append("## 本地论文库\n")
            papers: Dict[str, List[Dict[str, Any]]] = {}
            for r in local_results:
                paper = r.get("paper", "Unknown")
                if paper not in papers:
                    papers[paper] = []
                papers[paper].append(r)

            for paper, chunks in papers.items():
                parts.append(f"### {paper}")
                for chunk in chunks[:5]:
                    text = chunk.get("text", "")
                    if text:
                        parts.append(f"- {text}")
                parts.append("")

        return "\n".join(parts)

    def _fuse_knowledge_context(
        self,
        local_results: List[Dict],
        web_results: List[Dict]
    ) -> str:
        """将 local 和 web 结果融合为文本上下文"""
        parts = ["# 收集到的相关知识\n"]

        # 本地论文
        for r in local_results:
            paper = r.get("paper", "")
            text = r.get("text", "")
            if paper or text:
                parts.append(f"[本地文档] {paper}\n{text}")

        # 网页
        for r in web_results:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            if title or snippet:
                parts.append(f"[网页] {title}\n{snippet}\n{url}")

        return "\n\n".join(parts)

    async def regenerate_all(
        self,
        folder_hash: str,
        num_ideas: int = 3,
        idea_focus: str = "all"
    ) -> Tuple[List["ResearchIdea"], str, Dict[str, Any]]:
        """
        根据 folder hash 重新生成所有 ideas 以及初始周报

        Args:
            folder_hash: topic 的 folder hash（16位 MD5）
            num_ideas: 生成想法数量
            idea_focus: 想法聚焦方向

        Returns:
            Tuple[新生成的想法列表, 初始周报草稿, knowledge dict]
        """
        # 加载现有 context
        context_data = self._load_context(folder_hash)
        if not context_data:
            raise ValueError(f"Folder hash '{folder_hash}' 不存在或无 context.json")

        topic = context_data.get("topic", folder_hash)

        # 重建 knowledge dict
        knowledge = {
            "local_results": context_data.get("local_results", []),
            "web_results": context_data.get("web_results", []),
            "fused_context": self._fuse_knowledge_context(
                context_data.get("local_results", []),
                context_data.get("web_results", [])
            )
        }

        # 1. 重新生成所有 ideas（使用 VLM）
        ideas = await self.generate_ideas(
            knowledge_context=knowledge.get("fused_context", ""),
            research_domain=context_data.get("domain", ""),
            num_ideas=num_ideas,
            idea_focus=idea_focus
        )

        if not ideas:
            raise ValueError("Ideas 重新生成失败")

        # 2. 生成初始周报草稿（使用 VLM）
        initial_draft = await self._generate_initial_draft_vlm(ideas, topic, knowledge)

        # 3. 保存新 ideas 到文件（覆盖原有）
        self._regenerate_ideas_save(ideas, topic, knowledge, initial_draft)

        return ideas, initial_draft, knowledge

    def _regenerate_ideas_save(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Dict[str, Any],
        initial_draft: str
    ) -> None:
        """重新生成后保存 ideas 到文件"""
        import uuid as uuid_module

        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        # 保存共享 context
        self._save_context(topic, knowledge)

        # 删除旧的 idea 文件
        for f in folder.glob("*.json"):
            if f.name != "context.json":
                f.unlink()

        # 保存新的 initial_draft
        draft_file = folder / "initial_draft.md"
        with open(draft_file, "w", encoding="utf-8") as f:
            f.write(initial_draft)

        # 保存每个新的 idea
        for idea in ideas:
            idea_uuid = str(uuid_module.uuid4())[:8]
            idea_data = {
                "id": idea_uuid,
                "topic": topic,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "idea": {
                    "title": idea.title,
                    "description": idea.description,
                    "novelty": idea.novelty,
                    "methodology": idea.methodology,
                    "potential_challenges": idea.potential_challenges,
                    "related_work": idea.related_work,
                    "feasibility": idea.feasibility,
                    "inspiration_sources": idea.inspiration_sources
                }
            }
            file_path = folder / f"{idea_uuid}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(idea_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[IdeaEngine] 重新生成的想法已保存: {file_path}")

        # 更新 topic 索引
        index = self._get_topic_index()
        index[folder.name] = topic
        self._save_topic_index(index)

    async def generate_ideas(
        self,
        knowledge_context: str,
        research_domain: str = "",
        num_ideas: int = 3,
        idea_focus: str = "all",
        topic: str = ""
    ) -> List[ResearchIdea]:
        """
        基于知识上下文生成研究想法（VLM生成）

        Args:
            knowledge_context: 融合后的知识上下文
            research_domain: 研究领域
            num_ideas: 生成想法数量
            idea_focus: 侧重点 (novelty/feasibility/impact/all)
            topic: 用户原始研究主题/问题

        Returns:
            List[ResearchIdea]: 研究想法列表
        """
        logger.info(f"[IdeaEngine] 生成{num_ideas}个研究想法，topic={topic}")

        # 优先使用本地VLM生成ideas
        vlm_provider = await self._get_vlm_provider_async()
        if vlm_provider:
            logger.info("[IdeaEngine] 使用本地VLM生成ideas")
        else:
            logger.warning("[IdeaEngine] 本地VLM不可用，将使用云端LLM")

        focus_instruction = {
            "novelty": "特别强调创新性和独特贡献",
            "feasibility": "特别强调技术可行性和实现路径",
            "impact": "特别强调潜在影响力和应用价值",
            "all": "综合考虑创新性、可行性和影响力"
        }.get(idea_focus, "")

        prompt = f"""基于以下收集的知识上下文（包含相关论文的摘要和主要贡献），针对用户的研究主题，生成{num_ideas}个研究想法。

**用户研究主题：{topic}**

收集的知识（请仔细阅读，这些是与主题相关的参考资料，包含论文摘要和方法描述）：
{knowledge_context[:8000]}

{focus_instruction}

**分析现有工作的痛点**：
从上述论文中分析当前领域的主要问题和挑战：
1. 哪些问题还没有被很好地解决？
2. 现有方法的局限性是什么？（精度、速度、泛化能力、计算成本等）
3. 哪些场景或应用仍然困难？

**重要约束**：
- 想法必须与「{topic}」紧密相关，不能偏离主题
- 如果收集的知识中有与主题不相关的内容，请忽略
- 每个想法都要能追溯到上述参考资料中的具体内容
- 不要生成与主题无关的通用性想法
- **必须参考论文摘要的表述风格**，明确说明解决了什么问题

请为每个想法返回以下JSON格式的信息：

{{
    "ideas": [
        {{
            "title": "想法标题（参考摘要风格，明确研究问题）",
            "description": "详细描述该想法针对的具体问题，以及初步的解决思路",
            "novelty": "创新点：明确说明该想法解决了现有工作中的什么问题/痛点",
            "methodology": "方法论建议：具体的技术路线",
            "potential_challenges": ["挑战1", "挑战2"],
            "related_work": ["相关工作1（该想法与哪些论文相关）"],
            "feasibility": 0.0到1.0之间的浮点数,
            "inspiration_sources": ["灵感来源1（参考了哪些论文的具体贡献）"]
        }},
        ...
    ],
    "analysis_summary": "对现有工作的分析总结：当前领域的主要问题和尚未解决的痛点"
}}

请严格按照JSON格式返回，只返回JSON，不要包含其他文字。"""

        logger.info(f"[IdeaEngine] 生成ideas的prompt长度: {len(prompt)}")

        try:
            response_text = ""
            if vlm_provider:
                # 使用本地VLM生成
                logger.info("[IdeaEngine] 调用VLM text_chat...")
                response = await vlm_provider.text_chat(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=2048
                )

                logger.info(f"[IdeaEngine] VLM响应类型: {type(response)}")

                # 提取响应文本
                response_text = ""
                if hasattr(response, 'content'):
                    response_text = response.content
                    logger.info(f"[IdeaEngine] 从response.content提取，长度: {len(response_text)}")
                elif isinstance(response, dict):
                    response_text = response.get("content", "") or response.get("text", "")
                    logger.info(f"[IdeaEngine] 从dict提取，长度: {len(response_text)}")
                else:
                    response_text = str(response)
                    logger.info(f"[IdeaEngine] 强制转str，长度: {len(response_text)}")

                logger.info(f"[IdeaEngine] VLM原始响应前200字符: {response_text[:200]}")
            else:
                # Fallback: 使用云端LLM
                logger.info("[IdeaEngine] 使用云端LLM生成ideas")
                provider = self._get_llm_provider()
                if not provider:
                    logger.error("[IdeaEngine] 云端LLM provider也未初始化")
                    return []

                response = await provider.text_chat(
                    prompt=prompt,
                    contexts=[],
                    temperature=0.7,
                    max_tokens=4096
                )

                if hasattr(response, 'result_chain'):
                    chain = getattr(response.result_chain, 'chain', None)
                    if chain and len(chain) > 0:
                        first = chain[0]
                        if hasattr(first, 'get_text'):
                            response_text = first.get_text()
                        elif hasattr(first, 'text'):
                            response_text = first.text
                elif hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, dict):
                    response_text = response.get("content", "") or response.get("text", "")
                else:
                    response_text = str(response)

                logger.info(f"[IdeaEngine] 云端LLM响应长度: {len(response_text)}")

            logger.info(f"[IdeaEngine] 最终响应长度: {len(response_text)}")

            result = self._parse_json_response(response_text)

            if result and "ideas" in result:
                logger.info(f"[IdeaEngine] JSON解析成功，ideas数量: {len(result['ideas'])}")
                ideas = []
                for item in result["ideas"][:num_ideas]:
                    ideas.append(ResearchIdea(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        novelty=item.get("novelty", ""),
                        methodology=item.get("methodology", ""),
                        potential_challenges=item.get("potential_challenges", []),
                        related_work=item.get("related_work", []),
                        feasibility=item.get("feasibility", 0.5),
                        inspiration_sources=item.get("inspiration_sources", [])
                    ))
                return ideas
            else:
                logger.warning(f"[IdeaEngine] JSON解析失败或无ideas，response前100字符: {response_text[:100]}")
                return []

        except Exception as e:
            logger.error(f"[IdeaEngine] 创意生成失败: {e}")

        return []

    def _get_llm_provider(self):
        """获取LLM provider"""
        if not self.context:
            return None
        # 尝试获取当前正在使用的provider
        provider = getattr(self.context, 'get_using_provider', None)
        if provider:
            return provider()
        # fallback: 尝试通过provider_manager获取
        provider_manager = getattr(self.context, 'provider_manager', None)
        if provider_manager:
            inst_map = getattr(provider_manager, 'inst_map', None)
            if isinstance(inst_map, dict) and inst_map:
                return list(inst_map.values())[0]
        return None

    def _load_paper_urls(self) -> Dict[str, Any]:
        """
        从 milvus_abstracts_doc_stats.json 加载论文完整信息

        Returns:
            Dict[paper_id, paper_info_dict]
        """
        try:
            stats_path = Path(__file__).parent / "data" / "milvus_abstracts_doc_stats.json"
            if not stats_path.exists():
                logger.warning(f"[IdeaEngine] milvus_abstracts_doc_stats.json 不存在: {stats_path}")
                return {}
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            abstracts = stats.get("abstracts", {})
            logger.info(f"[IdeaEngine] 已加载 {len(abstracts)} 个论文信息")
            return abstracts
        except Exception as e:
            logger.warning(f"[IdeaEngine] 加载论文URL失败: {e}")
            return {}

    async def _generate_initial_draft_vlm(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        使用 VLM 生成初始周报草稿

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            knowledge: 知识检索结果

        Returns:
            初始周报草稿字符串
        """
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            logger.warning("[IdeaEngine] VLM 不可用，使用简单格式化")
            return self._format_ideas_as_markdown(ideas, topic)

        # 构建引用上下文（使用过滤后的图片）
        citations_context = ""
        media_instructions = ""

        # 加载论文URL映射
        paper_urls = self._load_paper_urls()
        logger.info(f"[IdeaEngine] [DEBUG] 已加载论文URL映射，数量: {len(paper_urls)}")

        if knowledge:
            local_results = knowledge.get("local_results", [])
            if local_results:
                # 调用图表过滤
                filtered_images = await self._filter_figures_by_relevance(local_results)
                logger.info(f"[IdeaEngine] 图表预过滤完成，保留 {len(filtered_images)} 张相关图片")

                # DEBUG: 打印过滤后的图片信息
                for i, img in enumerate(filtered_images, 1):
                    img_path = img.get('image_path', '')
                    img_caption = img.get('image_caption', '')
                    logger.info(f"[IdeaEngine] [DEBUG] 图片{i}: path={img_path}, caption={img_caption}")

                # 构建本地论文引用上下文（包含正文内容和URL）
                citations_context += "## 本地论文引用：\n"
                papers: Dict[str, List] = {}
                for r in local_results:
                    paper = r.get("paper", "Unknown")
                    if paper not in papers:
                        papers[paper] = []
                    papers[paper].append(r)
                for paper, chunks in papers.items():
                    # 从 milvus_abstracts_doc_stats.json 查找 URL（URL 在 metadata 中）
                    paper_key = paper
                    if paper.endswith('.pdf'):
                        paper_key = paper[:-4]
                    paper_info = paper_urls.get(paper_key, {})
                    metadata = paper_info.get('metadata', {})
                    paper_url = metadata.get('arxiv_url', '') or metadata.get('doi_url', '')
                    if paper_url:
                        citations_context += f"### [{paper}]({paper_url})\n"
                        logger.info(f"[IdeaEngine] [DEBUG] 找到论文URL: {paper} -> {paper_url}")
                    else:
                        citations_context += f"### {paper}\n"
                        logger.info(f"[IdeaEngine] [DEBUG] 未找到论文URL: {paper} (key: {paper_key})")
                    for chunk in chunks[:5]:
                        text = chunk.get("text", "")
                        if text:
                            citations_context += f"- {text[:300]}\n"
                    citations_context += "\n"

                # 添加图片信息（真实路径直接列出，caption 在上，路径在下）
                caption_cache: Dict[str, Dict[str, str]] = {}  # paper_folder -> {filename -> caption}
                media_lines: List[str] = ["\n## 可用图片（必须使用这些真实路径，不要生成新路径）：\n"]
                no_caption_images: List[Dict[str, int | str]] = []  # 供 VLM fallback

                for i, img in enumerate(filtered_images, 1):
                    img_path = img.get('image_path', '')
                    img_filename = Path(img_path).name
                    paper_folder = Path(img_path).parent.name
                    if paper_folder not in caption_cache:
                        caption_cache[paper_folder] = self._load_figure_captions(img_path)
                    fname_to_caption = caption_cache[paper_folder]
                    real_caption = fname_to_caption.get(img_filename, '')
                    if real_caption:
                        img_caption = real_caption
                        logger.info(f"[IdeaEngine] [DEBUG] 图片{i} 使用真实caption: {img_filename} -> {real_caption[:50]}...")
                    else:
                        img_caption = img_filename
                        logger.warning(f"[IdeaEngine] [DEBUG] 图片{i} 无真实caption: {img_filename}")
                        no_caption_images.append({"index": i, "path": img_path, "filename": img_filename})
                    media_lines.append(f"图 {i}：{img_caption}\n{img_path}\n")

                # VLM fallback：批量为无 caption 的图片生成描述
                if no_caption_images and vlm_provider:
                    vlm_descriptions = await self._vlm_describe_images_batch(vlm_provider, no_caption_images)
                    desc_map: Dict[int, str] = {int(desc["index"]): str(desc.get("caption", "")) for desc in vlm_descriptions if "index" in desc}
                    for li, line in enumerate(media_lines):
                        for idx, vlm_cap in desc_map.items():
                            if line.startswith(f"图 {idx}："):
                                # 替换 caption 部分，保留路径
                                parts = line.split('\n', 1)
                                if len(parts) == 2:
                                    media_lines[li] = f"图 {idx}：{vlm_cap}\n{parts[1]}"
                                logger.info(f"[IdeaEngine] [DEBUG] VLM 补充 caption: 图 {idx} -> {vlm_cap[:50]}...")
                                break

                media_instructions = ''.join(media_lines)

            # 添加网络搜索引用
            web_results = knowledge.get("web_results", [])
            if web_results:
                citations_context += "\n## 网络搜索引用：\n"
                for i, r in enumerate(web_results[:5], 1):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "")[:200]
                    if url:
                        citations_context += f"- [{title}]({url})\n"
                    else:
                        citations_context += f"- {title}\n"
                    if snippet:
                        citations_context += f"  摘要: {snippet}...\n"
                citations_context += "\n"

        ideas_summary = self._format_ideas_as_markdown(ideas, topic)

        prompt = f"""基于以下研究想法和参考资料，生成一个详细完整的组会周报。

研究主题：{topic}

研究想法：
{ideas_summary}

参考资料（RAG检索到的chunk，包含丰富信息，请充分利用）：
{citations_context}
{media_instructions}

请生成一个详细完整的组会周报，包含以下章节，每个章节都要有详细展开：
1. 背景动机：详细说明问题的背景、重要性、现有方法的不足（5-8句）
2. 相关工作：详细综述相关方法和论文，引用论文的具体贡献（5-8句）
3. 方法论：详细描述方法细节、工作流程、技术路线（5-10句）
4. 创新点：明确列出2-3个具体创新点，并解释为什么这些创新有效（5-8句）
5. 实验benchmark：详细说明实验设置、数据集、对比方法、评价指标（5-8句）
6. 挑战与解决方案：每个挑战都要详细说明原因和对应的具体解决方案（5-8句）
7. 下一步计划：具体的下一步研究方向和可行的改进思路（3-5句）
8. 参考文献：列出所有引用的论文和网页资源，**严格格式**：
`1. [**论文全名**](URL)`
- 数字序号列表，论文全名加粗，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL（如 `URL (URL)` ）、禁止纯文本URL、禁止不使用markdown链接

**重要**：
1. 参考资料中包含丰富的细节信息，请充分利用这些信息生成详细内容，不要简略！
2. **图表引用（核心规则，必须严格遵守）**：
   - **禁止在正文/方法论中使用 markdown 图片语法**，`![...](...)` 一律禁止出现！
   - 正文引用图片时，只用文字描述，如"如图1所示"、"如图2的实验结果"
   - **参考文献章节（8. 参考文献）中绝对禁止出现任何图片路径**，参考文献中如果需要引用方法图，只写纯文字如"NoPoSplat 方法流程图"，不得出现 /Users/ 或任何 .png .jpg 路径
   - **所有图片必须统一放在最后一个章节（9. 论文图表）**，放在参考文献之后，**每个图片占两行**（第一行是图号和caption，第二行是图片真实绝对路径），格式如下：

```
图 1 方法流程
/Users/xxx/data/figures/xxx/fig1.png
图 2 实验结果
/Users/xxx/data/figures/xxx/fig2.png
```

   - **必须使用可用图片中的真实路径**，直接复制粘贴，不要修改、不要生成新路径

   - 根据"可用图片"中提供的路径和caption，按上述格式填写
   - **序号必须连续**：图1、图2、图3...
   - 示例：`如图1所示，NoPoSplat在稀疏视图下展现出高质量的深度估计能力`（正文引用，不带图片语法）
3. **只有真正相关的图片才引用**，如果内容与某张图片无关，不要引用
4. **引用网络资源**：在相关工作章节中，如果某些方法或观点来自网络搜索结果，请使用 `[标题](URL)` 格式引用
5. **参考文献必须完整**：在"参考文献"章节中，**严格格式** `1. [**论文全名**](URL)` 列出所有本地论文和网络资源，**禁止裸URL或括号重复URL**
"""

        try:
            logger.info("[IdeaEngine] 使用 VLM 生成详细初始周报草稿...")
            draft = await self._vlm_chat_with_progress(
                vlm_provider,
                prompt=prompt,
                temperature=0.7,
                max_tokens=8192,
                task_name="VLM生成初始周报草稿"
            )
            return draft
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 生成失败: {e}，使用简单格式化")
            return ideas_summary

    async def to_feishu_markdown(
        self,
        ideas: List[ResearchIdea],
        topic: str = "",
        include_sources: bool = True,
        initial_draft: str = ""
    ) -> str:
        """
        将研究想法格式化为飞书文档兼容的Markdown格式（带VLM润色）

        流程：
        1. 如果有 initial_draft，使用它作为内容；否则从 ideas 生成本地格式化草稿
        2. VLM 润色内容（结构、格式、语言）
        3. 返回飞书兼容的Markdown格式

        格式规范：
        - 标题层级：# 一级 > ## 二级 > ### 三级
        - 列表格式：使用 - 或 1. ，保持一致性
        - 图片引用：使用 [图X] 格式
        - 公式格式：使用 $公式$ 行内公式
        - 飞书兼容：不使用复杂表格语法

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            include_sources: 是否包含灵感来源
            initial_draft: 预生成的周报草稿（来自 _generate_initial_draft_vlm）

        Returns:
            str: 飞书兼容的Markdown格式内容
        """
        if not ideas and not initial_draft:
            return ""

        # Step 1: 确定要润色的内容
        if initial_draft:
            # 使用预生成的草稿
            content_to_polish = initial_draft
        else:
            # 从 ideas 生成本地格式化草稿
            markdown_parts = [f"# {topic or '研究想法'}\n" if topic else "# 研究想法\n"]

            for i, idea in enumerate(ideas, 1):
                feasibility_bar = "★" * int(idea.feasibility * 5) + "☆" * (5 - int(idea.feasibility * 5))

                markdown_parts.append(f"## {i}. {idea.title}\n")
                markdown_parts.append(f"**可行性**: {feasibility_bar} ({idea.feasibility:.0%})\n")
                markdown_parts.append(f"\n### 描述\n{idea.description}\n")
                markdown_parts.append(f"\n### 创新点\n{idea.novelty}\n")
                markdown_parts.append(f"\n### 方法论\n{idea.methodology}\n")

                if idea.potential_challenges:
                    markdown_parts.append("\n### 潜在挑战\n")
                    for challenge in idea.potential_challenges:
                        markdown_parts.append(f"- {challenge}\n")

                if idea.related_work:
                    markdown_parts.append("\n### 相关工作\n")
                    for work in idea.related_work:
                        markdown_parts.append(f"- {work}\n")

                if include_sources and idea.inspiration_sources:
                    markdown_parts.append("\n### 灵感来源\n")
                    for source in idea.inspiration_sources:
                        markdown_parts.append(f"- {source}\n")

                markdown_parts.append("\n---\n")

            content_to_polish = "".join(markdown_parts)

        # Step 2: VLM 润色
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            logger.warning("[IdeaEngine] VLM不可用，返回未润色版本")
            return content_to_polish

        polish_prompt = f"""你是一个学术写作润色专家。请对以下研究想法内容进行润色，使其更加专业、流畅、符合学术规范。

要求：
1. 保持原有结构和关键信息
2. 优化语言表达，使其更加专业和准确
3. 改善句子结构，避免冗余
4. 确保格式规范（标题层级、列表符号等）
5. 输出必须是有效的Markdown格式

待润色的内容：
{content_to_polish}

请直接输出润色后的Markdown内容，不要包含其他解释或说明。"""

        try:
            polished_response = await vlm_provider.text_chat(
                prompt=polish_prompt,
                temperature=0.3,
                max_tokens=4096
            )

            if hasattr(polished_response, 'content'):
                polished = polished_response.content.strip()
            elif isinstance(polished_response, dict):
                polished = polished_response.get("content", "") or polished_response.get("text", "")
            else:
                polished = str(polished_response)

            if polished and len(polished) > len(content_to_polish) * 0.5:
                logger.info(f"[IdeaEngine] VLM润色完成，原始长度 {len(content_to_polish)}，润色后 {len(polished)}")
                return polished
            else:
                logger.warning("[IdeaEngine] VLM润色结果异常，返回未润色版本")
                return content_to_polish

        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM润色失败: {e}，返回未润色版本")
            return content_to_polish

    def _format_ideas_as_markdown(self, ideas: List["ResearchIdea"], topic: str) -> str:
        """将研究想法格式化为 Markdown"""
        output = f"# {topic}\n\n"
        for i, idea in enumerate(ideas, 1):
            output += f"## [{i}] {idea.title}\n\n"
            output += f"** novelty: ** {idea.novelty}\n\n"
            output += f"** methodology: ** {idea.methodology}\n\n"
            if idea.potential_challenges:
                output += f"** challenges: ** {', '.join(idea.potential_challenges)}\n\n"
            output += "---\n\n"
        return output

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """解析LLM返回的JSON响应"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON块
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # 尝试提取JSON对象
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"[IdeaEngine] JSON解析失败: {text[:200]}")
        return None

    def _extract_text_from_response(self, response) -> str:
        """从 LLM 响应中提取文本"""
        # 方法1：检查 result_chain（AstrBot 格式）
        if hasattr(response, 'result_chain'):
            chain = getattr(response.result_chain, 'chain', None)
            if chain and len(chain) > 0:
                first = chain[0]
                if hasattr(first, 'get_text'):
                    return first.get_text()
                elif hasattr(first, 'text'):
                    return first.text
        # 方法2：检查 content 属性（LlamaCpp 格式）
        if hasattr(response, 'content'):
            return response.content
        # 方法3：dict 格式
        if isinstance(response, dict):
            return response.get("content", "") or response.get("text", "")
        # 方法4：字符串格式
        return str(response)

    def find_topic_by_folder(self, folder_name: str) -> Optional[str]:
        """根据 folder_name 查找对应的 topic"""
        return self._get_topic_index().get(folder_name)

    def convert_to_research_ideas(self, ideas_list: List[Dict[str, Any]]) -> List["ResearchIdea"]:
        """
        将想法数据列表转换回 ResearchIdea 对象列表

        Args:
            ideas_list: load_ideas_by_uuids 返回的想法列表

        Returns:
            List[ResearchIdea]
        """
        research_ideas = []
        for data in ideas_list:
            if not isinstance(data, dict):
                logger.warning(f"[IdeaEngine] 跳过无效想法数据: {type(data)}")
                continue
            item = data.get("idea", {})
            research_ideas.append(ResearchIdea(
                title=item.get("title", ""),
                description=item.get("description", ""),
                novelty=item.get("novelty", ""),
                methodology=item.get("methodology", ""),
                potential_challenges=item.get("potential_challenges", []),
                related_work=item.get("related_work", []),
                feasibility=item.get("feasibility", 0.5),
                inspiration_sources=item.get("inspiration_sources", [])
            ))
        return research_ideas

    def save_ideas_to_file(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Dict[str, Any]
    ) -> List[Tuple[str, Path]]:
        """
        将多个想法及上下文保存到 topic 文件夹

        目录结构:
        ideas/
          topic_index.json
          <hash(topic)>/
            context.json          # 共享 context
            <uuid1>.json        # 单个 idea
            <uuid2>.json

        Args:
            ideas: 研究想法列表
            topic: 原始 topic
            knowledge: 知识检索结果

        Returns:
            List[Tuple[str, Path]]: [(uuid, 文件路径), ...]
        """
        import uuid as uuid_module

        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        # 详细日志：确认 knowledge 中的数据
        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])
        logger.info(f"[IdeaEngine] save_ideas_to_file 开始保存:")
        logger.info(f"  - topic: {topic}")
        logger.info(f"  - ideas 数量: {len(ideas)}")
        logger.info(f"  - local_results 数量: {len(local_results)}")
        logger.info(f"  - web_results 数量: {len(web_results)}")
        if local_results:
            for i, r in enumerate(local_results):
                text_len = len(r.get("text", ""))
                paper = r.get("paper", "?")[:40]
                logger.info(f"    local_result[{i}]: paper={paper}, text_len={text_len}")
        if web_results:
            for i, r in enumerate(web_results[:3]):
                title = r.get("title", "?")[:40]
                logger.info(f"    web_result[{i}]: title={title}")

        # 保存共享 context
        self._save_context(topic, knowledge)

        # 保存每个 idea 到 topic 文件夹
        results = []
        for idea in ideas:
            idea_uuid = str(uuid_module.uuid4())[:8]
            idea_data = {
                "id": idea_uuid,
                "topic": topic,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "idea": {
                    "title": idea.title,
                    "description": idea.description,
                    "novelty": idea.novelty,
                    "methodology": idea.methodology,
                    "potential_challenges": idea.potential_challenges,
                    "related_work": idea.related_work,
                    "feasibility": idea.feasibility,
                    "inspiration_sources": idea.inspiration_sources
                }
            }
            file_path = folder / f"{idea_uuid}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(idea_data, f, ensure_ascii=False, indent=2)
            results.append((idea_uuid, file_path))
            logger.info(f"[IdeaEngine] 想法已保存: {file_path}")

        # 更新 topic 索引（folder_name → topic）
        index = self._get_topic_index()
        index[folder.name] = topic
        self._save_topic_index(index)

        return results

    def load_ideas_by_topic(
        self, folder_hash: str
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        根据 folder hash 加载该 topic 下所有想法

        Args:
            folder_hash: folder 名称（MD5 hash）

        Returns:
            Tuple[List[想法dict], context dict]
        """
        folder = self._get_ideas_dir() / folder_hash
        if not folder.exists():
            logger.warning(f"[IdeaEngine] 未找到 folder_hash={folder_hash} 的文件夹")
            return [], None

        loaded = []
        for file_path in folder.glob("*.json"):
            if file_path.name == "context.json":
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    loaded.append(data)
                else:
                    logger.warning(f"[IdeaEngine] 想法文件格式错误（非 dict）: {file_path.name}")
            except (json.JSONDecodeError, IOError):
                logger.warning(f"[IdeaEngine] 跳过损坏的想法文件: {file_path.name}")

        # 加载 context
        context_data = self._load_context_by_folder(folder_hash)

        return loaded, context_data

    def _load_context_by_folder(self, folder_hash: str) -> Optional[Dict[str, Any]]:
        """根据 folder_hash 加载 context"""
        folder = self._get_ideas_dir() / folder_hash
        ctx_path = folder / "context.json"
        if not ctx_path.exists():
            return None
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return None
        except (json.JSONDecodeError, IOError):
            return None

    def load_ideas_by_uuids(
        self,
        uuids: List[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        根据 UUID 列表加载想法，同时加载共享 context

        通过扫描 topic_index.json 定位 UUID 所在的文件夹，无需 topic 参数

        Args:
            uuids: UUID 列表，如 ["a1b2c3d4", "e5f6g7h8"]

        Returns:
            Tuple[List[想法dict], context dict]
        """
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()
        loaded = []
        found_topic = None
        found_folder = None

        for folder_name, topic in index.items():
            folder = ideas_dir / folder_name
            if not folder.exists():
                continue
            for uid in uuids:
                file_path = folder / f"{uid}.json"
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            loaded.append(data)
                            if found_topic is None:
                                found_topic = topic
                                found_folder = folder_name
                        else:
                            logger.warning(f"[IdeaEngine] 想法文件格式错误（非 dict）: {uid}")
                    except (json.JSONDecodeError, IOError):
                        logger.warning(f"[IdeaEngine] 跳过损坏的想法文件: {uid}")

        # 加载 context
        context_data = None
        if found_folder:
            context_data = self._load_context_by_folder(found_folder)
            if context_data is None:
                context_data = {}
            context_data["_folder_hash"] = found_folder

        return loaded, context_data

    def delete_ideas_by_uuids(self, uuids: List[str]) -> Tuple[List[str], Optional[str]]:
        """
        根据 UUID 列表删除想法文件

        通过扫描 topic_index 定位 UUID 所在文件夹

        Args:
            uuids: UUID 列表

        Returns:
            Tuple[List[已删除的UUID], 所属topic]
        """
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()
        deleted = []
        found_topic = None

        for folder_name, topic in index.items():
            folder = ideas_dir / folder_name
            if not folder.exists():
                continue
            for uid in uuids:
                file_path = folder / f"{uid}.json"
                if file_path.exists():
                    file_path.unlink()
                    deleted.append(uid)
                    if found_topic is None:
                        found_topic = topic
                    logger.info(f"[IdeaEngine] 已删除想法: {file_path}")

        return deleted, found_topic

    def clear_ideas_by_topic(self, topic: str) -> Tuple[int, Optional[str]]:
        """
        清空指定 topic 下所有想法（保留 folder）

        Args:
            topic: 研究主题或 folder hash

        Returns:
            Tuple[已删除的文件数, 实际topic名称]
        """
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()

        # 判断是 topic 还是 folder hash
        folder_name = topic
        if topic not in index:
            folder_hash = self._topic_hash(topic)
            if folder_hash in index:
                folder_name = folder_hash
            else:
                return 0, None

        folder = ideas_dir / folder_name
        if not folder.exists():
            return 0, None

        # 统计要删除的文件（排除 context.json 和 initial_draft.md）
        json_files = list(folder.glob("*.json"))
        deleted_count = 0
        for f in json_files:
            f.unlink()
            deleted_count += 1
            logger.info(f"[IdeaEngine] 已删除想法文件: {f}")

        actual_topic = index.get(folder_name, folder_name)
        logger.info(f"[IdeaEngine] 已清空 topic「{actual_topic}」，删除 {deleted_count} 个想法文件")

        return deleted_count, actual_topic

    def delete_topic_by_hash(self, topic_or_hash: str) -> Tuple[bool, str, str]:
        """
        完全删除指定 topic（folder + 所有内容），包括 folder 本身

        Args:
            topic_or_hash: topic 名称或 folder hash（16位MD5）

        Returns:
            Tuple[是否成功, topic名称, folder_hash]
        """
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()

        # 判断是 folder hash 还是 topic 名称
        folder_name = topic_or_hash
        actual_topic = topic_or_hash
        found = False

        if topic_or_hash in index:
            # 直接是 folder hash
            actual_topic = index[topic_or_hash]
            folder_name = topic_or_hash
            found = True
        else:
            # 尝试作为 topic 名称查找
            for fh, tp in index.items():
                if tp == topic_or_hash:
                    folder_name = fh
                    actual_topic = tp
                    found = True
                    break

        if not found:
            return False, topic_or_hash, ""

        folder = ideas_dir / folder_name
        if not folder.exists():
            return False, actual_topic, folder_name

        # 删除整个 folder
        shutil.rmtree(folder)
        logger.info(f"[IdeaEngine] 已删除 topic folder: {folder}")

        # 从索引中移除
        if folder_name in index:
            del index[folder_name]
            self._save_topic_index(index)

        return True, actual_topic, folder_name

    def _save_ideas_append(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Dict[str, Any]
    ) -> List[Tuple[str, Path]]:
        """追加保存想法到已有 topic 文件夹（不覆盖已有想法）"""
        import uuid as uuid_module

        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        # 更新 topic 索引（folder_name → topic）
        index = self._get_topic_index()
        index[folder.name] = topic
        self._save_topic_index(index)

        results = []
        for idea in ideas:
            idea_uuid = str(uuid_module.uuid4())[:8]
            idea_data = {
                "id": idea_uuid,
                "topic": topic,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "idea": {
                    "title": idea.title,
                    "description": idea.description,
                    "novelty": idea.novelty,
                    "methodology": idea.methodology,
                    "potential_challenges": idea.potential_challenges,
                    "related_work": idea.related_work,
                    "feasibility": idea.feasibility,
                    "inspiration_sources": idea.inspiration_sources
                }
            }
            file_path = folder / f"{idea_uuid}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(idea_data, f, ensure_ascii=False, indent=2)
            results.append((idea_uuid, file_path))
            logger.info(f"[IdeaEngine] 追加想法已保存: {file_path}")

        return results

    async def add_ideas_to_topic(
        self,
        topic: str,
        num_ideas: int = 3,
        idea_focus: str = "all"
    ) -> Tuple[List["ResearchIdea"], Dict[str, Any]]:
        """
        为已有 topic 追加新想法（复用现有 context）

        Args:
            topic: 已有 topic
            num_ideas: 追加想法数量
            idea_focus: 想法聚焦方向

        Returns:
            Tuple[新生成的想法列表, 现有knowledge dict]
        """
        # 加载现有 context
        context_data = self._load_context(topic)
        if not context_data:
            raise ValueError(f"Topic '{topic}' 不存在，请先运行 /idea gen <topic> 生成想法")

        # 重建 knowledge dict（格式需与 search_knowledge 一致）
        knowledge = {
            "local_results": context_data.get("local_results", []),
            "web_results": context_data.get("web_results", []),
            "fused_context": self._fuse_knowledge_context(
                context_data.get("local_results", []),
                context_data.get("web_results", [])
            )
        }

        # 生成新想法
        ideas = await self.generate_ideas(
            knowledge_context=knowledge.get("fused_context", ""),
            research_domain=context_data.get("domain", ""),
            num_ideas=num_ideas,
            idea_focus=idea_focus
        )

        # 保存新想法（追加到 topic 文件夹）
        self._save_ideas_append(ideas, topic, knowledge)

        return ideas, knowledge

    async def analyze_topic(self, topic: str, depth: str = "standard") -> Optional["TopicAnalysis"]:
        """
        分析研究主题，生成搜索策略（简化版：使用VLM）

        Args:
            topic: 研究话题
            depth: 分析深度 (quick/standard/deep)

        Returns:
            TopicAnalysis: 结构化的主题分析
        """
        logger.info(f"[IdeaEngine] 分析主题: {topic}, 深度: {depth}")

        # 优先使用VLM分析
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            logger.warning("[IdeaEngine] VLM不可用，使用简单topic分析")
            return TopicAnalysis(
                domain="",
                keywords=[topic],
                search_queries=[topic],
                local_rag_queries=[topic],
                exploration_angles=[topic],
                summary=topic
            )

        prompt = f"""分析以下研究主题，生成结构化的信息收集计划：

研究主题：{topic}

请分析并返回以下JSON格式的信息：

{{
    "domain": "研究领域",
    "keywords": ["关键词1", "关键词2", ...],
    "search_queries": ["查询1", "查询2", ...],
    "local_rag_queries": ["本地检索词1", "本地检索词2", ...],
    "exploration_angles": ["角度1", "角度2", ...],
    "summary": "主题摘要"
}}

请严格按照JSON格式返回，不要包含其他文字。"""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1024
            )

            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, dict):
                response_text = response.get("content", "") or response.get("text", "")
            else:
                response_text = str(response)

            result = self._parse_json_response(response_text)

            if result:
                return TopicAnalysis(
                    domain=result.get("domain", ""),
                    keywords=result.get("keywords", []),
                    search_queries=result.get("search_queries", [topic]),
                    local_rag_queries=result.get("local_rag_queries", [topic]),
                    exploration_angles=result.get("exploration_angles", []),
                    summary=result.get("summary", topic)
                )
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM分析失败: {e}，使用简单分析")

        # Fallback: 简单分析
        return TopicAnalysis(
            domain="",
            keywords=[topic],
            search_queries=[topic],
            local_rag_queries=[topic],
            exploration_angles=[topic],
            summary=topic
        )

    async def search_knowledge(
        self,
        queries: List[str],
        local_rag_top_k: int = 5,
        web_top_k: int = 0
    ) -> Dict[str, Any]:
        """
        多源知识检索（支持本地RAG + 网络搜索）

        Args:
            queries: 搜索查询列表
            local_rag_top_k: 本地RAG召回数
            web_top_k: 网络搜索召回数

        Returns:
            Dict包含 local_results, web_results, fused_context
        """
        logger.info(f"[IdeaEngine] search_knowledge: 查询数={len(queries)}, local_k={local_rag_top_k}, web_k={web_top_k}")

        local_results = []
        web_results = []

        # 1. 本地RAG搜索
        if self._rag_engine and local_rag_top_k > 0:
            try:
                for query in queries[:5]:  # 限制查询数
                    result = await self._rag_engine.search(query, mode="retrieve")
                    sources = result.get("sources", [])
                    logger.info(f"[IdeaEngine] search_knowledge: query='{query[:50]}...' 返回 sources 数量: {len(sources)}")
                    for src in sources[:local_rag_top_k]:
                        src_metadata = src.get("metadata", {})
                        local_results.append({
                            "text": src.get("text", ""),
                            "paper": src_metadata.get("file_name", "Unknown"),
                            "page": str(src_metadata.get("page", "")),
                            "score": src.get("score", 0.0),
                            "metadata": {
                                "file_name": src_metadata.get("file_name", "Unknown"),
                                "page": str(src_metadata.get("page", "")),
                                "image_path": src_metadata.get("image_path"),
                                "image_caption": src_metadata.get("image_caption"),
                                "table_csv_path": src_metadata.get("table_csv_path"),
                                "table_png_path": src_metadata.get("table_png_path"),
                                "table_caption": src_metadata.get("table_caption"),
                            }
                        })
                logger.info(f"[IdeaEngine] 本地RAG检索完成，找到 {len(local_results)} 条结果")
            except Exception as e:
                logger.error(f"[IdeaEngine] 本地RAG搜索失败: {e}")

        # 2. 网络搜索（通过Bright Data MCP）
        bright_data_ok = self._check_bright_data_config()
        logger.info(f"[IdeaEngine] 网络搜索条件检查: web_top_k={web_top_k}, bright_data_ok={bright_data_ok}")
        if web_top_k > 0 and bright_data_ok:
            try:
                logger.info(f"[IdeaEngine] 开始网络搜索，查询数: {len(queries)}")
                web_results = await self._search_web(queries, web_top_k)
                logger.info(f"[IdeaEngine] 网络搜索完成，找到 {len(web_results)} 条结果")
            except Exception as e:
                logger.error(f"[IdeaEngine] 网络搜索失败: {e}")

        # 3. 融合上下文
        fused_context = self._fuse_knowledge_context(local_results, web_results)

        logger.info(f"[IdeaEngine] search_knowledge 返回: local_results={len(local_results)}, web_results={len(web_results)}")
        return {
            "local_results": local_results,
            "web_results": web_results,
            "fused_context": fused_context,
            "stats": {
                "web_count": len(web_results),
                "local_count": len(local_results)
            }
        }

    async def _generate_method_figures_with_paperbanana(self, ideas: List) -> List[Dict]:
        """调用 PaperBanana 服务生成方法图，返回飞书图片块（基于 ideas 列表）"""
        blocks = []
        if not ideas:
            return blocks
        for idea in ideas:
            method_text = getattr(idea, 'methodology', '') or ''
            title_text = getattr(idea, 'title', '') or ''
            if not method_text:
                continue
            try:
                image_path = await self._call_paperbanana(
                    method_text=method_text,
                    figure_caption=title_text
                )
                if image_path and os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    blocks.append({
                        "blockType": "image",
                        "options": {
                            "image": {
                                "base64": img_base64,
                                "caption": title_text
                            }
                        }
                    })
                    logger.info(f"[IdeaEngine] 方法图生成成功: {title_text[:30]}")
            except Exception as e:
                logger.warning(f"[IdeaEngine] 方法图生成失败 [{title_text[:20]}]: {e}")
        return blocks

    async def _generate_method_figures_with_paperbanana_from_text(self, method_text: str, topic: str, caption: Optional[str] = None) -> List[Dict]:
        """调用 PaperBanana 服务生成方法图（基于完整方法论文本）"""
        blocks = []
        if not method_text:
            return blocks
        figure_caption = caption or f"Methodology: {topic}"
        try:
            image_path = await self._call_paperbanana(
                method_text=method_text,
                figure_caption=figure_caption
            )
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                # 不设置 width/height，让飞书使用图片原始尺寸（避免单位不一致问题）
                blocks.append({
                    "blockType": "image",
                    "options": {
                        "image": {
                            "base64": img_base64,
                            "caption": figure_caption
                        }
                    }
                })
                logger.info(f"[IdeaEngine] 方法图生成成功: {topic[:30]}")
        except Exception as e:
            logger.warning(f"[IdeaEngine] 方法图生成失败 [{topic[:20]}]: {e}")
        return blocks

    async def _call_paperbanana(
        self,
        method_text: str,
        figure_caption: str = "",
        pipeline_mode: str = "demo_planner_critic",
        aspect_ratio: str = "16:9",
        figure_size: str = "14-17cm",
        max_critic_rounds: int = 1,
        model_name: str = "apiyi/gemini-3.1-pro-preview",
        image_gen_model: str = "apiyi/gemini-3.1-flash-image-preview",
        timeout: int = 120
    ) -> Optional[str]:
        """
        调用本地 PaperBanana 服务（http://127.0.0.1:7860/）生成方法图

        Args:
            method_text: 方法描述文本
            figure_caption: 图片标题
            pipeline_mode: Agent 流程模式
            aspect_ratio: 图片比例
            figure_size: 图片尺寸
            max_critic_rounds: 批评优化轮数
            model_name: 推理模型
            image_gen_model: 图像生成模型
            timeout: 超时秒数

        Returns:
            本地图片路径，失败返回 None
        """
        try:
            from gradio_client import Client
            client = Client("http://127.0.0.1:7860")

            # 先初始化：触发后端加载 apiyi 配置（与测试脚本一致）
            def _apply_keys():
                return client.predict(or_key="", g_key="", api_name="/apply_keys")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _apply_keys)

            def _call():
                return client.predict(
                    method_text=method_text,
                    caption_text=figure_caption,
                    pipe_mode=pipeline_mode,
                    ret_setting="none",
                    n_cands=1,
                    ar=aspect_ratio,
                    max_rounds=float(max_critic_rounds),
                    m_model=model_name,
                    img_model=image_gen_model,
                    figure_size=figure_size,
                    save_results="No",
                    api_name="/run_generate"
                )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _call)
            # /run_generate 返回 (generated_candidates, value_47, zip_download, status)
            # generated_candidates: list[dict(image=filepath, caption=str|None)]
            if isinstance(result, tuple) and len(result) >= 1:
                candidates = result[0]
                if isinstance(candidates, list) and candidates:
                    first = candidates[0]
                    if isinstance(first, dict) and "image" in first:
                        img_path = first["image"]
                        if os.path.exists(img_path):
                            logger.info(f"[IdeaEngine] PaperBanana 生成成功: {img_path}")
                            return img_path
            return None
        except Exception as e:
            logger.warning(f"[IdeaEngine] PaperBanana 调用失败: {e}")
            return None
