"""
工具函数模块：纯函数工具 + Mistune解析 + BrightData配置检查

所有不依赖 self 的函数均在此作为模块级函数提供。
"""

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import mistune

from astrbot.api import logger

from .base import IdeaEngineBase
from html.parser import HTMLParser
from html import unescape

if TYPE_CHECKING:
    from .datatypes import ResearchIdea


# ==================== 路径/哈希工具 ====================

def topic_hash(topic: str) -> str:
    """计算 topic 对应的 folder hash（MD5 hex 前16位）"""
    return hashlib.md5(topic.encode()).hexdigest()[:16]


# ==================== Markdown 样式处理 ====================

def strip_markdown_style(text: str) -> str:
    """移除 Markdown 样式标记，保留纯文本"""
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text


def strip_outer_markdown_style(text: str) -> str:
    """移除整行文本的外层 Markdown 样式标记"""
    if re.match(r'^(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)$', text):
        return strip_markdown_style(text)
    return text


# ==================== Mistune Markdown 解析 ====================

def create_feishu_markdown() -> mistune.Markdown:
    """创建带自定义插件的 mistune Markdown（LaTeX + 图表引用）"""

    def parse_fig_ref(md, m, state):
        state.append_token({'type': 'fig_ref', 'raw': m.group(0)})
        return m.end()

    def render_fig_ref(renderer, text):
        return f'<strong>{text}</strong>'

    def parse_latex(md, m, state):
        latex_match = m.group('latex')
        if latex_match:
            state.append_token({'type': 'latex', 'raw': latex_match[1:-1]})
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


def parse_html_with_html_parser(html: str) -> List[Dict[str, Any]]:
    """使用 Python 内置 html.parser 解析 HTML，返回 textStyles 格式"""

    class FeishuHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.result: List[Dict[str, Any]] = []
            self.current_text = ""
            self.styles: Dict[str, bool] = {}
            self.link_url: str | None = None
            self._in_eq = False
            self._eq_text = ""

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs) if attrs else {}
            if self.current_text and tag not in ('br', 'img'):
                self.result.append({"text": unescape(self.current_text), "style": dict(self.styles)})
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
                self.result.append({"text": unescape(self.current_text), "style": dict(self.styles)})
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
        parser.result.append({"text": unescape(parser.current_text), "style": dict(parser.styles)})
    merged: List[Dict[str, Any]] = []
    for item in parser.result:
        if merged and merged[-1].get('text') and item.get('text') and merged[-1].get('style') == item.get('style'):
            merged[-1]['text'] += item['text']
        else:
            merged.append(item)
    return merged


def parse_inline_styles(text: str) -> List[Dict[str, Any]]:
    """使用 mistune + html.parser 解析 Markdown 文本，返回飞书 textStyles 格式"""
    if not text:
        return [{"text": "", "style": {}}]
    try:
        md = create_feishu_markdown()
        html = md(text)
        result = parse_html_with_html_parser(cast(str, html))
        if result and any(item.get("text") or item.get("equation") for item in result):
            return result
    except Exception as e:
        logger.warning(f"[IdeaEngine] mistune 解析失败: {e}")
    return [{"text": text, "style": {}}]


# ==================== JSON / LLM 响应解析 ====================

def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM JSON issue)."""
    # Multiple passes to handle nested cases like },}
    for _ in range(3):
        text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def parse_json_response(text: str) -> Optional[Dict]:
    """从文本中解析 JSON（支持 ```json 包裹，容忍尾逗号）"""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    text = _strip_trailing_commas(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def extract_text_from_response(response) -> str:
    """从 LLM 响应中提取文本"""
    if hasattr(response, 'content'):
        return response.content
    if isinstance(response, dict):
        return response.get("content", "") or response.get("text", "")
    return str(response)


# ==================== 知识融合 ====================

def fuse_knowledge(local_results: List[Dict]) -> str:
    """将本地知识融合为统一上下文"""
    parts = ["# 收集到的相关知识\n"]
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
                    parts.append(f"- {text[:300]}")
            parts.append("")
    return "\n".join(parts)


def fuse_knowledge_context(local_results: List[Dict], web_results: List[Dict]) -> str:
    """将 local 和 web 结果融合为文本上下文"""
    parts = ["# 收集到的相关知识\n"]
    for r in local_results:
        paper = r.get("paper", "")
        text = r.get("text", "")
        if paper or text:
            parts.append(f"[本地文档] {paper}\n{text}")
    for r in web_results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        if title or snippet:
            parts.append(f"[网页] {title}\n{snippet}\n{url}")
    return "\n\n".join(parts)


# ==================== 论文 URL ====================

def load_paper_urls() -> Dict[str, Any]:
    """从 milvus_abstracts_doc_stats.json 加载论文完整信息"""
    try:
        stats_path = Path(__file__).parent.parent / "data" / "milvus_abstracts_doc_stats.json"
        if not stats_path.exists():
            return {}
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("abstracts", {})
    except Exception as e:
        logger.warning(f"[IdeaEngine] 加载论文URL失败: {e}")
        return {}


# ==================== 想法格式化（原始格式） ====================

def format_ideas_as_markdown(ideas: List["ResearchIdea"], topic: str) -> str:
    """将研究想法列表格式化为 Markdown（原始格式：novelty/methodology 字段）"""
    output = f"# {topic}\n\n"
    for i, idea in enumerate(ideas, 1):
        output += f"## [{i}] {idea.title}\n\n"
        output += f"** novelty: ** {idea.novelty}\n\n"
        output += f"** methodology: ** {idea.methodology}\n\n"
        if idea.potential_challenges:
            output += f"** challenges: ** {', '.join(idea.potential_challenges)}\n\n"
        output += "---\n\n"
    return output


# ==================== IdeaEngineUtils 类（需要 self 的方法） ====================

def _is_lark_cli_installed() -> bool:
    """检查 lark-cli 是否已安装（供 feishu_doc.py 和 utils.py 共享）。"""
    return shutil.which("lark-cli") is not None


class IdeaEngineUtils(IdeaEngineBase):
    """
    Bright Data 配置检查与 lark-cli 可用性。

    继承链：IdeaEngineBase → IdeaEngineUtils
    运行时属性（来自 IdeaEngineBase via MRO）：
        context        – AstrBot 上下文
        _get_ideas_dir – 获取 ideas 存储目录
    """

    def _check_bright_data_config(self) -> bool:
        """检查 Bright Data MCP 是否已配置"""
        try:
            # idea/utils.py → idea/ → astrbot_plugin_paperrag/ → plugins/ → data/
            data_dir = Path(__file__).resolve().parent.parent.parent.parent
            mcp_config_path = data_dir / "mcp_server.json"
            if not mcp_config_path.exists():
                logger.warning(f"[IdeaEngine] mcp_server.json 不存在 (path={mcp_config_path})，Bright Data 搜索将不可用")
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

    @staticmethod
    def _check_lark_cli(domain: str) -> dict[str, Any]:
        """检测 lark-cli 是否已安装并返回可用状态。

        lark-cli 作为独立 CLI 通过 subprocess 调用，不依赖 MCP 协议。

        Args:
            domain: 业务域名称，如 "doc", "wiki", "calendar", "sheets"

        Returns:
            {"available": bool, "domain": str, "error": str | None}
        """
        if not _is_lark_cli_installed():
            return {
                "available": False,
                "domain": domain,
                "error": "lark-cli not found. Install: npx @larksuite/cli@latest install",
            }
        return {"available": True, "domain": domain, "error": None}

