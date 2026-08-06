"""
引用处理模块 - 学术论文RAG专用

功能：
1. 使用 LLM 解析结构化引用信息（标题、作者、年份、期刊、DOI）
2. 识别正文中的引用标记（[1], [2], [1-3]等）
3. 建立正文章节与参考文献的双向关联
"""

import re
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from astrbot.api import logger
from .paper_link_resolver import PaperLinkResolver


@dataclass
class Reference:
    """结构化引用对象"""
    ref_id: str  # 如 "ref_1", "ref_2"
    raw_text: str  # 原始引用文本
    ref_title: str  # 论文标题
    ref_authors: str  # 作者
    ref_year: Optional[int]  # 年份
    ref_doi: Optional[str]  # DOI
    ref_venue: Optional[str]  # 期刊/会议
    ref_arxiv_url: Optional[str] = None  # arXiv URL（PaperLinkResolver 解析）
    ref_url: Optional[str] = None  # Generic web link fallback (Semantic Scholar, PDF, etc.)
    ref_source_arxiv_id: Optional[str] = None  # arXiv ID（LLM 从引用文本直接提取）
    ref_cited_by: List[str] = field(default_factory=list)  # 正文中引用此文献的位置（chunk索引）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "raw_text": self.raw_text,
            "ref_title": self.ref_title,
            "ref_authors": self.ref_authors,
            "ref_year": self.ref_year,
            "ref_doi": self.ref_doi,
            "ref_arxiv_url": self.ref_arxiv_url,
            "ref_url": self.ref_url,
            "ref_source_arxiv_id": self.ref_source_arxiv_id,
            "ref_venue": self.ref_venue
        }


@dataclass
class CitationInText:
    """文本中的引用标记"""
    ref_ids: List[str]  # 解析出的引用ID列表，如 ["ref_1", "ref_2"]
    position: int  # 在文本中的位置
    raw_text: str  # 原始匹配文本，如 "[1,2]"
    context: str  # 上下文（前后50字符）



# 参考文献部分的常见标题（精确匹配）
REFERENCE_SECTION_KEYWORDS = [
    'references', 'bibliography', 'works cited',
    'reference list', 'literature cited'
]

# 附录参考文献的标题（正文参考文献之后的部分）
APPENDIX_REFERENCE_KEYWORDS = [
    'appendix a. references', 'appendix b. references',
    'appendix c. references', 'appendix d. references',
    'supplementary references', 'additional references',
    'references s1', 'references s2',  # 补充材料中的编号
    's1 references', 's2 references',  # 补充材料另一种格式
]


def _is_reference_section_title(line_lower: str) -> bool:
    """
    判断一行是否是参考文献章节标题（精确匹配）

    避免误匹配如 "Reference Point Feature", "Reference Frame" 等论文章节
    """
    # 去掉首尾空格
    line_lower = line_lower.strip()

    # 处理章节编号前缀：如 "5. REFERENCES" -> "REFERENCES"
    # 匹配模式: 数字 + 标点 + 空格 开头
    line_clean = re.sub(r'^\d+[\.\)]\s+', '', line_lower)

    # 精确匹配（去掉首尾空格后完全相等）
    if line_clean in [kw.lower() for kw in REFERENCE_SECTION_KEYWORDS]:
        return True

    # 附录格式匹配: "Appendix A. References" 或 "References S1"
    for kw in APPENDIX_REFERENCE_KEYWORDS:
        if line_clean == kw.lower() or line_clean.startswith(kw.lower() + ' '):
            return True

    # 通用格式匹配: 参考文献 后面只接受特定标点结尾
    # 例如 "References." 或 "References:" 或 "References\n" 是可以接受的
    # 但 "Reference Point Feature" 不应该匹配
    for kw in ['references', 'bibliography']:
        if line_clean.startswith(kw.lower()) and len(line_clean) > len(kw):
            suffix = line_clean[len(kw):]
            # 后缀只能是: 句号、冒号、或空（行尾）
            if suffix and suffix[0] in '.:':
                return True
            if not suffix:  # 行尾直接结束
                return True

    return False


def _find_ref_section_end(
    lines_text: List[str],
    ref_start: int
) -> int:
    """
    从 ref_start 开始，找到参考文献部分的结束位置

    Args:
        lines_text: 文本行列表
        ref_start: 参考文献开始行索引

    Returns:
        参考文献结束位置（不包含）
    """
    ref_end = len(lines_text)

    for i, line in enumerate(lines_text[ref_start:], start=ref_start):
        stripped = line.strip()

        # ICLR 风格附录标题：单独一行的大写字母（"A"、"B"），或 "B.1" 小节标题。
        # 不会误匹配作者行（"B. Kim" 点号后跟空格+字母而非数字）和编号条目（"[1] ..."）
        if re.match(r'^[A-Z]\s*$', stripped) or re.match(r'^[A-Z]\.[0-9]+', stripped):
            ref_end = i
            break

        # 遇到附录/补充材料/Acknowledgment时截断
        # 匹配以这些关键词开头的行（更灵活，支持变体）
        if re.match(r'^(acknowledgment|appendix|supplementary)', stripped, re.IGNORECASE):
            ref_end = i
            break

    return ref_end


def _find_all_reference_sections(text: str) -> Dict[str, str]:
    """
    找到所有参考文献部分（支持正文+附录参考文献）

    策略：
    1. 检测每个 "References" 或 "Appendix X. References" 标题
    2. 每个标题单独作为一个 section
    3. 按在文本中的顺序处理

    Args:
        text: PDF 原始文本

    Returns:
        Dict[str, str]: section_name -> 参考文献文本
        例如: {"ref_1": "...", "ref_2": "..."}
        section_name 格式: ref_1, ref_2, ... 按顺序编号
    """
    lines_text = text.split('\n')
    sections: Dict[str, str] = {}

    # 查找所有参考文献标题位置
    ref_titles: List[Tuple[int, str]] = []  # (行索引, 标题名称)

    for i, line in enumerate(lines_text):
        line_stripped = line.strip().lower()

        # 使用精确匹配函数判断是否是参考文献标题
        if _is_reference_section_title(line_stripped):
            ref_titles.append((i, line.strip()))

    if not ref_titles:
        return {}

    # 处理每个参考文献部分
    ref_count = 0  # 全局参考文献部分计数器

    for idx, (start_line, title) in enumerate(ref_titles):
        ref_start = start_line + 1

        # 检测行号格式（跳过空行查找第一行非空内容）
        has_line_numbers = False
        for j in range(ref_start, min(ref_start + 10, len(lines_text))):
            first_line = lines_text[j].strip()
            if first_line:
                has_line_numbers = bool(re.match(r'^\[[0-9]+\]\s*\[[0-9]+\]', first_line)) or \
                                  bool(re.match(r'^\[[0-9]+\]\s*[0-9]+\.', first_line))
                break

        # 确定结束位置：使用下一个 section 的开始位置
        if idx + 1 < len(ref_titles):
            ref_end = ref_titles[idx + 1][0]
        else:
            # 最后一个 section，扫描到下一个可能的 section 标题或文本末尾
            ref_end = _find_ref_section_end(lines_text, ref_start)

            # 如果结束位置太靠后，尝试查找下一个 section
            if ref_end >= len(lines_text) - 5:
                for j in range(ref_start + 1, len(lines_text)):
                    line_lower = lines_text[j].strip().lower()
                    for kw in REFERENCE_SECTION_KEYWORDS + APPENDIX_REFERENCE_KEYWORDS:
                        if line_lower == kw or line_lower.startswith(kw + ' '):
                            ref_end = j
                            break
                    if ref_end != len(lines_text):
                        break

        if ref_start >= ref_end:
            continue

        # 清洗并拼接
        def clean_line(line: str) -> str:
            if has_line_numbers:
                cleaned = re.sub(r'^\[[0-9]+\]\s*', '', line)
                return cleaned
            return line

        result_lines = [clean_line(lines_text[i]) for i in range(ref_start, ref_end)]
        result = '\n'.join(result_lines)

        if result.strip():
            ref_count += 1
            section_name = f"ref_{ref_count}"
            sections[section_name] = result
            logger.info(f"📝 提取 [{title}] -> {section_name}: {len(result)} 字符, {ref_end - ref_start} 行")

    return sections


def _find_reference_section(text: str) -> Optional[str]:
    """找到参考文献部分（包含标题前的页码行）"""
    lines_text = text.split('\n')
    ref_start = -1

    # 找到 "References" 标题位置
    for i, line in enumerate(lines_text):
        line_stripped = line.strip().lower()
        for kw in REFERENCE_SECTION_KEYWORDS:
            if line_stripped == kw or line_stripped.startswith(kw + ' '):
                ref_start = i
                break
        if ref_start >= 0:
            break

    if ref_start < 0:
        return None

    # 往前扩展最多2行，包含 [Page X] 等页码标记（但不超出0）
    actual_start = max(0, ref_start - 2)

    # 自动检测是否存在行号
    first_line = lines_text[ref_start].strip()
    has_line_numbers = bool(re.match(r'^\[[0-9]+\]\s*\[[0-9]+\]', first_line)) or \
                       bool(re.match(r'^\[[0-9]+\]\s*[0-9]+\.', first_line))

    def clean_line(line: str) -> str:
        if has_line_numbers:
            return re.sub(r'^\[[0-9]+\]\s*', '', line)
        return line

    # 找到最后一个编号参考文献行
    ref_end = len(lines_text)
    for i, line in enumerate(lines_text[ref_start:], start=ref_start):
        stripped = clean_line(line).strip()
        if stripped.startswith('|') and stripped.count('|') >= 3:
            ref_end = i
            break
        if stripped.startswith('$') or stripped.endswith('$'):
            ref_end = i
            break
        # ICLR 风格附录标题：单独一行的大写字母（"A"、"B"），或 "B.1" 小节标题
        if re.match(r'^[A-Z]\s*$', stripped) or re.match(r'^[A-Z]\.[0-9]+', stripped):
            ref_end = i
            break
        if re.match(r'^(acknowledgment|appendix|supplementary)', stripped, re.IGNORECASE):
            ref_end = i
            break
        has_ref_number = bool(re.match(r'^\[[0-9]+\]', stripped)) or bool(re.match(r'^[0-9]+\.\s+[A-Z]', stripped))
        if has_ref_number:
            ref_end = i + 1

    if actual_start >= ref_end:
        return None

    result_lines = [clean_line(lines_text[i]) for i in range(actual_start, ref_end)]
    result = '\n'.join(result_lines)
    logger.info(f"📝 参考文献提取成功: {len(result)} 字符, {ref_end - actual_start} 行")
    return result



class CitationLinker:
    """
    引用链接器

    识别正文中出现的引用标记，如 [1], [1,2], [1-3], [1, 2, 5]
    以及 author-year 格式如 (Smith, 2020), Smith et al. (2020)
    并与提取的参考文献建立关联
    """

    # 匹配数字引用标记的正则
    CITATION_PATTERN = re.compile(r'\[(\d+(?:[,\-\s]+\d+)*)\]')

    # 匹配 author-year 引用格式的正则
    # 格式:
    #   - Smith, 2020
    #   - Smith (2020)
    #   - Smith et al. 2021 (AAAI格式，无逗号无括号)
    #   - (Smith et al. 2021) 括号包裹
    AUTHOR_YEAR_PATTERN = re.compile(
        r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?)\s*,\s*(\d{4})|'
        r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?)\s+\((\d{4})\)|'
        r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?)\s+(\d{4})\)'
    )

    # 匹配括号内的多引用: (Smith et al. 2021; Chen et al. 2024; Zhang, Liu, and Han 2024)
    MULTI_CITATION_PATTERN = re.compile(r'\(([^)]+)\)')

    # 匹配方括号内的多引用: [Barron et al. 2022; Duckworth et al. 2023]
    BRACKET_CITATION_PATTERN = re.compile(r'\[([^\]]+)\]')

    def find_citations_in_text(self, text: str) -> List[CitationInText]:
        """
        查找文本中所有引用标记

        Args:
            text: 文本内容

        Returns:
            CitationInText列表
        """
        citations = []

        for match in self.CITATION_PATTERN.finditer(text):
            ref_ids = self._parse_ref_ids(match.group(1))
            if ref_ids:
                # 获取上下文
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                citations.append(CitationInText(
                    ref_ids=[f"ref_{rid}" for rid in ref_ids],
                    position=match.start(),
                    raw_text=match.group(),
                    context=context
                ))

        return citations

    def _parse_ref_ids(self, ref_str: str) -> List[int]:
        """
        解析引用字符串为ID列表

        Args:
            ref_str: 如 "1, 2-5, 7"

        Returns:
            如 [1, 2, 3, 4, 5, 7]
        """
        ref_ids = []
        for part in re.split(r'[,\s]+', ref_str):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                try:
                    start, end = part.split('-', 1)
                    ref_ids.extend(range(int(start), int(end) + 1))
                except (ValueError, TypeError):
                    pass
            else:
                try:
                    ref_ids.append(int(part))
                except ValueError:
                    pass

        return ref_ids

    def _extract_first_author_surname(self, authors: str) -> Optional[str]:
        """
        从作者字符串提取第一作者姓氏

        Args:
            authors: 作者字符串，如 "Steven J. Gortler, et al." 或 "S. Karamcheti, et al."

        Returns:
            姓氏或 None
        """
        if not authors:
            return None

        authors = authors.strip()

        # 处理 "et al." 情况 - 取 "et al." 之前的词
        # 使用正则匹配 " et al." 模式，避免匹配到名字中的 "et"（如 Karamcheti）
        et_al_pattern = re.compile(r'\s+et\s+al', re.IGNORECASE)
        match = et_al_pattern.search(authors)
        if match:
            before_et = authors[:match.start()].strip()
            before_et = before_et.rstrip(',').rstrip()
            parts = before_et.split()
            if parts:
                surname = parts[-1].rstrip('.,')
                if surname.lower() not in ['jr', 'sr', 'md', 'phd', 'dr']:
                    return surname
                if len(parts) > 1:
                    surname2 = parts[-2].rstrip('.,')
                    if surname2.lower() not in ['jr', 'sr', 'md', 'phd', 'dr']:
                        return surname2

        # 处理 "and" 分隔 - 取第一个作者
        if ' and ' in authors:
            first_author = authors.split(' and ')[0].strip()
        elif '&' in authors:
            first_author = authors.split('&')[0].strip()
        else:
            first_author = authors

        # 处理逗号分隔 - 取第一部分
        if ',' in first_author:
            first_author = first_author.split(',')[0].strip()

        # 处理缩写格式: "S. Karamcheti" 或 "S Karamcheti"
        parts = first_author.split()
        if len(parts) >= 2:
            first_part = parts[0]
            # 检查第一部分是否是缩写格式: "S." 或 "S"
            if len(first_part) <= 2 and first_part[0].isupper():
                second_part = parts[1].rstrip('.,')
                if second_part.lower() not in ['jr', 'sr', 'md', 'phd', 'dr']:
                    return second_part

        # 否则取最后一个词作为姓氏
        if parts:
            surname = parts[-1].rstrip('.,')
            if surname.lower() in ['jr', 'sr', 'md', 'phd', 'dr']:
                surname = parts[-2].rstrip('.,') if len(parts) > 1 else surname
            return surname

        return None

    def _build_author_year_map(self, references: List[Reference]) -> Dict[str, str]:
        """
        从参考文献构建 author-year -> ref_id 映射

        Args:
            references: Reference列表

        Returns:
            映射字典，key为 "AuthorYear" 格式，value为 ref_id
        """
        author_year_map = {}
        for ref in references:
            authors = ref.ref_authors.strip()
            year = ref.ref_year
            if authors and year:
                # 提取第一作者姓氏
                surname = self._extract_first_author_surname(authors)
                if not surname:
                    continue

                year_str = str(year)
                surname_lower = surname.lower()

                # 基础格式: gortler1996
                key = f"{surname_lower}{year_str}"
                author_year_map[key] = ref.ref_id

                # 带 et al. 格式: gortler et al.1996
                key_et_al = f"{surname_lower} et al.{year_str}"
                author_year_map[key_et_al] = ref.ref_id

                # 处理年份后缀: 2023a, 2023b
                for suffix in ['a', 'b']:
                    author_year_map[f"{surname_lower}{year_str}{suffix}"] = ref.ref_id
                    author_year_map[f"{surname_lower} et al.{year_str}{suffix}"] = ref.ref_id
        return author_year_map

    def _match_author_in_map(self, author: str, year: str, author_year_map: Dict[str, str]) -> Optional[str]:
        """
        尝试将 author-year 匹配到 ref_id

        Args:
            author: 作者名
            year: 年份
            author_year_map: author-year -> ref_id 映射

        Returns:
            ref_id 或 None
        """
        year_str = str(year)

        # 首先尝试直接匹配（author 已经是姓氏）
        author_lower = author.lower().strip()
        variants = [
            f"{author_lower}{year_str}",  # gortler1996
            f"{author_lower} et al.{year_str}",  # gortler et al.1996
            f"{author_lower}et al.{year_str}",  # gortleret al.1996 (无空格)
        ]

        for key in variants:
            if key in author_year_map:
                return author_year_map[key]

        # 如果直接匹配失败，尝试从 author 中提取姓氏（如 "S. Karamcheti" -> "Karamcheti"）
        surname = self._extract_first_author_surname(author)
        if surname and surname.lower() != author_lower:
            surname_lower = surname.lower()
            variants = [
                f"{surname_lower}{year_str}",
                f"{surname_lower} et al.{year_str}",
                f"{surname_lower}et al.{year_str}",
            ]
            for key in variants:
                if key in author_year_map:
                    return author_year_map[key]

        return None

    def find_author_year_citations(self, text: str, author_year_map: Dict[str, str]) -> List[CitationInText]:
        """
        查找文本中的 author-year 引用并映射到 ref_id

        Args:
            text: 文本内容
            author_year_map: author-year -> ref_id 的映射

        Returns:
            CitationInText列表
        """
        citations = []
        seen_positions = set()  # 避免重复

        # 首先处理方括号内的多引用: [Barron et al. 2022; Duckworth et al. 2023]
        for bracket_match in self.BRACKET_CITATION_PATTERN.finditer(text):
            bracket_content = bracket_match.group(1).strip()

            # 检查是否包含分号（多引用）
            if ';' in bracket_content:
                # 分割多个引用
                parts = bracket_content.split(';')
                bracket_start = bracket_match.start()

                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # 尝试解析单个 author-year 引用
                    ref_ids = self._parse_single_author_year(part, author_year_map)
                    if ref_ids:
                        for ref_id in ref_ids:
                            citations.append(CitationInText(
                                ref_ids=[ref_id],
                                position=bracket_start,
                                raw_text=bracket_match.group(),
                                context=text[max(0, bracket_start - 50):min(len(text), bracket_start + len(bracket_match.group()) + 50)]
                            ))
            else:
                # 单个引用，使用原有逻辑
                ref_ids = self._parse_single_author_year(bracket_content, author_year_map)
                if ref_ids:
                    for ref_id in ref_ids:
                        citations.append(CitationInText(
                            ref_ids=[ref_id],
                            position=bracket_match.start(),
                            raw_text=bracket_match.group(),
                            context=text[max(0, bracket_match.start() - 50):min(len(text), bracket_match.end() + 50)]
                        ))

        # 处理括号内的 author-year 引用（原有逻辑）
        for match in self.AUTHOR_YEAR_PATTERN.finditer(text):
            # 获取作者和年份
            if match.group(1) and match.group(2):
                # 格式: Smith, 2020 或 Smith et al., 2020
                author = match.group(1).strip()
                year = match.group(2)
            elif match.group(3) and match.group(4):
                # 格式: Smith (2020) 或 Smith et al. (2020)
                author = match.group(3).strip()
                year = match.group(4)
            elif match.group(5) and match.group(6):
                # 格式: (Smith et al. 2021) - AAAI格式，括号包裹
                author = match.group(5).strip()
                year = match.group(6)
            else:
                continue

            # 避免同一位置重复匹配
            if match.start() in seen_positions:
                continue
            seen_positions.add(match.start())

            # 尝试多种作者名格式匹配
            ref_id = self._match_author_in_map(author, year, author_year_map)

            if ref_id:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                citations.append(CitationInText(
                    ref_ids=[ref_id],
                    position=match.start(),
                    raw_text=match.group(),
                    context=context
                ))

        return citations

    def _parse_single_author_year(self, text: str, author_year_map: Dict[str, str]) -> List[str]:
        """
        解析单个 author-year 引用文本

        Args:
            text: 引用文本，如 "Barron et al. 2022"
            author_year_map: author-year -> ref_id 的映射

        Returns:
            ref_id 列表
        """
        # 改进的正则：支持缩写格式如 "S. Karamcheti et al. 2024"
        # 使用非贪婪匹配避免吃掉年份前的空格
        pattern = re.compile(
            r'([A-Z]\.\s*[A-Z][a-z]+'  # 缩写+全名如 "S. Karamcheti"
            r'|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'  # 多单词作者名
            r')(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?'  # 可选的 et al. 或 and
            r'[\s,]*?'  # 可选逗号/空格（非贪婪）
            r'(?:[\(\[]?\s*)?'  # 可选的括号
            r'(\d{4}[a-z]?)'  # 年份，可选 a/b 后缀
        )
        match = pattern.search(text.strip())
        if match:
            author = match.group(1).strip()
            year = match.group(2)
            ref_id = self._match_author_in_map(author, year, author_year_map)
            if ref_id:
                return [ref_id]

        # Fallback: 尝试通用格式 "Author[ et al.] YEAR"
        # 支持: "Karamcheti 2024", "Karamcheti et al. 2024", "Karamcheti et al., 2024"
        parts = text.strip().split()
        for i in range(len(parts)):
            # 跳过 "et" 和 "al." 词
            if parts[i].lower() in ['et', 'al.', 'al']:
                continue
            # 尝试从第 i 个词开始匹配 Author YEAR
            for j in range(i + 1, len(parts) + 1):
                author_part = ' '.join(parts[i:j])
                year_part = None

                # 查找年份（四位数）
                for k in range(j, len(parts)):
                    if re.match(r'^\d{4}[a-z]?$', parts[k]):
                        year_part = parts[k]
                        break
                    # 跳过 "et" 和 "al." 词
                    if parts[k].lower() in ['et', 'al.', 'al', ',']:
                        continue

                if year_part:
                    # 尝试匹配
                    ref_id = self._match_author_in_map(author_part, year_part, author_year_map)
                    if ref_id:
                        return [ref_id]

        return []

    def link_citations_to_references(
        self,
        chunks: List[Any],
        references: List[Reference]
    ) -> List[Any]:
        """
        将正文章节与参考文献建立关联

        Args:
            chunks: 分块后的Node列表
            references: 提取的Reference列表

        Returns:
            更新后的chunks列表，每个chunk添加了cited_ref_ids元数据
        """
        if not references:
            return chunks

        # 构建 ref_id -> Reference 的映射
        ref_map = {ref.ref_id: ref for ref in references}

        # 构建 author-year -> ref_id 的映射（用于 author-year 格式引用）
        author_year_map = self._build_author_year_map(references)

        for i, chunk in enumerate(chunks):
            # 获取chunk的索引（用于被引用记录）
            chunk_idx = chunk.metadata.get('chunk_index', i)

            # 1. 查找数字引用 [1], [2,3]
            citations = self.find_citations_in_text(chunk.text)

            # 2. 查找 author-year 引用 (Smith, 2020)
            author_year_citations = self.find_author_year_citations(chunk.text, author_year_map)
            citations.extend(author_year_citations)

            # 收集此chunk引用的所有ref_ids
            cited_refs = set()
            for citation in citations:
                for ref_id in citation.ref_ids:
                    if ref_id in ref_map:
                        cited_refs.add(ref_id)
                        # 更新参考文献的被引用列表（使用chunk_index）
                        ref_map[ref_id].ref_cited_by.append(str(chunk_idx))

            # 将引用信息添加到chunk元数据
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}

            # 只存储引用的 ref_id 列表，具体信息从 paper_doc_stats.json 查找
            chunk.metadata['cited_ref_ids'] = sorted(cited_refs) if cited_refs else []

        return chunks


class LLMReferenceParser:
    """
    基于大模型的参考文献解析器

    使用 GPT-4o 解析参考文献的标题、作者、年份等信息，
    并通过 arXiv MCP 查询论文详情进行补充。

    特性：
    1. LLM 直接解析参考文献文本
    2. 使用 arXiv MCP 进行论文详情查询和补全
    3. 自动识别参考文献中的标题、作者、年份、DOI 等信息
    """

    # 系统提示词
    SYSTEM_PROMPT = """你是一个学术论文参考文献解析专家。你的任务是从论文的参考文献部分提取结构化信息。

参考文献格式可能非常复杂，包括但不限于：
- 序号. 作者: 标题. 期刊/会议, 年份.
- [序号] 作者. 标题. 期刊, 年份.
- 作者 (年份). 标题. 期刊.
- 带DOI的格式: 作者. 标题. DOI: xx.xxxx/xxxxx

你需要提取以下字段：
- title: 论文标题（最重要的字段）
- authors: 作者列表（多个作者用逗号分隔）
- year: 年份（4位数字）
- venue: 期刊/会议名称
- doi: DOI（如果有）

请仔细分析每条参考文献，准确提取上述信息。如果某些信息确实无法从文本中获得，请留空。"""

    # 批量解析提示词
    BATCH_PARSE_PROMPT = """你是一个学术论文参考文献解析专家。请批量解析以下参考文献。

参考文献格式可能包括：
- 序号. 作者: 标题. 期刊/会议, 年份.
- [序号] 作者. 标题. 期刊, 年份.
- 作者 (年份). 标题. 期刊.
- 各种变体格式

请为每条参考文献提取以下字段：
- title: 论文标题
- authors: 作者（多个作者用逗号分隔）
- year: 年份（4位数字）
- venue: 期刊/会议名称（如果有）
- doi: DOI（如果有）
- raw_snippet: 该条参考文献的原始文本片段（用于搜索补全）

请以JSON数组格式返回，不要包含任何其他内容：
[
    {{
        "title": "论文标题",
        "authors": "作者列表",
        "year": "年份",
        "venue": "期刊/会议",
        "doi": "DOI",
        "raw_snippet": "原始参考文献文本"
    }}
]

参考文献列表：
{reference_list}

只返回JSON数组，不要有其他内容："""

    # 整段参考文献解析提示词（让LLM自己分割+解析）
    SECTION_PARSE_PROMPT = """你是一个学术论文参考文献解析专家。下面是一篇论文的完整参考文献部分。

你的任务是：
1. 首先识别出参考文献部分中每一条单独的参考文献（参考文献可能跨多行）
2. 然后解析每条参考文献的详细信息

识别参考文献的技巧：
- 参考文献通常以数字编号 [1]、1. 或直接以作者名开头
- 每条参考文献通常以年份结尾（2021. 或 (2021)）
- 新引用通常从新的一行开始（该行以作者名或编号开头）
- 如果某行以大写字母开头且上一行以年份结尾，这是新引用的开始

请为每条识别出的参考文献提取以下字段：
- title: 论文标题（**重要：必须逐字完整复制标题，不要截断、缩写或省略任何部分**）
- authors: 作者（多个作者用逗号分隔，只填作者姓名，不填"et al"等）
- year: 年份（4位数字）
- venue: 期刊/会议名称（如果有）
- doi: DOI（如果有，只填DOI号）
- arxiv_id: arXiv ID（如果有，格式如 2412.01807，不含 arXiv: 前缀）
- raw_snippet: 该条参考文献的原始文本（从参考文献部分逐字复制，用于搜索补全时的回退查询）

请以JSON数组格式返回，只返回一个数组，不要有任何其他内容：
[
    {{
        "title": "论文完整标题",
        "authors": "作者1, 作者2, 作者3",
        "year": "2021",
        "venue": "期刊或会议名称",
        "doi": "10.xxxx/xxxxx",
        "arxiv_id": "2412.01807",
        "raw_snippet": "该条参考文献的原始文本片段"
    }}
]

参考文献部分：
{ref_section}

只返回JSON数组，不要有任何其他内容："""

    REF_REEXTRACT_PROMPT = """从以下参考文献条目中提取结构化信息。只提取论文本身的元数据，不要包含引用编号。

参考文献：
{raw_text}

请以JSON格式返回，只返回JSON对象，不要有任何其他内容：
{{
    "title": "论文标题（只提取标题本身，不要包含作者、期刊、年份）",
    "authors": "作者（多个用逗号分隔，不要包含et al.）",
    "year": "2024",
    "arxiv_id": "2412.01807"
}}

只返回JSON对象："""

    @staticmethod
    def _looks_like_polluted_title(title: str, authors: str = "") -> bool:
        """检测 title 是否被引用编号、作者名或 URL 污染。

        高置信度信号：
        1. 引用编号前缀 [N] / N. / N)
        2. "et al." 出现在标题中 → 确定是作者名
        3. title 完全等于 authors（LLM 将作者列表误认为标题）
        4. title 是 URL（LLM 误将链接当作标题）
        """
        import re
        t = (title or "").strip()
        if not t:
            return False
        if re.match(r'^\[\d+\]', t) or re.match(r'^\d+[\.\)]\s', t):
            return True
        if re.search(r'\bet al\.?\b', t):
            return True
        if re.match(r'^https?://', t):
            return True
        if authors:
            a = authors.strip().strip('.')
            if len(a) >= 5 and t.lower().strip('.') == a.lower():
                return True
        return False

    async def _re_extract_reference(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """将整条引用文本交 LLM 重新提取完整结构化信息。"""
        raw = (raw_text or "").strip()
        if not raw:
            return None
        prompt = self.REF_REEXTRACT_PROMPT.format(raw_text=raw)
        try:
            result = await self._call_llm(prompt)
            if not result:
                return None
            json_str = self._extract_json(result)
            if not json_str:
                logger.warning(f"📝 [引用重提取] 无法从 LLM 响应提取 JSON: {result[:120]}")
                return None
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                return None
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"📝 [引用重提取] JSON 解析失败: {e}")
            return None
        except Exception as e:
            logger.warning(f"📝 [引用重提取] LLM 调用异常: {e}")
            return None

    def __init__(
        self,
        llm_config: Dict[str, Any],
        arxiv_client: Any = None,
        link_resolver: Any = None,
    ):
        """
        Initialize LLM reference parser.

        Args:
            llm_config: Either a provider dict {"provider": provider_obj} (uses provider.text_chat()),
                        or a raw config dict {"model", "api_base", "api_key"} (uses direct HTTP for freeapi).
            arxiv_client: arXiv MCP client for paper detail queries (fallback).
            link_resolver: PaperLinkResolver instance for multi-source link resolution (preferred).
        """
        self.llm_config = llm_config
        self.arxiv_client = arxiv_client
        self._link_resolver = link_resolver

    async def _call_llm(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call LLM for reference parsing.

        Uses provider.text_chat() for all provider types (handles proxy,
        timeout, key rotation, format conversion). Falls back to raw HTTP
        (freeapi) when provider fails all retries.
        """
        if not self.llm_config:
            logger.error("📝 LLM 配置为空，无法调用 LLM")
            return None

        provider = self.llm_config.get("provider")

        if provider:
            result = await self._call_via_provider(prompt, provider, max_retries)
            if result is not None:
                return result
            # Provider 失败，回退到 freeapi（如果配置了）
            api_url = self.llm_config.get("api_base", "")
            if api_url:
                logger.warning("📝 Provider 全部重试失败，回退到 freeapi HTTP 调用")
                return await self._call_via_http(prompt, max_retries)
            return None

        return await self._call_via_http(prompt, max_retries)

    async def _call_via_provider(
        self, prompt: str, provider, max_retries: int
    ) -> Optional[str]:
        """Call LLM through AstrBot provider (handles all API formats).

        For Google Gemini providers, calls the client directly with thinking
        disabled and streaming enabled to avoid server-side timeout on
        long-thinking requests. For other providers, falls back to the
        standard stream or sync path.
        """
        is_gemini = type(provider).__name__ == 'ProviderGoogleGenAI'

        logger.info(
            f"📝 [LLM调用:provider] 开始请求，prompt长度: {len(prompt)} 字符"
            f"，provider: {type(provider).__name__}"
        )

        for attempt in range(max_retries):
            logger.info(f"📝 [LLM调用:provider] 尝试 {attempt + 1}/{max_retries}")
            try:
                if is_gemini:
                    content = await self._call_gemini_no_thinking(prompt, provider)
                elif callable(getattr(provider, 'text_chat_stream', None)):
                    content = await self._call_via_provider_stream(prompt, provider)
                else:
                    content = await self._call_via_provider_sync(prompt, provider)

                if content and len(content) >= 50:
                    logger.info(
                        f"📝 [LLM调用:provider] 提取到内容长度: {len(content)}"
                    )
                    return content

                if content:
                    logger.warning(
                        f"⚠️ [LLM调用:provider] 响应内容过短 ({len(content)} 字符)，"
                        f"可能被限流，5秒后重试..."
                    )
                    await asyncio.sleep(5)
                    continue
                else:
                    logger.warning("⚠️ [LLM调用:provider] 返回空内容")
            except asyncio.TimeoutError:
                logger.warning("⚠️ [LLM调用:provider] 请求超时 (910s)")
                await asyncio.sleep(3)
                continue
            except Exception as e:
                logger.warning(
                    f"⚠️ [LLM调用:provider] 请求异常: {type(e).__name__}: {e!r}"
                )
                await asyncio.sleep(3)
                continue

        logger.warning(f"⚠️ [LLM调用:provider] 重试 {max_retries} 次后仍失败")
        return None

    async def _call_gemini_no_thinking(
        self, prompt: str, provider
    ) -> Optional[str]:
        """Call Google Gemini directly with thinking disabled and streaming.

        Bypasses provider.text_chat_stream to set thinking_budget=0, which
        eliminates hidden reasoning tokens and dramatically speeds up
        structured output generation for reference parsing.
        """
        from google.genai import types

        model = provider.get_model()
        config = types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        contents = [types.UserContent(parts=[types.Part.from_text(text=prompt)])]

        accumulated = []
        result = await provider.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        async for chunk in result:
            if chunk.text:
                accumulated.append(chunk.text)

        return "".join(accumulated) if accumulated else None

    async def _call_via_provider_stream(
        self, prompt: str, provider
    ) -> Optional[str]:
        """Stream-based call that keeps connection alive for long requests.

        Providers emit incremental chunks (is_chunk=True) then a final
        response (is_chunk=False) with the full accumulated text.
        """
        final_text: Optional[str] = None
        async for resp in provider.text_chat_stream(prompt=prompt):
            if not resp.is_chunk and resp.completion_text:
                final_text = resp.completion_text
        return final_text

    async def _call_via_provider_sync(
        self, prompt: str, provider
    ) -> Optional[str]:
        """Non-stream call for providers without stream timeout issues."""
        original_timeout = getattr(provider, 'timeout', 120)
        client = getattr(provider, 'client', None)
        original_client_timeout = getattr(client, 'timeout', None)
        if client is not None:
            try:
                client.timeout = 900.0
            except (AttributeError, TypeError):
                pass
        provider.timeout = 900

        try:
            response = await asyncio.wait_for(
                provider.text_chat(prompt=prompt),
                timeout=910,
            )
            return response.completion_text
        finally:
            try:
                provider.timeout = original_timeout
            except (AttributeError, TypeError):
                pass
            if client is not None and original_client_timeout is not None:
                try:
                    client.timeout = original_client_timeout
                except (AttributeError, TypeError):
                    pass

    async def _call_via_http(
        self, prompt: str, max_retries: int
    ) -> Optional[str]:
        """Call LLM via raw HTTP (OpenAI-compatible API). Used for freeapi fallback."""
        url = f"{self.llm_config['api_base']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_config.get('api_key', 'sk-placeholder')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.llm_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.llm_config.get("max_tokens", 32768),
        }
        temperature = self.llm_config.get("temperature")
        if temperature is not None:
            data["temperature"] = temperature

        # Disable thinking mode for models that output reasoning by default
        # which breaks structured JSON response parsing
        model_lower = self.llm_config["model"].lower()
        if "glm" in model_lower or "deepseek" in model_lower:
            data["thinking"] = {"type": "disabled"}

        logger.info(f"📝 [LLM调用] 开始请求，prompt长度: {len(prompt)} 字符")

        for attempt in range(max_retries):
            logger.info(f"📝 [LLM调用] 尝试 {attempt + 1}/{max_retries}")
            try:
                async with aiohttp.ClientSession(trust_env=False) as session:
                    async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=900)) as resp:
                        logger.info(f"📝 [LLM调用] 收到响应状态码: {resp.status}")
                        if resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", 5))
                            logger.warning(f"⚠️ LLM API 速率限制 (429)，{retry_after}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                            await asyncio.sleep(retry_after)
                            continue
                        if resp.status == 500:
                            logger.warning(f"⚠️ LLM API 服务器错误 (500)，5秒后重试... (尝试 {attempt + 1}/{max_retries})")
                            await asyncio.sleep(5)
                            continue
                        if resp.status != 200:
                            text = await resp.text()
                            redacted = re.sub(r'(key|token|secret|Authorization)[=:]\s*\S+', r'\1=***', text, flags=re.IGNORECASE)
                            logger.warning(f"⚠️ LLM API 请求失败: HTTP {resp.status}, {redacted[:300]}")
                            return None
                        body = await resp.text()
                        try:
                            result = json.loads(body)
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ LLM API 返回非JSON，响应前200字符: {body[:200]}")
                            return None
                        logger.info(f"📝 [LLM调用] 响应解析成功")
            except asyncio.TimeoutError:
                logger.warning("⚠️ LLM API 请求超时")
                await asyncio.sleep(3)
                continue
            except Exception as e:
                logger.warning(f"⚠️ LLM API 请求异常: {e}")
                await asyncio.sleep(3)
                continue

            # 提取响应内容
            logger.info(f"📝 [LLM调用] 解析响应内容...")
            try:
                choices = result.get("choices", [])
                if not choices:
                    logger.warning("⚠️ LLM 返回空 choices")
                    return None
                finish_reason = choices[0].get("finish_reason", "")
                if finish_reason == "length":
                    logger.warning(
                        "⚠️ LLM 输出被截断 (finish_reason=length)，"
                        "参考文献解析可能不完整，建议检查结果"
                    )
                message = choices[0].get("message", {})
                content = message.get("content", "")
                logger.info(f"📝 [LLM调用] 提取到内容长度: {len(content) if content else 0}")
                if not content:
                    logger.warning("⚠️ [LLM调用] 返回空内容")
                    await asyncio.sleep(3)
                    continue
                if len(content) < 50:
                    logger.warning(
                        f"⚠️ [LLM调用] 响应内容过短 ({len(content)} 字符)，可能被限流，5秒后重试..."
                    )
                    await asyncio.sleep(5)
                    continue
                return content
            except Exception as e:
                logger.warning(f"⚠️ 解析 LLM 响应失败: {e}")
                return None

        logger.warning(f"⚠️ LLM API 重试 {max_retries} 次后仍失败")
        return None

    async def parse_reference_section(
        self,
        ref_section: str,
        ref_id_prefix: str = "ref",
        enable_fallback_search: bool = False,
        skip_resolution: bool = False,
    ) -> List[Reference]:
        """
        解析整段参考文献文本（让LLM自动分割+解析）

        超过 15000 字符时按序号边界分批，避免单次请求过大导致超时。

        Args:
            ref_section: 参考文献部分的完整文本（可能跨多行）
            ref_id_prefix: ref_id 前缀
            enable_fallback_search: 启用回退搜索

        Returns:
            Reference 对象列表
        """
        if not ref_section or not ref_section.strip():
            return []

        text_length = len(ref_section)
        logger.info(f"📝 开始 LLM 参考文献解析，文本长度: {text_length} 字符")

        # 超过 15000 字符时分批处理（约 4000 tokens，安全上限）
        if text_length > 15000:
            batches = self._split_reference_section_by_numbers(ref_section, max_chars=15000)
            logger.info(f"📝 分为 {len(batches)} 批进行处理（串行）")

            all_results = []
            for i, batch_text in enumerate(batches):
                if i > 0:
                    await asyncio.sleep(5)
                batch_prefix = f"{ref_id_prefix}_{i}"
                try:
                    results = await self._parse_single_batch(batch_text, batch_prefix, i)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"📝 批次 {batch_prefix} 处理异常: {e}")
                    continue

            # 重新编号
            for j, ref in enumerate(all_results):
                ref.ref_id = f"{ref_id_prefix}_{j + 1}"

            # 统一 arXiv 富化
            if not skip_resolution:
                await self._enrich_references(all_results, enable_fallback_search=enable_fallback_search)

            logger.info(f"📚 LLM 解析参考文献: 成功 {len(all_results)} 条 ({len(batches)} 批次)")
            return all_results

        # 正常单次处理
        results = await self._parse_single_batch(ref_section, ref_id_prefix, 0)
        if results and not skip_resolution:
            await self._enrich_references(results, enable_fallback_search=enable_fallback_search)
        return results

    def _split_reference_section_by_numbers(
        self,
        ref_section: str,
        max_chars: int = 15000
    ) -> List[str]:
        """
        按参考文献序号分割文本，确保每批不超过 max_chars 字符

        分割点: [1], [2], 1., 2., [12], [123] 等序号模式
        每批按序号分割，避免参考文献被从中间截断
        """
        lines = ref_section.split('\n')
        batches = []
        current_batch = []
        current_char_count = 0

        for line in lines:
            line_char_count = len(line)

            if current_char_count + line_char_count > max_chars and current_batch:
                is_new_ref = bool(re.match(r'^\s*\[?\d+\]?\s*[\.\:]', line.strip()))

                if not is_new_ref and current_char_count < max_chars * 0.9:
                    current_batch.append(line)
                    current_char_count += line_char_count
                    continue

                batches.append('\n'.join(current_batch))
                current_batch = []
                current_char_count = 0

            current_batch.append(line)
            current_char_count += line_char_count

        if current_batch:
            batches.append('\n'.join(current_batch))

        # 二次分割：强制截断仍超限的批次
        final_batches = []
        for batch in batches:
            if len(batch) > max_chars:
                sub_lines = batch.split('\n')
                sub_batch = []
                sub_char_count = 0
                for line in sub_lines:
                    if sub_char_count + len(line) > max_chars and sub_batch:
                        final_batches.append('\n'.join(sub_batch))
                        sub_batch = []
                        sub_char_count = 0
                    sub_batch.append(line)
                    sub_char_count += len(line)
                if sub_batch:
                    final_batches.append('\n'.join(sub_batch))
            else:
                final_batches.append(batch)

        return final_batches

    async def _parse_single_batch(
        self,
        ref_section: str,
        ref_id_prefix: str,
        batch_index: int
    ) -> List[Reference]:
        """
        解析单批参考文献文本

        Args:
            ref_section: 参考文献文本
            ref_id_prefix: ref_id 前缀
            batch_index: 批次索引（用于日志）

        Returns:
            Reference 对象列表
        """
        prompt = self.SECTION_PARSE_PROMPT.format(ref_section=ref_section)

        try:
            response = await self._call_llm(prompt)

            if not response:
                logger.warning(f"⚠️ 批次 {batch_index+1}: LLM 未返回有效响应")
                return []

            # 提取 JSON
            json_str = self._extract_json(response)
            if not json_str:
                # LLM 可能用自然语言拒绝解析（如输入不是参考文献列表）
                refusal_markers = [
                    "not a reference", "not a reference list", "not a reference section",
                    "no complete reference", "no references found", "no reference list",
                    "appendix",  # 实际观测: 模型会说明输入来自附录部分
                    "不是参考文献", "不是参考", "无法识别参考文献",
                ]
                lower_resp = response.lower()
                if any(marker in lower_resp for marker in refusal_markers):
                    logger.info(f"⚠️ 批次 {batch_index+1}: LLM 认为输入不是有效参考文献列表（跳过）: {response[:200]}")
                    return []
                logger.warning(f"⚠️ 批次 {batch_index+1}: 无法从 LLM 响应中提取 JSON，响应长度: {len(response)} 字符")
                logger.warning(f"========== LLM 原始输出 ==========\n{response}\n========== END ==========")
                return []

            parsed_list = json.loads(json_str)
            if not isinstance(parsed_list, list):
                logger.warning(f"⚠️ 批次 {batch_index+1}: LLM 返回的不是数组: {type(parsed_list)}")
                return []

            results = []
            for j, parsed in enumerate(parsed_list):
                try:
                    raw_snippet = parsed.get("raw_snippet", "") or ""
                    ref = Reference(
                        ref_id=f"{ref_id_prefix}_{j + 1}",
                        raw_text=raw_snippet,
                        ref_title=parsed.get("title", ""),
                        ref_authors=parsed.get("authors", ""),
                        ref_year=int(parsed["year"]) if str(parsed.get("year", "")).isdigit() else None,
                        ref_doi=parsed.get("doi") or None,
                        ref_venue=parsed.get("venue") or None,
                        ref_source_arxiv_id=parsed.get("arxiv_id") or None,
                    )
                    results.append(ref)
                except Exception as e:
                    logger.debug(f"⚠️ 批次 {batch_index+1} 解析第 {j} 条失败: {e}, 数据: {parsed}")
                    continue

            logger.info(f"📚 批次 {batch_index+1}: 解析成功 {len(results)} 条")
            return results

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ 批次 {batch_index+1}: JSON 解析失败: {e}")
            return []
        except Exception as e:
            logger.warning(f"⚠️ 批次 {batch_index+1}: 参考文献解析失败: {e}")
            return []

    async def parse_references(
        self,
        references: List[str],
        ref_id_prefix: str = "ref",
        enable_fallback_search: bool = False,
    ) -> List[Reference]:
        """
        解析参考文献列表

        Args:
            references: 参考文献原始文本列表
            ref_id_prefix: ref_id 前缀
            enable_fallback_search: 启用回退搜索

        Returns:
            Reference 对象列表
        """
        if not references:
            return []

        total = len(references)
        logger.info(f"📝 开始 LLM 参考文献解析，共 {total} 条...")

        # 一次请求解析所有参考文献
        results = await self._parse_batch(references, ref_id_prefix, 0)

        # 过滤掉解析失败的
        valid_results = [r for r in results if r is not None]

        # arXiv 富化：用 arXiv 官方元数据校验并补全
        await self._enrich_references(valid_results, enable_fallback_search=enable_fallback_search)

        logger.info(f"📚 LLM 解析参考文献: 成功 {len(valid_results)}/{total} 条")
        return valid_results

    async def _parse_batch(
        self,
        references: List[str],
        ref_id_prefix: str,
        start_index: int
    ) -> List[Optional[Reference]]:
        """批量解析一组参考文献"""
        if not references:
            return []

        # 构建参考文献列表文本（不截断原始引用）
        ref_list_text = "\n".join([
            f"[{j}] {ref}"
            for j, ref in enumerate(references)
        ])

        prompt = self.BATCH_PARSE_PROMPT.format(reference_list=ref_list_text)

        try:
            response = await self._call_llm(prompt)

            if not response:
                logger.warning("⚠️ LLM 未返回有效响应")
                return [None] * len(references)

            # 提取 JSON
            json_str = self._extract_json(response)

            if not json_str:
                logger.warning(f"⚠️ 无法从 LLM 响应中提取 JSON，响应长度: {len(response)} 字符")
                logger.warning(f"========== LLM 原始输出 ==========\n{response}\n========== END ==========")
                return [None] * len(references)

            parsed_list = json.loads(json_str)

            results = []
            for j, parsed in enumerate(parsed_list):
                try:
                    ref = Reference(
                        ref_id=f"{ref_id_prefix}_{start_index + j + 1}",
                        raw_text=references[j],
                        ref_title=parsed.get("title", ""),
                        ref_authors=parsed.get("authors", ""),
                        ref_year=int(parsed["year"]) if str(parsed.get("year", "")).isdigit() else None,
                        ref_doi=parsed.get("doi") or None,
                        ref_venue=parsed.get("venue") or None
                    )
                    results.append(ref)
                except Exception as e:
                    logger.debug(f"⚠️ 解析第 {j} 条失败: {e}")
                    results.append(None)

            return results

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON 解析失败: {e}")
            return [None] * len(references)
        except Exception as e:
            logger.warning(f"⚠️ 批量解析失败: {e}")
            return [None] * len(references)

    @staticmethod
    def _find_last_complete_json(text: str) -> Optional[str]:
        """从截断的 JSON 数组中提取最后一个完整的 JSON 对象，重新闭合数组。

        例如 "[{...}, {"title": "inc" → "[{...}]"
        """
        import re
        # 从末尾向前找最后一个完整的 }（JSON 对象结束）
        # 策略：找到最后一个 }, 标记处，往前确认对应的 { 位置，然后闭合数组
        rbrace_positions = [m.end() for m in re.finditer(r'\}', text)]
        if not rbrace_positions:
            return None

        # 从后往前试：逐个尝试在 } 后面加 ] 看是否能解析
        for pos in reversed(rbrace_positions):
            candidate = text[:pos] + ']'
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
        return None

    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 字符串"""

        # 记录原始长度
        original_len = len(text)

        # 清理文本：去除 BOM 和首尾空白
        text = text.strip().lstrip('\ufeff')

        # 如果清理后文本变短了，说明有 BOM 或空白
        if len(text) < original_len:
            logger.info(f"清理了 {original_len - len(text)} 个字符的 BOM/空白")

        def try_parse(t):
            """尝试解析 JSON，失败时修复无效转义后重试"""
            try:
                json.loads(t)
                return t, None
            except json.JSONDecodeError as e:
                # 检查是否是转义问题
                if "Invalid \\escape" in str(e) or "Invalid escape" in str(e):
                    # 修复常见的问题：\d, \s, \n, \t 等在非转义上下文
                    # 替换 {\d, \s, \t, \n 等为正确的转义
                    fixed = re.sub(r'\\([dDsStTn])', r'\\\\\1', t)
                    try:
                        json.loads(fixed)
                        return fixed, "fixed_escape"
                    except Exception:
                        pass
                return None, str(e)

        result, error = try_parse(text)
        if result:
            return result

        # 尝试提取 markdown 代码块中的 JSON
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            json_str = match.group(1).strip()
            result, _ = try_parse(json_str)
            if result:
                return result

        # 尝试找到 JSON 数组或对象（贪婪匹配）
        all_matches = list(re.finditer(r'(\{[\s\S]*\}|\[[\s\S]*\])', text))
        logger.info(f"[_extract_json] 找到 {len(all_matches)} 个潜在 JSON 匹配")
        for i, match in enumerate(all_matches):
            json_str = match.group(1)
            logger.info(f"[_extract_json] 匹配 {i+1}: 长度={len(json_str)}, 内容={repr(json_str)}")
            result, _ = try_parse(json_str)
            if result:
                logger.info(f"[_extract_json] 匹配 {i+1} 解析成功")
                return result

        # 截断挽救：LLM 输出可能因 max_tokens 被截断，尝试从已有数据中挽救
        truncated = text.strip()
        # 去掉 markdown 代码块前缀
        truncated = re.sub(r'^```(?:json)?\s*', '', truncated)
        # 如果以 [ 开头但不以 ] 结尾，尝试修复
        if truncated.startswith('[') and not truncated.rstrip().endswith(']'):
            last_complete = self._find_last_complete_json(truncated)
            if last_complete:
                logger.info(f"[_extract_json] 截断挽救: 从 {len(truncated)} 字符中恢复 {last_complete.count(chr(10))+1} 条记录")
                return last_complete

        return None

    async def _enrich_references(
        self, references: List[Reference], enable_fallback_search: bool = False
    ) -> None:
        """
        多源参考文献链接解析 + 元数据校验。

        策略（按优先级）：
        1. PaperLinkResolver（Crossref → OpenAlex → arXiv library）— 模糊匹配
        2. arXiv MCP（DOI 精确搜索）— 兜底
        3. Semantic Scholar 标题匹配 — 最终回退（仅 enable_fallback_search=True 时）
        4. 都失败则保留 LLM 原始解析结果，记录日志

        Args:
            references: Reference 对象列表（会被直接修改）
            enable_fallback_search: 启用回退搜索（reparse 时推荐开启）
        """
        if not references:
            return

        enriched_by_link_resolver = 0
        enriched_by_arxiv_mcp = 0
        enriched_by_web_search = 0

        sem = asyncio.Semaphore(10)

        async def enrich_one(ref: Reference) -> Dict[str, int]:
            """Enrich a single reference. Returns counts dict."""
            local_counts = {"link_resolver": 0, "arxiv_mcp": 0}

            if not ref or not ref.ref_title or len(ref.ref_title) <= 5:
                return local_counts

            # ---- 标题污染检测：若以编号或作者名开头，交 LLM 重提取 ----
            if self._looks_like_polluted_title(ref.ref_title, ref.ref_authors):
                raw_text = (ref.raw_text or "").strip()
                # raw_text 不足时，用 ref_title 本身作为回退文本（LLM 仍可从中提取干净标题）
                if not raw_text or len(raw_text) < 30:
                    if ref.ref_title and len(ref.ref_title) >= 30:
                        raw_text = ref.ref_title
                    else:
                        logger.debug(
                            f"📝 [引用重提取] 检测到污染但可用文本不足 "
                            f"({len(raw_text)} 字符)，跳过: {ref.ref_title[:80]}"
                        )
                        raw_text = None
                if raw_text:
                    logger.info(f"📝 [引用重提取] 检测到污染: {ref.ref_title[:80]}...")
                    extracted = await self._re_extract_reference(raw_text)

                    if extracted:
                        new_title = (extracted.get("title") or "").strip()
                        if new_title and len(new_title) >= 5:
                            logger.info(f"📝 [引用重提取] title: {new_title[:80]}")
                            ref.ref_title = new_title
                        new_authors = (extracted.get("authors") or "").strip()
                        if new_authors and not ref.ref_authors:
                            ref.ref_authors = new_authors
                        new_year = extracted.get("year")
                        if ref.ref_year is None:
                            try:
                                ref.ref_year = int(float(str(new_year)))
                            except (ValueError, TypeError):
                                pass
                        new_arxiv_id = (extracted.get("arxiv_id") or "").strip()
                        if new_arxiv_id and not ref.ref_source_arxiv_id:
                            ref.ref_source_arxiv_id = new_arxiv_id
                    else:
                        logger.warning(f"📝 [引用重提取] 重提取失败，继续使用原始字段")

            author_hint = ref.ref_authors or ""

            # ---- 第零优先：DataCite 直查（LLM 已提取 arXiv ID）----
            if ref.ref_source_arxiv_id and self._link_resolver is not None:
                try:
                    resolution = await self._link_resolver.resolve_by_arxiv_id(
                        ref.ref_source_arxiv_id
                    )
                    if resolution.has_any_url():
                        if resolution.doi_url and not ref.ref_doi:
                            ref.ref_doi = resolution.doi_url.replace("https://doi.org/", "")
                        if resolution.arxiv_url and not ref.ref_arxiv_url:
                            ref.ref_arxiv_url = resolution.arxiv_url
                        if resolution.matched_title:
                            ref.ref_title = resolution.matched_title
                        local_counts["link_resolver"] += 1
                        logger.info(
                            f"📝 [DataCite 直查] {ref.ref_source_arxiv_id} → "
                            f"{ref.ref_title[:60]}"
                        )
                        return local_counts
                except Exception as e:
                    logger.warning(
                        f"📝 [DataCite 直查] 异常: {ref.ref_source_arxiv_id} — {e}"
                    )

            # ---- 第一优先：PaperLinkResolver 多源解析 ----
            if self._link_resolver is not None:
                try:
                    resolution = await self._link_resolver.resolve_by_title(
                        ref.ref_title, author_hint=author_hint
                    )
                    if resolution.has_any_url():
                        if resolution.doi_url and not ref.ref_doi:
                            ref.ref_doi = resolution.doi_url.replace("https://doi.org/", "")
                        if resolution.arxiv_url and not ref.ref_arxiv_url:
                            ref.ref_arxiv_url = resolution.arxiv_url
                        if resolution.matched_title and resolution.resolution_score >= 85:
                            ref.ref_title = resolution.matched_title
                        local_counts["link_resolver"] += 1
                        logger.info(
                            f"📝 [多源解析] {resolution.backend} 匹配成功 "
                            f"(相似度 {resolution.resolution_score:.1f}%): "
                            f"{ref.ref_title[:60]}"
                        )
                        return local_counts
                    else:
                        logger.info(
                            f"📝 [多源解析] 未找到链接 "
                            f"(最佳 {resolution.backend}, {resolution.resolution_score:.1f}%): "
                            f"{ref.ref_title[:60]}"
                        )
                except Exception as e:
                    logger.warning(f"📝 [多源解析] 异常: {ref.ref_title[:60]} — {e}")

            # ---- 第二优先：arXiv MCP 兜底 ----
            if self.arxiv_client:
                try:
                    search_query = ref.ref_doi or ref.ref_title
                    result = await self.arxiv_client.call_tool_with_reconnect(
                        tool_name="search_arxiv",
                        arguments={"query": search_query, "max_results": 3}
                    )

                    if result is None:
                        logger.warning(f"📝 [arXiv MCP] 客户端返回 None，保留原始解析: {ref.ref_title[:60]}")
                        return local_counts

                    if not result.get("results"):
                        logger.info(f"📝 [arXiv MCP] 未搜到结果，保留原始解析: {ref.ref_title[:60]}")
                        return local_counts

                    matched = False
                    for paper in result.get("results", []):
                        paper_title = paper.get("title", "")
                        _ref_title = ref.ref_title if ref.ref_title else ""
                        if not _ref_title or not paper_title:
                            continue
                        if _ref_title.lower() != paper_title.lower():
                            continue

                        if paper.get("authors"):
                            ref.ref_authors = ", ".join(paper["authors"])
                        if paper.get("published_date"):
                            year_match = re.search(r'(\d{4})', paper["published_date"])
                            if year_match:
                                ref.ref_year = int(year_match.group(1))
                        if paper.get("doi"):
                            ref.ref_doi = paper.get("doi")

                        local_counts["arxiv_mcp"] += 1
                        matched = True
                        logger.debug(f"📝 [arXiv MCP] 已校验: {ref.ref_title[:60]}")
                        break

                    if not matched:
                        logger.info(f"📝 [arXiv MCP] 标题未完全匹配，保留原始解析: {ref.ref_title[:60]}")

                except Exception as e:
                    logger.warning(f"📝 [arXiv MCP] 查询异常，保留原始解析: {ref.ref_title[:60]} — {e}")

            return local_counts

        async def enrich_with_semaphore(ref: Reference) -> Dict[str, int]:
            async with sem:
                return await enrich_one(ref)

        tasks = [enrich_with_semaphore(ref) for ref in references if ref]
        if tasks:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for r in gathered:
                if isinstance(r, dict):
                    enriched_by_link_resolver += r.get("link_resolver", 0)
                    enriched_by_arxiv_mcp += r.get("arxiv_mcp", 0)
                elif isinstance(r, Exception):
                    logger.warning(f"📝 [批量增强] 单条引用处理异常: {r}")

        # ---- Fallback tiers: Semantic Scholar → DDG web search ----
        if enable_fallback_search and self._link_resolver is not None:
            # Helper: collect refs still missing links
            def _unresolved():
                return [r for r in references if r and r.ref_title
                        and len(r.ref_title) > 5
                        and not r.ref_doi and not r.ref_arxiv_url and not r.ref_url]

            # Helper: apply resolution result to a ref
            def _apply(ref, resolution):
                if resolution.doi_url and not ref.ref_doi:
                    ref.ref_doi = resolution.doi_url.replace("https://doi.org/", "")
                if resolution.arxiv_url and not ref.ref_arxiv_url:
                    ref.ref_arxiv_url = resolution.arxiv_url
                if resolution.url and not ref.ref_url:
                    ref.ref_url = resolution.url
                if resolution.matched_title and resolution.resolution_score >= 85:
                    ref.ref_title = resolution.matched_title

            # Tier 3: Semantic Scholar title match
            missing = _unresolved()
            if missing:
                logger.info(f"📝 [SemanticScholar] {len(missing)} refs to try...")
                ss_matched = 0
                for ref in missing[:10]:  # cap at 10
                    try:
                        r = await self._link_resolver._resolve_via_semantic_scholar(ref.ref_title)
                        if r.has_any_url():
                            _apply(ref, r)
                            enriched_by_web_search += 1
                            ss_matched += 1
                            logger.info(f"📝 [SemanticScholar] {r.backend} ({r.resolution_score:.0f}%): {ref.ref_title[:60]}")
                    except Exception as e:
                        logger.debug(f"📝 [SS] {ref.ref_title[:40]} — {type(e).__name__}")

                if ss_matched == 0:
                    logger.info(f"📝 [SemanticScholar] no matches (papers likely not in academic index)")

            # Tier 4: DDG web search for remaining refs
            missing = _unresolved()
            if missing:
                logger.info(f"📝 [WebSearch] {len(missing)} refs remain, searching web...")
                web_matched = 0
                for ref in missing[:10]:
                    try:
                        r = await self._link_resolver._resolve_via_web_search(ref.ref_title)
                        if r.has_any_url():
                            _apply(ref, r)
                            enriched_by_web_search += 1
                            web_matched += 1
                            logger.info(f"📝 [WebSearch] {r.backend}: {ref.ref_title[:60]}")
                    except Exception as e:
                        logger.debug(f"📝 [WS] {ref.ref_title[:40]} — {type(e).__name__}")

                if web_matched > 0:
                    logger.info(f"📝 [WebSearch] found {web_matched}")

        # 统计链接覆盖率
        has_doi = 0
        has_arxiv = 0
        missing_all = []
        for ref in references:
            if not ref:
                continue
            doi_ok = bool(ref.ref_doi)
            arxiv_ok = bool(ref.ref_arxiv_url)
            if doi_ok:
                has_doi += 1
            if arxiv_ok:
                has_arxiv += 1
            if not doi_ok and not arxiv_ok:
                title = (ref.ref_title or "")[:80]
                if title:
                    missing_all.append(title)

        parts = [
            f"多源解析 {enriched_by_link_resolver} 条",
            f"arXiv MCP {enriched_by_arxiv_mcp} 条",
        ]
        if enriched_by_web_search > 0:
            parts.append(f"回退搜索 {enriched_by_web_search} 条")
        logger.info(
            f"📝 参考文献富化完成: {', '.join(parts)}, "
            f"总计 {len(references)} 条"
        )
        logger.info(
            f"📝 链接覆盖: DOI {has_doi} | arXiv {has_arxiv} | "
            f"无链接 {len(missing_all)}"
        )
        if missing_all:
            logger.warning(
                f"📝 无链接参考文献 ({len(missing_all)} 条):\n" +
                "\n".join(f"  - {t}" for t in missing_all)
            )



def _paper_doc_stats_path():
    """返回 paper_doc_stats.json 的绝对路径。"""
    from pathlib import Path
    return Path(__file__).parent.parent / "data" / "paper_doc_stats.json"


def _load_paper_doc_stats() -> dict:
    """加载 paper_doc_stats.json，返回 {paper_key: stats_dict}。"""
    import json
    stats_path = _paper_doc_stats_path()
    if not stats_path.exists():
        return {}
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_paper_doc_stats(pdf_path: str, references: List[Reference]) -> None:
    """保存论文级参考文献解析结果到 data/paper_doc_stats.json。

    每篇论文一条记录，包含：
    - 统计信息（链接覆盖率、缺失列表）
    - references 字典（ref_id → 完整引用详情），供召回时查找
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone

    has_doi = 0
    has_arxiv = 0
    missing_refs: List[Dict[str, Any]] = []
    refs_detail: Dict[str, Dict[str, Any]] = {}

    for ref in references:
        if not ref:
            continue
        doi_ok = bool(ref.ref_doi)
        arxiv_ok = bool(ref.ref_arxiv_url)
        if doi_ok:
            has_doi += 1
        if arxiv_ok:
            has_arxiv += 1
        if not doi_ok and not arxiv_ok:
            missing_refs.append({
                "title": ref.ref_title or "",
                "authors": ref.ref_authors or "",
                "year": ref.ref_year,
            })

        refs_detail[ref.ref_id] = ref.to_dict()

    stats: Dict[str, Any] = {
        "file_name": Path(pdf_path).name,
        "total_refs": len(references),
        "has_doi": has_doi,
        "has_arxiv": has_arxiv,
        "missing_all": len(missing_refs),
        "missing_refs": missing_refs,
        "references": refs_detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    stats_path = _paper_doc_stats_path()
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    all_stats = _load_paper_doc_stats()
    paper_key = Path(pdf_path).name
    existing = all_stats.get(paper_key, {})
    existing.update(stats)
    all_stats[paper_key] = existing

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    logger.info(
        f"📝 论文参考文献统计已保存: {stats_path} "
        f"(DOI {has_doi}, arXiv {has_arxiv}, 无链接 {len(missing_refs)})"
    )


def compute_refstats() -> Dict[str, Any]:
    """直接从 paper_doc_stats.json 计算引用频次统计，无需查询 Milvus。

    每篇论文每个引用只计一次（同一论文多次引用同一文献不重复计数）。
    按标题归一化后跨论文聚合。

    Returns:
        {"references": [...], "total_refs": int, "total_papers": int}
    """
    all_stats = _load_paper_doc_stats()
    if not all_stats:
        return {"references": [], "total_refs": 0, "total_papers": 0}

    # title_norm → {"count": int, "raw_title": str, ...}
    title_counter: Dict[str, Dict[str, Any]] = {}
    # title_norm → set of paper_keys (用于同一论文内去重)
    paper_title_sets: Dict[str, set] = {}

    for paper_key, paper_stats in all_stats.items():
        if not isinstance(paper_stats, dict):
            continue
        refs = paper_stats.get("references", {})
        if not isinstance(refs, dict):
            continue

        for ref_id, ref in refs.items():
            if not isinstance(ref, dict):
                continue
            title = ref.get("ref_title", "").strip()
            if not title or len(title) < 5:
                continue

            title_norm = " ".join(title.lower().split())

            if title_norm not in title_counter:
                title_counter[title_norm] = {
                    "count": 0,
                    "raw_title": title,
                    "ref_authors": ref.get("ref_authors", ""),
                    "ref_year": ref.get("ref_year"),
                    "ref_doi": ref.get("ref_doi", ""),
                }
                paper_title_sets[title_norm] = set()

            if paper_key not in paper_title_sets[title_norm]:
                title_counter[title_norm]["count"] += 1
                paper_title_sets[title_norm].add(paper_key)

    total_refs = sum(info["count"] for info in title_counter.values())

    refs_list = []
    for title_norm, info in title_counter.items():
        refs_list.append({
            "title": info["raw_title"],
            "count": info["count"],
            "authors": info["ref_authors"],
            "year": info["ref_year"],
            "doi": info["ref_doi"],
        })
    refs_list.sort(key=lambda x: x["count"], reverse=True)

    return {
        "references": refs_list,
        "total_refs": total_refs,
        "total_papers": len(all_stats),
    }


def get_papers_with_zero_refs_from_json() -> Dict[str, Any]:
    """直接从 paper_doc_stats.json 获取参考文献数为 0 的论文列表。

    视为零引用的情况：
    - 无 references 字段或 references 为空
    - references 条目全部 title 为空（LLM 解析失败）

    Returns:
        {"papers": [...], "total_papers": int, "total_zero_ref": int}
    """
    all_stats = _load_paper_doc_stats()
    if not all_stats:
        return {"papers": [], "total_papers": 0, "total_zero_ref": 0}

    papers = []
    for paper_key, paper_stats in all_stats.items():
        if not isinstance(paper_stats, dict):
            continue
        refs = paper_stats.get("references", {})
        if not isinstance(refs, dict) or not refs:
            papers.append({"file_name": paper_stats.get("file_name", paper_key), "chunk_count": 0})
            continue
        # All refs have empty titles → effectively zero refs (LLM parse failure)
        if all(
            not (isinstance(r, dict) and r.get("ref_title", "").strip())
            for r in refs.values()
        ):
            papers.append({"file_name": paper_stats.get("file_name", paper_key), "chunk_count": 0})

    return {
        "papers": papers,
        "total_papers": len(all_stats),
        "total_zero_ref": len(papers),
    }


def classify_papers_for_repair() -> Dict[str, Any]:
    """Auto-classify papers from paper_doc_stats.json into two repair strategies.

    Strategy A — full_reparse: Papers with any no_title refs (LLM extraction failed),
        polluted titles (author names / citation numbers in title field), or completely
        unparsed. Needs full pipeline: PyMuPDF + LLM extraction + link resolution.

    Strategy B — link_only: Papers where ALL unlinked refs have clean, valid titles.
        Only link resolution failed. Lightweight repair: reload refs from JSON and
        re-run PaperLinkResolver enrichment. No LLM extraction, no PyMuPDF.

    Returns:
        {"full_reparse": [...], "link_only": [...], "total_papers": int}
        Each paper dict: {"file_name": str, "total": int, "linked": int,
                          "title_only": int, "no_title": int}
    """
    all_stats = _load_paper_doc_stats()
    if not all_stats:
        return {"full_reparse": [], "link_only": [], "total_papers": 0}

    full_reparse: list[dict] = []
    link_only: list[dict] = []

    for paper_key, paper_stats in all_stats.items():
        if not isinstance(paper_stats, dict):
            continue
        refs = paper_stats.get("references", {})
        file_name = paper_stats.get("file_name", paper_key)

        if not isinstance(refs, dict) or not refs:
            full_reparse.append({
                "file_name": file_name,
                "total": 0, "linked": 0, "title_only": 0, "no_title": 0,
            })
            continue

        total = len(refs)
        linked = 0
        title_only = 0
        no_title = 0
        for r in refs.values():
            if not isinstance(r, dict):
                continue
            has_link = bool(r.get("ref_doi") or r.get("ref_arxiv_url") or r.get("ref_url"))
            title = (r.get("ref_title") or "").strip()
            if has_link:
                linked += 1
            elif title:
                # Check for polluted title (author names / citation numbers).
                # Polluted titles can't match in Crossref → route to full reparse.
                if LLMReferenceParser._looks_like_polluted_title(
                    title, r.get("ref_authors", "")
                ):
                    no_title += 1
                else:
                    title_only += 1
            else:
                no_title += 1

        unlinked = title_only + no_title
        if unlinked == 0:
            continue

        entry = {
            "file_name": file_name,
            "total": total,
            "linked": linked,
            "title_only": title_only,
            "no_title": no_title,
        }

        if no_title > 0:
            full_reparse.append(entry)
        else:
            link_only.append(entry)

    full_reparse.sort(key=lambda p: (-(p["title_only"] + p["no_title"]), p["file_name"]))
    link_only.sort(key=lambda p: (-p["title_only"], p["file_name"]))

    return {
        "full_reparse": full_reparse,
        "link_only": link_only,
        "total_papers": len(all_stats),
    }


async def repair_links_for_paper(
    file_name: str,
    llm_config: Dict[str, Any],
    enable_fallback_search: bool = True,
) -> Dict[str, Any]:
    """Lightweight link-only repair: re-run link enrichment on stored refs.

    Loads existing references from paper_doc_stats.json, converts them back to
    Reference objects, and re-runs PaperLinkResolver enrichment. No LLM extraction,
    no PyMuPDF — only API calls to Crossref/OpenAlex/arXiv.

    Args:
        file_name: Paper file name (key in paper_doc_stats.json).
        llm_config: LLM config dict for optional polluted-title re-extraction.
        enable_fallback_search: Enable Semantic Scholar fallback for unresolved refs.

    Returns:
        {"file_name": str, "total": int, "linked_before": int, "linked_after": int,
         "newly_linked": int, "error": str | None}
    """
    all_stats = _load_paper_doc_stats()
    paper_stats = all_stats.get(file_name)
    if not paper_stats:
        return {"file_name": file_name, "error": f"Paper not found in stats: {file_name}"}

    refs_dict = paper_stats.get("references", {})
    if not isinstance(refs_dict, dict) or not refs_dict:
        return {"file_name": file_name, "error": "No references to repair"}

    # Count linked before
    linked_before = 0
    refs: list[Reference] = []
    for ref_id, r in refs_dict.items():
        if not isinstance(r, dict):
            continue
        if r.get("ref_doi") or r.get("ref_arxiv_url") or r.get("ref_url"):
            linked_before += 1
        refs.append(Reference(
            ref_id=ref_id,
            raw_text=r.get("raw_text", ""),
            ref_title=r.get("ref_title", ""),
            ref_authors=r.get("ref_authors", ""),
            ref_year=r.get("ref_year"),
            ref_doi=r.get("ref_doi"),
            ref_venue=r.get("ref_venue"),
            ref_arxiv_url=r.get("ref_arxiv_url"),
            ref_url=r.get("ref_url"),
            ref_source_arxiv_id=r.get("ref_source_arxiv_id"),
        ))

    total = len(refs)
    if total == 0:
        return {"file_name": file_name, "error": "No references to repair"}

    # Only repair refs that currently have no link
    refs_to_repair = [r for r in refs if not r.ref_doi and not r.ref_arxiv_url and not r.ref_url]
    if not refs_to_repair:
        return {
            "file_name": file_name, "total": total,
            "linked_before": linked_before, "linked_after": linked_before,
            "newly_linked": 0, "error": None,
        }

    # Run enrichment
    from .paper_link_resolver import PaperLinkResolver

    link_resolver = PaperLinkResolver(
        enable_crossref=True,
        enable_openalex=True,
        enable_arxiv_library=True,
        log_prefix=f"[LinkRepair:{file_name}]",
    )
    parser = LLMReferenceParser(llm_config, link_resolver=link_resolver)
    await parser._enrich_references(refs_to_repair, enable_fallback_search=enable_fallback_search)

    # Count linked after
    linked_after = sum(1 for r in refs if r.ref_doi or r.ref_arxiv_url or r.ref_url)
    newly_linked = linked_after - linked_before

    # Save updated refs back to JSON
    _save_paper_doc_stats(file_name, refs)

    logger.info(
        f"📝 [LinkRepair] {file_name}: {linked_before}→{linked_after} linked "
        f"(+{newly_linked}), {total - linked_after} still unlinked"
    )

    return {
        "file_name": file_name,
        "total": total,
        "linked_before": linked_before,
        "linked_after": linked_after,
        "newly_linked": newly_linked,
        "error": None,
    }


async def process_references_with_llm(
    pdf_path: str,
    chunks: List[Any],
    text: str,
    llm_config: Dict[str, Any],
    arxiv_client: Any = None,
    enable_fallback_search: bool = True,
    skip_resolution: bool = False,
) -> Tuple[List[Reference], List[Any]]:
    """
    使用 LLM 解析参考文献并建立引用关联

    支持正文+附录参考文献的PDF，会分拆处理后合并。

    Args:
        pdf_path: PDF 文件路径
        chunks: 分块后的 Node 列表
        text: PDF 原始文本
        llm_config: LLM 配置字典，包含 model、api_base、api_key
        arxiv_client: arXiv MCP 客户端（可选）
        enable_fallback_search: 启用网络搜索回退（仅 reparse/repair 命令时启用）

    Returns:
        (references列表, 更新后的chunks列表)
    """
    # 1. 提取所有参考文献部分
    ref_sections = _find_all_reference_sections(text)

    if not ref_sections:
        logger.debug("📝 未找到参考文献部分")
        return [], chunks

    # 2. 初始化多源链接解析器 + LLM 解析器
    link_resolver = PaperLinkResolver(
        enable_crossref=True,
        enable_openalex=True,
        enable_arxiv_library=True,
        log_prefix="[PaperLinkResolver:ref]",
    )
    llm_parser = LLMReferenceParser(llm_config, arxiv_client, link_resolver=link_resolver)
    all_references = []

    # ref_1, ref_2, ref_3... 自然顺序已经是正确的处理顺序
    section_names = sorted(ref_sections.keys())

    logger.info(f"📝 发现 {len(ref_sections)} 个参考文献部分: {section_names}")

    # 全局序号偏移
    global_offset = 0

    for section_name in section_names:
        ref_section = ref_sections[section_name]
        logger.info(f"📝 处理 {section_name} 参考文献，字符数: {len(ref_section)}")

        # 调用 LLM 解析
        refs = await llm_parser.parse_reference_section(
            ref_section, enable_fallback_search=enable_fallback_search, skip_resolution=skip_resolution
        )

        if not refs:
            logger.warning(f"⚠️ {section_name} 参考文献 LLM 解析失败")
            continue

        # 重新编号（加上全局偏移）
        for ref in refs:
            global_offset += 1
            ref.ref_id = f"ref_{global_offset}"

        all_references.extend(refs)
        logger.info(f"📚 {section_name} 解析成功: {len(refs)} 条，当前总计: {global_offset} 条")

    if not all_references:
        logger.warning("⚠️ 所有参考文献解析失败")
        return [], chunks

    logger.info(f"📚 LLM 解析成功: 共 {len(all_references)} 条参考文献")

    # 2.5. 保存论文级参考文献统计到 paper_doc_stats.json
    _save_paper_doc_stats(pdf_path, all_references)

    # 3. 建立引用关联
    linker = CitationLinker()
    chunks = linker.link_citations_to_references(chunks, all_references)

    return all_references, chunks
