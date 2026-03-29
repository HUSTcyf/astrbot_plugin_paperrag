"""
引用处理模块 - 学术论文RAG专用

功能：
1. 使用 Grobid 解析结构化引用信息（标题、作者、年份、期刊、DOI）
2. 识别正文中的引用标记（[1], [2], [1-3]等）
3. 建立正文章节与参考文献的双向关联
4. 备选方案：正则表达式解析（当 Grobid 不可用时）

Grobid: https://github.com/kermitt2/grobid
启动: docker run --rm -p 8070:8070 lfoppiano/grobid:latest
"""

import re
import json
import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from astrbot.api import logger


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
    ref_cited_by: List[str] = None  # 正文中引用此文献的位置（chunk索引）

    def __post_init__(self):
        if self.ref_cited_by is None:
            self.ref_cited_by = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "raw_text": self.raw_text,
            "ref_title": self.ref_title,
            "ref_authors": self.ref_authors,
            "ref_year": self.ref_year,
            "ref_doi": self.ref_doi,
            "ref_venue": self.ref_venue
        }


@dataclass
class CitationInText:
    """文本中的引用标记"""
    ref_ids: List[str]  # 解析出的引用ID列表，如 ["ref_1", "ref_2"]
    position: int  # 在文本中的位置
    raw_text: str  # 原始匹配文本，如 "[1,2]"
    context: str  # 上下文（前后50字符）


class ReferenceExtractor:
    """
    参考文献提取器

    使用正则表达式从参考文献部分提取结构化引用信息
    """

    # 常见引用格式的正则表达式
    REF_PATTERNS = [
        # 格式A: 1. Author: Title. Venue (Year) - 年份在末尾
        # 匹配 "1. Smith, A.: Title. CVPR (2024)"
        re.compile(
            r'^(\d+)\.\s*([^:]+):\s*(.+?)\.\s*(.+?)\s*\((\d{4})\)\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式B: 1. Author: Title. Venue (Month Year) - 年份在中间，带月份
        # 匹配 "2. Smith, A.: Title. J. ACM 58(3) (July 2023)"
        re.compile(
            r'^(\d+)\.\s*([^:]+):\s*(.+?)\.\s*(.+?)\s+\(([^)]+)\s+(\d{4})\)\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式C: 1. Author: Title. Venue, Year. - 年份在中间，逗号分隔
        # 匹配 "3. Smith, A.: Title. CVPR, 2024."
        re.compile(
            r'^(\d+)\.\s*([^:]+):\s*(.+?)\.\s*([^,]+),?\s*\((\d{4})\)\.?$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式D: 1. Author: Title. URL (Year) - 带URL，年份在末尾
        # 匹配 "4. Smith: Title. https://... (2024)"
        re.compile(
            r'^(\d+)\.\s*([^:]+):\s*(.+?)\.\s*(https?://\S+)\s*\((\d{4})\)\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式1: [1] Author. Title. Journal, Year.
        # 匹配 "[1] Adam Smith. Attention is all you need. NeurIPS, 2023."
        re.compile(
            r'^\[(\d+)\]\s*([^\]]+?)\.\s*([^,]+?)\.\s*([^,]+?),?\s*(\d{4})',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式2: [1] Author et al. (Year). Title.
        # 匹配 "[1] Vaswani et al. (2017). Attention is all you need."
        re.compile(
            r'^\[(\d+)\]\s*([^\]]+?)\s+et\s+al\.?\s*\((\d{4})\)\.\s*(.+?)\.\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式3: [1] Author, A. & Author, B. (Year). Title.
        # 匹配 "[1] Smith, A. & Jones, B. (2020). A survey of..."
        re.compile(
            r'^\[(\d+)\]\s*([^\]]+?)\s*\((\d{4})\)\.\s*(.+?)\.\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式4: [1] Author - Title (Year)
        # 匹配 "[1] Smith - Title (2020)"
        re.compile(
            r'^\[(\d+)\]\s*([^\]]+?)\s*-\s*(.+?)\s*\((\d{4})\)',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式5: Author, Initial. Year. Title. (无序号)
        # 匹配 "Wu, J. 2020. Essentials of Pattern Recognition: An..."
        re.compile(
            r'^([A-Z][a-z]+,\s*[A-Z])\.\s*(\d{4})\.\s*(.+?)\.\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式6: Author et al. Year. Title. (无序号，带et al)
        # 匹配 "Zhang et al. 2021. Paper title..."
        re.compile(
            r'^([A-Z][a-z]+\s+et\s+al\.?)\s+(\d{4})\.\s*(.+?)\.\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式7: Author, A. & Author, B. Year. Title. (无序号，多作者)
        # 匹配 "Smith, A. & Jones, B. 2022. Paper title..."
        re.compile(
            r'^([A-Z][a-z]+,\s*[A-Z])\.\s*&\s*([A-Z][a-z]+,\s*[A-Z])\.\s*(\d{4})\.\s*(.+?)\.\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
        # 格式8: Author1; Author2. Year. Title. (无序号，分号分隔)
        # 匹配 "Smith, A.; Jones, B. 2022. Paper title..."
        re.compile(
            r'^([A-Z][a-z]+,\s*[A-Z](?:\.\s*;\s*[A-Z][a-z]+,\s*[A-Z]\.)*)\s+(\d{4})\.\s*(.+?)\.\s*$',
            re.MULTILINE | re.IGNORECASE
        ),
    ]

    # DOI 正则表达式
    DOI_PATTERN = re.compile(r'(10\.\d{4,}/[^\s"\]>,;]+)')

    # 参考文献部分的常见标题
    REFERENCE_SECTION_KEYWORDS = [
        'references', 'reference', 'bibliography', 'works cited',
        'reference list', 'literature cited'
    ]

    def extract_references(self, text: str) -> List[Reference]:
        """
        从PDF文本中提取所有参考文献

        Args:
            text: PDF解析后的完整文本

        Returns:
            Reference对象列表
        """
        # 1. 找到参考文献部分
        ref_section = self._find_reference_section(text)
        if not ref_section:
            logger.debug("未找到参考文献部分")
            return []

        # 2. 分割成单独的行
        ref_lines = self._split_reference_lines(ref_section)
        logger.debug(f"📝 分割后得到 {len(ref_lines)} 个文本块")

        # 3. 解析每个引用
        references = []
        expected_seq = 1  # 期望的下一个序号
        skipped_formulas = 0
        skipped_tables = 0
        skipped_empty = 0
        parse_failed = 0

        for i, line in enumerate(ref_lines, 1):
            # 跳过公式引用（如 "[Formula on page 1] $ 36, 39, 57 $"）
            if '[Formula on page' in line:
                skipped_formulas += 1
                continue

            # 跳过表格内容
            if self._is_likely_table(line):
                skipped_tables += 1
                logger.debug(f"跳过表格内容: {line[:60]}...")
                continue

            # 跳过空行或太短的行
            if not line or len(line.strip()) < 10:
                skipped_empty += 1
                continue

            ref = self._parse_reference_line(line, i)
            if ref:
                # 验证序号连续性 - 从内容中提取实际序号（如 "1. Author" 或 "[1] Author"）
                import re
                # 匹配 "1. Author" 或 "[1] Author" 格式
                seq_match = re.match(r'^(\d+)\.|\[(\d+)\]', ref.raw_text)
                if seq_match:
                    # group(1) 是 "1." 格式, group(2) 是 "[1]" 格式
                    actual_seq = int(seq_match.group(1) or seq_match.group(2))
                    # 如果序号跳跃过大（超过10），与上一条合并
                    if actual_seq > expected_seq + 10:
                        if references:
                            prev_ref = references[-1]
                            prev_ref.raw_text += ' ' + ref.raw_text
                            if ref.ref_title:
                                prev_ref.ref_title += ' ' + ref.ref_title
                        continue
                    expected_seq = actual_seq + 1

                references.append(ref)
            else:
                parse_failed += 1

        logger.info(f"📚 共提取到 {len(references)} 条参考文献 (跳过公式: {skipped_formulas}, 跳过表格: {skipped_tables}, 跳过空行: {skipped_empty}, 解析失败: {parse_failed})")
        # 打印参考文献详情
        for i, ref in enumerate(references, 1):
            title = ref.ref_title
            authors = ref.ref_authors
            logger.info(f"   [{i}] {title}")
            logger.info(f"       作者: {authors}")
            logger.info(f"       年份: {ref.ref_year or 'N/A'}, DOI: {ref.ref_doi or 'N/A'}")
        return references

    def extract_references_from_pdf(self, pdf_path: str) -> List[Reference]:
        """
        从 PDF 文件提取参考文献

        Args:
            pdf_path: PDF 文件路径

        Returns:
            Reference对象列表
        """
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return self.extract_references(text)
        except Exception as e:
            logger.error(f"❌ PDF 文本提取失败: {e}")
            return []
        return references

    def _find_reference_section(self, text: str) -> Optional[str]:
        """找到参考文献部分"""
        lines = text.split('\n')
        ref_start = -1
        ref_end = len(lines)

        for i, line in enumerate(lines):
            line_stripped = line.strip().lower()
            # 检查是否是参考文献部分的标题（需要是独立的标题行，不是句子中的一部分）
            is_reference_header = False
            for kw in self.REFERENCE_SECTION_KEYWORDS:
                # 精确匹配：整行就是关键词，或者只有关键词+空格
                if line_stripped == kw or line_stripped.startswith(kw + ' '):
                    is_reference_header = True
                    break

            if is_reference_header:
                ref_start = i + 1
                continue

            # 如果已经找到参考文献部分，检查是否到达其他部分
            if ref_start >= 0:
                # 常见的新章节标题（表示参考文献部分结束）
                if re.match(r'^(appendix|acknowledg|author|bio|supplementary)', line_stripped):
                    ref_end = i
                    break

        if ref_start >= 0 and ref_start < ref_end:
            return '\n'.join(lines[ref_start:ref_end])

        return None

    def _split_reference_lines(self, ref_text: str) -> List[str]:
        """
        将参考文献文本分割成单独的行。

        注意：此方法主要用于有明确编号格式的参考文献（如 [1]、1.）。
        无序号格式的参考文献分割现在由 LLM（parse_reference_section）处理。
        """
        lines = ref_text.split('\n')
        result = []
        current = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 跳过纯数字行（页码碎片）
            if re.match(r'^\d+$', stripped):
                continue

            # 检查是否是新引用的开始
            # 格式: [1] Author... 或 1. Author... 或 \bibitem{1}...
            is_new_ref = (
                re.match(r'^\[\[\{]?\d+[\]]?', stripped) or
                stripped.startswith(r'\bibitem{') or
                re.match(r'^\d+\.\s+[A-Z]', stripped)
            )

            # 无序号但以 URL 开头的行
            if not is_new_ref and stripped.lower().startswith(('http://', 'https://', 'www.')):
                is_new_ref = True

            if is_new_ref:
                if current:
                    result.append(' '.join(current))
                current = [stripped]
            elif current:
                # 继续之前的引用，合并时处理连字符断行
                prev = current[-1]
                if prev.endswith('-'):
                    current[-1] = prev[:-1] + stripped
                else:
                    current.append(stripped)

        if current:
            result.append(' '.join(current))

        return result

    def _parse_reference_line(self, line: str, index: int) -> Optional[Reference]:
        """解析单行引用为Reference对象"""
        # 清理空白
        line = ' '.join(line.split())

        # 合并连字符断行（如 "Accessi- \n ble" → "Accessible"）
        line = re.sub(r'(\w)-\s+(\w)', r'\1\2', line)

        # 尝试用各种模式匹配
        for pattern in self.REF_PATTERNS:
            match = pattern.match(line)
            if match:
                groups = match.groups()

                # 判断是否有序号（原有格式）还是无序号（新格式）
                # 有序号格式: groups[0] 是数字
                # 无序号格式: groups[0] 是作者名（字母开头）
                if groups[0].isdigit():
                    ref_num = groups[0]
                else:
                    ref_num = str(index)  # 使用行号作为ref_num

                # 提取年份 - 在groups中查找纯数字的4位数
                year = None
                for g in reversed(groups):
                    if g and g.isdigit() and len(g) == 4:
                        year = int(g)
                        break

                ref = Reference(
                    ref_id=f"ref_{ref_num}",
                    raw_text=line,
                    ref_title=self._extract_title(groups),
                    ref_authors=self._extract_authors(groups),
                    ref_year=year,
                    ref_doi=self._extract_doi(line),
                    ref_venue=self._extract_venue(groups) if len(groups) > 3 else None
                )
                return ref

        # 如果没有匹配到标准格式，尝试提取DOI和基本信心
        doi = self._extract_doi(line)
        if doi:
            return Reference(
                ref_id=f"ref_{index}",
                raw_text=line,
                ref_title=self._extract_fallback_title(line),
                ref_authors="",
                ref_year=self._extract_year(line),
                ref_doi=doi,
                ref_venue=None
            )

        # 无法解析，检查是否看起来像有效标题
        fallback_title = self._extract_fallback_title(line)
        # 如果提取的"标题"实际上是作者名（如单个姓氏），则不使用
        if self._is_likely_author_name(fallback_title):
            fallback_title = ""  # 留空，这样在统计时会被过滤掉

        # 如果看起来像表格内容，则不使用
        if self._is_likely_table(line):
            logger.debug(f"跳过表格内容: {line[:50]}...")
            fallback_title = ""  # 留空，这样在统计时会被过滤掉

        logger.debug(f"无法解析引用格式: {line}... -> fallback_title='{fallback_title if fallback_title else '(empty)'}...'")
        return Reference(
            ref_id=f"ref_{index}",
            raw_text=line,
            ref_title=fallback_title,
            ref_authors="",
            ref_year=None,
            ref_doi=None,
            ref_venue=None
        )

    def _extract_title(self, groups: tuple) -> str:
        """从匹配组中提取标题"""
        # groups结构:
        # - 有序号格式: (ref_num, authors, title, venue/url, year) 或 (ref_num, authors, title, venue, year)
        # - 无序号格式5/6/8: (authors, year, title)
        # - 无序号格式7: (author1, author2, year, title)
        # title 通常在 index 2 或 3（取决于格式）
        if len(groups) == 3:
            # 格式5/6/8: (authors, year, title)
            return groups[2].strip()
        elif len(groups) == 4:
            # 格式7: (author1, author2, year, title)
            return groups[3].strip()
        elif len(groups) >= 3:
            return groups[2].strip()
        elif len(groups) >= 2:
            return groups[-1].strip()
        return ""

    def _extract_authors(self, groups: tuple) -> str:
        """从匹配组中提取作者"""
        if len(groups) == 4:
            # 格式7: (author1, author2, year, title) - 多作者用 & 合并
            return groups[0].strip() + " & " + groups[1].strip()
        elif len(groups) == 3:
            # 格式5/6/8: (authors, year, title) - 作者在 groups[0]
            return groups[0].strip()
        elif len(groups) >= 2:
            return groups[1].strip()
        return ""

    def _extract_venue(self, groups: tuple) -> Optional[str]:
        """从匹配组中提取期刊/会议"""
        if len(groups) >= 4:
            # groups[3] 是 venue 或 URL
            # 对于Format 0, groups[3]是URL而不是venue,此时venue为空
            venue = groups[3].strip()
            # 如果是URL格式（包含/或http），则不是venue
            if venue and not venue.startswith('http') and not venue.startswith('https'):
                return venue
        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """从文本中提取DOI"""
        match = self.DOI_PATTERN.search(text)
        return match.group(1) if match else None

    def _extract_year(self, text: str) -> Optional[int]:
        """从文本中提取年份"""
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        return int(year_match.group()) if year_match else None

    def _extract_fallback_title(self, text: str) -> str:
        """当无法解析时，提取一个近似的标题"""
        # 移除 [数字] 前缀和 DOI
        title = re.sub(r'^\[\d+\]\s*', '', text)
        title = re.sub(r'10\.\d{4,}/[^\s]+', '', title)
        # 取前100字符作为标题
        return title.strip()

    def _is_likely_author_name(self, text: str) -> bool:
        """
        检测文本是否可能是作者名而不是论文标题

        Args:
            text: 待检测的文本

        Returns:
            True 如果文本看起来像作者名，False 如果看起来像标题
        """
        if not text or len(text) < 3:
            return True

        text = text.strip()

        # 检查是否包含 "et al"（典型的作者引用格式）
        if 'et al' in text.lower():
            return True

        # 检查是否匹配常见的作者名格式：单个姓氏、名字首字母缩写
        # 例如: "Levine", "Smith, J.", "Wang, L."
        author_patterns = [
            r'^[A-Z][a-z]+$',  # 单个单词，首字母大写（可能是姓氏如 "Levine"）
            r'^[A-Z][a-z]+,\s*[A-Z]\.$',  # "Smith, J." 格式
            r'^[A-Z][a-z]+,\s*[A-Z]\.\s*[A-Z]\.$',  # "Smith, J. K." 格式
            r'^[A-Z][a-z]+\s+et\s+al\.?$',  # "Smith et al." 格式
        ]
        for pattern in author_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # 如果文本以方括号+数字开头（如 "[2] Angel Chang..."），这不是标题
        if re.match(r'^\[\d+\]', text):
            return True

        # 如果文本很短（<=3个单词）且全是字母，可能是作者名
        words = text.split()
        if len(words) <= 3 and len(words) > 0:
            # 检查是否全是简单的姓名格式
            all_simple_names = True
            for word in words:
                # 移除非字母字符检查是否是简单的名字/姓氏
                clean_word = re.sub(r'[^a-zA-Z]', '', word)
                if not clean_word:
                    continue
                # 简单名字通常首字母大写，其余小写，长度2-15
                if not (re.match(r'^[A-Z][a-z]{1,14}$', clean_word)):
                    all_simple_names = False
                    break
            if all_simple_names and len(words) <= 2:
                return True

        return False

    def _is_likely_table(self, text: str) -> bool:
        """
        检测文本是否可能是表格内容而不是论文标题

        Args:
            text: 待检测的文本

        Returns:
            True 如果文本看起来像表格内容
        """
        if not text:
            return False

        text = text.strip()

        # 统计管道符数量（表格的典型特征）
        pipe_count = text.count('|')

        # 如果有多个管道符，且在同一行或相邻行，可能是表格
        if pipe_count >= 3:
            # 检查是否包含表格相关的箭头符号或数字组合
            table_indicators = ['↑', '↓', '←', '→', ' LPIPS', ' SSIM', ' PSNR', 'RGB', ' FPS']
            has_table_indicator = any(indicator in text for indicator in table_indicators)

            # 检查是否包含大量数字（可能是表格的数值列）
            numbers_found = re.findall(r'\d+\.?\d*', text)
            has_many_numbers = len(numbers_found) >= 3

            # 检查是否包含时间格式（如 4min28s, 10min3s 等）
            time_patterns = [
                r'\d+min\d+s',  # 如 4min28s
                r'\d+min',      # 如 33min
                r'\d+s',        # 如 98s
            ]
            has_time_format = any(re.search(p, text) for p in time_patterns)

            if has_table_indicator or (has_many_numbers and has_time_format):
                return True

        # 检查是否主要是管道符分隔的数字和标题
        if pipe_count >= 5:
            return True

        # 检查是否包含 "===" 这种 markdown 表格分隔符
        if '===' in text:
            return True

        return False


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
    # 格式: (Smith, 2020), (Smith et al., 2020), Smith (2020), Smith et al. (2020)
    AUTHOR_YEAR_PATTERN = re.compile(
        r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?)\s*,\s*(\d{4})|'
        r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?)\s+\((\d{4})\)'
    )

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
                # 提取第一作者（处理 "Smith, A. & Jones, B." 格式）
                first_author = re.split(r'\s*&\s*|,', authors)[0].strip()
                # 移除末尾的点
                first_author = first_author.rstrip('.')
                key = f"{first_author}{year}"
                author_year_map[key.lower()] = ref.ref_id
                # 也存储 "Smith et al.YEAR" 格式以支持 "Smith et al." 变体
                if 'et al' not in first_author.lower():
                    key_et_al = f"{first_author} et al.{year}"
                    author_year_map[key_et_al.lower()] = ref.ref_id
        return author_year_map

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
            else:
                continue

            # 避免同一位置重复匹配
            if match.start() in seen_positions:
                continue
            seen_positions.add(match.start())

            # 尝试多种作者名格式匹配
            ref_id = None
            author_lower = author.lower()

            # 直接匹配
            if author_lower in author_year_map:
                ref_id = author_year_map[author_lower]
            else:
                # 尝试添加 " et al." 变体
                author_et_al = author_lower + ' et al.'
                if author_et_al in author_year_map:
                    ref_id = author_year_map[author_et_al]
                else:
                    # 尝试只用姓氏匹配
                    surname = author_lower.split()[0] if ' ' in author_lower else author_lower
                    surname_et_al = surname + ' et al.'
                    if surname in author_year_map:
                        ref_id = author_year_map[surname]
                    elif surname_et_al in author_year_map:
                        ref_id = author_year_map[surname_et_al]

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
            更新后的chunks列表，每个chunk添加了cited_references元数据
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

            if cited_refs:
                # 存储完整的引用信息（标题、作者、年份等），而不仅仅是 ref_id
                chunk.metadata['cited_references'] = [
                    {
                        "ref_id": rid,
                        "ref_title": ref_map[rid].ref_title,
                        "ref_authors": ref_map[rid].ref_authors,
                        "ref_year": ref_map[rid].ref_year,
                        "ref_doi": ref_map[rid].ref_doi,
                        "ref_venue": ref_map[rid].ref_venue,
                    }
                    for rid in cited_refs
                ]
            else:
                chunk.metadata['cited_references'] = []

        return chunks


def process_references_and_citations(
    text: str,
    chunks: List[Any]
) -> Tuple[List[Reference], List[Any]]:
    """
    完整的引用处理流程

    Args:
        text: PDF原始文本
        chunks: 分块后的Node列表

    Returns:
        (references列表, 更新后的chunks列表)
    """
    # 1. 提取参考文献
    extractor = ReferenceExtractor()
    references = extractor.extract_references(text)

    # 2. 建立引用关联
    if references:
        linker = CitationLinker()
        chunks = linker.link_citations_to_references(chunks, references)

    return references, chunks


class GrobidReferenceParser:
    """
    使用 Grobid 解析参考文献

    Grobid 是一个专门的学术文献解析工具，精度高：
    - GitHub: https://github.com/kermitt2/grobid
    - 启动: docker run --rm -p 8070:8070 lfoppiano/grobid:latest
    """

    GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
    TIMEOUT = 60

    def __init__(self):
        self.grobid_available = self._check_grobid()

    def _check_grobid(self) -> bool:
        """检查 Grobid 服务是否可用"""
        try:
            response = requests.get(
                f"{self.GROBID_URL}/api/health",
                timeout=5
            )
            if response.status_code == 200:
                logger.info("✅ Grobid 服务可用")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Grobid 服务不可用: {e}")
            logger.info("💡 启动 Grobid: docker run --rm -p 8070:8070 lfoppiano/grobid:latest")
        return False

    def parse_pdf(self, pdf_path: str) -> List[Reference]:
        """
        使用 Grobid 解析 PDF 文件中的参考文献

        Args:
            pdf_path: PDF 文件路径

        Returns:
            Reference 对象列表
        """
        if not self.grobid_available:
            logger.warning("⚠️ Grobid 不可用，参考文献解析可能不完整")
            return []

        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (Path(pdf_path).name, f, 'application/pdf')}
                logger.debug(f"📤 发送 PDF 到 Grobid: {pdf_path}")
                logger.debug(f"   URL: {self.GROBID_URL}/api/processReferences")

                response = requests.post(
                    f"{self.GROBID_URL}/api/processReferences",
                    files=files,
                    timeout=self.TIMEOUT
                )

                logger.debug(f"   响应状态码: {response.status_code}")
                logger.debug(f"   响应头 Content-Type: {response.headers.get('Content-Type', 'N/A')}")
                logger.debug(f"   响应内容长度: {len(response.content)} bytes")

            if response.status_code != 200:
                logger.error(f"❌ Grobid 解析失败: HTTP {response.status_code}")
                logger.error(f"   响应内容: {response.text[:500] if response.text else 'Empty'}")
                if response.status_code == 406:
                    logger.error("   原因: 服务端无法处理此 PDF 或格式不支持")
                return []

            # 解析 TEI XML 格式的响应
            try:
                root = ET.fromstring(response.content)
                references = self._parse_grobid_tei_response(root)
                logger.info(f"📚 Grobid 提取到 {len(references)} 条参考文献")
                return references
            except ET.ParseError as e:
                logger.error(f"❌ Grobid XML 解析失败: {e}")
                logger.debug(f"   XML 内容前200字符: {response.content[:200]}")
                return []

        except requests.exceptions.Timeout:
            logger.error("❌ Grobid 解析超时")
        except Exception as e:
            logger.error(f"❌ Grobid 解析异常: {e}")
            import traceback
            logger.debug(f"   详细错误: {traceback.format_exc()}")

        return []

    def parse_references_text(self, pdf_path: str) -> List[Reference]:
        """
        从 PDF 的参考文献文本解析（备选方案，当 Grobid 解析失败时）

        Args:
            pdf_path: PDF 文件路径

        Returns:
            Reference 对象列表
        """
        extractor = ReferenceExtractor()
        return extractor.extract_references_from_pdf(pdf_path)

    def _parse_grobid_response(self, data: dict) -> List[Reference]:
        """解析 Grobid JSON 响应"""
        references = []

        grobid_refs = data.get('references', [])
        if not grobid_refs:
            # 尝试 TEI 格式
            tei = data.get('tei', {})
            grobid_refs = tei.get('listBibl', {}).get('biblStruct', [])

        for i, ref in enumerate(grobid_refs, 1):
            try:
                # 提取各字段
                title = self._extract_grobid_title(ref)
                authors = self._extract_grobid_authors(ref)
                year = self._extract_grobid_year(ref)
                doi = self._extract_grobid_doi(ref)
                venue = self._extract_grobid_venue(ref)
                raw_text = self._extract_grobid_raw_text(ref)

                reference = Reference(
                    ref_id=f"ref_{i}",
                    raw_text=raw_text,
                    ref_title=title or raw_text if raw_text else f"Reference {i}",
                    ref_authors=authors,
                    ref_year=year,
                    ref_doi=doi,
                    ref_venue=venue
                )
                references.append(reference)

            except Exception as e:
                logger.debug(f"⚠️ 解析第 {i} 条参考文献失败: {e}")
                continue

        return references

    def _parse_grobid_tei_response(self, root: ET.Element) -> List[Reference]:
        """
        解析 Grobid TEI XML 响应

        TEI XML 结构:
        <TEI>
          <text>
            <body>
              <listBibl>
                <biblStruct>...</biblStruct>
              </listBibl>
            </body>
          </text>
        </TEI>
        """
        references = []

        # 定义 TEI 命名空间
        namespaces = {
            'tei': 'http://www.tei-c.org/ns/1.0'
        }

        # 查找所有 biblStruct 元素
        bibl_structures = root.findall('.//tei:listBibl/tei:biblStruct', namespaces)
        logger.debug(f"   TEI: 找到 {len(bibl_structures)} 个 biblStruct (带命名空间)")

        if not bibl_structures:
            # 尝试不带命名空间的查找
            bibl_structures = root.findall('.//listBibl/biblStruct')
            logger.debug(f"   TEI: 找到 {len(bibl_structures)} 个 biblStruct (不带命名空间)")

        if not bibl_structures:
            # 尝试查找 root 标签名
            logger.debug(f"   TEI: root.tag = {root.tag}")
            logger.debug(f"   TEI: root.text = {root.text[:200] if root.text else 'None'}...")
            # 打印所有子元素
            for i, child in enumerate(root):
                logger.debug(f"   TEI child[{i}]: tag={child.tag}, text={child.text[:50] if child.text else 'None'}...")

        for i, bibl in enumerate(bibl_structures, 1):
            try:
                # 提取各字段
                title = self._extract_title_from_tei(bibl, namespaces)
                authors = self._extract_authors_from_tei(bibl, namespaces)
                year = self._extract_year_from_tei(bibl, namespaces)
                doi = self._extract_doi_from_tei(bibl, namespaces)
                venue = self._extract_venue_from_tei(bibl, namespaces)
                raw_text = self._extract_raw_text_from_tei(bibl)

                reference = Reference(
                    ref_id=f"ref_{i}",
                    raw_text=raw_text,
                    ref_title=title or raw_text if raw_text else f"Reference {i}",
                    ref_authors=authors,
                    ref_year=year,
                    ref_doi=doi,
                    ref_venue=venue
                )
                references.append(reference)

            except Exception as e:
                logger.debug(f"⚠️ 解析第 {i} 条参考文献失败: {e}")
                continue

        return references

    def _extract_title_from_tei(self, bibl: ET.Element, namespaces: dict) -> str:
        """从 TEI biblStruct 中提取标题"""
        # 尝试 analytic/title
        analytic = bibl.find('tei:analytic', namespaces)
        if analytic is None:
            analytic = bibl.find('.//analytic')

        if analytic is not None:
            title = analytic.find('tei:title', namespaces)
            if title is None:
                title = analytic.find('.//title')
            if title is not None:
                return title.text or ''

        # 尝试 monogr/title
        monogr = bibl.find('tei:monogr', namespaces)
        if monogr is None:
            monogr = bibl.find('.//monogr')

        if monogr is not None:
            title = monogr.find('tei:title', namespaces)
            if title is None:
                title = monogr.find('.//title')
            if title is not None:
                return title.text or ''

        return ''

    def _extract_authors_from_tei(self, bibl: ET.Element, namespaces: dict) -> str:
        """从 TEI biblStruct 中提取作者"""
        authors = []

        # 查找所有 author 元素
        author_elems = bibl.findall('.//tei:author', namespaces)
        if not author_elems:
            author_elems = bibl.findall('.//author')

        for author in author_elems:
            persName = author.find('tei:persName', namespaces)
            if persName is None:
                persName = author.find('.//persName')

            if persName is not None:
                surname = persName.find('tei:surname', namespaces)
                if surname is None:
                    surname = persName.find('.//surname')
                forename = persName.find('tei:forename', namespaces)
                if forename is None:
                    forename = persName.find('.//forename')

                name_parts = []
                if forename is not None and forename.text:
                    name_parts.append(forename.text)
                if surname is not None and surname.text:
                    name_parts.append(surname.text)

                if name_parts:
                    authors.append(' '.join(name_parts))
            else:
                # 如果没有 persName，尝试直接获取文本
                if author.text:
                    authors.append(author.text.strip())

        return ', '.join(authors)

    def _extract_year_from_tei(self, bibl: ET.Element, namespaces: dict) -> Optional[int]:
        """从 TEI biblStruct 中提取年份"""
        # 尝试 imprint/date
        imprint = bibl.find('.//tei:imprint', namespaces)
        if imprint is None:
            imprint = bibl.find('.//imprint')

        if imprint is not None:
            date = imprint.find('tei:date', namespaces)
            if date is None:
                date = imprint.find('.//date')

            if date is not None:
                # 尝试从 when 属性获取
                when = date.get('when', '')
                if when and len(when) >= 4:
                    try:
                        return int(when[:4])
                    except ValueError:
                        pass

                # 尝试从文本获取
                if date.text:
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', date.text)
                    if year_match:
                        return int(year_match.group())

        return None

    def _extract_doi_from_tei(self, bibl: ET.Element, namespaces: dict) -> Optional[str]:
        """从 TEI biblStruct 中提取 DOI"""
        # 查找 idno 元素
        idnos = bibl.findall('.//tei:idno', namespaces)
        if not idnos:
            idnos = bibl.findall('.//idno')

        for idno in idnos:
            idno_type = idno.get('type', '')
            if 'doi' in idno_type.lower() or idno.text.startswith('10.'):
                return idno.text or ''

        # 尝试从 note 元素查找 DOI
        notes = bibl.findall('.//tei:note', namespaces)
        if not notes:
            notes = bibl.findall('.//note')

        for note in notes:
            if note.text and 'doi' in note.text.lower():
                import re
                doi_match = re.search(r'10\.\d{4,}/[^\s]+', note.text)
                if doi_match:
                    return doi_match.group()

        return None

    def _extract_venue_from_tei(self, bibl: ET.Element, namespaces: dict) -> Optional[str]:
        """从 TEI biblStruct 中提取期刊/会议名称"""
        # 尝试 monogr/title
        monogr = bibl.find('tei:monogr', namespaces)
        if monogr is None:
            monogr = bibl.find('.//monogr')

        if monogr is not None:
            title = monogr.find('tei:title', namespaces)
            if title is None:
                title = monogr.find('.//title')
            if title is not None and title.text:
                return title.text.strip()

        return None

    def _extract_raw_text_from_tei(self, bibl: ET.Element) -> str:
        """从 TEI biblStruct 中提取原始文本"""
        # 尝试获取 normalize 元素（原始文本）
        normalize = bibl.find('.//{http://www.tei-c.org/ns/1.0}normalize')
        if normalize is not None and normalize.text:
            return normalize.text.strip()

        # 尝试获取 admin/note 元素
        note = bibl.find('.//{http://www.tei-c.org/ns/1.0}note')
        if note is not None and note.text:
            return note.text.strip()

        # 尝试直接获取文本内容
        if bibl.text:
            return bibl.text.strip()

        # 返回序列化后的文本
        from xml.etree.ElementTree import tostring
        return tostring(bibl, encoding='unicode')[:200]

    def _extract_grobid_title(self, ref: dict) -> str:
        """从 Grobid 数据中提取标题"""
        # 尝试解析 title 元素
        title_elem = ref.get('title') or ref.get('analytic', {}).get('title')
        if isinstance(title_elem, list):
            title_elem = title_elem[0]
        if isinstance(title_elem, dict):
            return title_elem.get('#text', '') or title_elem.get('text', '')
        if isinstance(title_elem, str):
            return title_elem
        return ''

    def _extract_grobid_authors(self, ref: dict) -> str:
        """从 Grobid 数据中提取作者"""
        authors = []

        # 尝试多个位置查找作者
        author_elems = (
            ref.get('author') or
            ref.get('analytic', {}).get('author') or
            ref.get('monogr', {}).get('author') or
            []
        )

        if isinstance(author_elems, dict):
            author_elems = [author_elems]

        for author in author_elems:
            if isinstance(author, dict):
                persName = author.get('persName', {})
                surname = persName.get('surname', '')
                forename = persName.get('forename', '')
                if surname or forename:
                    full_name = f"{forename} {surname}".strip()
                    if full_name:
                        authors.append(full_name)
            elif isinstance(author, str):
                authors.append(author)

        return ', '.join(authors)

    def _extract_grobid_year(self, ref: dict) -> Optional[int]:
        """从 Grobid 数据中提取年份"""
        # 尝试从 date 元素提取
        date = ref.get('date') or ref.get('monogr', {}).get('date')
        if isinstance(date, dict):
            year = date.get('when', '') or date.get('#text', '')
            if year and len(year) >= 4:
                try:
                    return int(year[:4])
                except ValueError:
                    pass

        # 尝试从 monogr 提取
        monogr = ref.get('monogr', {})
        if isinstance(monogr, dict):
            imprint = monogr.get('imprint', {})
            if isinstance(imprint, dict):
                date = imprint.get('date')
                if date:
                    if isinstance(date, str) and len(date) >= 4:
                        try:
                            return int(date[:4])
                        except ValueError:
                            pass
                    elif isinstance(date, dict):
                        year = date.get('when', '') or date.get('#text', '')
                        if year and len(year) >= 4:
                            try:
                                return int(year[:4])
                            except ValueError:
                                pass

        return None

    def _extract_grobid_doi(self, ref: dict) -> Optional[str]:
        """从 Grobid 数据中提取 DOI"""
        # 尝试从 idno 元素提取
        idnos = ref.get('idno', []) or ref.get('monogr', {}).get('idno', [])
        if isinstance(idnos, dict):
            idnos = [idnos]

        for idno in idnos:
            if isinstance(idno, dict):
                if idno.get('type') == 'DOI':
                    return idno.get('#text', '') or idno.get('text', '')
            elif isinstance(idno, str) and idno.startswith('10.'):
                return idno

        # 尝试从 notes 提取
        notes = ref.get('note', [])
        if isinstance(notes, dict):
            notes = [notes]
        for note in notes:
            if isinstance(note, str) and 'doi' in note.lower():
                doi_match = re.search(r'10\.\d{4,}/[^\s]+', note)
                if doi_match:
                    return doi_match.group()

        return None

    def _extract_grobid_venue(self, ref: dict) -> Optional[str]:
        """从 Grobid 数据中提取期刊/会议名称"""
        monogr = ref.get('monogr', {})
        if isinstance(monogr, dict):
            # 尝试 title 元素
            title = monogr.get('title')
            if isinstance(title, list):
                title = title[0]
            if isinstance(title, dict):
                return title.get('#text', '') or title.get('text', '')
            if isinstance(title, str):
                return title

        return None

    def _extract_grobid_raw_text(self, ref: dict) -> str:
        """从 Grobid 数据中提取原始文本"""
        # 尝试多个位置
        raw = (
            ref.get('rawText') or
            ref.get('note') or
            ref.get('#text') or
            ''
        )

        if isinstance(raw, list):
            raw = ' '.join(str(r) for r in raw if r)
        if isinstance(raw, dict):
            raw = raw.get('#text', '') or raw.get('text', '')

        return str(raw) if raw else ''


def process_references_and_citations_grobid(
    pdf_path: str,
    chunks: List[Any],
    text: str = "",
    use_grobid: bool = False
) -> Tuple[List[Reference], List[Any]]:
    """
    参考文献处理流程：正则解析为主，Grobid 可选补充

    策略：
    1. 首先使用正则表达式解析（对大多数学术论文效果好）
    2. 如果 use_grobid=True 且 Grobid 可用，才使用 Grobid
    3. 合并结果并去重

    Args:
        pdf_path: PDF 文件路径
        chunks: 分块后的 Node 列表
        text: PDF 原始文本（可选）
        use_grobid: 是否使用 Grobid（默认关闭）

    Returns:
        (references列表, 更新后的chunks列表)
    """
    # 1. 首先使用正则表达式解析（更可靠）
    extractor = ReferenceExtractor()
    references = []

    if text:
        references = extractor.extract_references(text)
    else:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            references = extractor.extract_references(text)
        except Exception as e:
            logger.error(f"❌ PDF 文本提取失败: {e}")

    regex_count = len(references)
    if references:
        logger.info(f"📝 正则解析提取到 {regex_count} 条参考文献")
    else:
        logger.debug(f"📝 正则解析提取到 {regex_count} 条参考文献")

    # 2. 如果启用 Grobid，尝试 Grobid 解析作为补充
    if use_grobid:
        grobid_parser = GrobidReferenceParser()
        grobid_refs = []

        if grobid_parser.grobid_available:
            try:
                grobid_refs = grobid_parser.parse_pdf(pdf_path)
                grobid_count = len(grobid_refs)
                logger.debug(f"📚 Grobid 解析提取到 {grobid_count} 条参考文献")

                # 如果 Grobid 结果更多或不同，合并去重
                if grobid_count > regex_count * 0.5 and grobid_count > 0:
                    logger.info(f"🔄 合并正则({regex_count}) + Grobid({grobid_count}) 结果")
                    references = _merge_reference_lists(references, grobid_refs)
                elif grobid_count > 0 and regex_count == 0:
                    # 正则完全失败，使用 Grobid 结果
                    logger.info("📝 正则解析失败，使用 Grobid 结果")
                    references = grobid_refs

            except Exception as e:
                logger.debug(f"⚠️ Grobid 解析异常: {e}")
        else:
            logger.debug("⚠️ Grobid 不可用，跳过 Grobid 解析")
    else:
        logger.debug("📝 Grobid 未启用，使用正则解析")

    # 3. 建立引用关联
    if references:
        linker = CitationLinker()
        chunks = linker.link_citations_to_references(chunks, references)

    return references, chunks


def _merge_reference_lists(regex_refs: List[Reference], grobid_refs: List[Reference]) -> List[Reference]:
    """
    合并两个参考文献列表并去重

    Args:
        regex_refs: 正则解析的参考文献
        grobid_refs: Grobid 解析的参考文献

    Returns:
        合并去重后的参考文献列表
    """
    # 使用 DOI 和标题作为去重依据
    seen_titles = set()
    merged = []

    # 先添加正则解析的结果（通常更完整）
    for ref in regex_refs:
        title_lower = ref.ref_title.lower() if ref.ref_title else ""
        if title_lower and title_lower not in seen_titles:
            seen_titles.add(title_lower)
            merged.append(ref)

    # 添加 Grobid 中不同的结果
    for ref in grobid_refs:
        title_lower = ref.ref_title.lower() if ref.ref_title else ""
        if title_lower and title_lower not in seen_titles:
            seen_titles.add(title_lower)
            merged.append(ref)

    return merged


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

请以JSON数组格式返回，不要包含任何其他内容：
[
    {{
        "title": "论文标题",
        "authors": "作者列表",
        "year": "年份",
        "venue": "期刊/会议",
        "doi": "DOI"
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
- title: 论文标题
- authors: 作者（多个作者用逗号分隔，只填作者姓名，不填"et al"等）
- year: 年份（4位数字）
- venue: 期刊/会议名称（如果有）
- doi: DOI（如果有，只填DOI号）

请以JSON数组格式返回，只返回一个数组，不要有任何其他内容：
[
    {{
        "title": "论文标题",
        "authors": "作者1, 作者2, 作者3",
        "year": "2021",
        "venue": "期刊或会议名称",
        "doi": "10.xxxx/xxxxx"
    }}
]

参考文献部分：
{ref_section}

只返回JSON数组，不要有任何其他内容："""

    def __init__(
        self,
        llm_config: Dict[str, Any],
        arxiv_client: Any = None
    ):
        """
        初始化 LLM 参考文献解析器

        Args:
            llm_config: LLM 配置字典，包含：
                - model: 模型名称（如 "gpt-4o"）
                - api_base: API 基础 URL
                - api_key: API Key
            arxiv_client: arXiv MCP 客户端，用于查询论文详情
        """
        self.llm_config = llm_config
        self.arxiv_client = arxiv_client
        self._semaphore = None

    async def _call_llm(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """调用 LLM 生成文本（使用官方 OpenAI API），支持重试"""
        import aiohttp
        import asyncio
        import time

        # 获取或创建信号量（限制并发）
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(4)

        url = f"{self.llm_config['api_base']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_config.get('api_key', 'sk-placeholder')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.llm_config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 8192,  # 增加token限制以支持大量参考文献
        }

        logger.info(f"📝 [LLM调用] 开始请求，prompt长度: {len(prompt)} 字符")

        for attempt in range(max_retries):
            logger.info(f"📝 [LLM调用] 尝试 {attempt + 1}/{max_retries}")
            async with self._semaphore:
                logger.info(f"📝 [LLM调用] 获得信号量，开始请求...")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                            logger.info(f"📝 [LLM调用] 收到响应状态码: {resp.status}")
                            if resp.status == 429:
                                # 速率限制，等待后重试
                                retry_after = int(resp.headers.get("Retry-After", 5))
                                logger.warning(f"⚠️ LLM API 速率限制 (429)，{retry_after}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                                await asyncio.sleep(retry_after)
                                continue
                            if resp.status == 500:
                                # 服务器错误，等待后重试
                                logger.warning(f"⚠️ LLM API 服务器错误 (500)，5秒后重试... (尝试 {attempt + 1}/{max_retries})")
                                await asyncio.sleep(5)
                                continue
                            if resp.status != 200:
                                text = await resp.text()
                                logger.warning(f"⚠️ LLM API 请求失败: HTTP {resp.status}, 响应: {text[:500]}")
                                return None
                            result = await resp.json()
                            logger.info(f"📝 [LLM调用] 响应解析成功")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ LLM API 请求超时")
                    await asyncio.sleep(3)
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ LLM API 请求异常: {e}")
                    await asyncio.sleep(3)
                    continue

            # 提取响应内容（在信号量外部执行）
            logger.info(f"📝 [LLM调用] 解析响应内容...")
            try:
                choices = result.get("choices", [])
                if not choices:
                    logger.warning("⚠️ LLM 返回空 choices")
                    return None
                message = choices[0].get("message", {})
                content = message.get("content", "")
                logger.info(f"📝 [LLM调用] 提取到内容长度: {len(content) if content else 0}")
                return content
            except Exception as e:
                logger.warning(f"⚠️ 解析 LLM 响应失败: {e}")
                return None

        logger.warning(f"⚠️ LLM API 重试 {max_retries} 次后仍失败")
        return None

    async def parse_reference_section(
        self,
        ref_section: str,
        ref_id_prefix: str = "ref"
    ) -> List[Reference]:
        """
        解析整段参考文献文本（让LLM自动分割+解析）

        Args:
            ref_section: 参考文献部分的完整文本（可能跨多行）
            ref_id_prefix: ref_id 前缀

        Returns:
            Reference 对象列表
        """
        if not ref_section or not ref_section.strip():
            return []

        logger.info(f"📝 开始 LLM 参考文献解析（整段模式），文本长度: {len(ref_section)} 字符")

        prompt = self.SECTION_PARSE_PROMPT.format(ref_section=ref_section)
        logger.debug(f"📝 [LLM调试] Prompt长度: {len(prompt)} 字符")
        logger.debug(f"📝 [LLM调试] Prompt预览: {prompt[:300]}...")

        try:
            response = await self._call_llm(prompt)
            logger.debug(f"📝 [LLM调试] Raw响应长度: {len(response) if response else 0}")
            logger.debug(f"📝 [LLM调试] Raw响应预览: {response[:500] if response else 'None'}")

            if not response:
                logger.warning("⚠️ LLM 未返回有效响应")
                return []

            # 提取 JSON
            json_str = self._extract_json(response)
            if not json_str:
                logger.warning("⚠️ 无法从 LLM 响应中提取 JSON")
                return []

            parsed_list = json.loads(json_str)
            if not isinstance(parsed_list, list):
                logger.warning(f"⚠️ LLM 返回的不是数组: {type(parsed_list)}")
                return []

            results = []
            for j, parsed in enumerate(parsed_list):
                try:
                    ref = Reference(
                        ref_id=f"{ref_id_prefix}_{j + 1}",
                        raw_text="",  # 整段模式不保留raw_text
                        ref_title=parsed.get("title", ""),
                        ref_authors=parsed.get("authors", ""),
                        ref_year=int(parsed["year"]) if str(parsed.get("year", "")).isdigit() else None,
                        ref_doi=parsed.get("doi") or None,
                        ref_venue=parsed.get("venue") or None
                    )
                    results.append(ref)
                except Exception as e:
                    logger.debug(f"⚠️ 解析第 {j} 条失败: {e}, 数据: {parsed}")
                    continue

            logger.info(f"📚 LLM 解析参考文献: 成功 {len(results)} 条")
            return results

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON 解析失败: {e}")
            return []
        except Exception as e:
            logger.warning(f"⚠️ 参考文献解析失败: {e}")
            return []

    async def parse_references(
        self,
        references: List[str],
        ref_id_prefix: str = "ref"
    ) -> List[Reference]:
        """
        解析参考文献列表

        Args:
            references: 参考文献原始文本列表
            ref_id_prefix: ref_id 前缀

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

        # MCP 参考文献补全默认禁用（如需启用，取消注释以下代码）
        # if self.arxiv_client and valid_results:
        #     await self._enrich_from_arxiv(valid_results)

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
        logger.debug(f"📝 [LLM调试] Prompt长度: {len(prompt)} 字符")
        logger.debug(f"📝 [LLM调试] Prompt前200字符: {prompt[:200]}")

        try:
            response = await self._call_llm(prompt)
            logger.debug(f"📝 [LLM调试] Raw响应长度: {len(response) if response else 0}")
            logger.debug(f"📝 [LLM调试] Raw响应前500字符: {response[:500] if response else 'None'}")

            if not response:
                logger.warning("⚠️ LLM 未返回有效响应")
                return [None] * len(references)

            # 提取 JSON
            json_str = self._extract_json(response)
            logger.debug(f"📝 [LLM调试] 提取的JSON长度: {len(json_str) if json_str else 0}")
            logger.debug(f"📝 [LLM调试] 提取的JSON前300字符: {json_str[:300] if json_str else 'None'}")

            if not json_str:
                logger.warning("⚠️ 无法从 LLM 响应中提取 JSON")
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

    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 字符串"""
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # 尝试提取 markdown 代码块中的 JSON
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            json_str = match.group(1).strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass

        # 尝试找到 JSON 数组或对象
        for match in re.finditer(r'(\[[\s\S]*?\]|\{[\s\S]*?\})', text):
            json_str = match.group(1)
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                continue

        return None

    async def _enrich_from_arxiv(self, references: List[Reference]) -> None:
        """
        通过 arXiv MCP 补充论文信息

        Args:
            references: Reference 对象列表（会被直接修改）
        """
        if not self.arxiv_client:
            return

        for ref in references:
            if not ref:
                continue

            # 优先使用 DOI 搜索
            search_query = None
            if ref.ref_doi:
                search_query = ref.ref_doi
            elif ref.ref_title and len(ref.ref_title) > 5:
                search_query = ref.ref_title

            if not search_query:
                continue

            try:
                result = await self.arxiv_client.call_tool_with_reconnect(
                    tool_name="search_arxiv",
                    arguments={"query": search_query, "max_results": 3}
                )

                if not result or not result.get("results"):
                    continue

                # 找到最匹配的论文
                for paper in result.get("results", []):
                    paper_title = paper.get("title", "")
                    ref_title = ref.ref_title if ref.ref_title else ""

                    if not ref_title or not paper_title:
                        continue

                    # 标题必须完全相同（忽略大小写）
                    if ref_title.lower() != paper_title.lower():
                        continue

                    # 完全匹配，更新 Reference 对象
                    if paper.get("authors"):
                        ref.ref_authors = ", ".join(paper["authors"])
                    if paper.get("published_date"):
                        year_match = re.search(r'(\d{4})', paper["published_date"])
                        if year_match:
                            ref.ref_year = int(year_match.group(1))
                    if paper.get("doi"):
                        ref.ref_doi = paper.get("doi")

                    logger.debug(f"📝 arXiv 补充论文信息: {ref.ref_title}")
                    break

            except Exception as e:
                logger.debug(f"⚠️ arXiv 查询失败: {e}")
                continue


async def process_references_with_llm(
    pdf_path: str,
    chunks: List[Any],
    text: str,
    llm_config: Dict[str, Any],
    arxiv_client: Any = None
) -> Tuple[List[Reference], List[Any]]:
    """
    使用 LLM 解析参考文献并建立引用关联

    Args:
        pdf_path: PDF 文件路径
        chunks: 分块后的 Node 列表
        text: PDF 原始文本
        llm_config: LLM 配置字典，包含 model、api_base、api_key
        arxiv_client: arXiv MCP 客户端（可选）

    Returns:
        (references列表, 更新后的chunks列表)
    """
    # 1. 使用正则表达式提取参考文献部分（作为后备）
    extractor = ReferenceExtractor()
    ref_section = extractor._find_reference_section(text)

    if not ref_section:
        logger.debug("📝 未找到参考文献部分")
        return [], chunks

    # 📝 调试：查看参考文献部分内容
    logger.debug(f"📝 参考文献部分预览 (前500字符):\n{ref_section[:500]}")
    logger.debug(f"📝 参考文献部分总长度: {len(ref_section)} 字符, {len(ref_section.split(chr(10)))} 行")

    # 2. 直接将整段参考文献文本传给 LLM，让 LLM 自动分割+解析
    llm_parser = LLMReferenceParser(llm_config, arxiv_client)
    references = await llm_parser.parse_reference_section(ref_section)

    if not references:
        logger.warning("⚠️ LLM 解析参考文献失败，使用正则表达式作为后备")
        references = extractor.extract_references(text)
    else:
        logger.info(f"📚 LLM 解析成功: {len(references)} 条参考文献")

    # 4. 建立引用关联
    if references:
        linker = CitationLinker()
        chunks = linker.link_citations_to_references(chunks, references)

    return references, chunks
