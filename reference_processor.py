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

        for i, line in enumerate(ref_lines, 1):
            # 跳过公式引用（如 "[Formula on page 1] $ 36, 39, 57 $"）
            if '[Formula on page' in line:
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

        logger.info(f"📚 共提取到 {len(references)} 条参考文献")
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
        """将参考文献文本分割成单独的行"""
        lines = ref_text.split('\n')
        result = []
        current = []
        MIN_REF_LENGTH = 100  # 最小引用长度阈值

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
                stripped.startswith('[') or
                re.match(r'^\d+\.\s+[A-Z]', stripped)
            )

            # 无序号引用的启发式判断
            if not is_new_ref:
                # 以 URL 开头的行通常是新的完整引用
                if stripped.lower().startswith(('http://', 'https://', 'www.')):
                    is_new_ref = True
                # 如果当前累积文本已经很长（>=300字符），后续文本可能是新引用
                elif current and len(' '.join(current)) >= 300:
                    # 检查前一条引用是否看起来完整（以句号、doi或)结尾）
                    prev_text = ' '.join(current)
                    if prev_text.rstrip().endswith(('.', ')', 'doi:', 'org/', 'com/')):
                        is_new_ref = True
                # 如果一行以大写字母开头，且当前引用已完整，也可能新引用
                elif current and re.match(r'^[A-Z][a-z]+', stripped):
                    prev_text = ' '.join(current)
                    if prev_text.rstrip().endswith('.') and len(prev_text) >= MIN_REF_LENGTH:
                        is_new_ref = True

            if is_new_ref:
                if current:
                    result.append(' '.join(current))
                current = [stripped]
            elif current:
                # 继续之前的引用
                current.append(stripped)

        if current:
            result.append(' '.join(current))

        return result

    def _parse_reference_line(self, line: str, index: int) -> Optional[Reference]:
        """解析单行引用为Reference对象"""
        # 清理空白
        line = ' '.join(line.split())

        # 尝试用各种模式匹配
        for pattern in self.REF_PATTERNS:
            match = pattern.match(line)
            if match:
                groups = match.groups()
                ref_num = groups[0]

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

        # 无法解析，保存原始文本
        logger.debug(f"无法解析引用格式: {line[:100]}...")
        return Reference(
            ref_id=f"ref_{index}",
            raw_text=line,
            ref_title=line[:100],
            ref_authors="",
            ref_year=None,
            ref_doi=None,
            ref_venue=None
        )

    def _extract_title(self, groups: tuple) -> str:
        """从匹配组中提取标题"""
        # groups结构: (ref_num, authors, title, venue/url, year) 或 (ref_num, authors, title, venue, year)
        # title 始终在 index 2
        if len(groups) >= 3:
            return groups[2].strip()
        elif len(groups) >= 2:
            return groups[-1].strip()
        return ""

    def _extract_authors(self, groups: tuple) -> str:
        """从匹配组中提取作者"""
        if len(groups) >= 2:
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
        return title[:100].strip()


class CitationLinker:
    """
    引用链接器

    识别正文中出现的引用标记，如 [1], [1,2], [1-3], [1, 2, 5]
    并与提取的参考文献建立关联
    """

    # 匹配引用标记的正则
    CITATION_PATTERN = re.compile(r'\[(\d+(?:[,\-\s]+\d+)*)\]')

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

        for i, chunk in enumerate(chunks):
            # 获取chunk的索引（用于被引用记录）
            chunk_idx = chunk.metadata.get('chunk_index', i)

            citations = self.find_citations_in_text(chunk.text)

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
                chunk.metadata['cited_references'] = list(cited_refs)
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
                    ref_title=title or raw_text[:100] if raw_text else f"Reference {i}",
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
                    ref_title=title or raw_text[:100] if raw_text else f"Reference {i}",
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
