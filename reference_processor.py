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
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
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
    ref_cited_by: List[str] = field(default_factory=list)  # 正文中引用此文献的位置（chunk索引）

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



# 参考文献部分的常见标题
REFERENCE_SECTION_KEYWORDS = [
    'references', 'reference', 'bibliography', 'works cited',
    'reference list', 'literature cited'
]


def _find_reference_section(text: str) -> Optional[str]:
    """找到参考文献部分

    策略：
    1. 从 "References"（不区分大小写）标题之后开始
    2. 找到最后一个有效的编号参考文献行
    3. 遇到表格分隔行 | --- | --- | 时截断
    """
    lines_text = text.split('\n')
    ref_start = -1

    # 找到 "References" 标题位置
    for i, line in enumerate(lines_text):
        line_stripped = line.strip().lower()
        for kw in REFERENCE_SECTION_KEYWORDS:
            if line_stripped == kw or line_stripped.startswith(kw + ' '):
                ref_start = i + 1
                break
        if ref_start > 0:
            break

    if ref_start < 0:
        return None

    # 自动检测是否存在行号：检查第一条参考文献是否有双重编号格式
    # 模式如: "[998] [1]" 或 "[123] 1."
    first_line = lines_text[ref_start].strip()
    has_line_numbers = bool(re.match(r'^\[[0-9]+\]\s*\[[0-9]+\]', first_line)) or \
                       bool(re.match(r'^\[[0-9]+\]\s*[0-9]+\.', first_line))

    # 清洗行号格式：如 "[998] [1] ..." -> "[1] ..."
    def clean_line(line: str) -> str:
        if has_line_numbers:
            cleaned = re.sub(r'^\[[0-9]+\]\s*', '', line)
            return cleaned
        return line

    # 找到最后一个编号参考文献行
    # 策略：遇到包含特殊字符的行时停止（如表格分隔符 |、数学公式 $、附录等）
    ref_end = len(lines_text)

    for i, line in enumerate(lines_text[ref_start:], start=ref_start):
        stripped = clean_line(line).strip()

        # 遇到 Markdown 表格分隔行 | --- | --- | 直接截断
        if stripped.startswith('|') and stripped.count('|') >= 3:
            ref_end = i
            break

        # 遇到数学公式行（如 $...$ 或纯公式行）直接截断
        if stripped.startswith('$') or stripped.endswith('$'):
            ref_end = i
            break

        # 遇到附录/补充材料/Acknowledgment时截断
        if re.search(r'\b(Acknowledgment|Appendix|Supplementary Material)\b', stripped, re.IGNORECASE):
            ref_end = i
            break

        # 检查是否有新的参考文献编号（清洗后检查）
        has_ref_number = bool(re.match(r'^\[[0-9]+\]', stripped)) or bool(re.match(r'^[0-9]+\.\s+[A-Z]', stripped))

        if has_ref_number:
            ref_end = i + 1

    if ref_start >= ref_end:
        return None

    # 使用清洗后的行拼接结果
    result_lines = [clean_line(lines_text[i]) for i in range(ref_start, ref_end)]
    result = '\n'.join(result_lines)
    logger.info(f"📝 参考文献提取成功: {len(result)} 字符, {ref_end - ref_start} 行")
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
            "max_tokens": 16384,  # 模型最大支持 16384 tokens
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

        try:
            response = await self._call_llm(prompt)

            if not response:
                logger.warning("⚠️ LLM 未返回有效响应")
                return []

            # 提取 JSON
            json_str = self._extract_json(response)
            if not json_str:
                logger.warning(f"⚠️ 无法从 LLM 响应中提取 JSON，响应长度: {len(response)} 字符")
                logger.warning(f"========== LLM 原始输出 ==========\n{response}\n========== END ==========")
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

    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 字符串"""
        import re

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
                    except:
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
            logger.info(f"[_extract_json] 匹配 {i+1}: 长度={len(json_str)}, 前20字符={repr(json_str[:20])}")
            result, _ = try_parse(json_str)
            if result:
                logger.info(f"[_extract_json] 匹配 {i+1} 解析成功")
                return result

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
    # 1. 使用正则表达式提取参考文献部分
    ref_section = _find_reference_section(text)

    if not ref_section:
        logger.debug("📝 未找到参考文献部分")
        return [], chunks

    # 2. 直接将整段参考文献文本传给 LLM，让 LLM 自动分割+解析
    llm_parser = LLMReferenceParser(llm_config, arxiv_client)
    references = await llm_parser.parse_reference_section(ref_section)

    if not references:
        logger.warning("⚠️ LLM 解析参考文献失败")
    else:
        logger.info(f"📚 LLM 解析成功: {len(references)} 条参考文献")

    # 3. 建立引用关联
    if references:
        linker = CitationLinker()
        chunks = linker.link_citations_to_references(chunks, references)

    return references, chunks
