"""
Author-Year 引用匹配测试脚本

测试场景：
1. 引用格式: [Gortler et al. 1996; Levoy and Hanrahan 1996]
2. 引用格式: (Liu et al., 2023b; Alayrac et al., 2022)
3. 参考文献作者名变体: "Steven J. Gortler, et al." vs "Gortler et al."
4. 同年份后缀: 2023a, 2023b
"""

import re
from typing import List, Dict, Optional, Tuple


class AuthorYearMatcher:
    """增强的 author-year 引用匹配器"""

    def __init__(self):
        # 引用正则 - 方括号格式
        self.BRACKET_CITATION_PATTERN = re.compile(r'\[([^\]]+)\]')

        # 引用正则 - 括号格式
        self.PARENTHETICAL_CITATION_PATTERN = re.compile(r'\(([^)]+)\)')

        # 解析单个 author-year
        # 支持: "Gortler et al. 1996", "Gortler, 1996", "Gortler et al., 1996"
        # 支持: "S. Karamcheti" 或 "S Karamcheti" 缩写格式
        self.AUTHOR_YEAR_SINGLE = re.compile(
            r'([A-Z]\.\s*[A-Z][a-z]+'  # 缩写+全名如 "S. Karamcheti" 或 "S Karamcheti"
            r'|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'  # 多单词作者名
            r')(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?'  # 可选的 et al. 或 and
            r'[\s,]*?'  # 可选逗号/空格（非贪婪，避免吃掉年份前的空格）
            r'(?:[\(\[]?\s*)?'  # 可选的括号
            r'(\d{4}[a-z]?)'  # 年份，可选 a/b 后缀
            r'(?:\s*[\)\]]?\s*$)?'  # 可选的括号结束
        )

    def _extract_first_author_surname(self, authors: str) -> Optional[str]:
        """
        从作者字符串提取第一作者姓氏

        Args:
            authors: 作者字符串，如 "Steven J. Gortler, et al." 或 "Gortler" 或 "Marc Levoy and Pat Hanrahan" 或 "S. Karamcheti"

        Returns:
            姓氏或 None
        """
        if not authors:
            return None

        original = authors
        authors = authors.strip()

        # 处理 "et al." 情况 - 优先处理，取 "et al." 之前的词
        # 使用正则匹配 " et al." 模式，避免匹配到名字中的 "et"（如 Karamcheti）
        et_al_pattern = re.compile(r'\s+et\s+al', re.IGNORECASE)
        match = et_al_pattern.search(authors)
        if match:
            before_et = authors[:match.start()].strip()
            # 去除末尾可能的逗号
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

    def _extract_year_suffix(self, year_str: str) -> Tuple[str, str]:
        """提取年份和后缀"""
        match = re.match(r'(\d{4})([a-z]?)', year_str)
        if match:
            return match.group(1), match.group(2) or ''
        return year_str, ''

    def _build_author_year_map(self, references: List[Dict]) -> Dict[str, str]:
        """
        从参考文献构建 author-year -> ref_id 映射

        Args:
            references: [{"ref_id": "ref_1", "ref_authors": "...", "ref_year": 1996}, ...]

        Returns:
            {lowercase_key: ref_id}
        """
        author_year_map = {}

        for ref in references:
            ref_id = ref.get("ref_id", "")
            authors = ref.get("ref_authors", "").strip()
            year = ref.get("ref_year")

            if not authors or not year:
                continue

            year_str = str(year)
            surname = self._extract_first_author_surname(authors)
            if not surname:
                continue

            surname_lower = surname.lower()

            # 基础格式: gortler1996
            key = f"{surname_lower}{year_str}"
            author_year_map[key] = ref_id

            # 带 et al. 格式: gortler et al.1996
            key_et_al = f"{surname_lower} et al.{year_str}"
            author_year_map[key_et_al] = ref_id

            # 处理年份后缀: 2023a, 2023b
            year_str_a = f"{year_str}a"
            year_str_b = f"{year_str}b"
            author_year_map[f"{surname_lower}{year_str_a}"] = ref_id
            author_year_map[f"{surname_lower}{year_str_b}"] = ref_id
            author_year_map[f"{surname_lower} et al.{year_str_a}"] = ref_id
            author_year_map[f"{surname_lower} et al.{year_str_b}"] = ref_id

        return author_year_map

    def _match_single_author_year(self, author: str, year: str, author_year_map: Dict[str, str]) -> Optional[str]:
        """尝试匹配单个 author-year 到 ref_id"""
        year_str, suffix = self._extract_year_suffix(year)

        # 首先尝试直接匹配（author 已经是姓氏）
        author_lower = author.lower().strip()
        variants = [
            f"{author_lower}{year_str}{suffix}",  # gortler1996a
            f"{author_lower} et al.{year_str}{suffix}",  # gortler et al.1996a
            f"{author_lower}et al.{year_str}{suffix}",  # gortleret al.1996a (无空格)
        ]

        for key in variants:
            if key in author_year_map:
                return author_year_map[key]

        # 如果直接匹配失败，尝试从 author 中提取姓氏（如 "S. Karamcheti" -> "Karamcheti"）
        surname = self._extract_first_author_surname(author)
        if surname and surname.lower() != author_lower:
            surname_lower = surname.lower()
            variants = [
                f"{surname_lower}{year_str}{suffix}",
                f"{surname_lower} et al.{year_str}{suffix}",
                f"{surname_lower}et al.{year_str}{suffix}",
            ]
            for key in variants:
                if key in author_year_map:
                    return author_year_map[key]

        return None

    def find_citations(self, text: str, author_year_map: Dict[str, str]) -> List[Dict]:
        """
        在文本中查找 author-year 引用

        Args:
            text: 文本内容
            author_year_map: author-year -> ref_id 映射

        Returns:
            [{"ref_id": "ref_1", "raw_text": "...", "context": "..."}, ...]
        """
        citations = []
        seen = set()  # 避免重复

        # 1. 处理方括号格式: [Gortler et al. 1996; Levoy and Hanrahan 1996]
        for match in self.BRACKET_CITATION_PATTERN.finditer(text):
            bracket_content = match.group(1).strip()

            # 按分号分割多个引用
            parts = bracket_content.split(';')
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                ref_id = self._parse_and_match(part, author_year_map)
                if ref_id and ref_id not in seen:
                    seen.add(ref_id)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    citations.append({
                        "ref_id": ref_id,
                        "raw_text": match.group(),
                        "context": text[start:end]
                    })

        # 2. 处理括号格式: (Liu et al., 2023b; Alayrac et al., 2022)
        for match in self.PARENTHETICAL_CITATION_PATTERN.finditer(text):
            # 跳过已经是方括号的内容
            if text[match.start() - 1] == '[':
                continue

            paren_content = match.group(1).strip()

            # 按分号分割多个引用
            parts = paren_content.split(';')
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                ref_id = self._parse_and_match(part, author_year_map)
                if ref_id and ref_id not in seen:
                    seen.add(ref_id)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    citations.append({
                        "ref_id": ref_id,
                        "raw_text": match.group(),
                        "context": text[start:end]
                    })

        return citations

    def _parse_and_match(self, citation_text: str, author_year_map: Dict[str, str]) -> Optional[str]:
        """解析并匹配单个引用文本"""
        citation_text = citation_text.strip()

        # 尝试正则匹配
        match = self.AUTHOR_YEAR_SINGLE.search(citation_text)
        if match:
            author = match.group(1).strip()
            year = match.group(2).strip()
            return self._match_single_author_year(author, year, author_year_map)

        # Fallback: 尝试通用格式 "Author[ et al.] YEAR"
        # 支持: "Karamcheti 2024", "Karamcheti et al. 2024", "Karamcheti et al., 2024"
        parts = citation_text.split()
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
                    result = self._match_single_author_year(author_part, year_part, author_year_map)
                    if result:
                        return result

        # Fallback 2: 处理 "Author (YEAR)" 格式
        # 例如: "Gortler (1996)" 或 "Liu et al. (2023)"
        paren_year_pattern = re.compile(
            r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?)\s+\((\d{4}[a-z]?)\)'
        )
        paren_match = paren_year_pattern.search(citation_text)
        if paren_match:
            author = paren_match.group(1).strip()
            year = paren_match.group(2).strip()
            return self._match_single_author_year(author, year, author_year_map)

        return None


def test_basic_matching():
    """测试基础匹配功能"""
    print("=" * 60)
    print("测试 1: 基础 author-year 匹配")
    print("=" * 60)

    matcher = AuthorYearMatcher()

    # 参考文献
    references = [
        {"ref_id": "ref_1", "ref_authors": "Steven J. Gortler, et al.", "ref_year": 1996},
        {"ref_id": "ref_2", "ref_authors": "Marc Levoy and Pat Hanrahan", "ref_year": 1996},
        {"ref_id": "ref_3", "ref_authors": "Tianwei Liu, et al.", "ref_year": 2023},
        {"ref_id": "ref_4", "ref_authors": "Liu Liu, et al.", "ref_year": 2023},
    ]

    author_year_map = matcher._build_author_year_map(references)
    print(f"构建的映射: {author_year_map}")

    # 测试提取姓氏
    test_cases = [
        ("Steven J. Gortler, et al.", "Gortler"),
        ("Marc Levoy and Pat Hanrahan", "Levoy"),
        ("Tianwei Liu, et al.", "Liu"),
    ]

    print("\n提取姓氏测试:")
    for authors, expected in test_cases:
        result = matcher._extract_first_author_surname(authors)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{authors}' -> '{result}' (expected: '{expected}')")

    print()


def test_citation_formats():
    """测试各种引用格式"""
    print("=" * 60)
    print("测试 2: 各种引用格式解析")
    print("=" * 60)

    matcher = AuthorYearMatcher()

    references = [
        {"ref_id": "ref_1", "ref_authors": "Steven J. Gortler, et al.", "ref_year": 1996},
        {"ref_id": "ref_2", "ref_authors": "Marc Levoy and Pat Hanrahan", "ref_year": 1996},
        {"ref_id": "ref_3", "ref_authors": "Tianwei Liu, et al.", "ref_year": 2023},
        {"ref_id": "ref_4", "ref_authors": "Liu Liu, et al.", "ref_year": 2023},
    ]

    author_year_map = matcher._build_author_year_map(references)

    test_texts = [
        # 格式1: 方括号，分号分隔
        "Recent work [Gortler et al. 1996; Levoy and Hanrahan 1996] has shown...",

        # 格式2: 括号，分号分隔，逗号 - 常用格式
        "Previous studies (Liu et al., 2023b; Alayrac et al., 2022) have...",

        # 格式3: 多作者同年
        "Research (Liu et al., 2023a; Liu et al., 2023b) presents...",

        # 格式4: 完整段落引用
        "(Liu et al., 2023b; Chen et al., 2023; Karamcheti et al., 2024; Bai et al., 2023)",

        # 格式5: 无逗号但有括号
        "Similar approaches (Karamcheti et al. 2024) have demonstrated...",

        # 格式6: 缩写+全名格式
        "Following (S. Karamcheti et al. 2024), we propose...",
    ]

    print()
    for i, text in enumerate(test_texts, 1):
        print(f"文本 {i}: {text[:60]}...")

        # 调试：测试每个部分的匹配
        # 1. 测试方括号
        bracket_matches = matcher.BRACKET_CITATION_PATTERN.findall(text)
        if bracket_matches:
            print(f"  方括号内容: {bracket_matches}")

        # 2. 测试单个 author-year 正则
        for bm in bracket_matches:
            for part in bm.split(';'):
                part = part.strip()
                single_match = matcher.AUTHOR_YEAR_SINGLE.search(part)
                print(f"    '{part}' -> AUTHOR_YEAR_SINGLE: {single_match.group() if single_match else 'NO MATCH'}")

        # 3. 测试括号格式
        paren_matches = matcher.PARENTHETICAL_CITATION_PATTERN.findall(text)
        if paren_matches:
            print(f"  括号内容: {paren_matches}")

        citations = matcher.find_citations(text, author_year_map)
        print(f"  匹配到: {[c['ref_id'] for c in citations]}")
        print()


def test_realistic_scenario():
    """测试真实场景"""
    print("=" * 60)
    print("测试 3: 真实场景模拟")
    print("=" * 60)

    matcher = AuthorYearMatcher()

    # 模拟论文的参考文献（来自 2508.09977v2）
    references = [
        {"ref_id": "ref_1", "ref_authors": "A. Gortler, et al.", "ref_year": 1996},
        {"ref_id": "ref_2", "ref_authors": "M. Levoy and P. Hanrahan", "ref_year": 1996},
        {"ref_id": "ref_3", "ref_authors": "J. Liu, et al.", "ref_year": 2023},
        {"ref_id": "ref_4", "ref_authors": "J. Alayrac, et al.", "ref_year": 2022},
        {"ref_id": "ref_5", "ref_authors": "Z. Chen, et al.", "ref_year": 2023},
        {"ref_id": "ref_6", "ref_authors": "S. Karamcheti, et al.", "ref_year": 2024},
        {"ref_id": "ref_7", "ref_authors": "R. Bai, et al.", "ref_year": 2023},
    ]

    author_year_map = matcher._build_author_year_map(references)
    print(f"参考文献数量: {len(references)}")
    print(f"映射键数量: {len(author_year_map)}")
    print(f"映射: {author_year_map}")
    print()

    # 模拟 chunk 文本
    chunk_text = """
    3D Gaussian Splatting (3DGS) [1], [2] has revolutionized novel view synthesis.
    Although several recent surveys [10], [11], [12], [13] have documented the rapid
    development of 3DGS, existing methods still struggle with real-time rendering.

    The approach builds on previous work by Gortler et al. 1996 and Levoy and Hanrahan 1996.
    Recent progress includes Liu et al. 2023 which introduced key optimizations.

    Some recent approaches (Liu et al., 2023b; Alayrac et al., 2022) have achieved
    significant improvements. Others (Chen et al., 2023; Karamcheti et al., 2024; Bai et al., 2023)
    have explored different aspects of the problem.
    """

    citations = matcher.find_citations(chunk_text, author_year_map)
    print(f"在 chunk 中找到 {len(citations)} 个引用:")
    for c in citations:
        print(f"  - {c['ref_id']}: {c['raw_text']}")

    print()


def test_real_pdf():
    """从真实PDF提取文本测试"""
    print("=" * 60)
    print("测试 5: 从真实PDF提取文本")
    print("=" * 60)

    import os
    import sys

    # 添加父目录到路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF 未安装，跳过真实PDF测试")
        return

    # PDF 文件列表
    pdf_dir = "/Volumes/ext/Master/papers"
    pdf_files = [
        "2408.00714v2（SAM2）.pdf",
        "2508.09071v2(GeoVLA).pdf",
        "onthefly_nvs.pdf",
    ]

    matcher = AuthorYearMatcher()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        if not os.path.exists(pdf_path):
            print(f"\n文件不存在: {pdf_path}")
            continue

        print(f"\n{'='*40}")
        print(f"处理: {pdf_file}")
        print(f"{'='*40}")

        try:
            doc = fitz.open(pdf_path)
            all_text = ""
            for page in doc:
                all_text += page.get_text()
            doc.close()

            print(f"  提取文本长度: {len(all_text)} 字符")

            # 查找方括号内的 author-year 引用
            bracket_matches = matcher.BRACKET_CITATION_PATTERN.findall(all_text)
            print(f"  方括号引用数量: {len(bracket_matches)}")

            # 查找括号内的 author-year 引用
            paren_matches = matcher.PARENTHETICAL_CITATION_PATTERN.findall(all_text)
            print(f"  括号引用数量: {len(paren_matches)}")

            # 提取前10个引用样例
            print(f"  前10个方括号引用样例:")
            for i, m in enumerate(bracket_matches[:10]):
                if 'et al' in m or any(c.isdigit() for c in m):
                    print(f"    {i+1}. {m[:80]}")

            # 提取前10个括号引用样例
            print(f"  前10个括号引用样例:")
            for i, m in enumerate(paren_matches[:10]):
                if 'et al' in m or any(c.isdigit() for c in m):
                    print(f"    {i+1}. {m[:80]}")

        except Exception as e:
            print(f"  错误: {e}")


def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试 4: 边界情况")
    print("=" * 60)

    matcher = AuthorYearMatcher()

    # 只有一个作者的参考文献
    references = [
        {"ref_id": "ref_1", "ref_authors": "John Doe", "ref_year": 2020},
    ]

    author_year_map = matcher._build_author_year_map(references)
    print(f"单作者映射: {author_year_map}")

    # 测试同年同作者多论文 (2023a vs 2023b)
    references2 = [
        {"ref_id": "ref_1", "ref_authors": "J. Liu, et al.", "ref_year": 2023},
        {"ref_id": "ref_2", "ref_authors": "J. Liu, et al.", "ref_year": 2023},
    ]

    # 注意：这里会冲突，后者会覆盖前者
    # 这是个问题，需要在后续解决
    print(f"同年引用（会冲突）: {matcher._build_author_year_map(references2)}")

    print()


if __name__ == "__main__":
    test_basic_matching()
    test_citation_formats()
    test_realistic_scenario()
    test_edge_cases()

    print("=" * 60)
    print("测试完成!")
    print("=" * 60)
