"""
论文链接解析器

统一负责：
1. 按标题模糊搜索候选论文
2. 从 CORE / OpenAlex / arXiv library 提取可用 URL
3. 返回尽可能可靠的 arXiv / GitHub / DOI 链接

这份逻辑会被摘要索引与离线构建脚本共享，避免各自实现时
阈值、字段提取和回退策略不一致。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover - standalone / test fallback
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class LinkResolution:
    """链接解析结果。"""

    arxiv_url: str = ""
    github_url: str = ""
    doi_url: str = ""
    backend: str = ""
    resolution_source: str = ""
    resolution_score: float = 0.0
    matched_title: str = ""
    matched_identifier: str = ""
    score: float = 0.0

    def __post_init__(self):
        if not self.resolution_source:
            self.resolution_source = self.backend
        if not self.backend:
            self.backend = self.resolution_source
        if not self.resolution_score:
            self.resolution_score = self.score
        if not self.score:
            self.score = self.resolution_score

    def has_any_url(self) -> bool:
        return bool(self.arxiv_url or self.github_url or self.doi_url)


@dataclass
class PdfProbe:
    """PDF 元数据与首页探测结果。"""

    pdf_path: str
    metadata_title: str = ""
    metadata_author: str = ""
    metadata_subject: str = ""
    metadata_doi: str = ""
    metadata_arxiv_id: str = ""
    first_page_title: str = ""
    first_page_author: str = ""
    first_page_text: str = ""
    title_candidates: List[str] = field(default_factory=list)
    author_candidates: List[str] = field(default_factory=list)
    doi_candidates: List[str] = field(default_factory=list)
    arxiv_candidates: List[str] = field(default_factory=list)


class PaperLinkResolver:
    """基于标题的论文链接解析器。"""

    CROSSREF_THRESHOLD = 75.0
    CORE_THRESHOLD = 75.0
    OPENALEX_THRESHOLD = 75.0
    ARXIV_THRESHOLD = 70.0

    def __init__(
        self,
        core_api_key: str = "",
        enable_crossref: bool = True,
        enable_openalex: bool = True,
        enable_arxiv_library: bool = True,
        log_prefix: str = "[PaperLinkResolver]",
    ):
        self._core_api_key = core_api_key or ""
        self._enable_crossref = enable_crossref
        self._enable_openalex = enable_openalex
        self._enable_arxiv_library = enable_arxiv_library
        self._log_prefix = log_prefix
        self._last_request_time = 0.0

    @staticmethod
    def normalize_title(title: str) -> str:
        """清洗标题，便于 fuzzy matching。"""
        title = (title or "").strip()
        title = title.replace("$π_0$", "pi0").replace("$π_0.5$", "pi05")
        title = title.replace("$π$", "pi").replace("π", "pi")
        title = re.sub(r"\$([^$]+)\$", r"\1", title)
        title = re.sub(r"[^\w\s]", " ", title).strip().lower()
        return title

    @staticmethod
    def _clean_scalar(value: Any) -> str:
        """清洗单个标量字段。"""
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"none", "null"}:
            return ""
        return text

    @staticmethod
    def _normalize_free_text(value: str) -> str:
        """统一清洗自由文本，便于比较。"""
        value = (value or "").strip()
        value = re.sub(r"\s+", " ", value)
        return value

    @staticmethod
    def _strip_identifier_noise(value: str) -> str:
        """清理 DOI / arXiv ID 末尾夹带的装饰性符号。"""
        value = (value or "").strip()
        value = re.sub(r"[)\]}>⟩›.,;:]+$", "", value)
        return value.strip()

    @staticmethod
    def _looks_like_placeholder_title(title: str) -> bool:
        """判断标题是否像文件名、arXiv 编号这类占位内容。"""
        if not title:
            return True

        normalized = re.sub(r"\s+", " ", title).strip()
        if len(normalized) < 8:
            return True

        if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?(?:\([^)]+\))?", normalized):
            return True

        if re.fullmatch(r"[A-Za-z0-9._\-()]+", normalized) and len(normalized) <= 40:
            return True

        return False

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        """过滤显然不是标题的行。"""
        lowered = line.lower()
        noise_patterns = [
            r"^arxiv[:\s]",
            r"^\d{4}\.\d{4,5}(?:v\d+)?",
            r"^https?://",
            r"^copyright",
            r"^submitted",
            r"^accepted",
            r"^published",
            r"^keywords?[:\s]",
            r"^(author|authors)[:\s]",
            r"^(affiliation|institute|university|school|department)[:\s]",
            r"^doi[:\s]",
        ]
        for pattern in noise_patterns:
            if re.match(pattern, lowered, re.IGNORECASE):
                return True

        if "@" in line:
            return True

        if line.count(",") >= 3 and len(line) < 120:
            return True

        return False

    @staticmethod
    def _is_author_like_line(line: str) -> bool:
        """判断一行是否更像作者名而不是标题延续。"""
        normalized = re.sub(r"\s+", " ", line).strip(" ,;:|")
        if not normalized or len(normalized) > 100:
            return False

        if any(ch in normalized for ch in "@/\\"):
            return False

        words = normalized.split()
        if not 1 <= len(words) <= 8:
            return False

        stopwords = {"and", "of", "for", "the", "a", "an", "on", "in", "with", "to", "by", "from", "via"}
        if any(word.lower() in stopwords for word in words):
            return False

        cleaned_words = [re.sub(r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF'’´-]", "", word) for word in words]
        if any(not word for word in cleaned_words):
            return False

        return all(word[0].isupper() for word in cleaned_words)

    def _looks_like_title_continuation(self, prev_line: str, line: str) -> bool:
        """判断下一行是否像标题的续行，而不是作者行。"""
        prev = re.sub(r"\s+", " ", prev_line).strip(" -:|")
        nxt = re.sub(r"\s+", " ", line).strip()
        if not prev or not nxt:
            return False

        if prev.endswith(("-", ":", ";", ",", "—", "–", "/")):
            return True

        prev_words = prev.split()
        if prev_words:
            last_word = re.sub(r"[^A-Za-z\u00C0-\u024F\u1E00-\u1EFF]+", "", prev_words[-1].lower())
            if last_word in {"of", "for", "and", "in", "with", "to", "the", "a", "an", "on", "by", "from", "via"}:
                return True

        if self._is_author_like_line(line):
            return False

        if "@" not in nxt and "," not in nxt and len(nxt.split()) <= 8 and len(prev_words) <= 12:
            return True

        if len(prev_words) <= 4 and len(prev) < 45 and nxt[:1].islower():
            return True

        return False

    def _extract_title_candidates_from_text(self, full_text: str) -> List[str]:
        """从首页文本提取标题候选。"""
        if not full_text:
            return []

        lines = [re.sub(r"\s+", " ", line).strip() for line in full_text.split("\n")]
        title_candidates: List[str] = []
        pending_prefix = ""

        for line in lines[:40]:
            if not line:
                if title_candidates:
                    break
                continue

            lowered = line.lower()
            if re.match(r"^(abstract|摘要|summary|概述|references|参考文献)\b", lowered, re.IGNORECASE):
                break

            if self._is_noise_line(line):
                continue

            alpha_count = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", line))
            if alpha_count < 4:
                continue

            if len(line) < 8:
                if not title_candidates and not self._is_author_like_line(line):
                    prefix = line.strip(" -:|")
                    if prefix:
                        pending_prefix = f"{pending_prefix} {prefix}".strip() if pending_prefix else prefix
                continue

            candidate_line = line
            if pending_prefix:
                candidate_line = f"{pending_prefix} {candidate_line}".strip()
                pending_prefix = ""

            if not title_candidates:
                title_candidates.append(candidate_line)
                continue

            if self._looks_like_title_continuation(title_candidates[-1], candidate_line):
                title_candidates.append(candidate_line)
            break

        if not title_candidates:
            return []

        title = " ".join(title_candidates[:2]).strip(" -:|")
        return [title] if title else []

    @staticmethod
    def _extract_first_author_like_line(full_text: str, title_line: str = "") -> str:
        """从首页文本中找一个像作者行的候选。"""
        if not full_text:
            return ""

        title_norm = PaperLinkResolver.normalize_title(title_line)
        lines = [re.sub(r"\s+", " ", line).strip() for line in full_text.split("\n")]
        skip_title = bool(title_line)
        for line in lines[:50]:
            if not line:
                continue
            if skip_title and line == title_line:
                skip_title = False
                continue
            if title_norm:
                line_norm = PaperLinkResolver.normalize_title(line)
                if line_norm and (line_norm in title_norm or title_norm in line_norm):
                    continue
            if PaperLinkResolver._is_noise_line(line):
                continue
            if PaperLinkResolver._is_author_like_line(line):
                return line
        return ""

    @staticmethod
    def _extract_title_from_layout(page: Any) -> str:
        """从 PDF 首页版面布局中提取标题候选。"""
        try:
            layout = page.get_text("dict") or {}
        except Exception:
            return ""

        blocks: List[Tuple[float, float, str]] = []
        for block in layout.get("blocks", []) or []:
            if not isinstance(block, dict) or block.get("type") != 0:
                continue

            lines: List[str] = []
            max_font = 0.0
            for line in block.get("lines", []) or []:
                if not isinstance(line, dict):
                    continue
                span_texts: List[str] = []
                for span in line.get("spans", []) or []:
                    if not isinstance(span, dict):
                        continue
                    text = str(span.get("text", "") or "").strip()
                    if text:
                        span_texts.append(text)
                    try:
                        max_font = max(max_font, float(span.get("size", 0) or 0))
                    except Exception:
                        pass
                line_text = re.sub(r"\s+", " ", " ".join(span_texts)).strip()
                if line_text:
                    lines.append(line_text)

            text = re.sub(r"\s+", " ", " ".join(lines)).strip()
            if not text:
                continue

            bbox = block.get("bbox", [0, 0, 0, 0]) or [0, 0, 0, 0]
            try:
                y0 = float(bbox[1])
            except Exception:
                y0 = 0.0

            lowered = text.lower()
            if re.match(r"^(abstract|摘要|references|参考文献|introduction)\b", lowered, re.IGNORECASE):
                continue
            if "@" in text or "figure" in lowered or "table" in lowered:
                continue
            if len(text) < 10:
                continue
            if max_font < 10.0:
                continue
            if y0 > 220:
                continue

            blocks.append((max_font, -y0, text))

        if not blocks:
            return ""

        blocks.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return blocks[0][2]

    @staticmethod
    def _work_author_text(work: Dict[str, Any]) -> str:
        """从 work 中拼接作者相关文本，用于辅助排序。"""
        parts: List[str] = []

        def _append_value(value: Any):
            if not value:
                return
            if isinstance(value, str):
                cleaned = PaperLinkResolver._normalize_free_text(value)
                if cleaned:
                    parts.append(cleaned)
                return
            if isinstance(value, dict):
                for key in ("name", "display_name", "given", "family", "author_name"):
                    if value.get(key):
                        _append_value(value.get(key))
                        return
                return
            if isinstance(value, list):
                for item in value:
                    _append_value(item)
                return
            cleaned = PaperLinkResolver._normalize_free_text(str(value))
            if cleaned:
                parts.append(cleaned)

        for key in ("author", "authors", "authorships", "creator", "creators"):
            _append_value(work.get(key))

        return " ".join(parts)

    @staticmethod
    def _extract_identifier_candidates(text: str) -> Tuple[List[str], List[str]]:
        """从文本中提取 DOI / arXiv 候选。"""
        if not text:
            return [], []

        doi_candidates: List[str] = []
        arxiv_candidates: List[str] = []

        doi_pattern = re.compile(r"\b10\.\d{4,9}/[^\s<>\"]+", re.IGNORECASE)
        arxiv_pattern = re.compile(r"\b(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)

        for raw in doi_pattern.findall(text):
            doi = self._strip_identifier_noise(raw)
            if doi and doi not in doi_candidates:
                doi_candidates.append(doi)

        for match in arxiv_pattern.findall(text):
            arxiv_id = PaperLinkResolver._strip_identifier_noise(re.sub(r"v\d+$", "", match.strip()))
            if arxiv_id and arxiv_id not in arxiv_candidates:
                arxiv_candidates.append(arxiv_id)

        return doi_candidates, arxiv_candidates

    def extract_pdf_probe(self, pdf_path: str, max_chars: int = 3000) -> PdfProbe:
        """提取 PDF 元数据与首页候选信息。"""
        probe = PdfProbe(pdf_path=pdf_path)
        try:
            try:
                import pymupdf
            except Exception:  # pragma: no cover - environment fallback
                import fitz as pymupdf

            doc = pymupdf.open(pdf_path)
            try:
                metadata = doc.metadata or {}
                probe.metadata_title = self._clean_scalar(metadata.get("title", ""))
                probe.metadata_author = self._clean_scalar(metadata.get("author", ""))
                probe.metadata_subject = self._clean_scalar(metadata.get("subject", ""))
                probe.metadata_doi = self._clean_scalar(metadata.get("doi", ""))
                probe.metadata_arxiv_id = self._clean_scalar(metadata.get("arxiv_id", metadata.get("arxivId", "")))

                page_text = ""
                if len(doc) > 0:
                    page = doc[0]
                    page_text = page.get_text() or ""
                    probe.first_page_text = page_text[:max_chars]
                    probe.title_candidates = self._extract_title_candidates_from_text(probe.first_page_text)
                    layout_title = self._extract_title_from_layout(page)
                    if layout_title and (not probe.title_candidates or len(layout_title) >= len(probe.title_candidates[0])):
                        probe.title_candidates = [layout_title]
                        probe.first_page_title = layout_title
                    elif probe.title_candidates:
                        probe.first_page_title = probe.title_candidates[0]
                    probe.first_page_author = self._extract_first_author_like_line(
                        probe.first_page_text,
                        probe.first_page_title,
                    )

            finally:
                doc.close()

            combined_text = "\n".join(
                part for part in [
                    probe.metadata_title,
                    probe.metadata_author,
                    probe.metadata_subject,
                    probe.metadata_doi,
                    probe.metadata_arxiv_id,
                    page_text,
                ]
                if part
            )
            doi_candidates, arxiv_candidates = self._extract_identifier_candidates(combined_text)
            probe.doi_candidates = doi_candidates
            probe.arxiv_candidates = arxiv_candidates
            if probe.metadata_doi and probe.metadata_doi not in probe.doi_candidates:
                probe.doi_candidates.insert(0, self._strip_identifier_noise(probe.metadata_doi))
            if probe.metadata_arxiv_id and probe.metadata_arxiv_id not in probe.arxiv_candidates:
                probe.arxiv_candidates.insert(0, self._strip_identifier_noise(probe.metadata_arxiv_id))
        except Exception as e:
            logger.debug(f"PDF probe 失败 {pdf_path}: {e}")

        return probe

    @staticmethod
    def _title_similarity(query: str, candidate: str) -> float:
        """返回 0-100 的标题相似度。"""
        query_norm = PaperLinkResolver.normalize_title(query)
        cand_norm = PaperLinkResolver.normalize_title(candidate)
        if not query_norm or not cand_norm:
            return 0.0

        try:
            from rapidfuzz import fuzz

            return float(fuzz.token_set_ratio(query_norm, cand_norm))
        except Exception:
            return SequenceMatcher(None, query_norm, cand_norm).ratio() * 100.0

    @staticmethod
    def _work_title(work: Dict[str, Any]) -> str:
        """兼容不同 API 返回的标题字段。"""
        for key in ("title", "display_name", "name"):
            value = work.get(key, "")
            if value:
                return str(value)
        return ""

    @staticmethod
    def _iter_url_strings(work: Dict[str, Any]) -> Iterable[str]:
        """从 work 记录中收集候选 URL 字符串。"""
        def _yield_from_container(container: Any) -> Iterable[str]:
            if not container:
                return []
            if isinstance(container, str):
                return [container]
            if isinstance(container, dict):
                items: List[str] = []
                for key in ("landing_page_url", "pdf_url", "url", "downloadUrl", "doi", "arxiv", "arxiv_id", "arxivId"):
                    value = container.get(key)
                    if value:
                        items.append(str(value))
                return items
            if isinstance(container, list):
                items: List[str] = []
                for item in container:
                    items.extend(list(_yield_from_container(item)))
                return items
            return []

        for key in ("sourceFulltextUrls", "urls"):
            yield from _yield_from_container(work.get(key))
        yield from _yield_from_container(work.get("primary_location"))
        yield from _yield_from_container(work.get("best_oa_location"))
        yield from _yield_from_container(work.get("locations"))
        yield from _yield_from_container(work.get("ids"))

        download = work.get("downloadUrl", "") or ""
        if download:
            yield str(download)

        doi = work.get("doi", "") or ""
        if doi:
            yield str(doi)

    @classmethod
    def extract_arxiv_url_from_work(cls, work: Dict[str, Any]) -> str:
        """从 work 记录提取 arXiv URL。"""
        arxiv_id = work.get("arxivId", "") or work.get("arxiv_id", "")
        ids = work.get("ids", {}) or {}
        if not arxiv_id and isinstance(ids, dict):
            arxiv_id = ids.get("arxiv_id", "") or ids.get("arxivId", "") or ids.get("arxiv", "")
        if arxiv_id:
            arxiv_text = str(arxiv_id).strip()
            if arxiv_text.startswith("http://") or arxiv_text.startswith("https://"):
                match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)", arxiv_text, re.IGNORECASE)
                if match:
                    arxiv_id = re.sub(r"v\d+$", "", match.group(1))
                    return f"https://arxiv.org/abs/{arxiv_id}"
            arxiv_id = re.sub(r"v\d+$", "", arxiv_text)
            if re.fullmatch(r"\d{4}\.\d{4,5}", arxiv_id):
                return f"https://arxiv.org/abs/{arxiv_id}"

        identifiers = work.get("identifiers", []) or []
        for ident in identifiers:
            if isinstance(ident, dict) and ident.get("type") == "ARXIV_ID":
                arxiv_id = str(ident.get("identifier", ""))
                if arxiv_id:
                    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                    return f"https://arxiv.org/abs/{arxiv_id}"

        for alt in work.get("alternative_ids", []) or []:
            alt_text = str(alt)
            if "arxiv" in alt_text.lower():
                match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", alt_text, re.IGNORECASE)
                if match:
                    return f"https://arxiv.org/abs/{match.group(1)}"

        for value in cls._iter_url_strings(work):
            value_str = str(value)
            if "arxiv.org" in value_str.lower():
                match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)", value_str, re.IGNORECASE)
                if match:
                    arxiv_id = re.sub(r"v\d+$", "", match.group(1))
                    return f"https://arxiv.org/abs/{arxiv_id}"

        doi = str(work.get("doi", "") or "")
        if not doi and isinstance(ids, dict):
            doi = str(ids.get("doi", "") or "")
        if "arxiv." in doi.lower():
            arxiv_id = doi.split("arxiv.", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            return f"https://arxiv.org/abs/{arxiv_id}"

        return ""

    @classmethod
    def extract_github_url_from_work(cls, work: Dict[str, Any]) -> str:
        """从 work 记录提取 GitHub URL。"""
        for value in cls._iter_url_strings(work):
            value_str = str(value)
            if "github.com" in value_str.lower():
                match = re.search(r"github\.com/[\w\-]+/[\w\-]+", value_str, re.IGNORECASE)
                if match:
                    return f"https://{match.group()}"
        return ""

    @staticmethod
    def extract_doi_url_from_work(work: Dict[str, Any]) -> str:
        """从 work 记录提取 DOI URL。"""
        doi = PaperLinkResolver._strip_identifier_noise(str(work.get("doi", "") or "").strip())
        if not doi:
            return ""
        if doi.startswith("http://") or doi.startswith("https://"):
            return doi
        if doi.lower().startswith("10."):
            return f"https://doi.org/{doi}"
        if "doi.org/" in doi.lower():
            return doi
        return ""

    @staticmethod
    def _dedupe_works(works: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按稳定 key 去重候选结果。"""
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for work in works:
            key = (
                str(work.get("arxivId") or work.get("arxiv_id") or "").strip().lower()
                or str(work.get("doi") or "").strip().lower()
                or str(work.get("id") or "").strip().lower()
                or PaperLinkResolver.normalize_title(PaperLinkResolver._work_title(work))
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(work)
        return deduped

    @staticmethod
    def _work_from_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """将 Crossref item 规范化成统一 work 结构。"""
        title_list = item.get("title", []) or []
        title = ""
        if isinstance(title_list, list) and title_list:
            title = str(title_list[0])
        elif isinstance(title_list, str):
            title = title_list

        urls: List[str] = []
        if item.get("URL"):
            urls.append(str(item.get("URL")))
        for link in item.get("link", []) or []:
            if isinstance(link, dict):
                href = link.get("URL") or link.get("url") or ""
                if href:
                    urls.append(str(href))
        for alt in item.get("alternative-id", []) or []:
            if isinstance(alt, str) and alt.lower().startswith("http"):
                urls.append(alt)

        doi = str(item.get("DOI", "") or "")
        identifiers: List[Dict[str, Any]] = []
        if doi:
            identifiers.append({"type": "DOI", "identifier": doi})

        authors: List[str] = []
        for author in item.get("author", []) or []:
            if not isinstance(author, dict):
                continue
            name = " ".join(part for part in [str(author.get("given", "") or "").strip(), str(author.get("family", "") or "").strip()] if part)
            if not name:
                name = str(author.get("name", "") or "").strip()
            if name:
                authors.append(name)

        return {
            "title": title,
            "doi": doi,
            "URL": str(item.get("URL", "") or ""),
            "sourceFulltextUrls": urls,
            "downloadUrl": str(item.get("URL", "") or ""),
            "identifiers": identifiers,
            "authors": authors,
            "alternative_ids": [str(v) for v in item.get("alternative-id", []) or [] if v],
            "crossref_item": item,
        }

    @staticmethod
    def _clean_core_query(title: str) -> str:
        """CORE 查询清洗：去掉括号内容和多余标点。"""
        clean_title = re.sub(r"[:\(\[【].*?[)\]】]", "", title or "")
        clean_title = " ".join(clean_title.split())
        return clean_title.rstrip(".,;:")

    @staticmethod
    def _build_title_query_variants(title: str) -> List[str]:
        """为标题生成多个检索变体，优先使用更稳定的纯文本版本。"""
        variants: List[str] = []

        def _add(value: str):
            value = (value or "").strip()
            if value and value not in variants:
                variants.append(value)

        raw_title = (title or "").strip()
        if not raw_title:
            return variants

        # 先尝试去掉像 π0: / π0.5: / $π_0$: 这样的前缀，保留后半段更稳定的标题。
        stripped_title = raw_title
        compact_title = re.sub(r"\s+", " ", raw_title).strip()
        compact_title = re.sub(r"\s*([:：\-–—,.;])\s*", r"\1", compact_title)
        prefix_match = re.match(
            r"""^\s*(?:\$)?(?:π|pi)\s*[_\-]?\s*(?:\{?\d+(?:\.\d+)?\}?|\d+(?:\.\d+)?)?(?:\$)?\s*[:：\-–—]\s*(.+)$""",
            compact_title,
            re.IGNORECASE,
        )
        if prefix_match:
            stripped_title = prefix_match.group(1).strip()

        if stripped_title and stripped_title != raw_title:
            _add(stripped_title)

        article_stripped = re.sub(r"^(?:a|an|the)\s+", "", stripped_title, flags=re.IGNORECASE).strip()
        if article_stripped and article_stripped not in {raw_title, stripped_title}:
            _add(article_stripped)

        _add(raw_title)

        normalized = PaperLinkResolver.normalize_title(raw_title)
        if normalized and normalized != raw_title.lower():
            _add(normalized)

        stripped_normalized = PaperLinkResolver.normalize_title(stripped_title)
        if stripped_normalized and stripped_normalized not in variants:
            _add(stripped_normalized)

        return variants

    async def _search_crossref_candidates(self, title: str, limit: int = 5, author_hint: str = "") -> List[Dict[str, Any]]:
        """使用 Crossref 搜索候选结果。"""
        if not self._enable_crossref:
            return []

        try:
            query = (title or "").strip()
            if not query:
                return []

            logger.info(f"🔎 {self._log_prefix} Crossref 标题搜索: {query[:80]}")
            params = {
                "query.bibliographic": query,
                "rows": limit,
                "select": "DOI,title,URL,link,alternative-id,author,subject,created",
            }
            if author_hint:
                params["query.author"] = author_hint
            headers = {
                "User-Agent": "astrbot-paperrag/1.0 (mailto:astrbot@local)",
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get("https://api.crossref.org/works", params=params, headers=headers)
                response.raise_for_status()
                message = response.json().get("message", {}) or {}
                items = message.get("items", []) or []
                normalized = [self._work_from_crossref_item(item) for item in items if isinstance(item, dict)]
                return self._dedupe_works(normalized)
        except Exception as e:
            logger.warning(f"  → Crossref 标题搜索失败: {e}")
            return []

    async def _search_core_candidates(self, title: str, limit: int = 5, author_hint: str = "") -> List[Dict[str, Any]]:
        """使用 CORE API 搜索候选结果。"""
        if not self._core_api_key:
            return []

        try:
            queries = []
            cleaned = self._clean_core_query(title)
            if cleaned:
                queries.append(cleaned)
            if title and title not in queries:
                queries.append(title.strip())
            if cleaned and f'title:"{cleaned}"' not in queries:
                queries.append(f'title:"{cleaned}"')

            results: List[Dict[str, Any]] = []
            async with httpx.AsyncClient(timeout=30.0) as client:
                for query in queries:
                    if not query:
                        continue
                    logger.info(f"🔎 {self._log_prefix} CORE API 标题搜索: {query[:80]}")
                    response = await client.post(
                        "https://api.core.ac.uk/v3/search/works",
                        headers={
                            "Authorization": f"Bearer {self._core_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"q": query, "limit": limit},
                    )
                    response.raise_for_status()
                    chunk = response.json().get("results", []) or []
                    results.extend([w for w in chunk if isinstance(w, dict)])

            return self._dedupe_works(results)
        except Exception as e:
            logger.warning(f"  → CORE API 标题搜索失败: {e}")
            return []

    async def _search_openalex_candidates(self, title: str, limit: int = 5, author_hint: str = "") -> List[Dict[str, Any]]:
        """使用 OpenAlex 搜索候选结果。"""
        if not self._enable_openalex:
            return []

        try:
            import pyalex
            from pyalex import Works

            pyalex.config.email = "astrbot@local"
            query = self.normalize_title(title)
            logger.info(f"🔎 {self._log_prefix} OpenAlex 标题搜索: {title[:80]}")
            works = await asyncio.to_thread(lambda: Works().search(query).get(per_page=limit))
            normalized: List[Dict[str, Any]] = []
            for work in works:
                if not hasattr(work, "get"):
                    continue

                authorships = work.get("authorships", []) or []
                authors: List[str] = []
                for authorship in authorships:
                    if not isinstance(authorship, dict):
                        continue
                    author = authorship.get("author", {}) or {}
                    if isinstance(author, dict):
                        name = author.get("display_name", "") or author.get("name", "")
                        if name:
                            authors.append(str(name))

                normalized.append({
                    "title": work.get("title", "") or work.get("display_name", "") or "",
                    "arxiv_id": work.get("arxiv_id", "") or "",
                    "arxivId": work.get("arxivId", "") or "",
                    "doi": work.get("doi", "") or "",
                    "sourceFulltextUrls": work.get("sourceFulltextUrls", []) or [],
                    "downloadUrl": work.get("downloadUrl", "") or "",
                    "primary_location": work.get("primary_location", {}) or {},
                    "best_oa_location": work.get("best_oa_location", {}) or {},
                    "locations": work.get("locations", []) or [],
                    "identifiers": work.get("identifiers", []) or [],
                    "authorships": authorships,
                    "authors": authors,
                    "ids": work.get("ids", {}) or {},
                })
            return self._dedupe_works(normalized)
        except Exception as e:
            logger.warning(f"  → OpenAlex 标题搜索失败: {e}")
            return []

    async def _search_arxiv_library_candidates(self, title: str, limit: int = 5, author_hint: str = "") -> List[Dict[str, Any]]:
        """使用 arXiv library 搜索候选结果。"""
        if not self._enable_arxiv_library:
            return []

        try:
            import arxiv

            loop = asyncio.get_event_loop()
            elapsed = loop.time() - self._last_request_time
            if elapsed < 0.6:
                await asyncio.sleep(0.6 - elapsed)

            client = arxiv.Client()
            search = arxiv.Search(query=title, max_results=limit)
            results = await loop.run_in_executor(None, lambda: list(client.results(search)))
            self._last_request_time = loop.time()
            normalized: List[Dict[str, Any]] = []
            for result in results:
                entry_id = getattr(result, "entry_id", "") or ""
                normalized.append({
                    "title": getattr(result, "title", "") or "",
                    "entry_id": entry_id,
                    "arxivId": re.sub(r"^.*/", "", entry_id),
                })
            return normalized
        except Exception as e:
            logger.warning(f"  → arXiv library 标题搜索失败: {e}")
            return []

    def _build_direct_resolution(self, probe: PdfProbe, source: str) -> LinkResolution:
        """构建基于唯一标识符的直接解析结果。"""
        arxiv_id = ""
        if probe.metadata_arxiv_id:
            arxiv_id = probe.metadata_arxiv_id
        elif probe.arxiv_candidates:
            arxiv_id = probe.arxiv_candidates[0]

        doi = ""
        if probe.metadata_doi:
            doi = self._strip_identifier_noise(probe.metadata_doi)
        elif probe.doi_candidates:
            doi = self._strip_identifier_noise(probe.doi_candidates[0])

        normalized_arxiv_id = re.sub(r"v\d+$", "", arxiv_id) if arxiv_id else ""
        arxiv_url = f"https://arxiv.org/abs/{normalized_arxiv_id}" if normalized_arxiv_id else ""
        if doi and not doi.lower().startswith("http"):
            doi_url = f"https://doi.org/{doi}"
        else:
            doi_url = doi

        matched_identifier = arxiv_id or doi
        matched_title = probe.metadata_title or probe.first_page_title or ""

        return LinkResolution(
            arxiv_url=arxiv_url,
            doi_url=doi_url,
            backend=source,
            resolution_source=source,
            resolution_score=100.0,
            matched_title=matched_title,
            matched_identifier=matched_identifier,
            score=100.0,
        )

    async def resolve_from_pdf(self, pdf_path: str, title_hint: str = "") -> LinkResolution:
        """从 PDF 元数据和首页文本解析论文链接。"""
        probe = self.extract_pdf_probe(pdf_path)
        author_hint = probe.metadata_author or probe.first_page_author

        # 1) 唯一标识符优先：DOI / arXiv ID
        if probe.metadata_doi or probe.doi_candidates or probe.metadata_arxiv_id or probe.arxiv_candidates:
            resolution = self._build_direct_resolution(probe, source="PDF metadata/text")
            if resolution.has_any_url():
                if resolution.arxiv_url:
                    logger.info(f"  → arxiv (PDF identifier): {resolution.arxiv_url}")
                if resolution.doi_url and not resolution.arxiv_url:
                    logger.info(f"  → DOI (PDF identifier): {resolution.doi_url}")
                return resolution

        # 2) 标题候选：PDF 元数据标题 -> 首页标题 -> 上游标题提示
        candidate_titles: List[Tuple[str, str]] = []
        for label, candidate in (
            ("pdf metadata title", probe.metadata_title),
            ("pdf first-page title", probe.first_page_title),
            ("title hint", title_hint),
        ):
            candidate = (candidate or "").strip()
            if candidate and not self._looks_like_placeholder_title(candidate) and candidate not in [c for _, c in candidate_titles]:
                candidate_titles.append((label, candidate))

        best_resolution = LinkResolution()
        best_rank = 999
        for rank, (label, candidate) in enumerate(candidate_titles):
            resolution = await self.resolve_by_title(candidate, author_hint=author_hint)
            if not resolution.has_any_url():
                continue

            resolution.resolution_source = f"{label} -> {resolution.resolution_source or resolution.backend}"

            if (
                resolution.resolution_score > best_resolution.resolution_score
                or (
                    resolution.resolution_score == best_resolution.resolution_score
                    and rank < best_rank
                )
            ):
                best_resolution = resolution
                best_rank = rank

        return best_resolution

    @staticmethod
    def _author_score(author_hint: str, work: Dict[str, Any]) -> float:
        """根据作者线索计算辅助分数。"""
        author_hint = PaperLinkResolver._normalize_free_text(author_hint)
        if not author_hint:
            return 0.0

        author_text = PaperLinkResolver._work_author_text(work)
        if not author_text:
            return 0.0

        query_norm = PaperLinkResolver.normalize_title(author_hint)
        cand_norm = PaperLinkResolver.normalize_title(author_text)
        if not query_norm or not cand_norm:
            return 0.0

        if query_norm in cand_norm or cand_norm in query_norm:
            return 100.0

        return SequenceMatcher(None, query_norm, cand_norm).ratio() * 100.0

    @staticmethod
    def _combined_score(title_score: float, author_score: float) -> float:
        """把标题与作者线索合并成最终排序分数。"""
        if author_score <= 0:
            return title_score
        return min(100.0, title_score * 0.85 + author_score * 0.15)

    async def resolve_by_title(self, title: str, author_hint: str = "") -> LinkResolution:
        """按标题解析论文链接。"""
        search_title = (title or "").strip()
        if not search_title:
            return LinkResolution()

        query_norm = self.normalize_title(search_title)
        if not query_norm:
            return LinkResolution()

        title_queries = self._build_title_query_variants(search_title)
        if not title_queries:
            title_queries = [search_title]

        for source, searcher, threshold in (
            ("Crossref", self._search_crossref_candidates, self.CROSSREF_THRESHOLD),
            ("OpenAlex", self._search_openalex_candidates, self.OPENALEX_THRESHOLD),
            ("CORE API", self._search_core_candidates, self.CORE_THRESHOLD),
            ("arXiv library", self._search_arxiv_library_candidates, self.ARXIV_THRESHOLD),
        ):
            if source == "Crossref" and not self._enable_crossref:
                continue
            if source == "CORE API" and not self._core_api_key:
                continue
            if source == "OpenAlex" and not self._enable_openalex:
                continue
            if source == "arXiv library" and not self._enable_arxiv_library:
                continue

            works: List[Dict[str, Any]] = []
            used_query = ""
            for query in title_queries:
                works = await searcher(query, author_hint=author_hint)
                if works:
                    used_query = query
                    break
            if not works:
                logger.info(f"  → {source} 未返回候选结果")
                continue

            best_work = None
            best_score = 0.0
            for work in works:
                candidate_title = self._work_title(work)
                title_score = self._title_similarity(query_norm, candidate_title)
                score = self._combined_score(title_score, self._author_score(author_hint, work))
                if score > best_score:
                    best_score = score
                    best_work = work

            if not best_work or best_score < threshold:
                if source == "CORE API":
                    logger.info(f"  → CORE API 未找到足够相似的候选结果 (最佳 {best_score:.1f}%, query={used_query[:60]})")
                elif source == "OpenAlex":
                    logger.info(f"  → OpenAlex 未找到足够相似的候选结果 (最佳 {best_score:.1f}%, query={used_query[:60]})")
                else:
                    logger.info(f"  → arXiv library 未找到足够相似的候选结果 (最佳 {best_score:.1f}%, query={used_query[:60]})")
                continue

            arxiv_url = self.extract_arxiv_url_from_work(best_work)
            github_url = self.extract_github_url_from_work(best_work)
            doi_url = self.extract_doi_url_from_work(best_work)
            matched_title = self._work_title(best_work)

            if arxiv_url or github_url or doi_url:
                if arxiv_url:
                    logger.info(f"  → arxiv ({source}, 相似度 {best_score:.1f}%): {arxiv_url}")
                elif doi_url:
                    logger.info(f"  → DOI ({source}, 相似度 {best_score:.1f}%): {doi_url}")
                if github_url:
                    logger.info(f"  → github ({source}): {github_url}")
                return LinkResolution(
                    arxiv_url=arxiv_url,
                    github_url=github_url,
                    doi_url=doi_url,
                    backend=source,
                    resolution_source=source,
                    resolution_score=best_score,
                    matched_title=matched_title,
                    matched_identifier=(
                        best_work.get("arxivId", "")
                        or best_work.get("arxiv_id", "")
                        or best_work.get("DOI", "")
                        or best_work.get("doi", "")
                        or best_work.get("ids", {}).get("doi", "")
                        or ""
                    ),
                    score=best_score,
                )

            logger.info(f"  → {source} 命中但未提取到可用链接 (相似度 {best_score:.1f}%)")

        return LinkResolution()
