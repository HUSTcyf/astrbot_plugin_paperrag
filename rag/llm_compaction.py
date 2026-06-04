"""
LLM Compact Mixin - 仅用于提取 title/abstract/authors
"""

import re
from typing import Tuple, List, Any

from astrbot.api import logger
from provider.llm_utils import get_llm_provider
from rag.reference_processor import _find_all_reference_sections
from rag.llm_preprocessor import remove_reference_sections


LLM_EXTRACT_PROMPT_TEMPLATE = """从以下学术论文文本中提取标题、摘要和作者信息。

文本：
{text}

## 输出格式：
{{"title": "标题", "abstract": "摘要", "authors": ["作者1", "作者2"]}}"""


class LLMCompactionMixin:
    """
    Mixin class providing LLM-based title/abstract/authors extraction for HybridPDFParser.
    正文直接使用 docling 原文，不做 compact。
    """

    LLM_EXTRACT_PROMPT_TEMPLATE = LLM_EXTRACT_PROMPT_TEMPLATE

    async def _extract_metadata_with_llm(self, text: str) -> Tuple[str, str, List[str]]:
        """
        用 LLM 从论文首页文本中提取 title/abstract/authors。
        """

        provider = get_llm_provider(getattr(self, 'context', None), getattr(self, 'config', None))
        if provider is None:
            logger.warning("⚠️ LLM Provider 不可用，跳过 title/abstract/authors 提取")
            return "", "", []

        prompt = self.LLM_EXTRACT_PROMPT_TEMPLATE.format(text=text)

        try:
            response = await provider.text_chat(
                prompt=prompt,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"⚠️ LLM 调用失败: {e}")
            return "", "", []

        if hasattr(response, 'content'):
            content = response.content.strip()
        elif isinstance(response, dict):
            content = response.get("content", "").strip() or response.get("text", "").strip()
        else:
            content = str(response).strip()

        # 解析 JSON（去除 markdown 代码块包裹）
        content_clean = re.sub(r'```json\s*', '', content.strip())
        content_clean = re.sub(r'```\s*$', '', content_clean)

        import json
        try:
            data = json.loads(content_clean, strict=False)
            title = str(data.get("title", "")).strip().strip('"\n ')
            abstract = str(data.get("abstract", "")).strip().strip('"\n ')
            authors_raw = data.get("authors", [])
            if isinstance(authors_raw, list):
                authors = [str(a).strip() for a in authors_raw if str(a).strip()]
            else:
                authors = []
            logger.info(f"✅ LLM 提取: title={title[:50] if title else 'N/A'}, authors={len(authors)}人")
            return title, abstract, authors
        except json.JSONDecodeError:
            logger.warning(f"⚠️ LLM 响应 JSON 解析失败: {content[:100]}")
            return "", "", []

    async def _preprocess_documents_with_llm(self, documents: List[Any]) -> List[Any]:
        """
        预处理：LLM 只提取 title/abstract/authors，正文用 docling 原文（去掉参考文献）。
        """

        logger.info(f"🔄 文本预处理，共 {len(documents)} 个文档（LLM 仅提取 title/abstract/authors）...")

        for i, doc in enumerate(documents):
            is_first_page = (i == 0)
            if not hasattr(doc, 'metadata') or not doc.metadata:
                continue

            raw_text = doc.metadata.get("raw_text")
            if not raw_text:
                continue

            raw_text_str = str(raw_text)

            # 去掉参考文献部分
            ref_sections = _find_all_reference_sections(raw_text_str)
            if ref_sections:
                ref_chars = sum(len(v) for v in ref_sections.values())
                logger.info(f"📝 文档 {i+1} 移除 {len(ref_sections)} 处参考文献 ({ref_chars} 字符)")
                doc.text = remove_reference_sections(raw_text_str, ref_sections)
            else:
                doc.text = raw_text_str

            # 只对第一页提取 metadata（用 page1+page2 内容，即到 [Page 3] 之前）
            if is_first_page:
                page12_match = re.search(r'(\[Page \d+\].*?)(\n\[Page [3-9]|$)', raw_text_str, re.DOTALL)
                page12_text = page12_match.group(1) if page12_match else raw_text_str[:5000]
                try:
                    title, abstract, authors = await self._extract_metadata_with_llm(page12_text)
                    if title:
                        doc.metadata["extracted_title"] = title
                    if abstract:
                        doc.metadata["extracted_abstract"] = abstract
                    if authors:
                        doc.metadata["extracted_authors"] = authors
                except Exception as e:
                    logger.warning(f"⚠️ 文档 {i+1} LLM 提取异常: {e}")

        logger.info(f"✅ 文本预处理完成")
        return documents
