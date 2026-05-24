"""
Knowledge extraction agent — extracts verifiable facts from RAG Q&A sessions
and writes structured pages to the LLM Wiki (entities/, concepts/, comparisons/).

Pipeline: extract → critique → filter → write_to_wiki
Two independent LLM calls for correctness verification.
"""

from __future__ import annotations

import json
from typing import Any

from astrbot.api import logger
from idea.wiki import IdeaWikiEngine, slugify
from provider.llm_utils import call_llm, parse_json_response
from rag.token_utils import count_tokens, truncate_text_to_tokens


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT_HEADER = """\
You are a research knowledge extraction agent. Read the user question, AI answer,
and source paper chunks below. Extract **verifiable factual knowledge** to be stored
in a research wiki.

For each piece of extractable knowledge, determine its type:
- **entity**: a specific method, model, dataset, metric, or named artifact
  (e.g., "RRF", "BGE-M3", "MS MARCO")
- **concept**: a research concept, technique, paradigm, or idea
  (e.g., "reciprocal rank fusion", "late interaction retrieval")
- **comparison**: a comparative finding between entities/concepts
  (e.g., "ColBERT outperforms single-vector models on recall@100 but is slower")

For each page, write a concise Markdown body (2-5 paragraphs) with:
- Clear definition/description
- Key properties, formulas, or findings (if applicable)
- Relationships to other entities/concepts using [[wikilinks]]
- Source citations using [N] notation

Confidence levels:
- **high**: directly stated in sources with explicit numbers/quotes
- **medium**: reasonably inferred from sources
- **low**: speculative or weakly supported

Do NOT create pages for trivial observations or the user's question itself.
Only extract substantive, reusable research knowledge.

Output as valid JSON: {"pages": [{"type": "entity|concept|comparison", "title": "...", "slug": "...", "tags": [...], "confidence": "high|medium|low", "content_md": "..."}]}
If no extractable knowledge found: {"pages": []}"""  # noqa: E501


_CRITIQUE_PROMPT_HEADER = """\
You are a rigorous research fact-checker. Review each proposed wiki page against
the original source chunks and Q&A context below.

For EACH page, verify:
1. **Source support**: Is every factual claim directly supported by cited source(s)?
2. **Accuracy**: Are formulas, numbers, technical details correct vs. the sources?
3. **Over-generalization**: Does the page overstate what sources actually say?
4. **Completeness**: Is important context or caveat missing?
5. **Consistency**: Do any pages contradict each other?

For each page, assign a verdict:
- KEEP: accurate and well-supported → include unchanged
- REVISE: minor issues → fix content_md and/or adjust confidence
- REJECT: unsupported, inaccurate, or trivial → remove

Output as valid JSON: {"overall_confidence": "high|medium|low", "pages": [{"type": "...", "title": "...", "slug": "...", "tags": [...], "confidence": "high|medium|low", "verdict": "KEEP|REVISE|REJECT", "verification_notes": "...", "content_md": "..."}]}"""  # noqa: E501


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_SOURCE_TOKENS = 1500
_MAX_SOURCES_IN_PROMPT = 10
_VALID_PAGE_TYPES = {"entity", "concept", "comparison"}


def _format_sources_for_prompt(sources: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(sources, 1):
        if i > _MAX_SOURCES_IN_PROMPT:
            break
        metadata = s.get("metadata", {}) or {}
        fname = s.get("display_name") or metadata.get("file_name", "unknown")
        text = s.get("text", "")
        if count_tokens(text) > _MAX_SOURCE_TOKENS:
            text = truncate_text_to_tokens(text, _MAX_SOURCE_TOKENS) + "..."
        blocks.append(f"[{i}] {fname}\n{text}")
    return "\n\n".join(blocks)


def _parse_json(text: str) -> dict | None:
    """Sanitize Chinese quotes then delegate to parse_json_response."""
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return parse_json_response(text)


def _source_filenames(sources: list[dict], max_n: int = 5) -> list[str]:
    seen = set()
    result = []
    for s in sources:
        metadata = s.get("metadata", {}) or {}
        fname = s.get("display_name") or metadata.get("file_name", "unknown")
        if fname not in seen:
            seen.add(fname)
            result.append(fname)
            if len(result) >= max_n:
                break
    return result


def _coerce_page_type(ptype: str) -> str:
    return ptype if ptype in _VALID_PAGE_TYPES else "concept"


def _log_context_overflow(prompt_tokens: int, output_max: int, ctx_window: int, label: str) -> None:
    total = prompt_tokens + output_max
    if total > ctx_window:
        logger.warning(
            f"[KnowledgeExtract] {label} prompt may overflow context: "
            f"{prompt_tokens} prompt + {output_max} output = {total} > {ctx_window}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_knowledge_extraction(
    query: str,
    answer: str,
    sources: list[dict],
    context,
    config: dict | None = None,
) -> dict:
    """从 RAG Q&A 中提取可验证知识并写入 wiki。

    Returns:
        {"status": "written"|"rejected"|"error", "pages_written": int, "reason": str}
    """
    if not sources or not answer:
        return {"status": "error", "pages_written": 0, "reason": "无 sources 或 answer"}

    sources_text = _format_sources_for_prompt(sources)

    # ---- Step 1: Extract knowledge pages ----
    extract_prompt = (
        _EXTRACT_PROMPT_HEADER
        + f"\n\n## User Question\n{query}\n\n## AI Answer\n{answer}\n\n"
        + f"## Source Chunks\n{sources_text}"
    )

    ctx_window = (config or {}).get("llama_vlm_n_ctx")
    if isinstance(ctx_window, int) and ctx_window > 0:
        _log_context_overflow(count_tokens(extract_prompt), 4096, ctx_window, "Extract")

    try:
        raw = await call_llm(extract_prompt, context, config, temperature=0.2)
    except Exception as e:
        logger.error(f"[KnowledgeExtract] Extract LLM failed: {e}")
        return {"status": "error", "pages_written": 0, "reason": str(e)}

    data = _parse_json(raw)
    if not data:
        logger.warning("[KnowledgeExtract] Extract JSON parse failed")
        return {"status": "error", "pages_written": 0, "reason": "Extract JSON 解析失败"}

    pages = data.get("pages", [])
    if not pages:
        logger.info("[KnowledgeExtract] No extractable knowledge found")
        return {"status": "rejected", "pages_written": 0, "reason": "无可提取知识"}

    logger.info(f"[KnowledgeExtract] Extracted {len(pages)} candidate page(s)")

    # ---- Step 2: Critique and verify ----
    critique_prompt = (
        _CRITIQUE_PROMPT_HEADER
        + f"\n\n## Original Q&A\n**Question**: {query}\n**Answer**: {answer}\n\n"
        + f"## Source Chunks\n{sources_text}\n\n"
        + f"## Proposed Pages (JSON)\n{json.dumps(data, ensure_ascii=False)}"
    )

    if isinstance(ctx_window, int) and ctx_window > 0:
        _log_context_overflow(count_tokens(critique_prompt), 4096, ctx_window, "Critique")

    try:
        raw = await call_llm(critique_prompt, context, config, temperature=0.1)
    except Exception as e:
        logger.error(f"[KnowledgeExtract] Critique LLM failed: {e}")
        return {"status": "error", "pages_written": 0, "reason": str(e)}

    verified = _parse_json(raw)
    if not verified:
        logger.warning("[KnowledgeExtract] Critique JSON parse failed, using unverified pages")
        verified_pages = pages
        overall_confidence = "medium"
    else:
        verified_pages = verified.get("pages", pages)
        overall_confidence = verified.get("overall_confidence", "medium")

    # ---- Step 3: Filter ----
    accepted = []
    rejected = []
    for p in verified_pages:
        verdict = p.get("verdict", "KEEP")
        conf = p.get("confidence", "medium")
        if verdict == "REJECT" or conf == "low":
            rejected.append(p)
        else:
            accepted.append(p)

    if rejected:
        logger.info(
            f"[KnowledgeExtract] Rejected {len(rejected)}: "
            f"{[p.get('title', '?') for p in rejected]}"
        )

    if not accepted:
        return {
            "status": "rejected", "pages_written": 0,
            "reason": f"All {len(verified_pages)} pages rejected (overall={overall_confidence})",
        }

    # ---- Step 4: Write to wiki ----
    wiki = IdeaWikiEngine()
    wiki.ensure_schema()
    source_files = _source_filenames(sources)

    written = 0
    for p in accepted:
        try:
            ptype = _coerce_page_type(p.get("type", "concept"))
            ptitle = p.get("title", "Untitled")
            pslug = p.get("slug") or slugify(ptitle)
            wiki.save_page(
                page_type=ptype,
                title=ptitle,
                slug=pslug,
                content_md=p.get("content_md", ""),
                tags=p.get("tags", []),
                confidence=p.get("confidence", "medium"),
                sources_list=source_files,
            )
            wiki.append_log(
                "knowledge_extract",
                f"{ptype}/{pslug} confidence={p.get('confidence', '?')} "
                f"query={slugify(query)[:50]}"
            )
            written += 1
            logger.info(f"[KnowledgeExtract] Written: {ptype}/{ptitle}")
        except Exception as e:
            logger.error(f"[KnowledgeExtract] Write failed for {p.get('title', '?')}: {e}")

    return {
        "status": "written",
        "pages_written": written,
        "total_accepted": len(accepted),
        "overall_confidence": overall_confidence,
    }
