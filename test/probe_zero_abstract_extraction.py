#!/usr/bin/env python3
"""Probe zero-abstract PDF extraction without mutating any index files."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ASTRBOT_ROOT = PLUGIN_ROOT.parents[2]
CONFIG_PATH = ASTRBOT_ROOT / "data" / "config" / "astrbot_plugin_paperrag_config.json"
DATA_DIR = PLUGIN_ROOT / "data"
DEFAULT_PAPERS_DIR = "/Volumes/ext/Master/papers"

if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from rag.abstract_index import AbstractExtractor  # noqa: E402


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}", file=sys.stderr)
        return default


def _default_papers_dir() -> str:
    config = _load_json(CONFIG_PATH, {})
    if isinstance(config, dict):
        papers_dir = config.get("papers_dir")
        if papers_dir:
            return str(papers_dir)
    return DEFAULT_PAPERS_DIR


def _zero_abstract_papers() -> list[dict[str, Any]]:
    paper_stats = _load_json(DATA_DIR / "paper_doc_stats.json", {})
    abstract_stats = _load_json(DATA_DIR / "milvus_abstracts_doc_stats.json", {})
    abstracts = abstract_stats.get("abstracts", {}) if isinstance(abstract_stats, dict) else {}
    if not isinstance(paper_stats, dict) or not isinstance(abstracts, dict):
        return []

    papers: list[dict[str, Any]] = []
    for file_name, stats in paper_stats.items():
        if not isinstance(file_name, str) or not file_name.lower().endswith(".pdf"):
            continue
        paper_id = Path(file_name).stem
        entry = abstracts.get(paper_id)
        abstract_text = ""
        extracted_chars = None
        if isinstance(entry, dict):
            abstract_text = str(entry.get("abstract_text") or "").strip()
            metadata = entry.get("metadata", {})
            if isinstance(metadata, dict):
                extracted_chars = metadata.get("extracted_abstract_chars")
        if entry and abstract_text and extracted_chars != 0:
            continue
        papers.append({
            "paper_id": paper_id,
            "file_name": file_name,
            "chunk_count": stats.get("chunk_count", 0) if isinstance(stats, dict) else 0,
        })
    return papers


def _match_files(papers: list[dict[str, Any]], papers_dir: Path) -> list[dict[str, Any]]:
    file_map = {path.name: path for path in papers_dir.rglob("*.pdf")}
    file_map.update({path.name: path for path in papers_dir.rglob("*.PDF")})
    matched = []
    for paper in papers:
        file_path = file_map.get(str(paper["file_name"]))
        if file_path:
            matched.append({**paper, "file_path": file_path})
    return matched


def _filter_names(papers: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    if not names:
        return papers
    lowered = [name.lower() for name in names]
    return [
        paper for paper in papers
        if any(token in str(paper["file_name"]).lower() for token in lowered)
    ]


def _failure_reason(line_result: str | None, block_result: str | None) -> str:
    if block_result and not line_result:
        return "line extractor missed it; block extractor recovered it"
    if block_result and line_result and len(line_result) < 50:
        return "line extractor was too short; block extractor recovered it"
    if not block_result:
        return "no valid block-level abstract candidate"
    return "ok"


async def _probe_one(extractor: AbstractExtractor, paper: dict[str, Any]) -> None:
    file_path = Path(paper["file_path"])
    line_text = await extractor._parse_with_pymupdf(str(file_path))
    line_result = extractor._extract_abstract_text(line_text or "") if line_text else None
    block_result = await extractor._extract_abstract_from_pymupdf_blocks(str(file_path))

    try:
        import pymupdf

        doc = pymupdf.open(str(file_path))
        pages = len(doc)
        doc.close()
    except Exception:
        pages = "unknown"

    print(f"\n=== {paper['file_name']} ===")
    print(f"path: {file_path}")
    print(f"pages: {pages}")
    print(f"text chars: {len(line_text or '')}")
    print(f"line abstract chars: {len(line_result or '')}")
    print(f"block abstract chars: {len(block_result or '')}")
    print(f"reason: {_failure_reason(line_result, block_result)}")
    if block_result:
        print(f"block preview: {block_result[:500]}")
    elif line_result:
        print(f"line preview: {line_result[:500]}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", default=_default_papers_dir())
    parser.add_argument("--limit", type=int, default=0, help="Maximum papers to probe; 0 means all.")
    parser.add_argument("--names", nargs="*", default=[], help="Filename substrings to probe.")
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir).expanduser()
    if not papers_dir.exists():
        print(f"papers_dir does not exist: {papers_dir}", file=sys.stderr)
        return 2

    papers = _filter_names(_zero_abstract_papers(), args.names)
    matched = _match_files(papers, papers_dir)
    if args.limit > 0:
        matched = matched[:args.limit]

    print(f"zero-abstract papers: {len(papers)}")
    print(f"matched files: {len(matched)}")
    print(f"papers_dir: {papers_dir}")

    extractor = AbstractExtractor()
    for paper in matched:
        await _probe_one(extractor, paper)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
