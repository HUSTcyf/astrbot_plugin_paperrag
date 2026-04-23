#!/usr/bin/env python3
"""
论文链接解析探测脚本

默认只探测本地 PDF 元数据与首页文本，不发起网络请求。
加上 --resolve 后，会调用共享 resolver 继续做 Crossref / OpenAlex /
CORE / arXiv fallback，并打印最终 URL、来源与置信度。

用法示例:
    python test/test_link_resolution_probe.py
    python test/test_link_resolution_probe.py --resolve
    python test/test_link_resolution_probe.py --pdf /path/to/a.pdf --pdf /path/to/b.pdf --resolve
    python test/test_link_resolution_probe.py --root /Volumes/ext/Master/papers --sample 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.paper_link_resolver import PaperLinkResolver


DEFAULT_ROOTS = [
    Path("/Volumes/ext/Master/papers"),
    REPO_ROOT,
]


def load_core_api_key() -> str:
    """从插件配置里读取 CORE API key。"""
    candidates = [
        REPO_ROOT / "config" / "astrbot_plugin_paperrag_config.json",
        Path.home() / "AstrBot" / "data" / "config" / "astrbot_plugin_paperrag_config.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
            key = str(config.get("core_api_key", "") or "").strip()
            if key:
                return key
        except Exception:
            continue
    return ""


def find_pdf_files(roots: Sequence[Path]) -> List[Path]:
    """递归收集 PDF 文件。"""
    pdfs: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".pdf":
            pdfs.append(root)
            continue
        if root.is_dir():
            for path in root.rglob("*.pdf"):
                if path.name.startswith("._"):
                    continue
                pdfs.append(path)
    return sorted(set(pdfs))


def select_pdfs(paths: Sequence[Path], sample_size: int, seed: int) -> List[Path]:
    """从候选 PDF 中抽样。"""
    if sample_size <= 0 or len(paths) <= sample_size:
        return list(paths)
    rng = random.Random(seed)
    return sorted(rng.sample(list(paths), k=sample_size))


def format_lines(prefix: str, values: Iterable[str]) -> None:
    for value in values:
        print(f"{prefix}{value}")


def print_probe_summary(pdf_path: Path, probe) -> None:
    """打印本地探测结果。"""
    print(f"\n== {pdf_path.name} ==")
    print(f"path: {pdf_path}")
    print(f"metadata.title: {probe.metadata_title or '(empty)'}")
    print(f"metadata.author: {probe.metadata_author or '(empty)'}")
    print(f"metadata.subject: {probe.metadata_subject or '(empty)'}")
    print(f"metadata.doi: {probe.metadata_doi or '(empty)'}")
    print(f"metadata.arxiv_id: {probe.metadata_arxiv_id or '(empty)'}")
    print(f"first_page.title: {probe.first_page_title or '(empty)'}")
    print(f"first_page.author: {probe.first_page_author or '(empty)'}")

    if probe.title_candidates:
        format_lines("  title_candidate: ", probe.title_candidates)
    else:
        print("  title_candidate: (none)")

    if probe.doi_candidates:
        format_lines("  doi_candidate: ", probe.doi_candidates)
    else:
        print("  doi_candidate: (none)")

    if probe.arxiv_candidates:
        format_lines("  arxiv_candidate: ", probe.arxiv_candidates)
    else:
        print("  arxiv_candidate: (none)")


def print_resolution(resolution, fallback_title: str = "") -> None:
    """打印最终解析结果。"""
    final_title = resolution.matched_title or fallback_title
    print("resolution:")
    print(f"  source: {resolution.resolution_source or resolution.backend or '(empty)'}")
    print(f"  score: {resolution.resolution_score:.1f}")
    print(f"  final_title: {final_title or '(empty)'}")
    print(f"  matched_title: {resolution.matched_title or '(empty)'}")
    print(f"  matched_identifier: {resolution.matched_identifier or '(empty)'}")
    print(f"  arxiv_url: {resolution.arxiv_url or '(empty)'}")
    print(f"  doi_url: {resolution.doi_url or '(empty)'}")
    print(f"  github_url: {resolution.github_url or '(empty)'}")


async def probe_pdf(pdf_path: Path, resolve: bool, resolver: PaperLinkResolver) -> None:
    """探测单个 PDF。"""
    probe = resolver.extract_pdf_probe(str(pdf_path))
    print_probe_summary(pdf_path, probe)

    if not resolve:
        return

    print("resolution: running shared resolver ...")
    resolution = await resolver.resolve_from_pdf(str(pdf_path), title_hint=probe.metadata_title or probe.first_page_title)
    print_resolution(resolution, fallback_title=probe.metadata_title or probe.first_page_title)


async def main() -> int:
    parser = argparse.ArgumentParser(description="论文链接解析探测脚本")
    parser.add_argument("--pdf", action="append", default=[], help="指定一个 PDF 路径，可重复传入")
    parser.add_argument("--root", action="append", default=[], help="扫描 PDF 的根目录，可重复传入")
    parser.add_argument("--sample", type=int, default=5, help="未指定 PDF 时抽样数量，默认 5")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子，默认 42")
    parser.add_argument("--resolve", action="store_true", help="启用网络解析，打印最终 URL")
    parser.add_argument("--core-api-key", default="", help="手工指定 CORE API key")
    parser.add_argument("--no-crossref", action="store_true", help="禁用 Crossref")
    parser.add_argument("--no-openalex", action="store_true", help="禁用 OpenAlex")
    parser.add_argument("--no-arxiv-library", action="store_true", help="禁用 arXiv library")
    args = parser.parse_args()

    explicit_pdfs = [Path(p).expanduser() for p in args.pdf]
    if explicit_pdfs:
        pdfs = explicit_pdfs
    else:
        roots = [Path(p).expanduser() for p in args.root] if args.root else DEFAULT_ROOTS
        found = find_pdf_files(roots)
        pdfs = select_pdfs(found, args.sample, args.seed)

    if not pdfs:
        print("没有找到可探测的 PDF 文件。")
        return 1

    core_api_key = args.core_api_key.strip() or load_core_api_key()
    resolver = PaperLinkResolver(
        core_api_key=core_api_key,
        enable_crossref=not args.no_crossref,
        enable_openalex=not args.no_openalex,
        enable_arxiv_library=not args.no_arxiv_library,
        log_prefix="[Probe]",
    )

    print(f"PDF count: {len(pdfs)}")
    print(f"resolve mode: {'on' if args.resolve else 'off'}")
    print(f"crossref: {'on' if not args.no_crossref else 'off'}")
    print(f"openalex: {'on' if not args.no_openalex else 'off'}")
    print(f"arxiv_library: {'on' if not args.no_arxiv_library else 'off'}")

    for pdf_path in pdfs:
        if not pdf_path.exists():
            print(f"\n== {pdf_path} ==")
            print("path: (missing)")
            continue
        await probe_pdf(pdf_path, args.resolve, resolver)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
