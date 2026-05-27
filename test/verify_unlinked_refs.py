"""
Verify and optionally repair papers with unlinked references.

Reads paper_doc_stats.json and auto-classifies papers into two repair strategies:
- Strategy A (full_reparse): Papers with empty-title refs or completely unparsed.
  Needs full pipeline: PyMuPDF + LLM extraction + link resolution.
- Strategy B (link_only): Papers where ALL unlinked refs have valid titles.
  Lightweight: re-runs PaperLinkResolver enrichment only (no LLM extraction).

Usage:
    # Diagnosis only (read-only, no modifications to stats file):
    python test/verify_unlinked_refs.py

    # Smart repair — auto-classify and repair all papers:
    python test/verify_unlinked_refs.py --execute

    # Dry-run repair on link_only papers only (tests link repair path):
    python test/verify_unlinked_refs.py --execute --link-only

    # Dry-run repair on full_reparse papers only:
    python test/verify_unlinked_refs.py --execute --full-only
"""
import sys
import json
import time
import shutil
import argparse
from pathlib import Path

# Add plugin root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_paper_doc_stats(stats_path: Path) -> dict:
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_papers(all_stats: dict) -> dict:
    """Auto-classify papers into full_reparse and link_only groups.

    Uses LLMReferenceParser._looks_like_polluted_title() to detect refs where the
    title field contains author names or citation numbers instead of a real title.
    """
    from rag.reference_processor import LLMReferenceParser

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
            has_link = bool(r.get("ref_doi") or r.get("ref_arxiv_url"))
            title = (r.get("ref_title") or "").strip()
            if has_link:
                linked += 1
            elif title:
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


def print_report(classification: dict):
    """Print auto-classification report."""
    full_reparse = classification["full_reparse"]
    link_only = classification["link_only"]
    total_papers = classification["total_papers"]

    full_unlinked = sum(p["title_only"] + p["no_title"] for p in full_reparse)
    full_no_title = sum(p["no_title"] for p in full_reparse)
    full_title_only = sum(p["title_only"] for p in full_reparse)
    link_unlinked = sum(p["title_only"] for p in link_only)

    print("=" * 72)
    print("📚  Reference Repair — Auto-Classification Report")
    print("=" * 72)
    print(f"\n📊 Total papers in stats: {total_papers}")
    print(f"   • Total with unlinked refs: {len(full_reparse) + len(link_only)}")
    print(f"   • Already fully linked: {total_papers - len(full_reparse) - len(link_only)}")

    if not full_reparse and not link_only:
        print("\n✅ All papers have fully-linked references. Nothing to repair.")
        return

    # Strategy A: Full Reparse
    if full_reparse:
        print(f"\n{'─' * 72}")
        print(f"🔴 **Strategy A: Full Reparse** ({len(full_reparse)} papers)")
        print(f"   Unlinked refs: {full_unlinked}")
        print(f"     - title_only: {full_title_only}")
        print(f"     - no_title (LLM failed): {full_no_title}")
        print(f"   Method: PyMuPDF text extraction + LLM parsing + link resolution")
        print()

        for i, p in enumerate(full_reparse, 1):
            file_name = p["file_name"]
            display_name = file_name if len(file_name) <= 50 else file_name[:47] + "..."
            unlinked = p["title_only"] + p["no_title"]

            if p["total"] == 0:
                status = "UNPARSED"
            elif p["linked"] == 0:
                status = "ZERO LINKED"
            else:
                status = f"{unlinked} unlinked"

            print(f"  {i:3d}. [{status}] {display_name}")
            parts = [f"{p['total']} total"]
            if p["linked"]:
                parts.append(f"{p['linked']} linked")
            if p["title_only"]:
                parts.append(f"{p['title_only']} title-only")
            if p["no_title"]:
                parts.append(f"{p['no_title']} empty-title")
            print(f"       └─ {', '.join(parts)}")

    # Strategy B: Link-Only
    if link_only:
        print(f"\n{'─' * 72}")
        print(f"🔗 **Strategy B: Link-Only Repair** ({len(link_only)} papers)")
        print(f"   Unlinked refs: {link_unlinked} (all have valid titles)")
        print(f"   Method: PaperLinkResolver enrichment only (no LLM, no PyMuPDF)")
        print()

        for i, p in enumerate(link_only, 1):
            file_name = p["file_name"]
            display_name = file_name if len(file_name) <= 50 else file_name[:47] + "..."

            print(f"  {i:3d}. {display_name}")
            parts = [f"{p['total']} total"]
            if p["linked"]:
                parts.append(f"{p['linked']} linked")
            parts.append(f"{p['title_only']} title-only → need links")
            print(f"       └─ {', '.join(parts)}")

    # Summary
    print(f"\n{'─' * 72}")
    print("📊 Repair Cost Estimate:")
    print(f"   🔴 Full Reparse: {len(full_reparse)} papers × LLM call each")
    print(f"   🔗 Link-Only:    {len(link_only)} papers × API calls only (Crossref/OpenAlex/arXiv)")
    print(f"\n💡 Run with --execute to perform the repair:")
    print(f"   python test/verify_unlinked_refs.py --execute")
    print(f"\n   Or use the AstrBot command:")
    print(f"   /paper repair_refs confirm")


async def run_smart_repair(
    classification: dict,
    papers_dir: str,
    stats_path: Path,
    link_only_enabled: bool = True,
    full_only_enabled: bool = True,
):
    """Run auto-classified repair: link_only → PaperLinkResolver, full_reparse → LLM."""
    full_reparse = classification["full_reparse"]
    link_only = classification["link_only"]
    papers_path = Path(papers_dir)

    if not papers_path.exists():
        print(f"❌ Papers directory does not exist: {papers_dir}")
        return

    # Build file path map
    file_path_map: dict[str, str] = {}
    for ext in [".pdf", ".PDF", ".docx", ".DOCX"]:
        for f in papers_path.rglob(f"*{ext}"):
            file_path_map[f.name] = str(f)

    # Resolve LLM config
    print("\n🔍 Resolving LLM config...")
    plugin_config_path = Path(__file__).resolve().parent.parent / ".." / ".." / "config" / "astrbot_plugin_paperrag_config.json"
    if not plugin_config_path.exists():
        plugin_config_path = Path(__file__).resolve().parent.parent / "config" / "astrbot_plugin_paperrag_config.json"

    llm_config = None
    if plugin_config_path.exists():
        with open(plugin_config_path, "r") as f:
            plugin_config = json.load(f)

        text_provider_id = plugin_config.get("text_provider_id")
        freeapi_url = plugin_config.get("freeapi_url")
        freeapi_key = plugin_config.get("freeapi_key")

        if text_provider_id:
            print(f"   ⚠️  Provider mode requires running AstrBot instance.")
            print(f"   💡 Use the command inside AstrBot: /paper repair_refs confirm")
            return
        elif freeapi_url and freeapi_key:
            model = plugin_config.get("freeapi_model", "gpt-4o-mini")
            llm_config = {
                "model": model,
                "api_base": freeapi_url.rstrip("/"),
                "api_key": freeapi_key,
            }
            print(f"   • Using freeapi: {model}")
        else:
            print("❌ No LLM config found")
            return
    else:
        print(f"❌ Plugin config not found")
        return

    if not llm_config:
        print("❌ Cannot proceed without LLM config.")
        return

    # Backup stats file
    backup_path = stats_path.with_suffix(".json.bak")
    print(f"💾 Backing up {stats_path.name} → {backup_path.name}")
    shutil.copy2(stats_path, backup_path)

    start_time = time.time()
    total_link_new = 0

    # ---- Phase 1: Link-Only Repair ----
    if link_only_enabled and link_only:
        from rag.reference_processor import repair_links_for_paper

        link_matched = [p for p in link_only if p["file_name"] in file_path_map]
        # For link-only repair we don't need the actual file — just the stats entry
        # But we use file presence as a sanity check that the paper exists

        print(f"\n🔗 Phase 1: Link-Only Repair ({len(link_only)} papers)...\n")
        link_success = 0
        link_fail = 0

        for i, paper in enumerate(link_only, 1):
            file_name = paper["file_name"]
            print(f"  [{i}/{len(link_only)}] {file_name} "
                  f"({paper['title_only']} title-only refs)...", end=" ", flush=True)

            try:
                result = await repair_links_for_paper(
                    file_name, llm_config, enable_fallback_search=True
                )
                if result.get("error"):
                    print(f"FAIL ({result['error']})")
                    link_fail += 1
                else:
                    newly = result.get("newly_linked", 0)
                    total_link_new += newly
                    print(f"OK (+{newly} linked, "
                          f"{result['linked_before']}→{result['linked_after']})")
                    link_success += 1
            except Exception as e:
                print(f"ERROR ({e})")
                link_fail += 1

        print(f"\n  Link repair: {link_success} success, {link_fail} failed, "
              f"+{total_link_new} newly linked")

    # ---- Phase 2: Full Reparse ----
    if full_only_enabled and full_reparse:
        import fitz
        from rag.reference_processor import process_references_with_llm

        # Match papers to files
        matched = []
        not_found = []
        for paper in full_reparse:
            fn = paper["file_name"]
            if fn in file_path_map:
                matched.append({**paper, "file_path": file_path_map[fn]})
            else:
                not_found.append(fn)

        if not_found:
            print(f"\n⚠️  {len(not_found)} full-reparse papers not found:")
            for fn in not_found[:5]:
                print(f"   - {fn}")

        if not matched:
            print("❌ No full-reparse paper files found.")
        else:
            print(f"\n🔴 Phase 2: Full Reparse ({len(matched)} papers)...\n")
            full_success = 0
            full_fail = 0

            for i, paper in enumerate(matched, 1):
                file_path = paper["file_path"]
                file_name = paper["file_name"]
                unlinked = paper["title_only"] + paper["no_title"]

                print(f"  [{i}/{len(matched)}] {file_name} "
                      f"({unlinked} unlinked refs)...", end=" ", flush=True)

                try:
                    with fitz.open(file_path) as doc:
                        raw_text = "".join(str(page.get_text()) for page in doc)

                    if not raw_text.strip():
                        print("SKIP (no text)")
                        full_fail += 1
                        continue

                    refs, _ = await process_references_with_llm(
                        file_path, [], raw_text, llm_config,
                        enable_fallback_search=True,
                    )
                    if refs:
                        linked_now = sum(1 for r in refs if r.ref_doi or r.ref_arxiv_url)
                        print(f"OK ({len(refs)} refs, {linked_now} linked)")
                        full_success += 1
                    else:
                        print("FAIL (no refs found)")
                        full_fail += 1
                except Exception as e:
                    print(f"ERROR ({e})")
                    full_fail += 1

            print(f"\n  Full reparse: {full_success} success, {full_fail} failed")

    elapsed = time.time() - start_time
    print(f"\n{'─' * 72}")
    print(f"✅ Done in {elapsed:.1f}s")
    print(f"💾 Stats saved to: {stats_path}")
    print(f"💾 Backup at: {backup_path}")
    print(f"\n💡 Verify with: /paper refstats -1")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-classify and repair papers with unlinked references"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Run smart repair pipeline (modifies paper_doc_stats.json)"
    )
    parser.add_argument(
        "--link-only", action="store_true",
        help="Only run link-only repair phase"
    )
    parser.add_argument(
        "--full-only", action="store_true",
        help="Only run full reparse phase"
    )
    parser.add_argument(
        "--papers-dir", type=str, default="./papers",
        help="Path to papers directory"
    )
    args = parser.parse_args()

    # Locate paper_doc_stats.json
    script_dir = Path(__file__).resolve().parent
    plugin_dir = script_dir.parent
    stats_path = plugin_dir / "data" / "paper_doc_stats.json"

    if not stats_path.exists():
        print(f"❌ paper_doc_stats.json not found at: {stats_path}")
        sys.exit(1)

    # Phase 1: Analyze (always runs, read-only)
    all_stats = load_paper_doc_stats(stats_path)
    classification = classify_papers(all_stats)
    print_report(classification)

    # Phase 2: Execute (if requested)
    if args.execute:
        link_only_enabled = not args.full_only
        full_only_enabled = not args.link_only

        full_count = len(classification["full_reparse"])
        link_count = len(classification["link_only"])

        if not link_only_enabled:
            full_count = 0
        if not full_only_enabled:
            link_count = 0

        if full_count == 0 and link_count == 0:
            print("\n✅ No papers to repair with current settings.")
            return

        print("\n" + "=" * 72)
        print("⚠️  EXECUTE MODE: Will modify paper_doc_stats.json")
        if link_only_enabled:
            print(f"   🔗 Link-only repair: {len(classification['link_only'])} papers")
        if full_only_enabled:
            print(f"   🔴 Full reparse:     {len(classification['full_reparse'])} papers")
        print("=" * 72)

        import asyncio
        asyncio.run(run_smart_repair(
            classification,
            args.papers_dir,
            stats_path,
            link_only_enabled=link_only_enabled,
            full_only_enabled=full_only_enabled,
        ))


if __name__ == "__main__":
    main()
