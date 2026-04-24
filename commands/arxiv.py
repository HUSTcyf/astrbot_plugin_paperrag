"""arXiv and external paper sync commands for PaperRAG."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, cast

import requests
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .base import PluginCoreBase
from ..plugin_common import CoreAPIClient, _is_hidden_file

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


class ArxivCommandsMixin(PluginCoreBase):
    async def _paper_arxiv_list(self, event: AstrMessageEvent):
        """List all papers with arxiv URLs in markdown format"""
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return
        try:
            doc_stats_path = _PLUGIN_DIR / "data" / "milvus_abstracts_doc_stats.json"
            if not doc_stats_path.exists():
                yield event.plain_result("❌ 未找到论文索引文件")
                return
            with open(doc_stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            abstracts = data.get('abstracts', {})
            output = f"📚 **论文列表** ({len(abstracts)} 篇)\n\n"
            count = 0
            for v in abstracts.values():
                title = v.get('title', '未知标题')
                url = v.get('metadata', {}).get('arxiv_url', '')
                if url:
                    output += f"- [{title}]({url})\n"
                    count += 1
                else:
                    output += f"- {title} (无链接)\n"
            output += f"\n📊 总计: {len(abstracts)} 篇 | 有链接: {count} 篇"
            yield event.plain_result(output)
        except Exception as e:
            yield event.plain_result(f"❌ 获取论文列表失败: {e}")


    async def _paper_arxiv_add(self, event: AstrMessageEvent, query: str = '', max_results: int = 5):
        """Search CORE API and download papers, then add to database (Admin)

        Args:
            query: Search query for papers (e.g., paper title, authors, keywords)
            max_results: Maximum number of papers to download (default: 5)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if not query:
            yield event.plain_result("❌ Please provide search query\nUsage: /paper arxiv_add <query> [max_results]\nExample: /paper arxiv_add attention is all you need 3")
            return

        # Check CORE API key
        core_api_key = self.config.get("core_api_key", "")
        if not core_api_key:
            yield event.plain_result("❌ CORE API Key未配置\n请在插件配置中设置 core_api_key")
            return

        papers_dir = self.config.get("papers_dir", "./papers")

        # Ensure papers directory exists
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            papers_path.mkdir(parents=True, exist_ok=True)

        yield event.plain_result(f"🔍 在CORE搜索: \"{query}\"\n最大下载数量: {max_results}")

        try:
            # Step 1: Search CORE API
            yield event.plain_result("📡 正在搜索CORE...")
            core_client = CoreAPIClient(core_api_key)
            works = core_client.search_by_title(query, limit=max_results)

            if not works:
                yield event.plain_result("❌ 未找到相关论文")
                return

            yield event.plain_result(f"✅ 找到 {len(works)} 篇论文")

            # Step 2: Download each paper
            engine = self._get_engine()
            if not engine:
                yield event.plain_result("❌ RAG引擎未就绪")
                return

            successful = 0
            failed = 0
            skipped = 0

            for i, work in enumerate(works, 1):
                # 提取论文信息
                work_id = work.get('id', '')
                title = work.get('title', 'unknown')
                source_urls = work.get('sourceFulltextUrls', []) or []

                if not title:
                    logger.warning(f"⚠️ 论文信息缺少标题: {work}")
                    failed += 1
                    continue

                yield event.plain_result(f"\n📄 [{i}/{len(works)}] {title[:60]}...")

                # 提取下载URL（优先使用arXiv链接）
                pdf_url = None
                for url in source_urls:
                    if 'arxiv.org/pdf' in url:
                        pdf_url = url
                        break
                if not pdf_url and source_urls:
                    pdf_url = source_urls[0]

                if not pdf_url:
                    yield event.plain_result(f"   ⚠️ 无可下载链接")
                    failed += 1
                    continue

                # 确定文件名
                arxiv_id = core_client.extract_arxiv_id(work)
                if arxiv_id:
                    pdf_filename = f"{arxiv_id}.pdf"
                else:
                    safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                    pdf_filename = f"{work_id}_{safe_title}.pdf"

                pdf_path = papers_path / pdf_filename

                # 检查是否已存在
                if pdf_path.exists():
                    yield event.plain_result(f"   ⏭️ PDF已存在，跳过下载")
                    skipped += 1
                    result = await engine.add_paper(str(pdf_path))
                    if result.get("status") == "success":
                        successful += 1
                        yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                    continue

                # 下载PDF
                try:
                    yield event.plain_result(f"   📥 下载PDF: {pdf_url[:80]}...")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    }
                    pdf_response = requests.get(pdf_url, headers=headers, timeout=120, stream=True)

                    if pdf_response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        file_size = pdf_path.stat().st_size / (1024 * 1024)
                        yield event.plain_result(f"   ✅ 下载完成 ({file_size:.1f} MB)")

                        # Add to database
                        result = await engine.add_paper(str(pdf_path))
                        if result.get("status") == "success":
                            successful += 1
                            yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                        else:
                            yield event.plain_result(f"   ⚠️ 添加失败: {result.get('message', 'unknown')}")
                    else:
                        yield event.plain_result(f"   ❌ 下载失败: HTTP {pdf_response.status_code}")
                        failed += 1

                except Exception as e:
                    logger.error(f"下载论文失败: {e}")
                    yield event.plain_result(f"   ❌ 下载失败: {e}")
                    failed += 1

            # Summary
            output = f"""
📊 **CORE论文下载完成**

✅ 成功: {successful}
⏭️ 跳过: {skipped} (已存在)
❌ 失败: {failed}

📁 保存路径: {papers_dir}
💡 使用 /paper list 查看已添加的论文
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"arXiv操作失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ arXiv操作失败: {e}")


    async def _paper_arxiv_refs(self, event: AstrMessageEvent, top_k: int = 10, max_per_paper: int = 3):
        """Download highly-cited reference papers via CORE API and add to database (Admin)

        Args:
            top_k: Number of top-cited references to process (default: 10)
            max_per_paper: Maximum papers to download per reference (default: 3)
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG engine is not ready")
            return

        # Check CORE API key
        core_api_key = self.config.get("core_api_key", "")
        if not core_api_key:
            yield event.plain_result("❌ CORE API Key未配置\n请在插件配置中设置 core_api_key")
            return

        papers_dir = self.config.get("papers_dir", "./papers")
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            papers_path.mkdir(parents=True, exist_ok=True)

        yield event.plain_result(f"📊 正在获取高频引用论文统计...")

        try:
            # Step 1: Get reference statistics
            index_manager = engine._ensure_index_manager_initialized()
            stats = await index_manager.get_all_references()

            if "error" in stats:
                yield event.plain_result(f"❌ 获取统计失败: {stats['error']}")
                return

            references = stats.get("references", [])
            if not references:
                yield event.plain_result("📭 数据库中暂无参考文献信息\n💡 请先使用 /paper add 添加论文")
                return

            # Get top-k references
            top_refs = references[:top_k]
            yield event.plain_result(f"📚 找到 {len(references)} 种参考文献，取前 {len(top_refs)} 个高频引用")

            # Step 2: Search and download each reference paper via CORE API
            successful = 0
            failed = 0
            skipped = 0
            total_downloaded = 0

            core_client = CoreAPIClient(core_api_key)

            for i, ref in enumerate(top_refs, 1):
                title = ref.get("title", "")
                year = ref.get("year", "")

                if not title:
                    continue

                yield event.plain_result(f"\n[{i}/{len(top_refs)}] 📝 {title[:60]}...")

                try:
                    # Search CORE API
                    yield event.plain_result(f"   🔍 搜索: {title[:60]}...")
                    works = core_client.search_by_title(title, year=int(year) if year else None, limit=max_per_paper)

                    if not works:
                        yield event.plain_result(f"   ⚠️ 未找到相关论文")
                        failed += 1
                        continue

                    # Download first (most relevant) result
                    work = works[0]
                    source_urls = work.get("sourceFulltextUrls", []) or []

                    # 提取下载URL（优先使用arXiv链接）
                    pdf_url = None
                    for url in source_urls:
                        if 'arxiv.org/pdf' in url:
                            pdf_url = url
                            break
                    if not pdf_url and source_urls:
                        pdf_url = source_urls[0]

                    if not pdf_url:
                        yield event.plain_result(f"   ⚠️ 无可下载链接")
                        failed += 1
                        continue

                    # 确定文件名
                    arxiv_id = core_client.extract_arxiv_id(work)
                    if arxiv_id:
                        pdf_filename = f"{arxiv_id}.pdf"
                    else:
                        work_id = work.get('id', 'unknown')
                        safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                        pdf_filename = f"{work_id}_{safe_title}.pdf"

                    pdf_path = papers_path / pdf_filename

                    if pdf_path.exists():
                        yield event.plain_result(f"   ⏭️ PDF已存在，跳过")
                        skipped += 1
                        result = await engine.add_paper(str(pdf_path))
                        if result.get("status") == "success":
                            successful += 1
                            yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                        continue

                    # Download PDF
                    yield event.plain_result(f"   📥 下载: {pdf_filename}")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    }
                    pdf_response = requests.get(pdf_url, headers=headers, timeout=120, stream=True)

                    if pdf_response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        file_size = pdf_path.stat().st_size / (1024 * 1024)
                        yield event.plain_result(f"   ✅ 下载完成 ({file_size:.1f} MB)")

                        # Add to database
                        result = await engine.add_paper(str(pdf_path))
                        if result.get("status") == "success":
                            successful += 1
                            total_downloaded += 1
                            yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                        else:
                            yield event.plain_result(f"   ⚠️ 添加失败: {result.get('message', 'unknown')}")
                    else:
                        yield event.plain_result(f"   ❌ 下载失败: HTTP {pdf_response.status_code}")
                        failed += 1

                except Exception as e:
                    logger.error(f"处理论文失败: {e}")
                    yield event.plain_result(f"   ❌ 错误: {e}")
                    failed += 1

            # Summary
            output = f"""
📊 **CORE高频引用论文下载完成**

✅ 成功: {successful}
⏭️ 跳过: {skipped}
❌ 失败: {failed}
📥 新增下载: {total_downloaded}

📁 保存路径: {papers_dir}
💡 使用 /paper list 查看所有论文
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"CORE批量下载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 操作失败: {e}")


    async def _paper_arxiv_sync(self, event: AstrMessageEvent, confirm: str = ''):
        """Sync arxiv MCP downloaded papers to paperrag database (Admin)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result("⚠️ 即将扫描 MCP 已下载的论文并添加到数据库\n使用 /paper arxiv_sync confirm 确认执行")
            return

        # Get MCP storage path from configuration（支持跨平台配置）
        mcp_storage_path = self.config.get("arxiv_mcp_storage_path", "/Volumes/ext/arxiv")

        if not os.path.exists(mcp_storage_path):
            yield event.plain_result(f"❌ MCP存储路径不存在: {mcp_storage_path}")
            return

        engine = self._get_engine()
        if not engine:
            yield event.plain_result("❌ RAG引擎未就绪")
            return

        yield event.plain_result(f"📁 扫描MCP存储路径: {mcp_storage_path}")

        try:
            # Scan for PDF files
            mcp_path = Path(mcp_storage_path)
            pdf_files = list(mcp_path.glob("*.pdf"))

            # Filter out macOS metadata files
            pdf_files = [f for f in pdf_files if not _is_hidden_file(f)]

            if not pdf_files:
                yield event.plain_result("📭 MCP目录中没有找到PDF文件")
                return

            yield event.plain_result(f"📄 找到 {len(pdf_files)} 个PDF文件")

            # Get paperrag papers directory for display
            papers_dir = self.config.get("papers_dir", "./papers")

            successful = 0
            failed = 0
            already_in_db = 0

            for i, pdf_file in enumerate(pdf_files, 1):
                yield event.plain_result(f"\n[{i}/{len(pdf_files)}] 📄 {pdf_file.name}")

                # Check if already exists in paperrag directory
                papers_path = Path(papers_dir)
                dest_path = papers_path / pdf_file.name

                if dest_path.exists():
                    yield event.plain_result(f"   ⏭️ 论文已存在于paperrag目录，跳过")
                    already_in_db += 1
                    continue

                # Copy file to paperrag directory
                try:
                    import shutil
                    shutil.copy2(pdf_file, dest_path)
                    file_size = dest_path.stat().st_size / (1024 * 1024)
                    yield event.plain_result(f"   📋 已复制 ({file_size:.1f} MB)")
                except Exception as e:
                    logger.error(f"复制文件失败: {e}")
                    yield event.plain_result(f"   ❌ 复制失败: {e}")
                    failed += 1
                    continue

                # Add to database
                try:
                    result = await engine.add_paper(str(dest_path))
                    if result.get("status") == "success":
                        successful += 1
                        yield event.plain_result(f"   ✅ 已添加 (chunks: {result.get('chunks_added', 0)})")
                    else:
                        yield event.plain_result(f"   ⚠️ 添加失败: {result.get('message', 'unknown')}")
                        failed += 1
                except Exception as e:
                    logger.error(f"添加论文失败: {e}")
                    yield event.plain_result(f"   ❌ 添加失败: {e}")
                    failed += 1

            # Summary
            output = f"""
📊 **MCP论文同步完成**

✅ 成功: {successful}
⏭️ 跳过(已存在): {already_in_db}
❌ 失败: {failed}

📁 MCP路径: {mcp_storage_path}
📁 paperrag路径: {papers_dir}
💡 使用 /paper list 查看所有论文
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"MCP同步失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 同步失败: {e}")


    async def _paper_arxiv_cleanup(self, event: AstrMessageEvent, confirm: str = ''):
        """Clean up old versions of arxiv papers, keeping only latest versions (Admin)

        Args:
            confirm: Must be 'confirm' to proceed
        """
        if not self.enabled:
            yield event.plain_result("❌ Plugin is disabled")
            return

        if confirm != "confirm":
            yield event.plain_result("⚠️ 即将清理arXiv论文旧版本，只保留最新版本\n使用 /paper arxiv_cleanup confirm 确认执行")
            return

        mcp_storage_path = self.config.get("arxiv_mcp_storage_path", "/Volumes/ext/arxiv")

        if not os.path.exists(mcp_storage_path):
            yield event.plain_result(f"❌ MCP存储路径不存在: {mcp_storage_path}")
            return

        yield event.plain_result(f"🧹 扫描MCP存储路径: {mcp_storage_path}")

        try:
            import re
            from collections import defaultdict

            mcp_path = Path(mcp_storage_path)

            # Find all PDF files (excluding macOS metadata files)
            pdf_files = [f for f in mcp_path.glob("*.pdf") if not _is_hidden_file(f)]

            if not pdf_files:
                yield event.plain_result("📭 MCP目录中没有找到PDF文件")
                return

            # Group papers by base ID (without version suffix)
            # e.g., 2603.11298.pdf and 2603.11298v2.pdf -> base_id = 2603.11298
            papers_by_base = defaultdict(list)

            for pdf_file in pdf_files:
                filename = pdf_file.name
                # Match arxiv ID pattern: YYMM.NNNNN or YYMM.NNNNNvX
                match = re.match(r'^(\d{4}\.\d+)(v\d+)?\.pdf$', filename, re.IGNORECASE)
                if match:
                    base_id = match.group(1)  # e.g., "2603.11298"
                    version_str = match.group(2)  # e.g., "v2" or None
                    version = int(version_str[1:]) if version_str else 1
                    papers_by_base[base_id].append({
                        'file': pdf_file,
                        'version': version,
                        'is_latest': False
                    })
                else:
                    logger.debug(f"无法识别的文件名: {filename}")

            # Find papers with multiple versions
            multi_version_papers = {k: v for k, v in papers_by_base.items() if len(v) > 1}

            if not multi_version_papers:
                yield event.plain_result("✅ 没有发现多版本论文，无需清理")
                return

            yield event.plain_result(f"📋 发现 {len(multi_version_papers)} 篇多版本论文")

            # Mark latest versions
            deleted_count = 0
            kept_count = 0

            for base_id, versions in multi_version_papers.items():
                # Sort by version descending
                cast(List[Dict[str, Any]], versions).sort(key=lambda x: x['version'], reverse=True)

                # Mark latest as kept
                versions[0]['is_latest'] = True
                kept_count += 1

                # Delete old versions
                for v in versions[1:]:
                    old_file = cast(Path, v['file'])
                    version = cast(int, v['version'])
                    try:
                        file_size = old_file.stat().st_size / (1024 * 1024)
                        old_file.unlink()
                        deleted_count += 1
                        yield event.plain_result(
                            f"   🗑️ 删除旧版本: {old_file.name} (v{version}, {file_size:.1f} MB)"
                        )
                    except Exception as e:
                        logger.error(f"删除文件失败: {old_file}: {e}")
                        yield event.plain_result(f"   ❌ 删除失败: {old_file.name}")

            # Also clean up macOS metadata files
            metadata_files = [f for f in mcp_path.glob("._*")]
            metadata_count = 0
            for meta_file in metadata_files:
                try:
                    meta_file.unlink()
                    metadata_count += 1
                except Exception as e:
                    logger.error(f"删除metadata文件失败: {meta_file}: {e}")

            output = f"""
📊 **arXiv论文版本清理完成**

📄 多版本论文: {len(multi_version_papers)} 篇
✅ 保留最新版本: {kept_count} 个
🗑️ 删除旧版本: {deleted_count} 个
📦 清理metadata: {metadata_count} 个

💡 建议：修改 MCP 配置添加 --max-version=1 参数（如果支持）
"""
            yield event.plain_result(output.strip())

        except Exception as e:
            logger.error(f"清理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield event.plain_result(f"❌ 清理失败: {e}")
