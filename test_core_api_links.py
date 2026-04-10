#!/usr/bin/env python3
"""
测试 CORE API 链接提取脚本

用法：
    cd /path/to/astrbot_plugin_paperrag
    python test_core_api_links.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import httpx


class CoreAPIClient:
    """CORE API v3 客户端"""

    BASE_URL = "https://api.core.ac.uk/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def search_by_title(self, title: str, limit: int = 3) -> list:
        """根据论文标题搜索论文"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 精确匹配
                response = await client.post(
                    f"{self.BASE_URL}/search/works",
                    headers=self.headers,
                    json={"q": f'title:"{title}"', "limit": limit}
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                if results:
                    return results
        except Exception as e:
            print(f"      [警告] API 请求失败: {e}")
        return []

    def extract_arxiv_url(self, work: dict) -> Optional[str]:
        """从 work 记录提取 arxiv URL"""
        urls = work.get("sourceFulltextUrls", []) or []
        for url in urls:
            if isinstance(url, str) and "arxiv.org" in url:
                return url
        return None

    def extract_github_url(self, work: dict) -> Optional[str]:
        """从 work 记录提取 GitHub URL"""
        urls = work.get("sourceFulltextUrls", []) or []
        for url in urls:
            if isinstance(url, str) and "github.com" in url:
                return url
        # 也检查 downloadUrl
        download = work.get("downloadUrl", "") or ""
        if "github.com" in download:
            return download
        return None

    async def get_arxiv_by_title(self, title: str) -> tuple:
        """根据论文标题获取 arxiv 和 github 链接"""
        results = await self.search_by_title(title, limit=3)

        arxiv_url = None
        github_url = None

        for work in results:
            arxiv = self.extract_arxiv_url(work)
            github = self.extract_github_url(work)
            if arxiv:
                arxiv_url = arxiv
            if github:
                github_url = github

        return arxiv_url, github_url


def get_core_api_key() -> Optional[str]:
    """从配置文件读取 CORE API key"""
    config_paths = [
        Path(__file__).parent / "config" / "astrbot_plugin_paperrag_config.json",
        Path.home() / "AstrBot" / "data" / "config" / "astrbot_plugin_paperrag_config.json",
    ]
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8-sig") as f:
                    config = json.load(f)
                key = config.get("core_api_key", "")
                if key:
                    return key
            except Exception:
                pass
    return None


async def main():
    # 加载论文数据
    json_path = Path(__file__).parent / "data" / "milvus_abstracts_doc_stats.json"
    if not json_path.exists():
        print(f"错误: 未找到 {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    abstracts = data.get("abstracts", {})
    print(f"总论文数: {len(abstracts)}")

    # 获取 API key
    api_key = get_core_api_key()
    if not api_key:
        print("错误: 未找到 CORE_API_KEY，请检查配置文件")
        return

    print(f"CORE API Key: {api_key[:10]}...")

    # 筛选出没有 arxiv_url 的论文
    papers_without_url = []
    for paper_id, info in abstracts.items():
        metadata = info.get("metadata", {})
        if not metadata.get("arxiv_url"):
            papers_without_url.append((paper_id, info))

    print(f"需要查询的论文数: {len(papers_without_url)}")
    print()

    client = CoreAPIClient(api_key)

    results = {
        "found": [],      # 找到 arxiv 或 github 的
        "not_found": [],  # 都没找到
        "error": []       # 查询出错的
    }

    for i, (paper_id, info) in enumerate(papers_without_url):
        title = info.get("title", paper_id)
        print(f"[{i+1}/{len(papers_without_url)}] 查询: {title[:60]}...")

        try:
            arxiv_url, github_url = await client.get_arxiv_by_title(title)

            if arxiv_url or github_url:
                results["found"].append({
                    "paper_id": paper_id,
                    "title": title,
                    "arxiv_url": arxiv_url,
                    "github_url": github_url
                })
                print(f"    -> arxiv: {arxiv_url}")
                print(f"    -> github: {github_url}")
            else:
                results["not_found"].append({
                    "paper_id": paper_id,
                    "title": title
                })
                print(f"    -> 未找到链接")
        except Exception as e:
            results["error"].append({
                "paper_id": paper_id,
                "title": title,
                "error": str(e)
            })
            print(f"    -> 错误: {e}")

        # 避免请求过快
        if (i + 1) % 10 == 0:
            await asyncio.sleep(1)

    print()
    print("=" * 60)
    print("查询结果汇总")
    print("=" * 60)
    print(f"找到链接: {len(results['found'])}")
    print(f"未找到:   {len(results['not_found'])}")
    print(f"查询出错: {len(results['error'])}")
    print()

    if results["not_found"]:
        print("=== 未找到链接的论文 ===")
        for item in results["not_found"][:20]:
            print(f"  - {item['title'][:70]}...")
        if len(results["not_found"]) > 20:
            print(f"  ... 还有 {len(results['not_found']) - 20} 篇")

    # 保存结果到 JSON
    output_path = Path(__file__).parent / "data" / "core_api_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
