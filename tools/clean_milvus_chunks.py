"""
Milvus Chunk 噪声清洗脚本

功能：
1. 遍历 Milvus 中所有 chunks
2. 使用本地 LLM 判断每个 chunk 是否为噪声（参考文献/表格单元格/机构信息/无意义符号）
3. 如果是噪声：保留原始内容写入 garbage.json，从 Milvus 中删除
4. 保留原始 chunk 内容（不修改 text 字段）

用法：
    python tools/clean_milvus_chunks.py --test    # 测试模式：随机抽10条打印详细结果
    python tools/clean_milvus_chunks.py --dry-run # 只分析，不修改（默认）
    python tools/clean_milvus_chunks.py --execute # 实际执行删除
    python tools/clean_milvus_chunks.py --limit N # 只处理前 N 篇论文
"""

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# 抑制警告
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymilvus import connections, Collection

from provider.llama_cpp_vlm import get_llama_cpp_vlm_provider


# ============================================================================
# LLM 提示词（极简版：只做分类，不做规范化）
# ============================================================================

NOISE_CLASSIFY_PROMPT = """判断以下文本是否为无意义噪声（适合从 RAG 移除）：

噪声类型：
- 参考文献列表 [1] xxx
- 纯表格单元格（无完整语义）
- 机构 affiliation 信息
- 只有符号/数字/标点的无意义行
- 残缺公式片段或符号串
- 纯页码、页眉、页脚

文本：
{text}

只输出 JSON（不要其他内容）：{{"is_noise": true/false, "reason": "原因（5字以内）"}}
"""


# ============================================================================
# Milvus 操作
# ============================================================================

def connect_milvus(db_path: str) -> Collection:
    """连接到 Milvus 数据库"""
    alias = "cleaning"
    try:
        connections.connect(alias=alias, uri=db_path)
    except Exception as e:
        print(f"连接失败: {e}")
        raise

    from pymilvus import utility
    collections = utility.list_collections(using=alias)
    print(f"可用 collections: {collections}")

    if "paper_embeddings" in collections:
        col = Collection("paper_embeddings", using=alias)
    else:
        col = Collection(collections[0], using=alias)

    col.load()
    return col


def get_all_papers_from_stats(stats_path: str | Path) -> List[str]:
    """从 paper_doc_stats.json 直接读取所有论文的 file_name 列表"""
    stats_path = Path(stats_path)
    if not stats_path.exists():
        print(f"警告: {stats_path} 不存在，尝试从 Milvus 获取...")
        return []

    with open(stats_path, 'r', encoding='utf-8') as f:
        stats_data = json.load(f)

    return sorted(list(stats_data.keys()))


def fetch_chunks_by_paper(col: Collection, file_name: str) -> List[Dict[str, Any]]:
    """获取指定论文的所有 chunks"""
    try:
        results = col.query(
            expr=f'metadata["file_name"] == "{file_name}"',
            output_fields=["id", "text", "vector", "metadata"],
            limit=16384
        )
        return cast(List[Dict[str, Any]], results)
    except Exception:
        try:
            results = col.query(
                expr=f'metadata.file_name == "{file_name}"',
                output_fields=["id", "text", "vector", "metadata"],
                limit=16384
            )
            return cast(List[Dict[str, Any]], results)
        except Exception:
            return []


def delete_chunks_by_ids(col: Collection, ids: List[int]) -> None:
    """按 ID 删除 chunks"""
    if not ids:
        return
    batch_size = 1000
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        expr = "id in [" + ",".join(str(i) for i in batch) + "]"
        col.delete(expr)


# ============================================================================
# LLM 处理（串行，锁保护）
# ============================================================================

_llama_lock = asyncio.Lock()


async def process_single_chunk(client, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    使用 LLM 判断单个 chunk 是否为噪声（串行，锁保护）

    Returns:
        None = 跳过（处理失败或太短）
        {"action": "delete", "chunk": {...}, "reason": str} = 噪声待删除
    """
    chunk_id = chunk.get("id")
    text = chunk.get("text", "")

    if not text or len(text.strip()) < 5:
        return {
            "action": "delete",
            "chunk": chunk,
            "reason": "文本过短"
        }

    # 截断到 x 字符（足够判断噪声类型，不需要全文）
    prompt = NOISE_CLASSIFY_PROMPT.format(text=text)

    try:
        async with _llama_lock:
            response = await client.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.1,
                max_tokens=64,
            )

        content = ""
        if hasattr(response, "content"):
            content = response.content
        elif isinstance(response, dict):
            content = response.get("content", "") or response.get("text", "")
        else:
            content = str(response)

        # 解析 JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            return None

        result = json.loads(json_match.group(0))
        is_noise = result.get("is_noise", False)
        reason = result.get("reason", "")

        if is_noise:
            return {
                "action": "delete",
                "chunk": chunk,
                "reason": reason or "LLM判定为噪声"
            }

        return None

    except Exception as e:
        print(f"\n  [!] LLM处理失败 id={chunk_id}: {e}")
        return None


# ============================================================================
# 测试模式
# ============================================================================

async def test_mode(client, col: Collection, papers: List[str], num_samples: int = 10):
    """随机抽取 chunk，打印详细分类结果"""
    print(f"\n{'=' * 60}")
    print(f"测试模式：随机抽取 {num_samples} 条 chunk 测试")
    print(f"{'=' * 60}\n")

    all_chunks = []
    for paper in papers:
        chunks = fetch_chunks_by_paper(col, paper)
        for c in chunks:
            all_chunks.append((paper, c))

    if not all_chunks:
        print("未找到任何 chunks")
        return

    samples = random.sample(all_chunks, min(num_samples, len(all_chunks)))

    for idx, (paper, chunk) in enumerate(samples):
        text = chunk.get("text", "")
        chunk_id = chunk.get("id")
        print(f"\n--- 样本 {idx + 1}/{len(samples)} ---")
        print(f"论文: {paper}")
        print(f"ID: {chunk_id}")
        print(f"文本(前300字): {text[:300]}")
        print(f"文本长度: {len(text)} 字符")

        prompt = NOISE_CLASSIFY_PROMPT.format(text=text)
        try:
            async with _llama_lock:
                response = await client.text_chat(
                    prompt=prompt,
                    contexts=[],
                    temperature=0.1,
                    max_tokens=64,
                )
            content = ""
            if hasattr(response, "content"):
                content = response.content
            elif isinstance(response, dict):
                content = response.get("content", "") or response.get("text", "")
            else:
                content = str(response)

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                is_noise = result.get("is_noise", False)
                reason = result.get("reason", "")
                label = "噪声" if is_noise else "保留"
                print(f"分类结果: [{label}] reason={reason}")
                print(f"LLM原始输出: {content[:200]}")
            else:
                print(f"分类结果: [解析失败] 原始输出: {content[:200]}")
        except Exception as e:
            print(f"分类结果: [异常] {e}")

        print()


# ============================================================================
# 主流程（串行处理）
# ============================================================================

async def process_paper(client, col: Collection, file_name: str) -> Dict[str, Any]:
    """处理单篇论文的所有 chunks（串行）"""
    chunks = fetch_chunks_by_paper(col, file_name)
    if not chunks:
        return {"paper": file_name, "total": 0, "deleted": 0, "skipped": 0, "delete_ids": [], "delete_reasons": []}

    to_delete_ids = []
    delete_reasons = []

    for chunk in chunks:
        result = await process_single_chunk(client, chunk)
        if result is None:
            continue
        if result["action"] == "delete":
            to_delete_ids.append(chunk["id"])
            delete_reasons.append(result["reason"])

    return {
        "paper": file_name,
        "total": len(chunks),
        "deleted": len(to_delete_ids),
        "skipped": len(chunks) - len(to_delete_ids),
        "delete_ids": to_delete_ids,
        "delete_reasons": delete_reasons,
    }


async def main():
    parser = argparse.ArgumentParser(description="Milvus Chunk 噪声清洗工具（保留原始内容）")
    parser.add_argument("--test", action="store_true", help="测试模式：随机抽10条打印详细结果")
    parser.add_argument("--dry-run", action="store_true", default=False, help="只分析，不修改")
    parser.add_argument("--execute", action="store_true", help="实际执行删除")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇论文（0=全部）")
    parser.add_argument("--db-path", type=str, default=None, help="Milvus 数据库路径")
    args = parser.parse_args()

    plugin_dir = Path(__file__).parent.parent

    # 测试模式
    if args.test:
        db_path = args.db_path or str(plugin_dir / "data" / "milvus_papers.db")
        print(f"{'=' * 60}")
        print(f"测试模式")
        print(f"{'=' * 60}")
        print(f"数据库: {db_path}\n")

        print("连接 Milvus...")
        col = connect_milvus(db_path)

        stats_path = plugin_dir / "data" / "paper_doc_stats.json"
        papers = get_all_papers_from_stats(stats_path)
        print(f"总论文数: {len(papers)}")

        print("\n初始化 LLM...")
        client = get_llama_cpp_vlm_provider()
        await client.initialize()
        print("LLM 初始化完成\n")

        await test_mode(client, col, papers, num_samples=10)

        connections.disconnect("cleaning")
        print("\n测试完成!")
        return

    # dry-run 或 execute 模式
    if not args.execute:
        print("DRY RUN 模式：不会修改任何数据")
        print("   使用 --execute 标志来实际执行删除\n")

    db_path = args.db_path or str(plugin_dir / "data" / "milvus_papers.db")

    print(f"{'=' * 60}")
    print(f"Milvus Chunk 噪声清洗工具（保留原始内容）")
    print(f"{'=' * 60}")
    print(f"数据库: {db_path}")
    print()

    # 连接 Milvus
    print("连接 Milvus...")
    col = connect_milvus(db_path)

    # execute 模式下先备份数据库
    if args.execute:
        backup_dir = plugin_dir / "data" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"milvus_papers_backup_{timestamp}.db"
        print(f"\n⚠️  EXECUTE 模式：正在备份数据库...")
        print(f"   源文件: {db_path}")
        print(f"   备份文件: {backup_path}")
        shutil.copy2(db_path, backup_path)
        print(f"   备份完成!\n")

    # 获取论文列表
    print("获取论文列表...")
    stats_path = plugin_dir / "data" / "paper_doc_stats.json"
    papers = get_all_papers_from_stats(stats_path)
    if args.limit > 0:
        papers = papers[:args.limit]
        print(f"限制处理前 {args.limit} 篇论文")
    if not papers:
        print("警告: 未从 stats 文件获取到论文列表")
    print(f"总论文数: {len(papers)}")

    # 初始化 LLM
    print("\n初始化 LLM...")
    client = get_llama_cpp_vlm_provider()
    await client.initialize()
    print("LLM 初始化完成\n")

    # 统计
    total_chunks = 0
    total_deleted = 0
    garbage_entries = []

    # 按论文处理（串行）
    print(f"开始处理（按论文）...\n")
    for i, paper in enumerate(papers):
        paper_result = await process_paper(client, col, paper)

        total_chunks += paper_result["total"]
        total_deleted += paper_result["deleted"]

        if paper_result["deleted"] > 0:
            print(f"[{i+1}/{len(papers)}] {paper}: "
                  f"共{paper_result['total']} | 删除{paper_result['deleted']} | 跳过{paper_result['skipped']}")

        # 收集垃圾条目
        if paper_result.get("delete_ids"):
            chunks_map = {c["id"]: c for c in fetch_chunks_by_paper(col, paper)}
            for idx, cid in enumerate(paper_result["delete_ids"]):
                reason = paper_result["delete_reasons"][idx] if idx < len(paper_result["delete_reasons"]) else ""
                chunk = chunks_map.get(cid, {})
                garbage_entries.append({
                    "paper": paper,
                    "chunk_id": cid,
                    "text": chunk.get("text", ""),
                    "reason": reason,
                    "metadata": chunk.get("metadata", {}),
                })

            # 执行 Milvus 删除
            if args.execute:
                try:
                    delete_chunks_by_ids(col, paper_result["delete_ids"])
                except Exception as e:
                    print(f"  [!] 删除失败: {e}")

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(papers)} 篇论文", end="\r")

    print(f"\n\n{'=' * 60}")
    print("处理完成")
    print(f"{'=' * 60}")
    print(f"总 chunks: {total_chunks}")
    print(f"删除: {total_deleted} ({100*total_deleted/total_chunks:.1f}%)" if total_chunks > 0 else "删除: 0")
    print(f"跳过: {total_chunks - total_deleted}" if total_chunks > 0 else "跳过: 0")

    # 保存垃圾报告
    if garbage_entries:
        garbage_path = plugin_dir / "results" / "garbage_chunks.json"
        garbage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(garbage_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_deleted": total_deleted,
                "entries": garbage_entries,
                "db_path": db_path,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n噪声 chunk 已保存: {garbage_path}")

    if not args.execute:
        print("\n[DRY RUN] 未修改任何数据（使用 --execute 执行实际删除）")

    connections.disconnect("cleaning")
    print("\n完成!")


if __name__ == "__main__":
    asyncio.run(main())
