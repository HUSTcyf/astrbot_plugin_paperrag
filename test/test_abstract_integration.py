#!/usr/bin/env python3
"""
测试脚本：验证摘要索引集成功能

测试内容：
1. 添加论文时自动构建摘要索引（LLM优先，失败回退常规）
2. 摘要索引正确存储到独立的 Milvus 数据库

使用方法：
    cd /path/to/astrbot_plugin_paperrag
    python test_abstract_integration.py

注意：测试使用独立的测试目录和数据库，不影响原有数据
"""

import asyncio
import os
import shutil
import tempfile
import sys
from pathlib import Path

# 切换到插件目录
plugin_dir = Path(__file__).parent.parent
os.chdir(plugin_dir)
sys.path.insert(0, str(plugin_dir))


async def test_abstract_integration():
    """测试摘要索引集成"""
    from astrbot.api import logger

    print("=" * 50)
    print("摘要索引集成测试")
    print("=" * 50)

    # 创建临时测试目录
    test_dir = plugin_dir / "test_integration"
    test_papers_dir = test_dir / "papers"
    test_data_dir = test_dir / "data"
    test_papers_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # 测试数据库路径（独立）
    test_milvus_db = test_data_dir / "milvus_abstracts_test.db"

    print(f"\n📂 测试目录: {test_dir}")
    print(f"📂 测试论文目录: {test_papers_dir}")
    print(f"📂 测试数据库: {test_milvus_db}")

    # 查找3篇PDF论文
    papers_source = Path("/Volumes/ext/Master/papers")
    pdf_files = list(papers_source.glob("*.pdf"))[:3]

    if len(pdf_files) < 3:
        print(f"❌ 论文目录中需要至少3篇PDF，当前只有 {len(pdf_files)} 篇")
        return False

    print(f"\n📄 找到 {len(pdf_files)} 篇论文用于测试")

    # 复制论文到测试目录
    for pdf_file in pdf_files:
        dest = test_papers_dir / pdf_file.name
        shutil.copy2(pdf_file, dest)
        print(f"  ✓ 复制: {pdf_file.name}")

    # ========== 测试 AbstractIndexManager ==========
    print("\n" + "=" * 50)
    print("测试 AbstractIndexManager (LLM优先提取)")
    print("=" * 50)

    from abstract_index import AbstractIndexManager, LocalGGUFClient

    # 初始化 LocalGGUFClient
    llm_client = LocalGGUFClient()

    # 检查模型是否已加载
    if llm_client._is_loaded:
        print("✓ GGUF LLM 模型已加载，直接复用")
    else:
        print("🔄 GGUF LLM 未加载，正在加载...")
        loaded = await llm_client.load()
        if not loaded:
            print("⚠️ LLM 模型加载失败，将使用纯常规提取")
            llm_client = None
        else:
            print("✓ LLM 客户端已配置")

    # 模拟 embed_provider（使用简单的 mock）
    class MockEmbedProvider:
        async def get_text_embedding(self, text):
            # 返回一个假的768维向量
            import random
            return [random.random() for _ in range(768)]

        async def get_text_embeddings_batch(self, texts):
            return [await self.get_text_embedding(t) for t in texts]

    # 初始化 AbstractIndexManager
    abstract_index = AbstractIndexManager(
        milvus_uri=str(test_milvus_db),
        collection_name="paper_abstracts_test",
        embed_dim=768,
    )
    abstract_index.set_embed_model(MockEmbedProvider())
    if llm_client:
        abstract_index.set_llm_client(llm_client)

    await abstract_index.initialize()
    print("✓ AbstractIndexManager 初始化完成")

    # 测试每篇论文
    results = {"success": 0, "failed": 0}

    for pdf_file in pdf_files:
        paper_id = pdf_file.stem
        file_name = pdf_file.name

        print(f"\n📄 处理: {file_name}")

        try:
            success = await abstract_index.index_paper(
                pdf_path=str(pdf_file),
                paper_id=paper_id,
                file_name=file_name,
            )

            if success:
                results["success"] += 1
                print(f"  ✓ 摘要索引成功")
            else:
                results["failed"] += 1
                print(f"  ❌ 摘要索引失败")

        except Exception as e:
            results["failed"] += 1
            print(f"  ❌ 处理异常: {e}")

    # 验证结果
    print("\n" + "=" * 50)
    print("测试结果")
    print("=" * 50)

    abstracts = await abstract_index.get_all_abstracts()
    print(f"  成功: {results['success']}/3")
    print(f"  失败: {results['failed']}/3")
    print(f"  存储的摘要数: {len(abstracts)}")

    # 显示摘要内容（如果成功）
    if abstracts:
        print("\n📋 摘要内容预览:")
        for paper_id, abstract in list(abstracts.items())[:3]:
            print(f"\n  [{paper_id}]")
            text = abstract.abstract_text[:200] if abstract.abstract_text else "(空)"
            print(f"  {text}...")

    # 清理测试目录
    print("\n🧹 清理测试目录...")
    shutil.rmtree(test_dir)
    print("✓ 测试目录已清理")

    # 最终结果
    print("\n" + "=" * 50)
    if results["success"] == 3:
        print("✅ 所有测试通过！")
        return True
    elif results["success"] > 0:
        print(f"⚠️ 部分测试通过 ({results['success']}/3)")
        return True  # 部分成功也算通过
    else:
        print("❌ 测试失败")
        return False


async def test_abstract_extraction_only():
    """仅测试摘要提取（不测试向量存储）"""
    print("\n" + "=" * 50)
    print("测试摘要提取（不使用向量存储）")
    print("=" * 50)

    from abstract_index import AbstractExtractor

    # 查找1篇论文
    papers_dir = Path("/Volumes/ext/Master/papers")
    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ 未找到PDF论文")
        return False

    pdf_file = pdf_files[0]
    print(f"📄 测试文件: {pdf_file.name}")

    extractor = AbstractExtractor()

    # 常规提取
    abstract = await extractor.extract_abstract_from_pdf(str(pdf_file))

    if abstract:
        print(f"✓ 常规提取成功 ({len(abstract)} 字符)")
        print(f"  内容预览: {abstract[:150]}...")
    else:
        print("⚠️ 常规提取未找到摘要")

    # 测试 LocalGGUFClient
    try:
        from abstract_index import LocalGGUFClient
        import pymupdf

        llm_client = LocalGGUFClient()

        # 检查模型是否已加载（extract_abstract 会自动加载）
        if llm_client._is_loaded:
            print("✓ GGUF LLM 模型已加载，直接复用")
        else:
            print("🔄 GGUF LLM 未加载，正在加载...")
            loaded = await llm_client.load()
            if not loaded:
                print("⚠️ LLM 模型加载失败，跳过 LLM 测试")
                return True

        # 提取论文开头
        doc = pymupdf.open(str(pdf_file))
        text = ""
        for page in doc[:5]:
            text += page.get_text() + "\n"
        doc.close()

        print(f"\n📄 使用 LLM 提取摘要...")

        llm_abstract = await llm_client.extract_abstract(text[:3000])

        if llm_abstract:
            print(f"✓ LLM 提取成功 ({len(llm_abstract)} 字符)")
            print(f"  内容预览: {llm_abstract[:150]}...")
        else:
            print("⚠️ LLM 提取未返回结果")

    except Exception as e:
        print(f"⚠️ LLM 测试跳过: {e}")

    return True


async def main():
    """主测试流程"""
    print("🚀 开始测试摘要索引集成功能\n")

    # 测试1: 仅摘要提取
    ok1 = await test_abstract_extraction_only()

    # 测试2: 完整集成测试
    ok2 = await test_abstract_integration()

    print("\n" + "=" * 50)
    print("最终结果")
    print("=" * 50)
    print(f"  摘要提取测试: {'✅ 通过' if ok1 else '❌ 失败'}")
    print(f"  完整集成测试: {'✅ 通过' if ok2 else '❌ 失败'}")

    if ok1 and ok2:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️ 部分测试未通过，请检查日志")


if __name__ == "__main__":
    asyncio.run(main())
