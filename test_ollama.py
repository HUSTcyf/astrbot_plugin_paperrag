#!/usr/bin/env python3
"""
Ollama Embedding Provider 测试脚本
测试Ollama连接、embedding功能和性能
"""

import asyncio
import sys
import time

# 添加插件路径
sys.path.insert(0, '/Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag')

from ollama_embedding import create_ollama_provider, test_ollama_connection


async def test_basic_functionality():
    """测试基础功能"""
    print("=" * 60)
    print("🦙 Ollama Embedding Provider - 基础功能测试")
    print("=" * 60)

    # 1. 测试连接
    print("\n[1/4] 测试Ollama连接...")
    if not await test_ollama_connection():
        print(
            "\n❌ 连接失败！\n"
            "请确保：\n"
            "  1. Ollama服务正在运行: ollama serve\n"
            "  2. BGE-M3模型已下载: ollama pull bge-m3\n"
        )
        return False
    print("✅ 连接成功\n")

    # 2. 测试单个embedding
    print("[2/4] 测试单个embedding...")
    provider = create_ollama_provider(model="bge-m3")
    test_text = "Hello, this is a test for Ollama embedding!"
    embedding = await provider._embed_single(test_text)

    print(f"  文本: {test_text}")
    print(f"  向量维度: {len(embedding)}")
    print(f"  前5个值: {embedding[:5]}")
    print(f"  后5个值: {embedding[-5:]}")
    print("✅ 单个embedding成功\n")

    # 3. 测试批量embedding
    print("[3/4] 测试批量embedding...")
    test_texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning models require large datasets.",
        "Natural language processing enables human-computer interaction.",
        "Computer vision allows machines to interpret visual information.",
        "Deep learning uses neural networks with multiple layers."
    ]

    start = time.time()
    embeddings = await provider.get_embeddings(test_texts)
    elapsed = time.time() - start

    print(f"  文本数量: {len(test_texts)}")
    print(f"  耗时: {elapsed:.2f}秒")
    print(f"  平均速度: {len(test_texts)/elapsed:.1f} 文本/秒")
    print(f"  所有向量维度一致: {all(len(emb) == len(embeddings[0]) for emb in embeddings)}")
    print("✅ 批量embedding成功\n")

    # 4. 测试向量相似度
    print("[4/4] 测试向量相似度...")
    import numpy as np

    # 计算相似度矩阵
    embeddings_array = np.array(embeddings)
    similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
    norms = np.linalg.norm(embeddings_array, axis=1)
    similarity_matrix = similarity_matrix / np.outer(norms, norms)

    print("  相似度矩阵:")
    for i, text1 in enumerate(test_texts):
        print(f"  [{i+1}] {text1[:40]}...")
        for j, text2 in enumerate(test_texts):
            if i != j:
                sim = similarity_matrix[i][j]
                print(f"      -> [{j+1}] 相似度: {sim:.4f}")
    print("✅ 相似度计算成功\n")

    await provider._close()
    return True


async def test_performance():
    """测试性能"""
    print("=" * 60)
    print("🚀 Ollama Embedding Provider - 性能测试")
    print("=" * 60)

    provider = create_ollama_provider(
        model="bge-m3",
        batch_size=10
    )

    # 测试不同批量大小的性能
    batch_sizes = [1, 5, 10, 20, 50]

    print("\n批量大小性能对比:")
    print("-" * 60)
    print(f"{'批量大小':<10} {'文本数量':<10} {'耗时(秒)':<12} {'速度(文本/秒)':<15}")
    print("-" * 60)

    for batch_size in batch_sizes:
        texts = [f"Test text number {i}" for i in range(batch_size)]

        start = time.time()
        embeddings = await provider.get_embeddings(texts)
        elapsed = time.time() - start

        speed = batch_size / elapsed
        print(f"{batch_size:<10} {batch_size:<10} {elapsed:<12.2f} {speed:<15.1f}")

    print("-" * 60)
    print("✅ 性能测试完成\n")

    await provider._close()


async def test_error_handling():
    """测试错误处理"""
    print("=" * 60)
    print("🛡️  Ollama Embedding Provider - 错误处理测试")
    print("=" * 60)

    # 1. 测试无效服务地址
    print("\n[1/3] 测试无效服务地址...")
    try:
        provider = create_ollama_provider(
            base_url="http://invalid-host:11434",
            model="bge-m3",
            timeout=5.0
        )
        await provider._embed_single("test")
        print("❌ 应该抛出连接错误")
    except Exception as e:
        print(f"✅ 正确抛出错误: {type(e).__name__}")
        print(f"   错误信息: {str(e)[:80]}...")

    # 2. 测试无效模型
    print("\n[2/3] 测试无效模型...")
    try:
        provider = create_ollama_provider(
            model="invalid-model-name",
            timeout=30.0
        )
        await provider._embed_single("test")
        print("❌ 应该抛出模型不存在错误")
    except Exception as e:
        print(f"✅ 正确抛出错误: {type(e).__name__}")
        print(f"   错误信息: {str(e)[:80]}...")

    # 3. 测试超时
    print("\n[3/3] 测试超时处理...")
    try:
        provider = create_ollama_provider(
            model="bge-m3",
            timeout=0.001  # 极短超时
        )
        await provider._embed_single("test")
        print("⚠️  超时测试未触发（可能是网络太快）")
    except Exception as e:
        print(f"✅ 超时处理正常: {type(e).__name__}")

    print("\n✅ 错误处理测试完成\n")


async def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("🦙 Ollama Embedding Provider - 完整测试套件")
    print("=" * 60 + "\n")

    # 选择测试类型
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        print("请选择测试类型:")
        print("  1. 基础功能测试 (basic)")
        print("  2. 性能测试 (performance)")
        print("  3. 错误处理测试 (error)")
        print("  4. 全部测试 (all)")
        print()

        test_type = input("请输入选项 (1-4 或 test name): ").strip().lower()

        # 映射选项到测试名称
        test_map = {
            "1": "basic",
            "2": "performance",
            "3": "error",
            "4": "all"
        }
        test_type = test_map.get(test_type, test_type)

    try:
        if test_type in ["basic", "1", "all"]:
            await test_basic_functionality()

        if test_type in ["performance", "2", "all"]:
            await test_performance()

        if test_type in ["error", "3", "all"]:
            await test_error_handling()

        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
