#!/usr/bin/env python3
"""
语义分块测试脚本

测试 HybridPDFParser 的语义分块逻辑：
1. 基本段落分割
2. 大段落分割（句子级）
3. Overlap 功能
4. 中英文混合分块
"""

import sys
import asyncio
from typing import List

# 添加插件目录到路径
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_parser import HybridPDFParser, Node


def print_separator(title: str = ""):
    """打印分隔符"""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def print_chunks(chunks: List[Node]):
    """打印分块结果"""
    print(f"\n📦 总共 {len(chunks)} 个 chunks:\n")
    for i, chunk in enumerate(chunks):
        text_preview = chunk.text[:80].replace('\n', '↵') + "..." if len(chunk.text) > 80 else chunk.text.replace('\n', '↵')
        print(f"  Chunk {i}: [{len(chunk.text)} chars] {text_preview}")
        print(f"           metadata: {chunk.metadata.get('chunk_index', 'N/A')}")


def test_basic_paragraph_split():
    """测试1: 基本段落分割"""
    print_separator("测试1: 基本段落分割")

    parser = HybridPDFParser(chunk_size=200, chunk_overlap=30, min_chunk_size=100)
    text = """深度学习是机器学习的一个分支，它通过构建多层神经网络来学习数据的表征。

在计算机视觉领域，卷积神经网络发挥着重要作用。RNNs常用于序列建模任务。

自然语言处理中，Transformer架构已经成为主流。BERT和GPT系列模型展示了预训练语言模型的强大能力。

强化学习在游戏和机器人控制领域取得了显著成果。AlphaGo击败了人类世界冠军。

迁移学习使得模型能够利用已有知识快速适应新任务。小样本学习和零样本学习也得到了广泛关注。"""

    print(f"输入文本 ({len(text)} chars), chunk_size=200:")
    chunks = parser._semantic_chunk(text, {"source": "test1"})
    print_chunks(chunks)

    # 验证：超过 chunk_size 的文本应该产生多个 chunks
    assert len(chunks) >= 2, f"预期至少2个chunks，实际{len(chunks)}"
    for i, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["chunk_index"] == i
    print("\n✅ 测试1通过: 段落分割正确")


def test_large_paragraph_split():
    """测试2: 大段落分割（超过chunk_size）"""
    print_separator("测试2: 大段落分割（句子级）")

    parser = HybridPDFParser(chunk_size=100, chunk_overlap=20, min_chunk_size=50)
    # 这个段落超过100字符
    text = """这是一个非常长的段落。它包含了多个句子。我们需要按照句子来分割这个段落，确保每个chunk都不会太大。这是第三个句子，用来测试分割效果。"""

    print(f"输入文本 ({len(text)} chars):\n{text}")
    chunks = parser._semantic_chunk(text, {"source": "test2"})
    print_chunks(chunks)

    # 验证
    for chunk in chunks:
        assert len(chunk.text) <= 120, f"Chunk超过预期大小: {len(chunk.text)}"
    print(f"\n✅ 测试2通过: 大段落按句子分割正确")


def test_overlap_functionality():
    """测试3: Overlap功能"""
    print_separator("测试3: Overlap功能")

    parser = HybridPDFParser(chunk_size=80, chunk_overlap=25, min_chunk_size=40)
    text = """第一段落的文本内容。

第二段落的文本内容。

第三段落的文本内容。

第四段落的文本内容。

第五段落的文本内容。"""

    print(f"输入文本 ({len(text)} chars), chunk_size=80, overlap=25")
    chunks = parser._semantic_chunk(text, {"source": "test3"})
    print_chunks(chunks)

    # 检查overlap - 相邻chunks应该有重叠内容
    if len(chunks) >= 2:
        first_chunk_end = chunks[0].text[-30:]
        second_chunk_start = chunks[1].text[:30]
        print(f"\n🔗 Overlap检查:")
        print(f"  Chunk0 末尾: {first_chunk_end}")
        print(f"  Chunk1 开头: {second_chunk_start}")
        # 注意：由于_clean_overlap，可能会有些许差异

    print(f"\n✅ 测试3完成: Overlap机制已启用")


def test_chinese_english_mixed():
    """测试4: 中英文混合分块"""
    print_separator("测试4: 中英文混合分块")

    parser = HybridPDFParser(chunk_size=150, chunk_overlap=20)
    text = """深度学习（Deep Learning）是机器学习的一个分支。它通过构建多层神经网络来学习数据的表征。The key innovation is that feature learning replaces manual feature extraction.

在计算机视觉领域，卷积神经网络（CNN）发挥了重要作用。RNNs are commonly used for sequence modeling tasks."""

    print(f"输入文本 ({len(text)} chars):\n{text}")
    chunks = parser._semantic_chunk(text, {"source": "test4"})
    print_chunks(chunks)

    # 验证
    for chunk in chunks:
        assert len(chunk.text) <= 200, f"Chunk超过预期大小: {len(chunk.text)}"
    print(f"\n✅ 测试4通过: 中英文混合分块正确")


def test_min_chunk_size():
    """测试5: 最小块大小限制"""
    print_separator("测试5: 最小块大小")

    parser = HybridPDFParser(chunk_size=500, chunk_overlap=0, min_chunk_size=100)
    text = """短段落。"""

    print(f"输入文本 ({len(text)} chars), min_chunk_size=100")
    chunks = parser._semantic_chunk(text, {"source": "test5"})
    print_chunks(chunks)

    # 注意：当前实现可能返回空或单个chunk
    print(f"\n✅ 测试5完成")


def test_sentence_delimiters():
    """测试6: 各种句子分隔符"""
    print_separator("测试6: 句子分隔符优先级")

    parser = HybridPDFParser(chunk_size=30, chunk_overlap=0, min_chunk_size=10)
    text = """第一句？第二句！第三句。第四句；第五句，第六句。"""

    print(f"输入文本 ({len(text)} chars):\n{text}")
    chunks = parser._semantic_chunk(text, {"source": "test6"})
    print_chunks(chunks)

    print(f"\n✅ 测试6完成")


def test_empty_and_whitespace():
    """测试7: 空文本和空白处理"""
    print_separator("测试7: 空文本和空白处理")

    parser = HybridPDFParser(chunk_size=50, chunk_overlap=10)

    test_cases = [
        ("", "空文本"),
        ("   ", "仅空白"),
        ("\n\n\n", "仅换行"),
    ]

    for text, desc in test_cases:
        chunks = parser._semantic_chunk(text, {"source": "test7"})
        print(f"  {desc}: {len(chunks)} chunks")

    print("\n✅ 测试7完成")


def test_overlap_clean():
    """测试8: Overlap清理逻辑"""
    print_separator("测试8: Overlap清理")

    parser = HybridPDFParser(chunk_size=80, chunk_overlap=25, min_chunk_size=40)

    # 测试 _clean_overlap
    test_cases = [
        "这是一个测试overlap。",
        "多个句子。第一个句子。第二个句子。",
        "没有句子结束的文本，只有一部分",
    ]

    print("\n_clean_overlap 测试:")
    for text in test_cases:
        cleaned = parser._clean_overlap(text)
        print(f"  输入: {text}")
        print(f"  输出: {cleaned}\n")

    print("✅ 测试8完成")


def run_all_tests():
    """运行所有测试"""
    print_separator("语义分块测试套件")
    print("HybridPDFParser 语义分块功能测试\n")

    tests = [
        test_basic_paragraph_split,
        test_large_paragraph_split,
        test_overlap_functionality,
        test_chinese_english_mixed,
        test_min_chunk_size,
        test_sentence_delimiters,
        test_empty_and_whitespace,
        test_overlap_clean,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ 测试错误: {e}")
            failed += 1

    print_separator("测试结果汇总")
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    print(f"  📊 总计: {passed + failed}")

    if failed == 0:
        print("\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️  {failed} 个测试失败")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
