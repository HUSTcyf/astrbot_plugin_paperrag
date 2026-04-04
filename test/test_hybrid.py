#!/usr/bin/env python3
"""
测试混合架构RAG系统

功能：
1. 测试PDF解析
2. 测试索引创建
3. 测试混合检索

用法：
    python test_hybrid.py                     # 使用默认 papers 目录
    python test_hybrid.py --papers-dir /path/to/pdfs  # 指定目录
"""

import sys
import asyncio
import argparse
from pathlib import Path

# 添加插件路径
plugin_dir = Path(__file__).parent.parent
sys.path.insert(0, str(plugin_dir))

# 抑制警告
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from hybrid_parser import HybridPDFParser
from hybrid_index import HybridIndexManager
from hybrid_rag import HybridRAGEngine
from rag_engine import RAGConfig


class MockContext:
    """模拟AstrBot上下文"""
    pass


async def test_pdf_parsing(papers_dir: Path):
    """测试PDF解析"""
    print("=" * 60)
    print("测试1: PDF解析")
    print("=" * 60)
    print(f"📁 PDF目录: {papers_dir}")

    # 创建解析器
    parser = HybridPDFParser(
        enable_multimodal=True,
        chunk_size=512,
        chunk_overlap=50
    )

    if not papers_dir.exists():
        print("❌ papers目录不存在，跳过PDF解析测试")
        return

    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ papers目录中没有PDF文件，跳过PDF解析测试")
        return

    # 测试第一个PDF
    test_pdf = pdf_files[0]
    print(f"\n📄 测试文件: {test_pdf.name}")

    try:
        # 解析为Documents
        documents = parser.parse_pdf_to_documents(str(test_pdf))
        print(f"✅ 解析成功: {len(documents)} 个documents")

        if documents:
            doc = documents[0]
            print(f"   - 文本长度: {len(doc.text)} chars")
            print(f"   - 元数据: {list(doc.metadata.keys())}")
            print(f"   - 页数: {doc.metadata.get('total_pages', 0)}")
            print(f"   - 表格数: {doc.metadata.get('tables_count', 0)}")
            print(f"   - 公式数: {doc.metadata.get('formulas_count', 0)}")
            print(f"   - 图片数: {doc.metadata.get('images_count', 0)}")

        # 解析为Nodes
        nodes = await parser.parse_and_split(str(test_pdf))
        print(f"✅ 分块成功: {len(nodes)} 个nodes")

        if nodes:
            node = nodes[0]
            print(f"   - Node文本长度: {len(node.text)} chars")
            print(f"   - Node元数据: {list(node.metadata.keys())}")

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()


async def test_index_manager():
    """测试索引管理器"""
    print("\n" + "=" * 60)
    print("测试2: 索引管理器")
    print("=" * 60)

    try:
        # 创建索引管理器
        manager = HybridIndexManager(
            milvus_uri="./data/milvus_papers.db",
            collection_name="test_paper_embeddings",
            embed_dim=1024,
            hybrid_search=True
        )

        print("✅ HybridIndexManager初始化成功")

        # 获取统计信息
        stats = await manager.get_stats()
        print(f"📊 索引统计: {stats}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


async def test_hybrid_rag():
    """测试混合RAG引擎"""
    print("\n" + "=" * 60)
    print("测试3: 混合RAG引擎")
    print("=" * 60)

    try:
        # 创建配置
        config = RAGConfig(
            embedding_mode="ollama",
            ollama_config={"model": "bge-m3", "base_url": "http://localhost:11434"},
            milvus_lite_path="./data/milvus_papers.db",
            collection_name="test_paper_embeddings",
            embed_dim=1024,
            chunk_size=512,
            chunk_overlap=50,
            top_k=5,
            enable_multimodal=True
        )

        # 创建引擎
        context = MockContext()
        engine = HybridRAGEngine(config, context)

        print("✅ HybridRAGEngine初始化成功")

        # 获取统计信息
        stats = await engine.get_stats()
        print(f"📊 引擎统计: {stats}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="测试混合架构RAG系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    python test_hybrid.py
    python test_hybrid.py --papers-dir ./my_papers
    python test_hybrid.py -p /Volumes/data/papers
        """
    )
    parser.add_argument(
        "-p", "--papers-dir",
        type=str,
        default=None,
        help="PDF文件目录路径（默认：插件目录下的 papers 文件夹）"
    )
    return parser.parse_args()


async def main():
    """主测试函数"""
    args = parse_args()

    # 确定 papers 目录
    if args.papers_dir:
        papers_dir = Path(args.papers_dir)
    else:
        papers_dir = plugin_dir.parent / "papers"

    print("\n🧪 混合架构RAG系统测试\n")

    # 测试1: PDF解析
    await test_pdf_parsing(papers_dir)

    # 测试2: 索引管理器
    await test_index_manager()

    # 测试3: 混合RAG引擎
    await test_hybrid_rag()

    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
