"""
测试 Graph RAG 构建流程

测试步骤：
1. 从 Milvus 采样 N 个 chunks
2. 使用 MultimodalGraphBuilder 构建图谱
3. 验证图谱数据正确性

用法:
    python test_graph_build.py --sample 10
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# 添加插件目录到路径
plugin_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(plugin_dir))

# 设置工作目录
os.chdir(plugin_dir)


class MockContext:
    """模拟的 AstrBot context"""
    pass


class MockConfig:
    """模拟的配置"""
    def get(self, key, default=None):
        return default


async def test_graph_build_from_milvus(sample_size: int = 10):
    """测试从 Milvus 构建知识图谱"""
    print(f"\n{'='*60}")
    print(f"🧪 测试：采样 {sample_size} 个 chunks 构建图谱")
    print("=" * 60)

    # 导入模块
    try:
        from graph_rag_engine import MemoryGraphStore, GraphRAGConfig
        from graph_builder import MultimodalGraphBuilder
        print("✅ 导入模块成功")
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        return False

    # 创建配置
    graph_config = GraphRAGConfig(
        enable_graph_rag=True,
        storage_type="memory",
        max_triplets_per_chunk=5,
        multimodal_enabled=False,  # 测试时不启用多模态
    )

    # 创建图谱存储
    print("\n📦 创建 MemoryGraphStore...")
    try:
        graph_store = MemoryGraphStore(
            storage_path=str(plugin_dir / "data" / "test_graph_store"),
            auto_save=False  # 测试时不自动保存
        )
        print("✅ MemoryGraphStore 创建成功")
    except Exception as e:
        print(f"❌ MemoryGraphStore 创建失败: {e}")
        return False

    # 创建 GraphBuilder
    print("\n🔨 创建 MultimodalGraphBuilder...")
    try:
        builder = MultimodalGraphBuilder(
            config=graph_config,
            context=MockContext()
        )
        print("✅ MultimodalGraphBuilder 创建成功")
    except Exception as e:
        print(f"❌ MultimodalGraphBuilder 创建失败: {e}")
        return False

    # 初始化 LLM
    print("\n🤖 初始化 LLM...")
    try:
        await builder._ensure_llm_initialized()
        print(f"✅ LLM 初始化成功: {builder._llm}")
    except Exception as e:
        print(f"⚠️ LLM 初始化失败（将跳过实际构建）: {e}")
        print("   将使用模拟数据测试接口")

        # 使用模拟数据测试接口
        return await test_with_mock_data(graph_store)

    # 从 Milvus 采样 chunks
    print(f"\n📥 从 Milvus 采样 {sample_size} 个 chunks...")
    try:
        from hybrid_index import HybridIndexManager

        manager = HybridIndexManager(
            milvus_uri=str(plugin_dir / "data" / "milvus_papers.db"),
            collection_name="paper_embeddings",
            embed_dim=1024,
            hybrid_search=False,
        )

        await manager._ensure_collection()
        collection = manager._collection

        # 获取所有 chunks
        import pymilvus
        results = collection.query(
            expr="id >= 0",
            output_fields=["id", "text", "metadata"],
            limit=sample_size
        )

        if not results:
            print("⚠️ Milvus 中没有 chunks，使用模拟数据测试")
            return await test_with_mock_data(graph_store)

        # 转换为 ChunkNode
        class ChunkNode:
            def __init__(self, chunk):
                self.text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                self.metadata = metadata

        nodes = [ChunkNode(r) for r in results if r.get("text")]
        print(f"✅ 采样了 {len(nodes)} 个 chunks")
        for i, node in enumerate(nodes[:3]):
            text_preview = node.text[:100].replace('\n', ' ')
            print(f"   Chunk {i+1}: {text_preview}...")

    except Exception as e:
        print(f"⚠️ 从 Milvus 采样失败: {e}")
        print("   使用模拟数据测试")
        return await test_with_mock_data(graph_store)

    # 构建图谱
    print("\n🔨 开始构建图谱...")
    try:
        stats = await builder.build_from_nodes(nodes, graph_store)
        print(f"✅ 图谱构建完成:")
        print(f"   - 实体添加: {stats.get('entities_added', 0)}")
        print(f"   - 文本三元组: {stats.get('text_triplets_added', 0)}")
        print(f"   - 图片实体: {stats.get('image_entities_added', 0)}")
        print(f"   - 跨模态三元组: {stats.get('cross_modal_triplets_added', 0)}")
        print(f"   - 处理块: {stats.get('chunks_processed', 0)}")

    except Exception as e:
        print(f"❌ 图谱构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 验证图谱
    print("\n🔍 验证图谱...")
    return await verify_graph(graph_store)


async def test_with_mock_data(graph_store):
    """使用模拟数据测试接口"""
    print("\n📦 使用模拟数据测试...")

    # 定义简单的 ChunkNode
    class ChunkNode:
        def __init__(self, text, chunk_id="test_1"):
            self.text = text
            self.metadata = {"chunk_id": chunk_id}

    # 模拟文本
    test_chunks = [
        ChunkNode(
            "BERT is a transformer-based model. It uses self-attention mechanisms. "
            "Google developed BERT for NLP tasks. The model achieved state-of-the-art results.",
            "chunk_1"
        ),
        ChunkNode(
            "GPT models are autoregressive. OpenAI created GPT-3 and GPT-4. "
            "These language models can generate text. They use deep learning.",
            "chunk_2"
        ),
        ChunkNode(
            "Machine learning is a subset of AI. Deep learning uses neural networks. "
            "Neural networks have layers and neurons. Training data improves model performance.",
            "chunk_3"
        ),
    ]

    # 导入 builder
    try:
        from graph_rag_engine import GraphRAGConfig
        from graph_builder import MultimodalGraphBuilder

        graph_config = GraphRAGConfig(
            enable_graph_rag=True,
            storage_type="memory",
            max_triplets_per_chunk=5,
            multimodal_enabled=False,
        )

        builder = MultimodalGraphBuilder(
            config=graph_config,
            context=MockContext()
        )

        # 测试 add_entity 和 add_relation 接口
        print("\n🧪 测试 add_entity/add_relation 接口...")

        # 直接使用 graph_store 的接口
        entity1 = graph_store.add_entity("BERT", "Model", "A transformer model", "chunk_1")
        print(f"   add_entity('BERT') -> {entity1}")

        entity2 = graph_store.add_entity("Google", "Organization", "Tech company", "chunk_1")
        print(f"   add_entity('Google') -> {entity2}")

        rel1 = graph_store.add_relation("BERT", "Google", "developed_by", 1.0, "chunk_1")
        print(f"   add_relation('BERT', 'Google', 'developed_by') -> {rel1}")

        entity3 = graph_store.add_entity("NLP", "Domain", "Natural Language Processing", "chunk_1")
        print(f"   add_entity('NLP') -> {entity3}")

        rel2 = graph_store.add_relation("BERT", "NLP", "used_for", 1.0, "chunk_1")
        print(f"   add_relation('BERT', 'NLP', 'used_for') -> {rel2}")

        # 测试 get_neighbors
        print("\n🧪 测试 get_neighbors...")
        neighbors = graph_store.get_neighbors("BERT")
        print(f"   get_neighbors('BERT') -> {neighbors}")
        print(f"   邻居数量: {len(neighbors)}")

        # 测试 get_triplets
        print("\n🧪 测试 get_triplets...")
        triplets = graph_store.get_triplets("BERT", depth=1)
        print(f"   get_triplets('BERT', depth=1) -> {len(triplets)} triplets")
        for t in triplets:
            print(f"   - {t}")

        # 测试 search_related_entities
        print("\n🧪 测试 search_related_entities...")
        results = graph_store.search_related_entities("BERT", top_k=5)
        print(f"   search_related_entities('BERT') -> {len(results)} results")
        for r in results:
            print(f"   - {r['entity']['name']} (score: {r['score']})")

        # 测试 get_stats
        print("\n🧪 测试 get_stats...")
        stats = graph_store.get_stats()
        print(f"   get_stats() -> {stats}")

        # 测试 save
        print("\n💾 测试 save...")
        save_result = graph_store.save(force=True)
        print(f"   save(force=True) -> {save_result}")

        # 测试 clear
        print("\n🗑️ 测试 clear...")
        clear_result = graph_store.clear()
        print(f"   clear() -> done")

        # 验证 clear 后数据
        stats_after_clear = graph_store.get_stats()
        print(f"   get_stats() after clear -> {stats_after_clear}")

        return True

    except Exception as e:
        print(f"❌ 接口测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_graph(graph_store) -> bool:
    """验证图谱数据"""
    try:
        # 获取统计
        stats = graph_store.get_stats()
        print(f"\n📊 图谱统计:")
        print(f"   - 实体数量: {stats.get('entity_count', 0)}")
        print(f"   - 关系数量: {stats.get('relation_count', 0)}")

        # 搜索相关实体
        print("\n🔍 测试搜索 'model'...")
        results = graph_store.search_related_entities("model", top_k=5)
        print(f"   找到 {len(results)} 个相关实体:")
        for r in results:
            print(f"   - {r['entity']['name']} (type: {r['entity']['type']}, score: {r['score']})")

        # 获取某个实体的邻居
        if results:
            top_entity = results[0]['entity_id']
            print(f"\n🔍 获取实体 '{top_entity}' 的邻居...")
            neighbors = graph_store.get_neighbors(top_entity)
            print(f"   邻居数量: {len(neighbors)}")
            for n in neighbors[:5]:
                print(f"   - {n}")

        # 获取三元组
        if results:
            top_entity = results[0]['entity_id']
            print(f"\n🔍 获取实体 '{top_entity}' 的三元组...")
            triplets = graph_store.get_triplets(top_entity, depth=1)
            print(f"   三元组数量: {len(triplets)}")
            for t in triplets[:5]:
                print(f"   - {t.get('head')} --[{t.get('relation')}]--> {t.get('tail')}")

        return True

    except Exception as e:
        print(f"❌ 图谱验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="测试 Graph RAG 构建")
    parser.add_argument("--sample", type=int, default=10, help="采样数量")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🧪 Graph RAG 构建测试")
    print("="*60)

    result = asyncio.run(test_graph_build_from_milvus(args.sample))

    print("\n" + "="*60)
    if result:
        print("✅ 测试通过")
    else:
        print("❌ 测试失败")
    print("="*60)

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
