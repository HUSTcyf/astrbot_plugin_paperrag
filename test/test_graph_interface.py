"""
独立测试 Graph RAG 接口

不依赖 AstrBot 环境，直接测试 SimplePropertyGraphStoreAdapter 的接口。
"""

import sys
from pathlib import Path

# 添加插件目录到路径
plugin_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(plugin_dir))


# Mock astrbot.api.logger
class MockLogger:
    def info(self, msg): print(f"   [INFO] {msg}")
    def warning(self, msg): print(f"   [WARN] {msg}")
    def error(self, msg): print(f"   [ERROR] {msg}")
    def debug(self, msg): pass

import types
astrbot_mock = types.ModuleType('astrbot')
astrbot_api_mock = types.ModuleType('astrbot.api')
astrbot_api_mock.logger = MockLogger()
sys.modules['astrbot'] = astrbot_mock
sys.modules['astrbot.api'] = astrbot_api_mock


def test_interface():
    """测试 SimplePropertyGraphStoreAdapter 的接口"""
    print("\n" + "="*60)
    print("🧪 测试 SimplePropertyGraphStoreAdapter 接口")
    print("="*60)

    # 导入模块
    print("\n📦 导入模块...")
    try:
        from graph_rag_engine import SimplePropertyGraphStoreAdapter, PersistentPropertyGraphStoreAdapter
        from llama_index.core.graph_stores import SimplePropertyGraphStore
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

    # 创建适配器
    print("\n📦 创建适配器...")
    try:
        store = SimplePropertyGraphStore()
        adapter = SimplePropertyGraphStoreAdapter(store)
        print("✅ 适配器创建成功")
    except Exception as e:
        print(f"❌ 适配器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 1: add_entity
    print("\n🧪 测试 1: add_entity")
    try:
        entity1 = adapter.add_entity("BERT", "Model", "A transformer model", "chunk_1")
        print(f"   add_entity('BERT', 'Model', ...) -> {entity1}")
        assert entity1 == "BERT", f"Expected 'BERT', got {entity1}"

        entity2 = adapter.add_entity("Google", "Organization", "Tech company", "chunk_1")
        print(f"   add_entity('Google', 'Organization', ...) -> {entity2}")

        entity3 = adapter.add_entity("NLP", "Domain", "Natural Language Processing", "chunk_1")
        print(f"   add_entity('NLP', 'Domain', ...) -> {entity3}")

        # 再次添加相同实体应该返回相同 ID
        entity1_again = adapter.add_entity("BERT", "Model", "Updated description", "chunk_2")
        print(f"   add_entity('BERT', ...) again -> {entity1_again}")
        assert entity1_again == "BERT", "Duplicate entity should return same ID"
        print("✅ add_entity 测试通过")
    except Exception as e:
        print(f"❌ add_entity 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 2: add_relation
    print("\n🧪 测试 2: add_relation")
    try:
        rel1 = adapter.add_relation("BERT", "Google", "developed_by", 1.0, "chunk_1")
        print(f"   add_relation('BERT', 'Google', 'developed_by') -> {rel1}")
        assert rel1 is not None, "Relation should be created"
        assert "BERT" in rel1 and "developed_by" in rel1 and "Google" in rel1

        rel2 = adapter.add_relation("BERT", "NLP", "used_for", 1.0, "chunk_1")
        print(f"   add_relation('BERT', 'NLP', 'used_for') -> {rel2}")

        rel3 = adapter.add_relation("Google", "NLP", "focuses_on", 1.0, "chunk_1")
        print(f"   add_relation('Google', 'NLP', 'focuses_on') -> {rel3}")
        print("✅ add_relation 测试通过")
    except Exception as e:
        print(f"❌ add_relation 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 3: add_image_entity
    print("\n🧪 测试 3: add_image_entity")
    try:
        img_entity = adapter.add_image_entity(
            "Figure_1",
            "/path/to/image.png",
            "A neural network diagram",
            "diagram",
            "chunk_1"
        )
        print(f"   add_image_entity('Figure_1', ...) -> {img_entity}")
        assert img_entity == "Figure_1"

        # 添加图片与其他实体的关系
        img_rel = adapter.add_relation("Figure_1", "NeuralNetwork", "shows", 1.0, "chunk_1")
        print(f"   add_relation('Figure_1', 'NeuralNetwork', 'shows') -> {img_rel}")
        print("✅ add_image_entity 测试通过")
    except Exception as e:
        print(f"❌ add_image_entity 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 4: add_table_entity
    print("\n🧪 测试 4: add_table_entity")
    try:
        table_entity = adapter.add_table_entity(
            "Table_1",
            "Experimental results",
            "chunk_1"
        )
        print(f"   add_table_entity('Table_1', ...) -> {table_entity}")
        assert table_entity == "Table_1"
        print("✅ add_table_entity 测试通过")
    except Exception as e:
        print(f"❌ add_table_entity 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 5: get_neighbors
    print("\n🧪 测试 5: get_neighbors")
    try:
        neighbors = adapter.get_neighbors("BERT")
        print(f"   get_neighbors('BERT') -> {len(neighbors)} neighbors")
        for n in neighbors:
            print(f"      - {n}")

        # 测试不存在的实体
        neighbors_empty = adapter.get_neighbors("NonExistent")
        print(f"   get_neighbors('NonExistent') -> {len(neighbors_empty)} neighbors")
        print("✅ get_neighbors 测试通过")
    except Exception as e:
        print(f"❌ get_neighbors 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 6: get_triplets
    print("\n🧪 测试 6: get_triplets")
    try:
        triplets = adapter.get_triplets("BERT", depth=1)
        print(f"   get_triplets('BERT', depth=1) -> {len(triplets)} triplets")
        for t in triplets:
            print(f"      - {t}")

        # 多跳
        triplets_multi = adapter.get_triplets("BERT", depth=2)
        print(f"   get_triplets('BERT', depth=2) -> {len(triplets_multi)} triplets")
        print("✅ get_triplets 测试通过")
    except Exception as e:
        print(f"❌ get_triplets 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 7: search_related_entities
    print("\n🧪 测试 7: search_related_entities")
    try:
        results = adapter.search_related_entities("BERT", top_k=5)
        print(f"   search_related_entities('BERT') -> {len(results)} results")
        for r in results:
            print(f"      - {r['entity']['name']} (type: {r['entity']['type']}, score: {r['score']})")

        results2 = adapter.search_related_entities("Google", top_k=5)
        print(f"   search_related_entities('Google') -> {len(results2)} results")

        # 测试模糊搜索
        results3 = adapter.search_related_entities("model", top_k=5)
        print(f"   search_related_entities('model') -> {len(results3)} results")
        print("✅ search_related_entities 测试通过")
    except Exception as e:
        print(f"❌ search_related_entities 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 8: get_entity
    print("\n🧪 测试 8: get_entity")
    try:
        entity = adapter.get_entity("BERT")
        print(f"   get_entity('BERT') -> {entity}")
        assert entity is not None
        assert entity["name"] == "BERT"
        assert entity["type"] == "Model"
        assert entity["description"] == "A transformer model"

        entity2 = adapter.get_entity("NonExistent")
        print(f"   get_entity('NonExistent') -> {entity2}")
        assert entity2 is None
        print("✅ get_entity 测试通过")
    except Exception as e:
        print(f"❌ get_entity 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 9: get_stats
    print("\n🧪 测试 9: get_stats")
    try:
        stats = adapter.get_stats()
        print(f"   get_stats() -> {stats}")
        assert "entity_count" in stats
        assert "relation_count" in stats
        print("✅ get_stats 测试通过")
    except Exception as e:
        print(f"❌ get_stats 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试 10: clear
    print("\n🧪 测试 10: clear")
    try:
        adapter.clear()
        stats_after = adapter.get_stats()
        print(f"   get_stats() after clear -> {stats_after}")
        assert stats_after["entity_count"] == 0
        print("✅ clear 测试通过")
    except Exception as e:
        print(f"❌ clear 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_persistent_adapter():
    """测试 PersistentPropertyGraphStoreAdapter"""
    print("\n" + "="*60)
    print("🧪 测试 PersistentPropertyGraphStoreAdapter")
    print("="*60)

    try:
        from graph_rag_engine import PersistentPropertyGraphStoreAdapter
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

    # 创建临时存储路径
    import tempfile
    import shutil
    temp_dir = Path(tempfile.mkdtemp()) / "test_graph"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\n📦 创建 PersistentPropertyGraphStoreAdapter...")
        adapter = PersistentPropertyGraphStoreAdapter(
            storage_path=str(temp_dir),
            auto_save=False
        )
        print("✅ 创建成功")

        # 添加测试数据
        print("\n📝 添加测试数据...")
        adapter.add_entity("TestEntity", "Type", "A test entity", "chunk_1")
        adapter.add_relation("TestEntity", "AnotherEntity", "relates_to", 1.0, "chunk_1")

        stats = adapter.get_stats()
        print(f"   添加后统计: {stats}")

        # 测试 save
        print("\n💾 测试 save...")
        result = adapter.save(force=True)
        print(f"   save(force=True) -> {result}")
        assert result == True

        # 创建新实例并加载
        print("\n📦 创建新实例并加载...")
        adapter2 = PersistentPropertyGraphStoreAdapter(
            storage_path=str(temp_dir),
            auto_save=False
        )
        stats2 = adapter2.get_stats()
        print(f"   加载后统计: {stats2}")

        # 验证数据恢复
        entity = adapter2.get_entity("TestEntity")
        print(f"   get_entity('TestEntity') -> {entity}")
        assert entity is not None
        assert entity["name"] == "TestEntity"

        # 测试 clear(delete_storage=True)
        print("\n🗑️ 测试 clear(delete_storage=True)...")
        adapter2.clear(delete_storage=True)
        stats3 = adapter2.get_stats()
        print(f"   clear后统计: {stats3}")

        # 新实例应该没有数据
        adapter3 = PersistentPropertyGraphStoreAdapter(
            storage_path=str(temp_dir),
            auto_save=False
        )
        stats4 = adapter3.get_stats()
        print(f"   新实例统计: {stats4}")

        print("✅ PersistentPropertyGraphStoreAdapter 测试通过")
        return True

    except Exception as e:
        print(f"❌ PersistentPropertyGraphStoreAdapter 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir.parent, ignore_errors=True)


def test_multimodal_graph_builder_integration():
    """测试与 MultimodalGraphBuilder 的集成"""
    print("\n" + "="*60)
    print("🧪 测试 MultimodalGraphBuilder 集成")
    print("="*60)

    try:
        from graph_rag_engine import PersistentPropertyGraphStoreAdapter, GraphRAGConfig
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

    # 创建配置
    config = GraphRAGConfig(
        enable_graph_rag=True,
        storage_type="memory",
        max_triplets_per_chunk=5,
        multimodal_enabled=False,
    )

    # 创建存储
    import tempfile
    import shutil
    temp_dir = Path(tempfile.mkdtemp()) / "test_graph"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        store = PersistentPropertyGraphStoreAdapter(
            storage_path=str(temp_dir),
            auto_save=False
        )

        # 定义简单的 Node 类
        class SimpleNode:
            def __init__(self, text, metadata=None):
                self.text = text
                self.metadata = metadata or {}

        # 创建测试节点
        nodes = [
            SimpleNode(
                "BERT is a language model. It uses transformer architecture. Google developed BERT.",
                {"chunk_id": "doc_1"}
            ),
            SimpleNode(
                "GPT is also a language model. OpenAI created GPT. These models are different.",
                {"chunk_id": "doc_2"}
            ),
        ]

        print(f"\n📝 创建了 {len(nodes)} 个测试节点")

        # 由于 MultimodalGraphBuilder 需要 LLM，我们只测试接口兼容性
        print("\n🔍 测试存储接口是否与 MultimodalGraphBuilder 兼容...")

        # 检查必要的方法存在
        required_methods = [
            'add_entity', 'add_relation', 'add_image_entity', 'add_table_entity',
            'get_entity', 'get_neighbors', 'get_triplets',
            'search_related_entities', 'get_stats', 'clear'
        ]

        for method in required_methods:
            if not hasattr(store, method):
                print(f"❌ 缺少方法: {method}")
                return False
            print(f"   ✅ {method}")

        print("✅ 所有必要方法都存在")
        print("✅ MultimodalGraphBuilder 集成测试通过")
        return True

    except Exception as e:
        print(f"❌ MultimodalGraphBuilder 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir.parent, ignore_errors=True)


def main():
    print("\n" + "="*60)
    print("🧪 Graph RAG 接口测试")
    print("="*60)

    all_passed = True

    # 测试 1: 基本接口
    if not test_interface():
        all_passed = False

    # 测试 2: 持久化
    if not test_persistent_adapter():
        all_passed = False

    # 测试 3: MultimodalGraphBuilder 集成
    if not test_multimodal_graph_builder_integration():
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
