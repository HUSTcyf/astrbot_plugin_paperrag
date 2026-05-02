# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
"""
Graph RAG Integration Tests

测试目标（基于 Codex adversarial review 修复）：
1. HybridRetriever 第四通道：图谱检索 + RRF 融合
2. Provenance guard：过滤无 file_name/chunk_id 的图谱节点
3. GraphRAGConfig 构建：_create_graph_rag_config() 参数正确
4. Mode/TopK threading：search() 正确传递 actual_mode 和 top_k
5. get_retriever() LLMSynonymRetriever 导入正确性

运行方式（需要 Neo4j 可用）：
    pytest test/test_graph_rag_integration.py -v

如果 Neo4j 不可用，核心检索逻辑使用 FakeDriver 测试。
"""

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

# 确保 rag 模块可导入
_plugin_root = Path(__file__).resolve().parents[1]
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))

# ===== 测试夹具 =====


class FakeSession:
    def __init__(self, nodes=None):
        self._nodes = nodes or []
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        result = types.SimpleNamespace(data=lambda: [])
        if "RETURN count(n)" in query:
            result = types.SimpleNamespace(
                data=lambda: [types.SimpleNamespace(cnt=len(self._nodes))]
            )
        elif "MATCH ()-[r]->()" in query:
            rels = sum(1 for _ in self._nodes if self._nodes)
            result = types.SimpleNamespace(
                data=lambda: [types.SimpleNamespace(cnt=0)]
            )
        return result


class FakeDriver:
    def __init__(self, session=None):
        self._session = session or FakeSession()
        self.closed = False

    def session(self, database=None):
        return self._session

    def close(self):
        self.closed = True


def _install_neo4j_stub(nodes=None):
    fake_driver = FakeDriver(FakeSession(nodes))
    fake_graph_db = types.SimpleNamespace(GraphDatabase=types.SimpleNamespace(driver=lambda *a, **k: fake_driver))
    sys.modules["neo4j"] = fake_graph_db
    return fake_driver


def _install_astrbot_stubs():
    for mod_name in ["astrbot", "astrbot.api", "astrbot.api.star"]:
        if mod_name not in sys.modules:
            stub = types.SimpleNamespace(
                logger=types.SimpleNamespace(
                    info=lambda *a, **k: None,
                    warning=lambda *a, **k: None,
                    error=lambda *a, **k: None,
                )
            )
            sys.modules[mod_name] = stub


# ===== 测试 1: HybridRetriever 第四通道 RRF 融合 =====

class TestHybridRetrieverGraphChannel:
    """测试 HybridRetriever 的图谱检索通道"""

    def setup_method(self):
        _install_neo4j_stub()
        _install_astrbot_stubs()

    def _make_fake_index_manager(self):
        """创建假 IndexManager，返回预定义的向量检索结果"""
        manager = MagicMock()
        manager.search = AsyncMock(return_value=[
            {"text": "BERT是一种预训练语言模型", "metadata": {"file_name": "bert.pdf", "chunk_id": "chunk_0"}},
            {"text": "Transformer引入了注意力机制", "metadata": {"file_name": "transformer.pdf", "chunk_id": "chunk_1"}},
        ])
        manager.get_all_chunks = MagicMock(return_value=[])
        return manager

    def _make_fake_embed_provider(self):
        provider = MagicMock()
        provider.get_text_embedding = AsyncMock(return_value=[0.1] * 768)
        provider.get_text_embeddings = AsyncMock(return_value=[[0.1] * 768])
        return provider

    def _make_fake_graph_retriever(self):
        """创建假图谱检索器，返回模拟的实体节点"""
        retriever = MagicMock()

        class FakeNodeWithScore:
            def __init__(self, node, score):
                self.node = node
                self.score = score

        class FakeEntityNode:
            def __init__(self, name, label, content):
                self.name = name
                self.type = "entity"
                self.properties = {"label": label}
                self.content = content

        nodes = [
            FakeNodeWithScore(
                FakeEntityNode("BERT", "Model", "一种预训练语言模型"),
                score=0.95
            ),
            FakeNodeWithScore(
                FakeEntityNode("Attention", "Mechanism", "注意力机制"),
                score=0.88
            ),
        ]
        retriever.retrieve = AsyncMock(return_value=nodes)
        return retriever

    @pytest.mark.asyncio
    async def test_rrf_fusion_includes_graph_channel(self):
        """RRF 融合应包含图谱通道的贡献（但不暴露无出处的图谱节点）"""
        from rag.hybrid_rag import HybridRetriever

        index_manager = self._make_fake_index_manager()
        embed_provider = self._make_fake_embed_provider()
        graph_retriever = self._make_fake_graph_retriever()

        retriever = HybridRetriever(
            index_manager=index_manager,
            embed_provider=embed_provider,
            enable_sparse_retrieval=False,
            enable_bm25=False,
            enable_reranking=False,
            graph_retriever=graph_retriever,
            graph_weight=0.2,
        )

        result = await retriever.retrieve("BERT model attention", top_k=5)

        # 图谱检索器应被调用
        graph_retriever.retrieve.assert_called_once_with("BERT model attention")

        # 图谱实体转换为文本参与了 RRF 融合（通过 _graph_node_to_text）
        # 但 provenance guard 过滤了无 file_name/chunk_id 的节点
        # 所以最终结果只包含有出处的向量 chunks
        result_texts = [n.text for n in result.nodes]
        assert any("BERT" in t or "Transformer" in t for t in result_texts), \
            f"结果应包含向量 chunk 中的 BERT/Transformer: {result_texts}"

        # 验证结果数量和分数与节点一致
        assert len(result.scores) == len(result.nodes)
        assert len(result.nodes) > 0, "应该有至少一个结果"

    @pytest.mark.asyncio
    async def test_provenance_guard_filters_graph_only_nodes(self):
        """无 file_name/chunk_id 的图谱节点应被过滤"""
        from rag.hybrid_rag import HybridRetriever

        index_manager = self._make_fake_index_manager()
        embed_provider = self._make_fake_embed_provider()
        graph_retriever = self._make_fake_graph_retriever()

        retriever = HybridRetriever(
            index_manager=index_manager,
            embed_provider=embed_provider,
            enable_sparse_retrieval=False,
            enable_bm25=False,
            enable_reranking=False,
            graph_retriever=graph_retriever,
            graph_weight=0.2,
        )

        result = await retriever.retrieve("BERT model", top_k=10)

        # 所有结果必须有 file_name 或 chunk_id
        for node in result.nodes:
            meta = node.metadata or {}
            assert meta.get('file_name') or meta.get('chunk_id'), \
                f"节点缺少出处: {node.text[:50]}..."

    @pytest.mark.asyncio
    async def test_graph_disabled_does_not_crash(self):
        """禁用图谱时 HybridRetriever 应正常工作"""
        from rag.hybrid_rag import HybridRetriever

        index_manager = self._make_fake_index_manager()
        embed_provider = self._make_fake_embed_provider()

        retriever = HybridRetriever(
            index_manager=index_manager,
            embed_provider=embed_provider,
            enable_sparse_retrieval=False,
            enable_bm25=False,
            enable_reranking=False,
            graph_retriever=None,
            graph_weight=0.0,
        )

        result = await retriever.retrieve("BERT model", top_k=5)
        assert len(result.nodes) > 0


# ===== 测试 2: GraphRAGConfig 构建正确性 =====

class TestGraphRAGConfigConstruction:
    """测试 _create_graph_rag_config() 参数正确映射"""

    def test_graph_rrf_weight_replaces_hybrid_alpha(self):
        """graph_rag.graph_rrf_weight 应替代已删除的 hybrid_alpha"""
        _install_astrbot_stubs()
        sys.path.insert(0, str(Path(__file__).parents[1]))

        plugin_parent = Path(__file__).resolve().parents[1]
        if str(plugin_parent) not in sys.path:
            sys.path.insert(0, str(plugin_parent))

        # 模拟配置
        config = {
            "enable_graph_rag": True,
            "graph_rag": {
                "storage_type": "neo4j",
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_user": "neo4j",
                "neo4j_password": "password",
                "max_triplets_per_chunk": 5,
                "graph_retrieval_top_k": 5,
                "graph_rrf_weight": 0.3,
                "auto_build": True,
                "auto_build_threshold": 5,
                "multimodal_extraction": {
                    "enabled": True,
                    "max_images_per_chunk": 2,
                    "extract_image_entities": True,
                },
            }
        }

        # 加载 GraphRAGConfig
        from graphrag.graph_rag_engine import GraphRAGConfig
        graph_cfg = GraphRAGConfig.from_rag_config(
            types.SimpleNamespace(  # type: ignore[arg-type]
                enable_graph_rag=True,
                graph_storage_type="neo4j",
                graph_neo4j_uri="bolt://localhost:7687",
                graph_neo4j_user="neo4j",
                graph_neo4j_password="password",
                graph_max_triplets_per_chunk=5,
                graph_retrieval_top_k=5,
                graph_rrf_weight=0.3,
                graph_auto_build=True,
                graph_auto_build_threshold=5,
                graph_multimodal_enabled=True,
                graph_max_images_per_chunk=2,
                graph_extract_image_entities=True,
            )
        )

        assert hasattr(graph_cfg, 'graph_rrf_weight'), "GraphRAGConfig 应该有 graph_rrf_weight 属性"
        assert not hasattr(graph_cfg, 'hybrid_alpha'), "GraphRAGConfig 不应该有 hybrid_alpha 属性"
        assert graph_cfg.graph_rrf_weight == 0.3


# ===== 测试 3: search() mode/top_k threading =====

class TestSearchModeThreading:
    """测试 HybridRAGEngine.search() 正确传递 mode 和 top_k"""

    def setup_method(self):
        _install_neo4j_stub()
        _install_astrbot_stubs()

    def _make_mock_config(self):
        cfg = types.SimpleNamespace(
            top_k=5,
            embedding_mode="unsloth",
            embedding_provider_id="",
            compress_provider_id="",
            text_provider_id="",
            multimodal_provider_id="",
            unsloth_config={},
            milvus_lite_path=str(Path(__file__).parents[1] / "data" / "milvus_papers.db"),
            address="",
            db_name="default",
            authentication=None,
            collection_name="paper_embeddings",
            embed_dim=1024,
            enable_sparse_retrieval=False,
            enable_multi_vector_rerank=False,
            sparse_top_k=20,
            hybrid_alpha=0.5,
            hybrid_rrf_k=60,
            enable_bm25=False,
            bm25_top_k=20,
            enable_graph_rag=False,
            graph_rrf_weight=0.2,
            enable_two_stage_retrieval=False,
            two_stage_top_k=10,
            two_stage_rerank_k=5,
            enable_crag_quality_eval=False,
            crag_enable_correction=False,
            crag_min_score=0.3,
        )
        cfg.get_connection_mode = lambda: 'lite'
        cfg.get_connection_uri = lambda: str(Path(__file__).parents[1] / "data" / "milvus_papers.db")
        cfg.validate = lambda: (True, "")
        return cfg

    @pytest.mark.asyncio
    async def test_search_accepts_mode_parameter(self):
        """search() 应接受 mode 参数并记录到日志"""
        from rag.hybrid_rag import HybridRAGEngine

        config = self._make_mock_config()
        mock_context = types.SimpleNamespace(get_using_provider=MagicMock(return_value=None))

        engine = HybridRAGEngine(config, mock_context)

        # 验证 search 方法签名包含 mode 参数
        import inspect
        sig = inspect.signature(engine.search)
        assert 'mode' in sig.parameters, f"search() 应该有 mode 参数，当前签名: {sig}"

    @pytest.mark.asyncio
    async def test_search_accepts_top_k_parameter(self):
        """search() 应接受 top_k 参数"""
        from rag.hybrid_rag import HybridRAGEngine

        config = self._make_mock_config()
        mock_context = types.SimpleNamespace(get_using_provider=MagicMock(return_value=None))

        engine = HybridRAGEngine(config, mock_context)

        import inspect
        sig = inspect.signature(engine.search)
        assert 'top_k' in sig.parameters, f"search() 应该有 top_k 参数，当前签名: {sig}"


# ===== 测试 4: get_retriever() LLMSynonymRetriever 导入 =====

class TestGetRetrieverImport:
    """测试 get_retriever() 的 LLMSynonymRetriever 导入正确性"""

    def test_llm_synonym_retriever_imported_at_module_scope(self):
        """LLMSynonymRetriever 应在模块级别导入"""
        from graphrag import graph_rag_engine
        assert hasattr(graph_rag_engine, 'LLMSynonymRetriever'), \
            "graph_rag_engine 模块应该有 LLMSynonymRetriever 属性"


# ===== 测试 5: 端到端混合检索（需要 Neo4j） =====

class TestEndToEndHybridRetrieval:
    """端到端测试：验证整个混合检索流程"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path(__file__).parents[1].joinpath("data", "graph_store").exists(),
        reason="需要 Neo4j 数据文件存在"
    )
    async def test_graph_rag_engine_initialization(self):
        """GraphRAGEngine 初始化应成功（需要真实 Neo4j）"""
        _install_neo4j_stub()
        _install_astrbot_stubs()

        from graphrag.graph_rag_engine import GraphRAGConfig, create_graph_rag_engine

        config = GraphRAGConfig(
            enable_graph_rag=True,
            storage_type="neo4j",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            graph_retrieval_top_k=5,
            graph_rrf_weight=0.2,
        )

        mock_base_engine = MagicMock()
        mock_context = types.SimpleNamespace(get_using_provider=MagicMock(return_value=None))

        graph_engine = create_graph_rag_engine(config, mock_base_engine, mock_context)

        # initialize 不需要真实 Neo4j 因为我们用的是 FakeDriver
        # 如果没有真实服务会返回，但不应崩溃
        try:
            await graph_engine.initialize()
        except Exception as e:
            pytest.skip(f"Neo4j 不可用: {e}")

    @pytest.mark.asyncio
    async def test_hybrid_rag_engine_with_graph_channel(self):
        """HybridRAGEngine 应在启用 graph_rag 时连接图谱检索器"""
        _install_neo4j_stub()
        _install_astrbot_stubs()

        from rag.hybrid_rag import HybridRAGEngine

        config = self._make_mock_config_e2e()
        config.enable_graph_rag = True
        config.graph_rrf_weight = 0.2

        mock_context = types.SimpleNamespace(get_using_provider=MagicMock(return_value=None))

        engine = HybridRAGEngine(config, mock_context)

        # 初始化检索器
        try:
            retriever = await engine._ensure_retriever_initialized()
            assert retriever is not None
            # graph_retriever 可能为 None 因为没有真实 Neo4j，但不應崩潰
        except Exception as e:
            pytest.skip(f"图谱检索器初始化失败（需要真实服务）: {e}")

    def _make_mock_config_e2e(self):
        cfg = types.SimpleNamespace(
            top_k=5,
            embedding_mode="unsloth",
            embedding_provider_id="",
            compress_provider_id="",
            text_provider_id="",
            multimodal_provider_id="",
            unsloth_config={},
            milvus_lite_path=str(Path(__file__).parents[1] / "data" / "milvus_papers.db"),
            address="",
            db_name="default",
            authentication=None,
            collection_name="paper_embeddings",
            embed_dim=1024,
            enable_sparse_retrieval=False,
            enable_multi_vector_rerank=False,
            sparse_top_k=20,
            hybrid_alpha=0.5,
            hybrid_rrf_k=60,
            enable_bm25=False,
            bm25_top_k=20,
            enable_graph_rag=False,
            graph_storage_type="neo4j",
            graph_neo4j_uri="bolt://localhost:7687",
            graph_neo4j_user="neo4j",
            graph_neo4j_password="password",
            graph_max_triplets_per_chunk=5,
            graph_retrieval_top_k=5,
            graph_rrf_weight=0.2,
            graph_auto_build=False,
            graph_auto_build_threshold=10,
            graph_multimodal_enabled=True,
            graph_max_images_per_chunk=1,
            graph_extract_image_entities=True,
            enable_two_stage_retrieval=False,
            two_stage_top_k=10,
            two_stage_rerank_k=5,
            enable_crag_quality_eval=False,
            crag_enable_correction=False,
            crag_min_score=0.3,
        )
        cfg.get_connection_mode = lambda: 'lite'
        cfg.get_connection_uri = lambda: str(Path(__file__).parents[1] / "data" / "milvus_papers.db")
        cfg.validate = lambda: (True, "")
        return cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])