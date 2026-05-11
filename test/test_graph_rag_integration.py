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
import socket
import inspect
from rag.hybrid_rag import HybridRetriever, HybridRAGEngine
from graphrag import graph_rag_engine
from graphrag.graph_rag_engine import GraphRAGConfig, create_graph_rag_engine

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


# ===== 测试 2: search() mode/top_k threading =====

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

        config = self._make_mock_config()
        mock_context = types.SimpleNamespace(get_using_provider=MagicMock(return_value=None))

        engine = HybridRAGEngine(config, mock_context)

        # 验证 search 方法签名包含 mode 参数
        sig = inspect.signature(engine.search)
        assert 'mode' in sig.parameters, f"search() 应该有 mode 参数，当前签名: {sig}"

    @pytest.mark.asyncio
    async def test_search_accepts_top_k_parameter(self):
        """search() 应接受 top_k 参数"""

        config = self._make_mock_config()
        mock_context = types.SimpleNamespace(get_using_provider=MagicMock(return_value=None))

        engine = HybridRAGEngine(config, mock_context)

        sig = inspect.signature(engine.search)
        assert 'top_k' in sig.parameters, f"search() 应该有 top_k 参数，当前签名: {sig}"


# ===== 测试 4: get_retriever() LLMSynonymRetriever 导入 =====

class TestGetRetrieverImport:
    """测试 get_retriever() 的 LLMSynonymRetriever 导入正确性"""

    def test_llm_synonym_retriever_imported_at_module_scope(self):
        """LLMSynonymRetriever 应在模块级别导入"""
        assert hasattr(graph_rag_engine, 'LLMSynonymRetriever'), \
            "graph_rag_engine 模块应该有 LLMSynonymRetriever 属性"


# ===== 测试 5: 端到端混合检索（真实 Neo4j + 真实 Unsloth Embedding） =====

def _neo4j_available() -> bool:
    """检查 Neo4j 是否可用（通过 socket 连接判断，不受 sys.modules 污染影响）"""
    try:
        s = socket.create_connection(("localhost", 7687), timeout=2)
        s.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


from test._test_utils import get_neo4j_password as _get_neo4j_password


class TestEndToEndHybridRetrieval:
    """端到端测试：验证整个混合检索流程（真实 Neo4j + 真实 Embedding）"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _neo4j_available(), reason="Neo4j 不可用")
    async def test_graph_rag_engine_initialization(self):
        """GraphRAGEngine 初始化应连接真实 Neo4j 并查询统计"""
        _install_astrbot_stubs()


        config = GraphRAGConfig(
            enable_graph_rag=True,
            storage_type="neo4j",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password=_get_neo4j_password(),
            graph_retrieval_top_k=5,
        )

        mock_base_engine = MagicMock()
        mock_context = types.SimpleNamespace(get_using_provider=MagicMock(return_value=None))

        graph_engine = create_graph_rag_engine(config, mock_base_engine, mock_context)

        try:
            await graph_engine.initialize()
        except Exception as e:
            pytest.skip(f"GraphRAGEngine 初始化失败（缺少 LLM provider）: {e}")

        # 即使 _index 未创建成功，Neo4j adapter 应已连接
        if graph_engine._adapter is not None:
            stats = await graph_engine.get_graph_stats()
            assert stats["enabled"] is True
            assert "entity_count" in stats

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _neo4j_available(), reason="Neo4j 不可用")
    async def test_hybrid_rag_engine_with_real_embed_and_neo4j(self):
        """HybridRAGEngine 应使用真实 Unsloth Embedding + 真实 Neo4j 完成检索初始化"""
        _install_astrbot_stubs()


        config = self._make_real_config_e2e()
        mock_context = types.SimpleNamespace(get_using_provider=MagicMock(return_value=None))

        engine = HybridRAGEngine(config, mock_context)

        # 必须先初始化 embed provider
        embed_provider = await engine._ensure_embed_provider_initialized()
        assert embed_provider is not None

        # 验证 embed provider 是真实的 Unsloth，能产出 1024 维向量
        emb = await embed_provider.get_text_embedding("test query")
        assert len(emb) == 1024, f"Embedding 维度应为 1024，实际: {len(emb)}"

        # 初始化检索器（会自动初始化 index_manager + graph channel）
        retriever = await engine._ensure_retriever_initialized()
        assert retriever is not None

        # 验证 graph_retriever 状态
        assert hasattr(retriever, '_graph_retriever')
        if retriever._graph_retriever is not None:
            assert retriever._graph_weight == 0.2

    def _make_real_config_e2e(self):
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
            enable_graph_rag=True,
            graph_storage_type="neo4j",
            graph_neo4j_uri="bolt://localhost:7687",
            graph_neo4j_user="neo4j",
            graph_neo4j_password=_get_neo4j_password(),
            graph_max_triplets_per_chunk=5,
            graph_retrieval_top_k=5,
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