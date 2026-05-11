import asyncio
import gzip
import json
import sys
import tempfile
import types
from pathlib import Path

from test.test_main_split import _install_astrbot_stubs, _collect


class FakeGraphConfig:
    storage_type = "neo4j"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"


class FakeSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        return types.SimpleNamespace(data=lambda: [])

    def begin_transaction(self):
        return self

    def rollback(self):
        pass

    def commit(self):
        pass


class FakeDriver:
    def __init__(self, session):
        self._session = session
        self.closed = False

    def session(self, database=None):
        return self._session

    def close(self):
        self.closed = True


def _install_neo4j_stub(session):
    fake_driver = FakeDriver(session)

    class FakeGraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            return fake_driver

    sys.modules["neo4j"] = types.SimpleNamespace(GraphDatabase=FakeGraphDatabase)  # type: ignore[assignment]
    return fake_driver


def _make_plugin(tmp_path):
    # Clear possibly incomplete stubs from other test modules so the full
    # installation (including astrbot.api.event) can proceed.
    for mod_name in ["astrbot.api.event", "astrbot.api.star", "astrbot.api", "astrbot"]:
        sys.modules.pop(mod_name, None)
    _install_astrbot_stubs()

    plugin_parent = Path(__file__).resolve().parents[2]
    if str(plugin_parent) not in sys.path:
        sys.path.insert(0, str(plugin_parent))

    main_mod = __import__("astrbot_plugin_paperrag.main", fromlist=["PaperRAGPlugin"])
    graph_mod = __import__("astrbot_plugin_paperrag.commands.graph", fromlist=["_PLUGIN_DIR"])
    graph_mod._PLUGIN_DIR = tmp_path

    from astrbot.api.star import Context
    plugin = main_mod.PaperRAGPlugin(
        context=Context(),
        config={"enable_graph_rag": True, "graph_rag": {"storage_type": "neo4j"}},
    )
    plugin.enabled = True
    plugin._create_graph_rag_config = lambda: FakeGraphConfig()
    return plugin


def test_online_restore_preserves_props_and_recreates_relationships_by_backup_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        backup_dir = tmp_path / "data" / "graph_store"
        backup_dir.mkdir(parents=True)
        backup_file = backup_dir / "neo4j_backup_test.json.gz"
        backup = {
            "nodes": [
                {
                    "id": "old-node-1",
                    "labels": ["Paper", "Bad Label"],
                    "props": {"title": "Paper A", "year": 2026, "nested": {"skip": True}},
                },
                {
                    "id": "old-node-2",
                    "labels": ["Entity"],
                    "props": {"name": "Method"},
                },
            ],
            "relationships": [
                {
                    "rel_type": "MENTIONS",
                    "start_id": "old-node-1",
                    "end_id": "old-node-2",
                    "props": {"weight": 0.8, "source": "abstract"},
                },
                {
                    "rel_type": "BAD TYPE",
                    "start_id": "old-node-2",
                    "end_id": "old-node-1",
                    "props": {"ok": True},
                },
            ],
        }
        with gzip.open(backup_file, "wt", encoding="utf-8") as f:
            json.dump(backup, f)

        session = FakeSession()
        fake_driver = _install_neo4j_stub(session)
        plugin = _make_plugin(tmp_path)

        result = asyncio.run(plugin._restore_backup("neo4j_backup_test.json.gz", FakeGraphConfig()))

        assert result == {"status": "success", "nodes": 2, "relations": 2}
        assert fake_driver.closed is True

        queries = [query for query, _params in session.calls]
        assert any("CREATE (n:`Paper`)" in query for query in queries)
        assert any("CREATE (n:`Entity`)" in query for query in queries)
        assert not any("Bad Label" in query for query in queries)
        assert not any("k:" in query for query in queries)
        assert not any("elementId" in query for query in queries)
        assert any("MATCH (a {__backup_id: $start_id}), (b {__backup_id: $end_id})" in query for query in queries)
        assert any("CREATE (a)-[r:`MENTIONS`]->(b)" in query for query in queries)
        assert any("CREATE (a)-[r:`REL`]->(b)" in query for query in queries)
        assert session.calls[-1][0] == "MATCH (n) REMOVE n.__backup_id"

        node_params = [params["props"] for query, params in session.calls if query.startswith("CREATE (n")]
        assert node_params[0] == {"title": "Paper A", "year": 2026, "__backup_id": "old-node-1"}
        assert node_params[1] == {"name": "Method", "__backup_id": "old-node-2"}

        rel_params = [params for query, params in session.calls if "CREATE (a)-[r:" in query]
        assert rel_params[0] == {
            "start_id": "old-node-1",
            "end_id": "old-node-2",
            "props": {"weight": 0.8, "source": "abstract"},
        }


def test_restore_rejects_offline_directory_backup_with_clear_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        backup_dir = tmp_path / "data" / "graph_store" / "neo4j_backup_offline"
        backup_dir.mkdir(parents=True)

        plugin = _make_plugin(tmp_path)
        result = asyncio.run(plugin._restore_backup("neo4j_backup_offline", FakeGraphConfig()))

        assert result["status"] == "error"
        assert "离线目录备份不支持" in result["message"]
        assert "仅支持在线 JSON 备份" in result["message"]


def test_offline_backup_success_message_does_not_claim_graph_restore_support():
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = _make_plugin(Path(tmpdir))
        plugin._offline_backup = lambda _config: _async_value({
            "status": "success",
            "backup_file": "data/graph_store/neo4j_backup_offline",
            "size": "1.0 KB",
            "nodes": "多个",
            "relations": "多个",
        })

        class Event:
            def plain_result(self, text):
                return text

        output = asyncio.run(_collect(plugin._paper_graph_backup(Event(), mode="offline")))

        assert any("离线目录备份需手动恢复" in msg for msg in output)
        assert not any("/paper graph_restore data/graph_store/neo4j_backup_offline" in msg for msg in output)


async def _async_value(value):
    return value
