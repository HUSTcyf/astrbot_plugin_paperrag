import asyncio
import importlib
import sys
import tempfile
import types
from pathlib import Path


def _install_astrbot_stubs():

    class DummyLogger:
        def debug(self, *args, **kwargs):
            pass

        info = warning = error = debug

    class DummyCommandGroup:
        def __init__(self, func):
            self.func = func

        def __get__(self, instance, owner):
            if instance is None:
                return self
            return self.func.__get__(instance, owner)

        def command(self, _name):
            def decorator(func):
                return func

            return decorator

    def command_group(_name):
        def decorator(func):
            return DummyCommandGroup(func)

        return decorator

    def permission_type(_permission):
        def decorator(func):
            return func

        return decorator

    class DummyPermissionType:
        ADMIN = "admin"

    def register(*_args, **_kwargs):
        def decorator(cls):
            return cls

        return decorator

    class Context:
        def __init__(self, *args, **kwargs):
            pass

        def register_llm_tool(self, *args, **kwargs):
            pass

        def unregister_llm_tool(self, *args, **kwargs):
            pass

    class Star:
        def __init__(self, context, *args, config=None, **kwargs):
            self.context = context

        async def terminate(self):
            pass

    class AstrMessageEvent:
        pass

    class MessageChain:
        def message(self, text):
            pass

        def file_image(self, path):
            pass

    sys.modules["astrbot"] = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    api_mod.logger = DummyLogger()
    sys.modules["astrbot.api"] = api_mod

    event_mod = types.ModuleType("astrbot.api.event")
    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.filter = types.SimpleNamespace(
        command_group=command_group,
        permission_type=permission_type,
        PermissionType=DummyPermissionType,
    )
    sys.modules["astrbot.api.event"] = event_mod

    star_mod = types.ModuleType("astrbot.api.star")
    star_mod.Context = Context
    star_mod.Star = Star
    star_mod.register = register
    sys.modules["astrbot.api.star"] = star_mod

    message_mod = types.ModuleType("astrbot.core.message.message_event_result")
    message_mod.MessageChain = MessageChain
    sys.modules["astrbot.core.message.message_event_result"] = message_mod


class DummyEvent:
    def __init__(self):
        self.messages = []

    def plain_result(self, text):
        self.messages.append(text)
        return text

    async def send(self, _chain):
        return None


async def _collect(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results


def test_zero_abstract_detection_and_reparse_command():
    _install_astrbot_stubs()

    plugin_parent = Path(__file__).resolve().parents[2]
    if str(plugin_parent) not in sys.path:
        sys.path.insert(0, str(plugin_parent))

    main_mod = importlib.import_module("astrbot_plugin_paperrag.main")
    paper_mod = importlib.import_module("astrbot_plugin_paperrag.commands.paper")

    class FakeAbstractManager:
        def __init__(self, order, delete_vectors_result=True):
            self.order = order
            self.delete_vectors_result = delete_vectors_result
            self.deleted = []
            self.vectors_deleted = []
            self.indexed = []

        async def delete_paper(self, paper_id):
            self.deleted.append(paper_id)
            return True

        async def delete_paper_vectors_only(self, paper_id):
            self.order.append(f"delete_vectors:{paper_id}")
            self.vectors_deleted.append(paper_id)
            return self.delete_vectors_result

        async def index_paper(self, **kwargs):
            self.order.append(f"index:{kwargs['paper_id']}")
            self.indexed.append(kwargs)
            return True

    class FakeEngine:
        def __init__(self, abstract_manager):
            self.abstract_manager = abstract_manager

        async def _ensure_abstract_manager_initialized(self):
            return self.abstract_manager

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = tmp_path / "data"
        papers_dir = tmp_path / "papers"
        data_dir.mkdir(parents=True)
        papers_dir.mkdir(parents=True)

        (papers_dir / "has_abstract.pdf").write_bytes(b"%PDF-1.4 has_abstract")
        (papers_dir / "missing_abstract.pdf").write_bytes(b"%PDF-1.4 missing_abstract")

        (data_dir / "paper_doc_stats.json").write_text(
            """{
  "has_abstract.pdf": {"file_name": "has_abstract.pdf", "chunk_count": 12, "added_time": "2026-04-24 10:00:00"},
  "missing_abstract.pdf": {"file_name": "missing_abstract.pdf", "chunk_count": 8, "added_time": "2026-04-24 10:05:00"},
  "notes.txt": {"file_name": "notes.txt", "chunk_count": 1, "added_time": "2026-04-24 10:06:00"}
}""",
            encoding="utf-8",
        )
        (data_dir / "milvus_abstracts_doc_stats.json").write_text(
            """{
  "abstracts": {
    "has_abstract": {
      "paper_id": "has_abstract",
      "file_name": "has_abstract.pdf",
      "title": "Has Abstract",
      "abstract_text": "This abstract exists and should not be listed as missing.",
      "metadata": {"extracted_abstract_chars": 58}
    }
  }
}""",
            encoding="utf-8",
        )

        paper_mod._PLUGIN_DIR = tmp_path

        from astrbot.api.star import Context
        plugin = main_mod.PaperRAGPlugin(context=Context(), config={"papers_dir": str(papers_dir)})
        plugin.enabled = True

        result = plugin._get_papers_with_zero_abstracts()
        assert result["total_papers"] == 2
        assert result["total_zero_abstract"] == 1
        assert result["papers"][0]["file_name"] == "missing_abstract.pdf"

        stats_event = DummyEvent()
        stats_output = asyncio.run(_collect(plugin.cmd_abstractstats(stats_event, -1)))
        assert any("无摘要的论文" in msg for msg in stats_output)
        assert any("missing_abstract.pdf" in msg for msg in stats_output)

        order = []
        abstract_manager = FakeAbstractManager(order)
        plugin._get_engine = lambda: FakeEngine(abstract_manager)

        async def fake_extract_success(file_path):
            order.append(f"extract:{Path(file_path).stem}")
            return "This re-extracted abstract is long enough to be inserted into the abstract vector index safely."

        plugin._extract_missing_abstract_text = fake_extract_success
        paper_stats_before = (data_dir / "paper_doc_stats.json").read_text(encoding="utf-8")
        abstract_stats_before = (data_dir / "milvus_abstracts_doc_stats.json").read_text(encoding="utf-8")

        reparse_event = DummyEvent()
        reparse_output = asyncio.run(_collect(plugin.cmd_reparse_zero_abstract(reparse_event, "confirm")))
        assert abstract_manager.deleted == []
        assert abstract_manager.vectors_deleted == ["missing_abstract"]
        assert len(abstract_manager.indexed) == 1
        assert abstract_manager.indexed[0]["paper_id"] == "missing_abstract"
        assert abstract_manager.indexed[0]["file_name"] == "missing_abstract.pdf"
        assert abstract_manager.indexed[0]["abstract_text"].startswith("This re-extracted abstract")
        assert order == [
            "extract:missing_abstract",
            "delete_vectors:missing_abstract",
            "index:missing_abstract",
        ]
        assert (data_dir / "paper_doc_stats.json").read_text(encoding="utf-8") == paper_stats_before
        assert (data_dir / "milvus_abstracts_doc_stats.json").read_text(encoding="utf-8") == abstract_stats_before
        assert any("Abstract Reparse Complete" in msg for msg in reparse_output)

        failure_order = []
        failing_manager = FakeAbstractManager(failure_order)
        plugin._get_engine = lambda: FakeEngine(failing_manager)

        async def fake_extract_failure(file_path):
            failure_order.append(f"extract:{Path(file_path).stem}")
            return None

        plugin._extract_missing_abstract_text = fake_extract_failure
        failure_event = DummyEvent()
        asyncio.run(_collect(plugin.cmd_reparse_zero_abstract(failure_event, "confirm")))

        assert failing_manager.deleted == []
        assert failing_manager.vectors_deleted == []
        assert failing_manager.indexed == []
        assert failure_order == ["extract:missing_abstract"]

        missing_vector_order = []
        missing_vector_manager = FakeAbstractManager(missing_vector_order, delete_vectors_result=False)
        plugin._get_engine = lambda: FakeEngine(missing_vector_manager)

        async def fake_extract_success_missing_vector(file_path):
            missing_vector_order.append(f"extract:{Path(file_path).stem}")
            return "This re-extracted abstract is long enough to be inserted even when no old vector exists."

        plugin._extract_missing_abstract_text = fake_extract_success_missing_vector
        missing_vector_event = DummyEvent()
        asyncio.run(_collect(plugin.cmd_reparse_zero_abstract(missing_vector_event, "confirm")))

        assert missing_vector_manager.deleted == []
        assert missing_vector_manager.vectors_deleted == ["missing_abstract"]
        assert len(missing_vector_manager.indexed) == 1
        assert missing_vector_order == [
            "extract:missing_abstract",
            "delete_vectors:missing_abstract",
            "index:missing_abstract",
        ]
