import asyncio
import ast
import importlib
import sys
import tempfile
import types
from pathlib import Path


def _install_astrbot_stubs():
    if "astrbot.api" in sys.modules:
        return

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
        def register_llm_tool(self, *args, **kwargs):
            pass

        def unregister_llm_tool(self, *args, **kwargs):
            pass

    class Star:
        def __init__(self, context):
            self.context = context

        async def terminate(self):
            pass

    class AstrMessageEvent:
        pass

    class MessageChain:
        def __init__(self):
            self.parts = []

        def message(self, text):
            self.parts.append(text)

        def file_image(self, path):
            self.parts.append(path)

    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    event_mod = types.ModuleType("astrbot.api.event")
    star_mod = types.ModuleType("astrbot.api.star")
    message_mod = types.ModuleType("astrbot.core.message.message_event_result")

    api_mod.logger = DummyLogger()
    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.filter = types.SimpleNamespace(
        command_group=command_group,
        permission_type=permission_type,
        PermissionType=DummyPermissionType,
    )
    star_mod.Context = Context
    star_mod.Star = Star
    star_mod.register = register
    message_mod.MessageChain = MessageChain

    sys.modules["astrbot"] = astrbot_mod
    sys.modules["astrbot.api"] = api_mod
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.star"] = star_mod
    sys.modules["astrbot.core.message.message_event_result"] = message_mod


class DummyEvent:
    def __init__(self):
        self.messages = []

    def plain_result(self, text):
        self.messages.append(text)
        return text

    async def send(self, _chain):
        return None


class FakeEngine:
    def __init__(self):
        self.add_calls = []
        self.clear_calls = 0

    async def add_paper(self, file_path):
        self.add_calls.append(file_path)
        return {"status": "success", "chunks_added": 3}

    async def clear(self):
        self.clear_calls += 1
        return {"status": "success", "message": "cleared"}


async def _collect(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results


def test_split_main_keeps_command_shell_and_paper_commands_callable():
    _install_astrbot_stubs()

    plugin_parent = Path(__file__).resolve().parents[2]
    if str(plugin_parent) not in sys.path:
        sys.path.insert(0, str(plugin_parent))

    main_mod = importlib.import_module("astrbot_plugin_paperrag.main")
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    main_ast = ast.parse(main_path.read_text(encoding="utf-8"))
    plugin_class = next(
        node for node in main_ast.body
        if isinstance(node, ast.ClassDef) and node.name == "PaperRAGPlugin"
    )
    base_names = [base.id for base in plugin_class.bases if isinstance(base, ast.Name)]
    assert base_names[:4] == [
        "PaperCommandsMixin",
        "ArxivCommandsMixin",
        "GraphCommandsMixin",
        "IdeaCommandsMixin",
    ]

    command_groups = []
    subcommands = []
    for node in plugin_class.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            decorator_text = ast.unparse(decorator)
            if "command_group" in decorator_text:
                command_groups.append(decorator_text)
            if ".command(" in decorator_text:
                subcommands.append(decorator_text)

    assert command_groups == [
        "filter.command_group('paper')",
        "filter.command_group('idea')",
    ]
    for command in [
        "paper_commands.command('search')",
        "paper_commands.command('abstractstats')",
        "paper_commands.command('graph_restore')",
        "paper_commands.command('graph_link')",
        "idea_commands.command('list')",
    ]:
        assert command in subcommands

    for command_file in (Path(__file__).resolve().parents[1] / "commands").glob("*.py"):
        source = command_file.read_text(encoding="utf-8")
        assert "@filter.command_group" not in source
        assert ".command(" not in source

    from astrbot.api.star import Context as Ctx
    plugin = main_mod.PaperRAGPlugin(context=Ctx(), config={})
    for method_name in [
        "cmd_search",
        "cmd_graph_restore",
        "cmd_graph_link",
        "cmd_idea_list",
    ]:
        assert hasattr(plugin, method_name)

    calls = []

    async def fake_paper_search(event, query="", top_k=5):
        calls.append(("search", query, top_k))
        yield "search-ok"

    async def fake_graph_restore(event, backup_file=""):
        calls.append(("graph_restore", backup_file))
        yield "restore-ok"

    async def fake_graph_link(event, action="status"):
        calls.append(("graph_link", action))
        yield "link-ok"

    async def fake_idea_list(event):
        calls.append(("idea_list",))
        yield "idea-list-ok"

    plugin._paper_search = fake_paper_search
    plugin._paper_graph_restore = fake_graph_restore
    plugin._paper_graph_link = fake_graph_link
    plugin._idea_list = fake_idea_list

    event = DummyEvent()
    assert asyncio.run(_collect(plugin.cmd_search(event, "query", 7))) == ["search-ok"]
    assert asyncio.run(_collect(plugin.cmd_graph_restore(event, "demo.json.gz"))) == ["restore-ok"]
    assert asyncio.run(_collect(plugin.cmd_graph_link(event, "status"))) == ["link-ok"]
    assert asyncio.run(_collect(plugin.cmd_idea_list(event))) == ["idea-list-ok"]
    assert calls == [
        ("search", "query", 7),
        ("graph_restore", "demo.json.gz"),
        ("graph_link", "status"),
        ("idea_list",),
    ]
