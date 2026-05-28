"""Shared test utilities — not a test file itself."""

import asyncio
import json
import sys
import types
from pathlib import Path


# ============================================================================
# Astrbot stub installation
# ============================================================================

def install_astrbot_stubs():
    """Install dummy astrbot modules so plugin imports don't fail."""

    class DummyLogger:
        def debug(self, *args, **kwargs): pass
        info = warning = error = debug

    class DummyCommandGroup:
        def __init__(self, func): self.func = func

        def __get__(self, instance, owner):
            return self if instance is None else self.func.__get__(instance, owner)

        def command(self, _name):
            def decorator(func): return func
            return decorator

    def command_group(_name):
        def decorator(func): return DummyCommandGroup(func)
        return decorator

    def permission_type(_permission):
        def decorator(func): return func
        return decorator

    class DummyPermissionType:
        ADMIN = "admin"

    def register(*_args, **_kwargs):
        def decorator(cls): return cls
        return decorator

    class Context:
        def register_llm_tool(self, *args, **kwargs): pass
        def unregister_llm_tool(self, *args, **kwargs): pass

    class Star:
        def __init__(self, context, *args, config=None, **kwargs): self.context = context
        async def terminate(self): pass

    class AstrMessageEvent: pass

    class MessageChain:
        def __init__(self): self.parts = []
        def message(self, text): self.parts.append(text)
        def file_image(self, path): self.parts.append(path)

    api_mod = types.SimpleNamespace(logger=DummyLogger())
    event_mod = types.SimpleNamespace(
        AstrMessageEvent=AstrMessageEvent,
        filter=types.SimpleNamespace(
            command_group=command_group,
            permission_type=permission_type,
            PermissionType=DummyPermissionType,
        ),
    )
    star_mod = types.SimpleNamespace(Context=Context, Star=Star, register=register)
    message_mod = types.SimpleNamespace(MessageChain=MessageChain)

    sys.modules["astrbot"] = types.ModuleType("astrbot")
    sys.modules["astrbot.api"] = api_mod  # type: ignore[assignment]
    sys.modules["astrbot.api.event"] = event_mod  # type: ignore[assignment]
    sys.modules["astrbot.api.star"] = star_mod  # type: ignore[assignment]
    sys.modules["astrbot.core.message.message_event_result"] = message_mod  # type: ignore[assignment]


# ============================================================================
# Lightweight test doubles
# ============================================================================

class DummyEvent:
    """Minimal AstrMessageEvent stub that records plain_result calls."""

    def __init__(self):
        self.messages: list[str] = []

    def plain_result(self, text: str) -> str:
        self.messages.append(text)
        return text

    async def send(self, _chain):
        return None


async def collect_async(async_gen):
    """Consume an async generator into a list."""
    results = []
    async for item in async_gen:
        results.append(item)
    return results


# ============================================================================
# Config helpers
# ============================================================================

def get_neo4j_password() -> str:
    """Read Neo4j password from plugin config, with safe fallback."""
    config_path = Path(__file__).resolve().parent.parent / "data" / "config" / "astrbot_plugin_paperrag_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            return cfg.get("graph_rag", {}).get("neo4j_password", "")
        except Exception:
            pass
    return ""
