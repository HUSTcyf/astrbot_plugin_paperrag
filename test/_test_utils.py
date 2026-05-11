"""Shared test utilities — not a test file itself."""

import json
from pathlib import Path


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
