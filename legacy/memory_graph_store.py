"""
Legacy MemoryGraphStore 实现 - 已弃用

此模块保留了基于 LlamaIndex SimplePropertyGraphStore 的内存存储实现，
支持磁盘持久化（knowledge_graph.json.gz）。

当前插件已切换为仅支持 Neo4j 存储，此文件仅供历史参考。

迁移日期: 2026-04-30
弃用原因: 生产环境统一使用 Neo4j，memory 模式的持久化机制不再维护
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from astrbot.api import logger

# 注意: 此文件依赖 SimplePropertyGraphStoreAdapter，运行时需要从主模块导入
# from ..graphrag.graph_rag_engine import SimplePropertyGraphStoreAdapter


class SimplePropertyGraphStoreAdapter:
    """
    简化适配器存根 - 实际实现请参考 graphrag/graph_rag_engine.py

    此处仅用于 PersistentPropertyGraphStoreAdapter 的继承。
    如需恢复此模块，请将 SimplePropertyGraphStoreAdapter 的完整实现
    复制到此处，或将导入路径改为从 graph_rag_engine 引入。
    """
    pass


class PersistentPropertyGraphStoreAdapter(SimplePropertyGraphStoreAdapter):
    """
    带持久化的适配器（已弃用）

    封装 LlamaIndex SimplePropertyGraphStore，提供磁盘持久化。
    数据保存在 data/graph_store/knowledge_graph.json.gz。

    此类已从主代码中移除，仅保留在 legacy 目录供参考。
    如需恢复功能：
    1. 将 SimplePropertyGraphStoreAdapter 的完整实现从 graph_rag_engine.py 复制到此文件
    2. 取消下方所有方法的注释
    3. 在 graph_rag_engine.py 中恢复 MemoryGraphStore 别名
    """

    STORAGE_FILENAME = "knowledge_graph.json.gz"
    METADATA_FILENAME = "graph_metadata.json"

    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        save_interval: int = 100
    ):
        raise NotImplementedError(
            "PersistentPropertyGraphStoreAdapter 已弃用。"
            "请使用 Neo4j 存储模式。"
            "如需恢复此功能，请从 git 历史中恢复完整实现。"
        )


# 向后兼容别名
MemoryGraphStore = PersistentPropertyGraphStoreAdapter
