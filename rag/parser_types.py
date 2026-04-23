from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Node:
    """简化的Node类（替代llama-index的BaseNode）"""
    text: str
    metadata: Dict[str, Any]
