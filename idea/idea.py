"""
研究创意生成引擎

本文件已重构为 idea/ 模块。
请使用：from .idea import IdeaEngine

向后兼容：直接从 idea 模块导入
"""

from .idea import IdeaEngine, ResearchIdea, TopicAnalysis

__all__ = ["IdeaEngine", "ResearchIdea", "TopicAnalysis"]
