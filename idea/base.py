"""
基类
"""

from pathlib import Path

from astrbot.api import logger


class IdeaEngineBase:
    """
    研究创意生成引擎基类

    使用流程：
    1. generate_ideas - 基于本地RAG结果生成研究想法
    2. _generate_initial_draft_vlm - VLM生成周报草稿
    3. to_feishu_markdown - 格式化输出
    """

    def __init__(self, context, rag_engine=None, **kwargs):
        """
        初始化创意引擎

        Args:
            context: AstrBot上下文（用于LLM/VLM调用）
            rag_engine: RAG引擎实例
        """
        super().__init__(**kwargs)
        self.context = context
        self._rag_engine = rag_engine

    def _get_ideas_dir(self) -> Path:
        """获取想法存储根目录，不存在则创建"""
        # base.py → idea/ → astrbot_plugin_paperrag/ → plugins/ → data/
        data_dir = Path(__file__).resolve().parent.parent.parent.parent
        ideas_dir = data_dir / "plugin_data" / "astrbot_plugin_paperrag" / "ideas"
        ideas_dir.mkdir(parents=True, exist_ok=True)
        return ideas_dir
