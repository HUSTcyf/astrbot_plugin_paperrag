"""
研究创意生成引擎

使用方式：
  from idea import IdeaEngine
  engine = IdeaEngine(context=self.context, rag_engine=rag_engine)
"""

from .datatypes import ResearchIdea, TopicAnalysis
from .feishu_doc import IdeaEngineFeishuDoc
from .agentic_workflow import run_agentic_ideas
from .wiki import IdeaWikiEngine


class IdeaEngine(IdeaEngineFeishuDoc):
    """研究创意生成引擎

    继承链（线性化）：
      IdeaEngineBase
        → IdeaEngineUtils
          → IdeaEngineIdeas
            → IdeaEngineVM          # 独立，不在链中（直接组合）
              → IdeaEngineMarkdown
                → IdeaEngineCitations
                  → IdeaEngineGeneration
                    → IdeaEnginePaperBanana
                      → IdeaEngineFeishuDoc
                        → IdeaEngine

    使用流程：
    1. generate_ideas   - 基于本地RAG结果生成研究想法
    2. _generate_initial_draft_vlm - VLM生成周报草稿
    3. to_feishu_markdown - 格式化输出
    """
    pass


__all__ = [
    "IdeaEngine",
    "IdeaWikiEngine",
    "ResearchIdea",
    "TopicAnalysis",
    "run_agentic_ideas",
]
