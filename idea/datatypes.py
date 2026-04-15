"""
数据模型
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ResearchIdea:
    """研究想法"""
    title: str
    description: str
    novelty: str
    methodology: str
    potential_challenges: List[str]
    related_work: List[str]
    feasibility: float
    inspiration_sources: List[str]


@dataclass
class TopicAnalysis:
    """主题分析结果"""
    domain: str
    keywords: List[str]
    search_queries: List[str]
    local_rag_queries: List[str]
    exploration_angles: List[str]
    summary: str
