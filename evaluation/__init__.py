# -*- coding: utf-8 -*-
"""
Ragas 自动化评估模块
为 AstrBot 论文 RAG 插件提供：测试集自动生成 + RAG 评估 + 报告生成
"""

from .ragas_generator import RagasTestsetGenerator, EvalSample
from .ragas_evaluator import RagasEvaluator, EvaluationResult
from .report_generator import ReportGenerator

__all__ = [
    "RagasTestsetGenerator",
    "EvalSample",
    "RagasEvaluator",
    "EvaluationResult",
    "ReportGenerator",
]

__version__ = "1.0.0"
