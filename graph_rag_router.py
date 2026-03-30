"""
Graph RAG Router - 用户意图识别与智能路由

根据用户查询类型，智能选择检索策略：
1. vector - 向量检索（默认，事实性问答、概念定义）
2. graph_local - 图谱局部检索（实体关系、多跳推理）
3. graph_global - 图谱全局检索（宏观趋势、领域总结）
4. hybrid - 混合检索（组合策略）
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from astrbot.api import logger


class RetrievalMode(Enum):
    """检索模式枚举"""
    VECTOR = "vector"
    GRAPH_LOCAL = "graph_local"
    GRAPH_GLOBAL = "graph_global"
    HYBRID = "hybrid"
    AUTO = "auto"


# 系统提示词 - 引导 LLM 进行意图识别和工具选择
ROUTER_SYSTEM_PROMPT = """你是一个学术研究助手，负责分析用户问题并选择最合适的检索工具。

## 可用工具

1. **vector_search** - 向量语义检索
   - 用于：事实性问答、概念定义、单文档内容查询、模糊语义搜索
   - 适用问法："什么是X"、"X的定义"、"X的细节"、"X是如何工作的"

2. **graph_local_search** - 图谱局部检索
   - 用于：实体关系查询、多跳推理、对比分析、引用查询
   - 适用问法："X和Y的关系"、"X引用了谁"、"X和Y有什么区别"、"X的作者是谁"
   - 注意：需要提取问题中的关键实体名称

3. **graph_global_search** - 图谱全局检索
   - 用于：领域趋势分析、宏观总结、整体数据集特征
   - 适用问法："X领域的发展趋势"、"总结X方向的研究现状"、"X的演进路线"

## 决策规则

1. **优先级**：能不用图谱就不用，默认优先向量检索
2. **实体关系**：问题涉及多个实体关系时用 graph_local
3. **宏观问题**：问趋势、总结、现状时用 graph_global
4. **安全兜底**：不确定时默认使用 vector_search

## 输出格式

请严格按以下 JSON 格式输出分析结果，不要包含其他内容：
```json
{
  "mode": "vector|graph_local|graph_global|hybrid",
  "thinking": "你的思考过程（中文，50字以内）",
  "entities": ["如果选择graph_local，列出识别到的关键实体"],
  "query_refine": "优化后的检索关键词"
}
```

## 示例

问：Transformer是如何工作的？
输出：{"mode": "vector", "thinking": "询问技术原理和定义，属于语义搜索", "entities": [], "query_refine": "Transformer 工作原理"}

问：BERT和GPT的关系是什么？
输出：{"mode": "graph_local", "thinking": "询问两个实体间的关系，需要图谱检索", "entities": ["BERT", "GPT"], "query_refine": "BERT GPT 关系 对比"}

问：深度学习在NLP领域的发展趋势？
输出：{"mode": "graph_global", "thinking": "询问宏观趋势和领域总结", "entities": [], "query_refine": "深度学习 NLP 发展趋势"}
"""


@dataclass
class RouterResult:
    """路由结果"""
    mode: RetrievalMode
    thinking: str
    entities: List[str]
    query_refine: str


class GraphRAGRouter:
    """
    Graph RAG 路由引擎

    负责任务识别和检索策略选择
    """

    # 简单规则模式匹配（快速判断，不调用 LLM）
    PATTERN_RULES = [
        # 图谱局部检索模式
        (r'(.*?)和(.*?)的关系', RetrievalMode.GRAPH_LOCAL, ['关系查询']),
        (r'(.*?)和(.*?)区别', RetrievalMode.GRAPH_LOCAL, ['对比分析']),
        (r'(.*?)引用了(.*?)', RetrievalMode.GRAPH_LOCAL, ['引用查询']),
        (r'(.*?)与(.*?)的', RetrievalMode.GRAPH_LOCAL, ['关系查询']),
        (r'比较(.*?)和(.*?)', RetrievalMode.GRAPH_LOCAL, ['对比分析']),
        (r'(.*?)是谁提出的', RetrievalMode.GRAPH_LOCAL, ['来源查询']),
        (r'谁发明了(.*?)', RetrievalMode.GRAPH_LOCAL, ['来源查询']),

        # 图谱全局检索模式
        (r'(.*?)的发展趋势', RetrievalMode.GRAPH_GLOBAL, ['趋势分析']),
        (r'(.*?)的研究现状', RetrievalMode.GRAPH_GLOBAL, ['现状分析']),
        (r'(.*?)领域.*?总结', RetrievalMode.GRAPH_GLOBAL, ['总结分析']),
        (r'总结.*?(.*?)领域', RetrievalMode.GRAPH_GLOBAL, ['总结分析']),
        (r'(.*?)的演进', RetrievalMode.GRAPH_GLOBAL, ['演进分析']),
        (r'(.*?)的未来', RetrievalMode.GRAPH_GLOBAL, ['趋势分析']),

        # 向量检索模式（定义类）
        (r'什么是(.*?)', RetrievalMode.VECTOR, ['定义查询']),
        (r'(.*?)的定义', RetrievalMode.VECTOR, ['定义查询']),
        (r'如何(实现|使用|训练|改进)(.*?)', RetrievalMode.VECTOR, ['技术细节']),
    ]

    def __init__(self, context: Any = None, llm_provider: Any = None):
        """
        初始化路由引擎

        Args:
            context: AstrBot 上下文
            llm_provider: LLM Provider（可选，用于复杂意图识别）
        """
        self.context = context
        self.llm_provider = llm_provider

    def route(self, query: str, force_mode: Optional[RetrievalMode] = None) -> RouterResult:
        """
        根据用户查询路由到合适的检索模式

        Args:
            query: 用户查询
            force_mode: 强制模式（跳过意图识别）

        Returns:
            RouterResult - 包含模式、思考过程、实体、优化后的查询
        """
        # 如果强制指定模式，直接返回
        if force_mode and force_mode != RetrievalMode.AUTO:
            return RouterResult(
                mode=force_mode,
                thinking=f"强制使用{force_mode.value}模式",
                entities=self._extract_entities_simple(query),
                query_refine=query
            )

        # 1. 先尝试规则匹配（快速路径）
        rule_result = self._match_rules(query)
        if rule_result:
            logger.debug(f"🔀 规则匹配命中: {rule_result.mode.value} - {rule_result.thinking}")
            return rule_result

        # 2. 简单关键词判断
        keyword_result = self._match_keywords(query)
        if keyword_result:
            return keyword_result

        # 3. 默认使用向量检索（安全兜底）
        return RouterResult(
            mode=RetrievalMode.VECTOR,
            thinking="默认使用向量检索",
            entities=[],
            query_refine=query
        )

    def _match_rules(self, query: str) -> Optional[RouterResult]:
        """使用规则模式匹配"""
        for pattern, mode, intent in self.PATTERN_RULES:
            match = re.search(pattern, query)
            if match:
                entities = [g for g in match.groups() if g] if match.groups() else []
                return RouterResult(
                    mode=mode,
                    thinking=f"规则匹配: {intent[0]}",
                    entities=entities,
                    query_refine=query
                )
        return None

    def _match_keywords(self, query: str) -> Optional[RouterResult]:
        """使用关键词简单判断"""
        query_lower = query.lower()

        # 全局检索关键词
        global_keywords = ['趋势', '总结', '现状', '演进', '发展', '未来', '整体', '领域', '方向']
        if any(kw in query_lower for kw in global_keywords):
            return RouterResult(
                mode=RetrievalMode.GRAPH_GLOBAL,
                thinking="关键词命中: 全局检索",
                entities=[],
                query_refine=query
            )

        # 局部检索关键词（关系）
        local_keywords = ['关系', '区别', '对比', '引用', '谁', '哪个', '比较']
        if any(kw in query_lower for kw in local_keywords):
            entities = self._extract_entities_simple(query)
            return RouterResult(
                mode=RetrievalMode.GRAPH_LOCAL,
                thinking="关键词命中: 图谱局部检索",
                entities=entities,
                query_refine=query
            )

        return None

    def _extract_entities_simple(self, query: str) -> List[str]:
        """简单实体提取（从问句中提取被询问的核心概念）"""
        entities = []

        # 提取 "X和Y" 或 "X与Y" 格式的实体
        patterns = [
            r'([^和\s]+?)和([^和\s]+)',
            r'([^与\s]+?)与([^与\s]+)',
            r'([^?\s]+?)和([^?\s]+?)的区别',
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                for g in match.groups():
                    if g and len(g) > 1:
                        entities.append(g.strip())

        return entities[:5]  # 最多返回5个实体

    async def route_with_llm(self, query: str) -> RouterResult:
        """
        使用 LLM 进行意图识别（更准确但更慢）

        Args:
            query: 用户查询

        Returns:
            RouterResult
        """
        if not self.llm_provider:
            return self.route(query)

        try:
            # 构建消息
            messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]

            # 调用 LLM
            response = await self.llm_provider.chat(messages)

            # 解析 JSON 响应
            result = self._parse_llm_response(response, query)

            if result:
                logger.debug(f"🔀 LLM路由: {result.mode.value} - {result.thinking}")
                return result
            else:
                # 解析失败，降级到规则匹配
                logger.warning("LLM 响应解析失败，降级到规则匹配")
                return self.route(query)

        except Exception as e:
            logger.error(f"LLM 路由失败: {e}，降级到规则匹配")
            return self.route(query)

    def _parse_llm_response(self, response: str, original_query: str) -> Optional[RouterResult]:
        """解析 LLM 的 JSON 响应"""
        try:
            # 提取 JSON
            json_str = response.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            data = json.loads(json_str)

            mode_str = data.get("mode", "vector").lower()
            if mode_str == "graph_local":
                mode = RetrievalMode.GRAPH_LOCAL
            elif mode_str == "graph_global":
                mode = RetrievalMode.GRAPH_GLOBAL
            elif mode_str == "hybrid":
                mode = RetrievalMode.HYBRID
            else:
                mode = RetrievalMode.VECTOR

            return RouterResult(
                mode=mode,
                thinking=data.get("thinking", ""),
                entities=data.get("entities", []),
                query_refine=data.get("query_refine", original_query)
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"JSON 解析错误: {e}")
            return None


def create_router(context: Any = None, llm_provider: Any = None) -> GraphRAGRouter:
    """
    创建路由引擎实例

    Args:
        context: AstrBot 上下文
        llm_provider: LLM Provider

    Returns:
        GraphRAGRouter 实例
    """
    return GraphRAGRouter(context=context, llm_provider=llm_provider)
