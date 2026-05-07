# PaperRAG Agentic 架构文档

## 概述

PaperRAG 在传统 RAG 基础上实现了两套基于 **LangGraph StateGraph** 的 Agentic 工作流，通过引入查询路由、并行检索、自我审查和迭代优化等 Agent 范式，显著提升了论文知识检索与 Idea 生成的质量。

| 工作流 | 传统模式 | Agentic 模式 | 配置开关 |
|--------|---------|-------------|---------|
| 论文检索 `/paper arag` | 单路检索 → 一次生成 | 查询分类 → 向量∥图谱并行检索 → 合成 → 质量检查反馈循环 | `enable_agentic_rag` |
| 论文检索 `/paper react` | —（无传统等价物） | ReAct Agent 自主推理 + 7 工具动态调用 | `enable_agentic_rag` |
| Idea 生成 `/idea explore` | analyze → search → generate（一次完成） | analyze → search → generate → critique ⇄ debate ⇄ refine 迭代循环 | `enable_agentic_ideas` |

---

## 1. Agentic RAG — 论文知识检索

### 1.1 工作流拓扑

```
User Query
    │
    ▼
┌──────────┐     query_type: fact/comparison/review/citation
│  Router  │     graph_weight: 0.0-0.8
└────┬─────┘
     │ Send() 并行派发
     ├──────────────────┐
     ▼                  ▼
┌────────────┐   ┌─────────────┐
│ Vector     │   │ Graph       │
│ Search     │   │ Search      │
│ (语义检索)  │   │ (知识图谱)   │
└─────┬──────┘   └──────┬──────┘
      │                 │
      └───────┬─────────┘
              ▼
        ┌──────────┐
        │ _barrier │  ← 汇合同步
        └────┬─────┘
             ▼
       ┌────────────┐
       │ Synthesize │  ← 融合多源上下文，LLM 生成答案
       └─────┬──────┘
             ▼
      ┌──────────────┐    质量不足 (retry < 2)
      │ Quality Check│ ─────────────────────────┐
      └──────┬───────┘                          │
             │ 通过                               ▼
             ▼                          ┌────────────┐
      ┌──────────────┐                  │ Synthesize │  (重新生成)
      │ Final Output │  ← 格式化 + 引用 └────────────┘
      └──────────────┘
```

### 1.2 各节点职责

| 节点 | 核心逻辑 | 输入 | 输出 |
|------|---------|------|------|
| **Router** | LLM 将查询分类为 `fact/comparison/review/citation`，分配图谱权重；LLM 不可用时回退关键词匹配 | `query` | `query_type`, `graph_weight` |
| **Vector Search** | 调用 `HybridRAGEngine.search(mode="retrieve")` 进行语义向量检索 | `query`, `top_k` | `retrieved_nodes` |
| **Graph Search** | 调用 `GraphRAGEngine.search(mode="hybrid")` 从 Neo4j 知识图谱获取实体、关系和文本块；`graph_weight=0` 时跳过 | `query`, `graph_weight` | `graph_entities`, `graph_relations`, `retrieved_nodes` |
| **Synthesize** | 将向量检索结果 + 图谱实体/关系统一成结构化上下文，LLM 生成带引用的答案 | `query`, `retrieved_nodes`, `graph_entities`, `graph_relations` | `draft`, `citations` |
| **Quality Check** | LLM 评估答案质量，输出质量问题和修正建议；存在质量问题时路由回 Synthesize 重试（最多 2 次修正递进：补充细节 → 重新全面生成） | `query`, `draft`, `retry_count` | `quality_issues`, `retry_count` |
| **Final Output** | 格式化答案，附加检索结果摘要和参考文献 DOI 列表 | `draft`, `citations`, `retrieved_nodes` | `final_answer` |

### 1.3 关键技术点

- **LangGraph `Send()` 并行派发**：Router 输出后，`vector_search` 和 `graph_search` 同时执行，`_barrier` 节点汇合。相比传统串行检索，总延迟近似 `max(T_vector, T_graph)` 而非 `T_vector + T_graph`。
- **动态图谱权重**：Router 根据查询类型自动调节 `graph_weight`——citation 类查询权重最高（0.8），comparison 类次之（0.6），fact 和 review 类为 0.3；graph_search 在 weight=0 时完全跳过。
- **优雅降级**：Neo4j 不可用时 graph_search 自动跳过；所有 LLM 调用失败均返回安全默认值，不阻断整个流水线。

### 1.4 ReAct Tool-Using Agent（`/paper react`）

与静态 DAG 的固定拓扑不同，ReAct Agent 让 LLM **自主推理并选择工具**，形成 `Thought → Action → Observation → Thought → ...` 的推理循环。

**工作流拓扑**：

```
User Query
    │
    ▼
┌────────────────────────────────────────────┐
│              ReAct Agent Loop              │
│                                            │
│  ┌──────────┐     tool_call     ┌────────┐ │
│  │  Agent   │ ───────────────→  │  Tool  │ │
│  │ (LLM推理) │                  │ Exec   │ │
│  └──────────┘                  └───┬────┘ │
│       ▲                            │      │
│       │       observation          │      │
│       └────────────────────────────┘      │
│                                            │
│  最多 10 轮迭代，LLM 自行决策何时结束        │
└────────────────────────────────────────────┘
    │
    ▼
┌──────────────┐
│ Final Output │  ← 汇总推理过程和最终答案
└──────────────┘
```

**7 个自主工具**：

| 工具 | 功能 | 适用场景 |
|------|------|---------|
| `vector_search(query)` | 语义向量检索论文片段 | 事实查询、概念解释 |
| `graph_search(query)` | Neo4j 知识图谱结构化查询 | 实体关系、方法对比 |
| `list_documents()` | 列出所有已索引论文 | 了解知识库覆盖范围 |
| `graph_stats()` | 知识图谱实体/关系统计 | 评估图谱可用性 |
| `get_paper_info(filename)` | 论文详细元数据（标题、作者、摘要、DOI 等） | 引用溯源、论文定位 |
| `reference_stats(top_k)` | 参考文献引用频次统计 | 高影响力文献发现 |
| `abstract_stats(top_k)` | 摘要提取统计 | 概览知识库主题分布 |

**关键设计**：
- **LLM 自主决策**：Agent 自行判断需要调用哪些工具、以什么顺序调用，无需预设检索策略
- **硬上限保护**：最多 10 轮 tool-calling 迭代，防止无限循环
- **优雅降级**：任何工具调用失败时返回错误信息给 Agent，Agent 可自行调整策略或跳过
- **与静态 DAG 互补**：`/paper arag` 适合已知结构的查询（并行检索→合成→质检），`/paper react` 适合需要动态探索的开放式问题

---

## 2. Agentic Idea Engine — 研究 Idea 生成

### 2.1 工作流拓扑

```
Research Topic
      │
      ▼
┌──────────┐
│ Analyze  │  ← LLM 分析主题（领域、关键词、检索查询、探索角度）
└────┬─────┘
      ▼
┌──────────┐
│ Search   │  ← 多源知识检索（本地 RAG + Web 搜索）
└────┬─────┘
      ▼
┌──────────┐
│ Generate │  ← LLM 基于知识上下文生成研究 Idea（可配置数量，默认 3 个）
└────┬─────┘
      ▼
      ┌─────────────────────────────────────────────────────┐
      │           Critique ⇄ Debate ⇄ Refine Loop          │
      │                                                     │
      │  ┌──────────┐    phase=refine & rounds left         │
      │  │ Critique │───────────────────┐                  │
      │  └────┬─────┘                   │                  │
      │       │ phase=done (通过)        ▼                  │
      │       │              ┌──────────┐                  │
      │       │              │  Debate  │  (多视角辩论)     │
      │       │              └────┬─────┘                  │
      │       │                   │ → critique (重新评估)   │
      │       │                   │ → refine  (轮次耗尽)    │
      │       │                   │ → save   (异常)         │
      │       │                   ▼                         │
      │       │              ┌──────────┐                  │
      │       │              │  Refine  │  补充检索+重新生成 │
      │       │              └────┬─────┘                  │
      │       │                   │ → critique (重新评估)   │
      │       └───────────────────┘                         │
      │          (最多 2 轮 debate + 3 轮 refine)            │
      └─────────────────────────────────────────────────────┘
      │ (phase=done)
      ▼
┌──────────┐
│  Save    │  ← 持久化 Idea + 上下文数据
└────┬─────┘
      ▼
┌──────────────┐
│ Final Output │  ← 格式化最终结果
└──────────────┘
```

### 2.2 各节点职责

| 节点 | 核心逻辑 | 输入 | 输出 |
|------|---------|------|------|
| **Analyze** | 调用 `IdeaEngine.analyze_topic()` —— LLM 分析主题，输出领域、关键词、搜索查询、探索角度 | `topic`, `depth` | `topic_analysis` |
| **Search** | 调用 `IdeaEngine.search_knowledge()` —— 多查询串行检索本地 RAG + Web，融合去重 | `topic_analysis` | `context_data`（含 `fused_context`、`local_results`、`web_results`） |
| **Generate** | 调用 `IdeaEngine.generate_ideas()` —— LLM 基于融合知识上下文生成研究 Idea | `topic`, `context_data`, `topic_analysis` | `ideas`（含 title、description、novelty、methodology、feasibility 等） |
| **Critique** | LLM 审查 Idea 质量 —— 输出置信度评分、审查意见、每项 Idea 的独立评分、缺失证据列表 | `ideas`, `context_data` | `critique`, `confidence`, `missing_evidence`, `idea_scores`, `phase` |
| **Debate** | 多视角辩论优化 —— LLM 从不同学术立场辩论 Idea 的优势与局限，生成辩论记录和修正建议 | `ideas`, `critique`, `context_data`, `debate_round` | `debate_history`, `phase`, `debate_round++` |
| **Refine** | 将 missing_evidence 作为新查询补充检索，合并丰富后的上下文，重新调用 generate_ideas | `ideas`, `missing_evidence`, `context_data` | `ideas`（更新后）、`context_data`（丰富后）、`iteration++`、`phase` |
| **Save** | 将最终 Idea 和上下文数据持久化到文件系统 | `ideas`, `topic`, `context_data` | `saved_paths` |
| **Final Output** | 格式化为 Markdown 输出（含置信度、审查意见、Idea 详情） | `ideas`, `critique`, `confidence` | `final_output` |

### 2.3 关键技术点

- **Critique-Debate-Refine 迭代循环**：核心创新。Critique 节点判断 Idea 质量不足（`phase=refine`）且辩论轮次未耗尽时，路由到 Debate 进行多视角辩论；辩论结束后回到 Critique 重新评估。Debate 轮次耗尽后进入 Refine（补充检索 + 重新生成），Refine 完成后再次回到 Critique。默认最多 2 轮 debate + 3 轮 refine。
- **多视角辩论优化**：当 Critique 给出负面评价时，Debate 节点让 LLM 从不同学术立场（支持方 vs 质疑方）辩论每个 Idea 的优势与局限，产生的辩论记录作为 Critique 重新评估和 Refine 补充检索的依据。
- **基于反馈的补充检索**：Refine 将 Critique 和 Debate 识别的缺失证据转化为新的检索查询，扩充知识上下文后重新生成——传统 RAG 的知识检索是一次性、不可修正的。
- **量化质量评估**：Critique 输出置信度分数（0.0-1.0）和每项 Idea 的独立评分，提供可量化的质量信号。
- **硬上限保护**：`_max_debate_rounds`（默认 2）和 `_max_iterations`（默认 3）限制防止无限循环；Critique、Debate 或 Refine 失败时优雅退出（`phase = "done"`）。

---

## 3. 与传统 RAG 的核心区别

### 3.1 传统 RAG 的局限性

传统 RAG 管线是**单次通过**的：

```
query → [学术意图检查] → [路由: retrieve/rag] → [混合检索（含 RRF 图谱融合）] → [generate answer]
```

```
topic → [analyze] → [search] → [generate ideas] → 输出
```

虽然传统路径也具备意图检查、模式路由和 RRF 知识图谱融合，但存在以下局限：

- **检索策略固定**：路由仅在 `retrieve`/`rag` 两种模式间选择，不会根据查询内容（fact vs citation）动态调节图谱权重或调整检索策略
- **图谱结果不透明**：知识图谱以 RRF 分数融合进统一结果集，用户无法看到独立的实体、关系等结构化知识
- **无自我纠错**：Idea 生成后无质量评估，用户无法知道置信度或改进方向；知识检索也无后续修正机会
- **无反馈循环**：检索结果差 → 答案差，无法补救；生成结果差 → 无法基于批评再优化

### 3.2 Agentic RAG 的核心优势

| 维度 | 传统 RAG | Agentic RAG |
|------|---------|------------|
| **查询理解** | 路由仅在 retrieve/rag 模式间选择 | Router 细粒度分类（fact/comparison/review/citation），动态调节图谱权重 |
| **图谱检索** | RRF 分数融合进统一结果集，实体/关系不可见 | 独立并行分支，输出结构化实体和关系三元组，Synthesize 阶段显式利用 |
| **知识来源可见性** | 返回文本块列表，图谱贡献不可区分 | 向量来源和图谱来源显式标注（`source: "vector"/"graph"`），可追溯 |
| **延迟** | T_hybrid（含 RRF 融合） | max(T_vector, T_graph) —— 并行分支不叠加延迟 |
| **答案合成** | 单一 prompt 基于融合文本生成 | 结构化上下文（检索结果 + 图谱实体 + 图谱关系分节组织），LLM 显式理解图谱知识 |

### 3.3 Agentic Idea Engine 的核心优势

| 维度 | 传统 Idea 生成 | Agentic Idea Engine |
|------|--------------|-------------------|
| **质量审查** | 无，一次性生成 | LLM 自我审查，输出置信度和逐项评分 |
| **迭代优化** | 0 轮 | 最多 2 轮 debate + 3 轮 refine 迭代 |
| **知识补充** | 检索一次，不可修改 | Refine 阶段根据批评意见补充检索，丰富上下文后重新生成 |
| **可观测性** | 黑盒输出 | 每步执行可追踪（`steps` 记录全流程），审查意见可见 |
| **故障处理** | 异常抛给调用方 | 每节点独立 try/except，优雅降级 |

### 3.4 量化对比（Demo Mock 数据示例）

以下为 `tools/generate_demo_report.py --dry-run` 生成的示意性对比数据，展示 Agentic 模式在各项指标上的预期差异：

| 指标 | 传统模式 | Agentic 模式 | 差异 |
|------|---------|-------------|------|
| RAG 平均来源数 | ~3 条 | ~5 条（含知识图谱实体/关系） | +67%（多源并行） |
| 图谱可见性 | 仅 RRF 融合分数，实体/关系不可见 | 100% 查询获独立实体/关系输出 | 结构可视化 |
| Idea 迭代轮次 | 0（一次生成） | debate + refine / 主题 | 新增能力 |
| Idea 置信度 | 无评估 | 72%-78%（可量化） | 新增能力 |
| 审查意见 | 无 | 每主题具体改进建议 | 新增能力 |

实际数字取决于知识库覆盖度、Neo4j 图谱数据量和 VLM 推理质量。

---

## 4. 技术实现亮点

### 4.1 LangGraph StateGraph

两套工作流均基于 LangGraph 的 `StateGraph` 框架，使用 `TypedDict` 定义状态 Schema，Pydantic 模型做节点输入输出校验：

```python
class AgenticRAGState(TypedDict):
    query: str
    query_type: NotRequired[str]           # router 输出
    graph_weight: NotRequired[float]       # router 输出
    retrieved_nodes: Annotated[list[dict], _keep_last_n(10)]  # 多源合并，上限 10
    graph_entities: NotRequired[list[dict]]
    graph_relations: NotRequired[list[dict]]
    draft: NotRequired[str]
    citations: NotRequired[list[str]]
    final_answer: NotRequired[str]
    steps: Annotated[list[str], add_messages]  # 全流程执行追踪
```

### 4.2 统一 Provider 解析链

v2.0.0 重构后，所有 LLM 调用统一通过 `provider/llm_utils.py` 中的 `call_llm()` / `call_llm_json()` 发起，内部调用 `get_llm_provider()` 按 **4 步优先级链** 解析 Provider：

1. **插件配置 `text_provider_id`** → `provider_manager.inst_map[id]` — 用户显式选择优先
2. **本地 VLM 单例**（`get_llama_cpp_vlm_provider()`，Qwen3.5 9B/4B GGUF）— 完全离线场景
3. **当前会话 Provider**（`context.get_using_provider()`）— AstrBot 会话级别配置
4. **inst_map 中首个具备 `text_chat` 的 Provider** — 兜底：遍历 inst_map 查找 `callable(getattr(p, 'text_chat', None))`

所有节点（Router、Synthesize、Critique、Debate 等）共享同一套解析链，无差异。返回的 Provider 会经过 `callable(text_chat)` 安全校验，无法使用时抛出 `RuntimeError("无可用 LLM provider")`，由各节点独立 try/except 捕获后返回安全默认值。

### 4.3 优雅降级策略

- 所有节点独立 try/except，失败返回安全默认值
- Neo4j 不可用 → graph_search 跳过，向量检索仍正常工作
- LLM 不可用 → 回退关键词分类、跳过 Critique / Debate
- `max_iterations` 硬上限防止无限循环

### 4.4 状态管理

- `retrieved_nodes` 使用 `_keep_last_n(10)` reducer：多源结果合并后自动截断，防止状态膨胀
- `steps` 使用 `add_messages` reducer：自动累积全流程步骤，提供端到端可观测性

---

## 5. 配置与集成

### 5.1 配置开关

在 `_conf_schema.json` 中定义，`data/config/astrbot_plugin_paperrag_config.json` 中实际配置：

```json
{
  "enable_agentic_rag": true,     // 启用 Agentic RAG（/paper arag, /paper react）
  "enable_agentic_ideas": true,   // 启用 Agentic Idea（/idea explore）
  "enable_graph_rag": true        // 启用知识图谱（两个 Agentic 工作流的前提）
}
```

### 5.2 命令行集成

```python
# commands/paper.py — /paper search 分发
if self.config.get("enable_agentic_rag", False):
    async for result in self._agentic_rag(event, query=query, top_k=top_k):
        yield result
    return
# ...否则继续执行传统管线（_paper_search）

# commands/idea.py — /idea explore 分发
if self.config.get("enable_agentic_ideas", False):
    yield event.plain_result(f"🧠 Agentic Idea 生成中...\n主题: {topic}")
    result = await run_agentic_ideas(topic=topic, context=self.context, ...)
    # 流式输出各步骤 + 最终结果
    return
# ...否则继续执行传统管线（_idea_explore）
```

分发是完全的二选一开关，无混合模式。

---

## 6. 文件结构

```
agentic_rag/                      # Agentic RAG 包
├── __init__.py                   # 公共 API：run_agentic_rag(), run_agentic_rag_stream()
├── workflow.py                   # 静态 DAG StateGraph 定义 + 编译
├── state.py                      # AgenticRAGState TypedDict
├── engine_utils.py               # 引擎工厂（get_engine, get_graph_engine）
├── react_workflow.py             # ReAct Agent StateGraph 定义 + 编译
├── react_agent.py                # ReAct Agent 推理节点
├── react_tools.py                # 7 个自主工具（vector_search, graph_search, list_documents...）
├── react_state.py                # AgenticReActState TypedDict
├── test/                         # Agentic RAG 测试
└── nodes/
    ├── router.py                 # 查询分类 + 图谱权重分配
    ├── vector_search.py          # 向量语义检索
    ├── graph_search.py           # Neo4j 知识图谱检索
    ├── synthesize.py             # 多源上下文融合 + LLM 答案生成
    ├── quality_check.py          # 答案质量评估 + 反馈路由
    ├── final_output.py           # 答案格式化 + 参考文献

idea/                             # Idea 引擎包（含 Agentic + 传统）
├── agentic_workflow.py           # LangGraph StateGraph 定义 + run_agentic_ideas()
├── agentic_state.py              # AgenticIdeaState TypedDict
├── datatypes.py                  # ResearchIdea, TopicAnalysis 数据结构
├── generation.py                 # 传统 Idea 引擎（analyze_topic, search_knowledge, generate_ideas）
└── nodes/
    ├── analyze.py                # 主题分析
    ├── search.py                 # 多源知识检索
    ├── generate.py               # Idea 生成
    ├── critique.py               # LLM 质量审查
    ├── debate.py                 # 多视角辩论优化
    ├── refine.py                 # 补充检索 + 重新生成
    ├── save.py                   # 文件持久化
    └── final_output.py           # 格式化输出
```

---

## 7. 总结

PaperRAG 的 Agentic 方案在传统 RAG 基础上引入了一套完整的 **Agent 思考循环**：

- **路由决策**（Router）替代固定检索策略，让系统根据查询内容自适应调整行为
- **并行执行**（Send 派发）打破串行瓶颈，在不增加延迟的前提下引入知识图谱作为第二知识源
- **自我审查**（Critique）为生成结果提供了量化的质量信号和具体的改进方向
- **辩论优化**（Debate）从多元学术视角审视 Idea，打破单一模型的思维定式
- **迭代优化**（Critique-Debate-Refine Loop）实现了基于反馈的知识补充和重新生成，让系统具备纠错能力

这套方案的架构选择（LangGraph + TypedDict + Pydantic + 优雅降级）在不增加外部依赖的前提下，提供了结构化、可观测、可扩展的 Agent 工作流框架。
