# Paper RAG 插件架构

## 核心流程

```
PDF/DOCX/TXT → HybridPDFParser → Chunks → HybridIndexManager → Milvus
                                   ↓
                          多模态提取 (docling)
                      (文字/公式/表格/图片)

查询 /paper search → HybridRAGEngine.search()
                     ├─ 稠密向量检索 (BGE-M3 via Unsloth/API)
                     ├─ 稀疏权重检索 (BGE-M3 ABSPEC)
                     ├─ BM25 精确匹配
                     └─ 知识图谱通道 (Neo4j, 可选)
                     ↓
                RRF 融合 → ColBERT 重排序 → CRAG 质量评估 → call_llm() 生成回答

Agentic 查询 /paper arag|react → LangGraph workflow
                     ├─ Router (查询分类: fact/comparison/citation/review)
                     ├─ vector_search ∥ graph_search (并行)
                     ├─ synthesize (多源聚合生成)
                     ├─ quality_check → 反馈循环
                     └─ final_output
```

## 核心组件

### 1. provider/ — 统一模型服务层

- **`provider/llama_cpp_vlm.py`** — Llama.cpp VLM 单例（Qwen3.5 9B/4B GGUF），自动 9B→4B 降级
- **`provider/llm_utils.py`** — 统一 LLM 调用：
  - `get_llm_provider()` — 4 步优先级解析（config → VLM → session → inst_map）
  - `call_llm()` — LLM 调用 + 文本提取
  - `call_llm_json()` — LLM 调用 + JSON 解析
  - `extract_text_from_response()` — 兼容 VLM、cloud provider（result_chain）、dict

### 2. rag/ — 核心 RAG 引擎

- **`hybrid_rag.py`** — HybridRAGEngine：4 通道混合检索 + RRF 融合
- **`hybrid_parser.py`** — HybridPDFParser：PDF 多模态解析 + 语义分块
- **`hybrid_index.py`** — HybridIndexManager：Milvus Lite / 远程服务器
- **`multimodal_extractor.py`** — 多模态提取器（docling）：图片/表格/公式
- **`abstract_index.py`** — 摘要索引：提取标题/摘要，支持两阶段检索
- **`colbert_storage.py`** — ColBERT 多向量存储（FAISS-based）
- **`text_splitter.py`** — 语义感知分块（段落 → 句子 → 子句）
- **`llm_compaction.py`** — LLM 元数据提取（标题/摘要/作者）
- **`llm_preprocessor.py`** — LLM 文本预处理
- **`paper_link_resolver.py`** — 论文链接/DOI/arXiv ID 解析
- **`reference_processor.py`** — 参考文献解析

### 3. agentic_rag/ — Agentic RAG 工作流

两种 LangGraph 架构：

- **静态 DAG** (`workflow.py`)：router → [vector_search ∥ graph_search] → synthesize → quality_check → final_output
- **ReAct Tool-Using Agent** (`react_workflow.py`)：agent ↔ tool_executor 循环，7 个自主工具

### 4. graphrag/ — 知识图谱 RAG

- **`graph_builder.py`** — 多模态知识图谱构建器（JSON Schema + closed-set）
- **`graph_rag_engine.py`** — Graph RAG 引擎（Neo4j 后端）
- **`graph_rag_router.py`** — 图谱查询路由
- **`triplet_schema.json`** — 9 类实体 + 14 类关系 schema
- **`multimodal_schema.json`** — 多模态三元组 schema
- **`visualize_neo4j_html.py`** — Neo4j HTML 可视化

### 5. idea/ — 研究想法生成

两种模式：

- **线性流水线** (`generation.py`)：analyze → search → generate → save
- **Agentic 工作流** (`agentic_workflow.py`)：analyze → search → generate → critique → debate → refine → save
- `nodes/` — 8 个 LangGraph 节点（analyze, search, generate, critique, debate, refine, save, final_output）
- `websearch.py` — 多源网络搜索
- `feishu_doc.py` — 飞书文档导出
- `paperbanana.py` — PaperBanana 方法图生成

### 6. embedding/ — Embedding 模块

- **`unsloth_embedding.py`** — 本地 BGE-M3（MPS/CUDA/CPU）：稠密向量 + 稀疏权重 + ColBERT 多向量
- **`embedding_providers.py`** — Embedding 工厂
- **`flag_embedding.py`** — FlagEmbedding 封装

### 7. commands/ — 命令层

- `base.py` — PluginCoreBase + 图片工具 + 学术意图检测
- `paper.py` — `/paper search|list|add|addf|clear|delete|rebuild|rebuildf|refstats|abstractstats`
- `arxiv.py` — `/paper arxiv_list|add|refs|sync|cleanup`
- `graph.py` — `/paper graph_build|rebuild|stats|clear|backup|restore|backup_list|link`
- `idea.py` — `/idea gen|list|show|add|del|delete|clear|explore|analyze|search|generate|tofeishu|regen`
- `retrieval_helpers.py` — 检索辅助 + LLM 工具注册

## 检索架构

### 三通道混合检索 + RRF 融合（单阶段路径）

| 通道 | 配置项 | 权重控制 |
|------|------|---------|
| 稠密向量 | 始终启用 | RRF k=60 |
| 稀疏权重 | `enable_sparse_retrieval` | RRF k=60 |
| BM25 精确 | `enable_bm25` | RRF k=60 |

### 两阶段检索（`enable_two_stage_retrieval`）

```
Stage 1    摘要向量检索 → top-20 candidates
Stage 1.5  ColBERT rerank → top-6 papers（摘要通道配额）
Stage 1.6  知识图谱独立召回 → top-2 papers（图谱通道配额，不参与 rerank）
           合并去重 → 最多 8 篇论文进入 Stage 2
Stage 2    在选中论文内检索 chunks → top-k 最终结果
```

图谱通道使用实体关系作为结构性信号（如 `DUSt3R → enables → InstantSplat`），与摘要的文本相似度信号互补。
图谱召回的论文不经过 ColBERT rerank，避免文本打分过滤掉结构上相关但摘要匹配度低的论文。

### 可选增强

- **ColBERT 多向量重排序** (`enable_multi_vector_rerank`)：token-level late-interaction
- **CRAG 质量评估** (`enable_crag_quality_eval`)：自动质量评估 + 可选纠偏重搜

## 知识图谱 Schema

### 实体类型（9 类）

Method, Model, Task, Dataset, Metric, Component, Limitation, Application, Baseline

### 关系类型（14 类）

ADDRESSES, PROPOSES, USES_COMPONENT, EVALUATED_ON, ACHIEVES, COMPARES_WITH, LIMITED_BY, APPLIES_TO, EXTENDS, TRAINS_ON, IMPLEMENTS, OUTPERFORMS, REQUIRES, ABLATES_ON

## 配置参数（核心）

```python
RAGConfig(
    # Embedding
    embedding_mode="unsloth",     # "api" | "unsloth"
    embed_dim=1024,              # BGE-M3 = 1024, Gemini = 768

    # LLM Provider
    text_provider_id="",          # 文本 LLM（cloud provider ID）
    multimodal_provider_id="",    # 多模态 LLM（cloud provider ID）
    text_llm_temperature=0.7,     # 文本 LLM 温度
    text_llm_max_tokens=2048,     # 文本 LLM max tokens

    # Llama.cpp VLM（本地）
    llama_vlm_model_path="./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf",
    llama_vlm_mmproj_path="./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf",
    llama_vlm_temperature=0.7,
    llama_vlm_n_ctx=16384,
    llama_vlm_n_gpu_layers=99,
    llama_vlm_max_tokens=2560,

    # Milvus
    milvus_lite_path="",
    address="",                   # 远程地址
    db_name="default",
    collection_name="paper_embeddings",

    # 分块
    chunk_size=512,
    chunk_overlap=0,
    min_chunk_size=128,
    use_semantic_chunking=True,

    # 检索
    top_k=5,
    similarity_cutoff=0.5,

    # 混合检索
    enable_sparse_retrieval=True,
    enable_bm25=True,
    enable_multi_vector_rerank=False,
    hybrid_alpha=0.5,

    # Agentic RAG
    enable_agentic_rag=False,
    enable_agentic_ideas=False,

    # Graph RAG
    enable_graph_rag=False,

    # 两阶段检索
    enable_two_stage_retrieval=False,

    # CRAG
    enable_crag_quality_eval=True,
    crag_enable_correction=False,
    crag_min_score=0.5,
)
```

## 文件结构

```
astrbot_plugin_paperrag/
├── main.py                      # 插件入口
├── metadata.yaml                 # 插件元数据
├── _conf_schema.json             # 配置 schema（WebUI）
│
├── provider/                     # 统一模型服务层
│   ├── __init__.py
│   ├── llama_cpp_vlm.py          # Llama.cpp VLM 单例
│   └── llm_utils.py              # 统一 LLM 调用工具
│
├── rag/                          # 核心 RAG 引擎
│   ├── rag_engine.py             # 配置 & 工厂函数
│   ├── hybrid_parser.py          # PDF 解析 + 语义分块
│   ├── hybrid_index.py           # Milvus 索引管理
│   ├── hybrid_rag.py             # HybridRAGEngine
│   ├── multimodal_extractor.py   # 多模态提取器 (docling)
│   ├── abstract_index.py         # 摘要索引
│   ├── colbert_storage.py        # ColBERT 多向量存储 (FAISS)
│   ├── text_splitter.py          # 语义感知分块
│   ├── llm_compaction.py         # LLM 元数据提取
│   ├── llm_preprocessor.py       # LLM 文本预处理
│   ├── paper_link_resolver.py    # 论文链接解析
│   └── reference_processor.py    # 参考文献解析
│
├── agentic_rag/                  # Agentic RAG
│   ├── __init__.py               # 入口 + State
│   ├── workflow.py               # 静态 DAG 工作流
│   ├── react_workflow.py         # ReAct 工作流
│   ├── react_agent.py            # ReAct Agent 节点
│   ├── react_tools.py            # 7 个自主工具
│   ├── react_state.py            # ReAct State
│   ├── state.py                  # DAG State
│   ├── engine_utils.py           # 引擎工厂
│   └── nodes/
│       ├── router.py             # 查询分类
│       ├── vector_search.py      # 向量检索
│       ├── graph_search.py       # 图谱检索
│       ├── synthesize.py         # 多源聚合生成
│       ├── quality_check.py      # 质量检查
│       └── final_output.py       # 最终输出
│
├── graphrag/                     # 知识图谱 RAG
│   ├── graph_builder.py          # 图谱构建器
│   ├── graph_rag_engine.py       # Graph RAG 引擎 (Neo4j)
│   ├── graph_rag_router.py       # 图谱查询路由
│   ├── triplet_schema.json       # 9 实体 + 14 关系 schema
│   ├── multimodal_schema.json    # 多模态三元组 schema
│   └── visualize_neo4j_html.py   # Neo4j HTML 可视化
│
├── idea/                         # 研究想法生成
│   ├── generation.py             # 线性流水线
│   ├── agentic_workflow.py       # Agentic 工作流
│   ├── ideas.py                  # 想法管理
│   ├── websearch.py              # 多源网络搜索
│   ├── feishu_doc.py             # 飞书文档导出
│   ├── paperbanana.py            # PaperBanana 方法图
│   ├── vm.py                     # Viewpoint Model
│   ├── utils.py                  # 共享工具
│   ├── citations.py              # 引用提取
│   ├── markdown.py               # Markdown 格式化
│   ├── idea_schema.gbnf          # GBNF grammar (idea)
│   └── nodes/
│       ├── analyze.py            # 主题分析
│       ├── search.py             # 知识检索
│       ├── generate.py           # Idea 生成
│       ├── critique.py           # 质量审查
│       ├── debate.py             # 辩论优化
│       ├── refine.py             # 迭代改进
│       ├── save.py               # 持久化保存
│       └── final_output.py       # 最终输出
│
├── embedding/                    # Embedding 模块
│   ├── unsloth_embedding.py      # BGE-M3 (稠密+稀疏+ColBERT)
│   ├── embedding_providers.py    # Embedding 工厂
│   └── flag_embedding.py         # FlagEmbedding 封装
│
├── commands/                     # 命令层
│   ├── base.py                   # PluginCoreBase + 图片工具
│   ├── paper.py                  # /paper 命令
│   ├── arxiv.py                  # /paper arxiv_* 命令
│   ├── graph.py                  # /paper graph_* 命令
│   ├── idea.py                   # /idea 命令
│   └── retrieval_helpers.py      # 检索辅助 + LLM 工具注册
│
├── tools/                        # 独立工具脚本
│   ├── generate_demo_report.py   # 对比报告生成
│   ├── build_graph_from_milvus.py
│   ├── build_abstract_index.py
│   ├── clean_milvus_chunks.py
│   ├── download_models.py
│   ├── extract_figure_captions.py
│   ├── repair_table_csvs.py
│   ├── clean_testset_contexts.py
│   └── export_bloom.py
│
├── evaluation/                   # 评估
│   ├── run_evaluation_ragas.py   # RAGAS 评估
│   ├── run_evaluation_qasper.py  # QASPER 评估
│   ├── evaluate_retrieval.py     # 检索对比
│   └── ragas_evaluator.py        # RAGAS 指标计算
│
├── test/                         # 测试
│   ├── test_closed_set_schema.py
│   ├── test_graph_backup_restore.py
│   ├── test_graph_rag_integration.py
│   ├── test_colbert_storage.py
│   ├── test_chunk_tokenization.py
│   └── ...
│
├── agentic_rag/test/             # Agentic RAG 测试
├── idea/test/                    # Idea 测试
│
├── datasets/                     # 测试数据集 (QASPER)
├── legacy/                       # 遗留代码（已废弃，仅供对比）
└── docs/                         # 文档
    ├── ARCHITECTURE.md           # 本文档
    ├── CHANGELOG.md              # 变更记录
    ├── AGENTIC_ARCHITECTURE.md   # Agentic 架构详解
    ├── cypher_queries.md         # Neo4j Cypher 参考
    ├── INDEX.md                  # 文档索引
    └── changelog/                # 按版本拆分的变更记录
```
