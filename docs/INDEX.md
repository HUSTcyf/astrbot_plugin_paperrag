# 文档索引

## 核心文档

| 文档 | 说明 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构设计、组件说明、文件结构、配置参数 |
| [AGENTIC_ARCHITECTURE.md](AGENTIC_ARCHITECTURE.md) | Agentic RAG + Agentic Ideas LangGraph 工作流详解 |
| [CHANGELOG.md](CHANGELOG.md) | 变更记录（版本索引见 [docs/changelog/](changelog/INDEX.md)） |
| [cypher_queries.md](cypher_queries.md) | Neo4j Cypher 查询参考 |
| [FEISHU_BLOCK_STYLING.md](FEISHU_BLOCK_STYLING.md) | 飞书文档块样式更新技术方案 |

## 命令速查

### /paper — 论文检索与管理

| 命令 | 权限 | 说明 |
|------|------|------|
| `/paper search <query>` | 公开 | RAG 搜索并生成回答 |
| `/paper search <query> retrieve` | 公开 | 仅检索，不生成回答 |
| `/paper arag <query>` | 公开 | Agentic RAG 复杂查询（静态 DAG） |
| `/paper react <query>` | 公开 | ReAct Tool-Using Agent 查询 |
| `/paper list` | 公开 | 列出所有已索引文档 |
| `/paper refstats` | 公开 | 参考文献标题频次统计 |
| `/paper abstractstats` | 公开 | 摘要提取统计 |
| `/paper add <目录>` | 管理员 | 批量添加论文 |
| `/paper addf <文件>` | 管理员 | 添加单个论文 |
| `/paper delete <文件名>` | 管理员 | 删除指定论文 |
| `/paper clear confirm` | 管理员 | 清空知识库 |
| `/paper rebuild <目录> confirm` | 管理员 | 清空并重建知识库 |
| `/paper rebuildf <文件> confirm` | 管理员 | 重建单个论文 |

### /paper arxiv — arXiv 论文管理

| 命令 | 权限 | 说明 |
|------|------|------|
| `/paper arxiv_list` | 公开 | 列出带 arXiv URL 的论文 |
| `/paper arxiv_add <id>` | 管理员 | 通过 arXiv ID 搜索、下载并添加 |
| `/paper arxiv_refs <id>` | 管理员 | 下载高引参考文献 |
| `/paper arxiv_sync` | 管理员 | 同步 MCP 下载的论文 |
| `/paper arxiv_cleanup` | 管理员 | 清理旧版本 arXiv 论文 |

### /paper graph — 知识图谱管理

| 命令 | 权限 | 说明 |
|------|------|------|
| `/paper graph_build` | 公开 | 从已索引文档构建图谱 |
| `/paper graph_rebuild confirm` | 管理员 | 重建图谱 |
| `/paper graph_stats` | 公开 | 图谱统计信息 |
| `/paper graph_link <实体>` | 公开 | 查询实体关系 |
| `/paper graph_clear` | 管理员 | 清空图谱 |
| `/paper graph_backup` | 管理员 | 备份图谱到 JSON |
| `/paper graph_restore <文件>` | 管理员 | 从备份恢复图谱 |
| `/paper graph_backup_list` | 公开 | 列出可用备份 |

### /idea — 研究想法生成

| 命令 | 权限 | 说明 |
|------|------|------|
| `/idea gen <主题>` | 公开 | 生成研究想法（线性流水线） |
| `/idea explore <主题>` | 公开 | 探索研究想法（Agentic 迭代优化） |
| `/idea analyze <主题>` | 公开 | 分析研究主题 |
| `/idea search <主题>` | 公开 | 多源知识检索 |
| `/idea generate <主题>` | 公开 | 从知识上下文生成想法 |
| `/idea list` | 公开 | 列出所有已保存主题 |
| `/idea show <主题>` | 公开 | 查看主题下的想法 |
| `/idea add <主题>` | 公开 | 追加想法到已有主题 |
| `/idea del <UUID>` | 公开 | 删除指定想法 |
| `/idea delete <主题>` | 公开 | 删除整个主题 + 文件夹 |
| `/idea clear <主题>` | 公开 | 清空主题下想法（保留文件夹） |
| `/idea tofeishu <主题>` | 公开 | 创建飞书文档 |
| `/idea regen <UUID>` | 公开 | 重新生成指定想法 |

## 配置文件

插件配置通过 AstrBot WebUI 管理，完整 schema 见 `_conf_schema.json`（约 60 个配置项）。

核心配置项分类：

- **LLM Provider**：`text_provider_id`, `multimodal_provider_id`, `text_llm_temperature`, `text_llm_max_tokens`
- **Llama.cpp VLM**：`llama_vlm_model_path`, `llama_vlm_mmproj_path`, `llama_vlm_temperature`, `llama_vlm_n_ctx`, `llama_vlm_max_tokens`
- **Milvus**：`milvus_lite_path`, `address`, `db_name`, `collection_name`
- **Embedding**：`embedding_mode` (unsloth/api), `embed_dim`, `unsloth.*`
- **检索**：`top_k`, `similarity_cutoff`, `enable_sparse_retrieval`, `enable_bm25`, `enable_multi_vector_rerank`
- **Agentic**：`enable_agentic_rag`, `enable_agentic_ideas`
- **Graph RAG**：`enable_graph_rag`, `graph_rag.*`
- **分块**：`chunk_size`, `chunk_overlap`, `min_chunk_size`, `use_semantic_chunking`

## 核心文件速查

| 目录/文件 | 功能 |
|------|------|
| `provider/llama_cpp_vlm.py` | Llama.cpp VLM 单例管理 |
| `provider/llm_utils.py` | 统一 LLM 调用 (get_llm_provider / call_llm / call_llm_json) |
| `rag/hybrid_rag.py` | HybridRAGEngine：4 通道混合检索 |
| `rag/hybrid_parser.py` | PDF 解析 + 语义分块 |
| `rag/hybrid_index.py` | Milvus 索引管理 |
| `rag/multimodal_extractor.py` | 多模态提取器 (docling) |
| `rag/abstract_index.py` | 摘要索引 |
| `rag/colbert_storage.py` | ColBERT 多向量存储 |
| `agentic_rag/workflow.py` | 静态 DAG 工作流 |
| `agentic_rag/react_workflow.py` | ReAct 工作流 |
| `graphrag/graph_builder.py` | 知识图谱构建器 |
| `graphrag/graph_rag_engine.py` | Graph RAG 引擎 (Neo4j) |
| `idea/generation.py` | 线性 Idea 生成 |
| `idea/agentic_workflow.py` | Agentic Idea 工作流 |
| `embedding/unsloth_embedding.py` | BGE-M3 本地 Embedding |

## 测试

```bash
python -m pytest test/ agentic_rag/test/ idea/test/ -v
```

- `test/` — 核心 RAG 测试（17 个文件）
- `agentic_rag/test/` — Agentic RAG 节点测试（8 个文件）
- `idea/test/` — Idea 生成测试（2 个文件）

---

**最后更新**: 2026-05-07
