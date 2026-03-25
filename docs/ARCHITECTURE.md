# Paper RAG 插件架构

## 核心组件

```
PDF → HybridPDFParser → Nodes → HybridIndexManager → HybridRAGEngine → Response
          ↓                                      ↓
    多模态提取                              Milvus存储
  (文字/公式/图片)                         (向量+元数据)
```

### 1. HybridPDFParser (`hybrid_parser.py`)
- PDF多模态解析：文字、公式、图片
- **语义分块**：按段落/句子分割，保持语义完整
- **Overlap支持**：块之间保持语义连贯

### 2. HybridIndexManager (`hybrid_index.py`)
- Milvus向量存储管理（独立连接别名，避免冲突）
- 支持 Lite 模式 / 远程服务器
- 异步操作支持

### 3. HybridRAGEngine (`hybrid_rag.py`)
- 完整RAG流程：解析 → 分块 → 索引 → 检索 → 生成
- 支持多模态查询（图片输入）
- GLM LLM 生成

## 关键特性

### 语义分块策略
1. 优先按 `\n\n` 分割段落
2. 大段落按句子分割（中英文句号分隔）
3. 超长句子按子句分割（逗号等）
4. Overlap 保持语义连贯

### 分块优先级
```
段落(\n\n) > 句子(。！？.!? ) > 子句(，,；;)
```

### 多模态处理
- **文字**: 直接向量化
- **公式**: LaTeX格式保留，向量化
- **图片**: 跳过（暂不向量化）
- **表格**: 跳过（暂不向量化）

## 配置参数

```python
RAGConfig(
    # 模型配置
    glm_api_key="",                    # GLM API密钥
    glm_model="glm-4.7-flash",        # 文本模型
    glm_multimodal_model="glm-4.6v-flash",  # 多模态模型

    # Embedding配置
    embedding_mode="api",               # "api" 或 "ollama"
    embed_dim=768,                     # 向量维度

    # Milvus配置
    milvus_lite_path="./data/milvus_papers.db",
    collection_name="paper_embeddings",

    # 分块配置
    chunk_size=512,                    # 目标块大小
    chunk_overlap=0,                   # 重叠大小
    min_chunk_size=100,                # 最小块大小

    # 检索配置
    top_k=5,

    # 多模态
    enable_multimodal=True,
)
```

## 文件结构

```
astrbot_plugin_paperrag/
├── main.py                    # 插件入口
├── rag_engine.py              # 配置和工厂函数
├── hybrid_parser.py           # PDF解析+语义分块
├── hybrid_index.py            # Milvus索引管理
├── hybrid_rag.py              # RAG引擎
├── multimodal_extractor.py    # 多模态提取器
├── embedding_providers.py     # Embedding提供者
├── reference_processor.py     # 参考文献解析
├── milvus_manager.py          # Milvus管理器
├── ollama_embedding.py        # Ollama Embedding
├── llama_index_reranker.py    # 重排序
├── reranker.py                # 重排序封装
└── docs/
    ├── ARCHITECTURE.md        # 本文档
    ├── CHANGELOG.md          # 变更记录
    └── INDEX.md              # 文档索引
```
