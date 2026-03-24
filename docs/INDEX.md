# 文档索引

## 核心文档

| 文档 | 说明 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构设计、组件说明、配置参数 |
| [CHANGELOG.md](CHANGELOG.md) | 变更记录 |

## 使用指南

### 命令

```bash
/paper add                              # 添加论文
/paper search <query>                   # RAG搜索
/paper search <query> mode=retrieve     # 仅检索
/paper list                             # 列出文档
/paper clear confirm                    # 清空知识库
```

### 配置参数

```json
{
  "embedding_mode": "ollama",
  "ollama_config": {"model": "bge-m3"},
  "milvus_lite_path": "./data/milvus_papers.db",
  "collection_name": "paper_embeddings",
  "embed_dim": 1024,
  "chunk_size": 512,
  "chunk_overlap": 0,
  "glm_api_key": "",
  "glm_model": "glm-4.6v-flash"
}
```

## 核心文件

| 文件 | 功能 |
|------|------|
| `hybrid_parser.py` | PDF解析 + 语义分块 |
| `hybrid_index.py` | Milvus索引管理 |
| `hybrid_rag.py` | RAG引擎 |

## 测试

```bash
python3 test_hybrid.py              # 集成测试
python3 test_semantic_chunker.py   # 分块测试
```

---

**最后更新**: 2026-03-25
