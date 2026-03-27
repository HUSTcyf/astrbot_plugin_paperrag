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
/paper search <query> retrieve          # 仅检索
/paper list                             # 列出文档
/paper rebuild <目录> confirm           # 清空并重建知识库
/paper clear confirm                    # 清空知识库
```

### 配置参数

```json
{
  "embedding_mode": "ollama",
  "ollama_config": {"model": "bge-m3"},
  "embed_dim": 1024,
  "chunk_size": 512,
  "chunk_overlap": 0,
  "text_provider_id": "",
  "multimodal_provider_id": "",
  "llama_vlm_model_path": "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf",
  "llama_vlm_mmproj_path": "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf",
  "llama_vlm_max_tokens": 2560,
  "llama_vlm_temperature": 0.7
}
```

## 核心文件

| 文件 | 功能 |
|------|------|
| `main.py` | 插件入口、命令注册 |
| `rag_engine.py` | 配置和工厂函数 |
| `hybrid_parser.py` | PDF解析 + 语义分块 |
| `hybrid_index.py` | Milvus索引管理 |
| `hybrid_rag.py` | RAG引擎 |
| `multimodal_extractor.py` | 多模态提取器 |
| `embedding_providers.py` | Embedding提供者 |
| `reference_processor.py` | 参考文献解析 |
| `llama_cpp_vlm_provider.py` | Llama.cpp VLM本地推理 |

## 测试

```bash
python3 test_hybrid.py              # 集成测试
python3 test_semantic_chunker.py   # PDF解析和分块测试
```

---

**最后更新**: 2026-03-27
