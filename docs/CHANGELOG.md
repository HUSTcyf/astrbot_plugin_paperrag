# 变更记录

## 2026-03-25

### 新增功能

- **`/paper rebuild <目录> confirm`** - 清空并重建知识库命令
  - 清空现有向量数据库
  - 扫描指定目录重新导入
  - 显示进度和统计信息

- **多模态查询支持**
  - 支持图片输入查询
  - 自动使用 `glm-4.6v-flash` 多模态模型
  - Base64 编码图片

### 语义分块改进

- 按段落(`\n\n`) > 句子(。！？) > 子句(，,) 优先级分割
- 支持 `chunk_overlap` 保持语义连贯
- `_clean_overlap()` 避免句子中间断开

### 架构优化

- 使用 `HybridIndexManager` 直接管理 Milvus（避免 llama-index 冲突）
- 独立连接别名 `paperrag_hybrid`
- `TYPE_CHECKING` 避免循环导入
- `cast()` 修复 Pylance 类型检查

### Bug 修复

- 修复 metadata JSON 序列化问题（处理 `Rect` 等特殊类型）
- 修复连接复用问题（断开旧连接再创建新连接）
- 修复 Milvus 路径问题（使用插件目录下的默认路径）
- 修复 `list_documents` 显示每个文件单独统计

### 配置变更

| 配置项 | 旧默认值 | 新默认值 |
|--------|---------|---------|
| `embedding_mode` | `api` | `ollama` |
| `glm_model` | `glm-4.7-flash` | `glm-4.6v-flash` |
| `milvus_lite_path` | `./data/milvus_papers.db` | 空（插件目录下） |

---

## 2026-03-24

### 混合架构重构

- 新增 `HybridPDFParser`：保留多模态提取 + 语义分块
- 新增 `HybridIndexManager`：直接使用 pymilvus（避免 llama-index 冲突）
- 新增 `HybridRAGEngine`：完整 RAG 流程
- 默认模型改为 `glm-4.6v-flash`

### 移除的功能

- ~~表格向量化~~（暂不处理）
- legacy 目录（已删除）
- 旧版 `milvus_manager.py`（被 `hybrid_index.py` 替代）

### 新增参数

```python
glm_multimodal_model = "glm-4.6v-flash"  # 多模态模型
min_chunk_size = 100                       # 最小块大小
```

---

## 历史

- 完整重构记录见 [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)（如保留）
- 迁移指南见 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)（如保留）
