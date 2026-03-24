# 变更记录

## v2.0.0 (2026-03-25) - 重大架构重构

### 架构变更

#### 核心架构演进

| 阶段 | 组件 | 说明 |
|------|------|------|
| v1.x | llama-index | 使用 llama-index 管理索引和检索 |
| v2.0 | HybridIndexManager | 直接使用 pymilvus，避免冲突 |
| v2.0 | HybridPDFParser | 自定义 PDF 解析 + 语义分块 |
| v2.0 | 方案B 图片存储 | 原图 + VLM 路由 |

#### 方案B：图片存储与检索（学术 RAG 最优解）

**核心思路**：保存原图而非 CLIP 向量，VLM 加载原图生成答案

```
PDF解析 → 原图保存到磁盘 → metadata 存 image_path → 检索时关联 → VLM 加载原图
```

**优势**：
- 保留原始图表细节（坐标轴数值、公式符号等）
- 避免 CLIP 向量化信息丢失
- VLM 直接看图回答，减少幻觉
- 成本可控（仅视觉问题用 VLM）

#### VLM 路由逻辑

```python
if query_has_visual_keyword OR sources_have_images:
    use VLM (glm-4.6v-flash)
else:
    use LLM (glm-4.7-flash)
```

---

## 2026-03-25

### 新增功能

- **`/paper rebuild <目录> confirm`** - 清空并重建知识库命令
- **多模态查询支持（方案B）**
  - 支持图片输入查询
  - 自动使用 `glm-4.6v-flash` 多模态模型
  - Base64 编码图片
  - VLM 路由自动判断

- **图片存储架构（方案B）**
  - PDF 解析时将图片保存到磁盘（`{pdf_name}_p{page}_i{index}.png`）
  - Milvus metadata 中存储 `image_path` 而非图像向量
  - 基于页码 proximity 关联图片到文本块
  - VLM 加载原始图片生成答案

- **图片合并提取**
  - 跳过基于图注的去重
  - 保留所有相同 Figure 的子图
  - 计算外接矩形合并为完整大图

### 语义分块改进

- 按段落(`\n\n`) > 句子(。！？) > 子句(，,) 优先级分割
- 支持 `chunk_overlap` 保持语义连贯
- `_clean_overlap()` 避免句子中间断开

### 架构优化

- 使用 `HybridIndexManager` 直接管理 Milvus（避免 llama-index 冲突）
- 独立连接别名 `paperrag_hybrid`
- `TYPE_CHECKING` 避免循环导入
- `cast()` 修复 Pylance 类型检查
- 支持 FlagEmbedding 重排序

### Bug 修复

- 修复 metadata JSON 序列化问题（处理 `Rect` 等特殊类型）
- 修复连接复用问题（断开旧连接再创建新连接）
- 修复 Milvus 路径问题（使用插件目录下的默认路径）
- 修复 `list_documents` 显示每个文件单独统计

### 配置变更

| 配置项 | 旧默认值 | 新默认值 |
|--------|---------|---------|
| `embedding_mode` | `api` | `ollama` |
| `glm_model` | `glm-4.6v-flash` | `glm-4.7-flash`（文本模型） |
| `glm_multimodal_model` | 无 | `glm-4.6v-flash`（多模态模型） |
| `milvus_lite_path` | `./data/milvus_papers.db` | 空（插件目录下） |
| `multimodal.enabled` | `false` | `true` |
| `figures_dir` | papers/figures | 插件目录下 data/figures |

---

## 2026-03-24

### 混合架构重构

- 新增 `HybridPDFParser`：保留多模态提取 + 语义分块
- 新增 `HybridIndexManager`：直接使用 pymilvus（避免 llama-index 冲突）
- 新增 `HybridRAGEngine`：完整 RAG 流程
- 默认模型：`glm_model` = `glm-4.7-flash`，`glm_multimodal_model` = `glm-4.6v-flash`

### 移除的功能

- ~~表格向量化~~（暂不处理）
- legacy 目录（已删除）
- 旧版 `milvus_manager.py`（被 `hybrid_index.py` 替代）

### 新增参数

```python
glm_multimodal_model = "glm-4.6v-flash"  # 多模态模型（用于图片问答）
min_chunk_size = 100                       # 最小块大小
figures_dir = ""                           # 图片存储目录
enable_reranking = False                  # 重排序功能
reranking_model = "BAAI/bge-reranker-v2-m3"  # 重排序模型
```
