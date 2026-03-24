# 📚 Paper RAG Plugin v2.0 - 用户指南

本地论文库RAG检索插件，为AstrBot提供智能的论文检索和问答能力（支持多模态VLM问答）。

## ✨ 核心功能

- 🔍 **智能检索**：基于语义相似度快速定位相关文档片段
- 💡 **AI问答**：结合检索内容生成准确、有引用的答案
- 📄 **多格式支持**：PDF、Word、TXT、Markdown、HTML
- 🖼️ **多模态提取**：自动识别PDF中的图片、表格、公式
- 🖼️ **多模态查询**：支持图片输入进行VLM问答（方案B：原图+VLM）
- 💾 **本地存储**：所有数据存储在本地，保护隐私
- ⚡ **缓存加速**：常用查询结果缓存，响应更快
- 🦙 **Ollama支持**：使用本地Ollama服务进行免费无限制的向量化（推荐）
- 🏎️ **重排序支持**：FlagEmbedding加速检索精度

---

## 🚀 快速开始（5分钟）

### 第一步：安装插件

```bash
cd ~/AstrBot/data/plugins/astrbot_plugin_paperrag
pip install -r requirements.txt
pip install llama-index-core llama-index-vector-stores-milvus FlagEmbedding
```

### 第二步：配置插件

**方式A：使用Ollama本地Embedding（推荐，免费无限制）**

在 **AstrBot WebUI → 插件 → paper_rag → 插件配置** 中：

| 配置项 | 值 | 说明 |
|-------|-----|------|
| Embedding模式 | `Ollama本地模式` | 使用Ollama |
| 向量嵌入维度 | `1024` | BGE-M3固定1024维 |
| Ollama模型名称 | `bge-m3` | 模型名称 |
| glm_api_key | `glm-4.7-flash`（可选） | 用于RAG回答生成 |
| 论文文件存放目录 | `./papers` | PDF存放路径 |
| 启用插件 | ✅ | - |

> ⚠️ **使用Ollama前需要先安装和配置**：
> ```bash
> # 1. 安装Ollama
> curl -fsSL https://ollama.com/install.sh | sh
>
> # 2. 下载BGE-M3模型
> ollama pull bge-m3
>
> # 3. 启动服务
> ollama serve
> ```
> 详细配置见：[OLLAMA_GUIDE.md](docs/OLLAMA_GUIDE.md)

**方式B：使用Gemini API（快速，有配额限制）**

| 配置项 | 值 | 说明 |
|-------|-----|------|
| Embedding模式 | `API模式` | 使用API |
| Embedding 服务提供商 | `gemini_embedding` | Gemini Embedding API |
| 向量嵌入维度 | `768` | Gemini固定768维 |
| glm_api_key | `glm-4.7-flash`（可选） | 用于RAG回答生成 |
| 论文文件存放目录 | `./papers` | PDF存放路径 |
| 启用插件 | ✅ | - |

### 第三步：使用插件

```bash
# 1. 创建论文目录并放入PDF文件
mkdir papers
cp ~/Downloads/*.pdf papers/

# 2. 添加文档到知识库
/paper add

# 3. 搜索论文
/paper search 这篇论文的主要创新点是什么？
```

---

## 📖 使用说明

### 命令速查

| 命令 | 功能 | 示例 |
|------|------|------|
| `/paper search <问题>` | 搜索并生成回答 | `/paper search attention机制的原理` |
| `/paper search <问题> retrieve` | 仅检索相关片段 | `/paper search CNN retrieve` |
| `/paper list` | 查看已收录文档 | `/paper list` |
| `/paper add [目录]` | 添加文档（需管理员） | `/paper add ~/Documents/papers` |
| `/paper rebuild [目录] confirm` | 清空并重建知识库 | `/paper rebuild ./papers confirm` |
| `/paper clear confirm` | 清空知识库（需管理员） | `/paper clear confirm` |

### 使用示例

**示例1：添加论文**

```
你: /paper add
Bot: 🔍 扫描目录: ./papers
Bot: 📄 发现 10 个文档文件
Bot: ⏳ 开始导入...
Bot: ✅ [1/10] deep_learning.pdf - 85 个片段
Bot: ✅ [2/10] transformer.pdf - 92 个片段
...
Bot: ✅ 导入完成
Bot: 📊 总计: 10 个文件, 850 个片段
Bot: 💡 提示: 使用 /paper search [问题] 来检索文档
```

**示例2：搜索问答**

```
你: /paper search 什么是注意力机制？
Bot: 🔍 正在检索文档库...
Bot:
Bot: 💡 **回答**
Bot:
Bot: 注意力机制（Attention Mechanism）是一种神经网络架构...
Bot:
Bot: 📚 **参考文献**
Bot:
Bot: [1] **attention_is_all_you_need.pdf** (片段 #12)
Bot: > The attention mechanism allows the model to focus on...
```

---

## ⚙️ 配置详解

### 基础配置

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `enabled` | 启用插件 | `true` | ✅ |
| `embedding_mode` | Embedding模式 | `ollama` | `ollama`（推荐免费）/ `api` |
| `embedding_provider_id` | Embedding Provider ID（API模式） | `gemini_embedding` | Gemini / OpenAI |
| `glm_api_key` | 智谱AI API密钥 | - | 用于RAG回答生成 |
| `glm_model` | GLM文本模型 | `glm-4.7-flash` | `glm-4.7-flash` / `glm-4` |
| `glm_multimodal_model` | GLM多模态模型 | `glm-4.6v-flash` | `glm-4.6v-flash`（用于图片问答） |
| `papers_dir` | 论文目录 | `./papers` | `./papers` |
| `figures_dir` | 图片存储目录 | `data/figures` | 插件目录下的 data/figures |
| `embed_dim` | 向量维度 | `768` | `1024` (BGE-M3) / `768` (Gemini) / `1536` (OpenAI) |

> 💡 **Embedding模式对比**：
> - **Ollama模式**：免费、无限制、隐私保护、需要安装Ollama
> - **API模式**：快速、有配额限制、需要API密钥

### Ollama本地Embedding配置

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `ollama.base_url` | Ollama服务地址 | `http://localhost:11434` | 默认 |
| `ollama.model` | Ollama模型名称 | `bge-m3` | `bge-m3`（推荐）/ `nomic-embed-text` |
| `ollama.batch_size` | 并发批处理大小 | `10` | `10-20`（根据硬件） |
| `ollama.timeout` | 请求超时（秒） | `120.0` | 默认 |

> 🦙 **Ollama详细配置指南**：[OLLAMA_GUIDE.md](docs/OLLAMA_GUIDE.md)

### 检索配置

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `top_k` | 返回片段数 | `5` | `5` |
| `similarity_cutoff` | 相似度阈值 | `0.3` | `0.3` |

> 💡 **Embedding Provider 说明**：插件使用 AstrBot 中配置的 Embedding Provider。推荐使用 Ollama（免费无限制）。

### 分块配置

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `chunk_size` | 分块大小（字符） | `512` | `512` (论文) / `384` (文档) |
| `chunk_overlap` | 块间重叠 | `0` | `0` |
| `min_chunk_size` | 最小块大小 | `100` | `100` |
| `use_semantic_chunking` | 智能分块 | `true` | ✅ |

### 多模态配置

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `enable_multimodal` | 启用多模态 | `true` |
| `multimodal.extract_images` | 提取图片 | `true` |
| `multimodal.extract_tables` | 提取表格 | `true` |
| `multimodal.extract_formulas` | 提取公式 | `true` |
| `multimodal.nms_iou_threshold` | 图片去重阈值 | `0.5` |
| `multimodal.enable_nms` | 启用NMS去重 | `true` |

> 💡 **VLM路由说明**：查询含视觉关键词（"图"、"表格"、"公式"等）或检索结果关联图片时，自动使用 `glm_multimodal_model` 进行多模态回答。

**生产环境推荐配置**：
```json
{
    "enable_multimodal": true,
    "multimodal": {
        "extract_images": false,
        "extract_tables": true,
        "extract_formulas": true
    }
}
```

### 重排序配置

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `enable_reranking` | 启用重排序 | `false` | ✅（提升精度） |
| `reranking_model` | 重排序模型 | `BAAI/bge-reranker-v2-m3` | 默认 |
| `reranking_device` | 运行设备 | `auto` | `auto`（自动检测） |
| `reranking_adaptive` | 自适应模式 | `true` | ✅ |
| `reranking_batch_size` | 批处理大小 | `32` | `32-64` |
| `reranking_threshold` | 分数阈值 | `0.0` | `0.0` |

> 💡 **重排序说明**：
> - **性能提升**：检索精度提升15-25%
> - **延迟增加**：200-500ms（MPS加速）
> - **内存占用**：约2GB
> - **依赖安装**：`pip install -U FlagEmbedding`

**配置场景示例**：

1. **新手/默认配置**（推荐）
```json
{
  "enable_reranking": true
}
```

2. **Apple Silicon Mac**（MPS加速）
```json
{
  "enable_reranking": true,
  "reranking_device": "mps",
  "reranking_batch_size": 64
}
```

3. **NVIDIA GPU**（CUDA加速）
```json
{
  "enable_reranking": true,
  "reranking_device": "cuda",
  "reranking_batch_size": 128
}
```

4. **低内存/CPU**
```json
{
  "enable_reranking": true,
  "reranking_device": "cpu",
  "reranking_batch_size": 16
}
```

5. **高精度模式**
```json
{
  "enable_reranking": true,
  "reranking_model": "BAAI/bge-reranker-large",
  "reranking_threshold": 0.3
}
```

---

## 💡 使用技巧

### 1. 选择合适的分块大小

| 文档类型 | 推荐chunk_size | 说明 |
|---------|---------------|------|
| 学术论文 | `512-768` | 保留更多上下文 |
| 技术文档 | `384-512` | 平衡精度和速度 |
| 长篇报告 | `768-1024` | 减少分块数量 |

### 2. 提高搜索准确度

- **使用具体问题**：避免太宽泛的问题
- **包含关键词**：提问时使用专业术语
- **调整top_k**：增加返回片段数（默认5）
- **调整相似度阈值**：提高 `similarity_cutoff` 过滤低质量结果
- **使用图片查询**：支持图片输入进行多模态检索

### 3. 加速导入

- **使用Ollama**：本地批量向量化，无API限制（推荐）
- **批量Embedding**：自动启用批量处理
- **批量导入**：一次性添加多个PDF
- **禁用图片提取**：设置 `multimodal.extract_images: false`
- **使用SSD**：将Milvus数据库放在SSD上

---

## ❓ 常见问题

### Q1: 提示"RAG引擎未就绪"

**原因**：Embedding Provider未配置或配置错误

**解决**：
1. 检查 WebUI → 设置 → 模型提供商
2. 确认已添加 Embedding Provider
3. 检查插件配置中的 Provider ID 是否正确

### Q2: 导入后chunks=0

**原因**：PDF是扫描版（无文本层）或依赖未安装

**解决**：
1. 确认PDF不是扫描版
2. 安装依赖：`pip install -r requirements.txt
pip install llama-index-core llama-index-vector-stores-milvus`
3. 运行测试：`python test_pdf.py paper.pdf`

### Q3: 搜索结果不准确

**原因**：分块大小不合适或相似度阈值过高

**解决**：
1. 调整 `chunk_size`（尝试增大）
2. 降低 `similarity_cutoff`（如0.2）
3. 增加 `top_k`（返回更多结果）

### Q4: 提示"RPD配额耗尽"

**原因**：Gemini API每日1000次请求限制

**解决**：
1. 已优化批量调用，通常足够使用
2. 在 [AI Studio](https://aistudio.google.com/) 绑定账单，配额提升至150,000+
3. 切换其他Embedding Provider（如OpenAI）

### Q5: transformers导入错误

**原因**：未安装图片向量化依赖（正常）

**解决**：无需处理，系统会自动降级到文本模式，不影响使用

### Q6: 重排序功能不可用

**原因**：FlagEmbedding未安装

**解决**：
```bash
pip install -U FlagEmbedding
```

**MPS加速不可用**：
- 检查macOS版本 ≥ 12.3
- 更新PyTorch: `pip install --upgrade torch`

### Q7: 批量请求超过100个文本错误

**症状**: `at most 100 requests can be in one batch`

**原因**: Gemini API单次批量请求限制为100个文本

**解决**: ✅ 插件已自动处理，会自动分批无需手动干预

**技术细节**:
- 插件自动检测文本数量
- 超过100个时自动分批处理（每批100个）
- 完全透明，不影响使用体验

---

## 📚 详细文档导航

本插件提供完整的文档体系，涵盖用户使用、开发配置和本地Embedding等方面。

### 🎯 快速导航

| 需求 | 推荐文档 | 说明 |
|------|---------|------|
| **快速上手** | 当前文档 | 用户指南，5分钟快速开始 |
| **文档导航** | [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) | 完整文档索引 |
| **开发调试** | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 开发者文档 |
| **配置Ollama** | [docs/OLLAMA_QUICKREF.md](docs/OLLAMA_QUICKREF.md) | 快速参考 |
| **深入Ollama** | [docs/OLLAMA_GUIDE.md](docs/OLLAMA_GUIDE.md) | 完整指南 |

### 📂 docs/ 目录

**核心文档**：

1. **[DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)** - 📚 文档导航
   - 快速定位所需文档
   - 文档阅读路径
   - 常见问题索引

2. **[DEVELOPMENT.md](docs/DEVELOPMENT.md)** - 🔧 开发指南
   - 技术架构说明
   - 核心组件介绍
   - 开发工作流程
   - 故障排除（10个常见问题）

3. **[OLLAMA_GUIDE.md](docs/OLLAMA_GUIDE.md)** - 🦙 Ollama完整指南
   - Ollama安装和配置
   - 模型选择和下载
   - 性能优化建议
   - 故障排除

4. **[OLLAMA_QUICKREF.md](docs/OLLAMA_QUICKREF.md)** - 🦙 Ollama快速参考
   - 一分钟开始
   - 配置对比
   - 快速故障排除

**修复文档**：

5. **[FIX_LIST_EMPTY_ISSUE.md](docs/FIX_LIST_EMPTY_ISSUE.md)** - 数据库路径问题修复
6. **[FIX_MILVUS_API_COMPAT.md](docs/FIX_MILVUS_API_COMPAT.md)** - API兼容性修复

### 📌 文档定位

- **用户入口**: README.md (本文档)
- **文档索引**: docs/DOCUMENTATION_INDEX.md
- **开发者文档**: docs/DEVELOPMENT.md
- **Ollama配置**: docs/OLLAMA_GUIDE.md, docs/OLLAMA_QUICKREF.md

---

## 📞 获取帮助

- **完整文档**：查看 [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) 浏览所有文档
- **开发文档**：查看 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) 了解技术细节
- **问题反馈**：通过 GitHub Issues 提交问题
- **日志查看**：AstrBot 控制台输出

---

## 📄 许可证

MIT License

## 🙏 致谢

- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 聊天机器人框架
- [Milvus](https://milvus.io/) - 向量数据库
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF解析
