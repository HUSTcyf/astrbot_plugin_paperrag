# 📚 Paper RAG Plugin - 用户指南

本地论文库RAG检索插件，为AstrBot提供智能的论文检索和问答能力。

## ✨ 核心功能

- 🔍 **智能检索**：基于语义相似度快速定位相关文档片段
- 💡 **AI问答**：结合检索内容生成准确、有引用的答案
- 📄 **多格式支持**：PDF、Word、TXT、Markdown、HTML
- 🖼️ **多模态提取**：自动识别PDF中的图片、表格、公式
- 💾 **本地存储**：所有数据存储在本地，保护隐私
- ⚡ **缓存加速**：常用查询结果缓存，响应更快

---

## 🚀 快速开始（5分钟）

### 第一步：安装插件

```bash
cd ~/AstrBot/data/plugins/astrbot_plugin_paperrag
pip install -r requirements.txt
```

### 第二步：配置Embedding Provider

在 **AstrBot WebUI → 设置 → 模型提供商** 中添加：

| 配置项 | 值 |
|-------|-----|
| 类型 | Gemini |
| ID | `gemini_embedding` |
| API Key | [获取密钥](https://makersuite.google.com/app/apikey) |
| 模型 | `gemini-embedding-2-preview` |

### 第三步：配置插件

在 **AstrBot WebUI → 插件 → paper_rag → 插件配置** 中：

| 配置项 | 值 |
|-------|-----|
| Embedding 服务提供商 | `gemini_embedding` |
| LLM Provider ID | `glm-4.7-flash`（可选） |
| 论文文件存放目录 | `./papers` |
| 启用插件 | ✅ |

### 第四步：使用插件

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
| `embedding_provider_id` | Embedding Provider ID | - | `gemini_embedding` |
| `llm_provider_id` | LLM Provider ID（可选） | - | `glm-4.7-flash` |
| `papers_dir` | 论文目录 | `./papers` | `./papers` |
| `embed_dim` | 向量维度 | `768` | `768` (Gemini) / `1536` (OpenAI) |
| `top_k` | 返回片段数 | `5` | `5` |
| `similarity_cutoff` | 相似度阈值 | `0.3` | `0.3` |

### 分块配置

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `chunk_size` | 分块大小（字符） | `512` | `512` (论文) / `384` (文档) |
| `chunk_overlap` | 块间重叠 | `0` | `0`（避免bug） |
| `min_chunk_size` | 最小块大小 | `100` | `100` |
| `use_semantic_chunking` | 智能分块 | `true` | ✅ |

> 💡 **提示**：`chunk_overlap` 建议设为 0，避免已知分块bug。

### 多模态配置

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `enable_multimodal` | 启用多模态 | `true` |
| `multimodal.extract_images` | 提取图片 | `true` |
| `multimodal.extract_tables` | 提取表格 | `true` |
| `multimodal.extract_formulas` | 提取公式 | `true` |
| `multimodal.nms_iou_threshold` | 图片去重阈值 | `0.5` |
| `multimodal.enable_nms` | 启用NMS去重 | `true` |

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

### 3. 加速导入

- **批量导入**：一次性添加多个PDF
- **禁用图片提取**：设置 `extract_images: false`
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
2. 安装依赖：`pip install -r requirements.txt`
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

---

## 📞 获取帮助

- **开发文档**：查看 [DEVELOPMENT.md](DEVELOPMENT.md) 了解技术细节
- **问题反馈**：通过 GitHub Issues 提交问题
- **日志查看**：AstrBot 控制台输出

---

## 📄 许可证

MIT License

## 🙏 致谢

- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 聊天机器人框架
- [Milvus](https://milvus.io/) - 向量数据库
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF解析
