# 📚 Paper RAG Plugin

本地论文库RAG检索插件，支持多模态PDF向量化（文本、图片、表格、公式）和智能问答。

## 🆕 新功能

### 多模态PDF处理（v2.0）

- ✅ **图片提取**：自动识别并保留PDF中的图片
- ✅ **表格提取**：使用pdfplumber提取表格，支持Markdown格式
- ✅ **公式识别**：识别LaTeX公式（`$$...$$`, `\(...\)`, `\begin{equation}`）
- ✅ **智能分块**：图片/表格/公式完整保留，不分块
- ✅ **优雅降级**：依赖不可用时自动回退到文本模式
- ✅ **语义分块**：基于标题和段落的智能分块策略

### 分块策略

```
文本内容：语义分块（按标题、段落边界）
图片内容：完整保留（独立chunk，支持视觉检索）
表格内容：完整保留（独立chunk，Markdown格式）
公式内容：完整保留（独立chunk，LaTeX格式）
```

## 🎯 核心特性

- ✅ **本地向量存储**: Milvus Lite进行本地向量存储，保护论文隐私
- ✅ **高质量嵌入**: 集成AstrBot Embedding Provider，灵活配置嵌入模型
- ✅ **智能问答**: 基于检索增强生成(RAG)技术，提供准确的论文相关回答
- ✅ **多种检索模式**: 支持纯检索模式和RAG生成模式
- ✅ **缓存优化**: 内置LRU缓存，提升重复查询性能
- ✅ **引用标注**: 自动标注引用来源，方便追溯原文

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag
pip install -r requirements.txt
```

**基础依赖**（必需）：
- pymilvus[milvus_lite] - 向量数据库
- PyMuPDF - PDF解析
- pdfplumber - 表格提取
- python-docx - Word文档
- pillow - 图像处理

**可选依赖**（用于图片向量化）：
- transformers - HuggingFace模型
- torch - PyTorch

> 💡 如果未安装transformers，系统会自动降级到文本模式，不会影响核心功能。

### 2. 配置 Embedding Provider

在 **AstrBot WebUI → 设置 → 模型提供商** 中添加 Embedding Provider：

**示例配置（Gemini）**:
- 类型: `Gemini`
- ID: `gemini_embedding`
- API Key: 你的 Gemini API Key
- 模型: `gemini-embedding-2-preview`

获取 API Key: [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. 配置 LLM Provider（可选）

如需使用RAG生成功能，在 **AstrBot WebUI → 设置 → 模型提供商** 中添加 LLM Provider：

**示例配置（智谱GLM）**:
- 类型: `zhipu`
- ID: `glm-4.7-flash`
- API Key: 你的智谱 API Key
- 模型: `glm-4.7-flash`

## ⚙️ 配置选项

在插件配置中添加：

```json
{
    "enabled": true,
    "embedding_provider_id": "gemini_embedding",
    "llm_provider_id": "glm-4.7-flash",
    "milvus_lite_path": "./data/milvus_papers.db",

    "use_semantic_chunking": true,
    "chunk_size": 512,
    "chunk_overlap": 50,
    "min_chunk_size": 100,

    "enable_multimodal": true,
    "multimodal": {
        "enabled": true,
        "extract_images": true,
        "extract_tables": true,
        "extract_formulas": true
    }
}
```

### 配置说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `chunk_size` | 目标块大小（字符数） | 512 |
| `chunk_overlap` | 块间重叠大小 | 50 |
| `min_chunk_size` | 最小块大小 | 100 |
| `use_semantic_chunking` | 启用语义分块 | true |
| `enable_multimodal` | 启用多模态提取 | true |
| `multimodal.extract_images` | 提取图片 | true |
| `multimodal.extract_tables` | 提取表格 | true |
| `multimodal.extract_formulas` | 提取公式 | true |

## 📖 使用方法

### 命令

```bash
/paper search <问题>              # 搜索文档并回答
/paper search <问题> retrieve     # 仅检索，不生成答案
/paper list                       # 列出所有文档
/paper add [目录]                 # 添加文档到知识库
/paper clear confirm              # 清空知识库
```

### 示例

**1. 添加文档**：
```
/paper add ~/Documents/papers
```

输出：
```
📄 Found 10 document files
⏳ Starting import...
📖 [1/10] Parsing: example.pdf
✅ [1/10] example.pdf - 85 chunks (including 3 tables, 2 formulas)
```

**2. 搜索文档**：
```
/paper search What is the attention mechanism?
```

输出：
```
💡 Answer

The attention mechanism is a neural network architecture that...

📚 References

[1] attention.pdf (chunk #5)
> The attention mechanism computes weighted sums of values...

[2] transformer.pdf (chunk #12)
> Table: Attention head comparison
```

## 📊 性能优化建议

### 分块大小配置

| 场景 | chunk_size | chunk_overlap | 说明 |
|-----|-----------|---------------|------|
| 学术论文 | 512-768 | 50-100 | 较大的块保留更多上下文 |
| 技术文档 | 384-512 | 30-50 | 平衡精度和速度 |
| 长文档 | 768-1024 | 100-150 | 减少分块数量 |

### 多模态配置

**生产环境**（稳定，推荐）：
```json
{
    "enable_multimodal": true,
    "multimodal": {
        "enabled": false,
        "extract_images": false,
        "extract_tables": true,
        "extract_formulas": true
    }
}
```

**开发环境**（全功能）：
```json
{
    "enable_multimodal": true,
    "multimodal": {
        "enabled": true,
        "extract_images": true,
        "extract_tables": true,
        "extract_formulas": true
    }
}
```

## 🔧 技术架构

### 多模态处理流程

```
PDF文件
   ↓
[多模态提取]
   ├─ 文本 → PyMuPDF
   ├─ 图片 → PyMuPDF (位置信息)
   ├─ 表格 → pdfplumber (Markdown)
   └─ 公式 → LaTeX正则
   ↓
[智能分块]
   ├─ 文本 → 语义分块
   ├─ 图片 → 完整保留
   ├─ 表格 → 完整保留
   └─ 公式 → 完整保留
   ↓
[向量化]
   ├─ 文本 → Embedding Provider
   ├─ 表格/公式 → 文本嵌入
   └─ 图片 → SigLIP (可选)
   ↓
[Milvus存储]
   ↓
[检索 + RAG生成]
```

### 优雅降级机制

```
启动 → 检查依赖
   ↓
transformers可用?
   ├─ 是 → 启用图片向量化
   └─ 否 → 禁用图片（不报错）
   ↓
pdfplumber可用?
   ├─ 是 → 启用表格提取
   └─ 否 → 禁用表格（不报错）
   ↓
核心文本功能始终可用
```

## 🐛 故障排除

### 问题：添加文档后chunks=0

**可能原因**：
1. PDF是扫描版（无文本层）
2. 依赖未安装

**解决**：
```bash
# 测试PDF
python test_pdf.py paper.pdf

# 安装依赖
pip install -r requirements.txt
```

### 问题：transformers导入错误

**原因**：未安装多模态依赖（正常）

**解决**：无需处理，系统会自动降级到文本模式

### 问题：表格提取失败

**原因**：pdfplumber未安装

**解决**：
```bash
pip install pdfplumber
```

### 问题：分块过多警告

**原因**：文档生成了超过50个chunk

**影响**：可能影响检索性能，但不影响功能

**解决**：调整 `chunk_size` 参数，增大分块大小

## 📚 文档

- [DEVELOPMENT.md](DEVELOPMENT.md) - 开发者指南
- [DEPLOYMENT.md](DEPLOYMENT.md) - 部署指南
- [MULTIMODEL.md](MULTIMODEL.md) - 多模态功能详情

## 🙏 致谢

- [SigLIP](https://arxiv.org/abs/2303.15343) - 视觉编码器
- [Milvus](https://milvus.io/) - 向量数据库
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF解析
- [pdfplumber](https://github.com/jsvine/pdfplumber) - 表格提取

## 📄 许可证

MIT License
