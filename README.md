# 📚 Paper RAG Plugin

本地论文库RAG检索插件，支持多模态PDF向量化（文本、图片、表格、公式）和智能问答。

## 🆕 最新更新 (2026-03-23)

### v2.1 更新内容

#### 性能优化
- ✅ **批量 Embedding**：每篇PDF仅1次API调用，节省80-90% RPD配额
  - 从 `N chunks = N次调用` 优化为 `N chunks = 1次批量调用`
  - 处理能力提升约50倍（1000 RPD从约20篇PDF提升到约1000篇PDF）

#### 代码质量提升
- ✅ **日志精简**：删除冗余日志，减少70-80%日志输出
  - 移除所有 `[STEP X]`、`[WRAPPER STEP X]`、`[EMBED STEP X]` 调试日志
  - 移除属性访问、连接模式、索引类型等详细日志
  - 保留关键错误和警告信息

#### Bug修复
- ✅ **类型安全**：修复Pylance类型检查警告
  - 修复 `drop_collection()` 未await问题
  - 修复 `None` 类型参数问题
  - 修复视觉编码器类型访问问题

#### 文件过滤
- ✅ **Meta文件过滤**：自动跳过macOS `._` 开头的元数据文件
  - 避免导入系统生成的隐藏文件
  - 提升导入准确性

---

## 🎯 核心特性

- ✅ **本地向量存储**: Milvus Lite进行本地向量存储，保护论文隐私
- ✅ **高质量嵌入**: 集成AstrBot Embedding Provider，灵活配置嵌入模型
- ✅ **智能问答**: 基于检索增强生成(RAG)技术，提供准确的论文相关回答
- ✅ **多种检索模式**: 支持纯检索模式和RAG生成模式
- ✅ **缓存优化**: 内置LRU缓存，提升重复查询性能
- ✅ **引用标注**: 自动标注引用来源，方便追溯原文

### 多模态PDF处理（v2.0）

- ✅ **图片提取**：自动识别并保留PDF中的图片
  - 🆕 **NMS去重**：基于 Bbox 的 Non-Maximum Suppression 过滤重复图片
  - 🆕 **智能优选**：小图与大图重叠时，优先保留大图
- ✅ **表格提取**：使用pdfplumber提取表格，支持Markdown格式
- ✅ **公式识别**：识别LaTeX公式（`$$...$$`, `\(...\)`, `\begin{equation}`）
- ✅ **智能分块**：图片/表格/公式完整保留，不分块
- ✅ **优雅降级**：依赖不可用时自动回退到文本模式
- ✅ **语义分块**：基于标题和段落的智能分块策略

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ~/AstrBot/data/plugins/astrbot_plugin_paperrag
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

---

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
    "chunk_overlap": 0,
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
| `chunk_overlap` | 块间重叠大小 | 0（禁用以避免无限循环） |
| `min_chunk_size` | 最小块大小 | 100 |
| `use_semantic_chunking` | 启用语义分块 | true |
| `enable_multimodal` | 启用多模态提取 | true |
| `multimodal.extract_images` | 提取图片 | true |
| `multimodal.extract_tables` | 提取表格 | true |
| `multimodal.extract_formulas` | 提取公式 | true |

#### 多模态高级配置（可选）

```json
{
    "enable_multimodal": true,
    "multimodal": {
        "enabled": true,
        "extract_images": true,
        "extract_tables": true,
        "extract_formulas": true,
        "nms_iou_threshold": 0.5,
        "enable_nms": true
    }
}
```

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `nms_iou_threshold` | NMS IoU阈值（0-1），越小过滤越严格 | 0.5 |
| `enable_nms` | 是否启用NMS图片去重 | true |

---

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

---

## 📊 性能优化建议

### 分块大小配置

| 场景 | chunk_size | chunk_overlap | 说明 |
|-----|-----------|---------------|------|
| 学术论文 | 512-768 | 0（推荐） | 较大的块保留更多上下文，禁用重叠避免bug |
| 技术文档 | 384-512 | 0（推荐） | 平衡精度和速度，禁用重叠避免bug |
| 长文档 | 768-1024 | 0（推荐） | 减少分块数量，禁用重叠避免bug |

> ⚠️ **重要提示**：由于已知无限循环bug，建议将 `chunk_overlap` 设置为 0。如需启用重叠，请确保 `chunk_overlap < chunk_size / 2`。

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

---

## 🔧 技术架构

### 多模态处理流程

```
PDF文件
   ↓
[多模态提取]
   ├─ 文本 → PyMuPDF
   ├─ 图片 → PyMuPDF (位置信息)
   │   └─ 🆕 NMS去重 → 过滤重复/重叠图片
   ├─ 表格 → pdfplumber (Markdown)
   └─ 公式 → LaTeX正则
   ↓
[智能分块]
   ├─ 文本 → 语义分块
   ├─ 图片 → 完整保留
   ├─ 表格 → 完整保留
   └─ 公式 → 完整保留
   ↓
[批量向量化] 🆕
   ├─ 文本 → Embedding Provider (1次API调用)
   ├─ 表格/公式 → 文本嵌入 (批量)
   └─ 图片 → SigLIP (可选)
   ↓
[Milvus存储]
   ↓
[检索 + RAG生成]
```

### 图片去重机制

```
图片提取 → 多级过滤 → 保留唯一图片
   ↓
【第一级】尺寸过滤
   ├─ 宽度 < 10px → 移除
   ├─ 高度 < 10px → 移除
   └─ 使用 image.size（原始图片尺寸）
   ↓
【第二级】图注去重 🆕
   ├─ 提取图注编号（如 "Figure 1"）
   ├─ 相同图注的图片 → 判定为同图的不同版本
   ├─ 只保留尺寸最大的图片
   └─ 过滤缩略图、预览图、低分辨率版本
   ↓
【第三级】尺寸去重
   ├─ 相同 (宽, 高) → 判定为重复
   ├─ 保留第一次出现的图片
   └─ 过滤图标、分隔线等重复元素
   ↓
【第四级】NMS位置去重
   ├─ 计算所有图片的 Bbox 面积
   ├─ 按面积降序排序
   ├─ 逐对比较 Bbox 重叠
   │   ├─ IoU > 0.5 → 判定为重叠
   │   ├─ 面积大的图片 → 保留
   │   └─ 面积小的图片 → 移除
   └─ 返回过滤后的图片列表
```

**去重原理**：
- **尺寸过滤**：在调整大小前检查原始尺寸，过滤过小图片
- **图注去重**：学术论文中同一图注对应多分辨率版本（缩略图、预览图、完整图），只保留最高清版本
- **尺寸去重**：相同尺寸的图片通常是重复元素（logo、图标），只保留一份
- **NMS去重**：处理同一图片在PDF中多次嵌入的情况，过滤位置重叠的小图

### 批量Embedding优化 🆕

```python
# 旧方式：逐条调用（N次API调用）
for chunk in chunks:
    embedding = await get_text_embedding(chunk)  # ❌ 浪费RPD

# 新方式：批量调用（1次API调用）
embeddings = await embed(chunks)  # ✅ 节省RPD
```

**RPD节省效果**：
- 1篇PDF(50 chunks): 从50次调用 → 1次调用 (节省98%)
- 100篇PDF: 从5000次调用 → 100次调用 (节省98%)
- 1000 RPD配额: 处理能力从~20篇 → ~1000篇

---

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

### 问题：RPD配额耗尽

**原因**：Gemini Embedding API达到每日1000次请求限制

**解决方案**：
1. **已优化**：批量调用已启用，节省80-90% RPD
2. **升级配额**：在 [AI Studio](https://aistudio.google.com/) 绑定账单，RPD从1000提升到150,000+
3. **切换模型**：使用OpenAI等其他Embedding Provider
4. **本地模型**：部署 BGE-M3 等本地嵌入模型

---

## 📚 开发指南

### 插件架构

```
main.py (插件入口)
  ├── PaperRAGPlugin (Star)
  │   ├── __init__() - 初始化
  │   ├── _get_engine() - 获取RAG引擎（懒加载）
  │   ├── 缓存管理
  │   └── 命令处理器
  │
rag_engine.py (核心引擎)
  ├── RAGConfig - 配置数据类
  ├── EmbeddingProviderWrapper - Embedding Provider 包装
  ├── MilvusLiteStore - 向量存储
  ├── PaperRAGEngine - RAG引擎主类
  └── 文档解析器
```

### 设计模式

1. **单例模式**: RAG 引擎懒加载，全局唯一实例
2. **包装器模式**: 统一 Embedding Provider 接口
3. **策略模式**: 支持多种文档解析策略
4. **缓存模式**: LRU 缓存提升性能

### 核心组件

#### 1. PaperRAGPlugin (main.py)

```python
@register(
    "paper_rag",
    "YourName",
    "本地论文库RAG检索插件",
    "1.0.0",
    "https://github.com/your/repo"
)
class PaperRAGPlugin(Star):
    def __init__(self, context: Context, config: dict = {}):
        super().__init__(context)
        self.config = config or {}
        self.context = context

        # 插件配置
        self.enabled = self.config.get("enabled", True)

        # 缓存配置
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl_seconds", 3600)
        self.cache_max_size = self.config.get("cache_max_entries", 100)
        self._response_cache = {}

        # RAG引擎（懒加载）
        self._engine = None
        self._config_valid = False
```

#### 2. RAGConfig (rag_engine.py)

```python
@dataclass
class RAGConfig:
    """RAG配置类"""
    # Provider配置
    embedding_provider_id: str = ""
    llm_provider_id: str = ""

    # Milvus配置
    milvus_lite_path: str = "./data/milvus_papers.db"
    collection_name: str = "paper_embeddings"

    # 检索配置
    embed_dim: int = 768
    top_k: int = 5
    similarity_cutoff: float = 0.3

    # 论文目录
    papers_dir: str = "./papers"
```

### 开发工作流

```bash
# 进入插件目录
cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag

# 安装依赖
uv pip install -r requirements.txt

# 修改代码
vim main.py  # 或 rag_engine.py

# 重启 AstrBot
# 修改会自动热加载（如果启用）
```

### 类型注解

```python
from typing import List, Dict, Any, Optional, cast

async def search(self, query: str, mode: str = "rag") -> Dict[str, Any]:
    """明确的返回类型注解"""
    pass

def _get_engine(self) -> Optional[PaperRAGEngine]:
    """可能返回 None 的类型"""
    pass
```

---

## 🧪 测试指南

### 一键测试

```bash
cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag
python test_pdf.py /path/to/your/paper.pdf
```

### 测试流程

#### 步骤1: 安装依赖

```bash
pip install -r requirements.txt
```

#### 步骤2: 准备测试PDF

将测试PDF文件放到可访问的位置。

#### 步骤3: 运行测试

```bash
python test_pdf.py ~/Documents/test_paper.pdf
```

### 测试结果解读

#### 完美通过

```
🎉 所有测试通过! (5/5)
```
→ 插件功能完整，可以正常使用

#### 部分通过

```
⚠️  部分测试失败 (4/5)
```
→ 核心功能可用，部分功能降级

---

## 📄 许可证

MIT License

## 🙏 致谢

- [SigLIP](https://arxiv.org/abs/2303.15343) - 视觉编码器
- [Milvus](https://milvus.io/) - 向量数据库
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF解析
- [pdfplumber](https://github.com/jsvine/pdfplumber) - 表格提取
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 聊天机器人框架
