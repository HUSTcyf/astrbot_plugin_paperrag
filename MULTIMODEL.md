# 多模态PDF向量化功能

Paper RAG插件现已集成多模态向量化功能，支持图片、表格、公式的智能提取和检索。

## 核心特性

### 1. 智能内容提取

| 内容类型 | 提取方式 | 分块策略 | 向量化 |
|---------|---------|---------|--------|
| **文本** | PyMuPDF | 语义分块 | ✅ 文本嵌入 |
| **图片** | PyMuPDF | ❌ 不分块（完整保留） | ✅ SigLIP视觉编码（可选） |
| **表格** | pdfplumber | ❌ 不分块（完整保留） | ✅ 文本嵌入（Markdown） |
| **公式** | LaTeX正则 | ❌ 不分块（完整保留） | ✅ 文本嵌入（LaTeX） |

### 2. 优雅降级

- ✅ 默认启用多模态功能
- ✅ 依赖不可用时自动降级到文本模式
- ✅ 不会因缺少依赖而崩溃
- ✅ 清晰的日志提示当前状态

### 3. 智能分块策略

```
PDF内容
   ↓
[多模态提取]
   ├─ 文本 → [语义分块] → 与多模态内容拼接
   ├─ 图片 → [完整保留] → 独立chunk
   ├─ 表格 → [完整保留] → 独立chunk
   └─ 公式 → [完整保留] → 独立chunk
   ↓
[向量存储]
```

## 使用方法

### 配置

在AstrBot插件配置中添加：

```json
{
    "use_semantic_chunking": true,
    "enable_multimodal": true,
    "multimodal": {
        "enabled": true,
        "extract_images": true,
        "extract_tables": true,
        "extract_formulas": true
    }
}
```

### 命令

```bash
/paper add /path/to/papers  # 添加文档（自动提取多模态内容）
/paper search "attention mechanism"  # 检索
/paper list  # 查看文档列表
```

## 依赖说明

### 基础依赖（必需）

```bash
pip install -r requirements.txt
```

包含：pymilvus, PyMuPDF, pdfplumber, python-docx, pillow

### 可选依赖（用于图片向量化）

```bash
pip install transformers torch
```

**注意**：如果未安装transformers，系统会自动降级到文本模式，不会影响核心功能。

## 技术细节

### 图片/表格/公式为什么不分块？

1. **保留完整性**：图片、表格、公式是独立的内容单元，分块会破坏语义
2. **便于检索**：完整的表格/公式更容易精确匹配
3. **上下文保留**：可以与相邻的文本chunk拼接，提供上下文

### 向量化策略

```python
# 图片块（不分块）
{
    "text": "[Image: Figure 1: Architecture]",
    "metadata": {
        "chunk_type": "image",
        "page_number": 3,
        "caption": "Figure 1: Architecture"
    }
}

# 表格块（完整Markdown）
{
    "text": "| Column 1 | Column 2 |\n|----------|----------|\n| Value 1  | Value 2  |",
    "metadata": {
        "chunk_type": "table",
        "is_table": true
    },
    "table_data": "完整的Markdown表格"
}

# 公式块（完整LaTeX）
{
    "text": "$$ E = mc^2 $$",
    "metadata": {
        "chunk_type": "formula"
    },
    "formula_latex": "E = mc^2"
}
```

## 日志输出

### 正常模式
```
✅ 多模态提取器启用: 图片, 表格, 公式
✅ 提取完成:
   • 图片: 12
   • 表格: 5
   • 公式: 23
✅ 语义分块生成: 85 个片段
```

### 降级模式
```
⚠️  transformers 不可用
📝 回退到文本向量化模式
⚠️  图片提取被禁用
✅ 提取完成:
   • 表格: 5
   • 公式: 23
```

## 性能考虑

- **表格提取**：pdfplumber 对复杂表格可能较慢
- **内存占用**：图片数据会占用额外内存
- **存储空间**：完整的表格/公式会增加存储需求

## 故障排除

### 问题：表格提取失败

**原因**：pdfplumber未安装或PDF格式不支持

**解决**：
```bash
pip install pdfplumber
```

### 问题：transformers导入错误

**原因**：未安装多模态依赖（正常，不影响使用）

**解决**：无需处理，系统会自动降级

### 问题：chunk数量过多

**原因**：表格/公式完整保留，可能产生大量chunk

**解决**：调整配置，禁用不需要的功能
