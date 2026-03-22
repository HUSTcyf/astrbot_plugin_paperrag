# Paper RAG 插件 - 测试指南

本文档提供完整的测试流程和命令，帮助您验证插件功能。

## 🚀 快速测试

### 一键测试

```bash
cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag

# 运行完整测试
python test_pdf.py /path/to/your/paper.pdf
```

## 📋 详细测试流程

### 步骤1: 安装依赖

```bash
# 进入插件目录
cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag

# 安装基础依赖
pip install -r requirements.txt
```

**基础依赖包括**：
- pymilvus[milvus_lite] - 向量数据库
- PyMuPDF - PDF解析
- pdfplumber - 表格提取
- python-docx - Word文档
- pillow - 图像处理

**可选依赖**（用于图片向量化）：
```bash
pip install transformers torch
```

### 步骤2: 准备测试PDF

将测试PDF文件放到可访问的位置：

```bash
# 示例路径
~/Documents/test_paper.pdf
~/Downloads/paper.pdf
```

### 步骤3: 运行测试

```bash
# 完整测试（推荐）
python test_pdf.py ~/Documents/test_paper.pdf
```

## 📊 测试输出说明

### 测试1: 依赖检查

**预期输出**：
```
==================================================================
  📦 测试1: 依赖检查
==================================================================

🔍 核心依赖:
  ✅ pymilvus        - Milvus 向量数据库
  ✅ fitz            - PyMuPDF (PDF解析)
  ✅ pdfplumber      - 表格提取
  ✅ docx            - Word文档支持
  ✅ PIL             - Pillow (图像处理)

🔍 可选依赖 (多模态):
  ❌ transformers    - HuggingFace (视觉编码)
  ❌ torch           - PyTorch (深度学习)

✅ 所有核心依赖已安装
⚠️  缺少可选依赖（多模态功能将自动降级）:
   pip install transformers torch
```

**诊断**：
- ✅ 所有核心依赖通过 → 继续测试
- ❌ 缺少核心依赖 → 安装：`pip install pymilvus PyMuPDF pdfplumber python-docx pillow`

### 测试2: PDF基础提取

**预期输出**：
```
==================================================================
  📄 测试2: PDF基础提取
==================================================================

📁 文件: example.pdf
📏 大小: 1234.5 KB
📄 总页数: 16

  页 1: 3520 字符
    预览: Abstract This paper presents a novel approach...
  页 2: 4120 字符
    预览: 1. Introduction Recent advances in deep learning...
  页 3: 2890 字符
    预览: 2. Related Work Our approach builds upon...

📊 提取统计:
  • 有文本的页数: 16/16
  • 总字符数: 58940
  • 总图片数: 12
  • 文本密度: 3683.8 字符/页

✅ PDF包含可提取的文本
```

**诊断**：
- ✅ 文本提取成功 → 继续测试
- ❌ 无文本（扫描版）→ 需要OCR转换
- ⚠️ 文本很少 → 可能不适合RAG

### 测试3: 多模态提取

**预期输出**：
```
==================================================================
  🎨 测试3: 多模态提取
==================================================================

🔍 初始化多模态提取器...
✅ 提取器可用: True
   图片提取: ✅
   表格提取: ✅
   公式提取: ✅

📖 开始提取...

✅ 提取完成!

📊 提取结果:
  • 图片数量: 12
  • 表格数量: 5
  • 公式数量: 23
  • 文本长度: 58940 字符

🖼️  提取的图片（前3个）:
  [1] 页 3, 大小: 800x600
      图注: Figure 1: Architecture overview
  [2] 页 7, 大小: 1024x768
      图注: Figure 2: Performance comparison
  [3] 页 12, 大小: 640x480

📋 提取的表格（前2个）:
  [1] 页 5, 4行 x 3列
      表注: Table 1: Model configurations
      预览: | Model | Layers | Params |...
  [2] 页 9, 6行 x 4列
      表注: Table 2: Experimental results
      预览: | Method | Accuracy | Precision |...
```

**诊断**：
- ✅ 多模态提取成功 → 可以看到图片、表格、公式的统计
- ⚠️ 部分功能禁用 → 依赖不可用，但核心功能正常
- ❌ 提取失败 → 查看错误信息

### 测试4: 语义分块

**预期输出**：
```
==================================================================
  🧩 测试4: 语义分块
==================================================================

🔍 初始化解析器和分块器...
✅ 解析器和分块器初始化成功

📖 开始解析和分块...

✅ 分块完成!

📊 分块统计:
  • 总块数: 85

🏷️  块类型分布:
  • paragraph: 65
  • title: 12
  • table: 5
  • formula: 3

📝 分块示例（前3个）:
  [1] 类型: title, 长度: 45 字符
      内容: Abstract This paper presents a novel approach...
  [2] 类型: paragraph, 长度: 512 字符
      内容: 1. Introduction Recent advances in deep learning have...
  [3] 类型: paragraph, 长度: 498 字符
      内容: In this work, we propose a new method for...
```

**诊断**：
- ✅ 分块成功 → 显示块类型分布和示例
- ⚠️ 块数量过多（>50）→ 可能影响性能，但可用
- ❌ 分块失败 → 查看错误信息

### 测试5: 向量化功能

**预期输出**：
```
==================================================================
  🔢 测试5: 向量化功能
==================================================================

⚠️  注意: 此测试需要配置 AstrBot Embedding Provider
💡 如果未配置，将跳过此测试

✅ AstrBot API 可用

💡 向量化测试需要在 AstrBot 环境中运行
   使用 /paper add 命令进行完整测试
```

## 🎯 完整流程测试（在AstrBot中）

### 准备工作

1. **启动AstrBot**
   ```bash
   cd /path/to/AstrBot
   python main.py
   ```

2. **配置Embedding Provider**

   在 WebUI 中配置：
   - 类型: `Gemini`
   - ID: `gemini_embedding`
   - API Key: 你的API密钥
   - 模型: `gemini-embedding-2-preview`

### 测试命令

```bash
# 1. 添加文档
/paper add ~/Documents/papers

# 预期输出：
# 📄 Found 10 document files
# ⏳ Starting import...
# 📖 [1/10] Parsing: paper1.pdf
# ✅ [1/10] paper1.pdf - 85 chunks (including 3 tables, 2 formulas)
# ...

# 2. 列出文档
/paper list

# 预期输出：
# 📚 **Document Library**
#
# 1. ✅ **paper1.pdf**
#    └─ Chunks: 85
#    └─ Added: 2026-03-22 12:00:00
# ...

# 3. 搜索文档
/paper search What is the attention mechanism?

# 预期输出：
# 💡 **Answer**
#
# The attention mechanism is a neural network architecture...
#
# 📚 **References**
#
# [1] **attention.pdf** (chunk #5)
# > The attention mechanism computes weighted sums...
```

## 🔧 故障排除

### 问题1: ImportError: No module named 'fitz'

**解决**：
```bash
pip install PyMuPDF
```

### 问题2: ImportError: No module named 'pdfplumber'

**解决**：
```bash
pip install pdfplumber
```

### 问题3: 分块数量为0

**可能原因**：
- PDF是扫描版（无文本层）
- PDF文件损坏

**解决**：
```bash
# 使用独立测试工具诊断
python test_pdf.py /path/to/pdf

# 查看详细错误信息
```

### 问题4: transformers导入错误

**原因**：未安装多模态依赖（正常）

**解决**：
- 无需处理，系统会自动降级到文本模式
- 如需图片向量化，安装：`pip install transformers torch`

### 问题5: AstrBot中无法添加文档

**检查项**：
1. Embedding Provider是否已配置
2. PDF文件路径是否正确
3. 文件权限是否正确

**解决**：
```bash
# 检查文件权限
ls -l ~/Documents/papers

# 检查AstrBot日志
tail -f /path/to/astrbot/logs/astrbot.log
```

## 📊 测试结果解读

### 完美通过

```
🎉 所有测试通过! (5/5)
```
→ 插件功能完整，可以正常使用

### 部分通过

```
⚠️  部分测试失败 (4/5)
```
→ 核心功能可用，部分功能降级
→ 查看失败的测试，确认是否影响使用

### 全部失败

```
❌ 部分测试失败 (2/5)
```
→ 需要修复依赖或配置问题
→ 查看错误信息，按照故障排除步骤修复

## 🎓 测试最佳实践

1. **先测试依赖**：确保所有核心依赖已安装
2. **使用简单PDF**：首次测试使用简单的文本PDF
3. **逐步测试**：从基础提取到多模态，逐步验证
4. **检查日志**：失败时查看详细错误信息
5. **降级模式**：即使transformers不可用，核心功能仍可用

## 📞 获取帮助

如遇到问题：
1. 查看本文档的故障排除部分
2. 查看 README.md 了解功能说明
3. 查看 DEVELOPMENT.md 了解技术细节
4. 运行 `python test_pdf.py <file>` 查看详细诊断
