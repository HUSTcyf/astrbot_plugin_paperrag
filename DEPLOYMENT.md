# 🚀 Paper RAG Plugin 部署指南

完整的部署步骤和配置说明。

## 📋 部署前检查清单

在开始部署前，请确保您的环境满足以下要求：

- [ ] Python 3.10+ 已安装
- [ ] AstrBot 正常运行
- [ ] 已准备 Embedding Provider（推荐 Gemini）
- [ ] 已准备 LLM Provider（可选，用于RAG生成）
- [ ] 至少 16GB 可用内存
- [ ] 至少 500MB 可用磁盘空间

## 🔧 第一步：安装插件依赖

### 1.1 进入插件目录

```bash
cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag
```

### 1.2 安装Python依赖

**方法1：使用 pip**
```bash
pip install -r requirements.txt
```

**方法2：使用 uv（推荐）**
```bash
uv pip install -r requirements.txt
```

### 1.3 验证安装

运行以下命令验证关键包已正确安装：

```bash
python -c "from pymilvus import MilvusClient; print('✅ Milvus OK')"
python -c "import fitz; print('✅ PyMuPDF OK')"
```

预期输出：
```
✅ Milvus OK
✅ PyMuPDF OK
```

## 🔑 第二步：配置 Embedding Provider

### 2.1 获取 Gemini API Key

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 点击 "Create API Key"
3. 复制生成的 API Key（格式：`AIzaSy...`）
4. 保存密钥，稍后配置使用

### 2.2 在 AstrBot 中配置 Embedding Provider

1. 打开 AstrBot WebUI（通常是 `http://localhost:6185`）
2. 进入 **设置 → 模型提供商**
3. 点击 **添加 Provider**，选择 **Gemini**
4. 填写配置：
   - **ID**: `gemini_embedding`
   - **API Key**: 你的 Gemini API Key
   - **模型**: `gemini-embedding-2-preview`
5. 点击 **保存**

**其他 Embedding Provider 选项**：
- OpenAI: `text-embedding-3-small` (1536维)
- 本地模型: Ollama 等

## 🔑 第三步：配置 LLM Provider（可选）

如果需要使用 RAG 生成功能，配置 LLM Provider：

### 3.1 获取智谱 API Key

1. 访问 [智谱AI开放平台](https://open.bigmodel.cn/usercenter/apikeys)
2. 注册/登录账号
3. 点击 "创建API Key"
4. 复制生成的 API Key
5. 确认使用模型：`glm-4.7-flash`（免费版，1 QPS）

### 3.2 在 AstrBot 中配置 LLM Provider

1. 在 AstrBot WebUI 中，进入 **设置 → 模型提供商**
2. 点击 **添加 Provider**，选择 **zhipu**
3. 填写配置：
   - **ID**: `glm-4.7-flash`
   - **API Key**: 你的智谱 API Key
   - **模型**: `glm-4.7-flash`
4. 点击 **保存**

**其他 LLM Provider 选项**：
- OpenAI: `gpt-4o-mini`, `gpt-4o`
- Anthropic: `claude-3-haiku`
- 本地模型: Ollama 等

## ⚙️ 第四步：配置插件

### 4.1 打开插件配置

1. 在 AstrBot WebUI 中，进入 **插件** 页面
2. 找到 **paper_rag** 插件
3. 点击 **插件配置**

### 4.2 填写配置

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| **Embedding 服务提供商** | 选择 Embedding Provider | 点击按钮选择 `gemini_embedding` |
| **LLM Provider ID** | LLM Provider 的 ID | `glm-4.7-flash`（可留空） |
| **Milvus Lite数据库路径** | 向量数据库路径 | `./data/milvus_papers.db` |
| **向量嵌入维度** | 嵌入向量维度 | `768`（与 Gemini 匹配） |
| **检索返回的文档片段数量** | 每次检索返回的片段数 | `5` |
| **相似度阈值** | 相似度过滤阈值（0-1） | `0.3` |
| **论文文件存放目录** | PDF/Word等文档目录 | `./papers` |
| **启用插件** | 控制插件是否工作 | ✅ |
| **启用缓存** | 是否启用查询缓存 | ✅ |
| **缓存生存时间** | 缓存有效时间（秒） | `3600` |
| **最大缓存条目** | 最多缓存条数 | `100` |

### 4.3 保存配置

点击 **保存** 按钮，AstrBot 可能会提示重启。

### 4.4 创建论文目录

```bash
# 在插件目录下创建论文目录
mkdir -p papers
```

## 📥 第五步：导入论文

### 5.1 准备论文文件

将PDF论文文件复制到 `papers` 目录：

```bash
# 示例：复制单个论文
cp ~/Downloads/paper1.pdf papers/

# 或批量复制
cp ~/Downloads/*.pdf papers/
```

支持的文件格式：
- PDF (.pdf)
- Word (.docx, .doc)
- 纯文本 (.txt, .md)
- HTML (.html, .htm)

### 5.2 导入论文

在聊天中发送命令：

```
/paper add
```

插件会自动：
1. 扫描 `papers` 目录
2. 解析文档文件
3. 生成嵌入向量
4. 存储到向量数据库

预期输出：
```
🔍 扫描目录: ./papers
📄 发现 X 个文档文件
⏳ 开始导入...
✅ 导入: paper1.pdf (23 个片段)
✅ 导入: paper2.pdf (18 个片段)
...

✅ **导入完成**

📊 统计信息:
  • 文件数: X
  • 片段数: XX
  • 耗时: XX.X 秒

💡 提示: 使用 /paper search [问题] 来检索文档
```

## 🧪 第六步：测试功能

### 6.1 测试论文列表

```
/paper list
```

预期输出：
```
📚 **文档库列表**

1. ✅ **paper1.pdf**
   └─ 片段数: 23
   └─ 添加时间: 2026-03-21T10:30:00

2. ✅ **paper2.pdf**
   └─ 片段数: 18
   └─ 添加时间: 2026-03-21T10:31:00

📊 总计: 2 个文档
```

### 6.2 测试论文搜索

**RAG模式（生成回答）**：
```
/paper search 这篇论文的主要贡献是什么
```

预期输出：
```
🔍 正在检索文档库...
问题: 这篇论文的主要贡献是什么

💡 **回答**

根据检索到的文档内容，本文的主要贡献是...

📚 **参考文献**

[1] **paper1.pdf** (片段 #5)
> 我们提出了一个新的方法...

[2] **paper2.pdf** (片段 #12)
> 相关研究表明...
```

**检索模式（仅返回片段）**：
```
/paper search 卷积神经网络 mode=retrieve
```

预期输出：
```
📚 **文档检索结果**

[1] **paper1.pdf** (相似度: 0.856)
卷积神经网络（CNN）是一种专门用于处理网格状数据...

[2] **paper2.pdf** (相似度: 0.743)
CNN通过卷积层自动提取特征...
```

### 6.3 测试缓存功能

重复相同的查询，第二次应该更快（使用缓存）。

## 🐛 第七步：故障排除

### 问题1: 插件未加载

**症状**: 在AstrBot WebUI中看不到插件

**解决方案**:
1. 检查插件目录是否在正确位置：`/Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag`
2. 查看 AstrBot 日志获取详细错误信息
3. 确认所有依赖已安装：`pip list | grep -E "pymilvus|PyMuPDF"`
4. 验证 `_conf_schema.json` 格式正确

### 问题2: Embedding Provider 加载失败

**症状**: 搜索时提示 "❌ RAG引擎未就绪"

**解决方案**:
1. 在 AstrBot WebUI → 设置 → 模型提供商 中确认 Embedding Provider 已添加
2. 检查插件配置中的 Provider ID 是否与设置中的 ID 一致（如 `gemini_embedding`）
3. 测试 Provider 是否正常工作
4. 查看详细错误日志

### 问题3: 向量维度不匹配

**症状**: 提示"嵌入维度必须是64的倍数"或向量操作失败

**解决方案**:
- Gemini Embedding 2 Preview: 768维
- OpenAI text-embedding-3-small: 1536维
- 确认插件配置中的 `embed_dim` 与 Provider 模型匹配

### 问题4: 文档解析失败

**症状**: 导入时显示 "❌ PDF解析失败" 或 "❌ Word文档解析失败"

**解决方案**:
1. 确认文档文件未损坏
2. 尝试用对应的应用程序打开文件
3. 检查文件权限
4. 安装额外依赖：`pip install python-docx unstructured`

### 问题5: 内存不足

**症状**: 系统变慢或崩溃

**解决方案**:
1. 减少同时导入的文档数量
2. 调整 `embed_dim` 参数（降低维度）
3. 减小 `top_k` 值
4. 关闭其他应用程序

## 📊 第八步：性能优化

### 8.1 调整嵌入维度

编辑插件配置，降低嵌入维度可以减少存储：

```python
# 在插件配置中
"embed_dim": 512  # 可选：256, 512, 768, 1024
```

**注意**: 维度必须与 Embedding Provider 的模型匹配。

### 8.2 调整检索参数

```python
# 在插件配置中
"top_k": 3,                      # 减少返回结果
"similarity_cutoff": 0.5         # 提高相似度阈值
```

### 8.3 禁用 LLM 生成

如果不需要 AI 生成回答，只使用检索功能：

```python
# 在插件配置中
"llm_provider_id": ""  # 留空
```

### 8.4 调整缓存设置

```python
# 在插件配置中
"cache_enabled": true,
"cache_ttl_seconds": 7200,      # 延长缓存时间
"cache_max_entries": 200         # 增加缓存条目
```

## ✅ 部署完成检查清单

- [ ] 所有依赖已安装
- [ ] Embedding Provider 已配置并测试
- [ ] LLM Provider 已配置（如需要）
- [ ] 插件已在 AstrBot 中配置
- [ ] 论文目录已创建
- [ ] 至少导入了一篇论文
- [ ] 搜索功能正常工作
- [ ] 列表功能正常工作

## 🎉 开始使用

现在您可以：

1. **搜索文档**: 使用 `/paper search [问题]` 检索相关文档
2. **管理文档**: 使用 `/paper list` 和 `/paper add` 管理文档库
3. **调整参数**: 根据需要修改插件配置优化性能

### 常用命令速查

| 命令 | 功能 |
|------|------|
| `/paper search [问题]` | 搜索并生成回答 |
| `/paper search [问题] mode=retrieve` | 仅检索相关片段 |
| `/paper list` | 列出已收录的文档 |
| `/paper add` | 添加文档（管理员） |
| `/paper add [目录]` | 从指定目录添加（管理员） |
| `/paper clear 确认` | 清空知识库（管理员） |

## 📞 获取帮助

- 查看日志：AstrBot 控制台输出
- 查看文档：[README.md](README.md)
- 开发文档：[DEVELOPMENT.md](DEVELOPMENT.md)
- 提交问题：GitHub Issues

## 🔗 相关资源

- [AstrBot GitHub](https://github.com/AstrBotDevs/AstrBot)
- [Milvus Lite 文档](https://milvus.io/docs/milvus_lite.md)
- [Gemini Embedding API](https://ai.google.dev/gemini-api/docs/embeddings)
- [智谱 GLM API](https://open.bigmodel.cn/dev/api)
