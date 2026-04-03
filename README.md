# 📚 Paper RAG Plugin v1.9.1 - 用户指南

本地论文库RAG检索插件，为AstrBot提供智能的论文检索和问答能力（支持多模态VLM问答）。

> **版本说明**：当前版本 v1.9.1，完整更新历史见 [CHANGELOG.md](CHANGELOG.md)

## ✨ 核心功能

- 🔍 **混合检索**：BM25 关键词 + 向量语义双路召回 + RRF 分数融合，兼顾精确术语匹配与语义理解
- 💡 **AI问答**：结合检索内容生成准确、有引用的答案
- 📄 **多格式支持**：PDF、Word、TXT、Markdown、HTML
- 🖼️ **多模态提取**：自动识别PDF中的图片、表格、公式
- 🖼️ **多模态查询**：支持图片输入进行VLM问答（原图+VLM）
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
```

### 第二步：配置插件

**方式A：使用Ollama本地Embedding（推荐，免费无限制）**

在 **AstrBot WebUI → 插件 → paper_rag → 插件配置** 中：

| 配置项 | 值 | 说明 |
|-------|-----|------|
| Embedding模式 | `Ollama本地模式` | 使用Ollama |
| 向量嵌入维度 | `1024` | BGE-M3固定1024维 |
| Ollama模型名称 | `bge-m3` | 模型名称 |
| 文本问答Provider | （从AstrBot选取） | 用于RAG回答生成 |
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
| 文本问答Provider | （从AstrBot选取） | 用于RAG回答生成 |
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
| `/paper addf <文件路径>` | 添加单个文件（需管理员） | `/paper addf ./papers/attention.pdf` |
| `/paper delete <文件名>` | 删除指定论文（需管理员） | `/paper delete attention.pdf` |
| `/paper rebuild [目录] confirm` | 清空并重建知识库 | `/paper rebuild ./papers confirm` |
| `/paper clear confirm` | 清空知识库（需管理员） | `/paper clear confirm` |
| `/paper refstats [top_k]` | 查看参考文献引用统计（需管理员） | `/paper refstats 20` |
| `/paper refstats -1` | 列出无参考文献的论文 | `/paper refstats -1` |
| `/paper reparse_zero_ref confirm` | 批量重新解析无参考文献的论文（需管理员） | `/paper reparse_zero_ref confirm` |
| `/paper arxiv_add <关键词> [数量]` | 从arXiv搜索下载论文并添加（需管理员） | `/paper arxiv_add attention is all you need 3` |
| `/paper arxiv_refs [top_k] [每篇数量]` | 下载高频引用论文（需管理员） | `/paper arxiv_refs 10 3` |
| `/paper arxiv_sync confirm` | 同步MCP已下载论文到数据库（需管理员） | `/paper arxiv_sync confirm` |
| `/paper arxiv_cleanup confirm` | 清理arXiv论文旧版本（需管理员） | `/paper arxiv_cleanup confirm` |
| `/paper graph_build` | 构建知识图谱（需管理员） | `/paper graph_build` |
| `/paper graph_stats` | 查看图谱统计信息 | `/paper graph_stats` |
| `/paper graph_rebuild confirm` | 重建知识图谱（清空+重建） | `/paper graph_rebuild confirm` |
| `/paper graph_clear confirm` | 清空知识图谱（需管理员） | `/paper graph_clear confirm` |
| `/paper graph_backup [online\|offline]` | 备份图谱（需管理员） | `/paper graph_backup online` |
| `/paper graph_backup_list` | 列出可用备份 | `/paper graph_backup_list` |
| `/paper graph_restore [文件名]` | 恢复图谱备份（需管理员） | `/paper graph_restore neo4j_backup_xxx.json.gz` |
| `/paper graph_link [status\|create\|remove]` | 管理Neo4j符号链接 | `/paper graph_link status` |

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

## 🔬 arXiv 集成功能

插件提供多种方式获取论文并添加到数据库。

### arXiv MCP 同步

已配置的 arXiv MCP 服务器（`/Volumes/ext/arxiv`）下载的论文可以同步到 paperrag 数据库：

```
/paper arxiv_sync        # 查看待处理数量
/paper arxiv_sync confirm # 执行同步
```

### arXiv 论文清理

清理 arXiv 下载目录中的旧版本论文，只保留最新版本：

```
/paper arxiv_cleanup        # 查看待清理数量
/paper arxiv_cleanup confirm # 执行清理
```

**说明**：
- 自动识别同一论文的多个版本（如 `2603.11298.pdf` 和 `2603.11298v2.pdf`）
- 删除旧版本，保留最高版本号
- 同时清理 macOS 元数据文件（`._*`）

### 从 arXiv 搜索下载

使用 arXiv MCP 搜索论文并下载：

```
/paper arxiv_add <搜索关键词> [最大数量]
```

**示例**：
```
你: /paper arxiv_add attention is all you need 3
Bot: 🔍 在arXiv搜索: "attention is all you need"
Bot: 📡 正在搜索arXiv...
Bot: ✅ 找到 3 篇论文
Bot: 📄 [1/3] Attention Is All You Need
Bot:    📥 下载PDF: https://arxiv.org/pdf/1706.03762.pdf
Bot:    ✅ 下载完成 (8.2 MB)
Bot:    ✅ 已添加到数据库 (chunks: 45)
...
```

### 自动下载高频引用论文

根据已有文献的参考文献统计，自动下载被引用最多的论文：

```
/paper arxiv_refs [top_k] [每篇最大下载数]
```

**示例**：
```
你: /paper arxiv_refs 10 3
Bot: 📊 正在获取高频引用论文统计...
Bot: 📚 找到 156 种参考文献，取前 10 个高频引用
Bot: [1/10] 📝 Attention Is All You Need
Bot:    🔍 搜索: Attention Is All You Need Vaswani 2017
Bot:    📥 下载: 1706.03762v5.pdf
Bot:    ✅ 已添加到数据库
...
```

**工作流程**：
1. 调用 `/paper refstats` 获取高频引用论文列表
2. 对每个高频引用，使用标题+作者+年份构建搜索查询
3. 从 arXiv 下载相关论文
4. 自动添加到数据库
5. 跳过已存在的 PDF 文件

### 参考文献统计

查看数据库中论文的引用频次统计：

```
/paper refstats [top_k]
```

**示例输出**：
```
📚 **参考文献统计**

📊 统计概览:
   • 涉及论文种类: 156
   • 引用总条次: 892
   • 处理文档块: 234

🔝 **Top 20 高频引用论文**

 1. [ 15次] **Attention Is All You Need**
    └─ Vaswani, A. et al. (2017)
 2. [ 12次] **BERT: Pre-training of Deep Bidirectional**
    └─ Devlin, J. et al. (2018)
 3. [  8次] **Language Models are Few-Shot Learners**
    └─ Brown, T. et al. (2020)
...
```

---

## 📚 LLM 参考文献解析

插件支持使用 LLM（GPT-4o-mini）自动解析参考文献的标题、作者、年份等信息。

### 工作原理

1. **整段文本解析**：将参考文献部分（可能跨多行）作为整体发送给 LLM
2. **自动识别边界**：LLM 根据学术引用格式自动识别每条参考文献的边界
3. **结构化提取**：解析出标题、作者、年份、期刊、DOI 等字段
4. **双向关联建立**：将正文中的引用与参考文献关联

### 配置

LLM 参考文献解析使用 `evaluation/freeapi.json` 中的 API 配置：

```json
{
    "API_URL": "https://free.v36.cm",
    "API_KEY": "sk-..."
}
```

如需修改 API，请编辑 `evaluation/freeapi.json` 文件。

### 特性

| 特性 | 说明 |
|------|------|
| **自动边界识别** | 无需正则表达式启发式规则，LLM 自动识别跨行引用 |
| **并发控制** | 最多 4 个并发请求，避免 API 限流 |
| **自动重试** | HTTP 429/500 错误自动重试 |
| **表格过滤** | 自动检测并跳过表格，避免误解析 |
| **后备方案** | LLM 解析失败时自动降级到正则表达式解析 |

### 配置项

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `enable_llm_reference_parsing` | 启用 LLM 参考文献解析 | `true` |

### API 配置说明

API 配置从 `evaluation/freeapi.json` 读取，包含：
- `API_URL`: API 基础地址
- `API_KEY`: API 密钥

> 💡 如需使用其他 API 服务，修改 `evaluation/freeapi.json` 中的配置即可。

### MCP 参考文献补全（可选）

默认禁用 MCP（arXiv）参考文献 enrichment。如需启用，在 `reference_processor.py` 中取消注释：

```python
# MCP 参考文献补全（如需启用，取消注释以下代码）
# if self.arxiv_client and valid_results:
#     await self._enrich_from_arxiv(valid_results)
```

---

## ⚙️ 配置详解

### 基础配置

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `enabled` | 启用插件 | `true` | ✅ |
| `embedding_mode` | Embedding模式 | `ollama` | `ollama`（推荐免费）/ `api` |
| `embedding_provider_id` | Embedding Provider ID（API模式） | `gemini_embedding` | Gemini / OpenAI |
| `compress_provider_id` | 上下文压缩LLM | 空 | 从AstrBot提供商选取 |
| `text_provider_id` | 文本问答LLM | 空 | 从AstrBot提供商选取 |
| `multimodal_provider_id` | 多模态问答LLM | 空 | 从AstrBot提供商选取（用于图片问答） |
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
| `enable_bm25` | 启用 BM25 混合检索 | `false` | ✅ |
| `bm25_top_k` | BM25 召回数量 | `20` | `20-50` |
| `hybrid_alpha` | RRF 融合权重 | `0.5` | `0.5`（平等权重） |

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

> 💡 **VLM路由说明**：满足以下任一条件时，自动使用 `multimodal_provider_id` 配置的多模态模型进行回答：
> - 查询含视觉关键词（"图"、"表格"、"公式"、"架构"等）
> - 查询询问数量/比较/性能指标（"How many...", "Which is better...", "accuracy"等）
> - 检索文本内容提到 Figure/Table/Algorithm 等
> - 检索结果关联图片或图表 captions

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

### Llama.cpp 本地 VLM 配置（当 multimodal_provider_id 为空时使用）

当未配置 `multimodal_provider_id` 时，插件会自动使用本地 Llama.cpp VLM 进行图片问答。

**自动降级**：插件会优先使用 9B 模型，9B 模型不存在或加载失败时自动降级到 4B 模型。

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `llama_vlm_model_path` | GGUF 模型路径 | `./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf` | 9B/4B 均可 |
| `llama_vlm_mmproj_path` | mmproj 视觉编码器路径 | `./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf` | 与模型配套 |
| `llama_vlm_n_ctx` | 上下文窗口大小 | `4096` | `4096` |
| `llama_vlm_n_gpu_layers` | GPU 加速层数 | `99` | `99`（全部 GPU） |
| `llama_vlm_max_tokens` | 最大生成 token 数 | `2560` | `512-4096` |
| `llama_vlm_temperature` | 生成温度 | `0.7` | `0.7` |

> 💡 **Llama.cpp VLM 优势**：
> - 模型常驻内存，首次加载后推理快速（~1秒）
> - 支持多图输入
> - Apple Metal GPU 加速
> - 完全本地运行，无需 API
> - **自动降级**：9B 不可用时自动使用 4B

**安装步骤**：

1. 安装 llama-cpp-python（含多模态支持）：
```bash
# macOS Apple Silicon
CMAKE_ARGS="-DGGML_METAL=on -DLLAMA_MTMD=on" pip install llama-cpp-python

# NVIDIA GPU
# CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_MTMD=on" pip install llama-cpp-python
```

2. 模型下载（插件首次初始化时会自动下载，也可手动执行）：
```bash
# 9B 模型（约 5.6GB Q4 量化）
mkdir -p models
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir ./models/Qwen3.5-9B-GGUF
hf download unsloth/Qwen3.5-9B-GGUF mmproj-BF16.gguf --local-dir ./models/Qwen3.5-9B-GGUF

# 4B 模型（约 2.7GB Q4 量化，备用）
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf --local-dir ./models/Qwen3.5-4B-GGUF
hf download unsloth/Qwen3.5-4B-GGUF mmproj-BF16.gguf --local-dir ./models/Qwen3.5-4B-GGUF
```

3. 配置路径（基于插件目录）：
```
llama_vlm_model_path = ./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf
llama_vlm_mmproj_path = ./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf
```

4. 验证安装：
```bash
python -c "
from llama_cpp import Llama
llama = Llama('./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf', mmproj='./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf')
print('✅ Llama.cpp VLM 安装成功')
"
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

### 混合检索配置（BM25 + 向量）

BM25 混合检索通过 **双路召回 + 分数融合** 兼顾关键词精确匹配与语义理解：

| 配置项 | 说明 | 默认值 | 推荐值 |
|-------|------|--------|--------|
| `enable_bm25` | 是否启用混合检索 | `false` | ✅ |
| `bm25_top_k` | BM25 召回数量 | `20` | `20-50` |
| `hybrid_alpha` | RRF 向量权重（0=纯BM25，1=纯向量） | `0.5` | `0.5` |
| `hybrid_rrf_k` | RRF 常数 k | `60` | 默认 |

> **适用场景**：查询中包含明确关键词（论文标题、专业术语、作者名等）时，BM25 混合检索效果显著优于纯向量检索。

**检索流程**：
```
Query
  ├─ BM25 关键词搜索 ──→ 倒排索引命中 ──→ top_k 候选
  └─ 向量语义搜索 ──→ Milvus COSINE ──→ top_k 候选
                        ↓
              RRF 分数融合 ──→ 排序取 top_k
```

**典型配置**：

1. **默认（推荐）** — 关键词与语义兼顾
```json
{
  "enable_bm25": true,
  "bm25_top_k": 20,
  "hybrid_alpha": 0.5
}
```

2. **强关键词匹配** — 适合专有名词、技术术语查询
```json
{
  "enable_bm25": true,
  "bm25_top_k": 50,
  "hybrid_alpha": 0.3
}
```

3. **强语义理解** — 适合复杂问题、同义表述查询
```json
{
  "enable_bm25": true,
  "bm25_top_k": 20,
  "hybrid_alpha": 0.7
}
```

> ⚠️ **注意**：`bm25_top_k` 应大于最终 `top_k`，确保 RRF 融合时有足够候选；`enable_bm25` 与 `enable_reranking` 可同时开启，混合检索结果再经重排序二次优化。

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
2. 安装依赖：`pip install -r requirements.txt`
3. 运行测试：`python test_semantic_chunker.py paper.pdf`

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

**原因**：FlagEmbedding与transformers版本不兼容

**解决**：
```bash
# 确保 transformers 版本 < 5.0
pip install "transformers>=4.40.0,<5.0"
pip install -U FlagEmbedding
```

**已知冲突**：
- `mlx-lm`、`mlx-vlm` 与 `transformers<5.0` 冲突，如已安装请卸载
- transformers 5.x 移除了 `is_torch_fx_available`，导致 FlagEmbedding 导入失败

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

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 用户指南（本文档） |
| [CHANGELOG.md](CHANGELOG.md) | 变更记录 |
| [docs/INDEX.md](docs/INDEX.md) | 文档索引 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 技术架构说明 |

---

## 📈 Qasper 数据集评估

插件支持使用 Qasper 数据集评估 RAG 系统的性能。

### 免费 API 配置

RAGAS 评估流程需要使用 **GPT-4o-mini** 生成测试问题和评估答案。

**免费 API 获取方法**：
- 项目地址：[free_chatgpt_api](https://github.com/popjane/free_chatgpt_api)
- 提供免费的 GPT-4o-mini API

**配置步骤**：

1. **在 WebUI 中配置**（推荐）：
   - 插件配置 → `FreeAPI 服务地址`：填入如 `https://free.v36.cm`
   - 插件配置 → `FreeAPI 密钥`：填入从 free_chatgpt_api 获取的 API Key

2. **或手动编辑配置文件**：
   ```
   /Users/chenyifeng/AstrBot/data/config/astrbot_plugin_paperrag_config.json
   ```
   添加以下字段：
   ```json
   {
     "freeapi_url": "https://free.v36.cm",
     "freeapi_key": "你的API密钥"
   }
   ```

### Qasper 数据集说明

**Qasper 数据集不包含 PDF 文件**，只包含从论文提取的文本内容（full_text）。

### 评估流程

```
1. 下载数据集 (qasper_downloader.py)
       ↓
2. 索引论文到 Milvus (index_qasper.py)
       ↓
3. 生成 predictions (run_evaluation_qasper.py --generate)
       ↓
4. 运行评估 (run_evaluation_qasper.py --evaluate 或 --all)
```

### 快速开始

```bash
# 1. 下载数据集
cd datasets
python qasper_downloader.py

# 2. 索引论文到 Milvus
cd evaluation
python index_qasper.py --reinit

# 3. 生成 Predictions（支持断点续传）
python run_evaluation_qasper.py --generate

# 4. 运行评估
python run_evaluation_qasper.py --evaluate

# 或一步完成
python run_evaluation_qasper.py --all
```

### 命令行参数

**index_qasper.py**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--split` | 数据集划分 | `all` |
| `--reinit` | 重新初始化数据库 | False |

**run_evaluation_qasper.py**：
| 参数 | 说明 |
|------|------|
| `--generate` | 仅生成 predictions（支持断点续传） |
| `--evaluate` | 仅运行评估 |
| `--all` | 生成 + 评估 |
| `--no_resume` | 禁用断点续传，重新生成所有预测 |
| `--llm_only` | 纯LLM基线模式：不进行检索，直接使用LLM回答（用于基线对比） |
| `--bert_score` | 使用BERTScore F1进行语义评估（更适合QASPER长文档自由形式答案） |
| `--limit N` | 限制处理的问题数量（默认0不限制，用于快速测试） |

### 评估指标

- **Answer F1**: 答案 F1 分数（基于词汇重叠）
- **Answer BERT F1**: 答案 BERTScore F1（基于语义相似度，更适合QASPER长文档）
- **Answer F1 by type**: 按答案类型 (extractive/abstractive/boolean/none) 的 F1
- **Evidence F1**: 证据 F1 分数
- **Missing predictions**: 缺失预测数量

> 💡 **为什么需要 BERTScore F1？**
> QASPER 数据集平均上下文长度超过 23,000 字符，自由形式答案导致相同语义可用不同表达。
> 研究表明：Cosine F1 = 0.22（过于严格），BERTScore F1 = 0.62（更合理）。
> 布尔问题（No/False, Yes/True）语义等价但词汇不同，BERTScore 更公平。

### 输出文件

```
evaluation_output/
├── predictions.jsonl       # 预测结果
└── evaluation_results.json # 评估指标
```

### 详细文档

详见 [evaluation/README_qasper.md](evaluation/README_qasper.md)

---

## 📋 待实现功能 (ToDo)

以下功能正在规划中，将在未来的版本中逐步实现。

### ✅ Graph RAG 模块

将现有向量检索升级为**图增强 RAG**，支持多跳推理和关系查询。

**已实现功能**：
- [x] Graph RAG 引擎（`graph_rag_engine.py`）
- [x] 知识图谱检索器与融合检索器（`graph_retriever.py`）
- [x] 图谱构建器 - LLM 三元组抽取（`graph_builder.py`）
- [x] 用户意图识别与智能路由（`graph_rag_router.py`）
- [x] Memory 图谱存储（默认）
- [x] Neo4j 图数据库支持（可选）
- [x] 混合检索模式（向量 + 图谱 RRF 融合）
- [x] 关系查询引擎（支持"A 和 B 的关系"类问题）
- [x] 多跳推理增强
- [x] 手动/自动图谱构建
- [x] 新增命令：`/paper graph_build`、`/paper graph_stats`、`/paper graph_rebuild`、`/paper graph_clear`、`/paper graph_backup`、`/paper graph_restore`、`/paper graph_backup_list`、`/paper graph_link`

**技术方案**：基于 LlamaIndex PropertyGraphIndex 实现

**版本**：v1.7.3

### 🛠️ Graph RAG 修复 (v1.7.1)

**问题**：Qwen3.5 GGUF 模型在构建知识图谱时输出 thinking tokens 而非 JSON，导致解析失败。

**修复内容**：
- [x] 集成 `LlamaCppVLMProvider` 替代原有的 `LocalLLMProvider`
- [x] 修复 `system_prompt` 未被正确使用的问题
- [x] 增加 `max_tokens` 至 4096 避免 JSON 截断
- [x] 添加 `_strip_thinking_tokens()` 移除 `<think>...</think>` 块
- [x] 实现批量处理（batch_size=4）减少 LLM 调用次数
- [x] 修复 `add_relation()` 参数错误（使用 head/tail 字符串而非 ID）
- [x] Pure Text RAG 优先使用本地 LlamaCpp 模型

**批量处理优化**：
- 原调用次数：~1800 次（1800 chunks）
- 优化后调用次数：~450 次（batch_size=4）
- 预估 token 计算：4 chunks × ~500字符 + system prompt ≈ 2000-2300 tokens < 4096

**模型自动降级**：
- 优先使用 Qwen3.5-9B 模型
- 9B 不可用时自动降级到 Qwen3.5-4B 模型

---

### 🔲 创意生成引擎

融合本地知识库 + 网络搜索 + 创意生成，构建智能研究助手。

**计划功能**：
- [ ] 智能搜索规划（LLM 驱动的多源搜索查询生成）
- [ ] 网络信息增强（Semantic Scholar / Tavily / GitHub API）
- [ ] 多源知识融合（本地论文 + 网络结果统一上下文）
- [ ] 研究提案生成（基于结构化 prompt 的 idea 生成）
- [ ] 引文追踪与格式化

**技术方案**：基于 LangGraph 工作流编排

**预计版本**：v1.10

---

### 🔲 模型微调与强化学习（计划 v2.0+）

**长期规划**：

- [ ] RAG 系统评估数据收集与标注
- [ ] Reward Model 训练（基于人类反馈的奖励模型）
- [ ] RLHF 微调（使用 PPO/DPO 对检索/生成模型进行强化学习）
- [ ] 个性化适应（根据用户反馈持续优化）

**说明**：v2.0 之后实现，需要大量标注数据和训练资源。

---

> 💡 **欢迎贡献**：如果您对某个功能感兴趣，欢迎提交 PR 或参与讨论！

---

## 📞 获取帮助

- **问题反馈**：通过 GitHub Issues 提交问题
- **日志查看**：AstrBot 控制台输出

---

## 📄 许可证

MIT License

## 🙏 致谢

- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 聊天机器人框架
- [Milvus](https://milvus.io/) - 向量数据库
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF解析
