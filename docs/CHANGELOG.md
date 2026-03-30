# 变更记录

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.2] - 2026-03-30

### Added

- **断点续传功能** (`run_evaluation_qasper.py`)
  - 新增 `--resume` 参数（默认启用），跳过已有有效答案的问题
  - 新增 `--no_resume` 参数，禁用断点续传，重新生成所有预测
  - 新增 `load_existing_predictions()` 函数，从 JSONL 文件加载已有预测
  - 新增 `backup_predictions()` 函数，生成前自动备份原始文件
  - 新增 `save_predictions()` 函数，定期保存防止中断丢失数据
  - 处理网络错误导致部分问题未生成回答的场景

### Changed

- **FreeAPI 配置迁移**
  - `evaluation/freeapi.json` 配置文件已迁移到插件配置 `_conf_schema.json`
  - 新增 `freeapi_url` 和 `freeapi_key` 配置项（通过 WebUI 配置）
  - 更新 `hybrid_rag.py`、`run_evaluation_qasper.py`、`run_evaluation_ragas.py` 读取配置逻辑
  - README 新增免费 API 获取说明

## [1.6.1] - 2026-03-30

### Changed

- **CORE API 替代 arXiv MCP**
  - 新增 `CoreAPIClient` 类实现 CORE API v3 搜索功能
  - `/paper arxiv_add` 和 `/paper arxiv_refs` 命令改用 CORE API
  - 优先使用 arXiv PDF 链接下载，否则使用 `sourceFulltextUrls[0]`
  - 无需配置 arXiv MCP 服务器

- **参考文献提取修复**
  - 修复 "Authors Suppressed Due to Excessive Length" 导致截断过早的问题
  - 移除 `end_keywords` 中的 'author' 匹配，改用精确的 `section_headers` 列表
  - 遇到 Markdown 表格分隔行 `| --- | --- |` 时直接截断
  - 简化 `_find_reference_section()` 函数逻辑：找到最后一个编号行后截断

### Added

- **CORE API Key 配置项**
  - 新增 `core_api_key` 配置项（从 https://core.ac.uk/services/api 获取）
  - 存储在插件配置中

## [1.6.0] - 2026-03-29

### Changed

- **LLM参考文献解析架构重构**
  - 改用整段参考文献文本传给LLM，让LLM自动识别和解析每条参考文献
  - 舍弃复杂的正则表达式启发式规则（年份检测、长度阈值等）
  - 避免因参考文献格式差异（跨行、无序号等）导致的分割错误

### Added

- **GPT-4o-mini参考文献解析**
  - 使用官方OpenAI API直接调用GPT-4o-mini（不使用ragas依赖）
  - 新增 `parse_reference_section()` 方法，支持整段文本解析
  - API配置从 `evaluation/freeapi.json` 读取

- **并发控制与重试机制**
  - Semaphore-based并发控制（最多4个并发请求）
  - HTTP 429（限流）自动重试，支持 Retry-After 解析
  - HTTP 500（服务器错误）自动重试

- **表格检测增强**
  - 新增 `_is_likely_table()` 方法
  - 通过管道符数量、时间格式、表格指示符等特征识别表格
  - 避免将表格误判为参考文献

- **调试日志增强**
  - 新增 `📝 [LLM调用]`、`📝 [LLM调试]` 日志
  - 记录Prompt长度、响应内容、JSON提取结果

### Removed

- **简化参考文献分割逻辑**
  - `_split_reference_lines()` 移除冗余启发式规则
  - 只保留明确的编号格式识别（`[1]`、`1.`、`\bibitem{}`）和URL检测
  - 无序号参考文献的解析全部由LLM处理

- **MCP参考文献 enrichment 默认禁用**
  - 取消注释即可启用arXiv MCP补全

## [1.5.0] - 2026-03-28

### Added

- **参考文献统计功能** (`refstats`)
  - 新增 `get_all_references()` 方法从 Milvus 数据库提取参考文献
  - 统计每篇论文标题的出现频次
  - 按频次排序显示高频引用论文

- **arXiv 论文下载功能** (`arxiv_add`)
  - 使用配置的 arXiv MCP 搜索论文
  - 从 arXiv 直接下载 PDF（`https://arxiv.org/pdf/{paper_id}.pdf`）
  - 自动添加到数据库

- **高频引用论文自动下载** (`arxiv_refs`)
  - 根据 `refstats` 统计自动识别高频引用论文
  - 使用标题+作者+年份构建搜索查询
  - 自动从 arXiv 下载并添加到数据库

- **MCP 论文同步功能** (`arxiv_sync`)
  - 将 arXiv MCP 已下载的论文同步到 paperrag 数据库
  - 扫描 MCP 存储路径（默认 `/Volumes/ext/arxiv`）
  - 复制到 `papers_dir` 并添加到数据库

- **arXiv 论文版本清理** (`arxiv_cleanup`)
  - 自动识别同一论文的多个版本（如 `2603.11298.pdf` 和 `2603.11298v2.pdf`）
  - 删除旧版本，只保留最高版本号
  - 同时清理 macOS 元数据文件（`._*`）

## [1.4.0] - 2026-03-27

### Added

- **Llama.cpp 本地 VLM Provider**
  - 新增 `llama_cpp_vlm_provider.py` 模块
  - 使用 llama-cpp-python 实现本地视觉语言模型推理
  - Apple Metal GPU 加速支持
  - 模型常驻内存，首次加载后推理快速（~1秒）

- **自动模型下载**
  - 初始化时自动检查模型文件是否存在
  - 不存在时自动从 HuggingFace 下载
  - 支持 9B 和 4B 双模型配置

- **9B → 4B 自动降级**
  - 优先使用 Qwen3.5-9B 模型
  - 9B 模型不存在或加载失败时自动降级到 4B 模型
  - 路径：`./models/Qwen3.5-9B-GGUF/` 和 `./models/Qwen3.5-4B-GGUF/`

- **响应长度扩展**
  - `llama_vlm_max_tokens` 从 512 扩展到 2560（5倍）
  - 支持更长的多模态回答

### Changed

- **异步架构优化**
  - 使用 `asyncio.run_in_executor` 避免阻塞事件循环
  - 解决同步阻塞和冷启动问题

- **配置默认值更新**
  - `llama_vlm_model_path`: `./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf`
  - `llama_vlm_mmproj_path`: `./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf`
  - `llama_vlm_max_tokens`: `2560`

### Removed

- **删除 HF_MLX 相关逻辑**
  - 移除 hf_mlx_provider.py 中的 HF_MLX 相关代码
  - 不再依赖 hf_mlx 包

- **删除图片数量限制**
  - 移除所有图片数量限制逻辑
  - 支持任意数量的图片输入

## [1.3.1] - 2026-03-26

### Added

- **删除指定论文功能**
  - 新增 `/paper delete <filename>` 命令
  - 支持根据文件名删除对应的向量数据
  - 需要管理员权限

### Fixed

- **VLM模式修复**
  - 修复多模态Provider不支持时的错误处理，现在会正确fallback到文本模式
  - 修复图片传递方式，优先使用本地文件路径，失败后降级到base64
  - 修复LLMResponse内容提取逻辑，正确解析AstrBot的响应格式

## [1.3.0] - 2026-03-25

### Added

- **参考文献检索功能**
  - 新增 `reference_processor.py` 模块
  - 支持从PDF中提取结构化参考文献（标题、作者、年份、DOI、期刊）
  - 支持多种引用格式（[1], 1. 等）
  - 建立正文章节与参考文献的双向关联
  - 支持 Grobid 解析（可选）

- **多模态VLM检索**
  - 新增 VLM 路由逻辑，自动判断是否使用多模态模型
  - 支持查询含视觉关键词（"图"、"表格"、"公式"等）
  - 支持图片输入进行VLM问答
  - 自动关联检索结果中的图片

- **Ollama本地Embedding支持**
  - 新增 `embedding_providers.py` 模块
  - 支持 Ollama 本地向量化（推荐BGE-M3模型）
  - 支持AstrBot Embedding Provider
  - LLM文本压缩功能（应对长文本超过8192限制）

- **语义分块优化**
  - 优先级：段落 > 句子 > 子句
  - 支持 `chunk_overlap` 保持语义连贯
  - 修复句子中间断开问题（`_clean_overlap`）

- **图片处理增强**
  - 图片合并提取（计算外接矩形合并子图）
  - 跳过图注去重，保留完整图表
  - 基于页码 proximity 关联图片到文本块

- **命令新增**
  - `/paper addf <file_path>` - 添加单个文件
  - `/paper rebuild <目录> confirm` - 清空并重建知识库
  - `/paper clear confirm` - 清空知识库

### Changed

- **架构演进**
  - v1.x: llama-index 管理索引和检索
  - v1.3: Hybrid架构（HybridPDFParser + HybridIndexManager + HybridRAGEngine）
  - 直接使用 pymilvus，避免与llama-index冲突

- **默认配置变更**
  - `embedding_mode`: `api` → `ollama`
  - `glm_model`: `glm-4.6v-flash` → `glm-4.7-flash`（文本模型）
  - `glm_multimodal_model`: 新增，默认 `glm-4.6v-flash`
  - `multimodal.enabled`: `false` → `true`

- **PDF解析重构**
  - 新增 `hybrid_parser.py`：保留多模态提取 + 语义分块
  - 图片存储方案：原图保存到磁盘，metadata存储image_path

### Fixed

- 修复 metadata JSON序列化问题（处理 `Rect` 等特殊类型）
- 修复连接复用问题（断开旧连接再创建新连接）
- 修复 Milvus 路径问题（使用插件目录下的默认路径）
- 修复 `list_documents` 显示每个文件单独统计
- 修复参考文献部分标题误识别（`here for a reference.` → `References`）
- 修复无序号参考文献的分割问题

---

## [1.2.0] - 2026-03-25

### Added

- Ollama Embedding Provider 完整支持
- 重排序功能（FlagEmbedding）

### Changed

- README文档结构重组
- 优化API调用和文件读取逻辑

---

## [1.1.0] - 2026-03-24

### Added

- 多模态提取器（图片、表格、公式）
- 同步/异步自适应函数

### Changed

- 语义分块器优化
- Bug修复

---

## [1.0.0] - 2026-03-22

### Added

- 初始版本
- 基础RAG功能（PDF解析、语义分块、Milvus存储）
- 支持PDF、Word、TXT、Markdown、HTML格式
- `/paper search` 搜索问答
- `/paper add` 添加文档
- `/paper list` 查看文档列表
