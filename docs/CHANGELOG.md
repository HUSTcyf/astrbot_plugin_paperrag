# 变更记录

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- (暂无)

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
