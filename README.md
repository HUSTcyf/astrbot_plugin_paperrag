# 📚 Paper RAG Plugin v2.2.4 — 用户指南

本地论文库 RAG 检索插件，为 AstrBot 提供智能论文检索、知识图谱增强问答、研究想法生成和远程 Claude Code 编程执行。支持多模态（图片/表格/公式）提取、Llama.cpp VLM 本地问答和 Agentic RAG（LangGraph 工作流）。

> **版本说明**：当前版本 v2.2.4，完整更新历史见 [CHANGELOG.md](docs/CHANGELOG.md)，按版本拆分索引见 [docs/changelog/INDEX.md](docs/changelog/INDEX.md)

### 本版变化 (v2.2.4)

- **Academic intent guard for LLM Tools**：新增 `_guard_academic_intent()` 预检查至 `_paper_search_tool`、`_agentic_rag_tool` 和 `_react_rag_tool`。非学术查询（问候、闲聊）现在会被提前拒绝，避免进入昂贵的 RAG 流水线，每次节省 30+ 秒。
- **强化 `_check_academic_intent`**：扩展负面检测覆盖 "不"/"no"/"非" 变体，添加不可识别 LLM 响应的警告日志。
- **强化 LLM Tool 描述**：为 `paper_search`、`paper_arag`、`paper_react` 工具描述添加 `【严格限制】` 前缀，阻止主 LLM 为非学术查询调用论文工具。

### 上版变化 (v2.2.2)

- **⚠️ LLM Provider 思考模式处理**：参考文献解析需要 LLM 返回**纯 JSON**。推理模型的思考/推理 tokens 会破坏 JSON 解析。目前**自动关闭**的模型：Gemini（`thinking_budget=0`）、DeepSeek（`"thinking": {"type": "disabled"}`）、GLM（同上）。其他 provider 使用默认配置（大部分模型默认不开启思考模式，但若使用 o-series/reasoner 模型需手动在 AstrBot 配置中关闭）。详见下方 [⚠️ LLM 思考模式说明](#-llm-思考模式说明)。
- **LLM Provider 多路径**：provider 三重路径（Gemini → stream → sync），HTTP 作为最终回退。
- **参考文献 4 层解析链路**：Crossref → OpenAlex → arXiv Library → Semantic Scholar → DDG 网络搜索。每层失败后自动退至下一层。
- **arXiv 限流保护**：`_arxiv_lock` 全局串行化 + `_MIN_ARXIV_INTERVAL=2s` 最小间隔，初始化健康检查用真实查询替代 `"test"` 缓存命中，`num_retries=0` 避免 429 时浪费重试时间。
- **字符分片回退**：移除 tiktoken 依赖，回归 `15000` 字符阈值（≈4000 tokens），按参考文献序号边界分割 + 二次强制拆分兜底。
- **并行富化加速**：`asyncio.gather` + `Semaphore(10)`，参考文献富化 5.4x 加速。
- **OpenAlex 异步化**：`pyalex` → `httpx` 原生异步 REST API。
- **智能引用修复**：`classify_papers_for_repair()` 自动分类论文为 full_reparse / link_only 两种策略，`/paper repair_refs confirm` 一键修复所有未链接引用。
- **新命令**：`/paper reparseref <file>`（单篇重解析）、`/paper repair_refs confirm`（智能批量修复）。
- **enable_fallback_search 默认开启**：正常 ingestion 流程也启用 Semantic Scholar + DDG fallback。
- **test/ 清理**：移除 8 个 mock-heavy 测试文件（4368 行），保留 10 个有真实代码路径覆盖的测试。
- 完整变更详见 [docs/changelog/2.2.1.md](docs/changelog/2.2.1.md)

### 上版变化 (v2.2.0)

- **新 LLM Tool：`paper_search`** — 轻量混合 RAG 检索，比 paper_arag/paper_react 更快速，适合简单的论文内容查询。内部走 `engine.search → 文本清洗 → LLM 生成回答`。
- **新 LLM Tool：`code_execute`** — Claude Code 远程编程执行器。AstrBot agent 可调用此工具在服务器上执行编程任务（写/改代码、调试、实验、git 操作等）。采用 `claude -p` 子进程模式，无需 cc-connect 即可使用。内置输入校验（危险命令拦截）和超时子进程清理。
- **安全模型**：code_execute 使用 `--allowedTools` 白名单限制 Claude Code 工具范围，危险命令（rm -rf /、curl|sh、sudo 等）自动拦截，权限不足时返回清晰的授权指引。
- **版本号修正**：`@register()` 装饰器版本号从遗留的 1.12.6 更新为 2.2.0，与 metadata.yaml 保持一致。
- 完整变更详见 [docs/changelog/2.2.0.md](docs/changelog/2.2.0.md)

### 上版变化 (v2.1.2)

- **PaperBanana 本地服务集成**：插件可启动和管理本地 PaperBanana 服务，新增 `paperbanana_project_path` 配置项，自动使用项目 `.venv` 中的 Python 启动方法图生成服务。
- **Feishu 导出修复**：修复 lark-cli vs MCP 网关选择、PaperBanana 临时文件过早清理、`__import__('re')` 惰性导入等多项问题；长引用上下文支持 token 预算分批处理。
- **Docling 逻辑图注编号**：使用 docling 原生 `caption_text()` API 提取论文真实图注编号（如 "Figure 3"），替代基于每页计数器的简单命名；caption 缺失时回退到全局 `unknown` 计数器（`{page}-unknown_{N}.png`）。
- **Cypher 自动修复**：Neo4j 查询缺失 RETURN 子句时自动追加 `RETURN *` 并重试。
- **已验证引用索引**：从 chunk metadata 提取 LLMReferenceParser + arXiv MCP 校验过的引用数据，构建权威引用索引供 LLM 生成使用。
- **Agentic RAG Tool 修复**：移除 `_FakeEvent` 临时方案，改为传入真实 `AstrMessageEvent`，支持多模态响应文本提取。
- **正则表达式移除**：`_find_figure_anchor` 和 `_clean_figure_references` 彻底移除正则，改用 `str.find()` + 逐字符扫描，消除正则引擎隐含语义导致的 bug。
- **规范化章节体系**：8 个 canonical 章节标题，prompt 强制要求 + 运行态空白归一化容错（如 "实验 Benchmark" → "实验Benchmark"）。
- **双锚点图表定位**：引用图表固定插入相关工作末尾，方法论图表固定插入方法论末尾，`str.find()` 精确定位章节边界。
- **锚点重新计算**：从 `clean_text`（去图片行后）计算锚点，确保锚点文本在实际上传文档中存在，修复 `--selection-with-ellipsis` 匹配失败。
- 完整变更详见 [docs/changelog/2.1.2.md](docs/changelog/2.1.2.md)

### 上版变化 (v2.1.1)

- **增量保存机制**：Ragas 评估从批量写入改为逐样本增量保存，中途崩溃不丢失已计算结果。
- **代码溯源元数据**：`raw_answers.json` 新增 `_metadata` 字段，嵌入 git commit hash 等元数据提升可复现性。
- **评估 LLM max_tokens 可配置**：新增 `--eval-llm-max-tokens` CLI 参数（默认 16384）。
- 详细变更见 [docs/changelog/2.1.1.md](docs/changelog/2.1.1.md)

---

## 🏗️ 系统架构

![PaperRAG Architecture](docs/paper_rag_new.png)

PaperRAG 是一个多层学术论文问答系统，支持从简单向量检索到 Agentic 自主推理的多种查询模式：

**检索流水线**：
- **单阶段 3 通道混合检索**：稠密向量（BGE-M3） + 稀疏权重（ABSPEC） + BM25 精确匹配，通过 RRF 融合
- **两阶段检索**（可选）：Stage 1 摘要匹配 → ColBERT 重排序 → 知识图谱独立召回 → Stage 2 chunk 检索
- **可选增强**：ColBERT 多向量重排序、CRAG 质量评估

**知识图谱增强**：`MultimodalGraphBuilder` 从论文 Chunk 中自动抽取知识三元组，使用 Neo4j 作为图存储后端，JSON Schema 精确约束 LLM 输出。支持 **9 类实体**（Method、Model、Task、Dataset、Metric、Component、Limitation、Application、Baseline）和 **14 类关系**（ADDRESSES、PROPOSES、USES_COMPONENT、EVALUATED_ON、ACHIEVES、COMPARES_WITH、LIMITED_BY、APPLIES_TO、EXTENDS、TRAINS_ON、IMPLEMENTS、OUTPERFORMS、REQUIRES、ABLATES_ON）。两阶段检索中，图谱作为独立召回通道（Stage 1.6）补充论文召回，不与向量/BM25 走同一 RRF 融合，避免文本打分过滤掉结构相关但摘要匹配度低的论文。

**Agentic 工作流**（LangGraph，可选启用）：
- **静态 DAG** (`/paper arag`)：router → [vector_search ∥ graph_search] 并行 → synthesize → quality_check 反馈循环
- **ReAct Tool-Using Agent** (`/paper react`)：LLM 自主推理 → 工具选择 → 结果观察 → 迭代，7 个自主工具

**Idea 生成**（可选启用）：analyze → search → generate → critique → debate → refine → save，支持自反思迭代优化和飞书文档导出。

**多模态支持**：Docling/PyMuPDF 提取 PDF 中的图片、表格、公式，`LlamaCppVLMProvider`（Qwen3.5 GGUF，9B/4B 自动降级）支持本地图片问答。

详细架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，Agentic 工作流详解见 [docs/AGENTIC_ARCHITECTURE.md](docs/AGENTIC_ARCHITECTURE.md)。

---

## ✨ 核心功能

- 🔍 **混合检索**：稠密向量 + 稀疏权重（ABSPEC）+ BM25 精确匹配 三通道 RRF 融合，知识图谱独立召回（两阶段）
- 🧠 **Agentic RAG**：LangGraph 静态 DAG + ReAct Tool-Using Agent（7 个自主工具），智能查询分类与多跳推理
- 🔧 **远程 Claude Code**：Agent 可调用 `code_execute` 工具在服务器上执行编程任务，`paper_search` 提供轻量 RAG 检索
- 🕸️ **知识图谱**：Neo4j 图数据库，9 类实体 + 14 类关系，LLM 自动三元组抽取，多模态图谱构建
- 💡 **Idea 生成**：线性 + Agentic 迭代优化（自反思 critique→debate→refine 循环），飞书文档导出
- 🖼️ **多模态处理**：PDF 图片/表格/公式提取，Llama.cpp VLM 本地图片问答（9B/4B 自动降级）
- 📄 **多格式支持**：PDF、Word、TXT、Markdown、HTML
- 💾 **完全本地**：Unsloth BGE-M3 本地 Embedding + Llama.cpp 本地 VLM + Milvus Lite 本地向量库
- ⚡ **缓存加速**：查询结果缓存
- 🏎️ **重排序**：ColBERT 多向量 MaxSim Late Interaction

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ~/AstrBot/data/plugins/astrbot_plugin_paperrag
pip install -r requirements.txt
```

### 1.1 安装飞书 CLI（可选，用于 `/idea tofeishu` 导出）

`/idea tofeishu` 支持通过 [lark-cli](https://www.npmjs.com/package/@larksuite/cli) 一键创建飞书文档。lark-cli 原生支持 Markdown 格式（包括 `[text](url)` 可点击链接和行内样式），优于纯 MCP 块管线。未安装时自动回退到 MCP 模式。

```bash
npx @larksuite/cli@latest install
```

安装后按提示完成飞书账号授权登录即可。

### 2. 配置插件

在 **AstrBot WebUI → 插件 → paper_rag → 插件配置** 中：

| 配置项 | 推荐值 | 说明 |
|-------|--------|------|
| Embedding模式 | `unsloth` | 免费无限制，BGE-M3 本地加载 |
| 向量嵌入维度 | `1024` | BGE-M3 固定 1024 维 |
| 文本问答 Provider | （选取） | 用于 RAG 答案生成 |
| 论文文件存放目录 | `./papers` | PDF 存放路径 |

> ⚠️ 首次使用时 BGE-M3 模型会自动下载到 `./models/bge-m3/`。也可手动下载：
> ```bash
> huggingface-cli download unsloth/bge-m3 --local-dir ./models/bge-m3
> ```

### 3. 使用

以下是从零构建向量数据库和知识图谱、再到查询和飞书撰写的完整流水线。

---

#### 第一步：论文导入与向量化

将 PDF 论文放入 `./papers` 目录（或插件配置中指定的 `论文文件存放目录`），然后执行导入：

```bash
# 批量导入目录下所有 PDF（首次构建）
/paper add ./papers

# 导入单篇论文
/paper addf ./papers/2303.05499v5.pdf

# 清空并重建（更新全部论文时使用，需 confirm）
/paper rebuild ./papers confirm

# 重建单篇论文
/paper rebuildf ./papers/2303.05499v5.pdf confirm
```

导入过程中，插件自动完成：
1. **PDF 解析**（Docling + PyMuPDF）：提取文本、图片、表格、公式
2. **图表保存**：有 caption 的图表以逻辑编号命名（如 `3-Figure3.png`），无 caption 的标记为 `unknown`（如 `9-unknown_1.png`）
3. **文本分块**（语义分块，512 token/块）
4. **向量嵌入**（BGE-M3，1024 维）→ 写入 Milvus
5. **稀疏索引**（ABSPEC + BM25）用于混合检索

```bash
# 查看已导入论文列表
/paper list

# 检查参考文献解析状态
/paper refstats        # 引用频次统计
/paper refstats -1     # 引用解析质量报告（成功率/失败率）
/paper abstractstats    # 摘要提取统计
/paper abstractstats -1 # 列出未提取摘要的论文

# 修复解析问题
/paper reparseref <文件名>              # 单篇引用轻量重解析
/paper repair_refs confirm              # 智能修复所有未链接引用
/paper reparse_zero_ref confirm         # 重解析零引用/空标题论文
/paper reparse_zero_abstract confirm    # 重提取缺失摘要
```

---

#### 第二步：知识图谱构建（可选，需在配置中开启 `enable_graph_rag`）

向量检索完成后，构建知识图谱可增强跨论文关联查询能力：

```bash
# 构建知识图谱（9 类实体 + 14 类关系）
/paper graph_build confirm          # 构建（支持 skip：graph_build confirm 30）

# 查看图谱统计
/paper graph_stats
# 输出示例：实体 1234 个，关系 3456 条，涉及 89 篇论文
```

图谱构建过程：
1. **三元组抽取**：LLM 从论文 chunk 中提取 `(Head, Relation, Tail)` 知识三元组
2. **实体规范化**：自动将 "our method"/"the proposed approach" 替换为论文专属标识符，避免不同论文的通用自指代词被错误合并
3. **别名去重**：4 层去重体系（`Full Name (ACRONYM)` 解析 → 同论文共现 → 跨论文首字母匹配 → `:ALIAS_OF` 关系链接）
4. **Neo4j 写入**：实体节点 + 关系边，支持 Cypher 查询

**Neo4j Browser 可视化**（`http://localhost:7474`）：

```cypher
-- 查看知识图谱全貌（所有节点和关系）
MATCH (n)-[r]->(m)
RETURN n, r, m
```

```bash
# 图谱维护
/paper graph_rebuild confirm   # 重建图谱（清空后重新构建）
/paper graph_clear confirm     # 清空图谱
/paper graph_backup              # 在线备份图谱（Cypher JSON 导出）
/paper graph_restore <文件>    # 从备份恢复
/paper graph_backup_list       # 列出可用备份
```

> **使用云端大模型构建图谱**：在插件配置中设置 `multimodal_provider_id`（如 `google_gemini/gemini-3.5-flash`、DeepSeek、Qwen 等），`graph_build` / `graph_rebuild` 会自动使用该 Provider 进行三元组提取。未配置时回退本地 GGUF 模型。

---

#### 第三步：检索与问答

```bash
# 基础 RAG 检索（3 通道混合检索：稠密向量 + ABSPEC 稀疏 + BM25）
/paper search transformer 的核心创新点是什么？

# 仅检索不生成回答
/paper search attention mechanism 的最新进展 retrieve

# Agentic RAG 复杂查询（需在配置中开启 enable_agentic_rag）
# 静态 DAG：router → 并行检索 → 综合 → 质量检查
/paper arag 3D Gaussian Splatting 和 NeRF 的优劣对比

# ReAct Agent 自主查询（LLM 自主选择工具 + 多轮推理）
/paper react 本地论文库中关于扩散模型的最新进展
```

**检索模式对比**：

| 模式 | 命令 | 适用场景 | 特点 |
|------|------|---------|------|
| 基础 RAG | `/paper search` | 简单问答、事实查询 | 快速，单轮检索 |
| Agentic DAG | `/paper arag` | 对比分析、多跳推理 | 并行检索 + 反馈循环 |
| ReAct Agent | `/paper react` | 开放式探索、复杂推理 | LLM 自主决策，7 个工具 |

---

#### 第四步：研究想法生成与飞书导出

基于已索引的论文库生成研究想法，可导出为飞书文档：

```bash
# 生成研究想法（线性流水线）
/idea gen 3D Gaussian Splatting 的改进方向

# Agentic 迭代优化（自反思 critique→debate→refine 循环）
/idea explore Novel View Synthesis 的最新趋势

# 查看已生成的想法
/idea list
/idea show "3D Gaussian Splatting 的改进方向"

# 导出为飞书文档（需先安装 lark-cli）
/idea tofeishu "3D Gaussian Splatting 的改进方向"

# 导出到指定飞书文件夹
/idea tofeishu "3D Gaussian Splatting 的改进方向" <folder_token>
```

**飞书导出前提**：
```bash
# 安装 lark-cli（一次性，自动支持 Markdown 可点击链接）
npx @larksuite/cli@latest install
# 按提示完成飞书账号授权登录
```

飞书文档包含完整的 8 章节结构（背景动机、相关工作、方法论、创新点、实验 Benchmark、挑战与解决方案、下一步计划、参考文献），图表自动嵌入对应章节。

---

#### 完整流水线示例

```bash
# 1) 导入论文 → 向量数据库
/paper add ./papers

# 2) 构建知识图谱
/paper graph_build confirm

# 3) 查询
/paper search 扩散模型的加速采样方法有哪些？

# 4) 生成想法并导出飞书
/idea gen diffusion model acceleration
/idea tofeishu "diffusion model acceleration"
```

---

## 📖 命令速查

### /paper — 检索与问答

| 命令 | 权限 | 说明 |
|------|------|------|
| `/paper search <query>` | 公开 | RAG 搜索并生成回答 |
| `/paper search <query> retrieve` | 公开 | 仅检索，不生成回答 |
| `/paper arag <query>` | 公开 | Agentic RAG 复杂查询（静态 DAG） |
| `/paper react <query>` | 公开 | ReAct Tool-Using Agent 自主查询 |
| `/paper list` | 公开 | 列出所有已索引文档 |
| `/paper refstats [top_k]` | 公开 | 参考文献引用频次统计 |
| `/paper refstats -1` | 公开 | 引用解析质量报告（成功率/失败率，列出失败论文） |
| `/paper abstractstats [top_k]` | 公开 | 摘要提取统计 |
| `/paper abstractstats -1` | 公开 | 列出未提取摘要的论文 |

### /paper — 论文管理（管理员）

| 命令 | 说明 |
|------|------|
| `/paper add [目录]` | 批量添加论文 |
| `/paper addf <文件>` | 添加单个论文 |
| `/paper delete <文件名>` | 删除指定论文 |
| `/paper clear confirm` | 清空知识库 |
| `/paper rebuild [目录] confirm` | 清空并重建 |
| `/paper rebuildf <文件> confirm` | 重建单个论文 |
| `/paper reparseref <文件>` | 单篇论文引用重解析（轻量） |
| `/paper repair_refs confirm` | 智能修复所有未链接引用 |
| `/paper reparse_zero_ref confirm` | 批量重解析零引用/空标题论文（轻量） |
| `/paper reparse_zero_abstract confirm` | 批量重提取缺失摘要 |

### /paper arxiv — arXiv 集成

| 命令 | 权限 | 说明 |
|------|------|------|
| `/paper arxiv_list` | 公开 | 列出带 arXiv URL 的论文 |
| `/paper arxiv_add <关键词> [数量]` | 管理员 | 搜索、下载并添加 |
| `/paper arxiv_refs [top_k] [数量]` | 管理员 | 下载高引参考文献 |
| `/paper arxiv_sync confirm` | 管理员 | 同步 MCP 下载的论文 |
| `/paper arxiv_cleanup confirm` | 管理员 | 清理旧版本 arXiv 论文 |

### /paper graph — 知识图谱管理

| 命令 | 权限 | 说明 |
|------|------|------|
| `/paper graph_build confirm [skip]` | 公开 | 构建知识图谱（后台，检查点断点续传） |
| `/paper graph_stats` | 公开 | 图谱统计信息 |
| `/paper graph_link [status\|create\|remove]` | 公开 | Neo4j 数据符号链接管理 |
| `/paper graph_rebuild confirm` | 管理员 | 重建图谱 |
| `/paper graph_clear confirm` | 管理员 | 清空图谱 |
| `/paper graph_backup [online\|offline]` | 管理员 | 备份图谱 |
| `/paper graph_restore <文件>` | 管理员 | 恢复图谱 |
| `/paper graph_backup_list` | 公开 | 列出可用备份 |

### /idea — 研究想法生成

| 命令 | 权限 | 说明 |
|------|------|------|
| `/idea gen <主题>` | 公开 | 生成研究想法（线性） |
| `/idea explore <主题>` | 公开 | 探索研究想法（Agentic 迭代优化） |
| `/idea analyze <主题>` | 公开 | 分析研究主题 |
| `/idea search <主题>` | 公开 | 多源知识检索 |
| `/idea generate <主题>` | 公开 | 从知识上下文生成想法 |
| `/idea list` | 公开 | 列出已保存主题 |
| `/idea show <主题>` | 公开 | 查看主题下想法 |
| `/idea add <主题>` | 公开 | 追加想法 |
| `/idea del <UUID>` | 公开 | 删除指定想法 |
| `/idea delete <主题>` | 公开 | 删除主题 + 文件夹 |
| `/idea clear <主题>` | 公开 | 清空主题（保留文件夹） |
| `/idea clean <主题>` | 公开 | 清理主题（删除文件夹） |
| `/idea tofeishu <主题> [folder_token]` | 公开 | 导出飞书文档（lark-cli 优先，支持可点击链接） |
| `/idea regen <UUID>` | 公开 | 重新生成指定想法 |

---

## ⚙️ 配置详解

插件通过 AstrBot WebUI 管理约 60 个配置项，完整 schema 见 `_conf_schema.json`。以下为核心配置分类：

### LLM Provider

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `text_provider_id` | 文本问答 LLM | 空（使用当前会话 Provider） |
| `multimodal_provider_id` | 多模态问答 LLM | 空（使用本地 VLM） |
| `text_llm_temperature` | 文本 LLM 温度 | `0.7` |
| `text_llm_max_tokens` | 文本 LLM 最大 token 数 | `2048` |

### ⚠️ LLM 思考模式说明

参考文献解析依赖 LLM 输出**纯 JSON**。如果 LLM 开启了思考/推理模式，响应中会包含 `<think>` 标签或 `reasoning_content` 字段，导致 JSON 解析失败，参考文献解析**静默返回空结果**。

**插件已自动关闭思考模式的 Provider：**

| Provider | 路径 | 方式 |
|----------|------|------|
| Google Gemini | `_call_via_provider` → `_call_gemini_no_thinking` | `ThinkingConfig(thinking_budget=0)` |
| DeepSeek | `_call_via_http`（HTTP 回退） | `"thinking": {"type": "disabled"}` |
| GLM（智谱） | `_call_via_http`（HTTP 回退） | `"thinking": {"type": "disabled"}` |

**以下情况不会自动关闭思考模式，需要用户手动处理：**

| 场景 | 原因 | 解决方案 |
|------|------|---------|
| DeepSeek/GLM 走 Provider 路径（非 HTTP 回退） | `_call_via_provider_stream/sync` 不修改请求参数 | 在 AstrBot Provider 配置中将模型设为**非 reasoning 版本**（如 `deepseek-chat` 替代 `deepseek-reasoner`） |
| OpenAI o-series（o3, o4-mini 等） | 插件不支持 `reasoning_effort` 参数 | 在 Provider 配置中使用非 reasoning 模型（如 `gpt-4o`） |
| Qwen/其他支持思考的模型 | 插件未针对这些模型做特殊处理 | 在 AstrBot Provider 配置中关闭思考模式，或使用非 reasoning 版本 |
| Anthropic Claude（Opus 4.6+ adaptive thinking） | Claude 默认不开启 extended thinking | 一般不需要处理；若 Provider 配置了 `thinking` 参数，去掉即可 |

**如何判断是否受此问题影响？**

1. 使用 `/paper reparseref <文件名>` 命令测试
2. 观察 AstrBot 日志，如果看到 `JSON 解析失败` 或 `无法从 LLM 响应中提取 JSON` 且 LLM 原始输出中有 `<think>` 标签或大量推理文本，说明思考模式未关闭
3. 同等条件下参考文献解析返回 0 条结果

**不想手动处理的临时方案**：配置 `freeapi_url` + `freeapi_key` 作为备用 LLM。当 Provider 路径失败时，插件会自动回退到 HTTP 路径（该路径对 DeepSeek/GLM 自动关闭思考模式）。

### Llama.cpp 本地 VLM

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `llama_vlm_model_path` | GGUF 模型路径 | `./models/Qwen3.5-9B-GGUF/...` |
| `llama_vlm_mmproj_path` | mmproj 路径 | `./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf` |
| `llama_vlm_n_ctx` | 上下文窗口 | `16384` |
| `llama_vlm_n_gpu_layers` | GPU 层数 | `99` |
| `llama_vlm_max_tokens` | 最大生成 token | `2560` |
| `llama_vlm_temperature` | 生成温度 | `0.7` |

> 自动降级：优先 9B 模型，不可用时自动切换到 4B。

### Embedding / Unsloth BGE-M3

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `embedding_mode` | 模式：`unsloth` 或 `api` | `unsloth` |
| `embed_dim` | 向量维度（BGE-M3=1024） | `768` |
| `unsloth.model_path` | BGE-M3 模型路径 | `./models/bge-m3` |
| `unsloth.device` | 运行设备：`mps`/`cuda`/`cpu` | `mps` |

### 检索配置

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `top_k` | 返回片段数 | `5` |
| `similarity_cutoff` | 相似度阈值 | `0.5` |
| `enable_sparse_retrieval` | 稀疏权重检索 (ABSPEC) | `true` |
| `enable_bm25` | BM25 精确匹配 | `true` |
| `enable_multi_vector_rerank` | ColBERT 重排序 | `false` |

### Agentic / Graph RAG

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `enable_agentic_rag` | 启用 Agentic RAG | `false` |
| `enable_agentic_ideas` | 启用 Agentic Idea Engine | `false` |
| `enable_graph_rag` | 启用 Graph RAG | `false` |


### 分块配置

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `chunk_size` | 分块字符数 | `512` |
| `chunk_overlap` | 块间重叠 | `0` |
| `min_chunk_size` | 最小块大小 | `128` |
| `use_semantic_chunking` | 语义分块 | `true` |

---

## 💡 使用技巧

- **选择合适的分块大小**：学术论文推荐 `512-768`，长篇报告推荐 `768-1024`
- **提高搜索准确度**：使用具体问题、包含专业术语、调整 `top_k` 和 `similarity_cutoff`
- **加速导入**：使用 Unsloth 模式（无 API 限制）、禁用图片提取（`multimodal.extract_images: false`）
- **Agentic 模式**：复杂对比、多跳推理、引用溯源类问题开启 `enable_agentic_rag`，使用 `/paper arag` 或 `/paper react`

---
## 🔧 Claude Code 远程编程执行

AstrBot agent 可以通过 paperrag 插件的 `code_execute` LLM Tool 在服务器上远程执行 Claude Code。Agent 先用 `paper_search`/`paper_arag`/`paper_react` 检索知识，整合上下文后调用 `code_execute` 完成编程任务。

**Agent 调用流程**：`paper_search/arag/react 检索知识 → 整合上下文 → code_execute 执行编程任务 → 返回结果`

### 前置条件：安装 Claude Code

```bash
# 安装 Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 验证安装
claude --version
claude -p "echo hello" --output-format text
```

### 配置 API 访问

code_execute 工具使用 `claude -p` 子进程模式，依赖当前 shell 环境中的 API 凭证：

```bash
# 官方 Anthropic API
export ANTHROPIC_AUTH_TOKEN="sk-ant-..."

# 第三方 API Proxy（如 OpenRouter / one-api）
export ANTHROPIC_BASE_URL="https://your-api-proxy.com"
export ANTHROPIC_AUTH_TOKEN="sk-your-api-key"

# 验证
claude -p "hello" --output-format text
```

> ⚠️ **AstrBot 进程环境变量**：如果用 launchd/systemd 启动 AstrBot，需在 service 配置中设置环境变量。
> launchd plist 示例：
> ```xml
> <key>EnvironmentVariables</key>
> <dict>
>     <key>ANTHROPIC_BASE_URL</key>
>     <string>https://your-api-proxy.com</string>
>     <key>ANTHROPIC_AUTH_TOKEN</key>
>     <string>sk-your-api-key</string>
> </dict>
> ```

### 安全模型

code_execute 默认以受限模式运行，不跳过权限检查：

| 机制 | 说明 |
|------|------|
| `--allowedTools` 白名单 | 限制为 Read、Write/Edit（插件目录）、Bash（git/python/pytest/pip）、Grep、Glob |
| 危险命令拦截 | `rm -rf /`、`curl \| sh`、`sudo`、`chmod 777`、`git push --force` 等自动拒绝 |
| 权限错误引导 | 当任务需要额外权限时，返回清晰的指引信息，引导用户在服务器上手动授权 |

### Agent 如何使用 code_execute

`code_execute` 作为 LLM Tool 注册在 AstrBot 中，Agent 可以像调用 `paper_search` 一样调用它：

1. Agent 收到用户编程请求（如 "帮我写一个数据可视化脚本"）
2. Agent 如需论文知识，先调用 `paper_search` 检索相关文献
3. Agent 将检索结果 + 用户需求整合为完整任务描述
4. Agent 调用 `code_execute(task=整合后的任务, timeout=300)`
5. Claude Code 在服务器上执行并返回结果
6. Agent 将结果回复给用户

### 可选：cc-connect 会话持久化（高级）

**默认不需要 cc-connect** —— 直接 `claude -p` 即可满足绝大多数场景。

如果需要**跨调用保持 Claude Code 会话上下文**（例如多轮交互式编程），可以安装 cc-connect：

```bash
# 安装 cc-connect
npm install -g @anthropic-ai/cc-connect

# 创建项目
cc-connect feishu setup --project paperrag

# 启动 cc-connect 服务
cc-connect start

# 验证服务状态
cc-connect status
```

cc-connect 配置文件 `~/.cc-connect/config.toml`：

```toml
[server]
port = 9025
host = "127.0.0.1"

[projects.paperrag]
path = "/Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag"

[projects.paperrag.options.agent]
model = "claude-sonnet-4-6"

[projects.paperrag.options.env]
ANTHROPIC_BASE_URL = "https://your-api-proxy.com"
ANTHROPIC_AUTH_TOKEN = "sk-your-api-key"
```

启动后可通过 `claude --resume <session_id>` 复用会话上下文：

```bash
# 创建新会话
claude --resume new --print "列出项目的主要模块" --output-format text

# 后续调用复用同一会话
claude --resume <session_id> --print "详细分析第一个模块" --output-format text
```

> 详情参考 [docs/FEISHU_CLAUDE_CODE_REMOTE.md](docs/FEISHU_CLAUDE_CODE_REMOTE.md)

---

## ❓ 常见问题

### RAG 引擎未就绪

检查 WebUI → 设置 → 模型提供商中是否已添加 Embedding Provider，确认插件配置中 Provider ID 正确。

### 导入后 chunks=0

确认 PDF 不是扫描版；确保 docling/PyMuPDF 依赖已安装：`pip install -r requirements.txt`

### 搜索结果不准确

尝试增大 `chunk_size`、降低 `similarity_cutoff`、增加 `top_k`，或开启稀疏检索和 BM25。

### ColBERT 重排序不可用

确保 `embedding_mode=unsloth`、BGE-M3 模型存在于 `models/bge-m3/`。MPS 内存不足时可改 `unsloth.device` 为 `cpu` 或关闭 `enable_multi_vector_rerank`。

### Llama.cpp VLM 不可用

检查 GGUF 模型和 mmproj 文件路径。插件支持 9B→4B 自动降级。安装命令：
```bash
CMAKE_ARGS="-DGGML_METAL=on -DLLAMA_MTMD=on" pip install llama-cpp-python
```

### /idea tofeishu 飞书文档创建失败

插件使用双路径架构：**lark-cli 优先**（`npx @larksuite/cli@latest install` 安装），**MCP 回退**。常见问题：

- **lark-cli 未安装**：自动回退到 MCP 模式（通过 `batch_create_feishu_blocks` 创建），功能正常但链接渲染为纯文本格式
- **飞书 MCP 未授权**：在飞书开放平台重新授权 MCP 应用，或安装 lark-cli 后重试
- **图片过多超时**：`+media-insert` 默认 30s 超时，可在日志中查看具体失败原因

---

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构设计、组件说明、文件结构 |
| [docs/AGENTIC_ARCHITECTURE.md](docs/AGENTIC_ARCHITECTURE.md) | Agentic RAG + Agentic Ideas 工作流详解 |
| [docs/FEISHU_CLAUDE_CODE_REMOTE.md](docs/FEISHU_CLAUDE_CODE_REMOTE.md) | 飞书远程操控 Claude Code 方案（含 cc-connect 安装配置） |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | 变更记录 |
| [docs/changelog/INDEX.md](docs/changelog/INDEX.md) | 按版本拆分的变更索引 |
| [docs/FEISHU_BLOCK_STYLING.md](docs/FEISHU_BLOCK_STYLING.md) | 飞书导出技术方案（lark-cli + MCP 双路径） |
| [docs/cypher_queries.md](docs/cypher_queries.md) | Neo4j Cypher 查询参考 |
| [docs/INDEX.md](docs/INDEX.md) | 文档索引 |

---

## 📞 获取帮助

- **问题反馈**：通过 [GitHub Issues](https://github.com/HUSTcyf/astrbot_plugin_paperrag/issues) 提交
- **日志查看**：AstrBot 控制台输出

---

## 📄 许可证

MIT License

## 🙏 致谢

- [AstrBot](https://github.com/AstrBotDevs/AstrBot) — 聊天机器人框架
- [Milvus](https://milvus.io/) — 向量数据库
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF 解析
- [Docling](https://github.com/DS4SD/docling) — 多模态文档解析
- [Neo4j](https://neo4j.com/) — 图数据库
- [PaperBanana](https://github.com/dwzhu-pku/PaperBanana) — AI 学术图表生成
- [Unsloth](https://unsloth.ai/) — BGE-M3 本地加速
