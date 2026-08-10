# RAGAS 评测闭环 — 当前实现说明（2026-08-08 起，后续会话持续演进）

> 目标：在 AstrBot 生产环境（uv tool env，Python 3.12）中复用 `evaluation/` 框架，
> 对 papers/ 的 20 篇论文跑 RAGAS 测试集生成 + 评测，支撑 finetune 微调闭环。
> 本文档只描述**当前存活**的实现；早期被否决/回滚的方案已从文档移除，仅存于 git 历史。

## 一、评测依赖锁定（`requirements-ragas.txt`）

- `ragas==0.4.3`：`evaluation/` 使用 ragas 0.4 私有 API，必须锁版本。
- `langchain-community<0.4.2`：ragas 0.4.3 硬 import `chat_models.vertexai`，
  该模块在 community 0.4.2 被移除，必须压版本。
- 该文件为独立环境（`pip install -r requirements-ragas.txt`），与插件主依赖分离。

## 二、MiniMax 专属兼容层（`evaluation/minimax_compat.py`，新增）

MiniMax-M3（Token Plan）与标准 OpenAI 兼容端点（智谱 GLM 等）有 3 处差异。
为不让这些差异污染 `ragas_generator.py` / `ragas_evaluator.py` 的核心逻辑，将其集中到
本独立模块；智谱等标准端点完全不走本模块分支。三处差异：

| 差异 | MiniMax | 标准端点 |
|---|---|---|
| 思考模式 | 默认输出 `<think>` 块破坏 JSON | 无思考模式 |
| embedding 请求 | `{model, texts, type:"query"}`（非标准） | `{model, input: texts}` |
| embedding 模型名 | `embo-01` | 标准命名（text-embedding-3-small 等） |

导出 helper：
- `is_minimax_endpoint(api_base)` — 按 URL 关键字判断
- `needs_thinking_disabled(api_base)` / `build_llm_request_fields(api_base)` /
  `apply_llm_request_fields(kwargs, api_base)` — thinking 走 `extra_body`（openai SDK ≥2.x
  不接受 thinking 顶层参数），response_format 走顶层
- `resolve_embedding_model(api_base, default_model)` — 仅当用户未显式指定（仍是占位默认值）
  时自动切换：MiniMax→`embo-01`，智谱→`embedding-3`
- `build_embedding_request_data(api_base, ...)` / `extract_embedding_vectors(api_base, ...)`

不归本模块（对任何 LLM 都适用，留在 `ragas_generator.py`）：
- `_normalize_json_response`（`<think>` 剥离 + JSON 修复）——通用防御
- 7 个 extractor monkey-patch + `Executor.results` 过滤——ragas 0.4.3 `run_async_tasks` bug
- 429 退避——通用限流处理

## 三、测试集生成（`evaluation/ragas_generator.py`）

- `OpenAICompatibleLLM(BaseRagasLLM)`：标准 OpenAI 兼容 chat completions 客户端，
  构造时按 `minimax_compat` 决定是否禁用 thinking；`_normalize_json_response` 剥离
  `<think>` 块 → 去 markdown code fence → `_repair_json_string` → 解 `{"text":"<json>"}`
  双重包装 → 字面量 null/空对象判错重试 → schema 回显检测重试。
- 429 限流：`RateLimitError` + `_compute_backoff`（优先 `Retry-After`，否则 5/10/20/40/60…
  长指数退避）+ 默认 `_max_rpm=30`（MiniMax Token Plan 限流敏感，96 会触发大量 429）。
- `OpenAICompatibleEmbeddings(BaseRagasEmbeddings)`：走 `minimax_compat` 构造请求/提取向量。
- `UnslothEmbeddingsWrapper(BaseRagasEmbeddings)`：本地 BGE-M3（复用插件
  `embedding/unsloth_embedding.py` 单例）的 ragas embedding 适配器。
- **transforms 容错补丁**：`run_async_tasks`（ragas/async_utils.py:228）在所有任务完成后
  无条件 raise 第一个异常——单个 chunk 的 filter 解析失败即中止整个管线。对
  `SummaryExtractor` / `HeadlinesExtractor` / `ThemesExtractor` / `NERExtractor` /
  `CustomNodeFilter` / `EmbeddingExtractor` / `HeadlineSplitter` 的方法做 try/except
  包装，失败只跳过该节点（finally 恢复原方法）；`Executor.results` 过滤混入结果的
  Exception（`raise_exceptions=False` 时 ragas 会把异常塞进结果列表，导致 TestsetSample
  构造失败 → 全量 0 样本）。
- `RagasTestsetGenerator`：milvus_chunks → KnowledgeGraph → transforms → Testset 的组装。
- 同步/异步路径均经 `minimax_compat` 注入 thinking disabled（标准端点请求体保持纯净）。

## 四、评测（`evaluation/ragas_evaluator.py`）

- `_LLMWithN`：支持 `n>1` 的 ragas LLM 适配器；`_make_agenerate_text` 组装请求体，
  thinking 经 `apply_llm_request_fields` 注入（标准端点空 dict）。
- `_get_embed_model`：按 `embedding_mode` 分发——`api` → `OpenAICompatibleEmbeddings`；
  `unsloth` → `UnslothEmbeddingsWrapper`（本地 BGE-M3）。
- `load_raw_answers`：兼容新旧 raw_answers 格式。
- 增量保存：每完成一个样本写 `raw_answers.json`，中途崩溃不丢已计算结果。
- `_get_git_info`：记录评测时的 git commit。

## 五、入口（`evaluation/run_evaluation_ragas.py`）

- Step 2.6：CLI/环境变量未提供 key 时，从主 AstrBot `cmd_config.json` 自动填充 LLM 凭据
  ——按 `text_provider_id` → provider → provider_source 查找；处理 key 为 **list**（AstrBot
  格式）的解包；`/anthropic` 端点转 `/v1`；embedding 模型名经 `resolve_embedding_model`
  自动解析（MiniMax→embo-01，智谱→embedding-3）。
- Step 2.55：`--eval-provider` 显式选择评测端点（关键字匹配 cmd_config provider id），
  覆盖插件 `text_provider_id`。
- `_PROVIDER_MODULES` 含 `minimax_token_plan` 映射。
- `--step generate|evaluate|all`，`--skip-rag` 跳过上下文检索等开关。

## 六、调试历史要点（已解决，不再改）

| 问题 | 根因 | 现行修复 |
|---|---|---|
| HTTP 401 | MiniMax `cmd_config.json` 中 key 是 list，`str(list)` 产生错误 token | Step 2.6 list 解包 |
| JSON 解析失败 | MiniMax-M3 配置 reasoning，输出 `<think>` 块 | thinking disabled + `<think>` 剥离 |
| schema 回显 | 模型回显 prompt 中 JSON schema 本体 | 检测 + 重试 |
| embeddings KeyError | MiniMax `/v1/embeddings` 非标准（texts/vectors） | minimax_compat 兼容层 |
| **0 样本** | ① 模型偶发输出字面量 null 未被拦截；② `run_async_tasks` 无条件 raise 首个异常，单个 filter 块失败中止管线 | null 拦截 + transforms 容错补丁 + Executor.results 过滤 |
| 429 风暴 | MiniMax Token Plan 限流敏感，高并发触发 | 长退避 + `_max_rpm=30` |
| `import ragas` 挂起 | 误判为 ragas 自身问题；实际是 WSL drvfs FS-cache 温度导致的间歇性 I/O 挂起 | 裸 import 成功，无需修复 |
