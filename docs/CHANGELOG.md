# Changelog

所有值得注意的插件变更都会记录在这个文件中。

## [1.7.3] - 2026-04-02

### 图表提取优化

#### 1. 图表索引与存储规范化

**文件**: `multimodal_extractor.py`, `hybrid_parser.py`

**目标**: 优化图表的索引、关联和存储，与 Qasper 评估数据集格式保持一致。

**变更**:

- `_extract_figure_number()` 返回纯编号（如 `"1"`, `"A1"`）而非 `"Figure 1"`
- `_find_figure_caption()` 图片数量超过图注数量时返回 `None`
- 新增 `_find_table_caption()` 方法提取 Table caption
- 新增 `get_figures_and_tables()` 生成 Qasper 格式的图表列表
- `_extract_tables_from_page()` 添加 `page_text` 参数并调用 caption 提取

---

#### 2. 修复子图文件名冲突

**文件**: `hybrid_parser.py`

**问题**: 同一 Figure 的多个子图互相覆盖

**变更**: 引入 `caption_variant_counter` 解决子图覆盖问题

```python
# 构建 figure_id：{page}-{type}{num}-{variant}
# 例如: 3-Table1-1.png, 3-Table1-2.png
```

---

#### 3. 文件命名格式对齐 Qasper

**文件**: `hybrid_parser.py`

| 方面 | 修改前 | 修改后 |
|------|--------|--------|
| 文件名 | `{pdf_name}_p{page}_i{idx}.png` | `{page}-{type}{num}-{variant}.png` |
| 示例 | `1911.10742_p3_i1.png` | `3-Table1-1.png` |

---

#### 4. 图表按论文分类存储

**文件**: `hybrid_parser.py`

**变更**: 图表按论文ID分类存储，目录结构与 Qasper 数据集一致

```
data/figures/
├── 1911.10742/
│   ├── 3-Table1-1.png
│   └── 5-Figure1-1.png
├── 1904.09131/
│   └── ...
```

---

#### 5. 修复 FlagEmbedding 与 transformers 版本冲突

**文件**: `requirements.txt`

**问题**: `transformers>=5.0` 与 `FlagEmbedding` 不兼容（`is_torch_fx_available` 被移除）

**变更**:
- `transformers>=4.40.0` → `transformers>=4.40.0,<5.0`
- 移除 `mlx-lm`（与 `transformers<5.0` 冲突）

---

## [1.7.2] - 2026-04-01

### 核心改进

#### 1. CRAG质量阈值优化 - 动态阈值策略

**文件**: `hybrid_rag.py`

**问题**: 固定检索质量阈值无法适应不同类型问题。Qasper有9.6%不可回答问题，需要更严格的阈值。

**变更**: 实现动态阈值策略（策略1 - 动态阈值）：
```python
# 根据问题类型调整阈值：none类型用更严格标准
unanswerable_patterns = [
    "does the paper", "does not", "doesn't", "is not mentioned",
    "not mention", "whether the paper", "is there any", "what is the name of",
    "who proposed", "who suggested", "when was", "where did"
]
is_none_type_query = any(p in query.lower() for p in unanswerable_patterns)
threshold = 0.35 if is_none_type_query else 0.15  # 不可回答问题用更高阈值
```

---

#### 2. 证据链验证 - 减少幻觉

**文件**: `hybrid_rag.py`

**问题**: 即使检索质量评分高，LLM仍可能产生幻觉答案。

**变更**: 实现证据链验证策略（策略4 - 证据链验证）：
```python
# 在生成答案前，先验证检索到的证据是否支持回答
evidence_prompt = f"""Extract the specific sentences from the following content that are relevant to answering: {query}

Content: {combined_text}

If no relevant evidence exists, respond with only: NO_EVIDENCE"""
```

---

#### 3. 提升检索召回率

**文件**: `hybrid_rag.py`

**变更**: 增加CRAG初筛召回数量：
```python
async def retrieve(self, query: str, top_k: int = 5, use_llm_fusion: bool = False,
                   initial_vector_k: int = 50,  # 原为 5
                   initial_bm25_k: int = 100) -> QueryResult:  # 原为 20
```

---

### 评估功能

#### 4. 纯LLM基线模式

**文件**: `evaluation/run_evaluation_qasper.py`

**功能**: 新增 `--llm_only` 参数，支持纯LLM基线对比评估：
```bash
python evaluation/run_evaluation_qasper.py --generate --llm_only --limit 50
```

**用途**: 对比RAG+CRAG与纯LLM的性能差异，验证检索增强的有效性。

---

### Bug修复

#### 5. 单例模式修复

**文件**: `graph_builder.py`, `llama_cpp_vlm_provider.py`

**问题**: `get_llama_cpp_vlm_provider()` 调用方式不一致，部分代码传递了参数导致调用失败。

**修复**:
- `get_llama_cpp_vlm_provider()` 简化为无参数版本，使用缓存/默认配置
- 新增 `get_cached_llama_cpp_provider()` 用于检查已初始化实例
- 使用 `init_llama_cpp_vlm_provider()` 进行显式初始化

---

#### 6. LLM-only模式修复

**文件**: `hybrid_rag.py`

**问题**: 证据链验证在无检索结果时仍然执行，导致纯LLM模式失败。

**修复**: 证据验证仅在有检索来源时执行：
```python
if sources:
    # Evidence verification only for RAG mode
    ...
else:
    # Pure LLM mode: use model knowledge directly
    text_prompt = f"""Answer the following question about a research paper..."""
```

---

#### 7. BERTScore F1 语义评估支持

**文件**: `datasets/qasper_evaluator.py`, `evaluation/run_evaluation_qasper.py`

**背景**: QASPER数据集包含长文档（平均23000+字符）和自由形式答案，传统Cosine F1评估（0.22）过于严格。BERTScore F1（0.62）更适合语义评估。

**功能**: 新增 `--bert_score` 参数，使用BERTScore进行语义评估：
```bash
python evaluation/run_evaluation_qasper.py --evaluate --bert_score
```

**优势**:
- 评估语义而非词汇重叠
- 对布尔问题（No/False, Yes/True）更公平
- 更适合长文档自由形式答案

---

#### 8. 增强VLM路由 - 检测chunk文本内容

**文件**: `hybrid_rag.py`

**功能**: 扩展 `_should_use_vlm()` 和 `_extract_image_paths_from_sources()`：

1. **检索文本内容检测**：若chunk提到 "Figure X", "Table Y" 等，自动触发VLM
2. **扩展查询模式检测**：问题询问数量/比较/性能时触发VLM

```python
# 检索文本视觉模式
STRICT_VISUAL_PATTERNS = [
    r'\bfigure\s*\d+', r'\btable\s*\d+', r'\bchart\s*\d+',
    r'\balgorithm\s*\d+', r'\barchitecture\s*\d+',
]

# 查询需要视觉模式
QUERY_NEEDS_VISUAL_PATTERNS = [
    r'how\s+(big|large|many|much)',
    r'which\s+is\s+(better|larger|smaller|...)',
    r'\baccuracy\b', r'\bprecision\b', r'\bf1\b',
]
```

---

#### 9. LLM-only模式VLM支持

**文件**: `evaluation/run_evaluation_qasper.py`

**功能**: 纯LLM基线模式也支持VLM图片问答

**实现**:
- 根据问题关键词判断是否需要VLM
- 若涉及视觉内容，自动加载论文图表传给VLM
- 非视觉问题保持纯文本模式

---

## [1.7.1] - 2026-03-31

### 核心修复

#### 1. VLM路由逻辑修复 - 支持Qasper数据集图表识别

**文件**: `hybrid_rag.py`

**问题**: VLM路由无法识别Qasper数据集的图表caption节点，导致即使检索到图表相关内容也不会触发VLM模式。

**变更**:

1. `_should_use_vlm()` 方法新增 `sources_have_figure_captions` 检查：
```python
# 检查检索结果是否包含图表caption节点（Qasper等数据集场景）
sources_have_figure_captions = any(
    src.get("metadata", {}).get("node_type") == "figure_caption"
    or src.get("metadata", {}).get("figure_file")
    for src in sources
)

# VLM条件：查询涉及视觉 OR 检索结果有图片 OR 检索结果有图表captions
use_vlm = query_has_visual or sources_have_images or sources_have_figure_captions
```

2. `_extract_image_paths_from_sources()` 方法新增Qasper图片路径构建：
```python
# 2. figure_file 字段（Qasper等数据集场景）
figure_file = metadata.get("figure_file")
paper_id = metadata.get("paper_id")
if figure_file and paper_id:
    # Qasper图片路径格式: {plugin_dir}/datasets/test_figures_and_tables/{paper_id}/{figure_file}
    qasper_fig_path = str(plugin_dir / "datasets" / "test_figures_and_tables" / paper_id / figure_file)
    if qasper_fig_path not in seen:
        seen.add(qasper_fig_path)
        image_paths.append(qasper_fig_path)
```

**Qasper图片目录结构**:
```
datasets/test_figures_and_tables/{paper_id}/{figure_file}
例如: datasets/test_figures_and_tables/1911.10742/3-Table1-1.png
```

**影响**: 现在使用`milvus_qasper_vision.db`时，检索到`figure_caption`节点会自动触发VLM模式并加载对应图片。

---

## 2026-03-30

### 核心修复

#### 2. Qasper评估 - 禁用BM25索引

**文件**: `evaluation/run_evaluation_qasper.py`

**问题**: BM25索引构建时加载了错误的数据库（`papers_dir`指向其他论文），导致BM25拿到0个chunks失败。

**变更**: 对Qasper评估显式禁用BM25，仅使用Milvus向量检索：
```python
enable_bm25=False,  # Qasper 评估禁用 BM25，只用 Milvus 向量检索
```

**原因**: Qasper数据集只有提取的文本（JSONL格式），不包含原始PDF文件，因此无法构建BM25索引。

---

#### 3. Qasper数据集格式统一

**文件**: `evaluation/index_qasper.py`

**问题**: text-only模式和vision模式使用不同的数据格式，导致对比不公平。

**变更**: 统一使用原始格式`qasper-test-v0.3.json`进行索引：
```python
# 统一使用原始格式文件 qasper-test-v0.3.json 进行公平对比
# 原始格式包含 figures_and_tables，可用于 vision 模式
raw_qasper_file = SCRIPT_DIR / "datasets" / "qasper-test-v0.3.json"
papers_data = load_qasper_raw_data(raw_qasper_file)
```

---

### 新增功能

#### 4. Qasper评估 - 支持选择数据库路径

**文件**: `evaluation/run_evaluation_qasper.py`

**新增参数**: `--milvus-qasper-path`

允许用户指定Qasper专用的Milvus数据库路径，便于在text-only和vision模式之间切换：
```bash
# text-only模式
python evaluation/run_evaluation_qasper.py --generate --milvus-qasper-path ./data/milvus_qasper_text.db

# vision模式
python evaluation/run_evaluation_qasper.py --generate --milvus-qasper-path ./data/milvus_qasper_vision.db
```

---

## [历史变更]

---

## 新增功能

### 1. Neo4j 服务状态检查

**文件**: `main.py`

**功能**: 插件启动时检查 Neo4j 服务是否运行，提示用户启动

**配置项**: `auto_start_neo4j` (默认: true)

```json
{
    "auto_start_neo4j": true,
    "graph_rag": {
        "storage_type": "neo4j",
        "neo4j_password": "password"
    }
}
```

**特点**:
- Milvus：使用内置 Lite 版本，无需安装，数据存储在 `milvus_papers.db`
- Neo4j：需要原生安装，插件只检查状态并提示用户

**Neo4jServiceManager 类**:
```python
class Neo4jServiceManager:
    async def ensure_neo4j_running() -> bool:
        """检查并确保 Neo4j 运行"""

    def get_connection_info() -> dict:
        """获取连接信息"""
```

---

## 核心修复

### 2. Graph Builder - 使用 LlamaCppVLMProvider

**文件**: `graph_builder.py`

**变更**:
- `MultimodalGraphBuilder._llm` 类型从 `Optional[LocalLLMProvider]` 改为 `Optional[Any]`，以接受 `LlamaCppVLMProvider`
- 新增 `_ensure_llm_initialized()` 方法，优先使用 `LlamaCppVLMProvider`

```python
# graph_builder.py:590
self._llm: Optional[Any] = None  # LlamaCppVLMProvider

# graph_builder.py:625-637
async def _ensure_llm_initialized(self):
    """确保 LLM 已初始化 - 使用 LlamaCppVLMProvider"""
    if self._llm is None:
        from .llama_cpp_vlm_provider import get_llama_cpp_vlm_provider
        self._llm = get_llama_cpp_vlm_provider(...)
    await self._llm.initialize()
```

**原因**: `LocalLLMProvider.chat()` 方法存在以下问题：
- `temperature=0.1` 过低
- 错误的 stop tokens 配置
- 调用方式不正确

而 `LlamaCppVLMProvider.text_chat()` 正常工作。

---

### 3. 批量处理优化

**文件**: `graph_builder.py`

**变更**:
- 新增 `BATCH_TRIPLET_EXTRACTION_PROMPT` 用于批量三元组抽取
- 新增 `_process_batch()` 方法，支持一次 LLM 调用处理 8 个 chunks
- 减少 LLM 调用次数：从 ~1800 次降至 ~450 次（针对 1800 个 chunks，batch_size=4）

```python
# graph_builder.py:667-678
batch_size = 4
total_batches = (len(nodes) + batch_size - 1) // batch_size

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(nodes))
    batch_nodes = nodes[start_idx:end_idx]
    result = await self._process_batch(batch_nodes, graph_store, batch_idx, total_batches)
```

**计算依据**:
- 4 chunks × ~500 字符 ≈ 2000 字符 ≈ 500-800 tokens
- 加上 system prompt ≈ 1500 tokens
- 总计 ≈ 2000-2300 tokens < 4096 (n_ctx)

---

### 4. 修复 add_relation 参数错误

**文件**: `graph_builder.py`

**变更**: 修正 `add_relation()` 调用参数，使用实体名称字符串而非 ID

```python
# 错误写法 (使用 head_id/tail_id)
rel_id = graph_store.add_relation(head_id=..., tail_id=..., relation=...)

# 正确写法 (使用 head/tail 字符串名称)
rel_id = graph_store.add_relation(
    head=head,        # 实体名称字符串
    tail=tail,        # 实体名称字符串
    relation=relation,
    weight=triplet.get("confidence", 1.0),
    chunk_id=chunk_id
)
```

**MemoryGraphStore.add_relation 签名**:
```python
def add_relation(
    head: str,       # entity name, not ID
    tail: str,       # entity name, not ID
    relation: str,
    weight: float = 1,
    chunk_id: str = ""
) -> Optional[str]
```

---

### 5. 增加 max_tokens 限制

**文件**: `graph_builder.py`

**变更**: `MultimodalLLMConfig.max_tokens` 从 1024 增加到 4096

```python
# graph_builder.py:620
return MultimodalLLMConfig(
    ...
    max_tokens=4096,  # 增加 token 限制以支持更大的 JSON 响应
    temperature=0.1,
    ...
)
```

**原因**: 批量处理 8 个 chunks 产生的 JSON 响应较大，1024 tokens 不足以完整输出，导致 JSON 被截断。

---

### 6. 移除 thinking tokens

**文件**: `graph_builder.py`

**变更**: 新增 `_strip_thinking_tokens()` 方法，递归移除 `<think>...</think>` 块

```python
# graph_builder.py:1143-1155
def _strip_thinking_tokens(self, text: str) -> str:
    """移除 <think>...</think> 块"""
    if not text:
        return text
    import re
    stripped = text
    while '<think>' in stripped and '</think>' in stripped:
        stripped = re.sub(r'<think>.*?</think>', '', stripped, flags=re.DOTALL)
    return stripped.strip()
```

并在 `_parse_json_response()` 和 `_parse_multimodal_response()` 中调用此方法。

---

### 7. 修复 LlamaCppVLMProvider.system_prompt 未使用

**文件**: `llama_cpp_vlm_provider.py`

**变更**: `text_chat()` 方法现在正确使用 `system_prompt` 参数

```python
# llama_cpp_vlm_provider.py:324-327
messages = []
if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
messages.append({"role": "user", "content": content})
```

**原因**: 之前 `system_prompt` 被接受但从未被使用，导致 prompt 注入无效。

---

### 8. Pure Text RAG 优先使用本地模型

**文件**: `hybrid_rag.py`

**变更**: `_ensure_llm_initialized()` 方法优先使用 LlamaCpp 本地模型

```python
# hybrid_rag.py:436-467
async def _ensure_llm_initialized(self) -> Any:
    # 优先使用 LlamaCpp 本地模型
    if LLAMA_CPP_VLM_AVAILABLE:
        try:
            provider = get_llama_cpp_vlm_provider(...)
            await provider.initialize()
            ...
            logger.info("✅ 使用 LlamaCpp 本地模型进行文本生成")
            return provider
        except Exception as e:
            logger.warning(f"⚠️ LlamaCpp 模型加载失败: {e}")

    # 如果本地模型不可用，尝试AstrBot当前会话的Provider（云端）
    ...
```

---

### 9. 添加 Chunk 加载进度日志

**文件**: `main.py`

**变更**: 添加 chunks 加载进度日志

```python
logger.info(f"[PaperRAG] 正在加载 chunks... ({len(chunks)} 个)")
```

---

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `PAPERRAG_GGUF_MODEL_PATH` | GGUF 模型路径 | `./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_XL.gguf` |

### 模型降级

如果 9B 模型加载失败，代码会自动降级到 4B 模型：
- `Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf` → `Qwen3.5-4B-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf`
- `mmproj-BF16.gguf` 同样降级

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `main.py` | 修改 | Neo4j 服务状态检查、chunk 加载进度日志 |
| `graph_builder.py` | 修改 | 批量处理、LlamaCppVLMProvider 集成、修复 add_relation |
| `llama_cpp_vlm_provider.py` | 修改 | 修复 system_prompt 未使用问题 |
| `hybrid_rag.py` | 修改 | VLM路由修复、优先使用本地 LlamaCpp 模型 |
| `evaluation/index_qasper.py` | 修改 | 统一数据集格式 |
| `evaluation/run_evaluation_qasper.py` | 修改 | 禁用BM25、支持数据库路径选择 |
| `_conf_schema.json` | 修改 | 添加 auto_start_neo4j 配置项 |

---

## 测试建议

1. **Graph RAG 图谱构建测试**:
   ```bash
   # 重新加载插件后，执行图谱构建
   /paper rebuild
   ```

2. **检查日志输出**:
   - `[Graph-LLM]` 前缀的日志显示知识图谱构建进度
   - `[Llama.cpp-VLM]` 前缀的日志显示模型加载状态
   - `[VLM路由]` 前缀的日志显示视觉路由决策

3. **常见问题**:
   - 如果图谱为空，检查 `[Graph-LLM]` 日志中是否有 JSON 解析错误
   - 如果模型加载失败，确认 GGUF 模型文件存在于 `models/` 目录
   - 如果VLM未按预期触发，检查 `[VLM路由]` 日志中的 `sources_have_figure_captions` 值

---

## [2026-04-02] v1.7.x - LLM-Only Reference Parsing

### 1. 移除正则表达式参考文献解析，保留纯 LLM 方案

**文件**: `reference_processor.py`

**变更**: 删除所有正则表达式-based 解析代码，仅保留 LLM 解析

**删除的组件**:
- `ReferenceExtractor` 类 (~470 行) - 正则表达式提取器
- `process_references_and_citations()` 函数 - 正则处理入口
- `GrobidReferenceParser` 类 (~480 行) - GROBID API 解析器
- `process_references_and_citations_grobid()` 函数 - GROBID 处理
- `_merge_reference_lists()` 函数 - 引用列表合并

**保留的组件**:
- `CitationLinker` 类 - 引用链接
- `LLMReferenceParser` 类 - LLM 解析器
- `process_references_with_llm()` 函数 - LLM 处理入口

**新增**:
- 模块级 `_find_reference_section()` 函数 - 使用关键词匹配定位参考文献部分

```python
REFERENCE_SECTION_KEYWORDS = [
    'references', 'reference', 'bibliography', 'works cited',
    'reference list', 'literature cited'
]

def _find_reference_section(text: str) -> Optional[str]:
    """找到参考文献部分"""
    ...
```

**效果**: 文件从 1913 行缩减至 ~890 行（减少 54%）

---

### 2. 修复 LLM 输出截断问题

**文件**: `reference_processor.py`

**问题**: 1087 行参考文献超出 max_tokens 限制导致截断

**修复**: 将 `_call_llm()` 中的 `max_tokens` 从 8192 提升至 16384

```python
"max_tokens": 16384,  # 模型最大支持 16384 tokens
```

**注意**: 如仍超出限制，需实现批量处理将参考文献分批处理

---

### 3. 更新 hybrid_parser.py 引用

**文件**: `hybrid_parser.py`

**变更**: 更新 import 语句，移除已删除的函数引用

```python
# 更新前
from .reference_processor import (
    ReferenceExtractor,
    CitationLinker,
    process_references_and_citations,
    process_references_and_citations_grobid,
    process_references_with_llm,
    Reference
)

# 更新后
from .reference_processor import (
    CitationLinker,
    process_references_with_llm,
    Reference
)
```

**引用处理逻辑**: 移除了正则 fallback，仅使用 LLM 解析

```python
if effective_llm_config:
    try:
        references, all_nodes = await process_references_with_llm(...)
    except Exception as e:
        logger.warning(f"⚠️ LLM引用处理失败: {e}")
        references = []
else:
    references = []
```

---

## 测试建议

1. **验证引用解析**:
   ```bash
   # 处理包含大量参考文献的 PDF
   /paper parse <pdf_path>
   ```
   检查日志中是否有截断警告。

2. **检查语法**:
   ```bash
   python3 -m py_compile reference_processor.py
   python3 -m py_compile hybrid_parser.py
   ```

---

### 4. 修复 CRAG 评估 JSON 解析问题

**文件**: `hybrid_rag.py`

**问题**: LLM 返回的 JSON 中 reasoning 字段包含未转义引号（如 `如"10,000 条对话"`），导致 JSON 解析失败：
```
Expecting ',' delimiter: line 4 column 168 (char 209)
```

**修复**: 使用正则 fallback 直接提取 JSON 字段值，不依赖 `json.loads()`

```python
# hybrid_rag.py:544-556
import re
score_match = re.search(r'"score"\s*:\s*([0-9.]+)', matched_json)
score = float(score_match.group(1)) if score_match else 0.5
level_match = re.search(r'"level"\s*:\s*"([^"]+)"', matched_json)
level = level_match.group(1) if level_match else "medium"
reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', matched_json)
reasoning = reasoning_match.group(1) if reasoning_match else ""
```

**同时修复**: `max_tokens` 从 200 提升至 1024，避免输出被截断

### 5. 修复 LlamaCppVLMProvider 忽略调用者传入的 max_tokens

**文件**: `llama_cpp_vlm_provider.py`

**问题**: `text_chat()` 方法接受 `max_tokens` 参数但从未使用，始终使用 `self.max_tokens`（实例初始化时的值）

**修复**: 修改 `text_chat()` 使用调用者传入的 `max_tokens`

```python
# llama_cpp_vlm_provider.py:333-341
effective_max_tokens = kwargs.get('max_tokens', self.max_tokens)
result = await loop.run_in_executor(
    None,
    lambda: llama.create_chat_completion(
        messages=messages,
        temperature=temp,
        max_tokens=effective_max_tokens,  # 使用传入值而非 self.max_tokens
    )
)
```
