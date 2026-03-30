# PaperRAG 插件变更日志

## 版本 1.7.x - Graph RAG 修复与优化

### 问题描述

Graph RAG 知识图谱构建功能无法正常工作，Qwen3.5 GGUF 模型在构建知识图谱时输出 thinking tokens (`<think>...</think>`) 而非 JSON，导致解析失败。

---

## 核心修复

### 1. Graph Builder - 使用 LlamaCppVLMProvider

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

### 2. 批量处理优化

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

### 3. 修复 add_relation 参数错误

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

### 4. 增加 max_tokens 限制

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

### 5. 移除 thinking tokens

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

### 6. 修复 LlamaCppVLMProvider.system_prompt 未使用

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

### 7. Pure Text RAG 优先使用本地模型

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

### 8. 添加 Chunk 加载进度日志

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
| `graph_builder.py` | 修改 | 批量处理、LlamaCppVLMProvider 集成、修复 add_relation |
| `llama_cpp_vlm_provider.py` | 修改 | 修复 system_prompt 未使用问题 |
| `hybrid_rag.py` | 修改 | 优先使用本地 LlamaCpp 模型 |
| `main.py` | 修改 | 添加 chunk 加载进度日志 |

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

3. **常见问题**:
   - 如果图谱为空，检查 `[Graph-LLM]` 日志中是否有 JSON 解析错误
   - 如果模型加载失败，确认 GGUF 模型文件存在于 `models/` 目录
