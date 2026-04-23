# Test Chunk Tokenization - 测试说明

## 测试脚本位置
`test/test_chunk_tokenization.py`

## 运行方式

```bash
# 运行全部测试
.venv/bin/python3 test/test_chunk_tokenization.py

# 运行单个测试 (-t 1/2/3/4)
.venv/bin/python3 test/test_chunk_tokenization.py -t 4

# 运行多个测试
.venv/bin/python3 test/test_chunk_tokenization.py -t 1 -t 3
```

## 测试内容

| 测试 | 名称 | 说明 |
|------|------|------|
| Test1 | chunk_tokenization | 验证分块 token 数不超 512 |
| Test2 | colbert_no_truncation | 验证 ColBERT storage 不截断 |
| Test3 | e2e_pdf | 端到端 PDF 解析测试（含 LLM 预处理） |
| Test4 | local_llm_preprocessing | 本地 LLM 紧凑化测试（合成文本） |

## 关键修复记录

### 1. Tokens 单位统一
- 每个段落同时统计 **BGE tokens**（用于分块判断）和 **llama tokens**（用于 LLM 上下文限制）
- 不再混用两种单位

### 2. bin_capacity 改为 8192
- 与 llama n_ctx 一致，避免溢出

### 3. 安全边界改为 0
- 实测不需要额外安全边界

### 4. 溢出重试机制
- 首次溢出时自动将 max_output_tokens 减半重试

### 5. 短块合并策略
- `_post_process_chunks` 强制合并 <128 tokens 的短块
- 合并后如果超过 512 tokens，后续会通过 `_split_long_text` 拆分

### 6. min_chunk_size 默认值
- 从 100 改为 128

### 7. Llama 单例关闭
- `reset_llama_cpp_vlm_provider()` 正确调用 `llama.close()`

## 常见问题

**llama_decode returned -3**
- 原因：prompt + output 超过 n_ctx
- 解决：减少 max_output_tokens，或减小 bin 容量

**短块 (<100 tokens)**
- 原因：PDF 解析产生的碎片段落
- 解决：强制合并到相邻块