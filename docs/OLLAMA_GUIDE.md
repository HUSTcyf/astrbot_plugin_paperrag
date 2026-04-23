# Llama.cpp 本地 VLM 配置指南

使用 Llama.cpp 进行本地多模态推理，支持 GGUF 格式模型 + mmproj 视觉编码器，无需 API Key，完全免费、无限制、隐私保护。

## 安装步骤

### 1. 安装 llama-cpp-python

```bash
# macOS (Metal GPU 加速)
CMAKE_ARGS="-DGGML_METAL=on -DLLAMA_MTMD=on" pip install llama-cpp-python

# Linux (CUDA GPU 加速)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# 仅 CPU
pip install llama-cpp-python
```

### 2. 下载模型文件

推荐使用 Qwen2.5-VL（支持视觉功能）：

```bash
# 创建模型目录
mkdir -p ./models/Qwen3.5-9B-GGUF

# 下载 GGUF 模型（Qwen3.5-9B，支持视觉）
# 模型来源：HuggingFace Qwen 组织
wget https://huggingface.co/Qwen/Qwen3.5-9B-UD-Q4_K_XL-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf

# 下载对应的 mmproj 视觉编码器
wget https://huggingface.co/Qwen/Qwen3.5-9B-UD-Q4_K_XL-GGUF/resolve/main/mmproj-BF16.gguf
```

### 3. 验证安装

```python
from llama_cpp import Llama

# 测试模型加载
llm = Llama(
    model_path="./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf",
    n_ctx=4096,
    n_gpu_layers=99,
)
print("Llama.cpp 模型加载成功")
```

## 插件配置

在插件配置界面或配置文件中设置：

```json
{
  "embedding_mode": "unsloth",
  "multimodal_provider_id": "",
  "llama_vlm_model_path": "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf",
  "llama_vlm_mmproj_path": "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf",
  "llama_vlm_n_ctx": 8192,
  "llama_vlm_n_gpu_layers": 99,
  "llama_vlm_max_tokens": 2560,
  "llama_vlm_temperature": 0.7
}
```

### 配置说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `llama_vlm_model_path` | GGUF 模型文件路径 | `./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf` | 根据实际路径调整 |
| `llama_vlm_mmproj_path` | mmproj 视觉编码器路径 | `./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf` | 与模型配套 |
| `llama_vlm_n_ctx` | 上下文窗口大小 | `8192` | Qwen3.5 支持 262144 |
| `llama_vlm_n_gpu_layers` | GPU 加速层数 | `99` | `99`=全部加载到 GPU |
| `llama_vlm_max_tokens` | 最大生成 Token 数 | `2560` | 根据需求调整 |
| `llama_vlm_temperature` | 生成温度 | `0.7` | 较低=更确定性 |

## 性能优化

### GPU 层数调优

`n_gpu_layers` 控制加载到 GPU 的层数：

- **99 或 -1**：全部加载到 GPU（推荐 Apple Silicon）
- **内存有限**：设置为 `33`（半量）或 `0`（仅 CPU）
- **Windows CUDA**：建议 `99`

### 上下文窗口大小

较大的 `n_ctx` 支持更长上下文，但占用更多内存：

| n_ctx | 内存占用 | 适用场景 |
|-------|----------|----------|
| 4096 | ~8GB | 短文本问答 |
| 8192 | ~12GB | 标准使用 |
| 16384 | ~20GB | 长文档分析 |
| 32768 | ~35GB | 超长上下文 |

### 量化格式选择

| 量化格式 | 内存占用 | 质量损失 | 推荐场景 |
|----------|----------|----------|----------|
| Q4_K_XL | ~5.5GB | 最小 | 推荐使用 |
| Q5_K_M | ~6.5GB | 几乎无 | 内存充足时 |
| Q8_0 | ~9GB | 无 | 本地推理质量最高 |

## 常见问题

### Q: 报错 `model format not supported`

确保使用正确的 GGUF 格式模型，mmproj 文件需要与模型配套。

### Q: 内存不足 (OOM)

1. 降低 `n_gpu_layers`（从 99 → 33 → 0）
2. 减小 `n_ctx`（从 8192 → 4096）
3. 使用更小的量化格式（Q4_K_XL）

### Q: 视觉功能不工作

1. 确认 `mmproj` 文件存在且路径正确
2. 确认使用的是支持视觉的模型（如 Qwen2.5-VL）

### Q: 生成速度慢

1. 确认 GPU 正在被使用（macOS Metal / CUDA）
2. 降低 `n_ctx` 减少计算量
3. 使用更小的量化格式

## 与 API 模式对比

| 特性 | Llama.cpp | API 模式 |
|------|-----------|----------|
| 成本 | 免费 | 按量付费 |
| 隐私 | 数据本地处理 | 数据发送到第三方 |
| 速度 | 依赖本地硬件 | 依赖网络 |
| 限制 | 无 | 有 RPM/TPM 限制 |
| 稳定性 | 依赖本地服务 | 服务商保障 |
| 视觉支持 | 需要 mmproj | 通常内置 |

## 相关文档

- [Llama.cpp 官网](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Qwen 模型库](https://huggingface.co/Qwen)
