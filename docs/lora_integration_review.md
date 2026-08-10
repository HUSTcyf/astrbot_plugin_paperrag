# LoRA 微调模型接入 — 当前实现说明（FinetuneLLMProvider）

> 微调模型（Qwen3.5-0.8B LoRA）接入 AstrBot 运行时的最终方案与落地状态。
> 早期被否决的方案（llama-cpp GGUF LoRA、Unsloth QLoRA）均已回滚，仅存于 git 历史。

## 一、背景：为什么不走 llama-cpp（GGUF LoRA）

finetune 产物是 PEFT `.safetensors`（`adapter_config.json`：`peft_type=LORA`），
而 llama-cpp-python 的 `Llama(lora_path=...)` 要的是 **GGUF LoRA**（由 llama.cpp 的
`convert_lora_to_gguf.py` 产出）。仓库内无任何 safetensors→GGUF 转换，因此
`.safetensors` 无法被 `Llama(lora_path=)` 加载。llama-cpp 路线整体放弃，
`provider/llama_cpp_vlm.py` 中此前的 `lora_path` 参数已回滚删除。

## 二、为什么不走 Unsloth QLoRA

| 核查点 | 结论 |
|---|---|
| astrbot 运行环境装了 unsloth 吗 | 没装（`ModuleNotFoundError`） |
| 运行时有 GPU 吗 | 没有——astrbot env 的 torch 是 `+cu128`(CUDA 版)，但机器是 AMD GPU，`cuda.is_available()=False`，纯 CPU |
| 4-bit 量化在 CPU 上有意义吗 | 无——0.8B 才 ~1.7GB，4-bit 无收益，且 bitsandbytes 4-bit 内核在该 env 加载失败 |
| Unsloth 支持这个基座吗 | 有风险——`Qwen3_5ForConditionalGeneration` 是混合注意力（linear/full 交替），自定义 `target_modules`，Unsloth 自动映射不稳 |
| 插件已有的范式 | `embedding/unsloth_embedding.py` 已是"unsloth ImportError → transformers 直连"回退；为 LM 引入硬 unsloth 依赖会破坏这个设计 |

**结论**：放弃 Unsloth QLoRA。

## 三、采用方案：独立 transformers+peft 加载器（已落地）

与 finetune 脚本同构（`AutoTokenizer` → `AutoModelForCausalLM` → 可选 `PeftModel.from_pretrained`），
提取为插件可复用的 provider。

### 1. `provider/finetune_llm_provider.py`（新增）— `FinetuneLLMProvider`

- 惰性加载：首次 `chat()` 才 import torch/transformers 并加载模型，不拖慢插件启动。
- `async chat()`：`_ensure_loaded` + 线程池执行 generate，**不阻塞事件循环**（CLAUDE.md 约定）。
- `generate()`：同步路径（须先 `await ensure_loaded()`）。
- CPU 用 fp32（精度稳定）；CUDA 用 bf16（速度+显存）。
- adapter 目录校验（缺 `adapter_config.json` 抛 FileNotFoundError）。
- 相对路径解析到插件根目录。

### 2. 配置项（`_conf_schema.json` + `RAGConfig` 两处同步）

| 配置键 | 类型/默认 | 说明 |
|---|---|---|
| `finetune_llm_enabled` | bool / false | 是否启用微调模型 provider（默认关，仅评测对比用） |
| `finetune_base_model_dir` | str / `finetune/models/Qwen3.5-0.8B` | HF 格式基座目录 |
| `finetune_adapter_dir` | str / `""` | PEFT LoRA adapter 目录（空=只加载基座） |
| `finetune_llm_max_new_tokens` | int / 512 | 单次生成最大新 token 数 |

（`_conf_schema.json` 定义在 WebUI 可见；`RAGConfig` 在 `rag/rag_engine.py:42-44` 镜像。）

### 3. `commands/base.py` — lazy 单例接线

- `PluginCoreBase._finetune_llm` + `_finetune_llm_lock`（`threading.Lock`，仿 `_get_engine` 模式）。
- `_get_finetune_llm()`：仅当 `finetune_llm_enabled=true` 时实例化，否则返回 None；
  双线程锁防重入；实例化时打印 `model_info`。

### 4. 接入点定位

微调模型（0.8B）在论文问答上质量远不如生产 9B VLM，**不作为默认生成器**，
仅作为评测对比对象（finetune→评测闭环的真实意义）。

## 四、落地状态核对（2026-08-10）

| 检查项 | 状态 |
|---|---|
| `provider/finetune_llm_provider.py` 存在 | ✅ 已创建 |
| `_conf_schema.json` 新增 finetune 配置项 | ✅ `:102-128` |
| `RAGConfig` 镜像字段 | ✅ `rag_engine.py:42-45` |
| `commands/base.py` lazy 单例 `_get_finetune_llm` | ✅ `:233-259` |
| `llama_cpp_vlm.py` 的 `lora_path` 死代码 | ✅ 已回滚删除 |
| `_conf_schema.json` 的 `llama_vlm_lora_path` 配置 | ✅ 已删除 |
| 评测脚本对接（`finetune/eval_on_ragtestset.py` 用 FinetuneLLMProvider） | ✅ |
