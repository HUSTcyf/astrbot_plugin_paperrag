# -*- coding: utf-8 -*-
"""微调 LoRA 模型的本地推理 provider（transformers + peft 直连）。

加载范式与 finetune/gen_answers.py、finetune/mc_score.py 完全一致：
    AutoTokenizer → AutoModelForCausalLM → 可选 PeftModel.from_pretrained。

为什么不走 llama-cpp（GGUF LoRA）：
  - finetune 产物是 PEFT .safetensors（adapter_config.json peft_type=LORA），
    而 llama-cpp-python 的 Llama(lora_path=...) 要 GGUF LoRA；仓库内无 safetensors→GGUF 转换。
  - adapter 绑定 Qwen3.5-0.8B（hidden_size=1024/24 层），与生产 VLM Qwen3.5-9B-GGUF 维度不同，
    无法跨基座套用。
为什么不走 Unsloth QLoRA：
  - astrbot 运行环境未安装 unsloth；运行时纯 CPU（torch 是 +cu128 但机器是 AMD GPU），
    0.8B 才 ~1.7GB，4-bit 量化在 CPU 上无收益且 bitsandbytes 内核加载失败。
  - 基座是混合注意力的 Qwen3_5ForConditionalGeneration，Unsloth 对其 LoRA 注入支持不稳。
因此采用与 finetune 脚本同构的 transformers+peft 直连，CPU/GPU 自适应。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any, Optional

from astrbot.api import logger

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent


class FinetuneLLMProvider:
    """本地微调 LoRA 模型推理 provider（transformers + peft）。

    惰性加载：首次 generate 时才 import torch/transformers 并加载模型，
    避免拖慢插件启动（torch 是重依赖）。
    """

    def __init__(
        self,
        base_model_dir: str = "finetune/models/Qwen3.5-0.8B",
        adapter_dir: str = "",
        max_new_tokens: int = 512,
        num_threads: int = 16,
    ):
        """
        Args:
            base_model_dir: HF 格式基座模型目录（相对路径解析到插件根）
            adapter_dir: PEFT LoRA adapter 目录（空=只加载基座，不加 adapter）
            max_new_tokens: 单次生成的最大新 token 数
            num_threads: CPU 推理线程数（GPU 时忽略）
        """
        self.base_model_dir = self._resolve_path(base_model_dir)
        self.adapter_dir = self._resolve_path(adapter_dir) if adapter_dir else ""
        self.max_new_tokens = max_new_tokens
        self.num_threads = num_threads

        self._tokenizer: Any = None
        self._model: Any = None
        self._device: str = ""
        self._initialized = False
        self._lock = asyncio.Lock()

    def _resolve_path(self, path: str) -> str:
        """相对路径解析到插件根目录。"""
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve())
        return str((_PLUGIN_ROOT / candidate).resolve())

    def _load_sync(self) -> None:
        """同步加载模型（在持有 _lock 时于线程池中调用）。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(self.num_threads)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU 用 fp32（精度稳定）；CUDA 用 bf16（速度+显存）
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        logger.info(
            f"[FinetuneLLM] 加载基座 {self.base_model_dir} "
            f"(device={self._device}, dtype={dtype})"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model_dir, dtype=dtype
        ).to(self._device)

        if self.adapter_dir:
            adapter_cfg = Path(self.adapter_dir) / "adapter_config.json"
            if not adapter_cfg.exists():
                raise FileNotFoundError(
                    f"adapter 目录无效（缺 adapter_config.json）: {self.adapter_dir}"
                )
            from peft import PeftModel

            logger.info(f"[FinetuneLLM] 加载 LoRA adapter: {self.adapter_dir}")
            self._model = PeftModel.from_pretrained(self._model, self.adapter_dir)

        self._model.eval()
        self._initialized = True
        logger.info("[FinetuneLLM] ✅ 模型加载完成")

    async def _ensure_loaded(self) -> None:
        """惰性加载（首次调用时，线程池内执行，不阻塞事件循环）。"""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, self._load_sync)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """同步生成（调用前须已加载；未加载会抛 RuntimeError）。

        Args:
            prompt: 用户输入（不含 chat template 包装，本方法用 apply_chat_template 包装）
            max_new_tokens: 覆盖默认最大新 token 数
        """
        if not self._initialized:
            raise RuntimeError("[FinetuneLLM] 模型未加载，请先 await ensure_loaded()")
        return self._generate_inner(prompt, max_new_tokens)

    def _generate_inner(self, prompt: str, max_new_tokens: Optional[int]) -> str:
        """实际生成逻辑（与 finetune/gen_answers.py 同构）。"""
        import torch

        n = max_new_tokens or self.max_new_tokens
        # apply_chat_template(tokenize=True, return_tensors="pt") 返回 BatchEncoding，
        # 必须取 ["input_ids"] 才是 tensor（直接传 BatchEncoding 给 generate 会崩）
        ids = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )["input_ids"].to(self._device)
        with torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=n, do_sample=False)
        return self._tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    async def chat(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """异步生成（线程池执行，不阻塞事件循环——CLAUDE.md 约定）。

        首次调用会触发惰性加载。
        """
        await self._ensure_loaded()
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, self._generate_inner, prompt, max_new_tokens
            )

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    @property
    def model_info(self) -> str:
        """人类可读的模型描述（日志/调试用）。"""
        adapter = f" + adapter {Path(self.adapter_dir).name}" if self.adapter_dir else ""
        return f"{Path(self.base_model_dir).name}{adapter} @ {self._device or '未加载'}"
