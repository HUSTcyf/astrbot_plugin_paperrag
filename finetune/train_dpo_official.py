# -*- coding: utf-8 -*-
"""官方 trl DPOTrainer 跑 DPO：SFT adapter 基础上做偏好对齐

不手写 loss，直接用 HuggingFace 官方实现（trl 1.9.2）。

显存友好设计（官方机制，非 hack）：
- model 传 PeftModel(基础模型 + SFT adapter)，ref_model=None：
  trl 自动创建冻结的 "ref" adapter（SFT adapter 的副本）作为参考模型，
  只有 adapter 参数量，不加载第二个完整模型。
- precompute_ref_log_probs=True：训练前一次性算好全部参考 logprob，
  训练时仅一个模型在显存。

数据：data/dpo_pairs.jsonl，字段 {prompt, chosen, rejected}——trl 原生格式，
chat template 由 trl 自动应用（与 SFT 训练同分布）。

用法：
  python train_dpo_official.py --data data/dpo_pairs.jsonl \
      --model_dir models/Qwen3.5-0.8B \
      --sft_adapter checkpoints/lora-qwen3.5-0.8b-ragtruth \
      --output_dir checkpoints/lora-qwen3.5-0.8b-ragtruth-dpo \
      --beta 0.1 --lr 1e-5 --epochs 1 --batch_size 1
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--sft_adapter", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1, help="显存受限，780M 最多 1")
    p.add_argument("--max_length", type=int, default=4096,
                   help="prompt+chosen+rejected 拼接后的序列最大长度（对齐评测上下文）")
    p.add_argument("--max_steps", type=int, default=-1, help="调试用：提前停止")
    p.add_argument("--precompute_ref_batch_size", type=int, default=2,
                   help="预计算参考 logprob 的 batch（无梯度，可大于训练 batch）")
    p.add_argument("--num_threads", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.set_num_threads(args.num_threads)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)
    print(f"loading SFT adapter: {args.sft_adapter}")
    model = PeftModel.from_pretrained(model, args.sft_adapter)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    print(f"pairs: {len(rows)}")
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"],
                             "rejected": r["rejected"]} for r in rows])

    cfg = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_length=args.max_length,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=args.precompute_ref_batch_size,
        gradient_checkpointing=True,
        bf16=(device == "cuda"),
        max_steps=args.max_steps,
        save_strategy="no",
        logging_steps=5,
        report_to=[],
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg,
                         train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
