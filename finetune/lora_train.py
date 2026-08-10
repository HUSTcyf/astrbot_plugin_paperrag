# -*- coding: utf-8 -*-
"""Qwen3.5-0.8B 领域 LoRA 微调（CPU 可跑）

用法：
  python lora_train.py --data data/train.jsonl \
      --model_dir models/Qwen3.5-0.8B \
      --output_dir checkpoints/lora-qwen3.5-0.8b-vla

产出：
  checkpoints/<name>/adapter_model.safetensors   # LoRA adapter
  checkpoints/<name>/training_log.jsonl          # loss 记录（可用于画图）
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--max_length", type=int, default=0,
                   help="训练序列最大长度；0=不截断（默认，batch 内动态 pad）")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="scheduler warmup 步数（0=不做 warmup）")
    p.add_argument("--clip_grad_norm", type=float, default=1.0,
                   help="梯度裁剪范数（经典范式默认 1.0，设为 0 关闭）")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    p.add_argument("--max_steps", type=int, default=-1, help="调试用：提前停止")
    p.add_argument("--num_threads", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=1, help="显存受限，780M 最多 2")
    p.add_argument("--ckpt-every", type=int, default=100, help="每多少步保存断点（断电/卡死可续跑）")
    p.add_argument("--preencoded", default="", help="预编码 .pt 文件路径（跳过 apply_chat_template 编码）")
    p.add_argument("--restart", action="store_true", help="忽略已有断点，从头训练")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(42)

    if args.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        if device == "cpu" and args.dtype == "bf16":
            print("note: bf16 on CPU 慢且精度差，建议用 fp32")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    print(f"data: {len(rows)} rows, dtype={args.dtype}")

    if args.preencoded and Path(args.preencoded).exists():
        # 预编码缓存（绕过 apply_chat_template 的慢速编码）
        print(f"加载预编码缓存: {args.preencoded}")
        cached = torch.load(args.preencoded, map_location="cpu", weights_only=False)
        samples = cached["samples"]
        lens = cached["lens"]
        print(f"✅ 预编码加载完成: {len(samples)} 样本 (跳过 apply_chat_template)")
    else:
        def encode_sample(r):
            def ids(chat, gen_prompt):
                return tokenizer.apply_chat_template(
                    chat, tokenize=True, add_generation_prompt=gen_prompt,
                    return_dict=True, return_tensors="pt",
                )["input_ids"][0]
            # user 部分编码到 assistant 起始标记处，labels 据此 mask 掉指令（只对答案算 loss）
            user_len = ids([{"role": "user", "content": r["user"]}], True).numel()
            full = ids([{"role": "user", "content": r["user"]},
                        {"role": "assistant", "content": r["assistant"]}], False)
            if args.max_length > 0:
                # 头部全是 user 的 context 前部：截掉 excess 个，保证问题与答案完整保留
                excess = full.numel() - args.max_length
                if excess > 0:
                    full = full[excess:]
                    user_len = max(0, user_len - excess)
            return full, user_len

        samples = [encode_sample(r) for r in rows]
        lens = [s.numel() for s, _ in samples]
    print(f"max_len={max(lens)}, avg_len={sum(lens)//len(lens)}, "
          f"total_tokens_1epoch={sum(lens)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_path = Path(args.output_dir) / "training_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")

    # 断点续跑：检测到 resume_ckpt.pt 且未指定 --restart 时恢复
    ckpt_path = Path(args.output_dir) / "resume_ckpt.pt"
    start_step, base_elapsed = 0, 0.0
    if not args.restart and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["optimizer"])
        start_step, base_elapsed = ckpt["step"], ckpt["base_elapsed"]
        print(f"RESUME from step {start_step} (previous elapsed {base_elapsed:.0f}s)")

    n_steps = start_step
    t0 = time.time() - base_elapsed  # elapsed 计算自动含历史时长
    pad_id = tokenizer.pad_token_id or 0
    bs = args.batch_size

    # 总步数 = epochs × 每 epoch 的 batch 数（与训练循环的 range 语义一致）
    n_batches_per_epoch = (len(samples) + bs - 1) // bs
    total_steps = args.epochs * n_batches_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )
    # 断点续跑时把 scheduler 步进到已完成的步数（lr 曲线连续）
    for _ in range(start_step):
        scheduler.step()
    if start_step:
        print(f"scheduler restored: lr={scheduler.get_last_lr()[0]:.2e}")

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = n_batches_per_epoch
        for b in range(start_step, n_batches):
            chunk = samples[b * bs:(b + 1) * bs]
            L = max(s.numel() for s, _ in chunk)
            input_ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            labels = torch.full_like(input_ids, -100)
            for j, (s, user_len) in enumerate(chunk):
                input_ids[j, :s.numel()] = s
                labels[j, user_len:s.numel()] = s[user_len:]
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = (input_ids != pad_id).long()
            out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = out.loss
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            opt.step()
            scheduler.step()
            opt.zero_grad()
            total_loss += loss.item()
            n_steps += 1
            if n_steps % 10 == 0:
                el = time.time() - t0
                rec = {"step": n_steps, "epoch": epoch + 1, "loss": round(loss.item(), 4),
                       "elapsed_s": round(el), "tok_per_s": round(n_steps * max(lens) / el, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"step {n_steps} | loss {loss.item():.4f} | {rec['tok_per_s']} tok/s | {el:.0f}s")
            if n_steps % args.ckpt_every == 0:
                ckpt = {"step": n_steps, "base_elapsed": time.time() - t0,
                        "model": {n: p.detach().cpu() for n, p in model.named_parameters()
                                  if p.requires_grad},
                        "optimizer": opt.state_dict()}
                tmp = Path(args.output_dir) / "resume_ckpt.pt.tmp"
                torch.save(ckpt, tmp)
                os.replace(tmp, ckpt_path)
                print(f"  ckpt saved (step {n_steps})")
            if 0 < args.max_steps <= n_steps:
                print(f"max_steps reached, early stop")
                break
        print(f"epoch {epoch+1} done, avg loss {total_loss/max(n_batches,1):.4f}")
        if 0 < args.max_steps <= n_steps:
            break

    # 保存 adapter + 最终汇总
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    summary = {"rows": len(rows), "steps": n_steps, "elapsed_s": round(time.time() - t0),
               "lr": args.lr, "r": args.r, "alpha": args.alpha, "dtype": args.dtype,
               "warmup_steps": args.warmup_steps, "clip_grad_norm": args.clip_grad_norm,
               "max_length": args.max_length}
    log.write(json.dumps({"summary": summary}) + "\n")
    log.close()
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("resume checkpoint removed (training complete)")
    print(f"done: {n_steps} steps, {summary['elapsed_s']}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
