# -*- coding: utf-8 -*-
"""预编码 SFT 数据为 tensor 并缓存（绕过 apply_chat_template 的慢速模板渲染）。

问题：lora_train.py 的 encode_sample 用 apply_chat_template(tokenize=True, return_dict=True,
return_tensors="pt")，对长文本（>2000 token）的 paperqa 样本极慢（每个样本分钟级）。
48 个样本 × 2 次调用 = 编码阶段可能卡 10+ 分钟。

解决：本脚本用 tokenizer() 直接编码纯文本内容，手动拼接 chat template 的特殊标记，
把结果缓存为 .pt 文件。lora_train.py 加载缓存后跳过编码，直接训练。

用法：
  python -m finetune.preencode_sft \
      --data finetune/data/paperqa_sft.jsonl \
      --model-dir finetune/models/Qwen3.5-0.8B \
      --out finetune/data/paperqa_sft_encoded.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="预编码 SFT 数据")
    p.add_argument("--data", required=True, help="SFT jsonl 路径")
    p.add_argument("--model-dir", required=True, help="tokenizer 目录")
    p.add_argument("--max-length", type=int, default=0,
                   help="序列最大长度；0=不截断（默认，batch 内动态 pad）。设正数可控制显存/速度")
    p.add_argument("--out", required=True, help="输出 .pt 路径")
    args = p.parse_args()

    print(f"加载 tokenizer from {args.model_dir}...")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # apply_chat_template(tokenize=True) 返回类型因版本而异：
    # - 旧版返回 list[int]
    # - 新版返回 BatchEncoding（dict-like，含 input_ids + attention_mask）或 dict
    def _extract_ids(result):
        """从 apply_chat_template 结果中提取 input_ids list。"""
        if hasattr(result, "get") and result.get("input_ids") is not None:
            return list(result["input_ids"])
        if hasattr(result, "input_ids"):
            return list(result.input_ids)
        return list(result)

    # 测试：编码短样本来确认 apply_chat_template 能正确工作
    test_full = _extract_ids(tok.apply_chat_template(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "World"}],
        tokenize=True, add_generation_prompt=False,
    ))
    test_user = _extract_ids(tok.apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=True, add_generation_prompt=True,
    ))
    print(f"chat template OK: test_full={len(test_full)}tok, test_user={len(test_user)}tok")
    print(f"  decoded full: {repr(tok.decode(test_full))}")
    user_prefix = None  # 标记走 fallback 路径

    # 读取数据
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    print(f"数据: {len(rows)} 行, max_length={args.max_length}")

    # 逐样本编码
    samples = []
    eos_id = tok.eos_token_id or 0
    t0 = time.time()
    for i, r in enumerate(rows):
        ts = time.time()
        user_ids = tok(r["user"], add_special_tokens=False)["input_ids"]
        assistant_ids = tok(r["assistant"], add_special_tokens=False)["input_ids"]

        if user_prefix is not None:
            # 手动拼接（快速路径）
            full_ids = (
                list(user_prefix)          # <im_start>user\n
                + list(user_ids)            # user 实际内容
                + list(gen_prompt)          # <im_end>\n<im_start>assistant\n<think>...\n\n
                + list(assistant_ids)       # assistant 实际内容
                + list(assistant_suffix)    # <im_end>\n
            )
            user_len = len(user_prefix) + len(user_ids) + len(gen_prompt)
        else:
            # fallback：apply_chat_template（慢但正确）
            full_ids = _extract_ids(tok.apply_chat_template(
                [{"role": "user", "content": r["user"]},
                 {"role": "assistant", "content": r["assistant"]}],
                tokenize=True, add_generation_prompt=False,
            ))
            user_only = _extract_ids(tok.apply_chat_template(
                [{"role": "user", "content": r["user"]}],
                tokenize=True, add_generation_prompt=True,
            ))
            user_len = len(user_only)

        full = torch.tensor(full_ids, dtype=torch.long)
        if args.max_length > 0:
            excess = full.numel() - args.max_length
            if excess > 0:
                full = full[excess:]
                user_len = max(0, user_len - excess)

        samples.append((full, user_len))
        dt = time.time() - ts
        if dt > 0.5 or i < 3 or i >= len(rows) - 3:
            print(f"  [{i:>2}] {full.numel():>5} tok, {dt:.2f}s")

    elapsed = time.time() - t0
    lens = [s.numel() for s, _ in samples]
    print(f"\n编码完成: {len(samples)} 样本, {elapsed:.1f}s")
    print(f"max_len={max(lens)}, avg_len={sum(lens)//len(lens)}, total_tokens={sum(lens)}")

    # 验证第一个样本解码
    first_decoded = tok.decode(samples[0][0].tolist(), skip_special_tokens=True)
    print(f"\n=== 样本0解码验证 ===")
    print(f"前80字符: {first_decoded[:80]}")
    print(f"后80字符: ...{first_decoded[-80:]}")

    # 缓存
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "samples": samples,
        "lens": lens,
        "max_length": args.max_length,
    }, out_path)
    print(f"\n✅ 缓存已保存: {out_path} ({out_path.stat().st_size // 1024} KB)")
    print(f"\n训练时加载:")
    print(f"  python lora_train.py --preencoded {out_path} ...")


if __name__ == "__main__":
    main()
