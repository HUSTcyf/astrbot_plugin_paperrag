# -*- coding: utf-8 -*-
"""在 held-out eval 集上生成模型回答（base 或 base+LoRA）

输入：data/eval_ragas.jsonl（问题 + reference_contexts + reference）
输出：answers_<tag>.jsonl —— {question, answer, contexts, ground_truth}

用法：
  python gen_answers.py --model_dir models/Qwen3.5-0.8B --out answers_base.jsonl
  python gen_answers.py --model_dir models/Qwen3.5-0.8B \
      --adapter_dir checkpoints/lora-qwen3.5-0.8b-vla --out answers_lora.jsonl
"""
import argparse
import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL = "data/eval_ragas.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--adapter_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--eval", default=EVAL)
    p.add_argument("--max_new", type=int, default=192)
    p.add_argument("--num_threads", type=int, default=16)
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device: {device} dtype: {dtype}")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype).to(device)
    if args.adapter_dir:
        print(f"loading adapter: {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    print(f"eval rows: {len(rows)}")
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            q = r.get("question") or r.get("user_input")
            ctx = "\n".join(r.get("contexts") or r.get("reference_contexts") or [])
            # 评测 prompt 优先用数据自带的原始 prompt（与训练分布一致），否则用通用模板
            prompt = r.get("prompt") or ("请基于下面论文中的段落回答问题：\n" + ctx + "\n\n问题：" + q)
            # apply_chat_template(tokenize=True, return_tensors="pt") 返回 BatchEncoding，
            # 必须取 ["input_ids"] 才是 tensor，直接传 BatchEncoding 给 generate 会崩
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
            ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            f.write(json.dumps({"question": q, "answer": ans,
                                "contexts": r.get("contexts") or r.get("reference_contexts") or [],
                                "ground_truth": r.get("ground_truth") or r.get("reference")},
                               ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 5 == 0 or i == len(rows) - 1:
                print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")
    print(f"done -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
