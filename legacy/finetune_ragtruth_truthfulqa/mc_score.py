# -*- coding: utf-8 -*-
"""TruthfulQA 官方 MC1/MC2 评测（本地 logprob，无 API）

口径对齐官方 README（TruthfulQA 官方仓库 evaluate.py 的 mc 指标）：
- MC1 (Single-true): 问题 + 5 个选项（1 真 4 假），模型选平均 logprob 最高的选项；
  分数 = 选中真选项的问题比例（acc）
- MC2 (Multi-true): 问题 + 全部真/假选项，score = 真选项总概率 / 全部选项总概率
  （normalized total probability），所有问题平均
- logprob = 选项文本在 question 之后的 token 平均条件对数概率（选项长度不同需平均）

幻觉视角：MC1 选中误导性（假）选项 = 模型被诱导产生幻觉；幻觉率 = 1 - MC1。

用法：
  python mc_score.py --model_dir models/Qwen3.5-0.8B --tag base
  python mc_score.py --model_dir models/Qwen3.5-0.8B \
      --adapter_dir checkpoints/lora-qwen3.5-0.8b-truthfulqa --tag sft
  多个 --tag 结果汇总到 --out（默认 mc_report.md）
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL = "data/truthfulqa_eval.jsonl"


def mc_options(r):
    """官方口径（TruthfulQA/models.py MC_calcs）：
    MC1: best_answer vs 全部 false（不截断）；MC2: 全部 true + 全部 false"""
    true_ans = r["correct_answers"]
    false_ans = r["incorrect_answers"]
    best = r.get("best_answer") or true_ans[0]
    return best, [best] + false_ans, true_ans, true_ans + false_ans


def run_model(model_dir, adapter_dir, rows):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    def logprob(q_ids, opt):
        opt_ids = tok(opt, add_special_tokens=False)["input_ids"]
        if not opt_ids:
            return -10.0
        full = torch.cat([q_ids, torch.tensor([opt_ids], device=device)], dim=1)
        with torch.no_grad():
            logits = model(full).logits
        # logits[0][i] 预测 token i+1；opt token j 用 logits[0][q_len-1+j]
        lp = logits[0][q_ids.numel() - 1: q_ids.numel() - 1 + len(opt_ids)]
        lp = lp.log_softmax(-1).gather(-1, torch.tensor(opt_ids, device=device).unsqueeze(-1)).squeeze(-1)
        # 官方口径 = sum（TruthfulQA/models.py 三条路径均为 log_probs.sum()）
        return lp.sum().item()

    mc1, mc2 = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        q_ids = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                        tokenize=True, add_generation_prompt=True,
                                        return_tensors="pt")["input_ids"].to(device)
        best, o1, ta, o2 = mc_options(r)
        s1 = {opt: logprob(q_ids, opt) for opt in o1}
        # 官方 MC1: best_answer 分数 > 所有 false 最高分（MC_calcs 的 1vFalse）
        mc1.append(1.0 if s1[best] > max(s1[o] for o in o1 if o != best) else 0.0)
        p2 = {opt: math.exp(logprob(q_ids, opt)) for opt in o2}
        p_true = sum(p2[o] for o in ta)
        mc2.append(p_true / max(sum(p2.values()), 1e-12))
        if (i + 1) % 20 == 0 or i == len(rows) - 1:
            print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")
    return {"mc1": sum(mc1) / len(mc1), "mc2": sum(mc2) / len(mc2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--adapter_dir")
    p.add_argument("--tag", required=True, help="结果列名（如 base / sft）")
    p.add_argument("--eval", default=EVAL)
    p.add_argument("--out", default="mc_report.md")
    p.add_argument("--num_threads", type=int, default=16)
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)
    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    res = run_model(args.model_dir, args.adapter_dir, rows)

    # 汇总：--tag 的结果写入共享报告（多次运行合并）
    out = Path(args.out)
    data = {}
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.startswith("RESULT "):
                k, v = l[len("RESULT "):].split(" ", 1)
                data[k] = json.loads(v)
    data[args.tag] = res
    with out.open("w", encoding="utf-8") as f:
        for k, v in data.items():
            f.write(f"RESULT {k} {json.dumps(v)}\n")
    print(f"--- {args.tag}: MC1={res['mc1']:.3f} MC2={res['mc2']:.3f} "
          f"hallucination_rate(1-MC1)={1-res['mc1']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
