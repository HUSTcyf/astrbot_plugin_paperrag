# -*- coding: utf-8 -*-
"""TruthfulQA → DPO 偏好对 {prompt, chosen, rejected}

每问 1 对：chosen = best_answer（真答案），rejected = incorrect_answers[0]（幻觉答案）。
训练/评测划分与 SFT 一致（前 715 问训练，后 102 问评测，无泄漏）。

用法：
  python gen_dpo_truthfulqa.py --n-train 715 [--max-pairs 715]
"""
import argparse
import json

from datasets import load_dataset

PROMPT = "请回答下面这个问题：{q}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=715)
    p.add_argument("--max-pairs", type=int, default=715, help="DPO 对上限（时间紧可减量）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    rows = ds.shuffle(seed=args.seed)
    pairs = []
    for r in rows.select(range(args.n_train)):
        chosen = (r["best_answer"] or r["correct_answers"][0]).strip()
        rejected = r["incorrect_answers"][0].strip()
        if not chosen or not rejected:
            continue
        pairs.append({"prompt": PROMPT.format(q=r["question"].strip()),
                      "chosen": chosen, "rejected": rejected})
        if len(pairs) >= args.max_pairs:
            break

    with open("data/dpo_truthfulqa.jsonl", "w", encoding="utf-8") as f:
        for p_ in pairs:
            f.write(json.dumps(p_, ensure_ascii=False) + "\n")
    print(f"dpo pairs: {len(pairs)} -> data/dpo_truthfulqa.jsonl")


if __name__ == "__main__":
    main()
