# -*- coding: utf-8 -*-
"""TruthfulQA (generation) → 训练集 + 评测集（事实性问答，官方 GPT-judge 评测口径）

- 训练：817 条切 n-train（默认 717）
- 评测：其余 100 条（seed 42 固定，无泄漏）

训练格式（对齐 lora_train.py）：{user, assistant}
评测格式（对齐 gen_answers/score_official）：{question, contexts, ground_truth, prompt}
"""
import argparse
import json

from datasets import load_dataset

PROMPT = "请回答下面这个问题：{q}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=717)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    rows = ds.shuffle(seed=args.seed)
    print(f"total {len(rows)} rows: train {args.n_train} + eval {len(rows)-args.n_train}")

    def answer(r):
        return (r["best_answer"] or r["correct_answers"][0]).strip()

    with open("data/truthfulqa_train.jsonl", "w", encoding="utf-8") as f:
        for r in rows.select(range(args.n_train)):
            q = r["question"].strip()
            f.write(json.dumps({"user": PROMPT.format(q=q), "assistant": answer(r)},
                               ensure_ascii=False) + "\n")

    with open("data/truthfulqa_eval.jsonl", "w", encoding="utf-8") as f:
        for r in rows.select(range(args.n_train, len(rows))):
            q = r["question"].strip()
            f.write(json.dumps({"question": q, "contexts": [],
                                "ground_truth": answer(r),
                                "prompt": PROMPT.format(q=q),
                                "best_answer": r["best_answer"],
                                "correct_answers": r["correct_answers"],
                                "incorrect_answers": r["incorrect_answers"]},
                               ensure_ascii=False) + "\n")

    print(f"train {args.n_train} -> data/truthfulqa_train.jsonl")
    print(f"eval  {len(rows)-args.n_train} -> data/truthfulqa_eval.jsonl")


if __name__ == "__main__":
    main()
