# -*- coding: utf-8 -*-
"""RAGTruth QA 子集 → LoRA 训练集 + 评测集

官方用法（README）：RAGTruth 语料 "both for training and evaluating"。
QA 子集：989 个源 × 6 模型回答，人工逐句标注幻觉 span（conflicting/fabrication/irrelevant）。

本脚本：
- 训练集：split=train + quality=good + 零幻觉标注的回答（正样本，"忠实回答"示例），
  user 直接用官方 prompt（含 "Unable to answer based on given passages" 拒答指令）
- 评测集：split=test + quality=good + 零幻觉标注，question 用官方 question 字段，
  contexts 用官方 passages（供 faithfulness 评估）

输出：
  data/ragtruth_train.jsonl    LoRA 训练集
  data/ragtruth_eval.jsonl     评测集（question/contexts/ground_truth）

用法：python ragtruth_prep.py [--n-train 200] [--n-eval 60] [--seed 42]
"""
import argparse
import json
import random

RESP = "RAGTruth/dataset/response.jsonl"
SRC = "RAGTruth/dataset/source_info.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-eval", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", help="只使用指定来源模型的回答（如 gpt-4-0613 单源教师）")
    args = p.parse_args()

    srcs = {}
    for l in open(SRC, encoding="utf-8"):
        s = json.loads(l)
        if s["task_type"] == "QA":
            srcs[s["source_id"]] = s
    rng = random.Random(args.seed)

    train, eval_rows = [], []
    for l in open(RESP, encoding="utf-8"):
        r = json.loads(l)
        s = srcs.get(r["source_id"])
        if s is None or r["quality"] != "good" or r["labels"]:
            continue
        if args.model and r["model"] != args.model:
            continue
        info = s["source_info"]
        row = {"type": "context_qa", "user": s["prompt"], "assistant": r["response"]}
        erow = {"question": info["question"], "contexts": [info["passages"]],
                "ground_truth": r["response"], "prompt": s["prompt"]}
        (train if r["split"] == "train" else eval_rows).append((row, erow))

    print(f"usable: train={len(train)} test={len(eval_rows)}")
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    train = train[:args.n_train]
    eval_rows = eval_rows[:args.n_eval]

    with open("data/ragtruth_train.jsonl", "w", encoding="utf-8") as f:
        for row, _ in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open("data/ragtruth_eval.jsonl", "w", encoding="utf-8") as f:
        for _, erow in eval_rows:
            f.write(json.dumps(erow, ensure_ascii=False) + "\n")
    tl = [len(r["assistant"]) for r, _ in train]
    ul = [len(r["user"]) for r, _ in train]
    print(f"saved: data/ragtruth_train.jsonl ({len(train)} rows, "
          f"user_avg={sum(ul)//len(ul)} chars, answer_avg={sum(tl)//len(tl)} chars), "
          f"data/ragtruth_eval.jsonl ({len(eval_rows)} rows)")


if __name__ == "__main__":
    main()
