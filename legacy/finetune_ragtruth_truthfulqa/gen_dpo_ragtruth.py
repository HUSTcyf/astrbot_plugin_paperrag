# -*- coding: utf-8 -*-
"""RAGTruth QA → DPO 偏好对（跨模型偏好蒸馏）

每个 train 源 1 对：
- chosen  = gpt-4-0613 的 good + 零幻觉回答（clean，教师回答）
- rejected = 同源其他模型中幻觉标注最多（labels 最长）的回答
源级隔离：只用 split=train 的源（评测 150 源零泄漏）。
prompt 用官方 prompt（含 passages，与 SFT 训练同分布）。

用法：python gen_dpo_ragtruth.py [--max-pairs 625] → data/dpo_ragtruth.jsonl
"""
import argparse
import json

RESP = "RAGTruth/dataset/response.jsonl"
SRC = "RAGTruth/dataset/source_info.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-pairs", type=int, default=625, help="对上限（每源 1 对，共 625）")
    args = p.parse_args()

    qa_srcs = {}
    for l in open(SRC, encoding="utf-8"):
        s = json.loads(l)
        if s["task_type"] == "QA":
            qa_srcs[s["source_id"]] = s

    by_src = {}
    for l in open(RESP, encoding="utf-8"):
        r = json.loads(l)
        if r["source_id"] in qa_srcs and r["split"] == "train":
            by_src.setdefault(r["source_id"], []).append(r)

    pairs = []
    for sid, rs in by_src.items():
        clean = [r for r in rs if r["model"] == "gpt-4-0613"
                 and r["quality"] == "good" and not r["labels"]]
        if not clean:
            continue
        hall = [r for r in rs if r["model"] != "gpt-4-0613" and r["labels"]]
        if not hall:
            continue
        rejected = max(hall, key=lambda r: len(r["labels"]))
        pairs.append({"prompt": qa_srcs[sid]["prompt"],
                      "chosen": clean[0]["response"],
                      "rejected": rejected["response"]})
        if len(pairs) >= args.max_pairs:
            break

    with open("data/dpo_ragtruth.jsonl", "w", encoding="utf-8") as f:
        for p_ in pairs:
            f.write(json.dumps(p_, ensure_ascii=False) + "\n")
    lens = [len(p_["prompt"]) + max(len(p_["chosen"]), len(p_["rejected"])) for p_ in pairs]
    print(f"dpo pairs: {len(pairs)} -> data/dpo_ragtruth.jsonl "
          f"(prompt+answer chars max={max(lens)})")


if __name__ == "__main__":
    main()
