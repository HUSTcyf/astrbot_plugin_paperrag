# -*- coding: utf-8 -*-
"""PaperQA → DPO 偏好对 {prompt, chosen, rejected}（FaithDPO 风格）

对 RAGAS testset 的每个问题构造 1 对：
- prompt   = SFT 同分布的 user prompt（带 RAG 上下文，testset_to_sft.build_user_prompt）
- chosen   = testset 的 answer（MiniMax-M3 教师答案，基于上下文生成 → 高 faithfulness）
- rejected = base 模型无上下文自由生成的答案（raw_answers_finetune_base_norag.json
             中同一问题的回答 → 脱离上下文、易幻觉）

思路（FaithDPO）：
  chosen 是"忠实于上下文"的输出，rejected 是"脱离上下文自由发挥"的输出；
  DPO 让模型在带上下文的 prompt 下学会偏好前者。

用法：
  python -m finetune.gen_dpo_paperqa \
      --testset results/testset.json \
      --rejected results/raw_answers_finetune_base_norag.json \
      --out finetune/data/dpo_paperqa.jsonl \
      --max-context-chars 4000
"""
import argparse
import json
import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

try:
    from finetune.testset_to_sft import build_user_prompt
except ImportError:
    from testset_to_sft import build_user_prompt  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser(description="将 RAGAS testset + base 无上下文答案构造为 DPO 对")
    p.add_argument("--testset", default="results/testset.json")
    p.add_argument("--rejected", default="results/raw_answers_finetune_base_norag.json")
    p.add_argument("--out", default="finetune/data/dpo_paperqa.jsonl")
    p.add_argument("--max-context-chars", type=int, default=4000,
                   help="与 SFT 训练一致的 context 字符配额")
    p.add_argument("--min-rejected-chars", type=int, default=20,
                   help="过滤过短的 rejected（模型没认真答，起不到教学作用）")
    args = p.parse_args()

    with open(args.testset, "r", encoding="utf-8") as f:
        samples = json.load(f)

    with open(args.rejected, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rejected_by_q = {r["question"]: r["answer"] for r in payload["results"]}

    pairs = []
    skipped = {"no_rejected": 0, "short_rejected": 0, "empty_gt": 0}
    for i, s in enumerate(samples):
        question = s.get("question", "")
        chosen = s.get("answer", "")
        contexts = s.get("contexts", [])

        if not question or not chosen:
            skipped["empty_gt"] += 1
            continue
        rejected = rejected_by_q.get(question, "")
        if not rejected:
            skipped["no_rejected"] += 1
            continue
        if len(rejected) < args.min_rejected_chars:
            skipped["short_rejected"] += 1
            continue

        prompt = build_user_prompt(question, contexts, args.max_context_chars)
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p_ in pairs:
            f.write(json.dumps(p_, ensure_ascii=False) + "\n")

    plens = [len(p_["prompt"]) for p_ in pairs]
    clens = [len(p_["chosen"]) for p_ in pairs]
    rlens = [len(p_["rejected"]) for p_ in pairs]
    print(f"✅ DPO pairs: {len(pairs)} -> {out_path}")
    print(f"   跳过: {skipped}")
    print(f"   prompt  chars: min={min(plens)}, avg={sum(plens)//len(plens)}, max={max(plens)}")
    print(f"   chosen  chars: min={min(clens)}, avg={sum(clens)//len(clens)}, max={max(clens)}")
    print(f"   rejected chars: min={min(rlens)}, avg={sum(rlens)//len(rlens)}, max={max(rlens)}")

    print(f"\n下一步训练:")
    print(f"  python -m finetune.train_dpo_official \\")
    print(f"      --data {out_path} \\")
    print(f"      --model_dir finetune/models/Qwen3.5-0.8B \\")
    print(f"      --sft_adapter finetune/checkpoints/lora-qwen3.5-0.8b-paperqa-9070 \\")
    print(f"      --output_dir finetune/checkpoints/lora-qwen3.5-0.8b-paperqa-9070-dpo \\")
    print(f"      --beta 0.1 --lr 1e-5 --epochs 1 --batch_size 1")


if __name__ == "__main__":
    main()
