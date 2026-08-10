# -*- coding: utf-8 -*-
"""将 RAGAS testset.json 转换为 LoRA SFT 训练数据（{user, assistant} jsonl）。

RAGAS testset 的每个样本包含：
  - question: MiniMax-M3 生成的问题
  - answer:   MiniMax-M3 生成的参考答案（= 教师回答 / ground_truth）
  - contexts: RAG 检索的论文片段（1~55 个 chunk）

这天然就是教师蒸馏数据：MiniMax-M3（大模型）做老师，Qwen3.5-0.8B（小模型）做学生。
映射成 lora_train.py 需要的 {user, assistant} 格式即可直接训练。

用法：
  python -m finetune.testset_to_sft \
      --testset results/testset.json \
      --out finetune/data/paperqa_sft.jsonl \
      --max-context-chars 4000 \
      --max-length-tokens 2048
"""
import argparse
import json
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """粗估 token 数（中英混合，约 3 字符/token）。"""
    return len(text) // 3


def build_user_prompt(question: str, contexts: list[str], max_context_chars: int) -> str:
    """构造训练用的 user prompt（与 gen_answers.py 风格一致）。

    逐个累加 context chunk，直到达到 max_context_chars 字符配额。
    短样本（绝大多数）完全不截断；只有 multi_hop 拼接了大量 chunk 的极端样本
    才会在配额处停止添加更多 chunk（保留前面的完整 chunk，跳过后面的）。
    """
    context_parts = []
    total = 0
    for ctx in contexts:
        if total + len(ctx) > max_context_chars:
            break  # 达到配额，停止添加更多 chunk（前面的已完整保留）
        context_parts.append(ctx)
        total += len(ctx)

    context_block = "\n\n".join(context_parts)
    return (
        "请基于下面论文中的段落回答问题：\n"
        f"{context_block}\n\n"
        f"问题：{question}"
    )


def convert(
    testset_path: str,
    output_path: str,
    max_context_chars: int,
    max_length_tokens: int,
) -> None:
    with open(testset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"加载 testset: {len(samples)} 个样本 from {testset_path}")

    converted = []
    skipped = 0
    for i, s in enumerate(samples):
        question = s.get("question", "")
        answer = s.get("answer", "")
        contexts = s.get("contexts", [])

        if not question or not answer or not contexts:
            print(f"  [{i}] ⚠️ 跳过（缺字段）: q={bool(question)}, a={bool(answer)}, ctx={len(contexts)}")
            skipped += 1
            continue

        user = build_user_prompt(question, contexts, max_context_chars)
        total_tokens = estimate_tokens(user + answer)

        converted.append({
            "user": user,
            "assistant": answer,
            # 元数据（训练不使用，仅供分析）
            "_meta": {
                "evolution_type": s.get("evolution_type", ""),
                "n_contexts": len(contexts),
                "est_tokens": total_tokens,
                "truncated": total_tokens > max_length_tokens,
            }
        })

    # 统计
    token_lens = [estimate_tokens(c["user"] + c["assistant"]) for c in converted]
    over_limit = sum(1 for t in token_lens if t > max_length_tokens)
    print(f"\n转换完成: {len(converted)} 条训练数据（跳过 {skipped} 条）")
    print(f"token 分布: min={min(token_lens)}, max={max(token_lens)}, avg={sum(token_lens)//len(token_lens)}")
    print(f"超过 max_length={max_length_tokens}: {over_limit}/{len(converted)} 条（训练时自动头部截断）")

    # 按 evolution_type 统计
    from collections import Counter
    types = Counter(c["_meta"]["evolution_type"] for c in converted)
    print(f"类型分布: {dict(types)}")

    # 写入 jsonl（去掉 _meta，lora_train.py 只读 user/assistant）
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(
                {"user": c["user"], "assistant": c["assistant"]},
                ensure_ascii=False,
            ) + "\n")

    print(f"\n✅ 已写入: {out_path} ({len(converted)} 行)")
    print(f"\n下一步训练:")
    print(f"  python -m finetune.lora_train \\")
    print(f"      --data {out_path} \\")
    print(f"      --model_dir finetune/models/Qwen3.5-0.8B \\")
    print(f"      --output_dir finetune/checkpoints/lora-qwen3.5-0.8b-paperqa \\")
    print(f"      --epochs 3 --max_length {max_length_tokens}")


def main():
    p = argparse.ArgumentParser(
        description="将 RAGAS testset.json 转为 LoRA SFT 训练数据"
    )
    p.add_argument(
        "--testset", default="results/testset.json",
        help="RAGAS 生成的测试集路径",
    )
    p.add_argument(
        "--out", default="finetune/data/paperqa_sft.jsonl",
        help="输出的 SFT 训练数据路径",
    )
    p.add_argument(
        "--max-context-chars", type=int, default=4000,
        help="每个样本 context 的最大字符数（截断长 context）",
    )
    p.add_argument(
        "--max-length-tokens", type=int, default=2048,
        help="训练时的 max_length（超出会在训练时头部截断，此处仅用于统计）",
    )
    args = p.parse_args()

    convert(
        testset_path=args.testset,
        output_path=args.out,
        max_context_chars=args.max_context_chars,
        max_length_tokens=args.max_length_tokens,
    )


if __name__ == "__main__":
    main()
