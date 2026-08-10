# -*- coding: utf-8 -*-
"""在 RAGAS 测试集上对比 base / base+LoRA 模型的回答质量。

闭环思路（与 finetune/gen_answers.py 同构，但对接 evaluation/ 的 raw_answers 格式）：
  1. 从 --step evaluate（系统 LLM）产出的 raw_answers.json 中提取
     question + contexts + ground_truth（复用同一批 RAG 检索结果，保证公平对比）。
  2. 用 FinetuneLLMProvider（0.8B base 或 base+LoRA）重新生成 answer。
  3. 输出 raw_answers_finetune_<tag>.json（格式与 raw_answers.json 完全一致）。
  4. 用 --step evaluate --skip-rag 对三组答案统一打分对比。

为什么不直接用 gen_answers.py：
  - gen_answers.py 的输入是 data/eval_ragas.jsonl（需手动准备 question+contexts），
    而本脚本直接从 raw_answers.json 读取，省去格式转换。
  - gen_answers.py 是纯同步批处理；本脚本用 FinetuneLLMProvider 的 async chat()，
    虽然底层还是单模型串行 generate，但接口与插件 provider 一致，便于后续复用。

用法：
  # 前提：已通过 --step evaluate 生成 results/raw_answers.json（含 question+contexts）

  # base 模型（Qwen3.5-0.8B，不加 LoRA）
  python -m finetune.eval_on_ragtestset \
      --raw-answers results/raw_answers.json \
      --base-model-dir finetune/models/Qwen3.5-0.8B \
      --tag base \
      --out results/raw_answers_finetune_base.json

  # base + LoRA adapter（以 ragtruth DPO 为例）
  python -m finetune.eval_on_ragtestset \
      --raw-answers results/raw_answers.json \
      --base-model-dir finetune/models/Qwen3.5-0.8B \
      --adapter-dir finetune/checkpoints/lora-qwen3.5-0.8b-ragtruth-gpt4-dpo \
      --tag lora_ragtruth_dpo \
      --out results/raw_answers_finetune_lora_ragtruth_dpo.json

  # 之后统一打分对比：
  python -m evaluation.run_evaluation_ragas --step evaluate --skip-rag \
      --raw-answers-path results/raw_answers_finetune_base.json
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# 插件根目录加入 sys.path（standalone 脚本范式）
_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

# 双重 import 模式（CLAUDE.md 约定）
try:
    from provider.finetune_llm_provider import FinetuneLLMProvider
except ImportError:
    from finetune_llm_provider import FinetuneLLMProvider  # type: ignore

# 复用 evaluation/ 的 git 信息 + raw_answers 格式工具
try:
    from evaluation.ragas_evaluator import load_raw_answers, _get_git_info
except ImportError:
    from ragas_evaluator import load_raw_answers, _get_git_info  # type: ignore


def build_prompt(question: str, contexts: list[str], no_context: bool = False) -> str:
    """构造回答 prompt（与 RAGQueryWrapper._generate_answer 的 prompt 风格一致）。

    用 system LLM 评测时的 prompt 模板：
      "You are answering a research question using excerpts from academic papers..."
    这里改为中文模板，与 finetune/gen_answers.py 的训练分布更接近（训练数据用中文 prompt）。
    no_context=True：不注入检索上下文，直接回答问题（测原始模型能力）。
    """
    if no_context:
        return question
    context_block = "\n\n".join(c for c in contexts if c)
    return (
        "请基于下面论文中的段落回答问题：\n"
        f"{context_block}\n\n"
        f"问题：{question}"
    )


async def run(
    raw_answers_path: str,
    output_path: str,
    base_model_dir: str,
    adapter_dir: str,
    max_new_tokens: int,
    num_threads: int,
    tag: str,
    no_context: bool = False,
) -> None:
    # 1. 加载 raw_answers.json（复用 evaluation 的 loader，兼容新旧格式）
    raw_data, meta = load_raw_answers(raw_answers_path)
    print(f"✅ 加载 {len(raw_data)} 个样本 from {raw_answers_path}")
    if meta:
        commit_short = meta.get("git_commit", "unknown")[:8]
        print(f"   原始元数据: commit={commit_short}, llm={meta.get('llm_model', '?')}")

    # 2. 初始化 FinetuneLLMProvider（惰性加载，首次 chat 时触发）
    provider = FinetuneLLMProvider(
        base_model_dir=base_model_dir,
        adapter_dir=adapter_dir,
        max_new_tokens=max_new_tokens,
        num_threads=num_threads,
    )
    print(f"🔧 FinetuneLLMProvider 配置:")
    print(f"   base_model_dir: {base_model_dir}")
    print(f"   adapter_dir: {adapter_dir or '(无，纯 base)'}")
    print(f"   max_new_tokens: {max_new_tokens}")
    print(f"   tag: {tag}")
    print(f"   no_context: {no_context}")

    # 3. 逐样本生成（本地模型 CPU 推理慢，不并发，顺序处理 + 增量保存）
    git_info = _get_git_info()
    base_payload = {
        "_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_info["commit"],
            "git_commit_date": git_info["commit_date"],
            "git_dirty": git_info["dirty"],
            "total_samples": len(raw_data),
            "success_count": 0,
            "mode": "rag" if not no_context else "no_context",
            "llm_model": f"finetune:{Path(base_model_dir).name}"
                         + (f"+{Path(adapter_dir).name}" if adapter_dir else ""),
            "llm_base_url": "local",
            "embedding_model": "(reused from baseline)",
            "embed_base_url": "(reused from baseline)",
            "answer_top_k": meta.get("answer_top_k", "?"),
            "max_concurrent": 1,
            "finetune_tag": tag,
            "no_context": no_context,
            "source_raw_answers": raw_answers_path,
        },
        "results": [],
    }

    # 先写入初始状态（增量保存范式，与 ragas_evaluator.evaluate 一致）
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(base_payload, f, ensure_ascii=False, indent=2)

    results: list[dict | None] = [None] * len(raw_data)
    success_count = 0
    t0 = time.time()

    for i, row in enumerate(raw_data):
        question = row["question"]
        contexts = row.get("contexts", [])
        ground_truth = row.get("ground_truth", "")
        prompt = build_prompt(question, contexts, no_context)

        try:
            answer = await provider.chat(prompt)
        except Exception as e:
            print(f"  [{i+1}/{len(raw_data)}] ❌ 生成失败: {e}")
            continue

        if not answer or not answer.strip():
            print(f"  [{i+1}/{len(raw_data)}] ⚠️ 空回答，跳过")
            continue

        results[i] = {
            "question": question,
            "answer": answer,
            "contexts": [] if no_context else contexts,
            "ground_truth": ground_truth,
            "latency_ms": 0,  # 本地推理延迟不记录（与 RAG 网络延迟无意义对比）
            "question_type": row.get("question_type", "unknown"),
            "has_multimodal": row.get("has_multimodal", False),
        }
        success_count += 1

        # 增量保存
        base_payload["_metadata"]["success_count"] = success_count
        base_payload["results"] = [r for r in results if r is not None]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(base_payload, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        avg = elapsed / success_count
        remaining = avg * (len(raw_data) - success_count)
        print(f"  [{i+1}/{len(raw_data)}] ✅ ({success_count} done, "
              f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    print(f"\n{'='*60}")
    print(f"🎉 完成: {success_count}/{len(raw_data)} 样本")
    print(f"   总耗时: {time.time()-t0:.0f}s")
    print(f"   输出: {out_path}")
    print(f"   模型: {provider.model_info}")
    print(f"{'='*60}")
    print(f"\n下一步打分对比:")
    print(f"  python -m evaluation.run_evaluation_ragas "
          f"--step evaluate --skip-rag --raw-answers-path {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="用 FinetuneLLMProvider 在 RAGAS 测试集上生成回答（base / base+LoRA）"
    )
    p.add_argument(
        "--raw-answers", required=True,
        help="baseline raw_answers.json 路径（--step evaluate 产出，含 question+contexts）"
    )
    p.add_argument(
        "--base-model-dir", default="finetune/models/Qwen3.5-0.8B",
        help="HF 格式基座模型目录（相对路径解析到插件根）",
    )
    p.add_argument(
        "--adapter-dir", default="",
        help="PEFT LoRA adapter 目录（空=纯 base，不加 adapter）",
    )
    p.add_argument(
        "--tag", default="finetune",
        help="标签（写入输出文件元数据，便于区分多次运行）",
    )
    p.add_argument(
        "--out", default="results/raw_answers_finetune.json",
        help="输出 raw_answers 格式 JSON 路径",
    )
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument(
        "--no-context", action="store_true",
        help="不注入 RAG 检索上下文，仅用问题直接回答（测原始模型能力）",
    )
    args = p.parse_args()

    asyncio.run(run(
        raw_answers_path=args.raw_answers,
        output_path=args.out,
        base_model_dir=args.base_model_dir,
        adapter_dir=args.adapter_dir,
        max_new_tokens=args.max_new_tokens,
        num_threads=args.num_threads,
        tag=args.tag,
        no_context=args.no_context,
    ))


if __name__ == "__main__":
    main()
