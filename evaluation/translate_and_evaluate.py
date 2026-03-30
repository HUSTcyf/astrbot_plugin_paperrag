#!/usr/bin/env python3
"""
翻译预测结果并评估

将 predictions.jsonl 中的中文答案翻译成英文，然后再进行 Qasper 官方评估。

使用插件的虚拟环境运行:
    ../.venv/bin/python evaluation/translate_and_evaluate.py

用法:
    python evaluation/translate_and_evaluate.py --all              # 翻译 + 评估
    python evaluation/translate_and_evaluate.py --translate       # 仅翻译
    python evaluation/translate_and_evaluate.py --evaluate        # 仅评估
    python evaluation/translate_and_evaluate.py --config /path/to/config.json
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

import sys
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent.parent / "config" / "astrbot_plugin_paperrag_config.json"
DEFAULT_PREDICTIONS = SCRIPT_DIR / "evaluation_output" / "predictions.jsonl"
DEFAULT_TRANSLATED = SCRIPT_DIR / "evaluation_output" / "predictions_translated.jsonl"
DEFAULT_GOLD = SCRIPT_DIR / "datasets" / "qasper-test-v0.3.json"
DEFAULT_EVAL_OUTPUT = SCRIPT_DIR / "evaluation_output"


def load_config(config_path: Path) -> dict:
    """加载插件配置"""
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def contains_chinese(text: str) -> bool:
    """检测文本是否包含中文"""
    if not text:
        return False
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))


def load_predictions(predictions_file: Path) -> list:
    """加载预测文件"""
    predictions = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                predictions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return predictions


def save_predictions(predictions: list, output_file: Path):
    """保存预测到 JSONL 文件"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


async def translate_text(api_key: str, base_url: str, text: str, model: str = "gpt-4o-mini") -> str:
    """使用 LLM API 翻译文本为英文"""
    importaiohttp = None
    try:
        import aiohttp
    except ImportError:
        print("  ⚠️  aiohttp 未安装，正在安装...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
        import aiohttp

    if not text or not text.strip():
        return text

    prompt = f"""Translate the following text to English. Only output the translated text, nothing else.

Text to translate:
{text}

English translation:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4096
    }

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"  ⚠️  API 错误 {response.status}: {error_text[:200]}")
                return text

            result = await response.json()
            translated = result.get("choices", [{}])[0].get("message", {}).get("content", text)
            return translated.strip()


async def translate_predictions(
    predictions: list,
    api_key: str,
    base_url: str,
    batch_size: int = 5,
    delay: float = 1.0,
    resume: bool = True
) -> list:
    """翻译预测结果中的中文答案"""
    print("\n" + "=" * 60)
    print("翻译中文答案到英文")
    print("=" * 60)

    translated_file = Path(str(DEFAULT_TRANSLATED).replace(".jsonl", "_progress.jsonl"))

    # 如果启用断点续传，加载已翻译的结果
    translated_map = {}
    if resume and translated_file.exists():
        with open(translated_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line.strip())
                    translated_map[p["question_id"]] = p
                except json.JSONDecodeError:
                    continue
        print(f"📂 已加载 {len(translated_map)} 条已翻译的预测")

    # 统计需要翻译的数量
    need_translate = []
    already_translated = []
    for pred in predictions:
        qid = pred.get("question_id", "")
        answer = pred.get("predicted_answer", "")

        if qid in translated_map:
            # 检查翻译后的内容是否仍含有中文
            translated_answer = translated_map[qid].get("predicted_answer", "")
            if contains_chinese(translated_answer):
                # 翻译后仍含中文，需要重新翻译
                need_translate.append({
                    "question_id": qid,
                    "original_answer": answer,
                    "predicted_answer": translated_answer
                })
                del translated_map[qid]  # 删除旧翻译，重新翻译
            else:
                already_translated.append(translated_map[qid])
        elif contains_chinese(answer):
            need_translate.append(pred)
        else:
            # 没有中文的保留原样
            already_translated.append(pred)

    print(f"📊 统计:")
    print(f"   总数: {len(predictions)}")
    print(f"   已有翻译: {len(already_translated)}")
    print(f"   需要翻译: {len(need_translate)}")

    if not need_translate:
        print("✅ 所有答案都已经是英文或已翻译")
        return predictions

    # 开始翻译
    translated_count = 0
    all_translated = list(translated_map.values())

    for i, pred in enumerate(need_translate):
        qid = pred.get("question_id", "")
        answer = pred.get("predicted_answer", "")

        if i % 10 == 0:
            print(f"  翻译进度: {i}/{len(need_translate)}...")

        # 翻译答案
        translated_answer = await translate_text(api_key, base_url, answer)

        new_pred = {
            "question_id": qid,
            "predicted_answer": translated_answer,
            "predicted_evidence": pred.get("predicted_evidence", [])
        }

        all_translated.append(new_pred)
        translated_map[qid] = new_pred
        translated_count += 1

        # 定期保存进度
        if i > 0 and i % batch_size == 0:
            save_predictions(list(translated_map.values()), translated_file)
            await asyncio.sleep(delay)

    # 最终保存
    save_predictions(all_translated, translated_file)
    print(f"\n✅ 翻译完成: {translated_count} 条")
    print(f"   翻译后文件: {translated_file}")

    return all_translated


def run_evaluator(predictions_file: Path, gold_file: Path, output_dir: Path, text_evidence_only: bool = False):
    """运行官方评估脚本"""
    print("\n" + "=" * 60)
    print("运行评估")
    print("=" * 60)

    if not predictions_file.exists():
        print(f"❌ Predictions 文件不存在: {predictions_file}")
        sys.exit(1)

    if not gold_file.exists():
        print(f"❌ Gold 文件不存在: {gold_file}")
        sys.exit(1)

    evaluator_script = SCRIPT_DIR / "datasets" / "qasper_evaluator.py"
    if not evaluator_script.exists():
        print(f"❌ 评估脚本不存在: {evaluator_script}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(evaluator_script),
        "--predictions", str(predictions_file),
        "--gold", str(gold_file),
    ]

    if text_evidence_only:
        cmd.append("--text_evidence_only")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "evaluation_results.json"

    print(f"\n运行评估命令:")
    print(f"  {' '.join(cmd)}")
    print(f"\n评估结果将保存到: {output_file}")

    import subprocess
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print("\n评估结果:")
        print(result.stdout)

        try:
            eval_result = json.loads(result.stdout)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 评估结果已保存到: {output_file}")
        except json.JSONDecodeError:
            print(f"\n⚠️ 无法解析评估结果为 JSON")

    if result.stderr:
        print("\n错误信息:")
        print(result.stderr)

    return result.returncode == 0


async def main_async(args):
    """异步主函数"""
    # 解析路径
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = DEFAULT_CONFIG_PATH

    predictions_file = Path(args.predictions) if args.predictions else DEFAULT_PREDICTIONS
    translated_file = Path(args.translated) if args.translated else DEFAULT_TRANSLATED
    gold_file = Path(args.gold) if args.gold else DEFAULT_GOLD
    output_dir = Path(args.output) if args.output else DEFAULT_EVAL_OUTPUT

    # 加载配置
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)

    api_key = config.get("freeapi_key", "")
    base_url = config.get("freeapi_url", "").rstrip("/")

    if not api_key:
        print("❌ freeapi_key 未配置，无法进行翻译")
        sys.exit(1)

    # ========== 翻译模式 ==========
    if args.translate or args.all:
        if not predictions_file.exists():
            print(f"❌ Predictions 文件不存在: {predictions_file}")
            sys.exit(1)

        predictions = load_predictions(predictions_file)
        print(f"加载了 {len(predictions)} 条预测")

        await translate_predictions(
            predictions,
            api_key,
            base_url,
            batch_size=args.batch_size,
            delay=args.delay,
            resume=not args.no_resume
        )

    # ========== 评估模式 ==========
    if args.evaluate or args.all:
        # 使用翻译后的文件进行评估
        eval_predictions = translated_file if translated_file.exists() else predictions_file
        print(f"使用预测文件: {eval_predictions}")

        run_evaluator(eval_predictions, gold_file, output_dir, args.text_evidence_only)

    if not args.translate and not args.evaluate and not args.all:
        print("\n请选择操作:")
        print("  1. 翻译中文答案")
        print("  2. 翻译 + 评估")
        print("  3. 仅评估（使用已翻译的文件）")
        print("  0. 退出")

        choice = input("\n请输入选项 (0-3): ").strip()

        if choice == "1":
            if not predictions_file.exists():
                print(f"❌ Predictions 文件不存在: {predictions_file}")
                sys.exit(1)
            predictions = load_predictions(predictions_file)
            print(f"加载了 {len(predictions)} 条预测")
            await translate_predictions(predictions, api_key, base_url, resume=True)

        elif choice == "2":
            if not predictions_file.exists():
                print(f"❌ Predictions 文件不存在: {predictions_file}")
                sys.exit(1)
            predictions = load_predictions(predictions_file)
            print(f"加载了 {len(predictions)} 条预测")
            await translate_predictions(predictions, api_key, base_url, resume=True)
            run_evaluator(translated_file, gold_file, output_dir, args.text_evidence_only)

        elif choice == "3":
            eval_predictions = translated_file if translated_file.exists() else predictions_file
            print(f"使用预测文件: {eval_predictions}")
            run_evaluator(eval_predictions, gold_file, output_dir, args.text_evidence_only)

        else:
            print("已退出")


def main():
    parser = argparse.ArgumentParser(
        description="翻译预测结果并评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluation/translate_and_evaluate.py --all              # 翻译 + 评估
  python evaluation/translate_and_evaluate.py --translate       # 仅翻译
  python evaluation/translate_and_evaluate.py --evaluate        # 仅评估
  python evaluation/translate_and_evaluate.py --config /path/to/config.json
        """
    )

    parser.add_argument("--translate", action="store_true", help="仅翻译中文答案")
    parser.add_argument("--evaluate", action="store_true", help="仅运行评估")
    parser.add_argument("--all", action="store_true", help="翻译 + 评估")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--predictions", type=str, default=None, help="原始预测文件路径")
    parser.add_argument("--translated", type=str, default=None, help="翻译后预测文件路径")
    parser.add_argument("--gold", type=str, default=None, help="Gold 标准文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=5, help="每批翻译的问题数 (默认: 5)")
    parser.add_argument("--delay", type=float, default=1.0, help="批次间延迟秒数 (默认: 1.0)")
    parser.add_argument("--text_evidence_only", action="store_true", help="仅使用文本证据")
    parser.add_argument("--no_resume", action="store_true", help="禁用断点续传，重新翻译所有答案")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
