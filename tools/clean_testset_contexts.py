"""
测试集上下文规范化脚本

功能：
  清理 contexts 中的：
  1. 字面字符串 \\n（反斜杠+n）→ 替换为空格
  2. 箭头符号行（⇧, ⇩, ↓, ↑ 等单独成行）→ 删除整行
  3. 多余空格 → 合并为单空格
  4. 原文中的真正换行符（\n）→ 保留（PDF解析产生的段落分隔）
"""

import argparse
import json
import re
from pathlib import Path


# 箭头符号（单独成行时视为噪声）
ARROW_LINES: set[str] = {'⇧', '⇩', '↓', '↑', '▸', '▾', '▣', '■', '□', '▪', '▫', '●', '○'}


def normalize_text(text: str) -> str:
    """清理学术文本中的各类噪声字符。"""
    if not text:
        return text

    # 1. 将字面字符串 \n（反斜杠+n）替换为空格
    #    匹配 \n 但不匹配真正的换行符（\n 是单个字符 10）
    text = text.replace('\\n', ' ')

    # 2. 移除单独一行的箭头符号（如 ⇧、⇩ 等）
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in ARROW_LINES:
            continue  # 跳过整行箭头
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 3. 将 2+ 个连续空格合并为 1 个
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="规范化 testset contexts")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plugin_dir = Path(__file__).parent.parent
    input_path = args.input or (plugin_dir / "results" / "testset.json")
    output_path = args.output or (plugin_dir / "results" / "testset_cleaned.json")

    print(f"读取: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    changed = 0

    for idx, item in enumerate(data):
        if "contexts" not in item or not isinstance(item["contexts"], list):
            continue
        original = item["contexts"][:] if item["contexts"] else []
        item["contexts"] = [normalize_text(ctx) for ctx in item["contexts"]]
        for a, b in zip(original, item["contexts"]):
            if a != b:
                changed += 1

        if (idx + 1) % 50 == 0:
            print(f"  进度: {idx + 1}/{total}")

    print(f"共处理 {total} 条记录，{changed} 个 context 有变化")

    print(f"保存: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("完成!")


if __name__ == "__main__":
    main()
