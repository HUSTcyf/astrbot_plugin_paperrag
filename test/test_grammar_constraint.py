"""
Grammar 约束测试：验证 llama.cpp grammar 文件是否能约束输出格式
"""

import sys
import asyncio
import json
import re
from pathlib import Path

plugin_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(plugin_root))

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

import logging
for noisy_module in ["pdfminer", "pdfplumber", "pypdf", "PIL", "faiss", "torch"]:
    logging.getLogger(noisy_module).setLevel(logging.WARNING)


def extract_json(text: str) -> dict | None:
    """从 grammar 约束的输出中提取 JSON"""
    # grammar 输出的 JSON 可能有多余空白，先清理
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


async def test_grammar_constraint():
    from idea.llama_cpp_vlm_provider import LlamaCppVLMProvider

    model_path = plugin_root / "models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    mmproj_path = plugin_root / "models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"
    grammar_path = plugin_root / "rag/compact_schema.gbnf"

    if not model_path.exists() or not mmproj_path.exists():
        print(f"[SKIP] 模型不存在")
        return True

    print(f"模型: {model_path.name}")
    print(f"grammar: {grammar_path.name}")

    vlm = LlamaCppVLMProvider(
        model_path=str(model_path),
        mmproj_path=str(mmproj_path),
        n_ctx=8192,
        n_gpu_layers=99,
        max_tokens=256,
    )

    await vlm.initialize()
    print("[OK] VLM 初始化成功\n")

    # 测试1: 无 grammar（对比用）
    print("=== 测试1: 无 grammar ===")
    resp1 = await vlm.text_chat(
        prompt='用JSON格式回答：{"question": "天空是什么颜色", "answer": "请直接给出答案"}'
    )
    print(f"自由输出: {resp1.content[:200]}...")
    print()

    # 测试2: 有 grammar（通过 text_chat 的 grammar 参数）
    print("=== 测试2: 有 grammar 约束 (通过 text_chat) ===")
    if not grammar_path.exists():
        print(f"[SKIP] grammar 文件不存在: {grammar_path}")
        return True

    # 使用 grammar 参数（字符串路径）
    resp2 = await vlm.text_chat(
        prompt='请用JSON格式回答，包含title和abstract和authors和compacted_text字段',
        grammar=str(grammar_path)
    )
    content2 = resp2.content
    print(f"Grammar约束输出: {content2[:300]}...")

    # 解析 JSON
    data = extract_json(content2)
    if data:
        print(f"[OK] JSON 解析成功")
        print(f"  keys: {list(data.keys())}")
        if "title" in data:
            print(f"  title: {data['title'][:50]}...")
        if "abstract" in data:
            print(f"  abstract: {data['abstract'][:50]}...")
        if "authors" in data:
            print(f"  authors: {data['authors']}")
        if "compacted_text" in data:
            print(f"  compacted_text: {data['compacted_text'][:50]}...")
    else:
        print(f"[FAIL] JSON 解析失败")

    # 清理
    try:
        vlm._llama.close()
    except Exception:
        pass

    from idea import llama_cpp_vlm_provider
    llama_cpp_vlm_provider.reset_llama_cpp_vlm_provider()

    print("\n[OK] 测试完成")
    return data is not None


if __name__ == "__main__":
    result = asyncio.run(test_grammar_constraint())
    sys.exit(0 if result else 1)
