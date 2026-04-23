"""
直接测试 llama-cpp-python 的 grammar 约束功能
"""

import sys
import asyncio
from pathlib import Path

plugin_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(plugin_root))

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

import logging
for noisy_module in ["pdfminer", "pdfplumber", "pypdf", "PIL", "faiss", "torch"]:
    logging.getLogger(noisy_module).setLevel(logging.WARNING)


def test_grammar_direct():
    """直接测试 llama.create_chat_completion 的 grammar 约束"""
    from llama_cpp import Llama, LlamaGrammar

    model_path = plugin_root / "models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    mmproj_path = plugin_root / "models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"
    grammar_path = plugin_root / "rag/test_schema.gbnf"

    if not model_path.exists():
        print(f"[SKIP] 模型不存在")
        return True

    print(f"加载模型...")
    llama = Llama(
        model_path=str(model_path),
        mmproj=str(mmproj_path),
        n_ctx=8192,
        n_gpu_layers=99,
        n_batch=32,
        verbose=False,
    )
    print("[OK] 模型加载成功\n")

    # 创建 Grammar 对象
    grammar_content = grammar_path.read_text()
    grammar = LlamaGrammar.from_string(grammar_content)
    print(f"Grammar 对象创建成功: {grammar}\n")

    # 测试1: 无 grammar
    print("=== 测试1: 无 grammar ===")
    messages = [{"role": "user", "content": "回答: 1+1等于几"}]
    result1 = llama.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=100,
    )
    print(f"输出: {result1['choices'][0]['message']['content']}")
    print()

    # 测试2: 有 grammar
    print("=== 测试2: 有 grammar 约束 ===")
    result2 = llama.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=100,
        grammar=grammar,
    )
    content2 = result2['choices'][0]['message']['content']
    print(f"Grammar约束输出: {content2}")

    # 检查是否符合 {"answer": "..."} 格式
    if '{"answer":' in content2 and '}' in content2:
        print("[OK] 输出符合 JSON 格式")
    else:
        print("[WARN] 输出不符合预期格式")

    llama.close()
    print("\n[OK] 测试完成")
    return True


if __name__ == "__main__":
    test_grammar_direct()
