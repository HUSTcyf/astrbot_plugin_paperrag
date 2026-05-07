"""JSON 工具函数 — 暂时未启用，保留备用"""

import re


def strip_thinking_tokens(text: str) -> str:
    """移除 Qwen3.5 thinking 模式产生的 <think>...</think> 块。

    Qwen3.5 模型在 think=True 时会输出 <think>...</think> 块，
    当前默认 think=False，此函数暂时未使用，保留备用。
    """
    if not text:
        return text
    stripped = text
    while '<think>' in stripped and '</think>' in stripped:
        stripped = re.sub(r'<think>.*?</think>', '', stripped, flags=re.DOTALL)
    brace = stripped.find('{')
    if brace >= 0:
        stripped = stripped[brace:]
    rbrace = stripped.rfind('}')
    if rbrace >= 0:
        stripped = stripped[:rbrace + 1]
    return stripped.strip()
