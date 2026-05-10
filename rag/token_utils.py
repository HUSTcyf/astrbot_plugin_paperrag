"""
Unified token counting for PaperRAG.

Primary: tiktoken cl100k_base (lightweight, always precise for LLM context estimation).
Optional: set an embedding tokenizer (BGE-M3, Unsloth, etc.) for chunk-level counting
          that matches the embedding model's vocabulary.

No char/4 estimation — every count is exact.
"""

import threading
from typing import Any, Optional

_encoder: Optional[Any] = None
_encoder_lock = threading.Lock()

_embedding_tokenizer: Optional[Any] = None
_et_lock = threading.Lock()


def _ensure_encoder():
    """Lazy-init tiktoken cl100k_base encoder (thread-safe singleton)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Precise token count using tiktoken cl100k_base.

    Suitable for LLM context window estimation (GPT-4, Claude, Qwen, etc.).
    """
    if not text:
        return 0
    return len(_ensure_encoder().encode(text))


def set_embedding_tokenizer(tokenizer: Any) -> None:
    """Register the embedding model tokenizer for chunk-level counting.

    Call this once after the embedding model (BGE-M3, Unsloth, etc.) is loaded.
    """
    global _embedding_tokenizer
    with _et_lock:
        _embedding_tokenizer = tokenizer


def count_embedding_tokens(text: str) -> int:
    """Count tokens with the embedding tokenizer; fall back to tiktoken."""
    if not text:
        return 0
    with _et_lock:
        tok = _embedding_tokenizer
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return count_tokens(text)


def get_embedding_tokenizer() -> Optional[Any]:
    """Return the currently configured embedding tokenizer (or None)."""
    with _et_lock:
        return _embedding_tokenizer
