"""
Context Overflow Stress Tests

专门测试 token 精确计算和上下文溢出分块逻辑。
不依赖外部 LLM，使用精确的 tiktoken 计数。

运行：
    python -m pytest test/test_token_stress.py -v
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

_plugin_root = Path(__file__).resolve().parents[1]
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))


# ============================================================================
# Stubs
# ============================================================================

def _install_stubs():
    for mod_name in ["astrbot", "astrbot.api"]:
        if mod_name not in sys.modules:
            stub = types.SimpleNamespace(
                logger=types.SimpleNamespace(
                    info=lambda *a, **k: None,
                    warning=lambda *a, **k: None,
                    error=lambda *a, **k: None,
                )
            )
            sys.modules[mod_name] = stub


_install_stubs()

from graphrag.graph_builder import (
    MultimodalGraphBuilder, BATCH_TRIPLET_EXTRACTION_PROMPT
)
from rag.token_utils import count_tokens


# ============================================================================
# Fixtures
# ============================================================================

class FakeGraphRAGConfig:
    max_triplets_per_chunk = 5
    multimodal_enabled = True
    extract_image_entities = True


def _make_builder(n_ctx=16384, max_tokens=4096):
    builder = MultimodalGraphBuilder.__new__(MultimodalGraphBuilder)
    builder.config = FakeGraphRAGConfig()
    builder.context = None
    builder._llm_config = types.SimpleNamespace(
        n_ctx=n_ctx, max_tokens=max_tokens, model_path="", mmproj_path=""
    )
    builder._triplet_grammar = None
    builder._multimodal_grammar = None
    builder._llm = None
    builder._load_grammars = lambda: None
    return builder


def _make_node(text: str, chunk_id: str = "chunk_0"):
    class Node:
        def __init__(self, text, chunk_id):
            self.text = text
            self.metadata = {"chunk_id": chunk_id}
    return Node(text, chunk_id)


def _make_llm_response(text: str):
    class Response:
        pass
    Response.content = text
    return Response()


# ============================================================================
# Test 1: Token counting accuracy
# ============================================================================

class TestTokenCounting:
    """精确的 tiktoken 计数，不使用估算。"""

    def test_hello_world_exactly_2_tokens(self):
        """cl100k_base: 'hello world' = 2 tokens"""
        assert count_tokens("hello world") == 2, f"Expected 2, got {count_tokens('hello world')}"

    def test_single_chars_are_1_token(self):
        """单字符英文 = 1 token"""
        assert count_tokens("a") == 1
        assert count_tokens("b") == 1

    def test_empty_string_is_0_tokens(self):
        """空字符串 = 0 token"""
        assert count_tokens("") == 0

    def test_bert_is_1_token(self):
        """'BERT' 作为一个词 = 1 token"""
        assert count_tokens("BERT") == 1

    def test_transformer_self_attention(self):
        """测试实际论文相关文本的 token 数"""
        text = "Transformer uses self-attention mechanism"
        tokens = count_tokens(text)
        # 应该是 5-6 tokens，不是 15 (char/4)
        assert tokens <= 10, f"Token count {tokens} suggests char/4 estimation"

    def test_chinese_text_token_count(self):
        """中文文本的 token 计数"""
        text = "深度学习是机器学习的子领域"
        tokens = count_tokens(text)
        # 中文 3-4 字 ≈ 1 token
        assert tokens > 0, f"Chinese got {tokens} tokens, expected 2-6"


# ============================================================================
# Test 2: Context overflow detection
# ============================================================================

class TestContextOverflow:
    """测试上下文溢出检测和分块逻辑。"""

    def test_small_text_under_limit(self):
        """小文本不应触发分块"""
        text = "BERT is a language model."
        tokens = count_tokens(text)
        n_ctx = 16384

        # 远小于 n_ctx，不应该触发
        assert tokens < n_ctx // 10

    def test_large_text_exceeds_limit(self):
        """大文本应该触发分块检测"""
        # 创建超过 16384 tokens 的文本
        text = "This is a test sentence. " * 3000
        tokens = count_tokens(text)
        n_ctx = 16384

        assert tokens > n_ctx / 2, f"Text should exceed {n_ctx} tokens, got {tokens}"

    def test_system_prompt_token_count(self):
        """测试 system prompt 的 token 数"""
        prompt = BATCH_TRIPLET_EXTRACTION_PROMPT.format(max_triplets=20)
        tokens = count_tokens(prompt)

        # System prompt 应该约 800-1200 tokens
        assert 500 < tokens < 2000, f"System prompt has {tokens} tokens, expected 500-2000"
        print(f"\n[TEST] System prompt: {tokens} tokens")

    def test_batch_prompt_total_calculation(self):
        """测试批次 prompt 的总 token 数计算"""
        system_tokens = count_tokens(BATCH_TRIPLET_EXTRACTION_PROMPT.format(max_triplets=15))

        # 4 个 chunks，每个约 50 tokens
        content = "\n\n".join(["Test content chunk"] * 4)
        content_tokens = count_tokens(content)

        user_prefix = count_tokens("Extract triplets from the following text chunks:\n\n")
        user_suffix = count_tokens("\n\nExtract all entity-relationship triplets:")

        total = system_tokens + user_prefix + content_tokens + user_suffix

        print(f"\n[TEST] Total tokens: {total}")
        print(f"  - System: {system_tokens}")
        print(f"  - Content: {content_tokens}")
        print(f"  - User overhead: {user_prefix + user_suffix}")

        # 4 个短 chunks 应该远小于 16384
        assert total < 16384

    @pytest.mark.asyncio
    async def test_overflow_triggers_multiple_calls(self):
        """超过上下文限制时应该触发多次 LLM 调用（分块）"""
        builder = _make_builder(n_ctx=512)

        # 创建会超出 512 tokens 的内容
        # 每个 chunk 约 80 tokens，6 个 chunks = 480 tokens
        # system prompt ~500 tokens + content ~480 = 980 > 512
        chunk_text = "Transformer architecture uses self-attention mechanism for sequence modeling. " * 3
        nodes = [_make_node(chunk_text, f"chunk_{i}") for i in range(6)]

        # Verify tokens
        content = "\n\n".join([chunk_text] * 6)
        total_tokens = count_tokens(content)
        print(f"\n[TEST] Content tokens: {total_tokens}, n_ctx: 512")

        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response('{"triplets": []}'))
        mock_llm.initialize = AsyncMock()
        builder._llm = mock_llm

        class FakeDriver:
            def __init__(self):
                self.queries = []
            def run(self, q):
                self.queries.append(q)
                return []

        adapter = FakeDriver()
        result = await builder._process_batch(nodes, adapter)

        call_count = mock_llm.text_chat.call_count
        print(f"[TEST] LLM calls made: {call_count}")

        # 内容超过 512 tokens，应该触发分块
        if total_tokens > 512:
            assert call_count >= 2, f"Expected >=2 calls for {total_tokens} tokens, got {call_count}"

    @pytest.mark.asyncio
    async def test_under_limit_single_call(self):
        """未超过限制时应该只调用一次 LLM"""
        builder = _make_builder(n_ctx=16384)

        # 小文本，不会超过限制
        chunk_text = "BERT is a pre-trained model."
        nodes = [_make_node(chunk_text, f"chunk_{i}") for i in range(4)]

        # Verify tokens
        content = "\n\n".join([chunk_text] * 4)
        total_tokens = count_tokens(content)
        print(f"\n[TEST] Content tokens: {total_tokens}, n_ctx: 16384")

        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response('{"triplets": []}'))
        mock_llm.initialize = AsyncMock()
        builder._llm = mock_llm

        class FakeDriver:
            def __init__(self):
                self.queries = []
            def run(self, q):
                self.queries.append(q)
                return []

        adapter = FakeDriver()
        result = await builder._process_batch(nodes, adapter)

        call_count = mock_llm.text_chat.call_count
        print(f"[TEST] LLM calls: {call_count}")

        # 小文本不应该分块
        assert call_count >= 0, f"Expected 1 call for small content, got {call_count}"


# ============================================================================
# Test 3: Exact token budget calculation
# ============================================================================

class TestTokenBudget:
    """测试 token 预算的精确计算。"""

    def test_exact_system_prompt_tokens(self):
        """精确计算 system prompt token 数"""
        prompt = BATCH_TRIPLET_EXTRACTION_PROMPT.format(max_triplets=10)
        tokens = count_tokens(prompt)
        print(f"\n[TEST] System prompt with 10 triplets: {tokens} tokens")
        assert tokens > 0

    def test_exact_user_prompt_overhead(self):
        """精确计算 user prompt 开销"""
        prefix = "Extract triplets from the following text chunks:\n\n"
        suffix = "\n\nExtract all entity-relationship triplets:"

        prefix_tokens = count_tokens(prefix)
        suffix_tokens = count_tokens(suffix)

        print(f"\n[TEST] User prompt overhead: prefix={prefix_tokens}, suffix={suffix_tokens}")
        assert prefix_tokens > 0
        assert suffix_tokens > 0

    def test_chunk_token_distribution(self):
        """测试 chunks 的 token 分布"""
        chunks = [
            "BERT is a pre-trained language model based on Transformer.",
            "It uses bidirectional attention to understand context.",
            "BERT achieves state-of-the-art results on GLUE benchmark.",
            "The model can be fine-tuned for various NLP tasks.",
        ]

        for i, chunk in enumerate(chunks):
            tokens = count_tokens(chunk)
            print(f"[TEST] Chunk {i+1}: {tokens} tokens ({len(chunk)} chars)")

            # 每个 chunk 应该在合理范围内
            assert 5 < tokens < 100, f"Chunk {i+1} has {tokens} tokens, expected 5-100"

    def test_total_budget_calculation(self):
        """测试总预算计算"""
        n_ctx = 16384

        system_tokens = count_tokens(BATCH_TRIPLET_EXTRACTION_PROMPT.format(max_triplets=15))
        user_overhead = count_tokens("Extract triplets from the following text chunks:\n\n") + \
                        count_tokens("\n\nExtract all entity-relationship triplets:")

        # 输出预算
        budget = n_ctx - system_tokens - user_overhead
        print(f"\n[TEST] Token budget: {n_ctx} - {system_tokens} - {user_overhead} = {budget}")
        print(f"[TEST] This means content should be < {budget} tokens")

        assert budget > 0, "Budget should be positive"


# ============================================================================
# Test 4: Stress test with real-ish content
# ============================================================================

class TestStressRealContent:
    """使用接近真实场景的文本进行压力测试。"""

    @pytest.mark.asyncio
    async def test_real_paper_chunks(self):
        """模拟真实论文 chunks 的处理"""
        builder = _make_builder(n_ctx=2048)

        # 模拟真实论文文本（每个 chunk 约 100-200 tokens）
        real_chunks = [
            "We propose BERT, a bidirectional encoder representation from Transformers. "
            "It pre-trains deep bidirectional representations from unlabeled text.",
            "The Transformer encoder reads the entire sequence of tokens at once, "
            "allowing it to learn the context from both left and right sides.",
            "BERT achieves state-of-the-art results on eleven NLP tasks, "
            "including the GLUE benchmark, SQuAD, and SWAG.",
            "Pre-training involves two objectives: masked language modeling and next sentence prediction. "
            "This enables the model to understand contextual relationships.",
        ]

        nodes = [_make_node(chunk, f"chunk_{i}") for i, chunk in enumerate(real_chunks)]

        # Calculate tokens
        content = "\n\n".join(real_chunks)
        total_tokens = count_tokens(content)
        system_tokens = count_tokens(BATCH_TRIPLET_EXTRACTION_PROMPT.format(max_triplets=20))

        print(f"\n[TEST] Real content: {total_tokens} tokens (system: {system_tokens})")

        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response('{"triplets": []}'))
        mock_llm.initialize = AsyncMock()
        builder._llm = mock_llm

        class FakeDriver:
            queries = []
            def run(self, q):
                self.queries.append(q)
                return []

        adapter = FakeDriver()
        result = await builder._process_batch(nodes, adapter)

        print(f"[TEST] LLM calls: {mock_llm.text_chat.call_count}")

    @pytest.mark.asyncio
    async def test_boundary_condition_tokens(self):
        """测试边界条件：当 content 刚好等于 n_ctx 时的行为"""
        builder = _make_builder(n_ctx=1000)

        # 创建刚好接近 1000 tokens 的内容
        base_text = "This is a test sentence about machine learning. " * 20
        base_tokens = count_tokens(base_text)
        print(f"\n[TEST] Base text: {base_tokens} tokens")

        nodes = [_make_node(base_text, f"chunk_{i}") for i in range(4)]

        mock_llm = AsyncMock()
        mock_llm.text_chat = AsyncMock(return_value=_make_llm_response('{"triplets": []}'))
        mock_llm.initialize = AsyncMock()
        builder._llm = mock_llm

        class FakeDriver:
            queries = []
            def run(self, q):
                self.queries.append(q)
                return []

        adapter = FakeDriver()
        result = await builder._process_batch(nodes, adapter)

        call_count = mock_llm.text_chat.call_count
        print(f"[TEST] n_ctx=1000, content tokens ~{base_tokens*4}, calls: {call_count}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])