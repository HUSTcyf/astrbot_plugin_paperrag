"""
测试脚本：验证 HybridPDFParser 的 token 切分是否严格 ≤ 512 tokens，
以及 ColBERT storage 是否零截断，并可视化切分结果。

运行方式：
  cd /path/to/astrbot_plugin_paperrag
  .venv/bin/python -m test.test_chunk_tokenization
"""

import sys
import traceback
from pathlib import Path

plugin_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(plugin_root))

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

# 屏蔽 pdfminer/pdfplumber 等库的 DEBUG 日志
import logging
for noisy_module in ["pdfminer", "pdfplumber", "pypdf", "PIL", "PIL.Image",
                     "astrbot_core", "astrbot_plugin", "faiss", "torch", "torchao"]:
    logging.getLogger(noisy_module).setLevel(logging.WARNING)

# BGE-M3 tokenizer 路径（注意：插件子目录下的 models）
BGE_MODEL_DIR = plugin_root / "models" / "bge-m3"
# 备选：插件根目录
if not BGE_MODEL_DIR.exists():
    BGE_MODEL_DIR = Path("/Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag/models/bge-m3")


def load_tokenizer():
    """加载 BGE-M3 tokenizer"""
    try:
        from transformers import AutoTokenizer
        if not (BGE_MODEL_DIR / "tokenizer.json").exists():
            print(f"  [WARN] BGE-M3 模型不存在: {BGE_MODEL_DIR}")
            return None
        tok = AutoTokenizer.from_pretrained(str(BGE_MODEL_DIR), local_files_only=True)
        print(f"  [OK] tokenizer 加载成功: {tok.__class__.__name__}, vocab_size={tok.vocab_size}")
        return tok
    except Exception as e:
        print(f"  [WARN] tokenizer 加载失败: {e}")
        return None


def test_chunk_tokenization():
    """测试 1：直接测试 _get_token_count 和 _semantic_chunk 的 token 边界控制"""
    print("\n" + "=" * 70)
    print("测试 1：_get_token_count 和语义分块 token 边界控制")
    print("=" * 70)

    tokenizer = load_tokenizer()

    from rag.hybrid_parser import HybridPDFParser

    parser = HybridPDFParser(
        chunk_size=512,
        chunk_overlap=50,
        min_chunk_size=100,
    )

    if tokenizer:
        parser._tokenizer = tokenizer
        print("  [OK] parser 已注入真实 tokenizer\n")
    else:
        print("  [WARN] 无 tokenizer，回退到 len(text)//4\n")

    # 准确性测试（中文 ≈ 1.5 chars/token，BPE tokenizer）
    test_texts = [
        ("短英文", "This is a short sentence.", 10),
        ("中等英文", " ".join(["word"] * 100), 110),
        ("长英文", " ".join(["word"] * 500), 510),
        # 中文 ≈ 1.5 chars/token（参考值，实际以 tokenizer 为准）
        ("短中文", "这是中文字符。" * 20, 120),   # ~80 chars → ~101 tokens
        ("长中文", "这是中文字符。" * 200, 540),  # ~800 chars → ~512 tokens
        ("中英混合", "This is English. 这是中文。 " * 50, 420),  # ~550 chars → ~401 tokens
    ]

    print("  _get_token_count 准确性测试（容差 ±20 tokens）：")
    print("  注：无真实 tokenizer 时使用 len(text)//4 估算，部分偏差属正常现象")
    all_pass = True
    for desc, text, max_exp in test_texts:
        tokens = parser._get_token_count(text)
        ok = tokens <= max_exp + 20
        status = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{status}] {desc}: {tokens} tokens (expected ≤ {max_exp + 20})")

    # 语义分块边界测试
    print("\n  _semantic_chunk token 边界测试：")
    long_text = (
        "Deep learning has revolutionized artificial intelligence in recent years. "
        "Neural networks with many layers can learn complex patterns from data. "
        "Transformer models have become the dominant architecture for NLP tasks. "
        "Attention mechanisms allow models to focus on relevant information. "
        "\n\n"
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from experience without being explicitly programmed. "
        "Supervised learning uses labeled data to train models. "
        "Unsupervised learning discovers hidden patterns in unlabeled data. "
        "\n\n"
        "Computer vision systems can interpret and understand images and videos. "
        "Object detection locates and classifies objects in images. "
        "Semantic segmentation assigns labels to every pixel in an image. "
        "Image generation models can create realistic photos from text descriptions. "
    )

    nodes = parser._semantic_chunk(long_text, {"file_name": "synthetic_test.pdf"})

    print(f"    生成了 {len(nodes)} 个 chunks\n")
    viz_chunks(nodes, parser, tokenizer)

    over_limit = []
    for i, node in enumerate(nodes):
        t = parser._get_token_count(node.text)
        if t > 512:
            over_limit.append((i, t))

    if over_limit:
        print(f"    [FAIL] {len(over_limit)} 个 chunks 超过 512 tokens:")
        for idx, tok in over_limit:
            print(f"           chunk #{idx}: {tok} tokens")
        all_pass = False
    else:
        print(f"    [OK] 所有 chunks ≤ 512 tokens")

    return all_pass


def test_colbert_storage_no_truncation():
    """测试 2：验证 ColBERT storage 的防截断断言"""
    print("\n" + "=" * 70)
    print("测试 2：ColBERT storage 防截断断言")
    print("=" * 70)

    import numpy as np
    import tempfile
    from rag.colbert_storage import ColBERTStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ColBERTStorage(tmpdir)

        print(f"  MAX_TOKENS_PER_CHUNK = {storage.MAX_TOKENS_PER_CHUNK}\n")

        # 合法：512 tokens（等于 MAX_TOKENS_PER_CHUNK）
        try:
            storage.add_chunks([np.random.randn(512, 1024).astype(np.float32)], ["chunk_ok"])
            print("  [OK] 512 tokens chunk 添加成功")
        except ValueError as e:
            print(f"  [FAIL] 512 tokens chunk 不应抛异常: {e}")
            return False

        # 超限：513 tokens → 应抛异常
        try:
            storage.add_chunks([np.random.randn(513, 1024).astype(np.float32)], ["chunk_over"])
            print("  [FAIL] 513 tokens chunk 应抛异常但没有")
            return False
        except ValueError as e:
            print(f"  [OK] 超限正确抛异常: {str(e)[:80]}...")

        # 边界：256 tokens
        try:
            storage.add_chunks([np.random.randn(256, 1024).astype(np.float32)], ["chunk_256"])
            print("  [OK] 256 tokens boundary chunk 添加成功")
        except ValueError as e:
            print(f"  [FAIL] 256 tokens 不应抛异常: {e}")
            return False

        # 验证 n_tokens 记录（513 token 的 chunk 抛异常后不会被添加）
        expected = [512, 256]
        actual = [m["n_tokens"] for m in storage._id_mapping]
        if actual == expected:
            print(f"  [OK] n_tokens 记录正确: {actual}")
        else:
            print(f"  [FAIL] n_tokens 记录错误: expected {expected}, got {actual}")
            return False

    return True


def viz_chunks(nodes, parser, tokenizer):
    """可视化 chunks 的切分结果"""
    print("  " + "-" * 68)
    print(f"  {'IDX':>4}  {'TOKENS':>7}  {'CHARS':>6}  {'TEXT PREVIEW':<48}")
    print("  " + "-" * 68)

    for i, node in enumerate(nodes):
        text = node.text
        char_count = len(text)
        token_count = parser._get_token_count(text)

        # Token 比例
        ratio = token_count / char_count if char_count > 0 else 0

        # 状态标记
        if token_count > 512:
            flag = " ⚠️ >512"
        elif token_count < 100:
            flag = " ⚠️ <100"
        else:
            flag = ""

        # 文本预览：前50字符 + "..." + 后30字符（超过100字符时）
        if len(text) > 80:
            preview = text[:50].replace("\n", "↵") + "..." + text[-30:].replace("\n", "↵")
        else:
            preview = text.replace("\n", "↵")

        # 断点标记
        boundary = ""
        if "\n\n" in text[:20]:
            boundary = " [para-break]"

        print(
            f"  {i:>4}  {token_count:>7}  {char_count:>6}  {preview:<48}{flag}{boundary}"
        )

    print("  " + "-" * 68)

    # 统计摘要
    token_counts = [parser._get_token_count(n.text) for n in nodes]
    print(
        f"  统计: chunks={len(nodes)}, "
        f"tokens=[{min(token_counts)}, {max(token_counts)}], "
        f"avg={sum(token_counts)//len(token_counts)}, "
        f"over_512={sum(1 for t in token_counts if t > 512)}, "
        f"under_100={sum(1 for t in token_counts if t < 100)}"
    )
    print()


def test_chunking_with_real_pdf():
    """测试 3：使用真实 PDF 进行端到端测试"""
    print("\n" + "=" * 70)
    print("测试 3：端到端 PDF 切分测试（可视化）")
    print("=" * 70)

    tokenizer = load_tokenizer()

    test_pdf_candidates = [
        plugin_root / "data" / "papers" / "test.pdf",
        Path("/Users/chenyifeng/AstrBot/data/skills/theme-factory/theme-showcase.pdf"),
    ]

    test_pdf = None
    for p in test_pdf_candidates:
        if p.exists():
            test_pdf = p
            break

    if test_pdf is None:
        print("  [SKIP] 未找到测试 PDF\n")
        return True

    print(f"  PDF: {test_pdf.name} ({test_pdf.stat().st_size / 1024:.1f} KB)\n")

    import asyncio
    from rag.hybrid_parser import HybridPDFParser

    async def run():
        parser = HybridPDFParser(
            chunk_size=512,
            chunk_overlap=50,
            min_chunk_size=100,
        )
        if tokenizer:
            parser._tokenizer = tokenizer

        nodes = await parser.parse_and_split(str(test_pdf))
        return parser, nodes

    parser, nodes = asyncio.run(run())

    print(f"  生成了 {len(nodes)} 个 chunks\n")
    viz_chunks(nodes, parser, tokenizer)

    # 详细报告超限和过短 chunks
    over_limit = []
    under_limit = []
    for i, node in enumerate(nodes):
        t = parser._get_token_count(node.text)
        if t > 512:
            over_limit.append((i, t, node.text[:120]))
        elif t < 100:
            under_limit.append((i, t, node.text[:120]))

    if over_limit:
        print(f"  [FAIL] {len(over_limit)} 个 chunks 超过 512 tokens:")
        for idx, tok, txt in over_limit[:5]:
            print(f"         #{idx} ({tok} tok): {txt!r:.80}...")
        ok = False
    else:
        print(f"  [OK] 无 chunks 超过 512 tokens")
        ok = True

    if under_limit:
        print(f"  [WARN] {len(under_limit)} 个 chunks 不足 100 tokens:")
        for idx, tok, txt in under_limit[:5]:
            print(f"         #{idx} ({tok} tok): {txt!r:.80}...")

    # ColBERT storage 验证（token 数 ≤ 512）
    print("\n  ColBERT storage 兼容性检查（MAX_TOKENS_PER_CHUNK=512）：")
    storage_issues = []
    for i, node in enumerate(nodes):
        t = parser._get_token_count(node.text)
        if t > 512:
            storage_issues.append((i, t))

    if storage_issues:
        print(f"  [FAIL] {len(storage_issues)} 个 chunks 超过 ColBERT 上限 512:")
        for idx, tok in storage_issues[:5]:
            print(f"         #{idx}: {tok} tokens — add_chunks 会抛异常")
        return False
    else:
        print(f"  [OK] 所有 chunks 可被 ColBERT storage 接受（≤ 512 tokens）")

    # 打印所有 chunks 完整内容
    print("\n" + "=" * 70)
    print("所有 Chunk 完整内容")
    print("=" * 70)
    for i, node in enumerate(nodes):
        t = parser._get_token_count(node.text)
        print(f"\n--- Chunk #{i} ({t} tokens, {len(node.text)} chars) ---")
        print(node.text)
        print()

    return ok


def main():
    print("=" * 70)
    print("PaperRAG Token 切分 & ColBERT Storage 截断测试")
    print(f"模型路径: {BGE_MODEL_DIR}")
    print("=" * 70)

    results = {}

    tests = [
        ("test1_chunk_tokenization", test_chunk_tokenization),
        ("test2_colbert_no_truncation", test_colbert_storage_no_truncation),
        ("test3_e2e_pdf", test_chunking_with_real_pdf),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [ERROR] {name} 异常: {e}")
            traceback.print_exc()
            results[name] = False

    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    all_pass = all(results.values())
    print(f"\n{'全部通过 ✓' if all_pass else '存在失败测试 ✗'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
