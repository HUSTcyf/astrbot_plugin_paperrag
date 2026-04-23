"""
测试脚本：验证 HybridPDFParser 的 token 切分是否严格 ≤ 512 tokens，
以及 ColBERT storage 是否零截断，并可视化切分结果。

运行方式：
  cd /path/to/astrbot_plugin_paperrag
  .venv/bin/python -m test.test_chunk_tokenization
"""

import sys
import time
import traceback
import argparse
from pathlib import Path

plugin_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(plugin_root))

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

# 屏蔽 pdfminer/pdfplumber 等库的 DEBUG 日志
import logging
logger = logging.getLogger("test_chunk_tokenization")
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

    # 准确性测试
    test_texts = [
        ("短英文", "This is a short sentence.", 10),
        ("中等英文", " ".join(["word"] * 100), 110),
        ("长英文", " ".join(["word"] * 500), 510),
        # 中文约 1.6 chars/token（XLM-Roberta tokenizer），按 1020 容差 ±50
        ("短中文", "这是中文字符。" * 20, 120),    # ~160 chars → ~101 tokens
        ("长中文", "这是中文字符。" * 200, 1020),  # ~1600 chars → ~1001 tokens
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

    # 语义分块边界测试（使用 LlamaIndex SemanticSplitterNodeParser）
    print("\n  LlamaIndex 语义分块边界测试：")
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

    async def get_llamaindex_chunks():
        llamaparser = await parser._get_llamaindex_semantic_parser()
        if llamaparser is None:
            return None
        from llama_index.core import Document as LIDocument
        lldoc = LIDocument(text=long_text, metadata={"file_name": "synthetic_test.pdf"})
        llnodes = llamaparser.get_nodes_from_documents([lldoc])
        from rag.hybrid_parser import Node
        nodes = []
        for i, n in enumerate(llnodes):
            nodes.append(Node(text=n.get_text(), metadata={"chunk_index": i})) # type: ignore
        return nodes

    import asyncio
    nodes = asyncio.run(get_llamaindex_chunks())
    if nodes is None:
        print("  [FAIL] LlamaIndex 语义分块不可用")
        return False

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
    if token_counts:
        print(
            f"  统计: chunks={len(nodes)}, "
            f"tokens=[{min(token_counts)}, {max(token_counts)}], "
            f"avg={sum(token_counts)//len(token_counts)}, "
            f"over_512={sum(1 for t in token_counts if t > 512)}, "
            f"under_100={sum(1 for t in token_counts if t < 100)}"
        )
    else:
        print(f"  统计: chunks=0 (无 chunks)")
    print()


def _preview_multiline_text(text: str, limit: int = 0) -> str:
    """生成适合日志打印的多行文本预览。limit=0 表示不截断。"""
    normalized = text.replace("\r\n", "\n").strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "\n...[truncated]..."


def test_chunking_with_real_pdf(skip_llm: bool = False):
    """测试 3：使用真实 PDF 进行端到端测试"""
    print("\n" + "=" * 70)
    print("测试 3：端到端 PDF 切分测试（可视化）")
    print("=" * 70)

    tokenizer = load_tokenizer()

    test_pdf_candidates = [
        Path("/Volumes/ext/Master/papers/2408.00714v2（SAM2）.pdf"),
        Path("/Volumes/ext/Master/papers/2408.07967v2(FlashGS).pdf"),
        Path("/Volumes/ext/Master/papers/2406.09246v3(openvla).pdf"),
        Path("/Volumes/ext/Master/papers/2403.20309v6（InstantSplat）.pdf")
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

    parser = HybridPDFParser(
        chunk_size=512,
        chunk_overlap=50,
        min_chunk_size=100,
    )
    if tokenizer:
        parser._tokenizer = tokenizer

    async def run():
        # 启用 LLM 预处理时：先做标题/摘要提取 + 参考文献过滤 + 紧凑化，再语义分块
        # --skip-llm 时跳过这一步会大幅加速
        nodes = await parser.parse_and_split(str(test_pdf), llm_config={})
        return nodes

    nodes = asyncio.run(run())

    print(f"  生成了 {len(nodes)} 个 chunks\n")
    viz_chunks(nodes, parser, tokenizer)

    # LLM 预处理结果检查：确保 chunk 文本非空
    print("  正文内容检查：")
    if not nodes:
        print("  [FAIL] 生成 chunks 数量为 0")
        ok = False
    else:
        empty_chunks = []
        for i, node in enumerate(nodes):
            chunk_text = node.text or ""
            if not chunk_text.strip():
                empty_chunks.append(i)

        if empty_chunks:
            print(f"  [FAIL] {len(empty_chunks)} 个 chunks 内容为空: {empty_chunks[:10]}")
            ok = False
        else:
            print(f"  [OK] 所有 chunks 内容均非空")
            ok = True

    sample_count = min(3, len(nodes))
    if sample_count > 0:
        print(f"  正文内容预览（前 {sample_count} 个 chunks）:")
        for i in range(sample_count):
            chunk_text = nodes[i].text or ""
            chunk_tokens = parser._get_token_count(chunk_text)
            print(f"    [Chunk #{i}] tokens={chunk_tokens} chars={len(chunk_text)}")
            print("    " + _preview_multiline_text(chunk_text, limit=0).replace("\n", "\n    "))
            print()

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
    elif ok:
        print(f"  [OK] 无 chunks 超过 512 tokens")

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

        print()

    # ========== 本地 LLM 预处理逻辑检查 ==========
    print("\n" + "=" * 70)
    print("本地 LLM 预处理逻辑检查")
    print("=" * 70)

    # 检查1：多模态数据关联（从 nodes metadata 统计）
    nodes_with_images = [n for n in nodes if n.metadata.get("image_refs") or n.metadata.get("image_caption")]
    nodes_with_tables = [n for n in nodes if n.metadata.get("has_table")]
    nodes_with_formulas = [n for n in nodes if n.metadata.get("has_formula")]
    print(f"  [CHECK] 图片关联到文本块: {len(nodes_with_images)}/{len(nodes)} 个 chunks 有图片引用")
    print(f"  [CHECK] 表格关联到文本块: {len(nodes_with_tables)}/{len(nodes)} 个 chunks 有表格引用")
    print(f"  [CHECK] 公式关联到文本块: {len(nodes_with_formulas)}/{len(nodes)} 个 chunks 有公式引用")

    # 检查2：参考文献（LLM处理，需配置 llm_config）
    # 由于测试未传入 llm_config，参考文献不会被处理
    print(f"  [INFO] 参考文献处理: 未配置 llm_config，跳过（需要传入 llm_config 才启用）")

    # 检查3：LLMReferenceParser 类是否存在
    try:
        from rag.reference_processor import LLMReferenceParser
        print(f"  [OK] LLMReferenceParser 已实现")
        # 检查系统提示词
        if LLMReferenceParser.SYSTEM_PROMPT:
            print(f"  [OK] 系统提示词已定义 ({len(LLMReferenceParser.SYSTEM_PROMPT)} 字符)")
    except ImportError as e:
        print(f"  [FAIL] LLMReferenceParser 未找到: {e}")

    # 检查4：LlamaCppVLMProvider 是否实现
    try:
        from idea.llama_cpp_vlm_provider import LlamaCppVLMProvider
        print(f"  [OK] LlamaCppVLMProvider 已实现")
        # 检查模型路径是否存在
        provider = LlamaCppVLMProvider.__new__(LlamaCppVLMProvider)
        default_model = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
        model_exists = Path(default_model).exists()
        mmproj_exists = Path("./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf").exists()
        print(f"  [CHECK] VLM模型: model={'存在' if model_exists else '不存在'} mmproj={'存在' if mmproj_exists else '不存在'}")
    except ImportError as e:
        print(f"  [FAIL] LlamaCppVLMProvider 未找到: {e}")

    return ok




async def test_local_llm_preprocessing() -> bool:
    """
    测试 4：使用本地大模型（LlamaCppVLMProvider）提取 title/abstract/authors
    正文直接使用 docling 原文，不做 compact。
    """
    print("\n" + "=" * 70)
    print("测试 4：本地大模型提取 title/abstract/authors")
    print("=" * 70)

    from pathlib import Path
    from rag.hybrid_parser import HybridPDFParser

    # 1. 加载 VLM Provider
    try:
        from idea.llama_cpp_vlm_provider import LlamaCppVLMProvider
    except ImportError as e:
        print(f"  [FAIL] LlamaCppVLMProvider 导入失败: {e}")
        return False

    plugin_dir = Path(__file__).parent.parent
    model_path = plugin_dir / "models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    mmproj_path = plugin_dir / "models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"

    if not model_path.exists():
        print(f"  [SKIP] VLM 模型不存在: {model_path}")
        return True
    if not mmproj_path.exists():
        print(f"  [SKIP] VLM mmproj 不存在: {mmproj_path}")
        return True

    print(f"  模型路径: {model_path.name}")
    print(f"  mmproj路径: {mmproj_path.name}")

    vlm_provider = LlamaCppVLMProvider(
        model_path=str(model_path),
        mmproj_path=str(mmproj_path),
        n_ctx=8192,
        n_gpu_layers=99,
        max_tokens=4096,
        temperature=0.0,
    )

    # 2. 初始化
    print("  正在初始化 VLM Provider...")
    try:
        await vlm_provider.initialize()
        print("  [OK] VLM 初始化成功")
    except Exception as e:
        print(f"  [FAIL] VLM 初始化失败: {e}")
        return False

    # 注入测试用的 vlm_provider（复用已初始化的单例）
    import idea.llama_cpp_vlm_provider as llama_module
    llama_module._vlm_provider_instance = vlm_provider

    parser = HybridPDFParser(
        chunk_size=512,
        chunk_overlap=50,
        min_chunk_size=100,
    )

    # 测试用例：合成论文首页文本
    test_cases = [
        {
            "name": "英文论文首页",
            "text": """Segment Anything Model 2 (SAM 2)

Authors: Alexandre Raj, Lorenzo Bianchi, et al.

Abstract: We introduce the Segment Anything Model 2 (SAM 2), a foundation model for solving promptable visual segmentation task. SAM 2 is equipped with a memory that stores information about previous interactions, which allows it to generate masklet predictions throughout video frames. Our model produces segmentation masks of the object of interest in single images and across video frames.

1. Introduction
Deep learning has revolutionized computer vision. The Segment Anything Model (SAM) was proposed to solve interactive segmentation. We extend SAM to video domain with SAM 2."""
        },
        {
            "name": "中文论文首页",
            "text": """基于深度学习的图像分割方法研究

张三, 李四, 王五

摘要：本文提出了一种新的图像分割方法。我们的方法使用深度神经网络来学习图像特征，并在多个基准数据集上取得了先进的性能。实验结果表明，我们的方法在准确率和速度方面都优于现有方法。

1. 引言
图像分割是计算机视觉中的基础任务之一。近年来，深度学习技术在图像分割领域取得了显著进展。"""
        },
    ]

    print(f"  开始 LLM 元数据提取测试（共 {len(test_cases)} 个测试用例）:\n")

    all_pass = True
    for case in test_cases:
        desc = case["name"]
        text = case["text"]

        print(f"  [{desc}]")
        start_time = time.perf_counter()
        try:
            title, abstract, authors = await parser._extract_metadata_with_llm(text)
        except Exception as e:
            print(f"       [FAIL] LLM 调用异常: {e}")
            all_pass = False
            continue
        elapsed = time.perf_counter() - start_time

        print(f"       耗时: {elapsed:.1f}s")
        print(f"       title: {title[:60]}{'...' if len(title) > 60 else ''}")
        print(f"       abstract: {abstract[:80]}{'...' if len(abstract) > 80 else ''}")
        print(f"       authors: {authors}")

        title_ok = bool(title and len(title) > 0)
        abstract_ok = bool(abstract and len(abstract) > 0)
        authors_ok = bool(authors and len(authors) > 0)

        if title_ok and abstract_ok and authors_ok:
            print(f"       [OK] 全部提取成功")
        elif title_ok or abstract_ok or authors_ok:
            print(f"       [WARN] 部分提取成功")
            all_pass = False
        else:
            print(f"       [FAIL] 提取失败")
            all_pass = False
        print()

    # 清理 VLM Provider
    try:
        if vlm_provider._llama is not None:
            vlm_provider._llama.reset()
    except Exception:
        pass
    llama_module.reset_llama_cpp_vlm_provider()

    print("  [OK] 本地大模型预处理测试完成")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="PaperRAG Token 切分 & ColBERT Storage 截断测试")
    parser.add_argument("-t", "--tests", nargs="+", choices=["1", "2", "3", "4"],
                        help="选择要运行的测试 (如: -t 1 3 4)")
    parser.add_argument("-a", "--all", action="store_true", help="运行所有测试")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 预处理（仅测分块逻辑，大幅加速）")
    args = parser.parse_args()

    print("=" * 70)
    print("PaperRAG Token 切分 & ColBERT Storage 截断测试")
    print(f"模型路径: {BGE_MODEL_DIR}")
    print("=" * 70)

    all_tests = [
        ("test1_chunk_tokenization", test_chunk_tokenization),
        ("test2_colbert_no_truncation", test_colbert_storage_no_truncation),
        ("test3_e2e_pdf", test_chunking_with_real_pdf),
        ("test4_local_llm_preprocessing", test_local_llm_preprocessing),
    ]

    # 确定要运行的测试
    if args.all or args.tests is None:
        tests_to_run = all_tests
    else:
        indices = [int(x) - 1 for x in args.tests]
        tests_to_run = [all_tests[i] for i in indices if 0 <= i < len(all_tests)]

    results = {}
    skip_llm = args.skip_llm

    for name, fn in tests_to_run:
        try:
            import asyncio
            # 只对接受 skip_llm 参数的测试函数传递该参数
            import inspect
            sig = inspect.signature(fn)
            if "skip_llm" in sig.parameters:
                if asyncio.iscoroutinefunction(fn):
                    results[name] = asyncio.run(fn(skip_llm=skip_llm))
                else:
                    results[name] = fn(skip_llm=skip_llm)
            else:
                if asyncio.iscoroutinefunction(fn):
                    results[name] = asyncio.run(fn())
                else:
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
