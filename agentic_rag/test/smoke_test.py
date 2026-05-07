#!/usr/bin/env python3
"""
端到端冒烟测试 — 验证完整 Agentic RAG 流程。

用法:
    source .venv/bin/activate
    python agentic_rag/test/smoke_test.py

可选参数:
    --no-real-engine  不连接真实 Milvus/Neo4j，只测节点逻辑
    --query "问题"    指定测试查询
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加插件根目录到 sys.path
_plugin_root = Path(__file__).parent.parent.parent
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))


async def smoke_test_isolated():
    """节点隔离测试（mock context，不连真实引擎）。"""
    print("\n" + "=" * 60)
    print("🧪 节点隔离冒烟测试")
    print("=" * 60)

    from agentic_rag.workflow import compile_workflow
    from unittest.mock import MagicMock

    app = compile_workflow()
    print(f"✅ workflow 编译成功: {type(app).__name__}")
    print(f"   节点: {list(app.nodes.keys())}")

    # Mock context（无真实引擎）
    ctx = MagicMock()
    ctx.config = {}
    ctx.provider_manager = MagicMock()
    ctx.provider_manager.get_provider = MagicMock(return_value=None)

    state = {
        "query": "这篇论文使用的方法是什么",
        "_context": ctx,
    }

    print(f"\n🚀 运行: query='{state['query']}'")
    try:
        result = await app.ainvoke(state)

        print(f"\n✅ 完成！")
        print(f"   query_type:     {result.get('query_type', 'N/A')}")
        print(f"   graph_weight:   {result.get('graph_weight', 'N/A')}")
        print(f"   retrieved_nodes: {len(result.get('retrieved_nodes', []))}")
        print(f"   graph_entities:  {len(result.get('graph_entities', []))}")
        print(f"   draft 字数:      {len(result.get('draft', ''))}")
        print(f"   citations:       {len(result.get('citations', []))}")
        print(f"   final_answer:   {len(result.get('final_answer', ''))} 字")
        print(f"   steps:")
        for step in result.get("steps", []):
            print(f"     - {step}")
        return True
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def smoke_test_full():
    """完整流程测试（连接真实 Milvus/Neo4j）。"""
    print("\n" + "=" * 60)
    print("🌍 完整流程冒烟测试（真实引擎）")
    print("=" * 60)

    from agentic_rag.workflow import compile_workflow
    from agentic_rag.engine_utils import get_engine

    app = compile_workflow()

    # 尝试获取真实引擎
    class FakeContext:
        config = {}
        provider_manager = None

    ctx = FakeContext()

    print("尝试获取 HybridRAGEngine...")
    engine = get_engine(ctx)
    if engine is None:
        print("⚠️  HybridRAGEngine 未就绪，跳过完整流程")
        return False

    print("✅ HybridRAGEngine 就绪")

    state = {
        "query": "这篇论文的主要贡献是什么",
        "_context": ctx,
    }

    print(f"\n🚀 运行: query='{state['query']}'")
    try:
        result = await app.ainvoke(state)

        print(f"\n✅ 完成！")
        print(f"   query_type:     {result.get('query_type', 'N/A')}")
        print(f"   graph_weight:   {result.get('graph_weight', 'N/A')}")
        print(f"   retrieved_nodes: {len(result.get('retrieved_nodes', []))}")
        print(f"   graph_entities:  {len(result.get('graph_entities', []))}")
        print(f"   draft 字数:      {len(result.get('draft', ''))}")
        print(f"   citations:       {len(result.get('citations', []))}")

        answer = result.get("final_answer", "")
        print(f"   final_answer:   {len(answer)} 字")
        if answer:
            print(f"\n--- 回答预览 ---")
            print(answer[:500])
            if len(answer) > 500:
                print("... (截断)")
            print("---")

        print(f"   steps:")
        for step in result.get("steps", []):
            print(f"     - {step}")
        return True
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description="Agentic RAG 端到端冒烟测试")
    parser.add_argument("--no-real-engine", action="store_true", help="不连接真实引擎")
    parser.add_argument("--query", type=str, default="这篇论文的方法是什么", help="测试查询")
    args = parser.parse_args()

    if args.no_real_engine:
        ok = await smoke_test_isolated()
    else:
        # 先跑隔离测试，失败后再试完整
        ok = await smoke_test_isolated()
        if not ok:
            print("\n⚠️ 隔离测试失败，跳过完整流程")
        else:
            print("\n🔄 隔离测试通过，继续完整流程...")
            await smoke_test_full()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
