#!/usr/bin/env python3
"""
端到端冒烟测试 — 验证完整 Agentic Idea workflow。

用法:
    source .venv/bin/activate
    python idea/test/smoke_test.py

可选参数:
    --no-real-engine  不连接真实引擎，只测节点逻辑
    --topic "研究主题"  指定测试主题
"""

import argparse
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

from idea.agentic_workflow import compile_workflow

_plugin_root = Path(__file__).parent.parent.parent
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))


async def smoke_test_isolated():
    """节点隔离测试（mock context，不连真实引擎）。"""
    print("\n" + "=" * 60)
    print("🧪 Agentic Idea 节点隔离冒烟测试")
    print("=" * 60)


    app = compile_workflow()
    print(f"✅ workflow 编译成功: {type(app).__name__}")
    print(f"   节点: {list(app.nodes.keys())}")

    # Mock context（无真实引擎）
    ctx = MagicMock()
    type(ctx).config = PropertyMock(return_value={})
    type(ctx).provider_manager = PropertyMock(return_value=MagicMock())

    state = {
        "topic": "稀疏3DGS开放词汇统一重建",
        "depth": "standard",
        "topic_analysis": None,
        "context_data": None,
        "ideas": [],
        "draft": None,
        "iteration": 0,
        "critique": None,
        "confidence": 0.0,
        "missing_evidence": [],
        "idea_scores": None,
        "phase": "analyze",
        "steps": [],
        "_num_ideas": 3,
        "_idea_focus": "all",
        "_local_rag_top_k": 5,
        "_web_top_k": 0,
        "_max_iterations": 2,
        "_context": ctx,
        "_rag_engine": None,
    }

    print(f"\n🚀 运行: topic='{state['topic']}'")
    try:
        result = await app.ainvoke(state)

        print(f"\n✅ 完成！")
        print(f"   topic_analysis domain: {result.get('topic_analysis', {}).get('domain', 'N/A')}")
        print(f"   context_data local:    {len(result.get('context_data', {}).get('local_results', []))}")
        print(f"   ideas count:          {len(result.get('ideas', []))}")
        print(f"   critique:             {result.get('critique', 'N/A')[:50]}...")
        print(f"   confidence:           {result.get('confidence', 0.0):.2f}")
        print(f"   iteration:            {result.get('iteration', 0)}")
        print(f"   final_output 字数:    {len(result.get('final_output', ''))} 字")
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
    parser = argparse.ArgumentParser(description="Agentic Idea 端到端冒烟测试")
    parser.add_argument("--no-real-engine", action="store_true", help="不连接真实引擎")
    parser.add_argument("--topic", type=str, default="稀疏3DGS开放词汇统一重建", help="测试主题")
    args = parser.parse_args()

    ok = await smoke_test_isolated()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
