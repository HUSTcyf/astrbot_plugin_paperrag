#!/usr/bin/env python3
"""
Bright Data MCP 网络搜索功能测试

测试步骤：
1. 检查 npx @brightdata/mcp 是否可用
2. 测试 search_engine 工具
3. 测试 search_engine_batch 工具
"""

import asyncio
import json
import subprocess
import sys


async def test_npx_available():
    """检查 npx 是否可用"""
    print("=" * 60)
    print("测试1: 检查 npx 是否可用")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✅ npx 版本: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ npx 不可用: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ npx 检查失败: {e}")
        return False


async def test_mcp_server():
    """测试 MCP 服务器启动"""
    print("\n" + "=" * 60)
    print("测试2: 检查 Bright Data MCP 服务器")
    print("=" * 60)

    try:
        # 简单测试 - 尝试列出工具
        result = subprocess.run(
            ["npx", "@brightdata/mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"输出: {result.stdout[:500]}")
        print(f"错误: {result.stderr[:500]}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ MCP服务器检查失败: {e}")
        return False


async def test_search_engine():
    """测试 search_engine 工具"""
    print("\n" + "=" * 60)
    print("测试3: 测试 search_engine 工具")
    print("=" * 60)

    # 构造JSON-RPC请求
    rpc_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_engine",
            "arguments": {
                "query": "large language model medical diagnosis",
                "num_results": 5,
                "source": "web"
            }
        }
    }

    # API Token
    api_token = "88b654f6-f6b0-4e8d-85d5-c50dc5e2d3c5"

    try:
        print("发送请求...")
        print(f"查询: 'large language model medical diagnosis'")
        print(f"来源: web")
        print(f"数量: 5")

        env = {**subprocess.os.environ, "API_TOKEN": api_token}

        proc = await asyncio.create_subprocess_exec(
            "npx", "@brightdata/mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps(rpc_request).encode()),
                timeout=60
            )
        except asyncio.TimeoutError:
            proc.kill()
            print("❌ 请求超时 (60s)")
            return False

        print(f"\n返回码: {proc.returncode}")

        if stderr:
            print(f"stderr: {stderr.decode()[:500]}")

        if stdout:
            try:
                response = json.loads(stdout.decode())
                result = response.get("result", {})

                if "results" in result:
                    results = result["results"]
                    print(f"\n✅ 搜索成功！获取到 {len(results)} 条结果:\n")

                    for i, item in enumerate(results, 1):
                        print(f"--- 结果 {i} ---")
                        print(f"标题: {item.get('title', 'N/A')}")
                        print(f"URL: {item.get('url', 'N/A')}")
                        print(f"摘要: {item.get('snippet', 'N/A')[:150]}...")
                        print()

                    return True
                else:
                    print(f"❌ 响应格式异常: {result}")
                    return False
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                print(f"原始输出: {stdout.decode()[:500] if stdout else 'None'}")
                return False
        else:
            print("❌ 无输出")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


async def test_search_engine_batch():
    """测试 search_engine_batch 工具"""
    print("\n" + "=" * 60)
    print("测试4: 测试 search_engine_batch 工具")
    print("=" * 60)

    rpc_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_engine_batch",
            "arguments": {
                "queries": [
                    "transformer attention mechanism",
                    "GPT-4 clinical applications",
                    "medical NLP research"
                ],
                "num_results": 3
            }
        }
    }

    api_token = "88b654f6-f6b0-4e8d-85d5-c50dc5e2d3c5"

    try:
        print("发送批量搜索请求...")
        print(f"查询数量: 3")
        print(f"每个查询结果数: 3")

        env = {**subprocess.os.environ, "API_TOKEN": api_token}

        proc = await asyncio.create_subprocess_exec(
            "npx", "@brightdata/mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps(rpc_request).encode()),
                timeout=90
            )
        except asyncio.TimeoutError:
            proc.kill()
            print("❌ 请求超时 (90s)")
            return False

        print(f"\n返回码: {proc.returncode}")

        if stdout:
            try:
                response = json.loads(stdout.decode())
                result = response.get("result", {})

                total_results = 0
                for query, data in result.items():
                    results = data.get("results", []) if isinstance(data, dict) else []
                    total_results += len(results)
                    print(f"\n✅ 查询 '{query[:40]}...': {len(results)} 条结果")

                print(f"\n✅ 批量搜索成功！共 {total_results} 条结果")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                print(f"原始输出: {stdout.decode()[:500]}")
                return False
        else:
            print("❌ 无输出")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


async def main():
    """主测试流程"""
    print("\n🔍 Bright Data MCP 网络搜索功能测试")
    print("=" * 60)

    results = {}

    # 测试1
    results["npx可用"] = await test_npx_available()

    # 测试2
    results["MCP服务器"] = await test_mcp_server()

    # 测试3 - 核心测试
    results["search_engine"] = await test_search_engine()

    # 测试4 - 批量测试
    results["search_engine_batch"] = await test_search_engine_batch()

    # 汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！Bright Data 网络搜索功能正常")
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
