"""
Bright Data 网络搜索功能（独立模块）
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger


class IdeaEngineWebSearch:
    """
    Bright Data MCP 网络搜索功能。

    包含：search_engine, discover, scrape_as_markdown, scrape_batch
    不依赖继承链，可被 IdeaEngineGeneration 直接调用。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mcp_proc: Optional[Any] = None
        self._mcp_lock = asyncio.Lock()

    async def _ensure_mcp_process(self) -> Optional[Any]:
        """获取或启动持久化的 MCP 进程（进程级单例）"""
        if self._mcp_proc is not None:
            try:
                if self._mcp_proc.returncode is None:
                    return self._mcp_proc
                self._mcp_proc.terminate()
                await asyncio.wait_for(self._mcp_proc.wait(), timeout=1)
            except Exception:
                pass
            self._mcp_proc = None

        # websearch.py → idea/ → astrbot_plugin_paperrag/ → plugins/ → data/
        data_dir = Path(__file__).resolve().parent.parent.parent.parent
        mcp_config_path = data_dir / "mcp_server.json"
        try:
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)
            api_token = mcp_config.get("mcpServers", {}).get("BrightData", {}).get("env", {}).get("API_TOKEN", "")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"[IdeaEngine] 无法读取 MCP 配置: {e}")
            return None

        if not api_token:
            logger.error(f"[IdeaEngine] BrightData API Token 未配置")
            return None

        env = {**os.environ, "API_TOKEN": api_token}
        proc = await asyncio.create_subprocess_exec(
            "npx", "@brightdata/mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        await asyncio.sleep(2)
        self._mcp_proc = proc
        return proc

    async def _close_mcp_process(self):
        """关闭 MCP 进程"""
        if self._mcp_proc:
            try:
                self._mcp_proc.terminate()
                await asyncio.wait_for(self._mcp_proc.wait(), timeout=5)
            except Exception:
                try:
                    self._mcp_proc.kill()
                except Exception:
                    pass
            self._mcp_proc = None

    async def _call_brightdata_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        通用 Bright Data MCP 工具调用方法

        支持的工具:
        - search_engine: 搜索引擎搜索
        - search_engine_batch: 批量搜索引擎搜索
        - scrape_as_markdown: 抓取单个页面为 Markdown
        - scrape_batch: 批量抓取页面为 Markdown
        - discover: AI 驱动的智能搜索

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            timeout: 超时时间（秒）

        Returns:
            Dict 包含工具执行结果
        """
        try:
            async with self._mcp_lock:
                proc = await self._ensure_mcp_process()
                if proc is None:
                    return {"success": False, "error": "MCP 进程启动失败"}

            # 构建请求
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }

            request_str = json.dumps(rpc_request) + "\n"
            logger.info(f"[IdeaEngine] Bright Data MCP 调用: {tool_name}, 参数: {json.dumps(arguments)[:200]}")

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=request_str.encode()),
                    timeout=timeout
                )

                if stderr:
                    stderr_text = stderr.decode()
                    logger.info(f"[IdeaEngine] Bright Data stderr: {stderr_text}")

                if stdout:
                    stdout_text = stdout.decode().strip()
                    response = None
                    for line in stdout_text.split('\n'):
                        line = line.strip()
                        if line and line.startswith('{'):
                            try:
                                parsed = json.loads(line)
                                if "result" in parsed:
                                    response = parsed
                                    break
                            except json.JSONDecodeError:
                                continue
                    if response is None:
                        try:
                            response = json.loads(stdout_text)
                        except json.JSONDecodeError as e:
                            logger.warning(f"[IdeaEngine] JSON 解析失败: {e}, 内容: {stdout_text}")
                            return {"success": False, "error": f"JSON 解析失败: {e}"}
                    content = response.get("result", {}).get("content", [])
                    logger.info(f"[IdeaEngine] Bright Data MCP 原始响应: response_keys={list(response.keys()) if response else None}, content长度={len(content) if content else 0}")

                    if content and len(content) > 0:
                        text = content[0].get("text", "")
                        if text:
                            try:
                                data = json.loads(text)
                                logger.info(f"[IdeaEngine] Bright Data MCP 解析成功, data_keys={list(data.keys()) if isinstance(data, dict) else 'list'}")
                                return {"success": True, "data": data}
                            except json.JSONDecodeError:
                                logger.info(f"[IdeaEngine] Bright Data MCP text非JSON，返回原文")
                                return {"success": True, "data": text}

                    logger.warning(f"[IdeaEngine] Bright Data MCP 无有效content或text为空")
                    return {"success": True, "data": None}

            except asyncio.TimeoutError:
                logger.warning(f"[IdeaEngine] Bright Data MCP 调用超时: {tool_name}")
                return {"success": False, "error": "调用超时"}

        except Exception as e:
            logger.error(f"[IdeaEngine] Bright Data MCP 调用失败: {e}")
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "未知错误"}

    async def _search_web(self, queries: List[str], top_k: int) -> List[Dict]:
        """通过网络搜索获取信息（通过Bright Data MCP）"""
        results = []
        for query in queries[:5]:
            try:
                result = await self._call_brightdata_mcp_tool(
                    "search_engine",
                    {"query": query, "num_results": top_k, "language": "zh-CN"}
                )
                if result.get("success"):
                    results.extend(result.get("results", []))
            except Exception as e:
                logger.warning(f"[IdeaEngine] 网络搜索失败: {e}")
        return results[:top_k]

    async def _search_engine_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量搜索引擎搜索"""
        return await self._call_brightdata_mcp_tool("search_engine_batch", {"queries": queries})

    async def _discover_search(
        self,
        query: str,
        intent: str = "",
        country: str = "US",
        num_results: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """AI 驱动的智能搜索"""
        return await self._call_brightdata_mcp_tool("discover", {
            "query": query, "intent": intent, "country": country, "num_results": num_results, **kwargs
        })

    async def _scrape_as_markdown(self, url: str) -> Dict[str, Any]:
        """抓取单个页面为 Markdown"""
        return await self._call_brightdata_mcp_tool("scrape_as_markdown", {"url": url})

    async def _scrape_batch_markdown(self, urls: List[str]) -> Dict[str, Any]:
        """批量抓取页面为 Markdown"""
        return await self._call_brightdata_mcp_tool("scrape_batch", {"urls": urls})

    async def test_brightdata_mcp(self, query: str) -> Dict[str, Any]:
        """测试 Bright Data MCP 搜索功能"""
        result = await self._call_brightdata_mcp_tool(
            tool_name="search_engine",
            arguments={
                "query": query,
                "num_results": 5,
                "source": "web"
            }
        )
        if result.get("success"):
            data = result.get("data", {})
            organic = data.get("organic", []) if isinstance(data, dict) else []
            return {
                "success": True,
                "results": organic
            }
        return result
