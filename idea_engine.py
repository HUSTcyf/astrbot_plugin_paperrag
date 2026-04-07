"""
研究创意生成引擎

整合Bright Data网络搜索 + 本地Paper RAG + LLM生成
"""

import json
import re
import asyncio
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import logger


@dataclass
class ResearchIdea:
    """研究想法"""
    title: str
    description: str
    novelty: str
    methodology: str
    potential_challenges: List[str]
    related_work: List[str]
    feasibility: float
    inspiration_sources: List[str]


@dataclass
class TopicAnalysis:
    """主题分析结果"""
    domain: str
    keywords: List[str]
    search_queries: List[str]
    local_rag_queries: List[str]
    exploration_angles: List[str]
    summary: str


class IdeaEngine:
    """
    研究创意生成引擎

    使用流程：
    1. analyze_topic - 分析研究主题
    2. search_knowledge - 收集知识（网络+本地）
    3. generate_ideas - 生成研究想法
    """

    def __init__(self, context, rag_engine=None):
        """
        初始化创意引擎

        Args:
            context: AstrBot上下文（用于LLM调用）
            rag_engine: RAG引擎实例
        """
        self.context = context
        self._rag_engine = rag_engine
        self._bright_data_available = True

    def _get_llm_provider(self):
        """获取LLM provider"""
        if not self.context:
            return None
        # 尝试获取当前正在使用的provider
        provider = getattr(self.context, 'get_using_provider', None)
        if provider:
            return provider()
        # fallback: 尝试通过provider_manager获取
        provider_manager = getattr(self.context, 'provider_manager', None)
        if provider_manager:
            inst_map = getattr(provider_manager, 'inst_map', None)
            if isinstance(inst_map, dict) and inst_map:
                return list(inst_map.values())[0]
        return None

    async def analyze_topic(self, topic: str, depth: str = "standard") -> TopicAnalysis:
        """
        分析研究主题，生成搜索策略

        Args:
            topic: 研究话题
            depth: 分析深度 (quick/standard/deep)

        Returns:
            TopicAnalysis: 结构化的主题分析
        """
        logger.info(f"[IdeaEngine] 分析主题: {topic}, 深度: {depth}")

        depth_config = {
            "quick": {"num_queries": 3, "num_angles": 2},
            "standard": {"num_queries": 5, "num_angles": 4},
            "deep": {"num_queries": 10, "num_angles": 6}
        }
        config = depth_config.get(depth, depth_config["standard"])

        prompt = f"""分析以下研究主题，生成结构化的信息收集计划：

研究主题：{topic}

请分析并返回以下JSON格式的信息：

{{
    "domain": "研究领域",
    "keywords": ["关键词1", "关键词2", ...],
    "search_queries": ["查询1", "查询2", ...],
    "local_rag_queries": ["本地检索词1", "本地检索词2", ...],
    "exploration_angles": ["角度1", "角度2", ...],
    "summary": "主题摘要"
}}

请严格按照JSON格式返回，不要包含其他文字。"""

        provider = self._get_llm_provider()
        if not provider:
            logger.error("[IdeaEngine] LLM provider未初始化")
            return None

        try:
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.1,
                max_tokens=2048
            )

            response_text = ""
            # 方法1：检查 result_chain（AstrBot 格式）
            if hasattr(response, 'result_chain'):
                chain = getattr(response.result_chain, 'chain', None)
                if chain and len(chain) > 0:
                    first = chain[0]
                    if hasattr(first, 'get_text'):
                        response_text = first.get_text()
                    elif hasattr(first, 'text'):
                        response_text = first.text
            # 方法2：检查 content 属性（LlamaCpp 格式）
            elif hasattr(response, 'content'):
                response_text = response.content
            # 方法3：dict 格式
            elif isinstance(response, dict):
                response_text = response.get("content", "") or response.get("text", "")
            # 方法4：字符串格式
            else:
                response_text = str(response)
            result = self._parse_json_response(response_text)

            if result:
                return TopicAnalysis(
                    domain=result.get("domain", ""),
                    keywords=result.get("keywords", []),
                    search_queries=result.get("search_queries", []),
                    local_rag_queries=result.get("local_rag_queries", []),
                    exploration_angles=result.get("exploration_angles", []),
                    summary=result.get("summary", "")
                )

        except Exception as e:
            logger.error(f"[IdeaEngine] 主题分析失败: {e}")

        return None

    async def search_knowledge(
        self,
        queries: List[str],
        local_rag_top_k: int = 5,
        web_top_k: int = 10
    ) -> Dict[str, Any]:
        """
        多源知识检索

        Args:
            queries: 搜索查询列表
            local_rag_top_k: 本地RAG召回数
            web_top_k: 网络搜索召回数

        Returns:
            Dict包含 web_results, local_results, fused_context
        """
        logger.info(f"[IdeaEngine] 检索知识，查询数: {len(queries)}")

        web_results = []
        local_results = []

        # 1. 本地RAG搜索
        if self._rag_engine and local_rag_top_k > 0:
            try:
                for query in queries[:5]:  # 限制查询数
                    result = await self._rag_engine.search(query, mode="retrieve")
                    sources = result.get("sources", [])
                    for src in sources[:local_rag_top_k]:
                        local_results.append({
                            "text": src.get("text", "")[:500],
                            "paper": src.get("metadata", {}).get("file_name", "Unknown"),
                            "page": str(src.get("metadata", {}).get("page", "")),
                            "score": src.get("score", 0.0)
                        })
            except Exception as e:
                logger.error(f"[IdeaEngine] 本地RAG搜索失败: {e}")

        # 2. 网络搜索（通过Bright Data MCP）
        if self._bright_data_available:
            try:
                web_results = await self._search_web(queries, web_top_k)
            except Exception as e:
                logger.error(f"[IdeaEngine] 网络搜索失败: {e}")

        # 3. 知识融合
        fused_context = self._fuse_knowledge(web_results, local_results)

        return {
            "web_results": web_results,
            "local_results": local_results,
            "fused_context": fused_context,
            "stats": {
                "web_count": len(web_results),
                "local_count": len(local_results)
            }
        }

    async def _search_web(self, queries: List[str], top_k: int) -> List[Dict]:
        """通过网络搜索获取信息（通过Bright Data MCP）"""
        results = []

        try:
            # API Token - 从 mcp_server.json 读取
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            try:
                with open(mcp_config_path, "r", encoding="utf-8") as f:
                    mcp_config = json.load(f)
                api_token = mcp_config.get("mcpServers", {}).get("BrightData", {}).get("env", {}).get("API_TOKEN", "")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"无法从 {mcp_config_path} 读取 BrightData API Token: {e}")

            if not api_token:
                raise ValueError("BrightData API Token 未配置")

            # 启动Bright Data MCP服务器
            env = {**os.environ, "API_TOKEN": api_token}

            proc = await asyncio.create_subprocess_exec(
                "npx", "@brightdata/mcp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            for query in queries[:5]:
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "search_engine",
                        "arguments": {
                            "query": query,
                            "num_results": top_k,
                            "source": "web"
                        }
                    }
                }

                # MCP协议要求每行一个JSON-RPC消息
                request_str = json.dumps(rpc_request) + "\n"

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=request_str.encode()),
                        timeout=30
                    )

                    if stdout:
                        response = json.loads(stdout.decode())
                        content = response.get("result", {}).get("content", [])

                        # 解析搜索结果
                        if content and len(content) > 0:
                            text = content[0].get("text", "")
                            if text:
                                try:
                                    data = json.loads(text)
                                    organic = data.get("organic", [])
                                    for item in organic:
                                        results.append({
                                            "title": item.get("title", ""),
                                            "url": item.get("link", ""),
                                            "snippet": item.get("description", "")
                                        })
                                except json.JSONDecodeError:
                                    pass

                except asyncio.TimeoutError:
                    logger.warning(f"[IdeaEngine] 查询超时: {query}")
                    continue

            # 关闭进程
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

        except Exception as e:
            logger.error(f"[IdeaEngine] Bright Data调用失败: {e}")

        return results

    async def generate_ideas(
        self,
        knowledge_context: str,
        research_domain: str = "",
        num_ideas: int = 3,
        idea_focus: str = "all"
    ) -> List[ResearchIdea]:
        """
        基于知识上下文生成研究想法

        Args:
            knowledge_context: 融合后的知识上下文
            research_domain: 研究领域
            num_ideas: 生成想法数量
            idea_focus: 侧重点 (novelty/feasibility/impact/all)

        Returns:
            List[ResearchIdea]: 研究想法列表
        """
        logger.info(f"[IdeaEngine] 生成{num_ideas}个研究想法")

        focus_instruction = {
            "novelty": "特别强调创新性和独特贡献",
            "feasibility": "特别强调技术可行性和实现路径",
            "impact": "特别强调潜在影响力和应用价值",
            "all": "综合考虑创新性、可行性和影响力"
        }.get(idea_focus, "")

        prompt = f"""基于以下收集的知识上下文，生成{num_ideas}个研究想法。

研究领域：{research_domain or "通用研究领域"}

收集的知识：
{knowledge_context[:8000]}

{focus_instruction}

请为每个想法返回以下JSON格式的信息：

{{
    "ideas": [
        {{
            "title": "想法标题",
            "description": "详细描述",
            "novelty": "创新点",
            "methodology": "方法论建议",
            "potential_challenges": ["挑战1", "挑战2"],
            "related_work": ["相关工作1", "相关工作2"],
            "feasibility": 0.8,
            "inspiration_sources": ["灵感来源1", "灵感来源2"]
        }},
        ...
    ],
    "analysis_summary": "对现有工作的分析总结"
}}

请严格按照JSON格式返回，只返回JSON，不要包含其他文字。"""

        provider = self._get_llm_provider()
        if not provider:
            logger.error("[IdeaEngine] LLM provider未初始化")
            return []

        try:
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                temperature=0.7,
                max_tokens=4096
            )

            response_text = ""
            # 方法1：检查 result_chain（AstrBot 格式）
            if hasattr(response, 'result_chain'):
                chain = getattr(response.result_chain, 'chain', None)
                if chain and len(chain) > 0:
                    first = chain[0]
                    if hasattr(first, 'get_text'):
                        response_text = first.get_text()
                    elif hasattr(first, 'text'):
                        response_text = first.text
            # 方法2：检查 content 属性（LlamaCpp 格式）
            elif hasattr(response, 'content'):
                response_text = response.content
            # 方法3：dict 格式
            elif isinstance(response, dict):
                response_text = response.get("content", "") or response.get("text", "")
            # 方法4：字符串格式
            else:
                response_text = str(response)
            result = self._parse_json_response(response_text)

            if result and "ideas" in result:
                ideas = []
                for item in result["ideas"][:num_ideas]:
                    ideas.append(ResearchIdea(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        novelty=item.get("novelty", ""),
                        methodology=item.get("methodology", ""),
                        potential_challenges=item.get("potential_challenges", []),
                        related_work=item.get("related_work", []),
                        feasibility=item.get("feasibility", 0.5),
                        inspiration_sources=item.get("inspiration_sources", [])
                    ))
                return ideas

        except Exception as e:
            logger.error(f"[IdeaEngine] 创意生成失败: {e}")

        return []

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """解析LLM返回的JSON响应"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON块
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # 尝试提取JSON对象
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"[IdeaEngine] JSON解析失败: {text[:200]}")
        return None

    def _fuse_knowledge(
        self,
        web_results: List[Dict],
        local_results: List[Dict]
    ) -> str:
        """将多源知识融合为统一上下文"""
        parts = ["# 收集到的相关知识\n"]

        # 网络资源
        if web_results:
            parts.append("## 网络资源\n")
            for i, r in enumerate(web_results[:10], 1):
                parts.append(f"{i}. **{r.get('title', '')}**")
                parts.append(f"   {r.get('snippet', '')}")
                parts.append("")

        # 本地论文
        if local_results:
            parts.append("## 本地论文库\n")
            papers = {}
            for r in local_results:
                paper = r.get("paper", "Unknown")
                if paper not in papers:
                    papers[paper] = []
                papers[paper].append(r)

            for paper, chunks in list(papers.items())[:5]:
                parts.append(f"### {paper}")
                for chunk in chunks[:2]:
                    text = chunk.get("text", "")[:300]
                    if text:
                        parts.append(f"- {text}...")
                parts.append("")

        return "\n".join(parts)

    async def to_feishu_markdown(
        self,
        ideas: List[ResearchIdea],
        topic: str = "",
        include_sources: bool = True
    ) -> str:
        """
        将研究想法格式化为飞书文档兼容的Markdown格式

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            include_sources: 是否包含灵感来源

        Returns:
            str: 飞书兼容的Markdown格式内容
        """
        if not ideas:
            return ""

        markdown_parts = [f"# {topic or '研究想法'}\n" if topic else "# 研究想法\n"]

        for i, idea in enumerate(ideas, 1):
            feasibility_bar = "★" * int(idea.feasibility * 5) + "☆" * (5 - int(idea.feasibility * 5))

            markdown_parts.append(f"## {i}. {idea.title}\n")
            markdown_parts.append(f"**可行性**: {feasibility_bar} ({idea.feasibility:.0%})\n")
            markdown_parts.append(f"\n### 描述\n{idea.description}\n")
            markdown_parts.append(f"\n### 创新点\n{idea.novelty}\n")
            markdown_parts.append(f"\n### 方法论\n{idea.methodology}\n")

            if idea.potential_challenges:
                markdown_parts.append("\n### 潜在挑战\n")
                for challenge in idea.potential_challenges:
                    markdown_parts.append(f"- {challenge}\n")

            if idea.related_work:
                markdown_parts.append("\n### 相关工作\n")
                for work in idea.related_work:
                    markdown_parts.append(f"- {work}\n")

            if include_sources and idea.inspiration_sources:
                markdown_parts.append("\n### 灵感来源\n")
                for source in idea.inspiration_sources:
                    markdown_parts.append(f"- {source}\n")

            markdown_parts.append("\n---\n")

        return "".join(markdown_parts)

    async def create_feishu_document(
        self,
        ideas: List[ResearchIdea],
        topic: str = "",
        folder_token: str = ""
    ) -> Dict[str, Any]:
        """
        创建飞书文档并写入研究想法

        Args:
            ideas: 研究想法列表
            topic: 研究主题
            folder_token: 飞书文件夹Token（可选）

        Returns:
            Dict包含 document_id, url, error 等信息
        """
        try:
            # 1. 格式化内容为飞书Markdown
            markdown_content = await self.to_feishu_markdown(ideas, topic)
            if not markdown_content:
                return {"error": "内容为空"}

            # 2. 生成飞书块格式
            blocks = self._markdown_to_feishu_blocks(markdown_content)

            # 3. 调用 feishu-mcp 创建文档
            doc_result = await self._call_feishu_mcp_create_doc(topic or "研究想法", folder_token)
            if doc_result.get("error"):
                return doc_result

            document_id = doc_result.get("document_id")
            if not document_id:
                return {"error": "文档创建失败，未返回文档ID"}

            # 4. 添加内容块
            block_result = await self._call_feishu_mcp_add_blocks(document_id, blocks)
            if block_result.get("error"):
                return {
                    "document_id": document_id,
                    "url": f"https://feishu.cn/document/{document_id}",
                    "warning": f"文档已创建但内容添加失败: {block_result.get('error')}"
                }

            return {
                "document_id": document_id,
                "url": f"https://feishu.cn/document/{document_id}",
                "title": topic or "研究想法",
                "blocks_created": len(blocks)
            }

        except Exception as e:
            logger.error(f"[IdeaEngine] 创建飞书文档失败: {e}")
            return {"error": str(e)}

    def _markdown_to_feishu_blocks(self, markdown_text: str) -> List[Dict]:
        """将Markdown文本转换为飞书块格式"""
        blocks = []
        lines = markdown_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.startswith("# ") and not line.startswith("## "):
                # 一级标题
                blocks.append({
                    "blockType": "heading",
                    "options": {
                        "heading": {
                            "level": 1,
                            "content": line[2:].strip()
                        }
                    }
                })
            elif line.startswith("## "):
                # 二级标题
                blocks.append({
                    "blockType": "heading",
                    "options": {
                        "heading": {
                            "level": 2,
                            "content": line[3:].strip()
                        }
                    }
                })
            elif line.startswith("### "):
                # 三级标题
                blocks.append({
                    "blockType": "heading",
                    "options": {
                        "heading": {
                            "level": 3,
                            "content": line[4:].strip()
                        }
                    }
                })
            elif line.startswith("**") and line.endswith("**"):
                # 加粗文本作为标题
                blocks.append({
                    "blockType": "text",
                    "options": {
                        "text": {
                            "textStyles": [
                                {
                                    "text": line[2:-2].strip(),
                                    "style": {"bold": True}
                                }
                            ]
                        }
                    }
                })
            elif line.startswith("- "):
                # 无序列表
                blocks.append({
                    "blockType": "bullet",
                    "options": {
                        "bullet": {
                            "content": line[2:].strip()
                        }
                    }
                })
            elif line.startswith("---"):
                # 分割线
                blocks.append({
                    "blockType": "horizontal_line",
                    "options": {}
                })
            elif line.strip() == "":
                # 空行，跳过
                pass
            else:
                # 普通文本
                text_content = line.strip()
                if text_content:
                    blocks.append({
                        "blockType": "text",
                        "options": {
                            "text": {
                                "textStyles": [{"text": text_content}]
                            }
                        }
                    })

            i += 1

        return blocks

    async def _call_feishu_mcp_create_doc(
        self,
        title: str,
        folder_token: str = ""
    ) -> Dict[str, Any]:
        """调用 feishu-mcp 创建文档"""
        try:
            # 从 mcp_server.json 读取配置
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            logger.info(f"[IdeaEngine] 读取 MCP 配置: {mcp_config_path}")
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)

            feishu_config = mcp_config.get("mcpServers", {}).get("feishu", {})
            env_vars = feishu_config.get("env", {})
            logger.info(f"[IdeaEngine] feishu env keys: {list(env_vars.keys())}")

            # 构建 MCP 请求
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "create_feishu_document",
                    "arguments": {
                        "title": title,
                        "folderToken": folder_token
                    }
                }
            }
            logger.info(f"[IdeaEngine] MCP 请求: {json.dumps(mcp_request)[:200]}")

            # 调用 feishu-mcp
            logger.info("[IdeaEngine] 启动 feishu-mcp 进程...")
            proc = await asyncio.create_subprocess_exec(
                "npx", "-y", "feishu-mcp@latest", "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **env_vars}
            )
            logger.info(f"[IdeaEngine] feishu-mcp PID: {proc.pid}")

            # 发送请求
            request_data = json.dumps(mcp_request).encode()
            logger.info(f"[IdeaEngine] 发送请求 ({len(request_data)} bytes)...")
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=request_data),
                timeout=120
            )

            logger.info(f"[IdeaEngine] feishu-mcp 返回: stdout={len(stdout)} bytes, stderr={len(stderr)} bytes")

            if stderr:
                stderr_text = stderr.decode(errors="replace")
                logger.warning(f"[IdeaEngine] feishu-mcp stderr: {stderr_text[:500]}")

            if stdout:
                stdout_text = stdout.decode(errors="replace")
                logger.info(f"[IdeaEngine] feishu-mcp stdout: {stdout_text[:1000]}")
                try:
                    response = json.loads(stdout_text)
                    logger.info(f"[IdeaEngine] 解析响应成功: {str(response)[:200]}")
                    result = response.get("result", {}).get("content", [{}])[0].get("text", "{}")
                    logger.info(f"[IdeaEngine] feishu-mcp result: {result[:500]}")
                    doc_info = json.loads(result)
                    document_id = doc_info.get("document_id") or doc_info.get("objToken") or doc_info.get("obj_token")
                    if document_id:
                        logger.info(f"[IdeaEngine] 文档创建成功: {document_id}")
                        return {"document_id": document_id}
                    return {"error": f"创建文档失败: {result}"}
                except json.JSONDecodeError as e:
                    logger.error(f"[IdeaEngine] JSON解析失败: {e}, stdout: {stdout_text[:500]}")
                    return {"error": f"JSON解析失败: {e}"}

            # 无 stdout 时打印更多信息
            logger.error(f"[IdeaEngine] feishu-mcp 无 stdout 输出, proc.returncode={proc.returncode}")
            if stderr:
                logger.error(f"[IdeaEngine] stderr: {stderr.decode(errors='replace')[:500]}")
            return {"error": f"feishu-mcp 无响应 (returncode={proc.returncode})"}

        except asyncio.TimeoutError:
            logger.error("[IdeaEngine] feishu-mcp 调用超时")
            return {"error": "feishu-mcp 调用超时"}
        except Exception as e:
            import traceback
            logger.error(f"[IdeaEngine] 调用 feishu-mcp 失败: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

    async def _call_feishu_mcp_add_blocks(
        self,
        document_id: str,
        blocks: List[Dict]
    ) -> Dict[str, Any]:
        """调用 feishu-mcp 添加内容块"""
        try:
            # 从 mcp_server.json 读取配置
            mcp_config_path = Path(__file__).parent.parent.parent / "mcp_server.json"
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)

            feishu_config = mcp_config.get("mcpServers", {}).get("feishu", {})
            env_vars = feishu_config.get("env", {})

            # 构建 MCP 请求
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "batch_create_feishu_blocks",
                    "arguments": {
                        "documentId": document_id,
                        "parentBlockId": "0",
                        "index": 0,
                        "blocks": blocks
                    }
                }
            }

            # 调用 feishu-mcp
            proc = await asyncio.create_subprocess_exec(
                "npx", "-y", "feishu-mcp@latest", "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **env_vars}
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps(mcp_request).encode()),
                timeout=120
            )

            if stderr:
                stderr_text = stderr.decode()
                logger.warning(f"[IdeaEngine] feishu-mcp blocks stderr: {stderr_text}")

            if stdout:
                response = json.loads(stdout.decode())
                result = response.get("result", {}).get("content", [{}])[0].get("text", "{}")
                result_data = json.loads(result)
                if result_data.get("totalBlocksCreated", 0) > 0:
                    return {"success": True, "blocks_created": result_data.get("totalBlocksCreated")}
                return {"error": result_data}

            return {"error": "feishu-mcp 无响应"}

        except json.JSONDecodeError as e:
            logger.error(f"[IdeaEngine] 解析 feishu-mcp 响应失败: {e}")
            return {"error": f"解析响应失败: {e}"}
        except asyncio.TimeoutError:
            return {"error": "feishu-mcp 调用超时"}
        except Exception as e:
            logger.error(f"[IdeaEngine] 调用 feishu-mcp 失败: {e}")
            return {"error": str(e)}
