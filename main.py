"""
AstrBot Paper RAG Plugin
本地文档库RAG检索插件
支持基于Gemini Embedding + Milvus Lite的文档检索和问答

支持的文档格式:
- PDF (.pdf) - 使用PyMuPDF高效解析
- Word文档 (.docx, .doc) - 使用python-docx解析
- 纯文本 (.txt, .md) - 支持UTF-8和GBK编码
- HTML (.html, .htm) - 使用unstructured解析
- 其他格式 - 通过unstructured库自动解析（需安装）
"""

import os
import sys

# 确保插件根目录在 sys.path 中，使 rag 包可被绝对导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import register

import logging
for _ln in ("neo4j",):
    logging.getLogger(_ln).setLevel(logging.WARNING)

from .commands import ArxivCommandsMixin, GraphCommandsMixin, IdeaCommandsMixin, PaperCommandsMixin, RemoteCodeMixin
from .commands.base import PluginCoreBase


@register(
    "paper_rag",
    "HUSTcyf",
    "本地文档库RAG检索插件 (支持PDF/Word/TXT/HTML, Gemini + Milvus Lite)",
    "2.2.3",
    "https://github.com/HUSTcyf/astrbot_plugin_paperrag.git"
)
class PaperRAGPlugin(PaperCommandsMixin, ArxivCommandsMixin, GraphCommandsMixin, IdeaCommandsMixin, RemoteCodeMixin, PluginCoreBase):
    """论文RAG检索插件"""

    def __init__(self, context, config: dict = {}):
        super().__init__(context, config)
        # 注册 agentic_rag 为 LLM Tool
        context.register_llm_tool(
            name="paper_arag",
            func_args=[
                {"type": "string", "name": "query", "description": "复杂论文问答查询，支持多跳推理、对比分析、引用溯源"},
                {"type": "integer", "name": "top_k", "description": "召回数，默认5", "default": 5}
            ],
            desc="用于需要多跳推理、对比分析、引用溯源的复杂论文问题",
            func_obj=self._agentic_rag_tool,
        )
        # 注册 ReAct Agent 为 LLM Tool
        context.register_llm_tool(
            name="paper_react",
            func_args=[
                {"type": "string", "name": "query", "description": "论文问答查询，Agent 自主决定使用哪些检索工具"},
                {"type": "integer", "name": "top_k", "description": "召回数，默认5", "default": 5}
            ],
            desc="用于论文问答的智能 Agent，可自主选择向量检索或知识图谱检索",
            func_obj=self._react_rag_tool,
        )
        # 注册基础 RAG 检索为 LLM Tool
        context.register_llm_tool(
            name="paper_search",
            func_args=[
                {"type": "string", "name": "query", "description": "论文检索查询，从本地论文库中搜索相关内容"},
                {"type": "integer", "name": "top_k", "description": "召回数，默认5", "default": 5}
            ],
            desc="基础论文 RAG 检索：直接搜索本地论文库并生成回答。比 paper_arag/paper_react 更轻量快速，适合简单的论文内容查询",
            func_obj=self._paper_search_tool,
        )
        # 注册 Claude Code 编程执行为 LLM Tool
        context.register_llm_tool(
            name="code_execute",
            func_args=[
                {"type": "string", "name": "task", "description": "完整的编程任务描述，需包含所有必要上下文和指令"},
                {"type": "integer", "name": "timeout", "description": "最大执行秒数，默认300", "default": 300},
            ],
            desc="使用 Claude Code 执行编程任务：写/改代码、调试、运行实验、重构、git 操作等。agent 应先调用 paper_search/paper_arag/paper_react 检索相关知识，整合后形成完整任务再调用此工具",
            func_obj=self._code_execute_tool,
        )

    # ==================== Paper 命令组 ====================
    @filter.command_group("paper")
    def paper_commands(self):
        """Paper RAG command group
        search         - Search documents and answer questions
        arag           - Agentic RAG complex query (static DAG)
        react          - Tool-Using Agent (ReAct mode, dynamic tool selection)
        list           - List indexed documents
        add            - Add documents to knowledge base (PDF/Word/TXT supported)
        addf           - Add a single document to knowledge base
        delete         - Delete a specific paper from knowledge base
        clear          - Clear knowledge base
        rebuild        - Clear and re-add all documents
        refstats       - Show reference title frequency statistics (-1 for zero-ref papers, dedup=1 for deduplicated count)
        arxiv_add      - Search arxiv and download papers, then add to database (Admin)
        arxiv_refs     - Download highly-cited reference papers from arxiv (Admin)
        arxiv_sync     - Sync MCP downloaded papers to paperrag database (Admin)
        arxiv_cleanup  - Clean up old versions of arxiv papers (Admin)
        graph_build    - Build knowledge graph from indexed documents
        graph_rebuild  - Rebuild knowledge graph from scratch (clear + rebuild)
        graph_stats    - Show knowledge graph statistics
        graph_clear    - Clear knowledge graph (Admin)
        abstractstats  - Show abstract extraction statistics (-1 for papers without abstracts)
        reparse_zero_abstract - Batch re-extract abstracts for papers without abstracts (Admin)
        reparseref     - Re-parse references for a single paper (Admin, no full rebuild)
        """
        pass

    @paper_commands.command("search")
    async def cmd_search(self, event: AstrMessageEvent, query: str = '', top_k: int = 5):
        """Search document library and answer questions"""
        async for result in self._paper_search(event, query=query, top_k=top_k):
            yield result

    @paper_commands.command("list")
    async def cmd_list(self, event: AstrMessageEvent):
        """List all documents in the library"""
        async for result in self._paper_list(event):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("add")
    async def cmd_add(self, event: AstrMessageEvent, directory: str = ''):
        """Add documents to knowledge base (Admin)"""
        async for result in self._paper_add(event, directory=directory):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("addf")
    async def cmd_add_file(self, event: AstrMessageEvent, file_path: str = ''):
        """Add a single document to knowledge base (Admin)"""
        async for result in self._paper_addf(event, file_path=file_path):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("clear")
    async def cmd_clear(self, event: AstrMessageEvent, confirm: str = ''):
        """Clear document knowledge base (Admin)"""
        async for result in self._paper_clear(event, confirm=confirm):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("delete")
    async def cmd_delete(self, event: AstrMessageEvent, file_name: str = ''):
        """Delete a specific paper from knowledge base (Admin)"""
        async for result in self._paper_delete(event, file_name=file_name):
            yield result

    @paper_commands.command("refstats")
    async def cmd_refstats(self, event: AstrMessageEvent, top_k: int = 20, dedup: int = 0):
        """Show reference title frequency statistics"""
        async for result in self._paper_refstats(event, top_k=top_k, dedup=dedup):
            yield result

    @paper_commands.command("abstractstats")
    async def cmd_abstractstats(self, event: AstrMessageEvent, top_k: int = 20):
        """Show abstract extraction statistics"""
        async for result in self._paper_abstractstats(event, top_k=top_k):
            yield result

    @paper_commands.command("arag")
    async def cmd_arag(self, event: AstrMessageEvent, query: str = '', top_k: int = 5):
        """Agentic RAG complex query (multi-hop reasoning / comparison / citation tracing)

        Args:
            top_k: Number of results to return (default: 5)
        """
        async for result in self._agentic_rag(event, query=query, top_k=top_k):
            yield result

    @paper_commands.command("react")
    async def cmd_react(self, event: AstrMessageEvent, query: str = '', top_k: int = 5):
        """Tool-Using Agent (ReAct mode) for paper Q&A

        Args:
            top_k: Number of results to return (default: 5)
        """
        async for result in self._react_rag(event, query=query, top_k=top_k):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("reparse_zero_ref")
    async def cmd_reparse_zero_ref(self, event: AstrMessageEvent, confirm: str = ''):
        """Batch re-parse papers with zero references (Admin)"""
        async for result in self._paper_reparse_zero_ref(event, confirm=confirm):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("reparse_zero_abstract")
    async def cmd_reparse_zero_abstract(self, event: AstrMessageEvent, confirm: str = ''):
        """Batch re-extract abstracts for papers without abstracts (Admin)"""
        async for result in self._paper_reparse_zero_abstract(event, confirm=confirm):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("reparseref")
    async def cmd_reparseref(self, event: AstrMessageEvent, file_name: str = ''):
        """Re-parse references for a single paper without full index rebuild (Admin)

        Extracts raw text from PDF via PyMuPDF and re-runs LLM reference parsing.
        Results saved to data/paper_doc_stats.json. Use when LLM timeout causes
        reference parsing to fail for a specific paper.
        """
        async for result in self._paper_reparseref(event, file_name=file_name):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("repair_refs")
    async def cmd_repair_refs(self, event: AstrMessageEvent, confirm: str = ''):
        """Auto-classify and repair all papers with unlinked references (Admin)

        Splits papers into two strategies automatically:
        - Full reparse: papers with empty-title refs (LLM extraction failed)
        - Link-only repair: papers where all unlinked refs have valid titles
        """
        async for result in self._paper_repair_refs(event, confirm=confirm):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("rebuild")
    async def cmd_rebuild(self, event: AstrMessageEvent, directory: str = '', confirm: str = ''):
        """Clear and rebuild document knowledge base (Admin)"""
        async for result in self._paper_rebuild(event, directory=directory, confirm=confirm):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("rebuildf")
    async def cmd_rebuild_file(self, event: AstrMessageEvent, file_name: str = ''):
        """Rebuild a single paper in knowledge base (Admin)"""
        async for result in self._paper_rebuildf(event, file_name=file_name):
            yield result

    # ==================== ArXiv 命令（属于 paper 组） ====================
    @paper_commands.command("arxiv_list")
    async def cmd_arxiv_list(self, event: AstrMessageEvent):
        """List all papers with arxiv URLs in markdown format"""
        async for result in self._paper_arxiv_list(event):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("arxiv_add")
    async def cmd_arxiv_add(self, event: AstrMessageEvent, query: str = '', max_results: int = 5):
        """Search CORE API and download papers, then add to database (Admin)"""
        async for result in self._paper_arxiv_add(event, query=query, max_results=max_results):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("arxiv_refs")
    async def cmd_arxiv_refs(self, event: AstrMessageEvent, top_k: int = 10, max_per_paper: int = 3):
        """Download highly-cited reference papers via CORE API and add to database (Admin)"""
        async for result in self._paper_arxiv_refs(event, top_k=top_k, max_per_paper=max_per_paper):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("arxiv_sync")
    async def cmd_arxiv_sync(self, event: AstrMessageEvent, confirm: str = ''):
        """Sync arxiv MCP downloaded papers to paperrag database (Admin)"""
        async for result in self._paper_arxiv_sync(event, confirm=confirm):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("arxiv_cleanup")
    async def cmd_arxiv_cleanup(self, event: AstrMessageEvent, confirm: str = ''):
        """Clean up old versions of arxiv papers, keeping only latest versions (Admin)"""
        async for result in self._paper_arxiv_cleanup(event, confirm=confirm):
            yield result

    # ==================== Graph 命令（属于 paper 组） ====================
    @paper_commands.command("graph_build")
    async def cmd_graph_build(self, event: AstrMessageEvent, confirm: str = '', skip: str = ''):
        """Build knowledge graph from indexed documents"""
        async for result in self._paper_graph_build(event, confirm=confirm, skip=skip):
            yield result

    @paper_commands.command("graph_stats")
    async def cmd_graph_stats(self, event: AstrMessageEvent):
        """Show knowledge graph statistics"""
        async for result in self._paper_graph_stats(event):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_rebuild")
    async def cmd_graph_rebuild(self, event: AstrMessageEvent, confirm: str = ''):
        """Rebuild knowledge graph from scratch (clear + rebuild)"""
        async for result in self._paper_graph_rebuild(event, confirm=confirm):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_clear")
    async def cmd_graph_clear(self, event: AstrMessageEvent, confirm: str = ''):
        """Clear knowledge graph (Admin)"""
        async for result in self._paper_graph_clear(event, confirm=confirm):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_backup")
    async def cmd_graph_backup(self, event: AstrMessageEvent, mode: str = 'online'):
        """Backup knowledge graph to JSON (Admin)"""
        async for result in self._paper_graph_backup(event, mode=mode):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @paper_commands.command("graph_restore")
    async def cmd_graph_restore(self, event: AstrMessageEvent, backup_file: str = ''):
        """Restore knowledge graph from backup (Admin)"""
        async for result in self._paper_graph_restore(event, backup_file=backup_file):
            yield result

    @paper_commands.command("graph_backup_list")
    async def cmd_graph_backup_list(self, event: AstrMessageEvent):
        """List available graph backups"""
        async for result in self._paper_graph_backup_list(event):
            yield result

    @paper_commands.command("graph_link")
    async def cmd_graph_link(self, event: AstrMessageEvent, action: str = 'status'):
        """Query knowledge graph for entity relationships"""
        async for result in self._paper_graph_link(event, action=action):
            yield result

    # ==================== Idea 命令组 ====================
    @filter.command_group("idea")
    def idea_commands(self):
        """研究创意生成命令组
        gen                  - 生成研究想法并保存（第一阶段）
        list                 - 列出所有已保存的 topic
        show                 - 显示单个 topic 下所有想法
        add                  - 为已有 topic 追加新想法
        del                  - 删除指定 UUID 的想法
        clear                - 清空指定 topic 下所有想法（保留 folder）
        delete               - 完全删除 topic（包括 folder）
        tofeishu             - 将想法创建为飞书文档（第二阶段）
        explore              - 探索研究想法（完整流程）
        analyze              - 分析研究主题
        search               - 多源知识检索
        generate             - 基于知识上下文生成想法
        """
        pass

    @idea_commands.command("gen")
    async def cmd_idea_gen(self, event: AstrMessageEvent, topic: str = ""):
        """生成研究想法并保存到文件（第一阶段）"""
        async for result in self._idea_gen(event, topic=topic):
            yield result

    @idea_commands.command("list")
    async def cmd_idea_list(self, event: AstrMessageEvent):
        """列出所有已保存的 topic 及其想法数量"""
        async for result in self._idea_list(event):
            yield result

    @idea_commands.command("show")
    async def cmd_idea_show(self, event: AstrMessageEvent, identifier: str = ""):
        """显示单个 topic 下的所有想法"""
        async for result in self._idea_show(event, identifier=identifier):
            yield result

    @idea_commands.command("add")
    async def cmd_idea_add(self, event: AstrMessageEvent, topic: str = "", num_ideas: int = 3):
        """为已有 topic 追加新想法"""
        async for result in self._idea_add(event, topic=topic, num_ideas=num_ideas):
            yield result

    @idea_commands.command("del")
    async def cmd_idea_del(self, event: AstrMessageEvent, ids: str = ""):
        """删除指定 UUID 的想法"""
        async for result in self._idea_del(event, ids=ids):
            yield result

    @idea_commands.command("delete")
    async def cmd_idea_delete(self, event: AstrMessageEvent, topic_or_hash: str = ""):
        """完全删除 topic（包括 folder）"""
        async for result in self._idea_delete(event, topic_or_hash=topic_or_hash):
            yield result

    @idea_commands.command("clear")
    async def cmd_idea_clear(self, event: AstrMessageEvent, topic: str = ""):
        """清空指定 topic 下所有想法（保留 folder）"""
        async for result in self._idea_clear(event, topic=topic):
            yield result

    @idea_commands.command("clean")
    async def cmd_idea_clean(self, event: AstrMessageEvent, action: str = "", confirm: str = ""):
        """扫描并清理孤立/空的 topic 文件夹"""
        async for result in self._idea_clean(event, action=action, confirm=confirm):
            yield result

    @idea_commands.command("explore")
    async def cmd_idea_explore(self, event: AstrMessageEvent, topic: str = "", depth: str = "standard", num_ideas: int = 3):
        """探索研究想法（完整流程）"""
        async for result in self._idea_explore(event, topic=topic, depth=depth, num_ideas=num_ideas):
            yield result

    @idea_commands.command("analyze")
    async def cmd_idea_analyze(self, event: AstrMessageEvent, topic: str = "", depth: str = "standard"):
        """分析研究主题"""
        async for result in self._idea_analyze(event, topic=topic, depth=depth):
            yield result

    @idea_commands.command("search")
    async def cmd_idea_search(self, event: AstrMessageEvent, queries: str = "", local_k: int = 5, web_k: int = 10):
        """多源知识检索"""
        async for result in self._idea_search(event, queries=queries, local_k=local_k, web_k=web_k):
            yield result

    @idea_commands.command("generate")
    async def cmd_idea_generate(self, event: AstrMessageEvent, context: str = "", domain: str = "", num: int = 3, focus: str = "all"):
        """基于知识上下文生成想法"""
        async for result in self._idea_generate(event, context=context, domain=domain, num=num, focus=focus):
            yield result

    @idea_commands.command("tofeishu")
    async def cmd_idea_tofeishu(self, event: AstrMessageEvent, ids: str = "", folder_token: str = "", refresh: str = "auto"):
        """将想法创建为飞书文档（第二阶段）"""
        async for result in self._idea_tofeishu(event, ids=ids, folder_token=folder_token, refresh=refresh):
            yield result

    @idea_commands.command("testblocks")
    async def cmd_idea_testblocks(self, event: AstrMessageEvent, folder_token: str = ""):
        """Test idea generation blocks"""
        async for result in self._idea_testblocks(event, folder_token=folder_token):
            yield result

    @idea_commands.command("regen")
    async def cmd_idea_regen(self, event: AstrMessageEvent, folder_hash: str = "", refresh: str = "auto", num: int = 3, focus: str = "all"):
        """Regenerate specific idea block"""
        async for result in self._idea_regen(event, folder_hash=folder_hash, refresh=refresh, num=num, focus=focus):
            yield result

    # ==================== Remote CC 命令组 ====================
    @filter.command_group("cc")
    def cc_commands(self):
        """远程 Claude Code 管理命令
        status         - 显示远程连接状态和 Claude Code 版本
        connect        - 测试到远程服务器的 SSH 连接
        install        - 安装/更新远程 Claude Code + CC-Connect
        exec           - 在远程服务器上执行编程任务
        config         - 查看当前远程执行配置
        """
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @cc_commands.command("status")
    async def cmd_cc_status(self, event: AstrMessageEvent):
        """Show remote connection status and Claude Code version"""
        async for result in self._cc_status(event):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @cc_commands.command("connect")
    async def cmd_cc_connect(self, event: AstrMessageEvent):
        """Test SSH connection to remote server"""
        async for result in self._cc_connect(event):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @cc_commands.command("install")
    async def cmd_cc_install(self, event: AstrMessageEvent):
        """Install/update Claude Code + CC-Connect on remote"""
        async for result in self._cc_install(event):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @cc_commands.command("exec")
    async def cmd_cc_exec(self, event: AstrMessageEvent, task: str = "", timeout: int = 0):
        """Execute a programming task on the remote server"""
        async for result in self._cc_exec(event, task=task, timeout=timeout):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @cc_commands.command("config")
    async def cmd_cc_config(self, event: AstrMessageEvent):
        """Show current remote execution configuration"""
        async for result in self._cc_config(event):
            yield result
