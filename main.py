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

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import register

from .commands import ArxivCommandsMixin, GraphCommandsMixin, IdeaCommandsMixin, PaperCommandsMixin
from .commands.base import PluginCoreBase


@register(
    "paper_rag",
    "HUSTcyf",
    "本地文档库RAG检索插件 (支持PDF/Word/TXT/HTML, Gemini + Milvus Lite)",
    "1.12.5",
    "https://github.com/HUSTcyf/astrbot_plugin_paperrag.git"
)
class PaperRAGPlugin(PaperCommandsMixin, ArxivCommandsMixin, GraphCommandsMixin, IdeaCommandsMixin, PluginCoreBase):
    """论文RAG检索插件"""

    # ==================== Paper 命令组 ====================
    @filter.command_group("paper")
    def paper_commands(self):
        """Paper RAG command group
        search         - Search documents and answer questions
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
        """
        pass

    @paper_commands.command("search")
    async def cmd_search(self, event: AstrMessageEvent, query: str = '', mode: str = "rag", top_k: int = 5):
        """Search document library and answer questions"""
        async for result in self._paper_search(event, query=query, mode=mode, top_k=top_k):
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
