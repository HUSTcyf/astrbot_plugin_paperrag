"""Shared plugin core used by command mixins and the plugin entry."""

from __future__ import annotations

import asyncio
import threading
import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from astrbot.api import logger
from astrbot.api.star import Context, Star

from ..plugin_common import SUPPORTED_DOC_EXTENSIONS, _is_hidden_file
from ..rag.rag_engine import RAGConfig, create_rag_engine

_PLUGIN_DIR = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    from ..graphrag.graph_rag_engine import GraphRAGConfig
    from ..rag.hybrid_rag import HybridRAGEngine


class PluginCoreBase(Star):
    """Runtime base class for PaperRAG mixins.

    This keeps shared state and helper methods out of `main.py`, while also
    giving editors a concrete definition for the attributes accessed from each
    command mixin via `self`.
    """

    enabled: bool
    config: Dict[str, Any]
    context: Context
    cache_enabled: bool
    cache_ttl: int
    cache_max_size: int
    _response_cache: Dict[str, Any]
    _papers_since_graph_build: int
    _engine: Optional["HybridRAGEngine"]
    _config_valid: bool
    _neo4j_thread: Any

    def __init__(self, context: Context, config: dict = {}):
        super().__init__(context)
        self.config = config or {}
        self.context = context

        # 插件配置
        self.enabled = self.config.get("enabled", True)

        # 缓存
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl_seconds", 3600)
        self.cache_max_size = self.config.get("cache_max_entries", 100)
        self._response_cache = {}

        # Graph RAG 自动构建追踪
        self._papers_since_graph_build = 0

        # RAG引擎（懒加载）
        self._engine = None
        self._config_valid = False

        # Graph RAG 引擎（懒加载，缓存复用）
        self._graph_engine = None

        # 后台服务线程追踪
        self._neo4j_thread = None

        # 并发安全锁
        self._engine_lock = threading.Lock()  # 保护 _get_engine 初始化
        self._auto_build_lock = asyncio.Lock()  # 保护 _papers_since_graph_build 计数
        self._response_cache_lock = threading.Lock()  # 保护 _response_cache（sync 方法访问）

        logger.info("📚 Document RAG Plugin initialized (支持PDF/Word/TXT/HTML)")

    def _configure_mps_memory(self) -> None:
        """Configure PyTorch MPS memory behavior before torch is imported.

        Apple Silicon uses unified memory. Setting the high watermark to 0.0
        disables PyTorch's private upper limit and lets macOS memory pressure /
        swap decide. This must happen before torch initializes MPS to be fully
        effective.
        """
        unsloth_config = self.config.get("unsloth", {}) if isinstance(self.config, dict) else {}
        device = str(unsloth_config.get("device", "mps")).lower()
        if device != "mps":
            return

        ratio = unsloth_config.get("mps_high_watermark_ratio", "0.0")
        if ratio is None:
            return

        ratio_text = str(ratio).strip()
        if not ratio_text:
            return

        existing = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        if existing and existing != ratio_text:
            logger.info(
                "[MPS] 保留已有 PYTORCH_MPS_HIGH_WATERMARK_RATIO="
                f"{existing}，插件配置值 {ratio_text} 未覆盖"
            )
            return

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", ratio_text)
        logger.info(
            f"[MPS] PYTORCH_MPS_HIGH_WATERMARK_RATIO={os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')} "
            "(需在 torch/MPS 初始化前设置；修改后建议重启 AstrBot)"
        )

    def _scan_documents(self, directory: str) -> List[Any]:
        """扫描目录中的支持文档文件"""
        papers_dir = directory or self.config.get("papers_dir", "./papers")
        doc_files: List[Any] = []
        for ext in SUPPORTED_DOC_EXTENSIONS:
            doc_files.extend(Path(papers_dir).glob(f"*{ext}"))
        for ext in SUPPORTED_DOC_EXTENSIONS:
            doc_files.extend(Path(papers_dir).glob(f"*{ext.upper()}"))
        return [f for f in doc_files if not _is_hidden_file(f)]

    def _create_graph_rag_config(self) -> "GraphRAGConfig":
        """创建 GraphRAGConfig（从配置中读取所有设置）"""
        from ..graphrag.graph_rag_engine import GraphRAGConfig

        graph_rag_config = self.config.get("graph_rag", {})
        multimodal_config = graph_rag_config.get("multimodal_extraction", {})
        return GraphRAGConfig(
            enable_graph_rag=True,
            storage_type=graph_rag_config.get("storage_type", "neo4j"),
            neo4j_uri=graph_rag_config.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=graph_rag_config.get("neo4j_user", "neo4j"),
            neo4j_password=graph_rag_config.get("neo4j_password", ""),
            max_triplets_per_chunk=graph_rag_config.get("max_triplets_per_chunk", 5),
            graph_retrieval_top_k=graph_rag_config.get("graph_retrieval_top_k", 5),
            graph_rrf_weight=graph_rag_config.get("graph_rrf_weight", 0.2),
            auto_build=graph_rag_config.get("auto_build", False),
            auto_build_threshold=graph_rag_config.get("auto_build_threshold", 10),
            multimodal_enabled=multimodal_config.get("enabled", True),
            max_images_per_chunk=multimodal_config.get("max_images_per_chunk", 1),
            extract_image_entities=multimodal_config.get("extract_image_entities", True),
        )

    async def _check_academic_intent(self, query: str) -> tuple[bool, str]:
        """用本地VLM判断用户问题是否与学术论文相关"""
        prompt = f"""你是一个严格的学术问题分类器。只判断用户问题是否明确涉及学术论文的阅读、检索、分析。

以下情况回答"否"：
- 日常对话、闲聊、新闻、娱乐
- 常识问题（如"今天吃什么"、"天气怎么样"）
- 与论文无关的技术问题
- 任何不涉及论文内容的问题

以下情况回答"是"：
- 询问某篇论文的方法、创新点、实验结果
- 要求检索论文、总结论文内容
- 关于论文中的公式、原理、数据集的技术问题
- 要求对比多篇论文的异同

问题: {query}

只需回答"是"或"否"，不要解释。"""
        try:
            from ..idea.llama_cpp_vlm_provider import get_cached_llama_cpp_provider

            vlm_provider = get_cached_llama_cpp_provider()
            if vlm_provider and vlm_provider._initialized and vlm_provider._llama:
                response = await vlm_provider.text_chat(prompt=prompt, image_urls=[], temperature=0.0)
                text = response.content.strip().lower() if hasattr(response, "content") else str(response).lower()
                logger.info(f"[_check_academic_intent] VLM原始回答: '{text}'")
                if text.startswith("否"):
                    return False, "非学术问题"
                return True, "学术问题"

            logger.info("[_check_academic_intent] VLM未就绪，触发初始化...")
            from ..idea.llama_cpp_vlm_provider import init_llama_cpp_vlm_provider

            model_dir = _PLUGIN_DIR / "models" / "Qwen3.5-9B-GGUF"
            model_path = model_dir / "Qwen3.5-9B-UD-Q4_K_XL.gguf"
            mmproj_path = model_dir / "mmproj-BF16.gguf"
            vlm_provider = init_llama_cpp_vlm_provider(
                model_path=str(model_path),
                mmproj_path=str(mmproj_path),
                n_ctx=self.config.get("llama_vlm_n_ctx", 16384),
                n_gpu_layers=99,
                max_tokens=25600,
                temperature=0.0,
            )
            await vlm_provider.initialize()
            if vlm_provider and vlm_provider._initialized:
                response = await vlm_provider.text_chat(prompt=prompt, image_urls=[], temperature=0.0)
                text = response.content.strip().lower() if hasattr(response, "content") else str(response).lower()
                logger.info(f"[_check_academic_intent] VLM原始回答: '{text}'")
                if text.startswith("否"):
                    return False, "非学术问题"
                return True, "学术问题"
            return True, "VLM不可用，默认进行检索"
        except Exception as e:
            logger.warning(f"[_check_academic_intent] 意图判断失败: {e}")
            return True, f"意图判断失败，默认检索: {e}"

    async def _llm_direct_answer(self, query: str) -> str:
        """由LLM直接回答问题（不经过RAG）"""
        try:
            text_provider_id = self.config.get("text_provider_id", "")

            if text_provider_id:
                provider_manager = getattr(self.context, "provider_manager", None)
                if not provider_manager:
                    logger.error("[_llm_direct_answer] provider_manager 不可用，无法获取云端 LLM")
                else:
                    llm_provider = provider_manager.get_provider(text_provider_id)
                    if not llm_provider:
                        llm_provider = provider_manager.get_provider(None)
                    if not llm_provider:
                        logger.error(f"[_llm_direct_answer] 无法获取 LLM Provider (id={text_provider_id})")
                    else:
                        response = await llm_provider.generate(query)
                        return response.text.strip() if hasattr(response, "text") else str(response)
            else:
                from ..idea.llama_cpp_vlm_provider import get_cached_llama_cpp_provider
                vlm_provider = get_cached_llama_cpp_provider()
                if not vlm_provider:
                    logger.error("[_llm_direct_answer] 本地 VLM Provider 不可用")
                else:
                    response = await vlm_provider.text_chat(prompt=query, temperature=0.7)
                    if hasattr(response, 'content'):
                        return response.content.strip()
                    else:
                        logger.warning(f"[_llm_direct_answer] VLM 响应格式无法识别: {type(response)}")

        except Exception as e:
            logger.warning(f"[_llm_direct_answer] 回答失败: {e}")
        return ""

    def _get_engine(self) -> Optional["HybridRAGEngine"]:
        """获取RAG引擎（单例模式，带缓存）"""
        if self._engine is None and not self._config_valid:
            with self._engine_lock:
                # 双重检查
                if self._engine is None:
                    return self._create_engine_inner()
        return self._engine

    def _create_engine_inner(self) -> Optional["HybridRAGEngine"]:
        """创建 RAG 引擎（仅在持有 _engine_lock 时调用）"""
        try:
            self._configure_mps_memory()

            raw_embedding_mode = self.config.get("embedding_mode", "unsloth")
            if raw_embedding_mode == "api":
                embedding_mode = "astrbot"
            elif raw_embedding_mode == "ollama":
                embedding_mode = "unsloth"
            else:
                embedding_mode = raw_embedding_mode

            rag_config = RAGConfig(
                embedding_mode=embedding_mode,
                embedding_provider_id=self.config.get("embedding_provider_id", ""),
                compress_provider_id=self.config.get("compress_provider_id", ""),
                text_provider_id=self.config.get("text_provider_id", ""),
                multimodal_provider_id=self.config.get("multimodal_provider_id", ""),
                unsloth_config=self.config.get("unsloth", {}),
                llama_vlm_model_path=self.config.get("llama_vlm_model_path", "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"),
                llama_vlm_mmproj_path=self.config.get("llama_vlm_mmproj_path", "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"),
                llama_vlm_max_tokens=self.config.get("llama_vlm_max_tokens", 25600),
                llama_vlm_temperature=self.config.get("llama_vlm_temperature", 0.7),
                llama_vlm_n_ctx=self.config.get("llama_vlm_n_ctx", 16384),
                llama_vlm_n_gpu_layers=self.config.get("llama_vlm_n_gpu_layers", 99),
                milvus_lite_path=self.config.get("milvus_lite_path", ""),
                address=self.config.get("address", ""),
                db_name=self.config.get("db_name", "default"),
                authentication=self.config.get("authentication", {}),
                collection_name=self.config.get("collection_name", "paper_embeddings"),
                embed_dim=self.config.get("embed_dim", 768),
                top_k=self.config.get("top_k", 5),
                similarity_cutoff=self.config.get("similarity_cutoff", 0.3),
                papers_dir=self.config.get("papers_dir", "./papers"),
                chunk_size=self.config.get("chunk_size", 512),
                chunk_overlap=self.config.get("chunk_overlap", 0),
                min_chunk_size=self.config.get("min_chunk_size", 100),
                use_semantic_chunking=self.config.get("use_semantic_chunking", True),
                enable_multimodal=self.config.get("multimodal", {}).get("enabled", True),
                figures_dir=self.config.get("figures_dir", ""),
                enable_sparse_retrieval=self.config.get("enable_sparse_retrieval", True),
                enable_multi_vector_rerank=self.config.get("enable_multi_vector_rerank", False),
                sparse_top_k=self.config.get("sparse_top_k", 20),
                hybrid_alpha=self.config.get("hybrid_alpha", 0.5),
                hybrid_rrf_k=self.config.get("hybrid_rrf_k", 60),
                enable_bm25=self.config.get("enable_bm25", True),
                bm25_top_k=self.config.get("bm25_top_k", 20),
                enable_two_stage_retrieval=bool(self.config.get("enable_two_stage_retrieval", False)),
                two_stage_top_k=self.config.get("two_stage_top_k") or 10,
                two_stage_rerank_k=self.config.get("two_stage_rerank_k") or 5,
                enable_crag_quality_eval=self.config.get("enable_crag_quality_eval", True),
                crag_enable_correction=self.config.get("crag_enable_correction", False),
                crag_min_score=self.config.get("crag_min_score", 0.3),
                enable_llm_reference_parsing=self.config.get("enable_llm_reference_parsing", True),
                freeapi_url=self.config.get("freeapi_url", ""),
                freeapi_key=self.config.get("freeapi_key", ""),
                core_api_key=self.config.get("core_api_key", ""),
                use_arxiv_api=self.config.get("use_arxiv_api", True),
                enable_graph_rag=self.config.get("enable_graph_rag", False),
                graph_storage_type=self.config.get("graph_rag", {}).get("storage_type", "neo4j"),
                graph_neo4j_uri=self.config.get("graph_rag", {}).get("neo4j_uri", "bolt://localhost:7687"),
                graph_neo4j_user=self.config.get("graph_rag", {}).get("neo4j_user", "neo4j"),
                graph_neo4j_password=self.config.get("graph_rag", {}).get("neo4j_password", ""),
                graph_max_triplets_per_chunk=self.config.get("graph_rag", {}).get("max_triplets_per_chunk", 5),
                graph_retrieval_top_k=self.config.get("graph_rag", {}).get("graph_retrieval_top_k", 5),
                graph_rrf_weight=self.config.get("graph_rag", {}).get("graph_rrf_weight", 0.2),
                graph_auto_build=self.config.get("graph_rag", {}).get("auto_build", False),
                graph_auto_build_threshold=self.config.get("graph_rag", {}).get("auto_build_threshold", 10),
            )

            valid, error_msg = rag_config.validate()
            if not valid:
                logger.error(f"❌ RAG配置无效: {error_msg}")
                self._config_valid = False
                return None

            self._engine = create_rag_engine(rag_config, self.context)
            self._config_valid = True
            return self._engine
        except Exception as e:
            logger.error(f"❌ RAG引擎初始化失败: {e}")
            self._config_valid = False
            return None

    async def _get_graph_engine(self):
        """获取 GraphRAGEngine（单例模式，带缓存）"""
        if self._graph_engine is not None and self._graph_engine._initialized:
            return self._graph_engine

        base_engine = self._get_engine()
        if not base_engine:
            logger.warning("[PaperRAG] 基础引擎未就绪，无法创建 Graph RAG 引擎")
            return None

        try:
            from ..graphrag.graph_rag_engine import GraphRAGEngine
            graph_config = self._create_graph_rag_config()
            engine_instance = GraphRAGEngine(graph_config, base_engine, self.context)
            await engine_instance.initialize()
            if not engine_instance._initialized:
                logger.warning("[PaperRAG] Graph RAG 引擎初始化未完成")
                return None
            self._graph_engine = engine_instance
            return self._graph_engine
        except Exception as e:
            logger.error(f"[PaperRAG] Graph RAG 引擎创建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._graph_engine = None
            return None

    def _get_cache_key(self, query: str, mode: str, top_k: int) -> str:
        return f"{query}|{mode}|{top_k}"

    def _get_cached_response(self, cache_key: str):
        if not self.cache_enabled:
            return None

        import time

        with self._response_cache_lock:
            if cache_key in self._response_cache:
                cached_data, timestamp = self._response_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"📦 使用缓存: {cache_key[:50]}...")
                    return cached_data
                del self._response_cache[cache_key]

        return None

    def _set_cached_response(self, cache_key: str, response):
        if not self.cache_enabled:
            return

        import time

        with self._response_cache_lock:
            if len(self._response_cache) >= self.cache_max_size:
                oldest_key = min(self._response_cache.keys(), key=lambda k: self._response_cache[k][1])
                del self._response_cache[oldest_key]

            self._response_cache[cache_key] = (response, time.time())

    async def _maybe_trigger_graph_auto_build(self, papers_added: int = 1) -> bool:
        if not self.config.get("enable_graph_rag", False):
            return False

        auto_build = self.config.get("graph_rag", {}).get("auto_build", False)
        if not auto_build:
            return False

        async with self._auto_build_lock:
            self._papers_since_graph_build += papers_added
            threshold = self.config.get("graph_rag", {}).get("auto_build_threshold", 10)

            if self._papers_since_graph_build >= threshold:
                logger.info(f"📚 自动构建知识图谱（已添加 {self._papers_since_graph_build} 篇论文）")
                self._papers_since_graph_build = 0

                try:
                    engine = self._get_engine()
                    if not engine:
                        return False

                    asyncio.create_task(self._run_graph_build_in_background(engine))
                    return True
                except Exception as e:
                    logger.error(f"自动构建知识图谱失败: {e}")
                    return False

            return False

    async def _run_graph_build_in_background(self, engine):
        """子类覆盖此方法执行后台图谱构建。基类提供空实现防止未覆盖时崩溃。"""
        pass

    async def terminate(self):
        logger.info("📚 Document RAG Plugin is unloading...")

        if self.config.get("enable_llm_tools", True):
            try:
                self.context.unregister_llm_tool("search_papers")
                self.context.unregister_llm_tool("retrieve_papers")
                logger.info("✅ Paper RAG LLM工具已注销")
            except Exception as e:
                logger.warning(f"注销LLM工具时出现警告: {e}")

        with self._response_cache_lock:
            self._response_cache.clear()

        try:
            from ..idea.llama_cpp_vlm_provider import reset_llama_cpp_vlm_provider

            reset_llama_cpp_vlm_provider()
            logger.info("[Llama.cpp-VLM] Provider 已清理")

            gc.collect()
            logger.info("[Llama.cpp-VLM] 垃圾回收完成")
        except Exception as e:
            logger.warning(f"[Llama.cpp-VLM] 清理时出现警告: {e}")

        await super().terminate()
