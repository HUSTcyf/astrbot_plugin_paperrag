# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Ragas 自动化评估主入口
从 Milvus 数据库读取 chunks → 生成测试集 → RAG 评估 → 报告生成

流程:
  1. 从 Milvus 提取全量 chunk 文本
  2. 按论文分组，构建 llama-index Document
  3. 调用 Ragas TestsetGenerator 生成问答对
  4. 使用 HybridRAGEngine 执行 RAG 查询
  5. 调用 Ragas Evaluator 计算 6 大指标
  6. 生成 HTML + Markdown 报告

用法:
  # 完整流程（提取文本 -> 生成测试集 -> 评估 -> 报告）
  python -m evaluation.run_evaluation_ragas --step all

  # 从 Milvus 提取全量文本（调试用）
  python -m evaluation.run_evaluation_ragas --step extract

  # 仅生成测试集（需已有 milvus_chunks.json）
  python -m evaluation.run_evaluation_ragas --step generate

  # 完整流程使用已有 chunks 文件（避免重复从数据库读取）
  python -m evaluation.run_evaluation_ragas --step all --use-existing-chunks

环境变量:
  EVAL_LLM_API_KEY 评估用 LLM API Key
"""

import asyncio
import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

# 确保 evaluation 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.hybrid_index import HybridIndexManager
import csv
import io
from collections import defaultdict
from llama_index.core import Document as LIDocument
from .ragas_generator import RagasTestsetGenerator
from .minimax_compat import resolve_embedding_model
import tempfile
import uuid
import re
from .ragas_evaluator import RagasEvaluator, load_raw_answers
from .report_generator import ReportGenerator
from rag.rag_engine import create_rag_engine, RAGConfig
from .ragas_generator import OpenAICompatibleLLM


# 多模态文档生成时，发送给 LLM 的文档数量倍数（test_size * MULTIMODAL_DOC_MULTIPLIER）
MULTIMODAL_DOC_MULTIPLIER = 2

# 默认 LLM URL（CLI 参数和配置解析共用）
_DEFAULT_LLM_URL = "https://open.bigmodel.cn/api/paas/v4"


def _atomic_write_json(path: str | Path, data: Any) -> None:
    """原子写入 JSON（写 .tmp 再 os.replace），防止中途崩溃导致文件损坏。"""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass
class _EvalLLMConfig:
    """评估脚本的 LLM 配置容器，消除参数爆炸。"""
    # 测试集生成用
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    llm_api_key: str = ""
    # 评估指标计算用
    eval_llm_model: str = "gpt-5.4-nano"
    eval_llm_base_url: str = ""
    eval_llm_max_tokens: int = 16384
    # Embedding 用
    embedding_model: str = "text-embedding-v4"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embedding_mode: str = "api"
    eval_embedding_mode: str = "api"
    max_rpm: int = 30


def _resolve_eval_config(args) -> tuple[_EvalLLMConfig, dict]:
    """从 CLI 参数、环境变量、插件配置、free.json 解析 LLM 配置。

    优先级：CLI 显式参数 > 环境变量 > 插件配置 freeapi > free.json vapi
    """
    llm = _EvalLLMConfig()

    # Step 1: 从 CLI 参数读取
    llm.llm_model = args.llm_model
    llm.llm_api_key = args.llm_api_key or args.eval_llm_api_key or os.getenv("EVAL_LLM_API_KEY", "")
    llm.embed_api_key = args.embed_api_key or os.getenv("EVAL_EMBED_API_KEY", "")
    llm.eval_llm_model = args.eval_llm_model
    llm.eval_llm_max_tokens = args.eval_llm_max_tokens
    llm.embedding_model = args.embedding_model
    llm.embedding_mode = args.embedding_mode
    llm.eval_embedding_mode = args.eval_embedding_mode
    llm.max_rpm = args.max_rpm

    # Step 2: 从插件配置读取 freeapi
    plugin_config_path = Path(__file__).parent.parent.parent.parent / "config" / "astrbot_plugin_paperrag_config.json"
    plugin_config = {}
    if plugin_config_path.exists():
        with open(plugin_config_path, "r", encoding="utf-8-sig") as f:
            plugin_config = json.load(f)
        config_freeapi_key = plugin_config.get("freeapi_key", "")
        config_freeapi_url = plugin_config.get("freeapi_url", "")
        if config_freeapi_key and not llm.llm_api_key:
            llm.llm_api_key = config_freeapi_key
        if config_freeapi_key and not llm.embed_api_key:
            llm.embed_api_key = config_freeapi_key
        if config_freeapi_url:
            llm.llm_base_url = config_freeapi_url + "/v1/"
            llm.embed_base_url = config_freeapi_url + "/v1/"
    else:
        llm.llm_base_url = args.llm_base_url
        llm.embed_base_url = args.embed_base_url

    # Step 2.5: CLI 显式 URL 覆盖插件配置
    if args.llm_base_url and args.llm_base_url != _DEFAULT_LLM_URL:
        llm.llm_base_url = args.llm_base_url
    if args.embed_base_url:
        llm.embed_base_url = args.embed_base_url

    # Step 2.6: 从主 AstrBot cmd_config.json 自动填充 LLM 凭据（OpenAI 兼容端点）
    # 触发条件：CLI/环境变量/freeapi 均未提供 key。此时连带覆盖 base_url，
    # 避免"真实 key + 无 key 免费端点"的组合错误。
    # Step 2.55: --eval-provider 显式选择评测端点（覆盖插件 text_provider_id）
    _eval_provider = getattr(args, "eval_provider", "auto")
    if _eval_provider != "auto":
        main_config_path = Path(__file__).parent.parent.parent.parent / "cmd_config.json"
        if main_config_path.exists():
            try:
                with open(main_config_path, "r", encoding="utf-8-sig") as f:
                    _main_cfg = json.load(f)
                # 按 --eval-provider 关键字匹配 cmd_config 里的 provider id
                _match = next(
                    (p["id"] for p in _main_cfg.get("provider", [])
                     if _eval_provider in p.get("id", "").lower()),
                    None,
                )
                if _match:
                    plugin_config = dict(plugin_config)
                    plugin_config["text_provider_id"] = _match
                    print(f"✅ --eval-provider={_eval_provider}：使用 provider {_match}")
            except Exception as e:
                print(f"⚠️ --eval-provider 解析失败: {e}")
    if not llm.llm_api_key:
        main_config_path = Path(__file__).parent.parent.parent.parent / "cmd_config.json"
        if main_config_path.exists():
            try:
                with open(main_config_path, "r", encoding="utf-8-sig") as f:
                    main_cfg = json.load(f)
                provider_config = None
                text_provider_id = plugin_config.get("text_provider_id", "")
                for p in main_cfg.get("provider", []):
                    if p.get("id") == text_provider_id:
                        provider_config = dict(p)
                        break
                if provider_config:
                    src_id = provider_config.get("provider_source_id", "")
                    if src_id:
                        for ps in main_cfg.get("provider_sources", []):
                            if ps.get("id") == src_id:
                                provider_config = {**ps, **provider_config}
                                break
                    raw_key = provider_config.get("key") or provider_config.get("api_key") or ""
                    if isinstance(raw_key, list):  # AstrBot provider_sources 的 key 可能是 list
                        raw_key = raw_key[0] if raw_key else ""
                    key = str(raw_key)
                    api_base = str(provider_config.get("api_base", "") or "")
                    # 转换为 OpenAI 兼容 base_url（最终拼接 /chat/completions 调用）
                    if api_base.endswith("/anthropic"):
                        # MiniMax 等 Anthropic 协议端点 → OpenAI 兼容
                        oai_base = api_base.replace("/anthropic", "/v1")
                    elif "/v4" in api_base or "/v1" in api_base:
                        # 智谱（/paas/v4/）等已是完整 OpenAI 兼容 base，不再追加 /v1
                        oai_base = api_base.rstrip("/")
                    else:
                        oai_base = api_base.rstrip("/") + "/v1"
                    if key and oai_base:
                        model = provider_config.get("model", "")
                        print(f"✅ 已从 AstrBot cmd_config.json 自动填充 LLM 凭据: "
                              f"{text_provider_id} ({model}) @ {oai_base}")
                        llm.llm_api_key = key
                        if not llm.embed_api_key:
                            llm.embed_api_key = key
                        cli_url_explicit = bool(args.llm_base_url) and args.llm_base_url != _DEFAULT_LLM_URL
                        if not cli_url_explicit:
                            llm.llm_base_url = oai_base
                        if not args.embed_base_url:
                            llm.embed_base_url = oai_base
                        # MiniMax 端点的 embedding 模型为 embo-01（非 OpenAI 命名）
                        resolved = resolve_embedding_model(oai_base, llm.embedding_model)
                        if resolved != llm.embedding_model:
                            llm.embedding_model = resolved
                            print(f"✅ MiniMax 端点，embedding 模型切换为 {llm.embedding_model}")
                        if args.llm_model == "gpt-4o-mini":
                            llm.llm_model = model or llm.llm_model
                        if args.eval_llm_model == "gpt-5.4-nano":
                            llm.eval_llm_model = model or llm.eval_llm_model
            except Exception as e:
                print(f"⚠️ 从 cmd_config.json 填充 LLM 凭据失败: {e}")

    # Step 3: 从 free.json 读取 vapi（用于评估指标计算）
    free_json_path = Path(__file__).parent.parent.parent.parent / "free.json"
    if free_json_path.exists():
        try:
            with open(free_json_path, "r", encoding="utf-8-sig") as f:
                free_cfg = json.load(f)
            vapi_cfg = free_cfg.get("vapi", {})
            if isinstance(vapi_cfg, dict):
                vapi_url = vapi_cfg.get("url", "")
                vapi_key = vapi_cfg.get("key", "")
                if vapi_url:
                    print(f"✅ 已从 free.json 加载 vapi: {vapi_url}")
                llm.eval_llm_base_url = args.eval_llm_base_url or (vapi_url + "/v1/" if vapi_url else "") or llm.llm_base_url
                if vapi_key and not llm.llm_api_key:
                    llm.llm_api_key = vapi_key
                    print("✅ 使用 vapi key 作为 API Key")
        except Exception as e:
            print(f"⚠️ 读取 free.json 失败: {e}")
            llm.eval_llm_base_url = args.eval_llm_base_url or llm.llm_base_url
    else:
        llm.eval_llm_base_url = args.eval_llm_base_url or llm.llm_base_url

    # Step 4: 打印配置信息
    print(f"\n📊 配置信息:")
    print(f"   步骤: {args.step}")
    print(f"   LLM 模型: {llm.llm_model}")
    print(f"   LLM API URL: {llm.llm_base_url}")
    print(f"   评估 LLM 模型: {llm.eval_llm_model}")
    print(f"   评估 LLM URL: {llm.eval_llm_base_url}")
    print(f"   Embedding 模型: {llm.embedding_model}")
    print(f"   Embedding API URL: {llm.embed_base_url}")
    print(f"   Embedding 模式: {llm.embedding_mode}")
    if llm.embedding_mode == "unsloth":
        print(f"   Embedding 模式: Unsloth (本地 BGE-M3)")

    if not llm.llm_api_key:
        print("⚠️ 警告: 未提供 API Key（设置 EVAL_LLM_API_KEY 环境变量或使用 --llm-api-key）")

    return llm, plugin_config


# provider type → 模块路径映射（触发 @register_provider_adapter）
_PROVIDER_MODULES = {
    "openai_chat_completion": "astrbot.core.provider.sources.openai_source",
    "zhipu_chat_completion": "astrbot.core.provider.sources.zhipu_source",
    "openrouter_chat_completion": "astrbot.core.provider.sources.openrouter_source",
    "googlegenai_chat_completion": "astrbot.core.provider.sources.gemini_source",
    "minimax_token_plan": "astrbot.core.provider.sources.minimax_token_plan_source",
}


def _init_system_llm_provider(plugin_config: dict):
    """从主 AstrBot 配置初始化系统 LLM provider，返回带 inst_map 的轻量 provider manager。"""
    import importlib
    text_provider_id = plugin_config.get("text_provider_id", "")
    if not text_provider_id:
        print("⚠️ 插件配置中未设置 text_provider_id，无法使用系统 LLM 生成答案")
        return None

    main_config_path = Path(__file__).parent.parent.parent.parent / "cmd_config.json"
    if not main_config_path.exists():
        print(f"⚠️ 主配置文件不存在: {main_config_path}")
        return None

    with open(main_config_path, "r", encoding="utf-8-sig") as f:
        main_cfg = json.load(f)

    # 查找匹配的 provider 条目
    provider_config = None
    for p in main_cfg.get("provider", []):
        if p.get("id") == text_provider_id:
            if not p.get("enable", True):
                print(f"⚠️ Provider {text_provider_id} 未启用")
                return None
            provider_config = dict(p)
            break

    if not provider_config:
        print(f"⚠️ 未在主配置中找到 provider: {text_provider_id}")
        return None

    # 合并 provider_source 配置
    src_id = provider_config.get("provider_source_id", "")
    if src_id:
        for ps in main_cfg.get("provider_sources", []):
            if ps.get("id") == src_id:
                merged = {**ps, **provider_config}
                merged["id"] = provider_config["id"]
                provider_config = merged
                break

    provider_type = provider_config.get("type", "")
    if not provider_type:
        print(f"⚠️ Provider {text_provider_id} 缺少 type 字段")
        return None

    # 确保 provider type 对应的 source 模块已加载（触发 @register_provider_adapter）
    # 与 AstrBot ProviderManager.dynamic_import_provider() 逻辑一致
    try:
        from astrbot.core.provider.register import provider_cls_map
    except ImportError as e:
        print(f"⚠️ 无法导入 provider_cls_map: {e}")
        return None

    if provider_type not in provider_cls_map:
        module_path = _PROVIDER_MODULES.get(provider_type)
        if not module_path:
            print(f"⚠️ 不支持的 provider 类型: {provider_type}")
            return None
        importlib.import_module(module_path)

    meta = provider_cls_map.get(provider_type)
    if meta is None:
        print(f"⚠️ 未注册的 provider 类型: {provider_type}")
        return None

    provider_settings = main_cfg.get("provider_settings", {})
    try:
        provider_inst = meta.cls_type(provider_config, provider_settings)
    except Exception as e:
        print(f"⚠️ 实例化 provider {text_provider_id} 失败: {e}")
        return None

    model = provider_config.get("model", "?")
    api_base = provider_config.get("api_base", "?")
    print(f"✅ 系统 LLM provider 初始化成功: {text_provider_id} ({model}) @ {api_base}")

    # 返回带 inst_map 的轻量 provider manager
    class _LightweightProviderManager:
        def __init__(self, inst_map):
            self.inst_map = inst_map

    return _LightweightProviderManager({text_provider_id: provider_inst})


class _EvalFakeContext:
    """用于评估脚本的模拟上下文，支持注入 LLM provider 和 provider_manager。"""

    def __init__(self, llm_provider=None, provider_manager=None):
        self.provider_manager = provider_manager
        self._llm_provider = llm_provider

    def get_using_provider(self):
        return self._llm_provider


# ============================================================================
# 步骤 1: 从 Milvus 提取全量文本
# ============================================================================

def create_index_manager() -> Any:
    """创建 HybridIndexManager 实例（复用现有配置）"""
    plugin_dir = Path(__file__).parent.parent
    config_path = plugin_dir / "data" / "cmd_config.json"

    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    rag_config = config.get("rag_config", {})
    milvus_config = rag_config.get("milvus", {})


    return HybridIndexManager(
        collection_name=rag_config.get("collection_name", "paper_embeddings"),
        embed_dim=rag_config.get("embed_dim", 1024),
        milvus_uri=str(plugin_dir / "data" / "milvus_papers.db"),
        uri=milvus_config.get("address", ""),
        db_name=milvus_config.get("db_name", "default"),
        authentication=milvus_config.get("authentication"),
        hybrid_search=False,
    )


async def extract_chunks_from_milvus(
    output_path: str = "results/milvus_chunks.json",
) -> List[Dict[str, Any]]:
    """
    从 Milvus 提取全量 chunk 文本（按论文逐篇加载）

    Returns:
        [{"text": str, "metadata": dict, "paper_id": str}, ...]
    """
    print(f"\n{'='*60}")
    print("📤 步骤 1/4: 从 Milvus 提取全量文本（按论文逐篇加载）")
    print("=" * 60)

    index_manager = create_index_manager()
    chunks = await index_manager.get_all_chunks()

    # 保存到文件（原子写入）
    _atomic_write_json(output_path, chunks)

    print(f"✅ 提取完成: {len(chunks)} chunks -> {output_path}")

    # 按论文统计
    paper_counts: Dict[str, int] = {}
    for c in chunks:
        pid = c.get("paper_id", "unknown")
        paper_counts[pid] = paper_counts.get(pid, 0) + 1

    print(f"📊 论文数: {len(paper_counts)}")
    print(f"📊 Chunk 分布（前10）:")
    for pid, cnt in sorted(paper_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {pid}: {cnt} chunks")

    return chunks


# ============================================================================
# 步骤 2: 将 chunks 构建为 llama-index Document（支持多模态）
# ============================================================================

def _read_table_csv_as_text(csv_path: str, max_rows: int = 50) -> str:
    """
    读取 CSV 表格文件并转换为可读的文本格式

    Args:
        csv_path: CSV 文件路径
        max_rows: 最大读取行数（避免表格过大）

    Returns:
        表格的文本表示
    """

    try:
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            return ""

        content = csv_path_obj.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            return ""

        # 限制行数
        header = rows[0] if rows else []
        data_rows = rows[1:max_rows]
        remaining = len(rows) - 1 - max_rows

        # 构建文本表示
        lines = []
        lines.append("| " + " | ".join(str(c) for c in header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in data_rows:
            cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
            lines.append("| " + " | ".join(cells) + " |")

        result = "\n".join(lines)
        if remaining > 0:
            result += f"\n... (还有 {remaining} 行)"

        return result
    except Exception as e:
        return f"[Table: {csv_path} (read error)]"


def _get_multimodal_context(node_metadata: Dict[str, Any]) -> str:
    """
    从 chunk metadata 中提取图片/表格上下文信息，转换为文本描述

    Args:
        node_metadata: chunk 的 metadata

    Returns:
        包含图片/表格描述的文本
    """
    context_parts = []

    # 处理图片
    image_path = node_metadata.get("image_path", "")
    if image_path:
        image_caption = node_metadata.get("image_caption", "")
        if image_caption:
            context_parts.append(f"[IMAGE: {image_caption}]\nFile: {image_path}")
        else:
            context_parts.append(f"[IMAGE]\nFile: {image_path}")

    # 处理多图片
    all_images = node_metadata.get("all_images", [])
    for img_info in all_images:
        if isinstance(img_info, dict):
            img_path = img_info.get("path", "")
            img_caption = img_info.get("caption", "")
            if img_path and img_path != image_path:  # 避免重复
                if img_caption:
                    context_parts.append(f"[IMAGE: {img_caption}]\nFile: {img_path}")
                else:
                    context_parts.append(f"[IMAGE]\nFile: {img_path}")

    # 处理表格
    table_path = node_metadata.get("table_path", "")
    if table_path:
        table_caption = node_metadata.get("table_caption", "")
        table_text = _read_table_csv_as_text(table_path)
        if table_text:
            caption_prefix = f"[TABLE: {table_caption}]\n" if table_caption else "[TABLE]\n"
            context_parts.append(f"{caption_prefix}File: {table_path}\n{table_text}")
        else:
            context_parts.append(f"[TABLE: {table_caption}]\nFile: {table_path}")

    return "\n\n".join(context_parts) if context_parts else ""


def chunks_to_documents(
    chunks: List[Dict[str, Any]],
    min_chunks_per_paper: int = 5,
    include_multimodal: bool = True,
    figures_dir: str = "",
) -> List[Any]:
    """
    将 Milvus chunk 列表转换为 llama-index Document 列表

    Args:
        chunks: extract_chunks_from_milvus 返回的 chunk 列表
        min_chunks_per_paper: 最少 chunk 数才生成 Document（过滤太少内容的论文）
        include_multimodal: 是否在文本中包含图片/表格描述
        figures_dir: figures 目录路径（为空则使用插件默认路径）

    Returns:
        llama-index Document 列表
    """

    # 确定 figures 目录
    if not figures_dir:
        plugin_dir = Path(__file__).parent.parent
        figures_dir = str(plugin_dir / "data" / "figures")

    papers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        pid = chunk.get("paper_id", "unknown")
        if pid and pid != "unknown":
            papers[pid].append(chunk)

    documents = []

    for paper_id, paper_chunks in papers.items():
        if len(paper_chunks) < min_chunks_per_paper:
            continue

        # 按 chunk id 排序，保证顺序
        paper_chunks.sort(key=lambda x: x.get("id", 0))

        # 构建带多模态信息的文档
        combined_parts = []
        has_multimodal = False
        multimodal_count = 0
        all_image_paths = []
        all_table_paths = []

        for c in paper_chunks:
            text = c.get("text", "")
            if not text:
                continue

            # 如果启用多模态且 metadata 中有图片/表格信息，追加描述
            if include_multimodal:
                metadata = c.get("metadata", {})
                if isinstance(metadata, dict):
                    multimodal_ctx = _get_multimodal_context(metadata)
                    if multimodal_ctx:
                        has_multimodal = True
                        multimodal_count += 1
                        text = text + "\n\n" + multimodal_ctx
                    # 收集路径信息
                    if metadata.get("image_path"):
                        all_image_paths.append(metadata["image_path"])
                    if metadata.get("table_path"):
                        all_table_paths.append(metadata["table_path"])

            combined_parts.append(text)

        combined_text = "\n\n".join(combined_parts)

        if not combined_text.strip():
            continue

        doc = LIDocument(
            text=combined_text,
            metadata={
                "paper_id": paper_id,
                "chunk_count": len(paper_chunks),
                "source": "milvus",
                "has_multimodal": has_multimodal,
                "multimodal_count": multimodal_count,
                "image_paths": all_image_paths,
                "table_paths": all_table_paths,
            }
        )
        documents.append(doc)

    print(f"📄 构建 {len(documents)} 篇论文 Document")
    if include_multimodal:
        multimodal_docs = sum(1 for d in documents if d.metadata.get("has_multimodal"))
        print(f"   其中 {multimodal_docs} 篇包含图片/表格")
    return documents


# ============================================================================
# 步骤 2.5: 多模态测试集生成（仅生成与图表关联的问答对）
# ============================================================================

def extract_multimodal_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从 chunks 中提取与图片/表格关联的 chunks

    检查以下字段：
    - metadata.image_path: 直接的图片路径
    - metadata.table_path: 直接的表格路径
    - metadata.all_images: 图片列表
    - metadata.multimodal_data.images: 多模态数据中的图片
    - metadata.multimodal_data.tables: 多模态数据中的表格

    Args:
        chunks: 所有 chunks 列表

    Returns:
        包含图片/表格信息的 chunks 列表
    """
    multimodal_chunks = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        # 检查是否有图片或表格
        has_image = bool(metadata.get("image_path"))
        has_table = bool(metadata.get("table_path"))

        # 也检查 all_images 字段
        all_images = metadata.get("all_images", [])
        if not has_image and isinstance(all_images, list) and len(all_images) > 0:
            has_image = True

        # 检查 multimodal_data 中的 images 和 tables
        multimodal_data = metadata.get("multimodal_data", {})
        if isinstance(multimodal_data, dict):
            # multimodal_data.images 是一个列表
            mm_images = multimodal_data.get("images", [])
            if not has_image and isinstance(mm_images, list) and len(mm_images) > 0:
                has_image = True

            # multimodal_data.tables 是一个列表
            mm_tables = multimodal_data.get("tables", [])
            if not has_table and isinstance(mm_tables, list) and len(mm_tables) > 0:
                has_table = True

        if has_image or has_table:
            # 添加标记以便后续处理
            chunk = dict(chunk)  # 复制避免修改原数据
            chunk["_multimodal_type"] = []
            if has_image:
                chunk["_multimodal_type"].append("image")
            if has_table:
                chunk["_multimodal_type"].append("table")
            multimodal_chunks.append(chunk)

    print(f"🖼️ 找到 {len(multimodal_chunks)} 个与图片/表格关联的 chunks")
    return multimodal_chunks


def extract_multimodal_chunks_with_context(
    chunks: List[Dict[str, Any]],
    context_before: int = 1,
    context_after: int = 1,
    max_context_chunks_per_paper: int = 10,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    提取多模态 chunks 并附带同论文的上下文 chunks

    对于每个多模态 chunk，会添加：
    1. 同论文中相邻的普通 chunks（前后各 N 个）
    2. 同论文中的随机采样普通 chunks（最多 M 个）

    Args:
        chunks: 所有 chunks 列表
        context_before: 每个多模态 chunk 前面取几个普通 chunks
        context_after: 每个多模态 chunk 后面取几个普通 chunks
        max_context_chunks_per_paper: 每个论文最多添加的上下文 chunks 总数

    Returns:
        (多模态 chunks, 上下文 chunks) 元组
    """

    # 按 paper_id 分组
    chunks_by_paper = defaultdict(list)
    for i, chunk in enumerate(chunks):
        paper_id = chunk.get("metadata", {}).get("paper_id", "") or chunk.get("paper_id", "")
        if paper_id:
            chunks_by_paper[paper_id].append((i, chunk))

    # 复用 extract_multimodal_chunks 做多模态检测，获取带 _multimodal_type 标记的 chunks
    multimodal_with_tags = extract_multimodal_chunks(chunks)
    multimodal_ids = {c.get("id") for c in multimodal_with_tags}
    multimodal_indices = {i for i, c in enumerate(chunks) if c.get("id") in multimodal_ids}

    # 收集上下文 chunks
    context_indices = set()

    for paper_id, paper_chunks in chunks_by_paper.items():
        multimodal_in_paper = [idx for idx, _ in paper_chunks if idx in multimodal_indices]

        # 对于每个多模态 chunk，添加相邻的普通 chunks
        for mm_idx in multimodal_in_paper:
            # 前面的上下文
            for offset in range(1, context_before + 1):
                ctx_idx = mm_idx - offset
                if ctx_idx >= 0 and ctx_idx not in multimodal_indices and ctx_idx not in context_indices:
                    context_indices.add(ctx_idx)

            # 后面的上下文
            for offset in range(1, context_after + 1):
                ctx_idx = mm_idx + offset
                if ctx_idx < len(chunks) and ctx_idx not in multimodal_indices and ctx_idx not in context_indices:
                    context_indices.add(ctx_idx)

        # 如果上下文还不够，随机采样一些普通 chunks
        existing_count = len(context_indices)
        if existing_count < max_context_chunks_per_paper:
            non_multimodal = [idx for idx, _ in paper_chunks
                            if idx not in multimodal_indices and idx not in context_indices]
            needed = max_context_chunks_per_paper - existing_count
            # 按论文内顺序采样，保持均匀分布
            non_multimodal.sort()
            step = max(1, len(non_multimodal) // (needed * 2))
            sampled = non_multimodal[::step][:needed]
            for idx in sampled:
                context_indices.add(idx)

    # 构建返回列表
    multimodal_result = [chunks[i] for i in range(len(chunks)) if i in multimodal_indices]
    context_result = [chunks[i] for i in range(len(chunks)) if i in context_indices]

    print(f"🖼️ 多模态 chunks: {len(multimodal_result)}")
    print(f"📝 上下文 chunks: {len(context_result)}")

    return multimodal_result, context_result


def build_multimodal_documents(
    multimodal_chunks: List[Dict[str, Any]],
    figures_dir: str = "",
) -> List[Any]:
    """
    将多模态 chunks 构建为独立的 Document（每个 chunk 一个 Document）

    每个 Document 包含：
    - 原始文本
    - 图片路径和标题（如果有）
    - 表格内容和路径（如果有）

    Args:
        multimodal_chunks: extract_multimodal_chunks 返回的 chunks
        figures_dir: figures 目录路径

    Returns:
        多模态 Document 列表
    """

    if not figures_dir:
        plugin_dir = Path(__file__).parent.parent
        figures_dir = str(plugin_dir / "data" / "figures")

    documents = []
    for chunk in multimodal_chunks:
        metadata = chunk.get("metadata", {})
        text = chunk.get("text", "")
        multimodal_types = chunk.get("_multimodal_type", [])

        # 构建多模态描述（带格式指令前缀）
        parts = [MULTIMODAL_DOC_PREFIX, text]

        # 处理图片
        image_path = metadata.get("image_path", "")
        if image_path:
            image_caption = metadata.get("image_caption", "")
            if image_caption:
                parts.append(f"[IMAGE: {image_caption}]\nFile: {image_path}")
            else:
                parts.append(f"[IMAGE]\nFile: {image_path}")

        # 处理其他图片
        all_images = metadata.get("all_images", [])
        for img_info in all_images:
            if isinstance(img_info, dict):
                img_path = img_info.get("path", "")
                img_caption = img_info.get("caption", "")
                if img_path and img_path != image_path:
                    if img_caption:
                        parts.append(f"[IMAGE: {img_caption}]\nFile: {img_path}")
                    else:
                        parts.append(f"[IMAGE]\nFile: {img_path}")

        # 处理表格
        table_path = metadata.get("table_path", "")
        processed_table_paths = set()  # 避免重复处理

        if table_path:
            table_caption = metadata.get("table_caption", "")
            table_text = _read_table_csv_as_text(table_path)
            if table_caption:
                parts.append(f"[TABLE: {table_caption}]\nFile: {table_path}")
            else:
                parts.append(f"[TABLE]\nFile: {table_path}")
            if table_text:
                parts.append(table_text)
            processed_table_paths.add(table_path)

        # 也处理 multimodal_data.tables 中的表格（包含 markdown 内容）
        multimodal_data = metadata.get("multimodal_data", {})
        if isinstance(multimodal_data, dict):
            mm_tables = multimodal_data.get("tables", [])
            if isinstance(mm_tables, list):
                for table_info in mm_tables:
                    if isinstance(table_info, dict):
                        # multimodal_data.tables 中的表格没有 path 字段，直接使用 markdown 内容
                        tbl_markdown = table_info.get("markdown", "")
                        tbl_caption = table_info.get("caption", "")
                        page_num = table_info.get("page_number", "")
                        tbl_idx = table_info.get("table_index", "")

                        if tbl_markdown:
                            # 生成表格标识符（用于去重）
                            tbl_identifier = f"p{page_num}_t{tbl_idx}"
                            if tbl_identifier not in processed_table_paths:
                                if tbl_caption:
                                    parts.append(f"[TABLE {tbl_identifier}: {tbl_caption}]")
                                else:
                                    parts.append(f"[TABLE {tbl_identifier}]")
                                parts.append(tbl_markdown)
                                processed_table_paths.add(tbl_identifier)

        combined_text = "\n\n".join(parts)

        # 嵌入 chunk_id 用于后续匹配（不会影响 Ragas 生成）
        chunk_id = chunk.get("id", "")
        if chunk_id:
            combined_text = f"[CHUNK_ID:{chunk_id}]\n\n{combined_text}"

        doc = LIDocument(
            text=combined_text,
            metadata={
                "paper_id": chunk.get("paper_id", ""),
                "chunk_id": chunk_id,
                "source": "multimodal",
                "has_multimodal": True,  # 标记为多模态文档
                "multimodal_types": multimodal_types,
                "image_path": image_path,
                "table_path": table_path,
                "image_caption": metadata.get("image_caption", ""),
                "table_caption": metadata.get("table_caption", ""),
            }
        )
        documents.append(doc)

    print(f"📄 构建 {len(documents)} 个多模态 Document")
    return documents


# 多模态文档前缀指令
MULTIMODAL_DOC_PREFIX = """【重要格式要求】
- 答案中的论文引用必须使用标准 Markdown 链接格式，如 [论文名](url)
- 禁止将引用放在代码块中
- 例如：正确格式是 [NoPoSplat](https://arxiv.org/abs/2502.12138)

"""


async def generate_multimodal_testset(
    chunks: List[Dict[str, Any]],
    test_size: int = 20,
    output_path: str = "results/testset.json",
    llm: _EvalLLMConfig = _EvalLLMConfig(),
    context_before: int = 1,
    context_after: int = 1,
) -> List[Any]:
    """
    生成多模态问答对（专门针对图片和表格）

    会提取多模态 chunks 并附带同论文的上下文 chunks，使生成的问答能基于更完整的语境。

    Args:
        chunks: 所有 chunks 列表
        test_size: 生成问题数量
        output_path: 输出路径（会将结果 append 到已有文件）
        llm: LLM 配置容器
        context_before: 每个多模态 chunk 前面取几个普通 chunks 作为上下文
        context_after: 每个多模态 chunk 后面取几个普通 chunks 作为上下文

    Returns:
        生成的多模态样本列表
    """
    print(f"\n{'='*60}")
    print(f"🖼️ 步骤: 生成 {test_size} 个多模态评测问题（图片/表格）")
    print("=" * 60)

    # 提取多模态 chunks 和上下文 chunks
    multimodal_chunks, _ = extract_multimodal_chunks_with_context(
        chunks,
        context_before=context_before,
        context_after=context_after,
    )

    if not multimodal_chunks:
        print("⚠️ 没有找到与图片/表格关联的 chunks")
        return []

    # 构建多模态 documents
    multimodal_docs = build_multimodal_documents(multimodal_chunks)

    if not multimodal_docs:
        print("⚠️ 没有成功构建多模态文档")
        return []

    # 使用 Ragas 生成测试集
    generator = _create_testset_generator(llm)

    print(f"📊 多模态文档数量: {len(multimodal_docs)}")

    # 使用足够的文档
    trimmed_docs = multimodal_docs[:min(test_size * MULTIMODAL_DOC_MULTIPLIER, len(multimodal_docs))]

    # 使用临时文件避免覆盖主文件（generate_testset 会直接写入文件）
    temp_output = str(Path(tempfile.gettempdir()) / f"ragas_temp_{uuid.uuid4().hex}.json")

    print(f"正在调用 LLM 生成 {test_size} 个多模态问答对...")
    samples = await generator.generate_testset(
        documents=trimmed_docs,
        test_size=test_size,
        output_path=temp_output,
    )

    # 构建 chunk_id -> metadata 的映射（从原始 chunks）
    # 注意：chunk.get("id") 返回的是 int，需要转为 string 用于匹配
    chunk_id_to_metadata = {}
    for chunk in multimodal_chunks:
        chunk_id = str(chunk.get("id", ""))
        if chunk_id:
            metadata = chunk.get("metadata", {})
            chunk_id_to_metadata[chunk_id] = {
                "image_path": metadata.get("image_path", ""),
                "table_path": metadata.get("table_path", ""),
                "image_caption": metadata.get("image_caption", ""),
                "table_caption": metadata.get("table_caption", ""),
            }

    # 从 contexts[0] 解析 chunk_id 并找回 metadata
    multimodal_samples = []
    matched_count = 0
    unmatched_count = 0

    for sample in samples:
        contexts = sample.contexts if hasattr(sample, 'contexts') else []
        source_text = contexts[0] if contexts else ""

        # 解析 chunk_id
        match = re.search(r'\[CHUNK_ID:([^\]]+)\]', source_text)
        chunk_id = match.group(1) if match else ""
        metadata = chunk_id_to_metadata.get(chunk_id, {})

        if chunk_id and metadata:
            matched_count += 1
        else:
            unmatched_count += 1

        sample.metadata = {
            "is_multimodal": True,
            "multimodal_types": ["image", "table"],
            "image_path": metadata.get("image_path", ""),
            "table_path": metadata.get("table_path", ""),
            "image_caption": metadata.get("image_caption", ""),
            "table_caption": metadata.get("table_caption", ""),
            "is_context": False,
        }
        multimodal_samples.append(sample)

    print(f"📊 Metadata 匹配: {matched_count} 成功, {unmatched_count} 未匹配")

    # Append 到现有 testset.json
    existing_samples = []
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        try:
            with open(output_path_obj, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_samples = existing_data
                print(f"📖 读取已有测试集: {len(existing_samples)} 个样本")
            else:
                print(f"⚠️ 测试集格式错误（非列表）: {type(existing_data)}")
        except Exception as e:
            print(f"⚠️ 读取现有 testset 失败: {e}")
    else:
        print(f"📝 测试集文件不存在，将创建新文件")

    # 合并样本（带去重）
    new_sample_dicts = [s.to_dict() for s in multimodal_samples]

    # 去重：基于问题文本的 MD5 hash 去重，避免长文本截断导致的问题
    def get_question_key(sample: dict) -> str:
        q = sample.get("question", "")
        return hashlib.md5(q.encode()).hexdigest()[:16]

    seen_questions = set()
    deduped_existing = []
    for sample in existing_samples:
        q_key = get_question_key(sample)
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            deduped_existing.append(sample)

    deduped_new = []
    for sample in new_sample_dicts:
        q_key = get_question_key(sample)
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            deduped_new.append(sample)

    all_samples = deduped_existing + deduped_new

    print(f"🔄 合并（去重后）: {len(deduped_existing)} 已有 + {len(deduped_new)} 新增 = {len(all_samples)} 总计")
    if len(existing_samples) != len(deduped_existing) or len(new_sample_dicts) != len(deduped_new):
        print(f"   ⚠️ 去除 {len(existing_samples) - len(deduped_existing) + len(new_sample_dicts) - len(deduped_new)} 个重复样本")

    # 路径安全验证
    try:
        output_path_obj = Path(output_path).resolve()
        expected_dir = Path(__file__).parent.parent.joinpath("results").resolve()
        if not output_path_obj.is_relative_to(expected_dir):
            raise ValueError(f"Output path must be within {expected_dir}")
    except Exception as e:
        print(f"⚠️ 路径验证失败: {e}")
        raise

    # 原子写入
    _atomic_write_json(output_path_obj, all_samples)

    print(f"✅ 多模态测试集已生成并追加到: {output_path}")
    print(f"   新增 {len(multimodal_samples)} 个多模态样本")
    print(f"   总计 {len(all_samples)} 个样本")

    # 清理临时文件
    try:
        Path(temp_output).unlink()
    except Exception:
        pass

    return multimodal_samples


def _create_testset_generator(llm: _EvalLLMConfig) -> RagasTestsetGenerator:
    """从 LLM 配置创建 RagasTestsetGenerator — generate_testset_from_documents 和 generate_multimodal_testset 共用。"""
    return RagasTestsetGenerator(
        llm_model=llm.llm_model,
        llm_base_url=llm.llm_base_url,
        llm_api_key=llm.llm_api_key,
        embedding_model=llm.embedding_model,
        embed_base_url=llm.embed_base_url,
        embed_api_key=llm.embed_api_key,
        embedding_mode=llm.embedding_mode,
        max_rpm=llm.max_rpm,
    )


# ============================================================================
# 步骤 3: 生成测试集
# ============================================================================

async def generate_testset_from_documents(
    documents: List[Any],
    test_size: int = 50,
    output_path: str = "results/testset.json",
    llm: _EvalLLMConfig = _EvalLLMConfig(),
) -> List[Any]:
    """使用 Ragas 生成测试集"""
    print(f"\n{'='*60}")
    print(f"📝 步骤 2/4: 生成 {test_size} 个评测问题")
    print("=" * 60)

    generator = _create_testset_generator(llm)

    samples = await generator.generate_testset(
        documents=documents,
        test_size=test_size,
        output_path=output_path,
    )

    print(f"✅ 测试集生成完成: {len(samples)} 个样本 -> {output_path}")
    return samples


def _create_ragas_evaluator(llm: _EvalLLMConfig, answer_top_k: int = 5) -> RagasEvaluator:
    """从 LLM 配置创建 RagasEvaluator — run_evaluation 和 run_evaluation_from_raw_answers 共用。"""
    return RagasEvaluator(
        llm_model=llm.eval_llm_model,
        llm_base_url=llm.eval_llm_base_url,
        llm_api_key=llm.llm_api_key,
        embedding_model=llm.embedding_model,
        embed_base_url=llm.embed_base_url,
        embed_api_key=llm.embed_api_key,
        embedding_mode=llm.eval_embedding_mode,
        answer_top_k=answer_top_k,
        llm_max_tokens=llm.eval_llm_max_tokens,
    )


# ============================================================================
# 步骤 4: 执行评估
# ============================================================================

async def run_evaluation(
    query_engine: Any,
    testset_path: str,
    output_path: str = "results/evaluation_results.csv",
    max_concurrent: int = 5,
    llm: _EvalLLMConfig = _EvalLLMConfig(),
    answer_top_k: int = 5,
    context=None,
    config: Optional[dict] = None,
    mode: str = "rag",
) -> Any:
    """执行 Ragas 评估"""
    mode_label = "Agentic RAG" if mode == "agentic" else "RAG"
    print(f"\n{'='*60}")
    print(f"📊 步骤 3/4: 执行 Ragas 评估 [{mode_label}]")
    print("=" * 60)
    print(f"📊 指标评估: {llm.eval_llm_model} @ {llm.eval_llm_base_url}")

    evaluator = _create_ragas_evaluator(llm, answer_top_k)

    results = await evaluator.evaluate(
        query_engine=query_engine,
        testset_path=testset_path,
        output_path=output_path,
        max_concurrent=max_concurrent,
        context=context,
        config=config,
        mode=mode,
    )

    print(f"✅ 评估完成 -> {output_path}")
    return results


async def run_evaluation_from_raw_answers(
    raw_answers_path: str,
    output_path: str = "results/evaluation_results.csv",
    max_concurrent: int = 5,
    llm: _EvalLLMConfig = _EvalLLMConfig(),
    answer_top_k: int = 5,
) -> Any:
    """从已有 raw_answers.json 执行 Ragas 评估（跳过 RAG 推理）"""
    print(f"\n{'='*60}")
    print("📊 步骤 3/4: 执行 Ragas 评估（跳过 RAG 推理）")
    print("=" * 60)

    evaluator = _create_ragas_evaluator(llm, answer_top_k)

    results = await evaluator.evaluate_from_raw_answers(
        raw_answers_path=raw_answers_path,
        output_path=output_path,
        max_concurrent=max_concurrent,
    )

    print(f"✅ 评估完成 -> {output_path}")
    return results


# ============================================================================
# 步骤 5: 生成报告
# ============================================================================

def generate_reports(
    results_path: str,
    output_dir: str = "results",
    plugin_version: str = "1.0.0",
    paper_name: str = "AstrBot Paper RAG (Milvus)",
) -> Dict[str, str]:
    """生成 HTML + Markdown 报告"""
    print(f"\n{'='*60}")
    print("📋 步骤 4/4: 生成评测报告")
    print("=" * 60)


    reporter = ReportGenerator(results_path)
    html_path = reporter.generate_html_report(
        str(Path(output_dir) / "evaluation_report.html"),
        plugin_version=plugin_version,
        paper_name=paper_name,
    )
    md_path = reporter.generate_markdown_report(
        str(Path(output_dir) / "evaluation_report.md"),
        plugin_version=plugin_version,
        paper_name=paper_name,
    )

    print(f"✅ 报告生成完成:")
    print(f"   HTML: {html_path}")
    print(f"   Markdown: {md_path}")

    return {"html": html_path, "markdown": md_path}


def _build_rag_config(plugin_dir: Path, top_k: int | None = None,
                      plugin_config: dict | None = None) -> tuple:
    """从插件配置文件构建 RAGConfig 和 raw config dict（两处调用共享）。

    如已通过 _resolve_eval_config() 加载 plugin_config，可传入避免重复读文件。
    """
    if plugin_config is not None:
        rag_cfg = plugin_config
    else:
        config_path = plugin_dir.parent.parent / "config" / "astrbot_plugin_paperrag_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8-sig") as f:
                rag_cfg = json.load(f)
        else:
            rag_cfg = {}

    config = RAGConfig(
        embedding_mode=rag_cfg.get("embedding_mode", "unsloth"),
        embedding_provider_id=rag_cfg.get("embedding_provider_id", ""),
        compress_provider_id=rag_cfg.get("compress_provider_id", ""),
        text_provider_id=rag_cfg.get("text_provider_id", ""),
        milvus_lite_path=str(plugin_dir / "data" / "milvus_papers.db"),
        address=rag_cfg.get("milvus", {}).get("address", ""),
        db_name=rag_cfg.get("milvus", {}).get("db_name", "default"),
        collection_name=rag_cfg.get("collection_name", "paper_embeddings"),
        embed_dim=rag_cfg.get("embed_dim", 1024),
        top_k=top_k if top_k is not None else rag_cfg.get("top_k", 5),
        similarity_cutoff=rag_cfg.get("similarity_cutoff", 0.5),
        chunk_size=rag_cfg.get("chunk_size", 512),
        min_chunk_size=rag_cfg.get("min_chunk_size", 100),
        use_semantic_chunking=rag_cfg.get("use_semantic_chunking", True),
        enable_sparse_retrieval=rag_cfg.get("enable_sparse_retrieval", True),
        enable_multi_vector_rerank=rag_cfg.get("enable_multi_vector_rerank", False),
        sparse_top_k=rag_cfg.get("sparse_top_k", 20),
        hybrid_alpha=rag_cfg.get("hybrid_alpha", 0.5),
        hybrid_rrf_k=rag_cfg.get("hybrid_rrf_k", 60),
        enable_bm25=rag_cfg.get("enable_bm25", True),
        bm25_top_k=rag_cfg.get("bm25_top_k", 20),
        enable_two_stage_retrieval=rag_cfg.get("enable_two_stage_retrieval", False),
        enable_crag_quality_eval=rag_cfg.get("enable_crag_quality_eval", True),
        crag_enable_correction=rag_cfg.get("crag_enable_correction", True),
        crag_min_score=rag_cfg.get("crag_min_score", 0.5),
        enable_graph_rag=rag_cfg.get("enable_graph_rag", False),
        graph_storage_type=rag_cfg.get("graph_rag", {}).get("storage_type", "neo4j"),
        graph_neo4j_uri=rag_cfg.get("graph_rag", {}).get("neo4j_uri", "bolt://localhost:7687"),
        graph_neo4j_user=rag_cfg.get("graph_rag", {}).get("neo4j_user", "neo4j"),
        graph_neo4j_password=rag_cfg.get("graph_rag", {}).get("neo4j_password", ""),
        graph_retrieval_top_k=rag_cfg.get("graph_rag", {}).get("graph_retrieval_top_k", 5),
        graph_max_triplets_per_chunk=rag_cfg.get("graph_rag", {}).get("max_triplets_per_chunk", 5),
    )
    return config, rag_cfg


# ============================================================================
# 完整流程
# ============================================================================

async def _run_rag_eval(
    testset_path: str,
    results_path: str,
    llm: _EvalLLMConfig,
    sys_provider_mgr,
    top_k: int | None,
    max_concurrent: int,
    answer_top_k: int,
    mode: str = "rag",
    plugin_config: dict | None = None,
) -> Any:
    """创建 HybridRAG 引擎并执行评估 — run_full_pipeline 和 --step evaluate 共用。"""
    plugin_dir = Path(__file__).parent.parent
    config, rag_cfg = _build_rag_config(plugin_dir, top_k, plugin_config)
    fake_context = _EvalFakeContext(provider_manager=sys_provider_mgr)
    mode_label = "Agentic RAG" if mode == "agentic" else "RAG"
    print(f"🔧 初始化 HybridRAG 引擎 [{mode_label}]...")
    engine = create_rag_engine(config, fake_context)
    print("✅ HybridRAG 引擎创建成功")
    return await run_evaluation(
        query_engine=engine,
        testset_path=testset_path,
        output_path=results_path,
        max_concurrent=max_concurrent,
        llm=llm,
        answer_top_k=answer_top_k,
        context=fake_context,
        config=rag_cfg,
        mode=mode,
    )


async def run_full_pipeline(
    output_dir: str = "results",
    test_size: int = 50,
    multimodal_test_size: int = 0,
    max_concurrent: int = 5,
    llm: _EvalLLMConfig = _EvalLLMConfig(),
    plugin_version: str = "1.0.0",
    use_existing_chunks: bool = False,
    existing_chunks_path: str = "results/milvus_chunks.json",
    top_k: int | None = None,
    multimodal_context_before: int = 1,
    multimodal_context_after: int = 1,
    answer_top_k: int = 5,
    provider_manager=None,
    mode: str = "rag",
    plugin_config: dict | None = None,
) -> dict:
    """
    完整评测流程

    Args:
        use_existing_chunks: 是否使用已提取的 chunks 文件（避免重复从 Milvus 读取）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========== 步骤 1: 提取文本 ==========
    chunks_path = existing_chunks_path if use_existing_chunks else str(output_path / "milvus_chunks.json")

    if use_existing_chunks and Path(chunks_path).exists():
        print(f"\n{'='*60}")
        print("📤 步骤 1/4: 加载已有 chunks 文件")
        print("=" * 60)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✅ 加载 {len(chunks)} chunks from {chunks_path}")
    else:
        chunks = await extract_chunks_from_milvus(
            output_path=chunks_path,
        )

    if not chunks:
        return {"success": False, "error": "No chunks extracted from Milvus"}

    # ========== 步骤 2: 构建 Document + 生成测试集 ==========
    documents = chunks_to_documents(chunks, min_chunks_per_paper=5, include_multimodal=True)

    if not documents:
        return {"success": False, "error": "No valid documents created"}

    testset_path = str(output_path / "testset.json")
    await generate_testset_from_documents(
        documents=documents,
        test_size=test_size,
        output_path=testset_path,
        llm=llm,
    )

    # ========== 步骤 2.5: 生成多模态测试集（追加到同一文件） ==========
    if multimodal_test_size > 0:
        await generate_multimodal_testset(
            chunks=chunks,
            test_size=multimodal_test_size,
            output_path=testset_path,
            llm=llm,
            context_before=multimodal_context_before,
            context_after=multimodal_context_after,
        )

    # ========== 步骤 3+4: 创建引擎 + 执行评估 ==========
    results_path = str(output_path / "evaluation_results.csv")
    await _run_rag_eval(
        testset_path=testset_path,
        results_path=results_path,
        llm=llm,
        sys_provider_mgr=provider_manager,
        top_k=top_k,
        max_concurrent=max_concurrent,
        answer_top_k=answer_top_k,
        mode=mode,
        plugin_config=plugin_config,
    )

    # ========== 步骤 5: 生成报告 ==========
    reports = generate_reports(
        results_path=results_path,
        output_dir=str(output_path),
        plugin_version=plugin_version,
        paper_name="AstrBot Paper RAG (Milvus DB)",
    )

    # ========== 完成 ==========
    print(f"\n{'='*60}")
    print("🎉 评测完成！")
    print(f"{'='*60}")
    print(f"📁 结果目录: {output_dir}/")
    print(f"   • Chunks: {chunks_path}")
    print(f"   • 测试集: {testset_path}")
    print(f"   • 评估结果: {results_path}")
    print(f"   • HTML报告: {reports['html']}")
    print(f"   • Markdown报告: {reports['markdown']}")
    print(f"{'='*60}")

    return {
        "success": True,
        "chunks": chunks_path,
        "testset": testset_path,
        "results": results_path,
        "reports": reports,
    }


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ragas 自动化评测工具 - 从 Milvus 数据库生成测试集并评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（提取文本 -> 生成测试集 -> 评估 -> 报告）
  python -m evaluation.run_evaluation_ragas --step all

  # 从 Milvus 提取全量文本（调试用）
  python -m evaluation.run_evaluation_ragas --step extract

  # 仅生成测试集（需已有 milvus_chunks.json）
  python -m evaluation.run_evaluation_ragas --step generate

  # 使用已有 chunks 文件（避免重复从数据库读取）
  python -m evaluation.run_evaluation_ragas --step all --use-existing-chunks

  # 指定测试集大小
  python -m evaluation.run_evaluation_ragas --step all --test-size 100

环境变量:
  EVAL_LLM_API_KEY 评估用 LLM API Key
        """
    )

    # 步骤参数
    parser.add_argument(
        "--step",
        choices=["all", "extract", "generate", "evaluate", "multimodal"],
        default="all",
        help="执行步骤: all=完整流程, extract=仅提取文本, generate=仅生成测试集, evaluate=需提供引擎, multimodal=仅生成多模态测试集"
    )

    # 输出
    parser.add_argument("--output-dir", default="results", help="输出目录")

    # 测试集配置
    parser.add_argument("--test-size", type=int, default=50, help="生成测试问题数量")
    parser.add_argument("--top-k", type=int, default=None, help="检索时返回的 top_k（覆盖配置文件）")
    parser.add_argument("--multimodal-test-size", type=int, default=0, help="多模态测试问题数量（默认0，需显式指定）")
    parser.add_argument("--multimodal-context-before", type=int, default=1, help="多模态 chunk 前面的上下文 chunks 数量")
    parser.add_argument("--multimodal-context-after", type=int, default=1, help="多模态 chunk 后面的上下文 chunks 数量")

    # 评测端点选择：MiniMax Token Plan（余额多）vs 智谱 GLM（标准兼容、无补丁开销）
    parser.add_argument(
        "--eval-provider",
        choices=["auto", "minimax", "zhipu"],
        default="auto",
        help="评测用 LLM 端点：auto=按插件 text_provider_id，minimax=MiniMax-M3 Token Plan，zhipu=智谱 GLM-5.2（标准 OpenAI 兼容，无思考模式补丁开销）",
    )

    # LLM 配置（用于生成测试集）
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM 模型名称")
    parser.add_argument("--llm-base-url", default=_DEFAULT_LLM_URL, help="LLM API 基础 URL")
    parser.add_argument("--llm-api-key", default="", help="LLM API Key（可使用环境变量）")

    # Eval LLM 配置（用于评估指标计算）
    parser.add_argument("--eval-llm-model", default="gpt-5.4-nano", help="评估用 LLM 模型名称")
    parser.add_argument("--eval-llm-base-url", default="", help="评估用 LLM API 基础 URL（留空则使用 free.json 中的 vapi 配置）")
    parser.add_argument("--eval-llm-api-key", default="", help="评估用 LLM API Key")
    parser.add_argument("--eval-llm-max-tokens", type=int, default=16384,
                        help="评估用 LLM max_tokens（默认 16384，推理模型需更高值容纳 reasoning tokens）")

    # Embedding 配置
    parser.add_argument("--embedding-model", default="text-embedding-v4", help="Embedding 模型名称")
    parser.add_argument("--embed-base-url", default="", help="Embedding API 基础 URL（使用 freeapi_url）")
    parser.add_argument("--embed-api-key", default="", help="Embedding API Key")
    parser.add_argument(
        "--embedding-mode",
        choices=["api", "unsloth"],
        default="api",
        help="Embedding 模式: api=使用远程API, unsloth=本地 BGE-M3 (默认: api)"
    )

    # 评估用 Embedding 配置
    parser.add_argument(
        "--eval-embedding-mode",
        choices=["api", "unsloth"],
        default="api",
        help="评估指标用 Embedding 模式: api=使用远程API, unsloth=本地 BGE-M3 (默认)"
    )

    # 评测参数
    parser.add_argument("--mode", choices=["rag", "agentic"], default="rag",
                        help="评估模式: rag=普通RAG检索+LLM生成, agentic=Agentic RAG (LangGraph workflow)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="最大并发数")
    parser.add_argument("--answer-top-k", type=int, default=5, help="用 top-K 个检索 chunk 生成答案（默认5）")
    parser.add_argument("--max-rpm", type=int, default=30, help="RPM 限制（默认30：MiniMax Token Plan 限流敏感，过高会触发大量 429）")

    # 报告配置
    parser.add_argument("--plugin-version", default="1.0.0", help="插件版本")
    parser.add_argument("--paper-name", default="AstrBot Paper RAG", help="论文/系统名称")

    # 已有数据
    parser.add_argument("--use-existing-chunks", action="store_true", help="使用已有 chunks 文件（避免重复从 Milvus 读取）")
    parser.add_argument("--existing-chunks-path", default="results/milvus_chunks.json", help="已有 chunks 文件路径")
    parser.add_argument("--skip-rag", action="store_true", help="跳过 RAG 推理，直接从 raw_answers.json 读取已有结果进行评估")
    parser.add_argument("--raw-answers-path", default="results/raw_answers.json", help="raw_answers.json 路径（用于 --skip-rag）")
    parser.add_argument("--testset-path", default=None, help="指定测试集路径（覆盖默认 testset.json）")

    args = parser.parse_args()

    # 解析 LLM 配置（CLI > env > plugin config > free.json）
    llm, plugin_config = _resolve_eval_config(args)

    # 初始化系统 LLM provider（用于答案生成）
    sys_provider_mgr = _init_system_llm_provider(plugin_config)

    # 设置 RPM 限制
    OpenAICompatibleLLM.set_max_rpm(llm.max_rpm)

    # ========== 根据步骤执行 ==========
    if args.step == "all":
        asyncio.run(run_full_pipeline(
            output_dir=args.output_dir,
            test_size=args.test_size,
            multimodal_test_size=args.multimodal_test_size,
            max_concurrent=args.max_concurrent,
            llm=llm,
            plugin_version=args.plugin_version,
            use_existing_chunks=args.use_existing_chunks,
            existing_chunks_path=args.existing_chunks_path,
            top_k=args.top_k,
            multimodal_context_before=args.multimodal_context_before,
            multimodal_context_after=args.multimodal_context_after,
            answer_top_k=args.answer_top_k,
            provider_manager=sys_provider_mgr,
            mode=args.mode,
            plugin_config=plugin_config,
        ))

    elif args.step == "extract":
        path = args.existing_chunks_path
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        chunks = asyncio.run(extract_chunks_from_milvus(
            output_path=path,
        ))
        print(f"\n✅ 提取 {len(chunks)} chunks 完成")

    elif args.step == "generate":
        chunks_path = args.existing_chunks_path
        if not Path(chunks_path).exists():
            print(f"❌ 文件不存在: {chunks_path}")
            print("请先运行: python -m evaluation.run_evaluation_ragas --step extract")
            return
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        documents = chunks_to_documents(chunks, include_multimodal=True)
        testset_path = str(Path(args.output_dir) / "testset.json")

        async def _generate_step():
            await generate_testset_from_documents(
                documents=documents,
                test_size=args.test_size,
                output_path=testset_path,
                llm=llm,
            )
            if args.multimodal_test_size > 0:
                await generate_multimodal_testset(
                    chunks=chunks,
                    test_size=args.multimodal_test_size,
                    output_path=testset_path,
                    llm=llm,
                    context_before=args.multimodal_context_before,
                    context_after=args.multimodal_context_after,
                )

        asyncio.run(_generate_step())

    elif args.step == "evaluate":
        # ========== 仅执行评估（使用已有测试集）==========
        results_path = str(Path(args.output_dir) / "evaluation_results.csv")

        if args.skip_rag:
            # 跳过 RAG 推理，直接从 raw_answers.json 读取评估
            raw_answers_path = args.raw_answers_path
            if not Path(raw_answers_path).exists():
                print(f"❌ raw_answers.json 不存在: {raw_answers_path}")
                print("请先运行带 RAG 推理的评估命令，生成该文件")
                return
            # 读取 raw_answers（兼容新旧格式）
            try:
                raw_data, _ = load_raw_answers(raw_answers_path)
            except ValueError as e:
                print(f"❌ 无法识别的 raw_answers.json 格式: {e}")
                print("   文件必须是旧格式（JSON 数组）或新格式（含 _metadata 和 results 的 JSON 对象）")
                return
            actual_count = len(raw_data)
            print(f"✅ 跳过 RAG 推理，从已有结果评估")
            print(f"   raw_answers: {raw_answers_path}")
            print(f"   测试问题数量: {actual_count} (来自 raw_answers.json)")
            asyncio.run(run_evaluation_from_raw_answers(
                raw_answers_path=raw_answers_path,
                output_path=results_path,
                max_concurrent=args.max_concurrent,
                llm=llm,
                answer_top_k=args.answer_top_k,
            ))
        else:
            # 正常流程：RAG 推理 + 评估
            testset_path = args.testset_path if args.testset_path else str(Path(args.output_dir) / "testset.json")

            if not Path(testset_path).exists():
                print(f"❌ 测试集不存在: {testset_path}")
                print("请先运行: python -m evaluation.run_evaluation_ragas --step all --use-existing-chunks")
                return

            print(f"✅ 使用已有测试集: {testset_path}")

            asyncio.run(_run_rag_eval(
                testset_path=testset_path,
                results_path=results_path,
                llm=llm,
                sys_provider_mgr=sys_provider_mgr,
                top_k=args.top_k,
                max_concurrent=args.max_concurrent,
                answer_top_k=args.answer_top_k,
                mode=args.mode,
                plugin_config=plugin_config,
            ))

        # 生成报告
        generate_reports(
            results_path=results_path,
            output_dir=args.output_dir,
            plugin_version=args.plugin_version,
            paper_name=args.paper_name,
        )

        print(f"\n🎉 评估完成！结果: {results_path}")

    elif args.step == "multimodal":
        # ========== 仅生成多模态测试集 ==========
        chunks_path = args.existing_chunks_path
        if args.use_existing_chunks and Path(chunks_path).exists():
            print(f"\n{'='*60}")
            print("📤 加载已有 chunks 文件")
            print("=" * 60)
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            print(f"✅ 加载 {len(chunks)} chunks from {chunks_path}")
        else:
            chunks = asyncio.run(extract_chunks_from_milvus(output_path=chunks_path))

        if not chunks:
            print("❌ 没有可用的 chunks")
            return

        asyncio.run(generate_multimodal_testset(
            chunks=chunks,
            test_size=args.multimodal_test_size,
            output_path=str(Path(args.output_dir) / "testset.json"),
            llm=llm,
            context_before=args.multimodal_context_before,
            context_after=args.multimodal_context_after,
        ))
        print(f"\n🎉 多模态测试集生成完成！")


if __name__ == "__main__":
    main()
