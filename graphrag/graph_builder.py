"""
Graph Builder - 多模态知识图谱构建器

使用本地 LLM (Qwen3.5 GGUF) 从文档中抽取三元组，构建知识图谱。

支持：
1. 纯文本三元组抽取
2. 多模态（图+文）联合三元组抽取
3. 图片实体提取
4. 跨模态关系建立

Closed-set 实体类型 (9 类):
Method, Model, Task, Dataset, Metric, Component, Limitation, Application, Baseline

Closed-set 关系类型 (9 类):
ADDRESSES, PROPOSES, USES_COMPONENT, EVALUATED_ON, ACHIEVES, COMPARES_WITH, LIMITED_BY, APPLIES_TO, EXTENDS
"""

from __future__ import annotations

import json
import os
import re
import hashlib
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

from astrbot.api import logger
from llama_cpp import LlamaGrammar
from rag.token_utils import count_tokens


_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
_GRAMMAR_DIR = Path(__file__).resolve().parent

# ============================================================================
# Closed-set content-oriented relation schema
# ============================================================================

CLOSED_RELATION_TYPES: frozenset[str] = frozenset({
    "ADDRESSES", "PROPOSES", "USES_COMPONENT", "EVALUATED_ON",
    "ACHIEVES", "COMPARES_WITH", "LIMITED_BY", "APPLIES_TO", "EXTENDS",
    "TRAINS_ON", "IMPLEMENTS", "OUTPERFORMS", "REQUIRES", "ABLATES_ON",
})

CLOSED_ENTITY_TYPES: frozenset[str] = frozenset({
    "Method", "Model", "Task", "Dataset", "Metric",
    "Component", "Limitation", "Application", "Baseline",
})

RELATION_ALIASES: Dict[str, str] = {
    "based_on": "EXTENDS",
    "uses": "USES_COMPONENT",
    "achieves": "ACHIEVES",
    "outperforms": "OUTPERFORMS",
    "beats": "OUTPERFORMS",
    "improves": "EXTENDS",
    "proposes": "PROPOSES",
    "introduces": "PROPOSES",
    "trained_on": "TRAINS_ON",
    "trains_on": "TRAINS_ON",
    "applied_to": "APPLIES_TO",
    "compares_with": "COMPARES_WITH",
    "combines_with": "USES_COMPONENT",
    "integrates": "USES_COMPONENT",
    "depends_on": "USES_COMPONENT",
    "github": "IMPLEMENTS",
    "code": "IMPLEMENTS",
    "needs": "REQUIRES",
    "demands": "REQUIRES",
    "ablation": "ABLATES_ON",
    "ablates": "ABLATES_ON",
    "studies": "ABLATES_ON",
}

ENTITY_TYPE_ALIASES: Dict[str, str] = {
    "model/architecture": "Model",
    "method/technique": "Method",
    "optimizer/algorithm": "Method",
    "framework/library": "Component",
    "hyperparameter": "Component",
    "result/conclusion": "Metric",
    "application/domain": "Application",
    "comparison": "Baseline",
    "concept": "Method",
    "experiment": "Dataset",
    "other": "Method",
}

_EXT_TO_FIGURE_TYPE: Dict[str, str] = {
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".svg": "diagram", ".pdf": "document", ".gif": "image",
    ".tiff": "image", ".bmp": "image", ".webp": "image",
}


# ============================================================================
# VLM optimization: 哈希缓存 + 确定性降级
# ============================================================================

# VLM 缓存（进程内，key = SHA256(text)[:12]##SHA256(image_content)[:12]）
_VLM_CACHE: Dict[str, Dict[str, Any]] = {}


def _vlm_cache_key(text: str, image_path: str) -> str:
    """生成 VLM 缓存键（SHA256 前12位，跨运行一致）"""
    img_key = ""
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as f:
            img_key = hashlib.sha256(f.read()).hexdigest()[:12]
    elif image_path:
        img_key = hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:12]
    text_key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{text_key}##{img_key}"

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from .graph_rag_engine import GraphRAGConfig


# ============================================================================
# 配置
# ============================================================================

@dataclass
class LocalLLMConfig:
    """本地 LLM 配置"""
    model_path: str = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    mmproj_path: str = "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"
    n_ctx: int = 8192
    n_gpu_layers: int = 99
    max_tokens: int = 8192
    temperature: float = 0.1


# ============================================================================
# Prompt 模板（全英文）
# ============================================================================

BATCH_TRIPLET_EXTRACTION_PROMPT = """You are a strict academic content extractor. Extract entity-relation triplets ONLY about paper content from the given chunks. Each chunk is labeled with [Chunk X].

## Your Task
Extract meaningful content-level relationship triplets from the given paper text chunks.

## Output Format
```json
{{
  "triplets": [
    {{
      "head": "Head entity name (concise, max 30 chars)",
      "head_type": "Entity type (from closed set below)",
      "relation": "Natural language relation description",
      "relation_type": "Relation keyword (from closed set below)",
      "tail": "Tail entity name (concise, max 30 chars)",
      "tail_type": "Entity type (from closed set below)",
      "confidence": 0.95,
      "evidence": "[Chunk X]"
    }}
  ]
}}
```

## Entity Types (closed set — use EXACTLY one of these)
- Method: attention mechanism, optimization method, training technique, algorithm
- Model: BERT, GPT, Transformer, ResNet, named architectures
- Task: text classification, translation, QA, generation, detection
- Dataset: GLUE, ImageNet, COCO, benchmark names
- Metric: accuracy, F1, BLEU, precision, recall, loss, perplexity
- Component: layer type, module, sub-architecture, building block
- Limitation: weakness, constraint, failure mode, boundary condition
- Application: real-world use case, domain, deployment scenario
- Baseline: previous method, compared system, prior work

## Relation Types (closed set — relation_type MUST be one of these keywords)
- ADDRESSES: Paper/Method → Task or problem it targets
- PROPOSES: Paper/Method → Method or Model it introduces
- USES_COMPONENT: Method → Component or technique it incorporates
- EVALUATED_ON: Method → Dataset used for evaluation
- ACHIEVES: Method → Metric result or performance attained
- COMPARES_WITH: Method → Baseline it is compared against
- LIMITED_BY: Method → Limitation it suffers from
- APPLIES_TO: Method → Application domain it targets
- EXTENDS: Method → Prior work or model it builds upon
- TRAINS_ON: Model → Dataset used for training
- IMPLEMENTS: Model → Code repository providing implementation
- OUTPERFORMS: Method → Baseline it significantly exceeds
- REQUIRES: Method → Hardware or resource requirement
- ABLATES_ON: Method → Component contribution being studied

## Strict Rules
1. IGNORE ALL METADATA: authors, institutions, venues, dates, grants, affiliations
2. Extract ONLY the MOST IMPORTANT relationships (prioritize high-impact relations)
3. Entity names MUST come from the original text
4. relation_type MUST be chosen from the closed set above — never invent new ones
5. Confidence: 0.0-1.0 based on text clarity
6. **STRICT LIMIT**: Maximum {{max_triplets}} triplets TOTAL for ALL chunks combined
7. **IMPORTANT**: evidence field must ONLY contain "[Chunk X]" — do NOT include original text
8. Keep entity names concise (max 30 chars), use abbreviations if needed

## Example
Chunks:
[Chunk 1] BERT is based on the Transformer encoder architecture.
[Chunk 2] BERT achieves 86.4% accuracy on GLUE benchmark, outperforming ELMo.

Output:
```json
{{
  "triplets": [
    {{
      "head": "BERT",
      "head_type": "Model",
      "relation": "based on",
      "relation_type": "EXTENDS",
      "tail": "Transformer",
      "tail_type": "Model",
      "confidence": 0.98,
      "evidence": "[Chunk 1]"
    }},
    {{
      "head": "BERT",
      "head_type": "Model",
      "relation": "evaluated on GLUE",
      "relation_type": "EVALUATED_ON",
      "tail": "GLUE benchmark",
      "tail_type": "Dataset",
      "confidence": 0.95,
      "evidence": "[Chunk 2]"
    }},
    {{
      "head": "BERT",
      "head_type": "Model",
      "relation": "outperforms",
      "relation_type": "COMPARES_WITH",
      "tail": "ELMo",
      "tail_type": "Baseline",
      "confidence": 0.92,
      "evidence": "[Chunk 2]"
    }}
  ]
}}
```
"""

TRIPLET_EXTRACTION_PROMPT = """You are a strict academic content extractor. Extract entity-relation triplets ONLY about paper content.

## Your Task
Extract meaningful content-level relationship triplets from the given paper text.

## Output Format
```json
{{
  "triplets": [
    {{
      "head": "Head entity name (concise, max 30 chars)",
      "head_type": "Entity type (from closed set below)",
      "relation": "Natural language relation description",
      "relation_type": "Relation keyword (from closed set below)",
      "tail": "Tail entity name (concise, max 30 chars)",
      "tail_type": "Entity type (from closed set below)",
      "confidence": 0.95,
      "evidence": "Short phrase from text (max 50 chars)"
    }}
  ]
}}
```

## Entity Types (closed set — use EXACTLY one of these)
- Method: attention mechanism, optimization method, training technique, algorithm
- Model: BERT, GPT, Transformer, ResNet, named architectures
- Task: text classification, translation, QA, generation, detection
- Dataset: GLUE, ImageNet, COCO, benchmark names
- Metric: accuracy, F1, BLEU, precision, recall, loss, perplexity
- Component: layer type, module, sub-architecture, building block
- Limitation: weakness, constraint, failure mode, boundary condition
- Application: real-world use case, domain, deployment scenario
- Baseline: previous method, compared system, prior work

## Relation Types (closed set — relation_type MUST be one of these keywords)
- ADDRESSES: Paper/Method → Task or problem it targets
- PROPOSES: Paper/Method → Method or Model it introduces
- USES_COMPONENT: Method → Component or technique it incorporates
- EVALUATED_ON: Method → Dataset used for evaluation
- ACHIEVES: Method → Metric result or performance attained
- COMPARES_WITH: Method → Baseline it is compared against
- LIMITED_BY: Method → Limitation it suffers from
- APPLIES_TO: Method → Application domain it targets
- EXTENDS: Method → Prior work or model it builds upon
- TRAINS_ON: Model → Dataset used for training
- IMPLEMENTS: Model → Code repository providing implementation
- OUTPERFORMS: Method → Baseline it significantly exceeds
- REQUIRES: Method → Hardware or resource requirement
- ABLATES_ON: Method → Component contribution being studied

## Strict Rules
1. IGNORE ALL METADATA: authors, institutions, venues, dates, grants, affiliations
2. Extract ONLY the MOST IMPORTANT relationships (prioritize key findings and major contributions)
3. Entity names MUST come from the original text
4. relation_type MUST be chosen from the closed set above — never invent new ones
5. Confidence: 0.0-1.0 based on text clarity
6. **STRICT LIMIT**: Maximum {{max_triplets}} triplets TOTAL
7. **IMPORTANT**: evidence field must be a SHORT phrase (max 50 chars) — do NOT include long text snippets
8. Keep entity names concise (max 30 chars)

## Example
Input: "BERT is based on the Transformer encoder architecture and achieves 86.4% accuracy on GLUE benchmark, outperforming all previous models."

Output:
```json
{{
  "triplets": [
    {{
      "head": "BERT",
      "head_type": "Model",
      "relation": "based on",
      "relation_type": "EXTENDS",
      "tail": "Transformer",
      "tail_type": "Model",
      "confidence": 0.98,
      "evidence": "based on Transformer"
    }},
    {{
      "head": "BERT",
      "head_type": "Model",
      "relation": "achieves 86.4% on GLUE",
      "relation_type": "EVALUATED_ON",
      "tail": "GLUE benchmark",
      "tail_type": "Dataset",
      "confidence": 0.95,
      "evidence": "86.4% on GLUE"
    }},
    {{
      "head": "BERT",
      "head_type": "Model",
      "relation": "outperforms previous models",
      "relation_type": "COMPARES_WITH",
      "tail": "previous models",
      "tail_type": "Baseline",
      "confidence": 0.85,
      "evidence": "outperforms previous models"
    }}
  ]
}}
```
"""


MULTIMODAL_TRIPLET_EXTRACTION_PROMPT = """You are a multimodal academic content extractor. Extract entity-relation triplets from academic papers with images.

## Your Task
1. Extract triplets from the TEXT
2. Analyze the IMAGE and extract figure information
3. Establish CROSS-MODAL relations between text entities and figure

## Input
Text: {{text}}
Image Caption: {{image_caption}}
Image: (provided as image input)

## Entity Types (closed set — use EXACTLY one of these)
- Method: attention mechanism, optimization method, training technique, algorithm
- Model: BERT, GPT, Transformer, ResNet, named architectures
- Task: text classification, translation, QA, generation, detection
- Dataset: GLUE, ImageNet, COCO, benchmark names
- Metric: accuracy, F1, BLEU, precision, recall, loss, perplexity
- Component: layer type, module, sub-architecture, building block
- Limitation: weakness, constraint, failure mode, boundary condition
- Application: real-world use case, domain, deployment scenario
- Baseline: previous method, compared system, prior work

## Relation Types for text_triplets (closed set — relation_type MUST be one of these)
- ADDRESSES: Paper/Method → Task or problem it targets
- PROPOSES: Paper/Method → Method or Model it introduces
- USES_COMPONENT: Method → Component or technique it incorporates
- EVALUATED_ON: Method → Dataset used for evaluation
- ACHIEVES: Method → Metric result or performance attained
- COMPARES_WITH: Method → Baseline it is compared against
- LIMITED_BY: Method → Limitation it suffers from
- APPLIES_TO: Method → Application domain it targets
- EXTENDS: Method → Prior work or model it builds upon
- TRAINS_ON: Model → Dataset used for training
- IMPLEMENTS: Model → Code repository providing implementation
- OUTPERFORMS: Method → Baseline it significantly exceeds
- REQUIRES: Method → Hardware or resource requirement
- ABLATES_ON: Method → Component contribution being studied

## Strict Rules
1. IGNORE ALL METADATA: authors, institutions, venues, dates, grants, affiliations
2. text_triplets: relation_type MUST be from the closed set above
3. cross_modal_triplets: relation_type can be any string (e.g., "visualizes", "shows_results")

## Output Format
```json
{{
  "text_triplets": [
    {{
      "head": "Head entity",
      "head_type": "Entity type",
      "relation": "Natural language description",
      "relation_type": "ADDRESSES|PROPOSES|USES_COMPONENT|EVALUATED_ON|ACHIEVES|COMPARES_WITH|LIMITED_BY|APPLIES_TO|EXTENDS",
      "tail": "Tail entity",
      "tail_type": "Entity type",
      "confidence": 0.95,
      "evidence": "text snippet"
    }}
  ],
  "image_info": {{
    "figure_id": "{{figure_id}}",
    "description": "What is shown in the figure",
    "figure_type": "chart|photo|diagram|graph|table",
    "key_entities": ["Entity1", "Entity2"],
    "relations_shown": ["comparison", "performance", "trend"]
  }},
  "cross_modal_triplets": [
    {{
      "head": "{{figure_id}}",
      "relation": "visualizes or shows",
      "relation_type": "visualizes",
      "tail": "Entity or comparison being shown",
      "tail_type": "Entity type",
      "confidence": 0.9,
      "evidence": "Image shows X"
    }}
  ]
}}
```

## Example
Input:
Text: "Figure 2 shows the performance comparison between BERT and GPT on GLUE benchmark."
Image Caption: "Figure 2: Performance comparison on GLUE"

Output:
```json
{{
  "text_triplets": [
    {{
      "head": "BERT",
      "head_type": "Model",
      "relation": "compares with",
      "relation_type": "COMPARES_WITH",
      "tail": "GPT",
      "tail_type": "Model",
      "confidence": 0.9,
      "evidence": "performance comparison between BERT and GPT"
    }}
  ],
  "image_info": {{
    "figure_id": "Figure 2",
    "description": "Bar chart comparing BERT and GPT performance across 8 GLUE tasks",
    "figure_type": "chart",
    "key_entities": ["BERT", "GPT", "GLUE tasks"],
    "relations_shown": ["performance comparison", "accuracy scores"]
  }},
  "cross_modal_triplets": [
    {{
      "head": "Figure 2",
      "relation": "visualizes",
      "relation_type": "visualizes",
      "tail": "BERT vs GPT comparison",
      "tail_type": "Comparison",
      "confidence": 0.95,
      "evidence": "Figure 2 shows performance comparison"
    }}
  ]
}}
```
"""


# ============================================================================
# 多模态知识图谱构建器
# ============================================================================

class MultimodalGraphBuilder:
    """
    多模态知识图谱构建器

    使用本地 Qwen3.5 GGUF 模型从文本和图片中抽取三元组，构建知识图谱。

    支持：
    1. 纯文本三元组抽取
    2. 多模态联合抽取（图+文）
    3. 图片实体提取
    4. 跨模态关系建立
    """

    def __init__(
        self,
        config: "GraphRAGConfig",
        context: Any = None
    ):
        """
        初始化构建器

        Args:
            config: GraphRAGConfig 配置
            context: AstrBot 上下文
        """
        self.config = config
        self.context = context
        self._llm: Optional[Any] = None  # LlamaCppVLMProvider
        self._llm_config = self._get_llm_config()
        self._triplet_grammar: Optional[Any] = None
        self._multimodal_grammar: Optional[Any] = None

    def _get_llm_config(self) -> LocalLLMConfig:
        """获取 LLM 配置"""
        plugin_dir = _PLUGIN_ROOT

        def resolve_model_path(raw_path: str) -> str:
            path = Path(raw_path).expanduser()
            if path.is_absolute():
                return str(path.resolve())
            return str((plugin_dir / path).resolve())

        # 从配置获取 GGUF 模型路径
        model_path = resolve_model_path(os.environ.get(
            "PAPERRAG_GGUF_MODEL_PATH",
            str(plugin_dir / "models" / "Qwen3.5-9B-GGUF" / "Qwen3.5-9B-UD-Q4_K_XL.gguf")
        ))
        mmproj_path = str(plugin_dir / "models" / "Qwen3.5-9B-GGUF" / "mmproj-BF16.gguf")

        # 检查文件是否存在
        if not Path(model_path).exists():
            fallback = plugin_dir / "models" / "Qwen3.5-4B-GGUF" / Path(model_path).name
            if fallback.exists():
                model_path = str(fallback)

        if not Path(mmproj_path).exists():
            fallback_mmproj = plugin_dir / "models" / "Qwen3.5-4B-GGUF" / Path(mmproj_path).name
            if fallback_mmproj.exists():
                mmproj_path = str(fallback_mmproj)

        return LocalLLMConfig(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=self.config.llm_n_ctx if hasattr(self.config, 'llm_n_ctx') else 16384,
            n_gpu_layers=99,
            max_tokens=self.config.llm_max_tokens if hasattr(self.config, 'llm_max_tokens') else 8192,
            temperature=0.1,
        )

    async def _ensure_llm_initialized(self):
        """确保 LLM 已初始化 - 使用 LlamaCppVLMProvider"""
        if self._llm is None:
            from provider.llama_cpp_vlm import (
                get_cached_llama_cpp_provider,
                init_llama_cpp_vlm_provider,
            )
            # 优先复用已初始化的单例
            cached = get_cached_llama_cpp_provider()
            if cached is not None:
                self._llm = cached
            else:
                self._llm = init_llama_cpp_vlm_provider(
                    model_path=self._llm_config.model_path,
                    mmproj_path=self._llm_config.mmproj_path,
                    n_ctx=self._llm_config.n_ctx,
                    n_gpu_layers=self._llm_config.n_gpu_layers,
                    max_tokens=self._llm_config.max_tokens,
                    temperature=self._llm_config.temperature
                )
        await self._llm.initialize()
        self._load_grammars()

    def _load_grammars(self):
        """从 JSON schema 文件生成 grammar，约束 LLM 输出为合法 JSON"""

        triplet_path = _GRAMMAR_DIR / "triplet_schema.json"
        if triplet_path.exists():
            try:
                schema_text = triplet_path.read_text()
                grammar = LlamaGrammar.from_json_schema(schema_text)
                # 放宽 space 规则：允许任意空白符（不只是单个空格）
                grammar_text = str(grammar._grammar)
                grammar_text = grammar_text.replace('space ::= " "?', 'space ::= [ \\t\\n\\r]*')
                self._triplet_grammar = LlamaGrammar.from_string(grammar_text)
                logger.info(f"[Graph-LLM] 已加载 triplet grammar (space规则已放宽)")
            except Exception as e:
                logger.warning(f"[Graph-LLM] 加载 triplet grammar 失败: {e}")
        else:
            logger.warning(f"[Graph-LLM] triplet schema 文件不存在: {triplet_path}")

        multimodal_path = _GRAMMAR_DIR / "multimodal_schema.json"
        if multimodal_path.exists():
            try:
                schema_text = multimodal_path.read_text()
                grammar = LlamaGrammar.from_json_schema(schema_text)
                grammar_text = str(grammar._grammar)
                grammar_text = grammar_text.replace('space ::= " "?', 'space ::= [ \\t\\n\\r]*')
                self._multimodal_grammar = LlamaGrammar.from_string(grammar_text)
                logger.info(f"[Graph-LLM] 已加载 multimodal grammar (space规则已放宽)")
            except Exception as e:
                logger.warning(f"[Graph-LLM] 加载 multimodal grammar 失败: {e}")
        else:
            logger.warning(f"[Graph-LLM] multimodal schema 文件不存在: {multimodal_path}")

    async def build_from_nodes(
        self,
        nodes: List[Any],
        graph_store: Any
    ) -> Dict[str, int]:
        """
        从 Node 列表构建知识图谱

        Args:
            nodes: Node 列表
            graph_store: 图谱存储

        Returns:
            构建统计
        """
        stats = {
            "entities_added": 0,
            "text_triplets_added": 0,
            "image_entities_added": 0,
            "cross_modal_triplets_added": 0,
            "chunks_processed": 0,
            "chunks_with_images": 0,
            "chunks_failed": 0,
            "chunks_empty": 0
        }

        await self._ensure_llm_initialized()

        # 批量处理：每批 4 个 chunks
        # 安全计算：4 chunks × ~500字符 ≈ 2000字符 ≈ 500-800 tokens
        # 加上 system prompt ≈ 1500 tokens，总计 ≈ 2000-2300 tokens < 4096
        # prompt 中限制最多提取 {max_triplets} 个三元组，确保输出不会过长
        batch_size = 4
        total_batches = (len(nodes) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(nodes))
            batch_nodes = nodes[start_idx:end_idx]

            result = await self._process_batch(batch_nodes, graph_store, batch_idx, total_batches)

            if isinstance(result, Exception):
                stats["chunks_failed"] += len(batch_nodes)
                logger.warning(f"处理批次 {batch_idx + 1}/{total_batches} 失败: {result}")
            elif isinstance(result, dict):
                stats["entities_added"] += result.get("entities_added", 0)
                stats["text_triplets_added"] += result.get("text_triplets_added", 0)
                stats["image_entities_added"] += result.get("image_entities_added", 0)
                stats["cross_modal_triplets_added"] += result.get("cross_modal_triplets_added", 0)
                stats["chunks_with_images"] += result.get("chunks_with_images", 0)
                stats["chunks_processed"] += result.get("chunks_with_triplets", 0)
                stats["chunks_empty"] += result.get("chunks_empty", 0)
                # 批次统计日志
                batch_triplets = result.get("text_triplets_added", 0)
                batch_entities = result.get("entities_added", 0)
                batch_images = result.get("image_entities_added", 0)
                batch_cross = result.get("cross_modal_triplets_added", 0)
                logger.info(
                    f"[Graph-LLM] 批次 {batch_idx + 1}/{total_batches} 完成: "
                    f"实体+{batch_entities}, 文本三元组+{batch_triplets}, "
                    f"图片实体+{batch_images}, 跨模态三元组+{batch_cross}"
                )

        logger.info(
            f"✅ 图谱构建完成: "
            f"实体={stats['entities_added']}, "
            f"文本三元组={stats['text_triplets_added']}, "
            f"图片实体={stats['image_entities_added']}, "
            f"跨模态三元组={stats['cross_modal_triplets_added']}, "
            f"有效块={stats['chunks_processed']}, "
            f"空块={stats['chunks_empty']}"
        )

        return stats

    async def _process_batch(
        self,
        nodes: List[Any],
        graph_store: Any,
        batch_idx: int = 0,
        total_batches: int = 1
    ) -> Dict[str, Any]:
        """
        批量处理多个节点（一批 chunks 作为一次 LLM 调用）

        Args:
            nodes: Node 列表（一批）
            graph_store: 图谱存储
            batch_idx: 当前批次索引
            total_batches: 总批次数量

        Returns:
            批次统计
        """
        result = {
            "entities_added": 0,
            "text_triplets_added": 0,
            "image_entities_added": 0,
            "cross_modal_triplets_added": 0,
            "chunks_with_images": 0,
            "chunks_with_triplets": 0,
            "chunks_empty": 0
        }

        try:
            # 过滤掉太短的文本
            valid_nodes = []
            for node in nodes:
                text = node.text if hasattr(node, 'text') else str(node)
                if text and len(text) >= 50:
                    valid_nodes.append(node)

            if not valid_nodes:
                result["chunks_empty"] = len(nodes)
                return result

            # 检查哪些 chunks 有图片（提前检测，用于排除 prompt 和 VLM 处理）
            nodes_with_images = []
            nodes_with_images_set: set = set()
            for node in valid_nodes:
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                has_images = (
                    metadata.get("has_image", False) and
                    metadata.get("image_path") and
                    Path(metadata.get("image_path", "")).exists()
                )
                if has_images:
                    nodes_with_images.append(node)
                    nodes_with_images_set.add(id(node))

            # 构建批量 prompt（排除图像节点，它们由 VLM 单独处理）
            chunks_text = []
            chunk_label_to_valid_idx: dict[int, int] = {}  # [Chunk N] label → valid_nodes index
            text_only_count = 0
            for i, node in enumerate(valid_nodes):
                if id(node) in nodes_with_images_set and self.config.multimodal_enabled:
                    continue
                text = node.text if hasattr(node, 'text') else str(node)
                text_only_count += 1
                chunks_text.append(f"[Chunk {text_only_count}] {text}")
                chunk_label_to_valid_idx[text_only_count] = i

            if not chunks_text:
                # 全部是图像节点，跳过批处理 LLM 调用
                result["chunks_empty"] = 0
                triplets = []
            else:
                combined_text = "\n\n".join(chunks_text)
                system_prompt = BATCH_TRIPLET_EXTRACTION_PROMPT.format(
                    max_triplets=self.config.max_triplets_per_chunk * text_only_count
                )
                user_prompt = f"Extract triplets from the following text chunks:\n\n{combined_text}\n\nExtract all entity-relationship triplets:"

                # 检查是否超出上下文长度（精确计算）
                system_tokens = count_tokens(system_prompt)
                user_prefix_tokens = count_tokens("Extract triplets from the following text chunks:\n\n")
                user_suffix_tokens = count_tokens("\n\nExtract all entity-relationship triplets:")
                content_tokens = count_tokens(combined_text)
                # n_ctx 覆盖 prompt + 生成，必须为输出预留空间
                max_context = self._llm_config.n_ctx if hasattr(self, '_llm_config') else 4096
                max_output = self._llm_config.max_tokens if hasattr(self, '_llm_config') else 4096
                total_tokens = system_tokens + user_prefix_tokens + content_tokens + user_suffix_tokens + max_output

                if total_tokens > max_context:
                    logger.warning(
                        f"[Graph-LLM] ⚠️ 批次 {batch_idx + 1}/{total_batches} 超出上下文长度: "
                        f"{total_tokens} tokens (含输出预算 {max_output}) > {max_context} tokens，自动拆分"
                    )
                    # 预算 = n_ctx - system - overhead - max_tokens(输出)
                    budget = max_context - max_output - system_tokens - user_prefix_tokens - user_suffix_tokens
                    if budget <= 0:
                        budget = max_context // 4  # 极端情况：至少保留 1/4 给内容

                    # 统计每个 chunk 的 token 数
                    chunk_tokens_list = [count_tokens(c) for c in chunks_text]
                    avg_chunk_tokens = sum(chunk_tokens_list) / len(chunk_tokens_list) if chunk_tokens_list else 200

                    # 安全每组 chunk 数（每个 chunk 留 50% buffer）
                    safe_chunks = max(1, int(budget / (avg_chunk_tokens * 1.5)))
                    num_chunks = len(chunks_text)
                    num_splits = (num_chunks + safe_chunks - 1) // safe_chunks

                    logger.warning(
                        f"[Graph-LLM] 批次 {batch_idx + 1} 拆分为 {num_splits} 组 "
                        f"(共 {num_chunks} chunks，每组约 {safe_chunks} 个)"
                    )

                    if self._triplet_grammar is None:
                        logger.error(
                            "[Graph-LLM] Grammar 未加载，分块路径的约束生成已禁用。"
                            "请检查 triplet_schema.json。"
                        )

                    all_triplets = []
                    for i in range(num_splits):
                        start = i * safe_chunks
                        end = min(start + safe_chunks, num_chunks)
                        chunk_group = chunks_text[start:end]

                        try:
                            chunk_result = await self._call_llm_for_chunks(
                                chunk_group, self.config.max_triplets_per_chunk * len(chunk_group),
                                batch_idx, f"split{i+1}"
                            )
                            chunk_triplets = self._parse_json_response(chunk_result) if chunk_result else []

                            group_text = "\n\n".join(chunk_group)
                            group_tokens = count_tokens(system_prompt) + count_tokens(group_text) + 100
                            logger.info(f"[Graph-LLM] 分组 {i+1}/{num_splits}: {len(chunk_group)} chunks, 约 {group_tokens} tokens, {len(chunk_triplets)} triplets")

                            all_triplets.extend(chunk_triplets)
                        except Exception as group_err:
                            logger.error(
                                f"[Graph-LLM] 分组 {i+1}/{num_splits} 失败（已跳过，继续处理其余分组）: {group_err}"
                            )
                    triplets = all_triplets
                else:
                    # 调用 LLM（使用 GBNF grammar 约束输出）
                    assert self._llm is not None

                    if self._triplet_grammar is None:
                        logger.error(
                            "[Graph-LLM] Grammar 未加载，grammar 约束已禁用。"
                            "输出可能不符合预期格式。请检查 triplet_schema.json。"
                        )

                    response = await self._llm.text_chat(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_tokens=self._llm_config.max_tokens,
                        grammar=self._triplet_grammar,
                    )
                    response_text = response.content if hasattr(response, 'content') else str(response)

                    # 调试：记录完整响应
                    logger.info(f"[Graph-LLM] 批次 {batch_idx + 1} LLM响应长度: {len(response_text)}")

                    # 解析 JSON 响应
                    triplets = self._parse_json_response(response_text)

            # 如果有图片节点，使用多模态处理
            if nodes_with_images:
                result["chunks_with_images"] = len(nodes_with_images)
                if self.config.multimodal_enabled:
                    for node in nodes_with_images:
                        multimodal_result = await self._process_node(node, graph_store)
                        if isinstance(multimodal_result, dict):
                            result["entities_added"] += multimodal_result.get("entities_added", 0)
                            result["text_triplets_added"] += multimodal_result.get("text_triplets_added", 0)
                            result["image_entities_added"] += multimodal_result.get("image_entities_added", 0)
                            result["cross_modal_triplets_added"] += multimodal_result.get("cross_modal_triplets_added", 0)

            # Deterministic Chunk→Media edges from metadata (survives VLM failure)
            for node in valid_nodes:
                meta = node.metadata if hasattr(node, 'metadata') else {}
                if meta.get("has_image") and meta.get("image_path"):
                    graph_store.add_media_link(
                        chunk_id=meta.get("chunk_id", meta.get("file_name", "")),
                        media_path=meta["image_path"],
                        media_type="image",
                        caption=meta.get("image_caption", ""),
                    )

            # 添加文本三元组
            counted_entities: set = set()
            contributing_chunks: set = set()

            triplets = triplets if chunks_text else []
            for triplet in triplets:
                head = triplet.get("head", "").strip()
                relation = triplet.get("relation", "").strip()
                tail = triplet.get("tail", "").strip()

                if not head or not relation or not tail:
                    continue

                # 从 evidence 中提取 chunk 索引，通过 mapping 定位 valid_nodes
                evidence = triplet.get("evidence", "")
                chunk_idx = 0
                for label in range(1, text_only_count + 1):
                    if f"[Chunk {label}]" in evidence:
                        chunk_idx = chunk_label_to_valid_idx.get(label, 0)
                        break

                node = valid_nodes[chunk_idx]

                # Skip image nodes when multimodal enabled — VLM handles them separately
                if node in nodes_with_images and self.config.multimodal_enabled:
                    continue
                contributing_chunks.add(chunk_idx)
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                chunk_id = metadata.get("chunk_id", metadata.get("file_name", ""))

                graph_store.add_entity(
                    name=head,
                    entity_type=self._normalize_entity_type(triplet.get("head_type", "")),
                    chunk_id=chunk_id
                )
                graph_store.add_entity(
                    name=tail,
                    entity_type=self._normalize_entity_type(triplet.get("tail_type", "")),
                    chunk_id=chunk_id
                )

                rel_id = graph_store.add_relation(
                    head=head,
                    tail=tail,
                    relation=self._normalize_relation_type(triplet.get("relation_type", "")),
                    relation_description=relation,
                    weight=triplet.get("confidence", 1.0),
                    chunk_id=chunk_id
                )

                if rel_id:
                    result["text_triplets_added"] += 1
                    if head.lower() not in counted_entities:
                        result["entities_added"] += 1
                        counted_entities.add(head.lower())
                    if tail.lower() not in counted_entities:
                        result["entities_added"] += 1
                        counted_entities.add(tail.lower())

            result["chunks_with_triplets"] = len(contributing_chunks)
            if nodes_with_images and self.config.multimodal_enabled:
                non_image_count = len(valid_nodes) - len(nodes_with_images)
                result["chunks_empty"] = non_image_count - len(contributing_chunks)
            else:
                result["chunks_empty"] = len(valid_nodes) - len(contributing_chunks)

            return result

        except Exception as e:
            logger.error(f"[Graph-LLM] 批次 {batch_idx + 1}/{total_batches} 处理失败: {e}")
            return e

    async def _process_node(
        self,
        node: Any,
        graph_store: Any
    ) -> Optional[Dict[str, Any]]:
        """处理单个节点，失败时返回 Exception"""
        chunk_id = "unknown"
        try:
            text = node.text if hasattr(node, 'text') else str(node)
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            chunk_id = metadata.get("chunk_id", metadata.get("file_name", ""))

            if not text or len(text) < 50:
                return None

            # 检查是否有多模态内容
            has_images = (
                metadata.get("has_image", False) and
                metadata.get("image_path") and
                Path(metadata.get("image_path", "")).exists()
            )
            has_tables = metadata.get("has_table", False) and metadata.get("table_caption")

            # 提取 paper_id（用于图表/表格节点唯一性）
            file_name = metadata.get("file_name", "")
            paper_id = Path(file_name).stem if file_name else ""

            # Deterministic Chunk→Media edge (survives VLM failure)
            if has_images:
                graph_store.add_media_link(
                    chunk_id=chunk_id,
                    media_path=metadata.get("image_path", ""),
                    media_type="image",
                    caption=metadata.get("image_caption", ""),
                )

            if has_images and self.config.multimodal_enabled:
                # 多模态联合抽取
                image_path = metadata.get("image_path", "")
                result = await self._extract_multimodal_triplets(
                    text=text,
                    image_path=image_path,
                    image_caption=metadata.get("image_caption", ""),
                    chunk_id=chunk_id,
                    graph_store=graph_store,
                    paper_id=paper_id
                )
            else:
                # 纯文本抽取
                result = await self._extract_text_triplets(
                    text=text,
                    chunk_id=chunk_id,
                    graph_store=graph_store
                )

            # 处理表格实体
            if has_tables and self.config.extract_image_entities:
                table_caption = metadata.get("table_caption", "")
                table_id = self._extract_table_id(table_caption, text, paper_id)
                graph_store.add_table_entity(
                    table_id=table_id,
                    description=table_caption,
                    chunk_id=chunk_id
                )
                result["image_entities_added"] += 1  # 复用该字段统计表格

            result["has_images"] = 1 if has_images else 0
            result["has_triplets"] = (
                result.get("text_triplets_added", 0) > 0 or
                result.get("image_entities_added", 0) > 0 or
                result.get("cross_modal_triplets_added", 0) > 0
            )
            return result

        except Exception as e:
            logger.error(f"[Graph-LLM] 节点 {chunk_id} 处理失败: {e}")
            return e

    async def _call_llm_for_chunks(
        self,
        chunks_text: List[str],
        max_triplets: int,
        batch_idx: int,
        part_suffix: str
    ) -> str:
        """调用 LLM 处理指定 chunks，返回原始响应文本。调用前验证 token 预算。"""
        combined_text = "\n\n".join(chunks_text)
        system_prompt = BATCH_TRIPLET_EXTRACTION_PROMPT.format(max_triplets=max_triplets)
        user_prompt = f"Extract triplets from the following text chunks:\n\n{combined_text}\n\nExtract all entity-relationship triplets:"

        # 预检查：确保输入 + 输出不超 n_ctx
        input_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)
        max_context = self._llm_config.n_ctx if hasattr(self, '_llm_config') else 4096
        max_output = self._llm_config.max_tokens if hasattr(self, '_llm_config') else 4096
        if input_tokens + max_output > max_context:
            logger.warning(
                f"[Graph-LLM] 分组 {batch_idx + 1}.{part_suffix} token 预算紧张: "
                f"input={input_tokens} + output={max_output} = {input_tokens + max_output} > n_ctx={max_context}"
            )

        assert self._llm is not None
        response = await self._llm.text_chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=self._llm_config.max_tokens,
            grammar=self._triplet_grammar,
        )
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"[Graph-LLM] 批次 {batch_idx + 1}.{part_suffix} LLM响应长度: {len(response_text)}")
        return response_text

    async def _extract_text_triplets(
        self,
        text: str,
        chunk_id: str,
        graph_store: Any
    ) -> Dict[str, Any]:
        """纯文本三元组抽取"""
        result = {
            "entities_added": 0,
            "text_triplets_added": 0,
            "image_entities_added": 0,
            "cross_modal_triplets_added": 0
        }

        try:
            system_prompt = TRIPLET_EXTRACTION_PROMPT.format(max_triplets=self.config.max_triplets_per_chunk)
            user_prompt = f"## Input Text\n\n{text}\n\nExtract all entity-relationship triplets:"

            assert self._llm is not None
            response = await self._llm.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=self._llm_config.max_tokens,
                grammar=self._triplet_grammar,
            )
            response_text = response.content if hasattr(response, 'content') else str(response)
            triplets = self._parse_json_response(response_text)

            # Track entities counted in this extraction to avoid double-counting
            counted_entities: set = set()

            for triplet in triplets:
                head = triplet.get("head", "").strip()
                relation = triplet.get("relation", "").strip()
                tail = triplet.get("tail", "").strip()

                if not head or not relation or not tail:
                    continue

                graph_store.add_entity(
                    name=head,
                    entity_type=self._normalize_entity_type(triplet.get("head_type", "")),
                    chunk_id=chunk_id
                )
                graph_store.add_entity(
                    name=tail,
                    entity_type=self._normalize_entity_type(triplet.get("tail_type", "")),
                    chunk_id=chunk_id
                )

                rel_id = graph_store.add_relation(
                    head=head,
                    tail=tail,
                    relation=self._normalize_relation_type(triplet.get("relation_type", "")),
                    relation_description=relation,
                    weight=triplet.get("confidence", 1.0),
                    chunk_id=chunk_id
                )

                if rel_id:
                    result["text_triplets_added"] += 1
                    if head.lower() not in counted_entities:
                        result["entities_added"] += 1
                        counted_entities.add(head.lower())
                    if tail.lower() not in counted_entities:
                        result["entities_added"] += 1
                        counted_entities.add(tail.lower())

        except Exception as e:
            logger.warning(f"文本三元组抽取失败: {e}")

        return result

    async def _extract_multimodal_triplets(
        self,
        text: str,
        image_path: str,
        image_caption: str,
        chunk_id: str,
        graph_store: Any,
        paper_id: str = ""
    ) -> Dict[str, Any]:
        """
        多模态联合三元组抽取（哈希缓存 + 确定性降级）

        流水线：
        1. 缓存查询 → 命中则跳过 VLM
        2. 有图片路径 → 调用 VLM；无则直接降级
        3. 存储文本三元组（始终执行，不依赖 VLM）
        4. 无跨模态结果或置信度 <0.7 → 确定性降级
        5. 置信度足够 → 存储图片实体 + 跨模态三元组
        """
        result: Dict[str, Any] = {
            "entities_added": 0,
            "text_triplets_added": 0,
            "image_entities_added": 0,
            "cross_modal_triplets_added": 0
        }

        # ── 提取 figure_id ──────────────────────────────────────────────────
        figure_id = self._extract_figure_id(image_caption, text, paper_id)

        # ── 步骤1：哈希缓存查键 ───────────────────────────────────────────
        # 缓存键仅在有图片路径时有意义；跨运行一致性由图片内容哈希保证
        has_image_path = bool(image_path)
        cache_key = _vlm_cache_key(text, image_path) if has_image_path else None
        cached = _VLM_CACHE.get(cache_key) if cache_key else None

        if cached is not None:
            logger.info(f"[Graph-LLM] VLM 缓存命中，跳过调用: {cache_key}")
            data = cached
        elif has_image_path:
            # ── 步骤2：调用 VLM ───────────────────────────────────────────
            data = await self._call_vlm_multimodal(
                text=text,
                image_path=image_path,
                image_caption=image_caption,
                figure_id=figure_id,
            )
            if cache_key and (data.get("image_info") or data.get("cross_modal_triplets")):
                if len(_VLM_CACHE) > 500:
                    keys_to_remove = list(_VLM_CACHE.keys())[:250]
                    for k in keys_to_remove:
                        del _VLM_CACHE[k]
                _VLM_CACHE[cache_key] = data
        else:
            # ── 步骤3：无图片路径 → 确定性降级 ───────────────────────────
            data = {"text_triplets": [], "image_info": {}, "cross_modal_triplets": []}

        # ── 步骤3：存储文本三元组 ────────────────────────────────────────────
        text_triplets = data.get("text_triplets", [])
        counted_entities: set = set()
        for triplet in text_triplets[:self.config.max_triplets_per_chunk]:
            head = triplet.get("head", "").strip()
            relation = triplet.get("relation", "").strip()
            tail = triplet.get("tail", "").strip()
            if not head or not relation or not tail:
                continue
            graph_store.add_entity(
                name=head,
                entity_type=self._normalize_entity_type(triplet.get("head_type", "")),
                chunk_id=chunk_id
            )
            graph_store.add_entity(
                name=tail,
                entity_type=self._normalize_entity_type(triplet.get("tail_type", "")),
                chunk_id=chunk_id
            )
            rel_id = graph_store.add_relation(
                head=head,
                tail=tail,
                relation=self._normalize_relation_type(triplet.get("relation_type", "")),
                relation_description=relation,
                weight=float(triplet.get("confidence", 1.0)),
                chunk_id=chunk_id
            )
            if rel_id:
                result["text_triplets_added"] += 1
                if head.lower() not in counted_entities:
                    result["entities_added"] += 1
                    counted_entities.add(head.lower())
                if tail.lower() not in counted_entities:
                    result["entities_added"] += 1
                    counted_entities.add(tail.lower())

        # ── 步骤5：图片实体 + 跨模态三元组（带置信度阈值降级）──────────────
        image_info = data.get("image_info", {})
        cross_triplets = data.get("cross_modal_triplets", [])

        if not cross_triplets or not image_info:
            # VLM 未返回跨模态结果 → 直接降级
            return self._fallback_cross_modal(
                figure_id=figure_id,
                image_path=image_path,
                chunk_id=chunk_id,
                graph_store=graph_store,
                result=result,
            )

        avg_confidence = sum(
            float(t.get("confidence", 0)) for t in cross_triplets
        ) / max(len(cross_triplets), 1)

        if avg_confidence < 0.7:
            logger.info(f"[Graph-LLM] VLM 置信度 {avg_confidence:.2f} < 0.7，使用确定性降级边")
            return self._fallback_cross_modal(
                figure_id=figure_id,
                image_path=image_path,
                chunk_id=chunk_id,
                graph_store=graph_store,
                result=result,
            )

        # 置信度足够，存储图片实体 + 跨模态三元组
        if self.config.extract_image_entities:
            graph_store.add_image_entity(
                figure_id=figure_id,
                image_path=image_path,
                description=image_info.get("description", ""),
                figure_type=image_info.get("figure_type", "unknown"),
                chunk_id=chunk_id
            )
            result["image_entities_added"] += 1
            result["entities_added"] += 1

        for triplet in cross_triplets:
            head = figure_id
            relation = triplet.get("relation", "").strip()
            tail = triplet.get("tail", "").strip()
            if not relation or not tail:
                continue
            # Ensure tail entity exists in Neo4j (add_entity uses MERGE, idempotent)
            tail_type = triplet.get("tail_type", "Application")
            if tail.lower() not in graph_store:
                graph_store.add_entity(
                    name=tail,
                    entity_type=self._normalize_entity_type(tail_type),
                    chunk_id=chunk_id,
                )
            rel_id = graph_store.add_relation(
                head=head,
                tail=tail,
                relation=relation,
                weight=float(triplet.get("confidence", 0.9)),
                chunk_id=chunk_id
            )
            if rel_id:
                result["cross_modal_triplets_added"] += 1

        return result

    async def _call_vlm_multimodal(
        self,
        text: str,
        image_path: str,
        image_caption: str,
        figure_id: str,
    ) -> Dict[str, Any]:
        """调用 VLM 获取多模态三元组"""
        system_prompt = MULTIMODAL_TRIPLET_EXTRACTION_PROMPT.format(
            text=text,
            image_caption=image_caption or "No caption",
            figure_id=figure_id
        )
        user_prompt = f"""Analyze the image and extract cross-modal knowledge graph triplets.
Image Caption: {image_caption or 'No caption'}

Extract triplets:"""

        assert self._llm is not None
        response = await self._llm.text_chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
            image_urls=[image_path] if self.config.multimodal_enabled else None,
            max_tokens=self._llm_config.max_tokens,
            grammar=self._multimodal_grammar,
        )
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"[Graph-LLM] VLM响应长度: {len(response_text)}, 响应: {repr(response_text)}")
        return self._parse_multimodal_response(response_text)

    def _fallback_cross_modal(
        self,
        figure_id: str,
        image_path: str,
        chunk_id: str,
        graph_store: Any,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """确定性降级：仅创建 figure 实体节点。Chunk→Media 由 add_media_link() 保证。"""
        ext = Path(image_path).suffix.lower()
        figure_type = _EXT_TO_FIGURE_TYPE.get(ext, "unknown")
        if self.config.extract_image_entities:
            graph_store.add_image_entity(
                figure_id=figure_id,
                image_path=image_path,
                description="",
                figure_type=figure_type,
                chunk_id=chunk_id
            )
            result["image_entities_added"] += 1
            result["entities_added"] += 1

        return result

    def _stable_caption_hash(self, caption: str, paper_id: str = "", chunk_id: str = "", image_path: str = "") -> str:
        """基于稳定输入生成确定性哈希（跨运行一致）

        Args:
            caption: 图片 caption 文本
            paper_id: 论文标识
            chunk_id: chunk 标识
            image_path: 图片路径

        Returns:
            12 位十六进制哈希字符串
        """
        # 使用多个稳定字段组合，确保唯一性且跨运行一致
        components = [
            paper_id or "",
            chunk_id or "",
            image_path or "",
            caption.strip() if caption else "",
        ]
        combined = "|".join(components).encode('utf-8')
        # SHA256 是确定性的，输出前 12 位
        return hashlib.sha256(combined).hexdigest()[:12]

    def _extract_figure_id(self, caption: str, text: str, paper_id: str = "") -> str:
        """从图注或文本中提取 figure ID，并确保唯一性（跨论文不重名、跨运行一致）

        Args:
            caption: 图片的 caption 文本
            text: 关联的文本块
            paper_id: 论文标识（用于生成唯一 ID）

        Returns:
            唯一的 figure 标识符，格式: paper_id_Figure_N 或 paper_id_Table_N
        """
        # 从 caption 提取（caption 更准确，优先使用）
        patterns = [
            r'(Figure|Fig\.?|图)\s*(\d+[a-zA-Z]?)',
            r'(Table|表格)\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, caption or text, re.IGNORECASE)
            if match:
                prefix = match.group(1)
                num = match.group(2)
                if prefix.lower() in ["figure", "fig", "fig.", "图"]:
                    base = f"Figure {num}"
                elif prefix.lower() in ["table", "表格"]:
                    base = f"Table {num}"
                else:
                    base = "Unknown"

                # 使用 paper_id 确保唯一性
                if paper_id:
                    return f"{paper_id}_{base}"
                return base

        # 无法提取时，使用确定性哈希基于 caption 内容生成唯一标识
        if caption and len(caption.strip()) > 3:
            stable_hash = self._stable_caption_hash(caption, paper_id, "", "")
            if paper_id:
                return f"{paper_id}_Caption_{stable_hash}"
            return f"Caption_{stable_hash}"

        # 极端 fallback（caption 也为空）：使用确定性哈希
        stable_hash = self._stable_caption_hash("", paper_id, "", "")
        fallback_id = f"Fallback_{stable_hash}"
        if paper_id:
            return f"{paper_id}_{fallback_id}"
        return fallback_id

    def _extract_table_id(self, caption: str, text: str, paper_id: str = "") -> str:
        """从表注或文本中提取 table ID，并确保唯一性（跨论文不重名、跨运行一致）

        Args:
            caption: 表格的 caption 文本
            text: 关联的文本块
            paper_id: 论文标识（用于生成唯一 ID）

        Returns:
            唯一的 table 标识符，格式: paper_id_Table_N
        """
        # 从 caption 提取（caption 更准确，优先使用）
        patterns = [
            r'(Table|表格)\s*(\d+[a-zA-Z]?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, caption or text, re.IGNORECASE)
            if match:
                num = match.group(2)
                base = f"Table {num}"
                if paper_id:
                    return f"{paper_id}_{base}"
                return base

        # 无法提取时，使用确定性哈希基于 caption 内容生成唯一标识
        if caption and len(caption.strip()) > 3:
            stable_hash = self._stable_caption_hash(caption, paper_id, "", "")
            if paper_id:
                return f"{paper_id}_Table_{stable_hash}"
            return f"Table_{stable_hash}"

        # 极端 fallback（caption 也为空）：使用确定性哈希
        stable_hash = self._stable_caption_hash("", paper_id, "", "")
        fallback_id = f"Table_{stable_hash}"
        if paper_id:
            return f"{paper_id}_{fallback_id}"
        return fallback_id

    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to one of the 9 closed-set content-oriented types."""
        normalized = entity_type.strip()
        if normalized in CLOSED_ENTITY_TYPES:
            return normalized
        for ct in CLOSED_ENTITY_TYPES:
            if normalized.lower() == ct.lower():
                return ct
        aliased = ENTITY_TYPE_ALIASES.get(normalized.lower().strip())
        if aliased:
            return aliased
        logger.warning(f"[Graph-LLM] Unrecognized entity type '{entity_type}', defaulting to 'Method'")
        return "Method"

    def _normalize_relation_type(self, relation_type: str) -> str:
        """Normalize relation_type to one of the 9 closed-set predicates."""
        normalized = relation_type.strip().upper()
        if normalized in CLOSED_RELATION_TYPES:
            return normalized
        aliased = RELATION_ALIASES.get(relation_type.strip().lower())
        if aliased:
            return aliased
        logger.warning(f"[Graph-LLM] Unknown relation_type '{relation_type}', defaulting to 'USES_COMPONENT'")
        return "USES_COMPONENT"



    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """解析 JSON 响应。Grammar 已约束输出，理论上应该合法。"""
        if not response:
            logger.warning("[Graph-LLM] JSON 解析失败: 响应为空")
            return []

        json_str = response.strip().lstrip('﻿')
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(json_str)
            return self._extract_triplets(data)
        except json.JSONDecodeError as e:
            err_pos = e.pos if hasattr(e, 'pos') else 0

            # 尝试截断恢复：从后向前找最后一个完整的 JSON 块
            truncated_data = self._try_truncate_recover(json_str)
            if truncated_data is not None:
                logger.warning(f"[Graph-LLM] JSON 截断恢复成功，提取 {len(truncated_data.get('triplets', []))} 个三元组")
                return self._extract_triplets(truncated_data)

            # 恢复失败，记录完整响应用于调试
            logger.error(
                f"[Graph-LLM] JSON 解析失败: {e}，"
                f"错误位置: {err_pos}，响应长度: {len(json_str)}，"
                f"末尾内容: {repr(json_str[-200:] if len(json_str) > 200 else json_str)}"
            )
            # 记录原始响应到文件（用于调试）
            self._save_failed_response(response, err_pos)
            return []

    def _try_truncate_recover(self, json_str: str) -> Optional[Dict]:
        """尝试恢复被截断的 JSON。

        策略：找到最后一个完整的三元组对象（以 }, 结尾），
        然后关闭 JSON 数组和对象，重新解析。
        """
        if not json_str:
            return None

        # 策略1：找到最后一个 }, 并关闭 JSON
        # 从后向前搜索 },（完整三元组对象的结束标志）
        pos = len(json_str)
        while True:
            pos = json_str.rfind('},', 0, pos)
            if pos == -1:
                break
            # 尝试在这里关闭 JSON：}, → }]}
            candidate = json_str[:pos + 1] + ']}'
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and 'triplets' in data:
                    return data
            except json.JSONDecodeError:
                pass

        # 策略2：单三元组场景 — 找到 "evidence" 字段结束的 } 并关闭
        last_brace = json_str.rfind('}')
        if last_brace > 0:
            candidate = json_str[:last_brace + 1] + ']}'
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and 'triplets' in data:
                    return data
            except json.JSONDecodeError:
                pass

        # 策略3：原有的 ] 搜索（处理已有关闭括号但嵌套错误的情况）
        last_bracket = json_str.rfind(']')
        if last_bracket > 0:
            for i in range(last_bracket, 0, -1):
                try:
                    candidate = json_str[:i + 1]
                    if candidate.rstrip().endswith(']'):
                        data = json.loads(candidate)
                        if isinstance(data, dict) and 'triplets' in data:
                            return data
                except json.JSONDecodeError:
                    continue

        return None

    def _save_failed_response(self, response: str, err_pos: int):
        """保存解析失败的响应到文件用于调试"""

        debug_dir = _PLUGIN_ROOT / "data" / "debug"
        debug_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        filename = debug_dir / f"failed_response_{timestamp}.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Error position: {err_pos}\n")
                f.write(f"# Total length: {len(response)}\n")
                f.write(f"# Content around error (pos {err_pos - 50}:{err_pos + 100}):\n")
                f.write(repr(response[max(0, err_pos - 50):err_pos + 100]) + "\n\n")
                f.write("# Full response:\n")
                f.write(response)
            logger.info(f"[Graph-LLM] 失败响应已保存: {filename}")
        except Exception as ex:
            logger.warning(f"[Graph-LLM] 保存失败响应失败: {ex}")

    def _extract_triplets(self, data) -> List[Dict[str, Any]]:
        """从解析后的数据中提取三元组"""
        if isinstance(data, dict):
            triplets = data.get("triplets", [])
        elif isinstance(data, list):
            triplets = data
        else:
            return []

        if not isinstance(triplets, list):
            return []

        result = []
        for item in triplets:
            if not isinstance(item, dict):
                continue
            head = item.get("head")
            relation = item.get("relation")
            tail = item.get("tail")
            if head and relation and tail:
                result.append({
                    "head": str(head),
                    "head_type": item.get("head_type", ""),
                    "relation": str(relation),
                    "relation_type": item.get("relation_type", ""),
                    "tail": str(tail),
                    "tail_type": item.get("tail_type", ""),
                    "confidence": float(item.get("confidence", 0.5)),
                    "evidence": item.get("evidence", "")
                })

        result.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return result

    def _parse_multimodal_response(self, response: str) -> Dict[str, Any]:
        """解析多模态响应"""
        if not response or not isinstance(response, str):
            return {"text_triplets": [], "image_info": {}, "cross_modal_triplets": []}

        json_str = response.strip().lstrip('\ufeff')

        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"[Graph-LLM] 多模态 JSON 解析失败: {e}")
            # 截断恢复：从后向前找到最后一个完整的 }
            last_brace = json_str.rfind('}')
            if last_brace > 0:
                try:
                    truncated = json.loads(json_str[:last_brace + 1])
                    logger.info(f"[Graph-LLM] 多模态截断恢复成功")
                    return truncated
                except json.JSONDecodeError:
                    pass
            return {"text_triplets": [], "image_info": {}, "cross_modal_triplets": []}


# ============================================================================
# 便捷函数
# ============================================================================

async def build_graph_from_documents(
    documents: List[str],
    graph_store: Any,
    config: Any,
    context: Any = None
) -> Dict[str, int]:
    """
    便捷函数：从文档列表构建图谱

    Args:
        documents: 文档文本列表
        graph_store: 图谱存储
        config: GraphRAGConfig 配置
        context: AstrBot 上下文

    Returns:
        构建统计
    """
    class SimpleNode:
        def __init__(self, text: str, metadata: Dict[str, Any]):
            self.text = text
            self.metadata = metadata

    nodes = [
        SimpleNode(doc, {"chunk_id": f"doc_{i}"})
        for i, doc in enumerate(documents)
    ]

    builder = MultimodalGraphBuilder(config=config, context=context)
    return await builder.build_from_nodes(nodes, graph_store)
