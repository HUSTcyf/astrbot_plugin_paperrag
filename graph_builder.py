"""
Graph Builder - 多模态知识图谱构建器

使用本地 LLM (Qwen3.5 GGUF) 从文档中抽取三元组，构建知识图谱。

支持：
1. 纯文本三元组抽取
2. 多模态（图+文）联合三元组抽取
3. 图片实体提取
4. 跨模态关系建立

多模态知识图谱实体类型：
- Model/Architecture: BERT, GPT, Transformer
- Method/Technique: Attention,Pooling
- Task: 文本分类、翻译
- Dataset: GLUE, ImageNet
- Metric: Accuracy, F1
- Figure: 图片/图表实体
- Table: 表格实体
"""

import json
import asyncio
import os
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field

from astrbot.api import logger

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from .graph_rag_engine import MemoryGraphStore, Neo4jGraphStore, GraphRAGConfig


# ============================================================================
# 配置
# ============================================================================

@dataclass
class LocalLLMConfig:
    """本地 LLM 配置"""
    model_path: str = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    mmproj_path: str = "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = 99
    max_tokens: int = 1024
    temperature: float = 0.1


@dataclass
class MultimodalLLMConfig(LocalLLMConfig):
    """多模态 LLM 配置"""
    vision_enabled: bool = True


# ============================================================================
# 本地文本/多模态 LLM Provider
# ============================================================================

class LocalLLMProvider:
    """
    本地推理 Provider（使用 llama-cpp-python + GGUF 模型）

    支持：
    1. 纯文本推理（text only）
    2. 多模态推理（text + image）需要 mmproj
    """

    _instance: Optional["LocalLLMProvider"] = None
    _llama: Optional[Any] = None
    _lock: Optional[asyncio.Lock] = None
    _initialized: bool = False
    _vision_available: bool = False

    def __init__(self, config: LocalLLMConfig):
        self.config = config
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls, config: Optional[LocalLLMConfig] = None) -> "LocalLLMProvider":
        """获取单例实例"""
        if cls._instance is None:
            if config is None:
                config = LocalLLMConfig()
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（用于测试或重新初始化）"""
        cls._instance = None
        cls._llama = None
        cls._initialized = False
        cls._vision_available = False

    async def initialize(self) -> None:
        """初始化模型"""
        if self._initialized:
            return

        assert self._lock is not None, "Lock not initialized"
        async with self._lock:
            if self._initialized:
                return

            logger.info("[Graph-LLM] 正在初始化本地推理模型...")
            logger.info(f"[Graph-LLM] 模型路径: {self.config.model_path}")
            logger.info(f"[Graph-LLM] mmproj路径: {self.config.mmproj_path}")

            # 检查模型文件
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                fallback = model_path.parent.parent / "Qwen3.5-4B-GGUF" / model_path.name
                if fallback.exists():
                    logger.info(f"[Graph-LLM] 降级到 4B 模型: {fallback}")
                    self.config.model_path = str(fallback)

            # 检查 mmproj
            mmproj_path = Path(self.config.mmproj_path)
            if not mmproj_path.exists():
                fallback_mmproj = mmproj_path.parent.parent / "Qwen3.5-4B-GGUF" / mmproj_path.name
                if fallback_mmproj.exists():
                    logger.info(f"[Graph-LLM] mmproj 降级到 4B: {fallback_mmproj}")
                    self.config.mmproj_path = str(fallback_mmproj)

            try:
                import concurrent.futures
                from llama_cpp import Llama

                def _load_llama():
                    # 如果 mmproj 存在，启用 vision
                    if Path(self.config.mmproj_path).exists():
                        self._vision_available = True
                        logger.info("[Graph-LLM] Vision 模式可用（检测到 mmproj）")
                        return Llama(
                            model_path=self.config.model_path,
                            mmproj=self.config.mmproj_path,
                            n_ctx=self.config.n_ctx,
                            n_gpu_layers=self.config.n_gpu_layers,
                            n_batch=32,
                            verbose=False,
                        )
                    else:
                        logger.info("[Graph-LLM] 纯文本模式（未检测到 mmproj）")
                        self._vision_available = False
                        return Llama(
                            model_path=self.config.model_path,
                            n_ctx=self.config.n_ctx,
                            n_gpu_layers=self.config.n_gpu_layers,
                            n_batch=32,
                            verbose=False,
                        )

                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    self._llama = await loop.run_in_executor(executor, _load_llama)

                self._initialized = True
                logger.info("[Graph-LLM] ✅ 本地推理模型初始化完成")

            except Exception as e:
                logger.error(f"[Graph-LLM] ❌ 模型初始化失败: {e}")
                raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None
    ) -> str:
        """
        聊天接口

        Args:
            messages: [{"role": "system"/"user", "content": "..."}]
            images: 图片路径列表（可选，用于多模态）

        Returns:
            LLM 输出文本
        """
        if not self._initialized:
            await self.initialize()

        import concurrent.futures

        def _do_completion():
            try:
                prompt = self._build_prompt(messages)
                kwargs = {
                    "max_tokens": max(self.config.max_tokens, 4096),
                    "temperature": self.config.temperature,
                    "stop": ["<|im_end|>", "```"],  # 注意: 不添加 </think> 停止token，因为 JSON 在 thinking 之后
                }

                # 多模态模式
                if images and self._vision_available:
                    content = [
                        {"type": "image_url", "image_url": {"url": str(Path(img).resolve())}}
                        for img in images if Path(img).exists()
                    ]
                    content.append({"type": "text", "text": prompt})
                    kwargs["messages"] = [
                        {"role": "user", "content": content}
                    ]
                    logger.info(f"[Graph-LLM] 多模态调用: images={len(images) if images else 0}, prompt长度={len(prompt)}")
                else:
                    kwargs["prompt"] = prompt
                    logger.info(f"[Graph-LLM] 纯文本调用: prompt长度={len(prompt)}")

                assert self._llama is not None
                if self._vision_available and "messages" in kwargs:
                    result = self._llama.create_chat_completion(**kwargs)
                    # create_chat_completion 返回格式: choices[0].message.content
                    logger.info(f"[Graph-LLM] create_chat_completion result keys: {result.keys() if isinstance(result, dict) else type(result)}")
                    logger.info(f"[Graph-LLM] choices: {result.get('choices') if isinstance(result, dict) else 'N/A'}")
                    raw_content = result["choices"][0]["message"]["content"]
                    logger.info(f"[Graph-LLM] 原始响应类型: {type(raw_content)}, 长度: {len(str(raw_content)) if raw_content else 0}, 内容前500: {str(raw_content)[:500]}")
                    return raw_content.strip() if raw_content else ""
                else:
                    result = self._llama(**kwargs)
                    # __call__ 返回格式: choices[0].text
                    logger.info(f"[Graph-LLM] __call__ result keys: {result.keys() if isinstance(result, dict) else type(result)}")
                    logger.info(f"[Graph-LLM] choices: {result.get('choices') if isinstance(result, dict) else 'N/A'}")
                    raw_text = result["choices"][0]["text"]
                    logger.info(f"[Graph-LLM] 原始响应: {str(raw_text)[:500]}")
                    return raw_text.strip() if raw_text else ""
            except Exception as e:
                logger.error(f"[Graph-LLM] LLM调用失败: {e}")
                import traceback
                logger.error(f"[Graph-LLM] 详细错误: {traceback.format_exc()}")
                return ""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _do_completion)
        return result

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建 Qwen3.5 格式的 prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)


# ============================================================================
# Prompt 模板（全英文）
# ============================================================================

BATCH_TRIPLET_EXTRACTION_PROMPT = """You are an academic knowledge graph construction assistant. Extract structured entity-relationship triplets from multiple academic paper chunks.

## Your Task
Extract ALL meaningful relationship triplets from the given paper text chunks. Each chunk is labeled with [Chunk X].

## Output Format
```json
{{
  "triplets": [
    {{
      "head": "Head entity name (concise, max 30 chars)",
      "head_type": "Entity type",
      "relation": "Relation description (English)",
      "relation_type": "Relation type keyword",
      "tail": "Tail entity name (concise, max 30 chars)",
      "tail_type": "Entity type",
      "confidence": 0.95,
      "evidence": "Original text snippet from [Chunk X]"
    }}
  ],
  "entities": [
    {{
      "name": "Entity name",
      "type": "Entity type",
      "aliases": []
    }}
  ]
}}
```

## Entity Types
- Model/Architecture: BERT, GPT, Transformer, ResNet, network names
- Method/Technique: attention, pooling, optimization, training methods
- Task: text classification, translation, QA, generation
- Dataset: GLUE, ImageNet, COCO, benchmark names
- Metric: accuracy, F1, BLEU, precision, recall, loss
- Optimizer/Algorithm: Adam, SGD, learning rate
- Framework/Library: PyTorch, TensorFlow
- Author/Organization: researcher names, labs
- Venue: NeurIPS, ACL, ICML, conference names
- Other: anything not matching above

## Relation Types (English keywords)
- based_on, uses, achieves, outperforms, improves
- proposes, introduces, achieves, performs
- trained_on, applied_to, compares_with
- combines_with, integrates, depends_on
- publishes_in, collaborates_with

## Extraction Rules
1. Extract ALL meaningful relations from ALL chunks
2. Entity names MUST come from the original text
3. Use English relation descriptions
4. Confidence: 0.5-1.0 based on text clarity
5. Max {max_triplets} triplets per chunk
6. Include [Chunk X] in evidence to indicate source

## Example
Chunks:
[Chunk 1] BERT is based on the Transformer encoder architecture.
[Chunk 2] BERT achieves 86.4% accuracy on GLUE benchmark.

Output:
```json
{{
  "triplets": [
    {{
      "head": "BERT",
      "head_type": "Model/Architecture",
      "relation": "based on",
      "relation_type": "based_on",
      "tail": "Transformer",
      "tail_type": "Model/Architecture",
      "confidence": 0.98,
      "evidence": "BERT is based on the Transformer encoder [Chunk 1]"
    }},
    {{
      "head": "BERT",
      "head_type": "Model/Architecture",
      "relation": "achieves",
      "relation_type": "achieves",
      "tail": "GLUE benchmark",
      "tail_type": "Dataset",
      "confidence": 0.95,
      "evidence": "achieves 86.4% accuracy on GLUE benchmark [Chunk 2]"
    }}
  ]
}}
```
"""

TRIPLET_EXTRACTION_PROMPT = """You are an academic knowledge graph construction assistant. Extract structured entity-relationship triplets from academic papers.

## Your Task
Extract ALL meaningful relationship triplets from the given paper text.

## Output Format
```json
{{
  "triplets": [
    {{
      "head": "Head entity name (concise, max 30 chars)",
      "head_type": "Entity type",
      "relation": "Relation description (English)",
      "relation_type": "Relation type keyword",
      "tail": "Tail entity name (concise, max 30 chars)",
      "tail_type": "Entity type",
      "confidence": 0.95,
      "evidence": "Original text snippet"
    }}
  ],
  "entities": [
    {{
      "name": "Entity name",
      "type": "Entity type",
      "aliases": []
    }}
  ]
}}
```

## Entity Types
- Model/Architecture: BERT, GPT, Transformer, ResNet, network names
- Method/Technique: attention, pooling, optimization, training methods
- Task: text classification, translation, QA, generation
- Dataset: GLUE, ImageNet, COCO, benchmark names
- Metric: accuracy, F1, BLEU, precision, recall, loss
- Optimizer/Algorithm: Adam, SGD, learning rate
- Framework/Library: PyTorch, TensorFlow
- Author/Organization: researcher names, labs
- Venue: NeurIPS, ACL, ICML, conference names
- Other: anything not matching above

## Relation Types (English keywords)
- based_on, uses, achieves, outperforms, improves
- proposes, introduces, achieves, performs
- trained_on, applied_to, compares_with
- combines_with, integrates, depends_on
- publishes_in, collaborates_with

## Extraction Rules
1. Extract ALL meaningful relations from the text
2. Entity names MUST come from the original text
3. Use English relation descriptions
4. Confidence: 0.5-1.0 based on text clarity
5. Max {max_triplets} triplets per chunk

## Example
Input: "BERT is based on the Transformer encoder architecture and achieves 86.4% accuracy on GLUE benchmark, outperforming all previous models."

Output:
```json
{{
  "triplets": [
    {{
      "head": "BERT",
      "head_type": "Model/Architecture",
      "relation": "based on",
      "relation_type": "based_on",
      "tail": "Transformer",
      "tail_type": "Model/Architecture",
      "confidence": 0.98,
      "evidence": "BERT is based on the Transformer encoder"
    }},
    {{
      "head": "BERT",
      "head_type": "Model/Architecture",
      "relation": "achieves",
      "relation_type": "achieves",
      "tail": "GLUE benchmark",
      "tail_type": "Dataset",
      "confidence": 0.95,
      "evidence": "achieves 86.4% accuracy on GLUE benchmark"
    }},
    {{
      "head": "BERT",
      "head_type": "Model/Architecture",
      "relation": "outperforms",
      "relation_type": "outperforms",
      "tail": "previous models",
      "tail_type": "Model/Architecture",
      "confidence": 0.85,
      "evidence": "outperforming all previous models"
    }}
  ],
  "entities": [
    {{"name": "BERT", "type": "Model/Architecture", "aliases": []}},
    {{"name": "Transformer", "type": "Model/Architecture", "aliases": ["Transformer encoder"]}},
    {{"name": "GLUE benchmark", "type": "Dataset", "aliases": ["GLUE"]}}
  ]
}}
```
"""


MULTIMODAL_TRIPLET_EXTRACTION_PROMPT = """You are a multimodal knowledge graph construction assistant. Extract entity-relationship triplets from academic papers with images.

## Your Task
1. Extract triplets from the TEXT
2. Analyze the IMAGE and extract figure information
3. Establish CROSS-MODAL relations between text entities and figure

## Input
Text: {text}
Image Caption: {image_caption}
Image: (provided as image input)

## Output Format
```json
{{
  "text_triplets": [
    {{
      "head": "Head entity",
      "head_type": "Entity type",
      "relation": "Relation",
      "relation_type": "keyword",
      "tail": "Tail entity",
      "tail_type": "Entity type",
      "confidence": 0.95,
      "evidence": "text snippet"
    }}
  ],
  "image_info": {{
    "figure_id": "{figure_id}",
    "description": "What's in the figure (e.g., bar chart comparing A and B on metric X)",
    "figure_type": "chart|photo|diagram|graph|table",
    "key_entities": ["Entity1", "Entity2"],
    "relations_shown": ["comparison", "performance", "trend"]
  }},
  "cross_modal_triplets": [
    {{
      "head": "{figure_id}",
      "relation": "visualizes",
      "relation_type": "visualizes",
      "tail": "Entity or comparison being shown",
      "tail_type": "Entity type",
      "confidence": 0.9,
      "evidence": "Image shows X"
    }}
  ]
}}
```

## Image Figure Types
- chart: bar chart, line chart, pie chart (折线图、柱状图、饼图)
- photo: photograph, microscopic image (照片、显微图)
- diagram: architecture diagram, flowchart (架构图、流程图)
- graph: network graph, knowledge graph (网络图、知识图谱)
- table: data table (数据表格)

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
      "head_type": "Model/Architecture",
      "relation": "compares with",
      "relation_type": "compares_with",
      "tail": "GPT",
      "tail_type": "Model/Architecture",
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
    }},
    {{
      "head": "Figure 2",
      "relation": "shows_results",
      "relation_type": "shows_results",
      "tail": "BERT",
      "tail_type": "Model/Architecture",
      "confidence": 0.95,
      "evidence": "Bar chart comparing BERT and GPT"
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

    def _get_llm_config(self) -> MultimodalLLMConfig:
        """获取 LLM 配置"""
        plugin_dir = Path(__file__).parent.resolve()

        # 从配置获取 GGUF 模型路径
        model_path = os.environ.get(
            "PAPERRAG_GGUF_MODEL_PATH",
            str(plugin_dir / "models" / "Qwen3.5-9B-GGUF" / "Qwen3.5-9B-UD-Q4_K_XL.gguf")
        )
        mmproj_path = str(plugin_dir / "models" / "Qwen3.5-9B-GGUF" / "mmproj-BF16.gguf")

        # 检查文件是否存在
        if not Path(model_path).exists():
            fallback = Path(model_path).parent.parent / "Qwen3.5-4B-GGUF" / Path(model_path).name
            if fallback.exists():
                model_path = str(fallback)

        if not Path(mmproj_path).exists():
            fallback_mmproj = Path(mmproj_path).parent.parent / "Qwen3.5-4B-GGUF" / Path(mmproj_path).name
            if fallback_mmproj.exists():
                mmproj_path = str(fallback_mmproj)

        return MultimodalLLMConfig(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=4096,
            n_gpu_layers=99,
            max_tokens=32768,  # 无限制输出（32k，远超 n_ctx=4096）
            temperature=0.1,
            vision_enabled=self.config.multimodal_enabled
        )

    async def _ensure_llm_initialized(self):
        """确保 LLM 已初始化 - 使用 LlamaCppVLMProvider"""
        if self._llm is None:
            from .llama_cpp_vlm_provider import get_llama_cpp_vlm_provider
            self._llm = get_llama_cpp_vlm_provider(
                model_path=self._llm_config.model_path,
                mmproj_path=self._llm_config.mmproj_path,
                n_ctx=self._llm_config.n_ctx,
                n_gpu_layers=self._llm_config.n_gpu_layers,
                max_tokens=self._llm_config.max_tokens,
                temperature=self._llm_config.temperature
            )
        await self._llm.initialize()

    async def build_from_nodes(
        self,
        nodes: List[Any],
        graph_store: "Union[MemoryGraphStore, Neo4jGraphStore]"
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
        graph_store: "Union[MemoryGraphStore, Neo4jGraphStore]",
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

            # 构建批量 prompt
            chunks_text = []
            total_chars = 0
            for i, node in enumerate(valid_nodes):
                text = node.text if hasattr(node, 'text') else str(node)
                total_chars += len(text)
                chunks_text.append(f"[Chunk {i + 1}] {text}")

            combined_text = "\n\n".join(chunks_text)
            system_prompt = BATCH_TRIPLET_EXTRACTION_PROMPT.format(
                max_triplets=self.config.max_triplets_per_chunk * len(valid_nodes)
            )
            user_prompt = f"Extract triplets from the following text chunks:\n\n{combined_text}\n\nExtract all entity-relationship triplets:"

            # 检查是否超出上下文长度（粗略估算：1 token ≈ 4 字符）
            total_tokens = (len(system_prompt) + len(user_prompt)) // 4
            max_context = self._llm_config.n_ctx if hasattr(self, '_llm_config') else 4096
            if total_tokens > max_context:
                logger.warning(
                    f"[Graph-LLM] ⚠️ 批次 {batch_idx + 1}/{total_batches} 超出上下文长度: "
                    f"预估 {total_tokens} tokens > {max_context} tokens "
                    f"(chars: system={len(system_prompt)}, user={len(user_prompt)})"
                )

            # 调用 LLM
            assert self._llm is not None
            response = await self._llm.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 解析 JSON 响应
            triplets = self._parse_json_response(response_text)

            # 检查哪些 chunks 有图片
            nodes_with_images = []
            for node in valid_nodes:
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                has_images = (
                    metadata.get("has_image", False) and
                    metadata.get("image_path") and
                    Path(metadata.get("image_path", "")).exists()
                )
                if has_images:
                    nodes_with_images.append(node)

            # 如果有图片节点，使用多模态处理
            if nodes_with_images and self.config.multimodal_enabled:
                result["chunks_with_images"] = len(nodes_with_images)
                # 多模态处理：分别处理每个有图片的节点
                for node in nodes_with_images:
                    multimodal_result = await self._process_node(node, graph_store)
                    if isinstance(multimodal_result, dict):
                        result["image_entities_added"] += multimodal_result.get("image_entities_added", 0)
                        result["cross_modal_triplets_added"] += multimodal_result.get("cross_modal_triplets_added", 0)

            # 添加文本三元组
            for triplet in triplets:
                head = triplet.get("head", "").strip()
                relation = triplet.get("relation", "").strip()
                tail = triplet.get("tail", "").strip()

                if not head or not relation or tail:
                    continue

                # 从 evidence 中提取 chunk 索引
                evidence = triplet.get("evidence", "")
                chunk_idx = 0
                for i in range(1, len(valid_nodes) + 1):
                    if f"[Chunk {i}]" in evidence:
                        chunk_idx = i - 1
                        break

                node = valid_nodes[chunk_idx] if chunk_idx < len(valid_nodes) else valid_nodes[0]
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                chunk_id = metadata.get("chunk_id", metadata.get("file_name", ""))

                # 添加实体
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

                # 添加关系
                rel_id = graph_store.add_relation(
                    head=head,
                    tail=tail,
                    relation=relation,
                    weight=triplet.get("confidence", 1.0),
                    chunk_id=chunk_id
                )

                if rel_id:
                    result["text_triplets_added"] += 1
                    result["entities_added"] += 2
                result["text_triplets_added"] += 1

            if result["text_triplets_added"] > 0:
                result["chunks_with_triplets"] = len(valid_nodes)
            else:
                result["chunks_empty"] = len(valid_nodes)

            return result

        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            return result

    async def _process_node(
        self,
        node: Any,
        graph_store: "Union[MemoryGraphStore, Neo4jGraphStore]"
    ) -> Optional[Dict[str, Any]]:
        """处理单个节点"""
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

            if has_images and self.config.multimodal_enabled:
                # 多模态联合抽取
                image_path = metadata.get("image_path", "")
                result = await self._extract_multimodal_triplets(
                    text=text,
                    image_path=image_path,
                    image_caption=metadata.get("image_caption", ""),
                    chunk_id=chunk_id,
                    graph_store=graph_store
                )
            else:
                # 纯文本抽取
                result = await self._extract_text_triplets(
                    text=text,
                    chunk_id=chunk_id,
                    graph_store=graph_store
                )

            result["has_images"] = 1 if has_images else 0
            result["has_triplets"] = (
                result.get("text_triplets_added", 0) > 0 or
                result.get("image_entities_added", 0) > 0 or
                result.get("cross_modal_triplets_added", 0) > 0
            )
            return result

        except Exception as e:
            logger.error(f"处理节点失败: {e}")
            return None

    async def _extract_text_triplets(
        self,
        text: str,
        chunk_id: str,
        graph_store: "Union[MemoryGraphStore, Neo4jGraphStore]"
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
            user_prompt = f"## Input Text\n\n{text[:3000]}\n\nExtract all entity-relationship triplets:"

            assert self._llm is not None
            response = await self._llm.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            response_text = response.content if hasattr(response, 'content') else str(response)

            triplets = self._parse_json_response(response_text)

            for triplet in triplets:
                head = triplet.get("head", "").strip()
                relation = triplet.get("relation", "").strip()
                tail = triplet.get("tail", "").strip()

                if not head or not relation or not tail:
                    continue

                head_id = graph_store.add_entity(
                    name=head,
                    entity_type=self._normalize_entity_type(triplet.get("head_type", "")),
                    chunk_id=chunk_id
                )
                tail_id = graph_store.add_entity(
                    name=tail,
                    entity_type=self._normalize_entity_type(triplet.get("tail_type", "")),
                    chunk_id=chunk_id
                )

                rel_id = graph_store.add_relation(
                    head=head,
                    tail=tail,
                    relation=relation,
                    weight=triplet.get("confidence", 1.0),
                    chunk_id=chunk_id
                )

                if rel_id:
                    result["text_triplets_added"] += 1
                    result["entities_added"] += 2

        except Exception as e:
            logger.warning(f"文本三元组抽取失败: {e}")

        return result

    async def _extract_multimodal_triplets(
        self,
        text: str,
        image_path: str,
        image_caption: str,
        chunk_id: str,
        graph_store: "Union[MemoryGraphStore, Neo4jGraphStore]"
    ) -> Dict[str, Any]:
        """多模态联合三元组抽取"""
        result = {
            "entities_added": 0,
            "text_triplets_added": 0,
            "image_entities_added": 0,
            "cross_modal_triplets_added": 0
        }

        try:
            # 提取 figure_id（如 "Figure 2"）
            figure_id = self._extract_figure_id(image_caption, text)

            system_prompt = MULTIMODAL_TRIPLET_EXTRACTION_PROMPT.format(
                text=text[:2000],
                image_caption=image_caption or "No caption",
                figure_id=figure_id
            )
            user_prompt = f"""Analyze the image and extract cross-modal knowledge graph triplets.

Text: {text[:2000]}
Image Caption: {image_caption or 'No caption'}

Extract triplets:"""

            # 调用多模态 LLM
            assert self._llm is not None
            response = await self._llm.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                image_urls=[image_path] if self.config.multimodal_enabled else None
            )
            response_text = response.content if hasattr(response, 'content') else str(response)

            data = self._parse_multimodal_response(response_text)

            # 1. 存储文本三元组
            text_triplets = data.get("text_triplets", [])
            for triplet in text_triplets[:self.config.max_triplets_per_chunk]:
                head = triplet.get("head", "").strip()
                relation = triplet.get("relation", "").strip()
                tail = triplet.get("tail", "").strip()

                if not head or not relation or not tail:
                    continue

                head_id = graph_store.add_entity(
                    name=head,
                    entity_type=self._normalize_entity_type(triplet.get("head_type", "")),
                    chunk_id=chunk_id
                )
                tail_id = graph_store.add_entity(
                    name=tail,
                    entity_type=self._normalize_entity_type(triplet.get("tail_type", "")),
                    chunk_id=chunk_id
                )

                rel_id = graph_store.add_relation(
                    head=head,
                    tail=tail,
                    relation=relation,
                    weight=triplet.get("confidence", 1.0),
                    chunk_id=chunk_id
                )

                if rel_id:
                    result["text_triplets_added"] += 1
                    result["entities_added"] += 2

            # 2. 存储图片实体
            image_info = data.get("image_info", {})
            if image_info and self.config.extract_image_entities:
                figure_entity_id = graph_store.add_image_entity(
                    figure_id=figure_id,
                    image_path=image_path,
                    description=image_info.get("description", ""),
                    figure_type=image_info.get("figure_type", "unknown"),
                    chunk_id=chunk_id
                )
                result["image_entities_added"] += 1
                result["entities_added"] += 1

                # 3. 存储跨模态三元组
                cross_triplets = data.get("cross_modal_triplets", [])
                for triplet in cross_triplets:
                    head = triplet.get("head", figure_id)
                    relation = triplet.get("relation", "").strip()
                    tail = triplet.get("tail", "").strip()

                    if not relation or not tail:
                        continue

                    # 确保图片实体存在
                    graph_store.add_image_entity(
                        figure_id=figure_id,
                        image_path=image_path,
                        description=image_info.get("description", ""),
                        figure_type=image_info.get("figure_type", "unknown"),
                        chunk_id=chunk_id
                    )

                    rel_id = graph_store.add_relation(
                        head=head,
                        tail=tail,
                        relation=relation,
                        weight=triplet.get("confidence", 0.9),
                        chunk_id=chunk_id
                    )

                    if rel_id:
                        result["cross_modal_triplets_added"] += 1

        except Exception as e:
            logger.warning(f"多模态三元组抽取失败: {e}")
            # 回退到纯文本抽取
            return await self._extract_text_triplets(text, chunk_id, graph_store)

        return result

    def _extract_figure_id(self, caption: str, text: str) -> str:
        """从图注或文本中提取 figure ID"""
        import re

        # 从 caption 提取
        patterns = [
            r'(Figure|Fig\.?|图)\s*(\d+[a-zA-Z]?)',
            r'(Table|表格)\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, caption or text, re.IGNORECASE)
            if match:
                prefix = match.group(1)
                num = match.group(2)
                if prefix.lower() in ["figure", "fig.", "图"]:
                    return f"Figure {num}"
                elif prefix.lower() in ["table", "表格"]:
                    return f"Table {num}"

        return "Figure 1"

    def _normalize_entity_type(self, entity_type: str) -> str:
        """标准化实体类型"""
        type_mapping = {
            "model/architecture": "Model/Architecture",
            "method/technique": "Method/Technique",
            "task": "Task",
            "dataset": "Dataset",
            "metric": "Metric",
            "optimizer/algorithm": "Optimizer/Algorithm",
            "framework/library": "Framework/Library",
            "author/organization": "Author/Organization",
            "venue": "Venue",
            "hyperparameter": "Hyperparameter",
            "experiment setting": "Experiment Setting",
            "result/conclusion": "Result/Conclusion",
            "application/domain": "Application/Domain",
            "other": "Other",
        }
        return type_mapping.get(entity_type.lower().strip(), "Other")

    def _strip_thinking_tokens(self, text: str) -> str:
        """移除 Qwen3.5 thinking 模式产生的思考 tokens

        Qwen3.5 模型在 think=True 时会输出 <think>...</think> 块，
        这些内容不是 JSON，需要移除后再解析。
        """
        if not text:
            return text
        import re
        # 递归移除所有 <think>...</think> 块
        stripped = text
        while '<think>' in stripped and '</think>' in stripped:
            stripped = re.sub(r'<think>.*?</think>', '', stripped, flags=re.DOTALL)
        return stripped.strip()

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """解析 JSON 响应"""
        try:
            if not response:
                logger.warning(f"JSON 响应为空，原始类型: {type(response)}")
                return []
            json_str = response.strip().lstrip('\ufeff')  # 移除 UTF-8 BOM
            # 移除 thinking tokens
            json_str = self._strip_thinking_tokens(json_str)
            if not json_str:
                logger.warning("JSON 响应移除 thinking 后为空")
                return []
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            data = json.loads(json_str)

            if isinstance(data, dict):
                triplets = data.get("triplets", [])
                if isinstance(triplets, list):
                    result = []
                    for item in triplets:
                        if not all(k in item for k in ("head", "relation", "tail")):
                            continue
                        result.append({
                            "head": str(item.get("head", "")),
                            "head_type": item.get("head_type", ""),
                            "relation": str(item.get("relation", "")),
                            "relation_type": item.get("relation_type", ""),
                            "tail": str(item.get("tail", "")),
                            "tail_type": item.get("tail_type", ""),
                            "confidence": float(item.get("confidence", 0.5)),
                            "evidence": item.get("evidence", "")
                        })
                    result.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                    return result

            elif isinstance(data, list):
                return [
                    {
                        "head": str(item.get("head", "")),
                        "head_type": "",
                        "relation": str(item.get("relation", "")),
                        "relation_type": "",
                        "tail": str(item.get("tail", "")),
                        "tail_type": "",
                        "confidence": 0.5,
                        "evidence": ""
                    }
                    for item in data
                    if item.get("head") and item.get("relation") and item.get("tail")
                ]

            return []

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}")
            return []

    def _parse_multimodal_response(self, response: str) -> Dict[str, Any]:
        """解析多模态响应"""
        try:
            if not response or not isinstance(response, str):
                logger.warning(f"多模态响应无效: type={type(response)}, value={str(response)[:100] if response else 'None'}")
                return {"text_triplets": [], "image_info": {}, "cross_modal_triplets": []}

            json_str = response.strip().lstrip('\ufeff')  # 移除 UTF-8 BOM
            # 移除 thinking tokens
            json_str = self._strip_thinking_tokens(json_str)
            if not json_str:
                logger.warning(f"多模态响应移除 thinking 后为空，原始响应前200字符: {str(response)[:200]}")
                return {"text_triplets": [], "image_info": {}, "cross_modal_triplets": []}
            logger.info(f"[Graph-LLM] 移除 thinking 后响应前200字符: {json_str[:200]}")

            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.warning(f"多模态 JSON 解析失败: {e}, 响应内容: {str(response)[:200] if response else 'None'}")
            return {"text_triplets": [], "image_info": {}, "cross_modal_triplets": []}
        except Exception as e:
            logger.warning(f"多模态响应解析异常: {e}, type={type(response)}")
            return {"text_triplets": [], "image_info": {}, "cross_modal_triplets": []}


# ============================================================================
# 便捷函数
# ============================================================================

async def build_graph_from_documents(
    documents: List[str],
    graph_store: "Union[MemoryGraphStore, Neo4jGraphStore]",
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
