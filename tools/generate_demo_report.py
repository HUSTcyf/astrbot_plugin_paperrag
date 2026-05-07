#!/usr/bin/env python3
"""
PaperRAG Agentic vs Regular 性能对比 Demo.

生成自包含的 comparison_report.html，浏览器直接打开即可录制视频。

用法:
    cd /path/to/astrbot_plugin_paperrag
    python tools/generate_demo_report.py              # 真实运行
    python tools/generate_demo_report.py --dry-run    # Mock 模式，不调用任何本地 RAG 系统

输出:
    comparison_report.html（在插件根目录）
"""

from __future__ import annotations

import asyncio
import json
import sys
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── 路径设置 ─────────────────────────────────────────────
PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

CONFIG_PATH = Path("/Users/chenyifeng/AstrBot/data/config/astrbot_plugin_paperrag_config.json")
OUTPUT_PATH = PLUGIN_DIR / "comparison_report.html"

# ── 测试数据 ─────────────────────────────────────────────
RAG_QUERIES = [
    ("What is 3D Gaussian Splatting?", "fact"),
    ("3D Gaussian Splatting 与 NeRF 的对比分析", "comparison"),
    ("3D Gaussian Splatting 最新研究进展综述", "review"),
    ("3D Gaussian Splatting 奠基性工作及被引情况", "citation"),
]

IDEA_TOPICS = [
    "3D Gaussian Splatting 实时渲染优化",
    "NeRF 与 3DGS 结合的稀疏视图重建",
]

# ── Mock 数据（--dry-run 模式使用）─────────────────────────
MOCK_RAG_DATA = [
    {
        "query": "What is 3D Gaussian Splatting?",
        "expected_type": "fact",
        "regular_nodes": [
            {"text": "3D Gaussian Splatting represents scenes as a set of 3D Gaussians...", "score": 0.92, "metadata": {"file_name": "kerbl_3dgs_2023.pdf"}},
            {"text": "Unlike NeRF, 3DGS uses explicit point-based representation...", "score": 0.87, "metadata": {"file_name": "survey_3dgs_2024.pdf"}},
            {"text": "The key innovation is differentiable rasterization of Gaussian primitives.", "score": 0.81, "metadata": {"file_name": "gaussian_splatting_survey.pdf"}},
        ],
        "regular_node_count": 3,
        "regular_answer": "3D Gaussian Splatting (3DGS) 是一种基于显式点云表示的3D场景重建技术。它使用可微分的3D高斯椭球体来表示场景，通过可微分光栅化实现实时渲染。与NeRF的隐式MLP表示不同，3DGS使用显式的高斯基元，支持更快的训练和推理速度。",
        "regular_time": 2.3,
        "agentic_query_type": "fact",
        "agentic_graph_weight": 0.2,
        "agentic_graph_entities": [
            {"name": "3D Gaussian Splatting", "type": "Method"},
            {"name": "Differentiable Rasterization", "type": "Technique"},
            {"name": "Kerbl et al. 2023", "type": "Paper"},
        ],
        "agentic_graph_relations": [
            {"head": "3D Gaussian Splatting", "relation": "uses", "tail": "Differentiable Rasterization"},
            {"head": "Kerbl et al. 2023", "relation": "proposes", "tail": "3D Gaussian Splatting"},
        ],
        "agentic_nodes": [
            {"text": "3D Gaussian Splatting represents scenes as a set of 3D Gaussians...", "score": 0.92, "metadata": {"file_name": "kerbl_3dgs_2023.pdf"}, "source": "vector"},
            {"text": "Knowledge Graph: 3DGS uses Differentiable Rasterization.", "score": 0.85, "metadata": {"file_name": "neo4j"}, "source": "graph"},
            {"text": "Unlike NeRF, 3DGS uses explicit point-based representation...", "score": 0.87, "metadata": {"file_name": "survey_3dgs_2024.pdf"}, "source": "vector"},
            {"text": "The key innovation is differentiable rasterization of Gaussian primitives.", "score": 0.81, "metadata": {"file_name": "gaussian_splatting_survey.pdf"}, "source": "vector"},
            {"text": "Knowledge Graph: Kerbl et al. proposes 3D Gaussian Splatting.", "score": 0.78, "metadata": {"file_name": "neo4j"}, "source": "graph"},
        ],
        "agentic_node_count": 5,
        "agentic_answer": "3D Gaussian Splatting (3DGS) 是Kerbl等人于2023年提出的显式3D场景表示方法。其核心是使用3D高斯椭球体作为场景基元，通过可微分光栅化实现实时高质量渲染。\n\n根据知识图谱，3DGS的核心技术栈包括：可微分光栅化、自适应密度控制、球谐函数颜色编码。相比NeRF的隐式表示，3DGS在训练速度上快约100倍，渲染速度可达实时（30+ FPS）。",
        "agentic_citations": ["Kerbl et al. 2023", "Gaussian Splatting Survey 2024"],
        "agentic_steps": ["router: fact (graph_weight=0.2)", "vector_search: OK (3 nodes)", "graph_search: OK (2 entities, 2 relations)", "synthesize: OK"],
        "agentic_time": 4.1,
    },
    {
        "query": "3D Gaussian Splatting 与 NeRF 的对比分析",
        "expected_type": "comparison",
        "regular_nodes": [
            {"text": "NeRF uses MLP to represent scenes implicitly...", "score": 0.91, "metadata": {"file_name": "mildenhall_nerf_2020.pdf"}},
            {"text": "3DGS achieves real-time rendering at 30+ FPS...", "score": 0.88, "metadata": {"file_name": "kerbl_3dgs_2023.pdf"}},
        ],
        "regular_node_count": 2,
        "regular_answer": "NeRF使用MLP隐式表示场景，渲染速度慢；3DGS使用显式高斯表示，可以实时渲染。两者在重建质量上接近，但3DGS在训练和推理效率上具有显著优势。",
        "regular_time": 1.8,
        "agentic_query_type": "comparison",
        "agentic_graph_weight": 0.5,
        "agentic_graph_entities": [
            {"name": "NeRF", "type": "Method"},
            {"name": "3D Gaussian Splatting", "type": "Method"},
            {"name": "PSNR", "type": "Metric"},
            {"name": "Rendering Speed", "type": "Metric"},
        ],
        "agentic_graph_relations": [
            {"head": "3D Gaussian Splatting", "relation": "outperforms", "tail": "NeRF"},
            {"head": "NeRF", "relation": "measured_by", "tail": "PSNR"},
            {"head": "3D Gaussian Splatting", "relation": "excels_in", "tail": "Rendering Speed"},
        ],
        "agentic_nodes": [
            {"text": "NeRF uses MLP to represent scenes implicitly...", "score": 0.91, "metadata": {"file_name": "mildenhall_nerf_2020.pdf"}, "source": "vector"},
            {"text": "3DGS achieves real-time rendering at 30+ FPS...", "score": 0.88, "metadata": {"file_name": "kerbl_3dgs_2023.pdf"}, "source": "vector"},
            {"text": "Knowledge Graph: 3DGS outperforms NeRF in rendering speed.", "score": 0.82, "metadata": {"file_name": "neo4j"}, "source": "graph"},
        ],
        "agentic_node_count": 3,
        "agentic_answer": "3DGS与NeRF的对比分析（来源：知识图谱 + 向量检索）：\n\n1. **表示方式**：NeRF使用隐式MLP网络，3DGS使用显式3D高斯椭球体。\n2. **训练速度**：3DGS训练约30分钟，NeRF需数小时。知识图谱显示3DGS在训练效率上outperforms NeRF。\n3. **渲染速度**：3DGS可达30+ FPS实时渲染，NeRF需数秒/帧。\n4. **重建质量**：两者PSNR接近，NeRF在镜面反射场景略优。\n5. **编辑能力**：3DGS显式表示更易于场景编辑和操作。",
        "agentic_citations": ["Mildenhall et al. 2020", "Kerbl et al. 2023"],
        "agentic_steps": ["router: comparison (graph_weight=0.5)", "vector_search: OK (2 nodes)", "graph_search: OK (4 entities, 3 relations)", "synthesize: OK"],
        "agentic_time": 5.3,
    },
    {
        "query": "3D Gaussian Splatting 最新研究进展综述",
        "expected_type": "review",
        "regular_nodes": [
            {"text": "Recent advances include 4D Gaussian Splatting...", "score": 0.89, "metadata": {"file_name": "survey_2024_part2.pdf"}},
            {"text": "Dynamic scene reconstruction with deformable 3DGS...", "score": 0.85, "metadata": {"file_name": "dynamic_3dgs.pdf"}},
            {"text": "SLAM systems based on 3DGS show promising results.", "score": 0.82, "metadata": {"file_name": "gs_slam_2024.pdf"}},
            {"text": "Text-to-3D generation using Gaussian Splatting...", "score": 0.79, "metadata": {"file_name": "text2gs.pdf"}},
        ],
        "regular_node_count": 4,
        "regular_answer": "3DGS的最新进展主要包括：4D动态场景重建、3DGS-SLAM系统、文本到3D生成、以及压缩和加速技术。这些方向都在快速发展。",
        "regular_time": 2.1,
        "agentic_query_type": "review",
        "agentic_graph_weight": 0.4,
        "agentic_graph_entities": [
            {"name": "4D Gaussian Splatting", "type": "Method"},
            {"name": "3DGS-SLAM", "type": "Application"},
            {"name": "Dynamic Reconstruction", "type": "Task"},
        ],
        "agentic_graph_relations": [
            {"head": "4D Gaussian Splatting", "relation": "extends", "tail": "3D Gaussian Splatting"},
            {"head": "3DGS-SLAM", "relation": "based_on", "tail": "3D Gaussian Splatting"},
        ],
        "agentic_nodes": [
            {"text": "Recent advances include 4D Gaussian Splatting...", "score": 0.89, "metadata": {"file_name": "survey_2024_part2.pdf"}, "source": "vector"},
            {"text": "Knowledge Graph: 4DGS extends 3DGS to temporal domain.", "score": 0.84, "metadata": {"file_name": "neo4j"}, "source": "graph"},
            {"text": "Dynamic scene reconstruction with deformable 3DGS...", "score": 0.85, "metadata": {"file_name": "dynamic_3dgs.pdf"}, "source": "vector"},
            {"text": "SLAM systems based on 3DGS show promising results.", "score": 0.82, "metadata": {"file_name": "gs_slam_2024.pdf"}, "source": "vector"},
            {"text": "Text-to-3D generation using Gaussian Splatting...", "score": 0.79, "metadata": {"file_name": "text2gs.pdf"}, "source": "vector"},
            {"text": "Knowledge Graph: 3DGS-SLAM based_on 3DGS.", "score": 0.76, "metadata": {"file_name": "neo4j"}, "source": "graph"},
        ],
        "agentic_node_count": 6,
        "agentic_answer": "3D Gaussian Splatting最新研究进展综述（基于知识图谱和文献检索）：\n\n**1. 动态场景扩展**：4D Gaussian Splatting将3DGS扩展到时间维度（知识图谱：extends关系），支持动态场景重建。\n\n**2. SLAM系统**：3DGS-SLAM（based_on 3DGS）在实时SLAM中表现优异，同时提供高质量地图。\n\n**3. 生成式应用**：Text-to-3DGS使用扩散模型生成高斯场景，DreamGaussian等工作在消费级GPU上运行。\n\n**4. 压缩与加速**：Compact3D等方法将存储从GB级压缩到MB级。\n\n**5. 自动驾驶**：3DGS用于BEV感知和占用网络预测。",
        "agentic_citations": ["Survey 2024", "Dynamic 3DGS", "GS-SLAM 2024", "Text-to-3DGS"],
        "agentic_steps": ["router: review (graph_weight=0.4)", "vector_search: OK (4 nodes)", "graph_search: OK (3 entities, 2 relations)", "synthesize: OK"],
        "agentic_time": 6.7,
    },
    {
        "query": "3D Gaussian Splatting 奠基性工作及被引情况",
        "expected_type": "citation",
        "regular_nodes": [
            {"text": "Kerbl et al. '3D Gaussian Splatting for Real-Time Radiance Field Rendering' SIGGRAPH 2023.", "score": 0.95, "metadata": {"file_name": "kerbl_3dgs_2023.pdf"}},
            {"text": "The paper has been cited over 2000 times since publication.", "score": 0.72, "metadata": {"file_name": "citation_stats.pdf"}},
        ],
        "regular_node_count": 2,
        "regular_answer": "3DGS的奠基性工作是Kerbl等人2023年在SIGGRAPH发表的论文。该工作已被大量引用。",
        "regular_time": 1.5,
        "agentic_query_type": "citation",
        "agentic_graph_weight": 0.6,
        "agentic_graph_entities": [
            {"name": "Kerbl et al. 2023", "type": "Paper"},
            {"name": "SIGGRAPH 2023", "type": "Venue"},
            {"name": "Mildenhall et al. 2020", "type": "Paper"},
        ],
        "agentic_graph_relations": [
            {"head": "Kerbl et al. 2023", "relation": "published_in", "tail": "SIGGRAPH 2023"},
            {"head": "Kerbl et al. 2023", "relation": "cites", "tail": "Mildenhall et al. 2020"},
            {"head": "Kerbl et al. 2023", "relation": "cited_by_count", "tail": "2100+"},
        ],
        "agentic_nodes": [
            {"text": "Kerbl et al. '3D Gaussian Splatting for Real-Time Radiance Field Rendering' SIGGRAPH 2023.", "score": 0.95, "metadata": {"file_name": "kerbl_3dgs_2023.pdf"}, "source": "vector"},
            {"text": "Knowledge Graph: Kerbl et al. published_in SIGGRAPH 2023, cites Mildenhall, cited_by 2100+.", "score": 0.90, "metadata": {"file_name": "neo4j"}, "source": "graph"},
            {"text": "The paper has been cited over 2000 times since publication.", "score": 0.72, "metadata": {"file_name": "citation_stats.pdf"}, "source": "vector"},
        ],
        "agentic_node_count": 3,
        "agentic_answer": "3D Gaussian Splatting的奠基性工作及引用分析：\n\n**奠基论文**：Kerbl et al., \"3D Gaussian Splatting for Real-Time Radiance Field Rendering\", SIGGRAPH 2023（知识图谱确认：published_in SIGGRAPH 2023）。\n\n**引用关系**（知识图谱）：\n- cites → Mildenhall et al. (NeRF, ECCV 2020)\n- cited_by_count → 2100+（截至2025年初）\n\n**重要前驱工作**：Mildenhall等人的NeRF是该论文的核心baseline和灵感来源。\n\n**后续影响**：该工作催生了4DGS、3DGS-SLAM、Text-to-3DGS等多个子方向。",
        "agentic_citations": ["Kerbl et al. SIGGRAPH 2023", "Mildenhall et al. ECCV 2020"],
        "agentic_steps": ["router: citation (graph_weight=0.6)", "vector_search: OK (2 nodes)", "graph_search: OK (3 entities, 3 relations)", "synthesize: OK"],
        "agentic_time": 5.9,
    },
]

MOCK_IDEA_DATA = [
    {
        "topic": "3D Gaussian Splatting 实时渲染优化",
        "regular_ideas": [
            {"title": "Adaptive Level-of-Detail for 3DGS", "description": "根据视距动态调整高斯密度实现LOD渲染", "novelty": "首次将LOD应用于显式3DGS", "methodology": "基于屏幕空间投影的分层聚合", "feasibility": 0.7},
            {"title": "Hardware-Accelerated Splat Blending", "description": "利用GPU tensor core加速高斯blend", "novelty": "硬件协同设计", "methodology": "CUDA kernel优化", "feasibility": 0.6},
            {"title": "Importance-Based Gaussian Pruning", "description": "基于贡献度剪枝冗余高斯", "novelty": "混合重要性度量", "methodology": "梯度+可见性联合评分", "feasibility": 0.8},
        ],
        "regular_idea_count": 3,
        "regular_time": 8.2,
        "agentic_ideas": [
            {"title": "Foveated 3D Gaussian Rendering with Eye Tracking", "description": "利用眼动追踪实现注视点渲染，在VR/AR场景中降低外围视场高斯密度70%以上", "score": "8.5/10"},
            {"title": "Temporal Coherence-Aware Pruning Pipeline", "description": "基于帧间时序一致性动态剪枝，将时域冗余高斯识别率提升40%", "score": "7.8/10"},
            {"title": "Hardware-Aware Mixed-Precision Splatting", "description": "自适应混合精度渲染，在移动端GPU上实现2x加速，PSNR损失<0.3dB", "score": "8.2/10"},
        ],
        "agentic_idea_count": 3,
        "agentic_critique": "Idea #1 创新性高但需要额外硬件（眼动追踪），限制了适用范围。Idea #2 技术路径清晰，建议补充与Deformable 3DGS的对比实验。Idea #3 实用性强，建议明确目标移动设备型号。",
        "agentic_confidence": 0.72,
        "agentic_idea_scores": [
            {"dimension": "novelty", "score": 8.5},
            {"dimension": "feasibility", "score": 7.0},
            {"dimension": "impact", "score": 8.0},
        ],
        "agentic_steps": [
            "analyze: OK (domain=3D Vision)",
            "search: OK (5 local + 3 web)",
            "generate: OK (3 ideas)",
            "critique: OK (3 issues found)",
            "refine: OK (round 1, ideas updated)",
            "critique: OK (1 issue remaining)",
            "refine: OK (round 2, final)",
        ],
        "agentic_iterations": 2,
        "agentic_time": 18.5,
    },
    {
        "topic": "NeRF 与 3DGS 结合的稀疏视图重建",
        "regular_ideas": [
            {"title": "NeRF-init 3DGS Optimization", "description": "用NeRF预热3DGS的初始高斯位置", "novelty": "混合初始化策略", "methodology": "先训练轻量NeRF再迁移", "feasibility": 0.65},
            {"title": "Multi-View Consistency Regularization", "description": "引入多视图一致性约束提升稀疏视图质量", "novelty": "新型正则化项", "methodology": "几何一致性损失", "feasibility": 0.7},
            {"title": "Diffusion Prior Enhanced Sparse 3DGS", "description": "使用扩散模型prior填充缺失视图", "novelty": "生成模型辅助", "methodology": "Zero-1-to-3 + 3DGS融合", "feasibility": 0.55},
        ],
        "regular_idea_count": 3,
        "regular_time": 7.8,
        "agentic_ideas": [
            {"title": "Progressive Geometry-Image Co-optimization", "description": "NeRF隐式先验与3DGS显式表示的渐进式联合优化，在3视图输入下PSNR提升3.2dB", "score": "9.0/10"},
            {"title": "Epipolar Attention-Guided Sparse View Fusion", "description": "基于极线几何的跨视图注意力机制，替代传统多视图立体匹配，适用于非朗伯表面", "score": "8.7/10"},
            {"title": "Uncertainty-Aware Neural-Gaussian Hybrid Field", "description": "在不确定区域保留NeRF隐式表示，高置信区域使用3DGS，实现自适应混合表示", "score": "8.9/10"},
        ],
        "agentic_idea_count": 3,
        "agentic_critique": "Idea #1 技术路线扎实但需注意NeRF预训练的计算开销。Idea #2 极线注意力在稀疏视图下可能存在退化，建议补充3-view和5-view消融。Idea #3 混合表示切换策略是关键，建议使用贝叶斯不确定性估计。",
        "agentic_confidence": 0.78,
        "agentic_idea_scores": [
            {"dimension": "novelty", "score": 9.0},
            {"dimension": "feasibility", "score": 7.5},
            {"dimension": "impact", "score": 8.5},
        ],
        "agentic_steps": [
            "analyze: OK (domain=3D Vision)",
            "search: OK (4 local + 3 web)",
            "generate: OK (3 ideas)",
            "critique: OK (4 issues found)",
            "refine: OK (round 1, ideas updated)",
            "critique: OK (2 issues remaining)",
            "refine: OK (round 2, ideas updated)",
            "critique: OK (all resolved)",
        ],
        "agentic_iterations": 2,
        "agentic_time": 22.1,
    },
]


# ═══════════════════════════════════════════════════════════
# Fake Context（复用 evaluation 脚本模式）
# ═══════════════════════════════════════════════════════════

class FakeContext:
    """最小化 AstrBot Context，供 engine 和 agentic workflow 使用。"""

    def __init__(self, config: dict):
        self.config = config
        self.provider_manager = None  # agentic 节点自动回退到 local VLM

    def get_using_provider(self):
        """Agentic router 可能调用的方法。"""
        return None


# ═══════════════════════════════════════════════════════════
# Provider 工具
# ═══════════════════════════════════════════════════════════

async def _get_local_vlm_provider():
    """获取本地 VLM provider 单例（复用内存中已有实例，不创建新的）。"""
    try:
        from provider.llama_cpp_vlm import get_llama_cpp_vlm_provider
        provider = get_llama_cpp_vlm_provider()
        if provider is None:
            return None
        if not getattr(provider, "_initialized", False):
            await provider.initialize()
        return provider
    except Exception:
        pass
    return None


async def _call_llm(prompt: str, provider, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """通用 LLM 调用封装。"""
    response = await provider.text_chat(
        prompt=prompt, contexts=[], temperature=temperature, max_tokens=max_tokens,
    )
    if hasattr(response, "content"):
        return response.content
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("content", "") or response.get("text", "")
    return str(response)


async def _generate_answer_from_nodes(
    query: str, nodes: list[dict], provider
) -> str:
    """用检索到的 nodes 生成 RAG 回答（模拟常规 RAG 的 answer generation）。"""
    if not nodes or provider is None:
        return ""

    context_parts = []
    for i, node in enumerate(nodes[:5], 1):
        text = node.get("text", "")[:500]
        context_parts.append(f"[{i}] {text}")

    context_str = "\n\n".join(context_parts)
    prompt = (
        "你是一个学术论文问答助手。基于以下检索结果回答用户问题。"
        "引用来源时使用 [n] 格式。如果信息不足，明确指出。\n\n"
        f"用户问题: {query}\n\n检索结果:\n{context_str}\n\n请回答:"
    )
    return await _call_llm(prompt, provider, temperature=0.3)


# ═══════════════════════════════════════════════════════════
# RAG 对比
# ═══════════════════════════════════════════════════════════

async def run_rag_comparison(
    engine, context: FakeContext, config: dict, *, dry_run: bool = False
) -> list[dict]:
    """对每个查询执行常规 RAG 和 Agentic RAG，返回对比结果列表。"""
    if dry_run:
        print("  🧪 Dry-run mode: using mock RAG data")
        return MOCK_RAG_DATA

    from agentic_rag.workflow import compile_workflow

    provider = await _get_local_vlm_provider()
    results = []

    for query, expected_type in RAG_QUERIES:
        entry: dict[str, Any] = {"query": query, "expected_type": expected_type}
        t0 = time.time()

        # ── 常规 RAG ──
        try:
            reg_result = await engine.search(query, mode="retrieve", top_k=5)
            reg_nodes = []
            if reg_result is not None and hasattr(reg_result, "nodes"):
                for i, node in enumerate(reg_result.nodes):
                    reg_nodes.append({
                        "text": getattr(node, "text", ""),
                        "score": reg_result.scores[i] if i < len(reg_result.scores) else 1.0,
                        "metadata": getattr(node, "metadata", {}),
                    })
            elif isinstance(reg_result, list):
                reg_nodes = reg_result
            entry["regular_nodes"] = reg_nodes
            entry["regular_node_count"] = len(reg_nodes)

            # 用 local VLM 生成回答
            if provider and reg_nodes:
                entry["regular_answer"] = await _generate_answer_from_nodes(query, reg_nodes, provider)
            else:
                entry["regular_answer"] = ""
            entry["regular_time"] = round(time.time() - t0, 2)
        except Exception as e:
            entry["regular_nodes"] = []
            entry["regular_node_count"] = 0
            entry["regular_answer"] = f"[ERROR] {e}"
            entry["regular_time"] = round(time.time() - t0, 2)

        # ── Agentic RAG ──
        t1 = time.time()
        try:
            app = compile_workflow()
            state = await app.ainvoke({
                "query": query,
                "_context": context,
                "_config": config,
                "top_k": 5,
                "steps": [],
            })
            entry["agentic_query_type"] = state.get("query_type", "fact")
            entry["agentic_graph_weight"] = state.get("graph_weight", 0.0)
            entry["agentic_graph_entities"] = state.get("graph_entities", [])
            entry["agentic_graph_relations"] = state.get("graph_relations", [])
            entry["agentic_nodes"] = state.get("retrieved_nodes", [])
            entry["agentic_node_count"] = len(state.get("retrieved_nodes", []))
            entry["agentic_answer"] = state.get("final_answer", "")
            entry["agentic_citations"] = state.get("citations", [])
            entry["agentic_steps"] = state.get("steps", [])
            entry["agentic_time"] = round(time.time() - t1, 2)
        except Exception as e:
            import traceback
            entry["agentic_error"] = str(e)
            entry["agentic_traceback"] = traceback.format_exc()
            entry["agentic_query_type"] = "N/A"
            entry["agentic_graph_weight"] = 0.0
            entry["agentic_graph_entities"] = []
            entry["agentic_graph_relations"] = []
            entry["agentic_nodes"] = []
            entry["agentic_node_count"] = 0
            entry["agentic_answer"] = f"[ERROR] {e}"
            entry["agentic_citations"] = []
            entry["agentic_steps"] = []
            entry["agentic_time"] = round(time.time() - t1, 2)

        results.append(entry)
        print(f"  ✅ RAG query done: {query[:50]}...")

    return results


# ═══════════════════════════════════════════════════════════
# Idea 对比
# ═══════════════════════════════════════════════════════════

async def run_idea_comparison(
    engine, context: FakeContext, *, dry_run: bool = False
) -> list[dict]:
    """对每个 topic 执行常规 Idea 和 Agentic Idea，返回对比结果列表。"""
    if dry_run:
        print("  🧪 Dry-run mode: using mock Idea data")
        return MOCK_IDEA_DATA

    from idea import IdeaEngine
    from idea.agentic_workflow import run_agentic_ideas

    results = []

    for topic in IDEA_TOPICS:
        entry: dict[str, Any] = {"topic": topic}
        t0 = time.time()

        # ── 常规 Idea ──
        try:
            idea_engine = IdeaEngine(context=context, rag_engine=engine)
            analysis = await idea_engine.analyze_topic(topic, depth="standard")
            queries = (analysis.search_queries or [])[:5] + (analysis.local_rag_queries or [])[:3]
            if not queries:
                queries = [topic]
            knowledge = await idea_engine.search_knowledge(
                queries=queries, local_rag_top_k=10, web_top_k=5,
            )
            fused = knowledge.get("fused_context", "") if knowledge else ""
            ideas = await idea_engine.generate_ideas(
                knowledge_context=fused,
                research_domain=analysis.domain if analysis else "",
                num_ideas=3,
                topic=topic,
            )
            entry["regular_ideas"] = [
                {
                    "title": idea.title,
                    "description": (idea.description or "")[:200],
                    "novelty": (idea.novelty or "")[:150],
                    "methodology": (idea.methodology or "")[:150],
                    "feasibility": getattr(idea, "feasibility", 0.5),
                }
                for idea in ideas
            ]
            entry["regular_idea_count"] = len(ideas)
            entry["regular_time"] = round(time.time() - t0, 2)
        except Exception as e:
            import traceback
            entry["regular_error"] = str(e)
            entry["regular_traceback"] = traceback.format_exc()
            entry["regular_ideas"] = []
            entry["regular_idea_count"] = 0
            entry["regular_time"] = round(time.time() - t0, 2)

        # ── Agentic Idea ──
        t1 = time.time()
        try:
            agentic_result = await run_agentic_ideas(
                topic=topic, context=context, depth="standard",
                num_ideas=3, rag_engine=engine,
            )
            entry["agentic_ideas"] = agentic_result.get("ideas", [])
            entry["agentic_idea_count"] = len(agentic_result.get("ideas", []))
            entry["agentic_critique"] = agentic_result.get("critique", "")
            entry["agentic_confidence"] = agentic_result.get("confidence", 0.0)
            entry["agentic_idea_scores"] = agentic_result.get("idea_scores", [])
            entry["agentic_steps"] = agentic_result.get("steps", [])
            entry["agentic_final_output"] = agentic_result.get("final_output", "")
            entry["agentic_time"] = round(time.time() - t1, 2)

            # 计算迭代轮次
            iterations = sum(1 for s in entry["agentic_steps"] if "refine:" in s)
            entry["agentic_iterations"] = iterations
        except Exception as e:
            import traceback
            entry["agentic_error"] = str(e)
            entry["agentic_traceback"] = traceback.format_exc()
            entry["agentic_ideas"] = []
            entry["agentic_idea_count"] = 0
            entry["agentic_critique"] = f"[ERROR] {e}"
            entry["agentic_confidence"] = 0.0
            entry["agentic_idea_scores"] = []
            entry["agentic_steps"] = []
            entry["agentic_final_output"] = ""
            entry["agentic_iterations"] = 0
            entry["agentic_time"] = round(time.time() - t1, 2)

        results.append(entry)
        print(f"  ✅ Idea topic done: {topic[:50]}...")

    return results


# ═══════════════════════════════════════════════════════════
# HTML 报告生成
# ═══════════════════════════════════════════════════════════

CSS = """
:root {
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #212529;
    --muted: #6c757d;
    --accent: #2563eb;
    --green: #16a34a;
    --yellow: #ca8a04;
    --red: #dc2626;
    --border: #dee2e6;
    --tag-bg: #e9ecef;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    padding: 2rem; max-width: 1400px; margin: 0 auto;
}
h1 { font-size: 2rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.4rem; margin: 2rem 0 1rem; padding-bottom: 0.5rem;
     border-bottom: 2px solid var(--accent); }
h3 { font-size: 1.1rem; margin: 1rem 0 0.5rem; }
.header { text-align: center; margin-bottom: 2rem; }
.header .subtitle { color: var(--muted); font-size: 0.95rem; }
.header .config-badges { margin-top: 0.5rem; display: flex; gap: 0.5rem;
    justify-content: center; flex-wrap: wrap; }
.badge { display: inline-block; padding: 0.2em 0.6em; border-radius: 4px;
         font-size: 0.8rem; font-weight: 600; }
.badge-on  { background: #dcfce7; color: #166534; }
.badge-off { background: #fee2e2; color: #991b1b; }
.badge-info { background: #dbeafe; color: #1e40af; }

.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem; margin: 1rem 0; }
.summary-card { background: var(--card-bg); border-radius: 8px; padding: 1.2rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }
.summary-card .value { font-size: 2rem; font-weight: 700; color: var(--accent); }
.summary-card .label { font-size: 0.85rem; color: var(--muted); margin-top: 0.25rem; }
.summary-card.green .value { color: var(--green); }

.comparison-card { background: var(--card-bg); border-radius: 8px;
    padding: 1.5rem; margin: 1.5rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.comparison-card .query-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 1rem;
    padding: 0.5rem 0.75rem; background: #f0f4ff; border-radius: 4px; }

.side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
.side-left, .side-right { min-width: 0; }
.side-left h3 { color: var(--muted); }
.side-right h3 { color: var(--accent); }

.metric-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.5rem 0; }
.metric-tag { display: inline-block; padding: 0.15em 0.5em; border-radius: 4px;
              font-size: 0.8rem; background: var(--tag-bg); }
.metric-tag.green { background: #dcfce7; color: #166534; }
.metric-tag.blue  { background: #dbeafe; color: #1e40af; }

.answer-box { background: #f8f9fa; border: 1px solid var(--border); border-radius: 6px;
              padding: 1rem; margin: 0.75rem 0; font-size: 0.9rem; max-height: 300px;
              overflow-y: auto; word-break: break-word; }
.answer-box.empty { color: var(--muted); font-style: italic; }
.answer-box p { margin: 0.5rem 0; }
.answer-box ul, .answer-box ol { margin: 0.5rem 0; padding-left: 1.5rem; }
.answer-box li { margin: 0.25rem 0; }
.answer-box code { background: #e5e7eb; padding: 0.1em 0.3em; border-radius: 3px;
                   font-family: monospace; font-size: 0.85em; }
.answer-box pre { background: #1e293b; color: #e2e8f0; padding: 0.75rem;
                  border-radius: 6px; overflow-x: auto; }
.answer-box pre code { background: none; padding: 0; color: inherit; }
.answer-box strong { font-weight: 600; }
.answer-box em { font-style: italic; }
.answer-box h3, .answer-box h4, .answer-box h5 { margin: 0.75rem 0 0.25rem; }
.answer-box hr { border: none; border-top: 1px solid var(--border); margin: 0.75rem 0; }

.steps-list { font-size: 0.8rem; color: var(--muted); margin: 0.5rem 0; }
.steps-list li { list-style: none; padding: 0.15rem 0; font-family: monospace; }

.confidence-bar { height: 8px; border-radius: 4px; background: #e5e7eb; margin: 0.5rem 0; }
.confidence-bar .fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.confidence-bar .fill.low  { background: var(--red); }
.confidence-bar .fill.mid  { background: var(--yellow); }
.confidence-bar .fill.high { background: var(--green); }

.critique-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px;
                padding: 0.75rem; margin: 0.75rem 0; font-size: 0.9rem; }
.critique-box strong { color: #92400e; }
.critique-box p { margin: 0.4rem 0; }

.idea-item { padding: 0.75rem; margin: 0.5rem 0; border-left: 3px solid var(--border);
             background: #f8f9fa; border-radius: 0 6px 6px 0; }
.idea-item .idea-title { font-weight: 600; }
.idea-item .idea-score { font-size: 0.85rem; color: var(--accent); }

.takeaway { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
            padding: 1.25rem; margin: 1rem 0; }
.takeaway h4 { color: #166534; margin-bottom: 0.5rem; }
.takeaway ul { margin-left: 1.5rem; }
.takeaway li { margin: 0.25rem 0; }

.footer { text-align: center; color: var(--muted); font-size: 0.85rem;
          margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); }

.delta { font-weight: 600; }
.delta.pos { color: var(--green); }
.delta.neg { color: var(--red); }

@media (max-width: 900px) {
    .side-by-side { grid-template-columns: 1fr; }
    body { padding: 1rem; }
}
"""


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _markdown_to_html(text: str) -> str:
    """将 Markdown 文本转为 HTML（仅处理常见格式：bold, italic, code, lists, headings, paragraphs）。"""
    import re

    lines = text.split("\n")
    out: list[str] = []
    in_code_block = False
    list_buf: list[str] = []  # accumulate consecutive list items
    list_kind: str = ""  # "ul" or "ol"

    def _flush_list():
        nonlocal list_buf, list_kind
        if list_buf:
            tag = list_kind or "ul"
            out.append(f"<{tag}>" + "\n".join(list_buf) + f"</{tag}>")
            list_buf = []
            list_kind = ""

    i = 0
    while i < len(lines):
        line = lines[i]

        # code block (```)
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                _flush_list()
                out.append("<pre><code>")
            else:
                out.append("</code></pre>")
            i += 1
            continue

        if in_code_block:
            out.append(line)
            i += 1
            continue

        # empty line → paragraph break, flush list
        if not line.strip():
            _flush_list()
            i += 1
            continue

        stripped = line.strip()

        # heading: ### text or ## text
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading_match:
            _flush_list()
            level = min(len(heading_match.group(1)) + 2, 6)  # # → h3, ## → h4, ### → h5
            out.append(f"<h{level}>{heading_match.group(2)}</h{level}>")
            i += 1
            continue

        # ordered list: 1. text or 1) text
        ol_match = re.match(r"^(\d+)[.)]\s+(.+)$", stripped)
        if ol_match:
            if list_kind != "ol":
                _flush_list()
                list_kind = "ol"
            list_buf.append(f"<li>{ol_match.group(2)}</li>")
            i += 1
            continue

        # unordered list: - text or * text
        ul_match = re.match(r"^[-*]\s+(.+)$", stripped)
        if ul_match:
            if list_kind != "ul":
                _flush_list()
                list_kind = "ul"
            list_buf.append(f"<li>{ul_match.group(1)}</li>")
            i += 1
            continue

        # regular paragraph
        _flush_list()
        out.append(f"<p>{stripped}</p>")
        i += 1

    _flush_list()
    if in_code_block:
        out.append("</code></pre>")

    result = "\n".join(out)

    # inline formatting (must run after block-level to avoid list/heading conflicts)
    # bold: **text** or __text__
    result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", result)
    result = re.sub(r"__(.+?)__", r"<strong>\1</strong>", result)
    # italic: *text* or _text_ (but not inside words to avoid false positives)
    result = re.sub(r"\*(.+?)\*", r"<em>\1</em>", result)
    result = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", result)
    # inline code: `text`
    result = re.sub(r"`([^`]+)`", r"<code>\1</code>", result)
    # horizontal rule: --- or ***
    result = re.sub(r"^(---|\*\*\*)\s*$", "<hr>", result, flags=re.MULTILINE)

    return result


def _render_text(text: str) -> str:
    """安全地将文本渲染为 HTML：先转义 HTML 标签，再转换 Markdown 格式。"""
    safe = _escape_html(text)
    return _markdown_to_html(safe)


def _build_rag_card(entry: dict, index: int) -> str:
    """构建单个 RAG 查询的对比卡片 HTML。"""
    query = _escape_html(entry["query"])

    # ── 左侧：常规 RAG ──
    reg_answer = _render_text(entry.get("regular_answer", ""))
    reg_count = entry.get("regular_node_count", 0)
    reg_time = entry.get("regular_time", 0)

    left = f"""<div class="side-left">
        <h3>📋 Regular RAG</h3>
        <div class="metric-row">
            <span class="metric-tag">📄 {reg_count} sources</span>
            <span class="metric-tag">⏱ {reg_time}s</span>
        </div>"""

    if reg_answer:
        left += f"""<div class="answer-box">{reg_answer[:2000]}</div>"""
    else:
        left += """<div class="answer-box empty">（无 LLM 生成回答 — provider 未配置）</div>"""

    # 来源列表
    reg_nodes = entry.get("regular_nodes", [])
    if reg_nodes:
        left += """<details style="font-size:0.85rem;margin-top:0.5rem"><summary>📚 检索来源</summary><ul>"""
        for i, node in enumerate(reg_nodes[:5], 1):
            meta = node.get("metadata", {}) or {}
            fname = meta.get("file_name", "unknown")
            score = node.get("score", 0)
            left += f"<li>[{i}] {_escape_html(str(fname))} (score={score:.3f})</li>"
        left += "</ul></details>"

    left += "</div>"

    # ── 右侧：Agentic RAG ──
    ag_answer = _render_text(entry.get("agentic_answer", ""))
    ag_count = entry.get("agentic_node_count", 0)
    ag_time = entry.get("agentic_time", 0)
    qtype = entry.get("agentic_query_type", "N/A")
    gweight = entry.get("agentic_graph_weight", 0)
    gentities = entry.get("agentic_graph_entities", [])
    grelations = entry.get("agentic_graph_relations", [])

    # 高亮差异
    has_graph = len(gentities) > 0 or len(grelations) > 0

    right = f"""<div class="side-right">
        <h3>🤖 Agentic RAG</h3>
        <div class="metric-row">
            <span class="metric-tag blue">🔍 query_type={qtype}</span>
            <span class="metric-tag blue">🔗 graph_weight={gweight:.1f}</span>
            <span class="metric-tag green">📄 {ag_count} sources</span>"""

    if has_graph:
        right += f"""<span class="metric-tag green">🧠 +{len(gentities)} entities, +{len(grelations)} relations</span>"""

    right += f"""<span class="metric-tag">⏱ {ag_time}s</span></div>"""

    if ag_answer:
        right += f"""<div class="answer-box">{ag_answer[:2000]}</div>"""
    elif entry.get("agentic_error"):
        right += f"""<div class="answer-box" style="color:var(--red)">[ERROR] {_escape_html(entry['agentic_error'])}</div>"""
    else:
        right += """<div class="answer-box empty">（无输出）</div>"""

    # 图谱实体/关系
    if gentities:
        right += """<details style="font-size:0.85rem;margin-top:0.5rem"><summary>🧠 知识图谱实体</summary><ul>"""
        for e in gentities[:5]:
            name = _escape_html(str(e.get("name", e.get("id", "?"))))
            etype = _escape_html(str(e.get("type", "")))
            right += f"<li><strong>{name}</strong> ({etype})</li>"
        right += "</ul></details>"

    if grelations:
        right += """<details style="font-size:0.85rem;margin-top:0.5rem"><summary>🔗 知识图谱关系</summary><ul>"""
        for r in grelations[:5]:
            head = _escape_html(str(r.get("head", "?")))
            rel = _escape_html(str(r.get("relation", "?")))
            tail = _escape_html(str(r.get("tail", "?")))
            right += f"<li>{head} → <strong>{rel}</strong> → {tail}</li>"
        right += "</ul></details>"

    # Agentic 步骤
    steps = entry.get("agentic_steps", [])
    if steps:
        right += """<details style="font-size:0.8rem;margin-top:0.5rem"><summary>📋 执行步骤</summary><ul class="steps-list">"""
        for s in steps[-8:]:  # 最近 8 步
            right += f"<li>{_escape_html(str(s))}</li>"
        right += "</ul></details>"

    right += "</div>"

    return f"""<div class="comparison-card">
        <div class="query-title">[{index}] {query}</div>
        <div class="side-by-side">{left}{right}</div>
    </div>"""


def _build_idea_card(entry: dict, index: int) -> str:
    """构建单个 Idea topic 的对比卡片 HTML。"""
    topic = _escape_html(entry["topic"])

    # ── 左侧：常规 Ideas ──
    reg_ideas = entry.get("regular_ideas", [])
    reg_count = entry.get("regular_idea_count", 0)
    reg_time = entry.get("regular_time", 0)

    left = f"""<div class="side-left">
        <h3>📋 Regular Ideas</h3>
        <div class="metric-row">
            <span class="metric-tag">💡 {reg_count} ideas</span>
            <span class="metric-tag">⏱ {reg_time}s</span>
        </div>"""

    if reg_ideas:
        for idea in reg_ideas:
            title = _escape_html(idea.get("title", "?"))
            desc = _escape_html(idea.get("description", ""))[:150]
            feasibility = idea.get("feasibility", 0.5)
            left += f"""<div class="idea-item">
                <div class="idea-title">{title}</div>
                <div style="font-size:0.85rem;color:var(--muted)">{desc}...</div>
                <div style="font-size:0.8rem">可行性: {'★' * int(feasibility * 5)}{'☆' * (5 - int(feasibility * 5))}</div>
            </div>"""
    elif entry.get("regular_error"):
        left += f"""<div class="answer-box" style="color:var(--red)">[ERROR] {_escape_html(entry['regular_error'])}</div>"""
    else:
        left += """<div class="answer-box empty">（无 ideas 生成）</div>"""

    left += "</div>"

    # ── 右侧：Agentic Ideas ──
    ag_ideas = entry.get("agentic_ideas", [])
    ag_count = entry.get("agentic_idea_count", 0)
    ag_time = entry.get("agentic_time", 0)
    confidence = entry.get("agentic_confidence", 0)
    critique = entry.get("agentic_critique", "")
    idea_scores = entry.get("agentic_idea_scores", [])
    iterations = entry.get("agentic_iterations", 0)

    conf_pct = int(confidence * 100)
    conf_class = "high" if confidence >= 0.7 else ("mid" if confidence >= 0.4 else "low")

    right = f"""<div class="side-right">
        <h3>🤖 Agentic Ideas</h3>
        <div class="metric-row">
            <span class="metric-tag green">💡 {ag_count} ideas</span>
            <span class="metric-tag green">🔄 {iterations} iterations</span>
            <span class="metric-tag blue">⏱ {ag_time}s</span>
        </div>
        <div class="metric-row">
            <span class="metric-tag">📊 Confidence: {conf_pct}%</span>
        </div>
        <div class="confidence-bar"><div class="fill {conf_class}" style="width:{conf_pct}%"></div></div>"""

    if critique:
        right += f"""<div class="critique-box"><strong>📋 Critique:</strong> {_render_text(critique)}</div>"""

    if ag_ideas:
        for i, idea in enumerate(ag_ideas):
            title = _escape_html(idea.get("title", "?"))
            desc = _escape_html(idea.get("description", ""))[:150]
            # 查找对应 score
            score_text = ""
            if idea_scores and i < len(idea_scores):
                sc = idea_scores[i]
                score_text = f" | Score: {sc.get('score', '?')}/10"
            right += f"""<div class="idea-item">
                <div class="idea-title">{title}</div>
                <div class="idea-score">{score_text}</div>
                <div style="font-size:0.85rem;color:var(--muted)">{desc}...</div>
            </div>"""
    elif entry.get("agentic_error"):
        right += f"""<div class="answer-box" style="color:var(--red)">[ERROR] {_escape_html(entry['agentic_error'])}</div>"""
    else:
        right += """<div class="answer-box empty">（无 ideas 生成）</div>"""

    # 步骤
    steps = entry.get("agentic_steps", [])
    if steps:
        right += """<details style="font-size:0.8rem;margin-top:0.5rem"><summary>📋 执行步骤</summary><ul class="steps-list">"""
        for s in steps:
            right += f"<li>{_escape_html(str(s))}</li>"
        right += "</ul></details>"

    right += "</div>"

    return f"""<div class="comparison-card">
        <div class="query-title">[{index}] {topic}</div>
        <div class="side-by-side">{left}{right}</div>
    </div>"""


def _build_summary_section(rag_results: list[dict], idea_results: list[dict]) -> str:
    """构建汇总统计区。"""
    # RAG 统计
    rag_with_graph = sum(
        1 for r in rag_results
        if len(r.get("agentic_graph_entities", [])) > 0 or len(r.get("agentic_graph_relations", [])) > 0
    )
    avg_reg_sources = (
        sum(r.get("regular_node_count", 0) for r in rag_results) / len(rag_results)
    ) if rag_results else 0
    avg_ag_sources = (
        sum(r.get("agentic_node_count", 0) for r in rag_results) / len(rag_results)
    ) if rag_results else 0

    # Idea 统计
    avg_confidence = (
        sum(r.get("agentic_confidence", 0) for r in idea_results) / len(idea_results)
    ) if idea_results else 0
    total_iterations = sum(r.get("agentic_iterations", 0) for r in idea_results)

    return f"""<h2>📊 汇总统计</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{len(RAG_QUERIES)}</div>
            <div class="label">RAG 测试查询</div>
        </div>
        <div class="summary-card green">
            <div class="value">{rag_with_graph}/{len(RAG_QUERIES)}</div>
            <div class="label">查询命中知识图谱</div>
        </div>
        <div class="summary-card">
            <div class="value">{avg_reg_sources:.1f} → {avg_ag_sources:.1f}</div>
            <div class="label">平均来源数 (Regular → Agentic)</div>
        </div>
        <div class="summary-card">
            <div class="value">{len(IDEA_TOPICS)}</div>
            <div class="label">Idea 测试主题</div>
        </div>
        <div class="summary-card green">
            <div class="value">{avg_confidence:.0%}</div>
            <div class="label">Agentic 平均置信度</div>
        </div>
        <div class="summary-card green">
            <div class="value">{total_iterations}</div>
            <div class="label">总迭代优化轮次</div>
        </div>
    </div>"""


def _build_takeaways(rag_results: list[dict], idea_results: list[dict]) -> str:
    """构建关键发现区。"""
    # 动态生成关键发现
    bullets = []

    # RAG 发现
    rag_with_graph = sum(
        1 for r in rag_results
        if len(r.get("agentic_graph_entities", [])) > 0 or len(r.get("agentic_graph_relations", [])) > 0
    )
    if rag_with_graph > 0:
        bullets.append(
            f"<strong>知识图谱增强检索</strong>：{rag_with_graph}/{len(RAG_QUERIES)} 个查询通过并行图谱搜索"
            f"获得了额外的结构化知识（实体和关系），补充了纯向量检索无法捕获的语义关联。"
        )

    qtypes = set(r.get("agentic_query_type", "fact") for r in rag_results)
    if len(qtypes) > 1:
        bullets.append(
            f"<strong>智能查询分类</strong>：Agentic Router 自动将查询分类为 {', '.join(qtypes)} "
            f"等类型，并动态调整图谱权重，使不同意图的查询获得差异化的检索策略。"
        )

    # Idea 发现
    avg_confidence = (
        sum(r.get("agentic_confidence", 0) for r in idea_results) / len(idea_results)
    ) if idea_results else 0
    total_iterations = sum(r.get("agentic_iterations", 0) for r in idea_results)

    if total_iterations > 0:
        bullets.append(
            f"<strong>自我审查与迭代优化</strong>：Agentic Idea Engine 共执行 {total_iterations} 轮"
            f" critique→refine 迭代，自动识别缺失证据并补充检索，持续提升 idea 质量。"
        )

    if avg_confidence > 0:
        conf_label = "高" if avg_confidence >= 0.7 else "中"
        bullets.append(
            f"<strong>置信度量化评估</strong>：Agentic 模式为每组 ideas 提供置信度评分"
            f"（平均 {avg_confidence:.0%}，{conf_label}），便于用户判断 idea 的可靠程度。"
        )

    # 来源对比
    reg_total = sum(r.get("regular_node_count", 0) for r in rag_results)
    ag_total = sum(r.get("agentic_node_count", 0) for r in rag_results)
    if ag_total > reg_total:
        bullets.append(
            f"<strong>更丰富的检索来源</strong>：Agentic RAG 总共检索到 {ag_total} 条来源"
            f"（vs Regular {reg_total} 条），并行向量+图谱搜索带来更全面的文献覆盖。"
        )

    if not bullets:
        bullets.append("运行完成。请查看上方对比详情。")

    items = "\n".join(f"<li>{b}</li>" for b in bullets)
    return f"""<h2>💡 关键发现</h2>
    <div class="takeaway"><ul>{items}</ul></div>"""


def generate_html_report(
    rag_results: list[dict],
    idea_results: list[dict],
    config: dict,
    elapsed: float,
) -> str:
    """生成完整 HTML 报告。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 配置 badges
    badges_html = ""
    for key, label in [
        ("enable_agentic_rag", "Agentic RAG"),
        ("enable_agentic_ideas", "Agentic Ideas"),
        ("enable_graph_rag", "Graph RAG"),
    ]:
        val = config.get(key, False)
        cls = "badge-on" if val else "badge-off"
        badges_html += f'<span class="badge {cls}">{label}: {("ON" if val else "OFF")}</span>'

    badges_html += f'<span class="badge badge-info">VLM: Local (llama.cpp)</span>'
    badges_html += f'<span class="badge badge-info">Papers: 90 indexed</span>'

    # RAG cards
    rag_cards = "\n".join(_build_rag_card(r, i + 1) for i, r in enumerate(rag_results))

    # Idea cards
    idea_cards = "\n".join(_build_idea_card(r, i + 1) for i, r in enumerate(idea_results))

    summary = _build_summary_section(rag_results, idea_results)
    takeaways = _build_takeaways(rag_results, idea_results)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PaperRAG Agentic vs Regular — Performance Comparison</title>
<style>{CSS}</style>
</head>
<body>

<div class="header">
    <h1>PaperRAG Agentic vs Regular</h1>
    <p class="subtitle">Performance Comparison Report — 论文知识检索与 Idea 生成</p>
    <p class="subtitle">Generated: {now} | Engine init + {len(RAG_QUERIES)} queries + {len(IDEA_TOPICS)} topics = {elapsed:.1f}s</p>
    <div class="config-badges">{badges_html}</div>
</div>

{summary}

<h2>🔍 Section 1: RAG 检索对比</h2>
<p style="color:var(--muted);margin-bottom:1rem">
    左侧 Regular RAG = 单路向量检索 + LLM 回答；右侧 Agentic RAG = 查询分类 + 向量∥图谱并行检索 + LLM 合成。
</p>
{rag_cards}

<h2>💡 Section 2: Idea 生成对比</h2>
<p style="color:var(--muted);margin-bottom:1rem">
    左侧 Regular = analyze → search → generate 一次性生成；右侧 Agentic = analyze → search → generate → critique → refine 迭代优化。
</p>
{idea_cards}

{takeaways}

<div class="footer">
    Generated by <code>tools/generate_demo_report.py</code> — PaperRAG v1.12.5
</div>

</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

async def main(dry_run: bool = False):
    print("=" * 60)
    print("PaperRAG Agentic vs Regular — Comparison Demo")
    if dry_run:
        print("  🧪 DRY-RUN MODE — no local RAG/VLM/Neo4j calls")
    print("=" * 60)

    # 1. 加载配置（dry-run 模式也加载，用于 HTML 中的 config badges）
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        sys.exit(1)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    print(f"✅ Config loaded: {CONFIG_PATH}")

    # 2. 创建 FakeContext
    context = FakeContext(config)

    if dry_run:
        # Dry-run 模式：跳过所有重资源初始化，直接用 mock 数据跑 HTML 生成
        print("\n⏭️  Skipping engine init (dry-run)")
        print("⏭️  Skipping VLM check (dry-run)")
        print("⏭️  Skipping Neo4j check (dry-run)")
        engine = None  # not needed

        t_total = time.time()

        print(f"\n📊 Running RAG comparison ({len(RAG_QUERIES)} queries)...")
        rag_results = await run_rag_comparison(engine, context, config, dry_run=True)

        print(f"\n📊 Running Idea comparison ({len(IDEA_TOPICS)} topics)...")
        idea_results = await run_idea_comparison(engine, context, dry_run=True)

        elapsed = round(time.time() - t_total, 1)
        print(f"\n✅ All comparisons done in {elapsed}s")

        # 生成 HTML
        print(f"\n📄 Generating HTML report...")
        html = generate_html_report(rag_results, idea_results, config, elapsed)
        OUTPUT_PATH.write_text(html, encoding="utf-8")
        print(f"✅ Report saved: {OUTPUT_PATH}")
        print(f"   File size: {len(html):,} bytes")
        print(f"\n👉 Open in browser: file://{OUTPUT_PATH}")
        return

    # 3. 初始化引擎（直接创建，避免 engine_utils 的相对导入问题）
    print("\n🔧 Initializing HybridRAGEngine...")
    from rag.rag_engine import create_rag_engine, RAGConfig

    rag_config = RAGConfig(
        embedding_mode="unsloth",
        milvus_lite_path=config.get("milvus_lite_path", ""),
        collection_name=config.get("collection_name", "paper_embeddings"),
        embed_dim=config.get("embed_dim", 1024),
        top_k=config.get("top_k", 5),
        similarity_cutoff=config.get("similarity_cutoff", 0.3),
        chunk_size=config.get("chunk_size", 512),
        chunk_overlap=config.get("chunk_overlap", 0),
        min_chunk_size=config.get("min_chunk_size", 100),
        use_semantic_chunking=config.get("use_semantic_chunking", True),
        enable_sparse_retrieval=config.get("enable_sparse_retrieval", True),
        enable_multi_vector_rerank=config.get("enable_multi_vector_rerank", False),
        sparse_top_k=config.get("sparse_top_k", 20),
        hybrid_alpha=config.get("hybrid_alpha", 0.5),
        hybrid_rrf_k=config.get("hybrid_rrf_k", 60),
        enable_bm25=config.get("enable_bm25", True),
        bm25_top_k=config.get("bm25_top_k", 20),
        enable_two_stage_retrieval=bool(config.get("enable_two_stage_retrieval", False)),
        two_stage_top_k=config.get("two_stage_top_k") or 10,
        two_stage_rerank_k=config.get("two_stage_rerank_k") or 5,
        enable_crag_quality_eval=config.get("enable_crag_quality_eval", True),
        crag_enable_correction=config.get("crag_enable_correction", False),
        crag_min_score=config.get("crag_min_score", 0.3),
        enable_graph_rag=config.get("enable_graph_rag", False),
        graph_storage_type=config.get("graph_rag", {}).get("storage_type", "neo4j"),
        graph_neo4j_uri=config.get("graph_rag", {}).get("neo4j_uri", "bolt://localhost:7687"),
        graph_neo4j_user=config.get("graph_rag", {}).get("neo4j_user", "neo4j"),
        graph_neo4j_password=config.get("graph_rag", {}).get("neo4j_password", ""),
        graph_max_triplets_per_chunk=config.get("graph_rag", {}).get("max_triplets_per_chunk", 5),
        graph_retrieval_top_k=config.get("graph_rag", {}).get("graph_retrieval_top_k", 5),
        graph_rrf_weight=config.get("graph_rag", {}).get("graph_rrf_weight", 0.2),
        unsloth_config=config.get("unsloth", {}),
        llama_vlm_model_path=config.get("llama_vlm_model_path", ""),
        llama_vlm_mmproj_path=config.get("llama_vlm_mmproj_path", ""),
        llama_vlm_max_tokens=config.get("llama_vlm_max_tokens", 25600),
        llama_vlm_temperature=config.get("llama_vlm_temperature", 0.7),
        llama_vlm_n_ctx=config.get("llama_vlm_n_ctx", 16384),
        llama_vlm_n_gpu_layers=config.get("llama_vlm_n_gpu_layers", 99),
        enable_multimodal=config.get("multimodal", {}).get("enabled", True),
        figures_dir=config.get("figures_dir", ""),
        papers_dir=config.get("papers_dir", ""),
        enable_llm_reference_parsing=config.get("enable_llm_reference_parsing", True),
        freeapi_url=config.get("freeapi_url", ""),
        freeapi_key=config.get("freeapi_key", ""),
        core_api_key=config.get("core_api_key", ""),
        use_arxiv_api=config.get("use_arxiv_api", True),
        address=config.get("address", ""),
        db_name=config.get("db_name", "default"),
        authentication=config.get("authentication", {}),
    )
    engine = create_rag_engine(rag_config, context)
    if engine is None:
        print("❌ Failed to create HybridRAGEngine")
        sys.exit(1)
    print("✅ Engine initialized")

    # 3.5. Warmup: 执行一次空查询建立所有连接（Milvus, VLM）
    print("\n🔥 Warming up connections...")
    try:
        _ = await engine.search("warmup", mode="retrieve", top_k=1)
        print("   Milvus connection established")
    except Exception as e:
        print(f"   Milvus warmup failed (non-fatal): {e}")
    print("✅ Warmup complete")

    # 4. 检查 local VLM
    provider = await _get_local_vlm_provider()
    if provider:
        print("✅ Local VLM available (llama.cpp)")
    else:
        print("⚠️  Local VLM not available — answers will be empty")

    # 5. 检查 Neo4j
    try:
        from agentic_rag.engine_utils import get_graph_engine
        graph_engine = await get_graph_engine(context, config)
        if graph_engine is not None:
            print("✅ Neo4j Graph Engine available")
        else:
            print("⚠️  Neo4j Graph Engine not available — graph search will skip")
    except Exception as e:
        print(f"⚠️  Neo4j check failed (non-fatal): {e}")

    # 6. 运行对比
    t_total = time.time()

    print(f"\n📊 Running RAG comparison ({len(RAG_QUERIES)} queries)...")
    rag_results = await run_rag_comparison(engine, context, config)

    print(f"\n📊 Running Idea comparison ({len(IDEA_TOPICS)} topics)...")
    idea_results = await run_idea_comparison(engine, context)

    elapsed = round(time.time() - t_total, 1)
    print(f"\n✅ All comparisons done in {elapsed}s")

    # 7. 生成 HTML
    print(f"\n📄 Generating HTML report...")
    html = generate_html_report(rag_results, idea_results, config, elapsed)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"✅ Report saved: {OUTPUT_PATH}")
    print(f"   File size: {len(html):,} bytes")
    print(f"\n👉 Open in browser: file://{OUTPUT_PATH}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry_run))
