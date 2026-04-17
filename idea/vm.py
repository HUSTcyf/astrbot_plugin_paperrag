"""
VLM Provider 与图表过滤
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger

from .citations import IdeaEngineCitations


class IdeaEngineVM(IdeaEngineCitations):
    """VLM Provider 与图表过滤。继承链：... → IdeaEngineCitations → IdeaEngineVM"""

    def _get_vlm_provider(self):
        """获取本地VLM provider（LlamaCppVLMProvider）"""
        try:
            from .llama_cpp_vlm_provider import get_llama_cpp_vlm_provider
        except ImportError as e:
            logger.warning(f"[IdeaEngine] 无法导入 LlamaCppVLMProvider: {e}")
            return None

        try:
            vlm_provider = get_llama_cpp_vlm_provider()
            return vlm_provider
        except Exception as e:
            logger.warning(f"[IdeaEngine] 获取 VLM Provider 失败: {e}")
            return None

    async def _get_vlm_provider_async(self):
        """异步获取并初始化本地VLM provider"""
        vlm_provider = self._get_vlm_provider()
        if vlm_provider is None:
            return None

        if not vlm_provider._initialized:
            logger.info("[IdeaEngine] VLM Provider 未初始化，等待初始化...")
            await vlm_provider.initialize()

        return vlm_provider

    async def _vlm_chat_with_progress(self, vlm_provider, prompt: str, temperature: float, max_tokens: int, task_name: str = "VLM生成") -> str:
        """带进度提示的VLM调用，在推理过程中每10秒输出一次状态"""
        logger.info(f"[IdeaEngine] {task_name}开始，prompt长度: {len(prompt)}")

        async def progress_logger():
            elapsed = 0
            while True:
                await asyncio.sleep(10)
                elapsed += 10
                logger.info(f"[IdeaEngine] {task_name}进行中，已耗时{elapsed}秒...")

        progress_task = asyncio.create_task(progress_logger())

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

            if hasattr(response, 'content'):
                result = response.content
            elif isinstance(response, dict):
                result = response.get("content", "") or response.get("text", "")
            else:
                result = str(response)

            logger.info(f"[IdeaEngine] {task_name}完成，生成{len(result)}字符")
            return result

        except asyncio.CancelledError:
            progress_task.cancel()
            raise

    async def _vlm_describe_images_batch(
        self,
        vlm_provider,
        images: List[Dict[str, int | str]],
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> List[Dict[str, Any]]:
        """批量为图片生成文字描述（VLM fallback）"""
        if not images:
            return []

        image_list = []
        for img in images:
            image_list.append(f"本地图-{img['index']}: {img['filename']}\n  路径: {img['path']}")

        images_section = "\n".join(image_list)
        prompt = f"""你是一个学术图片描述助手。请为以下学术图片生成简短准确的描述文字（1-2句话）。

要求：
1. 直接描述图片内容，不要添加"如图所示"等引导语
2. 使用英文逗号分隔的主要信息描述
3. 不要超过50个字

图片列表：
{images_section}

请按以下JSON格式输出（每张图一行，不要有其他内容）：
{{"index": 1, "caption": "描述文字"}}
{{"index": 2, "caption": "描述文字"}}
"""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                temperature=temperature,
                max_tokens=512 * len(images),
            )
            raw = ""
            if hasattr(response, 'content'):
                raw = response.content
            elif isinstance(response, dict):
                raw = response.get("content", "") or response.get("text", "")
            else:
                raw = str(response)

            results: List[Dict[str, int | str]] = []
            for line in raw.strip().split('\n'):
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    obj = json.loads(line)
                    if "index" in obj and "caption" in obj:
                        results.append({"index": int(obj["index"]), "caption": obj["caption"]})
                except (json.JSONDecodeError, ValueError):
                    continue

            logger.info(f"[IdeaEngine] VLM 生成了 {len(results)} 个图片描述")
            return results

        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 图片描述失败: {e}")
            return []

    async def _filter_figures_by_relevance(
        self,
        local_results: List[Dict[str, Any]],
        relevance_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """图表预过滤：召回 chunk 关联论文中的所有图/表，用 rerank 选取最相关的。"""
        logger.info(f"[IdeaEngine] 图表预过滤开始，输入 {len(local_results)} 条结果")

        paper_figure_types: Dict[str, set] = {}
        paper_chunk_texts: Dict[str, str] = {}
        for result in local_results:
            metadata = result.get("metadata", {})
            image_path = metadata.get("image_path", "")
            if not image_path:
                continue
            fname = Path(image_path).name
            if fname.startswith(("Figure", "figure")):
                img_type = "Figure"
            elif fname.startswith(("Table", "table")):
                img_type = "Table"
            else:
                continue
            paper = result.get("paper", metadata.get("file_name", "Unknown"))
            if paper.endswith('.pdf'):
                paper = paper[:-4]
            if paper not in paper_figure_types:
                paper_figure_types[paper] = set()
                paper_chunk_texts[paper] = ""
            paper_figure_types[paper].add(img_type)
            chunk_text = result.get("text", "")
            if chunk_text and paper_chunk_texts[paper]:
                paper_chunk_texts[paper] += "\n"
            paper_chunk_texts[paper] += chunk_text

        if not paper_figure_types:
            logger.warning("[IdeaEngine] 没有找到关联的图表，跳过图表过滤")
            return []

        logger.info(f"[IdeaEngine] 找到 {len(paper_figure_types)} 篇关联论文: {list(paper_figure_types.keys())}")

        all_candidates: List[Dict[str, Any]] = []
        for paper, img_types in paper_figure_types.items():
            captions_data = self._load_captions_by_paper(paper)
            if not captions_data:
                logger.warning(f"[IdeaEngine] 论文无 caption 数据: {paper}")
                continue
            chunk_text = paper_chunk_texts.get(paper, "")

            media_base = Path(__file__).parent.parent / "data"
            figure_base = media_base / "figures"
            table_base = media_base / "tables"
            actual_folder: Optional[Path] = None
            for base in [figure_base, table_base]:
                if not base.exists():
                    continue
                for folder in base.iterdir():
                    if folder.is_dir() and folder.name.startswith(paper):
                        actual_folder = folder
                        break
                if actual_folder:
                    break

            for key, info in captions_data.items():
                caption = info.get("caption", "")
                filename = info.get("filename", "")
                page = info.get("page", "")
                is_figure = any(t == "Figure" for t in img_types) and "Figure" in key
                is_table = any(t == "Table" for t in img_types) and "Table" in key
                if not (is_figure or is_table):
                    continue
                img_dir = actual_folder if actual_folder else (figure_base if is_figure else table_base) / paper
                full_path = str(img_dir / filename) if filename else ""
                if not full_path or not Path(full_path).exists():
                    continue
                all_candidates.append({
                    "image_path": full_path,
                    "image_caption": caption or filename,
                    "paper": paper,
                    "page": page,
                    "chunk_text": chunk_text,
                    "result": None,
                    "caption": caption,
                    "filename": filename,
                })

        if not all_candidates:
            logger.warning("[IdeaEngine] 没有找到候选图表")
            return []

        logger.info(f"[IdeaEngine] 共有 {len(all_candidates)} 个候选图表，使用 ColBERT 重排序...")

        query = "相关研究内容：" + "\n".join(paper_chunk_texts.values())
        candidates_for_rerank = [
            {"text": c.get("caption", "") or c.get("filename", ""), "metadata": c, "score": 0.5}
            for c in all_candidates
        ]

        try:
            from ..embedding.unsloth_embedding import get_embedding_model
            model = get_embedding_model()
            doc_texts = [ca["text"] for ca in candidates_for_rerank]
            reranked_indices = model.colbert_rerank(query, doc_texts, top_k=len(doc_texts))
            reranked = []
            for idx, score in reranked_indices:
                reranked.append({"metadata": candidates_for_rerank[idx]["metadata"], "score": score})
            logger.info(f"[IdeaEngine] ColBERT rerank 完成，{len(reranked)} 个候选")
        except Exception as e:
            logger.warning(f"[IdeaEngine] ColBERT rerank 失败: {e}，使用原始顺序")
            reranked = [{"metadata": c, "score": 0.5} for c in all_candidates]

        filtered_images = []
        for item in reranked:
            score = item.get("score", 0.5)
            if score < relevance_threshold:
                continue
            c = item.get("metadata", {})
            filtered_images.append({
                "image_path": c.get("image_path", ""),
                "image_caption": c.get("image_caption", ""),
                "image_description": c.get("caption", ""),
                "image_score": score,
                "text_score": 0.5,
                "caption_richness": 1.0 if c.get("caption") else 0.3,
                "paper": c.get("paper", ""),
                "page": c.get("page", ""),
                "text": c.get("chunk_text", ""),
                "result": c.get("result"),
            })

        logger.info(f"[IdeaEngine] 图表预过滤完成（threshold={relevance_threshold}），返回 {len(filtered_images)} 张图片")
        return filtered_images

    def _load_captions_by_paper(self, paper_name: str) -> Dict[str, Any]:
        """加载指定论文的所有图表 caption 信息（自动查找匹配的 JSON 文件）。"""
        import json
        captions_dir = Path(__file__).parent.parent / "data" / "captions"
        if not captions_dir.exists():
            logger.warning(f"[IdeaEngine] caption 目录不存在: {captions_dir}")
            return {}

        exact_path = captions_dir / f"{paper_name}.json"
        if exact_path.exists():
            caption_file = exact_path
        else:
            matches = [f for f in captions_dir.iterdir() if f.name.startswith(paper_name) and f.suffix == ".json"]
            if matches:
                caption_file = matches[0]
                logger.debug(f"[IdeaEngine] caption 文件前缀匹配: {paper_name} -> {caption_file.name}")
            else:
                logger.warning(f"[IdeaEngine] 未找到 caption 文件: {paper_name}")
                return {}

        try:
            with open(caption_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"[IdeaEngine] caption JSON 解析失败: {caption_file}, error: {e}")
            return {}
        except OSError as e:
            logger.warning(f"[IdeaEngine] caption 文件读取失败: {caption_file}, error: {e}")
            return {}

    async def _extract_text_from_image(self, vlm_provider, image_path: str) -> str:
        """使用 VLM 从图片中提取文字描述并判断图片类型"""
        prompt = """请仔细阅读这张学术图片，按以下步骤处理：

**第一步：判断图片类型**
这是学术科研场景的图片，请先判断图片属于以下哪种类型：
- 表格（Table）：包含行列数据的表格
- 架构图（Architecture）：网络结构、系统架构、模型框架
- 方法图（Method）：算法流程、技术路线、步骤说明
- 统计分析图（Statistics）：柱状图、折线图、散点图、饼图等

**第二步：提取文字**
提取图片中所有可见的文字内容，包括：
1. 图表标题和副标题
2. 坐标轴标签和刻度
3. 图例说明
4. 公式和符号
5. 表格内容
6. 任何其他可见文字

**输出格式**：
如果图片有文字：
[图片类型] 提取的文字内容

如果图片无文字：
[图片类型] 无文字 - 简要描述图片内容（1-2句话）

请直接输出，不要解释。"""

        try:
            response = await vlm_provider.text_chat(
                prompt=prompt,
                image_urls=[image_path],
                temperature=0.1,
                max_tokens=512
            )

            if response and hasattr(response, 'content'):
                return response.content.strip()
            return ""
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM 提取文字失败: {image_path}, {e}")
            return ""
