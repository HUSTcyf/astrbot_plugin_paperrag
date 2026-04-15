"""
引用上下文构建与媒体 caption 增强
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import logger

from .markdown import IdeaEngineMarkdown


class IdeaEngineCitations(IdeaEngineMarkdown):
    """引用上下文构建与媒体 caption 增强。继承链：... → IdeaEngineMarkdown → IdeaEngineCitations"""

    def _build_citations_context(self, knowledge: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        构建引用上下文，包含本地检索和网络搜索的结果

        Args:
            knowledge: 知识检索结果

        Returns:
            Tuple[str, Dict]: (格式化的引用上下文字符串, 提取的媒体资源字典)
                媒体资源格式: {
                    "images": [{"index": "本地图-1", "path": str, "base64": str, "caption": str}, ...],
                    "tables": [{"index": "本地表-1", "csv_path": str, "png_path": str, "caption": str, "csv_content": str}, ...]
                }
        """
        if not knowledge:
            return "（无可用引用来源）", {"images": [], "tables": []}

        parts: List[str] = []
        extracted_media: Dict[str, Any] = {"images": [], "tables": []}
        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])

        # 本地检索引用
        local_image_idx = 0
        local_table_idx = 0
        if local_results:
            parts.append("## 本地论文检索引用：\n")
            for i, result in enumerate(local_results[:8], 1):
                paper = result.get("paper", "Unknown")
                page = result.get("page", "N/A")
                text = result.get("text", "")[:300]
                score = result.get("score", 0.0)
                metadata = result.get("metadata", {})
                file_name = metadata.get("file_name", "")

                arxiv_id = ""
                if file_name:
                    match = re.match(r'^(\d{4}\.\d{4,})', file_name)
                    if match:
                        arxiv_id = match.group(1)

                image_path = metadata.get("image_path")
                img_index = None
                img_base64 = None
                img_caption = None
                if image_path:
                    if os.path.exists(image_path):
                        local_image_idx += 1
                        img_index = f"本地图-{local_image_idx}"
                        img_caption = metadata.get("image_caption", f"图 {local_image_idx}")
                        try:
                            import base64
                            with open(image_path, "rb") as f:
                                img_base64 = base64.b64encode(f.read()).decode("utf-8")
                            extracted_media["images"].append({
                                "index": img_index,
                                "path": image_path,
                                "base64": img_base64,
                                "caption": img_caption,
                                "source_paper": paper,
                                "source_page": page
                            })
                        except Exception as e:
                            logger.warning(f"[IdeaEngine] 读取图片失败 {image_path}: {e}")
                            img_base64 = None
                    else:
                        logger.warning(f"[IdeaEngine] 图片元数据存在但文件缺失: {image_path}")

                if arxiv_id:
                    ref_str = f"{paper} (https://arxiv.org/abs/{arxiv_id})"
                else:
                    ref_str = paper
                if img_index:
                    parts.append(f"- {ref_str} (页码: {page}, 相关度: {score:.3f}, 图片: {img_index})\n")
                    if img_base64:
                        parts.append(f"  - 图片说明: {img_caption}\n")
                else:
                    parts.append(f"- {ref_str} (页码: {page}, 相关度: {score:.3f})\n")

                table_csv_path = metadata.get("table_csv_path")
                table_png_path = metadata.get("table_png_path")
                table_caption = metadata.get("table_caption", "")

                table_md_path = ""
                if table_csv_path:
                    table_md_path = table_csv_path.replace(".csv", ".md")
                if not table_png_path and table_csv_path:
                    table_png_path = table_csv_path.replace(".csv", ".png")
                    if not os.path.exists(table_png_path):
                        table_png_path = ""

                if table_csv_path or table_png_path:
                    local_table_idx += 1
                    tbl_index = f"本地表-{local_table_idx}"
                    csv_content = ""
                    if table_csv_path:
                        if not os.path.exists(table_csv_path):
                            logger.warning(f"[IdeaEngine] 表格CSV元数据存在但文件缺失: {table_csv_path}")
                        else:
                            try:
                                with open(table_csv_path, "r", encoding="utf-8") as f:
                                    csv_content = f.read()[:500]
                                extracted_media["tables"].append({
                                    "index": tbl_index,
                                    "csv_path": table_csv_path,
                                    "png_path": table_png_path,
                                    "md_path": table_md_path if os.path.exists(table_md_path) else "",
                                    "caption": table_caption,
                                    "csv_content": csv_content,
                                    "source_paper": paper,
                                    "source_page": page
                                })
                            except Exception as e:
                                logger.warning(f"[IdeaEngine] 读取表格失败 {table_csv_path}: {e}")
                                csv_content = "(无法读取)"
                    if table_png_path and not os.path.exists(table_png_path):
                        logger.warning(f"[IdeaEngine] 表格PNG元数据存在但文件缺失: {table_png_path}")

                    if not any(t["index"] == tbl_index for t in extracted_media["tables"]):
                        extracted_media["tables"].append({
                            "index": tbl_index,
                            "csv_path": table_csv_path or "",
                            "png_path": table_png_path or "",
                            "md_path": table_md_path if os.path.exists(table_md_path) else "",
                            "caption": table_caption,
                            "csv_content": csv_content,
                            "source_paper": paper,
                            "source_page": page
                        })
                    parts.append(f"    └─ 包含表格: {tbl_index} - {table_caption}\n")

                parts.append(f"    摘要: {text}...\n\n")

        if web_results:
            parts.append(f"## 网络搜索引用：\n")
            for i, result in enumerate(web_results[:5], 1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                snippet = result.get("snippet", "")[:300]
                link_str = f"[{title}]({url})" if url else title
                parts.append(f"- {link_str}\n")
                parts.append(f"  - 摘要: {snippet}...\n\n")

        if not parts:
            return "（无可用引用来源）", {"images": [], "tables": []}

        return "\n".join(parts), extracted_media

    def _is_simple_caption(self, caption: str) -> bool:
        """检查 caption 是否只是简单的编号而没有实际描述"""
        if not caption:
            return True
        patterns = [
            r'^图\s*\d+$',
            r'^Figure\s*\d+$',
            r'^fig\.\s*\d+$',
            r'^\d+$',
        ]
        for pattern in patterns:
            if re.match(pattern, caption.strip(), re.IGNORECASE):
                return True
        return False

    async def _enhance_media_captions(self, extracted_media: Dict, knowledge: Dict) -> Dict:
        """增强媒体 caption（简化版直接返回）"""
        return extracted_media
