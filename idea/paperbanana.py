"""
PaperBanana 图像生成（独立模块）
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional

from astrbot.api import logger

from .generation import IdeaEngineGeneration
from .websearch import IdeaEngineWebSearch


class IdeaEnginePaperBanana(IdeaEngineGeneration, IdeaEngineWebSearch):
    """PaperBanana 图像生成。继承链：... → IdeaEngineGeneration → IdeaEnginePaperBanana"""

    async def _generate_method_figures_with_paperbanana(self, ideas: List) -> List[Dict]:
        """调用 PaperBanana 服务生成方法图，返回飞书图片块（基于 ideas 列表）"""
        blocks = []
        if not ideas:
            return blocks
        for idea in ideas:
            method_text = getattr(idea, 'methodology', '') or ''
            title_text = getattr(idea, 'title', '') or ''
            combined = f"{title_text}\n{method_text}".strip()
            if not combined:
                continue
            try:
                img_path = await self._call_paperbanana(combined)
                if img_path:
                    blocks.append(self._make_image_block(img_path, title_text or "方法图"))
            except Exception as e:
                logger.warning(f"[IdeaEngine] PaperBanana 生成失败: {e}")
        return blocks

    async def _generate_method_figures_with_paperbanana_from_text(
        self, method_text: str, topic: str, caption: Optional[str] = None
    ) -> List[Dict]:
        """调用 PaperBanana 服务生成方法图（基于完整方法论文本）"""
        blocks = []
        if not method_text:
            return blocks
        figure_caption = caption or f"Methodology: {topic}"
        try:
            image_path = await self._call_paperbanana(
                method_text=method_text,
                figure_caption=figure_caption
            )
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                blocks.append({
                    "blockType": "image",
                    "options": {
                        "image": {
                            "base64": img_base64,
                            "caption": figure_caption
                        }
                    }
                })
                logger.info(f"[IdeaEngine] 方法图生成成功: {topic[:30]}")
        except Exception as e:
            logger.warning(f"[IdeaEngine] 方法图生成失败 [{topic[:20]}]: {e}")
        return blocks

    async def _call_paperbanana(
        self, method_text: str, figure_caption: str = ""
    ) -> Optional[str]:
        """调用 PaperBanana 服务获取图片路径"""
        if not method_text:
            return None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.paperbanana.com/generate",
                    json={"text": method_text},
                    headers={"Authorization": f"Bearer {os.environ.get('PAPERBANANA_TOKEN', '')}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    first = data[0] if isinstance(data, list) else data
                    if isinstance(first, dict) and "image" in first:
                        img_path = first["image"]
                        if os.path.exists(img_path):
                            return img_path
            return None
        except Exception as e:
            logger.warning(f"[IdeaEngine] PaperBanana 调用失败: {e}")
            return None
