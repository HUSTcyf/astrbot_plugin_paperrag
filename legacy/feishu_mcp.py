"""
Legacy: Feishu MCP 文档创建流水线（已废弃，保留备查）

原位置:
  - idea/feishu_doc.py: test_feishu_markdown_formats, MCP fallback in create_feishu_document
  - idea/markdown.py: _markdown_to_feishu_blocks, _convert_paren_paths_to_markdown,
                      _extract_inline_images, _make_image_block
  - idea/utils.py: _get_feishu_tool

当前方案: 仅使用 lark-cli 创建飞书文档。
"""

import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from astrbot.api import logger

# ── Reusable imports (kept in active codebase) ──
from idea.utils import strip_markdown_style, parse_inline_styles


# ═══════════════════════════════════════════════════════════════════════════════
# From idea/utils.py: _get_feishu_tool
# ═══════════════════════════════════════════════════════════════════════════════

def _get_feishu_tool(context):
    """获取飞书 MCP 工具（通过 provider_manager.llm_tools 查找）"""
    if not context:
        logger.error("[Legacy MCP] context 为 None")
        return None
    provider_manager = getattr(context, 'provider_manager', None)
    if not provider_manager:
        logger.error("[Legacy MCP] provider_manager 为 None")
        return None
    llm_tools = getattr(provider_manager, 'llm_tools', None)
    if not llm_tools:
        logger.error("[Legacy MCP] llm_tools 为 None")
        return None
    func_list = getattr(llm_tools, 'func_list', [])
    for tool in func_list:
        if 'feishu' in tool.name.lower():
            return tool
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# From idea/markdown.py: markdown → feishu block 转换
# ═══════════════════════════════════════════════════════════════════════════════

_IMG_MD_RE = re.compile(r'!\[(.*?)\]\(([/][^)]+\.(?:png|jpg|jpeg|webp|gif))\)')


def _convert_paren_paths_to_markdown(text: str) -> str:
    """将文本中的裸图片路径 (path) 转换为 markdown 图片语法 ![](path)"""
    converted_count = 0
    _EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'PNG', 'JPG', 'JPEG', 'WEBP', 'GIF']

    for ext in _EXTENSIONS:
        end_markers = [f'.{ext})', f'.{ext}）']
        for em in end_markers:
            idx = 0
            while True:
                idx = text.find(em, idx)
                if idx < 0:
                    break
                end_pos = idx + len(em) - 1
                if end_pos < 0:
                    idx += 1
                    continue
                paren_count = 1
                j = end_pos - 1
                while j >= 0 and paren_count > 0:
                    ch = text[j]
                    if ch in (')', '）'):
                        paren_count += 1
                    elif ch in ('(', '（'):
                        paren_count -= 1
                    j -= 1
                if paren_count == 0 and j >= 0:
                    path = text[j + 2:end_pos]
                    if path.startswith('/') and os.path.exists(path):
                        text = text[:j + 1] + f'![]({path})' + text[end_pos + 1:]
                        converted_count += 1
                        idx = j + len(f'![]({path})')
                        continue
                idx += 1

    return text


def _extract_inline_images(text: str) -> List[Dict[str, str]]:
    """从文本中分离行内图片和文本"""
    segments = []
    text_after_preprocess = _convert_paren_paths_to_markdown(text)
    pos = 0
    for m in _IMG_MD_RE.finditer(text_after_preprocess):
        if m.start() > pos:
            segments.append({"type": "text", "content": text_after_preprocess[pos:m.start()]})
        caption = m.group(1)
        path = m.group(2)
        if os.path.exists(path):
            segments.append({"type": "image", "path": path, "caption": caption})
        else:
            decoded = unquote(path)
            if os.path.exists(decoded):
                segments.append({"type": "image", "path": decoded, "caption": caption})
        pos = m.end()
    if pos < len(text_after_preprocess):
        segments.append({"type": "text", "content": text_after_preprocess[pos:]})
    if not segments:
        segments.append({"type": "text", "content": text})
    return segments


def _make_image_block(image_path: str, caption: str = "") -> Optional[Dict]:
    """根据本地图片路径构造飞书图片块"""
    try:
        if not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "blockType": "image",
            "options": {
                "image": {
                    "base64": img_base64,
                    "caption": caption,
                    "image_path": image_path
                }
            }
        }
    except Exception as e:
        logger.warning(f"[Legacy MCP] 读取图片失败 {image_path}: {e}")
        return None


def _markdown_to_feishu_blocks(markdown_text: str) -> List[Dict]:
    """将 Markdown 文本转换为飞书块格式（MCP 协议）"""
    blocks = []
    lines = markdown_text.split("\n")

    for line in lines:
        line = line.rstrip()

        if line.startswith("# ") and not line.startswith("## "):
            content = strip_markdown_style(line[2:].strip())
            blocks.append({
                "blockType": "heading",
                "options": {"heading": {"level": 1, "content": content}}
            })
        elif line.startswith("## ") and not line.startswith("### "):
            content = strip_markdown_style(line[3:].strip())
            blocks.append({
                "blockType": "heading",
                "options": {"heading": {"level": 2, "content": content}}
            })
        elif line.startswith("### "):
            content = strip_markdown_style(line[4:].strip())
            blocks.append({
                "blockType": "heading",
                "options": {"heading": {"level": 3, "content": content}}
            })
        elif line.startswith("---"):
            blocks.append({
                "blockType": "text",
                "options": {"text": {"textStyles": [{"text": "─────────────────────────────────", "style": {}}]}}
            })
        elif line.startswith("- ") or line.startswith("* "):
            raw_content = line[2:].strip()
            if raw_content:
                blocks.append({
                    "blockType": "list",
                    "options": {"list": {"content": raw_content, "isOrdered": False}},
                    "_textStyles": parse_inline_styles(raw_content)
                })
        elif re.match(r'^\d+[\.\)]\s', line):
            match = re.match(r'^(\d+[\.\)])\s+(.*)$', line)
            if match:
                raw_content = match.group(2).strip()
                if raw_content:
                    blocks.append({
                        "blockType": "list",
                        "options": {"list": {"content": raw_content, "isOrdered": True}},
                        "_textStyles": parse_inline_styles(raw_content)
                    })
        elif line.strip() == "":
            pass
        else:
            text_content = line.strip()
            if text_content:
                segments = _extract_inline_images(text_content)
                text_parts = []
                image_blocks = []
                for seg in segments:
                    if seg["type"] == "text":
                        text_parts.append(seg["content"])
                    elif seg["type"] == "image":
                        img_block = _make_image_block(seg["path"], seg["caption"])
                        if img_block is not None:
                            image_blocks.append(img_block)
                merged_text = "".join(text_parts)
                if merged_text.strip():
                    blocks.append({
                        "blockType": "text",
                        "options": {"text": {"textStyles": parse_inline_styles(merged_text)}}
                    })
                blocks.extend(image_blocks)

    return blocks


# ═══════════════════════════════════════════════════════════════════════════════
# From idea/feishu_doc.py: test_feishu_markdown_formats
# ═══════════════════════════════════════════════════════════════════════════════

async def test_feishu_markdown_formats(engine, folder_token: str = "") -> Dict[str, Any]:
    """测试用：列表样式 + 图片插入 + 引用链接（MCP 路径）"""
    from astrbot.core.agent.run_context import ContextWrapper
    ctx_wrapper = ContextWrapper(context=engine.context)

    provider_manager = getattr(engine.context, 'provider_manager', None)
    if not provider_manager:
        return {"success": False, "error": "provider_manager 不可用"}

    llm_tools = getattr(provider_manager, 'llm_tools', None)

    feishu_tool = add_blocks_tool = upload_image_tool = update_text_tool = get_blocks_tool = None
    if llm_tools:
        for tool in getattr(llm_tools, 'func_list', []):
            if tool.name == 'create_feishu_document':
                feishu_tool = tool
            elif tool.name == 'batch_create_feishu_blocks':
                add_blocks_tool = tool
            elif tool.name == 'upload_and_bind_image_to_block':
                upload_image_tool = tool
            elif tool.name == 'batch_update_feishu_block_text':
                update_text_tool = tool
            elif tool.name == 'get_feishu_document_blocks':
                get_blocks_tool = tool

    if not feishu_tool or not add_blocks_tool:
        return {"success": False, "error": "缺少必要工具"}

    draft_path = "/Users/chenyifeng/AstrBot/data/plugin_data/astrbot_plugin_paperrag/ideas/8a160941c48c813c/initial_draft.md"
    try:
        with open(draft_path, "r", encoding="utf-8") as f:
            test_markdown = f.read()
        test_markdown = unquote(test_markdown)
    except Exception as e:
        return {"success": False, "error": f"读取文件失败: {e}"}

    all_blocks = _markdown_to_feishu_blocks(test_markdown)
    image_count = sum(1 for b in all_blocks if b.get("blockType") == "image")

    create_result = await feishu_tool.call(ctx_wrapper, title="[测试] 列表样式+图片+引用", folderToken=folder_token or "")

    doc_info = {}
    if hasattr(create_result, 'content') and create_result.content:
        result_text = getattr(create_result.content[0], 'text', None) or str(create_result.content[0])
        try:
            doc_info = json.loads(result_text)
        except json.JSONDecodeError:
            pass

    document_id = (
        doc_info.get("document", {}).get("document_id")
        or doc_info.get("document_id")
        or doc_info.get("objToken")
        or doc_info.get("obj_token")
    )
    if not document_id:
        return {"success": False, "error": f"文档创建失败: {create_result}"}

    images_uploaded = 0
    current_index = 0
    text_batch: list = []
    batch_start_index = 0
    list_items_to_update: list = []

    async def flush_batch():
        nonlocal text_batch, batch_start_index
        if not text_batch:
            return
        result = await add_blocks_tool.call(
            ctx_wrapper, documentId=document_id,
            parentBlockId=document_id, index=batch_start_index, blocks=text_batch
        )
        text_batch = []

    for b in all_blocks:
        if b.get("blockType") == "image":
            await flush_batch()
            opts = b.get("options", {}).get("image", {})
            img_path = opts.get("image_path", "")
            img_base64 = opts.get("base64", "")
            is_temp_file = False
            if not img_path and img_base64:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(base64.b64decode(img_base64))
                tmp.close()
                img_path = tmp.name
                is_temp_file = True

            if img_path and os.path.exists(img_path):
                img_path = engine._ensure_png(img_path)
                img_caption = opts.get("caption", "")
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(img_path) as pil_img:
                        orig_w, orig_h = pil_img.size
                    img_width, img_height = orig_w, orig_h
                except Exception:
                    img_width, img_height = 768, 768
                img_result = await add_blocks_tool.call(
                    ctx_wrapper, documentId=document_id,
                    parentBlockId=document_id, index=current_index,
                    blocks=[{"blockType": "image", "align": 2, "options": {"image": {}}}]
                )
                image_block_id = None
                try:
                    if hasattr(img_result, 'content') and img_result.content:
                        r_text = getattr(img_result.content[0], 'text', None) or str(img_result.content[0])
                        r_data = json.loads(r_text)
                        image_info = r_data.get('imageBlocksInfo', {})
                        if isinstance(image_info, dict):
                            block_ids = image_info.get('blockIds', [])
                            if block_ids:
                                image_block_id = block_ids[0]
                except Exception:
                    pass

                if image_block_id and upload_image_tool:
                    upload_res = await upload_image_tool.call(
                        ctx_wrapper, documentId=document_id,
                        images=[{"blockId": image_block_id, "imagePathOrUrl": img_path}]
                    )
                    if upload_res and not getattr(upload_res, 'isError', True):
                        images_uploaded += 1
                        if img_caption and add_blocks_tool:
                            caption_block = [{
                                "blockType": "text",
                                "options": {
                                    "text": {
                                        "textStyles": [{"text": img_caption, "style": {"bold": True, "text_color": 7}}],
                                        "align": 2
                                    }
                                }
                            }]
                            await add_blocks_tool.call(
                                ctx_wrapper, documentId=document_id,
                                parentBlockId=document_id, index=current_index + 1,
                                blocks=caption_block
                            )
                            current_index += 1

                if is_temp_file and os.path.exists(img_path):
                    try:
                        os.unlink(img_path)
                    except OSError:
                        pass

            current_index += 1
            batch_start_index = current_index
        else:
            if b.get("blockType") == "list" and b.get("_textStyles"):
                list_content = b.get("options", {}).get("list", {}).get("content", "")
                list_items_to_update.append((list_content, b.get("_textStyles") or {}))
            text_batch.append(b)
            current_index += 1

    await flush_batch()

    # Update list styles
    updated_lists = 0
    if list_items_to_update and update_text_tool and get_blocks_tool:
        try:
            blocks_result = await get_blocks_tool.call(ctx_wrapper, documentId=document_id)
            blocks_text = ""
            if hasattr(blocks_result, 'content') and blocks_result.content:
                blocks_text = getattr(blocks_result.content[0], 'text', '') or str(blocks_result.content[0])
            all_doc_blocks = []
            try:
                if blocks_text:
                    decoder = json.JSONDecoder()
                    all_doc_blocks, _ = decoder.raw_decode(blocks_text)
            except Exception:
                pass

            def _normalize_text(t: str) -> str:
                return re.sub(r'\s+', ' ', t).strip()

            updates = []
            matched_block_ids = set()
            for list_text, text_styles in list_items_to_update:
                norm_list_text = _normalize_text(list_text)
                for block in all_doc_blocks:
                    block_id = block.get("block_id", "")
                    if block_id in matched_block_ids:
                        continue
                    block_type = block.get("block_type", 0)
                    if block_type not in (12, 13):
                        continue
                    block_data = block.get("bullet") or block.get("ordered") or {}
                    elements = block_data.get("elements", [])
                    block_text = ""
                    for elem in elements:
                        tr = elem.get("text_run", {})
                        if tr.get("content"):
                            block_text += tr["content"]
                    if _normalize_text(block_text) == norm_list_text:
                        matched_block_ids.add(block_id)
                        text_elements = []
                        for ts in text_styles:
                            if ts.get("equation"):
                                text_elements.append({"equation": ts["equation"], "style": ts.get("style", {})})
                            else:
                                text_elements.append({"text": ts.get("text", ""), "style": ts.get("style", {})})
                        updates.append({"blockId": block_id, "textElements": text_elements})
                        break

            if updates:
                upd_result = await update_text_tool.call(ctx_wrapper, documentId=document_id, updates=updates)
                if not (hasattr(upd_result, 'isError') and upd_result.isError):
                    updated_lists = len(updates)
        except Exception:
            pass

    url = f"https://feishu.cn/docx/{document_id}"
    return {
        "success": True, "document_id": document_id, "url": url,
        "blocks_created": len(all_blocks), "image_count": images_uploaded,
        "list_styles_updated": updated_lists,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# From idea/feishu_doc.py: MCP fallback in create_feishu_document
# ═══════════════════════════════════════════════════════════════════════════════

async def create_feishu_document_mcp(
    engine,
    *,
    clean_text: str,
    image_infos: list,
    generated_title: str,
    folder_token: str,
    weekly_report: str,
    provider_manager,
) -> Dict[str, Any]:
    """MCP 回退路径：创建空文档 → 手动插入文本块 → 上传图片。

    在原 create_feishu_document 中当 lark-cli 不可用时调用。
    """
    from astrbot.core.agent.run_context import ContextWrapper
    ctx_wrapper = ContextWrapper(context=engine.context)

    feishu_tool = _get_feishu_tool(engine.context)
    if not feishu_tool:
        return {"error": "未找到飞书 MCP 工具，且 lark-cli 不可用", "polished_content": weekly_report}

    # 1. 创建空文档
    create_result = await feishu_tool.call(ctx_wrapper, title=generated_title, folderToken=folder_token)
    result_text = ""
    if hasattr(create_result, 'content') and create_result.content:
        result_text = getattr(create_result.content[0], 'text', None) or str(create_result.content[0])

    if "请在浏览器打开以下链接进行授权" in result_text or "authorize" in result_text.lower():
        auth_url_match = re.search(r'https://accounts\.feishu\.cn/open-apis/authen/v1/authorize\S+', result_text)
        auth_url = auth_url_match.group(0) if auth_url_match else ""
        return {
            "error": "飞书授权已过期，请在浏览器中打开授权链接完成授权后重试",
            "auth_url": auth_url,
            "polished_content": weekly_report,
        }

    doc_info = {}
    try:
        doc_info = json.loads(result_text)
    except json.JSONDecodeError:
        pass

    document_id = (
        doc_info.get("document", {}).get("document_id")
        or doc_info.get("document_id")
        or doc_info.get("objToken")
        or doc_info.get("obj_token")
    )
    if not document_id:
        return {"error": f"文档创建失败: {result_text[:300]}", "polished_content": weekly_report}

    # 2. 转换并插入文本块
    add_blocks_tool = update_text_tool = get_blocks_tool = None
    if provider_manager:
        llm_tools = getattr(provider_manager, 'llm_tools', None)
        if llm_tools:
            for tool in getattr(llm_tools, 'func_list', []):
                if tool.name == 'batch_create_feishu_blocks':
                    add_blocks_tool = tool
                elif tool.name == 'batch_update_feishu_block_text':
                    update_text_tool = tool
                elif tool.name == 'get_feishu_document_blocks':
                    get_blocks_tool = tool

    if add_blocks_tool and clean_text:
        all_blocks = _markdown_to_feishu_blocks(clean_text)

        text_blocks = []
        list_items_to_update = []
        for b in all_blocks:
            if b.get("blockType") == "image":
                opts = b.get("options", {}).get("image", {})
                img_path = opts.get("image_path", "")
                if img_path:
                    image_infos.append({
                        "path": img_path,
                        "caption": opts.get("caption", ""),
                        "anchor": "",
                        "width": None, "height": None, "base64": opts.get("base64", ""),
                    })
            else:
                if b.get("blockType") == "list" and b.get("_textStyles"):
                    list_content = b.get("options", {}).get("list", {}).get("content", "")
                    list_items_to_update.append((list_content, b.get("_textStyles") or {}))
                text_blocks.append(b)

        CHUNK_SIZE = 20
        for chunk_start in range(0, len(text_blocks), CHUNK_SIZE):
            chunk = text_blocks[chunk_start:chunk_start + CHUNK_SIZE]
            await add_blocks_tool.call(
                ctx_wrapper, documentId=document_id,
                parentBlockId=document_id, index=chunk_start, blocks=chunk
            )

        # Update list block styles
        if list_items_to_update and update_text_tool and get_blocks_tool:
            try:
                blocks_result = await get_blocks_tool.call(ctx_wrapper, documentId=document_id)
                blocks_text = ""
                if hasattr(blocks_result, 'content') and blocks_result.content:
                    blocks_text = getattr(blocks_result.content[0], 'text', '') or str(blocks_result.content[0])
                all_doc_blocks = []
                try:
                    if blocks_text:
                        decoder = json.JSONDecoder()
                        all_doc_blocks, _ = decoder.raw_decode(blocks_text)
                except Exception:
                    pass

                def _normalize_text(t: str) -> str:
                    return re.sub(r'\s+', ' ', t).strip()

                updates = []
                matched_block_ids = set()
                for list_text, text_styles in list_items_to_update:
                    norm_list_text = _normalize_text(list_text)
                    for block in all_doc_blocks:
                        block_id = block.get("block_id", "")
                        if block_id in matched_block_ids:
                            continue
                        block_type = block.get("block_type", 0)
                        if block_type not in (12, 13):
                            continue
                        block_data = block.get("bullet") or block.get("ordered") or {}
                        elements = block_data.get("elements", [])
                        block_text = ""
                        for elem in elements:
                            tr = elem.get("text_run", {})
                            if tr.get("content"):
                                block_text += tr["content"]
                        if _normalize_text(block_text) == norm_list_text:
                            matched_block_ids.add(block_id)
                            text_elements = []
                            for ts in text_styles:
                                if ts.get("equation"):
                                    text_elements.append({"equation": ts["equation"], "style": ts.get("style", {})})
                                else:
                                    text_elements.append({"text": ts.get("text", ""), "style": ts.get("style", {})})
                            updates.append({"blockId": block_id, "textElements": text_elements})
                            break

                if updates:
                    for i in range(0, len(updates), 50):
                        batch = updates[i:i + 50]
                        await update_text_tool.call(ctx_wrapper, documentId=document_id, updates=batch)
            except Exception:
                pass

    # 3. 上传图片
    upload_image_tool = None
    add_blocks_tool_for_img = None
    if provider_manager:
        llm_tools = getattr(provider_manager, 'llm_tools', None)
        if llm_tools:
            for tool in getattr(llm_tools, 'func_list', []):
                if tool.name == 'upload_and_bind_image_to_block':
                    upload_image_tool = tool
                elif tool.name == 'batch_create_feishu_blocks':
                    if not add_blocks_tool_for_img:
                        add_blocks_tool_for_img = tool

    images_uploaded = 0
    if upload_image_tool and add_blocks_tool_for_img:
        for img_info in image_infos:
            img_path = img_info.get("path", "")
            img_caption = img_info.get("caption", "")
            img_base64 = img_info.get("base64", "")
            is_temp_file = False

            if img_base64 and (not img_path or not os.path.exists(img_path)):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(base64.b64decode(img_base64))
                tmp.close()
                img_path = tmp.name
                is_temp_file = True

            if not img_path or not os.path.exists(img_path):
                continue

            img_result = await add_blocks_tool_for_img.call(
                ctx_wrapper, documentId=document_id,
                parentBlockId=document_id, index=-1,
                blocks=[{"blockType": "image", "align": 2, "options": {"image": {}}}]
            )
            image_block_id = None
            try:
                if hasattr(img_result, 'content') and img_result.content:
                    r_text = getattr(img_result.content[0], 'text', None) or str(img_result.content[0])
                    r_data = json.loads(r_text)
                    image_info_data = r_data.get('imageBlocksInfo', {})
                    if isinstance(image_info_data, dict):
                        block_ids = image_info_data.get('blockIds', [])
                        if block_ids:
                            image_block_id = block_ids[0]
            except Exception:
                pass

            if image_block_id:
                upload_res = await upload_image_tool.call(
                    ctx_wrapper, documentId=document_id,
                    images=[{"blockId": image_block_id, "imagePathOrUrl": img_path}]
                )
                if upload_res and not getattr(upload_res, 'isError', True):
                    images_uploaded += 1
                    if img_caption:
                        caption_block = [{
                            "blockType": "text",
                            "options": {
                                "text": {
                                    "textStyles": [{"text": img_caption, "style": {"bold": True, "text_color": 7}}],
                                    "align": 2
                                }
                            }
                        }]
                        await add_blocks_tool_for_img.call(
                            ctx_wrapper, documentId=document_id,
                            parentBlockId=document_id, index=-1, blocks=caption_block
                        )

            if is_temp_file and os.path.exists(img_path):
                try:
                    os.unlink(img_path)
                except OSError:
                    pass

    url = f"https://feishu.cn/docx/{document_id}"
    return {
        "success": True, "document_id": document_id, "url": url,
        "images_uploaded": images_uploaded, "polished_content": weekly_report,
    }
