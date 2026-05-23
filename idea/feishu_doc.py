"""
飞书文档集成：测试方法与文档创建
"""

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from astrbot.api import logger

from .utils import _is_lark_cli_installed, topic_hash
from provider.llm_utils import extract_text_from_response
from .paperbanana import IdeaEnginePaperBanana
from astrbot.core.agent.run_context import ContextWrapper


class IdeaEngineFeishuDoc(IdeaEnginePaperBanana):
    """飞书文档集成。继承链：... → IdeaEnginePaperBanana → IdeaEngineFeishuDoc"""

    async def test_feishu_markdown_formats(self, folder_token: str = "") -> Dict[str, Any]:
        """
        测试用：列表样式 + 图片插入 + 引用链接
        """
        ctx_wrapper = ContextWrapper(context=self.context)

        provider_manager = getattr(self.context, 'provider_manager', None)
        if not provider_manager:
            return {"success": False, "error": "provider_manager 不可用"}

        llm_tools = getattr(provider_manager, 'llm_tools', None)

        # 收集工具
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

        # 从 initial_draft.md 读取内容测试
        draft_path = "/Users/chenyifeng/AstrBot/data/plugin_data/astrbot_plugin_paperrag/ideas/8a160941c48c813c/initial_draft.md"
        try:
            with open(draft_path, "r", encoding="utf-8") as f:
                test_markdown = f.read()
            test_markdown = unquote(test_markdown)  # URL解码
            logger.info(f"[Test] 读取测试文档: {draft_path}, 长度={len(test_markdown)}")
        except Exception as e:
            logger.error(f"[Test] 读取文件失败: {e}")
            return {"success": False, "error": f"读取文件失败: {e}"}

        # 转换为块
        all_blocks = self._markdown_to_feishu_blocks(test_markdown)
        image_count = sum(1 for b in all_blocks if b.get("blockType") == "image")
        list_count = sum(1 for b in all_blocks if b.get("blockType") == "list")
        list_with_styles = sum(1 for b in all_blocks if b.get("blockType") == "list" and b.get("_textStyles"))
        logger.info(f"[Test] 转换 {len(all_blocks)} 个块: {image_count} 图片, {list_count} 列表(其中 {list_with_styles} 个含样式)")

        # 创建飞书文档
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

        # 插入块（交错：文本批量，图片逐张两步上传）
        images_uploaded = 0
        current_index = 0
        text_batch: list = []
        batch_start_index = 0
        # 记录哪些 all_blocks 索引对应到列表块（需要后续更新样式）
        # (原始文本内容, _textStyles)
        list_items_to_update: list[tuple[str, dict]] = []

        async def flush_batch():
            nonlocal text_batch, batch_start_index
            if not text_batch:
                return
            result = await add_blocks_tool.call(
                ctx_wrapper, documentId=document_id,
                parentBlockId=document_id, index=batch_start_index, blocks=text_batch
            )
            if hasattr(result, 'isError') and result.isError:
                raw_text = ""
                if hasattr(result, 'content') and result.content:
                    raw_text = getattr(result.content[0], 'text', '') or str(result.content[0])
                logger.error(f"[Test] 文本块插入失败: {raw_text}")
            else:
                logger.info(f"[Test] 插入 {len(text_batch)} 个块 (index={batch_start_index})")
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
                    img_path = self._ensure_png(img_path)
                    img_caption = opts.get("caption", "")
                    logger.info(f"[Test] caption='{img_caption}'")
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(img_path) as pil_img:
                            orig_w, orig_h = pil_img.size
                        img_width, img_height = orig_w, orig_h
                        logger.info(f"[Test] 图片: {img_width}x{img_height}")
                    except Exception as e:
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
                    except Exception as e:
                        logger.error(f"[Test] 解析图片块ID失败: {e}")

                    if image_block_id and upload_image_tool:
                        upload_res = await upload_image_tool.call(
                            ctx_wrapper, documentId=document_id,
                            images=[{"blockId": image_block_id, "imagePathOrUrl": img_path}]
                        )
                        if upload_res and not getattr(upload_res, 'isError', True):
                            images_uploaded += 1
                            logger.info(f"[Test] 图片上传成功，添加caption: '{img_caption}'")
                            # 追加 caption 文本块（同级，不是子块）
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
                                cap_res = await add_blocks_tool.call(
                                    ctx_wrapper, documentId=document_id,
                                    parentBlockId=document_id,
                                    index=current_index + 1,
                                    blocks=caption_block
                                )
                                if hasattr(cap_res, 'isError') and cap_res.isError:
                                    err = getattr(cap_res.content[0], 'text', str(cap_res))[:200] if hasattr(cap_res, 'content') and cap_res.content else str(cap_res)
                                    logger.error(f"[Test] caption块追加失败: {err}")
                                else:
                                    logger.info(f"[Test] caption块追加成功")
                                    current_index += 1  # caption占一个block位置

                    # 上传完成后清理临时文件
                    if is_temp_file and os.path.exists(img_path):
                        try:
                            os.unlink(img_path)
                        except OSError:
                            pass

                current_index += 1
                batch_start_index = current_index
            else:
                # 记录带样式的列表块（使用原始 content 作为匹配键）
                if b.get("blockType") == "list" and b.get("_textStyles"):
                    list_content = b.get("options", {}).get("list", {}).get("content", "")
                    list_items_to_update.append((list_content, b.get("_textStyles") or {}))
                text_batch.append(b)
                current_index += 1

        await flush_batch()

        # 通过 get_feishu_document_blocks 获取所有块的 ID，按文本内容匹配列表块
        updated_lists = 0
        if list_items_to_update and update_text_tool and get_blocks_tool:
            try:
                blocks_result = await get_blocks_tool.call(ctx_wrapper, documentId=document_id)
                blocks_text = ""
                if hasattr(blocks_result, 'content') and blocks_result.content:
                    blocks_text = getattr(blocks_result.content[0], 'text', '') or str(blocks_result.content[0])

                # 解析 JSON：get_feishu_document_blocks 返回 JSON 数组，后面追加了特殊块提示文本
                # 使用 json.JSONDecoder().raw_decode() 自动忽略尾部内容（找到第一个完整 JSON 数组）
                all_doc_blocks = []
                try:
                    if blocks_text:
                        decoder = json.JSONDecoder()
                        all_doc_blocks, end_pos = decoder.raw_decode(blocks_text)
                        logger.info(f"[Test] JSON 解析成功，{len(all_doc_blocks)} 个块，忽略尾部 {len(blocks_text) - end_pos} 字符")
                except Exception as e:
                    logger.warning(f"[Test] JSON 解析失败: {e}")
                logger.info(f"[Test] 文档共有 {len(all_doc_blocks)} 个块")

                # 匹配：按文本内容找到列表块（空白符归一化后比较）
                def _normalize_text(t: str) -> str:
                    """归一化空白符：将多个连续空白符合并为一个，去除首尾空白"""
                    return re.sub(r'\s+', ' ', t).strip()

                updates = []
                matched_block_ids = set()  # 防止重复匹配同一块
                for list_text, text_styles in list_items_to_update:
                    norm_list_text = _normalize_text(list_text)
                    for block in all_doc_blocks:
                        block_id = block.get("block_id", "")
                        if block_id in matched_block_ids:
                            continue
                        block_type = block.get("block_type", 0)
                        # block_type 12=bullet, 13=ordered
                        if block_type not in (12, 13):
                            continue
                        # 从 block 中提取文本内容
                        block_data = block.get("bullet") or block.get("ordered") or {}
                        elements = block_data.get("elements", [])
                        block_text = ""
                        for elem in elements:
                            tr = elem.get("text_run", {})
                            if tr.get("content"):
                                block_text += tr["content"]
                        if _normalize_text(block_text) == norm_list_text:
                            block_id = block.get("block_id", "")
                            matched_block_ids.add(block_id)
                            logger.info(f"[Test] 匹配到列表块: block_id={block_id}, text={block_text[:50]}")
                            # 构建 textElements
                            text_elements = []
                            for ts in text_styles:
                                if ts.get("equation"):
                                    text_elements.append({"equation": ts["equation"], "style": ts.get("style", {})})
                                else:
                                    text_elements.append({"text": ts.get("text", ""), "style": ts.get("style", {})})
                            updates.append({"blockId": block_id, "textElements": text_elements})
                            break

                if updates:
                    logger.info(f"[Test] 更新 {len(updates)} 个列表块样式")
                    upd_result = await update_text_tool.call(
                        ctx_wrapper, documentId=document_id, updates=updates
                    )
                    if hasattr(upd_result, 'isError') and upd_result.isError:
                        err = ""
                        if hasattr(upd_result, 'content') and upd_result.content:
                            err = getattr(upd_result.content[0], 'text', '') or str(upd_result.content[0])
                        logger.error(f"[Test] 列表样式更新失败: {err}")
                    else:
                        updated_lists = len(updates)
                        logger.info(f"[Test] 列表样式更新成功 ({updated_lists} 个)")
            except Exception as e:
                logger.error(f"[Test] 获取或更新块样式失败: {e}")

        url = f"https://feishu.cn/docx/{document_id}"
        return {
            "success": True,
            "document_id": document_id,
            "url": url,
            "blocks_created": len(all_blocks),
            "image_count": images_uploaded,
            "list_styles_updated": updated_lists,
        }


    def _extract_methodology_section(self, text: str) -> str:
        """从周报文本中提取方法论章节内容"""
        lines = text.split("\n")
        in_methodology = False
        methodology_lines = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'^#{2,3}\s*(方法论|方法|methodology|Methodology|Method)', stripped, re.IGNORECASE):
                in_methodology = True
                continue
            elif in_methodology and stripped.startswith("#"):
                if re.match(r'^#{1,3}\s', stripped):
                    break
            if in_methodology:
                methodology_lines.append(line)
        if not methodology_lines:
            mid = len(lines) // 2
            methodology_lines = lines[mid:]
        return "\n".join(methodology_lines).strip()

    def _load_caption_for_paper(self, topic: str) -> Optional[str]:
        """从 data/captions/ 目录加载论文 caption"""
        captions_dir = Path(__file__).parent.parent / "data" / "captions"
        if not captions_dir.exists():
            return None
        topic_clean = topic.strip()
        for caption_file in captions_dir.glob("*.json"):
            filename_lower = caption_file.stem.lower()
            topic_lower = topic_clean.lower()
            topic_base = re.sub(r'v\d+$', '', topic_lower)
            if topic_base in filename_lower or filename_lower.startswith(topic_base.replace(' ', '')):
                try:
                    with open(caption_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data and isinstance(data, dict):
                        first_key = next(iter(data.keys()), None)
                        if first_key and "caption" in data[first_key]:
                            return data[first_key]["caption"]
                except Exception:
                    pass
        return None

    async def _generate_caption_with_vlm(self, topic: str, methodology_text: str) -> Optional[str]:
        """用 VLM 根据方法论生成 caption"""
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            return None
        prompt = f"""给定以下研究主题和方法论描述，请为该论文的方法流程图生成一个简洁的中文名称作为 caption。
研究主题：{topic}
方法论摘要：{methodology_text[:500]}
要求：
1. 用中文，起一个简洁的方法名称（5-15字）
2. 不要直接照搬研究主题
3. 直接输出名称，不要加任何前缀说明
Caption："""
        try:
            response = await vlm_provider.text_chat(prompt=prompt, contexts=[], temperature=0.3, max_tokens=128)
            return extract_text_from_response(response).strip() if extract_text_from_response(response) else None
        except Exception:
            return None

    async def _refactor_for_paperbanana(self, methodology_text: str, topic: str) -> str:
        """用本地 VLM 将方法论文本转述为 PaperBanana 学术图表格式"""
        vlm_provider = await self._get_vlm_provider_async()
        if not vlm_provider:
            return methodology_text
        prompt = f"""将以下方法论内容转述为适合生成科研图表的详细描述文本。
要求：
1. 保持学术严谨风格，包含每个模块/步骤的具体描述
2. 使用 Markdown 格式，以 ## Methodology: [主题] 开头
3. 使用 ### 二级标题划分不同模块
4. 在描述中加入 LaTeX 数学公式（$...$）来精确表达算法过程
主题：{topic}
方法论原文：{methodology_text}
请直接输出转述后的 Markdown 文本，不要添加任何说明："""
        try:
            response = await vlm_provider.text_chat(prompt=prompt, contexts=[], temperature=0.3, max_tokens=4096)
            result = extract_text_from_response(response)
            return result.strip() if result else methodology_text
        except Exception:
            return methodology_text

    async def create_feishu_document(
        self,
        ideas: List,
        topic: str = "",
        folder_token: str = "",
        knowledge: Optional[Dict[str, Any]] = None,
        table_format: str = "png",
        initial_draft: str = "",
        enable_paper_banana: bool = False
    ) -> Dict[str, Any]:
        """
        创建飞书文档并写入研究想法

        流程：
        1. 使用已有草稿或生成完整周报草稿（VLM）
        2. 提取方法论部分，用本地 VLM 转述为 PaperBanana 图表格式（可选）
        3. 调用 PaperBanana 生成方法图（可选）
        4. 将周报内容和图片插入飞书文档
        """
        try:
            # 1. 使用已有草稿或生成周报草稿
            if initial_draft:
                weekly_report = initial_draft
                logger.info("[IdeaEngine] 使用已有草稿，长度: %d", len(weekly_report))
            else:
                logger.info("[IdeaEngine] 生成完整周报草稿...")
                weekly_report = await self._generate_initial_draft_vlm(ideas, topic, knowledge)
            if not weekly_report:
                return {"error": "周报草稿生成失败", "polished_content": ""}

            # 后处理1：用 caption 匹配替换占位符图片路径为真实路径
            if knowledge:
                local_results = knowledge.get("local_results", [])
                if local_results:
                    weekly_report = self._replace_placeholder_paths_by_caption(weekly_report, local_results)
                    logger.info(f"[IdeaEngine] caption路径替换完成")

            # 2. LLM润色（两阶段：先用本地模型对引用生摘要，再用摘要+草稿润色）
            citations_context = ""
            verified_index = ""
            if knowledge:
                local_results = knowledge.get("local_results", [])
                web_results = knowledge.get("web_results", [])
                if local_results:
                    citations_context += "## 本地论文引用：\n"
                    papers: Dict[str, List] = {}
                    for r in local_results:
                        paper = r.get("paper", "Unknown")
                        if paper not in papers:
                            papers[paper] = []
                        papers[paper].append(r)
                    for paper, chunks in papers.items():
                        citations_context += f"### {paper}\n"
                        for chunk in chunks[:5]:
                            text = chunk.get("text", "")
                            if text:
                                citations_context += f"- {text}\n"
                        citations_context += "\n"

                    # 追加已验证的参考文献索引
                    verified_index = self._build_verified_reference_index(local_results)
                    if verified_index:
                        citations_context += verified_index + "\n\n"
                if web_results:
                    citations_context += "## 网络资源引用：\n"
                    for i, r in enumerate(web_results[:10], 1):
                        title = r.get("title", "")
                        url = r.get("url", "")
                        snippet = r.get("snippet", "")
                        if url:
                            citations_context += f"- [{title}]({url})\n"
                        else:
                            citations_context += f"- {title}\n"
                        if snippet:
                            citations_context += f"  摘要: {snippet}\n"
                    citations_context += "\n"

            # Plan B: 分步处理引用——步骤1生成核心记忆，步骤2用核心记忆润色
            if citations_context and len(citations_context) > 50:
                llm_provider = await self._get_vlm_provider_async()
                if llm_provider:
                    # --- 步骤1：生成核心记忆（超长引用分批处理）---
                    core_memory = ""
                    try:
                        from rag.token_utils import count_tokens, truncate_text_to_tokens

                        logger.info(f"[IdeaEngine] 步骤1：生成核心记忆，"
                                    f"引用: {len(citations_context)} chars, {count_tokens(citations_context)} tokens")

                        memory_prompt_template = """请对以下学术引用资料生成一段简洁的"核心观点记忆"（不超过800字），用于后续润色组会周报。

要求：
- 保留每个论文的：论文名、核心方法/技术路线、关键贡献/结论
- 去掉冗余的实验细节和重复信息
- 用简洁的要点列表组织，每条不超过2句
- 输出格式：直接输出压缩后的核心观点，不要加任何前缀说明
- **注意**：引用资料末尾的"已验证的参考文献索引"是经过校验的权威来源，其论文名、作者、年份、链接均为正确值，压缩时务必完整保留这些信息

引用资料：
{chunk}

核心观点记忆："""

                        n_ctx = getattr(llm_provider, 'n_ctx', 16384)
                        BATCH_MAX_TOKENS = 3072     # 批次摘要输出上限
                        MERGE_MAX_TOKENS = 4096      # 合并摘要输出上限
                        SAFETY_MARGIN = 100          # misc tokens 余量

                        template_body = memory_prompt_template.replace("{chunk}", "")
                        template_tokens = count_tokens(template_body)
                        available_tokens = n_ctx - template_tokens - BATCH_MAX_TOKENS - SAFETY_MARGIN

                        citations_tokens = count_tokens(citations_context)

                        if citations_tokens <= available_tokens:
                            # 单次就能放下
                            prompt = memory_prompt_template.format(chunk=citations_context)
                            prompt_tokens = count_tokens(prompt)
                            logger.info(f"[IdeaEngine] 单次处理, prompt: {prompt_tokens} tokens")
                            memory_response = await llm_provider.text_chat(
                                prompt=prompt, contexts=[], temperature=0.2,
                                max_tokens=BATCH_MAX_TOKENS,
                            )
                            core_memory = extract_text_from_response(memory_response) or ""
                        else:
                            # 分批处理
                            logger.info(f"[IdeaEngine] 引用过长 ({citations_tokens} tokens > "
                                        f"available {available_tokens}), n_ctx={n_ctx}, 启动分批处理")
                            citations = self._split_citations_into_batches(
                                citations_context, available_tokens
                            )
                            logger.info(f"[IdeaEngine] 分为 {len(citations)} 批")

                            partial_summaries = []
                            for i, chunk in enumerate(citations, 1):
                                prompt = memory_prompt_template.format(chunk=chunk)
                                prompt_tokens = count_tokens(prompt)
                                logger.info(f"[IdeaEngine] 批次 {i}/{len(citations)}, "
                                            f"chunk: {count_tokens(chunk)} tokens, "
                                            f"prompt: {prompt_tokens} tokens")
                                try:
                                    resp = await llm_provider.text_chat(
                                        prompt=prompt, contexts=[], temperature=0.2,
                                        max_tokens=BATCH_MAX_TOKENS,
                                    )
                                    summary = extract_text_from_response(resp) or ""
                                    if summary.strip():
                                        partial_summaries.append(summary.strip())
                                    logger.info(f"[IdeaEngine] 批次 {i}/{len(citations)} 完成, "
                                                f"摘要: {len(summary)} chars")
                                except Exception as batch_err:
                                    logger.warning(f"[IdeaEngine] 批次 {i}/{len(citations)} 失败: {batch_err}")
                                    continue

                            if not partial_summaries:
                                core_memory = truncate_text_to_tokens(citations_context, 2000)
                            elif len(partial_summaries) == 1:
                                core_memory = partial_summaries[0]
                            else:
                                # 合并多批摘要
                                merge_parts = []
                                for i, s in enumerate(partial_summaries):
                                    merge_parts.append(f"--- 第{i+1}段 ---\n{s}")
                                merge_body = "\n".join(merge_parts)
                                merge_prompt = f"""请将以下 {len(partial_summaries)} 段学术观点摘要合并为一段统一的"核心观点记忆"（不超过1200字）。

要求：
- 去重：相同论文的观点只保留一次
- 保留论文名、核心方法、关键贡献
- 用简洁的要点列表组织
- 直接输出合并后的核心观点，不要加前缀说明

各批次摘要：
{merge_body}

统一核心观点记忆："""
                                merge_prompt_tokens = count_tokens(merge_prompt)
                                logger.info(f"[IdeaEngine] 合并 prompt: {merge_prompt_tokens} tokens")
                                try:
                                    resp = await llm_provider.text_chat(
                                        prompt=merge_prompt, contexts=[], temperature=0.2,
                                        max_tokens=MERGE_MAX_TOKENS,
                                    )
                                    core_memory = extract_text_from_response(resp) or ""
                                    logger.info(f"[IdeaEngine] 合并 {len(partial_summaries)} 段完成, "
                                                f"最终: {len(core_memory)} chars")
                                except Exception as merge_err:
                                    logger.warning(f"[IdeaEngine] 合并失败: {merge_err}")
                                    core_memory = "\n\n".join(partial_summaries)

                        logger.info(f"[IdeaEngine] 核心记忆生成完成，"
                                    f"长度: {len(core_memory)} chars, {count_tokens(core_memory)} tokens")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] 核心记忆生成失败: {e}，使用原始引用摘要")
                        core_memory = truncate_text_to_tokens(citations_context, 2000)

                    # --- 步骤2：用核心记忆 + 草稿润色 ---
                    verified_section = f"\n\n## 已验证的参考文献索引（权威来源，必须使用这些标题和链接）：\n{verified_index}" if verified_index else ""
                    polish_prompt = f"""你是一个学术助手，负责对以下组会周报草稿进行润色和完善。

参考资料（核心记忆）：
{core_memory}{verified_section}

原始草稿：
{weekly_report}

**重要指令**：
- 在原文基础上适当扩展：每个简短的要点/列表项扩展为1-2句连贯段落
- 保持原文的整体结构和章节顺序，只做润色和扩展，不打乱框架
- 充分利用核心记忆中的信息，但不要直接复制，要融会贯通

**强制引用规则（必须严格遵守）**：
- **所有论文引用必须使用上述"已验证的参考文献索引"中的标题和链接**
- 该索引是经过 LLM+arXiv 校验的权威来源，论文名、作者、年份、DOI 均为正确值
- "本地论文引用"正文可能含 PDF 噪声，其论文名称不可直接使用
- 正文提到某篇论文时，必须在索引中查找对应条目，使用索引中的标题和 DOI 链接

格式要求：
- 包含章节：背景动机、相关工作、方法论、创新点、实验benchmark、挑战与解决方案、下一步计划、参考文献、论文图表
- **扩展原则**：将简短的要点列表扩展为连贯段落，但不能变成全新的内容
- **列表格式**：创新点和挑战与解决方案部分使用数字序号列表（如"1. 挑战一：xxx"）

**正文引用格式（重要）**：
- 正文中的引用：使用论文简称加markdown链接，如 [FLARE](https://arxiv.org/abs/2502.12138)、[NoPoSplat](https://arxiv.org/abs/2505.23716)
- **禁止在正文中使用论文全名或裸URL**
- **正文及正文中所有涉及引用的地方（论文简称如FLARE、方法名称、引用标记如[4][5]等）一律加粗**，不得有的加粗有的不加粗

**参考文献格式（重要，严格遵守）**：
- 放在最后一个章节
- 每行一条，**严格格式**：`1. [**论文全名**](URL)`
- 数字序号列表，全名加粗，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL（如 `URL (URL)` ）、禁止纯文本URL

**图表引用格式（重要，严格遵守）**：
- **禁止在正文中使用任何图片语法**：`!(..)`、`` ![](..) ``、`[图片:...](...)` 一律禁止
- 正文引用图片时只用文字描述，如"如图1所示"、"如图2所示"
- **参考文献章节中禁止出现任何图片路径**，参考文献中的方法图引用一律改为纯文字描述（如"NoPoSplat 方法流程图"），不得出现 /Users/ 等路径
- **不要生成"论文图表"章节**，论文图表已自动嵌入相关工作章节

请直接输出润色后的内容："""

                    try:
                        polish_total = len(core_memory) + len(weekly_report)
                        logger.info(f"[IdeaEngine] 步骤2：润色草稿，核心记忆: {len(core_memory)}, 草稿: {len(weekly_report)}, 总prompt: ~{polish_total}")
                        response = await llm_provider.text_chat(
                            prompt=polish_prompt,
                            contexts=[],
                            temperature=0.3,
                            max_tokens=32768
                        )
                        polished = extract_text_from_response(response)
                        if polished and len(polished) > 100:
                            weekly_report = polished
                            logger.info(f"[IdeaEngine] Plan B 润色完成，长度: {len(polished)}")
                        else:
                            logger.warning(f"[IdeaEngine] 润色结果过短，保持原内容")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] Plan B 润色失败: {e}，保持原内容")
                else:
                    logger.info("[IdeaEngine] 无LLM provider，跳过润色")
            else:
                # 无引用上下文时，直接润色草稿
                logger.info("[IdeaEngine] 无引用上下文，直接润色草稿")
                llm_provider = await self._get_vlm_provider_async()
                if llm_provider:
                    simplify_prompt = f"""你是一个学术助手，负责对以下组会周报草稿进行润色和完善。

原始草稿：
{weekly_report}

**注意**：草稿中的论文名可能包含 PDF 提取噪声。对于无法确认正确拼写的论文名，保持原文拼写，不要猜测修正。切勿编造不存在的论文名或链接。

**重要指令**：
- 在原文基础上适当扩展：每个简短的要点/列表项扩展为1-2句连贯段落
- 保持原文的整体结构和章节顺序，只做润色和扩展，不打乱框架

格式要求：
- 包含章节：背景动机、相关工作、方法论、创新点、实验benchmark、挑战与解决方案、下一步计划、参考文献
- **扩展原则**：将简短的要点列表扩展为连贯段落，但不能变成全新的内容
- **列表格式**：创新点和挑战与解决方案部分使用数字序号列表（如"1. 挑战一：xxx"）

**正文引用格式（重要）**：
- 正文中的引用：使用论文简称加markdown链接，如 [FLARE](https://arxiv.org/abs/2502.12138)、[NoPoSplat](https://arxiv.org/abs/2505.23716)
- **禁止在正文中使用论文全名或裸URL**
- **正文及正文中所有涉及引用的地方（论文简称如FLARE、方法名称、引用标记如[4][5]等）一律加粗**，不得有的加粗有的不加粗

**参考文献格式（重要，严格遵守）**：
- 放在最后一个章节
- 每行一条，**严格格式**：`1. [**论文全名**](URL)`
- 数字序号列表，全名加粗，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL（如 `URL (URL)` ）、禁止纯文本URL

**图表引用格式（重要，严格遵守）**：
- **禁止在正文中使用任何图片语法**：`!(..)`、`` ![](..) ``、`[图片:...](...)` 一律禁止
- 正文引用图片时只用文字描述，如"如图1所示"、"如图2所示"
- **参考文献章节中禁止出现任何图片路径**，参考文献中的方法图引用一律改为纯文字描述（如"NoPoSplat 方法流程图"），不得出现 /Users/ 等路径
- **不要生成"论文图表"章节**，论文图表已自动嵌入相关工作章节

请直接输出润色后的内容："""

                    try:
                        logger.info(f"[IdeaEngine] 直接润色草稿，原始长度: {len(weekly_report)}")
                        response = await llm_provider.text_chat(
                            prompt=simplify_prompt,
                            contexts=[],
                            temperature=0.3,
                            max_tokens=32768
                        )
                        polished = extract_text_from_response(response)
                        if polished and len(polished) > 100:
                            weekly_report = polished
                            logger.info(f"[IdeaEngine] 直接润色完成，长度: {len(polished)}")
                        else:
                            logger.warning(f"[IdeaEngine] 润色结果过短，保持原内容")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] 润色失败: {e}，保持原内容")
                else:
                    logger.info("[IdeaEngine] 无LLM provider，跳过润色")

            # 手动追加论文图表章节（用真实路径，不依赖LLM生成）
            weekly_report = self._append_figure_section(weekly_report, knowledge)

            # --- PaperBanana 方法图生成（可选）---
            figure_blocks = []
            if enable_paper_banana:
                # 3. 提取方法论章节内容
                methodology_text = self._extract_methodology_section(weekly_report)
                logger.info(f"[IdeaEngine] 方法论章节长度: {len(methodology_text)}")
                if len(methodology_text) < 50:
                    logger.warning("[IdeaEngine] 方法论章节过短，PaperBanana 图表可能质量不佳")

                # 4. 尝试从 captions 目录加载 caption，若无则用 VLM 生成
                paper_caption = self._load_caption_for_paper(topic)
                if not paper_caption:
                    paper_caption = await self._generate_caption_with_vlm(topic, methodology_text)
                logger.info(f"[IdeaEngine] PaperBanana caption: {paper_caption[:50] if paper_caption else 'None'}...")

                # 5. 用本地 VLM 将方法论转述为 PaperBanana 图表格式
                paperbanana_format_text = ""
                if methodology_text:
                    paperbanana_format_text = await self._refactor_for_paperbanana(methodology_text, topic)
                    logger.info(f"[IdeaEngine] PaperBanana 格式转述完成，长度: {len(paperbanana_format_text)}")

                # 6. 调用 PaperBanana 生成方法图
                if paperbanana_format_text:
                    logger.info("[IdeaEngine] 正在生成方法图（PaperBanana）...")
                    figure_blocks = await self._generate_method_figures_with_paperbanana_from_text(
                        paperbanana_format_text, topic, caption=paper_caption
                    )
                    logger.info(f"[IdeaEngine] PaperBanana 生成完成，共 {len(figure_blocks)} 张方法图")
            else:
                logger.info("[IdeaEngine] PaperBanana 未启用，跳过方法图生成")

            # 6. 使用本地 VLM 生成简洁标题
            generated_title = topic
            llm_provider = await self._get_vlm_provider_async()
            if llm_provider:
                title_prompt = f"""给定以下研究主题，请为飞书文档生成一个简洁、有意义、学术风格的标题。

研究主题：{topic}

要求：
1. 标题应该反映研究的核心内容，不要直接使用原始问题
2. 标题长度适中（5-15个字）
3. 可以包含 emoji 作为装饰
4. 直接输出标题，不要加任何说明

例如：
- 如果主题是"大模型在代码生成中的应用"，可以生成："🚀 代码生成新范式：大模型赋能编程"
- 如果主题是"多模态大模型研究"，可以生成："🔍 多模态大模型研究进展"

请直接输出标题："""
                try:
                    title_response = await llm_provider.text_chat(
                        prompt=title_prompt,
                        contexts=[],
                        temperature=0.7,
                        max_tokens=256
                    )
                    generated_title = extract_text_from_response(title_response)
                    generated_title = generated_title.strip() if generated_title else topic
                    logger.info(f"[IdeaEngine] LLM生成标题: {generated_title}")
                except Exception as e:
                    logger.warning(f"[IdeaEngine] 生成标题失败: {e}，使用原始主题")
                    generated_title = topic

            # 5. 获取飞书工具
            feishu_tool = self._get_feishu_tool()
            if not feishu_tool:
                return {"error": "未找到飞书 MCP 工具，请确认飞书 MCP 已配置并启用", "polished_content": weekly_report}

            from astrbot.core.agent.run_context import ContextWrapper
            ctx_wrapper = ContextWrapper(context=self.context)

            # 6. 保存草稿（提前保存，防止飞书API失败丢失）
            try:
                folder_hash = topic_hash(topic)
                draft_file = self._get_ideas_dir() / folder_hash / "initial_draft.md"
                draft_file.parent.mkdir(parents=True, exist_ok=True)
                with open(draft_file, "w", encoding="utf-8") as f:
                    f.write(weekly_report)
                logger.info(f"[IdeaEngine] 草稿已提前保存: {draft_file}")
            except Exception as e:
                logger.warning(f"[IdeaEngine] 提前保存草稿失败: {e}")

            # 7. 创建文档
            logger.info(f"[IdeaEngine] 创建飞书文档: {generated_title}, folder_token: {folder_token}")
            create_result = await feishu_tool.call(ctx_wrapper, title=generated_title, folderToken=folder_token)

            # 7. 解析响应
            result_text = ""
            if hasattr(create_result, 'content') and create_result.content:
                result_text = getattr(create_result.content[0], 'text', None) or str(create_result.content[0])

            # 检测授权过期/未授权（飞书 MCP 返回授权提示而非文档信息）
            if "请在浏览器打开以下链接进行授权" in result_text or "authorize" in result_text.lower():
                auth_url_match = re.search(r'https://accounts\.feishu\.cn/open-apis/authen/v1/authorize\S+', result_text)
                auth_url = auth_url_match.group(0) if auth_url_match else ""
                logger.error("[IdeaEngine] 飞书 MCP 未授权或授权已过期")
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

            logger.info(f"[IdeaEngine] 文档创建成功: {document_id}")

            # 8. 获取根块 ID 并插入内容
            root_block_id = document_id

            # 9. 将周报内容转换为飞书块格式（含行内样式）
            provider_manager = getattr(self.context, 'provider_manager', None)
            all_blocks = []
            if weekly_report:
                logger.info(f"[IdeaEngine] 周报内容长度: {len(weekly_report)}, 转换块数量: {len(weekly_report.split(chr(10)))}")
                logger.debug(f"[IdeaEngine] 原始内容末尾200字符: '''{weekly_report[-200:]}'''")
                weekly_report = self._normalize_figure_references(weekly_report)
                logger.debug(f"[IdeaEngine] normalize后内容末尾200字符: '''{weekly_report[-200:]}'''")
                figure_refs = re.findall(r'图\s*\d+', weekly_report)
                logger.info(f"[IdeaEngine] 正文图表引用: {len(figure_refs)}处，引用: {figure_refs}")
                # Prevent accidental markdown image rendering of bare paths
                weekly_report = self._convert_paren_paths_to_markdown(weekly_report)
                # Strip any stray ![image](path) the LLM may have generated
                weekly_report = re.sub(r'!\[image\]\(([/][^)]+\.(?:png|jpg|jpeg|webp|gif))\)', r'[\1]', weekly_report)
                logger.debug(f"[IdeaEngine] 路径替换后内容末尾200字符: '''{weekly_report[-200:]}'''")

                # Count markdown images — these should all become image blocks
                markdown_image_count = len(re.findall(r'!\[.*?\]\(/', weekly_report))
                logger.info(f"[IdeaEngine] markdown 图片总数: {markdown_image_count}")

                all_blocks = self._markdown_to_feishu_blocks(weekly_report)
                image_block_count = sum(1 for b in all_blocks if b.get("blockType") == "image")
                logger.info(f"[IdeaEngine] 转换后的块数量: {len(all_blocks)}，其中图片块: {image_block_count}")
                if markdown_image_count != image_block_count:
                    logger.warning(f"[IdeaEngine] ⚠️ markdown图片数({markdown_image_count})与飞书图片块数({image_block_count})不匹配（图片文件可能不存在）")
                if figure_refs and image_block_count < len(figure_refs):
                    logger.info(f"[IdeaEngine] 正文引用{len(figure_refs)}处图表，图片块{image_block_count}个 — 若引用数远大于图片块数，请检查知识库中论文的图表文件是否已提取")
                for i, b in enumerate(all_blocks):
                    if b.get("blockType") == "image":
                        opts = b.get("options", {}).get("image", {})
                        logger.info(f"[IdeaEngine] 图片块[{i}]: path='{opts.get('image_path', 'N/A')}', caption='{opts.get('caption', 'N/A')}'")

            # 将 PaperBanana 生成的方法图插入到方法论章节末尾
            if figure_blocks:
                method_insert_idx = self._find_methodology_end_index(all_blocks)
                logger.info(f"[IdeaEngine] 将 {len(figure_blocks)} 张方法图插入到索引 {method_insert_idx}")
                for i, fb in enumerate(figure_blocks):
                    all_blocks.insert(method_insert_idx + i, fb)

            if all_blocks:
                add_blocks_tool = None
                upload_image_tool = None
                update_text_tool = None
                get_blocks_tool = None
                if provider_manager:
                    llm_tools = getattr(provider_manager, 'llm_tools', None)
                    if llm_tools:
                        for tool in getattr(llm_tools, 'func_list', []):
                            if tool.name == 'batch_create_feishu_blocks':
                                add_blocks_tool = tool
                            elif tool.name == 'upload_and_bind_image_to_block':
                                upload_image_tool = tool
                            elif tool.name == 'batch_update_feishu_block_text':
                                update_text_tool = tool
                            elif tool.name == 'get_feishu_document_blocks':
                                get_blocks_tool = tool

                if add_blocks_tool:
                    # 交错插入：按原始顺序遍历，文本块批量插入，图片块逐张两步上传
                    images_uploaded = 0
                    current_index = 0
                    text_batch: list = []
                    batch_start_index = 0
                    list_items_to_update: list[tuple[str, dict]] = []

                    async def _flush_text_batch_async():
                        """异步刷出累积的文本块"""
                        nonlocal text_batch, batch_start_index, get_blocks_tool
                        if not text_batch:
                            return
                        CHUNK_SIZE = 20
                        for chunk_start in range(0, len(text_batch), CHUNK_SIZE):
                            chunk = text_batch[chunk_start:chunk_start + CHUNK_SIZE]
                            chunk_index = batch_start_index + chunk_start
                            result = await add_blocks_tool.call(
                                ctx_wrapper,
                                documentId=document_id,
                                parentBlockId=root_block_id,
                                index=chunk_index,
                                blocks=chunk
                            )
                            if hasattr(result, 'isError') and result.isError:
                                err = getattr(result.content[0], 'text', str(result))[:300] if hasattr(result, 'content') and result.content else str(result)
                                logger.error(f"[IdeaEngine] 文本块插入失败 (chunk {chunk_start}, index={chunk_index}): {err}")
                            else:
                                logger.info(f"[IdeaEngine] 插入 {len(chunk)} 个文本块 (index={chunk_index})")
                        text_batch = []

                    _plugin_root = Path(__file__).parent.parent
                    _use_larkcli = self._lark_cli_available()

                    def _extract_block_plain_text(block: dict) -> str:
                        """Extract plain text from any text-like feishu block for use as an anchor."""
                        opts = block.get("options", {})
                        # Heading / list / bullet / ordered: direct "content" field
                        for key in ("heading", "list", "bullet", "ordered"):
                            inner = opts.get(key, {})
                            if isinstance(inner, dict):
                                t = inner.get("content", "")
                                if t:
                                    return t
                        # Text block: text is in textStyles array
                        text_opts = opts.get("text", {})
                        if isinstance(text_opts, dict):
                            styles = text_opts.get("textStyles", [])
                            if styles:
                                parts = [s["text"] for s in styles if isinstance(s, dict) and s.get("text")]
                                return "".join(parts)
                        return ""

                    _last_anchor = ""
                    for b in all_blocks:
                        if b.get("blockType") == "image":
                            # Extract anchor from the last text block before flushing.
                            # Accept any non-empty text (even short headings);
                            # start...end disambiguation handles uniqueness.
                            _anchor = ""
                            for _tb in reversed(text_batch):
                                _t = _extract_block_plain_text(_tb)
                                if _t:
                                    _anchor = _t
                                    _last_anchor = _t
                                    break
                            if not _anchor:
                                _anchor = _last_anchor

                            await _flush_text_batch_async()

                            opts = b.get("options", {}).get("image", {})
                            img_path = opts.get("image_path", "")
                            img_base64 = opts.get("base64", "")
                            is_temp_file = False
                            if img_base64 and (not img_path or not os.path.exists(img_path)):
                                if img_path:
                                    logger.info(f"[IdeaEngine] 图片文件已被清理，从 base64 恢复: {img_path}")
                                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                                tmp.write(base64.b64decode(img_base64))
                                tmp.close()
                                img_path = tmp.name
                                is_temp_file = True
                            elif img_path and os.path.exists(img_path) and "data/temp" in img_path:
                                # PaperBanana images in data/temp/ are ephemeral
                                is_temp_file = True

                            if img_path and os.path.exists(img_path):
                                img_path = self._ensure_png(img_path)
                                img_width = opts.get("width")
                                img_height = opts.get("height")
                                if not img_width or not img_height:
                                    try:
                                        from PIL import Image as PILImage
                                        with PILImage.open(img_path) as pil_img:
                                            orig_w, orig_h = pil_img.size
                                        img_width, img_height = orig_w, orig_h
                                        logger.info(f"[IdeaEngine] 图片尺寸: {img_path} → {img_width}x{img_height}")
                                    except Exception:
                                        img_width, img_height = 768, 768
                                else:
                                    logger.info(f"[IdeaEngine] 图片尺寸 [来自block]: {img_width}x{img_height}")

                                img_caption = opts.get("caption", "")
                                inserted = False

                                # All images go through lark-cli with --selection-with-ellipsis
                                if _use_larkcli:
                                    rel_path = os.path.relpath(img_path, start=_plugin_root)
                                    if not rel_path.startswith("..") and not os.path.isabs(rel_path):
                                        try:
                                            lark_args = [
                                                "+media-insert",
                                                "--doc", f"https://feishu.cn/docx/{document_id}",
                                                "--type", "image",
                                                "--file", rel_path,
                                                "--width", str(img_width),
                                                "--height", str(img_height),
                                            ]
                                            if img_caption:
                                                lark_args += ["--caption", img_caption]
                                            if _anchor:
                                                # Use start...end disambiguation:
                                                # lark-cli matches BOTH prefix and suffix,
                                                # preventing false matches on common substrings
                                                text = _anchor.strip()
                                                if len(text) > 80:
                                                    anchor_key = f"{text[:40]}...{text[-40:]}"
                                                elif len(text) > 30:
                                                    anchor_key = f"{text[:15]}...{text[-15:]}"
                                                else:
                                                    anchor_key = text
                                                lark_args += ["--selection-with-ellipsis", anchor_key]
                                                logger.info(f"[IdeaEngine] lark-cli 定位插入 (anchor={anchor_key[:60]}...)")
                                            else:
                                                logger.info(f"[IdeaEngine] lark-cli 追加到末尾 (无锚点文本)")

                                            lark_res = self._call_lark_cli(
                                                "docs", lark_args, timeout=30,
                                                cwd=str(_plugin_root),
                                            )
                                            if lark_res["success"]:
                                                images_uploaded += 1
                                                inserted = True
                                                logger.info(f"[IdeaEngine] lark-cli 图片插入成功: {rel_path}")
                                            else:
                                                logger.warning(f"[IdeaEngine] lark-cli 图片插入失败: {lark_res.get('error', '')[:150]}")
                                        except Exception as e:
                                            logger.warning(f"[IdeaEngine] lark-cli 图片插入异常: {e}")
                                    else:
                                        logger.warning(f"[IdeaEngine] 图片路径不在插件目录内，跳过 lark-cli: {img_path}")

                                # Fallback: MCP only when lark-cli is not available
                                if not inserted:
                                    logger.info(f"[IdeaEngine] lark-cli 不可用，回退到 MCP")
                                    img_result = await add_blocks_tool.call(
                                        ctx_wrapper,
                                        documentId=document_id,
                                        parentBlockId=root_block_id,
                                        index=current_index,
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
                                    except Exception as e:
                                        logger.error(f"[IdeaEngine] 解析图片块ID失败: {e}")

                                    if image_block_id and upload_image_tool:
                                        upload_res = await upload_image_tool.call(
                                            ctx_wrapper,
                                            documentId=document_id,
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
                                                    parentBlockId=document_id,
                                                    index=current_index + 1,
                                                    blocks=caption_block
                                                )
                                                current_index += 1
                                        else:
                                            err_msg = ""
                                            if hasattr(upload_res, 'content') and upload_res.content:
                                                err_msg = getattr(upload_res.content[0], 'text', str(upload_res))[:200]
                                            logger.error(f"[IdeaEngine] MCP 图片上传失败: {err_msg}")
                                    else:
                                        if not image_block_id:
                                            logger.error(f"[IdeaEngine] 未获取到图片块ID，跳过上传")
                                        elif not upload_image_tool:
                                            logger.error(f"[IdeaEngine] upload_image_tool 不可用")

                                # 上传完成后清理临时文件
                                if is_temp_file and os.path.exists(img_path):
                                    try:
                                        os.unlink(img_path)
                                    except OSError:
                                        pass
                            else:
                                logger.error(
                                    f"[IdeaEngine] 图片块缺少有效数据 "
                                    f"(image_path={img_path!r}, has_base64={bool(img_base64)})，跳过"
                                )

                            current_index += 1
                            batch_start_index = current_index
                        else:
                            if b.get("blockType") == "list" and b.get("_textStyles"):
                                list_content = b.get("options", {}).get("list", {}).get("content", "")
                                list_items_to_update.append((list_content, b.get("_textStyles") or {}))
                            text_batch.append(b)
                            current_index += 1

                    await _flush_text_batch_async()

                    # 通过 get_feishu_document_blocks 获取块 ID，再更新列表样式
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
                                    all_doc_blocks, end_pos = decoder.raw_decode(blocks_text)
                                    logger.info(f"[IdeaEngine] JSON 解析成功，{len(all_doc_blocks)} 个块，忽略尾部 {len(blocks_text) - end_pos} 字符")
                            except Exception as e:
                                logger.warning(f"[IdeaEngine] JSON 解析失败: {e}")
                            logger.info(f"[IdeaEngine] 获取到 {len(all_doc_blocks)} 个文档块，准备更新 {len(list_items_to_update)} 个列表样式")

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
                                        logger.info(f"[IdeaEngine] 匹配列表块: block_id={block_id}, text={block_text[:30]}")
                                        break

                            if updates:
                                logger.info(f"[IdeaEngine] 更新 {len(updates)} 个列表块样式")
                                for i in range(0, len(updates), 50):
                                    batch = updates[i:i + 50]
                                    upd_result = await update_text_tool.call(
                                        ctx_wrapper,
                                        documentId=document_id,
                                        updates=batch
                                    )
                                    if hasattr(upd_result, 'isError') and upd_result.isError:
                                        err = getattr(upd_result.content[0], 'text', str(upd_result))[:300] if hasattr(upd_result, 'content') and upd_result.content else str(upd_result)
                                        logger.error(f"[IdeaEngine] 列表样式更新失败: {err}")
                                    else:
                                        logger.info(f"[IdeaEngine] 列表样式更新成功 ({len(batch)} 个块)")
                        except Exception as e:
                            logger.error(f"[IdeaEngine] 获取或更新块样式失败: {e}")

                    logger.info(f"[IdeaEngine] 文档写入完成: {images_uploaded} 张图片已上传, 总块数: {len(all_blocks)}")
                else:
                    logger.warning("[IdeaEngine] 未找到 batch_create_feishu_blocks 工具")
            else:
                logger.warning("[IdeaEngine] all_blocks 为空，跳过块插入")

            url = f"https://feishu.cn/docx/{document_id}"
            return {
                "success": True,
                "document_id": document_id,
                "url": url,
                "blocks_created": len(all_blocks),
                "polished_content": weekly_report
            }

        except Exception as e:
            logger.error(f"[IdeaEngine] 飞书文档创建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e), "polished_content": ""}

    # ==================== larksuite/cli 集成 ====================

    _LARK_CLI_SUBCMDS = frozenset({"doc", "docs", "wiki", "calendar", "sheets", "base", "im", "help"})

    @staticmethod
    def _lark_cli_available() -> bool:
        """检测 lark-cli 是否可用"""
        return _is_lark_cli_installed()

    @staticmethod
    def _call_lark_cli(subcmd: str, args: list[str], timeout: int = 30,
                       cwd: str | None = None) -> dict[str, Any]:
        """调用 lark-cli 命令（独立 CLI，不走 MCP）。

        Args:
            subcmd: 如 "doc", "wiki", "calendar"
            args: 如 ["create", "--title", "xxx"]
            timeout: 超时秒数
            cwd: 工作目录（用于解析相对路径；默认 None = 当前目录）

        Returns:
            {"success": True, "data": ...} 或 {"success": False, "error": ...}
        """
        if subcmd not in IdeaEngineFeishuDoc._LARK_CLI_SUBCMDS:
            raise ValueError(f"Unknown subcommand: {subcmd}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")

        cmd = ["lark-cli", subcmd] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env={**os.environ, "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1"},
            )
        except FileNotFoundError:
            logger.error("[lark-cli] 未找到 lark-cli，请安装: npx @larksuite/cli@latest install")
            return {"success": False, "error": "lark-cli not found"}
        except subprocess.TimeoutExpired as e:
            logger.error(
                f"[lark-cli] 命令超时 ({timeout}s): {' '.join(cmd)}\n"
                f"partial stderr: {e.stderr or '(none)'}"
            )
            return {"success": False, "error": f"Command timeout after {timeout}s"}
        except OSError as e:
            logger.error(f"[lark-cli] 系统调用失败: {e}")
            return {"success": False, "error": f"System error: {e}"}

        if result.returncode != 0:
            stderr_preview = (result.stderr or "").strip()[:500]
            logger.error(f"[lark-cli] {' '.join(cmd)} 失败 (rc={result.returncode}): {stderr_preview}")
            return {"success": False, "error": stderr_preview or f"exit code {result.returncode}"}

        stdout = result.stdout.strip()
        if not stdout:
            return {"success": True, "data": None}
        try:
            return {"success": True, "data": json.loads(stdout)}
        except json.JSONDecodeError:
            return {"success": True, "data": stdout}

    @staticmethod
    def _split_citations_into_batches(
        citations_context: str, max_tokens: int
    ) -> list[str]:
        """将引用文本按论文逐条拆分为不超过 max_tokens 的批次（精确 token 计数）。

        引用格式:
            - Paper Title
              摘要: snippet text

        贪心打包：每条引用是一个 block，用 count_tokens 精确度量，逐条加入当前批次。
        """
        from rag.token_utils import count_tokens

        # 按 "- " 开头的行分割为独立引用块
        blocks: list[str] = []
        current_block: list[str] = []
        for line in citations_context.split("\n"):
            if line.startswith("- ") and current_block:
                blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append("\n".join(current_block))

        batches: list[str] = []
        current_batch: list[str] = []
        current_tokens = 0
        for block in blocks:
            block_tokens = count_tokens(block)
            if current_batch and current_tokens + block_tokens > max_tokens:
                batches.append("\n".join(current_batch))
                current_batch = [block]
                current_tokens = block_tokens
            else:
                current_batch.append(block)
                current_tokens += block_tokens
        if current_batch:
            batches.append("\n".join(current_batch))

        return batches
