"""
飞书文档集成：测试方法与文档创建
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from astrbot.api import logger

from .utils import _is_lark_cli_installed, topic_hash
from provider.llm_utils import extract_text_from_response
from .paperbanana import IdeaEnginePaperBanana


class IdeaEngineFeishuDoc(IdeaEnginePaperBanana):
    """飞书文档集成。继承链：... → IdeaEnginePaperBanana → IdeaEngineFeishuDoc

    文档创建仅使用 lark-cli 路径。MCP 路径已移至 legacy/feishu_mcp.py。
    """

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
                except Exception as e:
                    logger.warning(f"[IdeaEngine] 读取 caption 文件失败: {caption_file} ({e})")
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
            response = await vlm_provider.text_chat(prompt=prompt, contexts=[], temperature=0.3)
            return extract_text_from_response(response).strip() if extract_text_from_response(response) else None
        except Exception as e:
            logger.warning(f"[IdeaEngine] VLM caption 生成失败: {e}")
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
            max_tokens_vlm = self._compute_vlm_max_tokens(vlm_provider, prompt)
            response = await vlm_provider.text_chat(prompt=prompt, contexts=[], temperature=0.3, max_tokens=max_tokens_vlm)
            result = extract_text_from_response(response)
            return result.strip() if result else methodology_text
        except Exception as e:
            logger.warning(f"[IdeaEngine] PaperBanana 重述 VLM 调用失败: {e}")
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
                cloud_provider = self.context.get_using_provider() if hasattr(self.context, 'get_using_provider') else None
                llm_provider = cloud_provider or await self._get_vlm_provider_async()
                use_cloud = cloud_provider is not None
                if llm_provider:
                    # --- 步骤1：生成核心记忆（云端 LLM 单次处理，本地 VLM 分批处理）---
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

                        BATCH_MAX_TOKENS = 3072     # 批次摘要输出上限
                        MERGE_MAX_TOKENS = 4096      # 合并摘要输出上限
                        SAFETY_MARGIN = 100          # misc tokens 余量

                        if use_cloud:
                            # 云端 LLM 上下文远超引用长度，无需分批，直接单次处理；不传 max_tokens，由 provider 自行决定
                            prompt = memory_prompt_template.format(chunk=citations_context)
                            logger.info(f"[IdeaEngine] 使用云端 LLM, 单次处理 {count_tokens(citations_context)} tokens")
                            memory_response = await llm_provider.text_chat(
                                prompt=prompt, contexts=[], temperature=0.2,
                            )
                            core_memory = extract_text_from_response(memory_response) or ""
                        else:
                            n_ctx = getattr(llm_provider, 'n_ctx', 16384)
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
- 包含章节：背景动机、相关工作、方法论、创新点、实验Benchmark、挑战与解决方案、下一步计划、参考文献
- **章节标题必须使用如下精确格式，不得自行修改**：
  `## 1. 背景动机`、`## 2. 相关工作`、`## 3. 方法论`、`## 4. 创新点`、`## 5. 实验Benchmark`、`## 6. 挑战与解决方案`、`## 7. 下一步计划`、`## 8. 参考文献`
- **扩展原则**：将简短的要点列表扩展为连贯段落，但不能变成全新的内容
- **列表格式**：创新点和挑战与解决方案部分使用数字序号列表（如"1. 挑战一：xxx"）

**正文引用格式（重要，严格遵守）**：
- **在方法名/论文名首次出现的位置直接替换为 markdown 链接**，如文中提到 PanoGS 应写为 `[PanoGS](url)`，提到 FLARE 应写为 `[FLARE](url)`
- **禁止**在句子末尾追加论文全名！错误示例：`PanoGS 提出了xxx。PanoGS: Gaussian-based Panoptic Segmentation...` ← 这种格式绝对不允许
- 正确示例：`[PanoGS](url) 提出利用金字塔三平面构建连续参数化特征空间...`
- 论文全名只允许出现在参考文献章节
- **禁止在正文中使用论文全名或裸URL**
**参考文献格式（重要，严格遵守）**：
- 放在最后一个章节
- 每行一条，**严格格式**：`1. [论文全名](URL)`
- 数字序号列表，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL（如 `URL (URL)` ）、禁止纯文本URL

**图表引用格式（重要，严格遵守）**：
- **禁止在正文中使用任何图片语法**：`!(..)`、`` ![](..) ``、`[图片:...](...)` 一律禁止
- **禁止在正文中编造具体的图/表编号**（如"如图1所示"、"如图2所示"），因为你还不知道哪些图表实际存在
- 如需引用图表，只使用泛指描述（如"相关实验结果如图所示"、"方法流程如图表所示"），不指定编号
- **参考文献章节中禁止出现任何图片路径**，参考文献中的方法图引用一律改为纯文字描述（如"NoPoSplat 方法流程图"），不得出现 /Users/ 等路径

请直接输出润色后的内容："""

                    try:
                        polish_total = len(core_memory) + len(weekly_report)
                        logger.info(f"[IdeaEngine] 步骤2：润色草稿，核心记忆: {len(core_memory)}, 草稿: {len(weekly_report)}, 总prompt: ~{polish_total}")
                        polish_provider = self.context.get_using_provider() if hasattr(self.context, 'get_using_provider') else None
                        if polish_provider:
                            response = await polish_provider.text_chat(
                                prompt=polish_prompt,
                                contexts=[],
                                temperature=0.3,
                            )
                            polished = extract_text_from_response(response)
                        else:
                            logger.warning("[IdeaEngine] Plan B 步骤2 无 polish provider，保持原内容")
                            polished = ""
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
- 包含章节：背景动机、相关工作、方法论、创新点、实验Benchmark、挑战与解决方案、下一步计划、参考文献
- **章节标题必须使用如下精确格式，不得自行修改**：
  `## 1. 背景动机`、`## 2. 相关工作`、`## 3. 方法论`、`## 4. 创新点`、`## 5. 实验Benchmark`、`## 6. 挑战与解决方案`、`## 7. 下一步计划`、`## 8. 参考文献`
- **扩展原则**：将简短的要点列表扩展为连贯段落，但不能变成全新的内容
- **列表格式**：创新点和挑战与解决方案部分使用数字序号列表（如"1. 挑战一：xxx"）

**正文引用格式（重要，严格遵守）**：
- **在方法名/论文名首次出现的位置直接替换为 markdown 链接**，如文中提到 PanoGS 应写为 `[PanoGS](url)`，提到 FLARE 应写为 `[FLARE](url)`
- **禁止**在句子末尾追加论文全名！错误示例：`PanoGS 提出了xxx。PanoGS: Gaussian-based Panoptic Segmentation...` ← 这种格式绝对不允许
- 正确示例：`[PanoGS](url) 提出利用金字塔三平面构建连续参数化特征空间...`
- 论文全名只允许出现在参考文献章节
- **禁止在正文中使用论文全名或裸URL**
**参考文献格式（重要，严格遵守）**：
- 放在最后一个章节
- 每行一条，**严格格式**：`1. [论文全名](URL)`
- 数字序号列表，URL作为markdown链接
- **禁止**：禁止裸URL、禁止括号内重复URL（如 `URL (URL)` ）、禁止纯文本URL

**图表引用格式（重要，严格遵守）**：
- **禁止在正文中使用任何图片语法**：`!(..)`、`` ![](..) ``、`[图片:...](...)` 一律禁止
- **禁止在正文中编造具体的图/表编号**（如"如图1所示"、"如图2所示"），因为你还不知道哪些图表实际存在
- 如需引用图表，只使用泛指描述（如"相关实验结果如图所示"、"方法流程如图表所示"），不指定编号
- **参考文献章节中禁止出现任何图片路径**，参考文献中的方法图引用一律改为纯文字描述（如"NoPoSplat 方法流程图"），不得出现 /Users/ 等路径

请直接输出润色后的内容："""

                    try:
                        logger.info(f"[IdeaEngine] 直接润色草稿，原始长度: {len(weekly_report)}")
                        polish_provider = self.context.get_using_provider() if hasattr(self.context, 'get_using_provider') else None
                        if polish_provider:
                            response = await polish_provider.text_chat(
                                prompt=simplify_prompt,
                                contexts=[],
                                temperature=0.3,
                            )
                            polished = extract_text_from_response(response)
                        else:
                            logger.warning("[IdeaEngine] 直接润色无 polish provider，保持原内容")
                            polished = ""
                        if polished and len(polished) > 100:
                            weekly_report = polished
                            logger.info(f"[IdeaEngine] 直接润色完成，长度: {len(polished)}")
                        else:
                            logger.warning(f"[IdeaEngine] 润色结果过短，保持原内容")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] 润色失败: {e}，保持原内容")
                else:
                    logger.info("[IdeaEngine] 无LLM provider，跳过润色")

            # 后处理2：移除虚构的图表引用（"如图 X 所示"等，在 prompt 禁止之外加一层保障）
            weekly_report, cleaned_count = self._clean_figure_references(weekly_report)

            # 后处理3：规范化章节标题，确保 str.find() 能精确定位
            weekly_report = self._normalize_section_headings(weekly_report)

            # 手动收集图表信息并确定插入锚点（不修改原文，无占位符）
            figure_infos_from_section, figure_anchors = self._append_figure_section(weekly_report, knowledge)

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
                    )
                    generated_title = extract_text_from_response(title_response)
                    generated_title = generated_title.strip() if generated_title else topic
                    logger.info(f"[IdeaEngine] LLM生成标题: {generated_title}")
                except Exception as e:
                    logger.warning(f"[IdeaEngine] 生成标题失败: {e}，使用原始主题")
                    generated_title = topic

            # 用 chunk 上下文丰富图表 caption（LLM 纯文本）
            if figure_infos_from_section and llm_provider:
                figure_infos_from_section = await self._enrich_figure_captions(
                    figure_infos_from_section, knowledge.get("local_results", []), llm_provider
                )

            # 5. 分离 markdown 图片（lark-cli 路径需单独上传图片）
            clean_text, image_infos = self._strip_markdown_images(weekly_report)
            logger.info(f"[IdeaEngine] 分离出 {len(image_infos)} 张图片，clean_text 长度: {len(clean_text)}")

            # 重新从 clean_text 计算锚点。_append_figure_section 在原稿（含图片行）
            # 上计算的锚点可能落在图片行尾，但图片行已被 strip 移除，导致锚点文本
            # 在实际上传的文档中不存在，--selection-with-ellipsis 匹配失败。
            figure_anchors = self._find_figure_anchors(clean_text)

            # 将 _append_figure_section 收集的图表合并到 image_infos（→ 相关工作末尾）
            if figure_infos_from_section:
                for fi in figure_infos_from_section:
                    image_infos.append({
                        "path": fi["path"],
                        "caption": fi["caption"],
                        "anchor_key": "related_work",
                        "width": None,
                        "height": None,
                        "base64": "",
                    })
                logger.info(f"[IdeaEngine] 合并引用图表: {len(figure_infos_from_section)} 个, "
                            f"锚点: {figure_anchors.get('related_work', 'N/A')!r}")

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

            # 7. 创建文档（仅使用 lark-cli）
            if not self._lark_cli_available():
                return {"error": "lark-cli 未安装，无法创建飞书文档。请安装: npx @larksuite/cli@latest install", "polished_content": weekly_report}

            _plugin_root = Path(__file__).parent.parent
            logger.info("[IdeaEngine] lark-cli 一键 markdown 创建飞书文档")
            try:
                tmp = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.md', prefix='feishu_',
                    dir=str(_plugin_root), delete=False, encoding='utf-8'
                )
                tmp.write(clean_text)
                tmp.close()
                rel_path = os.path.relpath(tmp.name, start=_plugin_root)

                lark_args = ["+create", "--title", generated_title, "--markdown", f"@{rel_path}"]
                if folder_token:
                    lark_args += ["--folder-token", folder_token]

                lark_res = self._call_lark_cli("docs", lark_args, timeout=30, cwd=str(_plugin_root))
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

                document_id = None
                if lark_res["success"] and lark_res.get("data"):
                    data = lark_res["data"]
                    if isinstance(data, dict):
                        inner = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
                        document_id = data.get("document_id") or inner.get("doc_id") or data.get("objToken")
                    if not document_id and isinstance(data, str):
                        document_id = data.strip()
                    if document_id:
                        logger.info(f"[IdeaEngine] lark-cli 创建文档成功: {document_id}")
                    else:
                        logger.warning(f"[IdeaEngine] lark-cli 创建成功但未返回 document_id: {lark_res['data']}")

                if not document_id:
                    return {"error": f"lark-cli 创建文档失败: {lark_res.get('error', '')[:200]}", "polished_content": weekly_report}
            except Exception as e:
                logger.error(f"[IdeaEngine] lark-cli 创建文档异常: {e}")
                return {"error": f"lark-cli 创建文档异常: {e}", "polished_content": weekly_report}

            # 8. 收集 PaperBanana 方法图（→ 方法论末尾）
            images_uploaded = 0
            for fb in figure_blocks:
                img_path = fb.get("path", "")
                if img_path:
                    image_infos.append({
                        "path": img_path,
                        "caption": fb.get("caption", ""),
                        "anchor_key": "methodology",
                        "width": None,
                        "height": None,
                        "base64": "",
                    })

            # 9. 插入所有图片（仅使用 lark-cli +media-insert）
            if image_infos:
                logger.info(f"[IdeaEngine] 开始插入 {len(image_infos)} 张图片")
                for img_info in image_infos:
                    img_path = img_info.get("path", "")
                    img_caption = img_info.get("caption", "")
                    img_width = img_info.get("width")
                    img_height = img_info.get("height")
                    is_temp_file = img_path and os.path.exists(img_path) and "data/temp" in img_path

                    if not img_path or not os.path.exists(img_path):
                        logger.error(f"[IdeaEngine] 图片路径不存在，跳过: {img_path!r}")
                        continue

                    img_path = self._ensure_png(img_path)
                    if not img_width or not img_height:
                        try:
                            from PIL import Image as PILImage
                            with PILImage.open(img_path) as pil_img:
                                orig_w, orig_h = pil_img.size
                            img_width, img_height = orig_w, orig_h
                        except Exception:
                            img_width, img_height = 768, 768

                    rel_path = os.path.relpath(img_path, start=_plugin_root)
                    if rel_path.startswith("..") or os.path.isabs(rel_path):
                        logger.warning(f"[IdeaEngine] 图片路径不在插件目录内，跳过: {img_path}")
                        continue

                    try:
                        lark_args = [
                            "+media-insert",
                            "--doc", f"https://feishu.cn/docx/{document_id}",
                            "--type", "image",
                            "--file", rel_path,
                            "--width", str(img_width),
                            "--height", str(img_height),
                        ]
                        anchor_key = img_info.get("anchor_key", "related_work")
                        anchor = figure_anchors.get(anchor_key) if figure_anchors else None
                        if anchor:
                            lark_args += ["--selection-with-ellipsis", anchor]
                        if img_caption:
                            lark_args += ["--caption", img_caption]
                        logger.info(f"[IdeaEngine] lark-cli 图片插入 (section={anchor_key}, "
                                    f"anchor={anchor!r})")

                        lark_res = self._call_lark_cli(
                            "docs", lark_args, timeout=30,
                            cwd=str(_plugin_root),
                        )
                        if lark_res["success"]:
                            images_uploaded += 1
                            logger.info(f"[IdeaEngine] lark-cli 图片插入成功: {rel_path}")
                        else:
                            logger.warning(f"[IdeaEngine] lark-cli 图片插入失败: {lark_res.get('error', '')[:150]}")
                    except Exception as e:
                        logger.warning(f"[IdeaEngine] lark-cli 图片插入异常: {e}")

                    if is_temp_file and os.path.exists(img_path):
                        try:
                            os.unlink(img_path)
                        except OSError:
                            pass

                logger.info(f"[IdeaEngine] 图片插入完成: {images_uploaded}/{len(image_infos)} 张")

            url = f"https://feishu.cn/docx/{document_id}"
            return {
                "success": True,
                "document_id": document_id,
                "url": url,
                "images_uploaded": images_uploaded,
                "polished_content": weekly_report
            }

        except Exception as e:
            logger.error(f"[IdeaEngine] 飞书文档创建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e), "polished_content": ""}

    # ==================== 图表引用清洗（不用正则） ====================

    @classmethod
    def _clean_figure_references(cls, text: str) -> tuple[str, int]:
        """移除文本中所有"如图 X 所示"引用（逐字符扫描，不用正则）。

        支持：如图 1 所示、如图1所示、(如图 1 所示)、（如图 1 所示）、
        如图 1-3 所示、如图 1、2、3 所示 等变体。
        不匹配泛指描述（"相关实验结果如图所示"）——必须有数字。
        """
        removed = 0
        i = 0
        n = len(text)
        # 从后往前处理，避免索引偏移
        spans_to_remove: list[tuple[int, int]] = []

        while i < n:
            # 找 "如图"
            pos = text.find('如图', i)
            if pos == -1:
                break

            # 跳过 "如图" 本身
            j = pos + 2

            # 跳过空白
            while j < n and text[j] in (' ', '\t', '　'):
                j += 1

            # 必须有数字
            if j >= n or not text[j].isdigit():
                i = pos + 2
                continue

            # 扫描数字及分隔符
            j += 1
            while j < n:
                ch = text[j]
                if ch.isdigit():
                    j += 1
                elif ch in ('-', '–', ',', '，', '、'):
                    # 范围或列表分隔符，后面必须有数字
                    j += 1
                    while j < n and text[j] in (' ', '\t', '　'):
                        j += 1
                    if j < n and text[j].isdigit():
                        j += 1
                    else:
                        break
                else:
                    break

            # 期望 "所示"
            k = j
            while k < n and text[k] in (' ', '\t', '　'):
                k += 1
            if k + 2 > n or text[k:k+2] != '所示':
                i = pos + 2
                continue

            # 确定移除范围
            remove_start = pos
            remove_end = k + 2

            # 扩展：前导开括号
            if remove_start > 0 and text[remove_start - 1] in ('(', '（'):
                remove_start -= 1
            # 扩展：尾随闭括号
            if remove_end < n and text[remove_end] in (')', '）'):
                remove_end += 1
            # 扩展：尾随分隔标点（不含句号，避免断句残留逗号）
            if remove_end < n and text[remove_end] in (',', '，', ';', '；'):
                remove_end += 1

            spans_to_remove.append((remove_start, remove_end))
            i = remove_end

        # 从后往前删除
        result = text
        for start, end in reversed(spans_to_remove):
            result = result[:start] + result[end:]
            removed += 1

        if removed:
            logger.info(f"[IdeaEngine] _clean_figure_references: 移除 {removed} 个虚构图表引用")
        return result, removed

    # ==================== larksuite/cli 集成 ====================

    _LARK_CLI_SUBCMDS = frozenset({"doc", "docs", "wiki", "calendar", "sheets", "base", "im", "help"})

    _IMG_MD_RE = re.compile(r'!\[(.*?)\]\(([/][^)]+\.(?:png|jpg|jpeg|webp|gif))\)')

    @classmethod
    def _strip_markdown_images(cls, text: str) -> tuple[str, list[dict]]:
        """去除 markdown 中的 ![...](path) 图片行，返回 (clean_text, image_infos).

        每行如果匹配到完整图片语法则整行移除，其余文本原样保留。
        从 caption 中提取图/表编号作为定位锚点。
        """
        images = []
        clean_lines = []
        for line in text.split('\n'):
            stripped = line.strip()
            m = cls._IMG_MD_RE.match(stripped)
            if m:
                caption = m.group(1)
                path = m.group(2)
                if not os.path.exists(path):
                    logger.warning(f"[IdeaEngine] _strip_markdown_images: 图片路径不存在，从 clean_text 移除但不加入上传队列: {path}")
                    continue
                images.append({"path": path, "caption": caption, "anchor_key": "related_work"})
                continue
            clean_lines.append(line)
        return '\n'.join(clean_lines), images

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
