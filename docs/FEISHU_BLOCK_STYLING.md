# 飞书文档导出技术方案

## 架构概览

`/idea tofeishu` 使用 **lark-cli 单路径**创建飞书文档（MCP 路径已移至 `legacy/feishu_mcp.py`）。

| 步骤 | 操作 | 工具/方法 |
|------|------|----------|
| 1 | 生成/润色草稿 | `_generate_initial_draft_vlm()` + Plan B LLM polish |
| 2 | 后处理：去虚构引用 | `_clean_figure_references()` |
| 3 | 后处理：规范化章节标题 | `_normalize_section_headings()` |
| 4 | 收集 knowledge 图表 | `_append_figure_section()` → `figure_infos` |
| 5 | 分离 markdown 图片 | `_strip_markdown_images()` → `clean_text` + `image_infos` |
| 6 | 创建文档 | `lark-cli docs +create --markdown @file` |
| 7 | 重算锚点 | `_find_figure_anchors(clean_text)` |
| 8 | 插入图片 | `lark-cli docs +media-insert --selection-with-ellipsis` |

## 规范化章节体系

8 个 canonical 章节标题，prompt 强制要求 + 运行态 `_normalize_section_headings()` 归一波折号、空白等 LLM 变体：

```
## 1. 背景动机
## 2. 相关工作
## 3. 方法论
## 4. 创新点
## 5. 实验Benchmark
## 6. 挑战与解决方案
## 7. 下一步计划
## 8. 参考文献
```

**空白归一化**：匹配前去除所有空格/制表符，容错 LLM 在中文与英文间插入空格（如 "实验 Benchmark" → "实验Benchmark"）。

## 双锚点图片定位

引用图表（knowledge figures）固定插入到相关工作末尾，方法论图表（PaperBanana）固定插入到方法论末尾。

| 锚点 key | 定位章节 | 适用图片 |
|----------|---------|---------|
| `related_work` | `## 2. 相关工作` 末尾最后一句 | 论文引用图表（`_append_figure_section` 收集） |
| `methodology` | `## 3. 方法论` 末尾最后一句 | PaperBanana 方法图（`figure_blocks`） |

锚点计算使用 `_find_section_anchor()`，纯 `str.find()` 定位章节边界，取区间内最后一行非空内容的末尾 60 字符作为锚点文本。

### 锚点重新计算

`create_feishu_document()` 在 `_strip_markdown_images()` 后从 `clean_text` 重新调用 `_find_figure_anchors()`。原因：`_append_figure_section()` 在原稿（含 `![...](path)` 图片行）上计算锚点，若锚点文本落在图片行尾部，`_strip_markdown_images()` 移除该行后，`--selection-with-ellipsis` 在实际上传的文档中匹配不到该文本，导致插入失败。

## 图表引用清洗

`_clean_figure_references()` 用逐字符扫描（非正则）移除 LLM 编造的 "如图 X 所示" 类引用：

- 扫描到 `如图` → 跳过空白 → 验证后续为数字 → 扫描数字及分隔符（`-`、`–`、`,`、`，`、`、`）→ 期望 `所示`
- 移除匹配片段，包括可选的前导括号 `（）` 和尾随逗号/分号分隔符
- **保留句号**（`.`、`。`），避免断句残留

## lark-cli 图片插入 API

```
lark-cli docs +media-insert \
  --doc https://feishu.cn/docx/{document_id} \
  --type image \
  --file {relative_path} \
  --width {w} --height {h} \
  --caption "{caption}" \
  --selection-with-ellipsis "{anchor_text}"
```

- `--file` 必须是项目根目录下的相对路径
- `--selection-with-ellipsis` 在文档中搜索锚点文本，找到后在其位置插入图片
- 锚点不存在时，图片追加到文档末尾

## lark-cli 创建文档 API

```
lark-cli docs +create \
  --title "{title}" \
  --markdown @{relative_md_path} \
  [--folder-token {token}]
```

- `--markdown @file` 一键上传 markdown 文件，自动处理 `[text](url)` → 可点击链接、`**bold**` → 加粗等样式

## 辅助方法

| 方法 | 位置 | 说明 |
|------|------|------|
| `_call_lark_cli()` | `idea/feishu_doc.py` | lark-cli 子进程封装，含超时和错误处理 |
| `_lark_cli_available()` | `idea/feishu_doc.py` | 检测 lark-cli 是否已安装 |
| `_strip_markdown_images()` | `idea/feishu_doc.py` | 分离 `![...](path)` 图片行 |
| `_ensure_png()` | `idea/markdown.py` | webp/非 PNG 格式转为 PNG |

## 文件位置

- lark-cli 路径实现：`idea/feishu_doc.py` → `create_feishu_document()`
- 章节规范化 + 锚点计算：`idea/markdown.py` → `_normalize_section_headings()`、`_find_figure_anchors()`、`_find_section_anchor()`
- 图表引用清洗：`idea/feishu_doc.py` → `_clean_figure_references()`
- MCP 遗留代码：`legacy/feishu_mcp.py`

---

**最后更新**: 2026-05-23
