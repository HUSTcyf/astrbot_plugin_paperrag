# 飞书文档块样式更新技术方案

## 背景

在向飞书文档插入列表块时，需要支持列表项内的 Markdown 样式（加粗、斜体、行内代码等）。由于飞书 MCP 的 `batch_create_feishu_blocks` 和 `createListBlock` API 只能创建带纯文本的列表块，样式需要通过 `batch_update_feishu_block_text` 单独更新。

## 核心问题

### 问题 1：batch_create_feishu_blocks 不返回 block_id

`batch_create_feishu_blocks` 工具的响应格式为：

```json
{
  "totalBlocksCreated": 3,
  "nextIndex": 3,
  "document_revision_id": "rev_xxx"
}
```

**普通块（列表/文本/标题）的 block_id 不在响应中返回**，只有图片块和白板块会通过 `imageBlocksInfo.blockIds` 返回 ID。

### 问题 2：列表块不支持直接创建带样式

飞书的列表块 API (`createListBlock`) 只接受纯文本字符串，不支持 `text_run` 样式的 `elements` 数组。因此必须分两步：
1. 创建列表块（含纯文本内容）
2. 调用 `batch_update_feishu_block_text` 更新为带样式的 `textElements`

## 解决方案：get_feishu_document_blocks + 内容匹配

### 流程

1. **创建所有块**：列表块先用纯文本创建，图片块用两步插入
2. **获取文档块列表**：调用 `get_feishu_document_blocks` 获取文档中所有块（含 ID）
3. **按内容匹配**：遍历文档块，按 `block_type`（12=bullet, 13=ordered）和文本内容匹配找到列表块
4. **更新样式**：调用 `batch_update_feishu_block_text` 批量更新列表块的文本样式

### 关键代码逻辑

```python
import re

def _normalize_text(t: str) -> str:
    """归一化空白符：将多个连续空白符合并为一个，去除首尾空白"""
    return re.sub(r'\s+', ' ', t).strip()

# 1. 记录需要更新样式的列表块
list_items_to_update: list[tuple[str, dict]] = []  # (原始文本内容, _textStyles)
for b in all_blocks:
    if b.get("blockType") == "list" and b.get("_textStyles"):
        list_content = b.get("options", {}).get("list", {}).get("content", "")
        list_items_to_update.append((list_content, b.get("_textStyles")))

# 2. 调用 get_feishu_document_blocks 获取块 ID
blocks_result = await get_blocks_tool.call(ctx_wrapper, documentId=document_id)

# 3. 解析响应（使用 json.JSONDecoder 自动忽略尾部内容）
all_doc_blocks = []
try:
    if blocks_text:
        decoder = json.JSONDecoder()
        all_doc_blocks, end_pos = decoder.raw_decode(blocks_text)
except Exception as e:
    logger.warning(f"JSON 解析失败: {e}")

# 4. 按文本内容匹配列表块（空白符归一化 + 防重复匹配）
updates = []
matched_block_ids = set()
for list_text, text_styles in list_items_to_update:
    norm_list_text = _normalize_text(list_text)
    for block in all_doc_blocks:
        block_id = block.get("block_id", "")
        if block_id in matched_block_ids:
            continue
        block_type = block.get("block_type", 0)
        if block_type not in (12, 13):  # bullet=12, ordered=13
            continue
        block_data = block.get("bullet") or block.get("ordered") or {}
        elements = block_data.get("elements", [])
        block_text = "".join(
            elem.get("text_run", {}).get("content", "")
            for elem in elements
        )
        if _normalize_text(block_text) == norm_list_text:
            matched_block_ids.add(block_id)
            text_elements = [
                {"text": ts.get("text", ""), "style": ts.get("style", {})}
                for ts in text_styles
            ]
            updates.append({"blockId": block_id, "textElements": text_elements})
            break

# 5. 批量更新样式（每批最多 50 个）
for i in range(0, len(updates), 50):
    batch = updates[i:i + 50]
    await update_text_tool.call(ctx_wrapper, documentId=document_id, updates=batch)
```

## JSON 解析注意事项

`get_feishu_document_blocks` 返回的响应格式为：
```
[{"block_id": "...", ...}]\n\n🖼️ 检测到 N 个图片块...   ← JSON 数组 + 追加提示文本
```

**关键**：`JSON.stringify(blocks, null, 2)` 格式化后的 JSON 数组后，追加了特殊块提示文本。如果直接 `json.loads()` 会因尾部多余内容报 `"Extra data"` 错误。

**正确方案**：使用 `json.JSONDecoder().raw_decode()` 自动解析第一个完整 JSON 数组并忽略尾部内容：
```python
decoder = json.JSONDecoder()
all_doc_blocks, end_pos = decoder.raw_decode(blocks_text)
logger.info(f"JSON 解析成功，自动忽略尾部 {len(blocks_text) - end_pos} 字符")
```

## 风险与限制

1. **文本匹配依赖内容唯一性**：如果两个列表项文本完全相同，只能匹配第一个（后续会被跳过），后者样式不会被应用
2. **Unicode 标准化**：暂未做 Unicode NFC/NFD 标准化，罕见字符可能匹配失败
3. **不支持跨块拆分**：如果列表项内容超长被拆分到多个块，无法正确匹配

## Feishu 块类型参考

| block_type | 类型 | 说明 |
|------------|------|------|
| 1 | doc | 文档根块 |
| 2 | text | 文本块 |
| 3-11 | heading | 标题块（1-9级） |
| 12 | bullet | 无序列表 |
| 13 | ordered | 有序列表 |
| 14 | code | 代码块 |
| 27 | image | 图片块 |
| 31 | table | 表格块 |
| 32 | table_cell | 表格单元格 |
| 40 | add_ons | Mermaid 等特殊块 |
| 43 | board | 白板块 |

## 相关 MCP 工具

| 工具名 | 功能 |
|--------|------|
| `batch_create_feishu_blocks` | 批量创建块 |
| `batch_update_feishu_block_text` | 批量更新块的文本样式 |
| `get_feishu_document_blocks` | 获取文档所有块（含 ID） |
| `upload_and_bind_image_to_block` | 上传图片并绑定到块 |

## 文件位置

- 实现：`idea_engine.py` 的 `test_feishu_markdown_formats` 和 `create_feishu_document` 方法
- 飞书 MCP：`/Users/chenyifeng/AstrBot/data/Feishu-MCP/`

---

**最后更新**: 2026-04-13
