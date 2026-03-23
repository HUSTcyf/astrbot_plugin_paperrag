# Paper RAG Plugin - 开发指南

本文档面向开发者，介绍插件的技术架构、实现细节和开发指南。

## 📋 目录

1. [最新更新](#最新更新)
2. [技术架构](#技术架构)
3. [核心组件](#核心组件)
4. [设计模式](#设计模式)
5. [开发工作流](#开发工作流)
6. [依赖安装](#依赖安装)
7. [故障排除](#故障排除)

---

## 🆕 最新更新 (2026-03-24)

### v2.4 主要更新

**Ollama本地Embedding支持**
- ✅ 通过HTTP API调用Ollama服务，避免进程冲突
- ✅ 完全免费、无限制、隐私保护
- ✅ 支持BGE-M3等多种本地模型
- ✅ 自动批处理和重试机制

**API模式优化**
- ✅ 批量Embedding处理（自动分批，符合API限制）
- ✅ 支持多种Embedding Provider（Gemini、OpenAI等）

**类型安全完善**
- ✅ 修复Pylance类型检查问题
- ✅ 优化类型注解，提升代码质量

**运行时错误修复**
- ✅ Clear函数处理None返回值
- ✅ 多模态相对导入修复

---

## 🏗️ 技术架构

### 整体架构

```
AstrBot框架 → PaperRAGPlugin → PaperRAGEngine → 外部依赖
                    ↓              ↓
                缓存层(LRU)    Embedding/Milvus/LLM
```

### 数据流程

```
PDF → 多模态提取 → 智能分块 → 批量向量化 → Milvus存储 → RAG生成
```

---

## 🧩 核心组件

### 1. PaperRAGPlugin (main.py)
**职责**: 插件入口、命令注册、缓存管理、懒加载引擎

### 2. PaperRAGEngine (rag_engine.py)
**职责**: 文档解析、向量化、检索、RAG生成

```python
class PaperRAGEngine:
    def __init__(self, config, context):
        self._store = None  # 延迟初始化
        self._embedding_wrapper = None
        self._llm_provider = None

    @property
    def embedding(self) -> Union[EmbeddingProviderWrapper, OllamaEmbeddingProvider]:
        """获取Embedding Provider（API或Ollama模式）"""
        self._ensure_embedding_initialized()
        return self._embedding_wrapper
```

### 3. EmbeddingProviderWrapper (rag_engine.py)
**职责**: 统一不同Provider接口，优先使用批量方法，自动分批处理

```python
async def embed(self, texts: str | List[str]) -> List[List[float]]:
    # 自动分批（Gemini限制100个/批）
    if len(texts) > BATCH_SIZE_LIMIT:
        for i in range(0, len(texts), BATCH_SIZE_LIMIT):
            batch = texts[i:i + BATCH_SIZE_LIMIT]
            embeddings = await embed_batch(batch)
```

### 4. OllamaEmbeddingProvider (ollama_embedding.py)
**职责**: 通过HTTP API调用本地Ollama服务进行embedding

```python
class OllamaEmbeddingProvider:
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 并发批处理
        tasks = [self._embed_single(text) for text in batch]
        batch_embeddings = await asyncio.gather(*tasks)
```

**优势**：
- 独立进程运行，无冲突
- 完全免费，无API限制
- 支持多种本地模型

### 5. MilvusStore (rag_engine.py)
**职责**: 向量数据库操作（增删查）

### 6. AdaptiveReranker (reranker.py)
**职责**: 智能重排序，提升检索精度15-25%

---

## 🎨 设计模式

### 1. 单例模式（懒加载）
延迟初始化资源密集型组件，减少启动时间和内存占用

### 2. 包装器模式
统一不同Provider接口，屏蔽差异，易于扩展

### 3. 策略模式
根据文件类型选择解析策略（PDF/DOCX/TXT）

### 4. 缓存模式（LRU）
减少重复计算和API调用，提升响应速度

---

## 💻 开发工作流

### 环境准备
```bash
cd ~/AstrBot/data/plugins/astrbot_plugin_paperrag
pip install -r requirements.txt
```

### 开发流程
```bash
# 1. 编辑代码
vim rag_engine.py

# 2. 语法检查
python3 -m py_compile rag_engine.py

# 3. 重启AstrBot（自动重载插件）

# 4. 测试功能
/paper add  # 导入PDF测试
```

### 调试技巧
```python
from astrbot.api import logger

logger.debug("调试信息")    # 开发调试
logger.info("普通信息")     # 重要事件
logger.warning("警告信息")  # 潜在问题
logger.error("错误信息", exc_info=True)  # 错误+堆栈
```

---

## 📦 依赖安装

### 核心依赖
```bash
pip install pymilvus[milvus_lite] PyMuPDF python-docx pdfplumber pillow
```

### 可选依赖（多模态与重排序）

**安装命令**：
```bash
# 使用requirements.txt（推荐）
~/.local/share/uv/tools/astrbot/bin/python -m pip install -r requirements.txt

# 手动安装
~/.local/share/uv/tools/astrbot/bin/python -m pip install \
  'transformers>=4.40.0' \
  'torch>=2.0.0' \
  'FlagEmbedding>=1.2.0'
```

**验证安装**：
```bash
~/.local/share/uv/tools/astrbot/bin/python -c "from FlagEmbedding import FlagModel; print('✅ 成功')"
```

---

## 🐛 故障排除

### 1. TypeError: 'NoneType' object is not callable
**原因**: Provider未正确初始化
**解决**: 检查Provider是否可用，验证embeddings返回值

### 2. Milvus连接失败
**原因**: 数据库目录权限或路径问题
**解决**: `Path(uri).parent.mkdir(parents=True, exist_ok=True)`

### 3. 向量维度不匹配
**原因**: embed_dim与模型不匹配
**解决**: Gemini用768维，OpenAI用1536维

### 4. FlagEmbedding未安装
**原因**: transformers版本不兼容
**解决**: 确保使用 `transformers>=4.40.0`

### 5. Gemini API批量限制错误
**症状**: `at most 100 requests can be in one batch`
**原因**: Gemini Embedding API单次批量请求最多100个文本
**解决**: 插件已自动实现分批处理，无需手动干预

**实现细节**：
```python
# 自动分批处理（BATCH_SIZE_LIMIT = 100）
if len(texts) > 100:
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        embeddings = await provider.get_embeddings(batch)
```

### 6. 分块无限循环
**原因**: chunk_overlap >= chunk_size / 2
**解决**: 设置 `chunk_overlap = 0`

### 7. Optional成员访问警告
**解决**: 使用`cast()`或类型守卫

```python
encoder = cast(VisionEncoder, self.vision_encoder)
vision_emb = encoder.encode_image(image_bytes)
```

### 8. Ollama接口兼容性错误
**症状**: `无法访问类"OllamaEmbeddingProvider"的属性"get_query_embedding"`
**原因**: OllamaEmbeddingProvider缺少get_query_embedding方法
**解决**: 已在ollama_embedding.py中添加所有必需方法

**添加的方法**：
```python
async def get_text_embedding(self, text: str) -> List[float]:
    """获取单个文本的embedding"""
    result = await self.embed([text])
    return result[0] if result and len(result) > 0 else []

async def get_query_embedding(self, query: str) -> List[float]:
    """获取查询嵌入"""
    return await self.get_text_embedding(query)
```

### 9. Dataclass可变默认值错误
**症状**: `mutable default <class 'dict'> for field ollama_config is not allowed`
**原因**: Python dataclass不能直接使用dict/list作为默认值
**解决**: 使用field(default_factory=dict)

**错误代码**：
```python
ollama_config: dict = {}  # ❌ 错误
```

**正确代码**：
```python
from dataclasses import dataclass, field

ollama_config: dict = field(default_factory=dict)  # ✅ 正确
```

---

## 🔗 参考资源

- [AstrBot GitHub](https://github.com/AstrBotDevs/AstrBot)
- [Milvus Lite 文档](https://milvus.io/docs/milvus_lite.md)
- [PyMuPDF 文档](https://pymupdf.readthedocs.io/)
