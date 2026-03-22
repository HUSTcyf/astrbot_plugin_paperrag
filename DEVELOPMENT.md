# Paper RAG Plugin - 开发指南

本文档面向开发者，介绍插件的技术架构、实现细节和开发指南。

## 📋 目录

1. [最新更新](#最新更新)
2. [技术架构](#技术架构)
3. [核心组件](#核心组件)
4. [设计模式](#设计模式)
5. [开发工作流](#开发工作流)
6. [测试指南](#测试指南)
7. [故障排除](#故障排除)

---

## 🆕 最新更新 (2026-03-23)

### v2.1 更新内容

#### 性能优化
- ✅ **批量 Embedding**：每篇PDF仅1次API调用，节省80-90% RPD配额
  ```python
  # 优化前：N chunks = N次API调用
  for chunk in chunks:
      embedding = await get_text_embedding(chunk)

  # 优化后：N chunks = 1次API调用
  embeddings = await embed(chunks)
  ```
  - RPD处理能力提升：1000 RPD从~20篇PDF → ~1000篇PDF

#### 代码质量提升
- ✅ **日志精简**：删除70-80%冗余日志
  - 移除 `[STEP X]`、`[WRAPPER STEP X]`、`[EMBED STEP X]` 调试日志
  - 移除连接模式、索引类型等详细日志
  - 保留错误和关键警告

#### Bug修复
- ✅ **类型安全**：修复所有Pylance类型检查警告
  - `drop_collection()` 添加 `await`
  - `hybrid_search()` 参数类型：`str` → `Optional[str]`
  - `vision_encoder.encode_image()` 添加类型断言

#### 文件过滤
- ✅ **Meta文件过滤**：自动跳过macOS `._` 开头的元数据文件
  ```python
  doc_files = [f for f in doc_files if not f.name.startswith("._")]
  ```

---

## 🏗️ 技术架构

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                  AstrBot 框架                       │
│  ┌──────────────────────────────────────────────┐  │
│  │         PaperRAGPlugin (main.py)            │  │
│  │  - 命令注册                                  │  │
│  │  - 配置管理                                  │  │
│  │  - 缓存层 (LRU)                              │  │
│  │  - 懒加载引擎                                │  │
│  └──────────────────────────────────────────────┘  │
│                       ↓                            │
│  ┌──────────────────────────────────────────────┐  │
│  │       PaperRAGEngine (rag_engine.py)        │  │
│  │  - 文档解析器                                │  │
│  │  - Embedding包装层                          │  │
│  │  - Milvus存储层                             │  │
│  │  - RAG生成器                                 │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                   外部依赖                          │
│  - AstrBot Provider (Embedding/LLM)               │
│  - Milvus Lite (向量数据库)                       │
│  - PyMuPDF/pdfplumber (PDF解析)                   │
└─────────────────────────────────────────────────────┘
```

### 数据处理流程

```
PDF文件
   ↓
┌─────────────────────────────────────┐
│     多模态提取 (multimodal_*)      │
│  ┌──────────────────────────────┐  │
│  │ 文本提取 → PyMuPDF          │  │
│  │ 图片提取 → PyMuPDF + NMS     │  │
│  │ 表格提取 → pdfplumber        │  │
│  │ 公式提取 → LaTeX正则         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│       智能分块 (semantic_chunker)   │
│  - 文本 → 语义分块                 │
│  - 图片/表格/公式 → 完整保留        │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│       批量向量化 🆕                  │
│  - 收集所有chunks                   │
│  - 一次性批量调用embed()            │
│  - 节省80-90% RPD                   │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│       Milvus存储                    │
│  - 文本向量 + 元数据                │
│  - COSINE相似度搜索                 │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│       RAG生成                       │
│  - 检索相关片段                     │
│  - 构建prompt                       │
│  - LLM生成答案                      │
└─────────────────────────────────────┘
```

### 图片去重机制

```
图片提取 → 四级过滤 → 保留唯一图片
   ↓
【第一级】尺寸过滤
   - 宽度 < 10px → 移除
   - 高度 < 10px → 移除
   - 使用原始尺寸（非调整后）

【第二级】图注去重 🆕
   - 提取图注编号（如 "Figure 1"）
   - 相同图注 → 判定为同图的不同版本
   - 只保留尺寸最大的图片
   - 过滤缩略图、预览图

【第三级】尺寸去重
   - 相同 (宽, 高) → 判定为重复
   - 保留第一次出现的图片
   - 过滤logo、图标等重复元素

【第四级】NMS位置去重
   - 按Bbox面积降序排序
   - 逐对比较IoU（交并比）
   - IoU > 0.5 → 判定为重叠
   - 保留面积大的，移除面积小的
```

**NMS去重原理**：
```
IoU (Intersection over Union) = 交集面积 / 并集面积

示例：
图片A: Bbox(100, 100, 200, 200), 面积=10000
图片B: Bbox(110, 110, 190, 190), 面积=6400

交集: (190-110) × (190-110) = 6400
并集: 10000 + 6400 - 6400 = 10000
IoU = 6400 / 10000 = 0.64 > 0.5 → 判定为重叠

结果：保留图片A（面积大），移除图片B
```

---

## 🧩 核心组件

### 1. PaperRAGPlugin (main.py)

**职责**：插件入口、命令注册、缓存管理

```python
@register("paper_rag", "Author", "Description", "1.0.0")
class PaperRAGPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.enabled = config.get("enabled", True)

        # 缓存配置
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_ttl = config.get("cache_ttl_seconds", 3600)
        self.cache_max_size = config.get("cache_max_entries", 100)
        self._response_cache = {}

        # RAG引擎（懒加载）
        self._engine = None

    def _get_engine(self) -> Optional[PaperRAGEngine]:
        """单例模式获取引擎"""
        if self._engine is None:
            rag_config = RAGConfig(...)
            self._engine = PaperRAGEngine(rag_config, self.context)
        return self._engine
```

### 2. PaperRAGEngine (rag_engine.py)

**职责**：文档解析、向量化、检索、RAG生成

```python
class PaperRAGEngine:
    def __init__(self, config: RAGConfig, context):
        self.config = config
        self.context = context

        # 延迟初始化组件
        self._store = None
        self._embedding_wrapper = None
        self._llm_provider = None

    @property
    def store(self) -> MilvusStore:
        """延迟初始化Milvus存储"""
        if self._store is None:
            self._store = MilvusStore(...)
        return self._store

    @property
    def embedding(self) -> EmbeddingProviderWrapper:
        """延迟初始化Embedding包装器"""
        if self._embedding_wrapper is None:
            provider = self._get_provider(self.config.embedding_provider_id)
            self._embedding_wrapper = EmbeddingProviderWrapper(provider)
        return self._embedding_wrapper
```

### 3. EmbeddingProviderWrapper (rag_engine.py)

**职责**：统一AstrBot Embedding Provider接口

```python
class EmbeddingProviderWrapper:
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    async def embed(self, texts: str | List[str]) -> List[List[float]]:
        """
        批量获取嵌入向量 🆕

        优化：从N次调用 → 1次批量调用
        """
        if isinstance(texts, str):
            texts = [texts]

        # 智能调用（同步/异步自适应）
        if hasattr(self.provider, "embed_texts"):
            embeddings = await self.provider.embed_texts(texts)
        elif hasattr(self.provider, "embed"):
            embeddings = await self.provider.embed(texts)
        else:
            raise Exception("不支持的embedding方法")

        return embeddings
```

### 4. MilvusStore (rag_engine.py)

**职责**：向量数据库操作（增删查）

```python
class MilvusStore:
    def __init__(self, uri: str, collection_name: str, dim: int):
        self.uri = uri
        self.collection_name = collection_name
        self.dim = dim
        self._alias = f"paper_rag_{collection_name}"

    async def add_documents(self, documents: List[Dict]) -> int:
        """批量添加文档"""
        await self._ensure_collection()
        collection = Collection(self.collection_name, using=self._alias)

        data = [
            {
                "vector": doc["embedding"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {})
            }
            for doc in documents
        ]

        collection.insert(data)
        collection.flush()
        return len(documents)

    async def search(self, query_vector: List[float], top_k: int):
        """向量搜索"""
        collection = Collection(self.collection_name, using=self._alias)
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["text", "metadata"]
        )
        return results
```

### 5. MultiModalIngestionEngine (multimodal_rag_engine.py)

**职责**：多模态PDF提取和向量化

```python
class MultiModalIngestionEngine:
    def __init__(self, vision_encoder, text_embedding_fn, ...):
        self.vision_encoder = vision_encoder
        self.text_embedding_fn = text_embedding_fn
        self.extractor = MultimodalPDFExtractor(...)

        # 优雅降级：检查依赖可用性
        self.vision_available = (
            vision_encoder is not None and
            getattr(vision_encoder, 'is_available', False)
        )

    async def process_pdf(self, pdf_path: str):
        """处理PDF（批量embedding）"""
        extracted = self.extractor.extract(pdf_path)
        chunks = []

        # 批量处理图片
        if extracted.images:
            image_texts = [img.caption for img in extracted.images]
            text_embeddings = await self.text_embedding_fn(image_texts)
            # ... 处理图片

        # 批量处理表格
        if extracted.tables:
            table_texts = [table.markdown for table in extracted.tables]
            text_embeddings = await self.text_embedding_fn(table_texts)
            # ... 处理表格

        return chunks
```

---

## 🎨 设计模式

### 1. 单例模式（懒加载）

**目的**：延迟初始化资源密集型组件

```python
class PaperRAGEngine:
    def __init__(self):
        self._store = None  # 延迟初始化

    @property
    def store(self):
        if self._store is None:
            self._store = MilvusStore(...)  # 首次访问时初始化
        return self._store
```

**优势**：
- 减少启动时间
- 降低内存占用
- 按需加载资源

### 2. 包装器模式

**目的**：统一不同Provider的接口

```python
class EmbeddingProviderWrapper:
    def embed(self, texts):
        # 适配不同Provider的方法
        if hasattr(self.provider, "embed_texts"):
            return self.provider.embed_texts(texts)
        elif hasattr(self.provider, "embed"):
            return self.provider.embed(texts)
        # ... 其他适配逻辑
```

**优势**：
- 屏蔽Provider差异
- 统一调用接口
- 易于扩展新Provider

### 3. 策略模式

**目的**：支持多种文档解析策略

```python
async def _parse_document(self, file_path: str):
    file_type = self._detect_file_type(file_path)

    # 根据文件类型选择解析策略
    if file_type == 'pdf':
        return await self._parse_pdf(file_path)
    elif file_type == 'docx':
        return await self._parse_docx(file_path)
    elif file_type == 'text':
        return await self._parse_text(file_path)
```

**优势**：
- 易于添加新格式支持
- 解耦解析逻辑
- 提高代码可维护性

### 4. 缓存模式（LRU）

**目的**：提升重复查询性能

```python
def _get_cached_response(self, cache_key: str):
    if cache_key in self._response_cache:
        cached_data, timestamp = self._response_cache[cache_key]
        if time.time() - timestamp < self.cache_ttl:
            return cached_data  # 命中缓存

    # 未命中，计算并缓存
    result = await self._compute_response(cache_key)
    self._set_cached_response(cache_key, result)
    return result
```

**优势**：
- 减少重复计算
- 降低API调用
- 提升响应速度

---

## 💻 开发工作流

### 环境准备

```bash
# 进入插件目录
cd ~/AstrBot/data/plugins/astrbot_plugin_paperrag

# 安装依赖
uv pip install -r requirements.txt

# 或使用pip
pip install -r requirements.txt
```

### 开发流程

```bash
# 1. 编辑代码
vim rag_engine.py  # 或 main.py, multimodal_rag_engine.py

# 2. 语法检查
python3 -m py_compile rag_engine.py

# 3. 重启AstrBot
# AstrBot会自动重新加载插件

# 4. 测试功能
# 在聊天中输入命令测试
```

### 调试技巧

```python
from astrbot.api import logger

# 使用不同级别的日志
logger.debug("调试信息")    # 开发调试
logger.info("普通信息")     # 重要事件
logger.warning("警告信息")  # 潜在问题
logger.error("错误信息")    # 错误发生

# 异常处理
try:
    result = await some_operation()
except Exception as e:
    logger.error(f"操作失败: {e}", exc_info=True)  # 包含堆栈信息
```

### 类型注解

```python
from typing import List, Dict, Any, Optional, cast

async def search(
    self,
    query: str,
    mode: str = "rag"
) -> Dict[str, Any]:
    """明确的类型注解"""
    pass

def _get_engine(self) -> Optional[PaperRAGEngine]:
    """可能返回None的返回类型"""
    pass

# 类型断言（解决Optional成员访问）
encoder = cast(VisionEncoder, self.vision_encoder)
vision_emb = encoder.encode_image(image_bytes)
```

---

## 🧪 测试指南

### 一键测试

```bash
cd ~/AstrBot/data/plugins/astrbot_plugin_paperrag
python test_pdf.py /path/to/paper.pdf
```

### 测试输出解读

#### 测试1: 依赖检查

```
✅ pymilvus        - Milvus 向量数据库
✅ fitz            - PyMuPDF (PDF解析)
✅ pdfplumber      - 表格提取
❌ transformers    - HuggingFace (可选)
```

**诊断**：
- ✅ 所有核心依赖 → 继续测试
- ❌ 缺少核心依赖 → `pip install pymilvus PyMuPDF pdfplumber`
- ⚠️ 缺少可选依赖 → 正常，不影响使用

#### 测试2: PDF基础提取

```
📄 总页数: 16
📊 有文本的页数: 16/16
📊 总字符数: 58940
```

**诊断**：
- 文本提取成功 → 继续测试
- 无文本（扫描版）→ 需要OCR转换
- 文本很少 → 可能不适合RAG

#### 测试3: 多模态提取

```
📊 提取结果:
  • 图片数量: 12
  • 表格数量: 5
  • 公式数量: 23
```

**诊断**：
- 多模态提取成功 → 功能正常
- 部分功能禁用 → 依赖不可用，但核心功能正常

---

## 🐛 故障排除

### 问题1: TypeError: 'NoneType' object is not callable

**症状**：调用embedding时出现类型错误

**原因**：Provider未正确初始化

**解决**：
```python
# 检查Provider是否可用
if not hasattr(provider, 'embed'):
    logger.error(f"Provider {provider_id} 不支持embed方法")
    return None

# 使用前验证
embeddings = provider.embed(texts)
if embeddings is None or len(embeddings) == 0:
    logger.error("Embedding返回空结果")
    return None
```

### 问题2: Milvus连接失败

**症状**：`MilvusException: connection failed`

**原因**：数据库目录权限或路径问题

**解决**：
```python
# 确保目录存在且有写权限
Path(uri).parent.mkdir(parents=True, exist_ok=True)

# 检查文件权限
import os
os.chmod(uri, 0o755)
```

### 问题3: 向量维度不匹配

**症状**：`向量维度必须是64的倍数`或插入失败

**原因**：embed_dim与模型不匹配

**解决**：
```python
# Gemini Embedding 2 Preview: 768维
embed_dim = 768

# OpenAI text-embedding-3-small: 1536维
embed_dim = 1536

# 验证维度
if embed_dim % 64 != 0:
    raise ValueError("嵌入维度必须是64的倍数")
```

### 问题4: 分块无限循环

**症状**：导入时卡住，CPU占用高

**原因**：chunk_overlap >= chunk_size / 2

**解决**：
```python
# 避免无限循环
if chunk_overlap >= chunk_size / 2:
    logger.warning("chunk_overlap过大，设置为0")
    chunk_overlap = 0
```

### 问题5: 协程未await

**症状**：Pylance警告 `reportUnusedCoroutine`

**原因**：async函数返回值未await

**解决**：
```python
# 错误
utility.drop_collection(collection_name)

# 正确
await utility.drop_collection(collection_name)
```

### 问题6: Optional成员访问警告

**症状**：Pylance警告 `reportOptionalMemberAccess`

**原因**：访问可能为None的对象成员

**解决**：
```python
# 错误
vision_emb = self.vision_encoder.encode_image(image_bytes)

# 正确
encoder = cast(VisionEncoder, self.vision_encoder)
vision_emb = encoder.encode_image(image_bytes)

# 或使用类型守卫
if self.vision_encoder is not None:
    vision_emb = self.vision_encoder.encode_image(image_bytes)
```

---

## 📚 扩展开发

### 添加新的文档格式支持

```python
async def _parse_markdown(self, file_path: str) -> List[str]:
    """解析Markdown文件"""
    import markdown

    with open(file_path, 'r') as f:
        text = f.read()

    # 提取代码块、标题等
    chunks = []
    # ... 分块逻辑

    return chunks

# 在_parse_document中添加
elif file_type == 'md':
    return await self._parse_markdown(file_path)
```

### 添加新的Embedding Provider

```python
class EmbeddingProviderWrapper:
    async def embed(self, texts: str | List[str]):
        # 添加新Provider的适配逻辑
        if hasattr(self.provider, "new_method"):
            return await self.provider.new_method(texts)
        # ... 现有逻辑
```

### 自定义分块策略

```python
class CustomChunker:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> List[str]:
        # 自定义分块逻辑
        chunks = []
        # ... 实现分块算法
        return chunks

# 在_parse_pdf中使用
chunker = CustomChunker(chunk_size=1024)
chunks = chunker.chunk(full_text)
```

---

## 🔗 参考资源

- [AstrBot GitHub](https://github.com/AstrBotDevs/AstrBot)
- [Milvus Lite 文档](https://milvus.io/docs/milvus_lite.md)
- [PyMuPDF 文档](https://pymupdf.readthedocs.io/)
- [pdfplumber 文档](https://github.com/jsvine/pdfplumber)
- [SigLIP 论文](https://arxiv.org/abs/2303.15343)
