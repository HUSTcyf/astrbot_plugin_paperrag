# Paper RAG Plugin 开发指南

## 📚 目录

1. [插件架构](#插件架构)
2. [核心组件](#核心组件)
3. [Provider 集成](#provider-集成)
4. [向量数据库](#向量数据库)
5. [文档解析](#文档解析)
6. [配置管理](#配置管理)
7. [命令系统](#命令系统)
8. [开发工作流](#开发工作流)
9. [最佳实践](#最佳实践)

---

## 插件架构

### 核心设计

```
main.py (插件入口)
  ├── PaperRAGPlugin (Star)
  │   ├── __init__() - 初始化
  │   ├── _get_engine() - 获取RAG引擎（懒加载）
  │   ├── 缓存管理
  │   └── 命令处理器
  │
rag_engine.py (核心引擎)
  ├── RAGConfig - 配置数据类
  ├── EmbeddingProviderWrapper - Embedding Provider 包装
  ├── MilvusLiteStore - 向量存储
  ├── PaperRAGEngine - RAG引擎主类
  └── 文档解析器
```

### 设计模式

1. **单例模式**: RAG 引擎懒加载，全局唯一实例
2. **包装器模式**: 统一 Embedding Provider 接口
3. **策略模式**: 支持多种文档解析策略
4. **缓存模式**: LRU 缓存提升性能

---

## 核心组件

### 1. PaperRAGPlugin (main.py)

```python
@register(
    "paper_rag",
    "YourName",
    "本地论文库RAG检索插件",
    "1.0.0",
    "https://github.com/your/repo"
)
class PaperRAGPlugin(Star):
    def __init__(self, context: Context, config: dict = {}):
        super().__init__(context)
        self.config = config or {}
        self.context = context

        # 插件配置
        self.enabled = self.config.get("enabled", True)

        # 缓存配置
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl_seconds", 3600)
        self.cache_max_size = self.config.get("cache_max_entries", 100)
        self._response_cache = {}

        # RAG引擎（懒加载）
        self._engine = None
        self._config_valid = False

    def _get_engine(self) -> Optional[PaperRAGEngine]:
        """获取RAG引擎（单例模式）"""
        if self._engine is None:
            # 创建 RAGConfig
            rag_config = RAGConfig(
                embedding_provider_id=self.config.get("embedding_provider_id", ""),
                llm_provider_id=self.config.get("llm_provider_id", ""),
                milvus_uri=self.config.get("milvus_uri", "./data/milvus_papers.db"),
                collection_name=self.config.get("collection_name", "paper_embeddings"),
                embed_dim=self.config.get("embed_dim", 768),
                top_k=self.config.get("top_k", 5),
                similarity_cutoff=self.config.get("similarity_cutoff", 0.3),
                papers_dir=self.config.get("papers_dir", "./papers")
            )

            # 初始化引擎（传递context以获取Provider）
            self._engine = PaperRAGEngine(rag_config, self.context)

        return self._engine
```

### 2. RAGConfig (rag_engine.py)

```python
@dataclass
class RAGConfig:
    """RAG配置类"""
    # Provider配置
    embedding_provider_id: str = ""
    llm_provider_id: str = ""

    # Milvus配置
    milvus_uri: str = "./data/milvus_papers.db"
    collection_name: str = "paper_embeddings"

    # 检索配置
    embed_dim: int = 768
    top_k: int = 5
    similarity_cutoff: float = 0.3

    # 论文目录
    papers_dir: str = "./papers"

    def validate(self) -> tuple[bool, str]:
        """验证配置"""
        if self.embed_dim % 64 != 0:
            return False, "嵌入维度必须是64的倍数"
        return True, ""
```

---

## Provider 集成

### Embedding Provider

```python
from astrbot.core.provider.provider import EmbeddingProvider

class EmbeddingProviderWrapper:
    """AstrBot Embedding Provider 包装类"""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def embed(self, texts: str | List[str]) -> List[List[float]]:
        """获取文本的嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]

        # 调用 provider 的 embed 方法
        embeddings = self.provider.embed(texts)  # type: ignore
        if not embeddings:
            raise Exception("Embedding provider 返回空结果")

        return embeddings

    async def get_text_embedding(self, text: str) -> List[float]:
        """获取文本嵌入（文档）"""
        result = self.embed([text])
        return result[0] if result and len(result) > 0 else []

    async def get_query_embedding(self, query: str) -> List[float]:
        """获取查询嵌入"""
        return await self.get_text_embedding(query)
```

### 在 PaperRAGEngine 中初始化 Provider

```python
class PaperRAGEngine:
    def __init__(self, config: RAGConfig, context):
        self.config = config
        self.context = context

        # 初始化 Embedding Provider
        self.embedding = None
        if config.embedding_provider_id:
            try:
                provider_manager = getattr(context, "provider_manager", None)
                inst_map = getattr(provider_manager, "inst_map", None)
                if isinstance(inst_map, dict):
                    provider = inst_map.get(config.embedding_provider_id)
                    if provider and callable(getattr(provider, "embed", None)):
                        self.embedding = EmbeddingProviderWrapper(provider)
                        logger.info(f"✅ Embedding Provider 加载成功")
            except Exception as e:
                logger.error(f"❌ Embedding Provider 加载失败: {e}")

        if not self.embedding:
            raise Exception("无法初始化 Embedding Provider，请检查配置")

        # LLM生成器（可选）- 类似的方式
```

---

## 向量数据库

### Milvus Lite 初始化

```python
class MilvusLiteStore:
    def __init__(self, uri: str, collection_name: str, dim: int):
        self.uri = uri
        self.collection_name = collection_name
        self.dim = dim

        # 确保目录存在
        Path(uri).parent.mkdir(parents=True, exist_ok=True)

        # 初始化客户端
        self.client = MilvusClient(uri)

        # 创建或加载集合
        self._init_collection()

    def _init_collection(self):
        """初始化集合"""
        if self.client.has_collection(self.collection_name):
            return

        # 创建新集合
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        # 创建索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 8, "efConstruction": 64}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
```

### 向量操作

```python
async def add_documents(self, documents: List[Dict[str, Any]]) -> int:
    """批量添加文档"""
    data = []
    for doc in documents:
        data.append({
            "vector": doc["embedding"],
            "text": doc["text"],
            "metadata": doc.get("metadata", {})
        })

    self.client.insert(collection_name=self.collection_name, data=data)
    return len(documents)

async def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """向量搜索"""
    results = self.client.search(
        collection_name=self.collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "metadata"]
    )

    # 格式化结果
    formatted_results = []
    for result in results[0]:
        formatted_results.append({
            "text": result["entity"]["text"],
            "metadata": result["entity"].get("metadata", {}),
            "score": result["distance"]
        })

    return formatted_results
```

---

## 文档解析

### 文件类型检测

```python
def _detect_file_type(self, file_path: str) -> str:
    """检测文件类型"""
    suffix = Path(file_path).suffix.lower()
    type_map = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',
        '.txt': 'text',
        '.md': 'text',
        '.html': 'html',
        '.htm': 'html',
    }
    return type_map.get(suffix, 'unknown')
```

### PDF 解析（PyMuPDF）

```python
async def _parse_pdf(self, file_path: str) -> List[str]:
    """解析PDF文件"""
    doc = fitz.open(file_path)
    text_chunks = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            # 分块处理
            chunk_size = 512
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if len(chunk.strip()) > 50:
                    text_chunks.append(chunk.strip())

    doc.close()

    # 警告：分块数量过多
    if len(text_chunks) > 50:
        logger.warning(f"⚠️ {filename}: 生成了 {len(text_chunks)} 个分块（超过50个），可能会影响检索性能")

    return text_chunks
```

### Word 解析（python-docx）

```python
async def _parse_docx(self, file_path: str) -> List[str]:
    """解析Word文档"""
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)
    full_text = []

    # 提取段落文本
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    # 提取表格文本
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join([cell.text.strip() for cell in row.cells])
            if row_text.strip():
                full_text.append(row_text)

    # 分块处理
    combined_text = "\n".join(full_text)
    chunk_size = 512
    text_chunks = []
    for i in range(0, len(combined_text), chunk_size):
        chunk = combined_text[i:i+chunk_size]
        if len(chunk.strip()) > 50:
            text_chunks.append(chunk.strip())

    # 警告：分块数量过多
    if len(text_chunks) > 50:
        logger.warning(f"⚠️ {filename}: 生成了 {len(text_chunks)} 个分块（超过50个），可能会影响检索性能")

    return text_chunks
```

### 通用解析器

```python
async def _parse_document(self, file_path: str) -> List[str]:
    """通用文档解析器 - 自动检测文件类型"""
    file_type = self._detect_file_type(file_path)

    if file_type == 'pdf':
        return await self._parse_pdf(file_path)
    elif file_type == 'docx':
        return await self._parse_docx(file_path)
    elif file_type in ('text', 'html', 'markdown'):
        return await self._parse_text(file_path)
    else:
        # 尝试用 unstructured 解析
        return await self._parse_with_unstructured(file_path)
```

---

## 配置管理

### _conf_schema.json

```json
{
    "embedding_provider_id": {
        "description": "Embedding 服务提供商",
        "type": "string",
        "hint": "选择用于向量嵌入的 Embedding Provider",
        "_special": "select_embedding_provider",
        "default": "gemini_embedding"
    },
    "llm_provider_id": {
        "description": "LLM Provider ID",
        "type": "string",
        "hint": "手动输入LLM Provider的ID",
        "obvious_hint": true,
        "default": "glm-4.7-flash"
    },
    "milvus_uri": {
        "description": "Milvus Lite数据库路径",
        "type": "string",
        "default": "./data/milvus_papers.db"
    },
    "embed_dim": {
        "description": "向量嵌入维度",
        "type": "int",
        "default": 768,
        "minimum": 128,
        "maximum": 4096
    },
    "top_k": {
        "description": "检索返回的文档片段数量",
        "type": "int",
        "default": 5,
        "minimum": 1,
        "maximum": 20
    },
    "similarity_cutoff": {
        "description": "相似度阈值",
        "type": "float",
        "default": 0.3,
        "minimum": 0,
        "maximum": 1
    },
    "papers_dir": {
        "description": "论文文件存放目录",
        "type": "string",
        "default": "./papers"
    }
}
```

### AstrBot 特殊配置

- `_special: "select_embedding_provider"` - 显示 Embedding Provider 选择器
- `_special: "select_provider"` - 显示 LLM Provider 选择器
- `obvious_hint: true` - 在 WebUI 中突出显示此配置项

---

## 命令系统

### 命令组

```python
@filter.command_group("文档")
def paper_commands(self):
    """文档RAG命令组"""
    pass

@paper_commands.command("搜索")
async def cmd_search(self, event: AstrMessageEvent,
                     query: str = None,
                     mode: str = "rag",
                     top_k: int = 5):
    """搜索文档库并回答问题"""
    # 实现逻辑
    pass

@paper_commands.command("列表")
async def cmd_list(self, event: AstrMessageEvent):
    """列出文档库中的所有文档"""
    # 实现逻辑
    pass
```

### 权限控制

```python
@filter.permission_type(filter.PermissionType.ADMIN)
@paper_commands.command("添加")
async def cmd_add(self, event: AstrMessageEvent, directory: str = None):
    """添加文档到知识库（管理员）"""
    # 实现逻辑
    pass

@filter.permission_type(filter.PermissionType.ADMIN)
@paper_commands.command("清空")
async def cmd_clear(self, event: AstrMessageEvent, confirm: str = None):
    """清空文档知识库（管理员）"""
    # 实现逻辑
    pass
```

---

## 开发工作流

### 1. 本地开发

```bash
# 进入插件目录
cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag

# 安装依赖
uv pip install -r requirements.txt

# 修改代码
vim main.py  # 或 rag_engine.py

# 重启 AstrBot
# 修改会自动热加载（如果启用）
```

### 2. 调试

```python
from astrbot.api import logger

# 使用日志
logger.debug(f"调试信息: {some_var}")
logger.info(f"普通信息")
logger.warning(f"警告信息")
logger.error(f"错误信息: {e}", exc_info=True)
```

### 3. 测试

```bash
# 在 AstrBot 聊天中测试命令
/paper list
/paper search 测试问题
```

---

## 最佳实践

### 1. 错误处理

```python
async def cmd_search(self, event: AstrMessageEvent, query: str = None):
    try:
        engine = self._get_engine()
        response = await engine.search(query)

        # 处理响应
        if response["type"] == "error":
            yield event.plain_result(f"❌ {response['message']}")
        else:
            yield event.plain_result(output)

    except Exception as e:
        logger.error(f"搜索失败: {e}", exc_info=True)
        yield event.plain_result(f"❌ 搜索失败: {str(e)}")
```

### 2. 资源清理

```python
async def terminate(self):
    """插件卸载时调用"""
    logger.info("📚 Document RAG Plugin 正在卸载...")

    # 清理资源
    self._response_cache.clear()

    # 注意：不需要显式关闭Milvus连接
    # MilvusClient会自动管理连接

    await super().terminate()
```

### 3. 性能优化

```python
# 使用缓存
def _get_cached_response(self, cache_key: str):
    """获取缓存的响应"""
    if not self.cache_enabled:
        return None

    if cache_key in self._response_cache:
        cached_data, timestamp = self._response_cache[cache_key]
        if time.time() - timestamp < self.cache_ttl:
            return cached_data

    return None

# 限流保护
for i, chunk in enumerate(chunks):
    if i % 5 == 0:
        await asyncio.sleep(0.5)  # 每5个chunk暂停
```

### 4. 类型注解

```python
from typing import List, Dict, Any, Optional

async def search(self, query: str, mode: str = "rag") -> Dict[str, Any]:
    """明确的返回类型注解"""
    pass

def _get_engine(self) -> Optional[PaperRAGEngine]:
    """可能返回 None 的类型"""
    pass
```

---

## 常见问题

### Q1: Provider 加载失败？

**原因**: Provider ID 不正确或 Provider 未配置

**解决方案**:
1. 在 AstrBot WebUI → 设置 → 模型提供商 中确认 Provider 已添加
2. 检查插件配置中的 Provider ID 是否与设置中的 ID 一致
3. 查看 AstrBot 日志获取详细错误信息

### Q2: 向量维度不匹配？

**原因**: `embed_dim` 配置与 Embedding Provider 的模型维度不匹配

**解决方案**:
- Gemini Embedding 2 Preview: 768维
- OpenAI text-embedding-3-small: 1536维
- 确认配置中的 `embed_dim` 与模型匹配

### Q3: Milvus 初始化失败？

**原因**: 数据库目录权限或路径问题

**解决方案**:
```python
# 确保目录存在且有写权限
Path(uri).parent.mkdir(parents=True, exist_ok=True)
```

---

## 扩展建议

### 1. 支持更多文档格式

- 添加 Markdown 解析器
- 支持 PPTX、EPUB 等格式
- 图片 OCR（使用 PaddleOCR 或 Tesseract）

### 2. 增强检索能力

- 混合检索（向量+关键词）
- 重排序（Cross-Encoder）
- 查询扩展

### 3. 改进用户体验

- WebUI 界面
- 文档预览
- 导入进度条
- 搜索结果高亮

---

## 参考资源

- [AstrBot GitHub](https://github.com/AstrBotDevs/AstrBot)
- [Milvus Lite 文档](https://milvus.io/docs/milvus_lite.md)
- [PyMuPDF 文档](https://pymupdf.readthedocs.io/)
- [python-docx 文档](https://python-docx.readthedocs.io/)
