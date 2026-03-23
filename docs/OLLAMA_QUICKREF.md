# 🦙 Ollama本地Embedding - 快速参考

## 一分钟开始

```bash
# 1. 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 下载BGE-M3模型
ollama pull bge-m3

# 3. 启动服务
ollama serve

# 4. 配置插件（在WebUI中）
embedding_mode = "ollama"
embed_dim = 1024
ollama.model = "bge-m3"

# 5. 导入文档
/paper add
```

## 配置对比

### API模式（之前）
```json
{
  "embedding_mode": "api",
  "embedding_provider_id": "gemini_embedding",
  "embed_dim": 768
}
```

### Ollama模式（现在）
```json
{
  "embedding_mode": "ollama",
  "embedding_provider_id": "",
  "embed_dim": 1024,
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "bge-m3",
    "batch_size": 10
  }
}
```

## 关键优势

| 特性 | API模式 | Ollama模式 |
|------|---------|-----------|
| 费用 | 150K次/天限制 | ✅ 完全免费 |
| 上下文长度 | 2048字符 | ✅ 无限制 |
| 进程稳定性 | ✅ 稳定 | ✅ 稳定（独立进程） |
| 隐私 | ❌ 上传云端 | ✅ 本地处理 |
| 速度 | 快 | 中等 |

## 故障排除

**问题**: 无法连接到Ollama
```bash
# 检查服务
ps aux | grep ollama

# 重启服务
pkill ollama && ollama serve
```

**问题**: 模型不存在
```bash
# 重新下载模型
ollama pull bge-m3
```

**问题**: 向量维度错误
```bash
# 清空数据库
/paper clear confirm

# 重新导入
/paper add
```

## 性能调优

**CPU-only**: `batch_size = 5-10`
**Apple Silicon**: `batch_size = 10-15`
**NVIDIA GPU**: `batch_size = 20-30`

## 模型选择

| 模型 | 维度 | 大小 | 速度 |
|------|-----|------|------|
| nomic-embed-text | 768 | 275MB | 最快 |
| bge-m3 | 1024 | 2.3GB | 中等 |
| mxbai-embed-large | 1024 | 650MB | 中等 |

详细文档: [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md)
