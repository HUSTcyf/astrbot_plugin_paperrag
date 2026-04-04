# Ollama 本地 Embedding 配置指南

使用 Ollama 进行本地向量化，无需 API Key，完全免费、无限制、隐私保护。

## 安装步骤

### 1. 安装 Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# 或通过 Homebrew (macOS)
brew install ollama
```

### 2. 拉取 Embedding 模型

```bash
# 推荐使用 bge-m3（支持中英文，效果好）
ollama pull bge-m3

# 备选：nomic-embed-text（纯英文）
ollama pull nomic-embed-text
```

### 3. 启动 Ollama 服务

```bash
# 启动服务（默认端口 11434）
ollama serve

# 或者在后台运行
ollama serve &
```

### 4. 验证安装

```bash
# 测试 bge-m3 模型
ollama run bge-m3 "Hello world"
```

## 插件配置

在插件配置界面或配置文件中设置：

```json
{
  "embedding_mode": "ollama",
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "bge-m3",
    "batch_size": 10,
    "timeout": 120.0
  }
}
```

### 配置说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `base_url` | Ollama 服务地址 | `http://localhost:11434` | 默认即可 |
| `model` | Embedding 模型名称 | `bge-m3` | `bge-m3`（推荐） |
| `batch_size` | 并发批处理大小 | `10` | `10-20`（根据硬件调整） |
| `timeout` | 请求超时（秒） | `120.0` | 默认 |

## 性能优化

### 批量大小调优

`batch_size` 控制同时处理的文档数量：
- **内存充足**：设置为 `20` 或更高，加快索引速度
- **内存有限**：设置为 `5-10`，避免 OOM

### GPU 加速（推荐）

Ollama 自动使用 GPU 加速。如需确认：

```bash
# 查看 GPU 使用情况
ollama run bge-m3 "test" --verbose
```

### 多线程配置

编辑 `~/.ollama/config.toml`：

```toml
# 限制并发请求数
max_parallel = 4

# 上下文大小
num_ctx = 8192
```

## 常见问题

### Q: 报错 `connection refused`

Ollama 服务未启动。执行：

```bash
ollama serve
```

### Q: 内存不足 (OOM)

降低 `batch_size`：

```json
{
  "ollama": {
    "batch_size": 5
  }
}
```

### Q: 向量化速度慢

1. 确认 GPU 正在被使用
2. 适当提高 `batch_size`
3. 使用 `bge-m3` 而非 `nomic-embed-text`

### Q: 模型下载慢

使用镜像或手动下载：

```bash
# 手动下载 bge-m3
ollama pull bge-m3

# 查看已下载模型
ollama list
```

## 与 API 模式对比

| 特性 | Ollama | API 模式 |
|------|--------|----------|
| 成本 | 免费 | 按量付费 |
| 隐私 | 数据本地处理 | 数据发送到第三方 |
| 速度 | 依赖本地硬件 | 依赖网络 |
| 限制 | 无 | 有 RPM/TPM 限制 |
| 稳定性 | 依赖本地服务 | 服务商保障 |

## 相关文档

- [Ollama 官网](https://ollama.com/)
- [Ollama 模型库](https://ollama.com/library)
- [bge-m3 模型](https://ollama.com/library/bge-m3)
