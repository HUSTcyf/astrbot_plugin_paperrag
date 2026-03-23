# 🦙 Ollama本地Embedding配置指南

## ✨ 优势

**为什么使用Ollama？**

| 特性 | Ollama本地模式 | Gemini API | OpenAI API |
|------|---------------|------------|------------|
| **费用** | ✅ 完全免费 | ❌ API限制/费用 | ❌ API费用 |
| **上下文长度** | 无限制 | 2048字符 | 8191字符 |
| **向量维度** | 768/1024维 | 768维 | 1536维 |
| **隐私保护** | ✅ 本地处理 | ❌ 上传云端 | ❌ 上传云端 |
| **网络依赖** | ❌ 无需网络 | ✅ 需要网络 | ✅ 需要网络 |
| **速度** | 中等（取决于硬件） | 快 | 快 |
| **配置难度** | 低（需安装Ollama） | 低（需API密钥） | 中（需API密钥） |

---

## 🚀 快速开始（3分钟）

### 第一步：安装Ollama

**macOS / Linux**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**验证安装**:
```bash
ollama --version
# 应该显示: ollama version is 0.1.x 或更高
```

### 第二步：下载BGE-M3模型

```bash
# 拉取BGE-M3模型（推荐，1024维）
ollama pull bge-m3

# 或其他可用模型:
# ollama pull nomic-embed-text  # 768维，更快
# ollama pull mxbai-embed-large  # 1024维，高精度
```

**模型大小参考**:
- bge-m3: ~2.3 GB
- nomic-embed-text: ~275 MB
- mxbai-embed-large: ~650 MB

### 第三步：启动Ollama服务

```bash
# 方式1：直接启动（前台运行）
ollama serve

# 方式2：后台启动（推荐）
nohup ollama serve > ~/.ollama/logs/server.log 2>&1 &

# 验证服务是否运行
curl http://localhost:11434/api/tags
# 应该返回已安装的模型列表
```

### 第四步：配置插件

在 **AstrBot WebUI → 插件 → paper_rag → 插件配置** 中：

| 配置项 | 值 | 说明 |
|-------|-----|------|
| Embedding模式 | `Ollama本地模式` | 选择ollama |
| Embedding Provider ID | （留空） | Ollama模式不需要 |
| 向量嵌入维度 | `1024` | BGE-M3固定1024维 |
| LLM Provider ID | `glm-4.7-flash`（可选） | 用于RAG回答生成 |
| Ollama服务地址 | `http://localhost:11434` | 默认地址 |
| Ollama模型名称 | `bge-m3` | 模型名称 |
| 并发批处理大小 | `10` | 同时处理的文本数 |
| 启用插件 | ✅ | - |

### 第五步：验证配置

```bash
# 1. 重启AstrBot
astrbot run

# 2. 导入测试文档
/paper add

# 3. 查询测试
/paper search 测试查询
```

**成功日志示例**:
```
🦙 初始化Ollama Embedding Provider
   - 服务地址: http://localhost:11434
   - 模型: bge-m3
   - 并发度: 10
   - 超时: 120.0秒

[导入文档时]
🦙 Ollama批量处理: 50 个文本（模型: bge-m3）
🔄 处理批次 1/5 (10 个文本，并发度: 10)
✅ Ollama完成: 50 个向量（维度: 1024）
```

---

## ⚙️ 高级配置

### Ollama配置选项

| 配置项 | 默认值 | 说明 |
|-------|--------|------|
| `ollama.base_url` | `http://localhost:11434` | Ollama服务地址 |
| `ollama.model` | `bge-m3` | 模型名称 |
| `ollama.timeout` | `120.0` | 请求超时（秒） |
| `ollama.batch_size` | `10` | 并发批处理大小 |
| `ollama.retry_attempts` | `3` | 失败重试次数 |

### 性能优化建议

**1. 调整并发度**

根据硬件配置调整 `ollama.batch_size`:

| 硬件配置 | 推荐batch_size | 预期速度 |
|---------|---------------|---------|
| CPU-only（4核） | 5-10 | ~5-10 文本/秒 |
| CPU-only（8核+） | 10-20 | ~10-20 文本/秒 |
| Apple Silicon M1/M2 | 10-15 | ~15-25 文本/秒 |
| NVIDIA GPU（8GB+） | 20-30 | ~30-50 文本/秒 |

**2. 选择合适的模型**

| 模型 | 维度 | 大小 | 速度 | 精度 | 推荐用途 |
|------|-----|------|------|------|---------|
| `nomic-embed-text` | 768 | 275MB | 最快 | 中 | 快速检索 |
| `bge-m3` | 1024 | 2.3GB | 中等 | 高 | 通用场景 |
| `mxbai-embed-large` | 1024 | 650MB | 中等 | 高 | 英文为主 |

**注意**: 切换模型后需要清空向量数据库并重新导入！

---

## 🐛 故障排除

### 1. 无法连接到Ollama服务

**症状**: `无法连接到Ollama服务 (http://localhost:11434)`

**解决**:
```bash
# 检查Ollama是否运行
ps aux | grep ollama

# 如果没有运行，启动服务
ollama serve

# 或后台启动
nohup ollama serve > ~/.ollama/logs/server.log 2>&1 &
```

### 2. 模型不存在

**症状**: `Ollama模型 'bge-m3' 不存在`

**解决**:
```bash
# 查看已安装的模型
ollama list

# 下载模型
ollama pull bge-m3
```

### 3. 向量维度不匹配

**症状**: 数据库已存在但向量维度不符

**解决**:
```bash
# 清空旧数据库（必须！）
/paper clear confirm

# 重新导入
/paper add
```

### 4. 速度太慢

**症状**: 导入速度很慢

**解决**:
1. 增加并发度: `ollama.batch_size = 20`
2. 使用更小的模型: `nomic-embed-text`
3. 检查CPU/GPU使用率: `htop` 或 `nvidia-smi`

### 5. 内存占用过高

**症状**: Ollama占用大量内存

**解决**:
1. 减少并发度: `ollama.batch_size = 5`
2. 使用更小的模型: `nomic-embed-text` (275MB)
3. 确保没有同时运行其他AI服务

---

## 📊 性能对比

**测试环境**: Apple M1 Pro, 16GB RAM

| 模型 | 100篇PDF导入时间 | 平均速度 | 内存占用 |
|------|----------------|---------|---------|
| nomic-embed-text | ~3分钟 | 25 文本/秒 | ~1GB |
| bge-m3 | ~5分钟 | 15 文本/秒 | ~3GB |
| mxbai-embed-large | ~4分钟 | 20 文本/秒 | ~2GB |

**API模式对比**:

| 模式 | 100篇PDF导入 | 费用 | 优势 |
|------|------------|------|------|
| Gemini API | ~2分钟 | 免费（有限额） | 最快 |
| Ollama (bge-m3) | ~5分钟 | 完全免费 | 无限制、隐私 |
| Ollama (nomic) | ~3分钟 | 完全免费 | 速度快 |

---

## 🔗 参考资源

- [Ollama官网](https://ollama.com/)
- [Ollama模型库](https://ollama.com/search)
- [BGE-M3模型详情](https://ollama.com/library/bge-m3)
- [Ollama API文档](https://github.com/ollama/ollama/blob/main/docs/api.md)

---

## 💡 常见问题

**Q: Ollama和之前的BGE-M3本地模式有什么区别？**

A: Ollama作为独立进程运行，通过HTTP API通信，完全避免了与AstrBot的进程冲突，更加稳定可靠。

**Q: 可以同时使用多个Ollama模型吗？**

A: 不建议。每次切换模型需要清空数据库并重新导入。建议选择一个最合适的模型长期使用。

**Q: Ollama支持GPU加速吗？**

A: 支持！如果你有NVIDIA GPU，Ollama会自动使用CUDA加速。可以通过 `ollama show bge-m3 --modelfile` 查看配置。

**Q: 如何完全卸载Ollama？**

A:
```bash
# 停止服务
pkill ollama

# 删除程序
rm $(which ollama)

# 删除模型和数据
rm -rf ~/.ollama
```

**Q: Ollama会影响AstrBot的性能吗？**

A: Ollama作为独立进程运行，不会直接导致AstrBot崩溃。但在导入文档时，两者会共同占用系统资源，建议根据硬件配置调整并发度。
