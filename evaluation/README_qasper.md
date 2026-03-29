# Qasper 评估

使用 RAG 系统生成 predictions 并进行评估。

## 重要说明

**Qasper 数据集不包含 PDF 文件**，只包含从论文提取的文本内容（full_text）。

因此评估流程为：
1. 使用数据集自带的 `full_text` 进行 RAG 索引
2. 基于索引后的内容进行问答评估

## 流程概览

```
1. 下载数据集 (qasper_downloader.py)
       ↓
2. 索引论文到 Milvus (index_qasper.py)
       ↓
3. 生成 predictions (run_evaluation.py --generate)
       ↓
4. 运行评估 (run_evaluation.py --evaluate 或 --all)
```

---

## 完整评估步骤

### 1. 下载数据集

```bash
cd datasets
python qasper_downloader.py
```

这会下载 Qasper 数据集（JSON 格式，包含论文文本）。

### 2. 索引论文到 Milvus

```bash
cd evaluation
python index_qasper.py --reinit
```

可选参数：
- `--split train|validation|test|all` - 指定要索引的数据集（默认 all）
- `--reinit` - 重新初始化数据库（清除旧数据）
- `--config /path/to/config.json` - 指定配置文件
- `--data_dir /path/to/data` - 指定数据目录

### 3. 生成 Predictions

```bash
python run_evaluation.py --generate
```

这会遍历测试集中的所有问题，使用 RAG 生成答案，保存到 `predictions.jsonl`。

### 4. 运行评估

```bash
python run_evaluation.py --evaluate
```

或使用 `--all` 一步完成：

```bash
python run_evaluation.py --all
```

---

## 命令行参数

### index_qasper.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--split` | 数据集划分 | `all` |
| `--reinit` | 重新初始化数据库 | False |
| `--config` | 配置文件路径 | `data/config/astrbot_plugin_paperrag_config.json` |
| `--data_dir` | 数据目录路径 | `./data/qasper` |

### run_evaluation.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--generate` | 仅生成 predictions | - |
| `--evaluate` | 仅运行评估 | - |
| `--all` | 生成 + 评估 | - |
| `--config` | 配置文件路径 | `data/config/astrbot_plugin_paperrag_config.json` |
| `--data_dir` | 数据目录路径 | `./data` |
| `--output` | 输出目录路径 | `./evaluation_output` |
| `--batch_size` | 每批处理问题数 | 10 |
| `--delay` | 批次间延迟(秒) | 1.0 |
| `--text_evidence_only` | 仅使用文本证据 | False |

---

## 输出文件

```
evaluation_output/
├── predictions.jsonl       # 预测结果
└── evaluation_results.json # 评估指标
```

## 评估指标

- **Answer F1**: 答案 F1 分数
- **Answer F1 by type**: 按答案类型 (extractive/abstractive/boolean/none) 的 F1
- **Evidence F1**: 证据 F1 分数
- **Missing predictions**: 缺失预测数量

---

## 故障排除

### "Milvus 连接失败"

检查配置：
```json
{
  "milvus_lite_path": "./milvus_lite.db",
  "collection_name": "paper_embeddings"
}
```

### "embed_dim 不匹配"

确保配置中 `embed_dim` 与 embedding 模型匹配：
- bge-m3 → 1024
- nomic-embed-text → 768

### 索引后检索不到内容

确认已运行 `--reinit` 重新初始化数据库。

---

## 脚本说明

### index_qasper.py

将 Qasper 数据集中的 `full_text` 字段解析为段落，直接索引到 Milvus。

**为什么不下载 PDF？**
Qasper 官方不提供 PDF 文件，仅提供提取的文本。数据集中已包含完整的论文内容（sections + paragraphs），直接索引这些内容即可进行 RAG 评估。

### run_evaluation.py

1. 初始化 RAG 引擎
2. 遍历测试集问题
3. 调用 `engine.search(question)` 获取 RAG 答案
4. 生成符合官方格式的 `predictions.jsonl`
5. 调用 `qasper_evaluator.py` 计算指标
