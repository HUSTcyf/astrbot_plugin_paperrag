# Qasper 数据集

Qasper (Question Answering over Scientific Papers) 是由 AllenAI 发布的科学论文问答数据集。

## 安装依赖

```bash
pip install datasets
```

或使用项目 requirements.txt：
```bash
pip install -r requirements.txt
```

## 使用方法

### 下载数据集

```bash
python datasets/qasper_downloader.py
```

### 查看数据集信息

```bash
python datasets/qasper_downloader.py --info
```

### 显示统计信息

```bash
python datasets/qasper_downloader.py --stats
```

### 提取论文用于 RAG

```bash
python datasets/qasper_downloader.py --extract
```

### 准备评估格式

```bash
python datasets/qasper_downloader.py --eval
```

### 执行全部操作

```bash
python datasets/qasper_downloader.py --all
```

## 输出文件

运行后将生成以下文件：

```
datasets/
├── data/
│   ├── qasper/                    # HuggingFace 数据集格式
│   │   ├── train.arrow
│   │   ├── validation.arrow
│   │   └── test.arrow
│   ├── qasper_papers_for_rag.json  # 论文全文 (用于 RAG)
│   └── evaluation_format/
│       ├── qasper_validation.jsonl  # 验证集
│       ├── qasper_test.jsonl         # 测试集
│       └── predictions_template.json # 预测模板
└── cache/
    └── datasets/                 # HuggingFace 缓存
```

## 数据集结构

```python
{
    "paper_title": "论文标题",
    "abstract": "摘要",
    "full_text": [
        {"section_name": "Introduction", "paragraph_text": "..."},
        {"section_name": "Method", "paragraph_text": "..."}
    ],
    "questions": [
        {
            "qid": "唯一ID",
            "question": "问题文本",
            "answer": {
                "answer_type": "free_text | yes_no | null",
                "answer_text": "答案"
            }
        }
    ]
}
```
