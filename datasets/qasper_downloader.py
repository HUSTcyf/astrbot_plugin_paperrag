#!/usr/bin/env python3
"""
Qasper (Question Answering over Scientific Papers) 数据集下载与管理脚本

数据集来源: AllenAI (https://qasper-led.org/)
Hugging Face: https://huggingface.co/datasets/allenai/qasper

功能:
1. 下载 Qasper 数据集
2. 解析数据结构
3. 提取论文全文用于 RAG
4. 生成评估格式的预测文件
5. 准备评估脚本所需格式

用法:
    python qasper_downloader.py                    # 下载数据集
    python qasper_downloader.py --info             # 查看数据集信息
    python qasper_downloader.py --stats            # 显示统计信息
    python qasper_downloader.py --extract -o ./output  # 提取论文到指定目录
    python qasper_downloader.py --eval -o ./output  # 准备评估格式到指定目录
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Tuple, Optional, Union

# 数据集保存路径
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CACHE_DIR = SCRIPT_DIR / "cache"

HF_DATASET_NAME = "allenai/qasper"

# 全局数据集函数引用
_datasets: Optional[Tuple[Callable, Callable]] = None


def _check_datasets() -> Tuple[Callable, Callable]:
    """检查 datasets 库是否可用，返回 (load_dataset, load_from_disk)"""
    global _datasets
    if _datasets is not None:
        return _datasets

    try:
        from datasets import load_dataset, load_from_disk
        _datasets = (load_dataset, load_from_disk)
        return _datasets
    except ImportError:
        print("请先安装 datasets 库: pip install datasets")
        sys.exit(1)


def ensure_dirs():
    """确保必要的目录存在"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _convert_qas_format(qas: list) -> list:
    """将 Qasper 原始 qas 格式转换为统一格式

    原始格式 (来自 qasper.py):
    {
        "question": ...,
        "question_id": ...,
        "answers": [{
            "answer": {
                "unanswerable": bool,
                "extractive_spans": [...],
                "yes_no": bool,
                "free_form_answer": str,
                "evidence": [...],
                "highlighted_evidence": [...]
            },
            "annotation_id": ...,
            "worker_id": ...
        }]
    }

    目标格式:
    {
        "qid": str,
        "question": str,
        "answer": {
            "answer_type": "free_text" | "yes_no" | "null",
            "answer_text": str
        }
    }
    """
    result = []
    for qa in qas:
        question = qa.get("question", "")
        question_id = qa.get("question_id", "")

        # 收集所有答案
        answers = qa.get("answers", [])
        if answers:
            first_answer = answers[0].get("answer", {})
            unanswerable = first_answer.get("unanswerable", False)
            yes_no = first_answer.get("yes_no")
            free_form = first_answer.get("free_form_answer", "")

            if unanswerable:
                answer_type = "null"
                answer_text = ""
            elif yes_no is not None:
                answer_type = "yes_no"
                answer_text = "yes" if yes_no else "no"
            elif free_form:
                answer_type = "free_text"
                answer_text = free_form
            else:
                answer_type = "null"
                answer_text = ""
        else:
            answer_type = "null"
            answer_text = ""

        result.append({
            "qid": question_id,
            "question": question,
            "answer": {
                "answer_type": answer_type,
                "answer_text": answer_text
            }
        })
    return result


def download_dataset() -> Any:
    """下载 Qasper 数据集

    数据来源: Qasper 官方 AWS S3
    - qasper-train-dev-v0.3.tgz (包含 train 和 dev 数据)
    - qasper-test-and-evaluator-v0.3.tgz (包含 test 数据)

    JSON 文件结构:
    - qasper-train-v0.3.json
    - qasper-dev-v0.3.json
    - qasper-test-v0.3.json
    """
    import json
    import tarfile
    import shutil
    from urllib.request import urlopen
    from urllib.error import URLError

    print(f"正在下载 Qasper 数据集 (v0.3)")
    print("=" * 60)

    output_path = DATA_DIR / "qasper"
    output_path.mkdir(parents=True, exist_ok=True)

    # 官方 S3 下载链接
    urls = {
        "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
    }

    # 内部文件名映射
    internal_files = {
        "train": "qasper-train-v0.3.json",
        "dev": "qasper-dev-v0.3.json",
        "test": "qasper-test-v0.3.json",
    }

    # 下载并解压
    for key, url in urls.items():
        tgz_file = CACHE_DIR / f"qasper-{key}-v0.3.tgz"

        if not tgz_file.exists():
            print(f"\n下载 {key} 数据 ({url.split('/')[-1]})...")
            print(f"  URL: {url}")
            try:
                response = urlopen(url, timeout=120)
                with open(tgz_file, "wb") as f:
                    shutil.copyfileobj(response, f)
                print(f"  下载完成: {tgz_file}")
            except (URLError, OSError) as e:
                print(f"  下载失败: {e}")
                print("\n请检查网络连接，或手动下载:")
                print(f"  wget '{url}' -O {tgz_file}")
                sys.exit(1)
        else:
            print(f"\n使用缓存: {tgz_file}")

        # 解压
        print(f"解压 {tgz_file}...")
        try:
            with tarfile.open(tgz_file, "r:gz") as tar:
                # 提取所有 JSON 文件
                for member in tar.getmembers():
                    for local_name, remote_name in internal_files.items():
                        if member.name.endswith(remote_name):
                            # 解压到目标目录
                            member.name = local_name + ".json"  # 重命名为 train.json, dev.json, test.json
                            tar.extract(member, output_path)
                            print(f"  提取: {member.name}")
                            break
        except tarfile.TarError as e:
            print(f"  解压失败: {e}")
            sys.exit(1)

    # 重命名 dev.json -> validation.json
    dev_file = output_path / "dev.json"
    val_file = output_path / "validation.json"
    if dev_file.exists() and not val_file.exists():
        dev_file.rename(val_file)
        print(f"  重命名: dev.json -> validation.json")

    # 转换 JSON 为 JSONL 格式（如果需要）
    for split in ["train", "validation", "test"]:
        json_file = output_path / f"{split}.json"
        jsonl_file = output_path / f"{split}.jsonl"

        if json_file.exists() and not jsonl_file.exists():
            print(f"\n转换 {json_file} -> {jsonl_file}...")
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            with open(jsonl_file, "w", encoding="utf-8") as f:
                for paper_id, paper_data in data.items():
                    record = {
                        "id": paper_id,
                        "paper_title": paper_data.get("title", ""),
                        "abstract": paper_data.get("abstract", ""),
                        "full_text": paper_data.get("full_text", []),
                        "questions": _convert_qas_format(paper_data.get("qas", [])),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  转换完成: {len(data)} 条记录")

    print(f"\n数据集下载完成! 保存位置: {output_path}")

    # 加载 JSONL 文件
    dataset = {}
    for split in ["train", "validation", "test"]:
        jsonl_file = output_path / f"{split}.jsonl"
        if jsonl_file.exists():
            print(f"\n加载 {split} 集...")
            data = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            dataset[split] = data
            print(f"  {split}: {len(data)} 条记录")

    # 返回一个简单的 dict 替代 Dataset 对象
    class SimpleDataset(dict):
        def keys(self):
            return super().keys()
        def __getitem__(self, key):
            return super().__getitem__(key)

    return SimpleDataset(dataset)


def load_local_dataset() -> Any:
    """从本地加载数据集"""
    import json

    output_path = DATA_DIR / "qasper"

    if not output_path.exists():
        print(f"本地数据集不存在，正在下载...")
        return download_dataset()

    print(f"从本地加载数据集: {output_path}")

    # 检查是否有任何数据文件
    jsonl_files = list(output_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"本地数据集为空，正在重新下载...")
        return download_dataset()

    # 从 JSONL 文件加载
    dataset = {}
    for split in ["train", "validation", "test"]:
        jsonl_file = output_path / f"{split}.jsonl"
        if jsonl_file.exists():
            data = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            dataset[split] = data

    class SimpleDataset(dict):
        def keys(self):
            return super().keys()
        def __getitem__(self, key):
            return super().__getitem__(key)

    return SimpleDataset(dataset)


def show_info(dataset: Any):
    """显示数据集基本信息"""
    print("\n" + "=" * 60)
    print("Qasper 数据集信息")
    print("=" * 60)

    print(f"\n数据集划分: {list(dataset.keys())}")
    print(f"训练集样本数: {len(dataset['train'])}")
    print(f"验证集样本数: {len(dataset['validation'])}")
    print(f"测试集样本数: {len(dataset['test'])}")

    # 显示第一条数据结构
    print("\n" + "-" * 60)
    print("数据结构示例 (训练集第一条):")
    print("-" * 60)

    sample = dataset['train'][0]
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:3000])


def show_stats(dataset: Any):
    """显示数据集统计信息"""
    print("\n" + "=" * 60)
    print("Qasper 数据集统计")
    print("=" * 60)

    total_papers = 0
    total_questions = 0
    total_paragraphs = 0

    answer_types = {"free_text": 0, "yes_no": 0, "null": 0}

    for split in dataset.keys():
        split_data = dataset[split]
        total_papers += len(split_data)

        for paper in split_data:
            questions = paper.get("questions", [])
            total_questions += len(questions)

            for q in questions:
                ans = q.get("answer", {})
                ans_type = ans.get("answer_type", "unknown")
                if ans_type in answer_types:
                    answer_types[ans_type] += 1

            full_text = paper.get("full_text", [])
            total_paragraphs += len(full_text)

    print(f"\n总论文数: {total_papers}")
    print(f"总问题数: {total_questions}")
    print(f"总段落数: {total_paragraphs}")

    print(f"\n答案类型分布:")
    for ans_type, count in answer_types.items():
        print(f"  {ans_type}: {count}")

    print(f"\n平均每篇论文问题数: {total_questions / max(total_papers, 1):.2f}")
    print(f"平均每篇论文段落数: {total_paragraphs / max(total_papers, 1):.2f}")


def extract_papers_for_rag(dataset: Any, output_file: Optional[Path] = None) -> list:
    """
    提取论文全文用于 RAG 处理

    返回格式:
    [{
        "paper_title": str,
        "abstract": str,
        "full_text": [{"section_name": str, "paragraph_text": str}, ...],
        "doi": str (可选),
        "arxiv_id": str (可选)
    }]
    """
    print("\n" + "=" * 60)
    print("提取论文全文用于 RAG")
    print("=" * 60)

    papers = []

    for split in dataset.keys():
        print(f"\n处理 {split} 集...")
        for i, paper in enumerate(dataset[split]):
            paper_title = paper.get("paper_title", "Unknown")

            # 提取 DOI 和 arXiv ID (如果存在)
            doc_id = paper.get("doc_id", {})
            doi = doc_id.get("doi") if isinstance(doc_id, dict) else None
            arxiv_id = doc_id.get("arxiv") if isinstance(doc_id, dict) else None

            paper_data = {
                "paper_title": paper_title,
                "abstract": paper.get("abstract", ""),
                "full_text": paper.get("full_text", []),
                "source_split": split,
                "doi": doi,
                "arxiv_id": arxiv_id
            }
            papers.append(paper_data)

            if (i + 1) % 500 == 0:
                print(f"  已处理 {i + 1} 篇论文...")

    print(f"\n共提取 {len(papers)} 篇论文")

    # 保存到文件
    if output_file is None:
        output_file = DATA_DIR / "qasper_papers_for_rag.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"已保存到: {output_file}")
    return papers


def generate_sample_predictions(dataset: Any, output_file: Optional[Path] = None) -> dict:
    """
    生成示例预测文件 (用于评估脚本)

    格式:
    {
        "question_id": {
            "predicted_answer": str,
            "predicted_evidence": [{"paragraph_index": int, "sentence_index": int}, ...]
        }
    }
    """
    print("\n" + "=" * 60)
    print("生成示例预测文件")
    print("=" * 60)

    predictions = {}

    for split in ["validation", "test"]:
        if split not in dataset:
            continue

        for paper in dataset[split]:
            questions = paper.get("questions", [])
            for q in questions:
                qid = q.get("qid", "")
                answer = q.get("answer", {})
                answer_text = answer.get("answer_text", "")

                # 简单示例：用答案文本作为预测
                predictions[qid] = {
                    "predicted_answer": answer_text,
                    "predicted_evidence": []
                }

    if output_file is None:
        output_file = DATA_DIR / "qasper_sample_predictions.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"已生成 {len(predictions)} 条预测记录")
    print(f"已保存到: {output_file}")
    print("\n提示: 此文件仅作为格式示例，实际预测需要运行模型生成")

    return predictions


def prepare_evaluation_format(dataset: Any, output_dir: Optional[Path] = None) -> Path:
    """
    准备官方评估脚本所需格式

    生成文件:
    - qasper_validation.jsonl: 验证集 (用于本地评估)
    - qasper_test.jsonl: 测试集 (需提交到 CodaLab)
    - sample_predictions.json: 预测文件模板
    """
    if output_dir is None:
        output_dir = DATA_DIR / "evaluation_format"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("准备评估格式文件")
    print("=" * 60)

    # 保存 validation.jsonl
    val_path = output_dir / "qasper_validation.jsonl"
    if "validation" in dataset:
        with open(val_path, "w", encoding="utf-8") as f:
            for paper in dataset["validation"]:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
        print(f"验证集: {val_path}")

    # 保存 test.jsonl
    test_path = output_dir / "qasper_test.jsonl"
    if "test" in dataset:
        with open(test_path, "w", encoding="utf-8") as f:
            for paper in dataset["test"]:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
        print(f"测试集: {test_path}")

    # 生成预测模板
    template_path = output_dir / "predictions_template.json"
    generate_sample_predictions(dataset, template_path)

    print(f"\n评估格式文件已保存到: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Qasper 数据集下载与管理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python qasper_downloader.py                    # 下载数据集
  python qasper_downloader.py --info             # 查看数据集信息
  python qasper_downloader.py --stats            # 显示统计信息
  python qasper_downloader.py --extract          # 提取论文用于 RAG
  python qasper_downloader.py --extract -o /path # 提取到指定目录
  python qasper_downloader.py --eval            # 准备评估格式
  python qasper_downloader.py --eval -o /path   # 评估文件到指定目录
  python qasper_downloader.py --all             # 执行全部操作
        """
    )

    parser.add_argument("--info", action="store_true", help="显示数据集信息")
    parser.add_argument("--stats", action="store_true", help="显示数据集统计")
    parser.add_argument("--extract", action="store_true", help="提取论文全文用于 RAG")
    parser.add_argument("--eval", action="store_true", help="准备评估格式文件")
    parser.add_argument("--all", action="store_true", help="执行全部操作")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="指定输出目录路径 (用于 --extract 和 --eval)")

    args = parser.parse_args()

    ensure_dirs()

    # 如果没有指定任何操作，下载数据集
    if not any([args.info, args.stats, args.extract, args.eval, args.all]):
        dataset = download_dataset()
        print("\n下载完成! 使用 --help 查看更多选项")
        return

    # 加载本地数据集
    dataset = load_local_dataset()

    # 处理输出路径
    output_path = Path(args.output) if args.output else None

    if args.all:
        show_info(dataset)
        show_stats(dataset)
        extract_papers_for_rag(dataset, output_path)
        prepare_evaluation_format(dataset, output_path)
        return

    if args.info:
        show_info(dataset)

    if args.stats:
        show_stats(dataset)

    if args.extract:
        extract_papers_for_rag(dataset, output_path)

    if args.eval:
        prepare_evaluation_format(dataset, output_path)


if __name__ == "__main__":
    main()
