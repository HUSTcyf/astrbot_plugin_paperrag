# -*- coding: utf-8 -*-
"""
评估报告生成器（HTML + Markdown）
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from astrbot.api import logger


class ReportGenerator:
    """评估报告生成器"""

    def __init__(self, results_path: str):
        """
        初始化报告生成器

        Args:
            results_path: 评估结果 CSV 文件路径
        """
        self.df = pd.read_csv(results_path)
        self.results_path = results_path

    def generate_html_report(
        self,
        output_path: str = "results/evaluation_report.html",
        plugin_version: str = "1.0.0",
        paper_name: str = "论文 RAG 系统",
    ) -> str:
        """
        生成 HTML 报告

        Args:
            output_path: 输出文件路径
            plugin_version: 插件版本
            paper_name: 论文/系统名称

        Returns:
            输出文件路径
        """
        # 计算统计信息
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_relevancy",
            "answer_correctness",
        ]

        stats: Dict[str, Dict[str, float]] = {}
        for m in metrics:
            if m in self.df.columns:
                mean_val = self.df[m].mean()
                std_val = self.df[m].std()
                min_val = self.df[m].min()
                max_val = self.df[m].max()
                # 处理 NaN
                if pd.isna(mean_val):
                    stats[m] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "valid": False}
                else:
                    stats[m] = {
                        "mean": float(mean_val),
                        "std": float(std_val) if not pd.isna(std_val) else 0.0,
                        "min": float(min_val) if not pd.isna(min_val) else 0.0,
                        "max": float(max_val) if not pd.isna(max_val) else 0.0,
                        "valid": True,
                    }

        # 按问题类型分组统计
        type_stats: Dict[str, Dict[str, float]] = {}
        if "question_type" in self.df.columns:
            for qtype in self.df["question_type"].unique():
                if pd.isna(qtype):
                    continue
                type_stats[qtype] = {}
                subset = self.df[self.df["question_type"] == qtype]
                for m in metrics:
                    if m in self.df.columns:
                        val = subset[m].mean()
                        type_stats[qtype][m] = float(val) if not pd.isna(val) else 0.0

        # 平均延迟
        avg_latency = 0.0
        if "latency_ms" in self.df.columns:
            avg_latency = self.df["latency_ms"].mean()
            if pd.isna(avg_latency):
                avg_latency = 0.0

        # 生成 HTML
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ragas 评测报告 - {paper_name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #1a1a1a; margin-bottom: 10px; }}
        h2 {{ color: #333; margin: 30px 0 15px; border-left: 4px solid #4CAF50; padding-left: 12px; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 30px; }}
        .meta span {{ margin-right: 20px; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
        .metric-value {{ font-size: 36px; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ color: #666; font-size: 13px; }}
        .metric-sub {{ color: #999; font-size: 12px; margin-top: 5px; }}
        .success {{ color: #4CAF50; }}
        .warning {{ color: #ff9800; }}
        .error {{ color: #f44336; }}
        table {{ background: white; border-collapse: collapse; width: 100%; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 14px 16px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #4CAF50; color: white; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        tr:last-child td {{ border-bottom: none; }}
        .suggestions {{ background: white; border-radius: 12px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 20px 0; }}
        .suggestions li {{ margin: 10px 0; padding-left: 25px; position: relative; color: #333; }}
        .suggestions li:before {{ content: "💡"; position: absolute; left: 0; }}
        .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 40px; padding: 20px; }}
        .score-bar {{ height: 8px; background: #eee; border-radius: 4px; margin-top: 8px; overflow: hidden; }}
        .score-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Ragas 评测报告</h1>
        <div class="meta">
            <span>📄 {paper_name}</span>
            <span>📅 {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
            <span>🔧 v{plugin_version}</span>
            <span>📝 {len(self.df)} 个样本</span>
        </div>
"""

        # 核心指标卡片
        html += """
        <h2>📈 核心指标</h2>
        <div class="card-grid">
"""
        metric_names = {
            "faithfulness": ("忠实度", "Faithfulness"),
            "answer_relevancy": ("回答相关性", "Answer Relevancy"),
            "context_precision": ("上下文精确率", "Context Precision"),
            "context_recall": ("上下文召回率", "Context Recall"),
            "context_relevancy": ("上下文相关性", "Context Relevancy"),
            "answer_correctness": ("答案正确性", "Answer Correctness"),
        }

        for m, (cn_name, en_name) in metric_names.items():
            if m in stats and stats[m]["valid"]:
                s = stats[m]
                color = "success" if s["mean"] >= 0.7 else "warning" if s["mean"] >= 0.5 else "error"
                bar_color = "#4CAF50" if s["mean"] >= 0.7 else "#ff9800" if s["mean"] >= 0.5 else "#f44336"
                html += f"""
            <div class="metric-card">
                <div class="metric-value {color}">{s["mean"]:.3f}</div>
                <div class="metric-label">{cn_name}</div>
                <div class="metric-sub">{en_name} ±{s["std"]:.3f}</div>
                <div class="score-bar"><div class="score-fill" style="width:{s['mean']*100}%;background:{bar_color}"></div></div>
            </div>
"""
        html += """
        </div>
"""

        # 详细统计表
        html += """
        <h2>📋 详细统计</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>平均值</th>
                <th>标准差</th>
                <th>最小值</th>
                <th>最大值</th>
                <th>评级</th>
            </tr>
"""
        for m, (cn_name, _) in metric_names.items():
            if m in stats and stats[m]["valid"]:
                s = stats[m]
                grade = "✅ 优秀" if s["mean"] >= 0.7 else "⚠️ 良好" if s["mean"] >= 0.5 else "❌ 需优化"
                html += f"""
            <tr>
                <td>{cn_name} ({m})</td>
                <td>{s["mean"]:.3f}</td>
                <td>{s["std"]:.3f}</td>
                <td>{s["min"]:.3f}</td>
                <td>{s["max"]:.3f}</td>
                <td>{grade}</td>
            </tr>
"""
        html += """
        </table>
"""

        # 按问题类型统计
        if type_stats:
            html += """
        <h2>📝 按问题类型统计</h2>
        <table>
            <tr>
                <th>问题类型</th>
"""
            for m, (cn_name, _) in metric_names.items():
                if m in stats:
                    html += f"<th>{cn_name}</th>"
            html += "</tr>\n"

            for qtype, mstats in type_stats.items():
                html += f"<tr><td><strong>{qtype}</strong></td>"
                for m in metrics:
                    if m in stats:
                        val = mstats.get(m, 0.0)
                        html += f"<td>{val:.3f}</td>"
                html += "</tr>\n"
            html += """
        </table>
"""

        # 低分样本分析
        if "faithfulness" in self.df.columns:
            low_faith = self.df[self.df["faithfulness"] < 0.5].head(10)
            if len(low_faith) > 0:
                html += """
        <h2>⚠️ 低分样本分析（忠实度 &lt; 0.5）</h2>
        <table>
            <tr>
                <th>#</th>
                <th>问题</th>
                <th>忠实度</th>
                <th>回答摘要</th>
            </tr>
"""
                for i, (_, row) in enumerate(low_faith.iterrows(), 1):
                    question = str(row.get("question", ""))[:80]
                    answer = str(row.get("answer", ""))[:60]
                    faith = row.get("faithfulness", 0)
                    html += f"""
            <tr>
                <td>{i}</td>
                <td title="{question}">{question}...</td>
                <td class="error">{faith:.3f}</td>
                <td title="{answer}">{answer}...</td>
            </tr>
"""
                html += """
        </table>
"""

        # 优化建议
        html += """
        <h2>💡 优化建议</h2>
        <div class="suggestions">
            <ul>
"""
        suggestions = []
        if "faithfulness" in stats and stats["faithfulness"]["valid"] and stats["faithfulness"]["mean"] < 0.7:
            suggestions.append("忠实度较低，建议：添加引用来源标注、降低 LLM 幻觉概率、优化检索内容质量")
        if "context_recall" in stats and stats["context_recall"]["valid"] and stats["context_recall"]["mean"] < 0.6:
            suggestions.append("召回率较低，建议：增加检索 Top-K、优化 Embedding 模型、改进分块策略（增大 chunk_size）")
        if "answer_correctness" in stats and stats["answer_correctness"]["valid"] and stats["answer_correctness"]["mean"] < 0.7:
            suggestions.append("正确性较低，建议：升级 LLM 模型、优化 Prompt 模板、添加答案后处理验证")
        if "context_precision" in stats and stats["context_precision"]["valid"] and stats["context_precision"]["mean"] < 0.5:
            suggestions.append("精确率较低，建议：优化重排序模型、调整 similarity_cutoff 阈值、过滤无关检索结果")

        if not suggestions:
            suggestions.append("各项指标表现良好，继续保持！")

        for s in suggestions:
            html += f"<li>{s}</li>\n"
        html += """
            </ul>
        </div>

        <div class="footer">
            <p>由 AstrBot Paper RAG 插件自动生成 | Ragas Evaluation Framework</p>
        </div>
    </div>
</body>
</html>
"""

        # 保存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML 报告已保存到: {output_path}")
        return output_path

    def generate_markdown_report(
        self,
        output_path: str = "results/evaluation_report.md",
        plugin_version: str = "1.0.0",
        paper_name: str = "论文 RAG 系统",
    ) -> str:
        """
        生成 Markdown 报告

        Args:
            output_path: 输出文件路径
            plugin_version: 插件版本
            paper_name: 论文/系统名称

        Returns:
            输出文件路径
        """
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_relevancy",
            "answer_correctness",
        ]

        metric_names = {
            "faithfulness": "忠实度",
            "answer_relevancy": "回答相关性",
            "context_precision": "上下文精确率",
            "context_recall": "上下文召回率",
            "context_relevancy": "上下文相关性",
            "answer_correctness": "答案正确性",
        }

        md = f"""# Ragas 评测报告

**系统**: {paper_name}
**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**版本**: v{plugin_version}
**评测样本数**: {len(self.df)}

---

## 核心指标

| 指标 | 平均值 | 标准差 | 最小值 | 最大值 | 评级 |
|-----|--------|--------|--------|--------|------|
"""

        for m, name in metric_names.items():
            if m in self.df.columns:
                mean_val = self.df[m].mean()
                std_val = self.df[m].std()
                min_val = self.df[m].min()
                max_val = self.df[m].max()
                if pd.isna(mean_val):
                    grade = "❌ N/A"
                    mean_str = std_str = min_str = max_str = "N/A"
                else:
                    grade = "✅ 优秀" if mean_val >= 0.7 else "⚠️ 良好" if mean_val >= 0.5 else "❌ 需优化"
                    mean_str = f"{mean_val:.3f}"
                    std_str = f"{std_val:.3f}" if not pd.isna(std_val) else "N/A"
                    min_str = f"{min_val:.3f}" if not pd.isna(min_val) else "N/A"
                    max_str = f"{max_val:.3f}" if not pd.isna(max_val) else "N/A"
                md += f"| {name} | {mean_str} | {std_str} | {min_str} | {max_str} | {grade} |\n"

        # 按问题类型统计
        if "question_type" in self.df.columns:
            md += "\n## 按问题类型统计\n\n"
            md += "| 问题类型 |"
            for m, name in metric_names.items():
                if m in self.df.columns:
                    md += f" {name} |"
            md += "\n|" + "---|" * (len([m for m in metric_names if m in self.df.columns]) + 1) + "\n"

            for qtype in self.df["question_type"].unique():
                if pd.isna(qtype):
                    continue
                subset = self.df[self.df["question_type"] == qtype]
                md += f"| **{qtype}** |"
                for m in metric_names:
                    if m in self.df.columns:
                        val = subset[m].mean()
                        md += f" {val:.3f} |" if not pd.isna(val) else " N/A |"
                md += "\n"

        # 优化建议
        md += "\n## 优化建议\n\n"
        suggestions = []
        if "faithfulness" in self.df.columns and self.df["faithfulness"].mean() < 0.7:
            suggestions.append("- ⚠️ **忠实度较低**：添加引用来源、降低 LLM 温度参数")
        if "context_recall" in self.df.columns and self.df["context_recall"].mean() < 0.6:
            suggestions.append("- ⚠️ **召回率较低**：增加检索 Top-K、优化 Embedding 模型")
        if "answer_correctness" in self.df.columns and self.df["answer_correctness"].mean() < 0.7:
            suggestions.append("- ⚠️ **正确性较低**：升级 LLM 模型、优化 Prompt")
        if not suggestions:
            suggestions.append("- ✅ 各项指标表现良好！")

        md += "\n".join(suggestions)
        md += "\n\n---\n*由 AstrBot Paper RAG 插件自动生成*\n"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"Markdown 报告已保存到: {output_path}")
        return output_path


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """使用示例"""
    generator = ReportGenerator("results/evaluation_results.csv")
    generator.generate_html_report()
    generator.generate_markdown_report()


if __name__ == "__main__":
    main()
