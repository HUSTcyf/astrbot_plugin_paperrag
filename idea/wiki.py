"""
LLM Wiki 存储引擎 — 将 Idea 生成结果写入 Karpathy 模式 Wiki 结构。

存储位置：~/.paperrag-wiki/（可通过 PAPERRAG_WIKI_PATH 环境变量覆盖）
"""

from __future__ import annotations

import re
import uuid as _uuid
from datetime import datetime
from pathlib import Path

from astrbot.api import logger


def slugify(text: str) -> str:
    """将任意文本转为合法的 slug。长文本使用 hash 后缀保证唯一性。"""
    import hashlib
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    if len(text) > 80:
        suffix = hashlib.md5(text.encode()).hexdigest()[:8]
        text = text[:80] + "-" + suffix
    return text


def _escape_yaml(text: str) -> str:
    """Escape a string value for safe inclusion in YAML frontmatter.

    Wraps in double quotes if the value contains newlines, colons, or
    leading characters that could be misinterpreted by YAML parsers.
    """
    if not text:
        return '""'
    if "\n" in text or "\r" in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if text[0] in r"#{}[]&*!|>%@" or text.endswith(":"):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if ":" in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if text.lower() in ("true", "false", "null", "yes", "no", "on", "off"):
        return f'"{text}"'
    return text


class IdeaWikiEngine:
    """
    管理 Wiki 目录的读写（PaperRAG + Hermes 共享记忆库）。

    目录结构:
      WIKI_ROOT/
        SCHEMA.md
        index.md
        log.md
        ideas/               # PaperRAG: 研究想法（自动管理）
          <topic-slug>/
            index.md
            context.md
            <idea-slug>.md
        entities/            # Hermes: 实体页（方法/模型/数据集）
        concepts/            # Hermes: 概念探索
        comparisons/         # Hermes: 对比分析
        queries/             # Hermes: 查询归档
        raw/                 # Layer 1: 不可变源材料

    路径优先级: 构造函数参数 > PAPERRAG_WIKI_PATH > WIKI_PATH > 默认 plugin_data 目录
    """

    # 模块级默认路径：跟随 AstrBot plugin_data 约定
    # wiki.py → idea/ → astrbot_plugin_paperrag/ → plugins/ → data/
    _DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent
    _DEFAULT_WIKI_ROOT = _DEFAULT_DATA_DIR / "plugin_data" / "astrbot_plugin_paperrag" / "wiki"

    def __init__(self, wiki_path: str | Path | None = None):
        import os as _os
        if wiki_path:
            self.root = Path(wiki_path).expanduser().resolve()
        else:
            wiki_env = (
                _os.environ.get("PAPERRAG_WIKI_PATH")
                or _os.environ.get("WIKI_PATH")
            )
            if wiki_env:
                self.root = Path(wiki_env).expanduser().resolve()
            else:
                self.root = IdeaWikiEngine._DEFAULT_WIKI_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- 路径 helpers ----

    def ideas_dir(self) -> Path:
        return self.root / "ideas"

    def topic_dir(self, topic: str) -> Path:
        return self.ideas_dir() / slugify(topic)

    def entity_dir(self) -> Path:
        return self.root / "entities"

    def concept_dir(self) -> Path:
        return self.root / "concepts"

    def comparison_dir(self) -> Path:
        return self.root / "comparisons"

    def _page_dir(self, page_type: str) -> Path:
        return {"entity": self.entity_dir(), "concept": self.concept_dir(),
                "comparison": self.comparison_dir()}[page_type]

    def _index_section_marker(self, page_type: str) -> str:
        return {"entity": "Entities", "concept": "Concepts",
                "comparison": "Comparisons"}[page_type]

    # ---- Schema bootstrap ----

    def ensure_schema(self) -> None:
        """首次运行时创建 SCHEMA.md、index.md、log.md"""
        self.root.mkdir(parents=True, exist_ok=True)

        schema_path = self.root / "SCHEMA.md"
        if not schema_path.exists():
            schema_path.write_text(DEFAULT_SCHEMA, encoding="utf-8")
            logger.info(f"[IdeaWiki] SCHEMA.md created at {self.root}")

        index_path = self.root / "index.md"
        if not index_path.exists():
            index_path.write_text(DEFAULT_INDEX, encoding="utf-8")

        log_path = self.root / "log.md"
        if not log_path.exists():
            log_path.write_text(
                f"# Idea Wiki Log\n\n## {datetime.now().strftime('%Y')}\n",
                encoding="utf-8"
            )

    # ---- index.md 操作 ----

    def add_topic_to_index(self, topic: str, topic_slug: str, summary: str = "") -> None:
        """在全局 index.md 的 Research Ideas 节下追加 topic 入口"""
        index_path = self.root / "index.md"
        entry = f"- [[{topic_slug}/index|{topic}]]"
        if summary:
            entry += f": {summary}"
        entry += f" _(updated {datetime.now().strftime('%Y-%m-%d')})_\n"

        content = ""
        if index_path.exists():
            content = index_path.read_text(encoding="utf-8")
        else:
            content = DEFAULT_INDEX

        if f"[[{topic_slug}/" in content:
            return

        # Insert under ## Research Ideas section if it exists
        for marker in ("## Research Ideas (PaperRAG)", "## Research Ideas"):
            idx = content.find(marker)
            if idx != -1:
                section_end = content.find("\n## ", idx + len(marker))
                if section_end == -1:
                    section_end = len(content)
                content = content[:section_end] + entry + content[section_end:]
                index_path.write_text(content, encoding="utf-8")
                return

        # Migration: no ## Research Ideas section found — create one
        # Find the main heading and insert before the first ## section after it
        main_heading = content.find("# Idea Wiki Index")
        if main_heading != -1:
            main_heading_end = content.find("\n", main_heading)
            if main_heading_end == -1:
                main_heading_end = main_heading + len("# Idea Wiki Index")

            first_section = content.find("\n## ", main_heading_end)
            suffix = content[first_section:] if first_section != -1 else ""

            # Collect orphaned wikilinks between main heading and first ## section
            between = content[main_heading_end:first_section] if first_section != -1 else content[main_heading_end:]
            orphaned = re.findall(r'^- \[\[.*$', between, re.MULTILINE)

            section_lines = [
                "",
                "## Research Ideas (PaperRAG)",
                "> Auto-generated by PaperRAG `/idea` command. Each topic folder contains an index.md with individual idea pages.",
                "",
            ]
            if orphaned:
                section_lines.extend(orphaned)
                section_lines.append("")
            section_lines.append(entry.strip())

            section_block = "\n".join(section_lines) + "\n"
            content = content[:main_heading_end] + section_block + suffix
            index_path.write_text(content, encoding="utf-8")
            return

        # Last resort: append at end
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(entry)

    # ---- log.md 操作 ----

    def append_log(self, action: str, detail: str) -> None:
        """追加操作到 log.md（每条日志带唯一 ID）"""
        log_path = self.root / "log.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry_id = str(_uuid.uuid4())[:8]
        line = f"- **{timestamp}** [`{entry_id}`] [{action}] {detail}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

    # ---- context.md（Raw Layer）----

    def save_context(self, topic: str, context_data: dict) -> Path:
        """保存原始 context 为 context.md（Raw Layer，不可变——仅首次写入）"""
        topic_slug = slugify(topic)
        topic_dir = self.topic_dir(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)

        path = topic_dir / "context.md"
        if path.exists():
            logger.info(f"[IdeaWiki] context.md already exists, skipping (immutable): {path}")
            return path

        lines = [
            "---",
            f"title: Raw Context — {_escape_yaml(topic)}",
            f"type: raw-context",
            f"topic: {topic_slug}",
            f"topic_title: {_escape_yaml(topic)}",
            f"aliases: []",
            f"cssclasses: [raw-context]",
            f"publish: false",
            f"created: {datetime.now().strftime('%Y-%m-%d')}",
            "---",
            "",
            "# Raw Context",
            "",
        ]

        local = context_data.get("local_results", [])
        web = context_data.get("web_results", [])

        if local:
            lines.append(f"## Local Results ({len(local)} items)\n")
            for r in local[:10]:
                lines.append(
                    f"- **{r.get('paper', '?')}**: "
                    f"{r.get('text', '')[:200]}..."
                )

        if web:
            lines.append(f"\n## Web Results ({len(web)} items)\n")
            for r in web[:5]:
                lines.append(
                    f"- [{r.get('title', '?')}]({r.get('url', '')}): "
                    f"{r.get('snippet', '')[:150]}..."
                )

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[IdeaWiki] context.md saved: {path}")
        return path

    # ---- topic index.md ----

    def init_topic_index(self, topic: str, analysis: dict | None = None) -> Path:
        """创建或更新 topic/index.md"""
        topic_dir = self.topic_dir(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)

        slug = slugify(topic)
        path = topic_dir / "index.md"

        if path.exists():
            return path  # 幂等，不覆盖

        lines = [
            "---",
            f"title: {_escape_yaml(topic)}",
            f"slug: {slug}",
            f"type: topic-index",
            f"topic_title: {_escape_yaml(topic)}",
            f"aliases: []",
            f"cssclasses: [topic-index]",
            f"publish: false",
            f"created: {datetime.now().strftime('%Y-%m-%d')}",
            "---",
            "",
            f"# {topic}",
            "",
        ]

        if analysis:
            lines.append("## Topic Analysis\n")
            summary = analysis.get("summary", "")
            if summary:
                lines.append(f"{summary}\n")
            keywords = analysis.get("keywords", [])
            if keywords:
                lines.append(f"**Keywords**: {', '.join(keywords)}\n")

        lines.append("\n## Ideas\n")
        lines.append("> Ideas generated under this topic appear below.\n")
        # Dataview query block — uncomment in Obsidian for dynamic listing
        lines.append("```dataview\n"
                     "TABLE feasibility, novelty_score, confidence\n"
                     f'FROM "ideas/{slug}"\n'
                     'WHERE type = "concept"\n'
                     "SORT feasibility DESC\n"
                     "```\n")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def add_idea_to_topic_index(
        self,
        topic: str,
        idea_slug: str,
        idea_title: str,
        scores: dict | None = None,
    ) -> None:
        """在 topic/index.md 的 Ideas 节下追加一条 idea 引用"""
        path = self.topic_dir(topic) / "index.md"
        if not path.exists():
            return

        score_str = ""
        if scores:
            score_parts = []
            for key, label in [("feasibility", "f"), ("novelty", "n"), ("score", "s")]:
                val = scores.get(key)
                if isinstance(val, (int, float)):
                    score_parts.append(f"{label}={val:.2f}")
            if score_parts:
                score_str = f" `[{', '.join(score_parts)}]`"

        entry = f"- [[{idea_slug}|{idea_title}]]{score_str}\n"

        content = path.read_text(encoding="utf-8")
        marker = "## Ideas"
        idx = content.find(marker)
        if idx == -1:
            content += f"\n{marker}\n{entry}"
        else:
            insert_pos = content.find("\n", idx)
            if insert_pos == -1:
                content += entry
            else:
                content = content[:insert_pos] + "\n" + entry + content[insert_pos:]

        path.write_text(content, encoding="utf-8")

    # ---- entity / concept / comparison 页面（Hermes 知识提取）----

    def save_page(
        self,
        page_type: str,
        title: str,
        slug: str,
        content_md: str,
        tags: list[str] | None = None,
        confidence: str = "medium",
        sources_list: list[str] | None = None,
        contested: bool = False,
        aliases: list[str] | None = None,
    ) -> Path:
        """写入 entity / concept / comparison 页面。

        Args:
            page_type: "entity" | "concept" | "comparison"
            title: 页面标题
            slug: 文件名 slug（不含 .md）
            content_md: Markdown 正文（不含 frontmatter）
            tags: 标签列表，必须来自 SCHEMA.md 定义的 taxonomy
            confidence: high | medium | low
            sources_list: 来源引用列表
            contested: 是否标记为存在争议
            aliases: 别名列表
        """
        page_dir = self._page_dir(page_type)
        page_dir.mkdir(parents=True, exist_ok=True)
        path = page_dir / f"{slug}.md"

        created_date = datetime.now().strftime('%Y-%m-%d')
        updated_date = created_date
        if path.exists():
            logger.warning(f"[IdeaWiki] {page_type} page already exists, updating: {path}")
            existing = path.read_text(encoding="utf-8")
            for line in existing.splitlines():
                if line.startswith("created:"):
                    created_date = line.split(":", 1)[1].strip()
                    break

        fm: list[str] = [
            "---",
            f"title: {_escape_yaml(title)}",
            f"slug: {slug}",
            f"cssclasses: [{page_type}]",
            f"publish: false",
            f"created: {created_date}",
            f"updated: {updated_date}",
            f"type: {page_type}",
        ]
        # YAML block list — handles commas, brackets, quotes in values safely
        if aliases:
            fm.append("aliases:")
            for a in aliases:
                fm.append(f"  - {_escape_yaml(a)}")
        else:
            fm.append("aliases: []")
        if tags:
            fm.append("tags:")
            for t in tags:
                fm.append(f"  - {t}")
        if sources_list:
            fm.append("sources:")
            for s in sources_list:
                fm.append(f"  - {_escape_yaml(s)}")
        fm.append(f"confidence: {confidence}")
        if contested:
            fm.append("contested: true")
        fm.append("---")

        content = "\n".join(fm) + "\n\n" + content_md.strip() + "\n"
        path.write_text(content, encoding="utf-8")
        logger.info(f"[IdeaWiki] {page_type} page saved: {path}")

        # 注册到 index.md
        self._register_page_in_index(slug, page_type, title)

        return path

    def _register_page_in_index(self, slug: str, page_type: str, title: str) -> None:
        """在全局 index.md 的对应节下注册新页面。"""
        index_path = self.root / "index.md"
        if not index_path.exists():
            return

        section = self._index_section_marker(page_type)
        entry = f"- [[{page_type}/{slug}|{title}]]\n"

        content = index_path.read_text(encoding="utf-8")
        marker = f"## {section}"
        idx = content.find(marker)
        if idx == -1:
            logger.warning(
                f"[IdeaWiki] Section '## {section}' not found in index.md; "
                f"page '{slug}' ({page_type}) not registered"
            )
            return

        if f"[[{page_type}/{slug}" in content:
            return  # 已注册，跳过

        # 插入到 section 第一个 ## 之前或文件末尾
        rest = content[idx + len(marker):]
        next_section = rest.find("\n## ")
        if next_section != -1:
            insert_pos = idx + len(marker) + next_section
            content = content[:insert_pos] + entry + content[insert_pos:]
        else:
            content = content.rstrip() + "\n" + entry + "\n"

        index_path.write_text(content, encoding="utf-8")

    # ---- 单个 idea.md ----

    def save_idea(
        self,
        topic: str,
        idea_dict: dict,
        idea_id: str,
        context_path: str | None = None,
        scores: dict | None = None,
        debate_rounds: int = 0,
    ) -> Path:
        """
        将单个 idea 写入为 Wiki 格式 Markdown。

        Args:
            topic: 原始 topic 字符串
            idea_dict: idea 字段（来自 ResearchIdea.to_dict()）
            idea_id: UUID（前 8 位用于 slug）
            context_path: 对应的 context.md 相对路径（provenance）
            scores: {feasibility, novelty, confidence, ...}
            debate_rounds: 经过了几轮 debate
        """
        title = idea_dict.get("title", "Untitled")
        slug = f"{slugify(title)}-{idea_id[:8]}"

        topic_dir = self.topic_dir(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)
        path = topic_dir / f"{slug}.md"

        # Preserve original created date if overwriting
        created_date = datetime.now().strftime('%Y-%m-%d')
        if path.exists():
            logger.warning(f"[IdeaWiki] idea already exists, updating: {path}")
            existing = path.read_text(encoding="utf-8")
            for line in existing.splitlines():
                if line.startswith("created:"):
                    created_date = line.split(":", 1)[1].strip()
                    break

        # frontmatter — escape YAML-sensitive values
        fm = [
            "---",
            f"title: {_escape_yaml(title)}",
            f"slug: {slug}",
            f"topic: {slugify(topic)}",
            f"topic_title: {_escape_yaml(topic)}",
            f"aliases: []",
            f"cssclasses: [research-idea]",
            f"publish: false",
            f"type: concept",
            f"idea_id: {idea_id}",
            f"created: {created_date}",
            f"updated: {datetime.now().strftime('%Y-%m-%d')}",
            f"debate_rounds: {debate_rounds}",
        ]

        # tags
        tags = ["idea"]
        if scores:
            conf = scores.get("confidence", "medium")
            tags.append(f"confidence:{conf}")
        fm.append(f"tags: [{', '.join(tags)}]")

        # scores
        if scores:
            fm.append(f"feasibility: {scores.get('feasibility', 0.5)}")
            fm.append(f"novelty_score: {scores.get('novelty', 0.5)}")

        # provenance
        sources = []
        if context_path:
            sources.append(f"context:{context_path}")
        fm.append(f"sources: [{', '.join(sources)}]")

        # related work
        related = idea_dict.get("related_work", [])
        if related:
            fm.append(f"related_work: [{', '.join(str(r)[:40] for r in related[:5])}]")

        fm.append("---")

        # body
        body = [f"# {title}", ""]

        novelty = idea_dict.get("novelty", "")
        if novelty:
            body.append("## Novelty\n")
            body.append(f"{novelty}\n")

        desc = idea_dict.get("description", "")
        if desc:
            body.append("## Description\n")
            body.append(f"{desc}\n")

        methodology = idea_dict.get("methodology", "")
        if methodology:
            body.append("## Methodology\n")
            body.append(f"{methodology}\n")

        challenges = idea_dict.get("potential_challenges", [])
        if challenges:
            body.append("## Challenges\n")
            for c in challenges:
                body.append(f"- {c}")
            body.append("")

        # Cross-idea links — discover sibling ideas for graph connectivity
        see_also = self._discover_sibling_ideas(topic, slug)
        if see_also:
            body.append("## See Also\n")
            for sib_slug, sib_title in see_also[:3]:
                body.append(f"- [[{sib_slug}|{sib_title}]]")
            body.append("")

        # linkback to topic index
        body.append(
            f"← [[{slugify(topic)}/index|Back to {topic}]]\n"
        )

        content = "\n".join(fm) + "\n\n" + "\n".join(body)
        path.write_text(content, encoding="utf-8")
        logger.info(f"[IdeaWiki] idea saved: {path}")

        return path

    def _discover_sibling_ideas(
        self, topic: str, exclude_slug: str
    ) -> list[tuple[str, str]]:
        """扫描同 topic 下已有 idea，返回 [(slug, title), ...] 用于 See Also 交叉链接"""
        siblings = []
        topic_dir = self.topic_dir(topic)
        if not topic_dir.exists():
            return siblings
        for md in sorted(topic_dir.glob("*.md")):
            if md.name in ("index.md", "context.md"):
                continue
            if md.stem == exclude_slug:
                continue
            try:
                content = md.read_text(encoding="utf-8")
                m = re.search(r"^slug:\s*(.+)$", content, re.MULTILINE)
                sib_slug = m.group(1).strip() if m else md.stem
                m = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
                sib_title = m.group(1).strip().strip('"') if m else md.stem
                siblings.append((sib_slug, sib_title))
            except Exception:
                pass
        return siblings

    # ---- Query helpers ----

    def search_ideas(self, query: str) -> list[tuple[Path, str]]:
        """
        简单 grep 检索，返回 (path, 匹配片段)。
        后续可替换为 semantic search 或 Hermes session_search。
        """
        results = []
        ideas_dir = self.ideas_dir()
        if not ideas_dir.exists():
            return results

        for md in ideas_dir.rglob("*.md"):
            if md.name in ("index.md", "context.md"):
                continue
            try:
                content = md.read_text(encoding="utf-8")
                if query.lower() in content.lower():
                    first = content.split("\n", 1)[0][:80]
                    results.append((md, first))
            except Exception:
                pass
        return results

    # ---- 路径访问器 ----

    def get_log_path(self) -> Path:
        return self.root / "log.md"

    def get_index_path(self) -> Path:
        return self.root / "index.md"

    def get_wiki_root(self) -> Path:
        return self.root


# ---- 默认 Schema 文本 ----

DEFAULT_SCHEMA = """\
# PaperRAG Idea Wiki Schema

> Shared knowledge base for PaperRAG (AstrBot) and Hermes Agent.
> Both systems read/write this wiki. PaperRAG auto-generates ideas;
> Hermes creates entity pages, comparisons, and query archives.

## Domain
AI/ML research ideation and knowledge management.

## Directory Structure
```
wiki/
├── SCHEMA.md           # This file — conventions, structure, tag taxonomy
├── index.md            # Sectioned content catalog
├── log.md              # Append-only action log (rotated yearly)
├── ideas/              # PaperRAG: auto-generated research ideas
│   └── <topic-slug>/
│       ├── index.md    # Topic overview with idea list
│       ├── context.md  # Raw Layer: search results (immutable)
│       └── <idea-slug>.md  # Individual idea page
├── entities/           # Hermes: entity pages (methods, models, datasets)
├── concepts/           # Hermes: concept/topic explorations
├── comparisons/        # Hermes: side-by-side analyses
├── queries/            # Hermes: filed query results
└── raw/                # Layer 1: immutable source material
    ├── articles/
    ├── papers/
    ├── transcripts/
    └── assets/
```

## Naming Conventions
- topic slug: lowercase + hyphens, max 60 chars
- idea slug: `<title-slug>-<uuid-first-8>`, max 80 chars
- File name must match frontmatter `slug`

## Tag Taxonomy

### For ideas/ (PaperRAG)
- idea-type: [hypothesis, method, application, survey, tool]
- status: [raw, generated, critiqued, refined, archived]
- domain: [ml, nlp, cv, rl, theory, system, data]
- confidence: [high, medium, low]

### For entities/ and concepts/ (Hermes)
- Models: model, architecture, benchmark, training
- People/Orgs: person, company, lab, open-source
- Techniques: optimization, fine-tuning, inference, alignment, data
- Meta: comparison, timeline, controversy, prediction

Rule: every tag on a page must appear in this taxonomy. Add new tags here first.

## Frontmatter

### Standard (all pages)
```yaml
---
title: Page Title
slug: page-slug
aliases: []
cssclasses: [page-type]
publish: false
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: entity | concept | comparison | summary | raw-context
tags: [from taxonomy above]
sources: [relative/path/to/source.md]
confidence: high | medium | low     # optional
contested: true                      # optional — unresolved contradictions
---
```

### Ideas (PaperRAG extensions)
```yaml
---
title: Idea Title
slug: idea-slug
topic: <topic-slug>
topic_title: <original topic name>
aliases: []
cssclasses: [research-idea]
publish: false
type: concept
tags: [idea-type, domain, status, confidence]
created: YYYY-MM-DD
updated: YYYY-MM-DD
idea_id: <uuid>
feasibility: <0.0-1.0>
novelty_score: <0.0-1.0>
debate_rounds: <int>
sources:
  - context:<context.md>
  - related:<other-idea-slug>
confidence: high | medium | low
---
```

### Raw sources (Layer 1)
```yaml
---
source_url: https://example.com
ingested: YYYY-MM-DD
sha256: <hex digest>
---
```

## Required Conventions
- Every idea page MUST have ## Novelty, ## Methodology, ## Challenges, ## See Also sections
- Every page MUST have at least 2 outbound [[wikilinks]] (backlink + cross-refs)
- When updating a page, ALWAYS bump the `updated` date
- ALL changes MUST be appended to `log.md`
- Every new page MUST be registered in `index.md` under the correct section
- log.md uses `## YYYY` year section headers, entries: `- **{timestamp}** [{uuid8}] [{action}] {detail}`
- Every page SHOULD set `publish: false` by default; set `publish: true` to publish via Quartz
"""

DEFAULT_INDEX = """\
# Idea Wiki Index

> Open this vault in Obsidian with the Dataview plugin to enable dynamic queries below.

## Research Ideas (PaperRAG)
> Auto-generated by PaperRAG `/idea` command. Each topic folder contains an index.md with individual idea pages.

```dataview
LIST FROM "ideas" WHERE type = "topic-index" SORT file.ctime DESC
```

## Entities (Hermes)
> Entity pages for methods, models, datasets, and other named research artifacts.

```dataview
LIST FROM "entities" SORT file.ctime DESC
```

## Concepts (Hermes)
> Concept explorations and topic deep-dives.

## Comparisons (Hermes)
> Side-by-side analyses and contrastive studies.

## Queries (Hermes)
> Filed query results worth preserving.
"""

