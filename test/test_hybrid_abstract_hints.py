import asyncio
import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace


class DummyLogger:
    def debug(self, *args, **kwargs):
        pass

    info = warning = error = debug


def _load_add_papers_method():
    source_path = Path(__file__).resolve().parents[1] / "rag" / "hybrid_rag.py"
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    plugin_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HybridRAGEngine"
    )
    add_papers = next(
        node for node in plugin_class.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "add_papers"
    )
    method_src = "".join(lines[add_papers.lineno - 1:add_papers.end_lineno])

    namespace = {
        "Any": object,
        "Dict": dict,
        "List": list,
        "logger": DummyLogger(),
        "os": os,
    }
    exec(
        "from typing import Any, Dict, List\n\n"
        "class ExtractedHybridRAGEngine:\n"
        f"{method_src}",
        namespace,
    )
    return namespace["ExtractedHybridRAGEngine"]


def test_add_papers_passes_parser_hints_into_abstract_index():
    hybrid_cls = _load_add_papers_method()

    class FakeParser:
        async def parse_and_split(self, file_path, llm_config, arxiv_client):
            return [
                SimpleNamespace(
                    text="chunk one",
                    metadata={
                        "file_name": "demo.pdf",
                        "extracted_title": "Parsed Title",
                        "extracted_abstract": "This abstract came from the parser and is long enough to be trusted directly.",
                        "github_url": "https://github.com/example/repo",
                        "title_source": "parser_llm",
                        "abstract_source": "parser_llm",
                    },
                )
            ]

    class FakeIndexManager:
        async def insert_nodes(self, nodes, embeddings):
            assert len(nodes) == len(embeddings) == 1
            return 1

    class FakeEmbedProvider:
        async def get_embeddings(self, texts):
            assert texts == ["chunk one"]
            return [[0.1, 0.2, 0.3]]

    class FakeAbstractManager:
        def __init__(self):
            self.calls = []
            self._abstract_cache = {}

        async def index_paper(self, **kwargs):
            self.calls.append(kwargs)
            self._abstract_cache[kwargs["paper_id"]] = SimpleNamespace(title=kwargs["title"])
            return True

    engine = hybrid_cls()
    engine.config = SimpleNamespace(enable_multi_vector_rerank=False, enable_llm_reference_parsing=False)
    engine._retriever = None
    engine._ensure_parser_initialized = lambda: FakeParser()
    engine._ensure_index_manager_initialized = lambda: FakeIndexManager()

    async def ensure_embed_provider():
        return FakeEmbedProvider()

    abstract_manager = FakeAbstractManager()

    async def ensure_abstract_manager():
        return abstract_manager

    engine._ensure_embed_provider_initialized = ensure_embed_provider
    engine._ensure_abstract_manager_initialized = ensure_abstract_manager

    results = asyncio.run(engine.add_papers(["/tmp/demo.pdf"]))

    assert results["successful"] == 1
    assert len(abstract_manager.calls) == 1

    call = abstract_manager.calls[0]
    assert call["paper_id"] == "demo"
    assert call["file_name"] == "demo.pdf"
    assert call["title"] == "Parsed Title"
    assert call["abstract_text"].startswith("This abstract came from the parser")
    assert call["metadata"]["github_url"] == "https://github.com/example/repo"
    assert call["metadata"]["title_source"] == "parser_llm"
    assert call["metadata"]["abstract_source"] == "parser_llm"
    assert call["metadata"]["extracted_title"] == "Parsed Title"
    assert call["metadata"]["extracted_abstract_chars"] == len(call["abstract_text"])
