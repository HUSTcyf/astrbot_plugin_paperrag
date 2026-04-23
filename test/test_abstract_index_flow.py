import asyncio
import tempfile

from rag.abstract_index import AbstractIndexManager
from rag.paper_link_resolver import LinkResolution


def test_index_paper_extracts_title_and_urls_in_one_flow():
    class FakeLLM:
        async def extract_title_and_abstract(self, text: str):
            assert "paper beginning" in text
            return (
                "LLM Confirmed Title",
                "This is a sufficiently long abstract body for the indexer to accept without fallback.",
            )

    class FakeResolver:
        async def resolve_from_pdf(self, pdf_path: str, title_hint: str = ""):
            assert pdf_path.endswith("demo.pdf")
            assert title_hint == "LLM Confirmed Title"
            return LinkResolution(
                arxiv_url="https://arxiv.org/abs/2503.01199",
                doi_url="https://doi.org/10.48550/arxiv.2503.01199",
                backend="OpenAlex",
                resolution_source="pdf first-page title -> OpenAlex",
                resolution_score=99.0,
                matched_title="LLM Confirmed Title",
                matched_identifier="2503.01199",
                score=99.0,
            )

    class FakeCollection:
        def __init__(self):
            self.inserted = []
            self.flushed = False

        def insert(self, data):
            self.inserted.append(data)

        def flush(self):
            self.flushed = True

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AbstractIndexManager(
                milvus_uri=f"{tmpdir}/milvus_abstracts.db",
                embed_dim=3,
                core_api_key="",
                use_arxiv_api=True,
            )
            manager._collection = FakeCollection()
            manager._embed_model = object()
            manager.set_llm_client(FakeLLM())
            manager._link_resolver = FakeResolver()

            async def fake_ensure_collection():
                return None

            async def fake_extract_beginning(pdf_path: str, max_chars: int = 3000):
                return "paper beginning with enough context for LLM extraction"

            async def fake_embed(text: str):
                assert "LLM Confirmed Title" in text
                return [0.1, 0.2, 0.3]

            manager._ensure_collection = fake_ensure_collection  # type: ignore[assignment]
            manager._extract_paper_beginning = fake_extract_beginning  # type: ignore[assignment]
            manager._embed_text = fake_embed  # type: ignore[assignment]

            ok = await manager.index_paper(
                pdf_path="/tmp/demo.pdf",
                paper_id="demo",
                file_name="demo.pdf",
                metadata={"source_kind": "pdf"},
            )

            assert ok is True
            assert len(manager._collection.inserted) == 1

            cached = manager._abstract_cache["demo"]
            assert cached.title == "LLM Confirmed Title"
            assert cached.abstract_text.startswith("This is a sufficiently long abstract body")
            assert cached.metadata["arxiv_url"] == "https://arxiv.org/abs/2503.01199"
            assert cached.metadata["doi_url"] == "https://doi.org/10.48550/arxiv.2503.01199"
            assert cached.metadata["resolution_source"] == "pdf first-page title -> OpenAlex"
            assert cached.metadata["matched_title"] == "LLM Confirmed Title"
            assert cached.metadata["matched_identifier"] == "2503.01199"
            assert cached.metadata["title_source"] == "llm"
            assert cached.metadata["abstract_source"] == "llm"

    asyncio.run(run())
