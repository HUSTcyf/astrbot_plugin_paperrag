import asyncio

from rag.paper_link_resolver import PaperLinkResolver, PdfProbe


def test_extract_arxiv_and_doi_urls_from_work():
    work = {
        "arxivId": "2405.12110v2",
        "sourceFulltextUrls": [
            "https://arxiv.org/pdf/2405.12110v2.pdf",
            "https://github.com/example/repo",
        ],
        "doi": "10.48550/arXiv.2405.12110",
        "ids": {
            "doi": "https://doi.org/10.48550/arXiv.2405.12110",
            "arxiv": "https://arxiv.org/abs/2405.12110v2",
        },
    }

    assert PaperLinkResolver.extract_arxiv_url_from_work(work) == "https://arxiv.org/abs/2405.12110"
    assert PaperLinkResolver.extract_github_url_from_work(work) == "https://github.com/example/repo"
    assert PaperLinkResolver.extract_doi_url_from_work(work) == "https://doi.org/10.48550/arXiv.2405.12110"


def test_strip_identifier_noise_removes_trailing_brackets():
    assert PaperLinkResolver._strip_identifier_noise("10.1145/3592433⟩") == "10.1145/3592433"
    assert PaperLinkResolver._strip_identifier_noise("2405.12110v2.") == "2405.12110v2"


def test_extract_identifier_candidates_handles_page_footers():
    resolver = PaperLinkResolver()
    text = "\n".join([
        "Mini-Splatting2: Building 360 Scenes within Minutes",
        "via Aggressive Gaussian Densification",
        "arXiv:2411.12788v1  [cs.CV]  19 Nov 2024",
        "DOI: 10.1145/3592433⟩",
    ])

    doi_candidates, arxiv_candidates = resolver._extract_identifier_candidates(text)
    assert doi_candidates == ["10.1145/3592433"]
    assert arxiv_candidates == ["2411.12788"]


def test_extract_title_candidates_handles_split_titles():
    resolver = PaperLinkResolver()

    mini_text = "\n".join([
        "Mini-Splatting2: Building 360 Scenes within Minutes",
        "via Aggressive Gaussian Densification",
        "Guangchi Fang, Bing Wang",
        "The Hong Kong Polytechnic University",
        "Abstract",
    ])

    assert resolver._extract_title_candidates_from_text(mini_text) == [
        "Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification"
    ]

    class FakePage:
        def get_text(self, mode="text"):
            if mode == "dict":
                return {
                    "blocks": [
                        {
                            "type": 0,
                            "bbox": [87.4, 95.0, 524.6, 134.2],
                            "lines": [
                                {
                                    "spans": [
                                        {"text": "LiteGS: A High-Performance Modular Framework for Gaussian", "size": 17.2},
                                    ]
                                },
                                {
                                    "spans": [
                                        {"text": "Splatting Training", "size": 17.2},
                                    ]
                                },
                            ],
                        },
                        {
                            "type": 0,
                            "bbox": [274.0, 149.9, 337.9, 161.9],
                            "lines": [
                                {
                                    "spans": [
                                        {"text": "Kaimin Liao", "size": 11.9},
                                    ]
                                }
                            ],
                        },
                    ]
                }
            return ""

    assert resolver._extract_title_from_layout(FakePage()) == (
        "LiteGS: A High-Performance Modular Framework for Gaussian Splatting Training"
    )


def test_resolve_from_pdf_uses_split_title_candidates_for_failed_samples():
    class StubResolver(PaperLinkResolver):
        def extract_pdf_probe(self, pdf_path: str, max_chars: int = 3000):
            if "MiniSplatting2" in pdf_path:
                return PdfProbe(
                    pdf_path=pdf_path,
                    first_page_title="Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification",
                    first_page_author="Guangchi Fang, Bing Wang",
                    title_candidates=[
                        "Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification"
                    ],
                )
            return PdfProbe(
                pdf_path=pdf_path,
                first_page_title="LiteGS: A High-Performance Modular Framework for Gaussian Splatting Training",
                first_page_author="Kaimin Liao",
                title_candidates=[
                    "LiteGS: A High-Performance Modular Framework for Gaussian Splatting Training"
                ],
            )

        async def _search_crossref_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

        async def _search_openalex_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            if "Mini-Splatting2" in title:
                return [
                    {
                        "title": "Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification",
                        "arxiv_id": "2411.12788v1",
                        "authors": ["Guangchi Fang", "Bing Wang"],
                    }
                ]
            return [
                {
                    "title": "LiteGS: A High-Performance Modular Framework for Gaussian Splatting Training",
                    "arxiv_id": "2503.01199v1",
                    "authors": ["Kaimin Liao"],
                }
            ]

        async def _search_core_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

        async def _search_arxiv_library_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

    resolver = StubResolver(core_api_key="", enable_arxiv_library=False)
    mini = asyncio.run(resolver.resolve_from_pdf("MiniSplatting2.pdf"))
    lite = asyncio.run(resolver.resolve_from_pdf("LiteGS.pdf"))

    assert mini.arxiv_url == "https://arxiv.org/abs/2411.12788"
    assert mini.matched_title == "Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification"
    assert lite.arxiv_url == "https://arxiv.org/abs/2503.01199"
    assert lite.matched_title == "LiteGS: A High-Performance Modular Framework for Gaussian Splatting Training"


def test_resolve_from_pdf_prefers_direct_pdf_identifiers():
    class StubResolver(PaperLinkResolver):
        def extract_pdf_probe(self, pdf_path: str, max_chars: int = 3000):
            return PdfProbe(
                pdf_path=pdf_path,
                metadata_title="Vision Transformers for Dense Prediction",
                metadata_author="René Ranftl",
                metadata_doi="10.5555/example.doi",
                first_page_title="Vision Transformers for Dense Prediction",
                first_page_author="René Ranftl",
            )

        async def resolve_by_title(self, title: str, author_hint: str = ""):
            raise AssertionError("title search should not run when DOI is present")

    resolution = asyncio.run(
        StubResolver(core_api_key="x", enable_arxiv_library=False).resolve_from_pdf("dummy.pdf")
    )

    assert resolution.backend == "PDF metadata/text"
    assert resolution.resolution_source == "PDF metadata/text"
    assert resolution.doi_url == "https://doi.org/10.5555/example.doi"
    assert resolution.matched_title == "Vision Transformers for Dense Prediction"
    assert resolution.matched_identifier == "10.5555/example.doi"


def test_resolve_from_pdf_uses_first_page_title_when_metadata_is_empty():
    class StubResolver(PaperLinkResolver):
        def extract_pdf_probe(self, pdf_path: str, max_chars: int = 3000):
            return PdfProbe(
                pdf_path=pdf_path,
                metadata_title="",
                metadata_author="",
                first_page_title="InstantSplat: Sparse-view Gaussian Splatting in Seconds",
                first_page_author="Junyi Zhu",
                title_candidates=["InstantSplat: Sparse-view Gaussian Splatting in Seconds"],
            )

        async def _search_crossref_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

        async def _search_openalex_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return [
                {
                    "title": "InstantSplat: Sparse-view Gaussian Splatting in Seconds",
                    "arxiv_id": "2405.12110v2",
                    "authors": ["Junyi Zhu", "Yue Wu"],
                }
            ]

        async def _search_core_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

        async def _search_arxiv_library_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

    resolution = asyncio.run(
        StubResolver(core_api_key="", enable_arxiv_library=False).resolve_from_pdf("dummy.pdf")
    )

    assert resolution.backend == "OpenAlex"
    assert resolution.resolution_source == "pdf first-page title -> OpenAlex"
    assert resolution.arxiv_url == "https://arxiv.org/abs/2405.12110"
    assert resolution.matched_title == "InstantSplat: Sparse-view Gaussian Splatting in Seconds"
    assert resolution.resolution_score >= 95.0


def test_resolver_uses_author_hint_to_choose_better_crossref_result():
    class StubResolver(PaperLinkResolver):
        async def _search_crossref_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return [
                {
                    "title": "Vision Transformers for Dense Prediction",
                    "authors": ["Someone Else"],
                    "arxivId": "0000.00000v1",
                },
                {
                    "title": "Vision Transformers for Dense Prediction",
                    "authors": ["René Ranftl", "Roberto"],
                    "arxivId": "2103.13413v1",
                },
            ]

        async def _search_openalex_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

        async def _search_core_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

        async def _search_arxiv_library_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

    resolution = asyncio.run(
        StubResolver(core_api_key="x", enable_arxiv_library=False).resolve_by_title(
            "Vision Transformers for Dense Prediction",
            author_hint="René Ranftl",
        )
    )

    assert resolution.backend == "Crossref"
    assert resolution.arxiv_url == "https://arxiv.org/abs/2103.13413"
    assert resolution.matched_identifier == "2103.13413v1"
    assert resolution.resolution_score >= 90.0


def test_openalex_fallback_still_works_when_arxiv_library_is_disabled():
    class StubResolver(PaperLinkResolver):
        async def _search_crossref_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

        async def _search_openalex_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return [
                {
                    "title": "InstantSplat: Sparse-view Gaussian Splatting in Seconds",
                    "arxiv_id": "2405.12110v2",
                    "authors": ["Junyi Zhu", "Yue Wu"],
                }
            ]

        async def _search_core_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

        async def _search_arxiv_library_candidates(self, title: str, limit: int = 5, author_hint: str = ""):
            return []

    resolution = asyncio.run(
        StubResolver(core_api_key="", enable_arxiv_library=False).resolve_by_title(
            "InstantSplat: Sparse-view Gaussian Splatting in Seconds"
        )
    )

    assert resolution.backend == "OpenAlex"
    assert resolution.arxiv_url == "https://arxiv.org/abs/2405.12110"
    assert resolution.resolution_source == "OpenAlex"
