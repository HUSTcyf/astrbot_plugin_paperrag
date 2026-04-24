from rag.abstract_index import AbstractExtractor


def _block(page: int, text: str):
    return (page, 0.0, 0.0, 100.0, 100.0, text)


def test_ieee_abstract_dash_block_is_extracted_completely():
    extractor = AbstractExtractor()
    blocks = [
        _block(0, "FAST: Efficient Action Tokenization for Vision-Language-Action Models"),
        _block(
            0,
            "Abstract—Autoregressive sequence models, such as Transformer-based "
            "vision-language action policies, can be tremendously effective for "
            "capturing complex and generalizable robotic behaviors. However, such "
            "models require us to choose a tokenization of continuous action signals.",
        ),
        _block(0, "I. INTRODUCTION"),
        _block(0, "Large, high-capacity Transformer models can be tremendously effective."),
    ]

    abstract = extractor._extract_abstract_from_blocks(blocks)

    assert abstract is not None
    assert abstract.startswith("Autoregressive sequence models")
    assert len(abstract) > 180
    assert "INTRODUCTION" not in abstract


def test_unlabeled_sam_style_first_page_abstract_is_extracted():
    extractor = AbstractExtractor()
    blocks = [
        _block(0, "SAM 3D: 3Dfy Anything in Images"),
        _block(0, "SAM 3D Team, Example Author, Another Author"),
        _block(
            0,
            "We present SAM 3D, a generative model for visually grounded 3D object "
            "reconstruction, predicting geometry, texture, and layout from a single "
            "image. SAM 3D excels in natural images, where occlusion and scene clutter "
            "are common. We achieve this with a human- and model-in-the-loop pipeline "
            "for annotating object shape, texture, and pose, providing visually grounded "
            "3D reconstruction data at unprecedented scale.",
        ),
        _block(0, "Introduction"),
    ]

    abstract = extractor._extract_abstract_from_blocks(blocks)

    assert abstract is not None
    assert abstract.startswith("We present SAM 3D")
    assert len(abstract) > 250


def test_hal_cover_page_does_not_hide_later_abstract_like_block():
    extractor = AbstractExtractor()
    blocks = [
        _block(0, "HAL Id: hal-04088161"),
        _block(0, "HAL is a multi-disciplinary open access archive for research documents."),
        _block(1, "3D Gaussian Splatting for Real-Time Radiance Field Rendering"),
        _block(
            1,
            "Radiance Field methods have recently revolutionized novel-view synthesis "
            "of scenes captured with multiple photos or videos. However, achieving high "
            "visual quality still requires neural networks that are costly to train and "
            "render. We introduce three key elements that allow us to achieve state-of-the-art "
            "visual quality while maintaining competitive training times and high-quality "
            "real-time novel-view synthesis at 1080p resolution.",
        ),
        _block(1, "1 INTRODUCTION"),
    ]

    abstract = extractor._extract_abstract_from_blocks(blocks)

    assert abstract is not None
    assert abstract.startswith("Radiance Field methods")
    assert "HAL is a multi-disciplinary" not in abstract


def test_line_fallback_does_not_truncate_abstract_dash_after_three_lines():
    extractor = AbstractExtractor()
    text = "\n".join(
        [
            "Abstract—Autoregressive",
            "sequence",
            "models,",
            "such",
            "as",
            "Transformer-based",
            "vision-language action policies, can be tremendously effective for capturing",
            "complex robotic behaviors and generalizable policies across tasks.",
            "I. INTRODUCTION",
        ]
    )

    abstract = extractor._extract_abstract_text(text)

    assert abstract is not None
    assert abstract.startswith("Autoregressive sequence models")
    assert len(abstract) > 100
    assert "INTRODUCTION" not in abstract
