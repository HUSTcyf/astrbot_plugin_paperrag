#!/usr/bin/env python3
"""使用 AstrBot Python 环境测试"""
import sys
import os
from pathlib import Path

plugin_dir = Path(__file__).parent
models_dir = plugin_dir / "models"

# 模拟 _configure_docling_globals() 设置的环境变量
os.environ["HF_HOME"] = str(models_dir)
os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print("=" * 60)
print("AstrBot Python 环境测试")
print("Python:", sys.executable)
print("=" * 60)

# 先导入 torch
print("\n[1] 导入 torch...")
import torch
print(f"  torch: {torch.__version__}")
print(f"  MPS available: {torch.backends.mps.is_available()}")

# 尝试 monkey patch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
print(f"  Monkey patched MPS available: {torch.backends.mps.is_available()}")

# 然后导入 docling
print("\n[2] 导入 docling.document_converter...")
from docling.document_converter import DocumentConverter, PdfFormatOption
print("  ✅ 导入成功")

print("\n[3] 创建 DocumentConverter...")
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings

settings.cache_dir = str(models_dir)
settings.artifacts_path = str(models_dir)

pipeline_options = PdfPipelineOptions(
    generate_picture_images=True,
    generate_page_images=False,
    do_table_structure=True,
    do_ocr=True,
    do_formula_enrichment=True,
    images_scale=2.0,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
print("  ✅ 创建成功")

print("\n[4] 执行 convert()...")
test_pdf = "/Volumes/ext/Master/papers/2403.20309v6(Instantsplat).pdf"
result = converter.convert(Path(test_pdf))
print(f"  ✅ 成功，items: {len(list(result.document.iterate_items()))}")

print("\n" + "=" * 60)
print("全部通过!")
print("=" * 60)
