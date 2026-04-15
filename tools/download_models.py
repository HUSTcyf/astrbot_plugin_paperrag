#!/usr/bin/env python3
"""
Docling 模型手动下载脚本

使用方法:
    python download_models.py

模型将下载到插件的 models/ 子目录
需要约 1.5-2GB 磁盘空间
"""

import sys
import os
from pathlib import Path

# 添加插件目录到 path
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

# 配置本地模型目录（与 DoclingExtractor 保持一致）
LOCAL_MODELS_DIR = plugin_dir / "models"

# 设置环境变量，使 huggingface_hub 缓存到本地
os.environ.setdefault("HF_HOME", str(LOCAL_MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(LOCAL_MODELS_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(LOCAL_MODELS_DIR))

# 设置 docling settings
from docling.datamodel.settings import settings
settings.cache_dir = LOCAL_MODELS_DIR
settings.artifacts_path = LOCAL_MODELS_DIR

from docling.utils.model_downloader import download_models


def main():
    LOCAL_MODELS_DIR.mkdir(exist_ok=True)

    print(f"📁 模型下载目录: {LOCAL_MODELS_DIR}")
    print(f"💾 预计需要: ~2GB")
    print()

    print("🚀 开始下载 docling 模型...")
    print("-" * 50)

    try:
        download_models(
            output_dir=LOCAL_MODELS_DIR,
            with_layout=True,           # 布局模型 (~500MB)
            with_tableformer=True,      # 表格结构模型
            with_tableformer_v2=False,  # TableFormerV2 (可选)
            with_code_formula=True,     # 公式识别模型 (~1GB)
            with_picture_classifier=True,# 图片分类器
            with_rapidocr=True,         # RapidOCR (默认, ~200MB)
            with_easyocr=False,          # EasyOCR (可选)
            with_smolvlm=False,         # SmolVLM (可选, 大模型)
            with_granitedocling=False,  # GraniteDocling (可选)
            with_granitedocling_mlx=False,
            with_smoldocling=False,
            with_smoldocling_mlx=False,
            with_granite_vision=False,
            with_granite_chart_extraction=False,
            progress=True,
            force=False,  # True: 强制重新下载
        )

        print("-" * 50)
        print("✅ 模型下载完成!")
        print()
        print(f"📁 模型保存位置: {LOCAL_MODELS_DIR}")
        print()

        # 列出下载的模型目录
        print("📦 已下载的模型:")
        for item in sorted(LOCAL_MODELS_DIR.iterdir()):
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"   📂 {item.name} ({size_mb:.1f} MB)")
            else:
                size_kb = item.stat().st_size / 1024
                print(f"   📄 {item.name} ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 如果需要使用镜像站，取消下面这行的注释:
    # import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    main()
