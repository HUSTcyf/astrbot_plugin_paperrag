#!/usr/bin/env python3
"""
模型手动下载脚本

功能：
    1. 下载 Docling 模型（PDF 多模态提取）
    2. 下载 BGE-M3 模型（Unsloth 本地加载，用于稀疏权重 + 稠密向量 + 多向量）

使用方法:
    python download_models.py                  # 下载所有模型
    python download_models.py --docling        # 仅下载 Docling 模型
    python download_models.py --bge-m3        # 仅下载 BGE-M3 模型
    python download_models.py --all --force   # 强制重新下载所有模型
"""

import sys
import os
import argparse
from pathlib import Path

# 添加插件目录到 path
# __file__ = .../astrbot_plugin_paperrag/tools/download_models.py
plugin_dir = Path(__file__).parent.parent  # 插件根目录
sys.path.insert(0, str(plugin_dir))

# 配置本地模型目录
LOCAL_MODELS_DIR = plugin_dir / "models"
BGE_M3_DIR = LOCAL_MODELS_DIR / "bge-m3"

# 设置环境变量，使 huggingface_hub 缓存到本地
os.environ.setdefault("HF_HOME", str(LOCAL_MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(LOCAL_MODELS_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(LOCAL_MODELS_DIR))


def download_bge_m3(force: bool = False) -> bool:
    """
    从 Huggingface 下载 BGE-M3 模型（Unsloth 版本）

    Args:
        force: 是否强制重新下载

    Returns:
        是否成功
    """
    print("=" * 50)
    print("📦 下载 BGE-M3 模型（Unsloth 版本）")
    print("=" * 50)
    print(f"📁 下载目录: {BGE_M3_DIR}")
    print(f"💾 预计需要: ~2.3GB")
    print()

    try:
        from huggingface_hub import snapshot_download

        print("🚀 开始下载 BGE-M3 模型...")
        print("   Repo: unsloth/bge-m3")
        print("-" * 50)

        BGE_M3_DIR.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id="unsloth/bge-m3",
            local_dir=str(BGE_M3_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,  # 断点续传
            force_download=force,   # 强制重新下载
        )

        print("-" * 50)
        print("✅ BGE-M3 模型下载完成!")
        print()
        print(f"📁 模型保存位置: {BGE_M3_DIR}")

        # 列出下载的文件
        print("📦 已下载的文件:")
        total_size = 0
        for item in sorted(BGE_M3_DIR.iterdir()):
            size = item.stat().st_size
            total_size += size
            if item.is_file():
                size_mb = size / (1024 * 1024)
                print(f"   📄 {item.name} ({size_mb:.1f} MB)")

        total_mb = total_size / (1024 * 1024)
        print(f"   📊 总大小: {total_mb:.1f} MB")

        return True

    except Exception as e:
        print(f"❌ BGE-M3 下载失败: {e}")
        return False


def download_docling(force: bool = False) -> bool:
    """
    下载 Docling 模型

    Args:
        force: 是否强制重新下载

    Returns:
        是否成功
    """
    print("=" * 50)
    print("📦 下载 Docling 模型（PDF 多模态提取）")
    print("=" * 50)

    LOCAL_MODELS_DIR.mkdir(exist_ok=True)

    print(f"📁 下载目录: {LOCAL_MODELS_DIR}")
    print(f"💾 预计需要: ~2GB")
    print()

    # 设置 docling settings
    from docling.datamodel.settings import settings
    settings.cache_dir = LOCAL_MODELS_DIR
    settings.artifacts_path = LOCAL_MODELS_DIR

    from docling.utils.model_downloader import download_models

    try:
        print("🚀 开始下载 Docling 模型...")
        print("-" * 50)

        download_models(
            output_dir=LOCAL_MODELS_DIR,
            with_layout=True,           # 布局模型 (~500MB)
            with_tableformer=True,      # 表格结构模型
            with_tableformer_v2=False,  # TableFormerV2 (可选)
            with_code_formula=True,     # 公式识别模型 (~1GB)
            with_picture_classifier=True,# 图片分类器
            with_rapidocr=True,        # RapidOCR (默认, ~200MB)
            with_easyocr=False,         # EasyOCR (可选)
            with_smolvlm=False,        # SmolVLM (可选, 大模型)
            with_granitedocling=False,  # GraniteDocling (可选)
            with_granitedocling_mlx=False,
            with_smoldocling=False,
            with_smoldocling_mlx=False,
            with_granite_vision=False,
            with_granite_chart_extraction=False,
            progress=True,
            force=force,
        )

        print("-" * 50)
        print("✅ Docling 模型下载完成!")
        print()
        print(f"📁 模型保存位置: {LOCAL_MODELS_DIR}")

        # 列出下载的模型目录
        print("📦 已下载的模型:")
        for item in sorted(LOCAL_MODELS_DIR.iterdir()):
            if item.is_dir() and item.name != "bge-m3":  # 排除 bge-m3
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"   📂 {item.name} ({size_mb:.1f} MB)")
            elif item.is_file():
                size_kb = item.stat().st_size / 1024
                print(f"   📄 {item.name} ({size_kb:.1f} KB)")

        return True

    except Exception as e:
        print(f"❌ Docling 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="模型下载脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_models.py              # 下载所有模型
  python download_models.py --docling    # 仅下载 Docling 模型
  python download_models.py --bge-m3      # 仅下载 BGE-M3 模型
  python download_models.py --all --force # 强制重新下载所有模型
        """
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--docling",
        action="store_true",
        help="仅下载 Docling 模型"
    )
    group.add_argument(
        "--bge-m3",
        action="store_true",
        help="仅下载 BGE-M3 模型（Unsloth 版本）"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="下载所有模型（默认）"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新下载（覆盖已有模型）"
    )

    args = parser.parse_args()

    # 默认下载所有
    download_all = args.all or (not args.docling and not args.bge_m3)

    print()
    print("=" * 50)
    print("📦 PaperRAG 模型下载工具")
    print("=" * 50)
    print()

    success = True

    if args.docling or download_all:
        success = download_docling(force=args.force) and success

    if args.bge_m3 or download_all:
        success = download_bge_m3(force=args.force) and success

    print()
    print("=" * 50)
    if success:
        print("✅ 所有模型下载完成!")
    else:
        print("⚠️ 部分模型下载失败，请查看上方错误信息")
        sys.exit(1)


if __name__ == "__main__":
    # 如果需要使用镜像站，取消下面这行的注释:
    # import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    main()
