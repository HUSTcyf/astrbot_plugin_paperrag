#!/usr/bin/env python3
"""
Paper RAG 插件完整流程测试
测试 PDF → 多模态提取 → 分块 → 向量化 的完整流程
"""

import sys
import os
from pathlib import Path


def print_header(title: str):
    """打印标题"""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def print_section(title: str):
    """打印小节标题"""
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


def test_dependencies():
    """测试1: 依赖检查"""
    print_header("📦 测试1: 依赖检查")

    dependencies = {
        'pymilvus': 'Milvus 向量数据库',
        'fitz': 'PyMuPDF (PDF解析)',
        'pdfplumber': '表格提取',
        'docx': 'Word文档支持',
        'PIL': 'Pillow (图像处理)'
    }

    optional_deps = {
        'transformers': 'HuggingFace (视觉编码)',
        'torch': 'PyTorch (深度学习)'
    }

    missing_required = []
    missing_optional = []

    print("🔍 核心依赖:")
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {module:15} - {description}")
        except ImportError:
            print(f"  ❌ {module:15} - {description}")
            missing_required.append(module)

    print()
    print("🔍 可选依赖 (多模态):")
    for module, description in optional_deps.items():
        try:
            __import__(module)
            print(f"  ✅ {module:15} - {description}")
        except ImportError:
            print(f"  ❌ {module:15} - {description}")
            missing_optional.append(module)

    print()
    if missing_required:
        print("❌ 缺少核心依赖，请安装:")
        print("   pip install pymilvus PyMuPDF pdfplumber python-docx pillow")
        return False
    else:
        print("✅ 所有核心依赖已安装")

    if missing_optional:
        print("⚠️  缺少可选依赖（多模态功能将自动降级）:")
        print("   pip install transformers torch")
    else:
        print("✅ 所有可选依赖已安装（完整多模态支持）")

    return True


def test_pdf_extraction(pdf_path: str):
    """测试2: PDF基础提取"""
    print_header("📄 测试2: PDF基础提取")

    try:
        import fitz
    except ImportError:
        print("❌ PyMuPDF 未安装")
        return False

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ 文件不存在: {pdf_path}")
        return False

    print(f"📁 文件: {pdf_path.name}")
    print(f"📏 大小: {pdf_path.stat().st_size / 1024:.1f} KB")

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        print(f"📄 总页数: {total_pages}")

        pages_with_text = 0
        total_chars = 0
        total_images = 0

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            text_length = len(text.strip())

            if text_length > 0:
                pages_with_text += 1
                total_chars += text_length

                # 显示前100个字符
                if page_num <= 3:  # 只显示前3页
                    preview = text.strip()[:100].replace('\n', ' ')
                    print(f"  页 {page_num}: {text_length} 字符")
                    print(f"    预览: {preview}...")

            # 统计图片
            images = page.get_images()
            if images:
                total_images += len(images)

        doc.close()

        print()
        print("📊 提取统计:")
        print(f"  • 有文本的页数: {pages_with_text}/{total_pages}")
        print(f"  • 总字符数: {total_chars}")
        print(f"  • 总图片数: {total_images}")
        print(f"  • 文本密度: {total_chars / total_pages:.1f} 字符/页")

        if pages_with_text == 0:
            print()
            print("❌ 诊断: PDF可能是扫描版（无文本层）")
            return False
        elif total_chars < 100:
            print()
            print("⚠️  诊断: 文本量很少")
            return False
        else:
            print()
            print("✅ PDF包含可提取的文本")
            return True

    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return False


def test_multimodal_extraction(pdf_path: str):
    """测试3: 多模态提取"""
    print_header("🎨 测试3: 多模态提取")

    try:
        from multimodal_extractor import MultimodalPDFExtractor
        from semantic_chunker import MULTIMODAL_AVAILABLE
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

    if not MULTIMODAL_AVAILABLE:
        print("⚠️  多模态模块不可用")
        return False

    print("🔍 初始化多模态提取器...")

    try:
        extractor = MultimodalPDFExtractor(
            extract_images=True,
            extract_tables=True,
            extract_formulas=True,
            fallback_to_text=True
        )

        print(f"✅ 提取器可用: {extractor.available}")
        print(f"   图片提取: {'✅' if extractor.extract_images else '❌'}")
        print(f"   表格提取: {'✅' if extractor.extract_tables else '❌'}")
        print(f"   公式提取: {'✅' if extractor.extract_formulas else '❌'}")

        print()
        print("📖 开始提取...")

        extracted = extractor.extract(pdf_path)

        print()
        print("✅ 提取完成!")
        print()
        print("📊 提取结果:")
        print(f"  • 图片数量: {len(extracted.images)}")
        print(f"  • 表格数量: {len(extracted.tables)}")
        print(f"  • 公式数量: {len(extracted.formulas)}")
        print(f"  • 文本长度: {len(extracted.text)} 字符")

        # 显示图片详情
        if extracted.images:
            print()
            print("🖼️  提取的图片（前3个）:")
            for i, img in enumerate(extracted.images[:3], 1):
                size = f"{img.image.size[0]}x{img.image.size[1]}" if img.image else "N/A"
                print(f"  [{i}] 页 {img.page_number}, 大小: {size}")
                if img.caption:
                    print(f"      图注: {img.caption}")

        # 显示表格详情
        if extracted.tables:
            print()
            print("📋 提取的表格（前2个）:")
            for i, table in enumerate(extracted.tables[:2], 1):
                rows = len(table.data) if table.data else 0
                cols = len(table.data[0]) if table.data and len(table.data) > 0 else 0
                print(f"  [{i}] 页 {table.page_number}, {rows}行 x {cols}列")
                if table.caption:
                    print(f"      表注: {table.caption}")
                if table.markdown:
                    preview = table.markdown[:100].replace('\n', ' ')
                    print(f"      预览: {preview}...")

        return True

    except Exception as e:
        print(f"❌ 提取失败: {e}")
        import traceback
        print(f"❌ 错误堆栈: {traceback.format_exc()}")
        return False


def test_semantic_chunking(pdf_path: str):
    """测试4: 语义分块"""
    print_header("🧩 测试4: 语义分块")

    try:
        from semantic_chunker import PDFParserAdvanced, SemanticChunker
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

    print("🔍 初始化解析器和分块器...")

    try:
        parser = PDFParserAdvanced(
            enable_multimodal=True
        )
        chunker = SemanticChunker(
            chunk_size=512,
            overlap=0,
            min_chunk_size=100
        )

        print("✅ 解析器和分块器初始化成功")
        print()
        print("📖 开始解析和分块...")

        chunks_dict = parser.parse_and_chunk(pdf_path, chunker)

        print()
        print("✅ 分块完成!")
        print()
        print("📊 分块统计:")
        print(f"  • 总块数: {len(chunks_dict)}")

        # 统计块类型
        chunk_types = {}
        for chunk in chunks_dict:
            chunk_type = chunk["metadata"].get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        print()
        print("🏷️  块类型分布:")
        for chunk_type, count in sorted(chunk_types.items()):
            print(f"  • {chunk_type}: {count}")

        # 显示块示例
        print()
        print("📝 分块示例（前3个）:")
        for i, chunk in enumerate(chunks_dict[:3], 1):
            text = chunk["text"][:150].replace('\n', ' ')
            metadata = chunk["metadata"]
            chunk_type = metadata.get("chunk_type", "unknown")
            print(f"  [{i}] 类型: {chunk_type}, 长度: {len(chunk['text'])} 字符")
            print(f"      内容: {text}...")

        return True

    except Exception as e:
        print(f"❌ 分块失败: {e}")
        import traceback
        print(f"❌ 错误堆栈: {traceback.format_exc()}")
        return False


def test_embedding():
    """测试5: 向量化功能"""
    print_header("🔢 测试5: 向量化功能")

    print("⚠️  注意: 此测试需要配置 AstrBot Embedding Provider")
    print("💡 如果未配置，将跳过此测试")
    print()

    # 检查是否可以导入AstrBot
    try:
        from astrbot.api import Context
        print("✅ AstrBot API 可用")
        print()
        print("💡 向量化测试需要在 AstrBot 环境中运行")
        print("   使用 /paper add 命令进行完整测试")
        return True
    except ImportError:
        print("⚠️  不在 AstrBot 环境中，跳过向量化测试")
        print("💡 请使用 AstrBot 的 /paper add 命令测试完整流程")
        return False


def main():
    """主测试函数"""
    if len(sys.argv) < 2:
        print("用法: python test_pdf.py <pdf_file_path>")
        print()
        print("示例:")
        print("  python test_pdf.py papers/example.pdf")
        print()
        print("测试流程:")
        print("  1. 依赖检查")
        print("  2. PDF基础提取")
        print("  3. 多模态提取")
        print("  4. 语义分块")
        print("  5. 向量化功能（需要AstrBot环境）")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print()
    print("🚀 Paper RAG 插件 - 完整流程测试")
    print(f"📁 测试文件: {pdf_path}")
    print()

    results = []

    # 测试1: 依赖检查
    results.append(("依赖检查", test_dependencies()))

    if not results[0][1]:
        print()
        print("❌ 依赖检查失败，请先安装必要的依赖")
        sys.exit(1)

    # 测试2: PDF基础提取
    results.append(("PDF提取", test_pdf_extraction(pdf_path)))

    if not results[1][1]:
        print()
        print("❌ PDF提取失败，请检查文件是否存在")
        sys.exit(1)

    # 测试3: 多模态提取
    results.append(("多模态提取", test_multimodal_extraction(pdf_path)))

    # 测试4: 语义分块
    results.append(("语义分块", test_semantic_chunking(pdf_path)))

    # 测试5: 向量化
    results.append(("向量化", test_embedding()))

    # 总结
    print_header("📊 测试总结")

    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name:15} {status}")

    print()
    passed = sum(1 for _, success in results if success)
    total = len(results)

    if passed == total:
        print(f"🎉 所有测试通过! ({passed}/{total})")
        print()
        print("💡 下一步:")
        print("  1. 在 AstrBot 中配置 Embedding Provider")
        print("  2. 使用 /paper add 命令添加文档")
        print("  3. 使用 /paper search 命令测试检索")
    else:
        print(f"⚠️  部分测试失败 ({passed}/{total} 通过)")
        print()
        print("💡 请查看上面的错误信息，修复问题后重试")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
