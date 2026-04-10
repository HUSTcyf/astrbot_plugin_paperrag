#!/usr/bin/env python3
"""
检查并统计 Milvus 数据库中的重复 chunk
"""

import sys
from pathlib import Path

# 抑制警告
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# 添加插件路径
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

try:
    from pymilvus import Collection, connections, utility
except ImportError:
    print("❌ 未安装 pymilvus")
    print("安装命令: pip install pymilvus[milvus_lite]")
    sys.exit(1)


def check_duplicates():
    """检查数据库中的重复文档"""

    db_path = "./data/milvus_papers.db"
    collection_name = "paper_embeddings"
    alias = "paperrag_check"

    print("=" * 60)
    print("📚 Milvus 数据库重复检查")
    print("=" * 60)
    print()

    # 连接 Milvus
    print(f"🔗 连接到数据库: {db_path}")
    try:
        connections.connect(
            alias=alias,
            uri=db_path
        )
        print("✅ 连接成功")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n提示: 请先安装 milvus-lite")
        print("安装命令: pip install pymilvus[milvus_lite]")
        return

    try:
        # 检查集合是否存在
        if not utility.has_collection(collection_name, using=alias):
            print(f"❌ 集合 '{collection_name}' 不存在")
            return

        # 加载集合
        collection = Collection(collection_name, using=alias)
        collection.load()

        # 获取总数
        total_count = collection.num_entities
        print(f"📊 总文档数: {total_count}\n")

        # 分批查询所有文档
        # Milvus 限制: topk + offset <= 16384
        # 使用较小的批次确保不超过限制
        print("🔍 扫描数据库中的重复...")

        batch_size = 500  # 减小批次大小
        offset = 0
        max_offset = 16000  # 最大 offset，确保 offset + batch_size <= 16384

        # 记录每个 (file_name, chunk_index) 对应的次数
        chunk_map = {}  # (file_name, chunk_index) -> count

        while offset < total_count:
            # 如果 offset 接近限制，使用更小的批次
            current_batch_size = min(batch_size, 16384 - offset)

            if current_batch_size <= 0:
                print(f"\n⚠️  已达到 Milvus 查询限制 (offset={offset})")
                print(f"  已扫描: {offset}/{total_count}")
                print(f"  剩余 {total_count - offset} 条记录未扫描")
                break

            results = collection.query(
                expr="",
                output_fields=["metadata"],
                limit=current_batch_size,
                offset=offset
            )

            for doc in results:
                metadata = doc.get("metadata", {})
                file_name = metadata.get("file_name", "unknown")
                chunk_index = metadata.get("chunk_index", -1)

                key = (file_name, chunk_index)
                if key not in chunk_map:
                    chunk_map[key] = 0
                chunk_map[key] += 1

            offset += len(results)
            print(f"  已扫描: {min(offset, total_count)}/{total_count}", end="\r")

        print()
        print()

        # 统计重复
        duplicates = {k: v for k, v in chunk_map.items() if v > 1}

        if not duplicates:
            print("✅ 未发现重复数据")
            return

        print(f"❌ 发现 {len(duplicates)} 个重复的 chunk:\n")

        # 统计信息
        total_dup_count = sum(v - 1 for v in duplicates.values())
        print(f"📊 统计:")
        print(f"  - 重复的 chunk 类型: {len(duplicates)}")
        print(f"  - 多余的记录数: {total_dup_count}")
        print(f"  - 去重后剩余: {total_count - total_dup_count}")
        print()

        # 显示前 20 个重复
        print("📋 重复详情（前 20 个）:")
        for i, ((file_name, chunk_index), count) in enumerate(list(duplicates.items())[:20]):
            print(f"  [{i+1}] {file_name} chunk #{chunk_index}: {count} 次")

        if len(duplicates) > 20:
            print(f"  ... 还有 {len(duplicates) - 20} 个重复项")

        print()
        print("=" * 60)
        print("\n💡 清理建议:")
        print("   在 AstrBot 中执行以下命令:")
        print("   1. /paper clear confirm")
        print("   2. /paper add")
        print("   这样可以彻底清除重复数据")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 断开连接
        try:
            connections.disconnect(alias)
            print("\n✅ 已断开连接")
        except:
            pass


if __name__ == "__main__":
    check_duplicates()
