"""
从 Milvus Lite 构建 Neo4j 知识图谱

用法:
    python build_graph_from_milvus.py

依赖:
    pip install llama-index-graph-stores-neo4j neo4j llama-index-llms-openai
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加插件目录到路径
plugin_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(plugin_dir))

# 设置工作目录
os.chdir(plugin_dir)


# ============================================================================
# 步骤 1: 从 Milvus 提取 chunks
# ============================================================================

async def extract_chunks_from_milvus(
    milvus_path: str = "./data/milvus_papers.db",
    collection_name: str = "paper_embeddings",
) -> List[Dict[str, Any]]:
    """从 Milvus Lite 提取全量 chunk 文本"""
    print(f"\n{'='*60}")
    print("📤 步骤 1: 从 Milvus 提取 chunks")
    print("=" * 60)

    try:
        from hybrid_index import HybridIndexManager

        manager = HybridIndexManager(
            milvus_uri=milvus_path,
            collection_name=collection_name,
            embed_dim=1024,
            hybrid_search=False,
        )

        chunks = await manager.get_all_chunks()
        print(f"✅ 提取完成: {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        raise


# ============================================================================
# 步骤 2: 转换为 LlamaIndex Document
# ============================================================================

def chunks_to_documents(chunks: List[Dict[str, Any]]) -> List[Any]:
    """将 chunks 转换为 LlamaIndex Document"""
    print(f"\n{'='*60}")
    print("📄 步骤 2: 转换为 LlamaIndex Document")
    print("=" * 60)

    from llama_index.core import Document
    from collections import defaultdict

    # 按 paper_id 分组
    papers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        pid = chunk.get("paper_id", "unknown")
        if pid and pid != "unknown":
            papers[pid].append(chunk)

    # 构建 Document
    documents = []
    for paper_id, paper_chunks in papers.items():
        # 按 chunk id 排序
        paper_chunks.sort(key=lambda x: x.get("id", 0))

        # 合并同一论文的 chunks
        combined_text = "\n\n".join(c.get("text", "") for c in paper_chunks if c.get("text"))

        if not combined_text.strip():
            continue

        doc = Document(
            text=combined_text,
            metadata={
                "paper_id": paper_id,
                "chunk_count": len(paper_chunks),
                "source": "milvus",
            }
        )
        documents.append(doc)

    print(f"✅ 构建 {len(documents)} 篇论文 Document")
    return documents


# ============================================================================
# 步骤 3: 构建 Neo4j 知识图谱
# ============================================================================

def build_neo4j_graph(
    documents: List[Any],
    neo4j_config: Optional[dict] = None,
) -> None:
    """使用 LlamaIndex 构建 Neo4j 知识图谱"""
    print(f"\n{'='*60}")
    print("🏗️ 步骤 3: 构建 Neo4j 知识图谱")
    print("=" * 60)

    neo4j_config = neo4j_config or {
        "url": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "neo4j_M73770",
    }

    try:
        from llama_index.graph_stores.neo4j import Neo4jGraphStore
        from llama_index.core import PropertyGraphIndex
        import llama_cpp

        # 获取模型路径（参照 llama_cpp_vlm_provider.py）
        plugin_dir = Path(__file__).parent.resolve()
        model_path = os.environ.get(
            "PAPERRAG_GGUF_MODEL_PATH",
            str(plugin_dir / "models" / "Qwen3.5-9B-GGUF" / "Qwen3.5-9B-UD-Q4_K_XL.gguf")
        )

        print(f"🤖 使用 LlamaCpp 加载模型: {model_path}")

        # 直接使用 llama_cpp.Llama（不需要 mmproj，用于纯文本）
        llama_llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=4096,  # 上下文窗口
            n_gpu_layers=99,  # GPU 加速层数
            n_batch=32,  # 批处理大小
            verbose=False,
        )

        class LlamaCppWrapper:
            """LlamaCpp 包装器，适配 llama_index 接口"""
            def __init__(self, llama_instance):
                self._llama = llama_instance
                self.model_name = "LlamaCpp-Qwen2.5"

            def complete(self, prompt: str, **kwargs) -> Any:
                """同步 complete 接口"""
                import json
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                result = self._llama.create_chat_completion(
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 4096),
                )
                return result["choices"][0]["message"]["content"]

        llm = LlamaCppWrapper(llama_llm)
        print(f"✅ LLM 初始化完成: {llm.model_name}")

        print(f"🤖 使用 LLM: {llm}")

        # 连接 Neo4j
        print(f"🔗 连接 Neo4j: {neo4j_config['url']}")
        graph_store = Neo4jGraphStore(
            url=neo4j_config["url"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            database="neo4j",
        )

        # 构建索引
        print("📊 开始构建知识图谱（这可能需要几分钟）...")
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            llm=llm,
            show_progress=True,
        )

        print("✅ 知识图谱构建完成！")

        # 验证
        verify_neo4j(graph_store)

    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install llama-index-graph-stores-neo4j neo4j")
        raise


def verify_neo4j(graph_store) -> None:
    """验证 Neo4j 中的数据"""
    print(f"\n{'='*60}")
    print("🔍 验证 Neo4j 数据")
    print("=" * 60)

    try:
        # 使用 Cypher 查询验证
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            graph_store.url,
            auth=(graph_store.username, graph_store.password)
        )

        with driver.session() as session:
            # 统计实体数
            result = session.run("MATCH (n) RETURN count(n) as count")
            record = result.single()
            entity_count = record["count"] if record else 0

            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            record = result.single()
            relation_count = record["count"] if record else 0

            print(f"✅ 实体数量: {entity_count}")
            print(f"✅ 关系数量: {relation_count}")

            # 显示部分实体
            result = session.run("MATCH (n) RETURN n.id, n.type LIMIT 10")
            print("\n📋 部分实体:")
            for record in result:
                print(f"   - {record['n.id']} ({record['n.type']})")

        driver.close()

    except Exception as e:
        print(f"⚠️ 验证查询失败: {e}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="从 Milvus 构建 Neo4j 知识图谱")
    parser.add_argument(
        "--milvus-path",
        default="./data/milvus_papers.db",
        help="Milvus Lite 数据库路径"
    )
    parser.add_argument(
        "--collection",
        default="paper_embeddings",
        help="Milvus 集合名称"
    )
    parser.add_argument(
        "--neo4j-url",
        default="bolt://localhost:7687",
        help="Neo4j 连接 URL"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j 用户名"
    )
    parser.add_argument(
        "--neo4j-password",
        default="",
        help="Neo4j 密码（如果为空，从环境变量 NEO4J_PASSWORD 读取）"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="仅提取 chunks，不构建图谱"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式：不加载 LLM，仅测试流程是否能走通"
    )

    args = parser.parse_args()

    # 获取密码
    neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD", "neo4j")

    neo4j_config = {
        "url": args.neo4j_url,
        "username": args.neo4j_user,
        "password": neo4j_password,
    }

    # 事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 步骤 1: 提取 chunks
        chunks = loop.run_until_complete(
            extract_chunks_from_milvus(args.milvus_path, args.collection)
        )

        # 保存 chunks 到文件（可选）
        chunks_file = "results/milvus_chunks.json"
        Path(chunks_file).parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"💾 Chunks 已保存到: {chunks_file}")

        if args.extract_only or args.dry_run:
            mode = "干跑模式" if args.dry_run else "仅提取模式"
            print(f"\n✅ {mode}，完成")
            print(f"   Chunks 数量: {len(chunks)}")
            print(f"   Document 数量: {len(chunks_to_documents(chunks))}")
            return

        # 步骤 2: 转换为 Document
        documents = chunks_to_documents(chunks)

        if not documents:
            print("❌ 没有可用的文档")
            return

        # 步骤 3: 构建 Neo4j 图谱
        build_neo4j_graph(documents, neo4j_config)

    finally:
        loop.close()

    print("\n" + "="*60)
    print("🎉 全部完成！")
    print("=" * 60)
    print("\n可通过以下方式查看图谱:")
    print("  1. Web UI: http://localhost:7474")
    print("  2. 命令行: cypher-shell -u neo4j -p <password>")


if __name__ == "__main__":
    main()
