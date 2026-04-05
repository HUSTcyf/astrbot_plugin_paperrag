# visualize_neo4j_html.py
# Neo4j 数据可视化，生成交互式 HTML（使用 pyvis）

from pyvis.network import Network
from neo4j import GraphDatabase
from pathlib import Path

# ============ 配置 ============
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j_M73770"  # TODO: 修改为你的密码

# 导出数量限制
LIMIT_NODES = 10000   # 节点数量限制
LIMIT_EDGES = 10000   # 边数量限制
# ============ 配置 END ============

OUTPUT_FILE = Path(__file__).parent / "graph_visualization.html"

# 节点颜色配置（与 graph_style.grass 一致）
NODE_COLORS = {
    "Paper": "#3498DB",
    "Author": "#E74C3C",
    "Reference": "#F39C12",
    "Concept": "#9B59B6",
    "Institution": "#1ABC9C",
    "Figure": "#2ECC71",
    "Table": "#E67E22",
    "Section": "#34495E",
    "Chunk": "#95A5A6",
}


def visualize_html():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # 创建网络
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
        cdn_resources="remote"
    )
    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.001,
        damping=0.09
    )

    # 节点尺寸配置
    NODE_SIZES = {
        "Paper": 30,
        "Author": 20,
        "Reference": 20,
        "Concept": 15,
        "Institution": 18,
        "Figure": 15,
        "Table": 15,
        "Section": 12,
        "Chunk": 8,
    }

    with driver.session() as session:
        # 查询节点
        print("查询节点...")
        result = session.run(f"""
            MATCH (n)
            RETURN id(n) as nid,
                   labels(n)[0] as type,
                   coalesce(n.title, n.name, n.ref_title, 'unknown') as label
            LIMIT {LIMIT_NODES}
        """)

        node_map = {}
        for record in result:
            nid = record["nid"]
            node_type = record["type"] or "Unknown"
            label = record["label"] or "unknown"
            label_short = label[:40] + "..." if len(label) > 40 else label

            color = NODE_COLORS.get(node_type, "#95A5A6")
            size = NODE_SIZES.get(node_type, 15)

            net.add_node(
                nid,
                label=label_short,
                title=f"[{node_type}]\n{label}",
                color=color,
                size=size
            )
            node_map[nid] = label

        print(f"添加了 {len(node_map)} 个节点")

        # 查询边
        print("查询关系...")
        result = session.run(f"""
            MATCH (n)-[r]->(m)
            RETURN id(n) as source, id(m) as target, type(r) as rel
            LIMIT {LIMIT_EDGES}
        """)

        edge_count = 0
        for record in result:
            source = record["source"]
            target = record["target"]
            rel = record["rel"]
            if source in node_map and target in node_map:
                net.add_edge(
                    source,
                    target,
                    title=rel,
                    label=rel,
                    color="rgba(255,255,255,0.3)"
                )
                edge_count += 1

        print(f"添加了 {edge_count} 条关系")

    driver.close()

    # 保存 HTML
    net.save_graph(str(OUTPUT_FILE))
    print(f"\n可视化已保存: {OUTPUT_FILE}")
    print(f"用浏览器打开: file://{OUTPUT_FILE.absolute()}")


if __name__ == "__main__":
    visualize_html()
