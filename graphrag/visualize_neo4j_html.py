# visualize_neo4j_html.py
# Neo4j 数据可视化，生成交互式 HTML（使用 pyvis）

from pyvis.network import Network
from neo4j import GraphDatabase
from pathlib import Path
import json

def _load_neo4j_password() -> str:
    """从插件配置文件读取 Neo4j 密码"""
    config_paths = [
        Path(__file__).parent.parent / "data" / "config" / "astrbot_plugin_paperrag_config.json",
        Path.home() / "AstrBot" / "data" / "config" / "astrbot_plugin_paperrag_config.json",
    ]
    for p in config_paths:
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                cfg = json.load(f)
                pw = cfg.get("graph_rag", {}).get("neo4j_password", "")
                if pw:
                    return pw
    raise RuntimeError("无法从配置文件中读取 neo4j_password，请检查 graph_rag.neo4j_password 配置")

# ============ 配置 ============
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = _load_neo4j_password()

# 导出数量限制
LIMIT_NODES = 10000   # 节点数量限制
LIMIT_EDGES = 10000   # 边数量限制
# ============ 配置 END ============

OUTPUT_FILE = Path(__file__).parent / "graph_visualization.html"

# 节点颜色配置（与 closed-set 实体类型一致）
NODE_COLORS = {
    "Method": "#3498DB",
    "Model": "#2ECC71",
    "Task": "#E74C3C",
    "Dataset": "#F39C12",
    "Metric": "#9B59B6",
    "Component": "#1ABC9C",
    "Limitation": "#E67E22",
    "Application": "#2C3E50",
    "Baseline": "#95A5A6",
    "Figure": "#FF6B6B",
    "Table": "#FFD93D",
}


def visualize_html():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # 创建网络
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",  # type: ignore[arg-type]
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
        "Method": 25,
        "Model": 30,
        "Task": 22,
        "Dataset": 20,
        "Metric": 15,
        "Component": 18,
        "Limitation": 15,
        "Application": 20,
        "Baseline": 18,
        "Figure": 12,
        "Table": 12,
    }

    with driver.session() as session:
        # 查询节点
        print("查询节点...")
        result = session.run("""
            MATCH (n)
            RETURN id(n) as nid,
                   labels(n)[0] as type,
                   coalesce(n.title, n.name, n.ref_title, 'unknown') as label
            LIMIT """ + str(LIMIT_NODES))  # type: ignore[arg-type]

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
        result = session.run("""
            MATCH (n)-[r]->(m)
            RETURN id(n) as source, id(m) as target, type(r) as rel
            LIMIT """ + str(LIMIT_EDGES))  # type: ignore[arg-type]

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
