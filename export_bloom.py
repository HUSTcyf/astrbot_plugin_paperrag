# export_bloom.py
# Neo4j 数据导出脚本，用于 Neo4j Bloom 可视化（CSV格式）

from neo4j import GraphDatabase
import csv
from pathlib import Path

# ============ 配置 ============
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j_M73770"  # TODO: 修改为你的密码

# 导出数量限制
LIMIT_NODES = 10000   # 节点数量限制
LIMIT_EDGES = 10000   # 边数量限制
# ============ 配置 END ============

OUTPUT_DIR = Path(__file__).parent / "bloom_export"
NODES_FILE = OUTPUT_DIR / "nodes.csv"
EDGES_FILE = OUTPUT_DIR / "edges.csv"

# 节点颜色配置（简化版本）
NODE_COLORS = {
    "Paper": "#3498DB",
    "Reference": "#F39C12",
    "Chunk": "#95A5A6",
    "Model/Architecture": "#3498DB",
    "Method/Technique": "#9B59B6",
    "Task": "#E67E22",
    "Metric": "#F39C12",
    "Dataset": "#1ABC9C",
    "Author/Organization": "#E74C3C",
    "Venue": "#8E44AD",
    "Optimizer/Algorithm": "#27AE60",
    "Framework/Library": "#16A085",
    "ImagePath": "#27AE60",
    "Figure": "#2ECC71",
    "Other": "#BDC3C7",
}


def get_node_color(node_type: str) -> str:
    """获取节点类型对应的颜色"""
    if node_type in NODE_COLORS:
        return NODE_COLORS[node_type]
    # 默认颜色：基于类型关键词匹配
    if "Figure" in node_type or "Table" in node_type:
        return "#2ECC71"
    if "Author" in node_type or "Organization" in node_type:
        return "#E74C3C"
    if "Paper" in node_type or "Reference" in node_type:
        return "#3498DB"
    if "Concept" in node_type or "Method" in node_type or "Technique" in node_type:
        return "#9B59B6"
    if "Dataset" in node_type or "Metric" in node_type:
        return "#1ABC9C"
    return "#95A5A6"


def get_node_type(labels: list) -> str:
    """根据标签获取简化的节点类型"""
    # 主要类型列表（按优先级）
    MAIN_TYPES = {
        "Chunk": ["Chunk"],
        "Paper": ["Paper"],
        "Reference": ["Reference"],
        "Model/Architecture": ["Model/Architecture"],
        "Method/Technique": ["Method/Technique"],
        "Task": ["Task"],
        "Metric": ["Metric"],
        "Dataset": ["Dataset"],
        "Author/Organization": ["Author/Organization", "Author"],
        "Venue": ["Venue"],
        "Optimizer/Algorithm": ["Optimizer/Algorithm"],
        "Framework/Library": ["Framework/Library"],
        "ImagePath": ["ImagePath"],
        "Figure": ["Figure:photo", "Figure:diagram", "Figure:chart", "Figure:table", "Figure:graph", "Figure"],
    }

    for main_type, type_aliases in MAIN_TYPES.items():
        for alias in type_aliases:
            if alias in labels:
                return main_type

    return "Other"


def get_display_label(label: str, max_len: int = 50) -> str:
    """截断标签，适合显示"""
    if not label:
        return "unknown"
    if len(label) <= max_len:
        return label
    return label[:max_len] + "..."


def export_for_bloom():
    OUTPUT_DIR.mkdir(exist_ok=True)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    nodes = []
    node_map = {}  # id -> node 映射，用于验证边

    with driver.session() as session:
        # 查询节点
        print("查询节点...")
        result = session.run(f"""
            MATCH (n)
            RETURN id(n) as nid,
                   labels(n) as labels,
                   coalesce(n.title, n.name, n.ref_title, n.label, 'unknown') as label,
                   properties(n) as props
            LIMIT {LIMIT_NODES}
        """)

        for record in result:
            nid = record["nid"]
            labels = record["labels"] or []
            props = record["props"] or {}

            # 使用简化的类型分类
            node_type = get_node_type(labels)
            label = get_display_label(record["label"])

            # 收集关键属性用于详情展示
            title = props.get("title") or props.get("name") or props.get("ref_title") or label
            paper_id = props.get("paper_id", "")

            nodes.append({
                "id": str(nid),
                "type": node_type,
                "label": label,
                "color": get_node_color(node_type),
                "title": str(title)[:200],
                "paper_id": str(paper_id),
            })
            node_map[nid] = nodes[-1]

        print(f"查询到 {len(nodes)} 个节点")

        # 查询边
        print("查询关系...")
        result = session.run(f"""
            MATCH (n)-[r]->(m)
            RETURN id(n) as source, id(m) as target, type(r) as rel_type
            LIMIT {LIMIT_EDGES}
        """)

        edges = []
        for record in result:
            source = record["source"]
            target = record["target"]
            rel_type = record["rel_type"] or "RELATED_TO"

            # 只添加两端节点都存在的边
            if source in node_map and target in node_map:
                edges.append({
                    "source": str(source),
                    "target": str(target),
                    "type": rel_type,
                })

        print(f"查询到 {len(edges)} 条关系")

    driver.close()

    # 保存节点 CSV
    with open(NODES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "type", "label", "color", "title", "paper_id"])
        writer.writeheader()
        writer.writerows(nodes)

    # 保存边 CSV
    with open(EDGES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "type"])
        writer.writeheader()
        writer.writerows(edges)

    print(f"\n导出完成:")
    print(f"  节点: {NODES_FILE}")
    print(f"  关系: {EDGES_FILE}")
    print(f"\n节点类型分布:")
    type_counts = {}
    for n in nodes:
        t = n.get("type", "Other")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    export_for_bloom()
