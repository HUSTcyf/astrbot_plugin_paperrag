# Neo4j 知识图谱常用 Cypher 查询

> 针对 PaperRAG 构建的学术论文知识图谱，基于实际数据结构整理。

---

## 1. 基本统计

```cypher
-- 节点总数
MATCH (n) RETURN count(n) AS node_count;

-- 关系总数
MATCH ()-[r]->() RETURN count(r) AS rel_count;

-- 按实体类型分组统计（降序）
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS count
ORDER BY count DESC;

-- 按关系类型分组统计（降序，Top 20）
MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count
ORDER BY count DESC LIMIT 20;
```

---

## 2. 实体查询

### 2.1 当前图谱支持的实体类型

| 标签 | 说明 |
|---|---|
| `Model/Architecture` | 模型/架构（如 BERT、GPT） |
| `Method/Technique` | 方法/技术（如 Attention、Fine-tuning） |
| `Dataset` | 数据集（如 ImageNet、COCO） |
| `Metric` | 评估指标（如 Accuracy、BLEU） |
| `Task` | 任务类型（如 Text Classification） |
| `Author/Organization` | 作者/机构 |
| `Venue` | 发表场所（如 NeurIPS、arXiv） |
| `Framework/Library` | 框架/库（如 PyTorch、TensorFlow） |
| `Optimizer/Algorithm` | 优化器/算法（如 Adam、SGD） |
| `Hyperparameter` | 超参数（如 Learning Rate） |
| `Figure:chart` | 图表：柱状图/折线图等 |
| `Figure:diagram` | 图表：架构图/流程图 |
| `Figure:graph` | 图表：曲线图 |
| `Figure:photo` | 图表：照片/实物图 |
| `Figure:table` | 图表：表格 |
| `Chunk` | 文本分块 |
| `ImagePath` | 图片路径 |
| `Entity` | 通用实体 |
| `Other` | 其他 |

### 2.2 按类型查询

```cypher
-- 查询所有模型
MATCH (n:`Model/Architecture`) RETURN n.name, n.description LIMIT 50;

-- 查询所有数据集
MATCH (n:Dataset) RETURN n.name, n.description LIMIT 50;

-- 查询所有评估指标
MATCH (n:Metric) RETURN n.name, n.description LIMIT 50;

-- 查询所有任务类型
MATCH (n:Task) RETURN DISTINCT n.name ORDER BY n.name;

-- 查询所有图表（含子类型）
MATCH (n) WHERE n:Figure OR n:`Figure:chart` OR n:`Figure:table`
RETURN labels(n)[0] AS figure_type, n.name LIMIT 50;
```

### 2.3 按名称搜索

```cypher
-- 精确匹配
MATCH (n {name: "BERT"}) RETURN n.name, labels(n), n.description;

-- 模糊搜索（包含关键词）
MATCH (n) WHERE n.name CONTAINS "transformer"
RETURN n.name, labels(n)[0] AS type LIMIT 20;

-- 模糊搜索（前缀匹配）
MATCH (n) WHERE n.name STARTS WITH "GPT"
RETURN n.name, labels(n)[0] AS type;

-- 搜索描述内容
MATCH (n) WHERE n.description CONTAINS "attention mechanism"
RETURN n.name, labels(n)[0] AS type, n.description LIMIT 20;
```

---

## 3. 关系查询

### 3.1 常见关系类型

> 关系类型由 LLM 从论文中提取，使用自然语言命名，数量众多（600+种）。
> 以下为高频出现的代表性关系：

| 关系类型 | 示例含义 |
|---|---|
| `based_on` | A 基于 B |
| `trained_on` | A 在 B 上训练 |
| `outperforms` | A 超过 B |
| `achieves` | A 达到 B |
| `uses` | A 使用 B |
| `proposes` | A 提出 B |
| `evaluated_on` | A 在 B 上评估 |
| `introduced_by` | A 由 B 引入 |
| `improves` | A 改进 B |
| `extends` | A 扩展 B |

### 3.2 关系查询

```cypher
-- 查询特定实体的所有关系
MATCH (n {name: "BERT"})-[r]-(m) RETURN n.name, type(r), m.name LIMIT 50;

-- 查询特定关系类型
MATCH (n)-[r]->(m) WHERE type(r) = "based_on"
RETURN n.name, m.name LIMIT 50;

-- 查询某模型是基于什么构建的
MATCH (n:`Model/Architecture`)-[r]->(m) WHERE type(r) = "based_on"
RETURN n.name, type(r), m.name;

-- 查询某数据集被哪些模型使用
MATCH (n)-[r]->(d:Dataset)
WHERE type(r) IN ["trained_on", "evaluated_on", "uses"]
RETURN n.name, type(r), d.name LIMIT 50;

-- 查询某方法的应用场景
MATCH (m:`Method/Technique`)-[r]-(n)
RETURN m.name, type(r), n.name, labels(n)[0] AS target_type LIMIT 50;
```

### 3.3 多跳查询

```cypher
-- 两跳关联网络
MATCH (n {name: "BERT"})-[r1]-(m)-[r2]-(p)
RETURN n.name, type(r1), m.name, type(r2), p.name LIMIT 50;

-- 模型 -> 方法 -> 数据集（技术链路）
MATCH (model:`Model/Architecture`)-[r1]-(method:`Method/Technique`)-[r2]-(dataset:Dataset)
RETURN model.name, type(r1), method.name, type(r2), dataset.name LIMIT 50;

-- 最短路径（两个实体之间）
MATCH p = shortestPath((a {name: "BERT"})-[*..6]-(b {name: "GPT"}))
RETURN p;
```

---

## 4. 图表与多模态查询

```cypher
-- 查询所有表格
MATCH (n:`Figure:table`) RETURN n.name, n.description LIMIT 30;

-- 查询所有架构图
MATCH (n:`Figure:diagram`) RETURN n.name, n.description LIMIT 30;

-- 查询图表关联的文本分块
MATCH (fig)-[r]-(chunk:Chunk)
WHERE fig:Figure OR labels(fig)[0] STARTS WITH "Figure"
RETURN fig.name, type(r), chunk.id LIMIT 30;

-- 查询图片路径
MATCH (n:ImagePath) RETURN n.name, n.description LIMIT 30;
```

---

## 5. 跨类型分析

```cypher
-- 每种实体类型关联的关系数量
MATCH (n)-[r]-()
RETURN labels(n)[0] AS type, count(DISTINCT r) AS rel_count
ORDER BY rel_count DESC;

-- 关联最多的实体（Top 20）
MATCH (n)-[r]-()
RETURN n.name, labels(n)[0] AS type, count(r) AS degree
ORDER BY degree DESC LIMIT 20;

-- 模型之间的对比关系
MATCH (a:`Model/Architecture`)-[r]-(b:`Model/Architecture`)
WHERE type(r) IN ["outperforms", "compares_to", "improves", "extends"]
RETURN a.name, type(r), b.name LIMIT 50;

-- 论文引用链路（通过 Venue 串联）
MATCH (a:`Model/Architecture`)-[r1]->(v:Venue)<-[r2]-(b:`Model/Architecture`)
RETURN a.name, v.name, b.name LIMIT 30;

-- 同一数据集上的模型对比
MATCH (m1:`Model/Architecture`)-[r1]-(d:Dataset)-[r2]-(m2:`Model/Architecture`)
RETURN m1.name, d.name, m2.name, type(r1), type(r2) LIMIT 50;
```

---

## 6. 可视化查询

```cypher
-- 全部数据（节点数量大时谨慎使用）
MATCH (n)-[r]->(m) RETURN n, r, m;

-- 限制数量
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 500;

-- 包含孤立节点
MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m;

-- 特定实体的子图
MATCH (n {name: "BERT"})-[r*1..2]-(m) RETURN n, r, m LIMIT 100;

-- 某类型的完整子图
MATCH (n:`Model/Architecture`)-[r]-(m)
RETURN n, r, m LIMIT 200;
```

---

## 7. 数据维护

```cypher
-- 清空全部数据
MATCH (n) DETACH DELETE n;

-- 删除特定类型的节点
MATCH (n:`Figure:table`) DETACH DELETE n;

-- 删除孤立节点（无任何关系）
MATCH (n) WHERE NOT (n)--() DELETE n;

-- 查看节点属性
MATCH (n {name: "BERT"}) RETURN properties(n);

-- 删除特定关系
MATCH ()-[r]->() WHERE type(r) = "uses" DELETE r;
```

---

## 8. 图谱质量检查

```cypher
-- 重复名称检测
MATCH (n) WITH n.name AS name, collect(n) AS nodes
WHERE size(nodes) > 1
RETURN name, size(nodes) AS dup_count, [x IN nodes | labels(x)[0]] AS types;

-- 无描述的实体
MATCH (n) WHERE n.description IS NULL OR n.description = ""
RETURN n.name, labels(n)[0] AS type LIMIT 50;

-- 无名称的异常节点
MATCH (n) WHERE n.name IS NULL
RETURN id(n), labels(n), properties(n) LIMIT 20;

-- 自环检测（节点指向自己的关系）
MATCH (n)-[r]->(n) RETURN n.name, type(r);
```
