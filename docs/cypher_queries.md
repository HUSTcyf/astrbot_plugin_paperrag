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

> 由 GBNF grammar 约束的 closed-set 9 类实体类型，LLM 输出被限制为以下类型之一。

| 标签 | 说明 |
|---|---|
| `Method` | 方法/技术（如 Attention Mechanism、Fine-tuning、Optimization） |
| `Model` | 模型/架构（如 BERT、GPT、Transformer、ResNet） |
| `Task` | 任务类型（如 Text Classification、Translation、QA） |
| `Dataset` | 数据集（如 GLUE、ImageNet、COCO） |
| `Metric` | 评估指标（如 Accuracy、F1、BLEU、Perplexity） |
| `Component` | 组件/模块（如 Layer Type、Sub-architecture、Building Block） |
| `Limitation` | 局限性（如 Weakness、Constraint、Boundary Condition） |
| `Application` | 应用场景（如 Real-world Use Case、Domain、Deployment） |
| `Baseline` | 基线方法（如 Previous Method、Compared System） |

**特殊节点类型**（由确定性逻辑创建，非 LLM 抽取）：

| 标签 | 说明 |
|---|---|
| `Figure_{type}` | 图片实体（如 `Figure_chart`、`Figure_diagram`、`Figure_image`），含 `image_path`、`description`、`figure_type`、`chunk_id` 属性 |
| `Table` | 表格实体，含 `description`、`chunk_id` 属性 |
| `Chunk` | 文本分块 |
| `Media` | 媒体文件（通过 `HAS_MEDIA` 关系连接到 Chunk） |

### 2.2 按类型查询

```cypher
-- 查询所有模型
MATCH (n:Model) RETURN n.name, n.description LIMIT 50;

-- 查询所有数据集
MATCH (n:Dataset) RETURN n.name, n.description LIMIT 50;

-- 查询所有评估指标
MATCH (n:Metric) RETURN n.name, n.description LIMIT 50;

-- 查询所有任务类型
MATCH (n:Task) RETURN DISTINCT n.name ORDER BY n.name;

-- 查询所有方法
MATCH (n:Method) RETURN n.name, n.description LIMIT 50;

-- 查询所有图表（含子类型）
MATCH (n) WHERE labels(n)[0] STARTS WITH 'Figure'
RETURN labels(n)[0] AS figure_type, n.name, n.image_path LIMIT 50;

-- 查询所有表格
MATCH (n:Table) RETURN n.name, n.description LIMIT 50;
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

### 3.1 当前图谱支持的关系类型

> 由 GBNF grammar 约束的 closed-set 9 类关系谓词，LLM 输出被限制为以下类型之一。
> 跨模态关系（cross-modal triplets）使用自由文本关系类型（如 `visualizes`、`shows_results`）。

| 关系类型 | 语义 | 示例 |
|---|---|---|
| `ADDRESSES` | 方法/论文 → 它所针对的任务或问题 | BERT → NLP Task |
| `PROPOSES` | 方法/论文 → 它所引入的方法或模型 | Paper → BERT |
| `USES_COMPONENT` | 方法 → 它所使用的组件或技术 | BERT → Transformer Encoder |
| `EVALUATED_ON` | 方法 → 用于评估的数据集 | BERT → GLUE |
| `ACHIEVES` | 方法 → 达到的指标或性能 | BERT → 86.4% Accuracy |
| `COMPARES_WITH` | 方法 → 与之比较的基线方法 | BERT → ELMo |
| `LIMITED_BY` | 方法 → 它所受的局限 | Method → Computational Cost |
| `APPLIES_TO` | 方法 → 目标应用领域 | Method → Medical Diagnosis |
| `EXTENDS` | 方法 → 它所基于的先前工作或模型 | BERT → Transformer |

**确定性关系类型**（由代码逻辑创建，非 LLM 抽取）：

| 关系类型 | 语义 |
|---|---|
| `HAS_MEDIA` | Chunk → Media 边（确定性，不受 VLM 失败影响） |

### 3.2 关系查询

```cypher
-- 查询特定实体的所有关系
MATCH (n {name: "BERT"})-[r]-(m) RETURN n.name, type(r), m.name LIMIT 50;

-- 查询特定关系类型（closed-set 谓词）
MATCH (n)-[r:`EVALUATED_ON`]->(m) RETURN n.name, m.name LIMIT 50;

-- 查询某模型是基于什么构建的
MATCH (n:Model)-[r:`EXTENDS`]->(m) RETURN n.name, type(r), m.name;

-- 查询某数据集被哪些方法评估
MATCH (n)-[r:`EVALUATED_ON`]->(d:Dataset)
RETURN n.name, type(r), d.name LIMIT 50;

-- 查询某方法使用的组件
MATCH (m:Method)-[r:`USES_COMPONENT`]->(c:Component)
RETURN m.name, type(r), c.name LIMIT 50;

-- 查询某方法的局限性
MATCH (m:Method)-[r:`LIMITED_BY`]->(l:Limitation)
RETURN m.name, type(r), l.name LIMIT 50;
```

### 3.3 多跳查询

```cypher
-- 两跳关联网络
MATCH (n {name: "BERT"})-[r1]-(m)-[r2]-(p)
RETURN n.name, type(r1), m.name, type(r2), p.name LIMIT 50;

-- 模型 -> 数据集 -> 其他模型（共享评估数据集）
MATCH (m1:Model)-[r1:`EVALUATED_ON`]->(d:Dataset)<-[r2:`EVALUATED_ON`]-(m2:Model)
RETURN m1.name, d.name, m2.name LIMIT 50;

-- 模型 -> 组件 -> 其他模型（共享组件）
MATCH (m1:Model)-[:`USES_COMPONENT`]->(c:Component)<-[:`USES_COMPONENT`]-(m2:Model)
WHERE m1.name < m2.name
RETURN m1.name, c.name, m2.name LIMIT 50;

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

-- 查询图片路径（存储在 Figure 节点的 image_path 属性中）
MATCH (n) WHERE labels(n)[0] STARTS WITH "Figure"
RETURN n.name, n.image_path, n.figure_type LIMIT 30;
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
MATCH (a:Model)-[r:`COMPARES_WITH`]-(b:Model)
RETURN a.name, type(r), b.name LIMIT 50;

-- 模型继承链（EXTENDS 关系）
MATCH (a:Model)-[r:`EXTENDS`*1..3]->(b:Model)
RETURN a.name, b.name, length(r) AS hops LIMIT 50;

-- 同一数据集上的模型对比
MATCH (m1:Model)-[r1:`EVALUATED_ON`]->(d:Dataset)<-[r2:`EVALUATED_ON`]-(m2:Model)
RETURN m1.name, d.name, m2.name LIMIT 50;
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
MATCH (n:Model)-[r]-(m)
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
