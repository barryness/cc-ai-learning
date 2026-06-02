# 向量数据库学习笔记：从 SQL 到语义搜索

> 从"为什么 MySQL 搜不出意思相近的内容"这个直觉问题开始。
> 不讲数学公式，只讲代码和直觉。

---

## 目录

1. [起点问题：MySQL 为什么搜不到"好吃的麻辣菜"](#1-起点问题mysql-为什么搜不到好吃的麻辣菜)
2. [向量数据库是什么](#2-向量数据库是什么)
3. [Embedding：把文字变成数字](#3-embedding把文字变成数字)
4. [ANN：近似最近邻搜索](#4-ann近似最近邻搜索)
5. [HNSW 索引：高速公路 → 主干道 → 小巷](#5-hnsw-索引高速公路--主干道--小巷)
6. [ChromaDB 核心概念：Client / Collection / Document](#6-chromadb-核心概念client--collection--document)
7. [Demo 代码逐段讲解](#7-demo-代码逐段讲解)
8. [两种存储模式：内存 vs 持久化](#8-两种存储模式内存-vs-持久化)
9. [混合查询：向量 + 元数据过滤](#9-混合查询向量--元数据过滤)
10. [Chroma vs FAISS vs Milvus：什么时候该升级](#10-chroma-vs-faiss-vs-milvus什么时候该升级)
11. [生产环境避坑指南](#11-生产环境避坑指南)
12. [总结：一张图理解全部](#12-总结一张图理解全部)

---

## 1. 起点问题：MySQL 为什么搜不到"好吃的麻辣菜"

### 场景

你有一个文档库，里面有一篇《麻婆豆腐的做法》。用户搜索："好吃的麻辣菜"。

```sql
SELECT * FROM documents WHERE content LIKE '%好吃的麻辣菜%';
-- 返回：0 行
```

为什么？因为文档里写的是"麻辣"，用户搜的是"好吃的"。**字面上一个匹配的都没有。**

### 换个思路

```sql
SELECT * FROM documents WHERE content LIKE '%麻辣%';
-- 返回：1 行 → 《麻婆豆腐的做法》
```

这次搜到了，"麻"和"辣"匹配上了。但这是你**手动**把查询词改成了"麻辣"——数据库自己不会做这件事。

### 再换个场景

文档是"iPhone 15 拍照效果很好"，用户搜"适合拍照的手机"。MySQL 怎么办？

```sql
SELECT * FROM documents WHERE content LIKE '%适合拍照的手机%';
-- 0 行

SELECT * FROM documents WHERE content LIKE '%拍照%' AND content LIKE '%手机%';
-- 还是 0 行（文档里没有"手机"这个词）
```

**MySQL 只能做字面匹配，完全不理解语义。**

### 这就是向量数据库要解决的问题

```
用户搜："适合拍照的手机"
向量数据库找到："iPhone 15 拍照效果很好" ← 语义最接近，但不是同一个词！
```

向量数据库的核心能力：**不要你告诉我关键词是什么，我自己理解你的意思，然后找意思最接近的内容。**

---

## 2. 向量数据库是什么

### 一句话

**向量数据库 = 专门存储和检索"向量"（高维数字列表）的数据库，通过计算向量之间的距离来判断语义相似度。**

### 类比：传统数据库 vs 向量数据库

| 概念 | 传统数据库（MySQL） | 向量数据库（ChromaDB） |
|------|---------------------|------------------------|
| 存什么 | 行和列（int, varchar, text） | 向量（float 数组，如 768 个 float） |
| 怎么查 | `WHERE name = 'x'` 精确匹配 | 给一个向量，找距离最近的 K 个向量 |
| 索引 | B-Tree（按大小排序） | HNSW 图（按距离分层） |
| 结果 | 确定的（有就有，没有就没有） | 近似的（找"足够近"的，精度换速度） |

### 直觉理解

想象一个地图：
- **传统数据库**：给你一个地址，告诉你那个地址有什么（精确查找）
- **向量数据库**：给你一个坐标，告诉你附近有什么（语义查找，按距离排序）

---

## 3. Embedding：把文字变成数字

### 为什么文字不能直接比

"麻婆豆腐" 和 "麻辣火锅" —— 人在脑子里能判断它们意思接近，但计算机只能看到字符 `0x9EBB 0x5A46 0x8C46...`，这些字节值之间没有任何语义关系。

### Embedding 做了什么

```
"麻婆豆腐" → Embedding 模型 → [0.12, -0.45, 0.78, 0.33, ...]  (768 个数字)
"麻辣火锅" → Embedding 模型 → [0.11, -0.43, 0.76, 0.35, ...]  (768 个数字)
"天气预报" → Embedding 模型 → [-0.67, 0.89, -0.12, -0.03, ...] (768 个数字)
```

- "麻婆豆腐" 和 "麻辣火锅" 的向量距离**很近** → 语义接近
- "麻婆豆腐" 和 "天气预报" 的向量距离**很远** → 语义不相关

### 本 Demo 用的 Embedding 模型

```python
class ChineseBertEmbedding(EmbeddingFunction):
    """
    使用 bert-base-chinese 生成句向量。
    通过 Mean Pooling 将 BERT 输出的每个 token 向量取平均，得到 768 维句向量。
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model = AutoModel.from_pretrained("bert-base-chinese")

    def __call__(self, texts):
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean Pooling：所有 token 向量取平均 → 一个 768 维向量
            pooled = mean_pooling(outputs, inputs["attention_mask"])
            # L2 归一化：让向量长度为 1
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled[0].numpy().tolist())
        return embeddings
```

### 为什么选 bert-base-chinese

| 模型 | 语言 | 维度 | 适合 |
|------|------|------|------|
| all-MiniLM-L6-v2 | 英文为主 | 384 | 英文 Demo，轻量快速 |
| bert-base-chinese | 中文 | 768 | 中文文档，语义理解准确 |
| text2vec-large-chinese | 中文 | 1024 | 对中文质量要求最高的场景 |

选 bert-base-chinese 因为它质量好、本地缓存加载、768 维足够表达中文语义。

---

## 4. ANN：近似最近邻搜索

### 暴力搜索的问题

如果每次查询都算"查询向量 vs 全部 100 万个文档向量"的距离：

```
100 万 个文档 × 768 维 = 7.68 亿次浮点运算
```

太慢了，而且随着文档量线性增长。

### ANN 的思想

**ANN = Approximate Nearest Neighbor = 近似最近邻**

不找"精确最近的那一个"，而是找"足够近的那几个"。用精度换速度。

```
暴力搜索：O(N) 时间，100% 精确
ANN 搜索：O(log N) 时间，99% 精确（通常够用了）
```

### 为什么 ANN 在向量检索中可以接受

因为语义搜索本来就不是精确的。用户搜"好吃的辣菜"，你返回麻婆豆腐还是水煮鱼，**都是对的**。和 SQL `WHERE id = 123` 必须精确到一行不同。

---

## 5. HNSW 索引：高速公路 → 主干道 → 小巷

### 这是一个什么结构

HNSW = Hierarchical Navigable Small World，分层的可导航小世界图。

用一个日常类比：

```
你要从北京去中关村某条小巷的某家咖啡店：

第 1 层（顶层）：北京 → 海淀区           ← 高速公路，节点很少
第 2 层（中层）：海淀区 → 中关村街道      ← 主干道，节点多一些
第 3 层（底层）：中关村街道 → 某条小巷     ← 小路，节点最密
```

HNSW 对向量做了同样的事：

```
Layer 2 (最稀疏):  向量A ←→ 向量Z          ← 大步跳，快速"定位到大区域"
Layer 1 (中间层):  向量A ←→ 向量M ←→ 向量Z ← 中步走，缩小范围
Layer 0 (最密集):  向量A→B→C→...→M→...→Z   ← 小步走，精确定位
```

### 搜索过程

```
1. 从顶层随机入口点开始
2. 在当前层做贪心搜索：不断走到离查询向量最近的邻居
3. 找不到更近的了 → 下降到下一层
4. 重复 2-3，到底层时返回收集到的 K 个最近向量
```

### 为什么这么快

- 顶层只有几十个节点 → 可以大步跨越
- 随着层数下降，精度逐渐提高
- 最终只访问了总向量的很小一部分 → O(log N)

### ChromaDB 中的 HNSW

ChromaDB 默认用 HNSW 做 ANN 索引。你在创建 Collection 时可以配置：

```python
collection = client.create_collection(
    name="my_docs",
    metadata={
        "hnsw:space": "cosine",       # 距离度量方式
        "hnsw:construction_ef": 100,  # 构建时的搜索宽度（越大越精确，越慢）
        "hnsw:M": 16,                 # 每个节点的最大连接数
    }
)
```

---

## 6. ChromaDB 核心概念：Client / Collection / Document

### 层级结构

```
ChromaDB
└── Client（客户端——连接到一个 ChromaDB 实例）
    └── Collection（集合——类似 SQL 的"表"）
        ├── Document 1（文档文本 + 向量 + 元数据）
        ├── Document 2
        └── Document N
```

### 每个概念对应到代码

```python
# Client：数据库连接
client = chromadb.Client()             # 内存模式
# client = chromadb.PersistentClient(path="./chroma_data")  # 持久化模式

# Collection：类似 CREATE TABLE
collection = client.create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,  # 自动把文本转成向量的函数
    metadata={"hnsw:space": "cosine"},
)

# Document：类似 INSERT INTO
collection.add(
    documents=["麻婆豆腐是一道经典的四川名菜..."],  # 原始文本
    metadatas=[{"category": "food"}],               # 附加属性（可过滤）
    ids=["doc_1"],                                  # 唯一 ID
)

# Query：类似 SELECT ... ORDER BY distance LIMIT 3
results = collection.query(
    query_texts=["好吃的辣菜"],  # 查询文本（自动转成向量）
    n_results=3,                # 返回 Top 3
)
```

### Collection 里发生了什么

```
当你 add() 时：
  文本 → embedding_function() → 768 维向量 → HNSW 索引 → 存储

当你 query() 时：
  查询文本 → embedding_function() → 768 维向量 → HNSW 搜索 → 返回最近的 K 个
```

关键：**embedding 函数在写入和查询时用同一个模型**。如果用了不同的模型，向量空间不对齐，查询结果就乱了。

---

## 7. Demo 代码逐段讲解

### 7.1 自定义 Embedding 函数

```python
class ChineseBertEmbedding(EmbeddingFunction):
```

为什么不用 ChromaDB 内置的 `DefaultEmbeddingFunction()`？

- 内置的是英文模型 `all-MiniLM-L6-v2`，对中文语义理解很差
- 搜"麻辣菜"可能返回篮球相关内容

自定义 Embedding 函数只需要实现 `__call__(self, texts)`，返回 `List[List[float]]`。ChromaDB 会在 add()/query() 时自动调用它。

### 7.2 Mean Pooling

```python
attention_mask = inputs["attention_mask"].unsqueeze(-1)
pooled = (outputs.last_hidden_state * attention_mask).sum(dim=1)
pooled = pooled / attention_mask.sum(dim=1)
```

BERT 输出的是**每个 token 一个向量**（一句话有 N 个 token 就有 N 个 768 维向量）。但我们需要**一个句子向量**。办法是对所有 token 向量取平均：

1. 把每个 token 向量乘以它的 attention_mask（0 或 1）→ padding 的 token 变成零向量
2. 求和 → 得到一个 768 维向量
3. 除以有效 token 数 → 得到平均值

### 7.3 L2 归一化

```python
pooled = F.normalize(pooled, p=2, dim=1)
```

把向量长度缩放到 1。好处：
- 余弦相似度 = 两个归一化向量的点积（计算更快）
- 避免长文本向量天然"更长"带来的偏差

### 7.4 关键词搜索（对比用）

```python
def simulate_keyword_search(query):
    query_chars = set(query) - stop_chars
    for text in documents:
        score = sum(1 for c in query_chars if c in text)
```

按字符出现次数打分，"有"和"一"这种单字会严重污染结果。这恰好展示了关键词搜索的致命弱点——它在中文场景几乎不可用，因为中文没有天然的空格分词。

### 7.5 语义搜索

```python
results = collection.query(
    query_texts=[query],
    n_results=3,
)
```

ChromaDB 自动：
1. 调用 embedding 函数把查询文本转成向量
2. 用 HNSW 在索引中搜索最近的 3 个文档向量
3. 返回原始文档文本和距离

---

## 8. 两种存储模式：内存 vs 持久化

### 内存模式（Demo 使用）

```python
client = chromadb.Client()
```

- 数据只存在进程内存中
- 进程结束，数据消失
- 适合：Demo、测试、临时探索
- 不需要任何配置

### 持久化模式（生产环境）

```python
client = chromadb.PersistentClient(path="./chroma_data")
```

- 数据写入磁盘（`./chroma_data/` 目录下）
- 进程重启数据还在
- 适合：本地应用、小规模生产
- 本质是 SQLite3 + HNSW 索引文件的组合

### 服务器模式（多客户端）

```bash
# 启动 ChromaDB 服务
chroma run --path ./chroma_data

# Python 客户端连接
client = chromadb.HttpClient(host="localhost", port=8000)
```

- 支持多个客户端同时读写
- 适合：团队协作、微服务架构

---

## 9. 混合查询：向量 + 元数据过滤

### 场景

用户想搜"科技类文档中和 AI 相关的内容"。

这不是一个纯语义问题——"和 AI 相关"是语义匹配，"科技类"是精确过滤。

### 代码

```python
results = collection.query(
    query_texts=["和人工智能相关的技术"],
    n_results=3,
    where={"category": "tech"},  # 先过滤，再在过滤结果中做向量检索
)
```

### 执行流程

```
1. where 过滤：从所有文档中筛出 category='tech' 的文档
2. 向量检索：在过滤后的文档子集中做 ANN 搜索
3. 返回 Top-K
```

### 支持的过滤操作

```python
# 等于
where={"category": "tech"}

# 不等于
where={"category": {"$ne": "sports"}}

# 属于列表
where={"category": {"$in": ["tech", "ai"]}}

# 数值比较
where={"word_count": {"$gt": 100}}

# 组合条件
where={"$and": [{"category": "tech"}, {"word_count": {"$gt": 100}}]}
```

注意：过滤发生在向量检索**之前**，所以返回结果数 ≤ n_results。如果过滤后只剩 2 篇文档，就算 `n_results=10` 也只会返回 2 篇。

---

## 10. Chroma vs FAISS vs Milvus：什么时候该升级

### 简单的决策逻辑

```
文档数 < 10万  → ChromaDB 完全够用（pip install 就行）
文档数 10万-100万 → pgvector 或 ChromaDB 持久化 + 合理分块
文档数 > 100万 → Milvus
离线/研究场景  → FAISS
已有 Elasticsearch → ES 原生向量功能（省一个服务）
```

### 三个工具的本质区别

| | ChromaDB | FAISS | Milvus |
|------|----------|-------|--------|
| **类型** | 数据库 | 检索库（库 ≠ 数据库） | 数据库 |
| **持久化** | 自动（SQLite） | 无（需要你手动保存） | 自动（MinIO/S3） |
| **CRUD** | add/update/delete/query | 只有 search | 全套 CRUD |
| **元数据过滤** | 原生支持 | 不支持 | 原生支持 |
| **分布式** | 不支持 | 不支持（单机） | 原生支持 |
| **适合场景** | 原型/小项目/学习 | 研究/离线批量 | 生产/大规模 |

### FAISS 是"库"不是"数据库"是什么意思

```python
# FAISS 的工作方式
import faiss
index = faiss.IndexFlatL2(768)       # 创建一个索引
index.add(vectors)                    # 把向量加进去
D, I = index.search(query_vector, k) # 搜索

# 进程结束，index 消失。
# 想持久化？自己写代码保存 numpy 数组到磁盘。
# 想新增一条？必须手动 update，没有 add 的便捷方法。
# 想加元数据过滤？自己维护一个 dict 映射。
```

FAISS 只做一件事，且做得极好：**向量相似度搜索**。它不管数据存哪里、怎么组织、怎么过滤——这些你自己搞定。

ChromaDB 在底层也可能用到了类似 HNSW 的算法（通过 hnswlib），但它帮你封装好了全部外围能力。

---

## 11. 生产环境避坑指南

### 坑1：Embedding 模型不一致

**场景**：导入文档时用模型 A，查询时用模型 B。

**结果**：向量空间完全错位，查询结果随机。

**解决**：**在 Collection 创建时绑定 embedding 函数**，ChromaDB 会一直复用。

### 坑2：文本分块不合理

**场景**：一本 500 页的书作为一个文档 embed 进去。

**结果**：整本书的 embedding 是一个"平均值"，丢失了大量细节。搜"第三章的数据结构"匹配到第四章去。

**解决**：合理分块（chunk），每块 200-500 字，块之间有 10-20% 重叠。

```python
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

### 坑3：只存向量，不存原始文本

**场景**：`collection.add(embeddings=vectors, ids=ids)` 只传向量。

**结果**：搜出来只有 ID 和一串数字，不知道原文档是什么。

**解决**：始终同时传 `documents` 参数，ChromaDB 会自动关联。

### 坑4：距离度量选错

| 场景 | 应选 |
|------|------|
| 文本语义搜索 | cosine（余弦相似度，只关心方向） |
| 图像相似搜索 | l2（欧几里得距离，关心绝对差异） |
| 推荐系统 | ip（内积，用户/物品向量的匹配强度） |

选错了不会报错，但排序结果会很怪。默认是 l2。

### 坑5：Python 3.14 + 科学计算包兼容性

详见 `python-env-guide.md` 踩坑记录。核心结论：**用 Python 3.12 做 ML 项目**，3.14 太新太多包没有 wheel。

---

## 12. 总结：一张图理解全部

```
                          向量数据库工作流程
                          ─────────────────

  写入路径：
  ┌──────────┐     ┌──────────────┐     ┌─────────────┐
  │ 原始文档  │ ──→ │ Embedding 模型│ ──→ │ 768维向量    │
  │ "麻婆豆腐" │     │ (bert-chinese)│     │ [0.12, -0.45, │
  └──────────┘     └──────────────┘     │  ...]        │
                                         └──────┬──────┘
                                                │ add()
                                         ┌──────▼──────┐
                                         │  HNSW 索引   │ ← ChromaDB Collection
                                         │  (向量+元数据) │
                                         └──────┬──────┘
                                                │
  查询路径：                                     │
  ┌──────────┐     ┌──────────────┐             │
  │ 用户查询  │ ──→ │ Embedding 模型│             │
  │ "好吃的辣菜"│     │ (同一个模型!)  │             │
  └──────────┘     └──────┬───────┘             │
                          │ query()             │
                          ▼                     ▼
                    ┌─────────────────────────────────┐
                    │  HNSW 搜索 → Top-K 最近向量       │
                    │  → 返回原始文档文本 + 相似度分数     │
                    └─────────────────────────────────┘

  核心认知：
  ┌────────────────────────────────────────────────────┐
  │ 向量数据库 = Embedding（语义转数字）+ ANN（快速搜索） │
  │                                                    │
  │ 它不是要替代 MySQL，而是补充 MySQL 做不到的事：       │
  │  → 理解语义、按意思搜索，而不是按字面匹配             │
  └────────────────────────────────────────────────────┘
```

---

## 下一步

1. **跑 Demo**：`python vector_db_demo.py`，观察三类查询的结果差异
2. **替换文档**：把测试文档换成你自己的数据，看检索质量
3. **读 comparison.md**：了解什么时候从 Chroma 升级到 Milvus
4. **学习 RAG**（005-rag-learning）：把向量检索 + LLM 生成组合起来，实现智能问答
