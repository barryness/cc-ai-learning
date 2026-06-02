# 练习6：BM25 混合检索

## 做了什么

1. 实现了 `BM25Retriever`（基于 `rank-bm25` 库）
2. 实现了 `rrf_fusion()` 融合函数
3. 创建对比脚本 `bm25_demo.py`，对比三种策略

## BM25 vs TF-IDF

| 特性 | TF-IDF | BM25 |
|------|--------|------|
| 词频处理 | 线性：词出现越多分数越高 | **饱和**：词频高到一定程度后不再加分 |
| 文档长度 | 无处理，长文档天然得分高 | **归一化**：长文档不被偏袒 |
| 可调参数 | 无 | k1（词频饱和度）、b（长度归一化） |
| 分数范围 | [0, 1]（余弦相似度标准化后） | [0, ∞) |
| 适合场景 | 语义相似查询 | 精确关键词匹配 |

### BM25 为什么比 TF-IDF 更适合关键词检索？

```
文档A（100字）: "Python Python Python Python Python"（出现5次）
文档B（100字）: "Python 教程"（出现1次）

TF-IDF: 文档A得分 ≈ 5x 文档B  → 过度偏袒词频
BM25:   文档A得分 ≈ 1.5x 文档B → 更合理，出现5次不代表5倍相关
```

## 测试代码

### `BM25Retriever` 核心实现

```python
class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        tokenized = [jieba.lcut(doc["content"]) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)  # rank-bm25 库

    def search(self, query, top_k=5):
        query_tokens = jieba.lcut(query)
        scores = self.bm25.get_scores(query_tokens)  # 每个文档一个分数
        top_indices = scores.argsort()[::-1][:top_k]
        # ... 组装返回
```

### RRF 融合

```python
def rrf_fusion(results_a, results_b, top_k=5, k=60):
    fused = {}
    for rank, doc in enumerate(results_a, 1):
        fused[doc["title"]] = {"doc": doc, "rrf_score": 1.0 / (k + rank)}
    for rank, doc in enumerate(results_b, 1):
        key = doc["title"]
        bonus = 1.0 / (k + rank)
        if key in fused:
            fused[key]["rrf_score"] += bonus  # 两种检索都找到 → 分数叠加
        else:
            fused[key] = {"doc": doc, "rrf_score": bonus}
    # 按 rrf_score 降序排列
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
```

**关键设计**：用排名而非原始分数来融合——TF-IDF 分数 [0,1] 和 BM25 分数 [0,几十] 不可直接比较，但排名是统一的尺度。

## 测试结果

### 场景1：语义查询 "公司办公地点在哪里？"

| 排名 | TF-IDF | BM25 | 混合(RRF) |
|------|--------|------|-----------|
| 1 | 办公政策 | 办公政策 | 办公政策 |
| 2 | 公司介绍 | 公司介绍 | 公司介绍 |
| 3 | 融资情况 | 大数据实时处理工具 | 融资情况 |
| 4 | 产品介绍 | 技术架构 | 大数据实时处理工具 |
| 5 | — | 融资情况 | 产品介绍 |

TF-IDF 更集中（4条），BM25 召回更广（5条），混合综合两者。

### 场景2：精确匹配 "Claude code是什么工具？"

| 排名 | TF-IDF | BM25 | 混合(RRF) |
|------|--------|------|-----------|
| 1 | vibe coding工具 | vibe coding工具 | vibe coding工具 |
| 2 | 大数据实时处理工具 | 产品介绍 | 大数据实时处理工具 |
| 3 | — | AI中的养龙虾 | 产品介绍 |
| 4 | — | Python函数式编程 | AI中的养龙虾 |
| 5 | — | 技术架构 | Python函数式编程 |

TF-IDF 只找到 2 条（"Claude code" 和文档词语重叠太少），BM25 找到 5 条（"code" 这个 token 扩散匹配到多篇文档）。混合检索在保留 TF-IDF 精确匹配的同时，用 BM25 补充了更多候选。

## 什么时候用哪种？

```
查询类型              → 推荐策略
────────────────────────────────────
"什么是..."           → 纯向量（语义理解）
"API v3.0 文档"       → 混合（精确版本号 + 语义）
"error code 500"      → 混合（精确错误码 + 相关文档）
"怎样提高性能"        → 纯向量（同义词多）
```

## 从 Demo 到生产

生产级 `advanced_rag.py` 的 `HybridRetriever` 就是这个模式的完整实现——向量检索（ChromaDB + Embedding）+ 关键词检索（替换为 BM25）+ RRF 融合。练习 6 的内容就是它核心逻辑的简化版。
