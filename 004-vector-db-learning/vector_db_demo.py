#!/usr/bin/env python3
"""
向量数据库最小检索系统 Demo
=============================

基于 ChromaDB 实现，演示向量数据库的核心概念：
  1. 为什么传统关键词搜索不够用
  2. 向量语义搜索如何工作
  3. ChromaDB 的基本操作（增、查、过滤）

运行方式：
  uv venv --python 3.12 && source .venv/bin/activate
  pip install chromadb torch transformers
  python vector_db_demo.py

Embedding 引擎：bert-base-chinese（768 维，本地缓存加载，无需联网）
"""

import numpy as np
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


# =============================================================================
# 自定义 Embedding 函数：使用 bert-base-chinese（本地缓存）
# =============================================================================

class ChineseBertEmbedding(EmbeddingFunction):
    """
    使用 bert-base-chinese 生成文本向量。

    通过 mean pooling 将 BERT 最后一层的 hidden states 聚合为单个句向量。
    这是最常用的句向量提取方式，简单但效果好。

    使用 local_files_only=True 确保从本地缓存加载，不依赖网络。
    """

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-chinese", local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            "bert-base-chinese", local_files_only=True
        )
        self.model.eval()
        self.device = "cpu"
        self.dim = 768

    def __call__(self, texts: Documents) -> Embeddings:
        import torch

        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling: 对所有 token 的 hidden states 取平均
            # attention_mask 用于排除 padding token
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (outputs.last_hidden_state * attention_mask).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1)
            # L2 归一化，使余弦相似度计算等价于点积
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled[0].cpu().numpy().tolist())
        return embeddings


# =============================================================================
# 准备数据 —— 模拟一个小型知识库
# =============================================================================

DOCUMENTS = [
    # 科技类
    ("Python 是一门解释型、面向对象的高级编程语言，以其简洁易读的语法而闻名。"
     "它广泛应用于 Web 开发、数据科学、人工智能和自动化运维。",
     "tech"),

    ("向量数据库是一种专门设计用于存储和检索高维向量数据的数据库系统。"
     "它通过近似最近邻（ANN）搜索算法，在海量向量中快速找到语义最相似的条目。"
     "ChromaDB 和 Milvus 是两种流行的向量数据库。",
     "tech"),

    ("大语言模型（LLM）如 GPT-4 和 Claude 使用 Transformer 架构，"
     "通过自注意力机制理解文本上下文。它们可以用于对话、代码生成和内容创作。",
     "tech"),

    # 美食类
    ("麻婆豆腐是一道经典的四川名菜，以豆腐和牛肉末为主料，"
     "配以豆瓣酱、花椒和辣椒，口感麻辣鲜香，是川菜的代表作之一。",
     "food"),

    ("意大利面和披萨是意大利美食的两大标志。正宗的意大利面讲究'al dente'的口感，"
     "而传统的那不勒斯披萨只需要面粉、番茄、马苏里拉奶酪和罗勒叶四种原料。",
     "food"),

    ("寿司是日本料理的代表，以醋饭搭配生鱼片或其他海鲜制成。"
     "高级寿司讲究食材的新鲜度和师傅的手艺，从米饭的温度到鱼片的厚度都有严格要求。",
     "food"),

    # 运动类
    ("篮球是一项由两支五人队伍进行的对抗性运动，球员需要将球投入对方的篮筐中得分。"
     "NBA 是全球最高水平的职业篮球联赛，诞生了乔丹、科比、詹姆斯等传奇球星。",
     "sports"),

    ("足球是全球最受欢迎的运动，每四年一届的世界杯吸引数十亿观众。"
     "这项运动起源于英国，现在已有超过 200 个国家和地区参与国际足联。",
     "sports"),

    ("马拉松是一项长跑运动，标准距离为 42.195 公里。这项运动起源于古希腊传说，"
     "现在已成为全球各大城市广受欢迎的全民健身项目。",
     "sports"),
]

TEST_QUERIES = [
    "怎样做一道有辣味的中国传统菜？",
    "我想了解 AI 相关的技术知识",
    "有什么适合团队参加的体育项目？",
]


# =============================================================================
# 初始化 ChromaDB
# =============================================================================

def init_chroma():
    """
    创建 ChromaDB 客户端和 Collection。

    Collection 类似 SQL 中的"表"——所有文档和向量都存在 Collection 里。
    使用自定义的 bert-base-chinese embedding 函数，输出 768 维向量。

    metadata 中的 hnsw:space 设为 cosine，让 HNSW 索引用余弦距离做 ANN 搜索，
    这对文本语义比对效果最好。
    """
    print("=" * 70)
    print("🔧 初始化 ChromaDB + bert-base-chinese Embedding 引擎")
    print("=" * 70)

    embedding_fn = ChineseBertEmbedding()

    client = chromadb.Client()
    collection = client.create_collection(
        name="knowledge_base",
        embedding_function=embedding_fn,
        metadata={
            "description": "小型知识库，包含科技、美食、运动三类文档",
            "hnsw:space": "cosine",
        },
    )

    print(f"  Collection 名称: {collection.name}")
    print(f"  Embedding 模型: bert-base-chinese (768 维)")
    print(f"  距离度量: cosine（余弦相似度）")
    print()
    return collection


# =============================================================================
# 数据导入
# =============================================================================

def import_documents(collection):
    """
    将文档导入到 ChromaDB Collection。

    每次 add() 都会自动调用 embedding 函数，将文本转为向量后存储。
    每个文档携带 metadata（category 标签），后续可按标签过滤。
    """
    print("=" * 70)
    print("📥 导入文档到向量数据库")
    print("=" * 70)

    for i, (text, category) in enumerate(DOCUMENTS):
        collection.add(
            documents=[text],
            metadatas=[{"category": category, "source": f"doc_{i+1}"}],
            ids=[f"doc_{i+1}"],
        )
        summary = text[:50].replace("\n", " ")
        print(f"  [{category:6s}] doc_{i+1}: {summary}...")

    print(f"\n  共导入 {len(DOCUMENTS)} 个文档（3 类 x 3 篇）")
    print()


# =============================================================================
# 对比实验：关键词搜索 vs 向量语义搜索
# =============================================================================

def simulate_keyword_search(query: str) -> list[str]:
    """
    模拟传统关键词搜索引擎的行为。

    原理：对查询分词，统计每个文档中出现了多少查询词。
    这就是传统搜索（MySQL LIKE / Elasticsearch 基础模式）的核心逻辑。

    致命缺陷：完全无法处理同义词、近义词、语义关联。
    「辣味」匹配不到「麻辣」，「AI」匹配不到「大语言模型」。
    """
    # 按字符切分（中文场景比 jieba 分词更直观地展示关键词搜索的局限）
    query_chars = set(query)
    # 过滤掉常见的无意义字符
    stop_chars = {"？", "的", "。", "，", "？", "！", " ", "是", "我", "有", "想", "做"}
    query_terms = query_chars - stop_chars

    scored = []
    for text, _ in DOCUMENTS:
        score = sum(1 for term in query_terms if term in text)
        scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    # 只返回有匹配的结果
    return [text for score, text in scored if score > 0]


def compare_search_methods(collection):
    """
    核心对比实验：
      左侧：传统关键词搜索 → 只看字面匹配，结果为空或错误
      右侧：向量语义搜索   → 理解语义含义，找到真正相关的内容

    这是整个 Demo 最重要的部分。
    """
    print("=" * 70)
    print("🔍 对比实验：关键词搜索 vs 向量语义搜索")
    print("=" * 70)

    for query in TEST_QUERIES:
        print(f"\n{'─' * 70}")
        print(f"  查询: 「{query}」")
        print(f"{'─' * 70}")

        # ---- 关键词搜索 ----
        print("\n  ❌ 传统关键词搜索（字面匹配）：")
        kw_results = simulate_keyword_search(query)
        if not kw_results:
            print("      → 无结果！（关键词不匹配，完全找不到相关内容）")
        else:
            for i, text in enumerate(kw_results, 1):
                print(f"      {i}. {text[:80]}...")

        # ---- 向量语义搜索 ----
        print("\n  ✅ 向量语义搜索（语义理解）：")
        results = collection.query(
            query_texts=[query],
            n_results=3,
        )
        for i, (doc_id, text, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
        ), 1):
            # Cosine 距离转相似度：1 - distance 就是余弦相似度
            similarity_pct = max(0, (1 - distance) * 100)
            print(f"      {i}. [{doc_id}] (相似度 {similarity_pct:.0f}%) {text[:80]}...")
    print()


# =============================================================================
# 元数据过滤 —— 向量检索 + 标量过滤的混合查询
# =============================================================================

def demonstrate_filtering(collection):
    """
    演示向量数据库的高级能力：混合查询（Hybrid Search）。

    「在科技类文档中，找和 AI 最相关的内容」
    → 先用 category='tech' 过滤，再在过滤结果中做向量检索。

    这种能力将向量数据库与纯检索库（FAISS）区分开来。
    """
    print("=" * 70)
    print("🎯 混合查询：向量检索 + 元数据过滤")
    print("=" * 70)

    query = "和人工智能相关的技术"
    categories_to_test = [None, "tech", "food", "sports"]

    for cat in categories_to_test:
        label = cat if cat else "无过滤（搜索全部）"
        print(f"\n  过滤条件: category = '{label}'")

        where_filter = {"category": cat} if cat else None
        results = collection.query(
            query_texts=[query],
            n_results=3,
            where=where_filter,
        )

        for i, (doc_id, text, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
        ), 1):
            similarity_pct = max(0, (1 - distance) * 100)
            print(f"      {i}. [{doc_id}] (相似度 {similarity_pct:.0f}%) {text[:80]}...")

    print()


# =============================================================================
# 向量可视化（概念演示）
# =============================================================================

def demonstrate_vector_space():
    """
    用 numpy 模拟一个"微型向量空间"，直观展示语义相似的本质。

    这不是 ChromaDB 操作，而是帮助建立直觉：
    为什么"语义相近 = 向量距离近"。
    """
    print("=" * 70)
    print("🧭 向量空间直觉：用 PCA 降维到 2D 可视化")
    print("=" * 70)

    # 对三组概念生成模拟向量
    np.random.seed(42)
    # 科技簇
    tech_vecs = np.random.randn(3, 768) * 0.3 + np.array([1.0, 0.0] + [0] * 766)
    # 美食簇
    food_vecs = np.random.randn(3, 768) * 0.3 + np.array([0.0, 1.0] + [0] * 766)
    # 运动簇
    sports_vecs = np.random.randn(3, 768) * 0.3 + np.array([-1.0, -0.5] + [0] * 766)

    all_vecs = np.vstack([tech_vecs, food_vecs, sports_vecs])
    labels = ["Python", "向量数据库", "大语言模型",
              "麻婆豆腐", "意大利面", "寿司",
              "篮球", "足球", "马拉松"]

    # 用前两个主成分做简易降维
    centered = all_vecs - all_vecs.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T

    print("\n  文档在 2D 语义空间中的位置：")
    print(f"  {'─' * 45}")
    for i, (label, (x, y)) in enumerate(zip(labels, projected)):
        category = ["科技", "科技", "科技", "美食", "美食", "美食", "运动", "运动", "运动"][i]
        print(f"  [{category}] {label:8s} → ({x:+.3f}, {y:+.3f})")

    print(f"\n  → 同类文档在向量空间中自然聚集，语义检索就是「找最近的邻居」")
    print()


# =============================================================================
# 距离度量详解
# =============================================================================

def demonstrate_distance_metrics():
    """向量相似度的三种度量方式。"""
    print("=" * 70)
    print("📐 向量距离度量方式")
    print("=" * 70)
    print("""
  常用的三种向量距离：

  1. 余弦相似度 (Cosine Similarity)
     → 公式: cos(θ) = (A·B) / (|A| x |B|)
     → 范围: [-1, 1]，1 表示完全相同方向
     → 只关注「方向」，不关注「长度」，最适用于文本语义比较

  2. 欧几里得距离 (L2 / Euclidean)
     → 公式: d = √(Σ(Ai - Bi)²)
     → 范围: [0, ∞)，0 表示完全相同
     → 同时考虑方向和长度，适用于图像相似度

  3. 点积 (Dot Product)
     → 公式: A·B = Σ(Ai x Bi)
     → 范围: (-∞, ∞)，越大越相似
     → 需要向量已做 L2 归一化（此时等价于余弦相似度）

  选型建议：
    - 文本语义搜索 → 余弦相似度（本 Demo 使用）
    - 图像相似度   → 欧几里得距离
    - 推荐系统     → 点积（归一化后）
""")


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  🧬 向量数据库最小检索系统")
    print("  从「为什么需要向量数据库」到「跑通第一个向量查询」")
    print("=" * 70)
    print()

    # Step 1: 初始化向量数据库
    collection = init_chroma()

    # Step 2: 导入文档
    import_documents(collection)

    # Step 3: 核心对比 —— 关键词 vs 语义
    compare_search_methods(collection)

    # Step 4: 元数据过滤
    demonstrate_filtering(collection)

    # Step 5: 向量空间直觉
    demonstrate_vector_space()

    # Step 6: 距离度量
    demonstrate_distance_metrics()

    # 总结
    print("=" * 70)
    print("  ✅ Demo 完成！")
    print("=" * 70)
    print("""
  核心收获：
    1. 传统搜索靠「字面匹配」 → 搜「辣味」找不到「麻辣」
    2. 向量搜索靠「语义理解」 → 搜「辣味的中餐」找到「麻婆豆腐」
    3. ChromaDB 让向量数据库像 SQLite 一样简单
    4. 混合查询（过滤 + 向量）是生产环境的标配

  进一步学习：
    → 对比阅读 comparison.md 了解各数据库差异
    → 把你的文档导入 Collection，观察中文检索效果
    → 研究 RAG（检索增强生成）：向量检索 + LLM 生成 = 智能问答
""")


if __name__ == "__main__":
    main()
