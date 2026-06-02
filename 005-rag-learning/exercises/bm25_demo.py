"""
=============================================================================
  练习6：BM25 混合检索对比 Demo
=============================================================================

演示三种检索策略的差异：
  1. 纯向量检索（TF-IDF）：语义相似
  2. 纯 BM25 关键词检索：精确字面匹配
  3. 混合检索（向量 + BM25 → RRF 融合）：取长补短

运行方式：
  cd rag-learning/demo && python3 ../exercises/bm25_demo.py
"""

import sys
import jieba
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))

from minimal_rag import SimpleRetriever, DOCUMENTS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


# ============================================================================
# BM25 关键词检索器
# ============================================================================
class BM25Retriever:
    """
    BM25（Best Match 25）：TF-IDF 的升级版，工业级关键词检索标配。

    比 TF-IDF 好在哪？
      1. 词频饱和：一个词出现100次 ≠ 相关性是10次的10倍，BM25 会"压平"
      2. 文档长度归一化：长文档不会因为词多就天然得分更高
      3. 有两个可调参数 k1（词频饱和度）和 b（长度归一化强度）
    """

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.contents = [doc["content"] for doc in documents]
        self.titles = [doc["title"] for doc in documents]

        # BM25 需要分词后的 token 列表（不是字符串）
        # 例：["公司 成立 于 2024 年", ...] → [["公司","成立","于","2024","年"], ...]
        tokenized = [jieba.lcut(c) for c in self.contents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 检索：对 query 分词，计算每篇文档的 BM25 分数"""
        query_tokens = jieba.lcut(query)             # jieba 分词
        scores = self.bm25.get_scores(query_tokens)  # BM25 打分，scores[i] = 文档i的得分

        # 按分数降序排序，取 top_k
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "title": self.titles[idx],
                    "content": self.contents[idx],
                    "score": round(float(scores[idx]), 4),
                })
        return results


# ============================================================================
# RRF 融合
# ============================================================================
def rrf_fusion(results_a: list[dict], results_b: list[dict], top_k: int = 5, k: int = 60):
    """
    Reciprocal Rank Fusion：把两种检索结果按排名融合。

    为什么用排名而不是分数？
      TF-IDF 分数范围 [0,1]，BM25 分数范围 [0, 几十]——没有可比性
      但排名是统一的尺度：第1名就是比第2名好
    """
    fused = {}  # title → {doc, rrf_score}

    # 检索A的排名贡献
    for rank, doc in enumerate(results_a, 1):
        key = doc["title"]
        fused[key] = {"doc": doc, "rrf_score": 1.0 / (k + rank)}

    # 检索B的排名贡献（如果文档在A中也出现了，分数叠加）
    for rank, doc in enumerate(results_b, 1):
        key = doc["title"]
        bonus = 1.0 / (k + rank)
        if key in fused:
            fused[key]["rrf_score"] += bonus      # 叠加 → 被两种检索都找到的文档排名上升
        else:
            fused[key] = {"doc": doc, "rrf_score": bonus}

    # 按 RRF 总分排序
    sorted_items = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    results = []
    for item in sorted_items[:top_k]:
        doc = item["doc"]
        doc["rrf_score"] = round(item["rrf_score"], 4)
        results.append(doc)
    return results


# ============================================================================
# 对比演示
# ============================================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         🔀 BM25 混合检索对比 Demo                          ║
║                                                              ║
║  对比三种检索策略在同一知识库上的表现                          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    tfidf_retriever = SimpleRetriever(DOCUMENTS)
    bm25_retriever = BM25Retriever(DOCUMENTS)

    # -------- 设计两个对比场景 --------
    scenarios = [
        {
            "name": "场景1：语义查询（同义词）",
            "query": "公司办公地点在哪里？",
            "note": "'办公地点' 和文档里的 '办公地址' 语义相同，但字面不同",
        },
        {
            "name": "场景2：精确匹配（专有名词）",
            "query": "Claude code是什么工具？",
            "note": "'Claude code' 是精确词，关键词检索应该比向量检索更准",
        },
    ]

    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"🔬 {sc['name']}")
        print(f"   查询: {sc['query']}")
        print(f"   说明: {sc['note']}")
        print(f"{'='*60}")

        vec_results = tfidf_retriever.retrieve(sc["query"], top_k=5)
        kw_results = bm25_retriever.search(sc["query"], top_k=5)
        hybrid_results = rrf_fusion(vec_results, kw_results, top_k=5)

        # 对比表格
        print(f"\n{'排名':<6} {'TF-IDF向量检索':<25} {'BM25关键词检索':<25} {'混合检索(RRF)':<25}")
        print("-" * 85)
        for rank in range(5):
            vec_title = vec_results[rank]["title"] if rank < len(vec_results) else "—"
            kw_title = kw_results[rank]["title"] if rank < len(kw_results) else "—"
            hy_title = hybrid_results[rank]["title"] if rank < len(hybrid_results) else "—"
            print(f"  第{rank+1}名  {vec_title:<23} {kw_title:<23} {hy_title:<23}")

    print(f"\n{'='*60}")
    print("💡 关键观察：")
    print("  - TF-IDF适合同义词/语义相近的查询（场景1）")
    print("  - BM25适合精确词/专有名词的匹配（场景2）")
    print("  - 混合检索综合两者排名，同时召回语义相关和精确匹配的文档")
    print(f"{'='*60}")
