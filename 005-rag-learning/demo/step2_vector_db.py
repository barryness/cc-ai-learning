"""
=============================================================================
  Step 2：向量数据库 —— 从 TF-IDF 到语义检索
=============================================================================

核心升级：TF-IDF（关键词匹配）→ ChromaDB + bert-base-chinese（语义理解）

为什么升级：
  TF-IDF: 搜"技术栈"找到"FastAPI" ← 不行，字面不同
  Embedding: 搜"技术栈"找到"后端使用FastAPI框架" ← 可以，语义相近

新增能力：
  - ChromaDB 持久化：文档索引存磁盘，重启不丢失
  - 语义搜索：理解同义词、近义词、上下文
  - 相似度分数：量化"有多相关"

运行方式：
  pip install chromadb transformers torch
  python step2_vector_db.py
"""

import os
from pathlib import Path

import chromadb
import jieba
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")


# ═══════════════════════════════════════════════════════════════════════════
# 自定义 Embedding 函数（bert-base-chinese，本地离线）
# ═══════════════════════════════════════════════════════════════════════════

class ChineseBertEmbedding:
    """
    使用 bert-base-chinese 生成 768 维句向量。

    与 Step1 的 TF-IDF 对比：
      TF-IDF: 向量维度 = 词汇表大小（稀疏，大部分是 0），只捕捉词频
      BERT:   向量维度 = 768（稠密），每个维度都是语义特征
    """

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
        self.model = AutoModel.from_pretrained("bert-base-chinese", local_files_only=True)
        self.model.eval()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import torch
        import torch.nn.functional as F
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean Pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled[0].cpu().numpy().tolist())
        return embeddings


# ═══════════════════════════════════════════════════════════════════════════
# 文档加载（同 Step1，复用了 load_documents 逻辑）
# ═══════════════════════════════════════════════════════════════════════════

def load_documents(data_dir: str = "data") -> list[dict]:
    base = Path(data_dir)
    if not base.exists():
        return _demo_docs()
    docs = []

    # 优先 PDF
    pdf_files = sorted(base.glob("*.pdf"))
    if pdf_files:
        try:
            import fitz
            for fp in pdf_files:
                text = ""
                with fitz.open(fp) as doc:
                    for page in doc:
                        text += page.get_text()
                text = text.strip()
                if text:
                    docs.append({"title": fp.stem, "content": text})
        except ImportError:
            pass

    # 回退 txt/md
    if not docs:
        for fp in sorted(list(base.glob("*.txt")) + list(base.glob("*.md"))):
            raw = fp.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            lines = raw.split("\n", 1)
            title = lines[0].lstrip("#").strip() if lines[0].startswith("#") else fp.stem
            content = lines[1].strip() if len(lines) > 1 else raw
            docs.append({"title": title, "content": content})

    return docs if docs else _demo_docs()


def _demo_docs():
    return [
        {"title": "公司介绍", "content": "AI学习科技有限公司成立于2024年，总部位于北京。公司专注于AI教育平台开发，已服务超过10万学员。"},
        {"title": "产品介绍", "content": "公司旗舰产品'AI Tutor'是一个智能学习助手，支持Python、Java、Go等20+编程语言的实时辅导。产品采用订阅制，月费29元，年费299元。"},
        {"title": "技术架构", "content": "AI Tutor后端使用FastAPI框架，数据库PostgreSQL，向量数据库Milvus，LLM对接DeepSeek和OpenAI。"},
        {"title": "融资情况", "content": "公司于2025年3月完成A轮融资5000万元，投资方包括红杉资本和创新工场。"},
        {"title": "办公政策", "content": "公司实行混合办公制度，每周一三五在办公室，周二周四可远程办公。年假15天，带薪病假7天。"},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 的核心升级：EmbeddingRetriever（替代 TfidfRetriever）
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingRetriever:
    """
    基于 ChromaDB 的语义检索器。

    ChromaDB 自动处理：
    1. 文档 → embedding_function() → 向量 → HNSW 索引
    2. 查询文本 → embedding_function() → 向量 → ANN 搜索
    3. 返回最接近的 K 个文档

    内部使用 HNSW 算法，O(log N) 搜索时间。
    """

    def __init__(self, documents: list[dict], persist_dir: str = None):
        self.docs = documents
        self.embedding_fn = ChineseBertEmbedding()

        # 持久化模式：数据存磁盘，重启保留
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()  # 内存模式

        # 如果 Collection 已存在，先删后建（重新索引）
        try:
            self.client.delete_collection("knowledge_base")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="knowledge_base",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # 批量导入
        self.collection.add(
            documents=[f"{d['title']}\n{d['content']}" for d in documents],
            metadatas=[{"title": d["title"], "source": d.get("source", "")} for d in documents],
            ids=[f"doc_{i}" for i in range(len(documents))],
        )
        print(f"✅ 已索引 {len(documents)} 篇文档到 ChromaDB (embedding: bert-base-chinese, 768维)")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = []
        for i, (doc_id, text, dist) in enumerate(zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            meta = results["metadatas"][0][i]
            # Cosine 距离 → 相似度百分比
            sim = max(0, round((1 - dist) * 100))
            docs.append({
                "title": meta.get("title", doc_id),
                "content": text[:300],
                "score": sim,
            })
        return docs


# ═══════════════════════════════════════════════════════════════════════════
# 生成器（同 Step1）
# ═══════════════════════════════════════════════════════════════════════════

def generate_answer(query: str, docs: list[dict]) -> str:
    if not docs:
        return "抱歉，知识库中没有找到相关信息。"

    context = "\n\n".join(
        f"[文档{i}] 标题：{d['title']}\n内容：{d['content']}\n语义相似度：{d['score']}%"
        for i, d in enumerate(docs, 1)
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "你是一个基于内部知识库回答问题的助手。规则：1. 优先使用下面提供的文档内容回答 2. 如果文档中没有相关信息，请说'知识库中暂无相关信息' 3. 回答时引用文档编号"},
            {"role": "user", "content": f"## 知识库检索结果\n\n{context}\n\n## 用户问题\n\n{query}\n\n请基于以上知识库内容回答用户问题。"},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════
# 对比实验：TF-IDF vs Embedding 检索
# ═══════════════════════════════════════════════════════════════════════════

def compare_retrieval(docs: list[dict]):
    """直观对比 TF-IDF 和 Embedding 对同一个查询的检索结果"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tokenized = [" ".join(jieba.lcut(f"{d['title']} {d['content']}")) for d in docs]
    tfidf_vec = TfidfVectorizer().fit_transform(tokenized)

    emb_retriever = EmbeddingRetriever(docs)

    test_queries = [
        ("技术方面用了哪些框架？", ["FastAPI", "PostgreSQL", "React"]),
        ("公司有什么好处？", ["混合办公", "年假", "病假"]),
    ]

    print("\n" + "=" * 60)
    print("🔬 对比实验：TF-IDF vs Embedding 检索")
    print("=" * 60)

    for query, expected_keywords in test_queries:
        print(f"\n查询: 「{query}」")
        print(f"期望找到包含以下关键词的文档: {expected_keywords}")

        # TF-IDF
        qv = TfidfVectorizer().fit(tokenized).transform([" ".join(jieba.lcut(query))])
        # Fix: use the same vectorizer
        qv2 = tfidf_vec.transform([" ".join(jieba.lcut(query))])
        tfidf_scores = cosine_similarity(qv2, tfidf_vec)[0]
        tfidf_top = tfidf_scores.argsort()[::-1][:3]

        print(f"\n  TF-IDF 检索:")
        for i in tfidf_top:
            if tfidf_scores[i] > 0:
                kw_hit = [k for k in expected_keywords if k in docs[i]['content']]
                print(f"    [{tfidf_scores[i]:.3f}] {docs[i]['title']} {'✅ 命中' if kw_hit else '❌ 未命中'} {kw_hit}")

        # Embedding
        emb_results = emb_retriever.retrieve(query, top_k=3)
        print(f"\n  Embedding 检索:")
        for d in emb_results:
            kw_hit = [k for k in expected_keywords if k in d['content']]
            print(f"    [{d['score']}%] {d['title']} {'✅ 命中' if kw_hit else '❌ 未命中'} {kw_hit}")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🧠 Step 2：向量数据库 —— ChromaDB + bert-base-chinese")
    print("=" * 60)

    docs = load_documents("data")
    print(f"✅ 共加载 {len(docs)} 篇文档\n")

    # 对比实验（可选——展示 TF-IDF 和 Embedding 的差异）
    compare_retrieval(docs)

    # RAG 问答
    retriever = EmbeddingRetriever(docs)

    queries = [
        "公司混合办公的政策是什么？",
        "公司技术架构是怎样的？",
        "AI Tutor产品价格多少？",
    ]

    print("\n" + "=" * 60)
    print("💬 RAG 问答")
    print("=" * 60)

    for q in queries:
        print(f"\n📝 用户: {q}")
        retrieved = retriever.retrieve(q, top_k=3)
        for i, d in enumerate(retrieved, 1):
            print(f"   [{i}] {d['title']} (语义相似度: {d['score']}%)")
        answer = generate_answer(q, retrieved)
        print(f"🤖 AI: {answer}")

    print(f"\n{'=' * 60}")
    print("✅ Step 2 完成！")
    print("接下来：step3_multi_pdf.py —— 支持多个 PDF + 自动分块")
    print("=" * 60)

    # 如果用了持久化 ChromaDB，清理
    # import shutil; shutil.rmtree("./chroma_data", ignore_errors=True)


if __name__ == "__main__":
    main()
