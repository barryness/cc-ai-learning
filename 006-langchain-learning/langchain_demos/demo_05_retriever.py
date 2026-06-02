"""
=============================================================================
  Demo 05：Retriever —— LangChain 的文档检索抽象层
=============================================================================

问题：之前写 RAG 用的是 ChromaDB 原生 API（collection.query()）。
      如果以后要换成 FAISS / Milvus / Pinecone 呢？

不用 LangChain 怎么办？
  → 每个向量数据库有自己的 API，切换时要重写所有检索代码
  → 痛点：和供应商绑定，换数据库 = 重构

LangChain Retriever 的价值：
  1. 统一接口：不管底层是 Chroma / FAISS / Milvus，都是 .invoke(query) → [Document]
  2. 自动处理 Embedding：VectorStore.from_documents() 自动 embed + 存入
  3. 过滤器：search_kwargs 支持元数据过滤
  4. 可组合：Retriever 可以作为 Chain 的一环（Demo 08 会看到）

运行方式：
  python demo_05_retriever.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# 知识库文档
# ═══════════════════════════════════════════════════════════════════════════

DOCUMENTS = [
    {"title": "公司介绍", "content": "AI学习科技有限公司成立于2024年，总部位于北京。"},
    {"title": "产品介绍", "content": "AI Tutor支持Python、Java、Go等20+编程语言，月费29元，年费299元。"},
    {"title": "技术架构", "content": "后端使用FastAPI，数据库PostgreSQL，向量数据库Milvus。"},
    {"title": "办公政策", "content": "混合办公，每周一三五到岗，年假15天，带薪病假7天。"},
    {"title": "融资情况", "content": "2025年A轮融资5000万元，投资方红杉资本和创新工场。"},
]


# ═══════════════════════════════════════════════════════════════════════════
# 自定义 Embedding（同之前的 bert-base-chinese）
# ═══════════════════════════════════════════════════════════════════════════

class ChineseBertEmbedding:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
        self.model = AutoModel.from_pretrained("bert-base-chinese", local_files_only=True)
        self.model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import torch
        import torch.nn.functional as F
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled[0].cpu().numpy().tolist())
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ═══════════════════════════════════════════════════════════════════════════
# 方式一：不用 LangChain —— 原生 ChromaDB
# ═══════════════════════════════════════════════════════════════════════════

def without_langchain():
    print("=" * 60)
    print("  方式一：原生 ChromaDB API")
    print("=" * 60)

    import chromadb

    emb_fn = ChineseBertEmbedding()
    client = chromadb.Client()
    try:
        client.delete_collection("demo5_raw")
    except Exception:
        pass
    collection = client.create_collection(
        name="demo5_raw",
        embedding_function=emb_fn,  # ChromaDB 原生接口接收 EmbeddingFunction
    )
    collection.add(
        documents=[f"{d['title']}\n{d['content']}" for d in DOCUMENTS],
        metadatas=[{"title": d["title"]} for d in DOCUMENTS],
        ids=[f"doc_{i}" for i in range(len(DOCUMENTS))],
    )

    # 检索 —— 注意 API 是 collection.query()
    results = collection.query(query_texts=["公司办公政策"], n_results=3)
    for i, (doc_id, text, dist) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    )):
        meta = results["metadatas"][0][i]
        print(f"  [{i+1}] {meta['title']} (距离: {dist:.3f})")

    print("""
    ❌ 原生 ChromaDB 的问题：
       1. API 不统一 —— FAISS/Milvus 的 API 完全不同
       2. 返回 dict → 每次都要 ["ids"][0][i] 这种深层取值
       3. 元数据过滤的语法因数据库而异
       4. 换数据库 = 重写所有检索代码
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 方式二：用 LangChain Retriever —— 统一抽象层
# ═══════════════════════════════════════════════════════════════════════════

def with_langchain_retriever():
    print("\n" + "=" * 60)
    print("  方式二：LangChain Retriever（统一接口）")
    print("=" * 60)

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    # 把自定义 embedding 包装成 LangChain Embeddings 接口
    class LCEmbeddings(Embeddings):
        def __init__(self):
            self._emb = ChineseBertEmbedding()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._emb.embed_documents(texts)

        def embed_query(self, text: str) -> str:
            return self._emb.embed_query(text)

    embeddings = LCEmbeddings()

    # LangChain Document 对象 —— 比 dict 更规范
    lc_docs = [
        Document(page_content=d["content"], metadata={"title": d["title"]})
        for d in DOCUMENTS
    ]

    # from_documents() 自动 embed + 存入
    vectorstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        collection_name="demo_05_lc",
        persist_directory="./chroma_demo05",
    )

    # 关键：as_retriever() 返回统一接口！
    # 不管底层是 Chroma / FAISS / Milvus，都是 retriever.invoke(query)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3},  # top_k
    )

    # 统一调用方式：invoke(query) → list[Document]
    docs = retriever.invoke("公司办公政策")

    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "?")
        print(f"  [{i}] {title}")
        print(f"      内容: {doc.page_content[:80]}...")

    print("""
    ✅ LangChain Retriever 的价值：
       1. 统一接口：retriever.invoke(query) → list[Document]
       2. 换数据库只改构造函数，检索代码不动
       3. Document 对象有 .page_content 和 .metadata，不再用 dict key
       4. as_retriever(search_kwargs) 统一配置 top_k / 过滤
       5. 可以直接作为 Chain 的一环（Demo 08 演示）
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 进阶：带元数据过滤的检索
# ═══════════════════════════════════════════════════════════════════════════

def demo_metadata_filter():
    print("\n" + "=" * 60)
    print("  进阶：元数据过滤检索")
    print("=" * 60)

    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    class LCEmbeddings(Embeddings):
        def __init__(self):
            self._emb = ChineseBertEmbedding()
        def embed_documents(self, texts):
            return self._emb.embed_documents(texts)
        def embed_query(self, text):
            return self._emb.embed_query(text)

    # 不同类别的文档
    lc_docs = [
        Document(page_content="FastAPI 是一个现代 Python Web 框架", metadata={"category": "技术", "difficulty": "中级"}),
        Document(page_content="Python 装饰器是修改函数行为的语法糖", metadata={"category": "技术", "difficulty": "入门"}),
        Document(page_content="公司年假15天，带薪病假7天", metadata={"category": "HR", "difficulty": "入门"}),
        Document(page_content="报销需部门经理审批", metadata={"category": "HR", "difficulty": "入门"}),
    ]

    vectorstore = Chroma.from_documents(
        lc_docs, LCEmbeddings(), collection_name="demo05_filter",
        persist_directory="./chroma_demo05",
    )

    # 只搜索 category=HR 的文档
    retriever_hr = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"category": "HR"}}
    )

    print("搜索「福利政策」，仅限 HR 类别:")
    docs = retriever_hr.invoke("福利政策")
    for doc in docs:
        print(f"  [{doc.metadata['category']}] {doc.page_content[:60]}...")

    print("\n  filter={'category': 'HR'} 只返回了 HR 文档 ✅")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🔍 Demo 05：Retriever —— LangChain 的检索抽象层")
    print("=" * 60)

    without_langchain()
    with_langchain_retriever()
    demo_metadata_filter()

    print("\n✅ Demo 05 完成！")
    print("关键收获：LangChain Retriever 统一了检索接口。")
    print("不管底层是 Chroma/FAISS/Milvus，一律 retriever.invoke(query) → list[Document]。\n")


if __name__ == "__main__":
    main()
