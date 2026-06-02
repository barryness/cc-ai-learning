"""
=============================================================================
  生产级 RAG Demo — 从玩具到工业级
=============================================================================

核心升级点：
  1. TF-IDF → Embedding 模型（sentence-transformers）：语义理解更强
  2. 内存数组 → ChromaDB 向量数据库：持久化、高效检索
  3. 整篇文档 → 语义分块：避免文档太长导致检索不准
  4. 单轮检索 → 混合检索 + 重排序：关键词 + 语义双重召回
  5. print → logging：可观测性

架构全景：

  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐
  │  PDF/网页/   │───▶│  文档分块器    │───▶│  Embedding    │
  │  数据库...   │    │  (Chunker)    │    │  (Encoder)    │
  └─────────────┘    └──────────────┘    └───────┬───────┘
                                                  │
                  ┌───────────────────────────────▼────────────────────┐
                  │                 向量数据库 (ChromaDB)                │
                  │   ┌─────────┐  ┌─────────┐  ┌─────────┐           │
                  │   │ Chunk 1 │  │ Chunk 2 │  │ Chunk N │  ...      │
                  │   │ vector  │  │ vector  │  │ vector  │           │
                  │   └─────────┘  └─────────┘  └─────────┘           │
                  └───────────────────────┬────────────────────────────┘
                                          │
  ┌──────────┐                    ┌───────▼────────┐
  │  用户问题  │──────────────────▶│   检索器        │
  └──────────┘                    │ (Retriever)    │
                                  │ 向量+关键词混合  │
                                  └───────┬────────┘
                                          │ top_k 文档
                                  ┌───────▼────────┐
                                  │   重排序器      │
                                  │ (Re-ranker)   │
                                  └───────┬────────┘
                                          │ 最相关文档
  ┌──────────┐                    ┌───────▼────────┐
  │   LLM    │◀───────────────────│  Prompt 构造   │
  │ 生成答案  │                    │  拼接上下文     │
  └──────────┘                    └────────────────┘
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass, field

import chromadb
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ============================================================================
# 配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("advanced-rag")

load_dotenv("../demo/.env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

# ============================================================================
# 1. 文档分块器（Chunker）
# ============================================================================
# 为什么需要分块？
# - Embedding 模型有最大输入长度限制（通常 512 tokens）
# - 文档太长，检索精度下降（一篇100页的文档匹配了其中一句话）
# - 分块后，检索返回的是"相关的段落"，而非整篇文档
# - Token ≠ 字符。1 个中文字 ≈ 1.5~2 tokens，1 个英文单词 ≈ 1.3 tokens
#
# 分块策略：
# - 固定大小分块（最简单）：每 N 个字符一块
# - 语义分块（更好）：按段落/句子边界切，保持语义完整
# - 递归分块（推荐）：先按大分隔符切，不够再按小分隔符切

@dataclass
class Chunk:
    """文档分块"""
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""


class SemanticChunker:
    """
    语义分块器：优先在段落边界切分，保持语义完整性。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块的重叠字符数
                       为什么要重叠？避免关键信息刚好被切断在边界
        例：chunk1 "...公司在2024年" chunk2 "完成了A轮融资5000万"
        如果没重叠 → 检索"公司融资"可能两块都匹配度不够
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str, title: str = "") -> list[Chunk]:
        """
        分段策略：
          1. 先按自然段落（\n\n）切分
          2. 短段落合并，长段落按句子再切
          3. 相邻块之间有 overlap
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果加上这段不会超长，就合并
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # 当前块满了，保存
                if current_chunk.strip():
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        metadata={"title": title, "chunk_index": chunk_idx},
                        chunk_id=f"{title}_chunk_{chunk_idx}",
                    ))
                    chunk_idx += 1

                    # 新块开头包含上一块的末尾（overlap）
                    overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        # 最后一块
        if current_chunk.strip():
            chunks.append(Chunk(
                text=current_chunk.strip(),
                metadata={"title": title, "chunk_index": chunk_idx},
                chunk_id=f"{title}_chunk_{chunk_idx}",
            ))

        logger.info(f"文档'{title}'被分为 {len(chunks)} 个 chunks")
        return chunks


# ============================================================================
# 2. Embedding 模型 & 向量数据库
# ============================================================================
class VectorStore:
    """
    向量存储和检索。
    用 ChromaDB 替代最小 Demo 中的 numpy 数组。
    """

    def __init__(self, collection_name: str = "rag_docs"):
        # 使用本地 Embedding 模型，无需 API Key
        # all-MiniLM-L6-v2：轻量模型，384维向量，速度很快
        # 工业级可换 text-embedding-3-small（OpenAI）或 bge-large-zh（中文优化）
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # ChromaDB 客户端（本地持久化）
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # 获取或创建 collection
        # 相当于关系数据库中的"表"
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 用余弦相似度
        )

    def add_documents(self, chunks: list[Chunk]):
        """将分块文档写入向量数据库"""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # sentence-transformers 自动完成：文本 → Tokenize → 模型推理 → 向量
        embeddings = self.encoder.encode(texts).tolist()

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"已写入 {len(chunks)} 个文档块到向量数据库")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义检索：把 query 转向量，在向量库中找最相似的 chunks"""
        query_embedding = self.encoder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        # 格式化和 Demo 相同的返回结构
        docs = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                docs.append({
                    "title": results["metadatas"][0][i].get("title", ""),
                    "content": results["documents"][0][i],
                    "score": round(1.0 - results["distances"][0][i], 4),
                    "chunk_id": results["ids"][0][i],
                })
        return docs


# ============================================================================
# 3. 关键词检索器（BM25 思想简化版）
# ============================================================================
class KeywordRetriever:
    """
    关键词匹配检索：弥补纯向量检索的不足。

    为什么需要关键词检索？
      向量检索擅长"语义相似"，但不擅长"精确匹配"。
      比如搜索"API-3.0"，向量检索可能返回 API 相关内容但不是 3.0 版本，
      而关键词检索可以直接匹配 "API-3.0"。
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """基于关键词重叠度打分"""
        query_words = set(query.lower().split())
        scored = []

        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            # 简单 Jaccard 相似度：共同词 / 总词数
            if not query_words:
                continue
            overlap = len(query_words & chunk_words)
            total = len(query_words | chunk_words)
            score = overlap / total if total > 0 else 0

            if score > 0:
                scored.append({
                    "title": chunk.metadata.get("title", ""),
                    "content": chunk.text,
                    "score": round(score, 4),
                    "chunk_id": chunk.chunk_id,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ============================================================================
# 4. 混合检索 + 重排序
# ============================================================================
class HybridRetriever:
    """
    混合检索器：向量检索 + 关键词检索 → RRF 融合

    RRF（Reciprocal Rank Fusion）原理：
      向量检索结果：chunk_A排第1, chunk_C排第2, chunk_B排第3
      关键词检索结果：chunk_B排第1, chunk_A排第4, chunk_D排第5

      用公式融合排名：score = 1/(k + rank_vector) + 1/(k + rank_keyword)
      其中 k 是常数（通常 60），防止单项排名为1时分数过高

      最终排名：综合考虑两种检索方式，提高召回率
    """

    def __init__(self, vector_store: VectorStore, keyword_retriever: KeywordRetriever):
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # 并行执行两种检索
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        keyword_results = self.keyword_retriever.search(query, top_k=top_k * 2)

        # RRF 融合
        fused = {}  # chunk_id → {doc, rrf_score}
        k = 60  # RRF 常数

        for rank, doc in enumerate(vector_results, 1):
            cid = doc["chunk_id"]
            fused[cid] = {"doc": doc, "rrf_score": 1.0 / (k + rank)}

        for rank, doc in enumerate(keyword_results, 1):
            cid = doc["chunk_id"]
            bonus = 1.0 / (k + rank)
            if cid in fused:
                fused[cid]["rrf_score"] += bonus
            else:
                fused[cid] = {"doc": doc, "rrf_score": bonus}

        # 按 RRF 分数排序
        sorted_items = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        results = []
        for item in sorted_items[:top_k]:
            doc = item["doc"]
            doc["score"] = round(item["rrf_score"], 4)
            results.append(doc)

        logger.info(f"混合检索：向量 {len(vector_results)} 条 + 关键词 {len(keyword_results)} 条 → 融合 {len(results)} 条")
        return results


# ============================================================================
# 5. 完整 RAG Pipeline
# ============================================================================
class AdvancedRAG:
    """生产级 RAG Pipeline"""

    def __init__(self):
        self.chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
        self.vector_store = VectorStore()
        self.keyword_retriever = None
        self.retriever = None
        self.all_chunks = []

    def ingest(self, documents: list[dict]):
        """导入文档：分块 → 写入向量库 → 建立关键词索引"""
        self.all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_text(doc["content"], title=doc["title"])
            self.all_chunks.extend(chunks)

        # 写入向量数据库
        self.vector_store.add_documents(self.all_chunks)

        # 建立关键词检索器
        self.keyword_retriever = KeywordRetriever(self.all_chunks)

        # 混合检索器
        self.retriever = HybridRetriever(self.vector_store, self.keyword_retriever)

        logger.info(f"导入完成：{len(documents)} 篇文档 → {len(self.all_chunks)} 个 chunks")

    def ask(self, query: str, top_k: int = 3, verbose: bool = True) -> str:
        """一次完整的 RAG Q&A"""
        if not self.retriever:
            return "请先导入文档（调用 ingest()）"

        # ---- 检索阶段 ----
        docs = self.retriever.search(query, top_k=top_k)

        if verbose:
            print(f"\n{'='*60}")
            print(f"📝 用户问题：{query}")
            print(f"{'='*60}")
            print(f"\n🔍 混合检索结果（共 {len(docs)} 条）：")
            for i, doc in enumerate(docs, 1):
                preview = doc['content'][:80].replace('\n', ' ')
                print(f"  [{i}] {doc['title']} | 分数: {doc['score']} | {preview}...")

        # ---- 生成阶段 ----
        if not docs:
            answer = "抱歉，知识库中没有找到相关信息。"
        else:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(
                    f"[文档{i}] 标题：{doc['title']}\n内容：{doc['content']}"
                )
            context = "\n\n".join(context_parts)

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个基于内部知识库回答问题的助手。优先使用提供的文档内容回答。如果文档中没有相关信息，明确告知用户。"},
                    {"role": "user", "content": f"## 知识库内容\n\n{context}\n\n## 用户问题\n\n{query}"},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            answer = response.choices[0].message.content

        if verbose:
            print(f"\n🤖 AI 回答：\n{answer}")
            print(f"{'='*60}\n")

        return answer


# ============================================================================
# 6. 运行
# ============================================================================
# 复用最小 Demo 的知识库
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))
from minimal_rag import DOCUMENTS

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🏭 生产级 RAG Demo                                ║
║                                                              ║
║  Embedding: sentence-transformers (all-MiniLM-L6-v2)        ║
║  向量数据库: ChromaDB (本地持久化)                            ║
║  分块策略: 语义分块 (500字符/块)                              ║
║  检索策略: 混合检索 (向量 + 关键词 + RRF融合)                 ║
╚══════════════════════════════════════════════════════════════╝
    """)

    rag = AdvancedRAG()
    rag.ingest(DOCUMENTS)

    test_queries = [
        "公司混合办公的政策是什么？",
        "公司的融资情况和投资方？",
        "AI Tutor支持哪些语言，多少钱？",
    ]

    for query in test_queries:
        rag.ask(query, top_k=3)
