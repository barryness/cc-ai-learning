"""
=============================================================================
  Step 3：多 PDF 支持 —— 批量文档知识库
=============================================================================

核心升级：单 PDF → 多 PDF + 自动分块 + 来源追溯

为什么需要分块（Chunking）：
  - Embedding 模型最大输入 512 token，50 页 PDF 远超限制
  - 整篇文档作为一个向量 → 语义被"平均"掉，检索精度极差
  - 分块后，检索返回的是"相关段落"而非整篇文档

新增能力：
  - 批量 PDF 导入：扫描 data/ 目录下所有 PDF
  - 递归分块器：按段落切 → 太长按句子切 → 按固定大小切
  - 来源追溯：每个结果标注 PDF 文件名 + 页码
  - 进度显示：大批量索引时显示进度

运行方式：
  pip install chromadb transformers torch pymupdf
  # 放 PDF 到 data/ 目录
  python step3_multi_pdf.py
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
# Embedding 函数（同 Step2，bert-base-chinese）
# ═══════════════════════════════════════════════════════════════════════════

class ChineseBertEmbedding:
    """使用 bert-base-chinese 生成 768 维句向量（Mean Pooling + L2 归一化）"""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
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
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled[0].cpu().numpy().tolist())
        return embeddings


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 核心新增：递归文本分块器
# ═══════════════════════════════════════════════════════════════════════════

class RecursiveChunker:
    """
    递归文本分块器。

    分块策略（优先级从高到低）：
      1. 先按段落（\n\n）切分
      2. 段落太长 → 按句子（。！？\n）切分
      3. 句子还太长 → 按固定大小硬切

    重叠设计：
      chunk_overlap 让相邻块有一小段重复内容，
      避免关键信息恰好被切在边界上。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    def split(self, text: str) -> list[str]:
        """递归切分文本，返回 chunk 列表"""
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """递归核心：用当前分隔符切分，超长的子块用下一级分隔符继续切"""
        if not separators:
            # 最后手段：按固定字符数硬切
            return self._split_by_size(text)

        sep = separators[0]
        next_seps = separators[1:]

        if sep == "":
            return self._split_by_size(text)

        parts = text.split(sep)

        chunks = []
        current = ""
        for part in parts:
            # 尝试把 part 追加到当前块
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                # 当前块已满，先保存
                if current:
                    chunks.append(current)
                # 如果 part 本身超长，递归用下一级分隔符切
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_recursive(part, next_seps)
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current = sub_chunks[-1]
                    else:
                        current = part
                else:
                    current = part

        if current:
            chunks.append(current)

        # 加入 overlap：每个 chunk 的末尾作为下一个 chunk 的开头
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap:]
                overlapped.append(prev_tail + chunks[i])
            return overlapped

        return chunks

    def _split_by_size(self, text: str) -> list[str]:
        """按固定大小硬切（最后手段）"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 文档加载（批量 PDF）
# ═══════════════════════════════════════════════════════════════════════════

def load_documents(data_dir: str = "data") -> list[dict]:
    """
    扫描 data/ 目录下所有 PDF，逐页提取文本。

    返回: [{"title": "文件名", "content": "文本", "source": "文件路径", "page": 页码}, ...]
    """
    base = Path(data_dir)
    if not base.exists():
        return _demo_docs()

    docs = []
    pdf_files = sorted(base.glob("*.pdf"))

    if pdf_files:
        try:
            import fitz
            for fp in pdf_files:
                with fitz.open(fp) as doc:
                    for page_num, page in enumerate(doc, 1):
                        text = page.get_text().strip()
                        if text:
                            docs.append({
                                "title": fp.stem,
                                "content": text,
                                "source": fp.name,
                                "page": page_num,
                            })
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
            docs.append({"title": title, "content": content, "source": fp.name, "page": 1})

    return docs if docs else _demo_docs()


def _demo_docs():
    return [
        {"title": "公司介绍", "content": "AI学习科技有限公司成立于2024年，总部位于北京。公司专注于AI教育平台开发，已服务超过10万学员。", "source": "demo", "page": 1},
        {"title": "产品介绍", "content": "公司旗舰产品'AI Tutor'是一个智能学习助手，支持Python、Java、Go等20+编程语言的实时辅导。产品采用订阅制，月费29元，年费299元。", "source": "demo", "page": 1},
        {"title": "技术架构", "content": "AI Tutor后端使用FastAPI框架，数据库PostgreSQL，向量数据库Milvus，LLM对接DeepSeek和OpenAI。", "source": "demo", "page": 1},
        {"title": "融资情况", "content": "公司于2025年3月完成A轮融资5000万元，投资方包括红杉资本和创新工场。", "source": "demo", "page": 1},
        {"title": "办公政策", "content": "公司实行混合办公制度，每周一三五在办公室，周二周四可远程办公。年假15天，带薪病假7天。", "source": "demo", "page": 1},
        {"title": "报销流程", "content": "员工报销通过OA系统提交，1000元以内部门经理审批，1000元以上需总监审批。每月15号统一打款。", "source": "demo", "page": 2},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 核心升级：ChunkedRetriever（带分块 + 来源追溯）
# ═══════════════════════════════════════════════════════════════════════════

class ChunkedRetriever:
    """
    支持多文档 + 自动分块的 ChromaDB 检索器。

    与 Step2 EmbeddingRetriever 的关键区别：
      Step2: 整篇文档 → 一个向量 → 精度低
      Step3: 文档 → 分块 → 每个块一个向量 → 检索返回"相关段落"
    """

    def __init__(self, documents: list[dict], persist_dir: str = "./chroma_data"):
        self.raw_docs = documents
        self.embedding_fn = ChineseBertEmbedding()
        self.chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)

        # 持久化 ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 重建 Collection
        try:
            self.client.delete_collection("knowledge_base_v3")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="knowledge_base_v3",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # 分块 + 批量索引
        self._index_documents()

    def _index_documents(self):
        """对所有文档分块并导入 ChromaDB"""
        chunk_texts = []
        chunk_metadatas = []
        chunk_ids = []

        total_pages = len(self.raw_docs)
        print(f"\n📄 共 {total_pages} 页文档，开始分块索引...")

        for doc_idx, doc in enumerate(self.raw_docs):
            chunks = self.chunker.split(doc["content"])
            source = doc.get("source", "unknown")
            page = doc.get("page", 1)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"
                chunk_texts.append(chunk)
                chunk_metadatas.append({
                    "title": doc["title"],
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_idx,
                })
                chunk_ids.append(chunk_id)

            if (doc_idx + 1) % 5 == 0 or doc_idx == total_pages - 1:
                print(f"  进度: {doc_idx + 1}/{total_pages} 页 → 已生成 {len(chunk_ids)} 个文本块")

        # 批量导入
        if chunk_texts:
            self.collection.add(
                documents=chunk_texts,
                metadatas=chunk_metadatas,
                ids=chunk_ids,
            )

        print(f"✅ 已索引 {total_pages} 页文档 → {len(chunk_ids)} 个文本块")
        print(f"   平均每页 {len(chunk_ids) / max(total_pages, 1):.1f} 个块")

    def retrieve(self, query: str, top_k: int = 3, source_filter: str = None) -> list[dict]:
        """
        语义检索，支持按来源文件过滤。

        参数:
          query: 用户问题
          top_k: 返回的 chunk 数量
          source_filter: 只搜索某个 PDF（文件名），None 表示全部
        """
        where_filter = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        docs = []
        for i, (doc_id, text, dist) in enumerate(zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            meta = results["metadatas"][0][i]
            sim = max(0, round((1 - dist) * 100))
            docs.append({
                "title": meta.get("title", ""),
                "content": text[:300],
                "score": sim,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", 0),
            })
        return docs

    def list_sources(self) -> list[str]:
        """列出知识库中所有来源文件"""
        sources = set(d.get("source", "unknown") for d in self.raw_docs)
        return sorted(sources)


# ═══════════════════════════════════════════════════════════════════════════
# 生成器（同 Step2）
# ═══════════════════════════════════════════════════════════════════════════

def generate_answer(query: str, docs: list[dict]) -> str:
    if not docs:
        return "抱歉，知识库中没有找到相关信息。"

    context = "\n\n".join(
        f"[文档{i}] 来源：{d['source']} (第{d['page']}页)\n标题：{d['title']}\n内容：{d['content']}\n语义相似度：{d['score']}%"
        for i, d in enumerate(docs, 1)
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "你是一个基于内部知识库回答问题的助手。规则：1. 优先使用下面提供的文档内容回答 2. 如果文档中没有相关信息，请说'知识库中暂无相关信息' 3. 回答时引用文档编号和来源"},
            {"role": "user", "content": f"## 知识库检索结果\n\n{context}\n\n## 用户问题\n\n{query}\n\n请基于以上知识库内容回答用户问题。"},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════
# 分块效果演示
# ═══════════════════════════════════════════════════════════════════════════

def demo_chunking():
    """演示分块器如何工作"""
    sample = (
        "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。"
        "它的核心思想是：在LLM生成答案之前，先从外部知识库中检索相关文档，"
        "然后将这些文档作为上下文提供给LLM，从而生成更准确、更有依据的答案。"
        "\n\n"
        "RAG的工作流程分为三个阶段：\n"
        "1. 检索（Retrieval）：根据用户问题，从知识库中找到最相关的文档\n"
        "2. 增强（Augmentation）：将检索到的文档和用户问题拼接成增强Prompt\n"
        "3. 生成（Generation）：LLM基于增强Prompt生成最终答案\n"
        "\n\n"
        "为什么需要RAG？首先，LLM的训练数据有截止日期，无法回答最新的问题。"
        "其次，LLM会产生幻觉，即编造不存在的事实。"
        "最重要的是，企业有自己的私有文档，LLM从未见过这些内容。"
        "RAG通过将外部知识注入Prompt，有效解决了这三个问题。"
    )

    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.split(sample)

    print("\n" + "=" * 60)
    print("📐 分块器演示 (chunk_size=200, overlap=30)")
    print("=" * 60)
    print(f"原文: {len(sample)} 字 → {len(chunks)} 个块\n")
    for i, chunk in enumerate(chunks, 1):
        # 高亮 overlap 部分（前 30 字=上一块的末尾）
        if i > 1 and len(chunk) > 30:
            print(f"  [块{i}] ({len(chunk)}字)")
            print(f"    ...{chunk[-50:]}")
        else:
            print(f"  [块{i}] ({len(chunk)}字)")
            print(f"    {chunk[:100]}...")
        if i < len(chunks):
            print(f"    ↓ overlap: {chunk[-30:]}\n")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  📚 Step 3：多 PDF 支持 —— 批量文档 + 自动分块")
    print("=" * 60)

    # 分块演示
    demo_chunking()

    # 加载文档
    docs = load_documents("data")
    print(f"\n✅ 共加载 {len(docs)} 页文档")

    # 建索引
    retriever = ChunkedRetriever(docs)

    # 列出知识库来源
    sources = retriever.list_sources()
    print(f"\n📂 知识库来源文件 ({len(sources)} 个):")
    for s in sources:
        page_count = sum(1 for d in docs if d.get("source") == s)
        print(f"   - {s} ({page_count} 页)")

    # RAG 问答
    queries = [
        "公司混合办公的政策是什么？",
        "公司的技术架构是怎样的？",
        "报销流程是什么？需要谁审批？",
    ]

    print("\n" + "=" * 60)
    print("💬 RAG 问答（含来源追溯）")
    print("=" * 60)

    for q in queries:
        print(f"\n📝 用户: {q}")
        retrieved = retriever.retrieve(q, top_k=3)
        for i, d in enumerate(retrieved, 1):
            print(f"   [{i}] {d['source']} 第{d['page']}页 块#{d['chunk_index']} (语义相似度: {d['score']}%)")
        answer = generate_answer(q, retrieved)
        print(f"🤖 AI: {answer}")

    print(f"\n{'=' * 60}")
    print("✅ Step 3 完成！")
    print("接下来：step4_chat_memory.py —— 多轮对话记忆")
    print("=" * 60)


if __name__ == "__main__":
    main()
