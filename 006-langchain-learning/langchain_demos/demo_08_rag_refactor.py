"""
=============================================================================
  Demo 08：用 LangChain 重构 RAG 系统
=============================================================================

对比：不用 LangChain 的 RAG（step3） vs 用 LangChain 的 RAG

关键发现：LangChain 能消除多少代码？

  手写版 (step3_multi_pdf.py):
    - 自己写 RecursiveChunker
    - 自己管理 ChromaDB Client/Collection
    - 自己拼接 Prompt 字符串
    - 自己解析检索结果（深层 dict 取值）
    - 自己组装 messages → LLM → 提取 answer
    → 约 300 行

  LangChain 版 (本文件):
    - RecursiveCharacterTextSplitter：一行代码
    - Chroma.from_documents()：自动 embed + 建索引
    - ChatPromptTemplate：模板和变量分离
    - retriever.invoke()：统一接口，返回 list[Document]
    - chain = prompt | llm | parser：声明式管道
    → 约 120 行（减少了 60%）

运行方式：
  python demo_08_rag_refactor.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# LangChain 版 RAG 完整实现
# ═══════════════════════════════════════════════════════════════════════════

def build_langchain_rag():
    """用 LangChain 构建完整的 RAG Pipeline"""

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # ── 1. 自定义 Embedding（和之前一样） ──
    class ChineseBertEmbedding:
        def __init__(self):
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
            self.model = AutoModel.from_pretrained("bert-base-chinese", local_files_only=True)
            self.model.eval()

        def _embed(self, texts):
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

        def embed_documents(self, texts):
            return self._embed(texts)

        def embed_query(self, text):
            return self._embed([text])[0]

    # ── 2. 文档加载 ──
    docs = _load_documents("data")
    print(f"✅ 加载 {len(docs)} 篇文档")

    # ── 3. 分块（LangChain 内置！不需要自己写 RecursiveChunker） ──
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    lc_docs = []
    for d in docs:
        chunks = text_splitter.split_text(d["content"])
        for i, chunk in enumerate(chunks):
            lc_docs.append(Document(
                page_content=chunk,
                metadata={"title": d["title"], "source": d.get("source", ""), "page": d.get("page", 1)},
            ))
    print(f"✅ 分块完成: {len(lc_docs)} 个文本块")

    # ── 4. 向量化 + 存 Chroma（一行代码！） ──
    embeddings = ChineseBertEmbedding()
    vectorstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        collection_name="rag_langchain",
        persist_directory="./chroma_rag_lc",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"✅ 向量索引完成 (ChromaDB + HNSW)")

    # ── 5. LLM ──
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.2)

    # ── 6. RAG Chain（关键：这是整个 Pipeline 的核心） ──
    # 声明式描述数据流：
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个基于内部知识库回答问题的助手。\n"
            "规则：\n"
            "1. 优先使用下面提供的文档内容回答\n"
            "2. 如果文档中没有相关信息，请说'知识库中暂无相关信息'\n"
            "3. 回答时引用来源"
        )),
        ("user", (
            "## 知识库检索结果\n\n"
            "{context}\n\n"
            "## 用户问题\n\n"
            "{question}\n\n"
            "请基于以上知识库内容回答用户问题。"
        )),
    ])

    # 辅助函数：把检索到的 Document 列表拼成字符串
    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[文档{i}] 来源：{d.metadata.get('source', '?')} (第{d.metadata.get('page', '?')}页)\n"
            f"标题：{d.metadata.get('title', '?')}\n"
            f"内容：{d.page_content[:300]}"
            for i, d in enumerate(docs, 1)
        )

    # 完整的 RAG Chain
    # 数据流：question → retriever → format_docs → context → prompt → llm → parser → answer
    rag_chain = (
        {
            "context": retriever | format_docs,     # 检索 + 格式化
            "question": RunnablePassthrough(),       # 原样传递用户问题
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ═══════════════════════════════════════════════════════════════════════════
# 文档加载（复用 step3 的逻辑）
# ═══════════════════════════════════════════════════════════════════════════

def _load_documents(data_dir: str = "data") -> list[dict]:
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
                            docs.append({"title": fp.stem, "content": text, "source": fp.name, "page": page_num})
        except ImportError:
            pass

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
        {"title": "报销流程", "content": "员工报销通过OA系统提交，1000元以内部门经理审批，1000元以上需总监审批。每月15号统一打款。交通补贴每月500元。", "source": "demo", "page": 2},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 对比演示
# ═══════════════════════════════════════════════════════════════════════════

def show_comparison():
    """展示手写版 vs LangChain 版的关键区别"""
    print("\n" + "=" * 60)
    print("📊 代码对比：手写 RAG vs LangChain RAG")
    print("=" * 60)

    comparisons = [
        ("分块器", "手写 RecursiveChunker 类（~50行）", "RecursiveCharacterTextSplitter()（1行）"),
        ("向量化+存储", "chromadb.Client() + create_collection() + add()（~15行）", "Chroma.from_documents()（1行）"),
        ("Prompt 构建", "f-string 拼接 + 手动组装 messages（~20行）", "ChatPromptTemplate.from_messages()（声明式）"),
        ("检索", "collection.query() → 深层 dict 取值（~15行）", "retriever.invoke() → list[Document]（1行）"),
        ("Pipeline", "手写函数调用链（~30行）", "RunnablePassthrough | prompt | llm | parser（4行）"),
    ]

    for item, manual, lc in comparisons:
        print(f"\n  {item}:")
        print(f"    手写: {manual}")
        print(f"    LC:   {lc}")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🚀 Demo 08：用 LangChain 重构 RAG 系统")
    print("=" * 60)

    # 代码量对比
    show_comparison()

    # 构建 RAG
    print("\n" + "=" * 60)
    print("  构建 LangChain RAG Pipeline")
    print("=" * 60)

    rag_chain, retriever = build_langchain_rag()

    # 问答测试
    queries = [
        "公司混合办公的政策是什么？",
        "AI Tutor产品价格多少？",
        "报销流程是什么？需要谁审批？",
    ]

    print("\n" + "=" * 60)
    print("  💬 RAG 问答")
    print("=" * 60)

    for q in queries:
        print(f"\n📝 用户: {q}")
        # 先展示检索结果
        docs = retriever.invoke(q)
        for i, d in enumerate(docs, 1):
            print(f"   [{i}] {d.metadata.get('title', '?')} — {d.metadata.get('source', '?')}")

        # RAG Chain 一站式调用
        answer = rag_chain.invoke(q)  # 注意：直接传字符串！RunnablePassthrough 透传
        print(f"🤖 AI: {answer}")

    print(f"\n{'=' * 60}")
    print("✅ Demo 08 完成！")
    print("\n📊 总结：LangChain 把 RAG 代码从 ~300 行减少到 ~120 行（-60%）")
    print("最关键的一行代码：")
    print('  rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser)')
    print("  这就是 LangChain 的声明式哲学：描述数据流，而不是写执行步骤。")
    print("=" * 60)


if __name__ == "__main__":
    main()
