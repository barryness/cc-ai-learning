"""
=============================================================================
  Step 4：聊天记忆 —— 多轮对话上下文
=============================================================================

核心升级：单轮 Q&A → 多轮对话（带记忆）

为什么需要聊天记忆：
  用户: "AI Tutor月费多少钱？"
  AI:   "月费29元。"
  用户: "那年费呢？"              ← 没提"AI Tutor"，依赖上下文
  AI:   "年费299元。"             ← 需要"记住"上一轮在聊 AI Tutor

如果没有记忆，LLM 看到孤立的"那年费呢？"会完全不知所云。

新增能力：
  - 对话历史管理：每轮自动记录 user + assistant 消息
  - 上下文理解：支持"那价格呢？"这类省略主语的追问
  - Token 预算管理：限制历史长度，避免超出 Context Window
  - /clear 命令：一键清空对话历史
  - 交互模式：持续对话，而非一次性 Q&A

运行方式：
  pip install chromadb transformers torch
  python step4_chat_memory.py
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")


# ═══════════════════════════════════════════════════════════════════════════
# Embedding 函数（同 Step2/3）
# ═══════════════════════════════════════════════════════════════════════════

class ChineseBertEmbedding:
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
# 检索器（同 Step2，简化为直接使用 ChromaDB）
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingRetriever:
    def __init__(self, documents: list[dict], persist_dir: str = "./chroma_data_v4"):
        self.docs = documents
        self.embedding_fn = ChineseBertEmbedding()

        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            self.client.delete_collection("knowledge_base_v4")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="knowledge_base_v4",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self.collection.add(
            documents=[f"{d['title']}\n{d['content']}" for d in documents],
            metadatas=[{"title": d["title"]} for d in documents],
            ids=[f"doc_{i}" for i in range(len(documents))],
        )
        print(f"✅ 已索引 {len(documents)} 篇文档")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = []
        for i, (doc_id, text, dist) in enumerate(zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )):
            meta = results["metadatas"][0][i]
            sim = max(0, round((1 - dist) * 100))
            docs.append({
                "title": meta.get("title", doc_id),
                "content": text[:300],
                "score": sim,
            })
        return docs


# ═══════════════════════════════════════════════════════════════════════════
# 文档加载
# ═══════════════════════════════════════════════════════════════════════════

def load_documents(data_dir: str = "data") -> list[dict]:
    base = Path(data_dir)
    if not base.exists():
        return _demo_docs()
    docs = []

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
# Step 4 核心升级：ChatSession —— 带记忆的对话管理
# ═══════════════════════════════════════════════════════════════════════════

class ChatSession:
    """
    多轮对话管理器。

    核心机制：
      1. 每轮对话后，把 user 问题和 assistant 回答存入 history
      2. 下一轮把 history 和当前问题一起发给 LLM
      3. LLM 看到完整上下文，就能理解"那价格呢？"指的是什么

    Token 预算管理：
      max_history_turns 限制保留的历史轮数，防止超出 LLM 的 Context Window。
      超出后，自动丢弃最早的对话（FIFO）。
    """

    def __init__(self, retriever: EmbeddingRetriever, max_history_turns: int = 10):
        self.retriever = retriever
        self.max_history_turns = max_history_turns
        self.history: list[dict] = []

    def ask(self, query: str, top_k: int = 3) -> str:
        """处理一次对话，自动管理历史和生成回答"""

        # --- 检索（可以结合历史做查询改写，Step5 会涉及） ---
        retrieved = self.retriever.retrieve(query, top_k=top_k)

        # --- 构建增强 Prompt（含历史对话） ---
        answer = self._generate_with_history(query, retrieved)

        # --- 更新历史 ---
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})

        # --- 截断历史 ---
        self._trim_history()

        return answer, retrieved

    def _generate_with_history(self, query: str, docs: list[dict]) -> str:
        """把检索结果 + 历史对话 + 当前问题一起发给 LLM"""

        if not docs:
            return "抱歉，知识库中没有找到相关信息。"

        # 检索到的文档
        context = "\n\n".join(
            f"[文档{i}] 标题：{d['title']}\n内容：{d['content']}\n语义相似度：{d['score']}%"
            for i, d in enumerate(docs, 1)
        )

        system_prompt = (
            "你是一个基于内部知识库回答问题的助手。\n"
            "规则：\n"
            "1. 优先使用下面提供的文档内容回答\n"
            "2. 如果文档中没有相关信息，请说'知识库中暂无相关信息'\n"
            "3. 回答时引用文档编号\n"
            "4. 如果用户的问题是追问（如'那价格呢？''还有什么功能？'），"
            "请结合对话历史理解用户真正想问什么"
        )

        # 构建 messages：system + 历史 + 当前
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        messages.append({
            "role": "user",
            "content": (
                f"## 知识库检索结果\n\n{context}\n\n"
                f"## 用户问题\n\n{query}\n\n"
                f"请基于以上知识库内容回答用户问题。"
            ),
        })

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=messages,
            max_tokens=500,
        )
        return response.choices[0].message.content

    def _trim_history(self):
        """截断历史，只保留最近的 max_history_turns 轮"""
        max_messages = self.max_history_turns * 2  # user + assistant = 一轮
        if len(self.history) > max_messages:
            removed = len(self.history) - max_messages
            self.history = self.history[-max_messages:]
            print(f"  ⚠️ 历史已满，移除了最早的 {removed} 条消息")

    def clear(self):
        """清空对话历史"""
        count = len(self.history) // 2
        self.history = []
        return count

    @property
    def turn_count(self) -> int:
        return len(self.history) // 2


# ═══════════════════════════════════════════════════════════════════════════
# 交互式对话
# ═══════════════════════════════════════════════════════════════════════════

def interactive_mode(chat: ChatSession):
    """交互式多轮对话模式"""
    print("\n" + "=" * 60)
    print("💬 进入交互对话模式")
    print("   命令：")
    print("     /clear  - 清空对话历史")
    print("     /history - 查看对话历史")
    print("     /exit   - 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not query:
            continue

        if query == "/exit":
            print("👋 再见！")
            break

        if query == "/clear":
            cleared = chat.clear()
            print(f"✅ 已清空 {cleared} 轮对话历史")
            continue

        if query == "/history":
            if not chat.history:
                print("  (无对话历史)")
            else:
                for i in range(0, len(chat.history), 2):
                    user_msg = chat.history[i]
                    assistant_msg = chat.history[i + 1] if i + 1 < len(chat.history) else None
                    print(f"\n  第{i//2 + 1}轮:")
                    print(f"    🧑: {user_msg['content'][:80]}...")
                    if assistant_msg:
                        print(f"    🤖: {assistant_msg['content'][:80]}...")
            continue

        answer, retrieved = chat.ask(query)
        print(f"\n🤖 AI: {answer}")
        print(f"   (本轮检索到 {len(retrieved)} 篇文档, 历史 {chat.turn_count} 轮)")


# ═══════════════════════════════════════════════════════════════════════════
# 多轮对话演示（自动模式）
# ═══════════════════════════════════════════════════════════════════════════

def demo_multi_turn(chat: ChatSession):
    """演示多轮对话的理解能力"""
    print("\n" + "=" * 60)
    print("🎭 多轮对话演示：上下文理解")
    print("=" * 60)

    conversation = [
        "AI Tutor产品月费多少钱？",
        "那年费呢？",            # 省略了主语"AI Tutor"
        "如果我按年订阅，比按月省多少？",  # 需要结合前两轮的回答做数学
        "公司办公政策是怎样的？",
        "远程是哪几天？",        # 省略了"办公政策"上下文
    ]

    for query in conversation:
        print(f"\n🧑 用户: {query}")
        answer, retrieved = chat.ask(query)
        print(f"🤖 AI: {answer}")

    print(f"\n📊 对话统计: 共 {chat.turn_count} 轮")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  💬 Step 4：聊天记忆 —— 多轮对话上下文")
    print("=" * 60)

    docs = load_documents("data")
    print(f"✅ 共加载 {len(docs)} 篇文档\n")

    retriever = EmbeddingRetriever(docs)
    chat = ChatSession(retriever, max_history_turns=10)

    # 自动演示多轮对话
    demo_multi_turn(chat)

    print(f"\n{'=' * 60}")
    print("✅ Step 4 完成！")
    print("接下来：step5_optimize_recall.py —— 混合检索 + 重排序")
    print("=" * 60)

    # 取消下面注释以进入交互模式：
    # chat.clear()
    # interactive_mode(chat)


if __name__ == "__main__":
    main()
