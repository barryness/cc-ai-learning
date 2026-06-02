"""
=============================================================================
  Step 1：基础 RAG —— 理解检索增强生成的核心链路
=============================================================================

目标：用最少的代码跑通 RAG 全流程。
技术栈：PyMuPDF 读 PDF → jieba 分词 → TF-IDF 检索 → DeepSeek 生成。

与工业级 RAG 的唯一区别：检索用的是 TF-IDF（关键词匹配），而非 Embedding（语义匹配）。
原理完全相同——都是"找到相关文档 → 拼进 Prompt → LLM 生成答案"。

运行方式：
  1. cp .env.example .env  → 填入 DEEPSEEK_API_KEY
  2. pip install pymupdf jieba scikit-learn openai python-dotenv
  3. 放一个 PDF 到 data/ 目录（或使用现有 txt/md 文件）
  4. python step1_basic.py
"""

import os
import sys
from pathlib import Path

import jieba
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")


# ═══════════════════════════════════════════════════════════════════════════
# 第 1 步：加载文档
# ═══════════════════════════════════════════════════════════════════════════

def load_documents(data_dir: str = "data") -> list[dict]:
    """
    从 data/ 目录加载所有支持的文档（.pdf / .txt / .md）。

    优先读取 PDF，没有 PDF 则回退到 txt/md。
    返回: [{"title": "...", "content": "..."}, ...]
    """
    base = Path(data_dir)
    if not base.exists():
        print(f"⚠️ 目录 '{data_dir}' 不存在，使用内置示例文档")
        return _demo_docs()

    docs = []

    # 先尝试读取 PDF
    pdf_files = sorted(base.glob("*.pdf"))
    if pdf_files:
        try:
            import fitz  # PyMuPDF
            for fp in pdf_files:
                text = ""
                with fitz.open(fp) as doc:
                    for page in doc:
                        text += page.get_text()
                text = text.strip()
                if text:
                    docs.append({"title": fp.stem, "content": text})
                    print(f"  [PDF] {fp.name} → {len(text)} 字")
        except ImportError:
            print("⚠️ pymupdf 未安装，无法读取 PDF，回退到 txt/md")

    # 回退：读取 txt/md
    if not docs:
        for fp in sorted(list(base.glob("*.txt")) + list(base.glob("*.md"))):
            raw = fp.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            lines = raw.split("\n", 1)
            title = lines[0].lstrip("#").strip() if lines[0].startswith("#") else fp.stem
            content = lines[1].strip() if len(lines) > 1 else raw
            docs.append({"title": title, "content": content})
            print(f"  [{fp.suffix}] {fp.name} → {len(content)} 字")

    return docs if docs else _demo_docs()


def _demo_docs():
    """内置示例文档 —— 当没有外部文件时使用"""
    print("  使用内置示例文档")
    return [
        {"title": "公司介绍", "content": "AI学习科技有限公司成立于2024年，总部位于北京。公司专注于AI教育平台开发，已服务超过10万学员。创始人张老师曾在Google和微软工作。"},
        {"title": "产品介绍", "content": "公司旗舰产品'AI Tutor'是一个智能学习助手，支持Python、Java、Go等20+编程语言的实时辅导。产品采用订阅制，月费29元，年费299元。"},
        {"title": "技术架构", "content": "AI Tutor后端使用FastAPI框架，数据库PostgreSQL，向量数据库Milvus，LLM对接DeepSeek和OpenAI。前端React+TypeScript，部署在阿里云ACK。"},
        {"title": "融资情况", "content": "公司于2025年3月完成A轮融资5000万元，投资方包括红杉资本和创新工场。资金将用于产品研发和市场推广。"},
        {"title": "办公政策", "content": "公司实行混合办公制度，每周一三五在办公室，周二周四可远程办公。年假15天，带薪病假7天。办公地点：北京海淀区中关村科技园A座15层。"},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 第 2 步：检索器 —— 找到相关文档
# ═══════════════════════════════════════════════════════════════════════════

class TfidfRetriever:
    """
    基于 TF-IDF 的检索器。

    原理（用人话）：
    - TF（词频）：一个词在文档中出现越多，越重要
    - IDF（逆文档频率）："的""了"这种在所有文档都出现的词，权重为0
    - 每个文档被转为一个稀疏向量，向量之间的距离 = 语义相关度

    注意：中文必须先 jieba 分词再 TF-IDF，否则每个汉字是一个"词"。
    """

    def __init__(self, documents: list[dict]):
        self.docs = documents
        # 把标题和正文拼接后分词（标题的权重通过文本内容自然体现）
        tokenized = [" ".join(jieba.lcut(f"{d['title']} {d['content']}")) for d in documents]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(tokenized)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([" ".join(jieba.lcut(query))])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]
        top_idx = scores.argsort()[::-1][:top_k]
        return [
            {**self.docs[i], "score": round(float(scores[i]), 3)}
            for i in top_idx if scores[i] > 0
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 第 3 步：生成器 —— 增强 Prompt + 调用 LLM
# ═══════════════════════════════════════════════════════════════════════════

def generate_answer(query: str, docs: list[dict]) -> str:
    """将检索到的文档拼入 Prompt，调用 LLM 生成答案"""

    if not docs:
        return "抱歉，知识库中没有找到相关信息。"

    # 拼接检索到的文档
    context = "\n\n".join(
        f"[文档{i}] 标题：{d['title']}\n内容：{d['content']}\n相关度：{d['score']}"
        for i, d in enumerate(docs, 1)
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "你是一个基于内部知识库回答问题的助手。\n"
                "规则：\n"
                "1. 优先使用下面提供的文档内容回答\n"
                "2. 如果文档中没有相关信息，请说'知识库中暂无相关信息'\n"
                "3. 回答时引用文档编号，如'根据[文档1]...'\n"
                "4. 保持回答简洁准确"
            )},
            {"role": "user", "content": (
                f"## 知识库检索结果\n\n{context}\n\n"
                f"## 用户问题\n\n{query}\n\n"
                f"请基于以上知识库内容回答用户问题。"
            )},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════
# 第 4 步：组装 Pipeline + 运行
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🧠 Step 1：基础 RAG —— TF-IDF 检索 + LLM 生成")
    print("=" * 60)

    docs = load_documents("data")
    print(f"\n✅ 共加载 {len(docs)} 篇文档")

    retriever = TfidfRetriever(docs)

    queries = [
        "公司混合办公的政策是什么？",
        "公司融资了多少钱？投资方有哪些？",
        "AI Tutor产品月费多少钱？",
    ]

    for q in queries:
        print(f"\n{'─' * 50}")
        print(f"📝 用户: {q}")

        retrieved = retriever.retrieve(q, top_k=3)
        print(f"🔍 检索到 {len(retrieved)} 篇相关文档:")
        for d in retrieved:
            print(f"   [{d['score']}] {d['title']}")

        answer = generate_answer(q, retrieved)
        print(f"🤖 AI: {answer}")

    print(f"\n{'=' * 60}")
    print("✅ Step 1 完成！")
    print("接下来：step2_vector_db.py —— 用向量数据库替换 TF-IDF")
    print("=" * 60)


if __name__ == "__main__":
    main()
