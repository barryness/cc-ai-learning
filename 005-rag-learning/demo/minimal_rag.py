"""
=============================================================================
  最小可运行 RAG Demo — 从零理解 Retrieval-Augmented Generation
=============================================================================

一句话：RAG = 先从知识库"检索"相关文档，再把文档和问题一起给 LLM 生成答案。

工作流程：
  用户提问 → 向量检索 → 找到最相关文档 → 拼接 Prompt → LLM 生成答案

为什么需要 RAG？
  1. LLM 训练数据有截止日期，不知道"今天"的事
  2. LLM 会产生幻觉（编造不存在的事实）
  3. 企业有自己的私有文档，LLM 没见过
  → RAG 把"外部知识"注入 Prompt，让 LLM 基于事实回答

运行方式：
  1. cp .env.example .env  → 填入你的 API Key
  2. pip install -r requirements.txt
  3. python3 minimal_rag.py
"""

import os
import numpy as np
import jieba
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 第1步：加载配置
# ============================================================================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-chat")


# ============================================================================
# 第2步：知识库 — 这就是 RAG 的"外部知识"
# ============================================================================
# 实际项目中，这些文档来自你的数据库、PDF、网页等
# 这里用硬编码的文档模拟一个"公司内部知识库"
DOCUMENTS = [
    {
        "title": "公司介绍",
        "content": "AI学习科技有限公司成立于2024年，总部位于北京。"
                   "公司专注于AI教育平台开发，已服务超过10万学员。"
                   "创始人张老师曾在Google和微软工作。",
    },
    {
        "title": "产品介绍",
        "content": "公司旗舰产品'AI Tutor'是一个智能学习助手，"
                   "支持Python、Java、Go等20+编程语言的实时辅导。"
                   "产品采用订阅制，月费29元，年费299元。",
    },
    {
        "title": "技术架构",
        "content": "AI Tutor后端使用FastAPI框架，数据库使用PostgreSQL，"
                   "向量数据库使用Milvus，LLM服务对接DeepSeek和OpenAI。"
                   "前端使用React+TypeScript，部署在阿里云ACK集群。",
    },
    {
        "title": "融资情况",
        "content": "公司于2025年3月完成A轮融资5000万元，"
                   "投资方包括红杉资本和创新工场。"
                   "资金将用于产品研发和市场推广。",
    },
    {
        "title": "办公政策",
        "content": "公司实行混合办公制度，每周一三五需要在办公室，"
                   "周二周四可远程办公。年假15天，带薪病假7天。"
                   "办公地点：北京海淀区中关村科技园A座15层。",
    },
    {
        "title": "Python函数式编程",
        "content": "map() 函数接收一个函数和一个可迭代对象，"
                   "返回一个迭代器，将函数应用到每个元素上。"
                   "filter() 过滤满足条件的元素。"
                   "reduce() 在functools模块中，累积计算结果。",
    },
    {
        "title": "机器学习基础",
        "content": "监督学习需要标注数据，无监督学习不需要标注。"
                   "过拟合指模型在训练集表现好、测试集表现差。"
                   "交叉验证是评估模型泛化能力的常用方法。",
    },
    {
        "title": "vibe coding工具",
        "content": "现在使用最强大的是Claude code"
                   "公认最好用的codex"
                   "以及opencode",
    },
    {
        "title": "AI中的养龙虾",
        "content": "养龙虾通常说的是openclaw",
    },
    {
        "title": "目前大数据实时处理技术使用的工具",
        "content": "多用flink,spark也有不少企业在使用",
    }
]


# ============================================================================
# 第3步：构建检索器（Retriever）— RAG 的核心组件
# ============================================================================
class SimpleRetriever:
    """
    最简检索器：用 TF-IDF 把文档转成向量，用余弦相似度做检索。

    实际项目中，TF-IDF 会被替换为 Embedding 模型（如 text-embedding-3-small），
    用向量数据库（如 Chroma/Milvus）存储和检索。但原理完全相同！
    """

    def __init__(self, documents: list[dict]):
        """
        初始化检索器：
        - 保存原始文档列表
        - 用 TF-IDF 把所有文档内容转成向量矩阵

        TF-IDF 原理（用人话）：
          - TF（词频）：一个词在文档中出现越多，越重要
          - IDF（逆文档频率）：一个词在所有文档都出现（如"的""了"），就不重要
          - TF×IDF = 词的权重
        """
        self.documents = documents                           # 保存原始文档
        self.contents = [doc["content"] for doc in documents] # 提取所有文档正文
        self.titles = [doc["title"] for doc in documents]     # 提取所有文档标题

        # 标题 + 正文 一起索引，标题往往包含最重要的关键词
        # 例："vibe coding工具" 在标题里，但正文没提 → 不加标题就检索不到
        full_texts = [f"{doc['title']} {doc['content']}" for doc in documents]

        # 关键：中文需要先分词再 TF-IDF
        # jieba 把 "公司成立于2024年" → ["公司", "成立", "于", "2024", "年"]
        # 用空格连起来，TF-IDF 就能按空格正确切词了
        tokenized = [" ".join(jieba.lcut(t)) for t in full_texts]

        # TF-IDF 向量化器：把文本变成数值向量
        # 文本 → 分词 → 计算 TF-IDF 权重 → 得到一个向量 [0.0, 0.5, 0.0, 0.3, ...]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(tokenized)
        # self.doc_vectors 是一个稀疏矩阵，shape = (N篇文档, 词汇表大小)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        检索：接收用户问题，返回最相关的 top_k 篇文档。

        流程：
          1. 把问题也转成 TF-IDF 向量（和文档用同一个 vectorizer）
          2. 计算问题向量 与 每篇文档向量 的余弦相似度
          3. 相似度从高到低排序，取前 top_k 篇
        """
        # 步骤1：问题转向量（也要先 jieba 分词，和文档用同样的处理）
        query_tokenized = " ".join(jieba.lcut(query))
        query_vector = self.vectorizer.transform([query_tokenized])

        # 步骤2：计算余弦相似度
        # cosine_similarity 返回一个矩阵，每个值在 [0, 1] 之间
        # 1.0 = 完全相关，0.0 = 毫不相关

        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        # similarities 是一个数组 [0.12, 0.05, 0.0, 0.08, 0.02, 0.0, 0.0]
        # 每个值代表 query 对应该文档的相似度

        # 步骤3：排序取 top_k
        # argsort 返回从小到大排序的索引，[::-1] 反转成从大到小
        top_indices = similarities.argsort()[::-1][:top_k]

        # 步骤4：组装返回结果
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0:  # 只返回有相关性的文档（相似度 > 0）
                results.append({
                    "title": self.titles[idx],
                    "content": self.contents[idx],
                    "score": round(float(score), 4),
                })
        return results


# ============================================================================
# 第4步：构建生成器（Generator）—— 调用 LLM
# ============================================================================
def generate_answer(query: str, retrieved_docs: list[dict], history: list[dict] = None) -> str:
    """
    将检索到的文档拼入 Prompt，让 LLM 基于它们回答。

    这是 RAG 最关键的一步：
      - 没有检索文档 → LLM 可能瞎编
      - 有了检索文档 → LLM "有据可查"，减少幻觉

    参数：
      history: 历史对话，格式 [{"role": "user", "content": "..."},
                               {"role": "assistant", "content": "..."}, ...]
    """

    if history is None:
        history = []

    # 如果没有检索到相关文档，让 LLM 老实说不知道
    if not retrieved_docs:
        return "抱歉，知识库中没有找到相关信息。"

    # ------ 拼接检索到的文档 ------
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"[文档{i}] 标题：{doc['title']}\n"
            f"内容：{doc['content']}\n"
            f"相关度：{doc['score']}"
        )
    context = "\n\n".join(context_parts)

    # ------ 构造 Prompt ------
    system_prompt = (
        "你是一个基于内部知识库回答问题的助手。\n"
        "规则：\n"
        "1. 优先使用下面提供的文档内容回答\n"
        "2. 如果文档中没有相关信息，请明确说'知识库中暂无相关信息'\n"
        "3. 回答时引用文档编号，如'根据[文档1]...'\n"
        "4. 保持回答简洁准确\n"
        "5. 如果用户的问题是追问（如'那价格呢'），结合历史对话理解上下文"
    )

    # 当前轮的用户消息：检索到的文档 + 用户问题
    current_message = (
        f"## 知识库检索结果\n\n{context}\n\n"
        f"## 用户问题\n\n{query}\n\n"
        f"请基于以上知识库内容回答用户问题。"
    )

    # ------ 组装 messages：system + 历史对话 + 当前问题 ------
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)                                  # 历史对话（多轮 user+assistant）
    messages.append({"role": "user", "content": current_message})  # 当前问题

    # ------ 调用 LLM ------
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )

    return response.choices[0].message.content


# ============================================================================
# 第5步：组装完整的 RAG Pipeline
# ============================================================================
class RAGPipeline:
    """
    完整 RAG Pipeline = 检索器 + 生成器

    数据流：
      用户提问 → Retriever.retrieve() → 相关文档列表
              → generate_answer() → LLM 回答（含历史对话）
    """

    def __init__(self, documents: list[dict]):
        self.retriever = SimpleRetriever(documents)
        self.history = []  # 多轮对话历史：[{"role": "user", "content": "..."}, ...]

    def ask(self, query: str, top_k: int = 3, verbose: bool = True) -> str:
        """
        一次完整的 RAG Q&A，支持多轮对话。

        参数：
          query:  用户问题
          top_k:  检索返回的文档数
          verbose: 是否打印检索过程（调试用）
        """
        # ---- 检索阶段 ----
        docs = self.retriever.retrieve(query, top_k=top_k)

        if verbose:
            print(f"\n{'='*60}")
            print(f"📝 用户问题：{query}")
            print(f"{'='*60}")
            if self.history:
                print(f"💬 历史对话：已记住 {len(self.history)//2} 轮")
            print(f"\n🔍 检索结果（共 {len(docs)} 条相关文档）：")
            for i, doc in enumerate(docs, 1):
                print(f"  [{i}] {doc['title']}  (相关度: {doc['score']})")

        # ---- 生成阶段 —— 传入历史对话 ----
        answer = generate_answer(query, docs, history=self.history)

        # ---- 更新历史：把本轮问答追加进去 ----
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})

        if verbose:
            print(f"\n🤖 AI 回答：")
            print(f"{answer}")
            print(f"{'='*60}\n")

        return answer

    def clear_history(self):
        """清空对话历史，开始新一轮对话"""
        self.history = []
        print("对话历史已清空。")


# ============================================================================
# 第6步：运行 Demo
# ============================================================================
if __name__ == "__main__":
    # 创建 RAG Pipeline
    rag = RAGPipeline(DOCUMENTS)

    print("""
╔══════════════════════════════════════════════════════════════╗
║               🚀 最小 RAG Demo —— 开工！                     ║
║                                                              ║
║  这是一个简化的 RAG 系统，展示核心原理。                       ║
║  检索器：TF-IDF + 余弦相似度（工业级会用 Embedding 模型）       ║
║  生成器：DeepSeek/OpenAI LLM                                 ║
║  知识库：10 篇公司内部文档                                     ║
║  新功能：多轮对话记忆                                         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # ====== 单轮测试 ======
    print("\n" + "─" * 60)
    print("📋 单轮问答测试")
    print("─" * 60)

    test_queries = [
        "公司混合办公的政策是什么？",
        "公司融资了多少钱？投资方有哪些？",
        "Python的map函数是做什么的？",
        "vibe coding工具有哪些？",
    ]

    for query in test_queries:
        rag.ask(query, top_k=5)

    # ====== 多轮对话测试 ======
    print("\n" + "─" * 60)
    print("💬 多轮对话测试（演示历史记忆）")
    print("─" * 60)

    # 清空之前的单轮历史，开始一次新的多轮对话
    rag.clear_history()

    # 第1轮：问一个完整问题
    q1 = "AI Tutor产品月费多少钱？"
    rag.ask(q1, top_k=5)

    # 第2轮：追问——这里没有再次提"AI Tutor"，依赖 LLM 从历史中理解
    q2 = "那年费呢？"
    rag.ask(q2, top_k=5)

    # 第3轮：再追问——进一步推理
    q3 = "如果我按年订阅，比按月省多少？"
    rag.ask(q3, top_k=5)

    print("✅ Demo 运行完毕！")
    print("接下来：阅读 README.md → 理解生产级架构 → 做练习题")
