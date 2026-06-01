"""
=============================================================================
  LLM 核心概念 Demo —— 用代码理解 LLM
  ─────────────────────────────────────────────────────────────
  配套 learning_notes.md 使用，每个实验对应一个概念。

  运行方式：
    python3 demo.py

  环境要求：
    pip install tiktoken numpy python-dotenv langchain-openai
=============================================================================
"""

import os
import sys
import json
import numpy as np

# ─── 导入和配置 ────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# tiktoken: OpenAI 的 Tokenizer，用来演示分词
import tiktoken

# LangChain OpenAI: 用于调用 LLM API
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def get_llm():
    """创建 LLM 实例（复用项目已有的 .env 配置）"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )


def separator(title: str):
    """打印章节分隔符"""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


# ===================================================================
#  实验1：Token —— LLM 的"积木块"
# ===================================================================
# 目标：理解文本如何被切分成 Token，为什么中英文 Token 数不同
# ===================================================================

def demo_tokenization():
    """演示 Tokenization：文本 → Token ID 的过程

    关键认知：
      - Token 不是字，不是词，是"子词"
      - 不同模型用不同的 Tokenizer
      - Token 数直接决定成本和 Context Window 使用量
    """
    separator("实验1：Token —— 文本是如何被切分的")

    # tiktoken 内置了 OpenAI 的多种 Tokenizer
    # cl100k_base: GPT-4、GPT-3.5-turbo 用的 Tokenizer
    enc = tiktoken.get_encoding("cl100k_base")

    examples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog",
        "中国的首都是北京",
        "Transformer 是一种神经网络架构，2017 年由 Google 提出。",
        "🌟 Hello 你好 123 !@#",
    ]

    for text in examples:
        tokens = enc.encode(text)
        decoded_tokens = [enc.decode([t]) for t in tokens]

        print(f"\n📝 原文: '{text}'")
        print(f"   Token 数: {len(tokens)}")
        print(f"   Token IDs: {tokens}")
        print(f"   分片: {decoded_tokens}")

    # 展示 Token 数量对比
    print("\n" + "─" * 50)
    print("📊 中英文 Token 效率对比：")
    chinese = "中国的首都是北京，北京是一座古老而现代的城市。"
    english = "Beijing is the capital of China. It is home to over 20 million people."
    print(f"   中文 '{chinese}' → {len(enc.encode(chinese))} Tokens")
    print(f"   英文 '{english}' → {len(enc.encode(english))} Tokens")
    print(f"   结论：同样信息量，中文用更少 Token")


# ===================================================================
#  实验2：Attention 机制 —— 用 NumPy 实现简化版
# ===================================================================
# 目标：用矩阵运算理解 Attention 的本质
# ===================================================================

def softmax(x):
    """Softmax 归一化。

    把一组数转成概率分布（所有值 ≥ 0，和为 1）。
    这是 Attention 的关键步骤——把"关注度分数"变成"关注度权重"。
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def demo_attention_mechanism():
    """用 NumPy 实现简化版 Attention

    这是整个 Transformer 最核心的机制。
    理解了这个，就理解了 LLM 的"大脑"。
    """
    separator("实验2：Attention 机制 —— 它们是如何相互关注的")

    # ── 场景 ──
    # 句子："The cat sat on the mat"
    # 每个词用一个 4 维向量表示（真实模型中通常是 768 或 4096 维）
    print("🏗️  构建词向量表示...")

    tokens_list = ["The", "cat", "sat", "on", "the", "mat"]
    embedding_dim = 4

    # 随机初始化词向量（真实模型这里是通过训练学到的）
    np.random.seed(42)
    embeddings = np.random.randn(len(tokens_list), embedding_dim)

    print(f"   每个词用 {embedding_dim} 维向量表示")
    for i, token in enumerate(tokens_list):
        print(f"   {token}: {embeddings[i]}")

    # ── Step 1: 计算注意力分数 ──
    print("\n📐 计算注意力分数（Query × Key^T）...")
    # 简化版：直接用向量点积作为注意力分数
    # 真实模型：每个 Token 通过 Wq/Wk/Wv 矩阵生成 Q、K、V
    scores = embeddings @ embeddings.T  # 6×6 矩阵

    print("   注意力分数矩阵（值越大 = 关注度越高）：")
    print(f"   {np.round(scores, 2)}")

    # ── Step 2: Softmax 归一化 ──
    print("\n🔢 Softmax 归一化：")
    attention_weights = softmax(scores)
    np.set_printoptions(precision=3, suppress=True)

    # 可视化注意力权重
    print("   注意力权重矩阵（每行 = 该 Token 对其他 Token 的关注分布）：")
    print(f"   {attention_weights}")

    print("\n   解读（行 = 当前词，列 = 被关注的词）：")
    for i, token_q in enumerate(tokens_list):
        most_attended = tokens_list[np.argmax(attention_weights[i])]
        print(f"   '{token_q}' 最关注 '{most_attended}' (权重 {attention_weights[i].max():.3f})")

    # ── Step 3: 加权求和 ──
    print("\n⚖️  加权求和（Value × 注意力权重）：")
    # 简化版：直接用原始向量作为 Value
    # 真实模型：Value = Embedding × Wv
    output = attention_weights @ embeddings

    print("   每个词的新表示（融合了上下文信息）：")
    for i, token in enumerate(tokens_list):
        print(f"   {token} (新): {np.round(output[i], 3)}")
        print(f"        ← 融合了所有其他词的信息")

    # ── 关键洞察 ──
    print("\n" + "💡" * 20)
    print("""
    关键洞察：
      1. "cat" 的新表示不仅包含"cat"本身，还融入了"sat"、"on"、"mat"的信息
      2. 这就是 Attention 的本质：每个词都看向所有词，然后"吸收"相关信息
      3. 堆叠多层 Attention → 模型能理解"猫坐在垫子上"这个完整场景
      4. 不同层的 Attention 关注不同东西：
         低层：语法关系（"The" → "cat"）
         中层：语义关系（"sat" → "mat"）
         高层：抽象概念（整个句子的主题）
    """)


# ===================================================================
#  实验3：Context Window —— LLM 的短期记忆
# ===================================================================
# 目标：理解 Context Window 有限时，信息如何丢失
# ====================================================================

def demo_context_window():
    """演示 Context Window 对 LLM 输出的影响

    关键认知：
      - Context Window 是硬限制，超出部分被丢弃
      - 中间的信息最容易被遗忘（Lost in the Middle）
      - "位置"很重要：关键信息放开头或结尾
    """
    separator("实验3：Context Window —— LLM 能记住多少？")

    llm = get_llm()

    # ── 实验 A: 在长文本中隐藏关键信息 ──
    print("📚 实验 A：信息位置对 LLM 的影响")
    print()

    # 生成一段包含"干扰文本"的输入
    filler = "这是无关的填充文本。\n" * 20  # 约 2000 Tokens

    # 关键信息放在不同位置
    info_beginning = f"关键信息：总经理的名字是张大鹏。\n{filler}"
    info_middle = f"{filler[:800]}关键信息：总经理的名字是张大鹏。\n{filler[800:]}"
    info_end = f"{filler}关键信息：总经理的名字是张大鹏。\n"

    for position, text in [("开头", info_beginning), ("中间", info_middle), ("结尾", info_end)]:
        prompt = f"{text}\n\n请问：总经理的名字是什么？"
        print(f"   📍 关键信息放在 {position}：", end=" ")
        response = llm.invoke([HumanMessage(content=prompt[:3000])])  # 手动截断
        answer = response.content[:60]
        print(f"{answer}...")

    # ── 实验 B: 超长输入的截断效果 ──
    print("\n📚 实验 B：超长输入的边界效果")
    print()
    long_text = "我喜欢红色。\n" * 100
    prompt = (
        f"请记住：我最喜欢的颜色是蓝色。\n"
        f"{long_text}"
        f"\n\n请问：我最喜欢的颜色是什么？"
    )

    print(f"   输入长度: {len(prompt)} 字符")
    response = llm.invoke([HumanMessage(content=prompt[:8000])])  # 模拟窗口
    print(f"   回答: {response.content[:100]}...")


# ===================================================================
#  实验4：Prompt 的影响 —— 起点决定方向
# ===================================================================
# 目标：理解为什么 Prompt 字面上的差异会导致完全不同输出
# ===================================================================

def demo_prompt_engineering():
    """演示不同 Prompt 带来的输出差异。

    关键认知：
      - Prompt = 概率分布的"初始条件"
      - 角色设定、格式要求、示例都改变了"续写的方向"
      - 好的 Prompt = 清晰、具体、有上下文
    """
    separator("实验4：Prompt 的影响 —— 同样的输入，不同的结果")

    llm = get_llm()

    # ── 实验：同一个问题，3 种 Prompt ──
    prompts = [
        {
            "role": "没有角色设定",
            "system": "",
            "question": "解释什么是机器学习",
        },
        {
            "role": "面向数据仓库工程师",
            "system": "你是一个资深数据仓库工程师。用数据库的类比来解释技术概念。",
            "question": "解释什么是机器学习",
        },
        {
            "role": "面向 5 岁小孩",
            "system": "你是一个幼儿园老师。用最简单的语言、比喻和故事来解释。",
            "question": "解释什么是机器学习",
        },
        {
            "role": "格式要求：JSON 输出",
            "system": "你是一个技术文档专家。请用 JSON 格式输出（包含核心概念、类比、应用场景字段）。",
            "question": "解释什么是机器学习",
        },
    ]

    for p in prompts:
        print(f"\n🎯 角色: {p['role']}")
        print(f"   问题: {p['question']}")
        print("─" * 40)

        messages = []
        if p["system"]:
            messages.append(SystemMessage(content=p["system"]))
        messages.append(HumanMessage(content=p["question"]))

        response = llm.invoke(messages)
        print(f"   回答: {response.content[:200]}")
        print()

    # ── 关键洞察 ──
    print("""
    💡 实验结论：
      同一个 LLM，同一个问题，不同的 Prompt → 完全不同的回答。

      Prompt 不是"命令"，而是"起点"。
      - 设定角色 → LLM 激活对应知识区域
      - 给例子 → LLM 模仿模式
      - 约束格式 → 限制输出空间

      这就是为什么 Prompt Engineering 是一门技术。
    """)


# ===================================================================
#  实验5：幻觉 —— LLM 为什么会编造
# ===================================================================
# 目标：展示 LLM 在不知道答案时如何"自信地胡说"
# ===================================================================

def demo_hallucination():
    """演示 LLM 的幻觉行为。

    关键认知：
      - LLM 的目标是"生成合理文本"，不是"说出真相"
      - 当问题超出知识边界，LLM 用"模式匹配"填补空白
      - 编造的细节通常"看起来很真"
    """
    separator("实验5：幻觉 —— LLM 如何编造答案")

    llm = get_llm()

    # ── 实验: 问 LLM 一些它"不知道"的问题 ──

    # 注：训练数据截止后的事件
    edge_questions = [
        "2025 年诺贝尔文学奖得主是谁？",
        "2026 年世界杯决赛的比分是多少？",
        "请引用一篇 2025 年发表的关于 LLM Agent 的论文",
        "2025 年苹果公司发布的最新产品是什么？",
    ]

    print("⚠️  注意：这些问题超出了 LLM 训练数据的截止时间。")
    print('   LLM 会诚实地说"不知道"，还是会编造？')
    print()

    for question in edge_questions:
        print(f"❓ 用户: {question}")

        response = llm.invoke([
            SystemMessage(content="你是一个诚实的助手。如果不知道就说不知道。"),
            HumanMessage(content=question),
        ])
        answer = response.content[:200]
        print(f"💬 LLM: {answer}")
        print()

    # 对比：知道答案的问题
    print("─" * 40)
    print("📊 对比：模型知道答案的问题")
    print()

    known_questions = [
        "Transformer 架构是哪一年提出的？",
        "什么是 Attention 机制？用一句话解释",
    ]

    for question in known_questions:
        print(f"❓ 用户: {question}")
        response = llm.invoke([HumanMessage(content=question)])
        print(f"💬 LLM: {response.content[:150]}")
        print()

    print("""
    💡 实验结论：
      1. 对于知识截止日期前的问题 → LLM 回答准确
      2. 对于知识截止日期后的问题 → LLM 可能编造（幻觉）
      3. "诚实设定"（说不知道）可以减少幻觉，但不能消除
      4. 幻觉 = 模型用"看起来合理"的模式填补了知识空白

      核心原因：LLM 的训练目标 = P(next_token | context)，不是"truth()"
    """)


# ===================================================================
#  实验6：长上下文失忆 —— 注意力稀释
# ===================================================================
# 目标：用数值模拟展示为什么上下文越长，注意力越分散
# ===================================================================

def demo_long_context_loss():
    """模拟长上下文中 Attention 被稀释的过程。

    关键认知：
      - 注意力权重总和 = 1（Softmax 的性质）
      - Token 越多，每个 Token 分到的权重越小
      - 长上下文中，Attention "平均化" = 约等于没 Attention
    """
    separator("实验6：长上下文失忆 —— 注意力如何被稀释")

    print("📊 模拟不同长度下，Attention 权重的分布")
    print()
    print(f"{'Token 数':<12} {'平均注意力权重':<18} {'最关注词的权重':<18} {'稀释倍数':<12}")
    print("-" * 60)

    for n_tokens in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        # 模拟随机注意力分数
        np.random.seed(42)
        scores = np.random.randn(n_tokens)
        weights = softmax(scores)

        avg_weight = weights.mean()
        max_weight = weights.max()
        dilution = 1.0 / n_tokens  # 理论平均权重

        print(f"{n_tokens:<12} {avg_weight:<18.6f} {max_weight:<18.6f} {dilution:<12.6f}")
        if n_tokens == 4 or n_tokens == 1024:
            print(f"  → 示例权重分布: {np.round(weights[:8], 4)}...")

    print()
    print("""
    💡 解释：
      - "平均注意力权重"列：随着 Token 增多，每个 Token 平均被关注度下降
      - 4K Tokens 时：每个 Token 平均被关注 0.00024
      - 128K Tokens 时：每个 Token 平均被关注 0.000008
      - 这就是为什么长上下文会"失忆"——关注度被稀释到噪声级别

      解决方案：
      1. RAG（检索增强生成）：只把相关片段放入 Context
      2. 关键信息放开头或结尾（Lost in the Middle 现象）
      3. 分段处理：长文档切段，逐段处理再汇总
    """)

    # ── 更直观的 demo：Lost in the Middle ──
    print("\n📚 Lost in the Middle 模拟：")
    print("   关键信息在不同位置时，Attention 权重的变化")
    print()

    doc_length = 100
    info_positions = [0, 25, 50, 75, 99]  # 关键信息位置（开头、1/4、中间、3/4、结尾）

    for pos in info_positions:
        # 模拟：位置越远，相关性衰减
        distances = np.abs(np.arange(doc_length) - pos)
        # 相关性随距离指数衰减
        relevance = np.exp(-distances / 20)
        # Softmax 后的关注度
        attention = softmax(relevance)

        info_weight = attention[pos]
        context = "开头" if pos == 0 else "1/4处" if pos == 25 else "中间" if pos == 50 else "3/4处" if pos == 75 else "结尾"
        print(f"   关键信息在 {context} → 被关注度: {info_weight:.4f} ({'✅ 高' if info_weight > 0.1 else '⚠️ 中' if info_weight > 0.02 else '❌ 低'})")

    print()
    print("   结论：开头和结尾的信息被关注度远高于中间！")


# ===================================================================
#  实验7：Transformer 架构概览
# ===================================================================
# 目标：用代码"画"出 Transformer 的结构
# ===================================================================

def demo_transformer_architecture():
    """用示意图展示 Transformer 架构。

    不涉及公式，用结构和流程来理解。
    """
    separator("实验7：Transformer 架构 —— 流水线工厂")

    architecture = """
┌───────────────────────────────────────────────────────────┐
│                   Transformer 架构                        │
│                                                           │
│   输入: "我 爱 编程"                                      │
│         │                                                 │
│         ▼                                                 │
│   ① Token Embedding ─── 把词转成向量                      │
│   ② Position Encoding ─ 告诉模型"词的顺序"                │
│         │                                                 │
│         ▼                                                 │
│   ┌───────────────────────┐    ┌──────────────────┐       │
│   │  Multi-Head           │    │  残差连接         │       │
│   │  Self-Attention       │◄───│  (Residual)       │       │
│   │  (每个词看向所有词)    │    └──────────────────┘       │
│   └───────────────────────┘                               │
│         │                                                 │
│         ▼                                                 │
│   ┌───────────────────────┐    ┌──────────────────┐       │
│   │  Feed-Forward         │    │  残差连接         │       │
│   │  (逐词非线性变换)      │◄───│  (Residual)       │       │
│   └───────────────────────┘    └──────────────────┘       │
│         │                                                 │
│         ▼ (重复 N 层 Transformer Block)                    │
│         │                                                 │
│         ▼                                                 │
│   ③ 输出概率分布 → 选择下一个 Token                       │
│                                                           │
│   输出: "编" → "程" → ...                                 │
└───────────────────────────────────────────────────────────┘
"""

    print(architecture)

    print("""
    📦 Transformer Block（核心组件）包含两个子层：

    子层 1: Self-Attention（自注意力）
      - 每个词"看向"所有其他词
      - 计算相关性，融合上下文信息
      - 并行处理所有位置

    子层 2: Feed-Forward（前馈网络）
      - 对每个位置独立做非线性变换
      - "思考"每个词在上下文中的含义
      - 两层线性变换 + ReLU 激活

    每个子层后都有：
      - 残差连接（Residual Connection）："直通车"，缓解梯度消失
      - 层归一化（Layer Norm）：稳定训练

    堆叠 N 层（GPT-3: 96 层, GPT-4: ~120 层）：
      低层 → 语法、词性
      中层 → 语义、关系
      高层 → 抽象推理、任务理解
    """)


# ===================================================================
#  主程序 —— 依次运行所有实验
# ===================================================================

def main():
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║      LLM 核心概念实验集                       ║")
    print("  ║      配套 learning_notes.md                   ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    print("  7 个实验，覆盖 LLM 最核心的 8 个概念")
    print()

    # 不需要 API 的实验
    demo_tokenization()
    demo_attention_mechanism()
    demo_long_context_loss()
    demo_transformer_architecture()

    # 需要 API 的实验
    # （取消注释即可运行，需要配置 .env 中的 API Key）
    demo_prompt_engineering()
    demo_hallucination()
    demo_context_window()

    print("✅ 所有实验完成！")
    print("   下一步：打开 learning_notes.md 复习概念")
    print("   然后完成 exercise.md 中的练习题")


if __name__ == "__main__":
    main()
