"""
=============================================================================
  LLM 入门实验台 —— 交互式网页版（不用写代码，点点按钮就行）
  ───────────────────────────────────────────────────────────────────────
  运行方式：
    1. pip install streamlit
    2. streamlit run llm_playground.py
    3. 浏览器会自动打开 http://localhost:8501
=============================================================================
"""

import streamlit as st
import numpy as np
import os
import json
from dotenv import load_dotenv
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="LLM 入门实验台",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LLM 入门实验台")
st.markdown("> 不用写一行代码，点点按钮就能理解 LLM 的核心概念")

# ─── 侧边栏：实验导航 ───────────────────────────────────
tab = st.sidebar.radio(
    "选择实验",
    [
        "🏠 首页",
        "🔤 实验1: Token 分切机",
        "👁️ 实验2: Attention 可视化",
        "📦 实验3: Context Window 模拟",
        "🎭 实验4: Prompt 实验室",
        "👻 实验5: 幻觉检测器",
    ]
)

# ─── 辅助函数 ────────────────────────────────────────

def softmax(x):
    e_x = np.exp(x - np.max(x, keepdims=True))
    return e_x / e_x.sum()


def get_llm():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )


def call_llm(messages):
    try:
        llm = get_llm()
        resp = llm.invoke(messages)
        return resp.content
    except Exception as e:
        return f"❌ 调用失败：{e}\n\n请检查 API 配置（.env 文件）"


# ════════════════════════════════════════════════════════
#  首页
# ════════════════════════════════════════════════════════

if tab == "🏠 首页":
    st.markdown("""
    ## LLM 是什么？

    **Large Language Model（大型语言模型）** 可以理解为"超级版输入法联想"。
    输入法会预测下一个词，LLM 也会——但它看得更远、理解更深、规模大了几万倍。

    ## 8 个核心概念，8 个实验

    | # | 概念 | 实验 |
    |---|------|------|
    | 1 | **Token** —— LLM 的"积木块" | 🔤 看看文本怎么被切碎 |
    | 2 | **Attention** —— "该看哪里" | 👁️ 可视化词与词的关注度 |
    | 3 | **Context Window** —— "短期记忆" | 📦 模拟窗口满的时候 |
    | 4 | **Transformer** —— "流水线工厂" | 📖 看架构图 |
    | 5 | **Prompt** —— "起点决定方向" | 🎭 同样问题，不同角色 |
    | 6 | **幻觉** —— "编故事" | 👻 测试 LLM 是否知道 |
    | 7 | **长上下文失忆** —— "万人体育场" | 📦 看注意力被稀释 |
    | 8 | **Embedding** —— "词向量" | 👁️ 可视化词与词的距离 |

    ## 怎么用

    左边选一个实验，开始探索！
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔤 **实验1**: 随便输段文字，看它被切成多少 Token")
    with col2:
        st.info("👁️ **实验2**: 一句话中，哪些词互相关注？")
    with col3:
        st.info("🎭 **实验4**: 同一个问题，不同角色回答完全不同")


# ════════════════════════════════════════════════════════
#  实验1: Token 分切机
# ════════════════════════════════════════════════════════

elif tab == "🔤 实验1: Token 分切机":
    st.header("🔤 Token 分切机")
    st.markdown("**Token = LLM 的'积木块'。** 不是字，不是词，是介于中间的'子词'。")

    with st.expander("📖 先理解 Token（点击展开）"):
        st.markdown("""
        - LLM 不看"字"，它看 **Token ID**（一串数字）
        - 英文 "Hello, world!" → 4 个 Token：['Hello', ',', ' world', '!']
        - 中文 "中国的首都是北京" → 6 个 Token：['中国', '的', '首', '都', '是', '北京']
        - Token 数 = 计费单位 + 上下文窗口的占用
        - 中文约 1 字 ≈ 1 Token，英文约 0.75 词 ≈ 1 Token
        """)

    st.subheader("✏️ 输入文本，看 Token 拆分结果")

    # 预设示例
    example = st.selectbox(
        "选择一个示例",
        ["", "Hello, world!", "中国的首都是北京",
         "Transformer是一种神经网络架构",
         "🌟 Hello 你好 123 !@#"]
    )
    user_text = st.text_area("或者自己输入", value=example, height=100,
                              help="输入任何文本，看它被拆成多少 Token")

    if st.button("🔪 切分 Token", type="primary"):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        text = user_text.strip()

        if not text:
            st.warning("请输入文本")
        else:
            tokens = enc.encode(text)
            decoded = [enc.decode([t]) for t in tokens]

            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Token 数", len(tokens))
            col2.metric("📝 字符数", len(text))
            col3.metric("📏 性价比", f"{len(text)/len(tokens):.1f}" if tokens else "0")

            st.subheader("🔍 拆分详情")
            st.markdown("每一块是一种颜色，就是 1 个 Token")

            # 彩色显示每个 Token
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                      "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
            html_parts = ""
            for i, t in enumerate(tokens):
                color = colors[i % len(colors)]
                html_parts += f'<span style="background:{color}; padding:2px 6px; margin:2px; border-radius:4px; font-size:18px" title="ID:{t}">{decoded[i]}</span>'
            st.markdown(f"<div style='line-height:2.5'>{html_parts}</div>",
                        unsafe_allow_html=True)

            # Token 列表
            with st.expander("查看 Token ID 列表"):
                for i, (t, d) in enumerate(zip(tokens, decoded)):
                    st.text(f"  Token #{i:3d}: ID {t:6d} → '{d}'")

    # Token 效率对比
    st.divider()
    st.subheader("📊 中文 vs 英文 Token 效率")
    st.markdown("同样意思，中文更省 Token（省钱！）")

    if st.button("🔄 对比中英文"):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        cn = "中国的首都是北京，北京是一座古老而现代的城市。"
        en = "Beijing is the capital of China. It is home to over 20 million people."

        cn_tokens = len(enc.encode(cn))
        en_tokens = len(enc.encode(en))

        col1, col2 = st.columns(2)
        col1.metric("🇨🇳 中文 Token 数", cn_tokens, help=cn)
        col2.metric("🇬🇧 英文 Token 数", en_tokens, help=en)
        st.success(f"💡 同样信息量，中文比英文少了 {en_tokens - cn_tokens} 个 Token！")


# ════════════════════════════════════════════════════════
#  实验2: Attention 可视化
# ════════════════════════════════════════════════════════

elif tab == "👁️ 实验2: Attention 可视化":
    st.header("👁️ Attention = 词与词之间的"关注"")
    st.markdown("每个词都会'看向'所有其他词，计算'我该多关注谁'。")

    with st.expander("📖 先理解 Attention（点击展开）"):
        st.markdown("""
        - 读"那家店在王府井"，看到"那家"时，你的大脑自动看向"店"、"王府井"
        - Attention 做的就是这事：**每个词都算一个"关注度分数"**
        - 分数越高 → 这个词对当前词"越重要"
        - 最后每个词都"吸收"了其他词的信息，变成"融合了上下文的新表示"
        """)

    sentence = st.text_input("输入一句话（英文，用空格分隔）",
                              value="The cat sat on the mat")
    words = sentence.strip().split()

    if len(words) < 2:
        st.warning("至少输入 2 个词")
    else:
        np.random.seed(42)
        dim = min(8, max(2, len(words)))
        embeddings = np.random.randn(len(words), dim)
        scores = embeddings @ embeddings.T
        weights = np.zeros_like(scores)
        for i in range(len(words)):
            weights[i] = softmax(scores[i])

        st.subheader("🔥 注意力热力图")
        st.markdown("**行 = 当前词，列 = 它关注的词。颜色越深 = 关注度越高**")

        import plotly.graph_objects as go
        fig = go.Figure(data=go.Heatmap(
            z=weights,
            x=words,
            y=words,
            colorscale="Viridis",
            text=np.round(weights, 3),
            texttemplate="%{text}",
            hovertemplate="'%{y}' 关注 '%{x}': %{text}<extra></extra>"
        ))
        fig.update_layout(
            height=400,
            xaxis_title="被关注的词",
            yaxis_title="当前词",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 解读
        st.subheader("💡 解读")
        for i, w in enumerate(words):
            top_idx = np.argmax(weights[i])
            top_word = words[top_idx]
            top_weight = weights[i][top_idx]
            st.markdown(f"- **'{w}'** 最关注 **'{top_word}'** (权重 {top_weight:.3f})")

        st.info("💡 每个词的新表示 = 所有其他词的'加权融合'。这就是 LLM 理解上下文的方式！")

# ════════════════════════════════════════════════════════
#  实验3: Context Window + 长上下文失忆
# ════════════════════════════════════════════════════════

elif tab == "📦 实验3: Context Window 模拟":
    st.header("📦 Context Window = LLM 的"短期记忆容量"")
    st.markdown("Token 越多，注意力越分散——就像在万人体育场找朋友。")

    with st.expander("📖 先理解（点击展开）"):
        st.markdown("""
        - Context Window = LLM 一次能"看到"的最大 Token 数
        - 4K (GPT-3.5) / 8K-128K (GPT-4) / 200K (Claude) / 1M (Gemini)
        - **问题**：Token 越多，每个 Token 分到的"关注度"越少
        - 就像：10 个人的房间找人容易，10000 人的体育场找人难
        """)

    st.subheader("📊 注意力稀释模拟")
    st.markdown("拖动滑块，看 Context Window 变大时，每个 Token 被关注的程度如何下降")

    n_tokens = st.slider("Token 数量", 4, 1024, 16, step=4,
                          help="模拟 Context Window 的大小")

    np.random.seed(42)
    scores = np.random.randn(n_tokens)
    weights = softmax(scores)

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, n_tokens + 1)),
        y=weights,
        mode="lines+markers",
        name="关注度",
        line=dict(color="#FF6B6B", width=2),
        marker=dict(size=4),
    ))
    fig.add_hline(y=1/n_tokens, line_dash="dash", line_color="gray",
                   annotation_text=f"平均关注度 = {1/n_tokens:.5f}")
    fig.update_layout(
        height=300,
        xaxis_title="Token 位置",
        yaxis_title="关注度权重",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("🧩 Token 数", n_tokens)
    col2.metric("📉 平均关注度", f"{weights.mean():.6f}")
    col3.metric("🎯 最高关注度", f"{weights.max():.4f}")

    st.info(f"""
    💡 **当 Context Window = {n_tokens} Tokens 时：**
    - 每个 Token 平均被关注 {weights.mean():.6f}（越低 = 越"失忆"）
    - 最高关注的 Token 也只占 {weights.max():.2%}
    - 这就是为什么长上下文会"失忆"——关注度被稀释了
    """)

    # Lost in the Middle 模拟
    st.divider()
    st.subheader("📚 Lost in the Middle 模拟")
    st.markdown("关键信息放在不同位置，LLM 的关注度变化：")

    doc_len = st.slider("文档长度（Token 数）", 20, 200, 100, step=10)
    positions = {"开头": 0, "1/4处": doc_len // 4, "中间": doc_len // 2,
                 "3/4处": doc_len * 3 // 4, "结尾": doc_len - 1}

    data = []
    for label, pos in positions.items():
        distances = np.abs(np.arange(doc_len) - pos)
        relevance = np.exp(-distances / (doc_len / 10))
        attn = softmax(relevance)
        data.append((label, attn[pos]))

    labels, values = zip(*data)
    fig2 = go.Figure(data=[go.Bar(
        x=labels, y=values,
        marker_color=["#4ECDC4" if v > 0.02 else "#FF6B6B" for v in values],
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    )])
    fig2.update_layout(height=300, xaxis_title="关键信息位置", yaxis_title="被关注度")
    st.plotly_chart(fig2, use_container_width=True)

    st.warning("⚠️ **结论**：开头和结尾记得最清，中间最容易被忘！重要信息放开头或结尾！")


# ════════════════════════════════════════════════════════
#  实验4: Prompt 实验室
# ════════════════════════════════════════════════════════

elif tab == "🎭 实验4: Prompt 实验室":
    st.header("🎭 Prompt 实验室")
    st.markdown("""
    **Prompt = 你给 LLM 的"起点"。起点不同，续写方向完全不同。**

    同一个问题，换一种说法，回答天差地别。
    """)

    with st.expander("📖 先理解（点击展开）"):
        st.markdown("""
        - LLM 是"续写机器"，你的 Prompt 就是它"从哪里开始想"
        - 设定角色 → 激活 LLM 不同的"知识区域"
        - 给例子 → LLM 会模仿例子的模式和风格
        - 约束格式 → 限制输出空间，减少意外
        """)

    question = st.text_input("✏️ 你想问什么？",
                              value="解释什么是机器学习")

    col1, col2 = st.columns(2)

    with col1:
        role = st.selectbox(
            "🎭 让 LLM 扮演什么角色？",
            ["普通助手（无角色）",
             "资深数据仓库工程师",
             "幼儿园老师（给 5 岁小孩讲）",
             "毒舌程序员",
             "古代诗人",
             "产品经理"]
        )

    with col2:
        output_format = st.selectbox(
            "📋 输出格式",
            ["自由格式", "JSON 格式", "Markdown 格式", "表格格式", "一句话概括"]
        )

    role_prompts = {
        "普通助手（无角色）": "",
        "资深数据仓库工程师": "你是一个资深数据仓库工程师。用数据库和数据仓库的类比来解释技术概念。回答要专业、有 SQL 示例。",
        "幼儿园老师（给 5 岁小孩讲）": "你是一个幼儿园老师。用最简单的语言、比喻和故事来解释。每句话不超过 20 个字。用小朋友能懂的方式说。",
        "毒舌程序员": "你是一个毒舌但技术很强的程序员。回答尖锐、幽默、一针见血。可以吐槽，但要说到点子上。",
        "古代诗人": "你是一个中国古代诗人。用诗意的语言回答问题，适当引用诗句，语言典雅。",
        "产品经理": "你是一个资深产品经理。从用户需求、场景、价值的角度回答问题。结构化思考。",
    }

    format_prompts = {
        "自由格式": "",
        "JSON 格式": "请用 JSON 格式输出。",
        "Markdown 格式": "请用 Markdown 格式输出，包含标题和列表。",
        "表格格式": "请用表格格式呈现主要信息。",
        "一句话概括": "请用一句话概括。不要超过 50 个字。",
    }

    if st.button("🚀 发送", type="primary"):
        if not question.strip():
            st.warning("请输入问题")
        else:
            with st.spinner("🤔 LLM 正在思考..."):
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = []
                if role_prompts[role]:
                    messages.append(SystemMessage(content=role_prompts[role]))
                if format_prompts[output_format]:
                    if messages:
                        system_content = messages[0].content
                        system_content += f"\n\n{format_prompts[output_format]}"
                        messages[0] = SystemMessage(content=system_content)
                    else:
                        messages.append(SystemMessage(content=format_prompts[output_format]))

                messages.append(HumanMessage(content=question))
                result = call_llm(messages)

                st.subheader("💬 回答")
                st.markdown(result)

                st.divider()
                st.caption(f"🎭 角色: {role}  |  📋 格式: {output_format}")

    # 对比实验
    st.divider()
    st.subheader("🆚 同一个问题，4 种角色对比")
    st.markdown("点一下，看看同样的问题，不同角色的回答差异有多大")

    demo_question = st.text_input("对比实验的问题", value="什么是机器学习")
    if st.button("🔄 开始对比", type="secondary"):
        from langchain_core.messages import HumanMessage, SystemMessage
        roles_to_compare = ["普通助手（无角色）", "幼儿园老师（给 5 岁小孩讲）",
                            "资深数据仓库工程师", "毒舌程序员"]
        tabs = st.tabs([r[:6] for r in roles_to_compare])
        for i, (role_name, tab_widget) in enumerate(zip(roles_to_compare, tabs)):
            with tab_widget:
                with st.spinner(f"调用 {role_name}..."):
                    messages = []
                    if role_prompts[role_name]:
                        messages.append(SystemMessage(content=role_prompts[role_name]))
                    messages.append(HumanMessage(content=demo_question))
                    result = call_llm(messages)
                    st.markdown(result)


# ════════════════════════════════════════════════════════
#  实验5: 幻觉检测器
# ════════════════════════════════════════════════════════

elif tab == "👻 实验5: 幻觉检测器":
    st.header("👻 幻觉检测器")
    st.markdown("""
    **幻觉 = LLM 编造看起来合理但实际错误的内容。**

    LLM 的目标是"生成最合理的下一段话"，不是"说出真相"。
    当问题超出它的知识范围，它可能会"自信地胡说"。
    """)

    with st.expander("📖 先理解为什么会产生幻觉（点击展开）"):
        st.markdown("""
        幻觉的 4 个原因：
        1. **训练目标 = 概率拟合，不是事实验证**——模型从未见过"对错标签"
        2. **知识截止**——训练数据停在某个时间点，之后的只能"编"
        3. **压缩损失**——几 TB 文本压缩成几百 GB 参数，细节必然丢失
        4. **采样随机性**——temperature > 0 引入了随机性，可能选了"错"的路径
        """)

    st.subheader("🧪 测试 LLM 是否知道答案")

    test_type = st.radio(
        "选择测试类型",
        [
            "🔥 知识边界测试（问未来的事，看 LLM 是否承认不知道）",
            "✅ 基础知识测试（问 LLM 应该知道的事）",
            "✏️ 自定义测试（自己输入问题）",
        ]
    )

    if "未来" in test_type:
        questions = [
            "2025 年诺贝尔文学奖得主是谁？",
            "2026 年世界杯决赛的比分是多少？",
            "2025 年苹果公司发布的最新产品是什么？",
            "2027 年的美国总统是谁？",
        ]
        for q in questions:
            if st.button(f"❓ {q}", use_container_width=True):
                with st.spinner("🤔 思考中..."):
                    from langchain_core.messages import HumanMessage, SystemMessage
                    result = call_llm([
                        SystemMessage(content="你是一个诚实的助手。如果不知道就说不知道，不要编造。"),
                        HumanMessage(content=q),
                    ])
                    has_hallucination = any(kw in result for kw in ["抱歉", "不知道", "尚未", "无法"])
                    if has_hallucination:
                        st.success("✅ LLM 承认不知道（没有幻觉）")
                    else:
                        st.error("⚠️ LLM 可能产生了幻觉！")
                    st.markdown(result)

    elif "基础" in test_type:
        questions = [
            "Transformer 架构是哪一年提出的？",
            "Attention 机制的核心公式叫什么？",
            "什么是 Few-shot Learning？",
            "GPT 中的 T 代表什么？",
        ]
        for q in questions:
            if st.button(f"❓ {q}", use_container_width=True):
                with st.spinner("🤔 思考中..."):
                    from langchain_core.messages import HumanMessage
                    result = call_llm([HumanMessage(content=q)])
                    st.success("✅ LLM 应该知道这个")
                    st.markdown(result)

    else:
        custom_q = st.text_input("输入你的问题", value="请引用一篇 2025 年发表的 AI 论文")
        if st.button("🔍 检测幻觉", type="primary"):
            with st.spinner("🤔 思考中..."):
                from langchain_core.messages import HumanMessage, SystemMessage
                result = call_llm([
                    SystemMessage(content="你是一个诚实的助手。如果不知道就说不知道。"),
                    HumanMessage(content=custom_q),
                ])
                st.markdown(result)
                st.divider()
                st.caption("💡 **判断方法**：如果回答中有具体名称、数字、引用但你不确定是否真实，可能是幻觉。可以用搜索引擎验证。")

    # 幻觉原理
    st.divider()
    with st.expander("📚 深入理解幻觉（附代码思路）"):
        st.markdown("""
        ### LLM 幻觉的根源

        ```
        训练阶段：
        喂入数据："爱因斯坦1922年访问了日本，并..."
        模型学到：爱因斯坦 → 1922 → 日本 → 访问

        推理阶段：
        用户问："爱因斯坦访问过中国吗？"
        → 模型激活模式：爱因斯坦 + 访问 + 地名
        → 选了"1922年"（和"访问"最常一起出现的词）
        → 编造了"上海"（模型推测：既然去了日本，大概也去了上海？）
        ```

        ### 减少幻觉的方法
        1. **明确要求"不知道就说不知道"**（System Prompt）
        2. **提供参考资料**（RAG：检索增强生成）
        3. **降低 temperature**（减少随机性）
        4. **要求 LLM 引用来源**（可验证的事实）
        5. **多次采样，交叉验证**（同一个问题问几次，看是否一致）
        """)

# ─── 页脚 ──────────────────────────────────────────
st.divider()
st.caption("🤖 LLM 入门实验台 · 配套 learning_notes.md · 2026")
