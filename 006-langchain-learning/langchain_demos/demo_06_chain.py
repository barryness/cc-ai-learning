"""
=============================================================================
  Demo 06：Chain —— 把多个步骤串联成一条流水线
=============================================================================

问题：RAG 有三个步骤：构建 Prompt → 调 LLM → 解析输出。
      每一步都要把上一步的输出传给下一步，代码越写越长。

不用 LangChain 怎么办？
  → 写一个 pipeline 函数，按顺序调用：
    messages = build_prompt(query)
    response = call_llm(messages)
    answer = parse_output(response)
  → 痛点：步骤多了像千层饼，加一步中间处理要改整个函数

LangChain Chain 的价值：
  1. LCEL (LangChain Expression Language) 用 | 管道串联
  2. 数据自动流转：上一步输出 → 下一步输入
  3. 可视化：chain.get_graph() 可以看到流程图
  4. 声明式：描述"做什么"而不是"怎么做"

运行方式：
  python demo_06_chain.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# 方式一：不用 LangChain —— 手工串联
# ═══════════════════════════════════════════════════════════════════════════

def without_langchain():
    print("=" * 60)
    print("  方式一：手工串联各步骤")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    concept = "Chain"

    # Step 1: 构建 Prompt
    messages = [
        {"role": "system", "content": "你是一个Python专家。回答不超过50字。"},
        {"role": "user", "content": f"解释{concept}的概念"},
    ]

    # Step 2: 调 LLM
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.3)

    # Step 3: 提取结果
    answer = response.choices[0].message.content

    # Step 4: 后处理（如果有的话）
    answer = answer.strip()

    print(f"结果: {answer}")

    print("""
    ❌ 手工串联的问题：
       1. 步骤线性写死，加一个步骤要插代码
       2. 想增加 retry / fallback 要在每个步骤里加
       3. 步骤间数据传递靠变量命名约定
       4. 想复用一个子流程 → 复制粘贴
       5. 想并行执行两个 LLM 调用 → 手写多线程
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 方式二：用 LangChain LCEL —— | 管道操作符
# ═══════════════════════════════════════════════════════════════════════════

def with_lcel_chain():
    print("\n" + "=" * 60)
    print("  方式二：LCEL | 管道 —— 声明式串联")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.3)

    # 用 | 把三个组件串起来
    # 数据流：{concept} → prompt → llm → parser → str
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是一个Python专家。回答不超过50字。"),
            ("user", "解释{concept}的概念"),
        ])
        | llm
        | StrOutputParser()
    )

    # 一个 invoke() 跑完全链路
    result = chain.invoke({"concept": "Chain"})
    print(f"结果: {result}")

    print("""
    ✅ LCEL | 管道的价值：
       1. A | B | C —— 声明式，一眼看出数据流向
       2. .invoke() 自动把上一步的输出传给下一步
       3. 要加步骤？在管道里加一个 | 即可
       4. 每个组件独立可测试（单独测 prompt / llm / parser）
       5. 支持并行、分支等高级模式（Demo 07 演示）
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 进阶：多个 Chain 的对比试验
# ═══════════════════════════════════════════════════════════════════════════

def demo_multiple_chains():
    print("\n" + "=" * 60)
    print("  进阶：多个 Chain —— 不同风格的同一个任务")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.3)
    parser = StrOutputParser()

    # Chain A: 简短风格
    chain_short = (
        ChatPromptTemplate.from_messages([
            ("system", "用一句话回答，不超过20字。"),
            ("user", "{question}"),
        ])
        | llm | parser
    )

    # Chain B: 详细风格（同样模板，换 system prompt）
    chain_detailed = (
        ChatPromptTemplate.from_messages([
            ("system", "详细回答，包含原因和例子，不超过100字。"),
            ("user", "{question}"),
        ])
        | llm | parser
    )

    question = "为什么要用 Chain？"
    print(f"问题: {question}\n")
    print(f"  简短回答: {chain_short.invoke({'question': question})}")
    print(f"  详细回答: {chain_detailed.invoke({'question': question})}")

    print("""
    两个 Chain 共用同一个 LLM 和 Parser，只换了 Prompt 模板。
    这就是声明式的威力 —— 像搭积木一样组合组件。
    """)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  ⛓️  Demo 06：Chain —— 把步骤串联成流水线")
    print("=" * 60)

    without_langchain()
    with_lcel_chain()
    demo_multiple_chains()

    print("\n✅ Demo 06 完成！")
    print("关键收获：LCEL 的 | 管道让代码变成声明式的数据流描述。")
    print("prompt | llm | parser —— 一个 invoke() 从头跑到尾。\n")


if __name__ == "__main__":
    main()
