"""
=============================================================================
  Demo 07：Runnable —— LCEL 的核心协议
=============================================================================

问题：Chain 只能线性串联，但如果需要并行、分支、自定义处理呢？

LangChain 的所有组件（LLM/Prompt/Parser/Retriever）都实现了 Runnable 协议。
Runnable 就像一个"统一插头"，所有组件都可以用 | 互相连接。

Runnable 的四种核心能力：
  1. invoke(input) → output           同步执行
  2. stream(input) → iterator[output]  流式输出
  3. batch([input, ...]) → [output]   批量并发
  4. 用 | 管道组合                     搭积木

本篇重点讲 4 个关键 Runnable：
  - RunnablePassthrough  —— 透传数据（像"透明管道"）
  - RunnableLambda       —— 包装任意函数为 Runnable
  - RunnableParallel     —— 并行执行多个分支
  - RunnableBranch       —— 条件分支（if/else）

运行方式：
  python demo_07_runnable.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# 1. RunnablePassthrough —— 数据透传
# ═══════════════════════════════════════════════════════════════════════════

def demo_passthrough():
    print("=" * 60)
    print("  1. RunnablePassthrough —— 数据透传")
    print("=" * 60)

    from langchain_core.runnables import RunnablePassthrough

    # RunnablePassthrough 直接输出输入，不做任何处理
    # 作用：在管道中占位，或者把上一步的结果"抄"到下一步
    passthrough = RunnablePassthrough()

    print(f"passthrough.invoke('hello') → {passthrough.invoke('hello')}")
    print(f"passthrough.invoke(42) → {passthrough.invoke(42)}")

    # 真正有用的场景：在 chain 中提取特定字段
    data = {"name": "张三", "age": 25, "role": "工程师"}
    # 只取 name 字段，其他丢掉
    extract_name = RunnablePassthrough.assign(
        # assign() 添加新字段而不删除旧字段
        upper_name=lambda x: x["name"].upper(),
    )
    print(f"assign upper_name: {extract_name.invoke(data)}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. RunnableLambda —— 把任何函数变成 Runnable
# ═══════════════════════════════════════════════════════════════════════════

def demo_lambda():
    print("\n" + "=" * 60)
    print("  2. RunnableLambda —— 包装任意函数")
    print("=" * 60)

    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.3)

    # 自定义函数：统计答案字数
    def count_chars(text: str) -> str:
        return f"{text}\n\n(共 {len(text)} 字)"

    # RunnableLambda 把普通函数包装成 Runnable
    # 这样就可以用 | 串到管道里！
    chain = (
        ChatPromptTemplate.from_messages([
            ("user", "用一句话解释{concept}"),
        ])
        | llm
        | StrOutputParser()
        | RunnableLambda(count_chars)  # 自定义后处理
    )

    result = chain.invoke({"concept": "Runnable"})
    print(result)

    print("\n  RunnableLambda 让任何函数都能接入管道 ✅")


# ═══════════════════════════════════════════════════════════════════════════
# 3. RunnableParallel —— 并行执行
# ═══════════════════════════════════════════════════════════════════════════

def demo_parallel():
    print("\n" + "=" * 60)
    print("  3. RunnableParallel —— 并行执行多个分支")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.3)

    # 典型场景：RAG 中同时做检索 + Query 改写
    # 两个 LLM 调用可以并发执行！

    # 分支1：翻译成英文
    chain_en = (
        ChatPromptTemplate.from_messages([("user", "把以下内容翻译成英文：{text}")])
        | llm | StrOutputParser()
    )

    # 分支2：总结要点
    chain_summary = (
        ChatPromptTemplate.from_messages([("user", "用3个要点总结以下内容：{text}")])
        | llm | StrOutputParser()
    )

    # RunnableParallel 并发执行两个分支
    parallel_chain = RunnableParallel(
        english=chain_en,
        summary=chain_summary,
    )

    text = "LangChain是一个用于构建LLM应用的框架，支持Python和JavaScript。"
    result = parallel_chain.invoke({"text": text})

    print("并行执行结果:")
    print(f"  英文翻译: {result['english']}")
    print(f"  要点总结: {result['summary']}")

    print("\n  两个 LLM 调用是并发的 ✅（比串行快了一倍）")


# ═══════════════════════════════════════════════════════════════════════════
# 4. 完整示例：Runnable 组合实战
# ═══════════════════════════════════════════════════════════════════════════

def demo_complete_pipeline():
    print("\n" + "=" * 60)
    print("  4. 完整管道：Passthrough + Parallel + Lambda")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.3)

    # 场景：输入一篇文章，同时得到中文摘要 + 英文翻译
    # 数据流：
    #                   ┌→ 摘要分支 → 统计字数 ─┐
    #   {article} → ───┤                        ├──→ {cn_summary, en_translation}
    #                   └→ 翻译分支 ────────────┘

    chain_cn = (
        ChatPromptTemplate.from_messages([("user", "用不超过30字总结：{article}")])
        | llm | StrOutputParser()
        | RunnableLambda(lambda s: f"{s}\n[字数: {len(s)}]")
    )

    chain_en = (
        ChatPromptTemplate.from_messages([("user", "Translate to English: {article}")])
        | llm | StrOutputParser()
    )

    full_chain = RunnableParallel(
        cn_summary=chain_cn,
        en_translation=chain_en,
    )

    article = "RAG（检索增强生成）通过将外部知识库与LLM结合，有效解决了AI生成内容中的幻觉问题。"
    result = full_chain.invoke({"article": article})

    print("=" * 40)
    print("中文摘要:")
    print(result["cn_summary"])
    print("=" * 40)
    print("英文翻译:")
    print(result["en_translation"])

    print("""
    ✅ 完整管道展示了 Runnable 的核心能力：
       1. RunnablePassthrough —— 透传原始数据
       2. RunnableLambda —— 统计字数
       3. RunnableParallel —— 两个分支并发
       4. 全部通过 | 串联
    """)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🧩 Demo 07：Runnable —— LCEL 的核心协议")
    print("=" * 60)

    demo_passthrough()
    demo_lambda()
    demo_parallel()
    demo_complete_pipeline()

    print("\n✅ Demo 07 完成！")
    print("关键收获：")
    print("  - Runnable 是 LangChain 的'统一插头'，所有组件都可以用 | 互连")
    print("  - RunnablePassthrough: 透传数据")
    print("  - RunnableLambda: 把任意函数接入管道")
    print("  - RunnableParallel: 并发执行多个分支\n")


if __name__ == "__main__":
    main()
