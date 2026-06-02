"""
=============================================================================
  Demo 01：LLM —— 为什么要用 LangChain 封装大模型调用？
=============================================================================

问题：不用 LangChain 怎么调 LLM？有什么痛点？

答案：原生 openai 库也很方便，LangChain 的 ChatOpenAI 核心价值在于：
  1. 统一接口：不管底层是 OpenAI / DeepSeek / 本地模型，invoke() 都一样
  2. 自动重试：网络抖动时自动 retry，不用手写 try/except
  3. Token 计数：自动返回 usage 信息
  4. 与 Chain/Runnable 无缝组合（后续 Demo 会看到）

运行方式：
  cd langchain_demos
  cp .env.example .env  → 填入 API Key
  python demo_01_llm.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# 方式一：不用 LangChain —— 原生 openai 库
# ═══════════════════════════════════════════════════════════════════════════

def without_langchain():
    print("=" * 60)
    print("  方式一：原生 openai 库")
    print("=" * 60)

    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话解释什么是 LangChain"}],
        temperature=0.3,
    )
    print(f"回答: {response.choices[0].message.content}")
    print(f"Token 用量: {response.usage}")

    print("""
    ✅ 能跑，功能完整
    ❌ 但每次都要传 model/messages/temperature，重复代码多
    ❌ 想切模型（如 DeepSeek → GPT-4）要改多处
    ❌ 想加 retry / 缓存 / 流式输出要手写
    ❌ 无法和 LangChain 的 PromptTemplate/Chain 组合
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 方式二：用 LangChain —— ChatOpenAI
# ═══════════════════════════════════════════════════════════════════════════

def with_langchain():
    print("\n" + "=" * 60)
    print("  方式二：LangChain ChatOpenAI")
    print("=" * 60)

    from langchain_openai import ChatOpenAI

    # 配置一次，到处复用
    llm = ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0.3,
        max_tokens=200,
    )

    # invoke() 是 LangChain 的统一调用入口
    # 参数是 messages 列表，和原生 API 格式兼容
    from langchain_core.messages import HumanMessage, SystemMessage

    result = llm.invoke([
        SystemMessage(content="你是一个简洁的AI助手，回答不超过30字。"),
        HumanMessage(content="什么是 LangChain？"),
    ])

    # result 是 AIMessage 对象，比原生 dict 更丰富
    print(f"类型: {type(result).__name__}")
    print(f"回答: {result.content}")
    print(f"Token 用量: {result.usage_metadata if hasattr(result, 'usage_metadata') else 'N/A'}")

    print("""
    ✅ 统一接口：invoke() 对所有模型都一样
    ✅ 模型切换：只改一处 ChatOpenAI(model=...) 即可
    ✅ 内置重试：网络故障自动 retry
    ✅ 消息类型：SystemMessage / HumanMessage / AIMessage 语义清晰
    ✅ 可组合：后续可以和 PromptTemplate / Chain 串联
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 进阶：流式输出 —— LangChain 的 stream()
# ═══════════════════════════════════════════════════════════════════════════

def demo_streaming():
    print("\n" + "=" * 60)
    print("  流式输出：stream()")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(
        model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.3,
    )

    print("逐 token 输出: ", end="", flush=True)
    for chunk in llm.stream([HumanMessage(content="用一句话解释 RAG")]):
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            print(content, end="", flush=True)
    print()


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🤖 Demo 01：LLM —— LangChain 如何封装大模型调用")
    print("=" * 60)

    without_langchain()
    with_langchain()
    demo_streaming()

    print("\n✅ Demo 01 完成！")
    print("关键收获：ChatOpenAI 不是黑魔法，只是把 OpenAI 客户端包装成了统一接口。")
    print("它的真正价值在后续 Demo —— 和 PromptTemplate / Chain 组合时体现。\n")


if __name__ == "__main__":
    main()
