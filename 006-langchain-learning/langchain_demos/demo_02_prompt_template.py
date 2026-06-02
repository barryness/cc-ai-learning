"""
=============================================================================
  Demo 02：PromptTemplate —— 告别 f-string 拼 Prompt
=============================================================================

问题：不用 LangChain 怎么构建 Prompt？有什么痛点？

  方式一：f-string（最原始）
    prompt = f"你是{role}，请用{lang}回答：{question}"
    痛点：变量多了像乱麻，没有类型检查，团队协作全靠命名约定

  方式二：LangChain PromptTemplate
    把 Prompt 当成"模板 + 变量"的数据结构，类型安全，可复用。

核心价值：
  1. 模板和变量分离 → 改 Prompt 不用翻代码
  2. 支持少样本示例（Few-shot Prompting）
  3. ChatPromptTemplate 天然支持多角色消息（system / user / assistant）
  4. 和 LLM 通过 | 管道组合（后续 Demo 详解）

运行方式：
  python demo_02_prompt_template.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# 方式一：不用 LangChain —— f-string 拼 Prompt
# ═══════════════════════════════════════════════════════════════════════════

def without_langchain():
    print("=" * 60)
    print("  方式一：f-string 拼 Prompt")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    role = "Python 专家"
    question = "装饰器是什么"
    requirements = "用比喻解释，不超过50字，不要代码"

    # f-string 拼接 —— 变量多了完全不可读
    system_prompt = (
        f"你是一个{role}。\n"
        f"回答要求：{requirements}"
    )
    user_prompt = f"请解释：{question}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    print(f"回答: {response.choices[0].message.content}")

    print("""
    ❌ f-string 拼接的痛点：
       1. 变量和模板混在一起，改模板要翻代码
       2. 没有类型检查 —— role 传了个 int 也不会报错
       3. 多角色对话的模板（system + few-shot + user）越来越乱
       4. 想复用同一个模板给不同问题用？复制粘贴
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 方式二：用 LangChain ChatPromptTemplate
# ═══════════════════════════════════════════════════════════════════════════

def with_langchain():
    print("\n" + "=" * 60)
    print("  方式二：LangChain ChatPromptTemplate")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.3)

    # 模板是数据结构，变量用 {花括号} 标记
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}。\n回答要求：{requirements}"),
        ("user", "请解释：{question}"),
    ])

    # .invoke() 传入变量字典 → 自动填充 → 生成 messages
    messages = prompt.invoke({
        "role": "Python 专家",
        "question": "装饰器是什么",
        "requirements": "用比喻解释，不超过50字，不要代码",
    })

    print(f"生成的 messages ({len(messages.messages)} 条):")
    for m in messages.messages:
        print(f"  [{m.type}] {m.content[:80]}...")

    result = llm.invoke(messages)
    print(f"\n回答: {result.content}")

    print("""
    ✅ ChatPromptTemplate 的好处：
       1. 模板和变量分离 —— 改 Prompt 不改代码逻辑
       2. 变量缺失会报错 —— 类型安全
       3. 支持 system / user / assistant 多角色
       4. 同一个模板可以复用无数次
       5. 后续可以和 LLM 用 | 管道组合
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 进阶：Few-shot Prompting —— 给 LLM 看几个例子
# ═══════════════════════════════════════════════════════════════════════════

def demo_few_shot():
    print("\n" + "=" * 60)
    print("  进阶：Few-shot Prompting（给 LLM 看例子）")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.1)

    # 例子库 —— 告诉 LLM 你想要什么格式
    examples = [
        {"input": "今天心情真好", "output": "POSITIVE"},
        {"input": "等了三小时还没到", "output": "NEGATIVE"},
        {"input": "还行吧没什么特别", "output": "NEUTRAL"},
    ]

    # Few-shot 模板：把例子格式化成 user/assistant 对话
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"),
        ("assistant", "{output}"),
    ])

    few_shot = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    # 最终 Prompt = few_shot 例子 + 当前问题
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个情感分析器，只输出 POSITIVE / NEGATIVE / NEUTRAL。"),
        few_shot,
        ("user", "{input}"),
    ])

    chain = final_prompt | llm

    test_cases = ["服务特别周到", "太让人失望了"]
    for text in test_cases:
        result = chain.invoke({"input": text})
        print(f"  「{text}」→ {result.content.strip()}")


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  📝 Demo 02：PromptTemplate —— 告别 f-string 拼 Prompt")
    print("=" * 60)

    without_langchain()
    with_langchain()
    demo_few_shot()

    print("\n✅ Demo 02 完成！")
    print("关键收获：PromptTemplate 把模板变成可复用的数据结构，变量和模板分离。")
    print("最重要的一行：prompt | llm —— 管道操作符，后续 Demo 会反复用到。\n")


if __name__ == "__main__":
    main()
