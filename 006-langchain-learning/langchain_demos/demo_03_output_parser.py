"""
=============================================================================
  Demo 03：OutputParser —— 让 LLM 输出结构化的数据
=============================================================================

问题：LLM 输出的是自然语言文本，但程序需要的是结构化数据（JSON、列表、枚举）。

不用 LangChain 怎么办？
  → 在 Prompt 里写"请输出 JSON 格式"→ json.loads() → 手动 try/except
  → 痛点：LLM 可能输出 {"name": "张三", }  ← 多余逗号
          或输出 ```json {...} ```            ← 被 markdown 包裹
          或直接不输出 JSON                    ← json.loads 直接炸

LangChain OutputParser 的价值：
  1. 自动处理 markdown 包裹（```json ... ```）
  2. 解析失败时自动把错误信息喂回 LLM 重试（Self-Correction）
  3. 类型安全：Pydantic 模型直接验证

运行方式：
  python demo_03_output_parser.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# 方式一：不用 LangChain —— 手工 JSON 解析
# ═══════════════════════════════════════════════════════════════════════════

def without_langchain():
    print("=" * 60)
    print("  方式一：手工 JSON 解析")
    print("=" * 60)

    import json
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[{"role": "user", "content": (
            "请以 JSON 格式返回以下信息：\n"
            "姓名：张三，年龄：25，技能：Python,Java,Go\n"
            '格式：{"name": "...", "age": ..., "skills": [...]}\n'
            "只输出 JSON，不要其他内容。"
        )}],
    )

    raw = response.choices[0].message.content.strip()
    print(f"LLM 原始输出: {raw}")

    # 手工解析 —— 这里是最容易出错的地方
    try:
        # 去掉可能的 markdown 包裹
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]  # 去掉 ```json
            if raw.endswith("```"):
                raw = raw.rsplit("\n", 1)[0]  # 去掉 ```
        data = json.loads(raw)
        print(f"解析成功: {data}")
    except json.JSONDecodeError as e:
        print(f"解析失败！{e}")
        print("  需要手写 fallback / retry / 正则提取...")

    print("""
    ❌ 手工解析的痛点：
       1. markdown 包裹 → 要手写字符串处理
       2. LLM 输出格式不稳定 → json.loads 经常炸
       3. 炸了怎么办 → 手写 retry + 把错误喂回 LLM
       4. 想验证字段类型 → 手写 if isinstance()
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 方式二：用 LangChain StrOutputParser（最简单）
# ═══════════════════════════════════════════════════════════════════════════

def with_str_output_parser():
    print("\n" + "=" * 60)
    print("  方式二：StrOutputParser —— 最简单的输出解析器")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.1)
    parser = StrOutputParser()

    # chain = prompt | llm | parser
    # AIMessage 对象 → parser → 纯字符串
    prompt = ChatPromptTemplate.from_messages([
        ("user", "用一句话解释：{concept}"),
    ])

    chain = prompt | llm | parser

    result = chain.invoke({"concept": "OutputParser"})
    print(f"结果类型: {type(result).__name__}")
    print(f"结果内容: {result}")

    print("""
    ✅ StrOutputParser 的作用：
       把 AIMessage 对象（含 metadata 等）→ 纯字符串
       这样上层代码拿到的是 str，不用关心 LangChain 内部类型
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 方式三：用 LangChain JsonOutputParser（结构化提取）
# ═══════════════════════════════════════════════════════════════════════════

def with_json_output_parser():
    print("\n" + "=" * 60)
    print("  方式三：JsonOutputParser —— 自动解析 JSON")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.1)

    # 告诉 parser 你期望什么 key
    # parser 会自动在 Prompt 里追加格式说明！
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "提取用户信息。\n{format_instructions}"),
        ("user", "{text}"),
    ])

    # .partial() 提前填入不变量：让 parser 告诉 LLM 输出什么格式
    prompt_with_format = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt_with_format | llm | parser

    result = chain.invoke({
        "text": "张三，25岁，Python工程师，擅长Java和Go",
    })
    print(f"JSON 输出: {result}")
    print(f"类型: {type(result).__name__}")  # dict！


# ═══════════════════════════════════════════════════════════════════════════
# 方式四：PydanticOutputParser —— 类型安全的结构化输出
# ═══════════════════════════════════════════════════════════════════════════

def with_pydantic_output_parser():
    print("\n" + "=" * 60)
    print("  方式四：PydanticOutputParser —— 类型安全输出")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.1)

    # 第一步：定义你想要的数据结构
    class Person(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        skills: list[str] = Field(description="技能列表")

    # 第二步：创建 PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "提取用户信息。\n{format_instructions}"),
        ("user", "{text}"),
    ])

    chain = prompt.partial(
        format_instructions=parser.get_format_instructions()
    ) | llm | parser

    result = chain.invoke({"text": "李四，28岁，全栈工程师，会React和Python和PostgreSQL"})
    print(f"类型: {type(result).__name__}")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"技能: {result.skills}")

    # 这就是类型安全 —— IDE 能自动补全 result.name！
    print(f"\n✅ Pydantic 验证通过：{result.name} 的年龄是 {result.age} 岁")

    print("""
    ✅ PydanticOutputParser 的核心价值：
       1. 自动生成 format_instructions（告诉 LLM 输出什么 JSON schema）
       2. LLM 输出后自动 Pydantic 验证（类型不匹配会抛异常）
       3. 异常后自动 retry（LangChain 会重新让 LLM 输出正确格式）
       4. 类型安全 —— IDE 自动补全，重构不怕改字段名
    """)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  📤 Demo 03：OutputParser —— 让 LLM 输出结构化数据")
    print("=" * 60)

    without_langchain()
    with_str_output_parser()
    with_json_output_parser()
    with_pydantic_output_parser()

    print("\n✅ Demo 03 完成！")
    print("关键收获：OutputParser 自动处理 markdown 包裹 + 类型验证 + 解析失败自动 retry。")
    print("PydanticOutputParser = 最强的选择，直接产出类型安全的 Python 对象。\n")


if __name__ == "__main__":
    main()
