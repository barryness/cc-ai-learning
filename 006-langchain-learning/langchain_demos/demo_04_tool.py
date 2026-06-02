"""
=============================================================================
  Demo 04：Tool —— 让 LLM 调用外部函数
=============================================================================

问题：LLM 只能"说"，不能"做"。怎么让它查数据库、调 API、发邮件？

不用 LangChain 怎么办？
  → 手写 function calling：
    1. 定义 JSON Schema 描述函数
    2. 调用 LLM 时传入 tools 参数
    3. 解析 LLM 返回的 tool_call → 执行对应函数 → 把结果再喂给 LLM
  → 痛点：Schema 手写容易出错，tool_call 到函数的 dispatch 要手写 if/else

LangChain @tool 的价值：
  1. 用 Python 函数 + 类型注解 + docstring 自动生成 JSON Schema
  2. bind_tools() 自动绑定到 LLM
  3. ToolMessage 标准化回传结果

运行方式：
  python demo_04_tool.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════
# 模拟外部数据源
# ═══════════════════════════════════════════════════════════════════════════

COMPANY_DB = {
    "AI学习科技": {"成立": 2024, "员工": 200, "总部": "北京"},
    "极客教育": {"成立": 2020, "员工": 500, "总部": "杭州"},
}

WEATHER_DB = {
    "北京": "晴，25°C，湿度40%",
    "杭州": "多云，28°C，湿度65%",
    "深圳": "阵雨，30°C，湿度80%",
}


# ═══════════════════════════════════════════════════════════════════════════
# 方式一：不用 LangChain —— 手工 Function Calling
# ═══════════════════════════════════════════════════════════════════════════

def without_langchain():
    print("=" * 60)
    print("  方式一：手工 Function Calling")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 手写 JSON Schema（容易出错，字段名打错不会报语法错）
    tools_schema = [{
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "查询公司基本信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "公司名称"},
                },
                "required": ["company_name"],
            },
        },
    }]

    # 第一次调用：LLM 决定要不要调 tool
    response = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "user", "content": "AI学习科技公司是哪年成立的？"}],
        tools=tools_schema,
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"LLM 想调用: {func_name}({args})")

        # 手工 dispatch（函数多了要写一堆 if/elif）
        if func_name == "get_company_info":
            result = COMPANY_DB.get(args["company_name"], {})
        else:
            result = {}

        # 第二次调用：把 tool 结果喂回 LLM
        response2 = client.chat.completions.create(
            model=MODEL, temperature=0,
            messages=[
                {"role": "user", "content": "AI学习科技公司是哪年成立的？"},
                {"role": "assistant", "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, ensure_ascii=False)},
            ],
        )
        print(f"最终回答: {response2.choices[0].message.content}")

    print("""
    ❌ 手工 Function Calling 的痛点：
       1. JSON Schema 手写，字段打错名不会报语法错
       2. tool_call → 函数 dispatch 要手写 if/elif，函数多了维护困难
       3. 多轮 tool 调用（先查A再查B）逻辑越来越复杂
    """)


# ═══════════════════════════════════════════════════════════════════════════
# 方式二：用 LangChain @tool —— 自动生成 Schema
# ═══════════════════════════════════════════════════════════════════════════

def with_langchain_tool():
    print("\n" + "=" * 60)
    print("  方式二：LangChain @tool 装饰器 + bind_tools()")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage

    # 用 @tool 装饰器，docstring 自动生成 Schema！
    @tool
    def get_company_info(company_name: str) -> dict:
        """查询公司基本信息，包括成立年份、员工数量、总部所在地"""
        return COMPANY_DB.get(company_name, {"error": "未找到该公司"})

    @tool
    def get_weather(city: str) -> str:
        """查询城市当前天气"""
        return WEATHER_DB.get(city, "未找到该城市的天气数据")

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0)

    # bind_tools() 把工具绑定到 LLM
    # LLM 看到用户消息 → 自动判断是否需要调 tool → 输出 tool_call
    llm_with_tools = llm.bind_tools([get_company_info, get_weather])

    # 第一轮：LLM 自动决定调用 get_company_info
    query = "AI学习科技公司总部在哪？员工多少人？"
    print(f"用户: {query}")

    ai_msg = llm_with_tools.invoke([HumanMessage(content=query)])

    # ai_msg.tool_calls 自动解析好了
    for tc in ai_msg.tool_calls:
        print(f"  LLM 决定调用: {tc['name']}({tc['args']})")
        # 按名字 dispatch（可以用 dict 映射，不需要 if/elif）
        tool_map = {"get_company_info": get_company_info, "get_weather": get_weather}
        func = tool_map[tc["name"]]
        result = func.invoke(tc["args"])
        print(f"  返回结果: {result}")
        # ToolMessage 是标准的回传格式
        from langchain_core.messages import ToolMessage
        tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
        # 把结果喂回 LLM
        final = llm_with_tools.invoke([HumanMessage(content=query), ai_msg, tool_msg])
        print(f"  最终回答: {final.content}")


# ═══════════════════════════════════════════════════════════════════════════
# 进阶：多工具链式调用
# ═══════════════════════════════════════════════════════════════════════════

def demo_multi_tool_chain():
    print("\n" + "=" * 60)
    print("  进阶：LLM 自动组合多个工具")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    @tool
    def get_company_info(company_name: str) -> dict:
        """查询公司基本信息"""
        return COMPANY_DB.get(company_name, {})

    @tool
    def get_weather(city: str) -> str:
        """查询城市天气"""
        return WEATHER_DB.get(city, "未知")

    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0)
    llm_with_tools = llm.bind_tools([get_company_info, get_weather])
    tool_map = {"get_company_info": get_company_info, "get_weather": get_weather}

    query = "查一下极客教育的总部在哪，然后告诉我那里的天气"
    print(f"用户: {query}")

    from langchain_core.messages import HumanMessage, ToolMessage
    messages = [HumanMessage(content=query)]

    # 多轮 tool 调用循环
    for round_num in range(3):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # 没有 tool_call → LLM 觉得可以回答了
            print(f"\n  最终回答: {ai_msg.content}")
            break

        for tc in ai_msg.tool_calls:
            print(f"  [第{round_num+1}轮] 调用 {tc['name']}({tc['args']})")
            func = tool_map[tc["name"]]
            result = func.invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))


# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  🔧 Demo 04：Tool —— 让 LLM 调用外部函数")
    print("=" * 60)

    without_langchain()
    with_langchain_tool()
    demo_multi_tool_chain()

    print("\n✅ Demo 04 完成！")
    print("关键收获：@tool 装饰器自动从函数签名+docstring 生成 JSON Schema。")
    print("bind_tools() 让 LLM 自动判断何时调哪个工具，不需要手写 dispatch。\n")


if __name__ == "__main__":
    main()
