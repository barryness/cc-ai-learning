"""
=============================================================================
  LangGraph 第2课：Tool Calling Agent —— 让 Agent 学会用工具
  ──────────────────────────────────────────────────────────────────────────
  目标：理解 Agent 的核心循环——LLM 决定"该调用工具还是直接回复"
        掌握 ToolNode、tools_condition、bind_tools
=============================================================================

第1课回顾：
  StateGraph = 状态机 + LLM
  我们手动写了 router 来判断下一步。

第2课进阶：
  让 LLM 自己决定"要不要用工具"——这才是真正的 Agent。

  Agent 循环：
    LLM 思考 → 需要工具吗？→ 调用工具 → 拿到结果 → LLM 继续思考 → ...
    ↑ 直到 LLM 认为"够了，我可以直接回复了"才结束

  LangGraph 内置支持：
    - ToolNode：自动执行工具调用的节点
    - tools_condition：自动判断"LLM 想调用工具还是想回复"
    - bind_tools：把工具"注册"给 LLM，让它知道有哪些工具可用

=============================================================================
"""

from typing import Annotated, TypedDict
import os
import json
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ============================================================================
# Step 1：定义 State
# ──────────────────────────────────────────────────────────────────────────
# Agent 的 State 非常简单——只需要消息列表。
# LangGraph 提供了 MessagesState，本质上就是：
#
#   class MessagesState(TypedDict):
#       messages: Annotated[list, add_messages]
#
# 我们自己定义是为了看得更清楚。
# ============================================================================

class AgentState(TypedDict):
    """Agent 状态。就一个字段：消息列表。

    add_messages 归约器的效果：
      节点 A 返回 {'messages': [msg1]}
      节点 B 返回 {'messages': [msg2]}
      → State 中 messages = [msg1, msg2]（追加，不是覆盖）
    """
    messages: Annotated[list, add_messages]


# ============================================================================
# Step 2：定义 Tool（工具）
# ──────────────────────────────────────────────────────────────────────────
# 工具是 Agent 的"手脚"。定义方式很简单：写一个函数，加 @tool 装饰器。
#
# @tool 做了什么：
#   1. 把函数转成 Tool 对象（含 name、description、args_schema）
#   2. description 会展示给 LLM，帮助 LLM 决定"什么时候用这个工具"
#   3. 函数签名自动解析为 inputSchema
#
# 重要：description 要写清楚"什么时候用"，因为 LLM 靠它做选择
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。支持 +、-、*、/、幂运算等。

    用法示例：
      - "25 + 3 * 8" → 49
      - "(100 + 50) / 3" → 50
      - "2 ** 10" → 1024

    当用户需要做数学计算时使用这个工具。
    """
    try:
        # 安全地计算表达式
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_time() -> str:
    """获取当前的日期和时间。

    当用户问"现在几点"、"今天星期几"、"当前日期"时使用。
    """
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S （%A）")


@tool
def get_weather(city: str) -> str:
    """查询某个城市当前的天气情况。

    当用户问"XX 城市天气怎么样"、"今天冷吗"、"要不要带伞"时使用。

    Args:
        city: 城市名称，如 "北京"、"上海"、"深圳"
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temp": 28, "condition": "晴天", "humidity": "30%"},
        "上海": {"temp": 32, "condition": "多云", "humidity": "65%"},
        "深圳": {"temp": 35, "condition": "阵雨", "humidity": "80%"},
        "杭州": {"temp": 30, "condition": "阴天", "humidity": "70%"},
        "广州": {"temp": 34, "condition": "雷阵雨", "humidity": "85%"},
    }

    if city in weather_data:
        w = weather_data[city]
        return (
            f"{city}天气：\n"
            f"  温度：{w['temp']}°C\n"
            f"  天气：{w['condition']}\n"
            f"  湿度：{w['humidity']}"
        )
    else:
        return f"暂无 {city} 的天气数据。支持的城市：{', '.join(weather_data.keys())}"


# ============================================================================
# Step 3：绑定 Tool 到 LLM
# ──────────────────────────────────────────────────────────────────────────
# bind_tools 是关键 API。它把工具列表"注册"给 LLM，使得：
#   1. LLM 知道有哪些工具可用（通过 tool 的 name 和 description）
#   2. 当 LLM 认为"这个问题需要工具"，返回 AIMessage.tool_calls
#   3. tool_calls 包含了：工具名、参数（LLM 自动生成的 JSON）
#
# 类比：
#   不加工具 = "面试官，我只能回答问题"
#   加了工具 = "面试官，我还能用计算器、查天气、查时间"
# ============================================================================

# 收集所有工具
tools = [calculator, get_time, get_weather]

# 初始化 LLM（使用 .env 中的 DeepSeek 配置）
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# 绑定工具——"注册"给 LLM
# 这之后 LLM 在生成回复时会考虑"我可以用 calculator 来算这个"
llm_with_tools = llm.bind_tools(tools)


# ============================================================================
# Step 4：定义 Agent 节点（LLM 决策节点）
# ──────────────────────────────────────────────────────────────────────────
# 这个节点是 Agent 的"大脑"。
# 它把当前所有消息发给 LLM，让 LLM 决定下一步做什么。
#
# LLM 的决定有两种：
#   1. 生成 tool_calls → 表示"我需要调用工具"
#   2. 生成普通回复 → 表示"我可以直接回答用户"
#
# LangGraph 的 tools_condition 会自动检查这一点。
# ============================================================================

def agent_node(state: AgentState) -> dict:
    """Agent 的"思考"节点。

    输入：当前 State（包含所有历史消息）
    输出：LLM 的新消息（可能包含 tool_calls）
    流程：把所有消息发给 LLM → LLM 决定下一步
    """
    messages = state["messages"]

    # 调用 LLM（LLM 内部会用 bind_tools 注册的工具信息）
    response = llm_with_tools.invoke(messages)

    # 如果 LLM 决定调用工具：
    #   response.tool_calls = [{'name': 'calculator', 'args': {'expression': '25+3*8'}, ...}]
    # 如果 LLM 决定直接回复：
    #   response.tool_calls = [] 或 None
    #   response.content = "答案是 49，因为 25 + 24 = 49"

    return {"messages": [response]}


# ============================================================================
# Step 5：构建 Agent StateGraph
# ──────────────────────────────────────────────────────────────────────────
# 这就是经典的 Agent 循环结构：
#
#   START → agent → tools_condition → (需要工具 → ToolNode → agent)
#                                   → (直接回复 → END)
#
# tools_condition 是 LangGraph 内置的条件边，它会检查：
#   如果 messages[-1] 有 tool_calls → 去 ToolNode
#   如果 messages[-1] 没有 tool_calls → 去 END
#
# 通俗理解：
#   agent 节点 = "大脑思考"
#   tools_condition = "大脑决定：需要动手还是直接说？"
#   ToolNode = "手，执行动作"
#   循环 = 思考→行动→思考→行动→...→直到直接说
# ============================================================================

def build_agent():
    """构建 Tool Calling Agent。"""
    print("=" * 60)
    print("  🏗️   构建 Tool Calling Agent...")
    print("=" * 60)

    builder = StateGraph(AgentState)

    # ---- 添加节点 ----
    # agent 节点：LLM 决策
    builder.add_node("agent", agent_node)
    # ToolNode：自动执行工具
    #   它会读取 messages[-1].tool_calls，逐个调用对应的工具函数
    #   然后把结果包装成 ToolMessage
    builder.add_node("tools", ToolNode(tools))

    # ---- 设置入口 ----
    builder.add_edge(START, "agent")

    # ---- 条件边：Agent 决策循环 ----
    # tools_condition 是内置路由：
    #   - 如果 LLM 返回了 tool_calls → "tools"
    #   - 如果 LLM 直接回复了 → END
    builder.add_conditional_edges(
        "agent",
        tools_condition,   # 内置路由函数
    )

    # ---- 工具执行完后回到 agent 继续思考 ----
    builder.add_edge("tools", "agent")

    graph = builder.compile()

    print("  ✅  Agent 构建完成！")
    print("  📊  结构: agent → (需要工具? tools → agent) / (直接回复? END)")
    print()

    return graph


# ============================================================================
# Step 6：运行演示
# ============================================================================

def run_demo():
    """运行多个场景，展示 Agent 的决策过程。"""
    graph = build_agent()

    # ---- 工具可用的模型列表 ----
    # 注意：不是所有模型都支持 tool calling。
    # OpenAI GPT-4、GPT-4o、DeepSeek V4 Flash 等支持。
    # 如果不支持，LLM 永远不会返回 tool_calls。

    scenarios = [
        "你好！今天天气怎么样？",
        "计算 25 + 3 * 8 等于多少？",
        "帮我查一下杭州的天气",
        "现在几点了？",
        "给我讲个笑话",
    ]

    for i, query in enumerate(scenarios, 1):
        print("=" * 60)
        print(f"  🎬  场景{i}：'{query}'")
        print("=" * 60)
        print()

        initial_state = {
            "messages": [
                SystemMessage(content="你是一个有用的 AI 助手。可以使用工具来获取信息。"),
                HumanMessage(content=query),
            ]
        }

        # stream 模式：一步步观察 Agent 的思考过程
        print("  📡  执行过程：")
        print()
        for step, output in enumerate(graph.stream(initial_state)):
            for node_name, result in output.items():
                if node_name == "agent" and "messages" in result:
                    msg = result["messages"][-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"      🤔 LLM 决定调用工具: {tc['name']}")
                            print(f"         参数: {tc['args']}")
                    else:
                        print(f"      💬 LLM 回复: {msg.content[:100]}...")
                elif node_name == "tools" and "messages" in result:
                    for msg in result["messages"]:
                        print(f"      🛠️  工具返回: {msg.content[:100]}")

        print()
        # 获取最终回复
        final = graph.invoke(initial_state)
        print(f"  📝  最终回复: {final['messages'][-1].content}")
        print()


# ============================================================================
# 图解 Agent 循环
# ============================================================================

def print_diagram():
    """打印 Agent 循环的架构图。"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    Tool Calling Agent 循环                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║                        ┌─────────┐                              ║
║               START ──►│  agent  │◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐      ║
║                        └────┬────┘                     │        ║
║                             │                          │        ║
║                    ┌────────▼────────┐                 │        ║
║                    │ tools_condition │—— 有 tool_call  │        ║
║                    └────────┬────────┘                 │        ║
║                             │                          │        ║
║                  ┌──────────┴──────────┐               │        ║
║                  │                     │               │        ║
║              ┌───▼───┐           ┌─────▼────┐         │        ║
║              │ tools │           │   END    │         │        ║
║              └───┬───┘           └──────────┘         │        ║
║                  │                                     │        ║
║                  └─────────────────────────────────────┘        ║
║                                                                  ║
║  流程详解：                                                        ║
║    1. agent 节点思考用户的问题                                     ║
║    2. tools_condition 检查 LLM 的决定：                            ║
║       - 如果要调用工具 → 进 tools 节点执行                          ║
║       - 如果要直接回复 → 进 END                                    ║
║    3. tools 节点执行完 → 回到 agent 继续思考（带工具结果）          ║
║    4. 循环直到 LLM 认为"够了，我可以回复了"                        ║
║                                                                  ║
║  关键理解：                                                        ║
║    Agent = LLM + 工具 + 循环                                       ║
║    LLM 是"大脑"——决定要不要用工具                                   ║
║    工具是"手脚"——执行具体操作                                       ║
║    循环是"毅力"——直到有最终答案才停                                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# 关键知识点总结
# ============================================================================

print_summary = """
┌──────────────────────────────────────────────────────────────┐
│                     第2课关键知识点                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. @tool 装饰器                                              │
│     • 把 Python 函数变成 LLM 可调用的工具                      │
│     • name + description → LLM 用来决定"什么时候用"            │
│     • 函数参数 → LLM 用来生成调用参数                          │
│                                                              │
│  2. bind_tools(tools)                                         │
│     • 把工具列表注册给 LLM                                    │
│     • LLM 的回复中会包含 tool_calls（如果有需要）              │
│                                                              │
│  3. ToolNode                                                  │
│     • LangGraph 内置节点，自动执行 tool_calls                  │
│     • 读取 messages[-1].tool_calls → 调用对应工具             │
│     • 结果包装成 ToolMessage → 追加回 messages                │
│                                                              │
│  4. tools_condition                                           │
│     • LangGraph 内置条件边                                    │
│     • 如果 messages[-1] 有 tool_calls → 去 ToolNode           │
│     • 否则 → END                                              │
│                                                              │
│  5. Agent 循环的核心模式                                      │
│     agent → decision → tools → agent → decision → ...        │
│     ↑______________________工具结果_____________________↑     │
│                                                              │
│  6. 和传统编程的区别                                          │
│     传统：if/else 硬编码流程                                  │
│     Agent：LLM 根据语义动态决定流程                            │
│     例如："25+3*8" → LLM 决定用 calculator                    │
│           "杭州天气" → LLM 决定用 get_weather                  │
│           用户不需要指定"用哪个工具"——LLM 自己判断             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print_diagram()
    run_demo()
