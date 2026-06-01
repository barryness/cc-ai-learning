"""
=============================================================================
  LangGraph 第1课：StateGraph 基础
  ──────────────────────────────────────────────────────────────────────────
  目标：理解 StateGraph 的核心概念——State、Node、Edge、Conditional Edge
        掌握 LangGraph 的"状态机"思考方式
=============================================================================

LangGraph 的本质：
  LLM 应用本质上是一个"有状态"的流程——每轮对话都有上下文，
  每个 Agent 决策都依赖之前的执行结果。

  StateGraph = 状态机 + LLM
    - State（状态）: 所有数据在 StateGraph 中以 State 形式流动
    - Node（节点）: 每个节点是一个函数，接收 State，返回 State 更新
    - Edge（边）: 定义节点之间的连接
    - Conditional Edge（条件边）: 根据 State 内容决定下一步走向

  类比：StateGraph 就像一张"流程图"，State 是流动的"数据表"，
       每个节点是"处理站"，边是"传送带"。

=============================================================================
"""

# ============================================================================
# Part 0：安装与导入
# ============================================================================
# pip install langgraph langchain-core langchain-openai

from typing import Annotated, TypedDict, Literal
from typing_extensions import TypedDict
import json
import os
from dotenv import load_dotenv

# 加载 .env 文件中的 API 配置
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI


# ============================================================================
# Part 1：定义 State（状态）
# ──────────────────────────────────────────────────────────────────────────
# State 是 StateGraph 的灵魂。所有数据都在 State 中传递。
# State 是一个 TypedDict，定义了图的"内存结构"。
#
# 关键概念：
#   - TypedDict：Python 的类型化字典，每个字段有固定类型
#   - Annotated：LangGraph 用这个标记来定义"归约器"（Reducer）
#   - add_messages：内置的消息归约器，自动追加消息到列表
#   - 归约器决定了"当多个节点修改同一个字段时，如何合并"
#
# 通俗理解：
#   State = Agent 的"工作台"——上面放着所有需要的数据
#   add_messages = 消息的"追加模式"——新消息不会覆盖旧消息，而是追加到列表末尾
#   普通字段 = "覆盖模式"——新值直接替换旧值
# ============================================================================

class AgentState(TypedDict):
    """Agent 的完整状态定义。

    这个 State 会在整个图的生命周期中传递和更新。
    """
    # ---- 核心字段 ----
    # Annotated[list, add_messages] 意味着：
    # 当多个节点都返回 {'messages': [...]} 时，
    # add_messages 会自动合并（追加）消息列表，而不是覆盖
    messages: Annotated[list, add_messages]

    # ---- Agent 内部状态 ----
    next_step: str          # 下一步要做什么（'analyze' | 'respond' | 'end'）
    analysis: str           # Agent 的分析结果
    final_answer: str       # 最终的回复


# ============================================================================
# Part 2：定义 Node（节点）
# ──────────────────────────────────────────────────────────────────────────
# Node 是 StateGraph 的执行单元。每个节点是一个 Python 函数：
#
#   函数签名：def node_name(state: AgentState) -> dict:
#   输入：当前 State（只读）
#   输出：dict（要更新的 State 字段）
#
# 重要：节点函数不直接修改 state，而是返回要更新的字段。
#       LangGraph 框架负责合并更新。
#
# 通俗理解：
#   每个 Node 就像一个"工作站"——工人（函数）从工作台（State）拿材料，
#   加工后把产物放回工作台。
# ============================================================================

# 先初始化 LLM
# 使用 .env 中的配置（DeepSeek V4 Flash，兼容 OpenAI API）
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)


def node_analyze(state: AgentState) -> dict:
    """节点1：分析用户输入。

    这个节点读取用户的消息，判断用户的意图。
    它是图中的"第一个工作站"。
    """
    print("  ──▶ [analyze] 正在分析用户输入...")

    # 从 State 中获取最新的用户消息
    last_message = state["messages"][-1].content
    print(f"      用户说: '{last_message[:60]}...' " if len(last_message) > 60
          else f"      用户说: '{last_message}'")

    # 用 LLM 分析意图
    response = llm.invoke([
        SystemMessage(content="你是一个分析助手。根据用户输入，输出 JSON："
                               '{"sentiment": "positive|negative|neutral", '
                               '"topic": "简短话题概括", '
                               '"needs_tool": true|false}'),
        HumanMessage(content=f"分析这段输入：{last_message}"),
    ])

    # 解析 LLM 的分析结果
    try:
        analysis = json.loads(response.content.strip().strip("```json").strip("```").strip())
    except:
        analysis = {"sentiment": "neutral", "topic": "general", "needs_tool": False}

    analysis_text = f"情感: {analysis['sentiment']}, 话题: {analysis['topic']}"
    print(f"      分析结果: {analysis_text}")

    # 决定下一步
    if analysis.get("needs_tool", False):
        next_step = "use_tool"
    else:
        next_step = "respond"

    return {
        "analysis": analysis_text,
        "next_step": next_step,
    }


def node_use_tool(state: AgentState) -> dict:
    """节点2：使用工具（模拟）。

    这个节点模拟"调用工具"的过程。
    在实际 Agent 中，这里会调用搜索引擎、计算器、文件系统等。
    """
    print("  ──▶ [use_tool] 正在执行工具...")

    # 模拟工具调用结果
    tool_result = "工具执行结果：已获取相关信息。当前时间 2026 年 6 月。"
    print(f"      工具返回: '{tool_result}'")

    # 将工具结果作为 AI 消息追加到消息列表
    msg = AIMessage(content=f"[工具结果] {tool_result}")

    return {
        "messages": [msg],       # add_messages 会追加
        "next_step": "respond",  # 工具用完后去回复
    }


def node_respond(state: AgentState) -> dict:
    """节点3：生成回复。

    根据分析和（可能有的）工具结果，生成最终回复。
    """
    print("  ──▶ [respond] 正在生成回复...")

    # 收集所有上下文
    context = f"分析结果：{state['analysis']}"

    system_prompt = f"""你是一个友好的 AI 助手。
    根据分析结果和对话历史生成回复。
    上下文：{context}"""

    # 调用 LLM 生成回复
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"][-3:],  # 最近 3 条消息作为上下文
    ])

    print(f"      回复: '{response.content[:80]}...' " if len(response.content) > 80
          else f"      回复: '{response.content}'")

    return {
        "messages": [AIMessage(content=response.content)],
        "final_answer": response.content,
        "next_step": "end",
    }


# ============================================================================
# Part 3：定义 Edge（边）
# ──────────────────────────────────────────────────────────────────────────
# Edge 决定了"处理完一个节点后，下一步去哪"。
#
# 两种边：
#   1. 普通边：无条件，A → B 是固定的
#   2. 条件边：根据 State 内容动态选择下一步
#
# 通俗理解：
#   普通边 = "流水线的固定传送带"
#   条件边 = "分拣员"——根据产品（State）上的标签，推送到不同处理线
# ============================================================================

def router_next_step(state: AgentState) -> str:
    """条件边的"路由函数"。

    这个函数检查 State 中的 next_step 字段，
    返回下一步要去哪个节点。

    返回值：节点的名称（字符串）
    特殊节点：END 表示流程图结束

    通俗理解：
      这就是"分拣员"——看看工作单（State）上写着下一步做什么，
      然后把工件推到对应的处理站。
    """
    print(f"  ──▶ [router]  下一步: '{state['next_step']}'")
    return state["next_step"]


# ============================================================================
# Part 4：构建 StateGraph
# ──────────────────────────────────────────────────────────────────────────
# 构建一个 StateGraph 的步骤：
#   1. 创建图：StateGraph(AgentState) —— 告诉图"你的内存结构是这个"
#   2. 添加节点：graph.add_node("name", func) —— 注册处理站
#   3. 添加边：graph.add_edge(A, B) —— 铺设固定传送带
#   4. 添加条件边：graph.add_conditional_edges(A, router, {B: ..., C: ...})
#      —— 铺设分拣传送带
#   5. 设置入口：graph.set_entry_point("start_node") —— 工件从哪上线
#   6. 编译：graph.compile() —— 把设计图变成可运行的"机器"
# ============================================================================

def build_graph():
    """构建完整的 StateGraph。"""
    print("=" * 60)
    print("  🏗️   构建 StateGraph...")
    print("=" * 60)

    # ---- 步骤1：创建图 ----
    # StateGraph<AgentState> 表示这个图操作的是 AgentState 类型的数据
    builder = StateGraph(AgentState)

    # ---- 步骤2：添加节点 ----
    # 每个节点就是一个"处理站"
    builder.add_node("analyze", node_analyze)   # 分析站
    builder.add_node("use_tool", node_use_tool)  # 工具站（可选）
    builder.add_node("respond", node_respond)    # 回复站

    # ---- 步骤3：设置入口 ----
    # START 是一个特殊节点，表示图的起点
    # 所有数据从这里开始流动
    builder.add_edge(START, "analyze")

    # ---- 步骤4：添加条件边 ----
    # 从 analyze 节点出发，用 router_next_step 函数决定下一步
    # 映射表：{'use_tool': 去工具站, 'respond': 去回复站}
    builder.add_conditional_edges(
        "analyze",
        router_next_step,
        {
            "use_tool": "use_tool",
            "respond": "respond",
        }
    )

    # ---- 步骤5：添加固定边 ----
    # 工具站处理完后，固定去回复站
    builder.add_edge("use_tool", "respond")

    # 回复站处理完后，流程图结束
    builder.add_edge("respond", END)

    # ---- 步骤6：编译 ----
    # compile() 会检查图结构的完整性（没有悬空的节点、所有路径可达等）
    graph = builder.compile()

    print("  ✅  StateGraph 构建完成！")
    print("  📊  节点: analyze → (use_tool?) → respond")
    print()

    return graph


# ============================================================================
# Part 5：运行 StateGraph
# ──────────────────────────────────────────────────────────────────────────
# graph.invoke(input) 是运行图的入口。
# input 是一个 dict，包含 State 的初始值。
# 返回最终 State。
# ============================================================================

def run_demo():
    """运行完整的 StateGraph 演示。"""
    graph = build_graph()

    # ---- 演示1：简单问答（不需要工具） ----
    print("=" * 60)
    print("  🎬  演示1：简单问答（不需要工具）")
    print("=" * 60)
    print()

    initial_state = {
        "messages": [HumanMessage(content="你好！今天天气怎么样？")],
        "next_step": "analyze",
        "analysis": "",
        "final_answer": "",
    }

    result = graph.invoke(initial_state)

    print()
    print(f"  📝  最终回复: {result['final_answer']}")
    print()

    # ---- 演示2：需要工具的请求 ----
    print("=" * 60)
    print("  🎬  演示2：需要工具辅助的请求")
    print("=" * 60)
    print()

    initial_state = {
        "messages": [HumanMessage(content="帮我查一下 Python 最新版本是多少？")],
        "next_step": "analyze",
        "analysis": "",
        "final_answer": "",
    }

    result = graph.invoke(initial_state)

    print()
    print(f"  📝  最终回复: {result['final_answer']}")
    print()


# ============================================================================
# Part 6：图解 StateGraph 工作流程
# ============================================================================
def print_diagram():
    """打印 StateGraph 的架构图。"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    StateGraph 工作流程                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   ┌───────┐                                                  ║
║   │ START │  ← 入口：用户输入到达                             ║
║   └───┬───┘                                                  ║
║       │                                                      ║
║       ▼                                                      ║
║   ┌─────────┐   ┌──────────────────────┐                     ║
║   │ analyze │───│ router_next_step()   │                     ║
║   └─────────┘   └──────────┬───────────┘                     ║
║       │                    │                                 ║
║       │  ┌─────────────────┼──────────────────┐              ║
║       │  │ needs_tool=true │ needs_tool=false │              ║
║       ▼  ▼                 ▼                  ▼              ║
║   ┌──────────┐        ┌──────────┐                           ║
║   │ use_tool │        │ respond  │                           ║
║   └─────┬────┘        └────┬─────┘                           ║
║         │                  │                                  ║
║         └──────────────────┘                                  ║
║                            │                                  ║
║                            ▼                                  ║
║                        ┌──────┐                               ║
║                        │ END  │  ← 出口：回复给用户            ║
║                        └──────┘                               ║
║                                                              ║
║   State = {                                                  ║
║     messages:   [...],    ← 对话消息（自动追加）               ║
║     next_step:  "...",    ← 路由标记                          ║
║     analysis:   "...",    ← 分析结果                          ║
║     final_answer: "..."   ← 最终回复                          ║
║   }                                                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# 关键知识点总结
# ============================================================================
"""
┌──────────────────────────────────────────────────────────────┐
│                     关键知识点总结                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. State（状态）                                             │
│     • 本质：一个 TypedDict，定义了图的"数据结构"               │
│     • 归约器（Reducer）：控制"如何合并更新"                    │
│       - add_messages：追加模式                                │
│       - 默认：覆盖模式（后写的覆盖先写的）                      │
│                                                              │
│  2. Node（节点）                                              │
│     • 本质：一个 Python 函数                                   │
│     • 输入：当前 State（只读）                                 │
│     • 输出：dict（要更新的字段）                               │
│     • 原则：一个节点只做一件事                                 │
│                                                              │
│  3. Edge（边）                                                │
│     • add_edge(A, B)：固定路径，A 执行完必然去 B               │
│     • add_conditional_edges(A, router, mapping)：             │
│       根据 State 动态选择路径                                  │
│                                                              │
│  4. 编译与执行                                                │
│     • builder.compile()：检查图结构完整性                       │
│     • graph.invoke(input_state)：执行图                        │
│     • 返回值：最终 State                                       │
│                                                              │
│  5. 核心思维：LangGraph = 状态机 + LLM                        │
│     • 不是"写代码控制流程"，而是"定义状态和转换规则"            │
│     • LLM 在节点内部工作，StateGraph 在节点之间工作             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print_diagram()
    run_demo()
