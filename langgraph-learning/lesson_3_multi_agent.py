"""
=============================================================================
  LangGraph 第3课：Multi-Agent 系统 —— 多 Agent 协作
  ──────────────────────────────────────────────────────────────────────────
  目标：理解多 Agent 协作模式——Supervisor 调度、专业 Agent 各司其职
        实现一个"研究报告生成系统"
=============================================================================

第2课回顾：
  一个 Agent = LLM + 工具 + 循环
  Agent 的"大脑"（LLM）自己决定什么时候用工具、什么时候回复。

第3课进阶：
  多个 Agent 协同工作——每个 Agent 有自己的专长和工具。
  Supervisor Agent 作为"项目经理"，把任务分给最合适的"专家 Agent"。

  为什么需要多 Agent？
    - 单一 Agent 的工具太多，LLM 容易选错
    - 不同任务需要不同的提示词和工具
    - 分工协作可以处理更复杂的任务

  类比：一个创业公司
    - CEO（Supervisor）分配任务
    - 研究员（Research Agent）负责查资料
    - 写手（Writer Agent）负责写文档
    - 大家都是"专家"，各司其职

=============================================================================
"""

import os
import json
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ============================================================================
# Step 1：定义 State
# ──────────────────────────────────────────────────────────────────────────
# 多 Agent 的 State 比单 Agent 多了几个字段：
#   - next_agent: Supervisor 决定"下一步交给谁"
#   - research_findings: Research Agent 的研究结果
#   - final_report: Writer Agent 的最终输出
#
# 通俗理解：这些额外字段是"部门间的交接单"
# ============================================================================

class MultiAgentState(TypedDict):
    """多 Agent 系统的状态。

    messages 是所有 Agent 共享的对话历史。
    额外字段是"协作缓冲区"——Agent 之间传递的数据。
    """
    messages: Annotated[list, add_messages]
    next_agent: str           # Supervisor 决定：'researcher' | 'writer' | 'FINISH'
    research_findings: str    # Research Agent 的研究结果
    final_report: str         # Writer Agent 的最终输出


# ============================================================================
# Step 2：初始化 LLM
# ──────────────────────────────────────────────────────────────────────────
# 注意：每个 Agent 用各自的 LLM 实例（可以配置不同的 prompt 和 model）
# ============================================================================

def create_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )


# ============================================================================
# Step 3：定义 Research Agent 的工具
# ──────────────────────────────────────────────────────────────────────────
# Research Agent 有"查资料"的能力
# ============================================================================

@tool
def search_web(query: str) -> str:
    """搜索互联网获取信息。当需要查找最新资料、事实、数据时使用。

    Args:
        query: 搜索关键词，越具体越好
    """
    # 模拟搜索结果
    knowledge_base = {
        "python": (
            "Python 是 Guido van Rossum 于 1991 年发布的编程语言。\n"
            "最新稳定版本：Python 3.13（2024年10月发布）。\n"
            "Python 以简洁的语法和丰富的生态著称。"
        ),
        "人工智能": (
            "人工智能（AI）是计算机科学的分支，旨在创造能模拟人类智能的系统。\n"
            "2025-2026 年主要趋势：\n"
            "  1. 多模态 AI 模型（文本+图像+音频）\n"
            "  2. AI Agent 自动化工作流\n"
            "  3. 小型高效模型的兴起（SLM）\n"
            "  4. AI 编程助手普及"
        ),
        "llm": (
            "大型语言模型（LLM）是基于 Transformer 架构的深度学习模型。\n"
            "主流模型：GPT-4o、Claude 4 Opus、Gemini 2.0、DeepSeek-V4。\n"
            "关键能力：文本生成、代码编写、推理分析、工具使用。"
        ),
        "langgraph": (
            "LangGraph 是 LangChain 团队开发的框架，用于构建有状态的 Agent 应用。\n"
            "核心概念：StateGraph、Node、Edge、Conditional Edge。\n"
            "应用场景：多 Agent 协作、复杂工作流、AI 应用后端。"
        ),
    }

    # 模糊匹配
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value

    return f"搜索 '{query}' 的结果：未找到相关记录（模拟搜索）"


@tool
def calculate(expression: str) -> str:
    """执行数学计算。当需要数据分析、统计计算时使用。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# Research Agent 的工具列表
research_tools = [search_web, calculate]

# 用于结构化输出的 parser
from langchain_core.output_parsers import JsonOutputParser


# ============================================================================
# Step 4：定义 Supervisor Agent
# ──────────────────────────────────────────────────────────────────────────
# Supervisor 是"项目经理"。
# 它不做具体工作，而是：
#   1. 看用户的问题是什么
#   2. 判断应该派哪个专家去处理
#   3. 当专家完成工作后，判断是否要交给另一个专家，或者结束
#
# 实现方式：
#   Supervisor 的 LLM 被要求输出 JSON：{"next": "researcher|writer|FINISH"}
# ============================================================================

SUPERVISOR_PROMPT = """你是一个 AI 项目经理（Supervisor），负责协调两个专家 Agent 完成任务。

专家团队：
  1. researcher（研究员）—— 负责搜索信息、查找资料、做计算和分析
  2. writer（写手）—— 负责根据研究结果撰写报告、文章、总结

你的职责：
  1. 分析用户请求，判断工作流程
  2. 如果需要查资料、做分析 → 先派 researcher
  3. 如果已有研究结果需要整理成报告 → 派 writer
  4. 如果是简单聊天（问候、闲聊等）→ 派 chat
  5. 如果任务完成（专家们已经完成了所有工作）→ next: FINISH

输出格式（必须是纯 JSON，不要任何其他文字）：
  {{"next": "researcher"}}
  {{"next": "writer"}}
  {{"next": "chat"}}
  {{"next": "FINISH"}}

工作流示例：
  - 用户问"介绍一下 AI 的发展趋势" → researcher → writer → FINISH
  - 用户问"计算 25*4 等于多少" → researcher → FINISH（不需要 writer）
  - 用户问"你好" → chat → FINISH（简单回复）

重要规则：
  - 第一步：看你收到的第一条用户请求
  - 第二步以后：看最新的 [研究结果/报告/回复] 来判断"做完了没有"
  - 如果某个专家已经完成了工作，派下一个专家或 FINISH
  - 一次只派一个专家。等专家完成后，你会再次被问到下一步"""


def supervisor_node(state: MultiAgentState) -> dict:
    """Supervisor 节点：决定下一步派哪个专家。

    这个函数是整个多 Agent 系统的"大脑"。
    它查看当前状态，决定谁应该接下来工作。
    """
    # 准备 Supervisor 的消息
    supervisor_messages = [SystemMessage(content=SUPERVISOR_PROMPT)]

    # 告诉 Supervisor 当前的状态——谁已经做了啥
    status = "当前状态：\n"
    if state.get("research_findings"):
        status += f"  研究结果：已有\n"
    if state.get("final_report"):
        status += f"  最终报告：已有\n"
    if not state.get("research_findings") and not state.get("final_report"):
        status += "  尚无任何结果\n"

    # 最新消息是用户请求
    last_msg = state["messages"][-1].content
    status += f"\n用户请求：{last_msg[:200]}"
    supervisor_messages.append(HumanMessage(content=status))

    # 调用 LLM 做决策
    llm = create_llm()
    response = llm.invoke(supervisor_messages)

    # 解析决策结果
    try:
        decision = json.loads(response.content.strip().strip("```json").strip("```").strip())
        next_agent = decision.get("next", "FINISH")
    except:
        next_agent = "FINISH"

    print(f"  ──▶ [Supervisor] 决策: next = {next_agent}")

    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"[Supervisor] 下一步: {next_agent}")],
    }


# ============================================================================
# Step 5：定义 Research Agent（研究员）
# ──────────────────────────────────────────────────────────────────────────
# Research Agent 负责查资料、做计算、整理信息。
# 它有自己的工具（search_web、calculate）和提示词。
# ============================================================================

RESEARCHER_PROMPT = """你是一个专业的研究员（Researcher Agent）。

你的职责：
  1. 使用 search_web 工具搜索用户想了解的信息
  2. 使用 calculate 工具做数据分析或计算
  3. 整理研究成果，输出一份结构化的"研究报告"

重要规则：
  - 先搜索信息，再整理结果
  - 如果用户需要多种信息，逐一搜索
  - 研究报告用中文，结构清晰

当你完成研究后，输出 "【研究完成】" 开头的结果。"""


def researcher_node(state: MultiAgentState) -> dict:
    """Research Agent 节点：执行研究任务。"""
    print("  ──▶ [Researcher] 开始研究工作...")

    # 创建 Research Agent 自己的 LLM（绑定研究工具）
    llm = create_llm().bind_tools(research_tools)

    # 收集上下文：用户的原始请求
    research_messages = [SystemMessage(content=RESEARCHER_PROMPT)]

    # 加上用户请求
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            research_messages.append(msg)
            break

    # Agent 循环：LLM 决定调工具还是回复
    max_rounds = 5
    for _ in range(max_rounds):
        response = llm.invoke(research_messages)
        research_messages.append(response)

        # 如果 LLM 想调工具
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"      🔍 Researcher 调用: {tc['name']}({tc['args']})")
                # 找到对应的工具函数
                for tool_fn in research_tools:
                    if tool_fn.name == tc["name"]:
                        result = tool_fn.invoke(tc["args"])
                        research_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                        print(f"        结果: {result[:80]}...")
                        break
        else:
            # LLM 直接回复了——研究完成
            break

    # 提取研究成果
    research_result = research_messages[-1].content if research_messages else "无研究结果"

    print(f"  ──▶ [Researcher] 完成！")

    return {
        "research_findings": research_result,
        "messages": [AIMessage(content=f"[研究员报告]\n{research_result}")],
    }


# ============================================================================
# Step 6：定义 Writer Agent（写手）
# ──────────────────────────────────────────────────────────────────────────
# Writer Agent 不调用工具，它专攻写作。
# 它读取 Research Agent 的研究成果，写出优雅的文档。
# ============================================================================

WRITER_PROMPT = """你是一个专业的写手（Writer Agent）。

你的职责：
  根据研究员提供的资料，撰写一份结构清晰、易读性强的报告。

写作要求：
  - 使用 Markdown 格式
  - 结构：标题 → 概述 → 详细内容 → 总结
  - 语言流畅自然
  - 适当使用列表、表格等格式

重要：基于研究员提供的资料撰写，不要编造信息。"""


def writer_node(state: MultiAgentState) -> dict:
    """Writer Agent 节点：撰写最终报告。"""
    print("  ──▶ [Writer] 开始撰写报告...")

    llm = create_llm()

    # Writer 不调工具，直接基于研究结果写作
    research = state.get("research_findings", "无研究资料")

    writer_messages = [
        SystemMessage(content=WRITER_PROMPT),
        HumanMessage(content=f"请根据以下研究资料撰写报告：\n\n{research}"),
    ]

    response = llm.invoke(writer_messages)

    print(f"  ──▶ [Writer] 完成！")
    print(f"      报告预览: {response.content[:100]}...")

    return {
        "final_report": response.content,
        "messages": [AIMessage(content=f"[最终报告]\n{response.content}")],
    }


# ============================================================================
# Step 6b：定义 Chat Agent（通用对话）
# ──────────────────────────────────────────────────────────────────────────
# 专门处理简单对话——打招呼、闲聊等
# ============================================================================

def chat_node(state: MultiAgentState) -> dict:
    """Chat Agent 节点：处理简单对话。"""
    print("  ──▶ [Chat] 处理对话...")

    llm = create_llm()

    chat_messages = [
        SystemMessage(content="你是一个友好的 AI 助手。用简短自然的语言回复用户。"),
        state["messages"][0] if isinstance(state["messages"][0], HumanMessage)
        else HumanMessage(content="你好！"),
    ]

    response = llm.invoke(chat_messages)

    print(f"      回复: {response.content[:80]}...")

    return {
        "messages": [AIMessage(content=response.content)],
    }


# ============================================================================
# Step 7：构建 Multi-Agent StateGraph
# ──────────────────────────────────────────────────────────────────────────
# 多 Agent 系统的图结构：
#
#   START → supervisor → (researcher → supervisor) / (writer → supervisor) / (END)
#                          ↑___________________________|
#
# 关键设计：supervisor 是"中枢"，每次 Agent 完成工作后都回到 supervisor
# ============================================================================

def router(state: MultiAgentState) -> str:
    """路由函数：根据 Supervisor 的决策，派给对应的 Agent。"""
    next_agent = state.get("next_agent", "FINISH")
    print(f"  ──▶ [Router] 路由到: {next_agent}")
    return next_agent


def build_multi_agent_system():
    """构建多 Agent 系统。"""
    print("=" * 60)
    print("  🏗️   构建 Multi-Agent 系统...")
    print("=" * 60)

    builder = StateGraph(MultiAgentState)

    # ---- 添加节点 ----
    builder.add_node("supervisor", supervisor_node)   # 项目经理
    builder.add_node("researcher", researcher_node)    # 研究员
    builder.add_node("writer", writer_node)            # 写手
    builder.add_node("chat", chat_node)                # 对话助手

    # ---- 入口 ----
    builder.add_edge(START, "supervisor")

    # ---- 条件边：Supervisor 决定派谁 ----
    builder.add_conditional_edges(
        "supervisor",
        router,
        {
            "researcher": "researcher",   # 派研究员
            "writer": "writer",           # 派写手
            "chat": "chat",               # 派对话助手
            "FINISH": END,                # 结束
        }
    )

    # ---- 所有 Agent 完成后都回到 Supervisor ----
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("chat", "supervisor")

    graph = builder.compile()

    print("  ✅  Multi-Agent 系统构建完成！")
    print("  📊  结构: supervisor → (researcher|writer|FINISH)")
    print()

    return graph


# ============================================================================
# Step 8：运行演示
# ============================================================================

def run_demo():
    """运行多 Agent 系统的多种场景。"""
    graph = build_multi_agent_system()

    scenarios = [
        {
            "title": "需要研究和写作的复杂任务",
            "query": "请介绍一下人工智能在2025-2026年的发展趋势，然后写一份简洁的报告",
        },
        {
            "title": "简单任务——仅需要计算",
            "query": "计算 1024 * 768 等于多少？",
        },
        {
            "title": "直接回复——不需要任何专家",
            "query": "你好！",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print("=" * 70)
        print(f"  🎬  场景{i}：{scenario['title']}")
        print(f"      用户: '{scenario['query']}'")
        print("=" * 70)
        print()

        initial_state = {
            "messages": [HumanMessage(content=scenario["query"])],
            "next_agent": "",
            "research_findings": "",
            "final_report": "",
        }

        final = graph.invoke(initial_state)

        print()
        # 展示最终输出
        if final.get("final_report"):
            print(f"  📝  最终报告:")
            print()
            print(final["final_report"])
        else:
            last_content = None
            for msg in reversed(final["messages"]):
                if isinstance(msg, AIMessage) and not msg.content.startswith("[Supervisor]"):
                    last_content = msg.content
                    break
            if last_content:
                print(f"  💬 回复: {last_content}")
            else:
                print(f"  💬 (无回复)")

        print()
        print("─" * 70)
        print()


# ============================================================================
# 图解 Multi-Agent 系统
# ============================================================================

def print_diagram():
    """打印多 Agent 系统的架构图。"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    Multi-Agent 系统架构                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║                         ┌──────────────┐                             ║
║                  START ─►│  supervisor  │◄──── return ────────┐     ║
║                         └──────┬───────┘                      │     ║
║                                │                              │     ║
║                    ┌───────────┼────────────┐                 │     ║
║                    │           │            │                 │     ║
║               ┌────▼───┐ ┌────▼────┐  ┌────▼────┐           │     ║
║               │researc │ │ writer  │  │ FINISH  │           │     ║
║               │her     │ │        │  │  (END)  │           │     ║
║               └───┬────┘ └────┬────┘  └─────────┘           │     ║
║                   │           │                              │     ║
║                   └───────────┘──────────────────────────────┘     ║
║                                                                      ║
║  工作流程：                                                            ║
║    1. 用户提出请求                                                    ║
║    2. Supervisor 分析：需要哪些专家参与？                               ║
║    3. → researcher：查资料、做计算                                     ║
║    4. → 回到 supervisor：研究完成了，下一步？                           ║
║    5. → writer：写报告                                                ║
║    6. → 回到 supervisor：报告写完了，下一步？                           ║
║    7. → FINISH：任务完成                                               ║
║                                                                      ║
║  为什么要分多个 Agent？                                                ║
║    - 单一 Agent 工具越多，LLM 越容易"混淆"                             ║
║    - 每个 Agent 的提示词可以针对性优化                                 ║
║    - Supervisor 做"窄决策"（选人）比单个 Agent 做"宽决策"（选工具+回复）更准  ║
║    - 可扩展：加一个新 Agent 不影响已有的 Agent                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# 总结
# ============================================================================

print_summary = """
┌──────────────────────────────────────────────────────────────┐
│                     第3课关键知识点                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Multi-Agent 模式                                          │
│     • 一个 Supervisor + N 个 Specialist Agents                │
│     • Supervisor 做决策（派谁），Specialist 做执行（干活）     │
│     • 任务完成后都回到 Supervisor，形成"中心辐射"结构          │
│                                                              │
│  2. 为什么有效                                                │
│     • 每个 Agent 的工具集更小 → LLM 选择更准确                 │
│     • 每个 Agent 的提示词更专注 → 行为更可预测                 │
│     • Supervisor 的"决策空间"小 → 出错概率低                   │
│                                                              │
│  3. 三种边模式                                                │
│     • 固定边（add_edge）：A → B                               │
│     • 条件边（add_conditional_edges）：根据 State 动态路由     │
│     • **循环边**：A → B → A（这是 Agent 循环的基础）           │
│                                                              │
│  4. Agent 间的通信                                            │
│     • 通过 State 共享数据（research_findings → writer）       │
│     • 通过 messages 记录执行历史                              │
│     • 通过 next_agent 字段传递"调度信号"                      │
│                                                              │
│  5. 和真实系统的差距                                           │
│     • 真实系统会有 error handling、retry、timeout              │
│     • 每个 Agent 可能是独立的 StateGraph（子图）               │
│     • Supervisor 可以检查 Agent 的输出质量，要求重做           │
│                                                              │
│  6. 什么时候用 Multi-Agent？                                   │
│     • 任务包含多个明显的阶段（调研→写作→审核）                 │
│     • 不同阶段需要不同的能力和知识                             │
│     • 单一 Agent 的工具太多（>10 个）                         │
│     • 需要审计每一步谁做了什么                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print_diagram()
    run_demo()
