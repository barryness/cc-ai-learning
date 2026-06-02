# LangGraph 学习笔记：从线性 Pipeline 到多 Agent 协作

> 从"LangChain Agent 不够用了"这个真实痛点出发。
> 不讲 API 文档，只讲为什么和怎么做。
> 
> 配套代码：`langgraph_demo.py`（3 个可运行 Demo）

---

## 目录

1. [为什么 LangChain Agent 不够？](#1-为什么-langchain-agent-不够)
2. [为什么需要 Graph？](#2-为什么需要-graph)
3. [核心概念：State](#3-核心概念state)
4. [核心概念：Node](#4-核心概念node)
5. [核心概念：Edge 和 Conditional Edge](#5-核心概念edge-和-conditional-edge)
6. [核心概念：Memory](#6-核心概念memory)
7. [Demo 1：简单工作流（问题→搜索→回答）](#7-demo-1简单工作流问题搜索回答)
8. [Demo 2：研究 Agent（拆解→搜索→总结→报告）](#8-demo-2研究-agent拆解搜索总结报告)
9. [Demo 3：多 Agent 系统（研究+写作+审核）](#9-demo-3多-agent-系统研究写作审核)
10. [LangGraph vs LangChain：什么时候用哪个](#10-langgraph-vs-langchain什么时候用哪个)
11. [生产环境注意事项](#11-生产环境注意事项)

---

## 1. 为什么 LangChain Agent 不够？

### 你已经学会了 LangChain

在 `006-langchain-learning` 里，你掌握了：
- `ChatPromptTemplate` → 构建 Prompt
- `ChatOpenAI` → 调用 LLM
- `@tool` + `bind_tools()` → Tool Calling
- `prompt | llm | parser` → LCEL 管道
- `retriever.invoke(query)` → 检索

你甚至用 LangChain 重构了 RAG 系统，代码从 300 行减少到 120 行。

### 但当你尝试构建真正自主的 Agent 时……

```python
# 用 LangChain 写一个"自主研究 Agent"：
# 需求：收到"研究 AI 趋势并写报告" → 自动搜索 → 自动总结 → 自动输出

# 你可能会这样写：
def research_agent(query):
    # Step 1: 拆解问题
    sub_qs = llm.invoke(f"把'{query}'拆成子问题").split("\n")

    # Step 2: 逐个搜索
    results = []
    for q in sub_qs:
        r = search_tool(q)
        results.append(r)

    # Step 3: 总结
    summary = llm.invoke(f"总结: {results}")

    # Step 4: 写报告
    report = llm.invoke(f"写报告: {summary}")

    # Step 5: 检查质量
    if len(report) < 100:       # ← 硬编码的检查逻辑
        report = llm.invoke(f"重写: {summary}")  # ← 手动重试

    return report
```

**问题在哪？**

| 问题 | 具体表现 |
|------|---------|
| **流程硬编码** | 拆解→搜索→总结→报告的步骤是写死的。如果搜索不满意想重新搜索？没法动态决定 |
| **没有状态管理** | `results`、`summary` 在函数内部，函数结束后就丢了。想追踪"Agent 中间干了什么"？做不到 |
| **条件逻辑脆弱** | `if len(report) < 100` — 这种硬编码的检查无法适应所有场景 |
| **循环难实现** | "搜索→不满意→改关键词再搜→满意→继续" — 需要 while 循环 + 手写退出条件 |
| **多人协作无法建模** | Research / Write / Review 三个 Agent 怎么交接？手写函数调用顺序？ |

**一句话：LangChain LCEL 擅长"线性 Pipeline"，但不擅长"有循环、有分支、有状态、有多角色的复杂工作流"。**

---

## 2. 为什么需要 Graph？

### 答案是：用"图"来建模 Agent 的工作流

```
LangChain LCEL（线性管道）:
  A → B → C → D
  数据从左到右，一步到底

LangGraph（状态图）:
       ┌───────┐
       │ 搜索   │←──────────┐
       └───┬───┘            │
           │                │
     ┌─────▼─────┐    ┌─────┴─────┐
     │ 结果够了？ │───▶│  总结报告  │
     └─────┬─────┘    └───────────┘
           │
    不够 → 回到搜索（循环）
```

### 图 vs 管道的本质区别

| | LangChain LCEL | LangGraph |
|------|------|------|
| 结构 | 线性管道 (`A|B|C`) | 有向图（节点+边） |
| 流程控制 | 固定顺序 | 动态路由（条件边） |
| 循环 | 不支持 | 天然支持（节点→节点→回来） |
| 状态 | 隐式（管道内传递） | 显式（State TypedDict） |
| 可观测 | 只能看最终输出 | 每个节点的输入输出都可追踪 |
| 多角色 | 需要手动编排 | 通过 State 天然共享 |

### 类比

```
LangChain LCEL = 工厂流水线
  产品从A站→B站→C站，固定路线，不能回头

LangGraph = 办公室协作
  经理（Supervisor）分配任务，员工（Agent）干活
  干完交回经理，经理决定下一步
  不满意 → 打回重做（循环！）
```

---

## 3. 核心概念：State

### 是什么

**State 是整个图的"共享内存"。** 所有节点都能读写它。

在 LangGraph 中，State 是一个 `TypedDict`：

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表（追加模式）
    search_results: str   # 搜索结果（覆盖模式）
    final_answer: str     # 最终答案（覆盖模式）
```

### 两个关键概念：Reducer（归约器）

```
Annotated[list, add_messages]  ← 追加模式
  节点A返回 {"messages": [msg1]}
  节点B返回 {"messages": [msg2]}
  → State.messages = [msg1, msg2]  (追加，不覆盖)

普通字段 (无 Annotated)  ← 覆盖模式
  节点A返回 {"search_results": "结果A"}
  节点B返回 {"search_results": "结果B"}
  → State.search_results = "结果B"  (后写的覆盖先写的)
```

### 直觉理解

```
State = Agent 的"工作台"
  ┌─────────────────────────────────┐
  │ messages: [用户问题, AI回复...]   │ ← 对话历史（不断追加）
  │ search_results: "搜索到..."      │ ← 本轮搜索结果
  │ final_answer: "报告如下..."      │ ← 最终输出
  └─────────────────────────────────┘

每个节点（Node）都是工人，从工作台上拿东西，做完放回去。
图（Graph）定义了工人之间的协作流程。
```

---

## 4. 核心概念：Node

### 是什么

**Node 是图中的一个执行单元。** 每个 Node 是一个 Python 函数：

```python
# Node 函数签名
def my_node(state: AgentState) -> dict:
    """
    输入：当前 State（整个工作台的快照）
    输出：dict（只包含要更新的字段）
    """
    # 读取 State
    last_msg = state["messages"][-1].content

    # 做处理...
    result = do_something(last_msg)

    # 返回要更新的字段
    return {
        "messages": [AIMessage(content=result)],
        "final_answer": result,
    }
```

### 关键规则

1. **Node 不直接修改 State** — 它返回一个 dict，框架负责合并
2. **返回值只包含要更新的字段** — 不需要返回整个 State
3. **一个 Node 只做一件事** — 搜索的只搜索，总结的只总结

### 直觉理解

```
Node = 办公室里的一个"岗位"
  搜索员：收到任务 → 搜索 → 把结果放回工作台
  写手：看到工作台上有搜索结果 → 写报告 → 放回工作台
  审核员：看到工作台上有报告 → 审核 → 放回工作台

每个岗位只看 State 中自己需要的部分，做完就放回去。
```

---

## 5. 核心概念：Edge 和 Conditional Edge

### Edge（普通边）

**固定路线：A 做完后必然去 B。**

```python
builder.add_edge("search", "answer")  # 搜索完 → 必然去回答
builder.add_edge("answer", END)       # 回答完 → 结束
```

### Conditional Edge（条件边）

**动态路线：A 做完后，根据 State 内容决定下一步。**

```python
def router(state: AgentState) -> str:
    """路由函数：看 State 决定去哪"""
    if state.get("search_satisfied"):
        return "answer"       # 搜索结果满意 → 去回答
    else:
        return "search"       # 不满意 → 重新搜索

builder.add_conditional_edges(
    "search",                          # 从 search 节点出发
    router,                            # 用这个函数决定
    {
        "answer": "answer",            # router 返回 "answer" → 去 answer 节点
        "search": "search",            # router 返回 "search" → 回 search 节点（循环！）
    }
)
```

### 直觉理解

```
普通边 = 工厂的固定传送带
  A站 → B站，产品必然流向 B

条件边 = 分拣机
  A站 → 检查产品标签 → 合格去B / 不合格回A（循环）
```

---

## 6. 核心概念：Memory

### Memory 的两个层次

```
┌──────────────────────────────────────────────┐
│ 层次 1：Short-term Memory（State）             │
│   一次 invoke() 内，State 在节点间传递          │
│   例：搜索结果 → 总结 → 报告，都在一次调用内     │
│   实现：State TypedDict                       │
├──────────────────────────────────────────────┤
│ 层次 2：Long-term Memory（Checkpointer）       │
│   跨多次 invoke()，状态持久化                   │
│   例：用户第1轮问"AI趋势"→Agent搜了→             │
│        用户第2轮追问"详细说说第2点"→              │
│        Agent 记得上一轮的搜索结果                │
│   实现：MemorySaver / SqliteSaver             │
└──────────────────────────────────────────────┘
```

### Checkpointer 的使用

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建内存检查点
checkpointer = MemorySaver()

# 编译时传入
graph = builder.compile(checkpointer=checkpointer)

# 每次调用时传入 thread_id（标识一次对话）
config = {"configurable": {"thread_id": "conversation_1"}}

# 第1轮
graph.invoke({"messages": [HumanMessage(content="研究AI趋势")]}, config)

# 第2轮（同一个 thread_id，Agent 记得上一轮的所有中间状态！）
graph.invoke({"messages": [HumanMessage(content="详细说说第2点")]}, config)
```

### 直觉理解

```
Short-term Memory (State) =
  一次会议中的白板 —— 会议结束就擦掉

Long-term Memory (Checkpointer) =
  会议纪要存档 —— 下次开会可以翻出来看

thread_id = 不同会议的编号 —— "项目A的第3次讨论"
```

---

## 7. Demo 1：简单工作流（问题→搜索→回答）

### 架构

```
START → search → answer → END

最简单的线性流程：
  1. search: 用工具搜索知识库
  2. answer: 基于搜索结果生成答案
```

### 为什么要这样做？

这是 LangGraph 的 "Hello World"。演示了：
- State 定义（最简单的 2 字段）
- Node 定义（2 个节点，各做一件事）
- 线性边（无分支、无循环）
- 和手写 pipeline 的对比（功能相同，但 LangGraph 版本可观测、可扩展）

### State 设计

```python
class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史
    search_results: str                       # 搜索结果
```

### 关键代码

```python
# 构建图
builder = StateGraph(SimpleState)
builder.add_node("search", search_node)
builder.add_node("answer", answer_node)
builder.add_edge(START, "search")
builder.add_edge("search", "answer")
builder.add_edge("answer", END)
graph = builder.compile()
```

---

## 8. Demo 2：研究 Agent（拆解→搜索→总结→报告）

### 架构

```
START → decompose → search → summarize → report → END

多步骤线性流程：
  1. decompose: LLM 把大问题拆成子问题
  2. search: 对每个子问题搜索
  3. summarize: 综合搜索结果
  4. report: 生成结构化报告
```

### 为什么要这样做？

Demo 1 只有 2 步，真实的研究任务要 4 步。这里演示了：
- 多步骤协作（每步的输出是下步的输入）
- State 字段随流程增长（search_results → summary → final_report）
- LLM 作为"处理器"（decompose/summarize/report）而不仅仅是"回答者"

### State 设计

```python
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    main_question: str          # 原始问题
    sub_questions: list[str]    # 拆解后的子问题
    search_results: dict        # {子问题: 搜索结果}
    summary: str                # 综合摘要
    final_report: str           # 最终报告
```

### 和 Demo 1 的关键区别

| | Demo 1 | Demo 2 |
|------|------|------|
| 节点数 | 2 | 4 |
| 复杂度 | 一问一答 | 拆解→搜索→总结→报告 |
| State | 2 字段 | 5 字段（每步一个产出） |
| LLM 角色 | 仅回答 | 拆解、总结、报告（3 次调用） |

---

## 9. Demo 3：多 Agent 系统（研究+写作+审核）

### 架构

```
         ┌──────────────────────┐
         │     Supervisor        │←──────────────────┐
         │   （项目经理）         │                    │
         └──────────┬───────────┘                    │
                    │                                │
       ┌────────────┼────────────┐                   │
       │            │            │                   │
       ▼            ▼            ▼                   │
  ┌─────────┐ ┌─────────┐ ┌─────────┐              │
  │Research │ │ Writer  │ │Reviewer │              │
  │ Agent   │ │ Agent   │ │ Agent   │              │
  └────┬────┘ └────┬────┘ └────┬────┘              │
       │           │           │                    │
       └───────────┴───────────┘────────────────────┘
               完成后都回到 Supervisor

典型流程：
  User: "写一篇 AI 趋势报告"
  → Supervisor: "先让研究 Agent 查资料"
  → Research Agent: 搜索+分析 → 放回 State
  → Supervisor: "资料有了，让 Writer 写报告"
  → Writer Agent: 基于研究结果写报告 → 放回 State
  → Supervisor: "报告写好了，让 Reviewer 审核"
  → Reviewer Agent: 审核 → 通过/打回修改
  → 如果打回：Writer 修改 → Reviewer 再审核（循环！）
  → 通过：FINISH
```

### 为什么要这样做？

Demo 2 的 4 个步骤是**线性写死**的，但真实场景中：
- 研究不满意 → 需要重新查（循环）
- 审核不通过 → 需要打回重写（循环）
- 不同任务需要不同的 Agent 组合（动态调度）

这就是 Supervisor 模式的价值：**用一个"经理 Agent"动态决策流程。**

### State 设计

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str              # Supervisor 决定：谁去干活
    task: str                    # 用户任务
    research_findings: str       # Research Agent 产出
    draft: str                   # Writer Agent 产出
    review_feedback: str         # Reviewer Agent 产出
    revision_count: int          # 修改次数（防止无限循环）
```

### Memory 演示

Demo 3 加入 Checkpointer，支持跨对话记忆：

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 第1轮：写报告
config = {"configurable": {"thread_id": "user_123"}}
graph.invoke({"messages": [...], "task": "研究AI趋势"}, config)

# 第2轮：追问（同一个 thread_id）
# Agent 记得上一轮的研究结果，不需要重新搜索！
graph.invoke({"messages": [HumanMessage(content="详细说说第二点")]}, config)
```

### 关键设计决策

**为什么用 Supervisor 而不是并行执行？**

```
并行执行（RunnableParallel）：
  研究、写作、审核同时进行
  → 但写作需要研究结果、审核需要写作结果 → 不能并行！

Supervisor 模式：
  按依赖顺序调度 → 研究完成 → 写作 → 审核
  审核不通过 → 打回写作 → 再审核（这是 LangChain 做不到的）
```

---

## 10. LangGraph vs LangChain：什么时候用哪个

```
你的应用流程是？

线性 → LangChain LCEL 足够
  │    例：RAG（检索→拼接Prompt→LLM→解析）
  │
  ├─ 有循环 → LangGraph
  │    例：搜索→不满意→改关键词再搜→满意→继续
  │
  ├─ 有动态分支 → LangGraph
  │    例：用户问技术问题→派技术Agent / 问HR问题→派HR Agent
  │
  ├─ 多角色协作 → LangGraph
  │    例：Research→Write→Review→打回重写
  │
  └─ 需要追踪中间状态 → LangGraph
       例：想知道"Agent 在第三步搜索了什么、结果如何"
```

### 两者配合使用

**LangGraph 内部大量使用 LangChain 的组件：**

```python
# LangGraph 的 Node 里用 LangChain 的组件
def researcher_node(state):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool

    # 这里全都是 LangChain 的积木块
    llm = ChatOpenAI(...)
    prompt = ChatPromptTemplate.from_messages([...])
    chain = prompt | llm | StrOutputParser()

    return {"research_findings": chain.invoke(state)}

# LangGraph 负责"流程控制"（什么时候调用谁）
# LangChain 负责"组件实现"（Prompt、LLM、Tool）
```

---

## 11. 生产环境注意事项

### 1. 防止无限循环

```python
class State(TypedDict):
    revision_count: int  # 关键：追踪重试次数

def supervisor(state):
    if state["revision_count"] > 3:
        return "FINISH"  # 强制结束，即使不满意
```

### 2. 错误处理

```python
def safe_node(state):
    try:
        result = risky_operation(state)
        return {"output": result}
    except Exception as e:
        return {"messages": [AIMessage(content=f"节点失败: {e}")]}
```

### 3. 超时控制

```python
# 对于可能耗时的节点（如搜索），设置超时
import signal

def search_with_timeout(state):
    signal.alarm(30)  # 30 秒超时
    try:
        result = search(state)
        signal.alarm(0)
        return result
    except TimeoutError:
        return {"search_results": "搜索超时"}
```

### 4. Checkpointer 选择

| Checkpointer | 场景 |
|------|------|
| `MemorySaver` | 开发/测试，进程重启数据丢失 |
| `SqliteSaver` | 单机部署，数据持久化到 SQLite |
| `PostgresSaver` | 生产环境，多实例共享 |

### 5. 可观测性

```python
# stream() 模式：逐步观察每个节点的输入输出
for step in graph.stream(initial_state):
    for node_name, output in step.items():
        print(f"[{node_name}] {output}")
```

---

## 总结

```
LangChain LCEL = 乐高积木 + 直线轨道
  → 适合：线性Pipeline（RAG、翻译、摘要）

LangGraph = 乐高积木 + 完整的铁路调度系统
  → 适合：有循环/分支/多角色的Agent系统

三个 Demo 的演进：
  Demo 1: 2节点线性 → 理解 State + Node + Edge
  Demo 2: 4节点线性 → 理解多步骤 + State 流转
  Demo 3: Supervisor模式 → 理解多Agent + 循环 + Memory

核心原则：
  LangGraph 的价值 = 让流程控制"可视化"和"可编程"
  不是写死 A→B→C，而是定义"在什么状态下、谁去干什么、干完去哪"
```

---

## 延伸阅读

- **运行 Demo**：`python langgraph_demo.py` — 本文档的配套代码，每个 Demo 可直接运行
- **官方文档**：[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- **前置知识**：`006-langchain-learning` — 理解 LangChain LCEL 和 Tool Calling
