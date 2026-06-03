# Agent 设计：从 LLM 到自主智能体

> 从 "调用一次模型" 到 "让模型自主完成一件事" —— Agent 设计的完整指南。

## 目录

1. [Agent 是什么](#1-agent-是什么)
2. [Planning 是什么](#2-planning-是什么)
3. [ReAct 是什么](#3-react-是什么)
4. [Tool Calling 是什么](#4-tool-calling-是什么)
5. [Reflection 是什么](#5-reflection-是什么)
6. [综合架构：一个完整的 Agent](#6-综合架构一个完整的-agent)
7. [Research Agent 实现分析](#7-research-agent-实现分析)
8. [Agent 设计模式对比](#8-agent-设计模式对比)
9. [生产级 Agent 的挑战](#9-生产级-agent-的挑战)

---

## 1. Agent 是什么

### 一句话定义

**Agent（智能体）= LLM + 工具 + 记忆 + 自主决策循环**

Agent 不是单次调用 LLM，而是让 LLM 进入一个"思考-行动-观察"的循环，直到完成任务。

### 从 Chat 到 Agent

```
Chat（聊天）:    用户 → LLM → 回复 → 结束
Agent（智能体）: 用户 → LLM → 思考 → 选择工具 → 执行 → 观察结果 → 思考 → ... → 完成任务
```

Chat 模式是"一问一答"，Agent 模式是"自主完成一个目标"。

### Agent 的四个核心要素

```
┌──────────────────────────────────────────────────┐
│                    Agent                          │
│                                                   │
│   ┌──────────┐    ┌──────────────┐               │
│   │   LLM    │    │   Planning   │               │
│   │ (大脑)   │◄──►│  (决策策略)   │               │
│   └──────────┘    └──────────────┘               │
│        │                 │                        │
│        ▼                 ▼                        │
│   ┌──────────┐    ┌──────────────┐               │
│   │  Tools   │    │   Memory     │               │
│   │ (手脚)   │    │  (记忆系统)   │               │
│   └──────────┘    └──────────────┘               │
│                                                   │
└──────────────────────────────────────────────────┘
```

| 要素 | 角色 | 类比 |
|------|------|------|
| **LLM（大脑）** | 理解任务、推理、决策 | 大脑皮层 |
| **Planning（策略）** | 拆解任务、规划步骤 | 前额叶 |
| **Tools（工具）** | 执行具体操作（搜索、计算、读写） | 手脚 |
| **Memory（记忆）** | 记住上下文、经验、用户偏好 | 海马体 |

### 一个最简单的 Agent

```python
# 最简 Agent：循环直到 LLM 说 STOP
def simple_agent(task: str) -> str:
    history = [{"role": "user", "content": task}]

    for _ in range(10):  # 防止无限循环
        response = llm.chat(history)

        if "STOP" in response:
            return response  # 任务完成

        if "SEARCH:" in response:
            query = extract_query(response)
            result = search(query)           # 执行工具
            history.append({"role": "user", "content": f"搜索结果: {result}"})

    return "达到最大步数"
```

关键点：
- **循环**：不是调用一次，而是一直循环直到完成
- **工具调用**：LLM 通过特定格式声明要调用的工具
- **观察反馈**：工具结果注入到对话中，LLM 据此调整下一步

---

## 2. Planning 是什么

### 核心思想

**Planning（规划）= 在行动之前先想清楚怎么做**

人类接到复杂任务时，不会直接动手，而是先拆解成子任务。Agent 也需要这个能力。

### 两种 Planning 策略

#### 2.1 隐式 Planning（ReAct 自带）

LLM 在每一步"思考"时自然形成了规划，但不显式列出。适合简单任务。

```
Thought: 用户想知道今天的天气。我需要先搜索。
Action: search("北京 今天 天气 2026-06-03")
Observation: 北京今天晴，22-35°C
Thought: 我已经拿到数据，可以回答了。
Final Answer: 北京今天晴，气温 22-35°C。
```

#### 2.2 显式 Planning（Plan-and-Execute）

先让 LLM 生成完整计划，再逐步执行。适合复杂多步任务。

```python
# 第一步：生成计划
plan = llm.chat("""
请为"调研 LangGraph 和 CrewAI 的差异"生成执行计划。
列出步骤，每步一个可执行的动作。
""")
# → 计划：
# 1. 搜索 "LangGraph 是什么"
# 2. 搜索 "CrewAI 是什么"
# 3. 搜索 "LangGraph vs CrewAI comparison"
# 4. 整理搜索结果为对比报告
# 5. 输出最终报告

# 第二步：按计划逐步执行
for step in plan:
    result = execute_step(step)
    if step_invalidated_by_result(step, result):
        plan = revise_plan(plan, result)  # 计划调整
```

### Planning 的权衡

| 维度 | 隐式 Planning | 显式 Planning |
|------|--------------|---------------|
| 延迟 | 低（边想边做） | 高（先规划再执行） |
| Token 消耗 | 少 | 多 |
| 复杂任务 | 容易丢失方向 | 有明显路线图 |
| 适应变化 | 灵活 | 需要重规划 |
| 可解释性 | 弱 | 强（计划可见） |

---

## 3. ReAct 是什么

### 核心思想

**ReAct = Reasoning（推理）+ Acting（行动）交替进行**

这是 Google DeepMind 2022 年提出的范式，目前是 Agent 设计的基础模型。

### ReAct 循环

```
┌─────────────────────────────────────────────────────┐
│                   ReAct Loop                         │
│                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│   │ Thought  │───►│  Action  │───►│ Observation  │  │
│   │ (推理)   │    │ (行动)   │    │  (观察)      │  │
│   └──────────┘    └──────────┘    └──────────────┘  │
│         ▲                              │            │
│         └──────────────────────────────┘            │
│                   循环直到 Final Answer              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 一个完整的 ReAct 示例

```
任务：LangGraph 最新版本是多少？它支持哪些 Python 版本？

Step 1:
  Thought: 我需要搜索 LangGraph 的最新版本信息。
  Action: search("LangGraph latest version 2026")
  Observation: LangGraph 1.2.0 released, supports Python 3.10-3.13

Step 2:
  Thought: 找到了版本号和 Python 支持范围，信息齐全。
  Final Answer: LangGraph 最新版是 1.2.0，支持 Python 3.10 到 3.13。
```

### ReAct 的 Prompt 模板

```python
REACT_PROMPT = """你是一个研究助手。按照以下格式回答：

Question: 用户的问题
Thought: 你当前的思考
Action: 要执行的动作，格式：tool_name(tool_input)
Observation: 工具返回的结果
... (Thought/Action/Observation 可重复)
Thought: 最终思考
Final Answer: 最终答案

可用工具：
- search(query: str): 搜索互联网
- calculator(expression: str): 计算数学表达式

Question: {question}
{history}
"""
```

### ReAct 为什么有效

1. **思考链可见**：每一步推理都可追溯，方便调试
2. **工具与推理解耦**：推理归 LLM，执行归工具，各司其职
3. **自我纠正**：观察到意外结果时，LLM 可以调整下一步
4. **可中断可恢复**：任何一步都可以检查、修改后继续

---

## 4. Tool Calling 是什么

### 核心思想

**Tool Calling = LLM 以结构化方式声明要调用哪个函数，而非生成自然语言指令**

这是 Function Calling 的泛化。OpenAI 发明了 Function Calling，Anthropic 的 Tool Use 与之类似，MCP 将其标准化。

### 三个关键步骤

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 1. 定义工具  │────►│ 2. LLM 决策  │────►│ 3. 执行结果  │
│ (Tool Spec) │     │ (返回调用)  │     │ (反馈LLM)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 工具定义（Tool Specification）

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索互联网获取最新信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    }
                },
                "required": ["path"]
            }
        }
    }
]
```

**三要素**：`name`（唯一标识）、`description`（何时使用）、`inputSchema`（参数 JSON Schema）

### LLM 的 Tool Call 响应

```json
{
    "role": "assistant",
    "content": null,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": "{\"query\": \"LangGraph latest version 2026\"}"
            }
        }
    ]
}
```

关键：LLM 不执行工具，只声明"要调用什么、传什么参数"。由 Agent 框架负责实际执行。

### Tool Calling vs 自然语言指令

```
旧方式（不可靠）:
  LLM 输出: "请搜索 LangGraph 最新版本"
  → 需要正则解析，格式不稳定

新方式（Tool Calling）:
  LLM 输出: {tool_calls: [{function: "search", arguments: {query: "..."}}]}
  → 结构化，可直接调度函数
```

---

## 5. Reflection 是什么

### 核心思想

**Reflection（反思）= Agent 审视自己的输出，发现不足并自我改进**

这是让 Agent 从"做完"到"做好"的关键机制。

### Reflection 循环

```
┌────────────────────────────────────────────────────┐
│               Reflection Loop                       │
│                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐ │
│   │ Generate │───►│  Review  │───►│   Revise     │ │
│   │ (生成)   │    │ (审视)   │    │   (修改)     │ │
│   └──────────┘    └──────────┘    └──────────────┘ │
│                         │               │           │
│                    检查通过？         不通过        │
│                         │               │           │
│                         ▼               │           │
│                     ┌──────────┐        │           │
│                     │  Output  │◄───────┘           │
│                     │ (输出)   │  通过后输出         │
│                     └──────────┘                    │
│                                                     │
└────────────────────────────────────────────────────┘
```

### 反思的 Prompt

```python
REFLECTION_PROMPT = """请审视你刚生成的回答，按以下维度和严格程度检查：

1. 事实准确性：所有数据是否有来源支撑？
2. 完整性：是否覆盖了用户问题的所有方面？
3. 逻辑性：论证链条是否有漏洞？
4. 清晰度：表述是否容易理解？

给出：
- 评分（1-10）
- 具体问题清单
- 改进建议

如果评分 >= 8，输出 "PASS"；否则输出具体的修改方案。
"""
```

### Reflection 的两种模式

| 模式 | 机制 | 适用场景 |
|------|------|----------|
| **内省式** | 同一个 LLM 审视自己的输出 | 事实性、完整性检查 |
| **对抗式** | 另一个 LLM 审视（Critic 角色） | 需要不同视角的场景 |
| **执行式** | 执行输出内容，检查是否成功 | 代码、配置生成 |

### 防止无限反思

```python
MAX_REFLECTION_ROUNDS = 3  # 最多修改 3 次

for round in range(MAX_REFLECTION_ROUNDS):
    output = agent.generate(task)
    review = agent.reflect(output)

    if review["score"] >= 8 or review["decision"] == "PASS":
        break  # 质量达标

    task = f"{task}\n\n修改建议：{review['feedback']}"
```

类比：代码 Review。写代码的人先提交 PR，Reviewer 检查，发现问题→修改→再检查，直到通过或达到修改上限。

---

## 6. 综合架构：一个完整的 Agent

### 架构图

```
                         ┌─────────────────────────────────┐
                         │        Agent Orchestrator        │
                         │                                  │
    用户任务 ──────────►  │  ┌───────────────────────────┐  │
                         │  │      Planning Loop         │  │
                         │  │  Plan → Execute → Observe  │  │
                         │  └───────────────────────────┘  │
                         │              │                   │
                         │     ┌────────┼────────┐          │
                         │     ▼        ▼        ▼          │
                         │  ┌─────┐ ┌──────┐ ┌──────────┐  │
                         │  │ReAct│ │Reflct│ │ Memory   │  │
                         │  │循环 │ │反思  │ │ 存取     │  │
                         │  └──┬──┘ └──┬───┘ └────┬─────┘  │
                         │     │       │          │         │
                         └─────┼───────┼──────────┼─────────┘
                               │       │          │
                               ▼       ▼          ▼
                         ┌──────────┐ ┌──────────────┐
                         │  Tools   │ │  Long-term   │
                         │ search   │ │  Memory      │
                         │ read     │ │  (vector DB) │
                         │ write    │ └──────────────┘
                         └──────────┘
```

### 数据流

```python
class Agent:
    def run(self, task: str) -> str:
        # 1. 规划阶段
        plan = self.plan(task)

        # 2. 执行循环（ReAct）
        for step in plan.steps:
            while not step.completed:
                thought = self.think(step)
                action = self.decide_action(thought)  # Tool Calling
                observation = self.execute(action)
                self.remember(thought, action, observation)  # Memory

            # 3. 反思阶段（可选）
            if step.should_reflect:
                review = self.reflect(step.result)
                if not review.passed:
                    step.revise(review.feedback)
                    continue  # 回到执行循环

        # 4. 生成最终输出
        return self.generate_final_output()
```

---

## 7. Research Agent 实现分析

### 功能架构

```
Research Agent
│
├── Phase 1: 基础版
│   ├── ReAct 推理循环
│   ├── search 工具（模拟）
│   └── 自动生成报告
│
├── Phase 2: + 记忆
│   ├── 短期记忆（对话历史）
│   ├── 长期记忆（研究主题向量化存储）
│   └── 上下文压缩（避免 token 爆炸）
│
└── Phase 3: + 反思
    ├── 报告自审（事实、完整、逻辑）
    ├── 自动修改循环
    └── 质量评分
```

### 关键设计决策

1. **搜索是模拟的**：避免依赖外部 API，用预设知识库 + 关键词匹配模拟搜索行为
2. **状态用 TypedDict**：兼容 LangGraph 风格，字段类型明确
3. **记忆用简单 dict**：演示概念，生产环境换向量数据库
4. **反思用评分制**：1-10 分，>= 8 通过，防止无限修改

### 工具设计

| 工具 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `search` | 搜索知识库 | query: str | 匹配结果列表 |
| `read_note` | 读笔记 | topic: str | 笔记内容 |
| `write_report` | 写报告 | content: str | 文件路径 |

### 防止幻觉的策略

- 所有结论必须引用 search 结果
- 反思阶段专门检查"是否有来源支撑"
- 无搜索结果时，Agent 应诚实报告而非编造

---

## 8. Agent 设计模式对比

| 模式 | 核心循环 | 优点 | 缺点 | 适用场景 |
|------|----------|------|------|----------|
| **ReAct** | Think → Act → Observe | 简单可靠 | 复杂任务容易迷路 | 搜索、问答 |
| **Plan-Execute** | Plan → Execute | 路线清晰 | 计划可能过时 | 多步工作流 |
| **ReWOO** | Plan → Tool use → Solve | 减少 LLM 调用 | 缺少中间纠正 | Token 敏感的批量任务 |
| **LLMCompiler** | 并行执行 DAG | 速度快 | 依赖关系需明确 | 可并行的多工具调用 |
| **Reflexion** | ReAct + 反思 | 质量高 | 延迟和成本翻倍 | 需要高质量输出的场景 |
| **Multi-Agent** | 多角色协作 | 专业化 | 协调开销大 | 复杂项目 |

### 选择决策树

```
任务复杂度？
│
├── 简单（1-2 步）→ 直接 ReAct
│
├── 中等（3-5 步）→ ReAct + Reflection
│
└── 复杂（> 5 步或有子任务）
    │
    ├── 子任务独立 → Multi-Agent（Supervisor 模式）
    └── 子任务有依赖 → Plan-Execute + Reflection
```

---

## 9. 生产级 Agent 的挑战

### 现实问题

| 挑战 | 表现 | 缓解方案 |
|------|------|----------|
| **Token 成本** | ReAct 循环消耗大量 token | 上下文压缩、摘要中间结果 |
| **幻觉** | LLM 编造工具调用结果 | 强制来源引用、反思检查 |
| **死循环** | Agent 在同一个工具上反复调用 | max_steps、循环检测、变化监控 |
| **工具失败** | 搜索超时、API 限流 | 重试机制、降级策略、超时处理 |
| **延迟** | 每步都要等 LLM 生成 | 并行工具调用、流式输出 |
| **可观测性** | 不知道 Agent 在做什么 | LangSmith、日志追踪、中间状态可视化 |

### 工程建议

1. **从小 Agent 开始**：一个 Agent 只做一件事，做对了再扩展
2. **工具设计是关键**：描述越清晰，LLM 选择越准确
3. **必须有上限**：max_steps、max_reflection_rounds、timeout
4. **日志即文档**：每一步的 Thought/Action/Observation 都应该记录
5. **先模拟后真实**：Demo 阶段用模拟工具，验证逻辑后再接入真实 API

---

## 延伸阅读

- [009-agent-learning/agent_demo.py](./agent_demo.py) — 三个阶段的完整实现代码
- [007-langgraph-learning](../007-langgraph-learning/langgraph_guide.md) — LangGraph 中 Agent 的图编排实现
- [008-mcp-learning](../008-mcp-learning/mcp_guide.md) — MCP 协议与 Tool Calling 的关系
- [006-langchain-learning](../006-langchain-learning/langchain_notes.md) — LangChain 中的 Agent 抽象
- [ReAct Paper (Google DeepMind, 2022)](https://arxiv.org/abs/2210.03629)
- [Reflexion Paper (2023)](https://arxiv.org/abs/2303.11366)
- [Plan-and-Execute Paper (2023)](https://arxiv.org/abs/2305.04091)
