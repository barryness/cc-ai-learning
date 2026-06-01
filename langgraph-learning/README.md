# LangGraph 学习

从 StateGraph 基础到 Multi-Agent 系统，一步步掌握 LangGraph。

## 课程结构

| 课程 | 文件 | 核心知识点 |
|------|------|-----------|
| 第1课：StateGraph 基础 | `lesson_1_state_graph.py` | State 定义、Node、Edge、Conditional Edge、编译执行 |
| 第2课：Tool Calling Agent | `lesson_2_tool_agent.py` | @tool、bind_tools、ToolNode、tools_condition、Agent 循环 |
| 第3课：Multi-Agent 系统 | `lesson_3_multi_agent.py` | Supervisor 调度、专业 Agent、多 Agent 协作模式 |

## 运行方式

```bash
cd langgraph-learning
python3 lesson_1_state_graph.py
python3 lesson_2_tool_agent.py
python3 lesson_3_multi_agent.py
```

## 先决条件

1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env`（参考 `.env` 文件中的 API Key 和 Base URL）

## 学习路径

### 第1课：StateGraph 基础

核心概念：
- **State**：TypedDict，定义图的"内存结构"
- **Node**：Python 函数，处理 State 并返回更新
- **Edge**：连接节点，定义流程
- **Conditional Edge**：根据 State 内容动态路由

从最简单的"分析→路由→回复"状态机开始，理解 LangGraph 的"状态机 + LLM"思考方式。

### 第2课：Tool Calling Agent

让 Agent 学会自主决定"用工具还是直接回复"：
- `@tool` 装饰器：函数 → LLM 可调用的工具
- `bind_tools()`：把工具注册给 LLM
- `ToolNode`：自动执行 tool_calls
- `tools_condition`：检查 LLM 是否想调工具

Agent 循环：思考 → 行动 → 思考 → 行动 → ... → 直到直接回复

### 第3课：Multi-Agent 系统

多 Agent 协作——Supervisor 调度 + 专业 Agent 各司其职：
- **Supervisor Agent**：项目经理，决定派谁
- **Researcher Agent**：研究员，搜索信息、计算分析
- **Writer Agent**：写手，根据研究结果撰写报告
- **Chat Agent**：对话助手，处理简单对话

架构：START → supervisor → (researcher|writer|chat) → supervisor → ... → FINISH

## LangGraph 核心思维

```
传统编程：  if/else 硬编码流程
LangGraph： 定义 State + 节点函数 + 边规则
           LLM 在节点内部工作，StateGraph 在节点之间工作
```

## 资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
