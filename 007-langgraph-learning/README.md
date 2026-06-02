# LangGraph 学习

> 从"LangChain Agent 不够用了"出发，掌握 StateGraph、多 Agent 协作和 Memory。

## 文件结构

```
007-langgraph-learning/
├── langgraph_guide.md      ← 完整教学指南（11 章：为什么 → 核心概念 → 3 个 Demo 架构设计）
├── langgraph_demo.py       ← 3 个可运行 Demo（从简单工作流到多 Agent 系统）
├── .env                    ← API 配置
├── requirements.txt        ← Python 依赖
└── README.md               ← 你在这里
```

## 快速开始

```bash
cd 007-langgraph-learning
pip install -r requirements.txt
python langgraph_demo.py           # 运行所有 Demo
python langgraph_demo.py --demo 1  # 只运行 Demo 1
python langgraph_demo.py --demo 2  # 只运行 Demo 2
python langgraph_demo.py --demo 3  # 只运行 Demo 3
```

## 学习路线

| Demo | 主题 | 架构 | 核心知识点 |
|------|------|------|-----------|
| Demo 1 | 简单工作流 | START → search → answer → END | State、Node、Edge |
| Demo 2 | 研究 Agent | decompose → search → summarize → report | 多步骤协作、State 流转 |
| Demo 3 | 多 Agent 系统 | Supervisor + Research + Writer + Reviewer | 条件边、循环、Memory |

## 设计哲学

```
LangChain LCEL = 乐高积木 + 直线轨道
  → 适合：线性 Pipeline（RAG、翻译、摘要）

LangGraph = 乐高积木 + 完整的铁路调度系统
  → 适合：有循环/分支/多角色的 Agent 系统
```

**核心原则：LangGraph 的价值 = 让流程控制"可视化"和"可编程"。** 不是写死 A→B→C，而是定义"在什么状态下、谁去干什么、干完去哪"。

## 前置知识

学习本模块前，建议先完成：
- `006-langchain-learning` — 理解 LangChain LCEL、Tool Calling、Runnable 协议

## 资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
