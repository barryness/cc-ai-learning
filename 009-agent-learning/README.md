# Agent 学习

> 从 ReAct 到 Reflection —— 构建自主研究智能体。

## 文件结构

```
009-agent-learning/
├── agent_design.md       ← 完整教学指南（Agent → Planning → ReAct → Tool Calling → Reflection）
├── agent_demo.py         ← Research Agent 实现（3 个渐进阶段）
└── README.md             ← 你在这里
```

## 快速开始

```bash
cd 009-agent-learning

# 运行所有 Demo
python agent_demo.py

# 或分阶段运行
python agent_demo.py --phase 1   # 基础 ReAct
python agent_demo.py --phase 2   # + 记忆系统
python agent_demo.py --phase 3   # + 反思机制
```

## Demo 三阶段

| 阶段 | 主题 | 新增能力 |
|------|------|----------|
| Phase 1 | 基础 Research Agent | ReAct 循环、模拟搜索、自动报告生成 |
| Phase 2 | + 记忆系统 | 短期对话记忆、长期知识记忆、上下文压缩 |
| Phase 3 | + 反思机制 | 自审评分、问题检测、自动修改循环 |

## Agent 架构概览

```
┌─────────────────────────────────────────┐
│              Research Agent              │
│                                          │
│   Planning → ReAct → Tools → Memory     │
│       ↑                         │       │
│       └── Reflection ←──────────┘       │
│                                          │
└─────────────────────────────────────────┘
```

## 核心工具

| 工具 | 功能 | 备注 |
|------|------|------|
| `search` | 搜索知识库 | 模拟实现，关键词匹配 |
| `write_report` | 写报告到文件 | 输出 .md 格式 |
| `recall` (Phase 2) | 检索长期记忆 | 关键词索引 |
| `review` (Phase 3) | 报告质量审查 | 5 维度评分 |
| `revise` (Phase 3) | 自动修改报告 | 基于反馈修正 |

## 学习路线

1. 阅读 `agent_design.md` — 理解 5 个核心概念
2. 运行 `python agent_demo.py --phase 1` — 体验 ReAct 循环
3. 阅读 Phase 1 源码 — 理解状态机和 LLM 模拟
4. 运行 Phase 2 — 观察记忆的存取和压缩
5. 运行 Phase 3 — 观察反思如何改进输出
6. 对照 `agent_design.md` 第 6-8 章 — 理解设计决策

## 前置知识

- `007-langgraph-learning` — Agent 的图编排实践（Supervisor 模式）
- `008-mcp-learning` — Tool Calling 的标准化协议
- `006-langchain-learning` — LangChain 中的 Agent 和 Tool 抽象

## 延伸资源

- [ReAct Paper (2022)](https://arxiv.org/abs/2210.03629)
- [Reflexion Paper (2023)](https://arxiv.org/abs/2303.11366)
- [Plan-and-Execute Paper (2023)](https://arxiv.org/abs/2305.04091)
- [Anthropic Agent Design Guide](https://docs.anthropic.com/en/docs/agents-and-tools)
