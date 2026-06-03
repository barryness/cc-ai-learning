# Multi-Agent 学习

> 从单 Agent 到多 Agent 协作 —— PM + Architect + Coder + Reviewer 共同完成项目。

## 文件结构

```
010-multi-agent-learning/
├── multi_agent_architecture.md  ← 完整教学指南（职责划分 → 通信 → 状态 → 冲突解决 → LangGraph 实现）
├── multi_agent_demo.py         ← Multi-Agent 协作 Demo（PM + Architect + Coder + Reviewer）
├── index.html                  ← 交互学习网页
├── README.md                   ← 你在这里
├── requirements.txt            ← Python 依赖
└── .env.example                ← API 配置模板
```

## 快速开始

```bash
cd 010-multi-agent-learning

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 3. 运行 Demo
python multi_agent_demo.py
python multi_agent_demo.py "开发一个待办事项 REST API"
```

## 四个 Agent

```
                       ┌──────────────┐
                       │  Supervisor  │
                       │  (协调调度)   │
                       └──────┬───────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │    PM    │       │Architect │       │  Coder   │
    │ 需求拆解  │       │ 架构设计  │       │ 代码实现  │
    └──────────┘       └──────────┘       └────┬─────┘
                                               │
                                               ▼
                                        ┌──────────┐
                                        │ Reviewer │
                                        │ 代码审查  │
                                        └────┬─────┘
                                             │
                                   ┌─────────┼─────────┐
                                   │                   │
                              approved            revise
                                   │                   │
                                   ▼                   ▼
                                 END          Coder/Architect
```

## Agent 职责

| Agent | 角色 | 输入 | 输出 |
|-------|------|------|------|
| **PM** | 项目经理 | 用户需求 | 任务列表（P0/P1/P2） |
| **Architect** | 架构师 | 任务列表 | 架构文档（技术选型/数据模型/API/代码结构） |
| **Coder** | 开发者 | 架构文档 | 可运行的 Python 代码 |
| **Reviewer** | 审查员 | 需求+架构+代码 | 评分 + approved/revise |

## 工作流

```
用户: "开发用户管理系统"
  → PM: 拆解为注册/登录/查询 3 个任务
  → Architect: FastAPI + SQLite + JWT 三层架构
  → Coder: 编写完整代码
  → Reviewer: 审查 → 发现问题（密码未 hash）
  → Coder: 修改代码
  → Reviewer: 重新审查 → approved
  → 输出完整项目
```

## 学习路线

1. 阅读 `009-agent-learning/agent_design.md` — 单 Agent 基础
2. 阅读 `multi_agent_architecture.md` — 理解 Multi-Agent 设计
3. 阅读 `multi_agent_demo.py` — 理解 LangGraph Supervisor 实现
4. 运行 Demo — 观察 4 个 Agent 如何协作
5. 自定义任务 — 用自己的需求测试

## 前置知识

- `007-langgraph-learning` — LangGraph StateGraph、条件边、Supervisor 模式
- `009-agent-learning` — Agent 概念、ReAct、Reflection
- `008-mcp-learning` — Tool Calling 机制
