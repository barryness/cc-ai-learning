# AI Learning Workspace

> 从 LLM 基础到 Multi-Agent 协作 —— 系统学习 AI 工程的 11 个核心主题。
> 每个概念配可运行 Demo + 教学指南 + 交互学习网页。

## 学习路径

```
001-llm → 002-prompt → 003-embedding → 004-vector-db → 005-rag
→ 006-langchain → 007-langgraph → 008-mcp
→ 009-agent → 010-multi-agent → 011-ai-coding
```

### 基础阶段

| # | 模块 | 核心内容 |
|---|------|----------|
| 001 | LLM 基础 | API 调用、Token、Temperature |
| 002 | Prompt 工程 | 设计原则、Few-shot、CoT |
| 003 | Embedding | 文本向量化、语义相似度 |
| 004 | Vector DB | ChromaDB、向量存储与检索 |

### 进阶阶段

| # | 模块 | 核心内容 |
|---|------|----------|
| 005 | RAG | 5 步渐进式检索增强生成 |
| 006 | LangChain | 7 大核心概念 + RAG 重构 |
| 007 | LangGraph | StateGraph → Supervisor 多 Agent |
| 008 | MCP | Tool Calling 标准化 + Server 开发 |

### 高级阶段

| # | 模块 | 核心内容 |
|---|------|----------|
| 009 | Agent 设计 | ReAct → Memory → Reflection |
| 010 | Multi-Agent | PM + Architect + Coder + Reviewer |
| 011 | AI Coding | CLAUDE.md / Context Engineering / Spec Driven |

## 项目结构

```
ai-learning/
├── index.html                  # 主页面（学习路径 + 模块导航）
├── CLAUDE.md                   # Claude Code 项目指令
├── AGENTS.md                   # 跨平台 Agent 指令
├── docs/learning-log.md        # 学习记录
│
├── 001-llm-learning/           # LLM 基础
├── 002-prompt-learning/        # Prompt 工程
├── 003-embedding-learning/     # Embedding
├── 004-vector-db-learning/     # Vector DB
├── 005-rag-learning/           # RAG
├── 006-langchain-learning/     # LangChain
├── 007-langgraph-learning/     # LangGraph
├── 008-mcp-learning/           # MCP
├── 009-agent-learning/         # Agent 设计
├── 010-multi-agent-learning/   # Multi-Agent
└── 011-ai-coding-learning/     # AI Coding
```

## 原则

- 所有概念必须配可运行 Demo（AST 语法验证通过）
- 从问题出发：Why → What → How
- 渐进式深入：Phase 1 → Phase 2 → Phase 3
- 每个模块包含 README + 教学指南 + Demo + 交互网页
- Python 标准库优先，减少外部依赖
- 代码注释用中文，标识符用英文

## 快速开始

```bash
# 浏览主页面
open index.html

# 进入任意模块学习
cd 005-rag-learning
python demo/minimal_rag.py
```

## 环境

- Python 3.10+
- macOS / Linux
- DeepSeek / OpenAI API（部分 Demo 需要）
