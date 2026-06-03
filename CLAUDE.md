# AI Learning Workspace

> Claude Code 项目指令。跨平台指令见 AGENTS.md。

## 项目定位

通过渐进式 Demo 系统学习 LLM 生态的 11 个核心主题。从底层概念到架构设计，每个模块独立可运行。

## 学习路线

```
001-llm-learning           # LLM 基础：API 调用、Token、Temperature
002-prompt-learning        # Prompt 工程：设计原则、模板、技巧
003-embedding-learning     # Embedding：文本向量化、语义相似度
004-vector-db-learning     # Vector DB：ChromaDB 入门、向量检索
005-rag-learning           # RAG：5 步渐进式检索增强生成
006-langchain-learning     # LangChain：7 大核心概念 + RAG 重构
007-langgraph-learning     # LangGraph：StateGraph → Supervisor 多 Agent
008-mcp-learning           # MCP：Tool Calling 标准化协议 + Server 开发
009-agent-learning         # Agent 设计：ReAct → Memory → Reflection
010-multi-agent-learning   # Multi-Agent：PM+Architect+Coder+Reviewer 协作
011-ai-coding-learning     # AI Coding：上下文工程 + Spec Driven + 项目模板
```

## 规则

- 所有概念必须配 Demo，所有 Demo 必须可运行（AST 语法验证）
- 代码注释用中文，标识符用英文
- 每个模块包含 README.md + 教学指南 + Demo 代码 + index.html
- 优先使用 Python 标准库，减少外部依赖
- 新模块目录命名：`0XX-topic-learning/`
- 每次新模块创建后必须更新 `docs/learning-log.md`

## 模块结构规范

```
0XX-topic-learning/
├── README.md            # 模块概述、快速开始、学习路线
├── xxx_guide.md         # 完整教学指南（或 xxx_notes/design/architecture.md）
├── xxx_demo.py          # 渐进式 Demo（Phase 1 → 2 → 3）
├── index.html           # 交互学习网页（深色主题，多 Tab）
└── requirements.txt     # Python 依赖（如需要）
```

## 教学指南写作规范

- 从问题出发：Why（为什么需要） → What（是什么） → How（怎么做）
- 使用 ASCII 架构图、对比表格、代码示例
- 每章结尾有"延伸阅读"链接到相关模块
- 最终章为"陷阱与最佳实践"或"生产级挑战"

## Index.html 风格规范

- 深色主题（--bg: #0f1117），每个模块用不同的 accent 色
- 多个 Tab 切换（position: sticky 顶部固定）
- card 组件承载内容块，grid-2/grid-3 做响应式布局
- code/pre 使用深色代码块，tag 做标签分类

## Demo 代码规范

- 文件头用 `"""多行注释"""` 说明架构和运行方式
- 每个阶段用独立的 build/run 函数
- 支持 `--phase N` 或 `--demo N` 选择阶段
- 模拟外部依赖（不阻塞用户运行）

## Git 约定

- Commit 消息格式：`type: 中文描述`
- Type: feat / docs / refactor / fix
- 示例：`feat: 新增 010-multi-agent-learning 模块`
- 不提交 .env 文件，提供 .env.example 模板

## Memory

项目记忆存储在 `docs/learning-log.md`，每次新模块需同步更新。
