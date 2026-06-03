# AGENTS.md — 跨平台 AI Agent 指令

> 兼容 Claude Code、Codex、Copilot、Cursor 等 AI 编程工具。
> Claude Code 专属配置见 CLAUDE.md。

## 项目定位

AI 学习工作坊 —— 通过渐进式 Demo 学习 LLM 生态的 11 个核心主题。

## 学习路径

```
001-llm-learning → 002-prompt-learning → 003-embedding-learning
→ 004-vector-db-learning → 005-rag-learning → 006-langchain-learning
→ 007-langgraph-learning → 008-mcp-learning → 009-agent-learning
→ 010-multi-agent-learning → 011-ai-coding-learning
```

## 通用编码规则

- 语言: Python 3.10+（标准库优先，减少依赖）
- 每个模块必须包含: README.md + 教学指南 + Demo 代码
- Demo 必须可运行（AST 语法验证通过）
- 代码注释用中文，标识符用英文
- 所有文件使用 UTF-8 编码

## 模块规范

每个模块遵循统一结构：
```
0XX-topic-learning/
├── README.md          # 模块概述、快速开始、学习路线
├── xxx_guide.md       # 完整教学指南（或 xxx_notes.md / xxx_design.md）
├── xxx_demo.py        # 可运行的渐进式 Demo
├── index.html         # 交互学习网页（深色主题）
└── requirements.txt   # 依赖（如需要）
```

## 文档风格

- 教学指南: 从问题出发（Why → What → How）
- Demo 代码: 渐进式（Phase 1 → Phase 2 → Phase 3）
- 网页: 多 Tab 交互式，深色主题，mobile responsive
- 学习日志: docs/learning-log.md 保持同步更新

## 约束

- 不在模块目录外创建教学文件
- 不引入新的依赖目录结构
- Git 提交遵循现有模块的 commit 风格
- 每次创建新模块后同步更新 docs/learning-log.md
