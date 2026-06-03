# AI Coding 最佳实践

> 从上下文工程到 Spec 驱动开发 —— 让 AI 成为可靠的开发伙伴。

## 文件结构

```
011-ai-coding-learning/
├── best_practice.md             ← 完整教学指南（10 章）
├── project_template/            ← 开箱即用的项目模板
│   ├── CLAUDE.md                ← 项目 AI 指令模板
│   ├── AGENTS.md                ← 跨平台 Agent 指令模板
│   ├── .claude/settings.json    ← Hooks 配置示例
│   ├── memory/                  ← 记忆系统模板
│   │   ├── MEMORY.md            ← 记忆索引
│   │   ├── user_role.md         ← 用户角色
│   │   ├── project_context.md   ← 项目上下文
│   │   ├── decisions.md         ← 技术决策记录
│   │   ├── architecture_decisions.md ← 架构决策
│   │   └── feedback.md          ← 协作反馈
│   ├── specs/                   ← Spec 文档模板
│   │   ├── 01-prd.md            ← PRD 模板
│   │   ├── 02-architecture.md   ← 架构设计模板
│   │   ├── 03-tasks.md          ← 任务拆解模板
│   │   ├── 04-code-review.md    ← 代码审查清单
│   │   ├── 05-test-plan.md      ← 测试计划模板
│   │   └── 06-deploy.md         ← 部署检查清单
│   └── skills/                  ← AI Skills 模板
│       ├── code-review.md       ← 代码审查 Skill
│       ├── test-writer.md       ← 测试生成 Skill
│       └── prd-generator.md     ← PRD 生成 Skill
├── index.html                   ← 交互学习网页
└── README.md                    ← 你在这里
```

## 快速开始

```bash
# 1. 阅读教学指南
open best_practice.md

# 2. 在新项目中使用模板
cp -r project_template/ ~/my-new-project/
cd ~/my-new-project/
# 编辑 CLAUDE.md 和 memory/ 文件

# 3. 开始 Spec Driven 开发
# 让 AI 基于 specs/01-prd.md 模板生成你的 PRD
```

## 学习路线

1. 阅读 `best_practice.md` 第 1-2 章 — 理解 AI Coding 的本质和 CLAUDE.md
2. 阅读 3-4 章 — AGENTS.md 和 Memory 系统
3. 阅读 5-6 章 — Skills 和 Hooks
4. 阅读 7-8 章 — Context Engineering 和 Spec Driven Development
5. 阅读 9 章 — 完整 AI 开发工作流
6. 浏览 `project_template/` — 理解每个模板的用途
7. 在自己的项目中实践

## 核心原则

```
AI 输出质量 = 上下文质量 × 任务清晰度
```

- **上下文质量**：CLAUDE.md + Memory + Skills + 正确的文件
- **任务清晰度**：PRD → Architecture → Tasks（Spec Driven）

## 前置知识

- 实际使用过 Claude Code 或其他 AI 编程工具
- 对软件开发流程（需求→设计→实现→测试→部署）有基本理解
