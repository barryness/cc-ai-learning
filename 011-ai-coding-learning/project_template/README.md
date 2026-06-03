# AI Coding 项目模板

> 开箱即用的 AI 协作开发项目模板。复制此目录到你的新项目即可开始。

## 快速开始

```bash
cp -r project_template/ my-new-project/
cd my-new-project

# 1. 编辑 CLAUDE.md — 填入项目信息
# 2. 编辑 memory/user_role.md — 填入你的信息
# 3. 开始第一个 Spec
#   "基于 project_template/specs/01-prd.md 模板，帮我写 XX 项目的 PRD"
```

## 目录结构

```
project_template/
├── CLAUDE.md                  # Claude Code 项目指令
├── AGENTS.md                  # 跨平台 Agent 指令
├── .claude/
│   └── settings.json          # Hooks 配置 + MCP Server 配置
├── memory/                    # AI 记忆系统
│   ├── MEMORY.md              # 记忆索引（AI 每次会话加载）
│   ├── user_role.md           # 用户角色
│   ├── project_context.md     # 项目上下文
│   ├── decisions.md           # 技术决策记录
│   ├── architecture_decisions.md  # 架构决策记录
│   └── feedback.md            # 协作反馈
├── specs/                     # Spec 文档
│   ├── 01-prd.md              # 产品需求文档
│   ├── 02-architecture.md     # 架构设计文档
│   ├── 03-tasks.md            # 任务拆解
│   ├── 04-code-review.md      # 代码审查清单
│   ├── 05-test-plan.md        # 测试计划
│   └── 06-deploy.md           # 部署检查清单
└── skills/                    # AI Skills
    ├── code-review.md         # 代码审查技能
    ├── test-writer.md         # 测试生成技能
    └── prd-generator.md       # PRD 生成技能
```

## 使用流程

```
1. 初始化模板 → 编辑 CLAUDE.md + memory/
2. PRD        → 让 AI 基于 01-prd.md 模板生成
3. Architecture → 让 AI 基于 02-architecture.md 模板生成
4. Tasks      → 让 AI 拆解为实现任务
5. Coding     → 逐 Task 让 AI 实现
6. Review     → 用 code-review Skill 审查
7. Test       → 用 test-writer Skill 生成测试
8. Deploy     → 按 06-deploy.md 检查清单上线
9. Retro      → 更新 memory/ 记录经验
```

## 配合 best_practice.md 阅读

模板中的每个文件在 `../best_practice.md` 中都有详细说明。
