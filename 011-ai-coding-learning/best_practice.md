# AI Coding 最佳实践：从上下文工程到 Spec 驱动开发

> AI 不是工具替代人，而是人通过好的上下文设计让 AI 精准协作。

## 目录

1. [重新理解 AI Coding](#1-重新理解-ai-coding)
2. [CLAUDE.md：项目的 AI 入口文件](#2-claudemd项目的-ai-入口文件)
3. [AGENTS.md：跨平台 Agent 指令](#3-agentsmd跨平台-agent-指令)
4. [Memory：跨会话的知识持久化](#4-memory跨会话的知识持久化)
5. [Skills：可复用的 AI 能力模板](#5-skills可复用的-ai-能力模板)
6. [Hooks：自动化事件触发](#6-hooks自动化事件触发)
7. [Context Engineering：上下文工程](#7-context-engineering上下文工程)
8. [Spec Driven Development：规约驱动开发](#8-spec-driven-development规约驱动开发)
9. [完整 AI 开发工作流](#9-完整-ai-开发工作流)
10. [反模式与常见错误](#10-反模式与常见错误)

---

## 1. 重新理解 AI Coding

### AI Coding 的本质

```
传统开发：        人 → 思考 → 写代码 → 调试 → 提交
AI 辅助开发：     人 → 描述意图 → AI 生成代码 → 人审查 → 提交
AI Coding 最佳实践：人 → 设计上下文 → AI 在约束内工作 → 人验收 → 提交
```

区别不在于"用不用 AI"，而在于**你花了多少精力设计 AI 的工作环境**。

### 核心等式

```
AI 输出质量 = 上下文质量 × 模型能力 × 任务清晰度

上下文质量 = 项目规则 + 记忆 + 模板 + 约束
模型能力 = 模型本身（你控制不了）
任务清晰度 = 你的 prompt 和 spec 写得多好
```

模型能力你无法改变，但上下文质量和任务清晰度完全由你控制。

### AI Coding 的三个层次

```
Level 1: 问答式（Ask）
  "帮我写一个排序函数"
  → AI 生成代码，粘贴到项目中
  问题：AI 不了解项目上下文，代码风格不一致

Level 2: 上下文感知（Context-Aware）
  配置 CLAUDE.md + 打开相关文件
  "帮我写一个排序函数"
  → AI 了解项目规范，生成符合风格的代码
  优势：代码风格一致，减少修改

Level 3: 规约驱动（Spec-Driven）
  先写 PRD → Architecture → Tasks
  每个阶段评审后再进入下一阶段
  "基于 tasks/03.md 实现用户认证模块"
  → AI 在清晰的规约下精准执行
  优势：方向正确、质量可控、可追溯
```

### 这里的核心教训

> AI Coding 的瓶颈从来不是模型不够聪明，而是你给它的上下文不够好。

---

## 2. CLAUDE.md：项目的 AI 入口文件

### 什么是 CLAUDE.md

`CLAUDE.md` 是 Claude Code 在启动时自动加载的项目级指令文件。它告诉 Claude：

- 这是什么项目
- 遵循什么规范
- 使用什么技术栈
- 有哪些约定和禁忌

### 放在哪里

```
项目根目录/CLAUDE.md          ← 项目级（推荐，纳入版本控制）
~/.claude/CLAUDE.md           ← 用户级（个人偏好，全局生效）
```

项目级优先，可以覆盖用户级。

### 怎么写一个好 CLAUDE.md

```markdown
# 项目名称

## 目标
（一句话说清项目是干什么的）

## 技术栈
- 语言: Python 3.12+
- 框架: FastAPI
- 数据库: PostgreSQL
- 测试: pytest

## 规则
- 所有函数必须有类型提示
- API 端点必须用 Pydantic 做请求/响应校验
- 测试覆盖率不低于 80%
- 不要引入超过 3 个新依赖

## 约定
- 文件命名: snake_case
- 类命名: PascalCase
- API 路径: /api/v1/resource

## 禁忌
- 不要在 API 层直接操作数据库
- 不要跳过测试直接提交
- 不要在生产代码中使用 print()
```

### CLAUDE.md 设计原则

| 原则 | 说明 | 例子 |
|------|------|------|
| **具体而非抽象** | "使用 pytest" 而非 "写好测试" | 差: "代码要干净" / 好: "函数不超过 20 行" |
| **约束而非建议** | "必须" 而非 "建议" | 差: "最好有类型提示" / 好: "所有函数必须有类型提示" |
| **说做什么也说不做什么** | 禁忌往往比规则更重要 | "不要在 API 层操作数据库" |
| **保持更新** | 发现新规则立即更新 | 踩坑后用 CLAUDE.md 防止再犯 |
| **短小精悍** | 每行一条规则，别写散文 | 控制在 200 行以内 |

### 真实案例

```
❌ 差的 CLAUDE.md：
  "这是一个 API 项目，要写好代码，注意安全，用合适的设计模式"
  → 全是空话，AI 没有任何可执行的约束

✅ 好的 CLAUDE.md：
  "FastAPI + PostgreSQL 项目。
   规则：
   - 所有 API 端点返回 {data, error} 格式
   - 数据库查询用 async SQLAlchemy 2.0
   - 不写同步函数调用异步函数
   - 每个路由文件对应一个 service 文件
   禁忌：
   - 不要用 raw SQL，用 ORM
   - 不要在前端组件里调 API，用 service 层封装"
  → 每条都可执行、可验证
```

---

## 3. AGENTS.md：跨平台 Agent 指令

### CLAUDE.md vs AGENTS.md

```
CLAUDE.md → Claude Code 专属
AGENTS.md → 多平台兼容（Claude Code、Codex、Copilot、Cursor 等）
```

`AGENTS.md` 是更通用的 Agent 指令文件。如果你同时使用多个 AI 编程工具，`AGENTS.md` 是公共配置，`CLAUDE.md` 是 Claude Code 专属配置。

### 协同策略

```
AGENTS.md：放跨平台通用规则
  - 项目目标
  - 通用编码规范
  - 技术栈声明

CLAUDE.md：放 Claude Code 专属配置
  - Claude Code 的 Skills 引用
  - Claude Code 特有的 Hooks 配置
  - Claude Code 的 Memory 路径

如果只用 Claude Code → 只需 CLAUDE.md
如果用多工具 → AGENTS.md 放通用规则 + CLAUDE.md 放专属配置
```

### AGENTS.md 模板

```markdown
# AGENTS.md — 跨平台 AI Agent 指令

## 项目背景
（项目的业务目标和技术定位）

## 通用编码规则
- 语言: Python 3.12+
- 测试: pytest with coverage
- 类型提示: 必需

## 架构约束
- 分层: router → service → repository
- 不跨层调用

## 工作约定
- 提交前必须通过 lint
- PR 描述必须引用 spec 文档
```

---

## 4. Memory：跨会话的知识持久化

### AI 编程的记忆问题

```
问题：
  会话 1: "我们决定用 Redis 做缓存，不用 Memcached"
  （对话结束，上下文消失）

  会话 2: "帮我加个缓存层"
  → AI 不知道上次讨论了什么，可能推荐 Memcached

解决：Memory 系统
  会话 1: → 存入 memory/cache-decision.md
  会话 2: → AI 自动加载 memory/，知道用 Redis
```

### Memory 的五种类型

```
memory/
├── MEMORY.md              # 索引文件（AI 每次加载）
├── user_role.md           # 用户角色（你是什么角色、目标是什么）
├── project_context.md     # 项目上下文（为什么做、约束条件）
├── decisions.md           # 重要决策记录（选了 Redis 而非 Memcached）
├── patterns.md            # 项目模式（这个项目常用的设计模式）
└── pitfalls.md            # 踩坑记录（上次 X 导致了 Y，下次注意）
```

### Memory 设计原则

1. **MEMORY.md 是索引，不是内容**：每行一条记录，< 150 字符，AI 每次会话加载
2. **记录 WHY 而非 WHAT**：WHAT 在代码里，WHY 在 Memory 里
3. **决策记录最珍贵**："我们选 X 而不是 Y，因为 Z"——这无法从代码推导
4. **随时更新**：上下文切换、决策变更时立即更新
5. **不过期假设**：写 Memory 时假设 3 个月后的 AI 会读到，给足上下文

### 何时写 Memory

| 场景 | 记录内容 |
|------|----------|
| 技术选型 | 为什么选 A 不选 B，当时的约束是什么 |
| Bug 根因 | 不是修了什么，而是为什么会产生这个 Bug |
| 架构决策 | 为什么分层是这样的，哪些是刻意设计 |
| 踩坑经验 | "上次在 X 场景下用了 Y 方法导致 Z 问题" |
| 用户偏好 | 用户喜欢什么风格、有什么习惯 |

---

## 5. Skills：可复用的 AI 能力模板

### 什么是 Skill

Skill 是预定义的 AI 工作流模板。它告诉 AI "当遇到这类任务时，按这个步骤做"。

```
没有 Skill：
  用户: "review 我的代码"
  AI: "看起来不错"（泛泛而谈，没有标准）

有 Skill：
  用户: "review 我的代码"
  AI: 加载 code-review skill
      → 1. 检查安全漏洞
      → 2. 检查逻辑错误
      → 3. 检查性能问题
      → 4. 检查代码风格
      → 5. 输出结构化报告
```

### Skill 的结构

```
skills/
├── code-review.md       # 代码审查技能
├── prd-generator.md     # PRD 生成技能
├── test-writer.md       # 测试生成技能
└── deploy-check.md      # 部署检查技能
```

### Skill 示例：code-review

```markdown
---
name: code-review
description: 代码审查 —— 安全、逻辑、性能、风格四维度检查
---

## 审查流程

### 1. 安全审查（优先级最高）
- [ ] 是否有 SQL 注入风险？
- [ ] 是否有 XSS 风险？
- [ ] 敏感信息是否硬编码？
- [ ] 输入校验是否充分？

### 2. 逻辑审查
- [ ] 边界条件是否处理？
- [ ] 错误处理是否完善？
- [ ] 并发场景是否安全？

### 3. 性能审查
- [ ] 是否有 N+1 查询？
- [ ] 是否有不必要的循环？
- [ ] 大数据量场景是否考虑？

### 4. 风格审查
- [ ] 命名是否清晰？
- [ ] 函数是否过长（> 30 行）？
- [ ] 是否有无用的代码？

## 输出格式
生成 Review Report，按严重程度排序：
- 🔴 Critical（必须修）
- 🟡 Warning（建议修）
- 🔵 Info（参考）
```

### Skills 的最佳实践

1. **做成 Checklist 而非散文**：可勾选、可验证
2. **定义输出格式**：让 AI 输出结构化报告而非随意评价
3. **按优先级排序**：安全检查永远第一
4. **持续迭代**：每次 Review 漏掉的问题，加到 Skill 里

---

## 6. Hooks：自动化事件触发

### 什么是 Hook

Hook 是在特定事件发生时自动执行的脚本或指令。

```
事件类型：
  - PreToolUse: 工具调用前触发（如执行 bash 命令前检查）
  - PostToolUse: 工具调用后触发（如写文件后自动 lint）
  - Notification: 通知事件
  - SessionStart: 会话开始时触发
  - SessionEnd: 会话结束时触发
  - PreCompaction: 上下文压缩前触发
```

### Hooks 配置示例

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "python -m ruff check \"$CLAUDE_TOOL_FILE_PATH\" --fix"
        }]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "if echo \"$CLAUDE_TOOL_COMMAND\" | grep -qE 'rm -rf|git push --force'; then echo 'DESTRUCTIVE COMMAND BLOCKED'; exit 1; fi"
        }]
      }
    ],
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "echo 'Session started at $(date)' >> .claude/session.log"
        }]
      }
    ]
  }
}
```

### Hook 的设计原则

| 原则 | 说明 |
|------|------|
| **自动化检查而非阻止** | lint 修代码 ok，阻止所有 git push 不 ok |
| **无副作用** | Hook 不应修改业务逻辑，只做质量检查 |
| **快速执行** | Hook 运行时间应 < 5 秒，否则影响体验 |
| **失败可见** | Hook 失败要有清晰的输出，告诉用户发生了什么 |

### 常用 Hook 场景

| 场景 | 事件 | 做什么 |
|------|------|--------|
| 代码格式化 | PostToolUse (Write/Edit) | 自动 ruff/black 格式化 |
| 安全检查 | PreToolUse (Bash) | 检测危险命令 (`rm -rf`, `git push --force`) |
| 测试提醒 | PostToolUse (Write/Edit) | 修改核心模块后提醒运行测试 |
| 日志记录 | SessionStart/End | 记录工作开始/结束时间 |
| 上下文压缩 | PreCompaction | 自动将关键信息写入 Memory |

---

## 7. Context Engineering：上下文工程

### 什么是上下文工程

> Context Engineering = 主动设计 AI 在工作时会看到什么信息

AI 的上下文窗口是"工作记忆"——你放进来的东西决定了 AI 的输出质量。

### 上下文设计矩阵

| 层次 | 内容 | 加载时机 | 目的 |
|------|------|----------|------|
| **系统层** | 模型能力、工具集 | 始终 | 定义能力边界 |
| **项目层** | CLAUDE.md、AGENTS.md | 会话开始 | 定义项目规则 |
| **记忆层** | MEMORY.md、决策记录 | 会话开始 | 提供历史上下文 |
| **任务层** | Spec 文档、Task 文件 | 任务切换时 | 定义当前任务 |
| **即时层** | 打开的文件、当前对话 | 实时 | 提供代码上下文 |

### 上下文窗口管理

```
上下文窗口 = 有限资源（如 200K tokens）

策略：
1. 精确导入：只加载与当前任务相关的文件
   而不是 "把所有文件都打开"
   CLAUDE.md 写好 → AI 知道去哪里找需要的信息

2. 分层加载：
   一级（始终）: CLAUDE.md、MEMORY.md
   二级（按需）: Spec 文档、架构文档
   三级（实时）: 当前编辑的文件、相关文件

3. 压缩策略：
   长对话中，定期摘要关键决策写入 Memory
   避免 "对话太长，AI 忘了开头说了什么"
```

### 上下文污染的常见形式

```
❌ 污染 1: 打开不相关的文件
  "帮我改 auth.py" → 同时打开了 15 个无关文件
  → AI 注意力被分散

❌ 污染 2: 过长的 CLAUDE.md
  CLAUDE.md 写了 500 行，包含代码示例
  → 规则淹没在细节中

❌ 污染 3: 过期的 Memory
  Memory 说 "用 MySQL 8.0"，但项目已经迁移到 PostgreSQL
  → AI 基于过期信息做决策

✅ 好的上下文：
  - 只打开与当前任务相关的 2-3 个文件
  - CLAUDE.md 控制在 100 行以内
  - Memory 定期清理过期记录
```

---

## 8. Spec Driven Development：规约驱动开发

### 传统 AI 开发的痛点

```
用户: "帮我做用户认证系统"
AI:   直接写代码（跳过需求分析、跳过架构设计）
      → 写出来的东西可能不是用户想要的
      → 没有设计文档，后续修改困难
      → 质量不可追溯

问题根源：让 AI 直接从模糊需求跳到代码实现
```

### Spec Driven 的核心思想

```
传统 AI 开发:
  需求 → AI 写代码 → 审查 → 修改 → 审查 → 修改 → ...

Spec Driven:
  需求 → PRD（确认）→ Architecture（确认）→ Tasks（确认）→ 逐 Task 实现 → Review
       ↑               ↑                    ↑
     每个阶段评审通过后才进入下一阶段
```

**关键差异**：在写代码之前，先把"要做什么"和"怎么做"想清楚并写成文档。

### Spec 的三层结构

```
specs/
├── 01-prd.md           # Product Requirements Doc: 要做什么
├── 02-architecture.md   # Architecture Design: 怎么做（宏观）
└── 03-tasks.md          # Task Breakdown: 分步实现计划
```

### 每层的目的

```
PRD（Product Requirements Doc）：
  回答: 用户要什么？验收标准是什么？
  产出: 用户故事、功能列表、验收条件
  评审人: 产品 / 业务方

Architecture（架构设计）：
  回答: 用什么技术？模块怎么划分？数据怎么流？
  产出: 技术选型、组件图、数据模型、API 契约
  评审人: Tech Lead / 架构师

Tasks（任务拆解）：
  回答: 分几步实现？每步做什么？依赖关系？
  产出: 任务列表（P0/P1/P2）、依赖关系、估时
  评审人: 开发者自己
```

### Spec Driven 的收益

| 维度 | 不用 Spec | 用 Spec |
|------|-----------|---------|
| **方向正确性** | 写着写着可能偏了 | 每个 Task 对照 Spec 检查 |
| **返工率** | 高（方向错了重写） | 低（方向在 PRD 阶段就确认了） |
| **可接手性** | 换人/新会话无法继续 | AI 读 Spec 就能继续 |
| **质量可追溯** | "我觉得做得对" | 对照验收标准逐条验证 |
| **AI 输出质量** | 不稳定 | 稳定（约束越多输出越准） |

---

## 9. 完整 AI 开发工作流

### 工作流全景

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI 开发全流程                                 │
│                                                                  │
│  [1] PRD        [2] Arch       [3] Tasks      [4] Coding        │
│  需求分析 ───→ 架构设计 ───→ 任务拆解 ───→ 逐 Task 实现         │
│      │              │              │              │               │
│      ▼              ▼              ▼              ▼               │
│  01-prd.md    02-arch.md    03-tasks.md    src/ 目录             │
│      │              │              │              │               │
│      └──────────────┴──────┬───────┘              │               │
│                            │                      │               │
│                    每个阶段 Review                │               │
│                    通过后才进入下一阶段             │               │
│                                                   ▼               │
│  [5] Review     [6] Test       [7] Deploy        [8] Retro       │
│  代码审查 ───→ 测试验证 ───→ 部署上线 ───→ 回顾优化              │
│      │              │              │              │               │
│      ▼              ▼              ▼              ▼               │
│  review.md     test-report    deploy-log    memory/更新           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 各阶段详解

#### [1] PRD — 产品需求文档

```
做什么：把模糊需求变成结构化的功能描述
谁参与：人（提出需求）+ AI（帮助结构化）
产出：specs/01-prd.md

内容：
  - 项目背景和目标
  - 用户故事
  - 功能列表（P0/P1/P2 优先级）
  - 验收标准（每条功能怎么才算完成）
  - 非功能需求（性能、安全、兼容性）

关键问题：
  "如果 PRD 写好了，另一个人（或另一个 AI）能基于它独立完成吗？"
  如果不能 → PRD 不够清晰
```

#### [2] Architecture — 架构设计

```
做什么：从 PRD 推导技术方案
谁参与：人（最终决策）+ AI（提出方案和 tradeoff 分析）
产出：specs/02-architecture.md

内容：
  - 技术选型及理由
  - 系统组件图
  - 数据模型设计
  - API 接口契约
  - 关键流程图
  - 安全设计
  - 已知风险和替代方案

关键问题：
  "如果换一个开发者，他能基于这个架构写出代码吗？"
```

#### [3] Tasks — 任务拆解

```
做什么：将架构设计拆解为小粒度的实现任务
谁参与：AI（拆解）+ 人（确认优先级和依赖关系）
产出：specs/03-tasks.md

拆分原则：
  - 每个 Task 独立可验证（有自己的验收条件）
  - 每个 Task 30 分钟到 2 小时
  - P0 优先，P1 次之，P2 可选
  - 明确依赖关系（Task B 依赖 Task A）

Task 模板：
  ### Task 1.1: 用户数据模型
  - 优先级: P0
  - 依赖: 无
  - 预估: 30 min
  - 描述: 创建 User model，字段包括...
  - 验收: pytest 通过，包含 3 个测试用例
```

#### [4] Coding — 逐 Task 实现

```
做什么：按 Task 顺序，逐个让 AI 实现
原则：一次只做一个 Task

操作步骤：
  1. 打开 specs/03-tasks.md
  2. 选择当前 Task
  3. 打开相关的架构文档和已有代码
  4. 告诉 AI："基于 specs/02-architecture.md 中的设计，
     实现 tasks/03-tasks.md 中的 Task 1.1"
  5. 确认代码符合预期
  6. 运行测试
  7. 提交
  8. 下一个 Task

为什么一次只做一个 Task：
  - AI 在狭窄的上下文中工作质量最高
  - 出问题容易定位
  - 每个 Task 可以独立验证
```

#### [5] Review — 代码审查

```
做什么：AI 先审查（快速），人再确认（最终）
产出：specs/04-review.md

流程：
  1. AI 对照 PRD 需求 → 检查功能覆盖
  2. AI 对照 Architecture → 检查架构一致性
  3. AI 运行 review Skill → 安全、逻辑、性能
  4. 人确认 AI 的审查结果
  5. 记录问题和修改

三层审查：
  第一层: AI 自动审查（快，覆盖 80% 问题）
  第二层: 人检查设计层面（架构一致性、需求覆盖）
  第三层: 运行测试（自动验证功能正确性）
```

#### [6] Test — 测试验证

```
做什么：多层测试确保质量
产出：specs/05-test-report.md

测试层次：
  - 单元测试：每个函数/方法的行为正确
  - 集成测试：模块间协作正确
  - API 测试：端点行为符合契约
  - 验收测试：对照 PRD 验收标准逐条验证

AI 的测试角色：
  - 根据 Task 描述生成测试用例
  - 根据已有代码生成测试
  - 根据 Bug 修复生成回归测试
```

#### [7] Deploy — 部署上线

```
做什么：安全地将代码部署到目标环境
产出：specs/06-deploy.md

检查清单：
  - [ ] 所有测试通过
  - [ ] Review 通过
  - [ ] 数据库迁移已准备
  - [ ] 环境变量已配置
  - [ ] 回滚方案已准备
  - [ ] 监控和告警已配置
```

#### [8] Retro — 回顾优化

```
做什么：总结本次开发的经验，更新 Memory 和 CLAUDE.md
产出：memory/ 更新

回顾问题：
  - AI 在哪一步反复出错？（→ 更新 CLAUDE.md 加约束）
  - 哪个 Task 拆分太粗/太细？（→ 调整 Task 模板）
  - 架构设计遗漏了什么？（→ 更新 Architecture 模板）
  - 有哪些可复用的 Skill？（→ 创建新 Skill）
```

---

## 10. 反模式与常见错误

### 十大反模式

| # | 反模式 | 后果 | 正确做法 |
|---|--------|------|----------|
| 1 | **没写 CLAUDE.md 就开始** | AI 不了解项目规范 | 先写好 CLAUDE.md |
| 2 | **直接让 AI 写代码，跳过 PRD** | 方向容易偏，返工率高 | PRD → Arch → Tasks → Code |
| 3 | **一次给 AI 太多任务** | 输出质量断崖式下降 | 一次一个 Task |
| 4 | **打开不相关的文件** | 上下文污染 | 只打开当前任务需要的文件 |
| 5 | **不用 Memory 记录决策** | 下次会话重复讨论 | 决策后立即写入 Memory |
| 6 | **Review 只看代码不看逻辑** | 漏掉设计问题 | 对照 PRD 和 Arch 审查 |
| 7 | **不写验收标准** | 不知道什么时候算完成 | PRD 就写好验收条件 |
| 8 | **忽视测试** | 回归 Bug 频发 | AI 生成实现后立即生成测试 |
| 9 | **过度依赖 AI** | 不理解的代码也合入 | 每行代码都要理解 |
| 10 | **CLAUDE.md 从不更新** | 规则过时，AI 行为退化 | 每次踩坑后更新 |

### 一句话总结

> **AI Coding 的质量，80% 取决于你给它设计的工作环境，20% 取决于你的 Prompt。模型反而是最小的变量。**

---

## 延伸阅读

- [011-ai-coding-learning/project_template/](./project_template/) — 完整项目模板（开箱即用）
- Claude Code 官方文档
- [Anthropic Context Engineering Guide](https://docs.anthropic.com/en/docs/agents-and-tools)
