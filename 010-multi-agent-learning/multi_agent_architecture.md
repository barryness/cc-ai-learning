# Multi-Agent 架构：从单 Agent 到多 Agent 协作

> 一个 Agent 做一件事，多个 Agent 分工协作完成复杂项目。

## 目录

1. [为什么需要 Multi-Agent](#1-为什么需要-multi-agent)
2. [Agent 职责划分](#2-agent-职责划分)
3. [Agent 通信机制](#3-agent-通信机制)
4. [状态共享设计](#4-状态共享设计)
5. [冲突解决策略](#5-冲突解决策略)
6. [Supervisor 模式详解](#6-supervisor-模式详解)
7. [PM + Architect + Coder + Reviewer 实战](#7-pm--architect--coder--reviewer-实战)
8. [Multi-Agent 设计决策](#8-multi-agent-设计决策)
9. [常见陷阱与最佳实践](#9-常见陷阱与最佳实践)

---

## 1. 为什么需要 Multi-Agent

### 单 Agent 的瓶颈

单 Agent 在处理复杂任务时会遇到三个天花板：

```
单 Agent 天花板：
┌─────────────────────────────────────────────────────┐
│  "帮我设计并实现一个 REST API 微服务"                  │
│                                                      │
│  一个 Agent 需要同时：                                │
│  ✗ 理解业务需求（PM 视角）                            │
│  ✗ 设计系统架构（架构师视角）                          │
│  ✗ 编写代码实现（开发视角）                            │
│  ✗ 检查代码质量（Review 视角）                        │
│                                                      │
│  结果：prompt 过长、上下文污染、逻辑混乱               │
└─────────────────────────────────────────────────────┘
```

### Multi-Agent 的分工优势

```
Multi-Agent 协作：
┌─────────────────────────────────────────────────────┐
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐│
│  │ PM Agent │→│Architect │→│  Coder   │→│Review││
│  │ 拆解需求  │  │ Agent   │  │ Agent   │  │ Agent││
│  │ 排优先级  │  │ 设计架构  │  │ 写代码   │  │ 审查  ││
│  └──────────┘  └──────────┘  └──────────┘  └──┬───┘│
│                           ↑                    │    │
│                           └── 打回修改 ←────────┘    │
│                                                      │
│  每个 Agent 只关注自己的领域，通过 Supervisor 协调     │
└─────────────────────────────────────────────────────┘
```

### 关键洞察

| 维度 | 单 Agent | Multi-Agent |
|------|----------|-------------|
| Prompt 长度 | 巨大（所有能力混在一起） | 短（每个 Agent 独立 System Prompt） |
| 上下文质量 | 信息过载、注意力分散 | 每个 Agent 只看相关信息 |
| 可维护性 | 改一处影响全局 | 改一个 Agent 不影响其他 |
| 可扩展性 | 加能力 = 加 prompt（边际递减） | 加能力 = 加 Agent（线性扩展） |
| 执行质量 | 一个 LLM 视角 | 多个 LLM 互审，质量更高 |

---

## 2. Agent 职责划分

### 核心原则：单一职责

> 每个 Agent 只做一件事，且做到极致。

### PM Agent

```
角色定位：项目经理 —— 理解需求、拆解任务、排优先级

输入：用户的自然语言需求
输出：结构化的任务列表 + 验收标准

核心能力：
  - 需求分析：理解用户真正想要什么
  - 任务分解：将大需求拆成可执行的小任务
  - 优先级排序：决定哪些先做、哪些后做
  - 范围控制：识别 MVP，避免过度设计

System Prompt 关键指令：
  "你是项目经理。你的职责是理解需求并拆解为可执行任务。
   不要设计架构，不要写代码。只产出任务列表。"
```

### Architect Agent

```
角色定位：架构师 —— 设计系统架构、数据模型、接口契约

输入：PM 的任务列表
输出：架构文档（组件图、数据模型、API 定义）

核心能力：
  - 架构设计：选择合适的技术方案和组件划分
  - 数据建模：定义核心数据结构
  - 接口设计：定义组件间的通信契约
  - 技术选型：选择合适的框架和工具

System Prompt 关键指令：
  "你是架构师。你的职责是根据需求设计系统架构。
   不要写具体代码实现。只产出架构设计文档。"
```

### Coder Agent

```
角色定位：开发者 —— 根据架构设计写出可运行的代码

输入：架构文档 + 任务列表
输出：可运行的代码 + 简要说明

核心能力：
  - 代码生成：根据架构设计写出实现
  - 细节决策：在架构约束下做出具体实现选择
  - 可运行性：确保代码可以直接执行
  - 异常处理：处理边界情况

System Prompt 关键指令：
  "你是开发者。根据架构设计写出具体代码实现。
   代码必须可运行、清晰、有必要的注释。
   不要修改架构设计，有疑问标注在代码注释中。"
```

### Reviewer Agent

```
角色定位：审查者 —— 检查代码质量、发现 Bug、验证需求覆盖

输入：需求 + 架构 + 代码
输出：审查意见（通过 / 修改建议）

核心能力：
  - 需求覆盖检查：代码是否实现了所有需求
  - 架构一致性：代码是否遵循了架构设计
  - 代码质量：命名、结构、边界处理
  - Bug 检测：逻辑错误、潜在问题

System Prompt 关键指令：
  "你是代码审查者。严格检查代码是否：
   1. 覆盖了所有需求
   2. 遵循了架构设计
   3. 代码清晰可维护
   给出通过或具体修改建议。不要自己写代码。"
```

### 职责边界表

| 维度 | PM | Architect | Coder | Reviewer |
|------|-----|-----------|-------|----------|
| 需求分析 | ✅ | - | - | ✅（验证） |
| 任务拆解 | ✅ | - | - | - |
| 架构设计 | - | ✅ | - | - |
| 代码实现 | - | - | ✅ | - |
| 质量检查 | - | - | - | ✅ |
| 修改建议 | - | 被 Review 后可改 | 被 Review 后可改 | ✅ |
| 最终决策 | ✅（验收） | - | - | ✅（技术质量） |

---

## 3. Agent 通信机制

### 三种通信模式

```
模式 1: 顺序流水线
  PM → Architect → Coder → Reviewer
  适合：需求明确、阶段分明的项目

模式 2: Supervisor 路由（本项目采用）
         ┌──→ PM
  User →│──→ Architect
        │──→ Coder
         └──→ Reviewer
  Supervisor 根据状态动态决定下一步派谁
  适合：需要灵活调度和循环反馈的场景

模式 3: 事件驱动
  Agent A 发布 → 消息总线 → Agent B/C/D 订阅
  适合：松耦合、异步协作的场景
```

### 本项目通信设计

```
通信媒介：State（共享状态对象）

所有 Agent 的通信都通过 State 进行：
  - Agent 从 State 读取上游产出
  - Agent 将结果写入 State
  - Supervisor 读取 State 决定路由

消息流：
  User Input
    → State.task
    → PM 读取 task → 写入 State.tasks
    → Architect 读取 tasks → 写入 State.architecture
    → Coder 读取 architecture → 写入 State.code
    → Reviewer 读取 task+architecture+code → 写入 State.review
    → 通过 → 输出；不通过 → 回到 Coder（最多 3 轮）
```

### 为什么用 State 而不是直接消息？

| 方式 | 优点 | 缺点 |
|------|------|------|
| **直接消息推送** | 实时、低延迟 | Agent 耦合、难以追溯 |
| **共享 State（本项目）** | 可追溯、解耦、方便调试 | 需要 State 设计合理 |
| **消息队列** | 高吞吐、异步 | 架构复杂、过度工程 |

---

## 4. 状态共享设计

### State 是 Multi-Agent 的核心

State 是"项目交接单"——每个 Agent 完成自己的工作后在 State 上签字，下一个 Agent 根据 State 的当前内容继续工作。

### 本项目 State 设计

```python
class TeamState(TypedDict):
    # ── 消息流（add_messages reducer 自动合并）──
    messages: Annotated[list, add_messages]

    # ── Supervisor 路由控制 ──
    next_agent: str              # Supervisor 决策结果

    # ── PM 产出 ──
    requirement: str             # 用户原始需求
    task_breakdown: str          # 任务拆解结果

    # ── Architect 产出 ──
    architecture: str            # 架构设计文档

    # ── Coder 产出 ──
    code: str                    # 生成的代码

    # ── Reviewer 产出 ──
    review_result: str           # "approved" / "revise"
    review_feedback: str         # 审查意见

    # ── 循环控制 ──
    revision_count: int          # 修改次数（防无限循环）
```

### State 设计原则

1. **按角色分区**：每个 Agent 的产出有独立字段，互不覆盖
2. **渐进填充**：字段随流程逐步填充，空字段表示该阶段未执行
3. **最小状态**：只存 Agent 间需要传递的信息，不存中间推理
4. **可回溯**：通过 messages 字段保留完整对话历史

### 一个完整的 State 演进

```
初始 State:
  { requirement: "用户管理系统 REST API", task_breakdown: "", architecture: "", code: "", ... }

PM 完成后:
  { ..., task_breakdown: "1. 用户CRUD API\n2. JWT认证\n3. 数据校验\n...", ... }

Architect 完成后:
  { ..., architecture: "## 架构设计\n- FastAPI框架\n- SQLite存储\n- 分层: router/service/model", ... }

Coder 完成后:
  { ..., code: "from fastapi import FastAPI\n...", ... }

Reviewer 完成后:
  { ..., review_result: "approved", review_feedback: "代码质量良好，覆盖所有需求", ... }
```

---

## 5. 冲突解决策略

### Multi-Agent 中的典型冲突

```
冲突场景 1: Reviewer 打回 Coder 的代码
  → 解决: Reviser 模式 —— Coder 按反馈修改，最多 3 轮

冲突场景 2: Architect 设计超出 PM 的范围
  → 解决: Reviewer 用 PM 的 task_breakdown 校验架构覆盖率

冲突场景 3: Coder 认为架构有问题
  → 解决: Coder 在代码注释中标注疑问，Reviewer 判断是否需要回到 Architect

冲突场景 4: Reviewer 和 Coder 对"好的代码"有分歧
  → 解决: 第三次修改后自动通过（Stop-the-World 机制）
```

### 三层冲突解决机制

```
第一层：角色边界预防
  System Prompt 明确职责边界
  PM 不管架构，Coder 不改架构
  → 90% 的冲突在源头消除

第二层：Reviewer 仲裁
  统一的质量判断标准
  检查维度：需求覆盖、架构一致、代码质量
  → 9% 的冲突由 Reviewer 裁决

第三层：Stop-the-World
  revision_count >= 3 → 强制通过
  Supervisor 最终决策 → FINISH
  → 1% 的死锁自动解除
```

### 冲突解决的 System Prompt 设计

```
PM Agent:
  "你是项目经理。只负责拆解需求。不要涉及技术选型。"

Architect Agent:
  "严格基于 PM 的任务列表进行设计。如果有任何需求模糊，
   在架构文档中标注【待澄清】，不要自行假设。"

Coder Agent:
  "严格遵循架构设计文档。如果发现架构设计有潜在问题，
   在代码注释中用【疑问】标注，但不要自行修改架构。"

Reviewer Agent:
  "你是最终质量把关者。如果 Coder 标注了【疑问】，
   你需要判断是否需要退回 Architect 重新设计。
   输出格式（JSON）：
   {"decision": "approved"/"revise", "feedback": "...", "back_to": "coder"/"architect"}"
```

---

## 6. Supervisor 模式详解

### 为什么用 Supervisor 而不是纯顺序流水线？

```
顺序流水线（固定路由）：
  PM → Architect → Coder → Reviewer → END
  问题：Reviewer 打回后怎么办？修改后是否需要重新审查？
  固定边无法处理循环和条件跳转。

Supervisor 模式（动态路由）：
  Supervisor 每一步都判断：
  - 当前阶段完成了什么？
  - 下一个应该派谁？
  - 是否需要重新审查？
  - 是否满足结束条件？

  这本质上是一个状态机，Supervisor 是状态转移函数。
```

### Supervisor 决策逻辑

```python
SUPERVISOR_SYSTEM = """你是项目经理（Supervisor），协调 4 个专家 Agent：

Agent 列表：
- pm: 需求分析，拆解为任务列表
- architect: 系统架构设计
- coder: 根据架构编写代码
- reviewer: 代码审查

决策规则（按优先级）：
1. 无任务列表 → pm
2. 有任务列表、无架构设计 → architect
3. 有架构设计、无代码 → coder
4. 有代码、未经审查 → reviewer
5. reviewer 要求修改 且 revision_count < 3 → coder（修改代码）
6. reviewer 要求回到架构 → architect（重新设计）
7. reviewer 通过 或 revision_count >= 3 → FINISH

输出 JSON: {"next": "pm"|"architect"|"coder"|"reviewer"|"FINISH"}"""
```

### 图结构

```
                        ┌──────────────┐
                        │  Supervisor  │
                        │  (路由决策)   │
                        └──────┬───────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
     ┌──────────┐       ┌──────────┐       ┌──────────┐
     │    PM    │       │Architect │       │  Coder   │
     │ 需求拆解  │       │ 架构设计  │       │ 代码实现  │
     └────┬─────┘       └────┬─────┘       └────┬─────┘
          │                  │                  │
          └──────────────────┴──────────────────┤
                                                │
                                                ▼
                                         ┌──────────┐
                                         │ Reviewer │
                                         │ 代码审查  │
                                         └────┬─────┘
                                              │
                                    ┌─────────┼─────────┐
                                    │                   │
                              通过 / >3次         打回修改
                                    │                   │
                                    ▼                   ▼
                                  END              Coder/Architect
```

---

## 7. PM + Architect + Coder + Reviewer 实战

### 完整工作流演示

```
用户输入: "开发一个用户管理系统，支持注册、登录、查询用户信息"

═══ Round 1 ═══
Supervisor: 无任务列表 → PM
  PM Agent 产出:
    任务列表:
    1. 用户注册 API (POST /register) - 用户名+密码，返回 token
    2. 用户登录 API (POST /login) - 用户名+密码，返回 JWT
    3. 用户查询 API (GET /users/{id}) - 需要认证
    4. 数据存储 - 用户表设计
    验收标准: 3 个 API 可运行，有基本的错误处理

═══ Round 2 ═══
Supervisor: 有任务列表、无架构 → Architect
  Architect Agent 产出:
    架构设计:
    - 框架: FastAPI (轻量、异步、自动文档)
    - 存储: SQLite + SQLAlchemy ORM
    - 认证: JWT (python-jose)
    - 结构: router → service → model 三层
    - API 设计: POST /register, POST /login, GET /users/me

═══ Round 3 ═══
Supervisor: 有架构、无代码 → Coder
  Coder Agent 产出:
    """完整的 Python 代码（~100 行）"""
    - FastAPI 应用初始化
    - User model (SQLAlchemy)
    - register/login/me 三个端点
    - JWT 生成和验证
    - 基本异常处理

═══ Round 4 ═══
Supervisor: 有代码、未审查 → Reviewer
  Reviewer Agent 审查:
    ✓ 需求覆盖: 3 个 API 全部实现
    ✓ 架构一致: 三层结构符合设计
    ✓ 代码质量: 变量命名清晰，有类型提示
    ✗ 问题: register 端点密码未 hash，存在安全风险
    → 决策: revise（打回修改）

═══ Round 5 ═══
Supervisor: reviewer 要求修改、revision_count=0 → Coder
  Coder Agent 修改:
    添加了 passlib 密码 hash + 登录验证逻辑修正

═══ Round 6 ═══
Supervisor: 有代码、需重新审查 → Reviewer
  Reviewer Agent 审查:
    ✓ 所有问题已修复
    ✓ 密码已 hash
    → 决策: approved ✓

═══ Round 7 ═══
Supervisor: reviewer 通过 → FINISH
  输出: 完整的可运行代码 + 架构文档 + 任务清单
```

### 关键设计决策

1. **Architect 和 Coder 分离的原因**：如果合并，LLM 容易跳过架构直接写代码，导致混乱
2. **Reviewer 不直接改代码**：保持审查者视角独立，避免"裁判下场踢球"
3. **PM 不需要审查代码**：PM 关注需求覆盖，Reviewer 关注代码质量，职责分开
4. **revision_count 限制**：防止 Coder 和 Reviewer 的对立导致无限循环

---

## 8. Multi-Agent 设计决策

### 何时用 Multi-Agent

```
                    任务是否需要多种专业知识？
                           │
              ┌────────────┴────────────┐
              │                         │
              是                        否
              │                         │
              ▼                         ▼
    子任务之间是否有依赖？          单 Agent 足够
              │
    ┌─────────┴─────────┐
    │                   │
    是                  否
    │                   │
    ▼                   ▼
  Supervisor 模式   独立并行 Agent
  （本项目采用）
```

### Agent 数量选择

| Agent 数 | 适用场景 | 风险 |
|----------|----------|------|
| 2-3 | 简单协作（Writer + Reviewer） | 角色不够细分 |
| 4-5 | 中等项目（PM + Arch + Coder + Review） | 协调成本可控 |
| 6-8 | 大项目（多 Coder、多 Reviewer） | 协调开销显著增长 |
| 8+ | 企业级 | 需要专职 Coordinator |

**经验法则**：Agent 数 = 项目中"需要不同 System Prompt 的角色数"。不要为了多而多。

### LangGraph vs 其他 Multi-Agent 框架

| 框架 | 协调方式 | 灵活度 | 学习曲线 |
|------|----------|--------|----------|
| **LangGraph** | 显式图 + State | 最高 | 中 |
| CrewAI | 预定义 Crew 抽象 | 中 | 低 |
| AutoGen | 对话驱动 | 中 | 中 |
| OpenAI Swarm | 函数路由 | 低 | 低 |

选择 LangGraph 的原因：需要 Reviewer 循环打回 + 灵活的 Supervisor 决策，这些需要图的条件边能力。

---

## 9. 常见陷阱与最佳实践

### 陷阱

| 陷阱 | 表现 | 缓解 |
|------|------|------|
| **角色过载** | 一个 Agent 承担太多角色 | 严格单一职责，该拆就拆 |
| **System Prompt 太弱** | Agent 做出不属于自己的决策 | 明确"你只能做 X，不能做 Y" |
| **上下文膨胀** | messages 列表无限制增长 | 定期摘要、压缩历史 |
| **死循环** | Reviewer 和 Coder 来回拉锯 | revision_count 上限 |
| **幻觉传递** | PM 编造需求 → Architect 基于假需求设计 | 每个 Agent 标注信息来源 |
| **过度工程化** | 简单任务用了 6 个 Agent | 先用最简单方案，按需加 Agent |

### 最佳实践

1. **先跑通顺序流，再加 Supervisor**：固定 PM→Arch→Coder→Review 跑通后，再引入 Supervisor 的动态路由
2. **每个 Agent 独立测试**：单独验证每个 Agent 的 System Prompt 效果
3. **State 最小化**：只存 Agent 间传递的信息，不要把所有中间推理都放进 State
4. **日志即文档**：每步打印 Agent 动作，方便调试
5. **用 LangSmith 做追踪**：生产环境用 LangSmith 追踪 Multi-Agent 的执行轨迹
6. **让 Reviewer 输出结构化 JSON**：`{"decision": "approved"/"revise", "feedback": "..."}`，方便 Supervisor 解析

---

## 延伸阅读

- [010-multi-agent-learning/multi_agent_demo.py](./multi_agent_demo.py) — PM + Architect + Coder + Reviewer 的完整 LangGraph 实现
- [007-langgraph-learning](../007-langgraph-learning/langgraph_guide.md) — LangGraph 基础知识（StateGraph、条件边、Memory）
- [009-agent-learning](../009-agent-learning/agent_design.md) — 单 Agent 设计基础（ReAct、Reflection、Tool Calling）
- [LangGraph Supervisor 文档](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Anthropic Building Effective Agents Guide](https://docs.anthropic.com/en/docs/agents-and-tools)
