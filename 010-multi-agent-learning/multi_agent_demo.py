"""
=============================================================================
  Multi-Agent Demo：PM + Architect + Coder + Reviewer
  ──────────────────────────────────────────────────────────────────────────
  配合 multi_agent_architecture.md 学习。

  四个 Agent 协作完成一个项目：
    PM Agent        — 需求分析、任务拆解
    Architect Agent — 系统架构设计
    Coder Agent     — 代码实现
    Reviewer Agent  — 代码审查 + 打回修改

  使用 LangGraph Supervisor 模式实现动态路由和 Reviewer 循环。

  运行方式：
    # 需要设置环境变量（参考 .env.example）
    cp .env.example .env  # 然后编辑 .env 填入 API 信息

    python multi_agent_demo.py                          # 默认任务
    python multi_agent_demo.py "开发一个待办事项 API"      # 自定义任务

  依赖：pip install langgraph langchain-core langchain-openai python-dotenv
=============================================================================
"""

import json
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("❌ 请设置 OPENAI_API_KEY 环境变量（或在 .env 文件中配置）")
    print("   cp .env.example .env  →  编辑 .env 填入 API Key")
    sys.exit(1)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def create_llm(temp=0):
    return ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=temp)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            State 定义                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TeamState(TypedDict):
    # 消息流（add_messages reducer 自动追加）
    messages: Annotated[list, add_messages]

    # Supervisor 路由
    next_agent: str

    # 用户输入
    requirement: str

    # PM 产出
    task_breakdown: str

    # Architect 产出
    architecture: str

    # Coder 产出
    code: str

    # Reviewer 产出
    review_result: str       # "approved" / "revise"
    review_feedback: str

    # 循环控制
    revision_count: int

    # 触发回到 Architect
    back_to_architect: bool


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         Supervisor                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SUPERVISOR_SYSTEM = """你是项目经理（Supervisor），协调 4 个专家 Agent 完成软件开发项目。

Agent 列表：
- pm: 需求分析专家，将需求拆解为结构化任务列表
- architect: 系统架构师，设计技术方案和数据结构
- coder: 开发者，根据架构设计编写可运行的代码
- reviewer: 代码审查员，检查代码质量、需求覆盖、架构一致性

决策规则（严格按优先级执行）：

1. 如果 task_breakdown 为空 → 派 pm（先做需求分析）
2. 如果 task_breakdown 不为空、architecture 为空 → 派 architect（需要架构设计）
3. 如果 architecture 不为空、code 为空 → 派 coder（需要写代码）
4. 如果 code 不为空、review_result 为空 → 派 reviewer（需要审查）
5. 如果 review_result 为 "revise" 且 revision_count < 3：
   - 如果 back_to_architect 为 true → 派 architect（重新设计架构）
   - 否则 → 派 coder（修改代码）
6. 如果 review_result 为 "approved" 或 revision_count >= 3 → FINISH

输出格式（纯 JSON，不要包含其他文字）：
{"next": "pm"} / {"next": "architect"} / {"next": "coder"} / {"next": "reviewer"} / {"next": "FINISH"}"""


def supervisor_node(state: TeamState) -> dict:
    llm = create_llm(temp=0)
    status = [
        f"需求: {state.get('requirement', '')[:100]}",
        f"任务拆解: {'已有' if state.get('task_breakdown') else '无'}",
        f"架构设计: {'已有' if state.get('architecture') else '无'}",
        f"代码: {'已有' if state.get('code') else '无'}",
        f"审查结果: {state.get('review_result', '无')}",
        f"修改次数: {state.get('revision_count', 0)}",
        f"回到架构: {state.get('back_to_architect', False)}",
    ]

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content="\n".join(status)),
    ])

    try:
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        decision = json.loads(raw)
        next_agent = decision.get("next", "FINISH")
    except (json.JSONDecodeError, IndexError):
        next_agent = "FINISH"

    print(f"\n{'─'*50}")
    print(f"  🧭 [Supervisor] → {next_agent}")
    print(f"{'─'*50}")
    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"[Supervisor] 决策: {next_agent}")],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      PM Agent（需求分析）                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PM_SYSTEM = """你是技术项目经理。你的唯一职责是将用户需求拆解为结构化的开发任务列表。

要求：
- 按"优先级"排序，分为 P0（核心）、P1（重要）、P2（锦上添花）
- 每个任务包含：名称、简要描述、验收标准
- 只拆解需求，不设计架构、不写代码
- 如果需求不明确，标注【待澄清】但不自行假设

输出格式（Markdown）：
## 需求分析
（1-2 句话概述）

## 任务列表
### P0（核心功能）
1. [任务名] - 描述 - 验收标准: XXX

### P1（重要功能）
2. [任务名] - 描述 - 验收标准: XXX

### P2（可选）
3. [任务名] - 描述 - 验收标准: XXX"""


def pm_node(state: TeamState) -> dict:
    print(f"  📋 [PM] 分析需求...")
    llm = create_llm(temp=0.3)

    response = llm.invoke([
        SystemMessage(content=PM_SYSTEM),
        HumanMessage(content=f"用户需求:\n{state['requirement']}"),
    ])

    task_breakdown = response.content
    summary = task_breakdown[:120].replace("\n", " ")
    print(f"      产出: {summary}...")

    return {
        "task_breakdown": task_breakdown,
        "messages": [AIMessage(content=f"[PM 任务拆解]\n{task_breakdown}")],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   Architect Agent（架构设计）                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECT_SYSTEM = """你是系统架构师。你的唯一职责是基于 PM 的任务列表设计技术方案。

要求：
- 严格基于 PM 的任务列表，不要超出范围
- 如果任务列表中有【待澄清】标注，在架构文档中也标注并在该处选择最合理的默认方案
- 选择合适的技术栈并说明理由
- 设计数据模型（字段、类型、关系）
- 设计 API 接口（方法、路径、参数、返回）
- 设计代码结构（文件组织、模块划分）
- 如果修改次数 > 0 且 Reviewer 指出架构问题，请修正架构设计

输出格式（Markdown）：
## 技术选型
- 框架: XXX（理由: XXX）

## 数据模型
### ModelName
- field: type（说明）

## API 设计
### POST /xxx
- 功能: XXX
- 请求体: {...}
- 响应: {...}

## 代码结构
```
project/
├── main.py
├── models.py
├── ...
```"""


def architect_node(state: TeamState) -> dict:
    revision = state.get("revision_count", 0)
    if revision > 0 and state.get("back_to_architect"):
        print(f"  🏗️  [Architect] 第{revision}次修改架构...")
    else:
        print(f"  🏗️  [Architect] 设计架构...")

    llm = create_llm(temp=0.3)

    user_msg = f"PM 任务列表:\n{state['task_breakdown']}"
    if state.get("review_feedback") and state.get("back_to_architect"):
        user_msg += f"\n\nReviewer 提出的架构问题:\n{state['review_feedback']}\n请修正架构设计。"

    response = llm.invoke([
        SystemMessage(content=ARCHITECT_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    architecture = response.content
    print(f"      产出: {architecture[:120].replace(chr(10), ' ')}...")

    return {
        "architecture": architecture,
        "messages": [AIMessage(content=f"[Architect 架构设计]\n{architecture}")],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     Coder Agent（代码实现）                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CODER_SYSTEM = """你是资深开发者。你的唯一职责是根据架构设计编写可运行的 Python 代码。

要求：
- 严格遵循架构设计文档中的技术选型和设计
- 代码必须完整、可直接运行（使用 `if __name__ == "__main__"` 提供 demo）
- 包含必要的异常处理和输入校验
- 代码清晰，关键逻辑有简洁注释
- 如果架构设计有疑问，用 # [疑问] 注释标注，但不要自行修改架构
- 如果 Reviewer 给了修改意见，请基于意见修改代码

输出格式：
```python
# 完整代码
```
（代码块之后可以附加一段简短的说明）"""


def coder_node(state: TeamState) -> dict:
    revision = state.get("revision_count", 0)
    if revision > 0:
        print(f"  💻 [Coder] 第{revision}次修改代码...")
    else:
        print(f"  💻 [Coder] 编写代码...")

    llm = create_llm()  # 代码生成用默认温度

    user_msg = f"需求:\n{state['requirement']}\n\n架构设计:\n{state['architecture']}"
    if state.get("review_feedback") and not state.get("back_to_architect"):
        user_msg += f"\n\nReviewer 修改意见:\n{state['review_feedback']}\n请根据意见修改代码。"

    response = llm.invoke([
        SystemMessage(content=CODER_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    code = response.content
    print(f"      产出: {code[:120].replace(chr(10), ' ')}...")

    return {
        "code": code,
        "messages": [AIMessage(content=f"[Coder 代码]\n{code}")],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                   Reviewer Agent（代码审查）                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

REVIEWER_SYSTEM = """你是严格的代码审查员。你的唯一职责是检查代码质量。

审查维度：
1. 需求覆盖：代码是否实现了所有 PM 任务（对照 task_breakdown）
2. 架构一致性：代码是否遵循了架构设计（对照 architecture）
3. 代码质量：命名是否清晰、结构是否合理、异常处理是否完善
4. 可运行性：代码能否直接运行？依赖是否明确？
5. 安全性：是否有基本的输入校验和安全处理？

输出格式（严格 JSON）：
{
  "decision": "approved",
  "score": 9,
  "feedback": "整体质量评价",
  "issues": [],
  "back_to_architect": false
}

或：

{
  "decision": "revise",
  "score": 5,
  "feedback": "具体问题描述和修改建议",
  "issues": ["问题1", "问题2"],
  "back_to_architect": false
}

decision 取值：
- "approved": 代码质量合格，可以交付
- "revise": 需要修改

back_to_architect 取值：
- true: 问题出在架构层，需要退回 Architect 重新设计
- false: 问题出在代码层，Coder 修改即可

评分标准：
- 9-10: 优秀，可直接交付
- 7-8: 良好，有少量改进空间
- 5-6: 一般，需要修改
- <5: 差，需要大幅重写

注意：
- 不要自己修改代码，只指出问题
- 如果 Coder 用 # [疑问] 标注了架构问题，请认真判断是否需要退回 Architect
- 小问题（命名、注释）给 approved，大问题（逻辑错误、遗漏功能、安全漏洞）给 revise"""


def reviewer_node(state: TeamState) -> dict:
    print(f"  🔍 [Reviewer] 审查代码...")
    llm = create_llm(temp=0.1)

    user_msg = f"""需求文档:
{state['task_breakdown']}

架构设计:
{state['architecture']}

待审查代码:
{state['code']}"""

    response = llm.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    # 解析 JSON
    try:
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        review = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        review = {
            "decision": "approved",
            "score": 7,
            "feedback": "Unable to parse review JSON, defaulting to approved.",
            "issues": [],
            "back_to_architect": False,
        }

    decision = review.get("decision", "approved")
    score = review.get("score", 0)
    feedback = review.get("feedback", "")
    back_to_architect = review.get("back_to_architect", False)

    print(f"      评分: {score}/10 → {decision}")
    if feedback:
        print(f"      反馈: {feedback[:200]}")

    return {
        "review_result": decision,
        "review_feedback": feedback,
        "revision_count": state.get("revision_count", 0) + (1 if decision == "revise" else 0),
        "back_to_architect": back_to_architect,
        "messages": [AIMessage(content=f"[Reviewer] 评分: {score}/10, 决策: {decision}\n反馈: {feedback}")],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         Graph 构建                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_router(state: TeamState) -> str:
    """条件边：Supervisor 决策后路由到对应 Agent"""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent in ("pm", "architect", "coder", "reviewer"):
        return next_agent
    return "FINISH"


def build_team_graph():
    """构建 Multi-Agent 协作图"""
    builder = StateGraph(TeamState)

    # 添加节点
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("pm", pm_node)
    builder.add_node("architect", architect_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reviewer", reviewer_node)

    # START → supervisor（每次循环都先经过 Supervisor 决策）
    builder.add_edge(START, "supervisor")

    # Supervisor → 各 Agent（条件路由）
    builder.add_conditional_edges("supervisor", build_router, {
        "pm": "pm",
        "architect": "architect",
        "coder": "coder",
        "reviewer": "reviewer",
        "FINISH": END,
    })

    # 各 Agent 完成后 → 回到 Supervisor
    builder.add_edge("pm", "supervisor")
    builder.add_edge("architect", "supervisor")
    builder.add_edge("coder", "supervisor")
    builder.add_edge("reviewer", "supervisor")

    return builder.compile()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         主入口                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    # 获取用户输入
    if len(sys.argv) > 1:
        requirement = " ".join(sys.argv[1:])
    else:
        requirement = "开发一个用户管理系统，支持用户注册、登录、查询个人信息"

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Multi-Agent 协作系统                                       ║")
    print("║   PM → Architect → Coder → Reviewer                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  📌 项目需求: {requirement}\n")

    graph = build_team_graph()

    initial_state: TeamState = {
        "messages": [HumanMessage(content=requirement)],
        "next_agent": "pm",
        "requirement": requirement,
        "task_breakdown": "",
        "architecture": "",
        "code": "",
        "review_result": "",
        "review_feedback": "",
        "revision_count": 0,
        "back_to_architect": False,
    }

    result = graph.invoke(initial_state)

    # 输出结果
    print("\n" + "=" * 60)
    print("  📦 协作完成！最终产出")
    print("=" * 60)

    print(f"\n{'─'*40}")
    print("  📋 PM 任务拆解:")
    print(f"{'─'*40}")
    print(result.get("task_breakdown", "(无)"))

    print(f"\n{'─'*40}")
    print("  🏗️  架构设计:")
    print(f"{'─'*40}")
    print(result.get("architecture", "(无)"))

    print(f"\n{'─'*40}")
    print("  💻 代码实现:")
    print(f"{'─'*40}")
    print(result.get("code", "(无)"))

    review = result.get("review_result")
    feedback = result.get("review_feedback")
    revisions = result.get("revision_count", 0)
    print(f"\n{'─'*40}")
    print(f"  🔍 审查结果: {review} | 修改次数: {revisions}")
    print(f"{'─'*40}")
    if feedback:
        print(f"  {feedback}")

    print(f"\n{'='*60}")
    print(f"  ✅ Multi-Agent 协作完成！")
    print(f"  总消息数: {len(result.get('messages', []))}")
    print(f"  修改次数: {revisions}")
    print(f"{'='*60}")

    # 保存代码到文件
    code = result.get("code", "")
    if code:
        # 提取代码块
        extracted = code
        if "```python" in extracted:
            extracted = extracted.split("```python")[1].split("```")[0]
        elif "```" in extracted:
            parts = extracted.split("```")
            if len(parts) >= 2:
                extracted = parts[1]

        filename = "/tmp/multi_agent_output.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(extracted.strip())
        print(f"\n  📄 代码已保存到: {filename}")


if __name__ == "__main__":
    main()
