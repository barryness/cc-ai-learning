"""
=============================================================================
  LangGraph 三大 Demo：从线性工作流到多 Agent 协作
  ──────────────────────────────────────────────────────────────────────────
  配合 langgraph_guide.md 学习。每个 Demo 独立可运行。

  Demo 1: 简单工作流（问题 → 搜索 → 回答）
  Demo 2: 研究 Agent（拆解 → 搜索 → 总结 → 报告）
  Demo 3: 多 Agent 系统（Supervisor + Research + Writer + Reviewer + Memory）

  运行方式：
    python langgraph_demo.py          # 运行所有 Demo
    python langgraph_demo.py --demo 1 # 只运行 Demo 1
    python langgraph_demo.py --demo 2 # 只运行 Demo 2
    python langgraph_demo.py --demo 3 # 只运行 Demo 3
=============================================================================
"""

import os
import sys
import json
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.getenv("OPENAI_API_KEY")

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def create_llm(temp=0):
    return ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=temp)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        Demo 1：简单工作流                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 架构：START → search → answer → END
#
# 这是 LangGraph 的 "Hello World"。演示了：
#   - State 定义（最简单的 2 字段）
#   - Node 定义（2 个节点，各做一件事）
#   - 线性边（无分支、无循环）
#
# 对比手写 pipeline：
#   手写: results = search(q); answer = llm.invoke(f"基于{results}回答{q}")
#   LangGraph: 定义 State + Node + Edge，框架负责执行和追踪
#   区别：手写版无法追踪中间状态，LangGraph 版每一步都可观测


class Demo1State(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: str


def demo1_search_node(state: Demo1State) -> dict:
    """搜索节点：从知识库中检索相关信息。"""
    query = state["messages"][-1].content
    print(f"  🔍 [search] 搜索: {query[:60]}...")

    # 模拟知识库搜索
    knowledge = {
        "python": "Python 由 Guido van Rossum 于 1991 年发布，最新稳定版 Python 3.13。",
        "ai": "人工智能（AI）2026 年趋势：多模态模型、AI Agent 自动化、SLM 小模型兴起。",
        "langgraph": "LangGraph 是构建有状态 Agent 应用的框架，核心概念：StateGraph、Node、Edge。",
        "rag": "RAG（检索增强生成）结合了信息检索和文本生成，减少 LLM 幻觉。",
        "embedding": "Embedding 将文本映射到向量空间，语义相似的文本向量距离更近。",
    }
    result = "未找到相关信息。"
    for k, v in knowledge.items():
        if k in query.lower():
            result = v
            break

    print(f"      结果: {result[:80]}...")
    return {"search_results": result}


def demo1_answer_node(state: Demo1State) -> dict:
    """回答节点：基于搜索结果生成答案。"""
    print(f"  💬 [answer] 生成回答...")

    llm = create_llm()
    response = llm.invoke([
        SystemMessage(content="你是一个基于知识库回答问题的助手。用中文简洁回答。"),
        HumanMessage(content=(
            f"知识库搜索结果：{state['search_results']}\n\n"
            f"用户问题：{state['messages'][-1].content}\n\n"
            f"请基于搜索结果回答。如果搜索结果不相关，请如实说明。"
        )),
    ])
    print(f"      回答: {response.content[:100]}...")
    return {"messages": [AIMessage(content=response.content)]}


def build_demo1():
    """构建 Demo 1 图：START → search → answer → END"""
    builder = StateGraph(Demo1State)
    builder.add_node("search", demo1_search_node)
    builder.add_node("answer", demo1_answer_node)
    builder.add_edge(START, "search")
    builder.add_edge("search", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def run_demo1():
    print("\n" + "=" * 65)
    print("  Demo 1：简单工作流（问题 → 搜索 → 回答）")
    print("=" * 65)

    graph = build_demo1()
    queries = ["什么是 LangGraph？", "Python 最新版本是什么？"]

    for q in queries:
        print(f"\n  📝 用户: {q}")
        result = graph.invoke({"messages": [HumanMessage(content=q)], "search_results": ""})
        final = result["messages"][-1].content
        print(f"  🤖 AI: {final}")

    print(f"\n  ✅ Demo 1 完成！")
    print("  💡 要点: 2 个节点 + 线性边 = 最简单的 LangGraph 工作流")
    print("     手写 pipeline 也能做，但 LangGraph 让每一步都可追踪。")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        Demo 2：研究 Agent                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 架构：START → decompose → search → summarize → report → END
#
# 比 Demo 1 多了两步：问题拆解 + 综合摘要。
# 演示了多步骤协作 —— 每步的输出是下一步的输入，State 字段逐步"生长"。
#
# 关键区别：
#   Demo 1: 用户问什么就搜什么（1 次 LLM 调用）
#   Demo 2: LLM 先拆解问题 → 逐个子问题搜索 → 综合摘要 → 生成报告（4 次 LLM 调用）


class Demo2State(TypedDict):
    messages: Annotated[list, add_messages]
    main_question: str
    sub_questions: list[str]
    search_results: str
    summary: str
    final_report: str


def demo2_decompose_node(state: Demo2State) -> dict:
    """拆解节点：LLM 把大问题拆成子问题。"""
    question = state["main_question"]
    print(f"  🧩 [decompose] 拆解问题: {question[:60]}...")

    llm = create_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "你是一个问题分析专家。把用户的问题拆解成 2-4 个子问题，每个子问题一句话。\n"
            "输出格式：每行一个子问题，不要编号，不要其他文字。"
        )),
        HumanMessage(content=f"请拆解这个问题：{question}"),
    ])

    sub_qs = [line.strip("- 123456789. ") for line in response.content.strip().split("\n") if line.strip()]
    sub_qs = sub_qs[:4]  # 最多 4 个

    for i, sq in enumerate(sub_qs, 1):
        print(f"      子问题{i}: {sq}")
    return {"sub_questions": sub_qs}


def demo2_search_node(state: Demo2State) -> dict:
    """搜索节点：对每个子问题分别搜索。"""
    print(f"  🔍 [search] 搜索 {len(state['sub_questions'])} 个子问题...")

    # 模拟知识库
    knowledge = {
        "python": "Python 由 Guido van Rossum 于 1991 年创建。最新版 Python 3.13 于 2024 年 10 月发布，引入了更好的 JIT 编译器。Python 以简洁语法和丰富生态著称，在 AI、Web 开发、自动化领域广泛应用。",
        "ai": "人工智能（AI）是计算机科学的重要分支。2026 年主要趋势包括：多模态大模型（同时处理文本、图像、音频）、AI Agent 自动化工作流（自主完成复杂任务）、小型高效模型（SLM）在端侧设备上的兴起、AI 编程助手（如 GitHub Copilot）的全面普及。",
        "langgraph": "LangGraph 是 LangChain 团队开发的 Agent 框架，核心理念是用有向图建模 Agent 工作流。关键概念：State（共享内存）、Node（执行单元）、Edge（流程控制）。与 LangChain LCEL 互补：LCEL 擅长线性 Pipeline，LangGraph 擅长循环/分支/多角色协作。",
        "rag": "RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的技术架构。工作流程：用户问题 → 检索相关文档 → 将文档作为上下文注入 Prompt → LLM 生成答案。RAG 能显著减少 LLM 的幻觉问题，在企业知识库问答中广泛应用。",
        "trend": "2026年 AI 领域关键趋势：1) Agent 化——AI 从被动回答转向主动执行任务；2) 多模态融合——文本/图像/视频/音频统一理解；3) 端侧部署——模型小型化，在手机和 IoT 设备上运行；4) AI 安全与对齐——确保 AI 行为符合人类价值观；5) 开源生态繁荣——Llama、Mistral、DeepSeek 等开源模型性能接近闭源。",
        "embedding": "Embedding（嵌入）是将文本、图像等数据映射为固定维度向量的技术。语义相似的文本在向量空间中距离更近。常用模型：OpenAI text-embedding-3、BGE、Sentence-BERT。应用场景：语义搜索、聚类、推荐系统、异常检测。",
    }

    results_parts = []
    for i, sq in enumerate(state["sub_questions"], 1):
        # 简单关键词匹配
        matched = None
        for k, v in knowledge.items():
            if k in sq.lower():
                matched = v
                break
        if not matched:
            # 尝试 LLM 中查找更多匹配
            for k, v in knowledge.items():
                if any(w in sq.lower() for w in k.split()):
                    matched = v
                    break
        content = matched or f"关于'{sq}'未找到详细信息（模拟搜索）"
        results_parts.append(f"[子问题{i}] {sq}\n搜索结果：{content}")
        print(f"      [{i}] {sq[:40]}... → {'找到' if matched else '未找到'}")

    return {"search_results": "\n\n---\n\n".join(results_parts)}


def demo2_summarize_node(state: Demo2State) -> dict:
    """总结节点：LLM 综合所有搜索结果。"""
    print(f"  📊 [summarize] 综合搜索结果...")

    llm = create_llm()
    response = llm.invoke([
        SystemMessage(content="你是一个信息分析专家。把多个搜索结果综合成一份简洁的摘要，提取关键信息。"),
        HumanMessage(content=(
            f"原始问题：{state['main_question']}\n\n"
            f"各子问题的搜索结果：\n{state['search_results']}\n\n"
            f"请输出一份 2-3 段的综合摘要。"
        )),
    ])
    print(f"      摘要: {response.content[:100]}...")
    return {"summary": response.content}


def demo2_report_node(state: Demo2State) -> dict:
    """报告节点：LLM 基于摘要生成结构化报告。"""
    print(f"  📝 [report] 生成报告...")

    llm = create_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "你是一个专业的技术报告撰写专家。基于提供的摘要，生成一份结构清晰的报告。\n"
            "使用 Markdown 格式，包含：标题、概述、详细内容、总结。"
        )),
        HumanMessage(content=(
            f"原始问题：{state['main_question']}\n\n"
            f"综合摘要：{state['summary']}\n\n"
            f"请生成最终报告。"
        )),
    ])
    print(f"      报告: {response.content[:120]}...")
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def build_demo2():
    """构建 Demo 2 图：START → decompose → search → summarize → report → END"""
    builder = StateGraph(Demo2State)
    builder.add_node("decompose", demo2_decompose_node)
    builder.add_node("search", demo2_search_node)
    builder.add_node("summarize", demo2_summarize_node)
    builder.add_node("report", demo2_report_node)
    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "report")
    builder.add_edge("report", END)
    return builder.compile()


def run_demo2():
    print("\n" + "=" * 65)
    print("  Demo 2：研究 Agent（拆解 → 搜索 → 总结 → 报告）")
    print("=" * 65)

    graph = build_demo2()
    question = "2026年人工智能的发展趋势是什么？"

    print(f"\n  📝 问题: {question}\n")
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "main_question": question,
        "sub_questions": [],
        "search_results": "",
        "summary": "",
        "final_report": "",
    })

    print(f"\n  {'─' * 60}")
    print(f"  📄 最终报告:\n")
    print(f"  {result['final_report']}")

    print(f"\n  ✅ Demo 2 完成！")
    print("  💡 要点: 4 个节点线性协作，State 字段随流程逐步填充")
    print("     每个节点的输出是下个节点的输入 —— State 就是这个'交接单'。")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     Demo 3：多 Agent 协作系统                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 架构：
#                ┌──────────────────────┐
#                │     Supervisor        │←──────────────────┐
#                │   （项目经理）         │                    │
#                └──────────┬───────────┘                    │
#                           │                                │
#              ┌────────────┼────────────┐                   │
#              │            │            │                   │
#              ▼            ▼            ▼                   │
#         ┌─────────┐ ┌─────────┐ ┌─────────┐              │
#         │Research │ │ Writer  │ │Reviewer │              │
#         │ Agent   │ │ Agent   │ │ Agent   │              │
#         └────┬────┘ └────┬────┘ └────┬────┘              │
#              │           │           │                    │
#              └───────────┴───────────┘────────────────────┘
#                      完成后都回到 Supervisor
#
# 典型流程：
#   User: "写一篇 AI 趋势报告"
#   → Supervisor: "先让 Research Agent 查资料"
#   → Research Agent: 搜索+分析 → 放回 State
#   → Supervisor: "资料有了，让 Writer 写报告"
#   → Writer Agent: 基于研究结果写报告 → 放回 State
#   → Supervisor: "报告写好了，让 Reviewer 审核"
#   → Reviewer Agent: 审核 → 通过/打回修改
#   → 如果打回：Writer 修改 → Reviewer 再审核（循环！）
#   → 通过：FINISH
#
# 关键设计：
#   1. Supervisor 模式：一个"经理 Agent"动态决策流程
#   2. Reviewer 循环：审核不通过 → 打回 Writer 重写（这是 LangChain 做不到的）
#   3. Memory（Checkpointer）：跨对话记忆，追问不需要重新搜索
#   4. 防无限循环：revision_count 限制最多修改 3 次


class Demo3State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    task: str
    research_findings: str
    draft: str
    review_feedback: str
    revision_count: int


# ── Supervisor ──

SUPERVISOR_SYSTEM = """你是一个 AI 项目经理（Supervisor），协调三个专家 Agent：

1. researcher — 研究员，负责搜索信息、分析资料
2. writer — 写手，根据研究结果撰写报告/文章
3. reviewer — 审核员，检查报告质量，决定通过或打回修改

你的决策规则（优先级从高到低）：
- 如果没有任何产出 → 先派 researcher
- 如果已有研究结果但没有草稿 → 派 writer
- 如果已有草稿但未审核（且修改次数 < 1） → 派 reviewer
- 如果 reviewer 要求修改（且修改次数 < 3） → 派 writer 修改
- 如果 reviewer 通过 或 修改次数 >= 3 → FINISH

输出格式（纯 JSON）：
{"next": "researcher"} / {"next": "writer"} / {"next": "reviewer"} / {"next": "FINISH"}"""


def supervisor_node(state: Demo3State) -> dict:
    llm = create_llm()
    status_parts = [f"任务: {state['task'][:200]}"]
    if state.get("research_findings"):
        status_parts.append("研究结果: 已有")
    if state.get("draft"):
        status_parts.append("草稿: 已有")
    if state.get("review_feedback"):
        status_parts.append(f"审核反馈: {state['review_feedback'][:100]}")
    status_parts.append(f"修改次数: {state.get('revision_count', 0)}")

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content="\n".join(status_parts)),
    ])

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        decision = json.loads(raw)
        next_agent = decision.get("next", "FINISH")
    except json.JSONDecodeError:
        next_agent = "FINISH"

    print(f"  ──▶ [Supervisor] → {next_agent}")
    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"[Supervisor] 决定: {next_agent}")],
    }


# ── Research Agent ──

RESEARCHER_SYSTEM = """你是研究员 Agent。搜索并分析用户想了解的信息。

模拟知识库（你只能基于这些信息回答）：
- AI趋势2026：多模态大模型成为主流（GPT-5、Claude 4、Gemini 3），AI Agent 从对话走向自动化执行，SLM 小模型在端侧爆发，AI 编程助手渗透率达 70%。
- LangGraph：LangChain 团队开发的 Agent 框架，核心是 StateGraph（状态图）。支持循环、条件分支、多 Agent 协作。与 LangChain LCEL 互补。
- RAG技术：检索增强生成，减少 LLM 幻觉。2026年演进方向：Agentic RAG（自主检索决策）、Graph RAG（知识图谱增强）、多模态 RAG（图片/表格检索）。
- AI安全：AI Alignment（对齐）、红队测试、内容审核、深度伪造检测是 2026 年 AI 安全四大支柱。
- 市场规模：全球 AI 市场 2026 年预计达 3000 亿美元，年增长率 35%。中国市场占比约 15%。

完成后输出：【研究完成】开头的研究报告。"""


def researcher_node(state: Demo3State) -> dict:
    print(f"  🔬 [Researcher] 开始研究...")
    llm = create_llm()
    response = llm.invoke([
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=f"研究任务: {state['task']}\n请给出详细的研究报告。"),
    ])
    findings = response.content
    print(f"      产出: {findings[:80]}...")
    return {
        "research_findings": findings,
        "messages": [AIMessage(content=f"[研究员报告]\n{findings}")],
    }


# ── Writer Agent ──

WRITER_SYSTEM = """你是专业写手 Agent。根据研究资料撰写报告。

要求：
- Markdown 格式，结构清晰（标题 → 概述 → 分点详述 → 总结）
- 语言流畅自然，适合阅读
- 如果审核员给了修改意见，请根据意见修改

修改时注意：
- 检查是否遗漏重要信息
- 检查结构是否合理
- 检查语言是否流畅"""


def writer_node(state: Demo3State) -> dict:
    revision = state.get("revision_count", 0)
    if revision > 0:
        print(f"  ✍️  [Writer] 第{revision}次修改草稿...")
    else:
        print(f"  ✍️  [Writer] 撰写草稿...")

    llm = create_llm()
    research = state.get("research_findings", "无研究资料")

    user_msg = f"研究资料:\n{research}\n\n请撰写报告。"
    if state.get("review_feedback"):
        user_msg += f"\n\n审核修改意见:\n{state['review_feedback']}\n\n请根据意见修改。"

    response = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    draft = response.content
    print(f"      产出: {draft[:80]}...")
    return {
        "draft": draft,
        "messages": [AIMessage(content=f"[草稿]\n{draft}")],
    }


# ── Reviewer Agent ──

REVIEWER_SYSTEM = """你是审核员 Agent。检查报告质量，决定通过或打回修改。

审核标准：
1. 内容完整性：是否覆盖了研究资料中的关键信息？
2. 结构合理性：是否有清晰的标题、概述、分点、总结？
3. 语言质量：是否流畅、专业、易读？
4. 准确性：是否有编造的内容？

输出格式（纯 JSON）：
- 通过：{"decision": "approved", "feedback": ""}
- 打回：{"decision": "revise", "feedback": "具体修改意见（引用原文段落，给出明确建议）"}

重要：打回必须给出具体、可执行的修改意见，不要泛泛地说"需要改进"。
如果修改次数已达 3 次，即使有小问题也应该通过。"""


def reviewer_node(state: Demo3State) -> dict:
    revision = state.get("revision_count", 0)
    print(f"  🔍 [Reviewer] 审核草稿（第{revision + 1}轮）...")

    llm = create_llm()
    draft = state.get("draft", "")
    research = state.get("research_findings", "")

    response = llm.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=(
            f"研究资料:\n{research[:1000]}\n\n"
            f"报告草稿:\n{draft[:1500]}\n\n"
            f"当前修改次数: {revision}\n"
            f"请审核并输出 JSON 决策。"
        )),
    ])

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        result = json.loads(raw)
        decision = result.get("decision", "approved")
        feedback = result.get("feedback", "")
    except json.JSONDecodeError:
        decision = "approved"
        feedback = ""

    if decision == "approved":
        print(f"      ✅ 审核通过！")
    else:
        print(f"      ❌ 打回修改: {feedback[:80]}...")

    return {
        "review_feedback": feedback if decision == "revise" else "",
        "revision_count": revision + 1,
        "messages": [AIMessage(content=(
            f"[审核结果] {'✅ 通过' if decision == 'approved' else '❌ 需修改: ' + feedback}"
        ))],
    }


# ── Router ──

def demo3_router(state: Demo3State) -> str:
    return state.get("next_agent", "FINISH")


# ── Build & Run ──

def build_demo3(with_memory=False):
    """构建 Demo 3 多 Agent 系统。with_memory=True 时启用 Checkpointer。"""
    builder = StateGraph(Demo3State)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor", demo3_router,
        {"researcher": "researcher", "writer": "writer", "reviewer": "reviewer", "FINISH": END}
    )

    # 所有 Agent 完成后回到 Supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("reviewer", "supervisor")

    if with_memory:
        from langgraph.checkpoint.memory import MemorySaver
        return builder.compile(checkpointer=MemorySaver())
    return builder.compile()


def run_demo3():
    print("\n" + "=" * 65)
    print("  Demo 3：多 Agent 系统（Supervisor + Research + Writer + Reviewer）")
    print("=" * 65)

    # ── 3a: 基础多 Agent 协作 ──
    print("\n  ── Part A: 基础多 Agent 协作 ──\n")

    graph = build_demo3()
    initial = {
        "messages": [HumanMessage(content="写一篇关于 2026 年 AI 发展趋势的报告")],
        "next_agent": "",
        "task": "写一篇关于 2026 年 AI 发展趋势的报告",
        "research_findings": "",
        "draft": "",
        "review_feedback": "",
        "revision_count": 0,
    }

    # 用 stream 模式观察每一步
    for step_output in graph.stream(initial, {"recursion_limit": 20}):
        for node_name, output in step_output.items():
            # 只展示关键信息
            if node_name == "supervisor":
                pass  # supervisor 的 print 已经展示了
            elif node_name == "reviewer":
                fb = output.get("review_feedback", "")
                if fb:
                    print(f"      📋 修改意见: {fb[:120]}...")

    # 最终结果
    final = graph.invoke(initial, {"recursion_limit": 20})
    print(f"\n  {'─' * 60}")
    print(f"  📄 最终报告:\n")
    print(f"  {final.get('draft', '(无)')[:800]}")

    # ── 3b: Memory 演示 ──
    print(f"\n  ── Part B: 跨对话 Memory（Checkpointer） ──\n")

    graph_with_mem = build_demo3(with_memory=True)
    config = {"configurable": {"thread_id": "demo3_session"}}

    # 第 1 轮：研究 AI 趋势
    print("  📝 第 1 轮: 研究 AI 趋势")
    graph_with_mem.invoke({
        "messages": [HumanMessage(content="研究一下 AI Agent 的发展趋势")],
        "next_agent": "",
        "task": "研究一下 AI Agent 的发展趋势",
        "research_findings": "",
        "draft": "",
        "review_feedback": "",
        "revision_count": 0,
    }, config)

    # 第 2 轮：追问（同一个 thread_id，Agent 记得上一轮的信息！）
    print("\n  📝 第 2 轮（追问，同一个 thread_id）: 详细说说多模态这一块")
    result2 = graph_with_mem.invoke({
        "messages": [HumanMessage(content="详细说说多模态这一块")],
    }, config)

    print(f"\n  🤖 追问回答: {result2['messages'][-1].content[:300]}...")

    print(f"\n  ✅ Demo 3 完成！")
    print("  💡 要点:")
    print("     1. Supervisor 模式：中心化调度，Agent 完成工作后回到 Supervisor")
    print("     2. Reviewer 循环：审核不通过 → 打回 Writer 修改 → 再审核")
    print("     3. Memory（Checkpointer）：同一个 thread_id，Agent 记住之前的所有状态")
    print("     4. 防无限循环：revision_count 限制最多修改 3 次")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              Main Entry                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    demo_arg = None
    if "--demo" in sys.argv:
        try:
            idx = sys.argv.index("--demo")
            demo_arg = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("用法: python langgraph_demo.py [--demo 1|2|3]")
            return

    print("=" * 65)
    print("  🚀 LangGraph 三大 Demo")
    print("  Guide: langgraph_guide.md  |  Demo: langgraph_demo.py")
    print("=" * 65)

    if demo_arg is None or demo_arg == 1:
        run_demo1()
    if demo_arg is None or demo_arg == 2:
        run_demo2()
    if demo_arg is None or demo_arg == 3:
        run_demo3()

    print(f"\n{'=' * 65}")
    print("  🎉 全部 Demo 完成！")
    print(f"  📖 下一步：阅读 langgraph_guide.md 理解背后的设计原理")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
