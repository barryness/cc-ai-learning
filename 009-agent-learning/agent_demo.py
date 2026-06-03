"""
Research Agent Demo —— 从 ReAct 到 Reflection 的渐进式实现

三个阶段：
  Phase 1: 基础 Research Agent（ReAct + 搜索 + 报告生成）
  Phase 2: + 记忆系统（短期对话记忆 + 长期知识记忆）
  Phase 3: + 反思机制（自审 + 自动修改 + 质量评分）

运行方式：
  python agent_demo.py            # 运行所有阶段
  python agent_demo.py --phase 1  # 只运行 Phase 1
  python agent_demo.py --phase 2  # 只运行 Phase 2
  python agent_demo.py --phase 3  # 只运行 Phase 3
"""

import json
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TypedDict

# ──────────────────────────────────────────────────────────────
# 模拟工具层 —— 用预设知识库模拟搜索，避免依赖外部 API
# ──────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    # Agent 相关
    "agent": [
        {"title": "什么是 AI Agent", "content": "AI Agent 是能够自主感知环境、做出决策并执行动作的智能系统。核心组件包括：LLM（大脑）、规划（拆解任务）、工具（执行动作）、记忆（存储经验）。Agent 与 Chatbot 的区别在于：Agent 是目标驱动的自主循环，Chatbot 是被动的单次问答。"},
        {"title": "Agent 架构模式", "content": "主流 Agent 架构包括：ReAct（推理-行动循环）、Plan-Execute（先规划后执行）、Multi-Agent（多角色协作）、Reflexion（带反思的 Agent）。选择哪种架构取决于任务复杂度：简单任务用 ReAct，多步任务用 Plan-Execute，需要高质量输出用 Reflexion。"},
        {"title": "ReAct 范式", "content": "ReAct（Reasoning + Acting）是 Google DeepMind 2022 年提出的 Agent 范式。核心循环为：Thought（推理当前状态）→ Action（选择并执行工具）→ Observation（观察工具结果）→ 循环直到得出 Final Answer。ReAct 的优势是推理过程可追溯、能自我纠正、与工具解耦。"},
    ],
    "planning": [
        {"title": "Agent 中的 Planning 策略", "content": "Planning 让 Agent 在行动前先思考方案。分为隐式规划（ReAct 每次思考时自然形成）和显式规划（先生成完整计划再逐步执行）。显式规划适用于复杂多步任务，但需要处理计划随执行调整的问题。常见技术包括：任务分解（Task Decomposition）、计划修订（Plan Revision）、子目标管理。"},
    ],
    "tool calling": [
        {"title": "Tool Calling 机制", "content": "Tool Calling 让 LLM 以结构化方式声明要调用哪个函数，而非生成自然语言。每个工具包含三个要素：name（名称）、description（功能描述）、inputSchema（参数 JSON Schema）。LLM 返回结构化 tool_call 对象，由 Agent 框架负责实际执行。OpenAI 的 Function Calling 和 Anthropic 的 Tool Use 本质上都是 Tool Calling。"},
        {"title": "MCP 协议", "content": "MCP（Model Context Protocol）是 Anthropic 提出的 Tool Calling 标准化协议。它基于 JSON-RPC 2.0，通过 stdio 或 SSE 通信。MCP 的三个角色：Host（AI 应用）、Client（协议客户端）、Server（工具提供方）。MCP 将 Tool Calling 从平台绑定中解放出来，让工具可以跨平台复用。"},
    ],
    "reflection": [
        {"title": "Reflection 反思机制", "content": "Reflection 让 Agent 审视自己的输出并自我改进。流程为：生成初始输出 → 自我审视（检查事实、完整、逻辑、清晰度）→ 发现不足 → 修改 → 再审视。为防止无限循环，通常设置最大修改轮次（如 3 次）和质量阈值（如评分 >= 8 通过）。Reflexion 论文（2023）系统化了这种方法。"},
        {"title": "Reflexion 论文核心思想", "content": "Reflexion 通过口头强化学习（Verbal Reinforcement Learning）让 Agent 从失败中学习。不同于传统 RL 的数值奖励，Reflexion 使用语言反馈作为学习信号。Agent 在 episode 结束后反思：什么做对了、什么做错了、下次应该怎么做。反思结果存入长期记忆，指导未来的行动。"},
    ],
    "langgraph": [
        {"title": "LangGraph 框架", "content": "LangGraph 是 LangChain 团队开发的图编排框架，用于构建有状态的 Agent。核心概念：StateGraph（带状态类型的图）、Node（处理节点）、Edge（边和条件边）、Checkpointer（持久化状态）。LangGraph 支持 Supervisor 模式的多 Agent 协作，其中 Supervisor 节点根据状态动态路由到不同 Worker。"},
        {"title": "LangGraph vs CrewAI", "content": "LangGraph 和 CrewAI 都是 Agent 框架但思路不同。LangGraph 是底层图编排框架，灵活但需要手动设计图结构。CrewAI 是高层抽象，定义了 Agent、Task、Crew 等概念，开箱即用但自由度较低。选择：需要精确控制流程 → LangGraph；快速构建多 Agent 应用 → CrewAI。"},
    ],
    # 通用知识
    "python": [
        {"title": "Python Agent 开发", "content": "Python 是 Agent 开发的主流语言。常用库：LangChain（Agent 抽象 + Tool 装饰器）、LangGraph（图编排）、OpenAI SDK（原生 Function Calling）、MCP Python SDK（MCP Server 开发）。Python 3.10+ 的类型提示（TypedDict、dataclass）非常适合定义 Agent 状态。"},
    ],
    "llm": [
        {"title": "LLM 在 Agent 中的角色", "content": "LLM 是 Agent 的推理引擎。关键能力：工具选择（根据任务选择合适的工具）、参数提取（从输入中提取工具参数）、结果合成（将多个工具结果合成为连贯回答）、自我纠错（发现错误后调整策略）。模型规模越大，这些能力越强，但 token 消耗也越高。"},
    ],
}


def simulate_search(query: str, top_k: int = 3) -> list[dict]:
    """模拟搜索引擎：在知识库中做关键词匹配"""
    results = []
    query_lower = query.lower()

    for category, articles in KNOWLEDGE_BASE.items():
        for article in articles:
            # 计算匹配度（标题匹配权重更高）
            title_match = sum(1 for word in query_lower.split() if word in article["title"].lower())
            content_match = sum(1 for word in query_lower.split() if word in article["content"].lower())
            score = title_match * 3 + content_match

            if score > 0:
                results.append({"title": article["title"], "content": article["content"], "category": category, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def simulate_write_report(content: str, filename: str) -> str:
    """模拟写报告到文件"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"报告已写入: {filename}"

# ──────────────────────────────────────────────────────────────
# Phase 1: 基础 Research Agent —— ReAct 循环
# ──────────────────────────────────────────────────────────────


class ResearchState(TypedDict):
    """Agent 状态"""
    task: str                          # 原始任务
    history: list[dict]                # 对话历史
    search_results: list[dict]         # 收集的搜索结果
    report: str                        # 最终报告
    steps: int                         # 已执行步数
    status: str                        # thinking / acting / done


# LLM 模拟器 —— 用规则 + 模板模拟 LLM 的 ReAct 推理
# 真实场景中替换为 OpenAI / Claude API 调用
class LLMSimulator:
    """模拟 LLM 的 ReAct 推理 —— 演示用，生产环境替换为真实 API"""

    def __init__(self):
        self.call_count = 0

    def react(self, state: ResearchState) -> dict:
        """模拟一次 ReAct 推理，返回 thought + action 或 final_answer"""
        self.call_count += 1
        step = state["steps"]
        results = state["search_results"]

        # 第一步：决定搜索方向
        if step == 0:
            return {
                "thought": f"收到研究任务：「{state['task']}」。我需要先搜索相关概念和背景知识。",
                "action": {"tool": "search", "args": {"query": state["task"]}},
                "is_final": False,
            }

        # 第二步：基于已有结果决定是否需要更多搜索
        if step == 1 and len(results) < 5:
            # 从已有结果中提取关键词，扩展搜索
            keywords = list(set(word for r in results for word in r["content"][:100].split()[:3]))
            return {
                "thought": f"第一轮搜索得到 {len(results)} 条结果。信息还不够全面，我需要补充搜索相关概念。",
                "action": {"tool": "search", "args": {"query": " ".join(keywords)}},
                "is_final": False,
            }

        # 第三步及以后：整合信息生成报告
        return {
            "thought": f"已收集 {len(results)} 条相关信息，覆盖面足够。我将整合这些资料生成研究报告。",
            "action": None,
            "is_final": True,
        }


def run_phase_1(task: str = "LangGraph Agent 设计模式") -> str:
    """Phase 1: 基础 Research Agent —— ReAct 搜索 + 自动报告生成"""

    print("\n" + "=" * 70)
    print("Phase 1: 基础 Research Agent（ReAct 循环）")
    print("=" * 70)
    print(f"研究任务: {task}\n")

    llm = LLMSimulator()
    state: ResearchState = {
        "task": task,
        "history": [],
        "search_results": [],
        "report": "",
        "steps": 0,
        "status": "thinking",
    }

    MAX_STEPS = 10

    while state["steps"] < MAX_STEPS:
        state["status"] = "thinking"

        # ReAct 推理
        result = llm.react(state)
        print(f"[Step {state['steps']}] Thought: {result['thought']}")

        if result["is_final"]:
            break

        # 执行工具
        action = result["action"]
        if action:
            state["status"] = "acting"
            tool_name = action["tool"]
            tool_args = action["args"]

            if tool_name == "search":
                print(f"  → Action: search(\"{tool_args['query']}\")")
                search_results = simulate_search(tool_args["query"])
                print(f"  → Observation: 找到 {len(search_results)} 条结果")
                state["search_results"].extend(search_results)

        state["steps"] += 1
        state["history"].append({
            "step": state["steps"],
            "thought": result["thought"],
            "action": str(action) if action else None,
            "results_count": len(state["search_results"]),
        })

    # 生成报告
    print(f"\n--- 生成研究报告 ---")
    report = generate_report(task, state)
    print(report)

    filename = "/tmp/research_report_phase1.md"
    simulate_write_report(report, filename)
    print(f"\n✅ Phase 1 完成: {filename}")

    return report


def generate_report(topic: str, state: ResearchState) -> str:
    """基于搜索结果自动生成报告"""
    results = state["search_results"]
    # 去重（按标题）
    seen = set()
    unique_results = []
    for r in results:
        if r["title"] not in seen:
            seen.add(r["title"])
            unique_results.append(r)

    sections = []
    for i, r in enumerate(unique_results, 1):
        sections.append(f"### {i}. {r['title']}\n\n{r['content']}\n")

    report = f"""# 研究报告: {topic}

> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 搜索步数: {state['steps']} 步 | 信息来源: {len(unique_results)} 条

---

## 核心发现

{chr(10).join(sections)}

---

## 结论

本研究基于 {len(unique_results)} 条信息源，覆盖了「{topic}」的主要方面。详细信息请参考上述各节。

*本报告由 Research Agent 自动生成*
"""
    return report


# ──────────────────────────────────────────────────────────────
# Phase 2: + 记忆系统
# ──────────────────────────────────────────────────────────────


@dataclass
class MemorySystem:
    """记忆系统 —— 短期 + 长期"""

    # 短期记忆：当前对话的完整历史
    short_term: list[dict] = field(default_factory=list)

    # 长期记忆：跨会话保存的知识（key → 摘要）
    long_term: dict[str, str] = field(default_factory=dict)

    # 记忆索引：主题 → 关键词列表（便于检索）
    index: dict[str, list[str]] = field(default_factory=dict)

    def remember(self, topic: str, summary: str):
        """存入长期记忆"""
        self.long_term[topic] = summary
        # 简单分词建立索引
        self.index[topic] = [w for w in topic.lower().split() if len(w) > 1]

    def recall(self, query: str) -> list[dict]:
        """从长期记忆检索相关知识"""
        query_words = set(w for w in query.lower().split() if len(w) > 1)
        matches = []

        for topic, summary in self.long_term.items():
            index_words = set(self.index.get(topic, []))
            overlap = query_words & index_words
            if overlap:
                matches.append({
                    "topic": topic,
                    "summary": summary,
                    "relevance": len(overlap) / len(query_words) if query_words else 0,
                })

        matches.sort(key=lambda x: x["relevance"], reverse=True)
        return matches

    def summarize_context(self, max_tokens: int = 500) -> str:
        """压缩短期记忆为摘要，避免 token 爆炸"""
        if len(self.short_term) <= 3:
            return "\n".join(str(m) for m in self.short_term)

        # 保留最近 2 条完整，更早的做摘要
        recent = self.short_term[-2:]
        older = self.short_term[:-2]
        older_summary = f"[前 {len(older)} 步摘要] " + " → ".join(
            m.get("thought", "")[:50] for m in older[-5:]
        )
        return older_summary + "\n" + "\n".join(str(m) for m in recent)


def run_phase_2(task: str = "Agent 记忆系统设计") -> str:
    """Phase 2: + 记忆系统 —— 跨会话知识保留"""

    print("\n" + "=" * 70)
    print("Phase 2: Research Agent + 记忆系统")
    print("=" * 70)

    memory = MemorySystem()

    # 模拟研究多个相关主题，积累长期记忆
    topics = [task, "Reflection 反思机制", "Multi-Agent 协作"]
    all_reports = []

    for i, topic in enumerate(topics):
        print(f"\n--- 子任务 {i+1}: {topic} ---")

        # 先检查长期记忆
        memories = memory.recall(topic)
        if memories:
            print(f"[Memory] 从长期记忆找到 {len(memories)} 条相关记录:")
            for m in memories[:2]:
                print(f"  - {m['topic']}: {m['summary'][:80]}...")

        # ReAct 搜索
        results = simulate_search(topic, top_k=3)
        print(f"[Search] 找到 {len(results)} 条结果")

        # 生成子报告
        seen = set()
        unique = []
        for r in results:
            if r["title"] not in seen:
                seen.add(r["title"])
                unique.append(r)

        sub_report = f"## {topic}\n\n" + "\n\n".join(
            f"**{r['title']}**: {r['content']}" for r in unique
        )
        all_reports.append(sub_report)

        # 存入长期记忆
        summary = f"研究了「{topic}」，获得 {len(unique)} 条信息。" + \
                  " 关键发现: " + (unique[0]["content"][:100] if unique else "无")
        memory.remember(topic, summary)
        memory.short_term.append({
            "topic": topic, "results": len(unique), "time": datetime.now().isoformat()
        })
        print(f"[Memory] 已存入长期记忆: {topic}")

        time.sleep(0.3)  # 模拟思考间隔

    # 整合所有子报告
    full_report = "# 综合研究报告: " + task + "\n\n" + "\n\n---\n\n".join(all_reports)
    full_report += "\n\n---\n\n## 记忆状态\n\n"
    full_report += f"- 短期记忆: {len(memory.short_term)} 条对话记录\n"
    full_report += f"- 长期记忆: {len(memory.long_term)} 个研究主题\n"
    full_report += f"- 记忆主题: {list(memory.long_term.keys())}\n"

    print(f"\n--- 整合报告 ---")
    print(full_report[:500] + "...")

    filename = "/tmp/research_report_phase2.md"
    simulate_write_report(full_report, filename)
    print(f"\n✅ Phase 2 完成: {filename}")

    return full_report


# ──────────────────────────────────────────────────────────────
# Phase 3: + 反思机制
# ──────────────────────────────────────────────────────────────


class ReflectionSystem:
    """反思系统 —— 自审 + 自动修改"""

    def __init__(self, max_rounds: int = 3, pass_threshold: int = 7):
        self.max_rounds = max_rounds
        self.pass_threshold = pass_threshold

    def review(self, report: str, task: str) -> dict:
        """审视报告质量，返回评分和修改建议"""

        issues = []
        score = 10

        # 检查 1: 事实引用（是否有具体信息来源）
        if "来源" not in report and "来自" not in report and "搜索" not in report:
            issues.append("缺少信息来源标注，所有结论应有出处")
            score -= 3

        # 检查 2: 结构完整性（是否有标题、正文、结论）
        if not report.startswith("#"):
            issues.append("缺少主标题，报告结构不完整")
            score -= 2

        if "结论" not in report and "## 总结" not in report:
            issues.append("缺少结论/总结部分")
            score -= 2

        # 检查 3: 内容丰富度
        sections = len([l for l in report.split("\n") if l.startswith("##")])
        if sections < 3:
            issues.append(f"内容不够丰富，只有 {sections} 个小节，建议至少 4 个")
            score -= 2

        # 检查 4: 与任务的相关性
        task_keywords = [w for w in task.lower().split() if len(w) > 1]
        matches = sum(1 for kw in task_keywords if kw in report.lower())
        if matches < len(task_keywords) * 0.5:
            issues.append("报告内容与任务关键词匹配度不足，可能偏离主题")
            score -= 2

        # 检查 5: 长度
        if len(report) < 500:
            issues.append("报告过短（< 500 字符），缺少足够的细节")
            score -= 2

        score = max(1, score)

        return {
            "score": score,
            "passed": score >= self.pass_threshold,
            "issues": issues,
            "suggestions": self._generate_suggestions(issues),
        }

    def _generate_suggestions(self, issues: list[str]) -> list[str]:
        """基于问题生成具体修改建议"""
        suggestions = []
        for issue in issues:
            if "来源" in issue:
                suggestions.append("在每条重要陈述后添加来源标记，如「[搜索结果 1]」")
            elif "标题" in issue or "结构" in issue:
                suggestions.append("添加 # 主标题和 ## 小节标题，形成层次清晰的结构")
            elif "结论" in issue:
                suggestions.append("在报告末尾添加「## 结论」小节，总结核心发现")
            elif "丰富" in issue or "小节" in issue:
                suggestions.append("为每个子主题创建独立的小节，填充搜索结果内容")
            elif "关键词" in issue or "偏离" in issue:
                suggestions.append("检查报告内容，确保每个小节都直接回应研究任务")
            elif "过短" in issue:
                suggestions.append("扩展每个小节，添加更多搜索结果细节和具体数据")
        return suggestions

    def revise(self, report: str, feedback: dict) -> str:
        """基于反馈修改报告"""
        lines = report.split("\n")
        new_lines = []

        for line in lines:
            new_lines.append(line)

        # 添加缺失的结构元素
        if "标题" in str(feedback.get("issues", [])):
            if not any(l.startswith("# ") for l in lines):
                new_lines.insert(0, "# 研究报告")
                new_lines.insert(1, "")

        if "结论" in str(feedback.get("issues", [])):
            new_lines.append("")
            new_lines.append("## 结论")
            new_lines.append("")
            new_lines.append("本研究综合多源信息，对主题进行了系统性分析。")
            new_lines.append("详细修改建议已采纳并整合到各小节中。")

        if "来源" in str(feedback.get("issues", [])):
            new_lines.insert(3, "")
            new_lines.insert(3, "> 信息来源：知识库搜索 | 生成时间：" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        return "\n".join(new_lines)


def run_phase_3(task: str = "Reflection 反思机制设计") -> str:
    """Phase 3: + 反思机制 —— 自审 + 自动修改"""

    print("\n" + "=" * 70)
    print("Phase 3: Research Agent + 反思机制")
    print("=" * 70)
    print(f"研究任务: {task}\n")

    reflector = ReflectionSystem(max_rounds=3, pass_threshold=7)

    # 先用 ReAct 生成初始报告
    results = simulate_search(task, top_k=5)

    # 生成初版报告（故意不完全，让反思机制发挥作用）
    initial_report = f"""## {task}

{results[0]['content']}

{results[1]['content'] if len(results) > 1 else ''}
"""
    print(f"[初始报告] {len(initial_report)} 字符, {initial_report.count(chr(10))} 行")
    print(f"内容预览: {initial_report[:150]}...\n")

    # 反思循环
    current_report = initial_report
    for round_num in range(1, reflector.max_rounds + 1):
        print(f"--- 反思第 {round_num} 轮 ---")

        review = reflector.review(current_report, task)
        print(f"  评分: {review['score']}/10")
        print(f"  问题: {review['issues']}")

        if review["passed"]:
            print(f"  ✅ 通过！评分 >= {reflector.pass_threshold}")
            break

        print(f"  ❌ 不通过。修改建议:")
        for s in review["suggestions"]:
            print(f"    - {s}")

        # 修改报告
        current_report = reflector.revise(current_report, review)
        print(f"  📝 已修改，新长度: {len(current_report)} 字符\n")

        if round_num == reflector.max_rounds:
            print(f"  ⚠️ 达到最大修改轮次 ({reflector.max_rounds})，停止反思")

    # 最终评估
    final_review = reflector.review(current_report, task)
    print(f"\n最终评分: {final_review['score']}/10")
    print(f"\n--- 最终报告 ---")
    print(current_report[:800])

    filename = "/tmp/research_report_phase3.md"
    simulate_write_report(current_report, filename)
    print(f"\n✅ Phase 3 完成: {filename}")

    return current_report


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main():
    phase = None
    if "--phase" in sys.argv:
        try:
            idx = sys.argv.index("--phase")
            phase = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("用法: python agent_demo.py [--phase 1|2|3]")
            return

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Research Agent Demo —— 渐进式 Agent 实现          ║")
    print("║       ReAct → Memory → Reflection                       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if phase is None or phase == 1:
        run_phase_1("Agent 设计模式与最佳实践")

    if phase is None or phase == 2:
        run_phase_2("Agent 架构演进")

    if phase is None or phase == 3:
        run_phase_3("反思机制在 Agent 中的应用")

    print("\n" + "=" * 70)
    print("全部 Demo 完成！")
    print("=" * 70)
    print("\n学习路径建议：")
    print("  1. agent_design.md — 理解 Agent 设计理论")
    print("  2. agent_demo.py — 对照源码理解实现")
    print("  3. 007-langgraph-learning — Agent 的图编排实践")
    print("  4. 008-mcp-learning — Tool Calling 标准化")


if __name__ == "__main__":
    main()
