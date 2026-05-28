# AI Learning Workspace

系统地学习和实践 AI 工程核心技术，所有概念配可运行 Demo。

## 学习路线

```
LLM 基础 → RAG → MCP → LangGraph → Multi-Agent → AI Coding
```

| 主题 | 状态 | 说明 |
|------|------|------|
| RAG | 学习中 | 检索增强生成，最小Demo+生产级Demo+8道习题 |
| MCP | 待开始 | Model Context Protocol，给LLM接工具 |
| LangGraph | 待开始 | 有状态多步Agent编排 |
| Multi-Agent | 待开始 | 多Agent协作模式 |
| AI Coding | 待开始 | AI辅助编程最佳实践 |

## 项目结构

```
ai-learning/
├── rag-learning/          # RAG 检索增强生成
│   ├── demo/              #   最小可运行Demo（200行，行行注释）
│   ├── production/        #   生产级Demo（ChromaDB+混合检索）
│   └── exercises/         #   8道练习题
│
├── mcp-learning/          # MCP（待开始）
├── langgraph-learning/    # LangGraph（待开始）
├── multi-agent-learning/  # Multi-Agent（待开始）
└── ai-coding-learning/    # AI Coding（待开始）
```

## 原则

- 所有概念必须配可运行 Demo
- 代码必须有详细行级注释
- 每个子项目包含 README + 练习题
- 优先使用 Python
- 踩坑必记录
- 每个模块独立学习记录

## 学习记录

[查看所有学习记录](docs/learning-log.md)，每个模块独立存放。

## 快速开始

```bash
# 当前已完成的模块
cd rag-learning/demo
cp .env.example .env        # 填入 DeepSeek 或 OpenAI API Key
pip install -r requirements.txt
python3 minimal_rag.py
```

## 环境

- Python 3.14
- macOS
- DeepSeek / OpenAI API
