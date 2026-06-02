# 🧠 RAG 检索增强生成 —— 企业知识库问答系统

> 从零构建本地 PDF 问答系统，5 步渐进演进。
> 不讲数学公式，只讲代码和直觉。

---

## 项目结构

```
005-rag-learning/
├── README.md              ← 你在这里
├── architecture.md        ← 完整架构文档（为什么需要 RAG → 每个组件怎么做）
├── tasks.md               ← 5 步任务清单（可打勾跟踪学习进度）
├── learning_notes.md      ← 教学笔记（概念讲解，配合 Demo 阅读）
├── index.html             ← 交互学习网页
└── demo/
    ├── step1_basic.py         ← Step 1: TF-IDF + 单 PDF 基础版
    ├── step2_vector_db.py     ← Step 2: ChromaDB 向量数据库
    ├── step3_multi_pdf.py     ← Step 3: 批量 PDF + 自动分块
    ├── step4_chat_memory.py   ← Step 4: 多轮对话记忆
    ├── step5_optimize_recall.py ← Step 5: 混合检索 + 重排序
    ├── requirements.txt       ← Python 依赖
    └── data/                  ← 测试文件放这里（PDF/txt/md）
```

---

## 快速开始

### 1. 环境搭建

```bash
cd 005-rag-learning/demo
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

推荐用 [DeepSeek](https://platform.deepseek.com)（便宜，中文好）。

### 3. 放 PDF 到 data/ 目录

```bash
mkdir -p data
# 把你的 PDF 文件放进去
```

### 4. 按顺序运行

```bash
python step1_basic.py        # 理解 RAG 核心链路
python step2_vector_db.py    # 向量数据库替代 TF-IDF
python step3_multi_pdf.py    # 支持多个 PDF
python step4_chat_memory.py  # 多轮对话
python step5_optimize_recall.py # 优化召回率
```

---

## 学习路线

| 步骤 | 核心概念 | 检索方式 | 新增能力 |
|------|---------|---------|---------|
| Step 1 | RAG 三阶段 | TF-IDF 关键词 | 基础问答 |
| Step 2 | Embedding + 向量DB | ChromaDB 语义 | 语义搜索 |
| Step 3 | Chunking 分块 | ChromaDB | 多 PDF + 来源追溯 |
| Step 4 | Context Window | ChromaDB | 多轮对话记忆 |
| Step 5 | 混合检索+重排序 | 向量+BM25+CrossEncoder | 高召回率 |

---

## 系统架构

```
PDF → 分块 → Embedding → ChromaDB（离线索引）
                              ↓
用户问题 → 混合检索 → 重排序 → Prompt 增强 → LLM 生成（在线查询）
```

详见 `architecture.md`。

---

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| PDF 解析 | PyMuPDF | 中文提取最稳定 |
| Embedding | bert-base-chinese | 本地缓存，离线可用 |
| 向量数据库 | ChromaDB | 轻量，零配置 |
| LLM | DeepSeek Chat | 便宜，中文好 |
| 重排序 | Cross-Encoder | 精排效果远好于双塔 |

---

## 前置知识

学习本模块前，建议先完成：
- `001-llm-learning` — 理解 LLM 基本原理
- `002-prompt-learning` — 理解 Prompt 设计
- `003-embedding-learning` — 理解 Embedding 和语义搜索
- `004-vector-db-learning` — 理解向量数据库
