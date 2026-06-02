# RAG 学习任务清单

> 从零构建本地 PDF 问答系统，5 步渐进演进。每步独立可运行。

---

## 前置准备

- [x] Python 3.12 环境
- [x] DeepSeek API Key（或 OpenAI 兼容的 API）
- [x] 准备 1~5 个测试 PDF 文件放在 `demo/data/` 下

---

## Step 1：基础版本 —— 理解 RAG 核心链路

**目标：** 用最小代码跑通 RAG 全流程。

- [ ] 用 PyMuPDF 读取单个 PDF 文本
- [ ] 用 jieba + TF-IDF 构建检索器
- [ ] 用 DeepSeek API 构建生成器
- [ ] 拼接 Prompt：检索结果 + 用户问题 → LLM
- [ ] 能回答"这份 PDF 讲了什么"类型的问题

**核心概念：** RAG = Retrieval + Augmentation + Generation
**文件：** `demo/step1_basic.py`
**无外部依赖：** 不需要向量数据库、不需要 Embedding 模型

---

## Step 2：向量数据库 —— 从 TF-IDF 到语义检索

**目标：** 用 ChromaDB 替换 TF-IDF，实现真正的语义搜索。

- [ ] 安装 ChromaDB + sentence-transformers
- [ ] 用 bert-base-chinese 做 Embedding
- [ ] 创建 ChromaDB Collection 存储文档向量
- [ ] 用语义搜索替换 TF-IDF 关键词搜索
- [ ] 对比 TF-IDF vs Embedding 的检索质量差异

**核心概念：** Embedding、向量数据库、语义搜索 vs 关键词搜索
**文件：** `demo/step2_vector_db.py`

---

## Step 3：多 PDF 支持 —— 批量文档知识库

**目标：** 支持导入 `data/` 目录下的所有 PDF，自动分块、自动索引。

- [ ] 扫描 `data/` 目录下所有 PDF
- [ ] 实现递归文本分块器（500 字/块，50 字重叠）
- [ ] 批量导入 ChromaDB（建索引时显示进度）
- [ ] 用户可以选择查询所有文档或指定文档
- [ ] 检索结果标注来源（哪个 PDF 的第几页）

**核心概念：** Chunking、增量索引、来源追溯
**文件：** `demo/step3_multi_pdf.py`

---

## Step 4：聊天记忆 —— 多轮对话上下文

**目标：** 支持追问、指代消解、上下文理解。

- [ ] 维护对话历史列表
- [ ] 每轮将历史消息传入 LLM
- [ ] 支持"那价格呢？"这种省略主语的追问
- [ ] 限制历史长度（最近 10 轮），避免超出 Context Window
- [ ] 实现 `/clear` 命令清空对话历史

**核心概念：** Context Window、多轮对话、Token 预算管理
**文件：** `demo/step4_chat_memory.py`

---

## Step 5：优化召回率 —— 混合检索 + 重排序

**目标：** 把检索质量从"能用"提升到"好用"。

- [ ] 实现 BM25 关键词检索（rank-bm25 库）
- [ ] 向量检索 + BM25 检索 → 结果融合（RRF 算法）
- [ ] 实现 Cross-Encoder 重排序（对 Top-20 重新精排）
- [ ] Query 改写：用户问题太短时自动扩展
- [ ] 定量评估：对比 Step2→Step5 的召回率提升

**核心概念：** 混合检索（Hybrid Search）、RRF 融合、重排序（Re-ranking）
**文件：** `demo/step5_optimize_recall.py`

---

## 最终输出

- [ ] `architecture.md` — 完整架构文档
- [ ] `tasks.md` — 本任务清单
- [ ] `learning_notes.md` — 教学笔记
- [ ] `index.html` — 交互学习网页
- [ ] `demo/` — 5 个渐进式 Demo，全部可运行
- [ ] `demo/data/` — 测试 PDF 文件

---

## 进阶方向（学完 5 步之后）

- [ ] 使用 BAAI/bge-large-zh-v1.5 替换 bert-base-chinese（召回率再提升 10-15%）
- [ ] 接入 Web UI（Streamlit/Gradio）
- [ ] 评估体系：用 RAGAS 框架量化评估检索和生成质量
- [ ] 多模态 RAG：PDF 中的图片/表格
