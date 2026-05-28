# 学习记录

## 2026-05-28 · RAG 入门

### 学到的概念

**RAG（Retrieval-Augmented Generation）= 开卷考试**

LLM 是闭卷考试——只能靠训练记忆回答，知识有截止日期，容易编造。RAG 是开卷考试——先从知识库检索相关文档，再把文档拼进 Prompt 让 LLM 参考着回答。

核心三步：**检索（Retrieve）→ 增强（Augment）→ 生成（Generate）**

深入理解了 5 个关键概念：

| 概念 | 一句话理解 |
|------|-----------|
| Embedding / 向量 | 把文本变成数字数组，语义相近的词向量也相近 |
| 向量数据库 | 专门存向量+做相似度搜索的数据库（ChromaDB/Milvus/Pinecone） |
| 文档分块 | 长文档必须切碎再检索，否则精度很差 |
| 混合检索 | 向量检索（语义）+ 关键词检索（精确），RRF 融合取长补短 |
| RRF | 融合算法：`score = 1/(60+向量排名) + 1/(60+关键词排名)` |

还搞清楚了 RAG vs Fine-tuning 的区别：RAG 给 LLM 外部参考资料（便宜、实时更新），Fine-tuning 改变 LLM 内部参数（适合教风格/格式）。

### 构建的代码

**`rag-learning/demo/minimal_rag.py`**（~200行）
- 检索器：TF-IDF + jieba 中文分词 + 余弦相似度
- 生成器：DeepSeek/OpenAI API
- 知识库：7 篇模拟公司内部文档
- 每行代码都有详细注释

**`rag-learning/production/advanced_rag.py`**（~280行）
- 检索器：sentence-transformers (all-MiniLM-L6-v2) + ChromaDB
- 分块器：语义分块（500字/块，50字重叠）
- 检索策略：向量检索 + 关键词检索 + RRF 融合
- 对比最小Demo展示了工业级的完整架构

**其他**
- 8 道练习题（基础→进阶→实战）
- README 含概念拆解、架构图、FAQ

### 踩坑记录

**问题**：首次运行 `minimal_rag.py`，5 个中文问题中 4 个检索到 0 条文档。

**原因**：TF-IDF 默认按空格切词。中文没有空格，每个汉字被当成一个独立的"词"。"公司混合办公的政策" → `['公','司','混','合','办','公',...]`，和文档几乎没有共同字符，相似度 ≈ 0。唯一命中的问题是包含英文 "AI Tutor" 的查询——英文有空格分词，撞大运匹配成功。

**修复**：引入 `jieba` 中文分词，在 TF-IDF 向量化之前把中文文本切成词语并用空格连接。"公司成立于2024年" → `"公司 成立 于 2024 年"`，TF-IDF 就能正确按空格切词了。

**启发**：
- TF-IDF 对不同语言适配完全不同，中文必须先分词
- 工业级直接用中文 Embedding 模型（BAAI/bge-large-zh-v1.5）一步到位
- RAG 出问题时先检查 tokenization 层，不要上来就调参数

### 工程化

- Git 初始化，3 次提交
- `.gitignore` 排除 `.env`（含 API Key）、`chroma_db/`、`__pycache__/`
- 推送到 GitHub: [BakerYoung/cc-ai-learning](https://github.com/BakerYoung/cc-ai-learning)

### 下一步

完成 RAG 练习 1-4（扩展知识库、调 top_k、换相似度算法、加对话历史）
