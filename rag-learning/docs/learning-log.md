# RAG 学习记录

## 2026-05-28

### 学习过程

**Q: 今天我要学习RAG。请用最容易理解的方式解释RAG，给我一个最小可运行Demo，一步一步带我运行，解释每行代码，扩展为生产级架构，最后给我练习题。**

→ RAG = 开卷考试。LLM 是闭卷考试（只能靠记忆），RAG 是开卷考试（先翻书再作答）。核心三步：检索 → 增强 → 生成。

构建了完整项目：

| 文件 | 内容 |
|------|------|
| `rag-learning/README.md` | 概念拆解、架构图、FAQ、学习路径 |
| `rag-learning/demo/minimal_rag.py` | 最小Demo（TF-IDF + 余弦相似度 + LLM），~200行，行行注释 |
| `rag-learning/production/advanced_rag.py` | 生产级Demo（ChromaDB + Embedding + 混合检索 + RRF），~280行 |
| `rag-learning/exercises/exercises.md` | 8道练习题（基础→进阶→实战） |

深入理解了 5 个关键概念：

| 概念 | 一句话理解 |
|------|-----------|
| Embedding / 向量 | 把文本变成数字数组，语义相近的词向量也相近 |
| 向量数据库 | 专门存向量+做相似度搜索的数据库（ChromaDB/Milvus/Pinecone） |
| 文档分块 | 长文档必须切碎再检索，否则精度很差 |
| 混合检索 | 向量检索（语义）+ 关键词检索（精确），RRF 融合取长补短 |
| RRF | 融合算法：`score = 1/(60+向量排名) + 1/(60+关键词排名)` |

还搞清楚了 RAG vs Fine-tuning 的区别：RAG 给 LLM 外部参考资料（便宜、实时更新），Fine-tuning 改变 LLM 内部参数（适合教风格/格式）。

---

**Q: 我配置了.env，然后运行python minimal_rag.py，但运行的并不是理想结果。**

→ 诊断发现两个问题：① macOS 上要用 `python3` 而非 `python`；② TF-IDF 对中文失效。

### 踩坑：中文分词问题

**现象**：5 个中文问题中 4 个检索到 0 条文档。唯一命中的是含英文 "AI Tutor" 的查询。

**根因**：TF-IDF 默认按空格切词。中文没有空格，每个汉字被当成一个独立的"词"。
```
"公司混合办公的政策" → ['公','司','混','合','办','公',...]
→ 和文档几乎没有共同字符 → 相似度 ≈ 0
```

**修复**：引入 `jieba` 中文分词，TF-IDF 之前先把中文文本切成词语：
```
"公司成立于2024年" → jieba → "公司 成立 于 2024 年" → TF-IDF 正确切词
```

**启发**：
- TF-IDF 对不同语言适配完全不同，中文必须先分词
- 工业级直接用中文 Embedding 模型（BAAI/bge-large-zh-v1.5）一步到位
- RAG 出问题时先检查 tokenization 层，不要上来就调参数

---

**Q: 你把这些改动也说明一下，然后我要引入git，帮我管理项目。**

→ 将踩坑记录补充到 README，修复运行命令 `python`→`python3`。Git 初始化，`.gitignore` 排除 `.env`（含 API Key）、`chroma_db/`、`__pycache__/`。首次提交 11 个文件。

---

**Q: git管理整个项目 / 提交到GitHub / 创建README / 记录学习过程 / 提问加进来 / 模块单独存放**

→ 一系列工程化操作：Git 管理全项目，推送到 [BakerYoung/cc-ai-learning](https://github.com/BakerYoung/cc-ai-learning)，创建根 README，建立对话式学习记录体系。

### 下一步

完成 RAG 练习 1-4（扩展知识库、调 top_k、换相似度算法、加对话历史）
