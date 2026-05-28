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

### 练习1：扩展知识库

> 详见 [exercises/exercise-01.md](../exercises/exercise-01.md)

新增了 3 篇文档（vibe coding工具、AI中的养龙虾、大数据实时处理工具），遇到一个新问题。

**Q: "vibe coding工具有哪些？"没有答案返回，我明确添加了资料。**

→ 直接复现验证：

```
📝 用户问题：vibe coding工具有哪些？
🔍 检索结果（共 0 条相关文档）  ← 果然检索不到
🤖 AI 回答：抱歉，知识库中没有找到相关信息。
```

### 踩坑：标题关键词 vs 内容索引

**排查过程**：

第一步——看文档内容和查询的 jieba 分词结果：

```
文档标题: vibe coding工具
文档内容: 现在使用最强大的是Claude code公认最好用的codex以及opencode
查询:     vibe coding工具有哪些？

查询分词: ['vibe', ' ', 'coding', '工具', '有', '哪些', '？']
文档分词: ['现在', '使用', '最', '强大', '的', '是', 'Claude', ' ', 'code', ...]
```

发现问题线索——"vibe coding"这两个词在查询里，但文档内容里完全没有。

第二步——进一步检查 TF-IDF 实际生效的 token（TF-IDF 默认只保留 2+ 字母数字字符，空格会被过滤掉）：

```
查询有效词: {'vibe', '工具', '哪些', 'coding'}
文档有效词: {'code', '使用', 'Claude', '最好', '现在', '以及', '强大', '公认', 'opencode', 'codex'}
共同有效词: set()  ← 空！一个共同词都没有 → 相似度 = 0
```

**根因**：TF-IDF 只索引 `content` 字段，完全忽略 `title`。"vibe coding" 这两个关键词只在标题出现，正文内容说的是 "Claude code"、"codex"、"opencode"——字面上完全不同。共同有效 token 为 0，余弦相似度直接为 0。

**修复**：将标题和内容拼接后一起索引。

```python
# 之前：只索引 content
self.contents = [doc["content"] for doc in documents]

# 之后：title + content 联合索引
full_texts = [f"{doc['title']} {doc['content']}" for doc in documents]
```

---

**追问: 之前没改索引时，"实时数据处理使用的什么工具？"却能出结果，为什么？**

→ 验证两个查询的 token 差异：

```
=== vibe coding ===
查询有效词: {'vibe', '工具', '哪些', 'coding'}
文档有效词: {'code', '使用', 'Claude', '最好', '现在', '以及', '强大', '公认', 'opencode', 'codex'}
共同有效词: set()  ← 空集，相似度 = 0

=== 实时数据处理 ===
查询有效词: {'实时', '数据处理', '使用', '什么', '工具'}
文档有效词: {'flink', '不少', '多用', '企业', '使用', 'spark'}
共同有效词: {'使用'}  ← 有一个共同词！相似度 > 0
```

"实时数据处理"的文档内容碰巧包含"使用"这个词，和查询里的"使用"对上了。虽然只有一个词、相关度只有 0.36，但足以让 TF-IDF 算出非零分数。而 "vibe coding" 文档内容和查询之间一个字面重叠都没有，分数严格为 0。

**深层启发**：
- TF-IDF 是纯字面匹配，不理解"vibe coding"和"Claude code"语义相关
- 检索到与否取决于有没有共同词——哪怕一个也行
- 工业级用 Embedding 模型做语义搜索，不依赖精确字面匹配
- RAG 检索字段设计很重要：title、summary、content 各自权重不同

---

**Q: git管理整个项目 / 提交到GitHub / 创建README / 记录学习过程 / 提问加进来 / 模块单独存放**

→ 一系列工程化操作：Git 管理全项目，推送到 [BakerYoung/cc-ai-learning](https://github.com/BakerYoung/cc-ai-learning)，创建根 README，建立对话式学习记录体系。

---

### 练习2：top_k 的影响

> 详见 [exercises/exercise-02.md](../exercises/exercise-02.md)

**核心发现**：top_k 在单文档问题上"看不出效果"，但在跨文档问题上非常关键。

构造了一个跨文档问题："公司的技术栈和产品价格分别是什么？"——技术栈在"技术架构"（排第3名），价格在"产品介绍"（排第5名）。

| top_k | 技术栈 | 价格 | 说明 |
|-------|--------|------|------|
| 2 | ❌ | ❌ | 前2名全是不相关文档 |
| 3 | ✅ | ❌ | 拿到技术栈，但价格文档排第5，漏了 |
| 5 | ✅ | ✅ | 两篇都拿到 |

同时发现 TF-IDF 的排名错误——"大数据实时处理工具"排第1，因为它和查询有字面重叠"技术""使用""工具"，但内容其实不相关。这是 TF-IDF 字面匹配的又一缺陷。

**top_k 选择原则**：简单查询 1-3，跨文档查询 5-10，本质是召回率 vs 精确率的权衡。

---

### 练习3：替换相似度算法

> 详见 [exercises/exercise-03.md](../exercises/exercise-03.md)

把余弦相似度替换成欧氏距离后，所有答案消失。

**根因**：余弦相似度是**越大越好**，代码 `argsort()[::-1]` 按降序取 top_k。欧氏距离是**越小越好**，同样是降序就变成"取最不相关的文档"。LLM 拿到全是不相关的文档，正确回复"知识库中暂无相关信息"。

**关键收获**：换检索算法必须同步改排序方向。余弦相似度在 RAG 里是标配——向量方向代表语义，长度代表信息密度，余弦只看方向天然适合语义比较。

---

### 练习4：多轮对话

> 详见 [exercises/exercise-04.md](../exercises/exercise-04.md)

给 RAG 加上对话历史记忆，支持追问。改了两处：`generate_answer()` 增加 `history` 参数把历史插入 messages，`RAGPipeline` 维护 `self.history` 列表并追加每轮问答。

演示效果：问"AI Tutor月费"→答29元 → 追问"年费"→答299元 → 追问"比按月省多少"→结合历史算出348-299=49元。

同时修了一个残留 bug：练习3改的欧氏距离没改回来，导致所有检索分数异常(1.4142)，改回余弦相似度后恢复正常。

### 基础练习完成小结

四个练习踩了四个坑，每个都揭示了一个 RAG 的核心问题：

| 练习 | 修复的问题 | 核心教训 |
|------|-----------|---------|
| 1. 扩展知识库 | 标题关键词检索不到 | 检索字段设计很重要 |
| 2. 调 top_k | 单文档问题看不出效果 | 跨文档问题才体现召回率价值 |
| 3. 换相似度 | 欧氏距离排序反向 | 算法方向必须和排序逻辑一致 |
| 4. 加对话历史 | previous_answer 没定义 | 历史要贯穿 pipeline 全链路 |

---

### 练习5：替换为真实文件

> 详见 [exercises/exercise-05.md](../exercises/exercise-05.md)

知识库从硬编码改为读取 `data/` 目录下的 `.txt` / `.md` 文件。新增 `load_documents_from_dir()` 函数：解析文件首行作为 title，其余作为 content，输出格式和之前 `DOCUMENTS` 完全一致，`RAGPipeline` 一行不改。

**关键收获**：数据加载和 RAG 逻辑解耦后，切换数据源只需换 loader，Pipeline 不感知。这是生产级 RAG 的第一个扩展点——从 `load_documents_from_dir()` 进化到 `DocumentLoader → Chunker → VectorStore`。

### 下一步

进阶练习 6（BM25 混合检索）或继续 MCP 学习
