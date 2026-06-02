# 练习5：替换为真实文档 —— 从文件构建知识库

## 改动内容

之前所有文档硬编码在 `DOCUMENTS` 列表里。现在改为从 `data/` 目录读取 `.txt` / `.md` 文件。

### 新增文件

```
demo/
├── data/                    # 文档目录（模拟真实知识库）
│   ├── 公司介绍.md
│   ├── 产品介绍.txt
│   ├── 技术架构.txt
│   ├── 融资情况.txt
│   └── 办公政策.txt
└── file_rag.py              # 文件版 RAG Demo
```

### 核心代码：`load_documents_from_dir()`

```python
def load_documents_from_dir(data_dir: str = "data") -> list[dict]:
    """
    从目录中读取所有 .txt / .md 文件，每篇解析为 {title, content}。
    解析规则：第一行去掉 # 作为 title，其余作为 content。
    """
    base = Path(data_dir)
    files = sorted(base.glob("*.txt")) + sorted(base.glob("*.md"))

    documents = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        lines = raw.split("\n", 1)
        first_line = lines[0].strip()

        if first_line.startswith("# "):
            title = first_line[2:].strip()
        elif first_line.startswith("#"):
            title = first_line[1:].strip()
        else:
            title = filepath.stem  # 无标题时用文件名

        content = lines[1].strip() if len(lines) > 1 else first_line
        documents.append({"title": title, "content": content})

    return documents
```

## 架构变化

```
之前：DOCUMENTS = [{...}, {...}]  ← 代码里写死
       ↓
      RAGPipeline(DOCUMENTS)

之后：data/*.txt → load_documents_from_dir() → [{...}, {...}]
       ↓
      RAGPipeline(documents)          ← 完全相同的接口
```

`RAGPipeline` 不需要任何改动——只要数据格式是 `[{"title": "...", "content": "..."}]` 就行。这就是接口设计的好处。

## 测试结果

```
加载: 产品介绍.txt → 标题='产品介绍' (117字)
加载: 办公政策.txt → 标题='办公政策' (101字)
加载: 技术架构.txt → 标题='技术架构' (140字)
加载: 融资情况.txt → 标题='融资情况' (90字)
加载: 公司介绍.md → 标题='公司介绍' (93字)
✅ 共加载 5 篇文档

Q: 公司混合办公的政策是什么？
  → 办公政策 (0.5293) → 正确回答 ✅

Q: 公司融资了多少钱？投资方有哪些？
  → 融资情况 (0.4861) → 正确回答 ✅

Q: AI Tutor的后端技术栈是什么？
  → 技术架构 (0.2025) → 正确回答 ✅
```

## 设计决策

### 为什么第一行特殊处理？

真实文档的标题通常嵌在内容里（Markdown 的 `#`、文本文件的首行）。把第一行抽出来作为结构化字段，这样做：
- 标题参与索引（练习1的教训）
- 检索结果展示时可以显示标题而非文件名
- 未来可以给标题更高的索引权重

### 为什么支持 .txt 和 .md？

真实知识库文档格式多样：Markdown 笔记、纯文本手册、甚至 HTML/PDF。这里先支持两种最常见的。扩展只需加 `glob("*.pdf")` + 对应的解析逻辑。

### 错误处理

```python
if not base.exists():
    return _default_documents()  # 目录不存在 → 用备用知识库
if not files:
    return _default_documents()  # 目录为空 → 用备用知识库
if not raw:
    continue                     # 空文件 → 跳过
```

降级策略：文件不可用时回退到硬编码的默认文档，而不是崩溃。

## 从练习到生产

这是 RAG 系统的第一个关键扩展点。真实项目里 `load_documents_from_dir()` 会进化成：

```
data/*.txt ──┐
data/*.pdf ──┼──→ DocumentLoader ──→ Chunker ──→ VectorStore
data/*.md  ──┤        ↑
API/爬虫 ────┘   支持更多格式
```

这就是 `production/advanced_rag.py` 里 `SemanticChunker` + `VectorStore` 的定位。
