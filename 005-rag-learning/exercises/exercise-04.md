# 练习4：增加对话历史 —— 多轮 RAG

## 目标

让 RAG 支持多轮对话：追问时不需重复完整上下文，LLM 能结合历史理解用户意图。

## 架构改动

改了两处，形成一个完整的数据流：

```
RAGPipeline.ask(query)
  │
  ├─ 检索阶段：self.retriever.retrieve(query) → 相关文档
  │
  ├─ 生成阶段：generate_answer(query, docs, history=self.history)
  │              │
  │              └─ messages = [system] + history + [current_query + docs]
  │
  └─ 更新历史：self.history.append(本轮 user + assistant)
```

### 改动1：`generate_answer()` 增加 `history` 参数

```python
# 修改前
def generate_answer(query: str, retrieved_docs: list[dict]) -> str:

# 修改后
def generate_answer(query: str, retrieved_docs: list[dict],
                    history: list[dict] = None) -> str:
```

messages 的组装逻辑：

```python
# 修改前：只有 system + 当前问题
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# 修改后：system + 历史对话 + 当前问题
messages = [{"role": "system", "content": system_prompt}]
messages.extend(history)                                    # 插入历史
messages.append({"role": "user", "content": current_message})  # 当前问题
```

`history` 格式：`[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]`，就是 OpenAI messages 格式。

### 改动2：`RAGPipeline` 维护历史

```python
class RAGPipeline:
    def __init__(self, documents):
        self.retriever = SimpleRetriever(documents)
        self.history = []  # 新增：对话历史

    def ask(self, query, top_k=3, verbose=True):
        docs = self.retriever.retrieve(query, top_k=top_k)
        answer = generate_answer(query, docs, history=self.history)  # 传入历史

        # 本轮问答追加到历史
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self):  # 新增：重置对话
        self.history = []
```

## 测试结果

### 多轮对话演示

```
轮1: AI Tutor产品月费多少钱？
  → 检索到 "产品介绍" (0.4922)
  → 回答: 月费29元

轮2: 年费价格是多少？  ← 没再提"AI Tutor"，从历史理解上下文
  → 检索到 "产品介绍" (0.1931)
  → 回答: 年费299元

轮3: 如果我按年订阅，比按月省多少？  ← 需要结合轮1的29元+轮2的299元
  → 检索到 "产品介绍" (0.1931)
  → 回答: 按月年总费用29×12=348元，年费299元，节省49元
```

轮3 是关键——LLM 正确使用了轮 1 的历史回答（月费 29 元）来计算，而月费 29 元在轮 3 的检索文档里虽然也有，但结合历史让推理更自然。

## 踩坑：用户最初的错误用法

用户尝试在 messages 里直接加：

```python
{"role": "assistant", "content": previous_answer}  # previous_answer 未定义！
```

问题：
1. `previous_answer` 没有作为参数传入 `generate_answer()`
2. 即使传入了，只能记住"上一轮回答"，不支持多轮
3. `RAGPipeline` 没有维护和传递历史的机制

## 关键设计决策

### 为什么不给每轮都做检索？

追问"那年费呢？"如果单独检索，TF-IDF 只能匹配到一个无关文档。但多轮 RAG 的流程是**每轮都检索 + 历史对话**——检索提供"当前轮的知识"，历史提供"上下文"。两者互补。

### 实际上，轮2的检索也有用

```
轮1检索: "AI Tutor产品月费多少钱？" → 产品介绍 (0.4922)
轮2检索: "年费价格是多少？" → 产品介绍 (0.1931)  ← 相关度虽然低了，但还能命中
```

如果轮2是"那年费呢？"（只有4个字），TF-IDF 检索就完全失效——共同词为0。这就是练习2/3反复证明的 TF-IDF 字面匹配限制。

### 什么时候清空历史？

- 用户切换话题（"换个问题，Python的map函数是什么"）→ 应该清空
- 多轮对话结束 → 应该清空
- 实际产品里可以用"意图识别"或让 LLM 判断是否需要重置

## 知识收获

- `messages` 就是对话记录，API 不区分"历史"和"当前"——都是按顺序排列的 role+content
- 多轮 RAG = 每轮检索 + 历史对话一起给 LLM
- 追问短依赖历史长——TF-IDF 短查询检索不到时，LLM 可以从历史里"猜"出上下文
