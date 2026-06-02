# LangChain 学习笔记：从手写 Pipeline 到声明式 AI 应用

> 从"手写 RAG 太累了"这个直觉问题开始。
> 不讲 API 文档，只讲为什么和怎么做。

---

## 目录

1. [起点问题：手写 RAG 到底累在哪？](#1-起点问题手写-rag-到底累在哪)
2. [LangChain 的核心哲学：一切皆 Runnable](#2-langchain-的核心哲学一切皆-runnable)
3. [LLM 抽象：ChatOpenAI 不是黑魔法](#3-llm-抽象chatopenai-不是黑魔法)
4. [PromptTemplate：模板和变量分离](#4-prompttemplate模板和变量分离)
5. [OutputParser：让 LLM 输出结构化数据](#5-outputparser让-llm-输出结构化数据)
6. [Tool：让 LLM 调用外部函数](#6-tool让-llm-调用外部函数)
7. [Retriever：检索的统一抽象层](#7-retriever检索的统一抽象层)
8. [Chain：声明式的数据流管道](#8-chain声明式的数据流管道)
9. [Runnable 协议：LCEL 的核心](#9-runnable-协议lcel-的核心)
10. [实战：用 LangChain 重构 RAG 全流程](#10-实战用-langchain-重构-rag-全流程)
11. [LangChain vs LangGraph：什么时候该升级](#11-langchain-vs-langgraph什么时候该升级)
12. [生产环境避坑指南](#12-生产环境避坑指南)

---

## 1. 起点问题：手写 RAG 到底累在哪？

### 你已经写过完整的 RAG 系统了

在 `005-rag-learning` 里，你从零实现了：
- PDF 解析 → 文本分块 → Embedding → ChromaDB 索引
- 语义检索 → Prompt 拼接 → LLM 生成

代码能跑，结果也不错。但回头看代码，有没有觉得：

```
每写一个新的 LLM 应用，都在重复这些：
  - openai_client.chat.completions.create(model=..., messages=..., temperature=...)
  - f"你是{role}，请用{lang}回答：{question}"  ← f-string 拼 Prompt
  - json.loads(response)  ← 手工解析输出
  - collection.query(query_texts=[...], n_results=...)  ← 深层 dict 取值
```

### 这些重复劳动的本质

| 你做的事情 | LangChain 的答案 |
|-----------|-----------------|
| `client.chat.completions.create(...)` | `ChatOpenAI().invoke(messages)` |
| `f"你是{role}，请回答{question}"` | `ChatPromptTemplate.from_messages([...])` |
| `json.loads(response.choices[0]...)` | `PydanticOutputParser(pydantic_object=Person)` |
| `if func_name == "xxx": result = ...` | `@tool` + `bind_tools()` |
| `collection.query(...)` → dict 取值 | `retriever.invoke(query)` → `list[Document]` |
| 手写 pipeline 函数 | `prompt | llm | parser`（管道操作符）|

**LangChain 不是给你新能力，而是把重复的模式标准化。**

---

## 2. LangChain 的核心哲学：一切皆 Runnable

### 一句话概括

```
LangChain = 一组标准化的"积木块" + 一个统一的"连接协议"
```

- **积木块**：ChatOpenAI、PromptTemplate、OutputParser、Retriever...
- **连接协议**：Runnable —— 所有积木块都实现了同样的接口
- **连接方式**：`|` 管道操作符（LCEL）

### Runnable 协议

每个"积木块"都有这三个方法：

```python
class Runnable:
    def invoke(input) -> output:    # 同步执行
        ...

    def stream(input) -> iterator:  # 流式输出
        ...

    def batch(inputs) -> outputs:   # 批量并发
        ...
```

因为所有组件都实现了 Runnable，所以可以无差别地用 `|` 连接：

```python
# 这三个组件类型完全不同，但都是 Runnable，所以可以 | 串联
chain = prompt | llm | parser
#       ↑        ↑      ↑
#   Runnable   Runnable  Runnable

chain.invoke({"question": "什么是RAG"})  # 从头跑到尾
```

### 数据流如何传递？

```python
chain = prompt | llm | parser

invoke({"question": "什么是RAG"})
       │
       ▼
    prompt.invoke()  →  ChatPromptValue  (messages 列表)
       │
       ▼
    llm.invoke()     →  AIMessage  (content + metadata)
       │
       ▼
    parser.invoke()  →  str  (纯文本)
```

**前一环节的输出 = 后一环节的输入。管道符自动完成数据传递。**

---

## 3. LLM 抽象：ChatOpenAI 不是黑魔法

### 不用 LangChain

```python
from openai import OpenAI
client = OpenAI(api_key="...", base_url="...")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "什么是LangChain？"}],
    temperature=0.3,
)
answer = response.choices[0].message.content
```

能用，但每次都要传 `model`、`messages`、`temperature`。想切模型？改所有调用点。

### 用了 LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="deepseek-chat", base_url="...", temperature=0.3)

# 配置一次，到处复用
result = llm.invoke([HumanMessage(content="什么是LangChain？")])
answer = result.content  # HumanMessage / AIMessage / SystemMessage 语义清晰
```

### 真正的好处在哪？

```python
# 好处1：切模型只改一行
llm = ChatOpenAI(model="gpt-4o")  # DeepSeek → OpenAI

# 好处2：和 Prompt 通过 | 组合
chain = prompt | llm

# 好处3：自动重试
llm = ChatOpenAI(model=..., max_retries=3)  # 网络抖动自动 retry
```

**ChatOpenAI 不是黑魔法，只是把 `client.chat.completions.create()` 包装成 Runnable。**
**它单独看不出优势，和 Prompt/Chain 组合时才真正发光。**

---

## 4. PromptTemplate：模板和变量分离

### 不用 LangChain

```python
# 方式A：f-string
prompt = f"你是{role}，请用{lang}回答：{question}"

# 痛点：
# - role/lang/question 混在字符串里
# - 变量多了像乱麻，少传一个不会报错（f-string 直接报 NameError 倒是会报）
# - 没法单独测试"模板是否合理"
```

### 用了 LangChain ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}。要求：{requirements}"),
    ("user", "请解释：{question}"),
])

# 模板是数据结构，变量是参数
messages = prompt.invoke({
    "role": "Python专家",
    "requirements": "用比喻解释，不超过50字",
    "question": "装饰器",
})
```

### 核心价值

1. **模板和变量分离**：改 Prompt 不改代码，改代码不改 Prompt
2. **类型检查**：变量缺失会报错（`KeyError`）而不是默默出奇怪输出
3. **Few-shot 支持**：`FewShotChatMessagePromptTemplate` 自动管理示例
4. **可复用**：同一个模板无限次 `.invoke()` 生成不同的 messages

---

## 5. OutputParser：让 LLM 输出结构化数据

### 问题的根源

LLM 返回自然语言，但程序要的是结构化数据（dict/list/enum）。

### 不用 LangChain

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "输出JSON格式: ..."}],
)
raw = response.choices[0].message.content
raw = raw.strip("```json").strip("```")  # 去 markdown 包裹
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    # 手写 fallback / retry / 正则提取...
    pass
```

### 用了 LangChain

```python
# StrOutputParser：AIMessage → str
chain = prompt | llm | StrOutputParser()
result = chain.invoke(...)  # 直接拿到 str

# JsonOutputParser：自动处理 markdown + 解析 + retry
parser = JsonOutputParser()
chain = prompt | llm | parser
result = chain.invoke(...)  # 直接拿到 dict

# PydanticOutputParser：类型安全！
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

parser = PydanticOutputParser(pydantic_object=Person)
chain = prompt | llm | parser
result = chain.invoke(...)  # 直接拿到 Person 对象
# result.name → IDE 自动补全！
```

### 自动纠错机制

```
LLM 输出格式不对 → Pydantic 验证失败 →
Parser 把错误信息 + 原始输出喂回 LLM →
LLM 重新输出正确格式 → 解析成功
```

**你不用写一行 retry 代码，LangChain 自动处理。**

---

## 6. Tool：让 LLM 调用外部函数

### 问题

LLM 只能"说"，不能"做"。怎么让它查数据库、调 API？

### 不用 LangChain

```python
# 1. 手写 JSON Schema
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
    }
}]

# 2. 解析 LLM 返回的 tool_call
tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)

# 3. 手工 dispatch
if tool_call.function.name == "get_weather":
    result = get_weather(args["city"])
elif tool_call.function.name == "get_stock":
    result = get_stock(args["symbol"])
# ... 10 个函数就写 10 个 elif

# 4. 回传结果
messages.append({"role": "tool", "content": json.dumps(result)})
```

### 用了 LangChain

```python
from langchain_core.tools import tool

# @tool 装饰器：函数签名 + docstring → 自动生成 JSON Schema
@tool
def get_weather(city: str) -> str:
    """查询城市当前天气"""
    return weather_db.get(city, "未知")

# bind_tools() 自动绑定
llm_with_tools = llm.bind_tools([get_weather, get_stock, ...])

# LLM 自动判断是否调 tool + 调哪个
ai_msg = llm_with_tools.invoke([HumanMessage(content="北京天气如何？")])
# ai_msg.tool_calls 已自动解析好

# dispatch 可以改用 dict 映射，不需要 if/elif
tool_map = {t.name: t for t in [get_weather, get_stock]}
for tc in ai_msg.tool_calls:
    result = tool_map[tc["name"]].invoke(tc["args"])
```

### 核心价值

- `@tool`：一个装饰器消除手写 JSON Schema
- `bind_tools()`：LLM 自动判断何时调哪个工具
- 工具可以像普通函数一样**独立测试**（不依赖 LLM）

---

## 7. Retriever：检索的统一抽象层

### 问题

你的 RAG 用了 ChromaDB。如果以后想换成 FAISS / Milvus / Pinecone？

### 不用 LangChain

```python
# ChromaDB 原生 API
results = collection.query(query_texts=["查询"], n_results=3)
for i in range(len(results["ids"][0])):
    meta = results["metadatas"][0][i]
    text = results["documents"][0][i]
    dist = results["distances"][0][i]
    # ...

# 换成 FAISS？
# results = faiss_index.search(query_vector, k=3)  ← API 完全不同
# → 所有检索代码要重写
```

### 用了 LangChain

```python
# 统一接口：不管底层是什么数据库
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke("查询")
# → list[Document]
#   Document(page_content="...", metadata={"title": "...", "source": "..."})

for doc in docs:
    print(doc.page_content)   # 统一属性访问
    print(doc.metadata["source"])  # 不再是深层 dict
```

### 核心价值

1. **供应商中立**：换数据库只改 `Chroma(...)` → `FAISS(...)`，检索代码不动
2. **统一返回类型**：`list[Document]`，不管底层是什么
3. **元数据过滤**：`search_kwargs={"filter": {"category": "HR"}}`
4. **可组合**：`retriever | format_docs` → 直接串到 RAG Chain

---

## 8. Chain：声明式的数据流管道

### 不用 LangChain 的手写 Pipeline

```python
def rag_pipeline(query):
    # Step 1: 检索
    docs = retriever.retrieve(query)
    # Step 2: 格式化
    context = format_docs(docs)
    # Step 3: 构建 Prompt
    messages = build_prompt(query, context)
    # Step 4: 调 LLM
    response = llm(messages)
    # Step 5: 提取结果
    return response.content

# 痛点：
# - 加一个中间步骤 → 改函数结构
# - 想并行两个操作 → 手写多线程
# - 想复用 Step 2+3 → 复制粘贴
```

### 用了 LangChain LCEL

```python
# 同样的逻辑，声明式表达
rag_chain = (
    {
        "context": retriever | format_docs,    # 分支1：检索 + 格式化
        "question": RunnablePassthrough(),      # 分支2：原样传递
    }
    | prompt   # 上一步的 dict 自动填充 {context} 和 {question}
    | llm      # AIMessage
    | parser   # 纯文本 str
)

# 一个 invoke() 跑完所有步骤
answer = rag_chain.invoke("公司年假多少天？")
```

### 声明式 vs 命令式

```
命令式（手写）:
  做A → 把结果存变量 → 做B → 把结果存变量 → 做C
  关注"怎么做"

声明式（LCEL）:
  A | B | C
  关注"数据怎么流"
```

---

## 9. Runnable 协议：LCEL 的核心

### 四个关键 Runnable

| Runnable | 作用 | 类比 |
|----------|------|------|
| `RunnablePassthrough` | 数据原样透传 | 透明管道 |
| `RunnableLambda` | 包装任意函数 | 适配器插头 |
| `RunnableParallel` | 并行执行多个分支 | 分流器 |
| `RunnableBranch` | 条件路由 | if/else 开关 |

### RunnablePassthrough

```python
# 场景：RAG Chain 中，question 不需要处理，原样传给 prompt
{
    "context": retriever | format_docs,   # 需要处理
    "question": RunnablePassthrough(),     # 不需要处理，透传
}
```

### RunnableLambda

```python
# 把普通函数变成 Runnable，接入管道
def count_words(text: str) -> str:
    return f"{text}\n\n(共 {len(text)} 字)"

chain = prompt | llm | StrOutputParser() | RunnableLambda(count_words)
```

### RunnableParallel

```python
# 两个 LLM 调用并发执行！
parallel = RunnableParallel(
    summary=chain_summary,
    translation=chain_translation,
)
# 同时得到摘要和翻译，时间 ≈ max(摘要耗时, 翻译耗时)，而非相加
```

---

## 10. 实战：用 LangChain 重构 RAG 全流程

### 重构对比

| 组件 | 手写版（~300行） | LangChain 版（~120行） | 减少 |
|------|-----------------|----------------------|------|
| 分块器 | `RecursiveChunker` 类 (~50行) | `RecursiveCharacterTextSplitter()` | 1行 |
| 向量存储 | `chromadb.Client()` + `create_collection()` + `add()` (~15行) | `Chroma.from_documents()` | 1行 |
| Prompt | f-string + 手动 messages (~20行) | `ChatPromptTemplate` 声明式 | 5行 |
| 检索 | `collection.query()` → dict 取值 (~15行) | `retriever.invoke()` | 1行 |
| Pipeline | 手写函数调用链 (~30行) | LCEL 管道 | 4行 |

### 最核心的一行代码

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)
```

**这一行就是整个 RAG Pipeline。** 数据流清晰可见：检索 → 格式化 → 注入 Prompt → LLM → 输出。

### 什么时候不用 LangChain？

LangChain 的价值在于**标准化重复模式**。如果你的应用：
- 只有一个简单的 LLM 调用 → 用原生 openai 库即可
- 有复杂的 Prompt + Chain + Tool + Retriever → LangChain 明显省代码
- 有复杂的条件分支、循环、状态管理 → 该升级到 LangGraph 了（见下一节）

---

## 11. LangChain vs LangGraph：什么时候该升级？

### LangChain 擅长的

```
线性流程：A → B → C → D
带分支：A → {B1 并行, B2 并行} → C
```

### LangChain 不擅长的

```
A → B → 如果B的结果>0.8 → C → D
      如果B的结果<0.2 → E → F（重新查询）
      如果B的结果介于 → G（人工审核）→ H
```

这种**动态路由、循环、状态管理**的场景，用 LangGraph（状态机/图编排）。

### 简单判断

| 你的需求 | 用这个 |
|---------|-------|
| 单次 LLM 调用 | 原生 openai |
| 线性/简单分支 Pipeline | LangChain LCEL |
| 多轮 Agent 循环、条件路由、人工审核 | LangGraph |
| 多 Agent 协作 | LangGraph + LangChain |

**LangChain 是乐高积木，LangGraph 是乐高图纸。两个配合使用。**

---

## 12. 生产环境避坑指南

### 1. 不要在 Chain 里隐藏副作用

```python
# ❌ 坏做法：在 RunnableLambda 里调外部 API（难以测试/调试）
def save_to_db(text):
    db.insert(text)  # 副作用！
    return text

# ✅ 好做法：把副作用放在 Chain 外面
result = chain.invoke(query)
db.insert(result)  # 明确可见
```

### 2. 注意 Token 消耗

```python
# Prompt 模板层数越多，LLM 看到的 context 越长
# 经常检查 chain.get_graph() 看看实际传了多少东西
```

### 3. API Key 管理

```python
# 从环境变量读取，不要硬编码
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # ✅
    # api_key="sk-..."                    # ❌ 永远不要硬编码
)
```

### 4. Debug 技巧

```python
# LangChain 的 verbose 模式
llm = ChatOpenAI(model=..., verbose=True)  # 打印每次 LLM 调用的详情

# 设置全局 debug
import langchain
langchain.debug = True  # 极其详细，包括每步的输入输出
```

### 5. 不要过度抽象

```python
# ❌ 为了一行简单的 LLM 调用建 5 层 Chain
chain = prompt_a | step_b | step_c | step_d | llm

# ✅ 简单的对话直接用原生 API
response = client.chat.completions.create(...)

# 复杂的 Pipeline 才用 Chain
# 简单的调用用原生 API
# 规则：重复 3 次以上的模式才考虑抽象
```

---

## 总结：一张图理解 LangChain

```
                   LangChain 的世界
    ┌──────────────────────────────────────────┐
    │                                           │
    │   所有的"积木块"都是 Runnable               │
    │                                           │
    │   PromptTemplate  ─┐                      │
    │   ChatOpenAI      ─┤                      │
    │   OutputParser    ─┼── Runnable 协议       │
    │   Retriever       ─┤   .invoke()          │
    │   Tool            ─┤   .stream()          │
    │   RunnableLambda  ─┤   .batch()           │
    │   RunnableParallel─┘                      │
    │                                           │
    │   用 | 管道连接：                           │
    │   prompt | llm | parser                   │
    │                                           │
    │   数据从左到右自动流转                       │
    └──────────────────────────────────────────┘

    怎么判断该不该用 LangChain？
    ┌─────────────────────────────────────┐
    │ 简单对话 → 原生 openai               │
    │ 复杂 Pipeline → LangChain LCEL      │
    │ 多 Agent/状态机 → LangGraph         │
    └─────────────────────────────────────┘

    核心原则：
    LangChain 不是为了"酷"而用，
    而是为了消除重复代码，让 Pipeline 可读、可测、可维护。
```

### Demo 对应关系

| Demo | 概念 | 文件 |
|------|------|------|
| Demo 01 | LLM 抽象 | `demo_01_llm.py` |
| Demo 02 | PromptTemplate | `demo_02_prompt_template.py` |
| Demo 03 | OutputParser | `demo_03_output_parser.py` |
| Demo 04 | Tool | `demo_04_tool.py` |
| Demo 05 | Retriever | `demo_05_retriever.py` |
| Demo 06 | Chain | `demo_06_chain.py` |
| Demo 07 | Runnable | `demo_07_runnable.py` |
| Demo 08 | RAG 重构 | `demo_08_rag_refactor.py` |

### 下一步学习

- 试试把 Demo 08 的 rag_chain 加上多轮对话记忆（`RunnableWithMessageHistory`）
- 用 `chain.get_graph()` 可视化你的 Pipeline
- 升级到 LangGraph：实现 Agent 循环（思考 → 调工具 → 观察结果 → 再思考）
- 用 LangSmith 追踪每次 LLM 调用的延迟、Token 消耗、错误
