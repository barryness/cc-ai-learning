# LLM 练习题

> 配套 `learning_notes.md` 和 `demo.py` 使用。
> 每个练习都对应一个核心概念。先读笔记，再跑 demo，最后做练习。

---

## 练习1：Token 直觉

### 背景

Token 是 LLM 的计价单位。理解 Token 能帮你估算成本、优化 Prompt。

### 任务

用 Python 计算以下文本的 Token 数，并回答问题。

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "Hello, how are you today?",
    "你今天怎么样？",
    # 加上你自己的文本
]

for text in texts:
    tokens = enc.encode(text)
    print(f"'{text[:30]}...' → {len(tokens)} Tokens")
```

### 问题

1. 同样表达"你好吗"，中文（"你好吗"）和英文（"How are you"）各用多少 Token？
2. 为什么中文每个字 ≈ 1 Token，而英文一个词 ≈ 1 Token？
3. 一段 1000 字的英文技术文档大概多少 Token？同样的内容写成中文呢？

### 验证

跑 `demo.py` 中的 `demo_tokenization()` 查看实际 Token 拆分结果。

---

## 练习2：API 调用练习

### 背景

这是你第一次用一个真实的 LLM API。目标是学会基本的调用模式。

### 任务

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 任务1：一次简单的问答
response = llm.invoke([HumanMessage(content="用一句话解释什么是 LLM")])
print(response.content)

# 任务2：修改温度参数，观察输出变化
# 提示：temperature=0 → 确定性输出，temperature=1.5 → 创造性输出

# 任务3：给 LLM 设定角色
# 提示：用 SystemMessage 设定角色
```

### 要求

1. 成功调用 LLM 并得到回复
2. 分别用 `temperature=0` 和 `temperature=1.5` 调用 3 次，观察输出的重复性
3. 用 `SystemMessage` 给 LLM 设定一个角色（如"你是一个 Linux 终端"）

### 思考

- `temperature=0` 时，多次调用结果一样吗？为什么？
- 角色设定如何改变了 LLM 的回复风格？

---

## 练习3：Prompt Engineering 对比

### 背景

同样的 LLM，不同的 Prompt → 完全不同的结果。

### 任务

用 3 种不同的 Prompt 问同一个问题，比较输出差异。

**问题**：解释什么是数据库索引

**Prompt 设计**：

| Prompt 编号 | 风格 | 提示 |
|------------|------|------|
| 1 | 学术风格 | "请从计算机科学角度解释..." |
| 2 | 给非技术人员 | "用生活类比如..." |
| 3 | 给 DBA | "从查询优化角度..." |

### 要求

1. 分别用 3 种 Prompt 调用 LLM
2. 输出中标注每个回答的特点（篇幅、用词、结构）
3. 写一段分析：为什么同一个模型会给出风格迥异的回答？

### 延伸

如果你问"数据库索引的缺点"，不同风格的 Prompt 会有什么差异？

---

## 练习4：观察 Context Window 的影响

### 背景

Context Window 有限时，LLM 会"丢失"较早的信息。

### 任务

构造一个实验：
1. 生成一段 3000 字的长文本
2. 在文本的**开头、中间、结尾**分别隐藏一条关键信息
3. 每次只放一条关键信息，问 LLM 能否正确回答

### 代码框架

```python
# 生成填充文本（每行一个数字，从 1 到 2000）
filler = "\n".join([f"计数：第 {i} 行" for i in range(1, 2001)])

# 1. 关键信息在开头
info_first = "秘密代码是：BLUE-2026\n" + filler

# 2. 关键信息在中间（第 1000 行左右）

# 3. 关键信息在结尾

# 分别问："秘密代码是什么？"
```

### 问题

1. 哪种位置 LLM 记得最清楚？
2. 如果文本再翻一倍（6000 行），结果会怎样？
3. 在实际应用中，如何避免这个问题的负面影响？

---

## 练习5：幻觉检测

### 背景

LLM 会自信地编造答案。学会识别幻觉是使用 LLM 的关键技能。

### 任务

向 LLM 问以下问题，判断哪些回答是幻觉：

```python
questions = [
    "2025 年图灵奖得主是谁？",
    "Transformer 的 'T' 代表什么？",
    "LangGraph 是什么框架？",
    "2024 年巴黎奥运会中国获得了多少枚金牌？",
    "Attention Is All You Need 论文的作者有哪些？",
]
```

### 检测方法

1. **知识边界检测**：问题涉及的时间在训练数据截止后吗？
2. **细节一致性**：回答中的数字、名字是否具体但不可验证？
3. **自信度测试**：问"你确定吗？"，看 LLM 是否改口
4. **交叉验证**：用搜索引擎验证可疑事实

### 输出格式

| 问题 | 判断 | 依据 |
|------|------|------|
| 2025 年图灵奖得主是谁？ | 幻觉/事实 | ... |

### 思考

- 要求"不知道就说不知道"能消除幻觉吗？
- 为什么 LLM 宁可编造也不说"不知道"？

---

## 练习6：综合实验 —— 设计你自己的 Prompt 系统

### 背景

综合运用所有学到的概念，设计一个实用的 Prompt 系统。

### 任务

设计一个"数据仓库助手" Prompt。要求：

1. **角色设定**：资深 DWH 工程师
2. **行为规则**：
   - 先用中文回答
   - 必须包含 SQL 例子
   - 超出知识范围要说不知道
3. **输出格式**：结构化（标题、说明、示例、注意事项）
4. **上下文管理**：如何处理用户的多轮追问？

### 代码框架

```python
system_prompt = """你是一个...（请自己填写）

行为规则：
1. ...
2. ...
3. ...

输出格式：
...
"""

# 测试你的 Prompt
test_cases = [
    "什么是缓慢变化维（SCD）？",
    "请解释星型模型和雪花模型的区别",
    "2027 年数据仓库的最新趋势是什么？",  # 应该触发"不知道"
]

for question in test_cases:
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])
    print(f"Q: {question}")
    print(f"A: {response.content[:200]}")
    print()
```

### 评估标准

| 标准 | 说明 |
|------|------|
| 回答准确性 | 技术概念是否正确 |
| 代码质量 | SQL 示例是否可运行 |
| 幻觉控制 | 对超范围问题是否诚实 |
| 格式一致性 | 输出是否结构化 |
| 用户体验 | 是否适合目标读者 |

---

## 答案要点（思考后再看）

<details>
<summary>练习1 答案要点</summary>

- 中文 "你好吗" → 3 Tokens，英文 "How are you" → 3 Tokens
- 中文每个字独立 Token，因为汉字是象形文字，BPE 算法难以合并
- 1000 字英文 ≈ 750 Tokens（平均 0.75 词/Token）
- 1000 字中文 ≈ 1000-1200 Tokens（含标点）
</details>

<details>
<summary>练习4 答案要点</summary>

- LLM 对开头和结尾的信息记忆最好（首因效应 + 近因效应）
- 中间的信息最容易丢失（Lost in the Middle）
- 解决方案：RAG、分段处理、关键信息放开头或结尾
</details>

<details>
<summary>练习5 答案要点</summary>

- 2025 年图灵奖（截止日期后）→ 大概率幻觉
- Transformer 的 T（基础概念）→ 事实
- 训练数据截止后的事件、极其冷门的知识、编造的引用 → 高度怀疑
</details>
