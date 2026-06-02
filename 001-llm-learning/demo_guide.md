# LLM 入门 —— 零代码实战指南

> 不用写代码，跟着步骤走，每次执行一条命令就能看到 LLM 的核心概念在运行。
> 每一步都告诉你"会看到什么"。

---

## 准备工作（约 5 分钟）

### 第一步：打开终端

Mac 的终端在 `启动台 → 其他 → 终端`（或者按 `Cmd + 空格` 搜索"终端"）。

打开后会看到一个黑色（或白色）窗口，有一个闪烁的光标等着你输入命令。

### 第二步：进入项目目录

复制粘贴下面这条命令到终端，按回车：

```bash
cd ~/data/workspace/learn_space/ai-learning/llm-learning
```

**你会看到**：光标移动到新的一行，没有报错就是成功。

### 第三步：检查环境

复制粘贴：

```bash
python3 --version
```

**你会看到**：`Python 3.14.1` 或类似版本号。

---

## 实验 1：Token —— 看看文本被切成多少"积木块"（约 2 分钟）

Token = LLM 的计数单位。中文 1 个字 ≈ 1 Token，英文 0.75 个词 ≈ 1 Token。

### 操作

复制粘贴：

```bash
python3 -c "
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')

for text in ['Hello, world!', '中国的首都是北京', 'Transformer是一种神经网络架构']:
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]
    print(f'原文: {text}')
    print(f'Token数: {len(tokens)}')
    print(f'拆分: {decoded}')
    print()
"
```

### 预期输出

```
原文: Hello, world!
Token数: 4
拆分: ['Hello', ',', ' world', '!']

原文: 中国的首都是北京
Token数: 6
拆分: ['中国', '的', '首', '都', '是', '北京']

原文: Transformer是一种神经网络架构
Token数: 16
拆分: ['Transformer', '是', '一', '种', ...]
```

### 你学到了什么

- **英文**："Hello, world!" 是 4 个 Token（词 + 标点都算）
- **中文**："中国的首都是北京" 是 6 个 Token（每个字/词独立）
- 混合语言："Transformer是一种神经网络架构" 更多 Token
- **这就是为什么 LLM 按 Token 计费**——你输入的每个字都会被拆碎

---

## 实验 2：Attention —— 词与词之间的"关注"（约 3 分钟）

### 操作

复制粘贴：

```bash
python3 -c "
import numpy as np

# 一句话：The cat sat on the mat
words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
np.random.seed(42)

# 每个词用数字向量表示（真实模型中是768维）
embeddings = np.random.randn(6, 4)

# 计算关注度分数：每个词和所有其他词的点积
scores = embeddings @ embeddings.T

# Softmax 转成概率
e_x = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
weights = e_x / e_x.sum(axis=-1, keepdims=True)

# 打印关注度矩阵
print('    ', '  '.join(f'{w:>6}' for w in words))
for i, w in enumerate(words):
    row = '  '.join(f'{weights[i][j]:.3f}' for j in range(6))
    print(f'{w:>6}: {row}')

# 解读
print()
for i, w in enumerate(words):
    top_j = np.argmax(weights[i])
    print(f'\"{w}\" 最关注 \"{words[top_j]}\" (权重 {weights[i][top_j]:.3f})')
"
```

### 预期输出

你会看到一个 6×6 的表格（注意力权重矩阵），以及每行解读：

```
      The     cat     sat      on     the     mat
  The: 0.692  0.282  0.009  0.007  0.001  0.009
  cat: 0.247  0.730  0.010  0.002  0.003  0.008
  sat: 0.025  0.031  0.245  0.087  0.532  0.079
   on: 0.000  0.000  0.001  0.991  0.004  0.004
  the: 0.001  0.001  0.089  0.072  0.814  0.023
  mat: 0.003  0.004  0.011  0.057  0.019  0.905

"The" 最关注 "The" (权重 0.692)
"cat" 最关注 "cat" (权重 0.730)
"sat" 最关注 "the" (权重 0.532)
"on" 最关注 "on" (权重 0.991)
"the" 最关注 "the" (权重 0.814)
"mat" 最关注 "mat" (权重 0.905)
```

### 你学到了什么

- 数字越大（越接近 1.0），表示该词越关注对方
- "sat" 最关注 "the"（权重 0.532）——"坐"这个动作和"垫子"相关
- "on" 最关注自己（权重 0.991）——介词通常不需要太多上下文
- 这就是 Attention 的本质：**每个词都看向所有词，计算谁更重要**

---

## 实验 3：Prompt 的影响力 —— 同样问题，不同回答（约 2 分钟）

### 操作

**先试普通模式**，复制粘贴：

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
llm = ChatOpenAI(model=os.getenv('MODEL_NAME'), api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))
r = llm.invoke([HumanMessage(content='解释什么是机器学习')])
print(r.content[:300])
"
```

**再试角色扮演**，复制粘贴：

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
llm = ChatOpenAI(model=os.getenv('MODEL_NAME'), api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

# 角色：幼儿园老师
messages = [
    SystemMessage(content='你是一个幼儿园老师。用最简单的语言、比喻和故事来解释，每句话不超过20个字。'),
    HumanMessage(content='解释什么是机器学习'),
]
r = llm.invoke(messages)
print(r.content[:300])
"
```

### 预期输出

**普通模式**：教科书式的回答，有定义、有例子、有条理。

**幼儿园老师模式**：用比喻和故事，简单语言，短句。比如：

> "机器学习就是教电脑自己学东西。就像教小朋友认苹果——给他看很多苹果，他就学会了。"

### 你学到了什么

- **同一个 LLM，同一个问题**，只是换了一句"角色设定"，回答完全不同
- **Prompt 不是命令，是起点**——你给 LLM 的"背景故事"决定了它从哪个角度回答
- 这就是 Prompt Engineering（提示工程）为什么是一门技术

---

## 实验 4：测试幻觉 —— LLM 会编造答案吗？（约 2 分钟）

### 操作

复制粘贴：

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
llm = ChatOpenAI(model=os.getenv('MODEL_NAME'), api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

questions = [
    '2025 年诺贝尔文学奖得主是谁？',          # 未来的事
    'Transformer 架构是哪一年提出的？',        # 应该知道
    '2026 年世界杯决赛比分是多少？',            # 未来的事
]

for q in questions:
    print(f'问: {q}')
    r = llm.invoke([
        SystemMessage(content='你是一个诚实的助手。如果不知道就说不知道，不要编造。'),
        HumanMessage(content=q),
    ])
    answer = r.content[:150]
    if '不知道' in answer or '抱歉' in answer or '尚未' in answer:
        print(f'答: ✅ 诚实地说了不知道')
    else:
        print(f'答: ⚠️ 可能产生了幻觉')
    print(f'   {answer}')
    print()
"
```

### 预期输出

```
问: 2025 年诺贝尔文学奖得主是谁？
答: ✅ 诚实地说了不知道
   抱歉，2025年的诺贝尔文学奖尚未颁发...

问: Transformer 架构是哪一年提出的？
答: ⚠️ 可能产生了幻觉  （其实是它知道答案）
   Transformer 架构是在 2017 年提出的...

问: 2026 年世界杯决赛比分是多少？
答: ✅ 诚实地说了不知道
   抱歉，2026年世界杯决赛目前尚未举行...
```

### 你学到了什么

- **对于知识截止日期内的**问题 → LLM 回答准确
- **对于未来的**事件 → 好的 LLM 会承认不知道
- **但并非所有 LLM 都这么诚实**——有些会编造看起来"合理"的答案
- 这就是**幻觉**：LLM 不是在"撒谎"，而是在"生成最合理的模式"

---

## 实验 5：Context Window —— 信息放在哪里更容易被记住（约 2 分钟）

### 操作

复制粘贴：

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
llm = ChatOpenAI(model=os.getenv('MODEL_NAME'), api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

filler = '这是无关紧要的填充内容。\n' * 30

# 关键信息在开头
text1 = '秘密代码是：GOLDEN-2026\n' + filler
print('📍 关键信息在开头:')
r1 = llm.invoke([HumanMessage(content=text1 + '\n\n请问：秘密代码是什么？')])
print(f'   回答: {r1.content[:80]}')
print()

# 关键信息在结尾
text2 = filler + '秘密代码是：GOLDEN-2026\n'
print('📍 关键信息在结尾:')
r2 = llm.invoke([HumanMessage(content=text2 + '\n\n请问：秘密代码是什么？')])
print(f'   回答: {r2.content[:80]}')
"
```

### 预期输出

```
📍 关键信息在开头:
   回答: 秘密代码是：GOLDEN-2026...

📍 关键信息在结尾:
   回答: 秘密代码是：GOLDEN-2026...
```

（如果文件更长，中间的信息可能会丢失——这就是 "Lost in the Middle" 现象）

### 你学到了什么

- **开头和结尾的信息 LLM 记得最清**
- **中间的信息最容易丢失**
- 这就是为什么写 Prompt 时，**最重要的指令放开头，最重要的输出放结尾**

---

## 实验 6：注意力稀释 —— 为什么长文本会"失忆"（约 2 分钟）

### 操作

复制粘贴：

```bash
python3 -c "
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

for n in [4, 16, 64, 256, 1024]:
    np.random.seed(42)
    scores = np.random.randn(n)
    weights = softmax(scores)
    print(f'Token 数: {n:5d}  |  平均关注度: {weights.mean():.6f}  |  最高关注度: {weights.max():.4f}')
"
```

### 预期输出

```
Token 数:     4  |  平均关注度: 0.250000  |  最高关注度: 0.5089
Token 数:    16  |  平均关注度: 0.062500  |  最高关注度: 0.2076
Token 数:    64  |  平均关注度: 0.015625  |  最高关注度: 0.0813
Token 数:   256  |  平均关注度: 0.003906  |  最高关注度: 0.1059
Token 数:  1024  |  平均关注度: 0.000977  |  最高关注度: 0.0271
```

### 你学到了什么

- Token 越少，LLM 对每个词的"关注"越集中
- Token 越多，关注度被稀释得越厉害
- 4 个 Token 时，最高关注度 0.51（一半的注意力给了最重要的词）
- 1024 个 Token 时，最高关注度只有 0.027（不到 3%）
- **这就是为什么超长上下文中 LLM 会"失忆"**——注意力被稀释了

---

## 收尾

### 你刚才学完了什么？

| 实验 | 概念 | 一句话总结 |
|------|------|-----------|
| 1 | Token | 文本被切成"积木块"，LLM 按块计费 |
| 2 | Attention | 每个词看向所有词，计算"谁更重要" |
| 3 | Prompt | 同一问题，角色设定不同→回答完全不同 |
| 4 | 幻觉 | LLM 可能编造答案，因为它目标是"续写"不是"说真话" |
| 5 | Context | 信息放开头或结尾最容易被记住 |
| 6 | 注意力稀释 | 越长越"失忆"，关注度被摊薄了 |

### 下一步

- 打开 `llm_playground.py`——有更友好的图形界面，点点按钮就能做实验
- 阅读 `learning_notes.md`——深入学习每个概念的技术原理
- 完成 `exercise.md`——巩固理解

### 遇到问题？

- **"找不到命令"** → 先运行 `cd ~/data/workspace/learn_space/ai-learning/llm-learning`
- **"没有这个模块"** → 运行 `pip install tiktoken numpy python-dotenv langchain-openai`
- **"API 错误"** → 检查 `.env` 文件是否有正确的 API Key
- 其他问题：截图给我，我帮你解决
