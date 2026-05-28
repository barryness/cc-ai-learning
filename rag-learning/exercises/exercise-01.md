# 练习1：扩展知识库

## 踩坑记录

### 坑1：TF-IDF 只索引 content，忽略 title

**问题**：新增"vibe coding工具"文档后，查询"vibe coding工具有哪些？"检索到 0 条文档。

**根因**：关键词只在 title 出现，content 里没有。TF-IDF 只索引 content。

**修复**：`SimpleRetriever.__init__` 改为 title + content 联合索引。

### 坑2：为什么"实时数据处理"没修也能搜到？

纯运气——文档内容碰巧有"使用"这个词和查询重叠。深入分析见下方测试代码。

## 测试代码

### test_token_overlap.py —— 分析两个查询的 token 差异

```python
"""
验证为什么 "vibe coding" 检索不到而 "实时数据处理" 能检索到。
结论：TF-IDF 是纯字面匹配，共同有效词 > 0 就有分数，= 0 就搜不到。
"""
import jieba
import re

# TF-IDF 默认 token_pattern：只保留 2+ 字母数字字符
TOKEN_PATTERN = re.compile(r'(?u)\b\w\w+\b')

def get_valid_tokens(text):
    """模拟 TF-IDF 实际生效的 token"""
    all_tokens = jieba.lcut(text)
    return {t for t in all_tokens if TOKEN_PATTERN.match(t)}

# === 案例1：vibe coding（搜不到） ===
vibe_content = '现在使用最强大的是Claude code公认最好用的codex以及opencode'
vibe_query = 'vibe coding工具有哪些？'

vq = get_valid_tokens(vibe_query)
vd = get_valid_tokens(vibe_content)
print('=== vibe coding ===')
print(f'查询有效词: {vq}')
print(f'文档有效词: {vd}')
print(f'共同有效词: {vq & vd}')  # 输出: set() → 相似度 = 0
print(f'→ 共同词为 0，检索失败\n')

# === 案例2：实时数据处理（能搜到） ===
data_content = '多用flink,spark也有不少企业在使用'
data_query = '实时数据处理使用的什么工具？'

dq = get_valid_tokens(data_query)
dd = get_valid_tokens(data_content)
print('=== 实时数据处理 ===')
print(f'查询有效词: {dq}')
print(f'文档有效词: {dd}')
print(f'共同有效词: {dq & dd}')  # 输出: {'使用'} → 相似度 > 0
print(f'→ 碰巧有 1 个共同词"使用"，检索成功（相关度仅 0.36）')

# === 实际输出 ===
# === vibe coding ===
# 查询有效词: {'vibe', '工具', '哪些', 'coding'}
# 文档有效词: {'code', '使用', 'Claude', '最好', '现在', '以及', '强大', '公认', 'opencode', 'codex'}
# 共同有效词: set()
# → 共同词为 0，检索失败
#
# === 实时数据处理 ===
# 查询有效词: {'实时', '数据处理', '使用', '什么', '工具'}
# 文档有效词: {'flink', '不少', '多用', '企业', '使用', 'spark'}
# 共同有效词: {'使用'}
# → 碰巧有 1 个共同词"使用"，检索成功（相关度仅 0.36）
```

### test_index_fix.py —— 验证标题+内容联合索引修复

```python
"""
验证检索器改为索引 title + content 后的效果。
修复前："vibe coding工具" 检索 0 条；修复后正确检索到。
"""
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS = [
    {"title": "vibe coding工具", "content": "现在使用最强大的是Claude code公认最好用的codex以及opencode"},
    {"title": "目前大数据实时处理技术使用的工具", "content": "多用flink,spark也有不少企业在使用"},
    {"title": "AI中的养龙虾", "content": "养龙虾通常说的是openclaw"},
]

def search(query, use_title=True):
    if use_title:
        texts = [f"{d['title']} {d['content']}" for d in DOCUMENTS]
    else:
        texts = [d['content'] for d in DOCUMENTS]

    tokenized = [" ".join(jieba.lcut(t)) for t in texts]
    vec = TfidfVectorizer()
    doc_vecs = vec.fit_transform(tokenized)

    q_tok = " ".join(jieba.lcut(query))
    q_vec = vec.transform([q_tok])
    sims = cosine_similarity(q_vec, doc_vecs)[0]

    for i, s in enumerate(sims):
        if s > 0:
            print(f"  [{DOCUMENTS[i]['title']}] 相似度: {s:.4f}")

print("=== 只索引 content（修复前） ===")
search("vibe coding工具有哪些？", use_title=False)  # 无输出，全为 0

print("\n=== 索引 title + content（修复后） ===")
search("vibe coding工具有哪些？", use_title=True)   # 正确命中

# === 实际输出 ===
# === 只索引 content（修复前） ===
# （无输出，所有文档相似度均为 0）
#
# === 索引 title + content（修复后） ===
#   [vibe coding工具] 相似度: 0.4709
```

## 修改的代码

`minimal_rag.py` 第 135-138 行：

```python
# 修改前
tokenized = [" ".join(jieba.lcut(c)) for c in self.contents]

# 修改后
full_texts = [f"{doc['title']} {doc['content']}" for doc in documents]
tokenized = [" ".join(jieba.lcut(t)) for t in full_texts]
```

## 知识点

- TF-IDF 是**纯字面匹配**，不理解语义。共同 token 数为 0 则相似度严格为 0
- RAG 检索字段设计很关键：title、summary、content 各有权重，title 往往含最重要的关键词
- 工业级用 Embedding 模型做语义搜索，"vibe coding" 和 "Claude code" 即使字面不同也能匹配
- 排查检索问题时，先把查询和文档的 token 打印出来对比——通常一眼就能发现问题
