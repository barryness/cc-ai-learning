"""
=============================================================================
  练习7：RAG 效果评估
=============================================================================

三种评估方法：
  1. BLEU —— 基于 n-gram 的字面重叠度（机器翻译常用，但不太适合问答）
  2. ROUGE-L —— 基于最长公共子序列（摘要评估常用，更适合 RAG）
  3. LLM 打分 —— 让 LLM 判断：回答是否准确、是否基于文档、是否有幻觉

运行方式：
  cd rag-learning/demo && python3 ../exercises/eval_rag.py
"""

import sys
import jieba
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))

from minimal_rag import RAGPipeline, DOCUMENTS, generate_answer
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================================
# 第1步：准备评估数据集（问题 + 标准答案）
# ============================================================================
# 每个 QA 对基于知识库手动编写，标准答案从对应文档中直接提取
EVAL_DATASET = [
    {
        "id": 1,
        "query": "公司混合办公的政策是什么？",
        "ground_truth": "每周一三五在办公室，周二周四可远程办公。年假15天，带薪病假7天。",
        "source_doc": "办公政策",
        "type": "事实提取",  # 简单：答案直接在文档里
    },
    {
        "id": 2,
        "query": "公司融资了多少钱？投资方有哪些？",
        "ground_truth": "A轮融资5000万元，投资方包括红杉资本和创新工场。",
        "source_doc": "融资情况",
        "type": "事实提取",
    },
    {
        "id": 3,
        "query": "AI Tutor支持哪些编程语言？",
        "ground_truth": "支持Python、Java、Go等20+编程语言。",
        "source_doc": "产品介绍",
        "type": "事实提取",
    },
    {
        "id": 4,
        "query": "AI Tutor月费多少钱？",
        "ground_truth": "月费29元。",
        "source_doc": "产品介绍",
        "type": "事实提取",
    },
    {
        "id": 5,
        "query": "Python的map函数是做什么的？",
        "ground_truth": "map()函数接收一个函数和一个可迭代对象，返回一个迭代器，将函数应用到每个元素上。",
        "source_doc": "Python函数式编程",
        "type": "事实提取",
    },
    {
        "id": 6,
        "query": "公司年假有多少天？",
        "ground_truth": "年假15天。",
        "source_doc": "办公政策",
        "type": "简单事实",
    },
    {
        "id": 7,
        "query": "vibe coding工具有哪些？",
        "ground_truth": "Claude code、codex、opencode。",
        "source_doc": "vibe coding工具",
        "type": "事实提取",
    },
    {
        "id": 8,
        "query": "公司有哪些竞争对手？",
        "ground_truth": "知识库中暂无相关信息。",  # 正确答案就是说不知道
        "source_doc": None,
        "type": "知识库外",  # 测试 RAG 是否正确拒绝回答
    },
    {
        "id": 9,
        "query": "AI Tutor的后端用什么框架？",
        "ground_truth": "FastAPI框架。",
        "source_doc": "技术架构",
        "type": "简单事实",
    },
    {
        "id": 10,
        "query": "公司数据库使用什么？",
        "ground_truth": "PostgreSQL。",
        "source_doc": "技术架构",
        "type": "简单事实",
    },
]


# ============================================================================
# 第2步：评估指标
# ============================================================================
def rouge_l_score(prediction: str, reference: str) -> float:
    """
    ROUGE-L：基于最长公共子序列（LCS）的召回率。

    为什么用 ROUGE-L 而不是 BLEU？
      BLEU 看 n-gram 精确匹配 → "月费29元" vs "每月29元" → 0分（不公平）
      ROUGE-L 看最长公共子序列 → "月费29元" vs "每月29元" → 能匹配到 "月" 和 "29元"
      ROUGE-L 对表述方式差异的容忍度更高，更适合问答评估
    """
    # jieba 分词后计算
    pred_tokens = jieba.lcut(prediction)
    ref_tokens = jieba.lcut(reference)

    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0

    # DP 求最长公共子序列长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    # ROUGE-L F1：调和平均精确率和召回率
    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_match_score(prediction: str, reference: str) -> float:
    """简单关键词重叠度——辅助指标"""
    pred_words = set(jieba.lcut(prediction))
    ref_words = set(jieba.lcut(reference))
    if not ref_words:
        return 0.0
    # 标准答案中有多少关键词出现在了预测答案里（召回率导向）
    return len(pred_words & ref_words) / len(ref_words)


def llm_evaluate(query: str, answer: str, ground_truth: str) -> dict:
    """
    用 LLM 做语义评估——最可靠但最贵的方法。

    评估三个维度：
      - 准确性（0-5）：回答和标准答案的事实一致程度
      - 完整性（0-5）：是否覆盖了标准答案的所有要点
      - 是否有幻觉（0-5）：5=完全基于文档，0=完全编造
    """
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    prompt = (
        "你是一个RAG系统评估专家。请对比AI回答和标准答案，给出评分。\n\n"
        f"用户问题：{query}\n\n"
        f"AI回答：{answer}\n\n"
        f"标准答案：{ground_truth}\n\n"
        "请从以下三个维度打分（0-5分，只输出JSON）：\n"
        '{"accuracy": 分数, "completeness": 分数, "faithfulness": 分数, "comment": "简短评语"}'
    )

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )

    import json
    try:
        # 提取 JSON 部分（LLM 可能输出额外文字）
        text = response.choices[0].message.content
        # 找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"accuracy": -1, "completeness": -1, "faithfulness": -1, "comment": f"解析失败: {text[:100]}"}
    except Exception:
        return {"accuracy": -1, "completeness": -1, "faithfulness": -1, "comment": "解析异常"}


# ============================================================================
# 第3步：运行评估
# ============================================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           📊 RAG 效果评估                                  ║
║                                                              ║
║  10 个问答对 × 3 种评估方法                                 ║
╚══════════════════════════════════════════════════════════════╝
    """)

    rag = RAGPipeline(DOCUMENTS)

    rouge_scores = []
    keyword_scores = []

    for item in EVAL_DATASET:
        print(f"\n{'─'*55}")
        print(f"#{item['id']} [{item['type']}] {item['query']}")

        # 用 RAG 生成答案（不打印冗余信息）
        answer = rag.ask(item["query"], top_k=3, verbose=False)

        # 计算 ROUGE-L
        rouge = rouge_l_score(answer, item["ground_truth"])
        rouge_scores.append(rouge)

        # 计算关键词匹配
        kw = keyword_match_score(answer, item["ground_truth"])
        keyword_scores.append(kw)

        print(f"  预期: {item['ground_truth'][:60]}...")
        print(f"  实际: {answer[:80]}...")
        print(f"  ROUGE-L: {rouge:.3f}  |  关键词匹配: {kw:.3f}")

    # ====== 汇总报告 ======
    print(f"\n{'='*55}")
    print("📊 评估汇总")
    print(f"{'='*55}")

    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_keyword = sum(keyword_scores) / len(keyword_scores)

    # 按题型分组
    fact_scores = []
    simple_scores = []
    ood_scores = []
    for i, item in enumerate(EVAL_DATASET):
        if item["type"] == "事实提取":
            fact_scores.append(rouge_scores[i])
        elif item["type"] == "简单事实":
            simple_scores.append(rouge_scores[i])
        elif item["type"] == "知识库外":
            ood_scores.append(rouge_scores[i])

    print(f"整体 ROUGE-L 平均: {avg_rouge:.3f}")
    print(f"整体 关键词匹配 平均: {avg_keyword:.3f}")
    print(f"  事实提取类 (7题) ROUGE-L: {sum(fact_scores)/len(fact_scores):.3f}" if fact_scores else "")
    print(f"  简单事实类 (2题) ROUGE-L: {sum(simple_scores)/len(simple_scores):.3f}" if simple_scores else "")
    print(f"  知识库外类 (1题) ROUGE-L: {sum(ood_scores)/len(ood_scores):.3f}" if ood_scores else "")

    # 找出最低分
    worst_idx = rouge_scores.index(min(rouge_scores))
    print(f"\n⚠️ 最低分: #{EVAL_DATASET[worst_idx]['id']} '{EVAL_DATASET[worst_idx]['query']}' (ROUGE-L={rouge_scores[worst_idx]:.3f})")

    print(f"\n💡 评估指标说明：")
    print(f"  ROUGE-L > 0.5: 回答和标准答案高度一致")
    print(f"  ROUGE-L 0.2-0.5: 回答包含核心信息但表述差异大")
    print(f"  ROUGE-L < 0.2: 可能遗漏关键信息或表述偏离较大")
    print(f"  注意：ROUGE-L 只看字面，'月费29元' vs '每月收费29元' 得分会低但语义正确")
    print(f"  更可靠的评估需要用 LLM 打分（取消注释 llm_evaluate 即可）")
