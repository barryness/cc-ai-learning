"""
=============================================================================
  文件版 RAG Demo — 从真实文件构建知识库
=============================================================================

与 minimal_rag.py 的唯一区别：知识库不是硬编码的，而是从 data/ 目录读取文件。

运行方式：
  python3 file_rag.py
"""

import os
import glob
from pathlib import Path
from minimal_rag import RAGPipeline


# ============================================================================
# 第1步：从文件系统加载文档
# ============================================================================
def load_documents_from_dir(data_dir: str = "data") -> list[dict]:
    """
    从目录中读取所有 .txt / .md 文件，每篇文档解析为 {title, content}。

    解析规则：
      - 第一行如果以 # 开头，去掉 # 作为 title
      - 其余内容作为 content
      - 文件名本身不参与解析（真实场景文档标题应该在内容里）

    参数：
      data_dir: 文档目录路径
    返回：
      [{"title": "公司介绍", "content": "AI学习科技..."}, ...]
    """
    base = Path(data_dir)
    if not base.exists():
        print(f"⚠️ 目录 '{data_dir}' 不存在，使用默认知识库")
        return _default_documents()

    # 收集所有 .txt 和 .md 文件
    files = sorted(base.glob("*.txt")) + sorted(base.glob("*.md"))
    if not files:
        print(f"⚠️ 目录 '{data_dir}' 下没有 .txt/.md 文件，使用默认知识库")
        return _default_documents()

    documents = []
    for filepath in files:
        # 读取文件全部内容
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            continue

        # 解析：第一行是标题，其余是正文
        lines = raw.split("\n", 1)
        first_line = lines[0].strip()

        # 去掉 markdown 的 # 前缀
        if first_line.startswith("# "):
            title = first_line[2:].strip()
        elif first_line.startswith("#"):
            title = first_line[1:].strip()
        else:
            title = filepath.stem  # 无标题时用文件名

        content = lines[1].strip() if len(lines) > 1 else ""
        if not content:
            content = first_line  # 只有一行时，整行作为内容

        documents.append({"title": title, "content": content})
        print(f"  加载: {filepath.name} → 标题='{title}' ({len(content)}字)")

    return documents


def _default_documents():
    """备用知识库——当 data/ 目录不存在时使用"""
    return [
        {"title": "公司介绍", "content": "AI学习科技有限公司成立于2024年，总部位于北京。"},
        {"title": "产品介绍", "content": "公司旗舰产品AI Tutor支持20+编程语言，月费29元。"},
    ]


# ============================================================================
# 第2步：运行
# ============================================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           📁 文件版 RAG Demo                               ║
║                                                              ║
║  知识库来源：data/ 目录下的 .txt / .md 文件                   ║
║  检索+生成：复用 minimal_rag.py 的 RAGPipeline               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 加载文档
    print("📂 加载文档：")
    documents = load_documents_from_dir("data")
    print(f"✅ 共加载 {len(documents)} 篇文档\n")

    # 喂给 RAG Pipeline
    rag = RAGPipeline(documents)

    test_queries = [
        "公司混合办公的政策是什么？",
        "公司融资了多少钱？投资方有哪些？",
        "AI Tutor的后端技术栈是什么？",
    ]

    for query in test_queries:
        rag.ask(query, top_k=3)
