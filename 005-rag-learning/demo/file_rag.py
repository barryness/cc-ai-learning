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
from pathlib import Path              # Path: 面向对象的路径操作，比字符串拼接更安全
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
    # Path() 把字符串路径变成 Path 对象，后续可以直接 .exists() .glob() .stem 等
    # 比 os.path 系列函数更直观，是 Python 3.4+ 推荐的做法
    base = Path(data_dir)

    # .exists() 检查目录是否存在，避免后续 open() 报错
    if not base.exists():
        print(f"⚠️ 目录 '{data_dir}' 不存在，使用默认知识库")
        return _default_documents()

    # .glob("*.txt") 用通配符匹配文件名，类似命令行的 ls *.txt
    # 返回值是生成器（generator），可以用 sorted() 转为排序后的列表
    # sorted() 按文件名字母排序，保证每次加载顺序一致
    files = sorted(base.glob("*.txt")) + sorted(base.glob("*.md"))
    if not files:
        print(f"⚠️ 目录 '{data_dir}' 下没有 .txt/.md 文件，使用默认知识库")
        return _default_documents()

    documents = []
    for filepath in files:
        # with open(...) as f: 是 Python 的上下文管理器
        # 好处：代码块结束后自动关闭文件，即使中途抛异常也会关
        # encoding="utf-8" 显式指定编码，避免中文乱码
        with open(filepath, "r", encoding="utf-8") as f:
            # .read() 读取文件全部内容，返回一个字符串
            # .strip() 去掉首尾的空白符（空格、换行、制表符等）
            raw = f.read().strip()

        if not raw:           # 跳过空文件（空字符串在 Python 里是 False）
            continue

        # .split("\n", 1) 按换行符切割，只切 1 次
        # 例："标题\n内容第一行\n内容第二行" → ["标题", "内容第一行\n内容第二行"]
        # maxsplit=1 保证只有第一行被切出来，正文里的换行符保留
        lines = raw.split("\n", 1)
        first_line = lines[0].strip()     # .strip() 去掉首行首尾空白

        # .startswith("# ") 判断字符串是否以指定前缀开头
        if first_line.startswith("# "):       # "# 标题" → 去掉前2个字符
            title = first_line[2:].strip()    # 切片 [2:] 从第3个字符取到末尾
        elif first_line.startswith("#"):      # "#标题" → 去掉前1个字符
            title = first_line[1:].strip()
        else:
            # filepath.stem: Path 对象属性，返回"不带后缀的文件名"
            # 例：Path("data/公司介绍.md").stem → "公司介绍"
            title = filepath.stem

        # lines[1] 是 split 后的第二部分（正文），如果文件只有一行则用空字符串
        content = lines[1].strip() if len(lines) > 1 else ""
        if not content:
            content = first_line  # 文件只有一行时，整行作为内容

        documents.append({"title": title, "content": content})
        # filepath.name: Path 对象属性，返回"文件名.后缀"
        # len(content): Python 内置函数，返回字符串长度（字符数，不是字节数）
        print(f"  加载: {filepath.name} → 标题='{title}' ({len(content)}字)")

    return documents


def _default_documents():
    """
    备用知识库——当 data/ 目录不存在时使用。

    函数名以 _ 开头是 Python 约定：表示"模块内部使用，外部不要直接调用"。
    不是强制的，但是社区通用惯例。
    """
    return [
        {"title": "公司介绍", "content": "AI学习科技有限公司成立于2024年，总部位于北京。"},
        {"title": "产品介绍", "content": "公司旗舰产品AI Tutor支持20+编程语言，月费29元。"},
    ]


# ============================================================================
# 第2步：运行
# ============================================================================
# if __name__ == "__main__" 是 Python 的入口守卫
# 当直接运行 python3 file_rag.py 时，__name__ 的值是 "__main__"，执行下面的代码
# 当被 import 导入时，__name__ 是 "file_rag"，不执行
# 这样这个文件既可以独立运行，也可以被其他脚本 import 复用函数
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
