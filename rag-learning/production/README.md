# 生产级 RAG 架构 Demo

## 与最小 Demo 的核心区别

| 维度 | 最小 Demo | 生产级 |
|------|----------|--------|
| 检索 | TF-IDF | Embedding 模型 (sentence-transformers) |
| 存储 | 内存数组 | 向量数据库 (ChromaDB) |
| 分块 | 整篇文档 | 语义分块 (Semantic Chunking) |
| 检索策略 | 单轮向量检索 | 混合检索 + 重排序 |
| 错误处理 | 无 | 完整异常处理 |
| 日志 | print | logging 模块 |

## 运行

```bash
cp ../demo/.env.example .env  # 填入 API Key
pip install -r requirements.txt
python advanced_rag.py
```
