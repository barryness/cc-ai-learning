# MCP 学习

> 从 Tool Calling 到标准化协议 —— 构建 LLM 可调用的工具服务。

## 文件结构

```
008-mcp-learning/
├── mcp_guide.md        ← 完整教学指南（11 章：Tool Calling → MCP 概念 → Claude Code 集成）
├── mcp_server.py       ← MCP Server 实现（文件系统 + SQLite，含自测模式）
└── README.md           ← 你在这里
```

## 快速开始

```bash
cd 008-mcp-learning

# 自测模式（模拟 LLM 调用所有工具）
python mcp_server.py --demo

# 启动 Server（stdio 模式，供 Claude Code 连接）
python mcp_server.py
```

## Server 提供的工具

| 工具 | 类别 | 功能 |
|------|------|------|
| `list_directory` | 文件系统 | 列出目录内容 |
| `read_file` | 文件系统 | 读取文件（多编码支持） |
| `search_files` | 文件系统 | 通配符搜索文件 |
| `write_file` | 文件系统 | 写入文件 |
| `db_list_tables` | SQLite | 列出数据库表 |
| `db_describe_table` | SQLite | 查看表结构 |
| `db_query` | SQLite | 执行 SELECT 查询（只读） |
| `db_export_table` | SQLite | 导出表数据为 JSON |

## 连接 Claude Code

在项目根目录或 `~/.claude/settings.json` 中添加：

```json
{
  "mcpServers": {
    "local-tools": {
      "command": "python3",
      "args": ["/path/to/008-mcp-learning/mcp_server.py"],
      "env": {
        "WORKSPACE_DIR": "/path/to/your/workspace",
        "DB_PATH": "/path/to/your/database.db"
      }
    }
  }
}
```

## 学习路线

1. 阅读 `mcp_guide.md` — 理解从 Tool Calling 到 MCP 的演进
2. 运行 `python mcp_server.py --demo` — 查看 8 个工具的实际效果
3. 阅读 `mcp_server.py` 源码 — 理解 Server 的实现细节
4. 配置 Claude Code 连接 — 在对话中实际使用
5. 扩展：添加你自己的工具

## 前置知识

- `006-langchain-learning` — 理解 Tool Calling 和 @tool 装饰器

## 资源

- [MCP 官方文档](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
