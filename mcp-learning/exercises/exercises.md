# MCP 练习题

5 道练习题，从修改现有 Server 到从零实现新 Server，逐步巩固 MCP 核心技能。

## 练习列表

| 编号 | 练习 | 难度 | 核心知识点 |
|------|------|------|-----------|
| 1 | 给文件系统 Server 添加 `delete_file` 工具 | ⭐ | Tool 声明 + 实现模式 |
| 2 | 添加 `get_file_info` 工具（文件元信息） | ⭐⭐ | Tool 设计 + 多字段返回 |
| 3 | 从零实现"备忘录" MCP Server | ⭐⭐⭐ | 完整 Server 开发 |
| 4 | MCP 错误处理最佳实践 | ⭐⭐ | 异常处理 + 用户体验 |
| 5 | 设计"好"的 Tool Description | ⭐⭐ | inputSchema 设计原则 |

## 预备知识

所有练习基于 `mcp-learning/filesystem-server/server.py` 的结构。核心模式：

```python
# 1. 在 list_tools() 中声明 Tool（name + description + inputSchema）
# 2. 在 call_tool() 中添加 if 分支实现逻辑
# 3. 返回 TextContent 列表
```
