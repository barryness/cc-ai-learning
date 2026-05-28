# MCP 学习记录

## 2026-05-28

### 学习过程

**Q: 请从零开始教我MCP。要求：用程序员视角、必须包含实际场景、必须实现一个MCP Server、必须解释Tool Calling、最后实现一个连接本地文件系统的MCP。**

### 核心概念理解

#### MCP 是什么？

MCP（Model Context Protocol）= LLM 和外部工具的通信协议。一句话：让 LLM 能够"调用函数"，而不仅仅是"生成文字"。

传统 LLM 的局限：
- 不能查实时数据（天气、股价、数据库）
- 不能执行操作（发邮件、创建文件、调用 API）
- 知识截止训练日期

MCP 的解决方案：定义一个标准协议，LLM（Client）通过它发现和调用外部工具（Server）。

#### 架构

```
┌──────────────┐   JSON-RPC (stdio)   ┌──────────────────┐
│  LLM Host    │ ◄──────────────────► │  MCP Server      │
│  (Claude等)  │   请求/响应           │  (工具提供方)     │
└──────────────┘                      └──────────────────┘

通信方式：stdin/stdout 管道，JSON-RPC 2.0 协议
```

#### 三个角色

| 角色 | 做什么 | 举例 |
|------|--------|------|
| MCP Host | 运行 LLM 的应用 | Claude Desktop、Cursor |
| MCP Client | Host 内部的协议客户端 | 连接 Server、管理 Tool 列表 |
| MCP Server | 提供具体功能的工具服务 | 天气查询、文件操作、数据库 |

#### Tool Calling 是什么？

LLM 调用工具的完整流程：
1. **初始化**：Client 连接 Server，Server 返回工具列表（Tool 对象数组）
2. **Tool 声明**：每个 Tool 包含 name（工具名）、description（功能描述）、inputSchema（参数 JSON Schema）
3. **LLM 决策**：LLM 根据用户问题 + 工具描述，选择最合适的 Tool，按 inputSchema 生成参数 JSON
4. **调用执行**：Client 将 (tool_name, arguments) 发给 Server，Server 执行并返回结果
5. **结果整合**：LLM 将工具返回结果融入最终回复

### Demo 实现

#### Demo 1：Hello MCP Server

实现了两个文件：
- `demo/server_hello.py`（~150行）：最小 MCP Server
  - 2 个 Tool：`get_weather`（模拟天气）、`calculate`（数学计算）
  - 使用 `@server.list_tools()` 声明工具菜单
  - 使用 `@server.call_tool()` 实现调用逻辑
  - `inputSchema` 按 JSON Schema 格式定义参数

- `demo/client_demo.py`（~140行）：MCP Client
  - `MCPToolCaller` 类封装连接、列出工具、调用工具、关闭
  - `decide_tool_call()` 模拟 LLM 的工具选择决策
  - 核心 API：`StdioServerParameters` → `stdio_client()` → `ClientSession`

#### Demo 2：文件系统 MCP Server

实现了 `filesystem-server/server.py`（~250行）：
- 4 个 Tool：`list_directory`、`read_file`、`write_file`、`search_files`
- 安全设计：`safe_path()` 防止路径穿越攻击（`../../etc/passwd`）
- 操作范围限制在 `~/mcp-workspace/`
- 文件大小限制 1MB

`filesystem-server/client_demo.py`（~85行）：
- 5 个场景演示：创建文件 → 查看目录 → 读取文件 → 搜索文件 → 安全测试

### 踩坑记录

#### Bug 1：MCP SDK 导入路径

**现象**：`from mcp import Client` 报错 `ModuleNotFoundError`

**根因**：mcp 包的公开 API 路径和直觉不同。

**修复**：
```python
# 正确的导入
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
```

#### Bug 2：filesystem server 的 `self._fmt_size()` 错误

**现象**：`list_directory` 报错 `name 'self' is not defined`

**根因**：`call_tool()` 是独立的 async 函数（不是类方法），但内部使用了 `self._fmt_size(size)`。`_fmt_size` 是模块级函数，`self` 不存在。

**修复**：`self._fmt_size(size)` → `_fmt_size(size)`

**教训**：在 `@server.call_tool()` 装饰的函数内，没有 `self`。这和 `@server.list_tools()` 一样都是模块级函数。

### Session 总结

本次 MCP 入门完成了：
1. 理解 MCP 协议架构（Host/Client/Server 三角色）
2. 理解 Tool Calling 的完整流程（5 步）
3. 实现 Hello MCP Server（get_weather + calculate）
4. 实现 MCP Client（模拟 LLM 决策）
5. 实现文件系统 MCP Server（4 个 Tool + 安全防护）
6. 发现并修复 2 个 Bug
