# MCP（Model Context Protocol）学习模块

## 一句话理解

MCP = LLM 的"函数调用协议"。让 LLM 不仅能"说"，还能"做"——查天气、读文件、调 API、操作数据库。

## 核心概念

### 为什么需要 MCP？

传统 LLM 的三大局限：
| 局限 | 说明 |
|------|------|
| 无法获取实时数据 | 不知道今天的天气、股价、新闻 |
| 无法执行操作 | 不能创建文件、发送邮件、调用 API |
| 知识有截止日期 | 训练数据是过去某个时间点的快照 |

MCP 的答案：定义一套标准协议，让 LLM 通过它发现和调用外部工具。

### 三角色架构

```
┌─────────────────┐
│   MCP Host       │  ← 运行 LLM 的应用（Claude Desktop、Cursor 等）
│  ┌─────────────┐ │
│  │ MCP Client  │ │  ← 协议客户端，管理连接和 Tool 列表
│  └──────┬──────┘ │
└─────────┼────────┘
          │ JSON-RPC over stdio
┌─────────┼────────┐
│  ┌──────┴──────┐ │
│  │ MCP Server  │ │  ← 工具提供方（天气 API、数据库、文件系统...）
│  └─────────────┘ │
└─────────────────┘
```

- **MCP Host**：AI 应用本身（Claude Desktop、Cursor 等）
- **MCP Client**：Host 内部的协议客户端，负责与 Server 通信
- **MCP Server**：提供具体功能的"工具服务"

### Tool Calling 五步流程

```
1. 初始化    Client 连接 Server，交换协议版本和能力
2. 发现工具  Server 返回 Tool 列表（name + description + inputSchema）
3. LLM 决策  LLM 根据用户意图选择 Tool，按 Schema 生成参数
4. 调用执行  Client 发送 (tool_name, arguments)，Server 执行返回结果
5. 结果整合  LLM 把工具返回结果融入最终回复
```

### Tool 的三要素

每个 Tool 必须声明：
- **name**：工具名称（LLM 用来标识）
- **description**：功能描述（LLM 用来理解**什么时候**调用）
- **inputSchema**：参数 JSON Schema（LLM 用来生成**正确的参数**）

```python
Tool(
    name="get_weather",
    description="获取指定城市的天气信息",
    inputSchema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"},
        },
        "required": ["city"],
    },
)
```

## 项目结构

```
mcp-learning/
├── README.md                   ← 本文件
├── demo/
│   ├── server_hello.py         ← 最小 MCP Server（2 个 Tool）
│   └── client_demo.py          ← MCP Client + 模拟 LLM 决策
├── filesystem-server/
│   ├── server.py               ← 文件系统 MCP Server（4 个 Tool + 安全防护）
│   ├── client_demo.py          ← 文件系统 Client Demo（5 个场景）
│   └── requirements.txt        ← 依赖（mcp>=1.0.0）
├── docs/
│   └── learning-log.md         ← 学习过程记录（踩坑、修复、总结）
└── exercises/
    └── (待添加)
```

## 快速开始

### 环境准备

```bash
# 激活虚拟环境
source ~/ai-learning-env/bin/activate

# 安装依赖
pip install mcp
```

### Demo 1：Hello MCP Server

```bash
# Terminal 1：启动 Server
cd mcp-learning/demo
python3 server_hello.py

# Terminal 2：运行 Client
cd mcp-learning/demo
python3 client_demo.py
```

Server 提供两个 Tool：
- `get_weather`：查询城市天气（模拟数据）
- `calculate`：执行数学表达式

### Demo 2：文件系统 MCP Server

```bash
# Terminal 1：启动 Server
cd mcp-learning/filesystem-server
python3 server.py

# Terminal 2：运行 Client
cd mcp-learning/filesystem-server
python3 client_demo.py
```

Server 提供四个 Tool：
- `list_directory`：列出目录内容
- `read_file`：读取文件
- `write_file`：写入文件
- `search_files`：按模式搜索文件

**安全机制**：
- 所有操作限制在 `~/mcp-workspace/` 目录
- 路径穿越攻击（`../../etc/passwd`）会被拦截
- 文件大小限制 1MB

## 关键 API

### Server 端

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

server = Server("my-server")

@server.list_tools()       # 声明 Tool 列表
async def list_tools() -> list[Tool]:
    ...

@server.call_tool()        # 处理 Tool 调用
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    ...

# 启动 Server
async with stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream, server.create_initialization_options())
```

### Client 端

```python
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# 连接 Server
params = StdioServerParameters(command="python3", args=["server.py"])
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # 列出工具
        tools = await session.list_tools()

        # 调用工具
        result = await session.call_tool("tool_name", {"arg": "value"})
```

## FAQ

**Q: MCP 和 Function Calling 有什么区别？**
A: Function Calling 是 OpenAI 定义的格式，LLM 在回复中直接输出 `{name, arguments}` JSON。MCP 是一个独立的中介服务，LLM 通过标准化协议发现和调用工具，与具体模型无关。

**Q: stdio 传输是什么意思？**
A: Server 进程通过 stdin 接收 JSON-RPC 请求，通过 stdout 返回结果。就像 Unix 管道 `echo 'request' | server`。好处是零网络配置、进程级隔离，适合本地工具。

**Q: 为什么 Tool 返回的是 `list[TextContent]` 而不是字符串？**
A: MCP 支持多模态内容。TextContent 是一种内容类型，未来可以扩展 ImageContent、EmbeddedResource 等。

**Q: pathlib 和 os.path 有什么区别？**
A: `pathlib` 是面向对象的路径处理库（Python 3.4+），`Path("/a/b") / "c"` 比 `os.path.join("/a/b", "c")` 更直观，推荐使用 pathlib。

## 学习路径

```
1. 理解 MCP 三角色架构
   ↓
2. 阅读 demo/server_hello.py（最小 Server 实现）
   ↓
3. 阅读 demo/client_demo.py（Client + LLM 决策模拟）
   ↓
4. 阅读 filesystem-server/server.py（生产级 Server + 安全设计）
   ↓
5. 做练习题（exercises/ 目录）
   ↓
6. 进阶：对接真实外部 API、数据库 MCP Server
```
