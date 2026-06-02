# MCP 学习笔记：从 Tool Calling 到标准化协议

> 从"LLM 怎么调用外部函数"这个真实问题出发。
> 不讲协议规范，只讲为什么和怎么做。

---

## 目录

1. [从 Tool Calling 开始](#1-从-tool-calling-开始)
2. [MCP 是什么](#2-mcp-是什么)
3. [为什么出现 MCP](#3-为什么出现-mcp)
4. [核心概念：MCP Client](#4-核心概念mcp-client)
5. [核心概念：MCP Server](#5-核心概念mcp-server)
6. [核心概念：Tool](#6-核心概念tool)
7. [核心概念：Resource](#7-核心概念resource)
8. [核心概念：Prompt](#8-核心概念prompt)
9. [实战：构建 MCP Server（文件系统 → SQLite）](#9-实战构建-mcp-server文件系统--sqlite)
10. [连接 Claude Code](#10-连接-claude-code)
11. [MCP vs Function Calling vs LangChain Tool](#11-mcp-vs-function-calling-vs-langchain-tool)

---

## 1. 从 Tool Calling 开始

### 你已经会了什么

在 `006-langchain-learning` 里，你掌握了 Tool Calling：

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}: 晴, 25°C"

# LLM 自动决定何时调用、传什么参数
llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("北京天气怎么样？")
# → LLM 返回 tool_calls: [{"name": "get_weather", "args": {"city": "北京"}}]
```

### 但这里有一个根本问题

**工具和 LLM 是"绑定"在一起的。** 工具定义写在代码里，换了 LLM 平台就要重新定义。

```
OpenAI 的 Tool Calling 格式:
  tools=[{"type": "function", "function": {"name": "...", "parameters": {...}}}]

Anthropic 的 Tool Use 格式:
  tools=[{"name": "...", "description": "...", "input_schema": {...}}]

Google Gemini 的 Function Calling 格式:
  tools=[{"functionDeclarations": [{"name": "...", "parameters": {...}}]}]
```

**每个平台都有自己的 Tool Calling 格式！** 如果你写了 10 个工具，想在 OpenAI、Anthropic、Gemini 三个平台上用——你就要维护 3 套工具定义。

### MCP 的答案：把工具变成"服务"

```
传统 Tool Calling:                     MCP:
                          
  App ─── LLM ─── Tool₁                  App ─── LLM ─── MCP Client ─── MCP Server
         │        Tool₂                                   │              ├── Tool₁
         │        Tool₃                   标准协议         │              ├── Tool₂
         └────────Tool₄                  (JSON-RPC)        │              └── Tool₃
                                                           │
  工具和 LLM 绑定在一起                   工具变成独立的"服务"，LLM 通过协议发现
```

---

## 2. MCP 是什么

### 一句话

**MCP（Model Context Protocol）= LLM 和外部工具的"USB-C 接口"。**

USB-C 出现之前，每种设备有自己的接口。USB-C 之后，一个接口通吃。MCP 对 LLM 工具调用做了一样的事——定义了统一的协议，让任何 LLM 都能发现和调用任何工具。

### 三角色架构

```
┌──────────────────────────────────────┐
│  MCP Host                             │  ← AI 应用（Claude Desktop、Cursor 等）
│  ┌────────────────────────────────┐   │
│  │  MCP Client                     │  │  ← 协议客户端，管理连接和 Tool 列表
│  │  • 连接 Server                  │  │
│  │  • 发现 Tool/Resource/Prompt    │  │
│  │  • 调用 Tool                    │  │
│  └────────────┬───────────────────┘   │
└───────────────┼───────────────────────┘
                │ JSON-RPC over stdio/SSE
┌───────────────┼───────────────────────┐
│  ┌────────────┴───────────────────┐   │
│  │  MCP Server                     │  │  ← 工具提供方
│  │  • 声明 Tool/Resource/Prompt    │  │
│  │  • 执行 Tool 调用               │  │
│  │  • 返回结果                     │  │
│  └────────────────────────────────┘   │
└───────────────────────────────────────┘
```

| 角色 | 做什么 | 举例 |
|------|--------|------|
| **MCP Host** | 运行 LLM 的应用 | Claude Desktop、Cursor、VS Code Copilot |
| **MCP Client** | Host 内部的协议客户端 | 连接 Server、管理 Tool 列表、转发调用 |
| **MCP Server** | 提供具体功能的服务 | 天气查询、文件操作、数据库访问、API 网关 |

### 类比

```
MCP Host   = 笔记本电脑
MCP Client = USB-C 接口（内置于电脑）
MCP Server = 外接设备（U盘、显示器、充电器...）

换一台电脑（Claude → Gemini），只要都有 USB-C（MCP 协议），
外接设备（你的 MCP Server）照常工作。
```

### 通信方式

```
MCP Host 启动 MCP Server 作为子进程
         │
         ├── stdin  → JSON-RPC 请求（list_tools, call_tool...）
         │
         └── stdout ← JSON-RPC 响应（Tool 列表、执行结果...）

协议格式：JSON-RPC 2.0
传输方式：stdio（本地）或 SSE（远程）
```

### 两种传输模式

| 模式 | 传输 | 场景 |
|------|------|------|
| **stdio** | stdin/stdout 管道 | 本地工具（文件系统、数据库、CLI） |
| **SSE** | HTTP Server-Sent Events | 远程工具（API 网关、云服务） |

---

## 3. 为什么出现 MCP

### 没有 MCP 之前的世界

```
你写了 10 个工具（天气、股票、文件、数据库、邮件...）

想在 Claude Desktop 用 → 按 Anthropic 格式定义 10 个 Function
想在 ChatGPT 用      → 按 OpenAI 格式定义 10 个 Function  
想在 Gemini 用       → 按 Google 格式定义 10 个 Function
想在 Cursor 用       → 按 Cursor 的格式再定义一遍

维护成本 = 工具数 × 平台数
```

### MCP 解决的核心问题

| 问题 | MCP 之前 | MCP 之后 |
|------|---------|---------|
| **工具定义** | 每个平台一种格式 | 一次定义，所有平台通用 |
| **工具部署** | 嵌入在应用代码里 | 独立进程，即插即用 |
| **工具发现** | 硬编码在代码里 | Client 自动发现 Server 提供的工具 |
| **工具复用** | 每个项目重新写 | 一个 Server 多个项目共用 |
| **安全隔离** | 和 LLM 同进程 | 独立进程，权限隔离 |

### 谁在推动 MCP

MCP 由 Anthropic 于 2024 年 11 月发布，作为一个开放协议。目前：

- **Claude Desktop** 原生支持 MCP（通过配置文件 `claude_desktop_config.json`）
- **Cursor** 支持 MCP
- **VS Code Copilot** 支持 MCP
- **Claude Code**（你正在用的）支持 MCP
- 社区已有数百个 MCP Server（文件系统、数据库、GitHub、Slack...）

---

## 4. 核心概念：MCP Client

### 是什么

**MCP Client 是 LLM 和 MCP Server 之间的"翻译官"。** 它运行在 MCP Host 内部，负责：

1. **连接管理**：启动 Server 进程，建立 stdio 管道
2. **协议握手**：交换协议版本和能力（initialize）
3. **工具发现**：调用 `list_tools()` 获取可用工具列表
4. **工具调用**：将 LLM 的 tool_call 转为 JSON-RPC 请求发给 Server
5. **结果返回**：将 Server 返回的结果传给 LLM

### 工作流程

```
用户: "北京天气怎么样？"
         │
         ▼
    ┌─────────┐
    │   LLM   │  分析意图 → 需要调用 get_weather 工具
    └────┬────┘
         │ tool_calls: [{"name": "get_weather", "args": {"city": "北京"}}]
         ▼
    ┌──────────────┐
    │  MCP Client  │  转发 → JSON-RPC: call_tool("get_weather", {"city": "北京"})
    └──────┬───────┘
           │ stdio (stdin)
           ▼
    ┌──────────────┐
    │  MCP Server  │  执行 get_weather("北京") → "北京: 晴, 25°C"
    └──────┬───────┘
           │ stdio (stdout)
           ▼
    ┌──────────────┐
    │  MCP Client  │  解析结果 → 返回给 LLM
    └──────┬───────┘
           │
           ▼
    ┌─────────┐
    │   LLM   │  整合结果 → "北京今天天气晴朗，温度 25°C"
    └─────────┘
```

### 在 Claude Code 中

Claude Code 内置了 MCP Client。你在 `CLAUDE.md` 或全局配置中声明 MCP Server，Claude Code 自动管理连接和工具列表。

---

## 5. 核心概念：MCP Server

### 是什么

**MCP Server 是一个独立的进程，提供一组 Tool/Resource/Prompt。** 它是 LLM 的"外接设备"。

### Server 的最小结构

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# 1. 创建 Server
server = Server("my-server")

# 2. 声明 Tool 列表（LLM 通过这个知道 Server 能做什么）
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [...]

# 3. 实现 Tool 调用（收到调用后执行）
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    ...

# 4. 启动 Server（监听 stdin/stdout）
async with stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream, ...)
```

### Server 提供三种能力

| 能力 | 用途 | 类比 |
|------|------|------|
| **Tool** | 让 LLM 执行操作 | 函数调用（查天气、读文件、发邮件） |
| **Resource** | 让 LLM 读取数据 | 文件读取（配置文件、文档、数据库记录） |
| **Prompt** | 给 LLM 提供提示词模板 | 预定义的 Prompt 模板（"帮我写周报"） |

### Server 的声明周期

```
1. 启动   → MCP Host 启动 Server 子进程
2. 初始化 → Client 和 Server 握手，交换能力列表
3. 发现   → Client 调用 list_tools() / list_resources() / list_prompts()
4. 服务   → Client 调用 call_tool() / read_resource() / get_prompt()
5. 关闭   → Host 关闭 Server 子进程
```

---

## 6. 核心概念：Tool

### 是什么

**Tool = LLM 可以调用的"函数"。** 这是 MCP 最核心的概念。每个 Tool 有三个要素：

```python
Tool(
    name="get_weather",           # 1. 名称：LLM 用来标识
    description="获取指定城市的天气信息",  # 2. 描述：LLM 用来理解"什么时候用"
    inputSchema={                 # 3. 参数 Schema：LLM 用来生成"正确的参数"
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如'北京'、'上海'",
            }
        },
        "required": ["city"],
    },
)
```

### 三要素的重要性

```
name 不对  → LLM 找不到这个工具
description 不好 → LLM 不知道该不该用
inputSchema 不准 → LLM 生成错误的参数 → 调用失败
```

**description 是最关键的部分。** LLM 靠描述来决定"什么时候调用这个工具"。描述要写清楚：
- 这个工具做什么
- 什么时候用（场景）
- 参数的含义

### Tool 的执行

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """LLM 调工具时，Server 收到 (name, arguments)，执行并返回结果"""
    if name == "get_weather":
        city = arguments["city"]
        weather = weather_api.query(city)  # 实际逻辑
        return [TextContent(type="text", text=f"{city}: {weather}")]
```

### 返回值必须是 `list[TextContent]`

```python
# ✅ 正确
return [TextContent(type="text", text="结果")]

# ✅ 也可以返回多条
return [
    TextContent(type="text", text="摘要：..."),
    TextContent(type="text", text="详情：..."),
]

# ❌ 错误：不能直接返回字符串
return "结果"
```

---

## 7. 核心概念：Resource

### 是什么

**Resource = LLM 可以"读取"的数据源。** 和 Tool 的区别：

| | Tool | Resource |
|------|------|------|
| 语义 | **做**事情（执行操作） | **读**数据（获取信息） |
| 副作用 | 有（写文件、发邮件、删记录） | 无（只读） |
| 参数 | 每次调用时传入 | 通过 URI 标识 |
| 类比 | 函数调用 | 文件读取 |

### Resource 的定义

```python
from mcp.types import Resource, TextResourceContents

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="file:///project/readme.md",    # 唯一标识
            name="项目 README",                   # 人类可读名
            description="项目的说明文档",          # 描述
            mimeType="text/markdown",            # MIME 类型
        ),
        Resource(
            uri="db:///users/table",
            name="用户表结构",
            description="数据库 users 表的 Schema",
            mimeType="application/json",
        ),
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "file:///project/readme.md":
        return Path("readme.md").read_text()
    elif uri == "db:///users/table":
        return get_table_schema("users")
```

### 什么时候用 Resource vs Tool

```
需要执行操作（创建、修改、删除）→ Tool
  例：write_file、send_email、create_ticket

只需要读取数据（配置、文档、数据库）→ Resource
  例：读取 README、获取数据库 Schema、查看日志

简单判断：如果功能是"获取 X"且无副作用 → Resource
          如果功能是"做 X"且有副作用 → Tool
```

---

## 8. 核心概念：Prompt

### 是什么

**Prompt = 预定义的提示词模板。** 让 LLM 快速获取特定场景的提示词。

### Prompt 的定义

```python
from mcp.types import Prompt, PromptMessage, TextContent

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="write_weekly_report",
            description="生成周报的提示词模板",
            arguments=[
                {"name": "project", "description": "项目名称", "required": True},
                {"name": "highlights", "description": "本周亮点", "required": False},
            ],
        ),
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> list[PromptMessage]:
    if name == "write_weekly_report":
        project = arguments.get("project", "")
        highlights = arguments.get("highlights", "无")
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"请为项目'{project}'撰写周报。\n本周亮点：{highlights}\n\n"
                         f"请按以下结构输出：\n1. 本周完成\n2. 下周计划\n3. 风险与问题"
                ),
            ),
        ]
```

### Tool / Resource / Prompt 总结

```
            ┌─────────────────────────────────────────┐
            │           MCP Server                    │
            │                                         │
            │  Tools       Resources      Prompts      │
            │  (执行)       (读取)        (模板)        │
            │  ───────     ────────      ───────      │
            │  查天气       读配置文件    周报模板       │
            │  写文件       数据库Schema   代码审查模板   │
            │  发邮件       日志内容      提交信息模板    │
            │  调API        文档内容      发布说明模板    │
            │                                         │
            │  LLM 主动调用   LLM 被动读取   LLM 请求模板 │
            └─────────────────────────────────────────┘
```

---

## 9. 实战：构建 MCP Server（文件系统 → SQLite）

### 设计思路

我们将构建一个 MCP Server，分两个层次：

```
层次 1：文件系统工具（基础）
  - list_directory: 列出目录内容
  - read_file: 读取文件
  - search_files: 按名称搜索文件
  - write_file: 写入文件

层次 2：SQLite 数据库工具（扩展）
  - db_list_tables: 列出所有表
  - db_describe_table: 查看表结构
  - db_query: 执行 SELECT 查询（只读）
  - db_export_table: 导出表数据
```

### 架构

```
┌────────────────────────────────────────────┐
│  mcp_server.py                              │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  FileSystemTools                     │    │
│  │  • list_directory                    │    │
│  │  • read_file                         │    │
│  │  • search_files                      │    │
│  │  • write_file                        │    │
│  │  • 安全: safe_path() 路径校验         │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  SQLiteTools                         │    │
│  │  • db_list_tables                    │    │
│  │  • db_describe_table                 │    │
│  │  • db_query (SELECT only)            │    │
│  │  • db_export_table                   │    │
│  │  • 安全: 只读连接，禁止 INSERT/DROP   │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  传输: stdio (stdin/stdout)                  │
│  协议: JSON-RPC 2.0                          │
└────────────────────────────────────────────┘
```

### 关键设计决策

**1. 为什么文件系统和 SQLite 放在同一个 Server？**

演示 MCP Server 的"服务聚合"模式——一个 Server 可以提供多组不相关的工具。LLM 根据用户意图选择调用哪个工具。

**2. SQLite 为什么只读？**

MCP Server 的权限原则：最小权限。Server 只授予 LLM 完成任务所需的最小权限。如果 LLM 不需要写入数据库，就不要开放写权限。

**3. 路径安全：safe_path()**

```python
def safe_path(user_path: str, allowed_root: str) -> Path:
    """防止路径穿越攻击"""
    full = Path(allowed_root) / user_path
    resolved = full.resolve()  # 展开 ../ 和符号链接
    
    if not str(resolved).startswith(str(Path(allowed_root).resolve())):
        raise PermissionError(f"拒绝访问：'{user_path}' 超出允许范围")
    
    return resolved

# safe_path("../../../etc/passwd") → PermissionError!
```

### 完整代码

见 `mcp_server.py`。运行方式：

```bash
# 作为 MCP Server 运行（stdio 模式，供 Claude Code 等连接）
python mcp_server.py

# 自测模式（内置 Client 模拟 LLM 调用）
python mcp_server.py --demo
```

---

## 10. 连接 Claude Code

### 方式 1：全局配置（推荐）

在 `~/.claude/settings.json` 中添加：

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

### 方式 2：项目级配置

在项目根目录的 `.mcp.json` 中：

```json
{
  "mcpServers": {
    "local-tools": {
      "command": "python3",
      "args": ["mcp_server.py"]
    }
  }
}
```

### 验证连接

在 Claude Code 中执行：

```
/list_tools
```

如果看到 `list_directory`、`read_file`、`db_query` 等工具，说明连接成功。

### 使用示例

然后在对话中直接说：

```
"帮我看看 workspace 目录下有什么文件"
"读取 workspace/readme.md 的内容"
"查询 sales.db 数据库中有哪些表"
"帮我查询 orders 表中 2026 年的订单总数"
```

Claude Code 会自动：
1. 发现你的 MCP Server 提供的工具
2. 根据你的问题选择合适的工具
3. 调用工具获取结果
4. 将结果融入回答

---

## 11. MCP vs Function Calling vs LangChain Tool

### 对比表

| | OpenAI Function Calling | LangChain @tool | MCP |
|------|------|------|------|
| **范围** | 单次对话内的工具 | 单个应用内的工具 | 跨应用、跨平台的标准 |
| **定义方式** | JSON Schema | Python 装饰器 | Python 装饰器 + JSON Schema |
| **部署** | 嵌入代码 | 嵌入代码 | 独立进程 |
| **传输** | HTTP API | 内存调用 | stdio / SSE |
| **平台绑定** | 仅 OpenAI | LangChain 生态 | 所有支持 MCP 的平台 |
| **工具复用** | 每个项目重新定义 | 可复用但不跨平台 | 一次定义到处使用 |
| **安全隔离** | 无 | 无 | 进程级隔离 |

### 什么时候用哪个

```
单个 OpenAI 项目，1-2 个简单工具  → OpenAI Function Calling
LangChain 项目，多个 LLM 组件     → LangChain @tool
多平台、多项目、需要复用的工具     → MCP Server
```

### 它们可以共存

```
MCP Server 负责"提供工具"（独立进程）
LangChain @tool 负责"使用工具"（在一个 Node 内调用 MCP 工具）
OpenAI Function Calling 是 LLM 层面的"工具调用格式"（所有方案都依赖它）
```

---

## 总结

```
MCP 的核心价值：

  标准化    → 一次定义，所有 LLM 平台通用
  解耦      → 工具和 LLM 分离，独立开发部署
  复用      → 一个 Server 多个项目共用
  安全      → 进程级隔离，权限可控
  生态      → 数百个现成的 MCP Server 可直接使用

一句话：
  OpenAI 发明了 Function Calling（LLM 调工具的能力）
  Anthropic 发明了 MCP（工具调用的标准化协议）
  LangChain 提供了 @tool（工具定义的最佳开发体验）
  
  三者配合：用 @tool 开发 → 用 MCP 发布 → 任何 LLM 都能用
```

---

## 延伸阅读

- **运行 Demo**：`python mcp_server.py --demo` — 内置自测模式
- **启动 Server**：`python mcp_server.py` — stdio 模式，供 Claude Code 连接
- **MCP 官方文档**：[modelcontextprotocol.io](https://modelcontextprotocol.io)
- **MCP Python SDK**：[github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)
- **前置知识**：`006-langchain-learning` — 理解 Tool Calling 和 @tool 装饰器
