"""
=============================================================================
  最小 MCP Server Demo —— 理解 Tool Calling
=============================================================================

MCP（Model Context Protocol）= LLM 和外部工具的通信协议
一句话：让 LLM 能够"调用函数"，而不仅仅是"生成文字"

架构：
  ┌──────────┐   Tool Call 请求    ┌──────────────┐
  │  LLM     │ ──────────────────▶ │  MCP Server  │
  │ (Client) │ ◀────────────────── │  (Tools)     │
  └──────────┘   返回执行结果       └──────────────┘

本 Demo 包含两个 Tool：
  1. get_weather —— 获取城市天气（模拟数据）
  2. calculate  —— 执行数学表达式（真实计算）

运行方式：
  Terminal 1: python3 server_hello.py           （启动 Server，等待连接）
  Terminal 2: python3 client_demo.py             （启动 Client，调用 Tool）
"""

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# ============================================================================
# 第1步：创建 MCP Server
# ============================================================================
# name: 给 Server 起个名字，LLM 会看到这个名字
# version: 版本号
server = Server("hello-world-mcp-server")


# ============================================================================
# 第2步：声明 Tool 列表
# ============================================================================
# LLM 通过这个函数知道 Server 提供了哪些功能
# 返回值是 Tool 对象列表，每个 Tool 描述了一个"可调用的函数"
@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    LLM 会先调用这个函数，拿到所有可用工具的"菜单"。
    然后根据用户问题，决定调用哪个工具。
    """
    return [
        Tool(
            name="get_weather",
            description="获取指定城市的天气信息",
            # inputSchema 告诉 LLM：调用这个工具需要传什么参数
            # 格式是 JSON Schema —— LLM 会按照这个 schema 生成参数
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'",
                    }
                },
                "required": ["city"],  # city 是必填参数
            },
        ),
        Tool(
            name="calculate",
            description="执行数学表达式计算，支持 + - * / () 等运算符",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2+3*4'、'(10+5)/3'",
                    }
                },
                "required": ["expression"],
            },
        ),
    ]


# ============================================================================
# 第3步：实现 Tool 调用逻辑
# ============================================================================
# 当 LLM 决定调用某个 Tool 时，Server 收到 (name, arguments)，
# 执行对应的逻辑，返回结果
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    每个 Tool 的实际执行逻辑都在这里。

    参数：
      name:      LLM 选择调用的工具名称（如 "get_weather"）
      arguments: LLM 根据 inputSchema 生成的参数（如 {"city": "北京"}）
    """
    if name == "get_weather":
        city = arguments["city"]
        # 模拟天气数据——真实场景这里会调天气 API
        weather_data = {
            "北京": {"温度": "25°C", "天气": "晴", "湿度": "45%"},
            "上海": {"温度": "28°C", "天气": "多云", "湿度": "65%"},
            "深圳": {"温度": "32°C", "天气": "雷阵雨", "湿度": "80%"},
        }
        info = weather_data.get(city, {"温度": "未知", "天气": "无数据", "湿度": "无数据"})

        # 必须以 TextContent 列表的形式返回
        return [TextContent(
            type="text",
            text=f"{city}天气：{info['天气']}，温度{info['温度']}，湿度{info['湿度']}",
        )]

    elif name == "calculate":
        expr = arguments["expression"]
        try:
            # eval 在生产环境不能直接用（代码注入风险）
            # 这里为了 Demo 简洁使用，生产级应使用 ast.literal_eval 或安全沙箱
            result = eval(expr, {"__builtins__": {}}, {})
            return [TextContent(
                type="text",
                text=f"{expr} = {result}",
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"计算错误：{str(e)}",
            )]

    else:
        return [TextContent(type="text", text=f"未知工具：{name}")]


# ============================================================================
# 第4步：启动 Server（stdio 传输）
# ============================================================================
# stdio = 标准输入输出（stdin/stdout）
# 这是 MCP 最基础的传输方式——Server 通过 stdin 接收 JSON-RPC 请求，
# 通过 stdout 返回结果。就像管道通信。
async def main():
    """
    stdio_server() 返回 (read_stream, write_stream)
    server.run() 在后台持续监听，处理来自 Client 的请求
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
