"""
=============================================================================
  MCP Client Demo —— 模拟 LLM 如何发现和调用 Tool
=============================================================================

这个 Client 做了 LLM（如 Claude Desktop）做的事情：
  1. 连接 MCP Server
  2. 拿到"工具菜单"（list_tools）
  3. 调用具体 Tool（call_tool）
  4. 拿到结果返回给用户

运行方式（两个终端）：
  Terminal 1: cd mcp-learning/demo && python3 server_hello.py
  Terminal 2: cd mcp-learning/demo && python3 client_demo.py
"""

import asyncio
import json
import re
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# ============================================================================
# MCP Client 封装
# ============================================================================
class MCPToolCaller:
    """
    封装 MCP Client 的完整交互流程。

    内部流程和 Claude Desktop 连接 MCP Server 完全一样：
      connect → list_tools → call_tool → close
    """

    def __init__(self, command: str, args: list[str]):
        self.command = command
        self.args = args

    async def connect(self):
        # stdio_client 启动 Server 子进程，通过 stdin/stdout 管道通信
        # StdioServerParameters: 封装启动参数（命令、参数、环境变量等）
        params = StdioServerParameters(command=self.command, args=self.args)
        self._ctx = stdio_client(params)
        read, write = await self._ctx.__aenter__()

        # ClientSession: 在管道之上封装 JSON-RPC 协议
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()  # 握手：交换协议版本和能力

    async def list_tools(self) -> list:
        """向 Server 请求工具列表。返回 Tool 对象列表。"""
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用一个具体的 Tool。返回文本结果。"""
        result = await self.session.call_tool(name, arguments)
        return result.content[0].text

    async def close(self):
        await self.session.__aexit__(None, None, None)
        await self._ctx.__aexit__(None, None, None)


# ============================================================================
# 模拟 LLM 的 Tool Calling 决策
# ============================================================================
def decide_tool_call(user_query: str, available_tools) -> tuple[str | None, dict]:
    """
    模拟 LLM 的决策：根据用户问题和可用工具列表，决定调用哪个工具、传什么参数。

    真实的 LLM 流程：
      1. 读取每个 Tool 的 name + description + inputSchema
      2. 理解用户意图 → 匹配到最合适的 Tool
      3. 根据 inputSchema 生成正确的参数 JSON
    """
    print(f"\n🧠 模拟 LLM 决策：")
    print(f"   用户问题：{user_query}")
    print(f"   可用工具：{[t.name for t in available_tools]}")

    for tool in available_tools:
        if tool.name == "get_weather":
            for city in ["北京", "上海", "深圳", "杭州", "广州"]:
                if city in user_query:
                    print(f"   → 决定调用 {tool.name}(city='{city}')")
                    return ("get_weather", {"city": city})

        elif tool.name == "calculate":
            if any(w in user_query.lower() for w in ["计算", "等于", "多少", "+", "-", "*", "/"]):
                expr_match = re.search(r'[\d\+\-\*/\(\)\.\s]+', user_query)
                if expr_match:
                    expr = expr_match.group().strip()
                    print(f"   → 决定调用 {tool.name}(expression='{expr}')")
                    return ("calculate", {"expression": expr})

    print(f"   → 没有匹配的工具")
    return (None, {})


# ============================================================================
# 运行
# ============================================================================
async def main():
    caller = MCPToolCaller("python3", ["server_hello.py"])

    print("🔌 连接到 MCP Server...")
    await caller.connect()
    print("✅ 连接成功！\n")

    # ---- 步骤1：获取工具菜单 ----
    tools = await caller.list_tools()
    print("📋 Server 提供的工具：")
    for tool in tools:
        props = tool.inputSchema.get("properties", {})
        params = ", ".join(f"{k}: {v.get('type','?')}" for k, v in props.items())
        print(f"   🔧 {tool.name}({params})")
        print(f"      {tool.description}\n")

    # ---- 步骤2：模拟多轮调用 ----
    queries = [
        "北京今天天气怎么样？",
        "计算 25 + 3 * 8 等于多少",
        "上海的天气如何？",
        "帮我算一下 (100 + 50) / 3",
    ]

    for query in queries:
        tool_name, args = decide_tool_call(query, tools)
        if tool_name:
            print(f"📞 调用 {tool_name}({json.dumps(args, ensure_ascii=False)})")
            result = await caller.call_tool(tool_name, args)
            print(f"📤 结果：{result}\n")
        else:
            print(f"⚠️ 无法处理：{query}\n")

    await caller.close()
    print("🔌 连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
