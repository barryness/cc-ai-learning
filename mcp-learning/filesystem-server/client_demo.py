"""
=============================================================================
  文件系统 MCP Client Demo
=============================================================================

演示 LLM 通过 MCP 操作本地文件的完整流程。

运行方式（两个终端）：
  Terminal 1: cd mcp-learning/filesystem-server && python3 server.py
  Terminal 2: cd mcp-learning/filesystem-server && python3 client_demo.py

安全提示：Server 限制操作范围在 ~/mcp-workspace/ 目录下
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))
from client_demo import MCPToolCaller


async def main():
    # server.py 在同一个目录下
    caller = MCPToolCaller("python3", ["server.py"])

    print("🔌 连接到文件系统 MCP Server...")
    await caller.connect()
    print("✅ 连接成功！\n")

    # 查看可用工具
    tools = await caller.list_tools()
    print(f"📋 Server 提供 {len(tools)} 个工具：")
    for tool in tools:
        print(f"   🔧 {tool.name}: {tool.description}")
    print()

    # ====== 场景1：创建项目文件 ======
    print("─" * 50)
    print("[场景1] 创建项目文件")
    print("─" * 50)
    r = await caller.call_tool("write_file", {
        "path": "my-project/README.md",
        "content": "# My MCP Project\n\n通过 MCP 创建的项目。\n\n## 运行\n\n```bash\npython3 main.py\n```",
    })
    print(r)
    r = await caller.call_tool("write_file", {
        "path": "my-project/main.py",
        "content": "#!/usr/bin/env python3\n\ndef hello():\n    print('Hello from MCP!')\n\nif __name__ == '__main__':\n    hello()\n",
    })
    print(r)

    # ====== 场景2：查看目录 ======
    print("\n─" * 50)
    print("[场景2] 查看项目目录")
    print("─" * 50)
    r = await caller.call_tool("list_directory", {"path": "my-project"})
    print(r)

    # ====== 场景3：读取文件 ======
    print("\n─" * 50)
    print("[场景3] 读取 main.py 内容")
    print("─" * 50)
    r = await caller.call_tool("read_file", {"path": "my-project/main.py"})
    print(r)

    # ====== 场景4：搜索文件 ======
    print("\n─" * 50)
    print("[场景4] 搜索所有 .py 文件")
    print("─" * 50)
    r = await caller.call_tool("search_files", {"pattern": "*.py"})
    print(r)

    # ====== 场景5：安全测试 ======
    print("\n─" * 50)
    print("[场景5] 安全测试：路径穿越攻击")
    print("─" * 50)
    r = await caller.call_tool("read_file", {"path": "../../../etc/passwd"})
    print(r)

    await caller.close()
    print("\n🔌 连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
