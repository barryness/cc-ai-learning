"""
=============================================================================
  文件系统 MCP Server —— 让 LLM 操作本地文件
=============================================================================

提供 5 个工具：
  1. list_directory —— 列出目录内容
  2. read_file      —— 读取文件内容
  3. write_file     —— 写入文件
  4. search_files   —— 按名称搜索文件
  5. delete_file    —— 删除文件

安全设计：
  - allowed_root: 只允许操作指定目录，防止 LLM 访问系统文件
  - 路径校验：所有路径必须位于 allowed_root 下
  - 文件大小限制：不读取超过 1MB 的文件

运行方式：
  Terminal 1: python3 server.py
  Terminal 2: python3 client_demo.py
"""

import asyncio
import os
import json
import time
import mimetypes
import fnmatch
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# ============================================================================
# 配置
# ============================================================================
ALLOWED_ROOT = os.path.expanduser("~/mcp-workspace")  # 安全限制：只允许这个目录
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB

server = Server("filesystem-mcp-server")


# ============================================================================
# 安全校验：防止路径穿越攻击
# ============================================================================
def safe_path(user_path: str) -> Path:
    """
    校验用户提供的路径，确保在 allowed_root 以内。

    防御场景：
      用户传 "../../../etc/passwd" → 拒绝，不允许跳出 allowed_root
      用户传 "/tmp"               → 拒绝，不在 allowed_root 内
    """
    # 如果用户给的是相对路径，拼到 allowed_root 下
    if not user_path.startswith("/"):
        full = Path(ALLOWED_ROOT) / user_path
    else:
        full = Path(user_path)

    # resolve() 展开所有 .. 和符号链接，得到真实路径
    resolved = full.resolve()

    # 确保解析后的路径在 allowed_root 以内
    root = Path(ALLOWED_ROOT).resolve()
    if not str(resolved).startswith(str(root)):
        raise PermissionError(f"拒绝访问：'{user_path}' 超出了允许的目录范围")

    return resolved


# ============================================================================
# 工具声明
# ============================================================================
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_directory",
            description="列出指定目录下的所有文件和子目录",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": f"目录路径，相对于 {ALLOWED_ROOT} 或绝对路径",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="read_file",
            description="读取文件的全部文本内容（限制 1MB 以内）",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_file",
            description="将文本内容写入文件（会覆盖已有内容）",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文本内容",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="search_files",
            description="按文件名模式搜索文件，支持通配符（如 *.py、test_*.txt）",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "文件名匹配模式，如 '*.py'、'test_*'",
                    },
                    "directory": {
                        "type": "string",
                        "description": "搜索目录，默认为根目录",
                    },
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="delete_file",
            description="删除指定文件（不能删除目录）",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要删除的文件路径",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="get_file_info",
            description="获取文件的详细信息：大小、修改时间、行数、类型等",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径",
                    },
                },
                "required": ["path"],
            },
        ),
    ]


# ============================================================================
# 工具实现
# ============================================================================
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "list_directory":
            path = safe_path(arguments["path"])
            if not path.exists():
                return [TextContent(type="text", text=f"目录不存在：{path}")]
            if not path.is_dir():
                return [TextContent(type="text", text=f"不是目录：{path}")]

            items = []
            for entry in sorted(path.iterdir()):
                # entry.iterdir() 遍历目录下的每个文件和子目录
                marker = "📁" if entry.is_dir() else "📄"
                size = entry.stat().st_size  # 文件大小（字节）
                items.append(f"  {marker} {entry.name}  ({_fmt_size(size)})")

            return [TextContent(
                type="text",
                text=f"目录 {path} 的内容：\n" + "\n".join(items) if items else "（空目录）",
            )]

        elif name == "read_file":
            path = safe_path(arguments["path"])
            if not path.exists():
                return [TextContent(type="text", text=f"文件不存在：{path}")]
            if path.stat().st_size > MAX_FILE_SIZE:
                return [TextContent(type="text", text=f"文件超过 1MB 限制，拒绝读取")]

            content = path.read_text(encoding="utf-8")
            return [TextContent(
                type="text",
                text=f"=== {path.name} ===\n{content}",
            )]

        elif name == "write_file":
            path = safe_path(arguments["path"])
            content = arguments["content"]

            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            # path.write_text() 写入文本，encoding='utf-8' 处理中文
            path.write_text(content, encoding="utf-8")
            return [TextContent(
                type="text",
                text=f"✅ 已写入 {len(content)} 个字符到 {path}",
            )]

        elif name == "search_files":
            pattern = arguments["pattern"]
            directory = arguments.get("directory", ".")
            base = safe_path(directory)

            matches = []
            # base.rglob(*) 递归遍历所有文件和子目录
            for fpath in base.rglob("*"):
                if fpath.is_file() and fnmatch.fnmatch(fpath.name, pattern):
                    # fnmatch.fnmatch 用 shell 风格通配符匹配文件名
                    matches.append(f"  📄 {fpath.relative_to(base)}")

            return [TextContent(
                type="text",
                text=f"搜索 '{pattern}' 结果（共 {len(matches)} 个）：\n" + "\n".join(matches) if matches else "未找到匹配文件",
            )]

        elif name == "delete_file":
            path = safe_path(arguments["path"])
            if not path.exists():
                return [TextContent(type="text", text=f"文件不存在：{path}")]
            if path.is_dir():
                return [TextContent(type="text", text=f"不能删除目录，请指定文件：{path}")]

            path.unlink()
            return [TextContent(
                type="text",
                text=f"✅ 已删除 {path}",
            )]


        elif name == "get_file_info":
            path = safe_path(arguments["path"])
            if not path.exists():
                return [TextContent(type="text", text=f"文件不存在：{path}")]

            stat = path.stat()
            info_lines = [
                f"文件路径：{path}",
                f"文件大小：{_fmt_size(stat.st_size)}（{stat.st_size} bytes）",
                f"修改时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))}",
                f"创建时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_ctime))}",
                f"权限：{oct(stat.st_mode)[-3:]}",
            ]

            # 如果是文本文件，统计行数
            if path.suffix in ('.py', '.txt', '.md', '.json', '.yaml', '.yml', '.html', '.css', '.js'):
                try:
                    content = path.read_text(encoding="utf-8")
                    lines = content.split('\n')
                    info_lines.append(f"行数：{len(lines)}")
                    # 空文件特殊处理：split('\n') 也会产生 1 个元素
                except UnicodeDecodeError:
                    info_lines.append("类型：二进制文件")

            return [TextContent(type="text", text="\n".join(info_lines))]

        else:
            return [TextContent(type="text", text=f"未知工具：{name}")]

    except PermissionError as e:
        return [TextContent(type="text", text=f"⛔ 权限拒绝：{str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"❌ 错误：{str(e)}")]


# ============================================================================
# 工具函数
# ============================================================================
def _fmt_size(size_bytes: int) -> str:
    """把字节数转成可读格式"""
    for unit in ["B", "KB", "MB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.0f}{unit}" if unit == "B" else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}GB"


# ============================================================================
# 启动
# ============================================================================
async def main():
    # 确保工作目录存在
    os.makedirs(ALLOWED_ROOT, exist_ok=True)
    print(f"📂 文件系统 MCP Server 启动", file=sys.stderr)
    print(f"   允许目录: {ALLOWED_ROOT}", file=sys.stderr)
    print(f"   等待 Client 连接...", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


import sys
if __name__ == "__main__":
    asyncio.run(main())
