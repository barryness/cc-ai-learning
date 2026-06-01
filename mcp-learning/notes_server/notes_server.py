"""
=============================================================================
  备忘录 MCP Server —— 让 LLM 管理用户的备忘录
=============================================================================

提供 4 个工具：
  1. add_note      —— 添加一条备忘录
  2. list_notes    —— 列出所有备忘录（支持按标签筛选）
  3. search_notes  —— 按关键词搜索
  4. delete_note   —— 删除指定备忘录

数据存储：~/mcp-workspace/notes.json

运行方式：
  Terminal 1: python3 notes_server.py
  Terminal 2: python3 test_notes_client.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# ============================================================================
# 配置
# ============================================================================
NOTES_FILE = Path(os.path.expanduser("~/mcp-workspace/notes.json"))
server = Server("notes-mcp-server")


# ============================================================================
# 数据存取
# ============================================================================
def load_notes() -> list[dict]:
    """从 JSON 文件加载所有备忘录"""
    if not NOTES_FILE.exists():
        return []
    return json.loads(NOTES_FILE.read_text(encoding="utf-8")).get("notes", [])


def save_notes(notes: list[dict]):
    """保存备忘录到 JSON 文件"""
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(
        json.dumps({"notes": notes}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def make_id() -> str:
    """生成唯一 ID，格式：YYYYMMDD-NNN"""
    today = datetime.now().strftime("%Y%m%d")
    notes = load_notes()
    today_notes = [n for n in notes if n["id"].startswith(today)]
    return f"{today}-{len(today_notes) + 1:03d}"


# ============================================================================
# 工具声明
# ============================================================================
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="add_note",
            description="创建一条新的备忘录。适用于记录待办事项、重要信息、灵感想法等场景。"
                        "创建的笔记会保存到本地 JSON 文件持久化存储。",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "备忘录标题，简洁概括内容。例如 '周五开会'、'学习计划'",
                    },
                    "content": {
                        "type": "string",
                        "description": "备忘录正文内容，支持多行文本。可以是详细的描述、代码片段、列表等。",
                    },
                    "tags": {
                        "type": "string",
                        "description": "标签，多个标签用英文逗号分隔。例如 '工作,重要,待办'。可选参数，不传则不添加标签。",
                    },
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="list_notes",
            description="列出所有已保存的备忘录。支持按标签精确筛选（需完整匹配标签名）。"
                        "返回每条备忘录的 ID、标题、标签列表和创建时间。",
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "按标签精确筛选，例如 '工作' 只返回含 '工作' 标签的笔记。不传则列出全部。",
                    },
                },
            },
        ),
        Tool(
            name="search_notes",
            description="在备忘录的标题和正文内容中搜索关键词（全文搜索），返回内容匹配的备忘录列表。"
                        "注意：仅搜索已保存内容，不支持模糊匹配或拼音搜索。",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，在标题和正文中执行子串匹配（大小写不敏感）。例如 '会议' 会匹配标题或内容中包含 '会议' 的备忘录。",
                    },
                },
                "required": ["keyword"],
            },
        ),
        Tool(
            name="delete_note",
            description="删除指定 ID 的备忘录，操作不可撤销。ID 必须从 list_notes 返回结果中获取。",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "备忘录 ID，格式如 '20260529-001'。通过 list_notes 工具获取可用的 ID。",
                    },
                },
                "required": ["note_id"],
            },
        ),
    ]


# ============================================================================
# 工具实现
# ============================================================================
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "add_note":
            title = arguments["title"]
            content = arguments["content"]
            tags_str = arguments.get("tags", "")

            notes = load_notes()
            note = {
                "id": make_id(),
                "title": title,
                "content": content,
                "tags": [t.strip() for t in tags_str.split(",") if t.strip()],
                "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            }
            notes.append(note)
            save_notes(notes)

            return [TextContent(
                type="text",
                text=f"已添加备忘录 [{note['id']}] {title}",
            )]

        elif name == "list_notes":
            tag = arguments.get("tag")
            notes = load_notes()

            if tag:
                notes = [n for n in notes if tag in n.get("tags", [])]

            if not notes:
                return [TextContent(type="text", text="（暂无备忘录）")]

            lines = []
            for n in notes:
                tags_str = ", ".join(n.get("tags", []))
                lines.append(f"  [{n['id']}] {n['title']}")
                lines.append(f"    标签: {tags_str}  |  创建: {n['created_at']}")

            return [TextContent(
                type="text",
                text=f"备忘录列表（共 {len(notes)} 条）：\n" + "\n".join(lines),
            )]

        elif name == "search_notes":
            keyword = arguments["keyword"].lower()
            notes = load_notes()

            matched = [
                n for n in notes
                if keyword in n["title"].lower() or keyword in n["content"].lower()
            ]

            if not matched:
                return [TextContent(type="text", text=f"未找到包含 '{keyword}' 的备忘录")]

            lines = []
            for n in matched:
                lines.append(f"  [{n['id']}] {n['title']}")
                lines.append(f"    {n['content'][:100]}{'...' if len(n['content']) > 100 else ''}")

            return [TextContent(
                type="text",
                text=f"搜索 '{keyword}' 结果（共 {len(matched)} 条）：\n" + "\n".join(lines),
            )]

        elif name == "delete_note":
            note_id = arguments["note_id"]
            notes = load_notes()

            new_notes = [n for n in notes if n["id"] != note_id]
            if len(new_notes) == len(notes):
                return [TextContent(type="text", text=f"备忘录不存在：{note_id}")]

            save_notes(new_notes)
            return [TextContent(type="text", text=f"已删除备忘录 [{note_id}]")]

        else:
            return [TextContent(type="text", text=f"未知工具：{name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"错误：{str(e)}")]


# ============================================================================
# 启动
# ============================================================================
async def main():
    print(f"📝 备忘录 MCP Server 启动", file=sys.stderr)
    print(f"   数据文件: {NOTES_FILE}", file=sys.stderr)
    print(f"   等待 Client 连接...", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
