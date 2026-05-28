# 练习3：从零实现"备忘录" MCP Server

## 任务

从零实现一个备忘录（Notes）MCP Server，让 LLM 能够管理用户的备忘录：添加、查看、搜索、删除。

## 学习目标

- 独立完成 MCP Server 的全部开发流程
- 学会用 JSON 文件做持久化存储
- 实践 Tool 设计（怎样拆功能才合理）

## 需求

实现 4 个 Tool：

| Tool | 功能 | 参数 |
|------|------|------|
| `add_note` | 添加一条备忘录 | title（标题）, content（内容）, tags（可选，逗号分隔标签） |
| `list_notes` | 列出所有备忘录 | tag（可选，按标签筛选） |
| `search_notes` | 按关键词搜索 | keyword |
| `delete_note` | 删除指定备忘录 | note_id |

数据存储在 `~/mcp-workspace/notes.json`，格式如下：

```json
{
  "notes": [
    {
      "id": "20260528-001",
      "title": "周五开会",
      "content": "下午3点讨论Q2规划",
      "tags": ["工作", "会议"],
      "created_at": "2026-05-28T16:30:00"
    }
  ]
}
```

## 实现提示

### 1. Server 骨架

```python
# notes_server.py
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

NOTES_FILE = Path(os.path.expanduser("~/mcp-workspace/notes.json"))
server = Server("notes-mcp-server")
```

### 2. 数据存取函数

```python
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
```

### 3. add_note 实现

```python
elif name == "add_note":
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
```

### 4. 完整实现要点

**search_notes**：用 `keyword.lower() in note['title'].lower() or keyword.lower() in note['content'].lower()` 做匹配。

**list_notes**：如果指定了 `tag` 参数，用 `tag in note['tags']` 筛选。

**delete_note**：用列表推导 `[n for n in notes if n['id'] != note_id]` 过滤后重新保存。

## 测试

```python
# test_notes_client.py
import asyncio
import sys
from pathlib import Path

# 复用 MCPToolCaller
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "demo"))
from client_demo import MCPToolCaller

async def main():
    caller = MCPToolCaller("python3", ["notes_server.py"])
    await caller.connect()
    print("✅ 已连接到备忘录 Server\n")

    # 添加几条备忘录
    r = await caller.call_tool("add_note", {
        "title": "买水果",
        "content": "苹果、香蕉、橙子各一斤",
        "tags": "生活,购物",
    })
    print(r)

    r = await caller.call_tool("add_note", {
        "title": "周五开会",
        "content": "下午3点讨论Q2规划，提前准备PPT",
        "tags": "工作,会议",
    })
    print(r)

    # 列出所有备忘录
    r = await caller.call_tool("list_notes", {})
    print(r)

    # 按标签筛选
    r = await caller.call_tool("list_notes", {"tag": "工作"})
    print(r)

    # 搜索
    r = await caller.call_tool("search_notes", {"keyword": "水果"})
    print(r)

    await caller.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## 关键知识点

### `json.dumps` 的参数

```python
json.dumps(data, ensure_ascii=False, indent=2)
```

- `ensure_ascii=False`：保留中文（否则输出 `中文`）
- `indent=2`：缩进 2 空格，生成可读的格式化 JSON

### `datetime.now().strftime()`

```python
datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # → "2026-05-28T16:30:00"
```

格式化码：`%Y` 年、`%m` 月、`%d` 日、`%H` 时、`%M` 分、`%S` 秒

## 思考题

1. 如果两个人同时操作同一个 notes.json，会出现什么问题？如何解决？
2. 当前的搜索引擎是简单的子串匹配，如果要支持模糊搜索（比如"水果"也能搜到"苹果"），你会怎么做？
