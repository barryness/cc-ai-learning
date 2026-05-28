# 练习2：添加 `get_file_info` 工具（文件元信息）

## 任务

在 `filesystem-server/server.py` 中添加 `get_file_info` 工具，返回文件的详细元信息（大小、修改时间、行数、类型等）。

## 学习目标

- 学习如何设计返回"结构化信息"的 Tool
- 熟悉 Python `pathlib` 和 `os` 模块的文件信息获取
- 理解 Tool 返回值的格式化

## 操作步骤

### 步骤1：声明 Tool

```python
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
```

### 步骤2：实现逻辑

```python
import os
import time
import mimetypes

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
```

### 步骤3：测试

```python
# 在 client_demo.py 中
print("─" * 50)
print("[场景X] 获取文件信息")
print("─" * 50)

r = await caller.call_tool("get_file_info", {"path": "my-project/main.py"})
print(r)
```

预期输出类似：
```
文件路径：/Users/yangming/mcp-workspace/my-project/main.py
文件大小：106B（106 bytes）
修改时间：2026-05-28 16:25:30
创建时间：2026-05-28 16:25:30
权限：644
行数：7
```

## 关键知识点

### `Path.stat()` 返回什么？

```python
stat = Path("file.txt").stat()
stat.st_size   # 文件大小（字节）
stat.st_mtime  # 最后修改时间（Unix 时间戳）
stat.st_ctime  # 创建时间 / 元数据修改时间（macOS 上是创建时间）
stat.st_mode   # 文件权限位（如 0o100644）
```

### `split('\n')` 的行数注意

```python
"".split('\n')       # → ['']        → len = 1（空文件也是 1 行）
"hello".split('\n')  # → ['hello']   → len = 1
"a\nb".split('\n')   # → ['a', 'b']  → len = 2
```

如果要去掉末尾换行符的空行：`content.rstrip('\n').split('\n')`

## 思考题

1. 如果要让 `get_file_info` 也支持目录（列出子文件数、总大小），怎么改？
2. `stat.st_mtime` 返回的是浮点数 Unix 时间戳，为什么要转成可读格式？
