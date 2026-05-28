# 练习1：给文件系统 Server 添加 `delete_file` 工具

## 任务

在 `filesystem-server/server.py` 中添加一个新的 `delete_file` 工具，让 LLM 能够删除文件。

## 学习目标

- 掌握向 MCP Server 添加新 Tool 的完整流程（声明 + 实现）
- 理解工具安全设计的重要性

## 操作步骤

### 步骤1：声明 Tool

在 `list_tools()` 的返回列表中添加一个新的 `Tool` 对象：

```python
Tool(
    name="delete_file",
    description="删除指定的文件（不可恢复，请谨慎使用）",
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
```

### 步骤2：实现 Tool 逻辑

在 `call_tool()` 函数中添加 `elif name == "delete_file":` 分支：

```python
elif name == "delete_file":
    path = safe_path(arguments["path"])
    if not path.exists():
        return [TextContent(type="text", text=f"文件不存在：{path}")]
    if path.is_dir():
        return [TextContent(type="text", text=f"'{path}' 是目录，请使用 rmdir 或手动删除")]
    
    path.unlink()  # Path.unlink() 删除文件（等同于 os.remove()）
    return [TextContent(
        type="text",
        text=f"已删除 {path}",
    )]
```

### 步骤3：测试

在 `client_demo.py` 中添加测试场景：

```python
# ====== 场景6：删除文件 ======
print("─" * 50)
print("[场景6] 删除文件测试")
print("─" * 50)

# 先创建一个临时文件
r = await caller.call_tool("write_file", {
    "path": "temp.txt",
    "content": "临时文件内容",
})
print(r)

# 删除它
r = await caller.call_tool("delete_file", {"path": "temp.txt"})
print(r)

# 确认已删除（应该报文件不存在）
r = await caller.call_tool("read_file", {"path": "temp.txt"})
print(r)
```

## 关键知识点

### `Path.unlink()` 方法

```python
from pathlib import Path

p = Path("/some/file.txt")
p.unlink()        # 删除文件
p.unlink(missing_ok=True)  # 文件不存在也不报错（Python 3.8+）
```

注意：`unlink()` 只能删除文件，不能删除非空目录。删除目录用 `Path.rmdir()`（只能删空目录）。

### 安全检查

虽然 `safe_path()` 已经做了路径校验，但删除操作还应考虑：
- 文件是否存在
- 是文件还是目录
- 是否需要确认机制（LLM 调用前应该询问）

## 思考题

1. 如果要实现一个"安全删除"（移到回收站而非永久删除），你会怎么设计？
2. 如果 LLM 误判了删除意图（用户说"帮我清理一下"），如何防止 LLM 误删重要文件？
