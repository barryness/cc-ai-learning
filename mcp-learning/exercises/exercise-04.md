# 练习4：MCP 错误处理最佳实践

## 任务

观察当前 `filesystem-server/server.py` 的错误处理设计，分析哪些场景覆盖了、哪些遗漏了，然后补全。

## 学习目标

- 理解 MCP 中错误返回的格式约定
- 学会识别和处理各种边界情况
- 理解为何 MCP Server 不能"崩溃"

## 当前错误处理分析

### 已覆盖的场景

| 场景 | 处理方式 | 代码位置 |
|------|---------|---------|
| 路径穿越攻击 | `PermissionError` 被最外层 `except` 捕获 | `safe_path()` raise → `call_tool()` except |
| 文件不存在 | 返回 "文件不存在" 文本 | 各分支 `if not path.exists()` |
| 未知 Tool 名 | 返回 "未知工具" 文本 | `else` 分支 |
| 通用异常 | 返回 `错误：{str(e)}` | 最外层 `except Exception` |

### 需要补全的场景

## 场景1：`list_directory` 对空目录的处理

**当前代码**：
```python
if not items:
    return "（空目录）"
```

**问题**：`items` 可能是空的，但此时返回的文本也可以更友好一些。实际已经处理好了。

## 场景2：`read_file` 的编码问题

**当前代码**：
```python
content = path.read_text(encoding="utf-8")
```

**问题**：如果文件不是 UTF-8 编码（如 GBK、Latin-1），`read_text()` 会抛出 `UnicodeDecodeError`。当前会被最外层 `except Exception` 捕获，返回通用错误。

**改进方案**：

```python
# 方案1：尝试多种编码
encodings = ["utf-8", "gbk", "latin-1"]
for enc in encodings:
    try:
        content = path.read_text(encoding=enc)
        break
    except UnicodeDecodeError:
        continue
else:
    return [TextContent(type="text", text=f"无法解码文件 {path}，不支持的编码")]
```

## 场景3：`write_file` 的权限问题

**当前代码**：
```python
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(content, encoding="utf-8")
```

**问题**：如果目录没有写权限，`mkdir()` 或 `write_text()` 会抛出 `PermissionError`。当前会被最外层捕获，但报错信息可能不够友好。

**改进方案**：

```python
try:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
except PermissionError:
    return [TextContent(
        type="text",
        text=f"没有写入权限：{path}。请检查目录权限。",
    )]
```

## 场景4：`search_files` 目录不存在

**当前代码**：
```python
base = safe_path(directory)
for fpath in base.rglob("*"):
```

**问题**：如果 `base` 目录不存在，`rglob()` 会抛出 `FileNotFoundError`（实际是直接返回空迭代器，但 `safe_path` 不会检查目录是否存在）。

**改进方案**：

```python
base = safe_path(directory)
if not base.exists():
    return [TextContent(type="text", text=f"目录不存在：{base}")]
if not base.is_dir():
    return [TextContent(type="text", text=f"不是目录：{base}")]
```

## 关键知识点：MCP 的错误处理哲学

### 为什么不能直接 raise/崩溃？

```
┌──────────┐  call_tool("read_file", {path: "/etc/passwd"})  ┌──────────┐
│  LLM     │ ──────────────────────────────────────────────► │  Server  │
│ (Client) │ ◀────────────────────────────────────────────── │          │
└──────────┘  返回错误文本（而非报错崩溃）                     └──────────┘
```

1. **Server 崩溃 = Client 断连**：如果 Server 进程崩溃，Client 的管道就断了，所有后续调用失败
2. **LLM 可以"理解"错误文本**：返回有意义的错误文本，LLM 能据此调整策略（换个路径、换个编码）
3. **错误是正常流程的一部分**：用户说"读取 /tmp/test.txt"，文件不存在 → 这不是 Bug，是正常的业务结果

### 错误返回的约定

MCP 规范没有强制错误格式，但最佳实践是：
- 返回 `TextContent`（而非 raise exception）
- 文本中包含足够的上下文信息（哪个文件、什么原因）
- 区分"用户可修复的错误"（文件不存在）和"系统错误"（磁盘满）

## 测试

修改后，运行以下测试验证错误处理：

```python
# 测试1：读取不存在的文件
r = await caller.call_tool("read_file", {"path": "不存在.txt"})
print("不存在文件:", r)

# 测试2：list 不存在的目录
r = await caller.call_tool("list_directory", {"path": "不存在目录"})
print("不存在目录:", r)

# 测试3：搜索不存在的目录
r = await caller.call_tool("search_files", {"pattern": "*.py", "directory": "不存在"})
print("搜索不存在目录:", r)

# 测试4：删除不存在的文件（需要先实现 delete_file）
r = await caller.call_tool("delete_file", {"path": "不存在.txt"})
print("删除不存在:", r)
```
