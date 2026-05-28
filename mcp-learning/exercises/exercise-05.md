# 练习5：设计"好"的 Tool Description

## 任务

对比不同的 Tool description 写法，理解怎样的描述能让 LLM 做出正确的工具选择。

## 学习目标

- 理解 LLM 如何使用 Tool 的 name、description、inputSchema
- 学会设计精准的 Tool 描述
- 避免常见的 Tool 设计陷阱

## 核心原理：LLM 如何选择 Tool？

```
用户问题 → LLM 读取每个 Tool 的 {name, description, inputSchema} 
         → 语义匹配：这个工具能解决用户的问题吗？
         → 生成参数：根据 inputSchema 中的 properties 和 required
```

Tool 的三要素各司其职：
- **name**：唯一标识（LLM 用名字来决定"要不要用"）
- **description**：功能说明（LLM 判断"什么时候用"的关键依据）
- **inputSchema**：参数规格（LLM 按这个生成参数 JSON）

## 对比实验

### 不好的 Tool 设计

```python
# 反例1：description 太模糊
Tool(
    name="do_stuff",
    description="处理一些事情",  # ← LLM 不知道什么时候用它
    inputSchema={
        "properties": {
            "data": {"type": "string"},  # ← 没有说明格式
        },
    },
)

# 反例2：name 和功能不匹配
Tool(
    name="get_weather",
    description="查询股票价格",  # ← name 和 description 矛盾，LLM 会困惑
    ...
)

# 反例3：缺少 required 约束
Tool(
    name="read_file",
    description="读取文件内容",
    inputSchema={
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
        },
        # ← 缺少 "required": ["path"]，LLM 可能不传这个参数
    },
)
```

### 好的 Tool 设计

```python
# 好例1：description 精准描述使用场景
Tool(
    name="read_file",
    description="读取文件的全部文本内容。适用于用户想看某个文件的内容时使用。限制 1MB 以内。",
    inputSchema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径，如 'project/main.py' 或 '/tmp/log.txt'",
            },
        },
        "required": ["path"],
    },
)

# 好例2：description 包含了"什么时候用"和"限制条件"
Tool(
    name="search_files",
    description="按文件名模式搜索文件（仅按文件名匹配，不在文件内容中搜索）。"
                "用于查找特定类型的文件。支持通配符如 '*.py'、'test_*'。",
    inputSchema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "文件名匹配模式，支持通配符。例如 '*.py' 搜索所有 Python 文件",
            },
            "directory": {
                "type": "string",
                "description": "搜索的起始目录，默认为根目录。例如 'my-project/'",
            },
        },
        "required": ["pattern"],
    },
)
```

## 设计原则

### 1. name：动词 + 名词，清晰无歧义

```
✅ 好的命名：
   read_file, write_file, delete_file, search_files, list_directory
   get_weather, calculate, add_note, send_email

❌ 不好的命名：
   file_op, do_thing, process, handle, execute
```

### 2. description：包含"什么时候用"+"能力边界"

```python
# ❌ 只说了做什么
"读取文件内容"

# ✅ 说了什么时候用 + 限制条件
"读取文件的全部文本内容，返回文件原文。适用于查看代码、配置、日志等文本文件。
 不支持二进制文件，文件大小限制 1MB。"
```

### 3. inputSchema：参数描述 = 给 LLM 的"填表指南"

```python
# ❌ 缺少说明，LLM 不知道怎么填
"path": {"type": "string"}

# ✅ 有说明 + 有示例，LLM 能正确生成参数
"path": {
    "type": "string",
    "description": "文件路径。可以是相对于工作目录的路径（如 'src/main.py'），
                   也可以是绝对路径（如 '/tmp/test.txt'）",
}
```

### 4. properties 的 description 要写出"格式"和"示例"

```python
# 布尔类型：说明 true/false 的含义
"overwrite": {
    "type": "boolean",
    "description": "如果文件已存在是否覆盖。true=覆盖，false=报错。默认为 false。",
}

# 数字类型：说明单位和范围
"limit": {
    "type": "integer",
    "description": "返回的最大条数，范围 1-100，默认 20",
}

# 枚举类型：列出所有合法值
"sort_by": {
    "type": "string",
    "enum": ["name", "size", "date"],
    "description": "排序字段：name=按文件名，size=按大小，date=按修改时间",
}
```

## 实战练习

### 题目：为"备忘录 Server"（练习3）重写 Tool 描述

重写 `add_note`、`search_notes`、`delete_note` 的 description 和参数说明，使其达到生产级质量。要求：

1. `add_note` 的 description 明确说明创建新笔记
2. `tags` 参数说明支持的格式（逗号分隔）
3. `search_notes` 的 description 明确它是"内容搜索"而非"标题搜索"
4. `delete_note` 说明需要 `note_id`（从 `list_notes` 返回结果中获取）

参考写法：

```python
Tool(
    name="add_note",
    description="创建一条新的备忘录。用于记录待办事项、重要信息、灵感想法等。",
    inputSchema={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "备忘录标题，简洁概括内容。例如：'周五开会'、'学习计划'",
            },
            "content": {
                "type": "string",
                "description": "备忘录正文内容",
            },
            "tags": {
                "type": "string",
                "description": "标签，多个标签用逗号分隔。例如：'工作,重要,待办'。可选参数。",
            },
        },
        "required": ["title", "content"],
    },
)
```

## 思考题

1. 如果两个 Tool 的功能有重叠（比如 `read_file` 和 `get_file_info` 都能获取文件内容），LLM 会怎么选择？如何避免歧义？
2. `description` 应该写多长？太长会不会让 LLM 的上下文不够用？

### 参考答案

1. LLM 会根据 description 中的"使用场景"来决定。`read_file` 的描述是"读取全部文本内容"，`get_file_info` 是"获取元信息"。当用户说"查看文件"时 LLM 优先选 `read_file`，说"文件多大/什么时候改的"时选 `get_file_info`。

2. description 用一句话说清"什么时候用"即可（10-30 字）。Token 预算留给 inputSchema 的参数描述，因为 LLM 生成参数时更需要细节。
