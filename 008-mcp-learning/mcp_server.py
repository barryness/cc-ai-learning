"""
=============================================================================
  MCP Server：文件系统 + SQLite 数据库
  ──────────────────────────────────────────────────────────────────────────
  配合 mcp_guide.md 学习。一个 Server 提供两组工具：

  文件系统工具                     SQLite 数据库工具
  ──────────────                  ─────────────────
  list_directory                   db_list_tables
  read_file                        db_describe_table
  search_files                     db_query (只读 SELECT)
  write_file                       db_export_table

  设计原则：
    - 最小权限：文件系统限制目录，SQLite 只读连接
    - 安全防护：safe_path() 防止路径穿越攻击
    - 一个 Server = 多组工具，LLM 自动选择

  运行方式：
    python mcp_server.py           # stdio 模式（供 Claude Code 连接）
    python mcp_server.py --demo    # 自测模式（内置 Client 模拟 LLM 调用）

  Claude Code 配置（.mcp.json 或 ~/.claude/settings.json）：
    {
      "mcpServers": {
        "local-tools": {
          "command": "python3",
          "args": ["/path/to/mcp_server.py"],
          "env": {
            "WORKSPACE_DIR": "/path/to/your/workspace",
            "DB_PATH": "/path/to/your/database.db"
          }
        }
      }
    }
=============================================================================
"""

import asyncio
import os
import sys
import json
import fnmatch
import sqlite3
import time
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# ═══════════════════════════════════════════════════════════════════════════════
# 配置（可通过环境变量覆盖）
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", os.path.expanduser("~/mcp-workspace"))
DB_PATH = os.environ.get("DB_PATH", os.path.expanduser("~/mcp-workspace/demo.db"))
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB

server = Server("local-tools-mcp-server")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          安全：路径校验                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def safe_path(user_path: str, root: str = None) -> Path:
    """
    校验用户路径，防止路径穿越攻击。

    攻击示例：
      user_path = "../../../etc/passwd"
      → resolve() 展开后变成 /etc/passwd
      → 不在 allowed_root 下 → 拒绝访问

    参数：
      user_path: 用户传入的相对路径
      root:      允许的根目录（默认 WORKSPACE_DIR）
    """
    root = root or WORKSPACE_DIR
    if not user_path.startswith("/"):
        full = Path(root) / user_path
    else:
        full = Path(user_path)

    resolved = full.resolve()
    allowed = Path(root).resolve()

    if not str(resolved).startswith(str(allowed)):
        raise PermissionError(
            f"路径穿越检测：'{user_path}' 不在允许的目录 '{root}' 内"
        )
    return resolved


def _fmt_size(size_bytes: int) -> str:
    """字节数 → 可读格式"""
    for unit in ["B", "KB", "MB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.0f}{unit}" if unit == "B" else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}GB"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        Tool 声明（工具菜单）                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@server.list_tools()
async def list_tools() -> list[Tool]:
    """LLM 首先调用这个函数，获取所有可用工具的"菜单"。"""
    return [
        # ── 文件系统工具 ──
        Tool(
            name="list_directory",
            description="列出指定目录下的所有文件和子目录。用于探索目录结构、查看项目文件。",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": f"目录路径，相对于 {WORKSPACE_DIR}。传 '.' 表示根目录。",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="read_file",
            description="读取文件的全部文本内容。支持多种编码（UTF-8、GBK、Latin-1）。限制 1MB 以内。",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径，相对于工作目录",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="search_files",
            description="按文件名模式搜索文件，支持通配符（如 *.py、test_*.txt）。递归搜索子目录。",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "文件名匹配模式，如 '*.py'、'test_*'、'README.*'",
                    },
                    "directory": {
                        "type": "string",
                        "description": "搜索的起始目录，默认为根目录",
                    },
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="write_file",
            description="将文本内容写入文件（会覆盖已有内容）。自动创建父目录。",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径，相对于工作目录",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文本内容",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        # ── SQLite 数据库工具 ──
        Tool(
            name="db_list_tables",
            description="列出 SQLite 数据库中的所有表。用于了解数据库中有哪些数据可用。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="db_describe_table",
            description="查看指定表的结构（列名、类型、是否可空、默认值）。用于理解表 Schema 后再查询。",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "要查看结构的表名",
                    },
                },
                "required": ["table_name"],
            },
        ),
        Tool(
            name="db_query",
            description="执行 SELECT 查询（只读）。支持 JOIN、WHERE、GROUP BY 等标准 SQL。结果限制 100 行。",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SELECT 查询语句。仅允许 SELECT 开头，禁止 INSERT/UPDATE/DELETE/DROP。",
                    },
                },
                "required": ["sql"],
            },
        ),
        Tool(
            name="db_export_table",
            description="导出指定表的全部数据为 JSON 格式。用于查看完整数据或做数据分析。默认限制 500 行。",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "要导出的表名",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最大返回行数，默认 500",
                    },
                },
                "required": ["table_name"],
            },
        ),
    ]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        Tool 实现（执行逻辑）                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """LLM 决定调用某个工具后，Server 在这里执行并返回结果。"""
    try:
        # ──── 文件系统工具 ────
        if name == "list_directory":
            return _handle_list_directory(arguments)
        elif name == "read_file":
            return _handle_read_file(arguments)
        elif name == "search_files":
            return _handle_search_files(arguments)
        elif name == "write_file":
            return _handle_write_file(arguments)
        # ──── SQLite 数据库工具 ────
        elif name == "db_list_tables":
            return _handle_db_list_tables()
        elif name == "db_describe_table":
            return _handle_db_describe_table(arguments)
        elif name == "db_query":
            return _handle_db_query(arguments)
        elif name == "db_export_table":
            return _handle_db_export_table(arguments)
        else:
            return [TextContent(type="text", text=f"未知工具：{name}")]

    except PermissionError as e:
        return [TextContent(type="text", text=f"⛔ 权限拒绝：{e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"❌ 错误：{e}")]


# ═══════════════════════════════════════════════════════════════════════════════
# 文件系统工具实现
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_list_directory(args: dict) -> list[TextContent]:
    path = safe_path(args["path"])
    if not path.exists():
        return [TextContent(type="text", text=f"目录不存在：{path}")]
    if not path.is_dir():
        return [TextContent(type="text", text=f"不是目录：{path}")]

    items = []
    for entry in sorted(path.iterdir()):
        marker = "📁" if entry.is_dir() else "📄"
        size = entry.stat().st_size
        items.append(f"  {marker} {entry.name}  ({_fmt_size(size)})")

    output = f"目录 {path} 的内容：\n" + ("\n".join(items) if items else "（空目录）")
    return [TextContent(type="text", text=output)]


def _handle_read_file(args: dict) -> list[TextContent]:
    path = safe_path(args["path"])
    if not path.exists():
        return [TextContent(type="text", text=f"文件不存在：{path}")]
    if path.stat().st_size > MAX_FILE_SIZE:
        return [TextContent(type="text", text=f"文件超过 1MB，拒绝读取（{_fmt_size(path.stat().st_size)}）")]

    # 尝试多种编码读取
    content = None
    for enc in ["utf-8", "gbk", "latin-1"]:
        try:
            content = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        return [TextContent(type="text", text=f"无法解码文件：{path}")]

    return [TextContent(type="text", text=f"=== {path.name} ===\n{content}")]


def _handle_search_files(args: dict) -> list[TextContent]:
    pattern = args["pattern"]
    directory = args.get("directory", ".")
    base = safe_path(directory)

    if not base.exists():
        return [TextContent(type="text", text=f"目录不存在：{base}")]
    if not base.is_dir():
        return [TextContent(type="text", text=f"不是目录：{base}")]

    matches = []
    for fpath in base.rglob("*"):
        if fpath.is_file() and fnmatch.fnmatch(fpath.name, pattern):
            matches.append(f"  📄 {fpath.relative_to(base)}  ({_fmt_size(fpath.stat().st_size)})")

    if not matches:
        return [TextContent(type="text", text=f"搜索 '{pattern}' ：未找到匹配文件")]
    return [TextContent(type="text", text=f"搜索 '{pattern}' 结果（共 {len(matches)} 个）：\n" + "\n".join(matches))]


def _handle_write_file(args: dict) -> list[TextContent]:
    path = safe_path(args["path"])
    content = args["content"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return [TextContent(type="text", text=f"✅ 已写入 {len(content)} 个字符到 {path}")]


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite 数据库工具实现
# ═══════════════════════════════════════════════════════════════════════════════

def _get_db_connection() -> sqlite3.Connection:
    """获取只读数据库连接。每次调用创建新连接，用完即关。"""
    db_file = Path(DB_PATH)
    if not db_file.exists():
        raise FileNotFoundError(f"数据库文件不存在：{DB_PATH}")

    # 以只读模式连接，防止 LLM 误操作修改数据
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row  # 让查询结果可以用列名访问
    return conn


def _handle_db_list_tables() -> list[TextContent]:
    conn = _get_db_connection()
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        if not tables:
            return [TextContent(type="text", text="数据库中没有表")]

        lines = [f"数据库 {DB_PATH} 中的表："]
        for t in tables:
            row_count = conn.execute(f"SELECT COUNT(*) FROM [{t['name']}]").fetchone()[0]
            lines.append(f"  📊 {t['name']}  ({row_count} 行)")
        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


def _handle_db_describe_table(args: dict) -> list[TextContent]:
    table = args["table_name"]
    conn = _get_db_connection()
    try:
        # PRAGMA table_info 返回列信息：cid, name, type, notnull, dflt_value, pk
        cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        if not cols:
            return [TextContent(type="text", text=f"表不存在：{table}")]

        lines = [f"表 {table} 结构："]
        for c in cols:
            flags = []
            if c["pk"]:
                flags.append("PK")
            if c["notnull"]:
                flags.append("NOT NULL")
            if c["dflt_value"] is not None:
                flags.append(f"DEFAULT {c['dflt_value']}")

            flag_str = f"  ({', '.join(flags)})" if flags else ""
            lines.append(f"  {c['name']}: {c['type']}{flag_str}")

        # 也列出索引
        indexes = conn.execute(
            f"SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='{table}'"
        ).fetchall()
        if indexes:
            lines.append("\n索引：")
            for idx in indexes:
                lines.append(f"  🔍 {idx['name']}")

        return [TextContent(type="text", text="\n".join(lines))]
    finally:
        conn.close()


def _handle_db_query(args: dict) -> list[TextContent]:
    sql = args["sql"].strip()

    # 安全检查：只允许 SELECT
    if not sql.upper().startswith("SELECT"):
        return [TextContent(
            type="text",
            text=f"⛔ 只允许 SELECT 查询。收到: {sql[:50]}..."
        )]

    # 禁止危险操作
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
    for keyword in dangerous:
        if keyword in sql.upper():
            return [TextContent(
                type="text",
                text=f"⛔ 查询包含禁止的关键词 '{keyword}'。只允许只读 SELECT。"
            )]

    conn = _get_db_connection()
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(100)  # 最多 100 行
        col_names = [d[0] for d in cursor.description]

        if not rows:
            return [TextContent(type="text", text="查询结果为空")]

        # 格式化为可读表格
        lines = [" | ".join(col_names)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))

        total = len(rows)
        has_more = len(cursor.fetchmany(1)) > 0
        suffix = f"\n（显示前 {total} 行" + ("，还有更多...）" if has_more else "）")

        return [TextContent(type="text", text="\n".join(lines) + suffix)]
    finally:
        conn.close()


def _handle_db_export_table(args: dict) -> list[TextContent]:
    table = args["table_name"]
    limit = min(int(args.get("limit", 500)), 1000)

    conn = _get_db_connection()
    try:
        rows = conn.execute(
            f"SELECT * FROM [{table}] LIMIT {limit}"
        ).fetchall()

        if not rows:
            return [TextContent(type="text", text=f"表 {table} 为空")]

        # 转为字典列表
        data = [dict(row) for row in rows]
        return [TextContent(
            type="text",
            text=f"表 {table} 数据（{len(data)} 行）：\n" +
                 json.dumps(data, ensure_ascii=False, indent=2, default=str)
        )]
    except sqlite3.OperationalError as e:
        return [TextContent(type="text", text=f"表不存在或无法访问：{e}")]
    finally:
        conn.close()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          Server 入口                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def run_stdio():
    """stdio 模式：供 Claude Code / Cursor 等 MCP Host 连接。"""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    print(f"📂 本地工具 MCP Server 启动", file=sys.stderr)
    print(f"   工作目录: {WORKSPACE_DIR}", file=sys.stderr)
    print(f"   数据库:   {DB_PATH}", file=sys.stderr)
    print(f"   等待 Client 连接...", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      Demo 模式（自测，模拟 LLM 调用）                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def run_demo():
    """
    自测模式：模拟 LLM 发现和调用工具的过程。
    不需要 MCP Host，直接在本进程内调用工具函数。
    """
    print("=" * 60)
    print("  🧪 MCP Server 自测模式")
    print("=" * 60)

    # 1. 创建工作目录和测试文件
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    _setup_demo_files()
    _setup_demo_database()

    # 2. 列出所有工具
    tools = await list_tools()
    print(f"\n📋 Server 提供 {len(tools)} 个工具：")
    for t in tools:
        props = t.inputSchema.get("properties", {})
        params = ", ".join(props.keys()) if props else "无参数"
        print(f"   🔧 {t.name}({params})")
        print(f"      {t.description[:60]}...")

    # 3. 模拟调用：文件系统工具
    print(f"\n{'─' * 60}")
    print("  📂 文件系统工具演示")
    print(f"{'─' * 60}")

    fs_scenarios = [
        ("list_directory", {"path": "."}, "列出根目录"),
        ("read_file", {"path": "hello.py"}, "读取 hello.py"),
        ("search_files", {"pattern": "*.py"}, "搜索所有 .py 文件"),
        ("write_file", {"path": "notes.txt", "content": "MCP 学习笔记\n1. Tool Calling\n2. MCP 协议\n3. 连接 Claude Code"}, "创建 notes.txt"),
    ]

    for tool_name, args, desc in fs_scenarios:
        print(f"\n  📝 {desc}")
        print(f"     → call_tool({tool_name}, {json.dumps(args, ensure_ascii=False)})")
        result = await call_tool(tool_name, args)
        for r in result:
            preview = r.text[:200] + "..." if len(r.text) > 200 else r.text
            for line in preview.split("\n"):
                print(f"     {line}")

    # 4. 模拟调用：SQLite 工具
    print(f"\n{'─' * 60}")
    print("  🗄️  SQLite 数据库工具演示")
    print(f"{'─' * 60}")

    db_scenarios = [
        ("db_list_tables", {}, "列出所有表"),
        ("db_describe_table", {"table_name": "products"}, "查看 products 表结构"),
        ("db_query", {"sql": "SELECT category, COUNT(*) as count, AVG(price) as avg_price FROM products GROUP BY category"}, "分类统计查询"),
        ("db_export_table", {"table_name": "products", "limit": 5}, "导出 products 前 5 行"),
    ]

    for tool_name, args, desc in db_scenarios:
        print(f"\n  📝 {desc}")
        print(f"     → call_tool({tool_name}, {json.dumps(args, ensure_ascii=False)})")
        result = await call_tool(tool_name, args)
        for r in result:
            preview = r.text[:300] + "..." if len(r.text) > 300 else r.text
            for line in preview.split("\n"):
                print(f"     {line}")

    # 5. 安全测试
    print(f"\n{'─' * 60}")
    print("  🛡️  安全测试")
    print(f"{'─' * 60}")

    print(f"\n  📝 路径穿越攻击测试")
    print(f"     → call_tool(read_file, path='../../../etc/passwd')")
    result = await call_tool("read_file", {"path": "../../../etc/passwd"})
    print(f"     {result[0].text}")

    print(f"\n  📝 SQL 注入防护测试")
    print(f"     → call_tool(db_query, sql='DROP TABLE products')")
    result = await call_tool("db_query", {"sql": "DROP TABLE products"})
    print(f"     {result[0].text}")

    print(f"\n{'=' * 60}")
    print("  ✅ 自测完成！")
    print(f"  💡 运行 python mcp_server.py（不带 --demo）启动 stdio 模式")
    print(f"     然后在 Claude Code 中连接即可使用。")
    print(f"{'=' * 60}")


def _setup_demo_files():
    """创建演示用的测试文件"""
    files = {
        "hello.py": (
            "#!/usr/bin/env python3\n"
            "\"\"\"Hello World 示例\"\"\"\n\n"
            "def greet(name: str) -> str:\n"
            "    return f\"Hello, {name}!\"\n\n"
            "if __name__ == \"__main__\":\n"
            "    print(greet(\"World\"))\n"
        ),
        "config.json": (
            "{\n"
            '  "app_name": "MCP Demo",\n'
            '  "version": "1.0.0",\n'
            '  "database": "demo.db",\n'
            '  "max_connections": 10\n'
            "}\n"
        ),
        "README.md": (
            "# MCP Demo 项目\n\n"
            "这是一个 MCP Server 的演示项目。\n\n"
            "## 功能\n\n"
            "- 文件系统操作\n"
            "- SQLite 数据库查询\n\n"
            "## 运行\n\n"
            "```bash\n"
            "python mcp_server.py --demo\n"
            "```\n"
        ),
    }
    for filename, content in files.items():
        filepath = Path(WORKSPACE_DIR) / filename
        if not filepath.exists():
            filepath.write_text(content, encoding="utf-8")


def _setup_demo_database():
    """创建演示用的 SQLite 数据库"""
    db_file = Path(DB_PATH)
    if db_file.exists():
        return  # 数据库已存在，不重复创建

    conn = sqlite3.connect(DB_PATH)
    try:
        # 创建 products 表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER DEFAULT 0,
                description TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 创建 orders 表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                total_price REAL NOT NULL,
                customer TEXT,
                order_date TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)

        # 插入示例数据
        products = [
            ("Python 编程入门", "图书", 59.0, 100, "适合编程初学者的 Python 教程"),
            ("深度学习实战", "图书", 89.0, 50, "基于 PyTorch 的深度学习实践"),
            ("MCP 协议指南", "图书", 39.0, 200, "MCP 协议从入门到精通"),
            ("机械键盘 K8", "硬件", 399.0, 30, "87键 茶轴 无线机械键盘"),
            ("4K 显示器", "硬件", 2499.0, 15, "27寸 4K IPS 广色域显示器"),
            ("VS Code 插件开发", "课程", 199.0, 999, "学习开发 VS Code 扩展"),
            ("LangGraph 实战", "课程", 299.0, 500, "从零构建多 Agent 系统"),
            ("AI 编程助手", "软件", 29.0, 9999, "月度订阅制 AI 编程助手"),
        ]
        conn.executemany(
            "INSERT INTO products (name, category, price, stock, description) VALUES (?, ?, ?, ?, ?)",
            products,
        )

        orders = [
            (1, 2, 118.0, "张三", "2026-05-01"),
            (3, 1, 39.0, "李四", "2026-05-15"),
            (4, 1, 399.0, "王五", "2026-05-20"),
            (7, 1, 299.0, "赵六", "2026-06-01"),
            (1, 3, 177.0, "张三", "2026-06-02"),
            (8, 5, 145.0, "李四", "2026-06-02"),
        ]
        conn.executemany(
            "INSERT INTO orders (product_id, quantity, total_price, customer, order_date) VALUES (?, ?, ?, ?, ?)",
            orders,
        )

        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--demo" in sys.argv:
        asyncio.run(run_demo())
    else:
        asyncio.run(run_stdio())
