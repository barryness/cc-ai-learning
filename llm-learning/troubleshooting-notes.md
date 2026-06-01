# llm_playground.py 踩坑记录

> 一个 Streamlit 应用的 6 次 Bug 修复，覆盖了 Python 编码、Streamlit 组件生命周期、模块导入策略、多版本 Python 环境管理等常见陷阱。

---

## 修改清单

### Bug 1：中文引号导致 SyntaxError

**报错**：`SyntaxError: invalid syntax` at line 195

**原因**：代码中使用了中文全角引号 `` " `` `` " ``（U+201C / U+201D），Python 把左引号 `"` 当作普通字符而非字符串分隔符，导致解析失败。

```python
# ❌ 错误
print(f'最关注" {word} "')
#           ↑ U+201C 左双引号

# ✅ 正确
print(f"最关注「{word}」")
```

**教训**：中文注释/字符串中不要用 `""`，改用 `「」` 或保持在英文引号 `""` 内部。

---

### Bug 2：tiktoken 解码结果破坏 HTML 渲染

**报错**：点击实验1「切分 Token」后页面渲染异常

**原因**：tiktoken 的 `decode()` 返回的 Token 文本包含特殊字符（如 `�` 替换字符 `�`），通过 `st.markdown(unsafe_allow_html=True)` 渲染时破坏了 HTML 结构。

```python
# ❌ 错误
decoded[i]  # 可能包含 � \x00 等破坏 HTML 的字符

# ✅ 修复
import html
html.escape(decoded[i])  # 转义 & < > 等 HTML 特殊字符
```

**教训**：任何用户数据、外部库输出进入 HTML 前，必须用 `html.escape()` 转义。

---

### Bug 3：st.tabs() 在按钮回调中导致 Streamlit 崩溃

**报错**：`StreamlitAPIException: DuplicateWidgetID` 或 `missing widget` 错误

**原因**：`st.tabs()` 创建了持久的 widget，但放在 `if st.button():` 条件块内部时，只有按钮被点击后 tabs 才存在。Streamlit 下次重渲染时找不到这些 tabs，状态不一致。

```python
# ❌ 错误
if st.button('开始对比'):
    with st.spinner(...):
        tab1, tab2 = st.tabs(['A', 'B'])  # tabs 在条件块内 → 崩溃
        ...

# ✅ 修复
if st.button('开始对比'):
    with st.spinner(...):
        results = {...}  # 先收集所有结果
    for name, result in results.items():
        with st.expander(f' {name}', expanded=True):  # expander 可以条件化
            st.markdown(result)
```

**教训**：`st.tabs()` 不能创建在 `if st.button()` 之类的条件分支内。需要条件化渲染时用 `st.expander()`。

---

### Bug 4：模块导入位置导致整个应用崩溃

**报错**：打开应用直接白屏，`ModuleNotFoundError: No module named 'langchain_openai'`

**原因**：LLM 相关库（`langchain_openai`、`langchain_core`）从按钮回调内移到了文件顶层 import。这导致即使实验 1-3 不需要这些依赖，整个应用也无法加载。

```python
# ❌ 错误：顶层导入可选依赖 → 应用启动就崩溃
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ✅ 修复：懒加载 — 只在按钮点击时才导入
def _load_llm_libs():
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        return (ChatOpenAI, HumanMessage, SystemMessage), None
    except Exception as e:
        return None, f'{e}'
```

然后再在按钮回调中调用 `_load_llm_libs()`，失败时显示友好错误而非崩溃。

**教训**：
- 顶层 import 只放**必需**依赖（streamlit, numpy, os, html, pathlib）
- 可选依赖必须**懒加载**——在用到的按钮/函数内部 import
- try/except 要 catch `Exception` 而非只 `ModuleNotFoundError`，避免遗漏 DLL 加载错误等

---

### Bug 5：多 Python 环境导致「安装了却没有」

**现象**：终端手动运行 `pip install` 装好了包，但 `streamlit run` 还是报 `ModuleNotFoundError`

**原因**：系统装了 3 个 Python（3.9 / 3.13 / 3.14），streamlit 在 3.9 下，pip 在 3.14 下。`pip install` 装到了 3.14，但 streamlit 用的是 3.9。

```
python3 (3.14) → pip 在 3.14 → 包装到 3.14
streamlit → 使用 3.9 → 找不到包
```

**排查方法**：
```bash
which -a python3        # 列出所有 Python
which -a streamlit      # 列出所有 streamlit
head -1 $(which streamlit)  # 看 streamlit 用哪个 Python
```

**修复**：删除旧版 Python，只保留 3.14。

**教训**：
- 永远用 `python3 -m pip install <包名>` 而非裸 `pip install`
- `which -a` 可以列出 PATH 中所有同名命令
- 多版本 Python 是这类问题的根源，只保留一个最好

---

### Bug 6：`@st.cache_resource` 缺失导致重复初始化

**原因**：`get_llm()` 每次调用都创建新的 `ChatOpenAI` 实例，浪费资源且可能触发 API 端的 rate limit。

```python
# ✅ 修复
@st.cache_resource
def get_llm():
    return ChatOpenAI(...)
```

**教训**：Streamlit 中创建连接/客户端实例的函数必须加 `@st.cache_resource`（非数据用 `@st.cache_data`）。

---

## 踩坑清单：Streamlit 开发注意事项

| # | 陷阱 | 原因 | 方案 |
|---|------|------|------|
| 1 | 中文 `""` 引号 | Python 不识别为字符串分隔符 | 用 `「」` 或 `""` |
| 2 | `unsafe_allow_html=True` + 外部数据 | 特殊字符破坏 HTML | `html.escape()` 转义 |
| 3 | `st.tabs()` 在条件块内 | Widget 生命周期与条件渲染冲突 | 用 `st.expander()` |
| 4 | 顶层导入可选依赖 | 一个异常整个 app 崩溃 | 懒加载 + try/except |
| 5 | 多 Python 版本 | pip 和 python 可能不匹配 | 统一用 `python3 -m pip` |
| 6 | 客户端实例无缓存 | 重复创建连接 | `@st.cache_resource` |

---

## 核心原则

1. **不要把所有 import 写顶层** —— 区分必需和可选，可选依赖懒加载
2. **区分 `st.tabs()` 和 `st.expander()`** —— tabs 不能条件化，expander 可以
3. **一个 Python，一个 pip** —— 多版本是万恶之源
4. **外部数据进 HTML 必须转义** —— `html.escape()` 是标配
5. **Streamlit 客户端实例必须缓存** —— `@st.cache_resource`
