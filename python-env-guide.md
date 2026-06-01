# Python 环境管理指南

> 记录当前 Python 环境配置，以及日常管理方法。

---

## 当前环境

| 项目 | 路径 |
|------|------|
| Python | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` |
| pip | `/Library/Frameworks/Python.framework/Versions/3.14/bin/pip` |
| 版本 | Python 3.14.3 |
| 已安装关键包 | streamlit, tiktoken, plotly, langchain-openai, langchain-core, numpy, python-dotenv, matplotlib, scikit-learn |

---

## 日常命令

### 查 Python 版本和位置

```bash
python3 --version
which python3
which pip
```

### 查已安装的包

```bash
pip list
pip list | grep <包名>
```

### 安装/卸载包

```bash
python3 -m pip install <包名>
python3 -m pip uninstall <包名>
```

### 查看某个包的信息

```bash
pip show <包名>
```

---

## 多版本问题

### 如果装了多个 Python

`which -a python3` 会列出 PATH 中所有 python3，排在前面的优先。

常见位置：
- `/Library/Frameworks/Python.framework/Versions/3.x/bin/` — 官方安装
- `/usr/local/bin/python3` — Homebrew 安装
- `/usr/bin/python3` — macOS 系统自带

### PATH 优先级

查看当前顺序：

```bash
echo $PATH | tr ':' '\n' | grep -i python
```

`/Library/Frameworks/.../3.14/bin` 必须在 3.9 之前出现。

---

## pip 与 Python 的一致性

### 确认 pip 对应哪个 Python

```bash
pip --version          # 会显示它属于哪个 Python
head -1 $(which pip)   # 查看 shebang
```

### 始终用这句安装，避免装错版本

```bash
python3 -m pip install <包名>
```

这保证包一定装到当前 `python3` 对应的环境里，避免 pip 指向不同 Python 的问题。

---

## 安装新 Python 版本的正确方式

1. 从 [python.org](https://www.python.org/downloads/) 下载官方安装包
2. 安装后确认路径：`/Library/Frameworks/Python.framework/Versions/3.x/bin/python3`
3. 安装完成后 `python3 -m pip install --upgrade pip`

---

## 删除旧 Python 版本

### 步骤

```bash
# 1. 找到旧版本路径
ls /Library/Frameworks/Python.framework/Versions/

# 2. 删除框架目录
sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.9
sudo rm -rf /Library/Frameworks/Python.framework/Versions/3.13

# 3. 清理符号链接
sudo rm -f /usr/local/bin/python3.9 /usr/local/bin/python3.9-config
sudo rm -f /usr/local/bin/python3.13 /usr/local/bin/python3.13-config

# 4. 清理 shell 配置文件中的 PATH
```

### 涉及的 shell 配置文件

| 文件 | 需删除的内容 |
|------|-------------|
| `~/.zprofile` | `PATH="/Library/Frameworks/Python.framework/Versions/3.9/bin:${PATH}"` |
| `~/.zprofile` | `PATH="/Library/Frameworks/Python.framework/Versions/3.13/bin:${PATH}"` |
| `~/.bash_profile` | `export PATH=$PATH:/usr/local/bin/python3.9` |
| `~/.bash_profile` | `export DJANGO_HOME=.../Python/3.7/...` + 对应的 PATH 行 |

### 清除 shell 补全缓存

删除旧版本后，终端自动补全可能仍显示旧版本号。这是 zsh 补全缓存的锅：

```bash
rm -f ~/.zcompdump*
```

重新打开终端后生效。

### 验证

```bash
# 确认框架目录只剩当前版本
ls /Library/Frameworks/Python.framework/Versions/

# 确认 PATH 干净（新终端窗口中执行）
echo $PATH | tr ':' '\n' | grep -i python

# 应只看到 3.14，不再出现 3.9 或 3.13
```

---

## 本机已完成的操作

- [x] 删除 Python 3.9 框架（2026-06-01）
- [x] 删除 Python 3.13 框架（2026-06-01）
- [x] 删除 Python 3.7 PATH 引用（2026-06-01）
- [x] 清理 ~/.zprofile 中 3.9/3.13 PATH（2026-06-01）
- [x] 清理 ~/.bash_profile 中 3.9/3.7 PATH（2026-06-01）
- [x] 清除 zsh 补全缓存 ~/.zcompdump*（2026-06-01）
- [x] 保留 Python 3.14 作为唯一版本
- [x] 安装全部 LLM 学习相关依赖
