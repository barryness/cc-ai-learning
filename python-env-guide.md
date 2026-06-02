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

# 4. 检查 uv 等工具安装的 Python
ls ~/.local/share/uv/python/          # 查看 uv 安装的版本
uv python list                         # 列出所有 uv 管理的 Python
rm -rf ~/.local/share/uv/python/cpython-3.11-*   # 删除不需要的版本
rm -f ~/.local/bin/python3.11                     # 删除 uv 创建的符号链接

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

## uv 包管理器

### 为什么用 uv

uv 是 Rust 写的 Python 包/项目管理器，对标 pip + venv + pyenv 的组合，核心优势：

- **快**：解析依赖和下载并行，比 pip 快 10-100 倍
- **Python 版本管理**：能下载、安装、切换任意 CPython 版本（类似 pyenv）
- **虚拟环境**：`uv venv` 创建，`uv pip install` 装包，不需要手动 activate
- **全局缓存**：所有下载的包和 Python 解释器都存在 `~/.cache/uv/`，同一版本只下载一次

### 安装

```bash
# macOS
brew install uv

# 或者
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 核心命令速查

```bash
# ---- Python 版本管理 ----
uv python list              # 列出已安装 + 可下载的 Python 版本
uv python install 3.12      # 安装指定版本（下载 + 解压到 ~/.local/share/uv/python/）
uv python uninstall 3.11    # 卸载指定版本

# ---- 虚拟环境 ----
uv venv                     # 在当前目录创建 .venv（使用当前默认 Python）
uv venv --python 3.12       # 指定 Python 版本创建虚拟环境
uv venv --python /usr/local/bin/python3.14  # 指定系统 Python 路径

# ---- 包管理 ----
uv pip install chromadb                          # 装包（自动用当前 venv，没有就系统环境）
uv pip install chromadb --python .venv/bin/python # 指定解释器
uv pip install "torch>=2.0" "transformers<5"      # 带版本约束
uv pip install --python .venv/bin/python -r requirements.txt  # 批量安装
uv pip list                                       # 列出已安装的包
uv pip uninstall chromadb                         # 卸载

# ---- 运行脚本 ----
uv run --python .venv/bin/python script.py        # 指定解释器运行
uv run python script.py                           # 自动检测 .venv
```

### 工作流：为新项目创建独立环境

```bash
# 1. 安装所需 Python 版本（只需要做一次，全局缓存）
uv python install 3.12

# 2. 创建项目虚拟环境
cd my-project
uv venv --python 3.12

# 3. 装依赖
uv pip install --python .venv/bin/python torch transformers chromadb

# 4. 运行脚本
uv run --python .venv/bin/python main.py
```

### 虚拟环境激活

uv 创建的 venv 和标准 Python venv 完全一样，可以手动激活：

```bash
source .venv/bin/activate   # 激活后 python / pip 自动指向 venv 内
which python                 # → .../my-project/.venv/bin/python
deactivate                   # 退出
```

### uv 安装的 Python 存在哪里

| 内容 | 路径 |
|------|------|
| Python 解释器 | `~/.local/share/uv/python/cpython-3.12.13-macos-x86_64-none/bin/python3.12` |
| 符号链接 | `~/.local/bin/python3.12` → 上面那个路径 |
| 包缓存 | `~/.cache/uv/wheels-v6/` |
| 项目 venv | 项目目录下的 `.venv/` |

---

## 常见踩坑记录

### 坑1：Python 3.14 太新，大量包没有 wheel

**现象**：
```
hint: You require CPython 3.14 (`cp314`), but we only found wheels for
`torch` with the following Python ABI tags: `cp310`, `cp311`, `cp312`, `cp313`
```

**根因**：Python 3.14 是预发布版本，多数科学计算包（torch、onnxruntime、pandas）还没有构建 cp314 wheel。`pip install` 会尝试从源码编译，但编译需要 C++ 工具链且大概率失败。

**解法**：用 uv 安装 Python 3.12（当前兼容性最好的版本）：

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install --python .venv/bin/python <你的包>
```

### 坑2：macOS x86_64 没有 PyTorch wheel

**现象**：
```
hint: Wheels are available for `torch` (v2.12.0) on the following
platforms: `manylinux_2_28_x86_64`, `macosx_14_0_arm64`, `win_amd64`
```

注意 `macosx_14_0_arm64` 只有 **arm64**（Apple Silicon），没有 x86_64（Intel Mac）。你如果是 Intel Mac，PyTorch >= 2.4 就没有预编译包。

**解法**：
- 装 PyTorch 2.2.2（最后支持 x86_64 macOS 的版本之一）：`uv pip install torch==2.2.2`
- 或者换 Apple Silicon Mac

### 坑3：PyTorch 版本与 transformers 版本互相制约

**现象**：
```
transformers: Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2
ValueError: Due to a serious vulnerability issue in `torch.load`, we now require
users to upgrade torch to at least v2.6
```

**根因**：transformers 新版本（4.50+）强制要求 torch >= 2.6（安全漏洞修复），而 macOS x86_64 又没有 torch >= 2.4 的 wheel。这是一个死锁。

**解法**：锁定 transformers 到兼容版本：

```bash
uv pip install torch==2.2.2
uv pip install "transformers>=4.30,<4.50" "sentence-transformers<3"
```

### 坑4：numpy 版本不兼容

**现象**：
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6
```

**根因**：PyTorch 2.2.2 是用 numpy 1.x 编译的，而 pip 可能自动装 numpy 2.x。两者 ABI 不兼容。

**解法**：
```bash
uv pip install "numpy<2"
```

### 坑5：HuggingFace 无法访问（网络问题）

**现象**：
```
ConnectTimeoutError: Connection to huggingface.co timed out
OSError: We couldn't connect to 'https://huggingface.co' to load the files
```

**根因**：`AutoModel.from_pretrained("xxx")` 默认先去 HuggingFace Hub 检查模型更新，网络不通就挂。

**解法**：

1. **离线加载**（模型已缓存时）：
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-chinese", local_files_only=True)
```

2. **设置镜像**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

3. **查看本地已缓存的模型**：
```bash
ls ~/.cache/huggingface/hub/models--*/
```

### 坑6：uv pip install 和 pip install 混用

**现象**：装了包但是 import 找不到。

**根因**：`pip install` 装到了系统 Python，`uv pip install --python .venv/bin/python` 装到了 venv。两者不互通。

**正确做法**：一个项目始终用一种方式。推荐：

```bash
# 方案A：全程 uv（推荐）
uv pip install --python .venv/bin/python <包名>

# 方案B：激活 venv 后用原生 pip
source .venv/bin/activate
pip install <包名>
```

### 坑7：uv venv 后忘记 source activate 导致包装错位置

**现象**：`pip install xxx` 后，运行脚本提示 `ModuleNotFoundError`。

**根因**：没激活 venv，pip 指向系统 Python，包装到了系统环境。

**验证方法**：
```bash
which python          # 如果是系统路径而非 .venv/ 下的，说明没激活
which pip             # 同理
```

**最佳实践**：用 `uv pip install --python .venv/bin/python` 明确指定解释器，完全绕过 pip PATH 的问题——这是 uv 最大的实用优势。

### 坑8：.venv 不要提交到 Git

`.venv/` 目录随平台和环境变化（macOS vs Linux，x86_64 vs arm64），提交到 Git 会造成各种奇怪问题。应该加入 `.gitignore`：

```
.venv/
__pycache__/
*.pyc
```

---

## 本机已完成的操作

- [x] 删除 Python 3.9 框架（2026-06-01）
- [x] 删除 Python 3.13 框架（2026-06-01）
- [x] 删除 Python 3.7 PATH 引用（2026-06-01）
- [x] 清理 ~/.zprofile 中 3.9/3.13 PATH（2026-06-01）
- [x] 清理 ~/.bash_profile 中 3.9/3.7 PATH（2026-06-01）
- [x] 删除 uv 安装的 Python 3.11（2026-06-01）
- [x] 清除 zsh 补全缓存 ~/.zcompdump*（2026-06-01）
- [x] 保留 Python 3.14 作为唯一版本
- [x] 安装全部 LLM 学习相关依赖
- [x] uv 安装 Python 3.12.13 用于 ChromaDB 项目（2026-06-02）
- [x] vector-db-learning 项目用 Python 3.12 + ChromaDB + bert-base-chinese（2026-06-02）
- [x] 解决 PyTorch/transformers/numpy 版本兼容链：torch 2.2.2 + transformers 4.49.0 + numpy<2（2026-06-02）
- [x] 整理 uv 使用指南 + 常见踩坑记录（2026-06-02）
