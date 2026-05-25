# 飞书远程操控 Claude Code 方案

## 目标

AstrBot agent 通过调用 Claude Code 进行编程和实验。agent 负责知识检索和任务编排，Claude Code 纯粹作为代码执行器。

## 核心原则

1. **Claude Code 不调用 paperrag** — Claude Code 是纯执行器，只做编程、调试、实验
2. **agent 负责编排** — AstrBot agent 先用 paperrag 的 `@llm_tool` 检索知识，整合后再发给 Claude Code
3. **不新增插件，不改架构** — `code_execute` 直接在 paperrag 插件内注册，与现有 `paper_arag`、`paper_react` 并列
4. **直接调 `claude -p` 子进程** — 同步获取结果，最简单可靠，不需要 cc-connect

## 架构

```
用户（飞书）
    │
    ▼
┌──────────────────────────────────────────┐
│              AstrBot                      │
│  ┌────────────────────────────────────┐  │
│  │         Agent Pipeline              │  │
│  │                                     │  │
│  │  paper_arag          (已有)          │  │
│  │  paper_react         (已有)          │  │  ← paperrag 插件
│  │  code_execute        (新增)          │  │    三个 LLM Tool 并列
│  │    → claude -p 子进程                │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

**数据流：**

```
1. 用户发消息
2. Agent 分析：需要哪些知识？
3. Agent 调用 paper_arag / paper_react 检索论文、引用等
4. Agent 整合检索结果 + 用户意图 → 形成完整编程任务
5. Agent 调用 code_execute(整合后的任务) → claude -p 子进程执行
6. Agent 收到 Claude Code 输出，回复用户
```

## 实现

只改两个文件，各加 ~10 行。

### 1. `commands/paper.py` — 新增 `_code_execute_tool` 方法

```python
# 权限错误关键字（模块级常量）
_PERMISSION_KEYWORDS = re.compile(
    r"permission|approval|authorization|not allowed|denied|requires? (human|user|manual)",
    re.IGNORECASE,
)

@staticmethod
def _validate_code_execute_task(task: str) -> str | None:
    """输入校验：危险模式检测。合法返回 None，非法返回错误信息。"""
    DANGEROUS_PATTERNS = [
        (r"rm\s+-rf\s+/\*?", "rm -rf /"),
        (r"curl\s+.*\|\s*(ba)?sh", "curl ... | sh"),
        (r"wget\s+.*\|\s*(ba)?sh", "wget ... | sh"),
        (r"git\s+push\s+(-f|--force)", "git push --force"),
        (r"\bsudo\b", "sudo"),
        (r"chmod\s+777", "chmod 777"),
        (r">\s*/dev/sd[a-z]", "> /dev/sdX (磁盘覆写)"),
    ]
    for pattern, label in DANGEROUS_PATTERNS:
        if re.search(pattern, task, re.IGNORECASE):
            logger.error(f"[code_execute] 危险模式拒绝: {label}")
            return f"任务包含潜在危险操作 ({label})，已被拒绝。请移除危险命令后重试。"
    return None

async def _code_execute_tool(self, event: AstrMessageEvent, task: str, timeout: int = 300) -> str:
    """LLM Tool wrapper: 启动 claude -p 子进程执行编程任务，同步返回结果。

    Agent 应先调用 paper_search/paper_arag/paper_react 检索相关知识，
    整合上下文后形成完整任务再调用此工具。

    Args:
        event: AstrMessageEvent (injected by framework)
        task: 完整的编程任务描述，需包含所有必要上下文和指令
        timeout: 最大执行秒数，默认300

    Returns:
        Claude Code 的输出文本
    """
    error = self._validate_code_execute_task(task)
    if error:
        return error

    work_dir = str(_PLUGIN_DIR)
    # 不跳过权限，使用 --allowedTools 限制可用工具
    cmd = [
        "claude", "-p", task,
        "--output-format", "text",
        "--allowedTools",
        "Read,Write(astrbot_plugin_paperrag/**),Edit(astrbot_plugin_paperrag/**),Bash(git:*,python:*,pytest:*,pip:*),Grep,Glob",
    ]

    logger.info(f"[code_execute] 执行: {task[:100]}...")
    process = None
    try:
        process = await asyncio.create_subprocess_exec(*cmd, cwd=work_dir, ...)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except FileNotFoundError:
        return "Claude Code 未安装或不在 PATH 中，请联系管理员安装。"
    except asyncio.TimeoutError:
        if process is not None:
            process.kill()
            await process.wait()
        return f"Claude Code 超时 ({timeout}s)，请缩小任务范围或增加超时。"

    output = stdout.decode("utf-8", errors="replace").strip()
    err_output = stderr.decode("utf-8", errors="replace").strip()

    if process.returncode != 0:
        # 检测权限错误 → 通知用户在服务器上手动授权执行
        if _PERMISSION_KEYWORDS.search(output) or _PERMISSION_KEYWORDS.search(err_output):
            return (
                f"Claude Code 执行此任务需要额外权限。\n"
                f"请在服务器上手动运行以下命令授权执行：\n"
                f"```bash\ncd '{work_dir}'\nclaude --dangerously-skip-permissions\n```\n"
                f"然后输入: {task[:300]}"
            )
        if not output:
            return f"Claude Code 退出码 {process.returncode}: {err_output[:500]}"

    return output if output else "(no output)"
```

### 2. `main.py` — 在 `__init__` 中注册

在现有两个 `context.register_llm_tool(...)` 后面添加：

```python
context.register_llm_tool(
    name="code_execute",
    func_args=[
        {"type": "string", "name": "task", "description": "完整的编程任务描述，需包含所有必要上下文和指令"},
        {"type": "integer", "name": "timeout", "description": "最大执行秒数，默认300", "default": 300},
    ],
    desc="使用 Claude Code 执行编程任务：写/改代码、调试、运行实验、重构、git 操作等。agent 应先调用 paper_arag/paper_react 检索相关知识，整合后形成完整任务再调用此工具。",
    func_obj=self._code_execute_tool,
)
```

### 3. 前置条件

```bash
# Claude Code 必须已安装且 API 可用
claude --version
claude -p "echo hello" --output-format text

# 如果用第三方 API proxy，确保 AstrBot 进程环境变量：
export ANTHROPIC_BASE_URL="https://your-api-proxy.com"
export ANTHROPIC_AUTH_TOKEN="sk-your-api-key"
# 如果用 launchd 启动 AstrBot，需要在 plist 中配置 EnvironmentVariables
```

## 安全模型

- **不跳过权限** — 不使用 `--dangerously-skip-permissions`，claude 在 `-p` 模式下自动使用 `--allowedTools` 限制工具范围
- **输入校验** — `_validate_code_execute_task()` 检测危险命令模式（rm -rf /、curl | sh、sudo 等）
- **权限错误处理** — 当 claude 退出码非零且 stderr 包含权限关键字时，返回清晰的指引信息，引导用户在服务器上手动授权执行
- **工具白名单** — 通过 `--allowedTools` 限制为 Read、Write/Edit（插件目录）、Bash（git/python/pytest/pip）、Grep、Glob
- **孤儿进程清理** — 超时时 kill 子进程并 wait 回收，防止资源泄漏

## 验证

```bash
# 1. 直接调试验证
cd /Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag
claude -p "列出 hybrid_index.py 中的主要函数" --output-format text --dangerously-skip-permissions

# 2. 重启 AstrBot，确认 tool 注册成功
grep "code_execute" /path/to/astrbot.log

# 3. 飞书端到端测试
# 输入: 帮我看看 hybrid_index.py 里 get_all_references 的实现
# 预期: Agent 调用 code_execute → Claude Code 读文件分析 → 返回结果
```

## 为什么不用 cc-connect

cc-connect 的 webhook 是异步的（返回 `{"status": "accepted"}` 立即返回），而 AstrBot tool 需要同步返回结果。`claude -p` 子进程天然同步，且省去额外部署。

后续如需会话持久化（跨调用保持上下文），可通过 `claude --resume <session_id>` 或接入 cc-connect 的 relay 机制实现。

## 参考

- [Claude Code CLI 文档](https://docs.anthropic.com/en/docs/claude-code)
