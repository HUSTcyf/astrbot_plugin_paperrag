"""Remote SSH Claude Code execution mixin for PaperRAG.

Provides:
- SSH connection management (password auth)
- Claude Code + CC-Connect installation detection and auto-install
- Remote code execution via claude -p over SSH
- /cc command group (status, connect, install, exec, config)
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from astrbot.api import logger

from .base import PluginCoreBase

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

_PLUGIN_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants (mirrored from paper.py for self-contained mixin)
# ---------------------------------------------------------------------------
_PERMISSION_KEYWORDS = re.compile(
    r"permission|approval|authorization|not allowed|denied|requires? (human|user|manual)",
    re.IGNORECASE,
)

_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-rf\s+/($|\*)", "rm -rf /"),
    (r"curl\s+.*\|\s*(ba)?sh", "curl ... | sh"),
    (r"wget\s+.*\|\s*(ba)?sh", "wget ... | sh"),
    (r"git\s+push\s+(-f|--force)", "git push --force"),
    (r"\bsudo\b", "sudo"),
    (r"chmod\s+777", "chmod 777"),
    (r">\s*/dev/(sd[a-z]|nvme\d+n\d+|xvd[a-z]|hd[a-z]|mmcblk\d+)", "> /dev/disk (磁盘覆写)"),
]

# ---------------------------------------------------------------------------
# SSH helpers (all blocking calls dispatched via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _ssh_connect(host: str, port: int, username: str, password: str, timeout: int):
    """Blocking SSH connect via paramiko. Runs in thread pool."""
    import paramiko  # lazy import

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=username,
        password=password or None,
        timeout=timeout,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def _ssh_exec(client, command: str, timeout: int) -> tuple[int, str, str]:
    """Blocking exec_command. Returns (exit_code, stdout, stderr)."""
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, out, err


def _ssh_exec_script(client, script: str, timeout: int) -> tuple[int, str, str]:
    """Execute a multi-line script via bash. Returns (exit_code, stdout, stderr)."""
    tmp_path = f"/tmp/paperrag_install_{uuid.uuid4().hex[:8]}.sh"
    sftp = client.open_sftp()
    try:
        sftp.putfo(io.BytesIO(script.encode("utf-8")), tmp_path)
        sftp.chmod(tmp_path, 0o700)
    finally:
        sftp.close()
    try:
        exit_code, out, err = _ssh_exec(client, f"bash {tmp_path}", timeout)
    finally:
        try:
            sftp = client.open_sftp()
            sftp.remove(tmp_path)
            sftp.close()
        except Exception:
            pass
    return exit_code, out, err


# ---------------------------------------------------------------------------
# RemoteCodeMixin
# ---------------------------------------------------------------------------

class RemoteCodeMixin(PluginCoreBase):
    """远程 SSH Claude Code 执行能力混入类，被 PaperRAGPlugin 继承。"""

    # ---- config helpers ---------------------------------------------------

    def _remote_cfg(self) -> dict:
        return self.config.get("remote_exec", {}) if isinstance(self.config, dict) else {}

    def _remote_enabled(self) -> bool:
        return bool(self._remote_cfg().get("enabled", False))

    def _sanitize_config_for_display(self, cfg: dict) -> dict:
        """返回密码脱敏后的配置副本。"""
        safe = dict(cfg)
        if safe.get("password"):
            safe["password"] = "********"
        return safe

    # ---- SSH client factory -----------------------------------------------

    async def _create_ssh_client(self):
        """创建并返回已连接的 paramiko SSHClient。

        在 asyncio.to_thread 中运行阻塞连接，避免阻塞事件循环。
        返回 (client, error_message)，二者必有一个为 None。
        """
        cfg = self._remote_cfg()
        host = cfg.get("host", "").strip()
        if not host:
            return None, "未配置远程主机，请在插件配置中设置 remote_exec.host。"

        port = int(cfg.get("port", 22))
        username = cfg.get("username", "root").strip()
        password = cfg.get("password", "").strip()
        timeout = int(cfg.get("connect_timeout", 15))

        if not password:
            return None, "未配置SSH密码，请在插件配置中设置 remote_exec.password。"

        try:
            client = await asyncio.to_thread(
                _ssh_connect, host, port, username, password, timeout,
            )
            return client, None
        except Exception as e:
            msg = str(e)
            if "Authentication" in msg:
                return None, f"SSH认证失败 {username}@{host}:{port}，请检查用户名和密码。"
            elif "Connection refused" in msg or "Errno 61" in msg:
                return None, f"连接被拒绝: {host}:{port}，请检查SSH服务是否运行。"
            elif "timed out" in msg.lower() or "Errno 60" in msg:
                return None, f"连接超时: {host}:{port}，请检查防火墙和网络。"
            elif "Name or service not known" in msg or "getaddrinfo" in msg:
                return None, f"无法解析主机名: {host}"
            else:
                return None, f"SSH连接失败: {msg}"

    async def _ssh_exec(self, client, command: str, timeout: int | None = None) -> tuple[int, str, str]:
        """在远程执行单条命令，在 asyncio.to_thread 中运行。"""
        if timeout is None:
            timeout = int(self._remote_cfg().get("exec_timeout", 600))
        return await asyncio.to_thread(_ssh_exec, client, command, timeout)

    # ---- remote readiness -------------------------------------------------

    async def _check_remote_claude_installed(self, client) -> tuple[bool, str]:
        """检查远程是否已安装 Claude Code。返回 (已安装, 版本号或错误信息)。"""
        exit_code, stdout, stderr = await self._ssh_exec(
            client, "which claude && claude --version 2>&1 || echo 'NOT_INSTALLED'", timeout=15,
        )
        output = stdout.strip()
        if "NOT_INSTALLED" in output:
            return False, "未安装"
        if exit_code != 0:
            return False, stderr.strip() or output
        version_line = output.split("\n")[0].strip()
        return True, version_line

    async def _check_remote_cc_connect_installed(self, client) -> tuple[bool, str]:
        """检查远程是否已安装 CC-Connect。"""
        exit_code, stdout, stderr = await self._ssh_exec(
            client, "which cc-connect && cc-connect --version 2>&1 || echo 'NOT_INSTALLED'", timeout=15,
        )
        output = stdout.strip()
        if "NOT_INSTALLED" in output:
            return False, "未安装"
        if exit_code != 0:
            return False, stderr.strip() or output
        return True, output.split("\n")[0].strip()

    async def _check_remote_node(self, client) -> tuple[bool, str]:
        """检查远程是否已安装 Node.js。"""
        exit_code, stdout, stderr = await self._ssh_exec(
            client, "which node && node --version 2>&1 || echo 'NOT_INSTALLED'", timeout=15,
        )
        output = stdout.strip()
        if "NOT_INSTALLED" in output:
            return False, "未安装"
        if exit_code != 0:
            return False, stderr.strip() or output
        return True, output.split("\n")[0].strip()

    async def _detect_remote_os(self, client) -> str:
        """检测远程操作系统和包管理器。返回 debian/rhel/arch/unknown。"""
        exit_code, stdout, _ = await self._ssh_exec(client, "cat /etc/os-release 2>/dev/null | head -5", timeout=10)
        os_info = stdout.lower()
        if "debian" in os_info or "ubuntu" in os_info:
            return "debian"
        if "rhel" in os_info or "centos" in os_info or "fedora" in os_info or "rocky" in os_info:
            return "rhel"
        if "arch" in os_info:
            return "arch"
        exit_code, stdout, _ = await self._ssh_exec(
            client, "which apt-get 2>/dev/null && echo 'debian' || (which yum 2>/dev/null && echo 'rhel') || (which pacman 2>/dev/null && echo 'arch') || echo 'unknown'",
            timeout=10,
        )
        return stdout.strip().split("\n")[-1] if stdout.strip() else "unknown"

    # ---- installation -----------------------------------------------------

    async def _install_claude_code_remote(self, client) -> str:
        """在远程安装 Node.js（如需要）、Claude Code、CC-Connect。

        返回可直接展示给用户的摘要信息。
        """
        messages: list[str] = []
        cfg = self._remote_cfg()
        work_dir = cfg.get("work_dir", "/root/paperrag_workspace").strip()

        # Step 1: 确保工作目录存在
        exit_code, out, err = await self._ssh_exec(client, f"mkdir -p {work_dir}", timeout=10)
        if exit_code != 0:
            messages.append(f"创建工作目录失败: {err}")

        # Step 2: 检查 Node.js
        node_ok, node_ver = await self._check_remote_node(client)
        if not node_ok:
            messages.append("未检测到 Node.js，正在安装 Node.js 18+...")
            os_type = await self._detect_remote_os(client)

            if os_type == "debian":
                node_install_script = (
                    "#!/bin/bash\n"
                    "set -e\n"
                    "curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>&1\n"
                    "apt-get install -y nodejs 2>&1\n"
                )
                exit_code, out, err = await _ssh_exec_script(client, node_install_script, timeout=120)
            elif os_type in ("rhel", "arch"):
                node_install_script = (
                    "#!/bin/bash\n"
                    "set -e\n"
                    "export NVM_DIR=\"$HOME/.nvm\"\n"
                    "if [ ! -s \"$NVM_DIR/nvm.sh\" ]; then\n"
                    "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash 2>&1\n"
                    "fi\n"
                    "[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\"\n"
                    "nvm install 20 2>&1\n"
                    "nvm use 20 2>&1\n"
                    "echo \"export NVM_DIR=\\\"$HOME/.nvm\\\"\" >> $HOME/.bashrc\n"
                    "echo '[ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\"' >> $HOME/.bashrc\n"
                )
                exit_code, out, err = await _ssh_exec_script(client, node_install_script, timeout=120)
            else:
                messages.append(f"错误：不支持的操作系统 ({os_type})，请手动安装 Node.js 18+。")
                return "\n".join(messages)

            if exit_code != 0:
                messages.append(f"Node.js 安装失败: {err[:300]}")
                return "\n".join(messages)

            node_ok, node_ver = await self._check_remote_node(client)
            if node_ok:
                messages.append(f"Node.js 安装成功: {node_ver}")
            else:
                messages.append("Node.js 安装完成但版本检测失败，继续后续步骤...")
        else:
            messages.append(f"Node.js 已安装: {node_ver}")

        # Step 3: 安装/更新 Claude Code
        cc_ok, cc_ver = await self._check_remote_claude_installed(client)
        if cc_ok:
            messages.append(f"Claude Code 已安装: {cc_ver}")
        else:
            messages.append("正在安装 Claude Code...")
            npm_cmd = "npm install -g @anthropic-ai/claude-code 2>&1"
            exit_code, out, err = await self._ssh_exec(client, npm_cmd, timeout=120)
            if exit_code != 0 and ("EACCES" in err or "permission denied" in (err + out).lower()):
                logger.info("[remote_code] npm 安装权限不足，尝试 sudo 重试")
                exit_code, out, err = await self._ssh_exec(client, f"sudo {npm_cmd}", timeout=120)

            if exit_code != 0:
                messages.append(f"Claude Code 安装失败: {err[:300]}")
                return "\n".join(messages)

            cc_ok, cc_ver = await self._check_remote_claude_installed(client)
            if cc_ok:
                messages.append(f"Claude Code 安装成功: {cc_ver}")
            else:
                messages.append("Claude Code npm 安装完成但 claude 命令未在 PATH 中找到。")

        # Step 4: 安装 CC-Connect
        cc_connect_ok, cc_connect_ver = await self._check_remote_cc_connect_installed(client)
        if cc_connect_ok:
            messages.append(f"CC-Connect 已安装: {cc_connect_ver}")
        else:
            messages.append("正在安装 CC-Connect...")
            exit_code, out, err = await self._ssh_exec(client, "npm install -g cc-connect 2>&1", timeout=120)
            if exit_code != 0 and ("EACCES" in err or "permission denied" in (err + out).lower()):
                exit_code, out, err = await self._ssh_exec(client, "sudo npm install -g cc-connect 2>&1", timeout=120)
            if exit_code != 0:
                messages.append(f"CC-Connect 安装失败: {err[:300]}")
            else:
                cc2_ok, cc2_ver = await self._check_remote_cc_connect_installed(client)
                if cc2_ok:
                    messages.append(f"CC-Connect 安装成功: {cc2_ver}")
                else:
                    messages.append("CC-Connect 已安装但版本检测失败。")

        return "\n".join(messages)

    async def _ensure_remote_ready(self) -> tuple[Any | None, str | None]:
        """连接到远程并确保 Claude Code 可用。

        返回 (client, error_message)。若 client 不为 None，则已连接并可用（调用方需负责关闭）。
        若启用了 auto_install 且 Claude Code 缺失，会自动尝试安装。
        """
        cfg = self._remote_cfg()

        client, error = await self._create_ssh_client()
        if error:
            return None, error

        installed, cc_ver = await self._check_remote_claude_installed(client)
        if not installed:
            if cfg.get("auto_install", True):
                logger.info("[remote_code] 远程未检测到 Claude Code，尝试自动安装...")
                install_result = await self._install_claude_code_remote(client)
                logger.info(f"[remote_code] 安装结果:\n{install_result}")

                installed, cc_ver = await self._check_remote_claude_installed(client)
                if not installed:
                    try:
                        client.close()
                    except Exception:
                        pass
                    return None, (
                        "远程服务器上 Claude Code 安装失败。\n\n"
                        "请手动安装：\n"
                        "1. SSH 登录到服务器\n"
                        "2. 安装 Node.js 18+: curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs\n"
                        "3. npm install -g @anthropic-ai/claude-code\n"
                        f"安装日志:\n{install_result}"
                    )
            else:
                try:
                    client.close()
                except Exception:
                    pass
                return None, (
                    "远程服务器上未安装 Claude Code。"
                    "请在插件配置中启用 auto_install 或运行 /cc install 进行安装。"
                )

        return client, None

    # ---- remote execution -------------------------------------------------

    async def _remote_code_execute(self, task: str, timeout: int | None = None) -> str:
        """在远程服务器上通过 Claude Code 执行编程任务。

        Args:
            task: 完整的编程任务描述。
            timeout: 最大执行秒数（None 则使用配置默认值）。

        Returns:
            Claude Code 输出文本或错误信息。
        """
        cfg = self._remote_cfg()
        if timeout is None:
            timeout = int(cfg.get("exec_timeout", 600))
        max_chars = int(cfg.get("max_output_chars", 50000))
        work_dir = cfg.get("work_dir", "/root/paperrag_workspace").strip()

        # 1. 任务安全校验
        from .paper import _DANGEROUS_PATTERNS as _DP
        for pattern, label in _DP:
            if re.search(pattern, task, re.IGNORECASE):
                logger.error(f"[remote_code] 危险模式被拒绝: {label}")
                return f"任务包含潜在危险操作 ({label})，已被拒绝。请移除危险命令后重试。"

        # 2. 连接并确保就绪
        client, error = await self._ensure_remote_ready()
        if error:
            logger.error(f"[remote_code] 远程未就绪: {error}")
            return f"远程执行失败: {error}"
        if client is None:
            return "远程执行失败：无法建立 SSH 连接。"

        try:
            # 3. 通过 SFTP 上传任务到远程临时文件（避免 shell 注入）
            task_id = uuid.uuid4().hex[:8]
            remote_task_file = f"/tmp/paperrag_task_{task_id}.txt"

            try:
                sftp = client.open_sftp()
                sftp.putfo(io.BytesIO(task.encode("utf-8")), remote_task_file)
                sftp.close()
            except Exception as e:
                logger.error(f"[remote_code] SFTP 上传失败: {e}")
                return f"上传任务到远程失败: {e}"

            # 4. 构建并执行命令（显式设置PATH，因为SSH非登录shell不会source profile）
            # --tools "" 禁用内置工具以避免与Agnes API的tool格式兼容性问题
            # ANTHROPIC_DEFAULT_*_MODEL 覆盖Claude Code默认模型为Agnes支持的模型
            cmd = (
                f"export PATH=/usr/local/bin:$PATH && "
                f"export ANTHROPIC_DEFAULT_OPUS_MODEL=agnes-2.0-flash && "
                f"export ANTHROPIC_DEFAULT_SONNET_MODEL=agnes-2.0-flash && "
                f"export ANTHROPIC_DEFAULT_HAIKU_MODEL=agnes-1.5-flash && "
                f"cd {work_dir} && "
                f"claude -p --model agnes-2.0-flash --tools \"\" "
                f"--output-format text "
                f"< {remote_task_file}"
            )

            logger.info(f"[remote_code] 远程执行中: {task[:100]}...")

            try:
                exit_code, stdout, stderr = await self._ssh_exec(client, cmd, timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(f"[remote_code] 执行超时 ({timeout}s)")
                return f"远程执行超时 ({timeout}s)，请缩小任务范围或增加 exec_timeout。"

            # 5. 清理远程临时文件
            try:
                sftp = client.open_sftp()
                sftp.remove(remote_task_file)
                sftp.close()
            except Exception:
                pass

            # 6. 解析输出
            output = stdout.strip()
            err_output = stderr.strip()

            # 权限检测
            perm_in_out = _PERMISSION_KEYWORDS.search(output)
            perm_in_err = _PERMISSION_KEYWORDS.search(err_output)
            if perm_in_out or perm_in_err:
                logger.warning(f"[remote_code] 检测到权限提示: {task[:100]}...")
                return (
                    f"远程 Claude Code 执行此任务需要额外权限。\n\n"
                    f"请 SSH 登录到服务器后手动执行：\n"
                    f"```bash\n"
                    f"ssh {cfg.get('username')}@{cfg.get('host')}\n"
                    f"export PATH=/usr/local/bin:$PATH\n"
                    f"cd {work_dir}\n"
                    f"claude -p --model agnes-2.0-flash --tools \"\" -p < {remote_task_file}\n"
                    f"```\n\n"
                    f"注意：--dangerously-skip-permissions 将绕过所有权限检查，请确认任务安全后再使用。"
                )

            # 检查退出码
            if exit_code != 0:
                logger.error(f"[remote_code] 非零退出 (rc={exit_code}): stderr={err_output[:300]}")
                if not output:
                    return f"远程 Claude Code 退出码 {exit_code}: {err_output[:500]}"

            if err_output:
                logger.warning(f"[remote_code] stderr: {err_output[:300]}")

            # 截断过长输出
            if len(output) > max_chars:
                output = output[:max_chars] + f"\n\n... [输出已截断，超过 {max_chars} 字符]"

            logger.info(f"[remote_code] 完成 ({len(output)} 字符)")
            return output if output else "（无输出）"

        finally:
            try:
                client.close()
            except Exception:
                pass

    # ---- /cc 命令处理 ----------------------------------------------------

    async def _cc_status(self, event: AstrMessageEvent):
        """显示远程连接状态和 Claude Code 版本。"""
        cfg = self._remote_cfg()
        if not cfg.get("enabled", False):
            yield event.plain_result("ℹ️ 远程执行**已禁用**。\n\n在插件配置中设置 `remote_exec.enabled = true` 以启用。")
            return

        yield event.plain_result("🔍 正在检查远程状态...")

        client, error = await self._create_ssh_client()
        if error:
            yield event.plain_result(f"❌ 无法连接到远程: {error}")
            return

        try:
            lines = ["✅ **SSH 已连接**"]
            lines.append(f"   主机: {cfg.get('host')}:{cfg.get('port')}")
            lines.append(f"   用户: {cfg.get('username')}")

            node_ok, node_ver = await self._check_remote_node(client)
            lines.append(f"   Node.js: {'✅ ' + node_ver if node_ok else '❌ ' + node_ver}")

            cc_ok, cc_ver = await self._check_remote_claude_installed(client)
            lines.append(f"   Claude Code: {'✅ ' + cc_ver if cc_ok else '❌ ' + cc_ver}")

            cc2_ok, cc2_ver = await self._check_remote_cc_connect_installed(client)
            lines.append(f"   CC-Connect: {'✅ ' + cc2_ver if cc2_ok else '❌ ' + cc2_ver}")

            exit_code, out, _ = await self._ssh_exec(
                client, f"test -d {cfg.get('work_dir', '/root/paperrag_workspace')} && echo 'exists' || echo 'missing'", timeout=5,
            )
            lines.append(f"   工作目录: {'✅ ' + cfg.get('work_dir', '') if 'exists' in out else '⚠️ ' + cfg.get('work_dir', '') + '（尚未创建）'}")

            os_type = await self._detect_remote_os(client)
            lines.append(f"   操作系统: {os_type}")

            yield event.plain_result("\n".join(lines))
        finally:
            try:
                client.close()
            except Exception:
                pass

    async def _cc_connect(self, event: AstrMessageEvent):
        """测试到远程服务器的 SSH 连接。"""
        cfg = self._remote_cfg()
        if not cfg.get("host"):
            yield event.plain_result("❌ 未配置远程主机，请在插件配置中设置 `remote_exec.host`。")
            return

        yield event.plain_result(f"🔗 正在测试 SSH 连接到 {cfg.get('username')}@{cfg.get('host')}:{cfg.get('port')}...")

        client, error = await self._create_ssh_client()
        if error:
            yield event.plain_result(f"❌ 连接失败: {error}")
            return

        try:
            exit_code, stdout, _ = await self._ssh_exec(
                client, "uname -a && cat /etc/os-release 2>/dev/null | head -3 || echo '(无法获取系统信息)'", timeout=10,
            )
            sys_info = stdout.strip()
            yield event.plain_result(f"✅ **SSH 连接成功！**\n\n```\n{sys_info[:800]}\n```")
        finally:
            try:
                client.close()
            except Exception:
                pass

    async def _cc_install(self, event: AstrMessageEvent):
        """在远程安装或更新 Claude Code + CC-Connect。"""
        cfg = self._remote_cfg()
        if not cfg.get("host"):
            yield event.plain_result("❌ 未配置远程主机，请在插件配置中设置 `remote_exec.host`。")
            return

        yield event.plain_result("📦 正在连接到远程并安装 Claude Code + CC-Connect...\n可能需要几分钟，请耐心等待。")

        client, error = await self._create_ssh_client()
        if error:
            yield event.plain_result(f"❌ 连接失败: {error}")
            return

        try:
            result = await self._install_claude_code_remote(client)
            has_error = any(
                line.startswith("错误") or line.startswith("安装失败") for line in result.split("\n")
            )
            prefix = "❌ 安装过程中出现问题：\n\n" if has_error else "✅ 安装完成：\n\n"
            yield event.plain_result(prefix + result)
        finally:
            try:
                client.close()
            except Exception:
                pass

    async def _cc_exec(self, event: AstrMessageEvent, task: str = "", timeout: int = 0):
        """在远程服务器上执行编程任务。

        Args:
            task: 编程任务描述。
            timeout: 覆盖执行超时（0 表示使用配置默认值）。
        """
        if not task.strip():
            yield event.plain_result("❌ 请提供任务描述。用法：`/cc exec <任务描述>`")
            return

        if not self._remote_enabled():
            yield event.plain_result("❌ 远程执行未启用。请在插件配置中设置 `remote_exec.enabled = true`。")
            return

        for pattern, label in _DANGEROUS_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                yield event.plain_result(f"❌ 任务包含潜在危险操作 ({label})，已被拒绝。")
                return

        yield event.plain_result(f"🚀 正在远程执行: `{task[:80]}...`")

        exec_timeout = timeout if timeout > 0 else None
        result = await self._remote_code_execute(task, timeout=exec_timeout)

        if len(result) > 2000:
            yield event.plain_result(result[:2000] + "\n\n...（显示已截断）")
        else:
            yield event.plain_result(result)

    async def _cc_config(self, event: AstrMessageEvent):
        """显示当前远程执行配置。"""
        cfg = self._remote_cfg()
        if not cfg:
            yield event.plain_result("ℹ️ 远程执行配置未设置。")
            return

        safe = self._sanitize_config_for_display(cfg)

        lines = ["**远程执行配置**\n"]
        lines.append(f"  启用: {safe.get('enabled', False)}")
        lines.append(f"  主机: {safe.get('host', '（未设置）')}")
        lines.append(f"  端口: {safe.get('port', 22)}")
        lines.append(f"  用户名: {safe.get('username', 'root')}")
        lines.append(f"  密码: {safe.get('password', '（未设置）')}")
        lines.append(f"  工作目录: {safe.get('work_dir', '/root/paperrag_workspace')}")
        lines.append(f"  连接超时: {safe.get('connect_timeout', 15)}s")
        lines.append(f"  执行超时: {safe.get('exec_timeout', 600)}s")
        lines.append(f"  自动安装: {safe.get('auto_install', True)}")
        lines.append(f"  最大输出字符: {safe.get('max_output_chars', 50000)}")

        yield event.plain_result("\n".join(lines))
