"""
PaperBanana 图像生成（独立模块）
"""

import asyncio
import atexit
import base64
import httpx
import os
import socket
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from astrbot.api import logger

from .generation import IdeaEngineGeneration


def _find_free_port(start: int = 8765, max_attempts: int = 20) -> int:
    """Find a free TCP port starting from `start`."""
    for port in range(start, start + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + max_attempts}")


class IdeaEnginePaperBanana(IdeaEngineGeneration):
    """PaperBanana 图像生成。继承链：... → IdeaEngineGeneration → IdeaEnginePaperBanana"""

    # ---- Server lifecycle (class-level) ---------------------------------

    _server_process: Optional[asyncio.subprocess.Process] = None
    _server_port: int = 0
    _server_ready: bool = False
    _server_lock = asyncio.Lock()
    _server_stderr_task: Optional[asyncio.Task] = None
    _cleanup_registered: bool = False

    @classmethod
    def _paperbanana_project_path(cls, config: dict) -> Optional[Path]:
        """Resolve the PaperBanana project path from config."""
        raw = (config or {}).get("paperbanana_project_path", "")
        if raw:
            p = Path(raw)
            if p.is_absolute() and p.exists():
                return p
        # Fallback: look for PaperBanana next to the plugin
        default = Path(__file__).parent.parent.parent.parent / "PaperBanana"
        if default.exists():
            return default
        return None

    @classmethod
    def _paperbanana_python(cls, project_path: Path) -> Optional[str]:
        """Return the path to PaperBanana's venv Python."""
        venv_python = project_path / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        import shutil
        sys_python = shutil.which("python3") or shutil.which("python")
        return sys_python

    @classmethod
    async def _stop_paperbanana_server(cls):
        """Stop the local PaperBanana server subprocess."""
        proc = None
        async with cls._server_lock:
            proc = cls._server_process
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
        except ProcessLookupError:
            pass  # Already exited
        except Exception as e:
            logger.warning(f"[PaperBanana] Error stopping server: {e}")
        finally:
            # Only clear state if this is still the same process
            async with cls._server_lock:
                if cls._server_process is proc:
                    cls._server_process = None
                    cls._server_port = 0
                    cls._server_ready = False
                if cls._server_stderr_task is not None:
                    cls._server_stderr_task.cancel()
                    cls._server_stderr_task = None
            logger.info("[PaperBanana] Local server stopped.")

    @classmethod
    async def _ensure_paperbanana_server(cls, config: dict):
        """Ensure the local PaperBanana server is running. Idempotent."""
        async with cls._server_lock:
            if (cls._server_ready and cls._server_port
                    and cls._server_process is not None
                    and cls._server_process.returncode is None):
                return

            project_path = cls._paperbanana_project_path(config)
            if not project_path:
                logger.warning("[PaperBanana] PaperBanana 项目路径未配置或不存在，跳过")
                return

            python_bin = cls._paperbanana_python(project_path)
            if not python_bin:
                logger.warning("[PaperBanana] 找不到可用的 Python，跳过")
                return

            server_script = Path(__file__).parent / "paperbanana_server.py"
            if not server_script.exists():
                logger.error(f"[PaperBanana] Server script not found: {server_script}")
                return

            port = _find_free_port()
            logger.info(f"[PaperBanana] Starting local server on port {port} using {python_bin}...")

            try:
                proc = await asyncio.create_subprocess_exec(
                    python_bin, str(server_script),
                    "--paperbanana-path", str(project_path),
                    "--port", str(port),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except Exception as e:
                logger.error(f"[PaperBanana] Failed to start server: {e}")
                return

            cls._server_process = proc

            # Background task to drain stderr so the pipe doesn't block
            async def _drain_stderr():
                try:
                    while True:
                        line = await proc.stderr.readline()
                        if not line:
                            break
                        decoded = line.decode("utf-8", errors="replace").strip()
                        if decoded:
                            logger.warning(f"[PaperBanana] [server:stderr] {decoded}")
                except Exception:
                    pass

            cls._server_stderr_task = asyncio.ensure_future(_drain_stderr())

            # Register atexit handler exactly once
            if not cls._cleanup_registered:
                _loop = asyncio.get_event_loop_policy().get_event_loop()
                atexit.register(lambda: _loop.run_until_complete(
                    cls._stop_paperbanana_server()))
                cls._cleanup_registered = True

            # Wait for the "Ready" line from stdout (non-blocking async read)
            deadline = asyncio.get_event_loop().time() + 60
            ready = False
            while asyncio.get_event_loop().time() < deadline:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    # Check if process died
                    if proc.returncode is not None:
                        logger.error(f"[PaperBanana] Server exited early (rc={proc.returncode})")
                        cls._server_process = None
                        return
                    continue

                if not line:
                    # EOF — process exited
                    await proc.wait()
                    logger.error(f"[PaperBanana] Server exited early (rc={proc.returncode})")
                    cls._server_process = None
                    return

                decoded = line.decode("utf-8", errors="replace").strip()
                logger.info(f"[PaperBanana] [server] {decoded}")
                if "Ready on port" in decoded:
                    cls._server_port = port
                    cls._server_ready = True
                    ready = True
                    break

            if not ready:
                logger.error("[PaperBanana] Server did not become ready within timeout")
                cls._server_process = None  # Don't block port — _stop not needed for timed-out process
                try:
                    proc.terminate()
                except Exception:
                    pass
                return

            logger.info(f"[PaperBanana] Server ready on port {port}")

    # ---- Image generation methods ---------------------------------------

    async def _generate_method_figures_with_paperbanana_from_text(
        self, method_text: str, topic: str, caption: Optional[str] = None
    ) -> List[Dict]:
        """调用 PaperBanana 服务生成方法图（基于完整方法论文本）"""
        blocks = []
        if not method_text:
            return blocks
        figure_caption = caption or f"Methodology: {topic}"
        try:
            image_path = await self._call_paperbanana(
                method_text=method_text,
                figure_caption=figure_caption
            )
            if image_path and os.path.exists(image_path):
                blocks.append({"path": image_path, "caption": figure_caption or ""})
                logger.info(f"[IdeaEngine] 方法图生成成功: {topic[:30]}")
        except Exception as e:
            logger.warning(f"[IdeaEngine] 方法图生成失败 [{topic[:20]}]: {e}")
        return blocks

    async def _call_paperbanana(
        self, method_text: str, figure_caption: str = ""
    ) -> Optional[str]:
        """调用本地 PaperBanana 服务生成方法图，返回图片文件路径。"""
        if not method_text:
            return None

        cfg_dict = getattr(self, '_plugin_config', {}) or {}

        if not cfg_dict.get("enable_paper_banana", False):
            logger.warning("[PaperBanana] enable_paper_banana 未启用，跳过方法图生成")
            return None

        # Ensure the local server is running
        await self._ensure_paperbanana_server(cfg_dict)

        if not self._server_ready:
            logger.warning("[PaperBanana] Local server not available, skipping generation")
            return None

        # Retry with exponential backoff for transient failures
        last_error = ""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=900.0)) as client:
                    # Pre-flight health check
                    health_url = f"http://127.0.0.1:{self._server_port}/health"
                    try:
                        resp = await client.get(health_url)
                        if resp.status_code != 200:
                            raise ConnectionError(f"Health check returned {resp.status_code}")
                    except Exception as he:
                        raise ConnectionError(f"Server health check failed: {he}") from he

                    response = await client.post(
                        f"http://127.0.0.1:{self._server_port}/generate",
                        json={"text": method_text, "caption": figure_caption},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success") and data.get("image_base64"):
                            img_bytes = base64.b64decode(data["image_base64"])
                            tmp = tempfile.NamedTemporaryFile(
                                suffix=".png", delete=False,
                                dir=Path(__file__).parent.parent / "data" / "temp"
                            )
                            Path(tmp.name).parent.mkdir(parents=True, exist_ok=True)
                            tmp.write(img_bytes)
                            tmp.close()
                            logger.info(f"[PaperBanana] Image saved to: {tmp.name}")
                            return tmp.name
                        else:
                            last_error = data.get("error", "unknown")
                            logger.warning(f"[PaperBanana] Generation failed (attempt {attempt + 1}/3): {last_error}")
                    elif response.status_code >= 500:
                        last_error = f"Server error: {response.status_code}"
                        logger.warning(f"[PaperBanana] Server error (attempt {attempt + 1}/3): {response.status_code}")
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        logger.warning(f"[PaperBanana] Unexpected response: {last_error}")
                        break  # Don't retry on 4xx
            except Exception as e:
                last_error = repr(e)
                logger.warning(f"[PaperBanana] Connection failed (attempt {attempt + 1}/3): {type(e).__name__}: {e}")

            if attempt < 2:
                delay = 2 ** attempt * 2
                await asyncio.sleep(delay)
                # Re-check server readiness before retry
                if not self._server_ready:
                    await self._ensure_paperbanana_server(cfg_dict)
                    if not self._server_ready:
                        logger.error("[PaperBanana] Server still unavailable after restart attempt")
                        return None

        logger.error(f"[PaperBanana] All retries exhausted: {last_error}")
        return None
