"""Tests for lark-cli integration methods."""
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

# Ensure plugin root is on sys.path for imports
import sys
from pathlib import Path
_plugin_root = Path(__file__).parent.parent.parent
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))

from idea.feishu_doc import IdeaEngineFeishuDoc
from idea.utils import IdeaEngineUtils


class TestLarkCliAvailable:
    """Tests for _lark_cli_available static method."""

    def test_available_when_found(self):
        with patch("shutil.which", return_value="/usr/local/bin/lark-cli"):
            assert IdeaEngineFeishuDoc._lark_cli_available() is True

    def test_unavailable_when_not_found(self):
        with patch("shutil.which", return_value=None):
            assert IdeaEngineFeishuDoc._lark_cli_available() is False


class TestCheckLarkCli:
    """Tests for _check_lark_cli static method on IdeaEngineUtils."""

    def test_available(self):
        with patch("shutil.which", return_value="/usr/local/bin/lark-cli"):
            result = IdeaEngineUtils._check_lark_cli("doc")
            assert result == {"available": True, "domain": "doc", "error": None}

    def test_unavailable(self):
        with patch("shutil.which", return_value=None):
            result = IdeaEngineUtils._check_lark_cli("wiki")
            assert result["available"] is False
            assert result["domain"] == "wiki"
            assert "lark-cli not found" in result["error"]


class TestCallLarkCli:
    """Tests for _call_lark_cli static method."""

    def test_success_json_output(self):
        stdout = json.dumps({"id": "doc-123", "url": "https://feishu.cn/doc/123"})
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
            result = IdeaEngineFeishuDoc._call_lark_cli("doc", ["create", "--title", "Test"])
            assert result["success"] is True
            assert result["data"] == {"id": "doc-123", "url": "https://feishu.cn/doc/123"}

    def test_success_plain_text_output(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="doc-abc-456\n", stderr="")
            result = IdeaEngineFeishuDoc._call_lark_cli("wiki", ["list"])
            assert result["success"] is True
            assert result["data"] == "doc-abc-456"

    def test_success_empty_output(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = IdeaEngineFeishuDoc._call_lark_cli("calendar", ["list"])
            assert result["success"] is True
            assert result["data"] is None

    def test_file_not_found_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = IdeaEngineFeishuDoc._call_lark_cli("doc", ["create"])
            assert result["success"] is False
            assert "not found" in result["error"].lower()

    def test_timeout_error(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["lark-cli"], timeout=30)):
            result = IdeaEngineFeishuDoc._call_lark_cli("doc", ["create"], timeout=5)
            assert result["success"] is False
            assert "timeout" in result["error"].lower()

    def test_os_error(self):
        with patch("subprocess.run", side_effect=OSError("no such process")):
            result = IdeaEngineFeishuDoc._call_lark_cli("doc", ["create"])
            assert result["success"] is False
            assert "System error" in result["error"]

    def test_nonzero_exit_code(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="unauthorized: invalid token")
            result = IdeaEngineFeishuDoc._call_lark_cli("doc", ["create"])
            assert result["success"] is False
            assert "unauthorized" in result["error"]

    def test_invalid_subcmd_rejected(self):
        """Invalid subcommands are rejected before subprocess call."""
        with pytest.raises(ValueError, match="Unknown subcommand"):
            IdeaEngineFeishuDoc._call_lark_cli("rm", ["-rf", "/"])

    def test_negative_timeout_rejected(self):
        """Negative timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            IdeaEngineFeishuDoc._call_lark_cli("doc", ["create"], timeout=-1)

    def test_zero_timeout_rejected(self):
        """Zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            IdeaEngineFeishuDoc._call_lark_cli("doc", ["create"], timeout=0)

    def test_cmd_construction(self):
        """Verify the correct subprocess command is constructed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
            IdeaEngineFeishuDoc._call_lark_cli("wiki", ["search", "--query", "test"])
            call_args = mock_run.call_args[0][0]
            assert call_args == ["lark-cli", "wiki", "search", "--query", "test"]

    def test_env_has_update_notifier_disabled(self):
        """Verify NO_UPDATE_NOTIFIER env var is set."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
            IdeaEngineFeishuDoc._call_lark_cli("doc", ["create"])
            env = mock_run.call_args[1]["env"]
            assert env["LARKSUITE_CLI_NO_UPDATE_NOTIFIER"] == "1"
            assert "PATH" in env  # inherits os.environ
