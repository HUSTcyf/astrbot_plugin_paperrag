"""PaperRAG plugin shared constants and lightweight helpers."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import requests
from astrbot.api import logger

SUPPORTED_DOC_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm']
"""支持的文档扩展名列表"""


def _is_hidden_file(file_path: Path) -> bool:
    """检测文件是否为 macOS 元数据文件（以 ._ 开头）"""
    return file_path.name.startswith("._")


class Neo4jServiceManager:
    """
    Neo4j 原生服务管理：检查/启动 Neo4j 服务

    使用方式：
        manager = Neo4jServiceManager()
        await manager.ensure_neo4j_running()
    """

    def __init__(
        self,
        neo4j_config: Optional[dict] = None
    ):
        self.neo4j_config = neo4j_config or {
            "host": "localhost",
            "port": 7687,
            "http_port": 7474,
            "user": "neo4j",
            "password": "password",
            "neo4j_home": "/usr/local/var/neo4j",  # macOS Homebrew 默认
        }

    def _is_neo4j_available(self) -> bool:
        """检查 Neo4j 是否可用"""
        try:
            result = subprocess.run(
                ["neo4j", "status"],
                capture_output=True,
                text=True,
                check=False
            )
            return "running" in result.stdout.lower() or result.returncode == 0
        except FileNotFoundError:
            pass

        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("localhost", self.neo4j_config["port"]))
            sock.close()
            return result == 0
        except Exception:
            return False

    async def ensure_neo4j_running(self) -> bool:
        """
        检测 Neo4j 是否可连接（不再自动启动）

        Returns:
            Neo4j 是否可连接
        """

        if self._is_neo4j_available():
            logger.info("[Neo4j] Neo4j 服务已连接")
            return True

        logger.warning("[Neo4j] Neo4j 不可连接，请确保 Neo4j 已启动")
        logger.info("[Neo4j] 可用命令:")
        logger.info("  neo4j start")
        logger.info("  brew services start neo4j")
        return False

    def get_connection_info(self) -> dict:
        """获取 Neo4j 连接信息"""
        return {
            "uri": f"bolt://{self.neo4j_config['host']}:{self.neo4j_config['port']}",
            "user": self.neo4j_config["user"],
            "password": self.neo4j_config["password"],
            "http_port": self.neo4j_config["http_port"],
        }


class CoreAPIClient:
    """CORE API v3 客户端 - 用于搜索和下载开放获取论文"""

    BASE_URL = "https://api.core.ac.uk/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def search_by_title(self, title: str, year: Optional[int] = None, limit: int = 5) -> list:
        """根据论文标题搜索论文"""

        query_parts = [f'title:"{title}"']
        if year:
            query_parts.append(f"publishedDate:{year}")
        query = " AND ".join(query_parts)

        try:
            response = requests.post(
                f"{self.BASE_URL}/search/works",
                headers=self.headers,
                json={"q": query, "limit": min(limit, 100)},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            logger.error(f"CORE API搜索失败: {e}")
            return []

    def extract_arxiv_id(self, work: dict) -> Optional[str]:
        """从 work 记录中提取 arXiv ID"""
        urls = work.get("sourceFulltextUrls", []) or []
        for url in urls:
            if url and "arxiv.org" in url:
                match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url)
                if match:
                    return match.group(1)
        return None
