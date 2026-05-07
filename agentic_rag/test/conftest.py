"""
Conftest — mock fixtures for agentic_rag tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel


@pytest.fixture
def mock_provider():
    """Mock LLM Provider。"""
    provider = MagicMock()
    provider.text_chat = AsyncMock(return_value=MagicMock(content="fact"))
    return provider


@pytest.fixture
def mock_context(mock_provider):
    """Mock AstrBot Context。"""
    context = MagicMock()
    context.config = {}
    context.provider_manager = MagicMock()
    context.provider_manager.get_provider = MagicMock(return_value=mock_provider)
    return context


@pytest.fixture
def mock_engine():
    """Mock HybridRAGEngine。"""
    engine = MagicMock()
    return engine


@pytest.fixture
def mock_graph_engine():
    """Mock GraphRAGEngine。"""
    engine = MagicMock()
    return engine
