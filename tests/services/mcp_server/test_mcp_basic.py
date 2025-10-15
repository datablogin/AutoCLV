"""
Basic tests for MCP Server

Tests for Phase 0: Infrastructure setup and basic functionality.
"""

import pytest
from analytics.services.mcp_server.main import mcp


@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """Test MCP server initializes correctly."""
    assert mcp.name == "Four Lenses Analytics"
    assert mcp.version == "0.1.0"


def test_mcp_health_check():
    """Test basic MCP server health check."""
    # Verify health_check tool is registered
    # The tool decorator registers the function, so we just verify the server has tools
    assert hasattr(mcp, "tool")
    # Verify the health_check function exists as a FunctionTool
    from analytics.services.mcp_server.main import health_check

    assert health_check is not None
    assert health_check.name == "health_check"
