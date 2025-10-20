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
    assert mcp.version == "2.0.0"  # Phase 5: Natural Language Interface


def test_mcp_health_check():
    """Test basic MCP server health check."""
    # Verify health_check module is imported (tool is registered via @mcp.tool decorator)
    from analytics.services.mcp_server.tools import health_check

    assert health_check is not None
    # The tool decorator registers the function with the MCP server
    assert hasattr(mcp, "tool")
