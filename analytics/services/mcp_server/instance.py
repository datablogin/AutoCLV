"""
MCP Server Instance

This module provides the global FastMCP instance that all tools register with.
It must be imported before tools are loaded to avoid circular imports.

Architecture:
- instance.py: Creates the mcp object (imported by main.py and all tool modules)
- main.py: Configures lifespan and runs the server
- tools/*.py: Import mcp from this module and register tools with @mcp.tool()
"""

from fastmcp import FastMCP

# Version constant (updated by main.py if needed)
VERSION = "2.0.0"

# Create the global MCP instance
# Note: lifespan will be configured in main.py before running
mcp = FastMCP(name="Four Lenses Analytics", version=VERSION)
