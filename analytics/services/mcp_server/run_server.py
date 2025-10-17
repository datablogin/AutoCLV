#!/usr/bin/env python3
"""
Standalone MCP server runner for Four Lenses Analytics.

This script explicitly imports all tool modules to ensure they're registered
with the MCP server before it starts accepting connections.
"""

import sys
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the main MCP server instance first
from analytics.services.mcp_server.main import mcp

# Explicitly import all tool modules to ensure registration
# This MUST happen before mcp.run() is called

# Print confirmation of tool registration
print(f"✓ MCP Server: {mcp.name} v{mcp.version}", file=sys.stderr)
print(f"✓ Registered tools: {len(mcp._tool_manager._tools)}", file=sys.stderr)
for name in sorted(mcp._tool_manager._tools.keys()):
    print(f"  - {name}", file=sys.stderr)
print("✓ Starting MCP server...", file=sys.stderr)
sys.stderr.flush()

# Run the server
if __name__ == "__main__":
    mcp.run()
