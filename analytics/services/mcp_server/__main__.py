"""Entry point for running MCP server as a module.

This allows running the server with: python -m analytics.services.mcp_server
"""

# Import main module which registers all tools
import analytics.services.mcp_server.main

if __name__ == "__main__":
    # Access the mcp instance and run it
    analytics.services.mcp_server.main.mcp.run()
