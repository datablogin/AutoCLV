"""Entry point for running MCP server as a module.

This allows running the server with: python -m analytics.services.mcp_server
"""

if __name__ == "__main__":
    try:
        # Import main module which registers all tools
        from analytics.services.mcp_server.main import mcp

        # Run the MCP server
        mcp.run()
    except ImportError as e:
        import sys

        print(f"Error: Failed to import MCP server: {e}", file=sys.stderr)
        print(
            "Ensure all dependencies are installed (uv sync) and PYTHONPATH is set correctly",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        import sys

        print(f"Error: Failed to start MCP server: {e}", file=sys.stderr)
        sys.exit(1)
