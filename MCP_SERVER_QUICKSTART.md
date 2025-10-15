# MCP Server Quick Start Guide

## Running the Server

```bash
# Start the MCP server (STDIO transport) - Recommended
uv run mcp-server

# Alternative: Run via Python module
uv run python -m analytics.services.mcp_server.main

# View available options
uv run mcp-server --help
```

## Running Tests

```bash
# Run all MCP server tests
uv run pytest tests/services/mcp_server/ -v

# Run with verbose output
uv run pytest tests/services/mcp_server/ -vv

# Run specific test
uv run pytest tests/services/mcp_server/test_mcp_basic.py::test_mcp_server_initialization -v
```

## Development Workflow

### After Modifying Dependencies

```bash
uv sync
```

### After Modifying Code

```bash
# Run tests
uv run pytest tests/services/mcp_server/ -v

# Import verification
uv run python -c "from analytics.services.mcp_server.main import mcp; print(f'✓ {mcp.name} v{mcp.version}')"
```

## Claude Desktop Integration

To connect Claude Desktop to the local MCP server:

1. **Locate Claude Desktop config file:**
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%/Claude/claude_desktop_config.json`

2. **Add MCP server configuration:**

```json
{
  "mcpServers": {
    "four-lenses-analytics": {
      "command": "/Users/robertwelborn/.local/bin/uv",
      "args": [
        "run",
        "--directory",
        "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a",
        "mcp-server"
      ]
    }
  }
}
```

**Note**: The `mcp-server` script is registered via the `pyproject.toml` `[project.scripts]` section.

3. **Restart Claude Desktop**

4. **Verify connection:**
   - Look for "Four Lenses Analytics" in the MCP server list
   - Try invoking the `health_check` tool

## Available Tools

### health_check

Returns the health status of the MCP server.

**Example response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "phase": "Phase 0 - Infrastructure Setup",
  "server_name": "Four Lenses Analytics"
}
```

## Troubleshooting

### ModuleNotFoundError: No module named 'fastmcp'

**Solution:** Run `uv sync` to install dependencies.

### VIRTUAL_ENV mismatch warning

This warning is normal when running with `uv` in a worktree. It doesn't affect functionality.

### Server won't start

1. Verify you're in the correct directory:
   ```bash
   pwd
   # Should be: /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a
   ```

2. Check dependencies are installed:
   ```bash
   uv run python -c "import fastmcp; print(fastmcp.__version__)"
   ```

3. Verify the package structure:
   ```bash
   ls -la analytics/services/mcp_server/
   # Should see: __init__.py, main.py, observability.py
   ```

## Directory Structure

```
analytics/
  services/
    mcp_server/
      __init__.py
      main.py              # MCP server entry point
      observability.py     # OpenTelemetry configuration

tests/
  services/
    mcp_server/
      __init__.py
      test_mcp_basic.py    # Infrastructure tests
```

## Next Steps (Phase 1)

Phase 1 will add:
- Foundation Services (Data Mart, RFM, Cohorts)
- Database connectivity
- Context-based state management
- Additional MCP tools

See `thoughts/shared/plans/2025-10-14-agentic-five-lenses-implementation.md` for the full implementation plan.
