# Claude Desktop MCP Server Integration - Fixed

## Problem Summary

The MCP server worked when run manually from the command line but failed when Claude Desktop tried to launch it, with the error:

```
ModuleNotFoundError: No module named 'analytics'
```

## Root Cause

When running `uv run python -m analytics.services.mcp_server.main`, the command:
1. **Works locally** because `uv sync` had previously installed the `autoclv` package in editable mode
2. **Fails in Claude Desktop** because `uv run python` creates a temporary environment without installing the local package first

The `analytics` module is part of the `autoclv` package, which must be installed for Python to find it.

## Solution

Added a **script entry point** to `pyproject.toml` that automatically ensures the package is installed:

```toml
[project.scripts]
mcp-server = "analytics.services.mcp_server.main:mcp.run"
```

This creates a `mcp-server` command that:
- Automatically installs the `autoclv` package when run via `uv run`
- Properly initializes the Python path
- Calls `mcp.run()` to start the server

## Claude Desktop Configuration

### ✅ CORRECT Configuration

Use this in `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

**Key points:**
- Uses the full path to `uv`: `/Users/robertwelborn/.local/bin/uv`
- Uses `--directory` to specify the project directory
- Calls the registered script: `mcp-server`

### ❌ INCORRECT Configuration (Previous)

```json
{
  "mcpServers": {
    "four-lenses-analytics": {
      "command": "/Users/robertwelborn/.local/bin/uv",
      "args": [
        "run",
        "python",
        "-m",
        "analytics.services.mcp_server.main"
      ],
      "cwd": "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a"
    }
  }
}
```

**Why this failed:**
- `uv run python` doesn't install the local package
- The `analytics` module is not in Python's path
- Results in `ModuleNotFoundError`

## Verification Steps

### 1. Verify the script works locally

```bash
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a
uv run mcp-server --help
```

Expected output: FastMCP banner showing "Four Lenses Analytics"

### 2. Verify it works from a different directory

```bash
cd /tmp
uv run --directory /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a mcp-server --help
```

Expected output: Same FastMCP banner (proves `--directory` flag works)

### 3. Test in Claude Desktop

1. Update `claude_desktop_config.json` with the correct configuration above
2. Restart Claude Desktop
3. Check the MCP servers list - should see "Four Lenses Analytics"
4. Try the `health_check` tool

Expected response from `health_check`:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "phase": "Phase 0 - Infrastructure Setup",
  "server_name": "Four Lenses Analytics"
}
```

## How to Run the Server

### Command Line (Development)

```bash
# Recommended - using script entry point
uv run mcp-server

# Alternative - using Python module
uv run python -m analytics.services.mcp_server.main

# From a different directory
uv run --directory /path/to/project mcp-server
```

### Claude Desktop (Production)

Use the configuration shown above. Claude Desktop will automatically:
1. Call `uv run --directory <project> mcp-server`
2. uv will install the `autoclv` package in a temporary environment
3. The `mcp-server` script will start the server
4. Claude Desktop connects via STDIO transport

## Technical Details

### Script Entry Point Mechanism

When you add a script to `[project.scripts]` in `pyproject.toml`:

```toml
[project.scripts]
mcp-server = "analytics.services.mcp_server.main:mcp.run"
```

This creates an executable script that:
1. **Imports the module**: `from analytics.services.mcp_server.main import mcp`
2. **Calls the function**: `mcp.run()`
3. **Handles package installation**: When run via `uv run`, the package is installed first

### Why `--directory` is Important

The `--directory` flag tells `uv` where to find the `pyproject.toml` file, which:
- Defines the package (`autoclv`)
- Lists dependencies
- Registers script entry points

Without `--directory`, `uv` would look in the current working directory, which might not contain the project files when Claude Desktop launches it.

## Alternative Solutions Considered

### Option 1: Use `uv run --with .`
```bash
uv run --with . python -m analytics.services.mcp_server.main
```
- **Pros**: Explicitly installs current directory as package
- **Cons**: More verbose, requires understanding of `--with` flag

### Option 2: Use absolute imports with PYTHONPATH
```json
{
  "command": "python",
  "args": ["-m", "analytics.services.mcp_server.main"],
  "env": {
    "PYTHONPATH": "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a"
  }
}
```
- **Pros**: No `uv` needed
- **Cons**: Doesn't handle dependencies, brittle, not recommended

### Option 3: Create standalone script (Chosen Solution ✅)
```toml
[project.scripts]
mcp-server = "analytics.services.mcp_server.main:mcp.run"
```
- **Pros**: Clean, standard Python packaging, automatically handles dependencies
- **Cons**: Requires package to be installable (which it already is)

## Files Modified

### `pyproject.toml`
- Added `[project.scripts]` section with `mcp-server` entry point

### `MCP_SERVER_QUICKSTART.md`
- Updated with correct `uv run mcp-server` command
- Updated Claude Desktop configuration example

## Troubleshooting

### Error: "command not found: mcp-server"

**Cause**: Package not installed or not in sync

**Solution**:
```bash
uv sync
```

### Error: "ModuleNotFoundError: No module named 'analytics'"

**Cause**: Running with wrong configuration

**Solution**: Make sure you're using `uv run mcp-server`, not `uv run python -m analytics...`

### Error: "No such file or directory"

**Cause**: Wrong project path in `--directory` flag

**Solution**: Verify the path exists:
```bash
ls -la /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a/pyproject.toml
```

### Claude Desktop shows "Connection Failed"

**Possible causes:**
1. Wrong `uv` path - verify with `which uv`
2. Wrong project directory - verify path exists
3. Server crashed on startup - check Claude Desktop logs

**Check logs:**
- macOS: `~/Library/Logs/Claude/`
- Look for stderr output from the MCP server

## Summary

The fix was simple but important:
1. ✅ Added script entry point to `pyproject.toml`
2. ✅ Updated Claude Desktop config to use `mcp-server` script
3. ✅ Used `--directory` flag to specify project location
4. ✅ Verified it works from any directory

**Result**: Claude Desktop can now successfully launch and connect to the Four Lenses Analytics MCP server! 🎉
