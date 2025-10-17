"""Test script to verify MCP tools are accessible via JSON-RPC protocol."""

import json
import subprocess
import sys

# Start the MCP server as a subprocess
server_proc = subprocess.Popen(
    [sys.executable, "-m", "analytics.services.mcp_server.main"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

try:
    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }

    server_proc.stdin.write(json.dumps(init_request) + "\n")
    server_proc.stdin.flush()

    # Read initialize response
    response = server_proc.stdout.readline()
    print(f"Initialize response: {response}")

    # Send initialized notification
    init_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }

    server_proc.stdin.write(json.dumps(init_notif) + "\n")
    server_proc.stdin.flush()

    # Send tools/list request
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }

    server_proc.stdin.write(json.dumps(tools_request) + "\n")
    server_proc.stdin.flush()

    # Read tools/list response
    response = server_proc.stdout.readline()
    print(f"\nTools/list response: {response}")

    # Parse and display tools
    try:
        resp_data = json.loads(response)
        if "result" in resp_data and "tools" in resp_data["result"]:
            tools = resp_data["result"]["tools"]
            print(f"\n✓ Found {len(tools)} tools:")
            for i, tool in enumerate(tools, 1):
                print(f"  {i}. {tool['name']}")
        else:
            print(f"\n✗ Unexpected response format: {resp_data}")
    except json.JSONDecodeError as e:
        print(f"\n✗ Failed to parse response: {e}")
        print(f"Raw response: {response}")

finally:
    # Clean up
    server_proc.terminate()
    server_proc.wait(timeout=5)
