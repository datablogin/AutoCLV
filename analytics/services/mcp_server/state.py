"""Shared state management for MCP server.

FastMCP's Context is per-request, so we need a shared state mechanism
to persist data (data mart, RFM metrics, cohorts) across tool calls.
"""

from typing import Any


class SharedState:
    """Thread-safe shared state storage for MCP tools."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Store a value in shared state."""
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from shared state."""
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in shared state."""
        return key in self._store

    def clear(self) -> None:
        """Clear all stored state."""
        self._store.clear()

    def keys(self) -> list[str]:
        """Get all keys in shared state."""
        return list(self._store.keys())


# Global shared state instance
_shared_state = SharedState()


def get_shared_state() -> SharedState:
    """Get the global shared state instance."""
    return _shared_state
