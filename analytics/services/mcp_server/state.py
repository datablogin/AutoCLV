"""Shared state management for MCP server.

FastMCP's Context is per-request, so we need a shared state mechanism
to persist data (data mart, RFM metrics, cohorts) across tool calls.
"""

import threading
from typing import Any


class SharedState:
    """Thread-safe shared state storage for MCP tools.

    Uses threading.RLock for thread-safe access to shared data.
    Implements basic size-based eviction to prevent unbounded memory growth.

    Protected keys (data_mart, rfm_metrics, etc.) are only evicted as a last resort
    when all keys in the store are protected.
    """

    MAX_ITEMS = 100  # Maximum number of items to store

    # Critical keys that should not be evicted unless absolutely necessary
    PROTECTED_KEYS = frozenset(
        {
            "data_mart",
            "rfm_metrics",
            "rfm_scores",
            "cohort_definitions",
            "cohort_assignments",
        }
    )

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._lock = threading.RLock()

    def set(self, key: str, value: Any) -> None:
        """Store a value in shared state.

        If MAX_ITEMS is reached, oldest items are evicted (FIFO).
        Protected keys are only evicted if all keys in the store are protected.

        Args:
            key: Storage key
            value: Value to store
        """
        with self._lock:
            # Check if eviction is needed
            if len(self._store) >= self.MAX_ITEMS and key not in self._store:
                # Try to evict non-protected key first
                evicted = False
                for k in self._store:
                    if k not in self.PROTECTED_KEYS:
                        del self._store[k]
                        evicted = True
                        break

                # If all keys are protected, evict oldest (first) key
                if not evicted:
                    first_key = next(iter(self._store))
                    del self._store[first_key]

            self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from shared state.

        Args:
            key: Storage key
            default: Value to return if key not found

        Returns:
            Stored value or default
        """
        with self._lock:
            return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in shared state.

        Args:
            key: Storage key

        Returns:
            True if key exists
        """
        with self._lock:
            return key in self._store

    def clear(self) -> None:
        """Clear all stored state.

        Thread-safe removal of all entries.
        """
        with self._lock:
            self._store.clear()

    def keys(self) -> list[str]:
        """Get all keys in shared state.

        Returns:
            List of all storage keys (copy, not live view)
        """
        with self._lock:
            return list(self._store.keys())


# Global shared state instance
_shared_state = SharedState()


def get_shared_state() -> SharedState:
    """Get the global shared state instance."""
    return _shared_state
