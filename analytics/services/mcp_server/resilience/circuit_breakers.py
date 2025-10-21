"""Circuit Breaker Pattern - Phase 4B

This module implements circuit breakers to prevent cascade failures when
external dependencies or intensive operations fail.

Circuit breaker states:
- CLOSED: Normal operation, requests flow through
- OPEN: Failure threshold exceeded, requests fail fast
- HALF_OPEN: Testing if service recovered, limited requests allowed

Usage:
    >>> breaker = get_circuit_breaker("file_operations")
    >>> result = breaker.call(load_large_dataset, "/path/to/data.csv")
"""

from collections.abc import Callable
from typing import Any

import structlog
from pybreaker import CircuitBreaker

logger = structlog.get_logger(__name__)

# Global circuit breakers registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    fail_max: int = 5,
    timeout_duration: int = 60,
    reset_timeout: int | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name.

    Circuit breakers are singletons per name. If a breaker with the given name
    already exists, it will be returned. Otherwise, a new one is created.

    Args:
        name: Unique name for this circuit breaker (e.g., "file_operations")
        fail_max: Maximum number of failures before opening the circuit (default: 5)
        timeout_duration: Seconds to keep circuit open before trying again (default: 60)
        reset_timeout: (Deprecated) Same as timeout_duration, kept for compatibility

    Returns:
        CircuitBreaker instance

    Example:
        >>> breaker = get_circuit_breaker("database", fail_max=3, timeout_duration=30)
        >>> try:
        ...     result = breaker.call(query_database, "SELECT * FROM users")
        ... except CircuitBreakerError:
        ...     # Circuit is open, fail fast without calling query_database
        ...     result = get_cached_data()
    """
    # Use reset_timeout if provided (for backward compatibility)
    if reset_timeout is not None:
        timeout_duration = reset_timeout

    if name not in _circuit_breakers:
        logger.info(
            "creating_circuit_breaker",
            name=name,
            fail_max=fail_max,
            timeout_duration=timeout_duration,
        )

        # Create circuit breaker with listeners for state changes
        breaker = CircuitBreaker(
            fail_max=fail_max,
            timeout_duration=timeout_duration,
            name=name,
            listeners=[_CircuitBreakerListener(name)],
        )

        _circuit_breakers[name] = breaker

    return _circuit_breakers[name]


class _CircuitBreakerListener:
    """Listener for circuit breaker state changes.

    Logs state transitions for monitoring and debugging.
    """

    def __init__(self, name: str):
        self.name = name

    def before_call(self, cb: CircuitBreaker, func: Callable, *args, **kwargs):
        """Called before executing a function through the circuit breaker."""
        logger.debug(
            "circuit_breaker_before_call",
            name=self.name,
            state=cb.current_state,
            fail_count=cb.fail_counter,
        )

    def success(self, cb: CircuitBreaker):
        """Called after a successful function execution."""
        logger.debug(
            "circuit_breaker_success",
            name=self.name,
            state=cb.current_state,
            fail_count=cb.fail_counter,
        )

    def failure(self, cb: CircuitBreaker, exc: Exception):
        """Called after a failed function execution."""
        logger.warning(
            "circuit_breaker_failure",
            name=self.name,
            state=cb.current_state,
            fail_count=cb.fail_counter,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def state_change(self, cb: CircuitBreaker, old_state, new_state):
        """Called when circuit breaker state changes."""
        logger.warning(
            "circuit_breaker_state_change",
            name=self.name,
            old_state=old_state.name,
            new_state=new_state.name,
            fail_count=cb.fail_counter,
        )


def get_circuit_breaker_status() -> dict[str, dict[str, Any]]:
    """Get current status of all circuit breakers.

    Returns:
        Dict mapping circuit breaker names to their status:
        {
            "file_operations": {
                "state": "closed",  # or "open", "half_open"
                "fail_count": 0,
                "fail_max": 5,
                "timeout_duration": 60
            },
            ...
        }

    Example:
        >>> status = get_circuit_breaker_status()
        >>> for name, info in status.items():
        ...     if info["state"] == "open":
        ...         print(f"Warning: {name} circuit is open!")
    """
    status = {}

    for name, breaker in _circuit_breakers.items():
        status[name] = {
            "state": breaker.current_state.name.lower(),
            "fail_count": breaker.fail_counter,
            "fail_max": breaker.fail_max,
            "timeout_duration": breaker.timeout_duration,
        }

    return status


def reset_all_circuit_breakers():
    """Reset all circuit breakers to closed state.

    This is useful for testing or after resolving underlying issues.
    Production use should be rare - circuit breakers should recover automatically.

    Example:
        >>> reset_all_circuit_breakers()  # Reset after maintenance window
    """
    logger.info("resetting_all_circuit_breakers", count=len(_circuit_breakers))

    for name, breaker in _circuit_breakers.items():
        try:
            breaker.reset()
            logger.info("circuit_breaker_reset", name=name)
        except Exception as e:
            logger.error(
                "circuit_breaker_reset_failed",
                name=name,
                error=str(e),
                error_type=type(e).__name__,
            )


# Pre-configured circuit breakers for common operations

# File operations circuit breaker
# Protects against file system failures (disk full, permissions, corrupted files)
file_operations_breaker = get_circuit_breaker(
    name="file_operations",
    fail_max=5,  # Open after 5 failures
    timeout_duration=60,  # Stay open for 60 seconds
)

# Large dataset loading circuit breaker
# Protects against memory/performance issues when loading large datasets
large_dataset_breaker = get_circuit_breaker(
    name="large_dataset",
    fail_max=3,  # Open after 3 failures (more sensitive)
    timeout_duration=120,  # Stay open for 2 minutes (longer recovery)
)

# External API circuit breaker (if needed in future)
# Protects against external API failures
external_api_breaker = get_circuit_breaker(
    name="external_api",
    fail_max=5,
    timeout_duration=60,
)
