"""Resilience patterns for MCP Server

This package provides resilience patterns for handling failures:
- Circuit breakers: Prevent cascade failures
- Retry logic: Automatic retry with exponential backoff (implemented in coordinator.py)
- Health monitoring: Track circuit breaker states
"""

from analytics.services.mcp_server.resilience.circuit_breakers import (
    get_circuit_breaker,
    get_circuit_breaker_status,
    reset_all_circuit_breakers,
)

__all__ = [
    "get_circuit_breaker",
    "get_circuit_breaker_status",
    "reset_all_circuit_breakers",
]
