"""Metrics package for MCP Server

This package provides metrics collection and export for monitoring.

Phase 4A: In-memory metrics collection
Phase 4B: Prometheus metrics export
"""

from analytics.services.mcp_server.metrics.prometheus_exporter import (
    decrement_active_analyses,
    get_metrics_text,
    increment_active_analyses,
    record_analysis_duration,
    record_lens_execution,
    start_metrics_server,
    update_circuit_breaker_state,
    update_system_metrics,
)

__all__ = [
    "start_metrics_server",
    "get_metrics_text",
    "record_lens_execution",
    "record_analysis_duration",
    "increment_active_analyses",
    "decrement_active_analyses",
    "update_system_metrics",
    "update_circuit_breaker_state",
]
