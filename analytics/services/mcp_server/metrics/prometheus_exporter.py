"""Prometheus Metrics Exporter - Phase 4B

This module exports MCP server metrics to Prometheus for monitoring and alerting.

Metrics exported:
- lens_execution_duration_seconds: Histogram of lens execution times
- lens_execution_total: Counter of total lens executions by status
- active_analyses: Gauge of currently active analyses
- system_memory_bytes: Gauge of process memory usage
- system_cpu_percent: Gauge of CPU usage

Usage:
    # Start Prometheus metrics server on port 8000
    >>> start_metrics_server(port=8000)

    # Metrics available at http://localhost:8000/metrics
"""

import time
from threading import Lock

import structlog
from prometheus_client import Counter, Gauge, Histogram, generate_latest, start_http_server

logger = structlog.get_logger(__name__)

# Metrics definitions
lens_execution_duration = Histogram(
    "lens_execution_duration_seconds",
    "Lens execution duration in seconds",
    ["lens_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

lens_execution_total = Counter(
    "lens_execution_total",
    "Total lens executions",
    ["lens_name", "status"],  # status: success or failure
)

active_analyses = Gauge("active_analyses", "Number of currently active analyses")

analysis_duration = Histogram(
    "analysis_duration_seconds",
    "Overall analysis duration in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

system_memory_bytes = Gauge("system_memory_bytes", "Process memory usage in bytes")

system_cpu_percent = Gauge("system_cpu_percent", "Process CPU usage percentage")

# Circuit breaker state gauge
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["breaker_name"],
)

# Global state
_metrics_server_started = False
_metrics_lock = Lock()


def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics HTTP server.

    Args:
        port: Port to expose metrics on (default: 8000)

    Raises:
        RuntimeError: If metrics server is already running

    Example:
        >>> start_metrics_server(port=9090)
        # Metrics available at http://localhost:9090/metrics
    """
    global _metrics_server_started

    with _metrics_lock:
        if _metrics_server_started:
            raise RuntimeError("Metrics server is already running")

        try:
            start_http_server(port)
            _metrics_server_started = True
            logger.info("prometheus_metrics_server_started", port=port)
        except Exception as e:
            logger.error(
                "prometheus_metrics_server_failed", port=port, error=str(e), error_type=type(e).__name__
            )
            raise


def get_metrics_text() -> bytes:
    """Get current Prometheus metrics in text format.

    Returns:
        Metrics in Prometheus text format

    Example:
        >>> metrics = get_metrics_text()
        >>> print(metrics.decode('utf-8'))
        # HELP lens_execution_duration_seconds Lens execution duration in seconds
        # TYPE lens_execution_duration_seconds histogram
        ...
    """
    return generate_latest()


def record_lens_execution(lens_name: str, duration_seconds: float, success: bool):
    """Record a lens execution.

    Args:
        lens_name: Name of the lens (e.g., 'lens1', 'lens5')
        duration_seconds: Execution duration in seconds
        success: Whether execution succeeded

    Example:
        >>> record_lens_execution('lens1', 1.5, True)
    """
    status = "success" if success else "failure"

    lens_execution_duration.labels(lens_name=lens_name).observe(duration_seconds)
    lens_execution_total.labels(lens_name=lens_name, status=status).inc()


def record_analysis_duration(duration_seconds: float):
    """Record an overall analysis duration.

    Args:
        duration_seconds: Analysis duration in seconds

    Example:
        >>> record_analysis_duration(5.2)
    """
    analysis_duration.observe(duration_seconds)


def increment_active_analyses():
    """Increment the count of active analyses.

    Call this when an analysis starts.

    Example:
        >>> increment_active_analyses()
    """
    active_analyses.inc()


def decrement_active_analyses():
    """Decrement the count of active analyses.

    Call this when an analysis completes (success or failure).

    Example:
        >>> decrement_active_analyses()
    """
    active_analyses.dec()


def update_system_metrics():
    """Update system resource metrics (memory, CPU).

    Call this periodically to update resource usage metrics.
    Requires psutil to be installed.

    Example:
        >>> update_system_metrics()  # Updates memory and CPU gauges
    """
    try:
        import psutil

        process = psutil.Process()

        # Memory usage
        memory_info = process.memory_info()
        system_memory_bytes.set(memory_info.rss)

        # CPU usage (non-blocking, 0.1s interval)
        cpu_percent = process.cpu_percent(interval=0.1)
        system_cpu_percent.set(cpu_percent)

    except ImportError:
        logger.debug("psutil_not_available_skipping_system_metrics")
    except Exception as e:
        logger.warning("system_metrics_update_failed", error=str(e), error_type=type(e).__name__)


def update_circuit_breaker_state(breaker_name: str, state: str):
    """Update circuit breaker state metric.

    Args:
        breaker_name: Name of the circuit breaker
        state: State name ('closed', 'open', 'half_open')

    Example:
        >>> update_circuit_breaker_state('file_operations', 'open')
    """
    state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state.lower(), 0)

    circuit_breaker_state.labels(breaker_name=breaker_name).set(state_value)
