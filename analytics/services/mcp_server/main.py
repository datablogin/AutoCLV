"""
Four Lenses Analytics MCP Server

This module provides the main MCP server for AutoCLV Four Lenses customer analytics.
Phase 0: Basic scaffold with observability setup.
"""

import logging
import sys
from contextlib import asynccontextmanager

import structlog

# Version and phase constants
VERSION = "2.0.0"
PHASE = (
    "Phase 5 - Natural Language Interface with LLM-powered query parsing and synthesis"
)

# Configure structlog to write to stderr, not stdout (to avoid interfering with MCP JSON protocol)
logging.basicConfig(
    format="%(message)s",
    stream=sys.stderr,
    level=logging.INFO,
)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def app_lifespan(app):
    """Initialize and cleanup MCP server resources.

    Phase 1: Minimal lifespan management (startup/shutdown logging only).
    Phase 4A/4B: Observability initialization with OpenTelemetry and metrics.

    Note: AppContext was removed as it's not accessed by tools in Phase 1.
    Configuration will be reintroduced in Phase 4 when needed for observability setup.
    """
    logger.info(
        "mcp_server_starting",
        version=VERSION,
        phase=PHASE,
    )

    # Phase 4A/4B: Initialize observability (OpenTelemetry tracing + metrics)
    import os

    from analytics.services.mcp_server.metrics import start_metrics_server
    from analytics.services.mcp_server.observability import configure_observability

    otlp_endpoint = os.getenv("OTLP_ENDPOINT")
    environment = os.getenv("ENVIRONMENT", "development")
    sampling_rate = float(os.getenv("SAMPLING_RATE", "1.0"))
    metrics_port = int(os.getenv("PROMETHEUS_METRICS_PORT", "8000"))

    tracer, meter = configure_observability(
        service_name="mcp-four-lenses",
        environment=environment,
        otlp_endpoint=otlp_endpoint,
        sampling_rate=sampling_rate,
    )

    logger.info(
        "observability_initialized",
        otlp_enabled=otlp_endpoint is not None,
        otlp_endpoint=otlp_endpoint,
        environment=environment,
        sampling_rate=sampling_rate,
    )

    # Phase 4B: Start Prometheus metrics HTTP server
    try:
        start_metrics_server(port=metrics_port)
        logger.info("prometheus_metrics_server_started", port=metrics_port)
    except RuntimeError as e:
        # Server already running (e.g., during hot reload)
        logger.warning("prometheus_metrics_server_already_running", error=str(e))
    except Exception as e:
        logger.error(
            "prometheus_metrics_server_failed", error=str(e), port=metrics_port
        )

    yield

    # Shutdown
    logger.info("mcp_server_stopping")


# Import MCP server instance (must be imported before tools to avoid circular imports)
from analytics.services.mcp_server.instance import mcp  # noqa: E402

# Configure lifespan
mcp.lifespan = app_lifespan


# Import Phase 1 Foundation Tools
# Import Phase 2 Lens Tools
# Import Phase 3 Orchestration Tool
# Import Phase 4A Observability Tools
# Import Phase 5 Conversational Analysis Tool
# These imports MUST happen before mcp.run() is called
# Each module registers its tools using the @mcp.tool() decorator
from analytics.services.mcp_server.tools import (  # noqa: E402, F401
    cohorts,
    conversational_analysis,
    data_loader,
    data_mart,
    debug_env,
    execution_metrics,
    health_check,
    lens1,
    lens2,
    lens3,
    lens4,
    lens5,
    orchestrated_analysis,
    rfm,
)

logger.info(
    "mcp_server_initialized",
    phase="Phase 5",
    tools_registered=16,
    new_tools=[
        "health_check",
        "get_execution_metrics",
        "reset_execution_metrics",
        "run_conversational_analysis",
        "debug_environment",
    ],
)


if __name__ == "__main__":
    mcp.run()
