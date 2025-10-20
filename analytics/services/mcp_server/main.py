"""
Four Lenses Analytics MCP Server

This module provides the main MCP server for AutoCLV Four Lenses customer analytics.
Phase 0: Basic scaffold with observability setup.
"""

import logging
import sys
from contextlib import asynccontextmanager

import structlog
from fastmcp import FastMCP

# Version and phase constants
VERSION = "2.0.0"
PHASE = "Phase 5 - Natural Language Interface with LLM-powered query parsing and synthesis"

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
    Phase 4: Will add observability initialization, resource pooling, and graceful shutdown.

    Note: AppContext was removed as it's not accessed by tools in Phase 1.
    Configuration will be reintroduced in Phase 4 when needed for observability setup.
    """
    logger.info(
        "mcp_server_starting",
        version=VERSION,
        phase=PHASE,
    )

    # Phase 1: No resources to initialize yet
    # Phase 4: Initialize observability, database pools, config, etc.

    yield

    # Shutdown
    logger.info("mcp_server_stopping")


# Initialize MCP server
mcp = FastMCP(name="Four Lenses Analytics", version=VERSION, lifespan=app_lifespan)


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
