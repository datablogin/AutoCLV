"""
Four Lenses Analytics MCP Server

This module provides the main MCP server for AutoCLV Four Lenses customer analytics.
Phase 0: Basic scaffold with observability setup.
"""

import logging
import sys
from contextlib import asynccontextmanager

import structlog
from fastmcp import Context, FastMCP

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
    logger.info("mcp_server_starting", phase="Phase 1 - Foundation Services")

    # Phase 1: No resources to initialize yet
    # Phase 4: Initialize observability, database pools, config, etc.

    yield

    # Shutdown
    logger.info("mcp_server_stopping")


# Initialize MCP server
mcp = FastMCP(name="Four Lenses Analytics", version="0.1.0", lifespan=app_lifespan)


@mcp.tool()
async def health_check(ctx: Context) -> dict:
    """
    Check health of MCP server.

    Returns:
        Health status for the MCP server
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "phase": "Phase 1 - Foundation Services",
        "server_name": "Four Lenses Analytics",
    }


# Import Phase 1 Foundation Tools
# This registers the tools with the MCP server
from analytics.services.mcp_server.tools import (
    cohorts,  # noqa: F401, E402
    data_mart,  # noqa: F401, E402
    rfm,  # noqa: F401, E402
)

# Import Phase 2 Lens Tools
from analytics.services.mcp_server.tools import (
    lens1,  # noqa: F401, E402
    lens2,  # noqa: F401, E402
    lens3,  # noqa: F401, E402
    lens4,  # noqa: F401, E402
)

logger.info("mcp_server_initialized", phase="Phase 2", tools_registered=7)


if __name__ == "__main__":
    mcp.run()
