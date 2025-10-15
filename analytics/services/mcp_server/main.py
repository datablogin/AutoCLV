"""
Four Lenses Analytics MCP Server

This module provides the main MCP server for AutoCLV Four Lenses customer analytics.
Phase 0: Basic scaffold with observability setup.
"""

from fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional
import structlog
import sys
import logging

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


@dataclass
class AppContext:
    """Application-wide resources for MCP server."""
    config: dict


@asynccontextmanager
async def app_lifespan(app):
    """Initialize and cleanup MCP server resources."""
    logger.info("mcp_server_starting", phase="Phase 0 - Infrastructure Setup")

    # Startup
    config = {
        "max_lookback_days": 730,
        "default_discount_rate": 0.1,
    }

    yield AppContext(config=config)

    # Shutdown
    logger.info("mcp_server_stopping")


# Initialize MCP server
mcp = FastMCP(
    name="Four Lenses Analytics",
    version="0.1.0",
    lifespan=app_lifespan
)


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
        "server_name": "Four Lenses Analytics"
    }


# Import Phase 1 Foundation Tools
# This registers the tools with the MCP server
from analytics.services.mcp_server.tools import data_mart  # noqa: F401, E402
from analytics.services.mcp_server.tools import rfm  # noqa: F401, E402
from analytics.services.mcp_server.tools import cohorts  # noqa: F401, E402

logger.info("mcp_server_initialized", phase="Phase 1", tools_registered=3)


if __name__ == "__main__":
    mcp.run()
