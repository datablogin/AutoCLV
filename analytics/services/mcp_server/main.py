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
    logger.info(
        "mcp_server_starting",
        version="1.1.0",
        phase="Phase 3 - COMPLETE: ALL LENSES FULLY OPERATIONAL!",
    )

    # Phase 1: No resources to initialize yet
    # Phase 4: Initialize observability, database pools, config, etc.

    yield

    # Shutdown
    logger.info("mcp_server_stopping")


# Initialize MCP server
mcp = FastMCP(name="Four Lenses Analytics", version="1.1.0", lifespan=app_lifespan)


@mcp.tool()
async def health_check(ctx: Context) -> dict:
    """
    Check health of MCP server.

    Returns:
        Health status for the MCP server
    """
    return {
        "status": "healthy",
        "version": "1.1.0",
        "phase": "Phase 3 - COMPLETE ✅🎉 ALL LENSES FULLY OPERATIONAL!",
        "server_name": "Four Lenses Analytics",
        "capabilities": [
            "Load transactions from file with automatic datetime parsing",
            "Auto-build data mart from loaded transactions",
            "Auto-calculate RFM metrics",
            "Auto-create customer cohorts",
            "Orchestrated multi-lens analysis with natural language queries",
            "Full Lens 1 support (Current Period Snapshot)",
            "Full Lens 4 support (Multi-Cohort Comparison)",
            "Full Lens 5 support (Overall Customer Base Health) - FULLY WORKING!",
            "Parallel lens execution for optimal performance",
            "Manual and automatic foundation building",
            "Detailed error tracking per lens with specific error messages",
        ],
        "version_history": [
            "v1.1.0: REALLY THE FINAL FIX! - health_score field (not overall_health_score)",
            "v1.0.5: Lens 5 total_customers accessed from health_score object",
            "v1.0.4: Added detailed error tracking - lens_errors shows specific error messages",
            "v1.0.3: Fixed Lens 5 import error (_identify_key_strengths vs _format_insights)",
            "v1.0.2: Fixed build_customer_data_mart to store period_aggregations",
            "v1.0.1: Fixed intent parsing - 'overall health' routes to Lens 5 only",
            "v1.0.0: Fixed Lens 5 orchestration - correct function signature",
            "v0.6.0: Fixed datetime parsing - enables auto-foundation building",
            "v0.5.1: Load transactions returns summary only",
            "v0.5.0: Fixed Lens 5 period_aggregations storage",
        ],
    }


# Import Phase 1 Foundation Tools
# Import Phase 2 Lens Tools
# Import Phase 3 Orchestration Tool
# These imports MUST happen before mcp.run() is called
# Each module registers its tools using the @mcp.tool() decorator
from analytics.services.mcp_server.tools import (  # noqa: E402, F401
    cohorts,
    data_loader,
    data_mart,
    lens1,
    lens2,
    lens3,
    lens4,
    lens5,
    orchestrated_analysis,
    rfm,
)

logger.info("mcp_server_initialized", phase="Phase 3", tools_registered=11)


if __name__ == "__main__":
    mcp.run()
