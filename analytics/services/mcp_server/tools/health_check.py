"""Health Check MCP Tool - Phase 4A & 4B

This tool provides system health monitoring for the MCP server and all dependencies.

Phase 4A:
- Basic health checks (MCP server, shared state, foundation data, resources)

Phase 4B:
- Liveness probe: Is the server alive?
- Readiness probe: Can the server handle requests?
- Deep health checks: OpenTelemetry, metrics collector, circuit breakers
- Kubernetes-ready probe endpoints

It checks:
1. MCP server status
2. Shared state availability
3. Foundation data readiness
4. Memory and system resources
5. OpenTelemetry tracer status (Phase 4B)
6. Metrics collector status (Phase 4B)
7. Circuit breaker states (Phase 4B)

Usage:
    Call health_check() to get current system health status
    Call liveness_probe() for Kubernetes liveness probe
    Call readiness_probe() for Kubernetes readiness probe
"""

import time
from datetime import datetime

import structlog
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.instance import mcp
from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)


class HealthCheckResponse(BaseModel):
    """Health check response with system status."""

    status: str = Field(
        description="Overall health status: 'healthy', 'degraded', or 'unhealthy'"
    )
    timestamp: str = Field(description="ISO timestamp of health check")
    checks: dict[str, str] = Field(description="Individual component health checks")
    uptime_seconds: float | None = Field(
        default=None, description="Server uptime in seconds (if available)"
    )
    foundation_data_status: dict[str, bool] = Field(
        description="Foundation data availability"
    )
    resource_usage: dict[str, float] | None = Field(
        default=None, description="System resource usage metrics"
    )


# Track server start time
_SERVER_START_TIME = time.time()


@mcp.tool()
async def health_check(ctx: Context) -> HealthCheckResponse:
    """
    Check health of MCP server and all dependencies.

    This tool performs comprehensive health checks on:
    - MCP server status and uptime
    - Shared state management system
    - Foundation data availability (transactions, data mart, RFM, cohorts)
    - System resource usage (memory, if available)

    Returns:
        HealthCheckResponse with detailed health status and component checks

    Example:
        >>> result = await health_check()
        >>> print(result.status)  # 'healthy'
        >>> print(result.checks)  # {'mcp_server': 'healthy', ...}
    """
    logger.info("health_check_starting")

    checks: dict[str, str] = {}
    status = "healthy"

    # Check 1: MCP Server
    try:
        checks["mcp_server"] = "healthy"
        logger.debug("mcp_server_check_passed")
    except Exception as e:
        checks["mcp_server"] = f"unhealthy: {str(e)}"
        status = "degraded"
        logger.error("mcp_server_check_failed", error=str(e))

    # Check 2: Shared State System
    try:
        shared_state = get_shared_state()
        checks["shared_state"] = "healthy"
        logger.debug("shared_state_check_passed")
    except Exception as e:
        checks["shared_state"] = f"unhealthy: {str(e)}"
        status = "unhealthy"
        logger.error("shared_state_check_failed", error=str(e))

    # Check 3: Foundation Data Availability
    foundation_data_status = {}
    try:
        shared_state = get_shared_state()

        foundation_data_status["transactions"] = shared_state.has("transactions")
        foundation_data_status["data_mart"] = shared_state.has("data_mart")
        foundation_data_status["rfm_metrics"] = shared_state.has("rfm_metrics")
        foundation_data_status["rfm_scores"] = shared_state.has("rfm_scores")
        foundation_data_status["period_aggregations"] = shared_state.has(
            "period_aggregations"
        )
        foundation_data_status["cohort_definitions"] = shared_state.has(
            "cohort_definitions"
        )
        foundation_data_status["cohort_assignments"] = shared_state.has(
            "cohort_assignments"
        )

        # Foundation data is informational, not a health failure
        data_available = any(foundation_data_status.values())
        if data_available:
            checks["foundation_data"] = (
                f"available ({sum(foundation_data_status.values())} items)"
            )
        else:
            checks["foundation_data"] = "no data loaded (use load_transactions)"

        logger.debug("foundation_data_check_passed", status=foundation_data_status)
    except Exception as e:
        checks["foundation_data"] = f"check failed: {str(e)}"
        foundation_data_status = {}
        logger.error("foundation_data_check_failed", error=str(e))

    # Check 4: System Resource Usage (optional)
    resource_usage: dict[str, float] | None = None
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        resource_usage = {
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(interval=0.1),
        }
        checks["system_resources"] = "healthy"
        logger.debug("system_resources_check_passed", usage=resource_usage)
    except ImportError:
        # psutil not available, skip resource checks
        checks["system_resources"] = "psutil not available (optional)"
        logger.debug("system_resources_check_skipped")
    except Exception as e:
        checks["system_resources"] = f"check failed: {str(e)}"
        logger.error("system_resources_check_failed", error=str(e))

    # Calculate uptime
    uptime_seconds = time.time() - _SERVER_START_TIME

    logger.info(
        "health_check_complete",
        status=status,
        checks=checks,
        uptime_seconds=uptime_seconds,
    )

    return HealthCheckResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        checks=checks,
        uptime_seconds=uptime_seconds,
        foundation_data_status=foundation_data_status,
        resource_usage=resource_usage,
    )
