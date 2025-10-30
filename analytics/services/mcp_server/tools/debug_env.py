"""Debug Environment Variables Tool

Quick diagnostic tool to check if environment variables are accessible to the MCP server.
"""

import os

import structlog
from fastmcp import Context
from pydantic import BaseModel

from analytics.services.mcp_server.instance import mcp

logger = structlog.get_logger(__name__)


class DebugEnvResponse(BaseModel):
    """Response with environment variable status."""

    anthropic_api_key_present: bool
    anthropic_api_key_length: int | None
    anthropic_api_key_prefix: str | None
    python_path_set: bool
    all_env_vars: list[str]


@mcp.tool()
async def debug_environment(ctx: Context) -> DebugEnvResponse:
    """
    Check environment variables accessible to the MCP server.

    This is a diagnostic tool to verify that ANTHROPIC_API_KEY and other
    environment variables are properly set and accessible.

    Returns:
        Environment variable status and details
    """
    await ctx.info("Checking environment variables...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    python_path = os.getenv("PYTHONPATH")

    # Get all environment variable names (not values for security)
    all_vars = sorted(list(os.environ.keys()))

    response = DebugEnvResponse(
        anthropic_api_key_present=api_key is not None,
        anthropic_api_key_length=len(api_key) if api_key else None,
        anthropic_api_key_prefix=api_key[:7] + "..."
        if api_key
        else None,  # Reduced from 15 to 7 chars
        python_path_set=python_path is not None,
        all_env_vars=all_vars,
    )

    if api_key:
        await ctx.info(f"✅ ANTHROPIC_API_KEY found ({len(api_key)} chars)")
    else:
        await ctx.error("❌ ANTHROPIC_API_KEY not found!")

    if python_path:
        await ctx.info(f"✅ PYTHONPATH set: {python_path}")
    else:
        await ctx.warning("⚠️  PYTHONPATH not set")

    logger.info(
        "environment_check_complete",
        api_key_present=api_key is not None,
        pythonpath_set=python_path is not None,
        total_env_vars=len(all_vars),
    )

    return response
