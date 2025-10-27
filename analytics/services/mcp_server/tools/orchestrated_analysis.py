"""Orchestrated Analysis MCP Tool - Phase 3 & 5

This tool provides the main entry point for running orchestrated Four Lenses analyses
using the LangGraph coordinator. It handles:
1. Natural language or structured query parsing (Phase 3: rule-based, Phase 5: optional LLM)
2. Automatic lens selection and orchestration
3. Parallel lens execution with dependency management
4. Result aggregation and insight synthesis (Phase 3: simple, Phase 5: optional LLM narrative)
5. Query result caching for cost optimization (Phase 5)

Usage:
    - "Give me a customer health snapshot" -> Runs Lens 1
    - "Compare periods" -> Runs Lens 2
    - "Show me cohort evolution" -> Runs Lens 3
    - "Which cohorts perform best" -> Runs Lens 4
    - "Overall customer base health" -> Runs Lens 5
    - "customer health and cohorts" -> Runs Lenses 1, 3, 4, 5 in parallel
"""

from typing import Any

import structlog
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.main import mcp
from analytics.services.mcp_server.orchestration.coordinator import (
    FourLensesCoordinator,
)
from analytics.services.mcp_server.orchestration.query_cache import get_query_cache

logger = structlog.get_logger(__name__)


class OrchestratedAnalysisRequest(BaseModel):
    """Request for orchestrated Four Lenses analysis."""

    query: str = Field(
        max_length=1000,
        description="Natural language query describing desired analysis. "
        "Examples: 'customer health snapshot', 'compare cohorts', "
        "'overall base health', 'lens1 and lens5'",
    )

    use_llm: bool = Field(
        default=False,
        description="Use LLM-powered parsing and synthesis (Phase 5). "
        "Default: False (uses rule-based parsing from Phase 3). "
        "Requires ANTHROPIC_API_KEY environment variable.",
    )

    use_cache: bool = Field(
        default=True,
        description="Use query result caching to reduce costs. "
        "Default: True. Only applies when use_llm=True.",
    )

    include_visualizations: bool = Field(
        default=False,
        description="Generate PNG visualizations for charts (Phase 3). "
        "Default: False to minimize token usage. "
        "Note: Each chart adds ~20-67KB base64 data, causing conversation limits. "
        "Set to True only when visualizations are explicitly needed.",
    )


class OrchestratedAnalysisResponse(BaseModel):
    """Orchestrated analysis response with aggregated results."""

    query: str
    lenses_executed: list[str]
    lenses_failed: list[str]
    insights: list[str]
    recommendations: list[str]
    execution_time_ms: float

    # Optional: Individual lens results
    lens1_result: dict[str, Any] | None = None
    lens2_result: dict[str, Any] | None = None
    lens3_result: dict[str, Any] | None = None
    lens4_result: dict[str, Any] | None = None
    lens5_result: dict[str, Any] | None = None

    # Optional LLM-generated narrative (Phase 5, if use_llm=True)
    narrative: str | None = None

    # Formatted outputs (Phase 3) - PNG images + Plotly JSON
    formatted_outputs: dict[str, Any] | None = None  # Charts as base64 PNG + JSON

    # Foundation status
    data_mart_ready: bool
    rfm_ready: bool
    cohorts_ready: bool

    # Error information
    error: str | None = None
    lens_errors: dict[str, str] | None = None  # Maps lens name to error message

    # Cache information (Phase 5)
    cache_hit: bool = False
    cache_stats: dict[str, Any] | None = None  # Cache hit rate, size, etc.


@mcp.tool()
async def run_orchestrated_analysis(
    request: OrchestratedAnalysisRequest, ctx: Context
) -> OrchestratedAnalysisResponse:
    """
    Run orchestrated Four Lenses analysis from natural language query.

    This tool uses LangGraph to:
    1. Parse the user's intent (which lenses to run) - Phase 3: rule-based, Phase 5: optional LLM
    2. Check foundation data readiness (data mart, RFM, cohorts)
    3. Execute lenses in optimal order with parallel execution
    4. Synthesize results into coherent insights - Phase 3: simple, Phase 5: optional LLM narrative
    5. Cache results to reduce costs (Phase 5, when use_llm=True)

    Example queries:
    - "Give me a snapshot of customer base health"
    - "Show me overall customer base health"
    - "Compare cohorts"
    - "customer health and cohort analysis"
    - "lens1 and lens5"

    Prerequisites:
    - Data mart must be built (run build_customer_data_mart first)
    - For Lens 1/2: RFM metrics must be calculated (run calculate_rfm_metrics)
    - For Lens 3/4/5: Cohorts must be created (run create_customer_cohorts)
    - For LLM features (use_llm=True): ANTHROPIC_API_KEY environment variable must be set

    Args:
        request: Natural language analysis query with optional LLM and caching flags
        ctx: FastMCP context

    Returns:
        Synthesized insights from relevant lenses with execution metadata
    """
    logger.info(
        "orchestrated_analysis_received",
        query=request.query[:50],
        use_llm=request.use_llm,
        use_cache=request.use_cache,
    )

    await ctx.info(f"Running orchestrated analysis: {request.query}")

    cache = get_query_cache()
    cache_hit = False

    try:
        # Check cache if enabled
        if request.use_cache:
            cached_result = cache.get(request.query, request.use_llm)
            if cached_result:
                cache_hit = True
                await ctx.info(
                    f"Cache hit! Returning cached result (hit rate: {cache.get_hit_rate():.1%})"
                )
                result = cached_result
            else:
                # Cache miss - run analysis
                coordinator = FourLensesCoordinator(use_llm=request.use_llm)
                result = await coordinator.analyze(
                    request.query,
                    use_cache=False,
                    include_visualizations=request.include_visualizations,
                )  # Cache handled at tool level

                # Store in cache
                cache.set(request.query, request.use_llm, result)
        else:
            # Caching disabled - run analysis directly
            coordinator = FourLensesCoordinator(use_llm=request.use_llm)
            result = await coordinator.analyze(
                request.query,
                use_cache=False,
                include_visualizations=request.include_visualizations,
            )  # Cache disabled

        # Report progress
        lenses_executed = result.get("lenses_executed", [])
        lenses_failed = result.get("lenses_failed", [])

        if lenses_executed:
            await ctx.report_progress(
                0.8, f"Executed {len(lenses_executed)} lens(es) successfully"
            )

        if lenses_failed:
            await ctx.warning(
                f"Failed to execute {len(lenses_failed)} lens(es): {', '.join(lenses_failed)}"
            )

        # Create response
        response = OrchestratedAnalysisResponse(
            query=result.get("query", request.query),
            lenses_executed=lenses_executed,
            lenses_failed=lenses_failed,
            insights=result.get("insights", []),
            recommendations=result.get("recommendations", []),
            execution_time_ms=result.get("execution_time_ms", 0.0),
            narrative=result.get("narrative"),  # Only present if use_llm=True
            formatted_outputs=result.get(
                "formatted_outputs"
            ),  # Phase 3 formatted outputs
            lens1_result=result.get("lens1_result"),
            lens2_result=result.get("lens2_result"),
            lens3_result=result.get("lens3_result"),
            lens4_result=result.get("lens4_result"),
            lens5_result=result.get("lens5_result"),
            data_mart_ready=result.get("data_mart_ready", False),
            rfm_ready=result.get("rfm_ready", False),
            cohorts_ready=result.get("cohorts_ready", False),
            error=result.get("error"),
            lens_errors=result.get("lens_errors"),
            cache_hit=cache_hit,
            cache_stats=cache.get_stats() if request.use_cache else None,
        )

        if cache_hit:
            await ctx.info("Analysis complete (cached result)")
        else:
            await ctx.info(
                f"Analysis complete: {len(lenses_executed)} lens(es) executed "
                f"in {result.get('execution_time_ms', 0):.0f}ms"
            )

        return response

    except Exception as e:
        logger.error(
            "orchestrated_analysis_tool_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        await ctx.error(f"Orchestrated analysis failed: {str(e)}")

        # Return error response
        return OrchestratedAnalysisResponse(
            query=request.query,
            lenses_executed=[],
            lenses_failed=[],
            insights=[],
            recommendations=[],
            execution_time_ms=0.0,
            data_mart_ready=False,
            rfm_ready=False,
            cohorts_ready=False,
            error=str(e),
        )
