"""Conversational Analysis MCP Tool - Phase 5

This tool provides conversational Four Lenses analysis with context maintenance.
It supports follow-up questions and maintains conversation history to resolve
references like "that", "last quarter", etc.

Usage:
    - Initial query: "Give me a customer health snapshot"
    - Follow-up: "Now compare that to last quarter"
    - Follow-up: "What about the January cohort specifically?"
    - Follow-up: "Show me retention rates"
"""

import os
from datetime import datetime
from typing import Any

import structlog
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.main import mcp
from analytics.services.mcp_server.orchestration.coordinator import (
    FourLensesCoordinator,
)

logger = structlog.get_logger(__name__)


class ConversationEntry(BaseModel):
    """Single entry in conversation history."""

    query: str
    timestamp: str
    lenses_executed: list[str]
    insights: list[str]
    recommendations: list[str]


class ConversationalAnalysisRequest(BaseModel):
    """Request for conversational Four Lenses analysis."""

    query: str = Field(
        max_length=1000,
        description="Natural language query or follow-up question",
    )

    conversation_history: list[dict[str, Any]] | None = Field(
        default=None,
        description="Previous queries and results for context (optional)",
    )

    use_llm: bool = Field(
        default=True,
        description="Use LLM-powered parsing and synthesis (recommended for conversational analysis)",
    )


class ConversationalAnalysisResponse(BaseModel):
    """Response with conversation context maintained."""

    # Current query results
    query: str
    lenses_executed: list[str]
    lenses_failed: list[str]
    insights: list[str]
    recommendations: list[str]
    execution_time_ms: float

    # Optional LLM-generated narrative (if use_llm=True)
    narrative: str | None = None

    # Conversation context
    conversation_history: list[dict[str, Any]]
    conversation_turn: int  # Which turn in the conversation (1, 2, 3, ...)

    # Foundation status
    data_mart_ready: bool
    rfm_ready: bool
    cohorts_ready: bool

    # Error information
    error: str | None = None
    lens_errors: dict[str, str] | None = None

    # Cost tracking (if use_llm=True)
    token_usage: dict[str, int] | None = None


@mcp.tool()
async def run_conversational_analysis(
    request: ConversationalAnalysisRequest, ctx: Context
) -> ConversationalAnalysisResponse:
    """
    Run conversational Four Lenses analysis with context maintenance.

    This tool maintains conversation history to support follow-up questions:
    - "Now compare that to last quarter"
    - "What about the January cohort specifically?"
    - "Show me retention rates"
    - "How does that compare to industry benchmarks?"

    The tool uses Claude to:
    1. Understand the context from previous queries
    2. Resolve references ("that", "last quarter", etc.)
    3. Generate coherent responses that build on previous analysis

    Prerequisites:
    - Data mart must be built (run build_customer_data_mart first)
    - For Lens 1/2: RFM metrics (run calculate_rfm_metrics)
    - For Lens 3/4/5: Cohorts (run create_customer_cohorts)
    - ANTHROPIC_API_KEY environment variable must be set

    Args:
        request: Query with optional conversation history
        ctx: FastMCP context

    Returns:
        Analysis results with updated conversation history

    Example conversation:
        Turn 1: "Give me a snapshot of customer base health"
        Turn 2: "Now show me cohort analysis"
        Turn 3: "Which cohort performs best?"
    """
    await ctx.info(f"Running conversational analysis: {request.query}")

    # Initialize conversation history
    conversation_history = request.conversation_history or []
    conversation_turn = len(conversation_history) + 1

    try:
        # Check for API key if LLM is requested
        if request.use_llm:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                error_msg = (
                    "ANTHROPIC_API_KEY environment variable required for LLM features. "
                    "Set use_llm=False to use rule-based parsing instead."
                )
                logger.error("anthropic_api_key_missing")
                await ctx.error(error_msg)

                return ConversationalAnalysisResponse(
                    query=request.query,
                    lenses_executed=[],
                    lenses_failed=[],
                    insights=[],
                    recommendations=[],
                    execution_time_ms=0.0,
                    conversation_history=conversation_history,
                    conversation_turn=conversation_turn,
                    data_mart_ready=False,
                    rfm_ready=False,
                    cohorts_ready=False,
                    error=error_msg,
                )

        # Create coordinator with LLM support
        coordinator = FourLensesCoordinator(use_llm=request.use_llm)

        # TODO (Phase 5.4 Enhancement): Use Claude to resolve follow-up references
        # For MVP, we pass the query as-is without reference resolution
        # Future enhancement: Analyze conversation_history to resolve "that", "last quarter", etc.

        # Run analysis
        result = await coordinator.analyze(request.query)

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

        # Add current query to conversation history
        current_entry = {
            "query": request.query,
            "timestamp": datetime.now().isoformat(),
            "lenses_executed": lenses_executed,
            "insights": result.get("insights", []),
            "recommendations": result.get("recommendations", []),
        }
        conversation_history.append(current_entry)

        # Get token usage if LLM was used
        token_usage = None
        if request.use_llm and coordinator.query_interpreter:
            interpreter_tokens = coordinator.query_interpreter.get_token_usage()
            synthesizer_tokens = coordinator.result_synthesizer.get_token_usage()

            token_usage = {
                "query_parsing_input": interpreter_tokens["input_tokens"],
                "query_parsing_output": interpreter_tokens["output_tokens"],
                "synthesis_input": synthesizer_tokens["input_tokens"],
                "synthesis_output": synthesizer_tokens["output_tokens"],
                "total_tokens": (
                    interpreter_tokens["total_tokens"]
                    + synthesizer_tokens["total_tokens"]
                ),
            }

            logger.info(
                "token_usage",
                conversation_turn=conversation_turn,
                total_tokens=token_usage["total_tokens"],
            )

        # Create response
        response = ConversationalAnalysisResponse(
            query=result.get("query", request.query),
            lenses_executed=lenses_executed,
            lenses_failed=lenses_failed,
            insights=result.get("insights", []),
            recommendations=result.get("recommendations", []),
            execution_time_ms=result.get("execution_time_ms", 0.0),
            narrative=result.get("narrative"),  # Only present if LLM was used
            conversation_history=conversation_history,
            conversation_turn=conversation_turn,
            data_mart_ready=result.get("data_mart_ready", False),
            rfm_ready=result.get("rfm_ready", False),
            cohorts_ready=result.get("cohorts_ready", False),
            error=result.get("error"),
            lens_errors=result.get("lens_errors"),
            token_usage=token_usage,
        )

        await ctx.info(
            f"Conversational analysis complete (turn {conversation_turn}): "
            f"{len(lenses_executed)} lens(es) executed in {result.get('execution_time_ms', 0):.0f}ms"
        )

        return response

    except Exception as e:
        logger.error(
            "conversational_analysis_failed",
            error=str(e),
            error_type=type(e).__name__,
            conversation_turn=conversation_turn,
        )
        await ctx.error(f"Conversational analysis failed: {str(e)}")

        # Return error response
        return ConversationalAnalysisResponse(
            query=request.query,
            lenses_executed=[],
            lenses_failed=[],
            insights=[],
            recommendations=[],
            execution_time_ms=0.0,
            conversation_history=conversation_history,
            conversation_turn=conversation_turn,
            data_mart_ready=False,
            rfm_ready=False,
            cohorts_ready=False,
            error=str(e),
        )
