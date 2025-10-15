"""Cohorts MCP Tool - Phase 1 Foundation Service"""

from datetime import datetime
from typing import Literal

import structlog
from customer_base_audit.foundation.cohorts import (
    assign_cohorts,
    create_monthly_cohorts,
    create_quarterly_cohorts,
    create_yearly_cohorts,
)
from customer_base_audit.foundation.customer_contract import CustomerIdentifier
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.main import mcp

logger = structlog.get_logger(__name__)


class CreateCohortsRequest(BaseModel):
    """Request to create cohort definitions."""

    cohort_type: Literal["monthly", "quarterly", "yearly"] = Field(
        description="Type of cohorts to create"
    )
    start_date: datetime | None = Field(
        default=None, description="Start date (defaults to earliest customer)"
    )
    end_date: datetime | None = Field(
        default=None, description="End date (defaults to latest customer)"
    )


class CohortResponse(BaseModel):
    """Cohort creation response."""

    cohort_count: int
    customer_count: int
    date_range: tuple[str, str]
    cohort_type: str
    assignment_summary: dict[str, int]


async def _create_customer_cohorts_impl(
    request: CreateCohortsRequest, ctx: Context
) -> CohortResponse:
    """Implementation of cohort creation logic."""
    await ctx.info(f"Creating {request.cohort_type} cohorts")

    # Get data mart from context
    mart = ctx.get_state("data_mart")
    if mart is None:
        raise ValueError("Data mart not found. Run build_customer_data_mart first.")

    # Extract customer identifiers (need acquisition dates)
    # For now, use first transaction date as acquisition proxy
    # Use first granularity only to avoid O(n*g) iteration
    first_granularity = list(mart.periods.keys())[0]
    periods = mart.periods[first_granularity]

    # Group by customer and get minimum period_start (O(n) instead of O(n*g))
    from itertools import groupby

    sorted_periods = sorted(periods, key=lambda p: (p.customer_id, p.period_start))
    customer_first_dates = {
        customer_id: next(group).period_start
        for customer_id, group in groupby(sorted_periods, key=lambda p: p.customer_id)
    }

    customers = [
        CustomerIdentifier(
            customer_id=cid, acquisition_ts=acq_date, source_system="transactions"
        )
        for cid, acq_date in customer_first_dates.items()
    ]

    await ctx.report_progress(0.3, "Creating cohort definitions...")

    # Create cohorts
    if request.cohort_type == "monthly":
        cohort_defs = create_monthly_cohorts(
            customers, request.start_date, request.end_date
        )
    elif request.cohort_type == "quarterly":
        cohort_defs = create_quarterly_cohorts(
            customers, request.start_date, request.end_date
        )
    else:
        cohort_defs = create_yearly_cohorts(
            customers, request.start_date, request.end_date
        )

    await ctx.report_progress(0.7, "Assigning customers to cohorts...")

    # Assign customers
    cohort_assignments = assign_cohorts(customers, cohort_defs)

    # Store in context
    ctx.set_state("cohort_definitions", cohort_defs)
    ctx.set_state("cohort_assignments", cohort_assignments)

    # Validate cohorts were created
    if not cohort_defs:
        raise ValueError(
            "No cohorts created in specified date range. "
            "Check that customers exist within the time period."
        )

    # Calculate summary using Counter for O(n) performance
    from collections import Counter

    assignment_summary = dict(Counter(cohort_assignments.values()))

    date_range = (
        min(c.start_date for c in cohort_defs).isoformat(),
        max(c.end_date for c in cohort_defs).isoformat(),
    )

    await ctx.info(f"Created {len(cohort_defs)} cohorts")

    return CohortResponse(
        cohort_count=len(cohort_defs),
        customer_count=len(customers),
        date_range=date_range,
        cohort_type=request.cohort_type,
        assignment_summary=assignment_summary,
    )


@mcp.tool()
async def create_customer_cohorts(
    request: CreateCohortsRequest, ctx: Context
) -> CohortResponse:
    """
    Create cohort definitions and assign customers.

    This tool creates time-based cohorts (monthly, quarterly, yearly) and
    assigns customers based on acquisition dates. Required for Lens 3 and 4.

    Args:
        request: Configuration for cohort creation

    Returns:
        Summary of created cohorts and assignments
    """
    return await _create_customer_cohorts_impl(request, ctx)
