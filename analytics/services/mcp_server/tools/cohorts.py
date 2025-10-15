"""Cohorts MCP Tool - Phase 1 Foundation Service"""

from fastmcp import Context
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
from customer_base_audit.foundation.cohorts import (
    create_monthly_cohorts,
    create_quarterly_cohorts,
    create_yearly_cohorts,
    assign_cohorts,
)
from customer_base_audit.foundation.customer_contract import CustomerIdentifier
from analytics.services.mcp_server.main import mcp
import structlog

logger = structlog.get_logger(__name__)


class CreateCohortsRequest(BaseModel):
    """Request to create cohort definitions."""
    cohort_type: Literal["monthly", "quarterly", "yearly"] = Field(
        description="Type of cohorts to create"
    )
    start_date: datetime | None = Field(
        default=None,
        description="Start date (defaults to earliest customer)"
    )
    end_date: datetime | None = Field(
        default=None,
        description="End date (defaults to latest customer)"
    )


class CohortResponse(BaseModel):
    """Cohort creation response."""
    cohort_count: int
    customer_count: int
    date_range: tuple[str, str]
    cohort_type: str
    assignment_summary: dict[str, int]


async def _create_customer_cohorts_impl(
    request: CreateCohortsRequest,
    ctx: Context
) -> CohortResponse:
    """Implementation of cohort creation logic."""
    await ctx.info(f"Creating {request.cohort_type} cohorts")

    # Get data mart from context
    mart = ctx.get_state("data_mart")
    if mart is None:
        raise ValueError(
            "Data mart not found. Run build_customer_data_mart first."
        )

    # Extract customer identifiers (need acquisition dates)
    # For now, use first transaction date as acquisition proxy
    customer_first_dates = {}
    for granularity, periods in mart.periods.items():
        for period in periods:
            if period.customer_id not in customer_first_dates:
                customer_first_dates[period.customer_id] = period.period_start
            else:
                customer_first_dates[period.customer_id] = min(
                    customer_first_dates[period.customer_id],
                    period.period_start
                )

    customers = [
        CustomerIdentifier(customer_id=cid, acquisition_ts=acq_date, source_system="transactions")
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

    # Calculate summary
    assignment_summary = {}
    for cohort_id in set(cohort_assignments.values()):
        assignment_summary[cohort_id] = sum(
            1 for cid in cohort_assignments.values() if cid == cohort_id
        )

    date_range = (
        min(c.start_date for c in cohort_defs).isoformat(),
        max(c.end_date for c in cohort_defs).isoformat()
    )

    await ctx.info(f"Created {len(cohort_defs)} cohorts")

    return CohortResponse(
        cohort_count=len(cohort_defs),
        customer_count=len(customers),
        date_range=date_range,
        cohort_type=request.cohort_type,
        assignment_summary=assignment_summary
    )


@mcp.tool()
async def create_customer_cohorts(
    request: CreateCohortsRequest,
    ctx: Context
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
