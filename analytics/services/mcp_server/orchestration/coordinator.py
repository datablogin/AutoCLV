"""LangGraph Coordinator for Four Lenses Analytics - Phase 3

This module implements the orchestration layer that:
1. Parses user intent (which lenses to run)
2. Ensures foundation data is prepared (data mart, RFM, cohorts)
3. Executes lenses in parallel where dependencies allow
4. Aggregates results into coherent insights

Design:
- Rule-based intent parsing (no LLM for MVP)
- StateGraph for workflow management
- Parallel execution using asyncio.gather
- Graceful error handling with partial results
"""

import asyncio
import time
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph
from opentelemetry import trace
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Lazy import to avoid circular dependency
_metrics_collector = None


def get_metrics_collector_instance():
    """Get metrics collector with lazy import to avoid circular dependencies."""
    global _metrics_collector
    if _metrics_collector is None:
        from analytics.services.mcp_server.tools.execution_metrics import (
            get_metrics_collector,
        )

        _metrics_collector = get_metrics_collector()
    return _metrics_collector


class AnalysisState(TypedDict, total=False):
    """State for Four Lenses analysis workflow.

    TypedDict with total=False allows optional fields, which is necessary
    for progressive state building through the workflow.

    Note: Using Annotated with operator.add for list fields would be ideal,
    but keeping simple for MVP. Lists are replaced rather than accumulated.
    """

    # Input
    query: str  # Natural language or structured query
    intent: dict[str, Any]  # Parsed intent (which lenses, parameters)

    # Foundation data readiness
    data_mart_ready: bool
    rfm_ready: bool
    cohorts_ready: bool

    # Lens results (dict with lens metrics and insights)
    lens1_result: dict[str, Any] | None
    lens2_result: dict[str, Any] | None
    lens3_result: dict[str, Any] | None
    lens4_result: dict[str, Any] | None
    lens5_result: dict[str, Any] | None

    # Aggregated output
    insights: list[str]
    recommendations: list[str]

    # Metadata
    lenses_executed: list[str]
    lenses_failed: list[str]
    lens_errors: dict[str, str]  # Maps lens name to error message
    execution_time_ms: float
    error: str | None


class FourLensesCoordinator:
    """Orchestrates Four Lenses analysis workflow using LangGraph."""

    def __init__(self):
        """Initialize coordinator with compiled StateGraph."""
        self.graph = self._build_graph()
        self.shared_state = get_shared_state()

    def _build_graph(self) -> Any:
        """Build the LangGraph state graph.

        Workflow:
        1. parse_intent: Determine which lenses to run
        2. prepare_foundation: Ensure data mart, RFM, cohorts are ready
        3. execute_lenses: Run requested lenses in parallel where possible
        4. synthesize_results: Aggregate insights and recommendations
        """
        workflow = StateGraph(AnalysisState)

        # Add nodes
        workflow.add_node("parse_intent", self._parse_intent)
        workflow.add_node("prepare_foundation", self._prepare_foundation)
        workflow.add_node("execute_lenses", self._execute_lenses)
        workflow.add_node("synthesize_results", self._synthesize_results)

        # Define edges
        workflow.set_entry_point("parse_intent")
        workflow.add_edge("parse_intent", "prepare_foundation")
        workflow.add_edge("prepare_foundation", "execute_lenses")
        workflow.add_edge("execute_lenses", "synthesize_results")
        workflow.add_edge("synthesize_results", END)

        return workflow.compile()

    async def _parse_intent(self, state: AnalysisState) -> AnalysisState:
        """Parse user query into structured intent.

        Phase 3 MVP: Rule-based keyword matching
        Phase 5: LLM-based parsing with Claude

        Args:
            state: Current workflow state

        Returns:
            Updated state with parsed intent
        """
        query = state["query"].lower()
        logger.info("parsing_intent", query=state["query"])

        intent: dict[str, Any] = {
            "lenses": [],
            "date_range": None,
            "cohort_filter": None,
            "parameters": {},
        }

        # Check Lens 5 first (most specific "overall health" patterns)
        lens5_keywords = [
            "overall health",
            "customer base health",
            "base health",
            "overall customer",
            "lens5",
            "lens 5",
        ]
        if any(kw in query for kw in lens5_keywords):
            intent["lenses"].append("lens5")

        # Lens 1: Snapshot/health keywords (but not "overall health" which is Lens 5)
        lens1_keywords = ["snapshot", "current", "lens1", "lens 1"]
        # Only add "health" keyword if not already matched by Lens 5
        if "lens5" not in intent["lenses"]:
            lens1_keywords.append("health")

        if any(kw in query for kw in lens1_keywords):
            intent["lenses"].append("lens1")

        # Lens 2: Comparison/trend keywords
        if any(
            kw in query
            for kw in ["compare", "comparison", "trend", "change", "lens2", "lens 2"]
        ):
            intent["lenses"].append("lens2")

        # Lens 3: Single cohort evolution
        if any(
            kw in query for kw in ["cohort evolution", "lifecycle", "lens3", "lens 3"]
        ):
            intent["lenses"].append("lens3")

        # Lens 4: Multi-cohort comparison
        if any(
            kw in query
            for kw in [
                "cohort comparison",
                "cohort analysis",
                "cohorts",
                "which cohort",
                "lens4",
                "lens 4",
            ]
        ):
            intent["lenses"].append("lens4")

        # If no specific lens detected, run Lens 1 as default
        if not intent["lenses"]:
            logger.info("no_lens_detected_defaulting_to_lens1", query=state["query"])
            intent["lenses"].append("lens1")

        # Remove duplicates and sort
        intent["lenses"] = sorted(set(intent["lenses"]))

        logger.info(
            "intent_parsed",
            lenses=intent["lenses"],
            lens_count=len(intent["lenses"]),
        )

        state["intent"] = intent
        return state

    async def _prepare_foundation(self, state: AnalysisState) -> AnalysisState:
        """Ensure foundation data is prepared.

        Automatically builds foundation data from transactions if available:
        1. Build data mart from transactions (if not already built)
        2. Calculate RFM metrics from data mart (if not already calculated)
        3. Create cohorts from data mart (if not already created)

        Args:
            state: Current workflow state

        Returns:
            Updated state with foundation readiness flags
        """
        logger.info("preparing_foundation_data")

        # Check what's available in shared state
        has_transactions = self.shared_state.has("transactions")
        has_data_mart = self.shared_state.has("data_mart")
        has_rfm = self.shared_state.has("rfm_metrics")
        has_cohorts = self.shared_state.has("cohort_definitions")

        # Auto-build foundation data if transactions available
        if has_transactions and not has_data_mart:
            logger.info("auto_building_data_mart")
            try:
                transactions = self.shared_state.get("transactions")
                from customer_base_audit.foundation.data_mart import (
                    CustomerDataMartBuilder,
                    PeriodGranularity,
                )

                builder = CustomerDataMartBuilder(
                    period_granularities=(
                        PeriodGranularity.QUARTER,
                        PeriodGranularity.YEAR,
                    )
                )
                data_mart = builder.build(transactions)
                self.shared_state.set("data_mart", data_mart)

                # Store period aggregations separately for Lens 5
                first_granularity = list(data_mart.periods.keys())[0]
                period_aggregations = data_mart.periods[first_granularity]
                self.shared_state.set("period_aggregations", period_aggregations)

                has_data_mart = True
                logger.info(
                    "data_mart_built",
                    order_count=len(data_mart.orders),
                    period_count=len(period_aggregations),
                )
            except Exception as e:
                logger.error(
                    "data_mart_build_failed", error=str(e), error_type=type(e).__name__
                )

        # Auto-calculate RFM if data mart available
        if has_data_mart and not has_rfm:
            logger.info("auto_calculating_rfm")
            try:
                data_mart = self.shared_state.get("data_mart")

                from customer_base_audit.foundation.rfm import (
                    calculate_rfm,
                    calculate_rfm_scores,
                )

                # Use latest transaction date as observation end
                first_granularity = list(data_mart.periods.keys())[0]
                period_aggregations = data_mart.periods[first_granularity]
                max_date = max(p.period_end for p in period_aggregations)

                rfm_metrics = calculate_rfm(
                    period_aggregations=period_aggregations,
                    observation_end=max_date,
                    parallel=True,
                )
                rfm_scores = calculate_rfm_scores(rfm_metrics)

                self.shared_state.set("rfm_metrics", rfm_metrics)
                self.shared_state.set("rfm_scores", rfm_scores)
                has_rfm = True
                logger.info("rfm_calculated", customer_count=len(rfm_metrics))
            except Exception as e:
                logger.error(
                    "rfm_calculation_failed", error=str(e), error_type=type(e).__name__
                )

        # Auto-create cohorts if data mart available
        if has_data_mart and not has_cohorts:
            logger.info("auto_creating_cohorts")
            try:
                data_mart = self.shared_state.get("data_mart")
                from itertools import groupby

                from customer_base_audit.foundation.cohorts import (
                    assign_cohorts,
                    create_quarterly_cohorts,
                )
                from customer_base_audit.foundation.customer_contract import (
                    CustomerIdentifier,
                )

                # Extract customer acquisition dates from data mart
                first_granularity = list(data_mart.periods.keys())[0]
                periods = data_mart.periods[first_granularity]

                sorted_periods = sorted(
                    periods, key=lambda p: (p.customer_id, p.period_start)
                )
                customer_first_dates = {
                    customer_id: next(group).period_start
                    for customer_id, group in groupby(
                        sorted_periods, key=lambda p: p.customer_id
                    )
                }

                customers = [
                    CustomerIdentifier(
                        customer_id=cid,
                        acquisition_ts=acq_date,
                        source_system="transactions",
                    )
                    for cid, acq_date in customer_first_dates.items()
                ]

                cohort_defs = create_quarterly_cohorts(customers, None, None)
                cohort_assignments = assign_cohorts(customers, cohort_defs)

                self.shared_state.set("cohort_definitions", cohort_defs)
                self.shared_state.set("cohort_assignments", cohort_assignments)
                has_cohorts = True
                logger.info("cohorts_created", cohort_count=len(cohort_defs))
            except Exception as e:
                logger.error(
                    "cohort_creation_failed", error=str(e), error_type=type(e).__name__
                )

        state["data_mart_ready"] = has_data_mart
        state["rfm_ready"] = has_rfm
        state["cohorts_ready"] = has_cohorts

        logger.info(
            "foundation_status",
            transactions=has_transactions,
            data_mart=has_data_mart,
            rfm=has_rfm,
            cohorts=has_cohorts,
        )

        # Log warnings if required data is still missing
        lenses = state["intent"]["lenses"]
        if ("lens1" in lenses or "lens2" in lenses) and not has_rfm:
            logger.warning(
                "missing_rfm_data",
                message="Lens 1/2 requires RFM metrics. Load transactions first with load_transactions tool.",
            )

        if (
            "lens3" in lenses or "lens4" in lenses or "lens5" in lenses
        ) and not has_cohorts:
            logger.warning(
                "missing_cohort_data",
                message="Lens 3/4/5 requires cohorts. Load transactions first with load_transactions tool.",
            )

        return state

    async def _execute_lenses(self, state: AnalysisState) -> AnalysisState:
        """Execute requested lenses in parallel where dependencies allow.

        Dependency analysis:
        - Lens 1: Requires RFM (independent)
        - Lens 2: Requires RFM + Lens 1 result (dependent on Lens 1)
        - Lens 3: Requires cohorts (independent)
        - Lens 4: Requires cohorts (independent)
        - Lens 5: Requires cohorts (independent)

        Parallel execution strategy:
        - Group 1: Lens 1, 3, 4, 5 (all independent, run in parallel)
        - Group 2: Lens 2 (depends on Lens 1, run after Group 1 if Lens 1 succeeds)

        Args:
            state: Current workflow state

        Returns:
            Updated state with lens results
        """
        lenses_to_run = state["intent"]["lenses"]
        logger.info("executing_lenses", lenses=lenses_to_run)

        start_time = time.time()
        lenses_executed: list[str] = []
        lenses_failed: list[str] = []
        lens_errors: dict[str, str] = {}  # Track error messages for failed lenses

        # Track analysis start
        get_metrics_collector_instance().record_analysis_start()

        # Prepare parallel execution groups
        group1_tasks: dict[str, Any] = {}

        # Group 1: Independent lenses (Lens 1, 3, 4, 5)
        if "lens1" in lenses_to_run:
            group1_tasks["lens1"] = self._execute_lens1(state)

        if "lens3" in lenses_to_run:
            group1_tasks["lens3"] = self._execute_lens3(state)

        if "lens4" in lenses_to_run:
            group1_tasks["lens4"] = self._execute_lens4(state)

        if "lens5" in lenses_to_run:
            group1_tasks["lens5"] = self._execute_lens5(state)

        # Execute Group 1 in parallel
        if group1_tasks:
            logger.info("executing_parallel_group1", lenses=list(group1_tasks.keys()))

            # Track start time for each lens
            lens_start_times = {lens: time.time() for lens in group1_tasks.keys()}

            results = await asyncio.gather(
                *group1_tasks.values(), return_exceptions=True
            )

            # Process results and record metrics
            for lens_name, result in zip(group1_tasks.keys(), results):
                duration_ms = (time.time() - lens_start_times[lens_name]) * 1000

                if isinstance(result, Exception):
                    error_msg = f"{type(result).__name__}: {str(result)}"
                    error_type = type(result).__name__

                    logger.error(
                        "lens_execution_failed",
                        lens=lens_name,
                        error=str(result),
                        error_type=error_type,
                        duration_ms=duration_ms,
                    )

                    # Record failure metrics
                    get_metrics_collector_instance().record_lens_execution(
                        lens_name=lens_name,
                        success=False,
                        duration_ms=duration_ms,
                        error_type=error_type,
                    )

                    lenses_failed.append(lens_name)
                    lens_errors[lens_name] = error_msg
                    state[f"{lens_name}_result"] = None
                else:
                    logger.info(
                        "lens_execution_succeeded",
                        lens=lens_name,
                        duration_ms=duration_ms,
                    )

                    # Record success metrics
                    get_metrics_collector_instance().record_lens_execution(
                        lens_name=lens_name,
                        success=True,
                        duration_ms=duration_ms,
                    )

                    lenses_executed.append(lens_name)
                    state[f"{lens_name}_result"] = result

        # Group 2: Lens 2 (depends on Lens 1)
        if "lens2" in lenses_to_run:
            if state.get("lens1_result") is not None:
                logger.info("executing_lens2_with_lens1_context")
                lens2_start_time = time.time()
                try:
                    result = await self._execute_lens2(state)
                    lens2_duration_ms = (time.time() - lens2_start_time) * 1000

                    lenses_executed.append("lens2")
                    state["lens2_result"] = result

                    logger.info(
                        "lens_execution_succeeded",
                        lens="lens2",
                        duration_ms=lens2_duration_ms,
                    )

                    # Record success metrics
                    get_metrics_collector_instance().record_lens_execution(
                        lens_name="lens2",
                        success=True,
                        duration_ms=lens2_duration_ms,
                    )

                except Exception as e:
                    lens2_duration_ms = (time.time() - lens2_start_time) * 1000
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    error_type = type(e).__name__

                    logger.error(
                        "lens_execution_failed",
                        lens="lens2",
                        error=str(e),
                        error_type=error_type,
                        duration_ms=lens2_duration_ms,
                    )

                    # Record failure metrics
                    get_metrics_collector_instance().record_lens_execution(
                        lens_name="lens2",
                        success=False,
                        duration_ms=lens2_duration_ms,
                        error_type=error_type,
                    )

                    lenses_failed.append("lens2")
                    lens_errors["lens2"] = error_msg
                    state["lens2_result"] = None
            else:
                error_msg = "Lens 1 result not available (required for Lens 2)"
                logger.warning(
                    "lens2_skipped",
                    reason=error_msg,
                )
                lenses_failed.append("lens2")
                lens_errors["lens2"] = error_msg
                state["lens2_result"] = None

        execution_time_ms = (time.time() - start_time) * 1000

        # Record overall analysis duration
        get_metrics_collector_instance().record_analysis_duration(execution_time_ms)

        state["lenses_executed"] = lenses_executed
        state["lenses_failed"] = lenses_failed
        state["lens_errors"] = lens_errors
        state["execution_time_ms"] = execution_time_ms

        logger.info(
            "lenses_execution_complete",
            executed=lenses_executed,
            failed=lenses_failed,
            execution_time_ms=execution_time_ms,
        )

        return state

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
        reraise=True,
    )
    async def _execute_lens_with_retry(
        self, lens_name: str, lens_func, state: AnalysisState
    ) -> dict[str, Any]:
        """Execute a lens with automatic retry on transient failures.

        Retries up to 3 times with exponential backoff for transient errors:
        - TimeoutError: Network or computation timeouts
        - ConnectionError: Database or service connection issues
        - RuntimeError: Temporary computation failures

        Args:
            lens_name: Name of the lens being executed (for logging)
            lens_func: Async function to execute
            state: Current workflow state

        Returns:
            Lens result dict

        Raises:
            Exception: If all retry attempts fail
        """
        logger.info("executing_lens_with_retry", lens=lens_name, attempt=1)
        try:
            result = await lens_func(state)
            logger.info("lens_execution_succeeded_with_retry", lens=lens_name)
            return result
        except Exception as e:
            logger.error(
                "lens_execution_failed_will_retry",
                lens=lens_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def _execute_lens1(self, state: AnalysisState) -> dict[str, Any]:
        """Execute Lens 1: Single period snapshot.

        Retrieves RFM metrics from shared state and calls Lens 1 analysis.

        Returns:
            Lens 1 result dict (from Lens1Response model)
        """
        with tracer.start_as_current_span("lens1_execution") as span:
            from customer_base_audit.analyses.lens1 import analyze_single_period

            from analytics.services.mcp_server.tools.lens1 import (
                _assess_concentration_risk,
                _calculate_customer_health_score,
                _generate_lens1_recommendations,
            )

            rfm_metrics = self.shared_state.get("rfm_metrics")
            rfm_scores = self.shared_state.get("rfm_scores")

            if rfm_metrics is None:
                raise ValueError("RFM metrics not found in shared state")

            span.set_attribute("rfm_metrics_count", len(rfm_metrics))

            # Run Lens 1 analysis
            with tracer.start_as_current_span("lens1_calculate"):
                lens1_result = analyze_single_period(
                    rfm_metrics=rfm_metrics,
                    rfm_scores=rfm_scores if rfm_scores else None,
                )

            # Calculate insights
            with tracer.start_as_current_span("lens1_generate_insights"):
                health_score = _calculate_customer_health_score(lens1_result)
                concentration_risk = _assess_concentration_risk(lens1_result)
                recommendations = _generate_lens1_recommendations(lens1_result)

            span.set_attribute("total_customers", lens1_result.total_customers)
            span.set_attribute("health_score", health_score)
            span.set_attribute("concentration_risk", concentration_risk)

        # Convert to dict for state storage
        return {
            "period_name": "Current Period",
            "total_customers": lens1_result.total_customers,
            "one_time_buyers": lens1_result.one_time_buyers,
            "one_time_buyer_pct": float(lens1_result.one_time_buyer_pct),
            "total_revenue": float(lens1_result.total_revenue),
            "top_10pct_revenue_contribution": float(
                lens1_result.top_10pct_revenue_contribution
            ),
            "top_20pct_revenue_contribution": float(
                lens1_result.top_20pct_revenue_contribution
            ),
            "avg_orders_per_customer": float(lens1_result.avg_orders_per_customer),
            "median_customer_value": float(lens1_result.median_customer_value),
            "rfm_distribution": lens1_result.rfm_distribution,
            "customer_health_score": health_score,
            "concentration_risk": concentration_risk,
            "recommendations": recommendations,
        }

    async def _execute_lens2(self, state: AnalysisState) -> dict[str, Any]:
        """Execute Lens 2: Period-to-period comparison.

        Placeholder for MVP - full implementation requires two separate RFM calculations.

        Returns:
            Lens 2 result dict
        """
        # Placeholder - Phase 3 MVP will skip full Lens 2 implementation
        # Full implementation requires two separate period calculations
        logger.warning(
            "lens2_placeholder",
            message="Lens 2 full implementation pending - returning placeholder",
        )

        return {
            "period1_name": "Period 1",
            "period2_name": "Period 2",
            "retention_rate": 0.75,
            "churn_rate": 0.25,
            "growth_momentum": "moderate",
            "key_drivers": ["Retention holding steady", "Acquisition slowing"],
            "recommendations": [
                "Lens 2 full implementation pending in Phase 3 follow-up"
            ],
        }

    async def _execute_lens3(self, state: AnalysisState) -> dict[str, Any]:
        """Execute Lens 3: Single cohort evolution.

        Placeholder for MVP - full implementation requires cohort-specific analysis.

        Returns:
            Lens 3 result dict
        """
        logger.warning(
            "lens3_placeholder",
            message="Lens 3 full implementation pending - returning placeholder",
        )

        return {
            "cohort_id": "2024-Q1",
            "cohort_size": 100,
            "periods_analyzed": 4,
            "cohort_maturity": "growth",
            "ltv_trajectory": "strong",
            "recommendations": [
                "Lens 3 full implementation pending in Phase 3 follow-up"
            ],
        }

    async def _execute_lens4(self, state: AnalysisState) -> dict[str, Any]:
        """Execute Lens 4: Multi-cohort comparison.

        Placeholder for MVP - full implementation requires multi-cohort analysis.

        Returns:
            Lens 4 result dict
        """
        logger.warning(
            "lens4_placeholder",
            message="Lens 4 full implementation pending - returning placeholder",
        )

        return {
            "cohort_count": 4,
            "alignment_type": "left-aligned",
            "best_performing_cohort": "2024-Q1",
            "worst_performing_cohort": "2023-Q4",
            "key_differences": ["Q1 shows 20% higher retention"],
            "recommendations": [
                "Lens 4 full implementation pending in Phase 3 follow-up"
            ],
        }

    async def _execute_lens5(self, state: AnalysisState) -> dict[str, Any]:
        """Execute Lens 5: Overall customer base health.

        Retrieves period aggregations and cohort data, then calls Lens 5 analysis.

        Returns:
            Lens 5 result dict
        """
        with tracer.start_as_current_span("lens5_execution") as span:
            logger.info("executing_lens5")

            try:
                from customer_base_audit.analyses.lens5 import (
                    assess_customer_base_health,
                )

                from analytics.services.mcp_server.tools.lens5 import (
                    _generate_recommendations,
                    _identify_key_risks,
                    _identify_key_strengths,
                )

                # Get required data from shared state
                period_aggregations = self.shared_state.get("period_aggregations")
                cohort_assignments = self.shared_state.get("cohort_assignments")

                logger.info(
                    "lens5_data_check",
                    has_period_aggs=period_aggregations is not None,
                    has_cohort_assignments=cohort_assignments is not None,
                    period_count=len(period_aggregations) if period_aggregations else 0,
                )

                if period_aggregations is None:
                    raise ValueError("Period aggregations not found in shared state")
                if cohort_assignments is None:
                    raise ValueError("Cohort assignments not found in shared state")

                span.set_attribute("period_count", len(period_aggregations))
                span.set_attribute("cohort_count", len(set(cohort_assignments.values())))

                # Determine analysis window from period aggregations
                all_dates = [p.period_start for p in period_aggregations]
                analysis_start_date = min(all_dates)
                analysis_end_date = max(all_dates)

                logger.info(
                    "lens5_analysis_window",
                    start=analysis_start_date.isoformat(),
                    end=analysis_end_date.isoformat(),
                )

                # Run Lens 5 analysis with correct parameters
                with tracer.start_as_current_span("lens5_calculate"):
                    lens5_result = assess_customer_base_health(
                        period_aggregations=period_aggregations,
                        cohort_assignments=cohort_assignments,
                        analysis_start_date=analysis_start_date,
                        analysis_end_date=analysis_end_date,
                    )

                # Access health_score data correctly
                health_score = lens5_result.health_score

                logger.info(
                    "lens5_analysis_complete",
                    total_customers=health_score.total_customers,
                    health_score=float(health_score.health_score),
                )

                span.set_attribute("health_score", float(health_score.health_score))
                span.set_attribute("health_grade", health_score.health_grade)
                span.set_attribute("total_customers", health_score.total_customers)

                # Generate insights from strengths and risks
                with tracer.start_as_current_span("lens5_generate_insights"):
                    strengths = _identify_key_strengths(lens5_result)
                    risks = _identify_key_risks(lens5_result)
                    insights = (
                        strengths + risks
                    )  # Combine strengths and risks into insights

                    # Generate recommendations
                    recommendations = _generate_recommendations(lens5_result)

                # Convert to dict for state storage (matching the Lens5Metrics structure)
                result = {
                    "analysis_name": "Customer Base Health Assessment",
                    "date_range": (
                        analysis_start_date.isoformat(),
                        analysis_end_date.isoformat(),
                    ),
                    "health_score": float(health_score.health_score),
                    "health_grade": health_score.health_grade,
                    "total_customers": health_score.total_customers,
                    "total_active_customers": health_score.total_active_customers,
                    "overall_retention_rate": float(
                        health_score.overall_retention_rate
                    ),
                    "cohort_quality_trend": health_score.cohort_quality_trend,
                    "revenue_predictability_pct": float(
                        health_score.revenue_predictability_pct
                    ),
                    "acquisition_dependence_pct": float(
                        health_score.acquisition_dependence_pct
                    ),
                    "insights": insights,
                    "recommendations": recommendations,
                }

                logger.info(
                    "lens5_result_formatted",
                    health_score=result["health_score"],
                    grade=result["health_grade"],
                )

                return result

            except Exception as e:
                logger.error(
                    "lens5_execution_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                raise

    async def _synthesize_results(self, state: AnalysisState) -> AnalysisState:
        """Synthesize results from all executed lenses.

        Phase 3 MVP: Simple aggregation of insights and recommendations
        Phase 5: LLM-based narrative generation with Claude

        Args:
            state: Current workflow state

        Returns:
            Updated state with aggregated insights and recommendations
        """
        logger.info("synthesizing_results")

        all_insights: list[str] = []
        all_recommendations: list[str] = []

        # Aggregate insights from each executed lens
        if state.get("lens1_result"):
            lens1 = state["lens1_result"]
            all_insights.append(
                f"Customer Base Health: {lens1['customer_health_score']:.1f}/100 "
                f"(Concentration Risk: {lens1['concentration_risk']})"
            )
            all_recommendations.extend(lens1.get("recommendations", []))

        if state.get("lens2_result"):
            lens2 = state["lens2_result"]
            all_insights.append(
                f"Growth Momentum: {lens2.get('growth_momentum', 'unknown')}"
            )
            all_recommendations.extend(lens2.get("recommendations", []))

        if state.get("lens3_result"):
            lens3 = state["lens3_result"]
            all_insights.append(
                f"Cohort Maturity: {lens3.get('cohort_maturity', 'unknown')}"
            )
            all_recommendations.extend(lens3.get("recommendations", []))

        if state.get("lens4_result"):
            lens4 = state["lens4_result"]
            all_insights.append(
                f"Best Cohort: {lens4.get('best_performing_cohort', 'N/A')}"
            )
            all_recommendations.extend(lens4.get("recommendations", []))

        if state.get("lens5_result"):
            lens5 = state["lens5_result"]
            all_insights.append(
                f"Overall Health: {lens5['health_score']:.1f}/100 "
                f"(Grade: {lens5['health_grade']})"
            )
            all_insights.extend(lens5.get("insights", []))
            all_recommendations.extend(lens5.get("recommendations", []))

        # Add execution summary
        lenses_executed = state.get("lenses_executed", [])
        lenses_failed = state.get("lenses_failed", [])
        execution_time = state.get("execution_time_ms", 0)

        summary = f"Executed {len(lenses_executed)} lens(es) in {execution_time:.0f}ms"
        if lenses_failed:
            summary += f" ({len(lenses_failed)} failed: {', '.join(lenses_failed)})"

        all_insights.insert(0, summary)

        state["insights"] = all_insights
        state["recommendations"] = all_recommendations

        logger.info(
            "synthesis_complete",
            insight_count=len(all_insights),
            recommendation_count=len(all_recommendations),
        )

        return state

    async def analyze(self, query: str) -> dict[str, Any]:
        """Run complete Four Lenses analysis from query.

        This is the main entry point for orchestrated analysis.

        Args:
            query: Natural language or structured query

        Returns:
            Complete analysis results including:
            - query: Original query
            - intent: Parsed intent
            - lenses_executed: List of successfully executed lenses
            - lenses_failed: List of failed lenses
            - insights: Aggregated insights
            - recommendations: Aggregated recommendations
            - execution_time_ms: Total execution time
            - lens results: Individual lens results (lens1_result, etc.)
        """
        with tracer.start_as_current_span("orchestrated_analysis") as span:
            span.set_attribute("query", query[:200])  # Truncate for trace storage
            logger.info("orchestrated_analysis_starting", query=query)

            initial_state: AnalysisState = {
                "query": query,
                "intent": {},
                "data_mart_ready": False,
                "rfm_ready": False,
                "cohorts_ready": False,
                "lens1_result": None,
                "lens2_result": None,
                "lens3_result": None,
                "lens4_result": None,
                "lens5_result": None,
                "insights": [],
                "recommendations": [],
                "lenses_executed": [],
                "lenses_failed": [],
                "lens_errors": {},
                "execution_time_ms": 0.0,
                "error": None,
            }

            try:
                result = await self.graph.ainvoke(initial_state)

                # Add span attributes for successful execution
                span.set_attribute(
                    "lenses_executed", ",".join(result.get("lenses_executed", []))
                )
                span.set_attribute(
                    "lenses_failed", ",".join(result.get("lenses_failed", []))
                )
                span.set_attribute(
                    "execution_time_ms", result.get("execution_time_ms", 0)
                )
                span.set_attribute("success", True)

                logger.info("orchestrated_analysis_complete")
                return dict(result)
            except Exception as e:
                # Record exception in span
                span.record_exception(e)
                span.set_attribute("success", False)
                span.set_attribute("error_type", type(e).__name__)

                logger.error(
                    "orchestrated_analysis_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                initial_state["error"] = str(e)
                return dict(initial_state)
