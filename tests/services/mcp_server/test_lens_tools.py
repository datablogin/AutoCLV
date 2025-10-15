"""
Integration tests for Lens MCP Tools (Phase 2)

Tests the four lens services and their integration with foundation tools.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Import the MCP tool implementations (not the wrapped versions)
from analytics.services.mcp_server.tools.lens1 import (
    _analyze_single_period_snapshot_impl,
    Lens1Request,
)
from analytics.services.mcp_server.tools.lens2 import (
    _analyze_period_comparison_impl,
    Lens2Request,
)
from analytics.services.mcp_server.tools.lens3 import (
    _analyze_cohort_lifecycle_impl,
    Lens3Request,
)
from analytics.services.mcp_server.tools.lens4 import (
    _compare_multiple_cohorts_impl,
    Lens4Request,
)

# Import foundation data structures
from customer_base_audit.foundation.rfm import RFMMetrics
from customer_base_audit.foundation.data_mart import (
    PeriodAggregation,
    PeriodGranularity,
)
from customer_base_audit.foundation.cohorts import (
    CohortDefinition,
    CustomerIdentifier,
)

# Import SharedState for testing
from analytics.services.mcp_server.state import get_shared_state


class MockContext:
    """Mock FastMCP Context for testing."""

    def __init__(self):
        self.state = {}
        self.messages = []
        self.progress_reports = []

    def get_state(self, key: str):
        """Get state value."""
        return self.state.get(key)

    def set_state(self, key: str, value):
        """Set state value."""
        self.state[key] = value

    async def info(self, message: str):
        """Log info message."""
        self.messages.append(("info", message))

    async def report_progress(self, progress: float, message: str):
        """Report progress."""
        self.progress_reports.append((progress, message))


@pytest.fixture(autouse=True)
def clear_shared_state():
    """Clear SharedState before each test to avoid contamination."""
    shared_state = get_shared_state()
    shared_state._store.clear()
    yield
    shared_state._store.clear()


@pytest.fixture
def mock_rfm_metrics():
    """Create mock RFM metrics for testing."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    metrics = []
    for i in range(100):
        total_spend = Decimal(str((i % 10 + 1) * 100))
        frequency = i % 5 + 1
        metrics.append(
            RFMMetrics(
                customer_id=f"cust_{i:03d}",
                observation_start=base_date,
                observation_end=base_date + timedelta(days=90),
                recency_days=i % 30 + 1,
                frequency=frequency,
                monetary=total_spend / frequency,  # Average transaction value
                total_spend=total_spend,
            )
        )

    return metrics


@pytest.fixture
def mock_period_aggregations():
    """Create mock period aggregations for testing."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    aggregations = []
    for period_idx in range(4):  # 4 quarters
        period_start = base_date + timedelta(days=period_idx * 90)
        period_end = period_start + timedelta(days=89)

        for customer_idx in range(100):
            aggregations.append(
                PeriodAggregation(
                    customer_id=f"cust_{customer_idx:03d}",
                    period_start=period_start,
                    period_end=period_end,
                    total_orders=customer_idx % 5 + 1,
                    total_spend=float((customer_idx % 10 + 1) * 100),
                    total_margin=float((customer_idx % 10 + 1) * 30),
                    total_quantity=(customer_idx % 5 + 1) * 10,
                    last_transaction_ts=period_end - timedelta(days=1),
                )
            )

    return aggregations


@pytest.fixture
def mock_cohort_definitions():
    """Create mock cohort definitions for testing."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    cohorts = []
    for quarter in range(4):
        start_date = base_date + timedelta(days=quarter * 90)
        end_date = start_date + timedelta(days=89)

        cohorts.append(
            CohortDefinition(
                cohort_id=f"2024-Q{quarter + 1}",
                start_date=start_date,
                end_date=end_date,
            )
        )

    return cohorts


@pytest.fixture
def mock_cohort_assignments():
    """Create mock cohort assignments for testing."""
    assignments = {}

    for quarter in range(4):
        for i in range(quarter * 25, (quarter + 1) * 25):
            assignments[f"cust_{i:03d}"] = f"2024-Q{quarter + 1}"

    return assignments


@pytest.fixture
def mock_data_mart(mock_period_aggregations):
    """Create mock data mart for testing."""

    class MockDataMart:
        def __init__(self, periods):
            self.periods = {PeriodGranularity.QUARTER: periods}

    return MockDataMart(mock_period_aggregations)


@pytest.mark.asyncio
async def test_lens1_basic_analysis(mock_rfm_metrics):
    """Test Lens 1 basic single-period analysis."""
    ctx = MockContext()

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("rfm_metrics", mock_rfm_metrics)
    shared_state.set("rfm_scores", [])

    request = Lens1Request(period_name="Q1 2024")

    response = await _analyze_single_period_snapshot_impl(request, ctx)

    # Verify response structure
    assert response.period_name == "Q1 2024"
    assert response.total_customers == 100
    assert response.customer_health_score >= 0
    assert response.customer_health_score <= 100
    assert response.concentration_risk in ("low", "medium", "high")
    assert len(response.recommendations) > 0

    # Verify SharedState was updated
    assert shared_state.get("lens1_result") is not None

    # Verify progress was reported
    assert len(ctx.progress_reports) > 0


@pytest.mark.asyncio
async def test_lens1_without_rfm_fails():
    """Test Lens 1 fails gracefully without RFM metrics."""
    ctx = MockContext()

    request = Lens1Request(period_name="Q1 2024")

    with pytest.raises(ValueError, match="RFM metrics not found"):
        await _analyze_single_period_snapshot_impl(request, ctx)


@pytest.mark.asyncio
async def test_lens2_period_comparison(mock_rfm_metrics):
    """Test Lens 2 period-to-period comparison."""
    ctx = MockContext()

    # Create two different period datasets
    period1_rfm = mock_rfm_metrics[:80]  # Simulate 80 customers in period 1
    period2_rfm = mock_rfm_metrics[20:]  # Simulate 80 customers in period 2 (60 overlap)

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("rfm_metrics", period1_rfm)
    shared_state.set("rfm_metrics_period2", period2_rfm)

    request = Lens2Request(period1_name="Q1 2024", period2_name="Q2 2024")

    response = await _analyze_period_comparison_impl(request, ctx)

    # Verify response structure
    assert response.period1_name == "Q1 2024"
    assert response.period2_name == "Q2 2024"
    assert response.period1_customers == 80
    assert response.period2_customers == 80
    assert response.retention_rate >= 0
    assert response.retention_rate <= 100
    assert response.churn_rate >= 0
    assert response.churn_rate <= 100
    assert response.growth_momentum in ("strong", "moderate", "declining", "negative")
    assert len(response.key_drivers) >= 0
    assert len(response.recommendations) > 0

    # Verify SharedState was updated
    assert shared_state.get("lens2_result") is not None


@pytest.mark.asyncio
async def test_lens2_missing_period_fails(mock_rfm_metrics):
    """Test Lens 2 fails gracefully with missing period data."""
    ctx = MockContext()

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("rfm_metrics", mock_rfm_metrics)
    # Missing period2

    request = Lens2Request()

    with pytest.raises(ValueError, match="Period 2 RFM metrics not found"):
        await _analyze_period_comparison_impl(request, ctx)


@pytest.mark.asyncio
async def test_lens3_cohort_evolution(
    mock_data_mart, mock_cohort_definitions, mock_cohort_assignments
):
    """Test Lens 3 single cohort evolution analysis."""
    ctx = MockContext()

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("data_mart", mock_data_mart)
    shared_state.set("cohort_definitions", mock_cohort_definitions)
    shared_state.set("cohort_assignments", mock_cohort_assignments)

    request = Lens3Request(cohort_id="2024-Q1")

    response = await _analyze_cohort_lifecycle_impl(request, ctx)

    # Verify response structure
    assert response.cohort_id == "2024-Q1"
    assert response.cohort_size == 25  # 25 customers per cohort
    assert response.periods_analyzed >= 0
    assert response.cohort_maturity in ("early", "growth", "mature", "declining")
    assert response.ltv_trajectory in ("strong", "moderate", "weak")
    assert len(response.recommendations) > 0

    # Verify curves are present
    assert isinstance(response.activation_curve, dict)
    assert isinstance(response.revenue_curve, dict)
    assert isinstance(response.retention_curve, dict)

    # Verify SharedState was updated
    assert shared_state.get("lens3_result") is not None


@pytest.mark.asyncio
async def test_lens3_invalid_cohort_fails(
    mock_data_mart, mock_cohort_definitions, mock_cohort_assignments
):
    """Test Lens 3 fails gracefully with invalid cohort ID."""
    ctx = MockContext()

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("data_mart", mock_data_mart)
    shared_state.set("cohort_definitions", mock_cohort_definitions)
    shared_state.set("cohort_assignments", mock_cohort_assignments)

    request = Lens3Request(cohort_id="INVALID_COHORT")

    with pytest.raises(ValueError, match="Cohort 'INVALID_COHORT' not found"):
        await _analyze_cohort_lifecycle_impl(request, ctx)


@pytest.mark.asyncio
async def test_lens4_multi_cohort_comparison(
    mock_data_mart, mock_cohort_assignments
):
    """Test Lens 4 multi-cohort comparison analysis."""
    ctx = MockContext()

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("data_mart", mock_data_mart)
    shared_state.set("cohort_assignments", mock_cohort_assignments)

    request = Lens4Request(alignment_type="left-aligned")

    response = await _compare_multiple_cohorts_impl(request, ctx)

    # Verify response structure
    assert response.cohort_count == 4  # 4 cohorts
    assert response.alignment_type == "left-aligned"
    assert len(response.cohort_summaries) == 4

    # Verify cohort summaries
    for summary in response.cohort_summaries:
        assert summary.cohort_size > 0
        assert summary.total_revenue >= 0
        assert summary.avg_revenue_per_member >= 0

    # Verify comparative insights
    assert response.best_performing_cohort is not None
    assert len(response.key_differences) >= 0
    assert len(response.recommendations) > 0

    # Verify SharedState was updated
    assert shared_state.get("lens4_result") is not None


@pytest.mark.asyncio
async def test_lens4_time_aligned_mode(mock_data_mart, mock_cohort_assignments):
    """Test Lens 4 with time-aligned mode."""
    ctx = MockContext()

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("data_mart", mock_data_mart)
    shared_state.set("cohort_assignments", mock_cohort_assignments)

    request = Lens4Request(alignment_type="time-aligned")

    response = await _compare_multiple_cohorts_impl(request, ctx)

    assert response.alignment_type == "time-aligned"
    assert response.cohort_count == 4


@pytest.mark.asyncio
async def test_lens4_with_margin(mock_data_mart, mock_cohort_assignments):
    """Test Lens 4 with margin analysis enabled."""
    ctx = MockContext()

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("data_mart", mock_data_mart)
    shared_state.set("cohort_assignments", mock_cohort_assignments)

    request = Lens4Request(include_margin=True)

    response = await _compare_multiple_cohorts_impl(request, ctx)

    assert response.cohort_count == 4


@pytest.mark.asyncio
async def test_full_workflow_integration(
    mock_rfm_metrics,
    mock_data_mart,
    mock_cohort_definitions,
    mock_cohort_assignments,
):
    """Test full workflow from foundation to all lenses."""
    ctx = MockContext()

    # Populate SharedState (used by tool implementation)
    shared_state = get_shared_state()
    shared_state.set("data_mart", mock_data_mart)
    shared_state.set("rfm_metrics", mock_rfm_metrics)
    shared_state.set("rfm_scores", [])
    shared_state.set("cohort_definitions", mock_cohort_definitions)
    shared_state.set("cohort_assignments", mock_cohort_assignments)

    # Run Lens 1
    lens1_response = await _analyze_single_period_snapshot_impl(
        Lens1Request(period_name="Q1 2024"), ctx
    )
    assert lens1_response.total_customers == 100

    # Run Lens 3 (cohort-based)
    lens3_response = await _analyze_cohort_lifecycle_impl(
        Lens3Request(cohort_id="2024-Q1"), ctx
    )
    assert lens3_response.cohort_size == 25

    # Run Lens 4 (multi-cohort)
    lens4_response = await _compare_multiple_cohorts_impl(
        Lens4Request(alignment_type="left-aligned"), ctx
    )
    assert lens4_response.cohort_count == 4

    # Verify all results stored in SharedState
    assert shared_state.get("lens1_result") is not None
    assert shared_state.get("lens3_result") is not None
    assert shared_state.get("lens4_result") is not None


def test_lens1_health_score_calculation():
    """Test Lens 1 health score calculation logic."""
    from analytics.services.mcp_server.tools.lens1 import (
        _calculate_customer_health_score,
    )
    from customer_base_audit.analyses.lens1 import Lens1Metrics

    # Create test metrics with values that trigger penalties
    # 75% one-time buyers should trigger -30 penalty
    # 65% top 10 concentration should trigger -10 penalty
    # Total expected score: 100 - 30 - 10 = 60
    metrics = Lens1Metrics(
        total_customers=100,
        one_time_buyers=75,
        one_time_buyer_pct=Decimal("75"),
        total_revenue=Decimal("10000"),
        top_10pct_revenue_contribution=Decimal("65"),
        top_20pct_revenue_contribution=Decimal("80"),
        avg_orders_per_customer=Decimal("2.5"),
        median_customer_value=Decimal("100"),
        rfm_distribution={},
    )

    score = _calculate_customer_health_score(metrics)
    assert 0 <= score <= 100
    assert score == 60.0  # Should be penalized: 100 - 30 - 10 = 60


def test_lens2_growth_momentum_assessment():
    """Test Lens 2 growth momentum assessment logic."""
    from analytics.services.mcp_server.tools.lens2 import _assess_growth_momentum
    from customer_base_audit.analyses.lens2 import Lens2Metrics, CustomerMigration
    from customer_base_audit.analyses.lens1 import Lens1Metrics

    # Create minimal test metrics
    period_metrics = Lens1Metrics(
        total_customers=100,
        one_time_buyers=50,
        one_time_buyer_pct=Decimal("50"),
        total_revenue=Decimal("10000"),
        top_10pct_revenue_contribution=Decimal("60"),
        top_20pct_revenue_contribution=Decimal("75"),
        avg_orders_per_customer=Decimal("2.5"),
        median_customer_value=Decimal("100"),
        rfm_distribution={},
    )

    # Create sets of customer IDs for migration
    retained_ids = set([f"cust_{i:03d}" for i in range(70)])
    churned_ids = set([f"cust_{i:03d}" for i in range(70, 100)])
    new_ids = set([f"cust_{i:03d}" for i in range(100, 130)])

    migration = CustomerMigration(
        retained=retained_ids,
        churned=churned_ids,
        new=new_ids,
        reactivated=set()
    )

    metrics = Lens2Metrics(
        period1_metrics=period_metrics,
        period2_metrics=period_metrics,
        migration=migration,
        retention_rate=Decimal("70"),
        churn_rate=Decimal("30"),
        reactivation_rate=Decimal("0"),
        customer_count_change=0,
        revenue_change_pct=Decimal("15"),
        avg_order_value_change_pct=Decimal("5"),
    )

    momentum = _assess_growth_momentum(metrics)
    assert momentum in ("strong", "moderate", "declining", "negative")
