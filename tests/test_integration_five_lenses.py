"""End-to-end integration test for the complete Five Lenses workflow.

This test verifies that the Five Lenses customer-base audit framework components
work together correctly, from raw transaction data through all five lenses of analysis.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from customer_base_audit.foundation.data_mart import (
    CustomerDataMartBuilder,
    PeriodGranularity,
)
from customer_base_audit.foundation.customer_contract import CustomerIdentifier
from customer_base_audit.foundation.cohorts import (
    CohortDefinition,
    CohortMetadata,
    assign_cohorts,
)
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from customer_base_audit.analyses.lens1 import analyze_single_period
from customer_base_audit.analyses.lens2 import analyze_period_comparison
from customer_base_audit.analyses.lens3 import analyze_cohort_evolution


def test_five_lenses_complete_pipeline():
    """Test complete Five Lenses workflow from raw data to insights.

    This integration test demonstrates the recommended workflow:
    1. Build data mart from raw transactions
    2. Calculate RFM metrics
    3. Define cohorts and assign customers
    4. Run Lens 1 (single period analysis)
    5. Run Lens 2 (period-to-period comparison)
    6. Run Lens 3 (cohort evolution)
    7. Verify all metrics are internally consistent

    This serves both as a test and as living documentation of the framework.
    """

    # ============================================================================
    # Step 1: Create sample transaction data
    # ============================================================================

    # Create realistic transaction data spanning 2 quarters
    # Q1 2023: 3 customers with varying purchase patterns
    # Q2 2023: 2 retained, 1 churned, 2 new customers
    transactions = [
        # Customer C1: High-value repeat buyer (retained in Q2)
        {
            "order_id": "O1",
            "customer_id": "C1",
            "event_ts": datetime(2023, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "unit_price": 100.0,
            "quantity": 2,
        },
        {
            "order_id": "O2",
            "customer_id": "C1",
            "event_ts": datetime(2023, 2, 20, 14, 30, 0, tzinfo=timezone.utc),
            "unit_price": 150.0,
            "quantity": 1,
        },
        {
            "order_id": "O3",
            "customer_id": "C1",
            "event_ts": datetime(2023, 4, 10, 9, 15, 0, tzinfo=timezone.utc),
            "unit_price": 120.0,
            "quantity": 3,
        },
        # Customer C2: One-time buyer in Q1 (churned)
        {
            "order_id": "O4",
            "customer_id": "C2",
            "event_ts": datetime(2023, 1, 25, 16, 45, 0, tzinfo=timezone.utc),
            "unit_price": 50.0,
            "quantity": 1,
        },
        # Customer C3: Moderate repeat buyer (retained in Q2)
        {
            "order_id": "O5",
            "customer_id": "C3",
            "event_ts": datetime(2023, 2, 5, 11, 20, 0, tzinfo=timezone.utc),
            "unit_price": 80.0,
            "quantity": 2,
        },
        {
            "order_id": "O6",
            "customer_id": "C3",
            "event_ts": datetime(2023, 5, 12, 13, 0, 0, tzinfo=timezone.utc),
            "unit_price": 90.0,
            "quantity": 1,
        },
        # Customer C4: New in Q2
        {
            "order_id": "O7",
            "customer_id": "C4",
            "event_ts": datetime(2023, 4, 20, 15, 30, 0, tzinfo=timezone.utc),
            "unit_price": 200.0,
            "quantity": 1,
        },
        # Customer C5: New in Q2, high-value
        {
            "order_id": "O8",
            "customer_id": "C5",
            "event_ts": datetime(2023, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
            "unit_price": 300.0,
            "quantity": 2,
        },
        {
            "order_id": "O9",
            "customer_id": "C5",
            "event_ts": datetime(2023, 6, 15, 14, 0, 0, tzinfo=timezone.utc),
            "unit_price": 250.0,
            "quantity": 1,
        },
    ]

    # ============================================================================
    # Step 2: Build data mart from transactions
    # ============================================================================

    mart_builder = CustomerDataMartBuilder(
        period_granularities=[PeriodGranularity.QUARTER]
    )
    data_mart = mart_builder.build(transactions)

    # Verify data mart structure
    assert len(data_mart.orders) == 9
    assert PeriodGranularity.QUARTER in data_mart.periods

    quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]

    # ============================================================================
    # Step 3: Calculate RFM metrics for each quarter
    # ============================================================================

    # Q1 2023: Jan 1 - Mar 31 (period_end is exclusive, so Apr 1)
    q1_end = datetime(2023, 4, 1, tzinfo=timezone.utc)
    q1_periods = [p for p in quarterly_periods if p.period_end == q1_end]
    q1_observation_end = datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc)
    q1_rfm = calculate_rfm(q1_periods, observation_end=q1_observation_end)

    # Q2 2023: Apr 1 - Jun 30 (period_end is exclusive, so Jul 1)
    q2_end = datetime(2023, 7, 1, tzinfo=timezone.utc)
    q2_periods = [p for p in quarterly_periods if p.period_end == q2_end]
    q2_observation_end = datetime(2023, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
    q2_rfm = calculate_rfm(q2_periods, observation_end=q2_observation_end)

    # Verify RFM calculations
    assert len(q1_rfm) == 3  # C1, C2, C3 active in Q1
    assert len(q2_rfm) == 4  # C1, C3, C4, C5 active in Q2

    # Calculate RFM scores for segmentation
    q1_scores = calculate_rfm_scores(q1_rfm)
    q2_scores = calculate_rfm_scores(q2_rfm)

    # Verify RFM score ranges (important for CLV segmentation accuracy)
    for score in q1_scores:
        assert 1 <= score.r_score <= 5, f"Invalid recency score: {score.r_score}"
        assert 1 <= score.f_score <= 5, f"Invalid frequency score: {score.f_score}"
        assert 1 <= score.m_score <= 5, f"Invalid monetary score: {score.m_score}"

    for score in q2_scores:
        assert 1 <= score.r_score <= 5, f"Invalid recency score: {score.r_score}"
        assert 1 <= score.f_score <= 5, f"Invalid frequency score: {score.f_score}"
        assert 1 <= score.m_score <= 5, f"Invalid monetary score: {score.m_score}"

    # ============================================================================
    # Step 4: Lens 1 - Single Period Analysis
    # ============================================================================

    # Analyze Q1 in isolation
    lens1_q1 = analyze_single_period(q1_rfm, rfm_scores=q1_scores)

    assert lens1_q1.total_customers == 3
    assert lens1_q1.one_time_buyers == 2  # C2 and C3 (C1 has 2 orders in Q1)
    assert lens1_q1.one_time_buyer_pct == Decimal("66.67")
    assert lens1_q1.total_revenue == Decimal(
        "560.00"
    )  # C1: 200+150=350, C2: 50, C3: 160

    # Analyze Q2 in isolation
    lens1_q2 = analyze_single_period(q2_rfm, rfm_scores=q2_scores)

    assert lens1_q2.total_customers == 4
    assert lens1_q2.one_time_buyers == 3  # C1, C3, C4 have 1 order each in Q2; C5 has 2
    assert lens1_q2.one_time_buyer_pct == Decimal("75.00")
    assert lens1_q2.total_revenue == Decimal(
        "1500.00"
    )  # C1: 360, C3: 90, C4: 200, C5: 850

    # ============================================================================
    # Step 5: Lens 2 - Period-to-Period Comparison
    # ============================================================================

    # Compare Q1 → Q2 to track customer migration
    lens2 = analyze_period_comparison(
        period1_rfm=q1_rfm,
        period2_rfm=q2_rfm,
        period1_metrics=lens1_q1,
        period2_metrics=lens1_q2,
    )

    # Verify Lens 1 metrics are correctly propagated to Lens 2
    assert lens2.period1_metrics == lens1_q1
    assert lens2.period2_metrics == lens1_q2

    # Verify migration patterns
    assert len(lens2.migration.retained) == 2  # C1, C3
    assert len(lens2.migration.churned) == 1  # C2
    assert len(lens2.migration.new) == 2  # C4, C5
    assert lens2.retention_rate == Decimal("66.67")  # 2/3
    assert lens2.churn_rate == Decimal("33.33")  # 1/3

    # Verify customer count change
    assert lens2.customer_count_change == 1  # 4 - 3

    # Verify revenue metrics
    revenue_change = (
        (lens1_q2.total_revenue - lens1_q1.total_revenue) / lens1_q1.total_revenue * 100
    )
    assert lens2.revenue_change_pct == revenue_change.quantize(Decimal("0.01"))

    # ============================================================================
    # Step 6: Lens 3 - Cohort Evolution
    # ============================================================================

    # Create Q1 cohort (customers acquired in Q1)
    cohort_metadata: CohortMetadata = {
        "description": "Q1 2023 Cohort",
        "acquisition_channel": "mixed",
        "created_by": "integration_test",
    }

    q1_cohort = CohortDefinition(
        cohort_id="2023-Q1",
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2023, 4, 1, tzinfo=timezone.utc),
        metadata=cohort_metadata,
    )

    # Assign customers to cohorts based on first transaction (use UTC timezone)
    customers = [
        CustomerIdentifier("C1", datetime(2023, 1, 15, tzinfo=timezone.utc), "transactions"),
        CustomerIdentifier("C2", datetime(2023, 1, 25, tzinfo=timezone.utc), "transactions"),
        CustomerIdentifier("C3", datetime(2023, 2, 5, tzinfo=timezone.utc), "transactions"),
        CustomerIdentifier("C4", datetime(2023, 4, 20, tzinfo=timezone.utc), "transactions"),
        CustomerIdentifier("C5", datetime(2023, 6, 1, tzinfo=timezone.utc), "transactions"),
    ]

    cohort_assignments = assign_cohorts(customers, [q1_cohort])
    q1_cohort_customers = frozenset(
        cid for cid, cohort_id in cohort_assignments.items() if cohort_id == "2023-Q1"
    )

    assert q1_cohort_customers == frozenset(["C1", "C2", "C3"])

    # Track Q1 cohort evolution over time
    all_periods = data_mart.periods[PeriodGranularity.QUARTER]
    lens3 = analyze_cohort_evolution(
        cohort_name=q1_cohort.cohort_id,
        acquisition_date=q1_cohort.start_date,
        period_aggregations=all_periods,
        cohort_customer_ids=list(q1_cohort_customers),
    )

    # Verify cohort metrics
    assert lens3.cohort_size == 3
    assert len(lens3.periods) == 2  # Q1 and Q2

    # Period 0 (Q1): All 3 customers should be active
    period_0 = lens3.periods[0]
    assert period_0.period_number == 0
    assert period_0.active_customers == 3
    assert period_0.cumulative_activation_rate == 1.0  # 100% as float

    # Period 1 (Q2): Only C1 and C3 retained (C2 churned)
    period_1 = lens3.periods[1]
    assert period_1.period_number == 1
    assert period_1.active_customers == 2
    assert period_1.cumulative_activation_rate == 1.0  # Ever-active (100% as float)

    # ============================================================================
    # Step 7: Cross-Lens Consistency Checks
    # ============================================================================

    # Lens 1 and Lens 2 should have consistent customer counts
    assert lens1_q1.total_customers == len(lens2.migration.retained) + len(
        lens2.migration.churned
    )
    assert lens1_q2.total_customers == len(lens2.migration.retained) + len(
        lens2.migration.new
    )

    # Lens 3 cohort size should match Q1 customer count
    assert lens3.cohort_size == lens1_q1.total_customers

    # Revenue should be consistent across lenses
    assert lens1_q1.total_revenue == lens2.period1_metrics.total_revenue
    assert lens1_q2.total_revenue == lens2.period2_metrics.total_revenue


@pytest.mark.slow
def test_five_lenses_pipeline_with_synthetic_data():
    """Test Five Lenses pipeline with larger synthetic dataset.

    This demonstrates the framework's ability to handle realistic data volumes
    and provides a more comprehensive integration test.
    """
    pytest.importorskip("customer_base_audit.synthetic_data")

    from customer_base_audit.synthetic_data.generator import (
        generate_customers_and_transactions,
    )
    from customer_base_audit.synthetic_data.scenarios import baseline_scenario

    # Generate synthetic data for 100 customers over 6 months
    config = baseline_scenario()
    config.num_customers = 100
    config.observation_start = datetime(2023, 1, 1)
    config.observation_end = datetime(2023, 6, 30)
    config.seed = 42  # Explicit seed for deterministic tests

    customers, transactions = generate_customers_and_transactions(config)

    # Build data mart
    mart_builder = CustomerDataMartBuilder(
        period_granularities=[PeriodGranularity.MONTH]
    )
    data_mart = mart_builder.build(transactions)

    # Calculate RFM for first and last month
    monthly_periods = data_mart.periods[PeriodGranularity.MONTH]

    jan_end = datetime(2023, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
    jan_periods = [p for p in monthly_periods if p.period_end <= jan_end]
    jan_rfm = calculate_rfm(jan_periods, observation_end=jan_end)

    jun_end = datetime(2023, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
    jun_periods = [
        p
        for p in monthly_periods
        if p.period_start >= datetime(2023, 6, 1, tzinfo=timezone.utc) and p.period_end <= jun_end
    ]
    jun_rfm = calculate_rfm(jun_periods, observation_end=jun_end)

    # Run Lens 1 on both periods
    lens1_jan = analyze_single_period(jan_rfm)
    lens1_jun = analyze_single_period(jun_rfm)

    # Run Lens 2 to compare
    lens2 = analyze_period_comparison(jan_rfm, jun_rfm)

    # Verify basic sanity checks
    assert lens1_jan.total_customers > 0
    assert lens1_jun.total_customers > 0
    assert 0 <= lens2.retention_rate <= 100
    assert 0 <= lens2.churn_rate <= 100

    # Verify consistency
    assert lens2.retention_rate + lens2.churn_rate <= Decimal("100.1")  # Rounding


def test_cohort_metadata_type_safety():
    """Test that CohortMetadata TypedDict provides type safety."""

    # Valid metadata using CohortMetadata structure
    metadata: CohortMetadata = {
        "description": "Test cohort",
        "campaign_id": "CAMP-001",
        "acquisition_channel": "paid-search",
    }

    cohort = CohortDefinition(
        cohort_id="test-cohort",
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2023, 2, 1, tzinfo=timezone.utc),
        metadata=metadata,
    )

    assert cohort.metadata["description"] == "Test cohort"
    assert cohort.metadata["campaign_id"] == "CAMP-001"

    # Custom metadata fields should also work (backward compatibility)
    custom_metadata = {
        "custom_field": "custom_value",
        "region": "US-WEST",
    }

    cohort_custom = CohortDefinition(
        cohort_id="test-cohort-2",
        start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2023, 2, 1, tzinfo=timezone.utc),
        metadata=custom_metadata,
    )

    assert cohort_custom.metadata["custom_field"] == "custom_value"
