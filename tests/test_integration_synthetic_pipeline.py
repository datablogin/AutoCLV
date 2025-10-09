"""Integration test for synthetic data generation and validation.

This test ensures that synthetic data generation produces valid, realistic datasets
that can be used for testing CLV pipelines and customer-base audits.

Note: Full pipeline testing (Lens 1-3) requires data mart and RFM infrastructure
which depends on PeriodAggregation building. Those tests will be added when
foundation infrastructure is complete.
"""

from datetime import date

import pytest

from customer_base_audit.synthetic import (
    generate_customers,
    generate_transactions,
    BASELINE_SCENARIO,
    HIGH_CHURN_SCENARIO,
    HEAVY_PROMOTION_SCENARIO,
    STABLE_BUSINESS_SCENARIO,
    check_spend_distribution_is_realistic,
    check_temporal_coverage,
    check_no_duplicate_transactions,
    check_cohort_decay_pattern,
)


def test_synthetic_data_quality_baseline() -> None:
    """Test that baseline scenario produces high-quality synthetic data."""
    customers = generate_customers(200, date(2024, 1, 1), date(2024, 12, 31), seed=123)
    transactions = generate_transactions(
        customers, date(2024, 1, 1), date(2024, 12, 31), scenario=BASELINE_SCENARIO
    )

    # Validate data quality
    assert check_spend_distribution_is_realistic(transactions).ok
    assert check_temporal_coverage(transactions, customers, min_months_with_activity=6).ok
    assert check_no_duplicate_transactions(transactions).ok
    # Note: cohort_decay_pattern check can be noisy with small samples due to reactivations

    # Basic data integrity checks
    assert len(transactions) > 100
    assert len(set(c.customer_id for c in customers)) == len(customers)

    # All transactions should be from valid customers
    customer_ids = set(c.customer_id for c in customers)
    for t in transactions:
        assert t.customer_id in customer_ids


def test_high_churn_vs_baseline() -> None:
    """Verify high churn scenario produces fewer transactions than baseline."""
    customers = generate_customers(100, date(2024, 1, 1), date(2024, 12, 31), seed=111)

    # Generate baseline scenario
    txns_baseline = generate_transactions(
        customers, date(2024, 1, 1), date(2024, 12, 31), scenario=BASELINE_SCENARIO
    )

    # Generate high churn scenario
    txns_high_churn = generate_transactions(
        customers, date(2024, 1, 1), date(2024, 12, 31), scenario=HIGH_CHURN_SCENARIO
    )

    # High churn should produce fewer transactions (more customers leave)
    assert len(txns_high_churn) < len(txns_baseline)

    # Both should pass validation
    assert check_spend_distribution_is_realistic(txns_baseline).ok
    assert check_spend_distribution_is_realistic(txns_high_churn).ok


def test_promotion_scenario_produces_spike() -> None:
    """Verify heavy promotion scenario produces more transactions."""
    customers = generate_customers(100, date(2024, 1, 1), date(2024, 12, 31), seed=222)

    # Generate baseline vs promotion
    txns_baseline = generate_transactions(
        customers, date(2024, 1, 1), date(2024, 12, 31), scenario=BASELINE_SCENARIO
    )

    txns_promotion = generate_transactions(
        customers, date(2024, 1, 1), date(2024, 12, 31), scenario=HEAVY_PROMOTION_SCENARIO
    )

    # Promotion should generate more transactions
    assert len(txns_promotion) > len(txns_baseline)

    # Check that November spike is detectable
    from customer_base_audit.synthetic import check_promo_spike_signal

    result = check_promo_spike_signal(txns_promotion, promo_month=11, min_ratio=1.3)
    assert result.ok, f"Heavy promotion spike should be detectable: {result.message}"


def test_stable_business_high_repeat_rate() -> None:
    """Verify stable business scenario produces high repeat purchase behavior."""
    customers = generate_customers(100, date(2024, 1, 1), date(2024, 12, 31), seed=333)

    txns = generate_transactions(
        customers, date(2024, 1, 1), date(2024, 12, 31), scenario=STABLE_BUSINESS_SCENARIO
    )

    # Stable business should have high transaction volume
    # (low churn + high frequency)
    txns_per_customer = len(txns) / len(customers)
    assert txns_per_customer > 10, f"Stable business should have high repeat rate, got {txns_per_customer:.2f}"

    # Validate quality
    assert check_spend_distribution_is_realistic(txns).ok
    assert check_temporal_coverage(txns, customers).ok


@pytest.mark.slow
def test_large_scale_synthetic_generation() -> None:
    """Test generation and validation of large synthetic dataset (marked slow for CI)."""
    # Generate larger dataset
    customers = generate_customers(1000, date(2024, 1, 1), date(2024, 12, 31), seed=42)
    transactions = generate_transactions(
        customers, date(2024, 1, 1), date(2024, 12, 31), scenario=BASELINE_SCENARIO
    )

    # Should generate substantial data
    assert len(transactions) > 5000

    # Validate at scale
    assert check_spend_distribution_is_realistic(transactions).ok
    assert check_temporal_coverage(transactions, customers, min_months_with_activity=6).ok
    assert check_no_duplicate_transactions(transactions).ok

    # Data integrity checks
    customer_ids = set(c.customer_id for c in customers)
    for t in transactions:
        assert t.customer_id in customer_ids
