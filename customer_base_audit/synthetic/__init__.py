"""Synthetic data generation and validation utilities.

This package helps produce realistic-but-fake datasets to exercise
customer audit and CLV pipelines without accessing production data.
"""

from .generator import (
    Customer,
    Transaction,
    ScenarioConfig,
    generate_customers,
    generate_transactions,
)
from .validation import (
    check_non_negative_amounts,
    check_reasonable_order_density,
    check_promo_spike_signal,
    check_spend_distribution_is_realistic,
    check_cohort_decay_pattern,
    check_no_duplicate_transactions,
    check_temporal_coverage,
)
from .scenarios import (
    BASELINE_SCENARIO,
    HIGH_CHURN_SCENARIO,
    PRODUCT_RECALL_SCENARIO,
    HEAVY_PROMOTION_SCENARIO,
    PRODUCT_LAUNCH_SCENARIO,
    SEASONAL_BUSINESS_SCENARIO,
    STABLE_BUSINESS_SCENARIO,
)

__all__ = [
    "Customer",
    "Transaction",
    "ScenarioConfig",
    "generate_customers",
    "generate_transactions",
    "check_non_negative_amounts",
    "check_reasonable_order_density",
    "check_promo_spike_signal",
    "check_spend_distribution_is_realistic",
    "check_cohort_decay_pattern",
    "check_no_duplicate_transactions",
    "check_temporal_coverage",
    "BASELINE_SCENARIO",
    "HIGH_CHURN_SCENARIO",
    "PRODUCT_RECALL_SCENARIO",
    "HEAVY_PROMOTION_SCENARIO",
    "PRODUCT_LAUNCH_SCENARIO",
    "SEASONAL_BUSINESS_SCENARIO",
    "STABLE_BUSINESS_SCENARIO",
]
