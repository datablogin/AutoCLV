"""Test Lens 4 Phase 2 features with synthetic data."""

from datetime import datetime

from customer_base_audit.synthetic import (
    BASELINE_SCENARIO,
    generate_customers,
    generate_transactions,
)
from customer_base_audit.foundation.data_mart import (
    CustomerDataMartBuilder,
    PeriodGranularity,
)
from customer_base_audit.foundation.customer_contract import CustomerIdentifier
from customer_base_audit.foundation.cohorts import (
    create_monthly_cohorts,
    assign_cohorts,
)
from customer_base_audit.analyses.lens4 import compare_cohorts


def main():
    """Test Lens 4 with synthetic data."""
    print("=" * 80)
    print("Testing Lens 4 Phase 2 with Synthetic Data")
    print("=" * 80)

    # Generate synthetic customers and transactions
    print("\n1. Generating synthetic data...")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)

    customers = generate_customers(
        n=1000,
        start=start_date.date(),
        end=datetime(2024, 3, 31).date(),  # 3 months of acquisitions
    )
    print(f"   Generated {len(customers)} customers")

    transactions = generate_transactions(
        customers=customers,
        scenario=BASELINE_SCENARIO,
        start=start_date.date(),
        end=end_date.date(),
    )
    print(f"   Generated {len(transactions)} transactions")

    # Build period aggregations (monthly)
    print("\n2. Building data mart...")

    # Convert Transaction objects to dictionaries
    transaction_dicts = [
        {
            "order_id": txn.order_id,
            "customer_id": txn.customer_id,
            "event_ts": txn.event_ts,
            "unit_price": txn.unit_price,
            "quantity": txn.quantity,
        }
        for txn in transactions
    ]

    mart_builder = CustomerDataMartBuilder(
        period_granularities=[PeriodGranularity.MONTH]
    )
    data_mart = mart_builder.build(transaction_dicts)
    period_aggregations = data_mart.periods[PeriodGranularity.MONTH]
    print(f"   Created {len(period_aggregations)} period aggregations")

    # Create monthly cohorts and assign customers
    print("\n3. Creating cohorts...")

    # Convert Customer objects to CustomerIdentifier objects
    customer_identifiers = [
        CustomerIdentifier(
            customer_id=c.customer_id,
            acquisition_ts=datetime.combine(c.acquisition_date, datetime.min.time()),
            source_system="synthetic",
        )
        for c in customers
    ]

    cohort_definitions = create_monthly_cohorts(customers=customer_identifiers)
    print(f"   Created {len(cohort_definitions)} monthly cohorts")

    cohort_assignments = assign_cohorts(
        customers=customer_identifiers,
        cohort_definitions=cohort_definitions,
    )
    print(f"   Assigned {len(cohort_assignments)} customers to cohorts")

    # Test LEFT-ALIGNED comparison
    print("\n" + "=" * 80)
    print("LEFT-ALIGNED COMPARISON (by cohort age)")
    print("=" * 80)

    metrics_left = compare_cohorts(
        period_aggregations=period_aggregations,
        cohort_assignments=cohort_assignments,
        alignment_type="left-aligned",
        include_margin=False,
    )

    print("\nResults:")
    print(f"  - Cohort decompositions: {len(metrics_left.cohort_decompositions)}")
    print(f"  - Time to second purchase: {len(metrics_left.time_to_second_purchase)}")
    print(f"  - Cohort comparisons: {len(metrics_left.cohort_comparisons)}")

    # Show some decompositions
    print("\n  Sample Cohort Decompositions (first 5):")
    for decomp in metrics_left.cohort_decompositions[:5]:
        print(
            f"    {decomp.cohort_id} Period {decomp.period_number}: "
            f"{decomp.active_customers}/{decomp.cohort_size} active "
            f"({decomp.pct_active}%), "
            f"AOF={decomp.aof}, AOV=${decomp.aov}, "
            f"Revenue=${decomp.total_revenue}"
        )

    # Show time to second purchase
    print("\n  Time to Second Purchase Analysis:")
    for ttsp in metrics_left.time_to_second_purchase:
        print(
            f"    {ttsp.cohort_id}: "
            f"{ttsp.customers_with_repeat} repeated ({ttsp.repeat_rate}%), "
            f"Median={ttsp.median_days} days, Mean={ttsp.mean_days} days"
        )

    # Show cohort comparisons
    print("\n  Cohort Comparisons (first 5):")
    for comp in metrics_left.cohort_comparisons[:5]:
        print(
            f"    {comp.cohort_a_id} vs {comp.cohort_b_id} at Period {comp.period_number}:"
        )
        print(f"      % Active: {comp.pct_active_change_pct}% change")
        print(f"      AOF: {comp.aof_change_pct}% change")
        print(
            f"      Revenue/Customer: ${comp.revenue_delta} ({comp.revenue_change_pct}%)"
        )

    # Test TIME-ALIGNED comparison
    print("\n" + "=" * 80)
    print("TIME-ALIGNED COMPARISON (by calendar period)")
    print("=" * 80)

    metrics_time = compare_cohorts(
        period_aggregations=period_aggregations,
        cohort_assignments=cohort_assignments,
        alignment_type="time-aligned",
        include_margin=False,
    )

    print("\nResults:")
    print(f"  - Cohort decompositions: {len(metrics_time.cohort_decompositions)}")
    print(f"  - Time to second purchase: {len(metrics_time.time_to_second_purchase)}")
    print(f"  - Cohort comparisons: {len(metrics_time.cohort_comparisons)}")

    # Show some decompositions
    print("\n  Sample Cohort Decompositions (first 5):")
    for decomp in metrics_time.cohort_decompositions[:5]:
        print(
            f"    {decomp.cohort_id} Period {decomp.period_number}: "
            f"{decomp.active_customers}/{decomp.cohort_size} active "
            f"({decomp.pct_active}%), "
            f"AOF={decomp.aof}, AOV=${decomp.aov}, "
            f"Revenue=${decomp.total_revenue}"
        )

    # Verify difference between alignments
    print("\n" + "=" * 80)
    print("VERIFICATION: Left-aligned vs Time-aligned")
    print("=" * 80)
    print(
        f"\nLeft-aligned has {len(metrics_left.cohort_decompositions)} decompositions"
    )
    print(f"Time-aligned has {len(metrics_time.cohort_decompositions)} decompositions")
    print(
        "\nIn time-aligned mode, all cohorts are shown in each calendar period they're active."
    )
    print(
        "In left-aligned mode, cohorts are shown by their age (periods since acquisition)."
    )

    # Show period distribution
    left_periods = set(d.period_number for d in metrics_left.cohort_decompositions)
    time_periods = set(d.period_number for d in metrics_time.cohort_decompositions)
    print(f"\nLeft-aligned period range: {min(left_periods)} to {max(left_periods)}")
    print(f"Time-aligned period range: {min(time_periods)} to {max(time_periods)}")

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
