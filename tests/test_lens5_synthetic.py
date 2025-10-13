"""Test Lens 5 with synthetic data."""

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
from customer_base_audit.analyses.lens5 import assess_customer_base_health


def main():
    """Test Lens 5 with synthetic data."""
    print("=" * 80)
    print("Testing Lens 5 (Overall Customer Base Health) with Synthetic Data")
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

    # Test Lens 5: Overall Customer Base Health
    print("\n" + "=" * 80)
    print("LENS 5: OVERALL CUSTOMER BASE HEALTH")
    print("=" * 80)

    metrics = assess_customer_base_health(
        period_aggregations=period_aggregations,
        cohort_assignments=cohort_assignments,
        analysis_start_date=start_date,
        analysis_end_date=end_date,
    )

    print("\nResults:")
    print(
        f"  - Cohort revenue contributions (C3 data): {len(metrics.cohort_revenue_contributions)}"
    )
    print(f"  - Cohort repeat behavior: {len(metrics.cohort_repeat_behavior)}")
    print(
        f"  - Analysis period: {metrics.analysis_start_date.date()} to {metrics.analysis_end_date.date()}"
    )

    # Show health score
    print("\n" + "-" * 80)
    print("HEALTH SCORE")
    print("-" * 80)
    health = metrics.health_score
    print(f"  Total Customers: {health.total_customers}")
    print(f"  Total Active Customers: {health.total_active_customers}")
    print(f"  Overall Retention Rate: {health.overall_retention_rate}%")
    print(f"  Cohort Quality Trend: {health.cohort_quality_trend}")
    print(f"  Revenue Predictability: {health.revenue_predictability_pct}%")
    print(f"  Acquisition Dependence: {health.acquisition_dependence_pct}%")
    print(f"  Health Score: {health.health_score} (Grade: {health.health_grade})")

    # Show cohort revenue contributions (C3 data) - first 10
    print("\n" + "-" * 80)
    print("COHORT REVENUE CONTRIBUTIONS (C3 Data) - First 10")
    print("-" * 80)
    for crp in metrics.cohort_revenue_contributions[:10]:
        print(
            f"  {crp.cohort_id} @ {crp.period_start.strftime('%Y-%m')}: "
            f"${crp.total_revenue} ({crp.pct_of_period_revenue}% of period), "
            f"{crp.active_customers} active, "
            f"${crp.avg_revenue_per_customer}/customer"
        )

    # Show cohort repeat behavior
    print("\n" + "-" * 80)
    print("COHORT REPEAT BEHAVIOR")
    print("-" * 80)
    for crb in metrics.cohort_repeat_behavior:
        print(
            f"  {crb.cohort_id}: "
            f"{crb.cohort_size} customers, "
            f"{crb.one_time_buyers} one-time, "
            f"{crb.repeat_buyers} repeat ({crb.repeat_rate}%), "
            f"Avg orders/repeat buyer: {crb.avg_orders_per_repeat_buyer}"
        )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_revenue = sum(
        crp.total_revenue for crp in metrics.cohort_revenue_contributions
    )
    total_active = sum(
        crp.active_customers for crp in metrics.cohort_revenue_contributions
    )

    print(f"  Total Revenue (analysis period): ${total_revenue}")
    print(f"  Total Active Customer-Periods: {total_active}")
    print(f"  Number of Cohorts: {len(metrics.cohort_repeat_behavior)}")

    # Repeat behavior summary
    total_repeat_buyers = sum(
        crb.repeat_buyers for crb in metrics.cohort_repeat_behavior
    )
    total_one_time = sum(crb.one_time_buyers for crb in metrics.cohort_repeat_behavior)
    overall_repeat_rate = (
        (total_repeat_buyers / (total_repeat_buyers + total_one_time) * 100)
        if (total_repeat_buyers + total_one_time) > 0
        else 0
    )

    print(f"\n  Overall Repeat Buyers: {total_repeat_buyers}")
    print(f"  Overall One-Time Buyers: {total_one_time}")
    print(f"  Overall Repeat Rate: {overall_repeat_rate:.2f}%")

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
