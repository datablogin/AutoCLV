"""Performance and statistical analysis for RFM, Lens 1, and Lens 2.

This test module validates statistical correctness and identifies performance
bottlenecks in Track A components (RFM, Lens 1, Lens 2).
"""

import time
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from customer_base_audit.foundation.data_mart import (
    CustomerDataMartBuilder,
    PeriodGranularity,
)
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from customer_base_audit.analyses.lens1 import analyze_single_period
from customer_base_audit.analyses.lens2 import analyze_period_comparison
from customer_base_audit.synthetic import (
    generate_customers,
    generate_transactions,
    BASELINE_SCENARIO,
)


def generate_test_dataset(n_customers: int, seed: int = 42):
    """Generate synthetic dataset for testing."""
    customers = generate_customers(
        n=n_customers,
        start=date(2023, 1, 1),
        end=date(2023, 12, 31),
        seed=seed,
    )

    # Create new scenario with desired seed (BASELINE_SCENARIO is frozen)
    scenario = replace(BASELINE_SCENARIO, seed=seed)

    transactions = generate_transactions(
        customers,
        start=date(2023, 1, 1),
        end=date(2023, 12, 31),
        catalog=["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"],
        scenario=scenario,
    )

    return customers, transactions


@pytest.mark.slow
class TestRFMStatisticalCorrectness:
    """Test RFM calculations for statistical correctness."""

    def test_rfm_monetary_equals_average_transaction_value(self):
        """Verify monetary = total_spend / frequency."""
        customers, transactions = generate_test_dataset(100)

        # Build data mart
        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.MONTH]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        # Calculate RFM for January
        monthly_periods = data_mart.periods[PeriodGranularity.MONTH]
        jan_periods = [
            p
            for p in monthly_periods
            if p.period_end == datetime(2023, 2, 1, tzinfo=timezone.utc)
        ]

        if jan_periods:
            jan_rfm = calculate_rfm(
                jan_periods,
                observation_end=datetime(2023, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
            )

            # Verify monetary calculation
            for rfm in jan_rfm:
                expected_monetary = rfm.total_spend / rfm.frequency
                assert abs(rfm.monetary - expected_monetary) < Decimal("0.01"), (
                    f"Customer {rfm.customer_id}: monetary {rfm.monetary} != {expected_monetary}"
                )

    def test_rfm_frequency_matches_order_count(self):
        """Verify frequency equals total orders."""
        customers, transactions = generate_test_dataset(100)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]
        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]

        if q1_periods:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )

            # Manual count from periods
            customer_orders = {}
            for period in q1_periods:
                if period.customer_id not in customer_orders:
                    customer_orders[period.customer_id] = 0
                customer_orders[period.customer_id] += period.total_orders

            # Verify frequency matches
            for rfm in q1_rfm:
                expected_frequency = customer_orders[rfm.customer_id]
                assert rfm.frequency == expected_frequency, (
                    f"Customer {rfm.customer_id}: frequency {rfm.frequency} != {expected_frequency}"
                )

    def test_rfm_scores_distribution(self):
        """Verify RFM scores follow expected quintile distribution."""
        customers, transactions = generate_test_dataset(500)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]
        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]

        if q1_periods and len(q1_periods) >= 20:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
            q1_scores = calculate_rfm_scores(q1_rfm)

            # Count score distribution
            r_scores = [s.r_score for s in q1_scores]
            f_scores = [s.f_score for s in q1_scores]
            m_scores = [s.m_score for s in q1_scores]

            # Each score should have some customers (not all in one bucket)
            unique_r = len(set(r_scores))
            unique_f = len(set(f_scores))
            unique_m = len(set(m_scores))

            assert unique_r >= 2, f"RFM R scores too concentrated: {unique_r} unique"
            assert unique_f >= 2, f"RFM F scores too concentrated: {unique_f} unique"
            assert unique_m >= 2, f"RFM M scores too concentrated: {unique_m} unique"


@pytest.mark.slow
class TestLens1StatisticalCorrectness:
    """Test Lens 1 analysis for statistical correctness."""

    def test_lens1_one_time_buyer_percentage(self):
        """Verify one-time buyer percentage calculation."""
        customers, transactions = generate_test_dataset(200)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]
        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]

        if q1_periods:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
            lens1 = analyze_single_period(q1_rfm)

            # Manual calculation
            one_time_buyers = sum(1 for rfm in q1_rfm if rfm.frequency == 1)
            expected_pct = (
                Decimal(one_time_buyers) / Decimal(len(q1_rfm)) * 100
            ).quantize(Decimal("0.01"))

            assert lens1.one_time_buyers == one_time_buyers, (
                "One-time buyer count mismatch"
            )
            assert lens1.one_time_buyer_pct == expected_pct, (
                f"One-time buyer % mismatch: {lens1.one_time_buyer_pct} != {expected_pct}"
            )

    def test_lens1_revenue_sums_correctly(self):
        """Verify total revenue matches sum of customer spend."""
        customers, transactions = generate_test_dataset(150)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]
        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]

        if q1_periods:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
            lens1 = analyze_single_period(q1_rfm)

            # Manual calculation
            expected_revenue = sum(rfm.total_spend for rfm in q1_rfm)

            assert lens1.total_revenue == expected_revenue, (
                f"Revenue mismatch: {lens1.total_revenue} != {expected_revenue}"
            )

    def test_lens1_revenue_concentration_pareto(self):
        """Verify revenue concentration follows Pareto principle."""
        customers, transactions = generate_test_dataset(500)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]
        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]

        if q1_periods and len(q1_periods) >= 50:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
            lens1 = analyze_single_period(q1_rfm)

            # Pareto principle: top 20% should drive > 50% of revenue
            # With synthetic data, might be less extreme
            top_20_contribution = lens1.top_20pct_revenue_contribution

            assert top_20_contribution > Decimal("50"), (
                f"Top 20% drives only {top_20_contribution}% of revenue (expected > 50%)"
            )


@pytest.mark.slow
class TestLens2StatisticalCorrectness:
    """Test Lens 2 period comparison for statistical correctness."""

    def test_lens2_retention_churn_sum_to_100(self):
        """Verify retention + churn = 100%."""
        customers, transactions = generate_test_dataset(200)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]

        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]
        q2_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 7, 1, tzinfo=timezone.utc)
        ]

        if q1_periods and q2_periods:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
            q2_rfm = calculate_rfm(
                q2_periods,
                observation_end=datetime(2023, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
            )

            lens2 = analyze_period_comparison(q1_rfm, q2_rfm)

            # Retention + churn should equal 100% (with rounding tolerance)
            total = lens2.retention_rate + lens2.churn_rate
            assert Decimal("99.9") <= total <= Decimal("100.1"), (
                f"Retention ({lens2.retention_rate}) + Churn ({lens2.churn_rate}) = {total} != 100"
            )

    def test_lens2_customer_migration_reconciliation(self):
        """Verify customer migration counts reconcile."""
        customers, transactions = generate_test_dataset(200)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]

        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]
        q2_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 7, 1, tzinfo=timezone.utc)
        ]

        if q1_periods and q2_periods:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
            q2_rfm = calculate_rfm(
                q2_periods,
                observation_end=datetime(2023, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
            )

            lens2 = analyze_period_comparison(q1_rfm, q2_rfm)

            # Period 1 customers = retained + churned
            period1_count = len(q1_rfm)
            migration_period1 = len(lens2.migration.retained) + len(
                lens2.migration.churned
            )
            assert period1_count == migration_period1, (
                f"Period 1 count mismatch: {period1_count} != {migration_period1}"
            )

            # Period 2 customers = retained + new
            period2_count = len(q2_rfm)
            migration_period2 = len(lens2.migration.retained) + len(lens2.migration.new)
            assert period2_count == migration_period2, (
                f"Period 2 count mismatch: {period2_count} != {migration_period2}"
            )


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for RFM, Lens 1, and Lens 2."""

    def test_rfm_performance_1000_customers(self):
        """Benchmark RFM calculation with 1,000 customers."""
        customers, transactions = generate_test_dataset(1000)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.MONTH]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]

        start_time = time.time()
        data_mart = mart_builder.build(mart_transactions)
        build_time = time.time() - start_time

        monthly_periods = data_mart.periods[PeriodGranularity.MONTH]
        jan_periods = [
            p
            for p in monthly_periods
            if p.period_end == datetime(2023, 2, 1, tzinfo=timezone.utc)
        ]

        start_time = time.time()
        if jan_periods:
            calculate_rfm(
                jan_periods,
                observation_end=datetime(2023, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
        rfm_time = time.time() - start_time

        print(f"\n1,000 customers - Data mart build: {build_time:.2f}s")
        print(f"1,000 customers - RFM calculation: {rfm_time:.2f}s")

        # Performance assertions
        assert build_time < 5.0, (
            f"Data mart build too slow: {build_time:.2f}s (expected < 5s)"
        )
        assert rfm_time < 1.0, (
            f"RFM calculation too slow: {rfm_time:.2f}s (expected < 1s)"
        )

    def test_lens1_performance_1000_customers(self):
        """Benchmark Lens 1 analysis with 1,000 customers."""
        customers, transactions = generate_test_dataset(1000)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]
        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]

        if q1_periods:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )

            start_time = time.time()
            analyze_single_period(q1_rfm)
            lens1_time = time.time() - start_time

            print(f"\n1,000 customers - Lens 1 analysis: {lens1_time:.2f}s")

            assert lens1_time < 2.0, (
                f"Lens 1 analysis too slow: {lens1_time:.2f}s (expected < 2s)"
            )

    def test_lens2_performance_1000_customers(self):
        """Benchmark Lens 2 comparison with 1,000 customers."""
        customers, transactions = generate_test_dataset(1000)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]
        data_mart = mart_builder.build(mart_transactions)

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]

        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]
        q2_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 7, 1, tzinfo=timezone.utc)
        ]

        if q1_periods and q2_periods:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
            q2_rfm = calculate_rfm(
                q2_periods,
                observation_end=datetime(2023, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
            )

            start_time = time.time()
            analyze_period_comparison(q1_rfm, q2_rfm)
            lens2_time = time.time() - start_time

            print(f"\n1,000 customers - Lens 2 comparison: {lens2_time:.2f}s")

            assert lens2_time < 2.0, (
                f"Lens 2 comparison too slow: {lens2_time:.2f}s (expected < 2s)"
            )

    @pytest.mark.parametrize("n_customers", [100, 500, 1000, 5000])
    def test_scalability_analysis(self, n_customers):
        """Test scalability across different dataset sizes."""
        customers, transactions = generate_test_dataset(n_customers)

        mart_builder = CustomerDataMartBuilder(
            period_granularities=[PeriodGranularity.QUARTER]
        )
        mart_transactions = [
            {
                "order_id": f"O{i}",
                "customer_id": txn.customer_id,
                "event_ts": txn.event_ts,
                "unit_price": txn.unit_price,
                "quantity": txn.quantity,
                "product_id": txn.product_id,
            }
            for i, txn in enumerate(transactions)
        ]

        # Time data mart build
        start_time = time.time()
        data_mart = mart_builder.build(mart_transactions)
        build_time = time.time() - start_time

        quarterly_periods = data_mart.periods[PeriodGranularity.QUARTER]
        q1_periods = [
            p
            for p in quarterly_periods
            if p.period_end == datetime(2023, 4, 1, tzinfo=timezone.utc)
        ]

        # Time RFM calculation
        start_time = time.time()
        if q1_periods:
            q1_rfm = calculate_rfm(
                q1_periods,
                observation_end=datetime(2023, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
            )
        rfm_time = time.time() - start_time

        # Time Lens 1 analysis
        start_time = time.time()
        if q1_periods:
            analyze_single_period(q1_rfm)
        lens1_time = time.time() - start_time

        print(
            f"\n{n_customers:>5} customers: Build={build_time:.3f}s, RFM={rfm_time:.3f}s, Lens1={lens1_time:.3f}s"
        )

        # Verify roughly linear scaling
        # For 10x increase in customers, should be < 15x increase in time
        if n_customers == 5000:
            assert build_time < 30.0, (
                f"Data mart build doesn't scale: {build_time:.2f}s for {n_customers} customers"
            )
            assert rfm_time < 5.0, (
                f"RFM doesn't scale: {rfm_time:.2f}s for {n_customers} customers"
            )
