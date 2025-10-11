"""Tests for RFM (Recency-Frequency-Monetary) calculation utilities."""

from datetime import datetime
from decimal import Decimal

import pytest

from customer_base_audit.foundation.data_mart import PeriodAggregation
from customer_base_audit.foundation.rfm import (
    RFMMetrics,
    RFMScore,
    calculate_rfm,
    calculate_rfm_scores,
)


class TestRFMMetrics:
    """Test RFMMetrics dataclass validation."""

    def test_valid_rfm_metrics(self):
        """Valid RFM metrics should be created successfully."""
        metrics = RFMMetrics(
            customer_id="C1",
            recency_days=10,
            frequency=5,
            monetary=Decimal("50.00"),
            observation_start=datetime(2023, 1, 1),
            observation_end=datetime(2023, 12, 31),
            total_spend=Decimal("250.00"),
        )
        assert metrics.customer_id == "C1"
        assert metrics.recency_days == 10
        assert metrics.frequency == 5
        assert metrics.monetary == Decimal("50.00")
        assert metrics.total_spend == Decimal("250.00")

    def test_negative_recency_raises_error(self):
        """Negative recency should raise ValueError."""
        with pytest.raises(ValueError, match="Recency cannot be negative"):
            RFMMetrics(
                customer_id="C1",
                recency_days=-1,
                frequency=5,
                monetary=Decimal("50.00"),
                observation_start=datetime(2023, 1, 1),
                observation_end=datetime(2023, 12, 31),
                total_spend=Decimal("250.00"),
            )

    def test_zero_frequency_raises_error(self):
        """Zero frequency should raise ValueError."""
        with pytest.raises(ValueError, match="Frequency must be positive"):
            RFMMetrics(
                customer_id="C1",
                recency_days=10,
                frequency=0,
                monetary=Decimal("50.00"),
                observation_start=datetime(2023, 1, 1),
                observation_end=datetime(2023, 12, 31),
                total_spend=Decimal("0.00"),
            )

    def test_negative_monetary_raises_error(self):
        """Negative monetary value should raise ValueError."""
        with pytest.raises(ValueError, match="Monetary value cannot be negative"):
            RFMMetrics(
                customer_id="C1",
                recency_days=10,
                frequency=5,
                monetary=Decimal("-50.00"),
                observation_start=datetime(2023, 1, 1),
                observation_end=datetime(2023, 12, 31),
                total_spend=Decimal("250.00"),
            )

    def test_monetary_mismatch_raises_error(self):
        """Monetary value not matching total_spend / frequency should raise ValueError."""
        with pytest.raises(ValueError, match="Monetary .* != total_spend / frequency"):
            RFMMetrics(
                customer_id="C1",
                recency_days=10,
                frequency=5,
                monetary=Decimal("100.00"),  # Should be 250 / 5 = 50
                observation_start=datetime(2023, 1, 1),
                observation_end=datetime(2023, 12, 31),
                total_spend=Decimal("250.00"),
            )


class TestCalculateRFM:
    """Test calculate_rfm function."""

    def test_empty_input_returns_empty_list(self):
        """Empty period aggregations should return empty list."""
        result = calculate_rfm([], datetime(2023, 12, 31))
        assert result == []

    def test_single_customer_single_period(self):
        """Calculate RFM for single customer with single period."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1),
                period_end=datetime(2023, 2, 1),
                total_orders=3,
                total_spend=150.0,
                total_margin=50.0,
                total_quantity=10,
            )
        ]
        observation_end = datetime(2023, 4, 15)
        rfm = calculate_rfm(periods, observation_end)

        assert len(rfm) == 1
        assert rfm[0].customer_id == "C1"
        assert rfm[0].frequency == 3
        assert rfm[0].total_spend == Decimal("150.00")
        assert rfm[0].monetary == Decimal("50.00")  # 150 / 3
        # Recency: from period_end (Feb 1) to observation_end (Apr 15)
        assert rfm[0].recency_days == (observation_end - datetime(2023, 2, 1)).days

    def test_single_customer_multiple_periods(self):
        """Calculate RFM for single customer across multiple periods."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1),
                period_end=datetime(2023, 2, 1),
                total_orders=3,
                total_spend=150.0,
                total_margin=50.0,
                total_quantity=10,
            ),
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 3, 1),
                period_end=datetime(2023, 4, 1),
                total_orders=2,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=5,
            ),
        ]
        observation_end = datetime(2023, 4, 15)
        rfm = calculate_rfm(periods, observation_end)

        assert len(rfm) == 1
        assert rfm[0].customer_id == "C1"
        assert rfm[0].frequency == 5  # 3 + 2
        assert rfm[0].total_spend == Decimal("250.00")  # 150 + 100
        assert rfm[0].monetary == Decimal("50.00")  # 250 / 5
        # Recency: from most recent period_end (Apr 1) to observation_end (Apr 15)
        assert rfm[0].recency_days == (observation_end - datetime(2023, 4, 1)).days

    def test_multiple_customers(self):
        """Calculate RFM for multiple customers."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1),
                period_end=datetime(2023, 2, 1),
                total_orders=3,
                total_spend=150.0,
                total_margin=50.0,
                total_quantity=10,
            ),
            PeriodAggregation(
                customer_id="C2",
                period_start=datetime(2023, 2, 1),
                period_end=datetime(2023, 3, 1),
                total_orders=1,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=2,
            ),
            PeriodAggregation(
                customer_id="C3",
                period_start=datetime(2023, 3, 1),
                period_end=datetime(2023, 4, 1),
                total_orders=10,
                total_spend=750.0,
                total_margin=250.0,
                total_quantity=50,
            ),
        ]
        observation_end = datetime(2023, 4, 15)
        rfm = calculate_rfm(periods, observation_end)

        assert len(rfm) == 3
        # Results should be sorted by customer_id
        assert rfm[0].customer_id == "C1"
        assert rfm[1].customer_id == "C2"
        assert rfm[2].customer_id == "C3"

        # Verify C1
        assert rfm[0].frequency == 3
        assert rfm[0].total_spend == Decimal("150.00")

        # Verify C2
        assert rfm[1].frequency == 1
        assert rfm[1].total_spend == Decimal("100.00")
        assert rfm[1].monetary == Decimal("100.00")  # 100 / 1

        # Verify C3
        assert rfm[2].frequency == 10
        assert rfm[2].total_spend == Decimal("750.00")
        assert rfm[2].monetary == Decimal("75.00")  # 750 / 10

    def test_one_time_buyer(self):
        """Calculate RFM for customer with single transaction."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1),
                period_end=datetime(2023, 2, 1),
                total_orders=1,
                total_spend=50.0,
                total_margin=15.0,
                total_quantity=1,
            )
        ]
        observation_end = datetime(2023, 12, 31)
        rfm = calculate_rfm(periods, observation_end)

        assert len(rfm) == 1
        assert rfm[0].frequency == 1
        assert rfm[0].monetary == Decimal("50.00")  # 50 / 1
        assert rfm[0].total_spend == Decimal("50.00")

    def test_accurate_recency_with_last_transaction_ts(self):
        """RFM recency should use actual transaction timestamp when available."""
        # Customer purchased on Dec 10, in a monthly period (Dec 1-31)
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 12, 1),
                period_end=datetime(2024, 1, 1),
                total_orders=1,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=1,
                last_transaction_ts=datetime(2023, 12, 10, 14, 30),  # Actual timestamp
            )
        ]
        observation_end = datetime(2024, 1, 15)
        rfm = calculate_rfm(periods, observation_end)

        assert len(rfm) == 1
        # Recency should be from Dec 10 to Jan 15
        expected_recency = (observation_end - datetime(2023, 12, 10, 14, 30)).days
        assert rfm[0].recency_days == expected_recency
        assert rfm[0].recency_days == 35  # Dec 10 14:30 to Jan 15 00:00 = 35 days

    def test_fallback_recency_without_last_transaction_ts(self):
        """RFM recency should fall back to period_end when last_transaction_ts is None."""
        # Same scenario but without last_transaction_ts (backward compatibility)
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 12, 1),
                period_end=datetime(2024, 1, 1),
                total_orders=1,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=1,
                last_transaction_ts=None,  # Fallback mode
            )
        ]
        observation_end = datetime(2024, 1, 15)
        rfm = calculate_rfm(periods, observation_end)

        assert len(rfm) == 1
        # Recency should be from Jan 1 (period_end) to Jan 15 = 14 days
        expected_recency = (observation_end - datetime(2024, 1, 1)).days
        assert rfm[0].recency_days == expected_recency
        assert rfm[0].recency_days == 14

    def test_recency_accuracy_difference(self):
        """Demonstrate the accuracy improvement from using actual timestamps."""
        # Customer purchased early in the month
        actual_purchase = datetime(2023, 12, 5, 10, 0)
        period_start = datetime(2023, 12, 1)
        period_end = datetime(2024, 1, 1)
        observation_end = datetime(2024, 1, 20)

        # With actual timestamp
        periods_with_ts = [
            PeriodAggregation(
                customer_id="C1",
                period_start=period_start,
                period_end=period_end,
                total_orders=1,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=1,
                last_transaction_ts=actual_purchase,
            )
        ]

        # Without timestamp (fallback)
        periods_without_ts = [
            PeriodAggregation(
                customer_id="C1",
                period_start=period_start,
                period_end=period_end,
                total_orders=1,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=1,
                last_transaction_ts=None,
            )
        ]

        rfm_with_ts = calculate_rfm(periods_with_ts, observation_end)
        rfm_without_ts = calculate_rfm(periods_without_ts, observation_end)

        # With timestamp: Jan 20 - Dec 5 10:00 = 45 days
        assert rfm_with_ts[0].recency_days == 45

        # Without timestamp: Jan 20 - Jan 1 = 19 days
        assert rfm_without_ts[0].recency_days == 19

        # Accuracy difference: 26 days (almost the entire month!)
        accuracy_improvement = (
            rfm_with_ts[0].recency_days - rfm_without_ts[0].recency_days
        )
        assert accuracy_improvement == 26

    def test_multiple_periods_uses_latest_transaction(self):
        """With multiple periods, recency should use the most recent transaction."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1),
                period_end=datetime(2023, 2, 1),
                total_orders=2,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=2,
                last_transaction_ts=datetime(2023, 1, 15),
            ),
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 3, 1),
                period_end=datetime(2023, 4, 1),
                total_orders=1,
                total_spend=50.0,
                total_margin=15.0,
                total_quantity=1,
                last_transaction_ts=datetime(2023, 3, 28),  # Most recent
            ),
        ]
        observation_end = datetime(2023, 4, 15)
        rfm = calculate_rfm(periods, observation_end)

        assert len(rfm) == 1
        # Should use March 28, not January 15
        expected_recency = (observation_end - datetime(2023, 3, 28)).days
        assert rfm[0].recency_days == expected_recency
        assert rfm[0].recency_days == 18

    def test_mixed_periods_with_and_without_timestamps(self):
        """Handle mix of periods with and without last_transaction_ts."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 1, 1),
                period_end=datetime(2023, 2, 1),
                total_orders=1,
                total_spend=50.0,
                total_margin=15.0,
                total_quantity=1,
                last_transaction_ts=None,  # Old data without timestamp
            ),
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 3, 1),
                period_end=datetime(2023, 4, 1),
                total_orders=1,
                total_spend=75.0,
                total_margin=20.0,
                total_quantity=1,
                last_transaction_ts=datetime(2023, 3, 15),  # New data with timestamp
            ),
        ]
        observation_end = datetime(2023, 4, 15)
        rfm = calculate_rfm(periods, observation_end)

        assert len(rfm) == 1
        # Should use the available timestamp from March period
        expected_recency = (observation_end - datetime(2023, 3, 15)).days
        assert rfm[0].recency_days == expected_recency
        assert rfm[0].recency_days == 31

    def test_future_dated_transaction_raises_error(self):
        """Transaction after observation_end should raise ValueError."""
        periods = [
            PeriodAggregation(
                customer_id="C1",
                period_start=datetime(2023, 12, 1),
                period_end=datetime(2024, 1, 1),
                total_orders=1,
                total_spend=100.0,
                total_margin=30.0,
                total_quantity=1,
                last_transaction_ts=datetime(2024, 2, 1),  # After observation_end
            )
        ]
        observation_end = datetime(2024, 1, 15)

        with pytest.raises(
            ValueError,
            match="Transaction timestamp .* cannot be after observation_end",
        ):
            calculate_rfm(periods, observation_end)


class TestCalculateRFMScores:
    """Test calculate_rfm_scores function."""

    def test_empty_input_returns_empty_list(self):
        """Empty RFM metrics should return empty list."""
        result = calculate_rfm_scores([])
        assert result == []

    def test_score_calculation_with_distinct_values(self):
        """Score RFM metrics into quintiles with distinct values."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            ),
            RFMMetrics(
                "C2",
                30,
                2,
                Decimal("75.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("150.00"),
            ),
            RFMMetrics(
                "C3",
                5,
                10,
                Decimal("100.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("1000.00"),
            ),
            RFMMetrics(
                "C4",
                50,
                1,
                Decimal("25.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("25.00"),
            ),
            RFMMetrics(
                "C5",
                20,
                8,
                Decimal("80.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("640.00"),
            ),
        ]
        scores = calculate_rfm_scores(metrics)

        assert len(scores) == 5

        # Results should be sorted by customer_id
        score_map = {s.customer_id: s for s in scores}

        # C3 should have high scores (low recency, high frequency, high monetary)
        assert score_map["C3"].r_score >= 4  # Most recent (recency_days = 5)
        assert score_map["C3"].f_score >= 4  # Highest frequency (10)
        assert score_map["C3"].m_score >= 4  # Highest monetary (100)

        # C4 should have low scores (high recency, low frequency, low monetary)
        assert score_map["C4"].r_score <= 2  # Least recent (recency_days = 50)
        assert score_map["C4"].f_score <= 2  # Lowest frequency (1)
        assert score_map["C4"].m_score <= 2  # Lowest monetary (25)

    def test_rfm_score_string_format(self):
        """RFM score string should be formatted as RFM (e.g., '543')."""
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            ),
            RFMMetrics(
                "C2",
                30,
                2,
                Decimal("75.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("150.00"),
            ),
        ]
        scores = calculate_rfm_scores(metrics)

        for score in scores:
            # RFM score should be 3 digits
            assert len(score.rfm_score) == 3
            assert score.rfm_score.isdigit()
            # Each digit should be 1-5
            assert all(1 <= int(d) <= 5 for d in score.rfm_score)

    def test_small_dataset_edge_case(self):
        """Small datasets should handle gracefully even with limited bins."""
        # Test with only 3 customers - may produce fewer than 5 bins
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            ),
            RFMMetrics(
                "C2",
                30,
                2,
                Decimal("75.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("150.00"),
            ),
            RFMMetrics(
                "C3",
                20,
                3,
                Decimal("60.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("180.00"),
            ),
        ]
        scores = calculate_rfm_scores(metrics)

        # Should still produce valid scores
        assert len(scores) == 3
        for score in scores:
            # All scores should be 1-5
            assert 1 <= score.r_score <= 5
            assert 1 <= score.f_score <= 5
            assert 1 <= score.m_score <= 5
            # RFM score should be valid format
            assert len(score.rfm_score) == 3
            assert score.rfm_score.isdigit()

        # Scores should be ordered correctly (C1 < C2 recency)
        score_map = {s.customer_id: s for s in scores}
        assert score_map["C1"].r_score > score_map["C2"].r_score  # C1 more recent

    def test_duplicate_values_edge_case(self):
        """Datasets with duplicate values should handle gracefully."""
        # All customers have same frequency - will create only 1 bin
        metrics = [
            RFMMetrics(
                "C1",
                10,
                5,
                Decimal("50.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("250.00"),
            ),
            RFMMetrics(
                "C2",
                30,
                5,
                Decimal("75.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("375.00"),
            ),
            RFMMetrics(
                "C3",
                20,
                5,
                Decimal("60.00"),
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                Decimal("300.00"),
            ),
        ]
        scores = calculate_rfm_scores(metrics)

        # Should still produce valid scores even with duplicate frequencies
        assert len(scores) == 3
        for score in scores:
            assert 1 <= score.r_score <= 5
            assert 1 <= score.f_score <= 5
            assert 1 <= score.m_score <= 5

        # All should have same frequency score since all have frequency=5
        score_map = {s.customer_id: s for s in scores}
        assert (
            score_map["C1"].f_score
            == score_map["C2"].f_score
            == score_map["C3"].f_score
        )


class TestRFMScore:
    """Test RFMScore dataclass validation."""

    def test_valid_rfm_score(self):
        """Valid RFM score should be created successfully."""
        score = RFMScore(
            customer_id="C1",
            r_score=5,
            f_score=4,
            m_score=3,
            rfm_score="543",
        )
        assert score.customer_id == "C1"
        assert score.r_score == 5
        assert score.f_score == 4
        assert score.m_score == 3
        assert score.rfm_score == "543"

    def test_invalid_r_score_raises_error(self):
        """R score outside 1-5 range should raise ValueError."""
        with pytest.raises(ValueError, match="r_score must be between 1 and 5"):
            RFMScore(
                customer_id="C1",
                r_score=6,
                f_score=4,
                m_score=3,
                rfm_score="643",
            )

    def test_invalid_f_score_raises_error(self):
        """F score outside 1-5 range should raise ValueError."""
        with pytest.raises(ValueError, match="f_score must be between 1 and 5"):
            RFMScore(
                customer_id="C1",
                r_score=5,
                f_score=0,
                m_score=3,
                rfm_score="503",
            )

    def test_invalid_m_score_raises_error(self):
        """M score outside 1-5 range should raise ValueError."""
        with pytest.raises(ValueError, match="m_score must be between 1 and 5"):
            RFMScore(
                customer_id="C1",
                r_score=5,
                f_score=4,
                m_score=10,
                rfm_score="54A",
            )

    def test_rfm_score_mismatch_raises_error(self):
        """RFM score string not matching individual scores should raise ValueError."""
        with pytest.raises(ValueError, match="rfm_score .* does not match"):
            RFMScore(
                customer_id="C1",
                r_score=5,
                f_score=4,
                m_score=3,
                rfm_score="555",  # Should be "543"
            )


class TestParallelProcessing:
    """Test parallel processing functionality in calculate_rfm."""

    def create_test_periods(self, num_customers: int) -> list[PeriodAggregation]:
        """Helper to create test period aggregations."""
        periods = []
        for i in range(num_customers):
            periods.append(
                PeriodAggregation(
                    customer_id=f"C{i}",
                    period_start=datetime(2023, 1, 1),
                    period_end=datetime(2023, 12, 31),
                    total_orders=5,
                    total_spend=250.0,
                    total_margin=75.0,
                    total_quantity=10,
                )
            )
        return periods

    def test_parallel_produces_same_results_as_serial(self):
        """Parallel processing produces identical results to serial processing."""
        periods = self.create_test_periods(1000)
        obs_end = datetime(2024, 1, 15)

        # Calculate using serial processing
        rfm_serial = calculate_rfm(periods, obs_end, parallel=False)

        # Calculate using parallel processing (force by lowering threshold)
        rfm_parallel = calculate_rfm(
            periods, obs_end, parallel=True, parallel_threshold=100
        )

        # Results should be identical (same order due to sorting)
        assert len(rfm_serial) == len(rfm_parallel)
        for serial, parallel in zip(rfm_serial, rfm_parallel):
            assert serial.customer_id == parallel.customer_id
            assert serial.recency_days == parallel.recency_days
            assert serial.frequency == parallel.frequency
            assert serial.monetary == parallel.monetary
            assert serial.total_spend == parallel.total_spend

    def test_parallel_flag_disabled(self):
        """When parallel=False, always uses serial processing."""
        periods = self.create_test_periods(1000)  # Would normally trigger parallel
        obs_end = datetime(2024, 1, 15)

        # Should use serial despite being above a low threshold
        rfm = calculate_rfm(
            periods, obs_end, parallel=False, parallel_threshold=100
        )

        assert len(rfm) == 1000
        assert rfm[0].customer_id == "C0"
        assert rfm[-1].customer_id == "C999"

    def test_parallel_threshold_respected(self):
        """Parallel processing only enabled when customer count >= threshold."""
        periods_small = self.create_test_periods(100)
        periods_large = self.create_test_periods(200)
        obs_end = datetime(2024, 1, 15)

        # Below threshold: should use serial
        rfm_small = calculate_rfm(
            periods_small, obs_end, parallel=True, parallel_threshold=150
        )
        assert len(rfm_small) == 100

        # At/above threshold: should use parallel
        rfm_large = calculate_rfm(
            periods_large, obs_end, parallel=True, parallel_threshold=150
        )
        assert len(rfm_large) == 200

        # Results should be sorted either way
        assert rfm_small[0].customer_id == "C0"
        assert rfm_large[0].customer_id == "C0"

    def test_custom_n_workers(self):
        """Custom n_workers parameter is respected."""
        periods = self.create_test_periods(1000)
        obs_end = datetime(2024, 1, 15)

        # Test with 2 workers
        rfm_2_workers = calculate_rfm(
            periods, obs_end, parallel=True, parallel_threshold=100, n_workers=2
        )

        # Test with 4 workers
        rfm_4_workers = calculate_rfm(
            periods, obs_end, parallel=True, parallel_threshold=100, n_workers=4
        )

        # Results should be identical regardless of worker count
        assert len(rfm_2_workers) == len(rfm_4_workers) == 1000
        for rfm2, rfm4 in zip(rfm_2_workers, rfm_4_workers):
            assert rfm2.customer_id == rfm4.customer_id
            assert rfm2.recency_days == rfm4.recency_days

    def test_empty_input_with_parallel_enabled(self):
        """Empty input returns empty list even with parallel enabled."""
        rfm = calculate_rfm(
            [], datetime(2024, 1, 15), parallel=True, parallel_threshold=0
        )
        assert rfm == []

    def test_single_customer_with_parallel_enabled(self):
        """Single customer works correctly with parallel processing."""
        periods = self.create_test_periods(1)
        obs_end = datetime(2024, 1, 15)

        rfm = calculate_rfm(
            periods, obs_end, parallel=True, parallel_threshold=1, n_workers=4
        )

        assert len(rfm) == 1
        assert rfm[0].customer_id == "C0"
        assert rfm[0].frequency == 5

    @pytest.mark.slow
    def test_parallel_performance_benefit(self):
        """Parallel processing provides performance benefit for large datasets."""
        import time

        # Create dataset with 100k customers
        periods = self.create_test_periods(100_000)
        obs_end = datetime(2024, 1, 15)

        # Measure serial time
        start_serial = time.time()
        rfm_serial = calculate_rfm(periods, obs_end, parallel=False)
        time_serial = time.time() - start_serial

        # Measure parallel time (with 4 workers)
        start_parallel = time.time()
        rfm_parallel = calculate_rfm(
            periods, obs_end, parallel=True, parallel_threshold=10_000, n_workers=4
        )
        time_parallel = time.time() - start_parallel

        # Verify results are identical
        assert len(rfm_serial) == len(rfm_parallel) == 100_000

        # Parallel should be faster (allow some variance for overhead)
        # We expect ~2-3x speedup with 4 workers, but use conservative 1.2x threshold
        speedup = time_serial / time_parallel
        assert (
            speedup > 1.2
        ), f"Parallel processing not faster: {time_serial:.2f}s vs {time_parallel:.2f}s (speedup: {speedup:.2f}x)"
